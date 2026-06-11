"""
sleep_monitor/morphology.py

Morphological cluster detection in the CAP differential (CLE-CRE) signal.

Key observation
---------------
In the CAP differential, each physiological event leaves a multi-peak imprint:

  Respiratory events: double-peak structure in the resp band (0.1-0.5 Hz).
    The bandpassed signal shows a characteristic crest per breath half-cycle.
    Within each crest, 1-2 local maxima may appear; the canonical "double peak"
    has 2 sub-peaks separated by 0.25-1.5 s.

  Cardiac events: triple-peak structure in the cardiac band (0.5-3.0 Hz).
    Each heartbeat produces a pulsation with 1-3 local maxima (systolic,
    reflected wave, diastolic components), spanning 0.05-0.6 s per beat.

Architecture: event-centred two-band pipeline
----------------------------------------------
1. preprocess_diff      - extract CLE-CRE window, filter into resp + cardiac bands
2. run_pipeline         - for each band:
     a. dominant_peaks  - find one strong peak per event (high prom, large min_dist)
     b. local_subpeaks  - scan +/-window around each dominant peak for sub-peaks
     c. classify        - resp (n_sub=2) / cardiac (n_sub=3) / other
3. events_to_rates      - count classified events -> sliding-window rate
4. acf_rates_from_cap   - ACF-based sliding-window rate from band-filtered CAP
                          (primary rate estimator, decoupled from event detection)
5. compute_rate_divisor - adaptive divisor: ratio of event rate to ACF rate
6. band_events_to_rates - convenience: event-based rate with adaptive divisor
7. gt_event_rates       - ACF-based GT rates from PSG Thorax / Pleth
8. gt_event_times_peaks - individual GT event times via peak detection
9. bland_altman         - aligned (mean, diff) arrays for Bland-Altman analysis
10. event_summary       - console summary

Decoupled rate estimation
--------------------------
Primary:   acf_rates_from_cap (uses existing sliding_rates / ACF on band signal)
Secondary: band_events_to_rates (event counting with adaptive divisor)

Both are validated against GT in the analysis script, so the contribution of the
morphological classification to rate accuracy is clearly visible.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from .config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI
from .sessions import SleepSession
from .rates import sliding_rates, rate_acf


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class MorphCfg:
    """
    Parameters for the event-centred two-band morphological pipeline.

    Each band has separate dominant-peak detection parameters and a local
    sub-peak scan window for morphology characterisation.
    """
    # --- Resp band (0.1-0.5 Hz) -------------------------------------------
    resp_bp_lo:  float = RESP_LO     # 0.1 Hz
    resp_bp_hi:  float = RESP_HI     # 0.5 Hz
    resp_bp_ord: int   = 3

    # Dominant peak: one prominent peak per breath half-cycle
    resp_dom_min_dist_s:  float = 1.20   # min distance between dominant peaks (s)
    resp_dom_prom_factor: float = 0.25   # prominence = factor * std(resp signal)

    # Sub-peak scan: look for fine structure within +/-window around each dominant
    resp_sub_win_s:       float = 1.20   # half-window for sub-peak scan (s)
    resp_sub_min_dist_s:  float = 0.20   # min distance between sub-peaks (s)
    resp_sub_prom_factor: float = 0.04   # prominence for sub-peaks (low)

    # Classification
    resp_n_min:       int   = 1     # sub-peak count -> "resp" event
    resp_n_max:       int   = 4     # narrow bandpass smooths crests to 1-peak; allow 1+
    resp_width_min:   float = 0.00  # 0 = allow single-peak events (width_s=0)
    resp_width_max:   float = 1.50

    # --- Cardiac band (0.5-3.0 Hz) ----------------------------------------
    card_bp_lo:  float = CARD_LO    # 0.5 Hz
    card_bp_hi:  float = CARD_HI    # 3.0 Hz
    card_bp_ord: int   = 3

    # Dominant peak: one prominent peak per heartbeat
    card_dom_min_dist_s:  float = 0.40   # min distance between dominant peaks (s)
    card_dom_prom_factor: float = 0.30   # prominence for dominant peaks

    # Sub-peak scan: look for fine structure within +/-window
    card_sub_win_s:       float = 0.35   # half-window for sub-peak scan (s)
    card_sub_min_dist_s:  float = 0.06   # min distance between sub-peaks (s)
    card_sub_prom_factor: float = 0.04   # prominence for sub-peaks (low)

    # Classification
    card_n_min:       int   = 2     # sub-peak count -> "cardiac" event (2-3)
    card_n_max:       int   = 4
    card_width_min:   float = 0.05  # cluster width (s)
    card_width_max:   float = 0.65

    # --- Display signal ---------------------------------------------------
    disp_bp_lo:  float = 0.05       # mild broadband filter for visualisation
    disp_bp_hi:  float = 8.0
    disp_bp_ord: int   = 2

    # --- Rate derivation -------------------------------------------------
    win_s:  float = 30.0            # sliding window length (s)
    step_s: float =  5.0            # step (s)


# ── Detected event ─────────────────────────────────────────────────────────────

@dataclass
class ClusterEvent:
    """
    One detected physiological event.

    Attributes
    ----------
    kind          : 'resp' | 'cardiac' | 'other'
    band          : 'resp' | 'cardiac'  (frequency band of detection)
    center_s      : dominant-peak time (s from window start)
    width_s       : time from first to last sub-peak (s); 0 when n_subpeaks==1
    n_subpeaks    : number of local maxima found in the local scan window
    peak_indices  : [dominant_peak_index]  (in the band-filtered window array)
    amplitudes    : amplitude at each sub-peak
    intervals_s   : inter-sub-peak intervals (length = n_subpeaks - 1)
    symmetry      : amplitude symmetry  1=symmetric  0=asymmetric
    """
    kind:         str
    band:         str
    center_s:     float
    width_s:      float
    n_subpeaks:   int
    peak_indices: List[int]
    amplitudes:   np.ndarray
    intervals_s:  np.ndarray
    symmetry:     float

    @property
    def amp_ratios(self) -> np.ndarray:
        mx = float(self.amplitudes.max()) + 1e-12
        return self.amplitudes / mx


# ── Internal filter ────────────────────────────────────────────────────────────

def _bp(x: np.ndarray, lo: float, hi: float,
        fs: float = FS, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, x.astype(np.float64))


# ── Step 1: preprocess ─────────────────────────────────────────────────────────

def preprocess_diff(
    session: SleepSession,
    start_hr: float,
    win_hr: float,
    cfg: Optional[MorphCfg] = None,
    acc_removal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract a time window from a session, compute CLE-CRE, and produce band-
    filtered signals for morphological analysis.

    Returns
    -------
    t_s    : (N,) seconds from window start
    sig    : (N,) broadband display-filtered CLE-CRE
    extras : dict — raw_diff, sig_resp, sig_card, gt_thorax_raw,
                    gt_pleth_raw, gt_ecg_raw, gt_resp, gt_card
    """
    if cfg is None:
        cfg = MorphCfg()

    t = session.time_hr
    mask = (t >= start_hr) & (t <= start_hr + win_hr)
    if not mask.any():
        raise ValueError(f'No samples in [{start_hr:.3f}, {start_hr+win_hr:.3f}] hr')
    idx = np.where(mask)[0]
    t_s = (t[idx] - t[idx[0]]) * 3600.0

    cap, psg = session.cap, session.psg
    raw_diff = (cap['CLE'][idx].astype(np.float64)
                - cap['CRE'][idx].astype(np.float64))
    fs = session.fs

    def _remove_acc(sig_raw, lo, hi, order=3):
        acc = cap['acc_mag'][idx].astype(np.float64)
        s = _bp(sig_raw, lo, hi, fs, order)
        a = _bp(acc,     lo, hi, fs, order)
        beta = np.dot(a, s) / (np.dot(a, a) + 1e-12)
        return s - beta * a

    if acc_removal:
        sig_disp = _remove_acc(raw_diff, cfg.disp_bp_lo, cfg.disp_bp_hi, cfg.disp_bp_ord)
        sig_resp = _remove_acc(raw_diff, cfg.resp_bp_lo, cfg.resp_bp_hi, cfg.resp_bp_ord)
        sig_card = _remove_acc(raw_diff, cfg.card_bp_lo, cfg.card_bp_hi, cfg.card_bp_ord)
    else:
        sig_disp = _bp(raw_diff, cfg.disp_bp_lo, cfg.disp_bp_hi, fs, cfg.disp_bp_ord)
        sig_resp = _bp(raw_diff, cfg.resp_bp_lo, cfg.resp_bp_hi, fs, cfg.resp_bp_ord)
        sig_card = _bp(raw_diff, cfg.card_bp_lo, cfg.card_bp_hi, fs, cfg.card_bp_ord)

    extras = {
        'raw_diff':      raw_diff,
        'sig_resp':      sig_resp,
        'sig_card':      sig_card,
        'gt_thorax_raw': psg['Thorax'][idx].astype(np.float64),
        'gt_pleth_raw':  psg['Pleth'][idx].astype(np.float64),
        'gt_ecg_raw':    psg['ECG'][idx].astype(np.float64),
        'gt_resp':       _bp(psg['Thorax'][idx].astype(np.float64),
                             RESP_LO, RESP_HI, fs),
        'gt_card':       _bp(psg['Pleth'][idx].astype(np.float64),
                             CARD_LO, CARD_HI, fs),
    }
    return t_s, sig_disp, extras


# ── Step 2: event-centred pipeline ─────────────────────────────────────────────

def _scan_local_subpeaks(
    sig: np.ndarray,
    dom_idx: int,
    half_win: int,
    min_dist: int,
    prom_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Within [dom_idx - half_win, dom_idx + half_win], find all local maxima
    that constitute the sub-peak fine structure around a dominant event.

    Returns (local_peak_indices_absolute, amplitudes).
    If no sub-peaks are found beyond the dominant itself, returns the dominant.
    """
    lo = max(0, dom_idx - half_win)
    hi = min(len(sig), dom_idx + half_win + 1)
    local = sig[lo:hi]

    if len(local) < 3:
        return np.array([dom_idx]), np.array([sig[dom_idx]])

    prom_thr = prom_factor * float(np.std(local)) if np.std(local) > 0 else 1e-12
    pks, _   = find_peaks(local, distance=min_dist, prominence=prom_thr)

    if len(pks) == 0:
        return np.array([dom_idx]), np.array([sig[dom_idx]])

    abs_pks = pks + lo
    return abs_pks, sig[abs_pks]


def run_pipeline(
    sig_raw: np.ndarray,
    t_s: np.ndarray,
    fs: float = FS,
    cfg: Optional[MorphCfg] = None,
    extras: Optional[dict] = None,
) -> List[ClusterEvent]:
    """
    Event-centred morphological pipeline on both frequency bands.

    For each band:
      1. Find dominant peaks (one per event, high prominence + min distance)
      2. Scan a local window around each dominant peak for fine sub-peaks
      3. Classify based on sub-peak count and width

    Parameters
    ----------
    sig_raw : broadband CLE-CRE (used only if band signals not in `extras`)
    t_s     : (N,) seconds from window start
    fs      : sampling rate
    cfg     : MorphCfg
    extras  : dict from preprocess_diff(); sig_resp / sig_card reused directly

    Returns
    -------
    List of ClusterEvent sorted by centre time.
    """
    if cfg is None:
        cfg = MorphCfg()

    if extras is not None and 'sig_resp' in extras:
        sig_resp = extras['sig_resp']
        sig_card = extras['sig_card']
    else:
        sig_resp = _bp(sig_raw, cfg.resp_bp_lo, cfg.resp_bp_hi, fs, cfg.resp_bp_ord)
        sig_card = _bp(sig_raw, cfg.card_bp_lo, cfg.card_bp_hi, fs, cfg.card_bp_ord)

    band_specs = [
        # (band_name, signal, dom_min_dist_s, dom_prom, sub_win_s, sub_min_dist_s, sub_prom,
        #  n_min, n_max, width_min, width_max)
        ('resp', sig_resp,
         cfg.resp_dom_min_dist_s, cfg.resp_dom_prom_factor,
         cfg.resp_sub_win_s, cfg.resp_sub_min_dist_s, cfg.resp_sub_prom_factor,
         cfg.resp_n_min, cfg.resp_n_max, cfg.resp_width_min, cfg.resp_width_max),
        ('cardiac', sig_card,
         cfg.card_dom_min_dist_s, cfg.card_dom_prom_factor,
         cfg.card_sub_win_s, cfg.card_sub_min_dist_s, cfg.card_sub_prom_factor,
         cfg.card_n_min, cfg.card_n_max, cfg.card_width_min, cfg.card_width_max),
    ]

    events: List[ClusterEvent] = []

    for (band, sig, dom_d_s, dom_p, sub_w_s, sub_d_s, sub_p,
         n_min, n_max, w_min, w_max) in band_specs:

        dom_dist = max(1, int(round(dom_d_s * fs)))
        dom_prom = dom_p * float(np.std(sig))
        dom_pks, _ = find_peaks(sig, distance=dom_dist, prominence=dom_prom)

        half_win = int(round(sub_w_s * fs))
        sub_dist = max(1, int(round(sub_d_s * fs)))

        for dp in dom_pks:
            abs_pks, amps = _scan_local_subpeaks(sig, int(dp), half_win, sub_dist, sub_p)
            times         = t_s[abs_pks]
            n_sub         = len(abs_pks)

            if n_sub > 1:
                width     = float(times[-1] - times[0])
                intervals = np.diff(times)
                sym       = 1.0 - abs(float(amps[0]) - float(amps[-1])) / (amps.max() + 1e-12)
            else:
                width     = 0.0
                intervals = np.array([0.0])
                sym       = 1.0

            # Classify
            if n_min <= n_sub <= n_max and w_min <= width <= w_max:
                kind = band   # 'resp' or 'cardiac'
            else:
                kind = 'other'

            events.append(ClusterEvent(
                kind         = kind,
                band         = band,
                center_s     = float(t_s[dp]),
                width_s      = width,
                n_subpeaks   = n_sub,
                peak_indices = [int(dp)],
                amplitudes   = amps.copy(),
                intervals_s  = intervals.copy(),
                symmetry     = sym,
            ))

    events.sort(key=lambda e: e.center_s)
    return events


# ── ACF-based rates from band-filtered CAP signal ─────────────────────────────

def acf_rates_from_cap(
    sig_resp: np.ndarray,
    sig_card: np.ndarray,
    fs: float = FS,
    win_s: float = 30.0,
    step_s: float = 5.0,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Sliding-window ACF rate estimates from band-filtered CAP signals.
    This is the primary rate estimator, independent of event detection.

    Returns
    -------
    (resp_t, resp_hz), (card_t, card_hz)
        Time axes (s) and rates (Hz) for respiratory and cardiac.
    """
    resp_t, resp_d = sliding_rates(sig_resp, RESP_LO, RESP_HI, fs, win_s, step_s)
    card_t, card_d = sliding_rates(sig_card, CARD_LO, CARD_HI, fs, win_s, step_s)
    return (resp_t, resp_d['acf']), (card_t, card_d['acf'])


# ── Adaptive divisor ───────────────────────────────────────────────────────────

def compute_rate_divisor(
    events: List[ClusterEvent],
    band: str,
    sig_band: np.ndarray,
    fs: float = FS,
) -> float:
    """
    Estimate the scaling factor k to convert raw event rate to the true
    physiological rate:  true_rate = raw_event_rate / k.

    k is the median ratio of (event rate) / (ACF rate) — a continuous value,
    NOT rounded to an integer.  Use calibrate_k_cardiac / calibrate_k_resp
    for session-level calibration; this function operates on a single segment.
    """
    ev_times = sorted(e.center_s for e in events if e.band == band)
    if len(ev_times) < 4:
        return 1.0

    event_rate = (len(ev_times) - 1) / (ev_times[-1] - ev_times[0])

    f_lo = RESP_LO if band == 'resp' else CARD_LO
    f_hi = RESP_HI if band == 'resp' else CARD_HI
    acf = rate_acf(sig_band, f_lo, f_hi, fs)

    if np.isnan(acf) or acf <= 0:
        return 1.0

    return max(0.5, event_rate / acf)


# ── Rate derivation from events ────────────────────────────────────────────────

def events_to_rates(
    events: List[ClusterEvent],
    kind: str,
    t_total_s: float,
    win_s: float = 30.0,
    step_s: float = 5.0,
    band: Optional[str] = None,
    rate_divisor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window rate from event centre times.

    Parameters
    ----------
    events       : output of run_pipeline()
    kind         : event kind filter ('resp', 'cardiac', 'other', or 'any')
    t_total_s    : total signal duration (s)
    win_s        : window length (s)
    step_s       : step (s)
    band         : additionally filter by band ('resp' or 'cardiac'); None=all
    rate_divisor : divide raw event rate by this factor

    Returns (t_centres_s, rates_hz).  NaN when < 2 events in window.
    """
    if kind == 'any':
        ev_times = np.array([e.center_s for e in events
                             if (band is None or e.band == band)])
    else:
        ev_times = np.array([e.center_s for e in events
                             if e.kind == kind
                             and (band is None or e.band == band)])

    t_centres = np.arange(win_s / 2.0, t_total_s - win_s / 2.0 + step_s, step_s)
    rates     = np.full(len(t_centres), np.nan)

    for i, tc in enumerate(t_centres):
        lo, hi  = tc - win_s / 2.0, tc + win_s / 2.0
        in_win  = ev_times[(ev_times >= lo) & (ev_times <= hi)]
        if len(in_win) >= 2:
            raw_hz   = (len(in_win) - 1) / float(in_win[-1] - in_win[0])
            rates[i] = raw_hz / rate_divisor

    return t_centres, rates


def band_events_to_rates(
    events: List[ClusterEvent],
    band: str,
    t_total_s: float,
    sig_band: Optional[np.ndarray] = None,
    fs: float = FS,
    win_s: float = 30.0,
    step_s: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convenience wrapper: rate from ALL events in a band with adaptive k.

    k is computed from the ratio of raw event rate to ACF rate (continuous,
    not rounded to integer).  When sig_band is unavailable, k defaults to 1.0
    (no correction).  For session-level calibration, use calibrate_k_cardiac
    or calibrate_k_resp from sleep_monitor.rates.

    Returns
    -------
    (t_centres, rates_hz, k_used)
    """
    if sig_band is not None and len(events) >= 4:
        div = compute_rate_divisor(events, band, sig_band, fs)
    else:
        div = 1.0

    t_c, r = events_to_rates(events, 'any', t_total_s, win_s, step_s,
                              band=band, rate_divisor=float(div))
    return t_c, r, div


# ── GT rates ───────────────────────────────────────────────────────────────────

def gt_event_rates(
    psg_sig: np.ndarray,
    fs: float = FS,
    band: str = 'resp',
    win_s: float = 30.0,
    step_s: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GT sliding-window ACF rates from PSG Thorax (resp) or Pleth (cardiac).
    Returns (t_s, rates_hz).
    """
    f_lo = RESP_LO if band == 'resp' else CARD_LO
    f_hi = RESP_HI if band == 'resp' else CARD_HI
    t_s, rate_dict = sliding_rates(psg_sig, f_lo, f_hi, fs, win_s, step_s)
    return t_s, rate_dict['acf']


def gt_event_times_peaks(
    psg_sig: np.ndarray,
    fs: float = FS,
    band: str = 'resp',
    prom_factor: float = 0.30,
) -> np.ndarray:
    """
    Detect individual event times (breath or beat) in a PSG signal.
    Returns (M,) times in seconds from signal start.
    """
    f_lo = RESP_LO if band == 'resp' else CARD_LO
    f_hi = RESP_HI if band == 'resp' else CARD_HI
    filtered = _bp(psg_sig, f_lo, f_hi, fs)
    min_dist = max(1, int(round(0.85 * fs / f_hi)))
    prom_thr = prom_factor * float(np.std(filtered))
    pks, _   = find_peaks(filtered, distance=min_dist, prominence=prom_thr)
    return pks / fs


# ── Bland-Altman ───────────────────────────────────────────────────────────────

def bland_altman(
    method_t: np.ndarray,
    method_r: np.ndarray,
    gt_t: np.ndarray,
    gt_r: np.ndarray,
    scale: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Align GT onto method time grid and compute Bland-Altman arrays.

    Rates are scaled by `scale` (60 -> Hz to br/min or BPM).

    Returns (means, diffs, stats_dict).
    stats keys: bias, loa_lo, loa_hi, mae, r, n.
    """
    empty = dict(bias=np.nan, loa_lo=np.nan, loa_hi=np.nan,
                 mae=np.nan, r=np.nan, n=0)
    valid_gt = ~np.isnan(gt_r)
    valid_m  = ~np.isnan(method_r)
    if valid_gt.sum() < 2 or valid_m.sum() < 2:
        return np.array([]), np.array([]), empty

    f_gt      = interp1d(gt_t[valid_gt], gt_r[valid_gt],
                         kind='linear', bounds_error=False, fill_value=np.nan)
    gt_interp = f_gt(method_t)
    ok        = valid_m & ~np.isnan(gt_interp)

    if ok.sum() < 3:
        return np.array([]), np.array([]), empty

    m     = method_r[ok] * scale
    g     = gt_interp[ok] * scale
    means = (m + g) / 2.0
    diffs = m - g
    bias  = float(np.mean(diffs))
    sd    = float(np.std(diffs))

    r_val = float(pearsonr(m, g)[0]) if len(m) >= 3 else np.nan

    return means, diffs, dict(
        bias   = bias,
        loa_lo = bias - 1.96 * sd,
        loa_hi = bias + 1.96 * sd,
        mae    = float(np.mean(np.abs(diffs))),
        r      = r_val,
        n      = int(ok.sum()),
    )


# ── Console summary ────────────────────────────────────────────────────────────

def event_summary(events: List[ClusterEvent], t_total_s: float) -> None:
    """Print a concise morphological event summary to stdout."""
    from collections import Counter, defaultdict

    total = len(events)
    if total == 0:
        print('No events detected.')
        return

    counts = Counter(e.kind for e in events)
    print(f'\nMorphological event summary  ({t_total_s:.0f} s = {t_total_s/60:.1f} min)')
    print(f'  Total events : {total}')

    for kind in ('resp', 'cardiac', 'other'):
        n = counts[kind]
        pct = 100 * n / total
        ev_t = sorted(e.center_s for e in events if e.kind == kind)
        rate_str = ''
        if len(ev_t) >= 2:
            hz = (len(ev_t) - 1) / (ev_t[-1] - ev_t[0])
            unit = 'br/min' if kind == 'resp' else ('BPM' if kind == 'cardiac' else 'ev/min')
            rate_str = f'  (naive {hz*60:.1f} {unit})'
        print(f'  {kind:8s}: {n:4d}  ({pct:.0f}%){rate_str}')

    print('\n  Sub-peak count per band:')
    for bnd in ('resp', 'cardiac'):
        n_dist: dict = defaultdict(int)
        for e in events:
            if e.band == bnd:
                n_dist[e.n_subpeaks] += 1
        dist_str = '  '.join(f'n={k}:{v}' for k, v in sorted(n_dist.items()))
        print(f'    {bnd}: {dist_str}')

    print('\n  Cluster width of classified events (s):')
    for kind in ('resp', 'cardiac'):
        ws = [e.width_s for e in events if e.kind == kind and e.n_subpeaks > 1]
        if ws:
            print(f'    {kind}: mean={np.mean(ws):.3f}  '
                  f'std={np.std(ws):.3f}  '
                  f'[{np.min(ws):.3f}, {np.max(ws):.3f}]')
