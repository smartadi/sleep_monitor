"""Low-band (0.1-3 Hz) and sigma-band SEC power at PSG-scored N2 sleep spindles.

Manuscript items produced
    §2.8 ¶162-163   spindle alignment, low vs sigma band, random-N2 and arousal controls
    §3.4 ¶198       N2 spindles per recording (351-2,134) and in total (14,305)
    §3.4 ¶199       low-band increase CLE/CRE 0.45-0.49 dB, CH 0.55 dB; "all 12
                    recordings"; absent at random N2; survives arousal exclusion
    §3.4 ¶200       SEC sigma change 0.02-0.03 dB; EEG sigma +3.45 dB
    Fig. 6 (¶202)   a centre-triggered low-band average, b per-spindle low vs sigma
                    histogram (CH), c per-recording low-band change per channel
    tables          outputs/tables/s07_spindles/*.csv (listed in run())

Method, in one paragraph. Each PSG-scored spindle is placed at its centre; those
whose centre falls in an N2 epoch are kept. For every spindle (and every control
time point) a +/-8 s window is cut from each channel (CLE, CRE, CLE-CRE, CH and
contact EEG), a short-time spectrogram is taken (128-sample segments, 96 overlap),
band power is averaged in dB over the band's frequency bins, and the per-event
change is the mean over the core |t| < 1 s minus the mean over the window's own
edges |t| > 5 s. Recording values are the mean over its spindles; the pooled value
is the mean over all spindles of all recordings. Two controls re-use the ERSP of
the same windows: random N2 time points, and spindles split by whether a scored
arousal lies within +/-5 s.

Ported from:
    analysis/spindles/spindle_loader.py          load_spindles, _find_spindle_file,
                                                 _SPINDLE_RE, _tod_sec
    analysis/spindles/spindle_lowband_detection.py
                                                 get_channel, stage_at,
                                                 channel_event_metrics, run_session,
                                                 main (per-session + pooled table),
                                                 make_figure (Fig. 6)
    analysis/spindles/spindle_ersp.py            session_ersp, main (sigma core table)
    analysis/spindles/spindle_ersp_control.py    load_arousals, core_spectrum, main
    writeup/paper/key_numbers.py                 _spindles (the paper's summary values)

Changes from the legacy code:
    1. CORRECTION -- the paper's 0.45-0.49 dB (CLE/CRE), CH 0.55 dB and EEG sigma
       3.45 dB came from key_numbers.py averaging all 13 rows of
       spindle_lowband_detection.csv, i.e. the 12 recordings AND the POOLED row.
       That is neither the per-recording mean nor the pooled per-spindle mean. Here
       the per-recording table (12 rows) and the pooled per-spindle values are kept
       in separate tables and both are registered against the paper.
    2. CORRECTION -- the spindle annotation file ('Spindle  K') also carries the
       technologist's K-complex marks (label 'K-Complex'); the legacy loader did
       not read the label, so every K-complex in N2 was analysed as a spindle and
       was also excluded from the control pool as if it were a spindle. Only
       'Spindle' marks are kept now. How many marks this removes, and the effect on
       every registered number, is written to kcomplex_removed.csv and
       kcomplex_effect.csv (the legacy result is recomputed for that comparison).
    3. NOTE -- the low band is 0.1-3 Hz in code (the f = 0 STFT bin is excluded;
       at 128-sample segments the band is the 0.78/1.56/2.34 Hz bins). The
       manuscript says 0-3 Hz. The computation is kept; the label is reported.
    4. NOTE -- the detection-rate controls are every N2 quarter-epoch point
       (0.25/0.5/0.75 of each N2 epoch) at least 3 s from a spindle: all of them,
       not count-matched. The random-N2 ERSP control is count-matched (one per
       spindle, epoch mid-points, drawn with replacement when there are fewer N2
       epochs than spindles) but is not required to be spindle-free. Methods ¶163
       says "count-matched spindle-free N2 windows"; neither control is both.
       Kept as is, noted in the numbers report.
    5. "Observed in all 12 recordings" is registered per channel: the recording
       mean is positive in 12/12 for CH and CLE-CRE but 11/12 for CLE and CRE.
    6. The Fig. 6b caption quotes 0.55 / 0.02 dB; the panel plots the pooled
       per-spindle distribution, whose means are registered against it.
    7. The legacy 0.5-3 Hz variant band, the unused RNG of the detection script,
       its print-outs and the --figure-only path are removed. The legacy spindle
       and arousal loaders are kept verbatim (seclib.load_arousals keeps events by
       start/end rather than by centre and does not skip '._' files, so it would
       not reproduce the legacy control).
    8. Fig. 6 is redrawn from the same arrays: panel titles shortened, low band
       labelled 0.1-3 Hz, per-recording medians kept as ticks in panel c.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.stats import trim_mean

import seclib
from seclib import figures
from seclib.numbers import Numbers

STAGE = 's07_spindles'
REQUIRES = []

FS = seclib.FS
N2_CODE = 2
WIN_HALF = 8.0            # +/- s extracted per event
CORE_HALF = 1.0           # |t| < CORE_HALF is the "during spindle" core
BASE_EDGE = 5.0           # |t| > BASE_EDGE is the event's own baseline
NPERSEG = 128
NOVERLAP = 96
FMAX = 45.0               # ERSP frequency ceiling
CTRL_MIN_GAP = 3.0        # s: detection controls are at least this far from a spindle
MAX_EVENTS = 400          # ERSP: events per recording per condition (random subset)
AROUSAL_NEAR = 5.0        # s: a spindle "has an arousal" if one is centred this close

SIGMA = (11.0, 16.0)
# NOTE (change 3): lower edge 0.1 Hz, not 0, so the f = 0 STFT bin is excluded:
# after per-segment mean removal it holds only residual within-window drift.
LOW = (0.1, 3.0)
BANDS = {'low_03': LOW, 'sigma': SIGMA}
TAG = {'low_03': 'low', 'sigma': 'sigma'}

CAP_CHANNELS = ['CLE', 'CRE', 'CLE-CRE', 'CH']
ALL_CHANNELS = CAP_CHANNELS + ['EEG']
ERSP_CHANNELS = ['EEG', 'CLE-CRE', 'CLE', 'CRE', 'CH']
CONTROL_CHANNELS = ['CH', 'CLE-CRE']
CONDITIONS = ['spindle', 'randN2', 'arousal', 'spindle_arous', 'spindle_noarous']
COLORS = {'CLE': '#4C72B0', 'CRE': '#55A868', 'CLE-CRE': '#C44E52', 'CH': '#8172B3'}
SHOW = {'CLE': 'CLE', 'CRE': 'CRE', 'CLE-CRE': 'CLE−CRE', 'CH': 'CH', 'EEG': 'EEG'}


# ── PSG annotations ───────────────────────────────────────────────────────────
# Line format: HH:MM:SS,mmm-HH:MM:SS,mmm; <value>;<label>
_SPINDLE_RE = re.compile(
    r'^(\d{2}):(\d{2}):(\d{2}),(\d{3})-(\d{2}):(\d{2}):(\d{2}),(\d{3});\s*'
    r'([0-9.]+);(.+)$'
)


def _tod_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _tod_to_hr(tod, session):
    """Wall-clock seconds-of-day -> hours from the SEC recording start."""
    ts = session.time_start
    if hasattr(ts, 'tz') and ts.tz is not None:
        ts = ts.tz_localize(None) if hasattr(ts, 'tz_localize') else ts.replace(tzinfo=None)
    start = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6
    off = tod - start
    off = np.where(off < -43200, off + 86400, off)
    off = np.where(off > 43200, off - 86400, off)
    return off / 3600.0


def _find_spindle_file(psg_dir, kind: str) -> Optional[str]:
    """kind = 'K' (duration file) or 'frequency'."""
    stem = 'Spindle  K' if kind == 'K' else 'Spindle frequency'
    pattern = str(psg_dir / 'PSG_analysis_*' / f'{stem}*.txt')
    matches = [m for m in glob.glob(pattern) if 'MACOSX' not in m]
    return sorted(matches)[0] if matches else None


def load_spindles(session, drop_kcomplex: bool = True) -> Optional[dict]:
    """Scored spindle centres (hours from SEC start) within the recording.

    Timing comes from the 'Spindle  K' file, or from the 'Spindle frequency' file
    when the former is missing (S4N2). Also returns how many K-complex marks the
    file held and how many of them would otherwise have been kept.
    """
    psg_dir = session.meta.get('psg_dir')
    src = _find_spindle_file(psg_dir, 'K') or _find_spindle_file(psg_dir, 'frequency')
    if src is None or session.time_start is None:
        return None
    starts, ends, is_kc = [], [], []
    with open(src, 'r', encoding='latin-1') as fh:
        for line in fh:
            mt = _SPINDLE_RE.match(line.strip())
            if not mt:
                continue
            starts.append(_tod_sec(*mt.group(1, 2, 3, 4)))
            ends.append(_tod_sec(*mt.group(5, 6, 7, 8)))
            is_kc.append(not mt.group(10).strip().lower().startswith('spindle'))
    if not starts:
        return None
    center_hr = 0.5 * (_tod_to_hr(np.array(starts), session)
                       + _tod_to_hr(np.array(ends), session))
    is_kc = np.array(is_kc)
    in_win = (center_hr >= 0.0) & (center_hr <= float(session.time_hr[-1]))
    keep = in_win & ~is_kc if drop_kcomplex else in_win   # CORRECTION: label filter
    return {'center_hr': center_hr[keep],
            'kc_center_hr': center_hr[in_win & is_kc],
            'n_kcomplex_file': int(is_kc.sum()),
            'source': os.path.basename(src)}


def load_arousal_centres(session) -> Optional[np.ndarray]:
    """Centre times (hours from SEC start) of scored 'Classification Arousal' events."""
    pat = str(session.meta['psg_dir'] / 'PSG_analysis_*' / 'Classification Arousal*.txt')
    fs = [m for m in glob.glob(pat)
          if 'MACOSX' not in m and os.path.basename(m)[:2] != '._']
    if not fs or session.time_start is None:
        return None
    starts, ends = [], []
    with open(sorted(fs)[0], 'r', encoding='latin-1') as fh:
        for line in fh:
            mt = _SPINDLE_RE.match(line.strip())
            if not mt:
                continue
            starts.append(_tod_sec(*mt.group(1, 2, 3, 4)))
            ends.append(_tod_sec(*mt.group(5, 6, 7, 8)))
    if not starts:
        return None
    center = 0.5 * (_tod_to_hr(np.array(starts), session)
                    + _tod_to_hr(np.array(ends), session))
    return center[(center >= 0) & (center <= float(session.time_hr[-1]))]


def get_channel(s, ch):
    if ch == 'EEG':
        return s.psg['EEG'].astype(np.float64)
    if ch == 'CLE-CRE':
        return s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    return s.cap[ch].astype(np.float64)


def stage_at(t_hr, prof):
    """Stage code of the scored epoch nearest each time (-1 if none within 30 s)."""
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(t_hr), -1, np.int8)
    for i, t in enumerate(t_hr):
        j = np.argmin(np.abs(tep - t))
        if abs(tep[j] - t) < 30.0 / 3600.0:
            out[i] = codes[j]
    return out


def _to_samples(t_hr):
    return np.round(np.asarray(t_hr) * 3600.0 * FS).astype(int)


# ── Per-event band power (detection table, Fig. 6) ────────────────────────────

def channel_event_metrics(sig, centers_samp, win_samp, want_trace=None):
    """Per-event band-power change in dB, core |t|<1 s vs the event's own |t|>5 s.

    Returns ({band: per-event dB array}, mean dB(t) curve of `want_trace`, t axis).
    """
    n = len(sig)
    per_band = {b: [] for b in BANDS}
    trace_acc, trace_k = None, 0
    tcen = core_t = base_t = None
    fmask = {}
    for c in centers_samp:
        a, b = c - win_samp, c + win_samp + 1
        if a < 0 or b > n:
            continue
        f, t, Sxx = spectrogram(sig[a:b], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        dB = 10.0 * np.log10(Sxx + 1e-12)
        if tcen is None:
            tcen = t - WIN_HALF          # the event sits WIN_HALF s into the window
            core_t = np.abs(tcen) < CORE_HALF
            base_t = np.abs(tcen) > BASE_EDGE
            fmask = {bn: (f >= lo) & (f <= hi) for bn, (lo, hi) in BANDS.items()}
        for bn in BANDS:
            band_dB = dB[fmask[bn]].mean(axis=0)
            base = band_dB[base_t].mean()
            per_band[bn].append(band_dB[core_t].mean() - base)
            if bn == want_trace:
                curve = band_dB - base
                trace_acc = curve if trace_acc is None else trace_acc + curve
                trace_k += 1
    out = {bn: np.array(v) for bn, v in per_band.items()}
    return out, (trace_acc / trace_k if trace_k else None), tcen


def detection_controls(prof, all_centers_hr):
    """Every N2 quarter-epoch point at least CTRL_MIN_GAP s from any spindle.

    NOTE (change 4): all such points are used -- not count-matched.
    """
    n2_starts = prof['t_ep_hr'][prof['codes'] == N2_CODE]
    cand = np.array([t0 + frac * 30.0 / 3600.0
                     for t0 in n2_starts for frac in (0.25, 0.5, 0.75)])
    if len(cand):
        d = np.min(np.abs(cand[:, None] - all_centers_hr[None, :]), axis=1) * 3600.0
        cand = cand[d >= CTRL_MIN_GAP]
    return _to_samples(cand)


def detect_session(s, spin_hr, all_centers_hr):
    """Per-channel, per-band spindle statistics for one recording."""
    cen_samp = _to_samples(spin_hr)
    ctrl_samp = detection_controls(s.sleep_profile, all_centers_hr)
    win = int(WIN_HALF * FS)
    per_channel, trig_low, t_axis = {}, {}, None
    for ch in ALL_CHANNELS:
        sig = get_channel(s, ch)
        db_e, trace, tcen = channel_event_metrics(sig, cen_samp, win, want_trace='low_03')
        db_c, _, _ = channel_event_metrics(sig, ctrl_samp, win)
        trig_low[ch] = trace
        t_axis = tcen if t_axis is None else t_axis
        per_channel[ch] = {}
        for bn in BANDS:
            de = db_e[bn][np.isfinite(db_e[bn])]
            dc = db_c[bn][np.isfinite(db_c[bn])]
            # detected = core power above the spindle's own baseline (chance 0.5)
            per_channel[ch][bn] = {
                'det_rate': float(np.mean(de > 0)),
                'mean_db': float(np.mean(de)),
                'median_db': float(np.median(de)),
                'trim_db': float(trim_mean(de, 0.1)),
                'null_rate': float(np.mean(dc > 0)),
                'n_spindles': int(len(de)),
                'db_per_spindle': de,
            }
    return {'n_controls': int(len(ctrl_samp)), 'per_channel': per_channel,
            'trig_low': trig_low, 't_axis': t_axis}


def detection_tables(sessions):
    """(per-recording table, pooled one-row table, per-spindle arrays)."""
    rows = []
    pooled_db = {ch: {bn: [] for bn in BANDS} for ch in ALL_CHANNELS}
    for res in sessions:
        row = {'session': res['label'], 'subject': res['subject'],
               'n_spindles_N2': res['n_spindles_N2'], 'n_controls': res['n_controls']}
        for ch in ALL_CHANNELS:
            pc = res['per_channel'][ch]
            for bn, tag in TAG.items():
                m = pc[bn]
                row[f'{ch}_{tag}_detrate'] = m['det_rate']
                row[f'{ch}_{tag}_meandB'] = m['mean_db']
                row[f'{ch}_{tag}_mediandB'] = m['median_db']
                row[f'{ch}_{tag}_trimdB'] = m['trim_db']
                pooled_db[ch][bn].append(m['db_per_spindle'])
            row[f'{ch}_low_nullrate'] = pc['low_03']['null_rate']
        rows.append(row)
    per_session = pd.DataFrame(rows)

    # pooled: spindle-weighted detection rate; dB statistics over all spindles
    pooled = {'session': 'POOLED', 'subject': 'ALL',
              'n_spindles_N2': int(per_session['n_spindles_N2'].sum())}
    arrays = {}
    for ch in ALL_CHANNELS:
        for bn, tag in TAG.items():
            num = sum(r['per_channel'][ch][bn]['det_rate'] * r['per_channel'][ch][bn]['n_spindles']
                      for r in sessions)
            den = sum(r['per_channel'][ch][bn]['n_spindles'] for r in sessions)
            pooled[f'{ch}_{tag}_detrate'] = num / den
            alldb = np.concatenate(pooled_db[ch][bn])
            arrays[f'db_{ch}_{bn}'] = alldb
            sem = float(alldb.std(ddof=1) / np.sqrt(len(alldb)))
            pooled[f'{ch}_{tag}_meandB'] = float(alldb.mean())
            pooled[f'{ch}_{tag}_mediandB'] = float(np.median(alldb))
            pooled[f'{ch}_{tag}_trimdB'] = float(trim_mean(alldb, 0.1))
            pooled[f'{ch}_{tag}_semdB'] = sem
            # 95% CI of the mean: for sigma, an upper bound on any spindle-locked change
            pooled[f'{ch}_{tag}_ci_lo'] = float(alldb.mean() - 1.96 * sem)
            pooled[f'{ch}_{tag}_ci_hi'] = float(alldb.mean() + 1.96 * sem)
            # share of the summed effect carried by the strongest 5% of events
            top = np.sort(alldb)[::-1][:max(1, len(alldb) // 20)]
            pooled[f'{ch}_{tag}_top5pct_share'] = (float(top.sum() / alldb.sum())
                                                   if alldb.sum() != 0 else np.nan)
        pooled[f'{ch}_low_nullrate'] = float(np.nanmean(per_session[f'{ch}_low_nullrate']))
    arrays['t_axis'] = sessions[0]['t_axis']
    for ch in ALL_CHANNELS:
        arrays[f'trig_low_{ch}'] = np.array([r['trig_low'][ch] for r in sessions])
    return per_session, pd.DataFrame([pooled]), arrays


def channel_summary(per_session, pooled):
    """One row per channel x band: per-recording, per-participant and pooled views.

    CORRECTION (change 1): the per-recording mean is over the 12 recordings only;
    the pooled per-spindle mean is reported beside it, never averaged into it.
    """
    p = pooled.iloc[0]
    rows = []
    for ch in ALL_CHANNELS:
        for tag in TAG.values():
            v = per_session[f'{ch}_{tag}_meandB']
            subj = per_session.groupby('subject')[f'{ch}_{tag}_meandB'].mean()
            rows.append({
                'channel': ch, 'band': tag,
                'recording_mean_dB': float(v.mean()),
                'recording_median_dB': float(v.median()),
                'recording_min_dB': float(v.min()), 'recording_max_dB': float(v.max()),
                'n_recordings_positive': int((v > 0).sum()), 'n_recordings': int(len(v)),
                'participant_mean_dB': float(subj.mean()),
                'n_participants_positive': int((subj > 0).sum()),
                'pooled_mean_dB': float(p[f'{ch}_{tag}_meandB']),
                'pooled_median_dB': float(p[f'{ch}_{tag}_mediandB']),
                'pooled_trim10_dB': float(p[f'{ch}_{tag}_trimdB']),
                'pooled_ci_lo_dB': float(p[f'{ch}_{tag}_ci_lo']),
                'pooled_ci_hi_dB': float(p[f'{ch}_{tag}_ci_hi']),
                'pooled_detrate': float(p[f'{ch}_{tag}_detrate']),
                'pooled_top5pct_share': float(p[f'{ch}_{tag}_top5pct_share']),
                'legacy_13row_mean_dB': float(pd.concat(
                    [v, pd.Series([p[f'{ch}_{tag}_meandB']])]).mean()),
            })
    return pd.DataFrame(rows)


# ── Event-related spectral perturbation (ERSP) and its controls ───────────────

def session_ersp(sig, centers_samp, half):
    """Mean baseline-corrected dB time-frequency map over events for one channel."""
    acc = faxis = tcen = None
    k = 0
    for c in centers_samp:
        a, b = c - half, c + half + 1
        if a < 0 or b > len(sig):
            continue
        f, t, Sxx = spectrogram(sig[a:b], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        fb = f <= FMAX
        dB = 10.0 * np.log10(Sxx[fb] + 1e-12)
        if acc is None:
            acc = np.zeros_like(dB)
            tcen = t - WIN_HALF
            faxis = f[fb]
        acc += dB
        k += 1
    if k == 0:
        return None
    mean_dB = acc / k
    base = mean_dB[:, np.abs(tcen) > BASE_EDGE].mean(axis=1, keepdims=True)
    return {'f': faxis, 't': tcen, 'ersp': mean_dB - base, 'k': k}


def ersp_session(s, spin_hr, rng):
    """Spindle ERSP core spectrum per channel (spindle_ersp.py): table rows."""
    if len(spin_hr) > MAX_EVENTS:
        spin_hr = rng.choice(spin_hr, size=MAX_EVENTS, replace=False)
    cen = _to_samples(spin_hr)
    rows = []
    for ch in ERSP_CHANNELS:
        r = session_ersp(get_channel(s, ch), cen, int(WIN_HALF * FS))
        core = r['ersp'][:, np.abs(r['t']) < CORE_HALF].mean(axis=1)
        sig = (r['f'] >= SIGMA[0]) & (r['f'] <= SIGMA[1])
        rows.append({'session': s.meta['label'], 'channel': ch,
                     'sigma_core_dB': float(core[sig].mean()),
                     'peak_abs_dB': float(np.max(np.abs(core))),
                     'peak_freq_hz': float(r['f'][np.argmax(np.abs(core))])})
    return rows


def _core_spectrum(sig, centers_hr, half, rng):
    if len(centers_hr) > MAX_EVENTS:
        centers_hr = rng.choice(centers_hr, size=MAX_EVENTS, replace=False)
    r = session_ersp(sig, _to_samples(centers_hr), half)
    if r is None:
        return None, None
    return r['f'], r['ersp'][:, np.abs(r['t']) < CORE_HALF].mean(axis=1)


def control_session(s, spin_hr, rng):
    """Core spectra for spindles and the controls (spindle_ersp_control.py).

    randN2: count-matched N2 epoch mid-points (NOTE change 4: not spindle-free).
    spindle_arous / spindle_noarous: spindles with / without a scored arousal
    centred within +/-AROUSAL_NEAR s (the arousal-exclusion sensitivity analysis).
    """
    prof = s.sleep_profile
    n2_starts = prof['t_ep_hr'][prof['codes'] == N2_CODE]
    rand_hr = rng.choice(n2_starts, size=min(len(spin_hr), len(n2_starts)),
                         replace=len(n2_starts) < len(spin_hr)) + 0.5 * 30.0 / 3600.0
    ar = load_arousal_centres(s)
    if ar is not None and len(ar):
        ar_n2 = ar[stage_at(ar, prof) == N2_CODE]
        d = np.min(np.abs(spin_hr[:, None] - ar[None, :]), axis=1) * 3600.0
        spin_ar, spin_no = spin_hr[d <= AROUSAL_NEAR], spin_hr[d > AROUSAL_NEAR]
    else:
        ar_n2, spin_ar, spin_no = np.array([]), np.array([]), spin_hr
    centres = {'spindle': spin_hr, 'randN2': rand_hr, 'arousal': ar_n2,
               'spindle_arous': spin_ar, 'spindle_noarous': spin_no}
    out, f_axis = {}, None
    for ch in CONTROL_CHANNELS:
        sig = get_channel(s, ch)
        for c in CONDITIONS:
            if len(centres[c]) < 15:
                continue
            f, core = _core_spectrum(sig, centres[c], int(WIN_HALF * FS), rng)
            if core is not None:
                out[(ch, c)] = core
                f_axis = f
    counts = {c: len(v) for c, v in centres.items()}
    return out, f_axis, counts


def control_table(stacks, f_axis):
    """Mean 0.1-3 Hz core change per channel x condition, over recordings."""
    lo = (f_axis >= LOW[0]) & (f_axis <= LOW[1])
    rows = []
    for ch in CONTROL_CHANNELS:
        for c in CONDITIONS:
            a = np.array(stacks[(ch, c)])
            rows.append({'channel': ch, 'cond': c,
                         'lowfreq_dB': round(float(np.nanmean(a[:, lo])), 3) if a.size else np.nan,
                         'n_sessions': a.shape[0] if a.ndim == 2 else 0})
    return pd.DataFrame(rows)


# ── Whole cohort ───────────────────────────────────────────────────────────────

def analyse(drop_kcomplex: bool = True) -> dict:
    """Run detection, ERSP and controls over the twelve recordings.

    The two ERSP generators are seeded and consumed in the same order as the
    legacy scripts (11 for the ERSP table, 3 for the controls), so with
    drop_kcomplex=False the legacy outputs are reproduced.
    """
    rng_ersp, rng_ctrl = np.random.default_rng(11), np.random.default_rng(3)
    det, ersp_rows, kc_rows, cnt_rows = [], [], [], []
    stacks = {(ch, c): [] for ch in CONTROL_CHANNELS for c in CONDITIONS}
    f_axis = None
    for s in seclib.iter_sessions():
        sp = load_spindles(s, drop_kcomplex=drop_kcomplex)
        prof = s.sleep_profile
        n2 = stage_at(sp['center_hr'], prof) == N2_CODE
        spin_hr = sp['center_hr'][n2]
        kc_n2 = int((stage_at(sp['kc_center_hr'], prof) == N2_CODE).sum()) \
            if len(sp['kc_center_hr']) else 0
        kc_rows.append({'session': s.meta['label'], 'source_file': sp['source'],
                        'kcomplex_marks_in_file': sp['n_kcomplex_file'],
                        'kcomplex_in_recording': len(sp['kc_center_hr']),
                        'kcomplex_in_N2': kc_n2})
        res = detect_session(s, spin_hr, sp['center_hr'])
        res.update(label=s.meta['label'], subject=s.meta['subject'],
                   n_spindles_N2=int(len(spin_hr)))
        det.append(res)
        ersp_rows += ersp_session(s, spin_hr, rng_ersp)
        out, f, counts = control_session(s, spin_hr, rng_ctrl)
        for key, core in out.items():
            stacks[key].append(core)
        f_axis = f if f is not None else f_axis
        cnt_rows.append({'session': s.meta['label'], **counts})
        print(f"  {s.meta['label']}: {len(spin_hr):5d} N2 spindles, "
              f"{res['n_controls']} controls, {kc_n2} K-complexes in N2")
    per_session, pooled, arrays = detection_tables(det)
    return {'per_session': per_session, 'pooled': pooled, 'arrays': arrays,
            'summary': channel_summary(per_session, pooled),
            'ersp': pd.DataFrame(ersp_rows),
            'control': control_table(stacks, f_axis),
            'control_counts': pd.DataFrame(cnt_rows),
            'kcomplex': pd.DataFrame(kc_rows)}


# ── Fig. 6 ─────────────────────────────────────────────────────────────────────

def make_figure(per_session, pooled, arrays):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.35], wspace=0.3)

    # A: centre-triggered low-band average, mean over recordings (CH +/- SEM)
    ax = fig.add_subplot(gs[0, 0])
    t = arrays['t_axis']
    for ch in CAP_CHANNELS:
        curves = arrays[f'trig_low_{ch}']
        m = curves.mean(axis=0)
        bold = ch == 'CH'
        ax.plot(t, m, color=COLORS[ch], lw=2.4 if bold else 1.3,
                alpha=1.0 if bold else 0.75, label=SHOW[ch])
        if bold:
            se = curves.std(axis=0) / np.sqrt(curves.shape[0])
            ax.fill_between(t, m - se, m + se, color=COLORS[ch], alpha=0.2,
                            label='CH ± SEM (12 recordings)')
    ax.axvline(0, color='k', ls='--', lw=1.0)
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xlim(-6, 6)
    ax.set_xlabel('Time from spindle centre (s)')
    ax.set_ylabel('0.1–3 Hz power (dB re own baseline)')
    ax.set_title('A  Spindle-centred low-band average', loc='left')
    ax.legend(fontsize=9, loc='lower right', frameon=False)

    # B: per-spindle change, low band vs sigma, CH, all spindles pooled
    ax = fig.add_subplot(gs[0, 1])
    db_low, db_sig = arrays['db_CH_low_03'], arrays['db_CH_sigma']
    bins = np.linspace(-6, 6, 61)
    ax.hist(db_sig, bins=bins, density=True, color='#999999', alpha=0.75,
            label=f'σ 11–16 Hz  (mean {db_sig.mean():+.2f}, median '
                  f'{np.median(db_sig):+.2f} dB)')
    ax.hist(db_low, bins=bins, density=True, color=COLORS['CH'], alpha=0.6,
            label=f'0.1–3 Hz  (mean {db_low.mean():+.2f}, median '
                  f'{np.median(db_low):+.2f} dB)')
    for arr, col in ((db_low, COLORS['CH']), (db_sig, '#555555')):
        ax.axvline(arr.mean(), color=col, lw=2)               # solid = mean
        ax.axvline(np.median(arr), color=col, lw=2, ls=':')   # dotted = median
    ax.axvline(0, color='k', ls='--', lw=1.0)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.set_xlabel('Per-spindle power change (dB)\nsolid = mean, dotted = median')
    ax.set_ylabel('Density')
    ax.set_title(f'B  CH, {len(db_low):,} N2 spindles', loc='left')
    ax.legend(fontsize=8.5, loc='upper left', framealpha=1.0, edgecolor='none')
    ax.text(0.98, 0.84, f'{100 * np.mean(db_low > 0):.1f}% of spindles\nabove own '
            f'baseline\n(chance 50%)', transform=ax.transAxes, ha='right', va='top',
            fontsize=9)

    # C: per-recording mean change per channel, medians as ticks
    ax = fig.add_subplot(gs[0, 2])
    labels = per_session['session'].tolist()
    x = np.arange(len(labels))
    w = 0.2
    for i, ch in enumerate(CAP_CHANNELS):
        xs = x + (i - 1.5) * w
        ax.bar(xs, per_session[f'{ch}_low_meandB'], w, color=COLORS[ch],
               label=f'{SHOW[ch]} mean', edgecolor='none')
        ax.scatter(xs, per_session[f'{ch}_low_mediandB'], marker='_', s=40, lw=1.4,
                   color='k', zorder=4)
    ax.scatter([], [], marker='_', s=40, lw=1.4, color='k', label='median')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel('0.1–3 Hz power change (dB)')
    ax.set_title('C  Per-recording low-band change', loc='left')
    ax.legend(fontsize=9, ncol=3, loc='upper right', frameon=False)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    return figures.save(fig, 'fig6_spindles', STAGE)


# ── Numbers ────────────────────────────────────────────────────────────────────

def _rng(vals, d=3):
    return f'{min(vals):.{d}f}–{max(vals):.{d}f}'


def register_numbers(res, legacy):
    nb = Numbers(STAGE)
    S = res['summary'].set_index(['channel', 'band'])
    L = legacy['summary'].set_index(['channel', 'band'])
    ps, lps = res['per_session'], legacy['per_session']
    ctl = res['control'].set_index(['channel', 'cond'])['lowfreq_dB']
    kc = res['kcomplex']
    sec34, cap = '§3.4', '§3.4 Fig. 6'
    kc_note = 'K-complex marks removed (CORRECTION 2); with them, as in the legacy code: '

    # Methods ¶162-163
    nb.add('spin_low_band', '§2.8 ¶163', 'low-frequency band analysed', '0.1–3',
           paper_value='low-frequency (0-3 Hz)', unit='Hz', source='LOW', status='DIFF',
           note='code excludes the f=0 STFT bin (0.78/1.56/2.34 Hz bins used); '
                'text says 0-3 Hz')
    nb.add('spin_sigma_band', '§2.8 ¶163', 'sigma band', '11–16',
           paper_value='sigma-band (11-16 Hz)', unit='Hz', source='SIGMA')
    nb.add('spin_controls', '§2.8 ¶163', 'baseline windows for spindles',
           f"{int(ps.n_controls.sum())} spindle-free N2 points for "
           f"{int(ps.n_spindles_N2.sum())} spindles",
           paper_value='count-matched baseline windows ... spindle-free N2 windows',
           source='detection_controls, control_session', status='DIFF',
           note='detection controls are ALL N2 quarter-epoch points >=3 s from a spindle '
                '(not count-matched); the random-N2 ERSP control is count-matched but '
                'not spindle-free (epoch mid-points, with replacement)')
    nb.add('spin_subject_avg', '§2.8 ¶163', 'subject-level averaging before group average',
           'per-recording and per-spindle averages',
           paper_value='Subject-level averages were calculated before the group average',
           source='channel_summary', status='DIFF',
           note='the §3.4 spindle numbers are per-recording or pooled-spindle means; '
                'participant means are registered as NEW (spin_*_participant)')
    nb.add('spin_arousal_window', '§2.8 ¶163', 'arousal-exclusion window', AROUSAL_NEAR,
           unit='s', source='AROUSAL_NEAR',
           note='spindle excluded if a Classification Arousal centre lies within ±5 s')

    # CORRECTION 2: K-complex marks
    nb.add('spin_kc_in_files', sec34, 'K-complex marks in the spindle files',
           int(kc.kcomplex_marks_in_file.sum()), unit='marks', source='load_spindles',
           note=f"{int((kc.kcomplex_marks_in_file > 0).sum())} of 12 files carry them")
    nb.add('spin_kc_removed_n2', sec34, 'K-complex marks in N2 the legacy code analysed '
           'as spindles', int(kc.kcomplex_in_N2.sum()), unit='marks',
           source='load_spindles', note='CORRECTION 2: now removed')

    # ¶198 counts
    n = ps.n_spindles_N2
    nb.add('spin_n_range', sec34, 'N2 spindles per recording',
           f'{n.min():,}–{n.max():,}', paper_value='351–2,134', source='analyse',
           note=kc_note + f'{lps.n_spindles_N2.min():,}–{lps.n_spindles_N2.max():,}')
    nb.add('spin_n_total', sec34, 'N2 spindles in total', f'{int(n.sum()):,}',
           paper_value='14,305 total', source='analyse',
           note=kc_note + f'{int(lps.n_spindles_N2.sum()):,}. Spindles with a full '
                f"±8 s window (the dB arrays): {len(res['arrays']['db_CH_low_03']):,}")

    # ¶199 low band -- CORRECTION 1: 12-recording mean and pooled kept apart
    def low_rows(key, chans, paper, label):
        def fmt(v):
            return _rng(v) if len(chans) > 1 else float(v[0])
        rec = [S.loc[(c, 'low'), 'recording_mean_dB'] for c in chans]
        pool = [S.loc[(c, 'low'), 'pooled_mean_dB'] for c in chans]
        old = [L.loc[(c, 'low'), 'legacy_13row_mean_dB'] for c in chans]
        nb.add(f'{key}_recmean', sec34, f'{label}: mean of the 12 recording means',
               fmt(rec), paper_value=paper, unit='dB', source='channel_summary',
               note='paper value = legacy mean of 13 rows incl. POOLED ('
                    + ', '.join(f'{SHOW[c]} {v:.3f}' for c, v in zip(chans, old))
                    + '); with K-complexes: '
                    + ', '.join(f"{SHOW[c]} {L.loc[(c, 'low'), 'recording_mean_dB']:.3f}"
                                for c in chans))
        nb.add(f'{key}_pooled', sec34, f'{label}: mean over all N2 spindles (pooled)',
               fmt(pool), paper_value=paper, unit='dB', source='detection_tables',
               note='with K-complexes: '
                    + ', '.join(f"{SHOW[c]} {L.loc[(c, 'low'), 'pooled_mean_dB']:.3f}"
                                for c in chans))
        nb.add(f'{key}_participant', sec34, f'{label}: mean of the 6 participant means',
               fmt([S.loc[(c, 'low'), 'participant_mean_dB'] for c in chans]),
               unit='dB', source='channel_summary',
               note='the averaging Methods ¶163 describes; identical to the recording '
                    'mean because every participant has exactly two recordings')

    low_rows('spin_low_temple', ['CLE', 'CRE'], '0.45–0.49 dB',
             'low-band change, CLE and CRE')
    nb.add('spin_low_diff_recmean', sec34, 'low-band change, CLE−CRE, mean of 12 '
           'recordings', S.loc[('CLE-CRE', 'low'), 'recording_mean_dB'], unit='dB',
           source='channel_summary',
           note='key_numbers.py put CLE−CRE in the quoted 0.45-0.49 range; pooled '
                f"{S.loc[('CLE-CRE', 'low'), 'pooled_mean_dB']:.3f}")
    low_rows('spin_low_ch', ['CH'], '0.55 dB', 'low-band change, CH')

    for ch in CAP_CHANNELS:
        nb.add(f'spin_all12_{ch}', sec34, 'recordings with a positive mean low-band '
               f'change, {SHOW[ch]}', int(S.loc[(ch, 'low'), 'n_recordings_positive']),
               paper_value='all 12 recordings', unit='of 12', source='channel_summary',
               note='with K-complexes: '
                    f"{int(L.loc[(ch, 'low'), 'n_recordings_positive'])}/12; participants "
                    f"positive {int(S.loc[(ch, 'low'), 'n_participants_positive'])}/6")
    nb.add('spin_detrate_ch', sec34, 'CH spindles with core power above own baseline '
           '(pooled)', 100 * S.loc[('CH', 'low'), 'pooled_detrate'], unit='%',
           source='detection_tables',
           note='"did not invariably coincide"; chance 50%. Pooled median '
                f"{S.loc[('CH', 'low'), 'pooled_median_dB']:+.3f} dB vs mean "
                f"{S.loc[('CH', 'low'), 'pooled_mean_dB']:+.3f} dB (tail-driven)")
    for ch in CONTROL_CHANNELS:
        nb.add(f'spin_randN2_{ch}', sec34, f'random-N2 control, {SHOW[ch]}, 0.1-3 Hz '
               'ERSP core change', ctl[(ch, 'randN2')], unit='dB', source='control_table',
               note='"absent at randomly selected N2 time points"; spindles '
                    f"{ctl[(ch, 'spindle')]:.3f} dB in the same ERSP")
        nb.add(f'spin_noarous_{ch}', sec34, 'spindles without an arousal within ±5 s, '
               f'{SHOW[ch]}, 0.1-3 Hz ERSP core change', ctl[(ch, 'spindle_noarous')],
               unit='dB', source='control_table',
               note='"persisted after excluding spindles associated with scored '
                    f"arousals\"; with an arousal {ctl[(ch, 'spindle_arous')]:.3f} dB, "
                    f"arousals themselves {ctl[(ch, 'arousal')]:.3f} dB")

    # ¶200 sigma
    rec_sig = [S.loc[(c, 'sigma'), 'recording_mean_dB'] for c in CAP_CHANNELS]
    pool_sig = [S.loc[(c, 'sigma'), 'pooled_mean_dB'] for c in CAP_CHANNELS]
    bound = [max(abs(S.loc[(c, 'sigma'), 'pooled_ci_lo_dB']),
                 abs(S.loc[(c, 'sigma'), 'pooled_ci_hi_dB'])) for c in CAP_CHANNELS]
    nb.add('spin_sigma_cap_recmean', sec34, 'SEC sigma change, 4 channels, mean of 12 '
           'recordings', _rng(rec_sig), paper_value='0.02–0.03 dB', unit='dB',
           source='channel_summary', note='legacy 13-row mean ' + _rng(
               [L.loc[(c, 'sigma'), 'legacy_13row_mean_dB'] for c in CAP_CHANNELS]))
    nb.add('spin_sigma_cap_pooled', sec34, 'SEC sigma change, 4 channels, pooled '
           'spindles', _rng(pool_sig), paper_value='0.02–0.03 dB', unit='dB',
           source='detection_tables',
           note=f'95% CI bound on |change| {_rng(bound)} dB: quote as a bound')
    nb.add('spin_eeg_sigma_recmean', sec34, 'EEG sigma change, mean of 12 recordings',
           S.loc[('EEG', 'sigma'), 'recording_mean_dB'], paper_value='3.45 dB', unit='dB',
           source='channel_summary',
           note='paper value = legacy 13-row mean '
                f"{L.loc[('EEG', 'sigma'), 'legacy_13row_mean_dB']:.3f}; with K-complexes "
                f"{L.loc[('EEG', 'sigma'), 'recording_mean_dB']:.3f}")
    nb.add('spin_eeg_sigma_pooled', sec34, 'EEG sigma change, pooled spindles',
           S.loc[('EEG', 'sigma'), 'pooled_mean_dB'], paper_value='3.45 dB', unit='dB',
           source='detection_tables',
           note=f"with K-complexes {L.loc[('EEG', 'sigma'), 'pooled_mean_dB']:.3f}")
    nb.add('spin_eeg_sigma_ersp', sec34, 'EEG sigma core change in the spindle ERSP '
           '(<=400 spindles per recording), mean of 12',
           float(res['ersp'].query("channel == 'EEG'").sigma_core_dB.mean()),
           unit='dB', source='ersp_session', note='independent corroboration')

    # ¶202 Fig. 6 caption
    curve = res['arrays']['trig_low_CH'].mean(axis=0)
    nb.add('fig6a_ch_peak', cap, 'Fig. 6A CH curve, peak', float(curve.max()),
           paper_value='largest increase in CH (0.55 dB)', unit='dB', source='make_figure',
           note='the curve averaged over |t|<1 s equals the 12-recording mean '
                f"{S.loc[('CH', 'low'), 'recording_mean_dB']:.3f} dB; CH is the largest "
                'channel either way')
    nb.add('fig6b_ch_low', cap, 'Fig. 6B CH low band, pooled-spindle mean (plotted)',
           S.loc[('CH', 'low'), 'pooled_mean_dB'], paper_value='0.55 dB', unit='dB',
           source='make_figure', note='panel B is the pooled per-spindle distribution')
    nb.add('fig6b_ch_sigma', cap, 'Fig. 6B CH sigma, pooled-spindle mean (plotted)',
           S.loc[('CH', 'sigma'), 'pooled_mean_dB'], paper_value='0.02 dB', unit='dB',
           source='make_figure')
    pos = {c: int(S.loc[(c, 'low'), 'n_recordings_positive']) for c in CAP_CHANNELS}
    nb.add('fig6c_all12', cap, 'Fig. 6C recordings positive on every channel',
           min(pos.values()), paper_value='across all 12 recordings', unit='of 12',
           source='channel_summary', note=', '.join(f'{SHOW[c]} {v}/12'
                                                     for c, v in pos.items()))
    return nb


def kcomplex_effect(res, legacy):
    """Every summary value with and without the K-complex marks, side by side."""
    a = res['summary'].set_index(['channel', 'band'])
    b = legacy['summary'].set_index(['channel', 'band'])
    cols = ['recording_mean_dB', 'pooled_mean_dB', 'pooled_median_dB',
            'pooled_detrate', 'n_recordings_positive']
    out = a[cols].add_suffix('__kcomplex_removed').join(
        b[cols].add_suffix('__with_kcomplex'))
    for c in cols:
        out[f'{c}__change'] = a[c] - b[c]
    ctl = res['control'].merge(legacy['control'], on=['channel', 'cond'],
                               suffixes=('__kcomplex_removed', '__with_kcomplex'))
    return out.reset_index(), ctl


# ── Stage entry ────────────────────────────────────────────────────────────────

def run():
    tab, cache = seclib.TAB_DIR / STAGE, seclib.CACHE_DIR / STAGE
    tab.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    print('  analysis, K-complex marks removed:')
    res = analyse(drop_kcomplex=True)
    print('  legacy mode, K-complex marks kept (to measure the correction):')
    legacy = analyse(drop_kcomplex=False)

    res['per_session'].to_csv(tab / 'detection_per_recording.csv', index=False)
    res['pooled'].to_csv(tab / 'detection_pooled.csv', index=False)
    res['summary'].to_csv(tab / 'detection_summary.csv', index=False)
    res['ersp'].to_csv(tab / 'ersp_sigma_core.csv', index=False)
    res['control'].to_csv(tab / 'ersp_controls.csv', index=False)
    res['control_counts'].to_csv(tab / 'ersp_control_event_counts.csv', index=False)
    res['kcomplex'].to_csv(tab / 'kcomplex_removed.csv', index=False)
    eff, eff_ctl = kcomplex_effect(res, legacy)
    eff.to_csv(tab / 'kcomplex_effect.csv', index=False)
    eff_ctl.to_csv(tab / 'kcomplex_effect_controls.csv', index=False)
    np.savez(cache / 'per_spindle_db.npz', **res['arrays'])

    print('  figure:', make_figure(res['per_session'], res['pooled'], res['arrays']))
    register_numbers(res, legacy).save()


if __name__ == '__main__':
    run()
