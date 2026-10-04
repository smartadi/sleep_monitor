"""Harmonic-comb episodes: detection per channel, cross-channel events, stage and REM timing, Figs. 8-9.

Manuscript items
    Methods ¶160   comb detector (>= 3 consecutive integer-related rungs >= 5 dB above
                   the local floor; bands >= 1.5 min, present >= 60% of their span;
                   cross-channel merge; +/-30 min stage occupancy; REM before onset /
                   after offset; nearest REM; matched random-NREM null)
    Results ¶219   22 events, nine sessions, six participants; example bands
                   0.15/0.28/0.42/0.68/0.95 Hz; 19 N2 + one each N1/N3/Wake
    Results ¶220   N2 ~0.9 at onset; REM 5.9% vs 1.7% before (x3.5), 0.7% vs 3.1% after;
                   REM ~30-8 min before onset; nearest REM median 30 / 51 min;
                   5 of 6 participants; N1 rises in the 10 min before onset
    Results ¶224   "approximately 10-30 min after REM"
    Fig. 8         S6N1 comb episodes: CH only (as the caption says) and all three channels
    Fig. 9         (a) stage occupancy +/-30 min around onset, (b) REM occupancy
    Tables         ladder_events, ladder_stage_summary, harmonic_ladders_long,
                   ladder_bands (every rung frequency), ladder_onset_occupancy

Ported from
    analysis/slow_wave/harmonic_ladder_overlay.py   constants, _sig, _stage_at,
        _enhance_spec, _rung_db, _comb_count, track_bands, detect_channel,
        draw_stage_ladder, overlay (per-channel summary rows and the figure)
    analysis/slow_wave/ladder_stage_relationship.py merge_spans, rem_intervals,
        rem_gaps, rem_frac, main (events, null, occupancy, per-subject direction)

Changes from the legacy code
    1. The onset-aligned occupancy behind "N2 ~0.9 at onset" (Fig. 9a) is saved as a
       table (ladder_onset_occupancy.csv); it was only ever plotted.
    2. The random-NREM null draws 200 points PER SESSION (event durations cycled),
       not 200 per event as Methods ¶160 says. Computation kept (it reproduces the
       quoted 1.7% / 3.1%); the discrepancy is registered.
    3. Fig. 8's caption says CH, but the legacy figure stacks CH, CLE and CRE. Both
       are produced: fig8_S6N1_CH (what the caption describes) and
       fig8_S6N1_3ch (the legacy layout).
    4. Every detected rung frequency is written out (ladder_bands.csv) and the episode
       best matching the example "0.15, 0.28, 0.42, 0.68, 0.95 Hz" is identified and
       registered; that example had no traceable source.
    5. The 3-minute minimum episode length (MIN_RUN_SEC) and the 1.5-min core /
       4-min gap bridge are not in Methods; registered as notes.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, spectrogram

from seclib import STAGE_COLORS, STAGE_LABELS, STAGE_ORDER, TAB_DIR, iter_sessions
from seclib.config import STAGE_ROW, STAGE_ROW_LABELS, STAGE_ROW_TICKS
from seclib.figures import save
from seclib.numbers import Numbers
from seclib.preprocessing import remove_acc_artifact

STAGE = 's06_harmonic_comb'
REQUIRES = []

CHANNELS = ['CH', 'CLE', 'CRE']
FIG8_SESSION = 'S6N1'
FMAX = 3.0
WIN_SEC = 30.0
STEP_SEC = 15.0

# ── Comb (episode) detector ──────────────────────────────────────────────────
# A window is a comb candidate when some fundamental f0 has >= MIN_RUNGS rungs
# that are peaks >= RUNG_DB above the local spectral background, including
# >= MIN_CONSEC consecutive rungs from f0 (searched over F0_LO..F0_HI).
KMAX = 12
F0_LO, F0_HI, F0_STEP = 0.15, 0.55, 0.006   # fundamental search grid (Hz)
RUNG_DB = 5.0            # a rung = peak >= this many dB above local background
MIN_RUNGS = 3            # a comb needs >= this many rungs ...
MIN_CONSEC = 3           # ... including >= this many CONSECUTIVE from f0
RUNG_TOL = 0.05          # search radius for a rung peak near k*f0 (Hz)
EXTEND_RUNGS = 2         # hysteresis: grow the episode while >= this many rungs persist
MIN_CORE = 6             # candidate runs shorter than 6 windows (1.5 min) are noise
GAP_BRIDGE = 16          # bridge dropouts up to ~4 min between cores
MIN_RUN_SEC = 180.0      # an episode is a sustained block >= 3 min

# ── Rungs: horizontal-band tracker inside each episode ───────────────────────
# Low-threshold peaks per window, linked across time; only bands that persist
# survive, at their own measured frequency (no forced k*f0 spacing).
PEAK_DIST_HZ = 0.06      # minimum separation between distinct bands (Hz)
BAND_DB = 3.0            # per-window peak height (dB above background)
BAND_PROM = 1.5          # ... minimum prominence
BAND_JUMP = 0.035        # max freq step between linked windows (Hz)
BAND_GAP = 3             # bridge only brief dropouts
MIN_BAND_SEC = 90.0      # a band must persist >= 1.5 min ...
BAND_COVER = 0.6         # ... and be present in >= 60% of its span
TSMOOTH = 7              # time-smoothing (windows) before band peak-finding

# ── Stage / REM timing ───────────────────────────────────────────────────────
REM = 0                  # stage code for REM
WIN_MIN = 30.0           # peri-event window for REM search / occupancy
TAUS = np.arange(-WIN_MIN, WIN_MIN + 0.01, 0.5)   # minutes, occupancy curve
N_NULL = 200             # random NREM control points PER SESSION
EXAMPLE_BANDS = [0.15, 0.28, 0.42, 0.68, 0.95]    # ¶219 example event


# ═════════════════════════════════════════════════════════════════════════════
# 1. Signal and spectrogram
# ═════════════════════════════════════════════════════════════════════════════

def _sig(session, ch):
    """Rate-band channel: OLS accelerometer regression, 0.05-4 Hz."""
    acc = session.cap['acc_mag'].astype(np.float64)
    return remove_acc_artifact(session.cap[ch].astype(np.float64), acc, 0.05, 4.0)


def _enhance_spec(sig, fs):
    """Per-column background-subtracted spectrogram: dB above a frequency-median
    filtered background (~0.4 Hz kernel), so narrow rungs stand out at their
    true height whatever the local power level."""
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=int(WIN_SEC * fs),
                            noverlap=int((WIN_SEC - STEP_SEC) * fs))
    m = f <= FMAX
    f, Sxx = f[m], Sxx[m]
    db = 10 * np.log10(Sxx + 1e-20)
    dfq = f[1] - f[0]
    k = max(3, int(0.4 / dfq) | 1)
    return f, t / 3600.0, db - median_filter(db, size=(k, 1), mode='nearest')


def _stage_at(sp, t_hr):
    if sp is None:
        return -1
    idx = np.searchsorted(sp['t_ep_hr'], t_hr, side='right') - 1
    if 0 <= idx < len(sp['codes']):
        return int(sp['codes'][idx])
    return -1


# ═════════════════════════════════════════════════════════════════════════════
# 2. Comb detection per channel (Methods ¶160)
# ═════════════════════════════════════════════════════════════════════════════

def _rung_db(col, freqs, fk):
    """Peak enhancement (dB above background) within RUNG_TOL of fk, or -inf."""
    lo = np.searchsorted(freqs, fk - RUNG_TOL)
    hi = np.searchsorted(freqs, fk + RUNG_TOL) + 1
    if lo >= len(freqs) or hi <= lo:
        return -np.inf
    return float(np.max(col[lo:hi]))


def _comb_count(col, freqs, f0, db):
    """Number of rungs of f0 that are peaks >= db above background, if the comb
    has >= MIN_CONSEC consecutive rungs from the fundamental (else 0)."""
    ks = [k for k in range(1, min(KMAX, int(FMAX / f0)) + 1)
          if _rung_db(col, freqs, k * f0) > db]
    sset = set(ks)
    consec, kk = 0, 1
    while kk in sset:
        consec += 1
        kk += 1
    return len(ks) if consec >= MIN_CONSEC else 0


def track_bands(enh, freqs, lo, hi):
    """Horizontal-band tracker over windows [lo, hi).

    Returns (freq_median, start_idx, end_idx) for every band lasting
    >= MIN_BAND_SEC and present in >= BAND_COVER of its span."""
    df = freqs[1] - freqs[0]
    dist = max(1, int(round(PEAK_DIST_HZ / df)))
    enh = median_filter(enh, size=(1, TSMOOTH | 1), mode='nearest')
    active, done = [], []
    for w in range(lo, hi):
        pk, _ = find_peaks(enh[:, w], height=BAND_DB, prominence=BAND_PROM, distance=dist)
        peaks = list(freqs[pk])
        used = set()
        for b in active:
            best, bd = -1, BAND_JUMP + 9
            for pi, fr in enumerate(peaks):
                if pi in used:
                    continue
                d = abs(fr - b['f'])
                if d < bd:
                    bd, best = d, pi
            if best >= 0 and bd <= BAND_JUMP:
                b['fs'].append(peaks[best]); b['f'] = peaks[best]
                b['end'] = w; b['gap'] = 0; used.add(best)
            else:
                b['gap'] += 1
        keep = []
        for b in active:
            (done if b['gap'] > BAND_GAP else keep).append(b)
        active = keep
        for pi, fr in enumerate(peaks):
            if pi not in used:
                active.append({'f': fr, 'fs': [fr], 'start': w, 'end': w, 'gap': 0})
    done += active
    min_len = int(round(MIN_BAND_SEC / STEP_SEC))
    out = []
    for b in done:
        span = b['end'] - b['start'] + 1
        if span >= min_len and len(b['fs']) / span >= BAND_COVER:
            out.append((float(np.median(b['fs'])), b['start'], b['end']))
    return out


def detect_channel(session, ch):
    """Comb episodes on one channel: (f, t_hr, enh, active[n_win], episodes)."""
    f, t_hr, enh = _enhance_spec(_sig(session, ch), session.fs)
    n_win = enh.shape[1]
    f0_grid = np.arange(F0_LO, F0_HI + 1e-9, F0_STEP)

    # per window: richest bright comb
    best_cnt = np.zeros(n_win, int)
    for i in range(n_win):
        col = enh[:, i]
        best_cnt[i] = max((_comb_count(col, f, fc, RUNG_DB) for fc in f0_grid), default=0)
    is_cand = best_cnt >= MIN_RUNGS

    # drop short candidate runs BEFORE bridging, so scattered noise cannot be
    # stitched into a night-spanning block
    core = np.zeros(n_win, bool)
    i = 0
    while i < n_win:
        if is_cand[i]:
            j = i
            while j < n_win and is_cand[j]:
                j += 1
            if j - i >= MIN_CORE:
                core[i:j] = True
            i = j
        else:
            i += 1
    is_cand = core
    # bridge dropouts between sustained cores
    idx = np.where(is_cand)[0]
    if len(idx) >= 2:
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= GAP_BRIDGE:
                is_cand[a + 1:b] = True

    min_run = int(round(MIN_RUN_SEC / STEP_SEC))
    active = np.zeros(n_win, bool)
    episodes = []
    i = 0
    while i < n_win:
        if is_cand[i]:
            j = i
            while j < n_win and is_cand[j]:
                j += 1
            if j - i >= min_run:
                # time extent: fit the block's f0, then extend while >= EXTEND_RUNGS persist
                cols = [enh[:, w] for w in range(i, j)]
                f0b = max(f0_grid, key=lambda fc: sum(_comb_count(c, f, fc, RUNG_DB) for c in cols))
                lo, hi = i, j
                while lo > 0 and _comb_count(enh[:, lo - 1], f, f0b, RUNG_DB) >= EXTEND_RUNGS:
                    lo -= 1
                while hi < n_win and _comb_count(enh[:, hi], f, f0b, RUNG_DB) >= EXTEND_RUNGS:
                    hi += 1
                active[lo:hi] = True
                episodes.append({'lo': lo, 'hi': hi, 'f0': float(f0b),
                                 'bands': track_bands(enh, f, lo, hi)})
                i = hi
            else:
                i = j
        else:
            i += 1
    return f, t_hr, enh, active, episodes


def channel_summary(session, ch, t_hr, active, episodes):
    """One row per channel with episodes (harmonic_ladders_long.csv)."""
    if not episodes:
        return None
    sp = session.sleep_profile
    stages = [_stage_at(sp, t_hr[i]) for i in np.where(active)[0]]
    stages = [s for s in stages if s >= 0]
    dom = STAGE_LABELS.get(max(set(stages), key=stages.count), '?') if stages else '?'
    n_rungs = [len(ep['bands']) for ep in episodes]
    f0s = [min(fr for fr, _, _ in ep['bands']) for ep in episodes if ep['bands']]
    return dict(session=session.label, subject=session.subject, channel=ch,
                f0_hz=round(float(np.median(f0s)) if f0s else np.nan, 3),
                median_rungs=int(np.median(n_rungs)) if n_rungs else 0,
                active_min=round(active.sum() * STEP_SEC / 60, 1),
                longest_min=round(max(ep['hi'] - ep['lo'] for ep in episodes) * STEP_SEC / 60, 1),
                dominant_stage=dom)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Events, stage and REM timing (Results ¶219-220)
# ═════════════════════════════════════════════════════════════════════════════

def merge_spans(spans, tol_hr=1.0 / 60):
    """Merge channel episodes that overlap (or touch within 1 min) into events."""
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= out[-1][1] + tol_hr:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(m) for m in out]


def rem_intervals(sp):
    """(start_hr, end_hr) of every REM epoch."""
    t = np.asarray(sp['t_ep_hr'], float)
    codes = np.asarray(sp['codes'])
    n = min(len(t), len(codes))
    return [(t[i], t[i + 1]) for i in range(n - 1) if int(codes[i]) == REM]


def rem_gaps(rem_ivs, t0, t1):
    """Minutes from onset back to the nearest REM end, and from offset forward to
    the nearest REM start (nan if none)."""
    before = [t0 - e for (s, e) in rem_ivs if e <= t0]
    after = [s - t1 for (s, e) in rem_ivs if s >= t1]
    db = min(before) * 60 if before else np.nan
    da = min(after) * 60 if after else np.nan
    return db, da


def rem_frac(sp, a_hr, b_hr):
    """Fraction of time in [a, b] hr scored REM (sampled every 0.5 min)."""
    ts = np.arange(a_hr, b_hr, 0.5 / 60)
    if len(ts) == 0:
        return np.nan
    st = np.array([_stage_at(sp, t) for t in ts])
    valid = st >= 0
    return float(np.mean(st[valid] == REM)) if valid.any() else np.nan


def analyse():
    """All twelve nights: channel episodes, merged events, occupancy, REM null."""
    events, rows, long_rows, band_rows = [], [], [], []
    occ = {s: np.zeros(len(TAUS)) for s in STAGE_ORDER}
    n_ev = 0
    rng = np.random.default_rng(7)
    fig8 = None

    for s in iter_sessions():
        sp = s.sleep_profile
        t = np.asarray(sp['t_ep_hr'], float)
        codes = np.asarray(sp['codes'])
        nrem_mask = np.isin(codes, [1, 2, 3])        # N3, N2, N1
        rem_ivs = rem_intervals(sp)

        spans, per_ch = [], {}
        for ch in CHANNELS:
            f, t_hr, enh, active, episodes = detect_channel(s, ch)
            per_ch[ch] = (f, t_hr, enh, active, episodes)
            row = channel_summary(s, ch, t_hr, active, episodes)
            if row:
                long_rows.append(row)
            for k, ep in enumerate(episodes):
                spans.append((t_hr[ep['lo']], t_hr[ep['hi'] - 1]))
                for fr, a, b in ep['bands']:
                    band_rows.append(dict(session=s.label, subject=s.subject, channel=ch,
                                          episode=k, ep_t0_hr=t_hr[ep['lo']],
                                          ep_t1_hr=t_hr[ep['hi'] - 1], ep_f0_hz=ep['f0'],
                                          band_hz=fr, band_t0_hr=t_hr[a], band_t1_hr=t_hr[b],
                                          band_min=(b - a + 1) * STEP_SEC / 60))
        if s.label == FIG8_SESSION:
            fig8 = (s.label, sp, per_ch)
        evs = merge_spans(spans)

        for (t0, t1) in evs:
            n_ev += 1
            ts = np.arange(t0, t1, 0.5 / 60)
            est = np.array([_stage_at(sp, tt) for tt in ts])
            dom = (STAGE_LABELS.get(int(np.bincount(est[est >= 0]).argmax()), '?')
                   if (est >= 0).any() else '?')
            db, da = rem_gaps(rem_ivs, t0, t1)
            pre = rem_frac(sp, t0 - WIN_MIN / 60, t0)
            post = rem_frac(sp, t1, t1 + WIN_MIN / 60)
            events.append(dict(session=s.label, subject=s.subject, t0_hr=round(t0, 3),
                               t1_hr=round(t1, 3), dur_min=round((t1 - t0) * 60, 1),
                               dom_stage=dom,
                               rem_min_before=round(db, 1) if np.isfinite(db) else np.nan,
                               rem_min_after=round(da, 1) if np.isfinite(da) else np.nan,
                               rem_frac_pre=round(pre, 3) if np.isfinite(pre) else np.nan,
                               rem_frac_post=round(post, 3) if np.isfinite(post) else np.nan))
            for ti, tau in enumerate(TAUS):
                st = _stage_at(sp, t0 + tau / 60)
                if st in occ:
                    occ[st][ti] += 1

        # CORRECTION (note only): Methods says 200 matched draws PER EVENT; the
        # null is 200 random NREM time points PER SESSION, event durations cycled.
        # Kept: it is the computation behind the quoted 1.7% / 3.1%.
        nrem_t = t[:len(codes)][nrem_mask[:len(t)]]
        null_pre, null_post = [], []
        if len(nrem_t) and evs:
            durs = [(b - a) for a, b in evs]
            for k in range(N_NULL):
                d = durs[k % len(durs)]
                rt = nrem_t[rng.integers(len(nrem_t))]
                null_pre.append(rem_frac(sp, rt - WIN_MIN / 60, rt))
                null_post.append(rem_frac(sp, rt + d, rt + d + WIN_MIN / 60))

        se = [e for e in events if e['session'] == s.label]
        warnings.filterwarnings('ignore', 'All-NaN slice', RuntimeWarning)  # S6N2: no REM
        if se:
            rows.append(dict(
                session=s.label, subject=s.subject, n_events=len(se),
                dom_stage_mode=pd.Series([e['dom_stage'] for e in se]).mode().iat[0],
                rem_frac_pre=np.nanmean([e['rem_frac_pre'] for e in se]),
                rem_frac_post=np.nanmean([e['rem_frac_post'] for e in se]),
                null_rem_frac_pre=np.nanmean(null_pre) if null_pre else np.nan,
                null_rem_frac_post=np.nanmean(null_post) if null_post else np.nan,
                med_rem_min_before=np.nanmedian([e['rem_min_before'] for e in se]),
                med_rem_min_after=np.nanmedian([e['rem_min_after'] for e in se]),
            ))
        print(f'  {s.label}: {len(evs)} events')

    # CORRECTION: the onset-aligned occupancy (Fig. 9, "N2 ~0.9 at onset") is
    # returned as a table, not only drawn.
    occupancy = pd.DataFrame({'tau_min': TAUS,
                              **{STAGE_LABELS[st]: occ[st] / max(n_ev, 1) for st in STAGE_ORDER}})
    occupancy['n_events'] = n_ev
    return (pd.DataFrame(events), pd.DataFrame(rows), pd.DataFrame(long_rows),
            pd.DataFrame(band_rows), occupancy, fig8)


def subject_rem_side(ev):
    """Per participant: is the nearest REM closer before onset or after offset?"""
    out = []
    for subj, g in ev.groupby('subject'):
        b = np.nanmedian(g['rem_min_before']) if g['rem_min_before'].notna().any() else np.nan
        a = np.nanmedian(g['rem_min_after']) if g['rem_min_after'].notna().any() else np.nan
        who = ('BEFORE' if (np.isfinite(b) and (not np.isfinite(a) or b < a))
               else ('AFTER' if np.isfinite(a) else '-'))
        out.append(dict(subject=subj, median_min_before=b, median_min_after=a, rem_nearer=who))
    return pd.DataFrame(out)


def example_match(bands):
    # CORRECTION: the ¶219 example (0.15, 0.28, 0.42, 0.68, 0.95 Hz) had no source.
    # Find the channel episode whose rung set best covers it: for each example
    # frequency the nearest detected rung; score = the worst of those distances.
    best = None
    for key, g in bands.groupby(['session', 'channel', 'episode']):
        fr = np.sort(g.band_hz.values)
        near = [fr[np.argmin(np.abs(fr - x))] for x in EXAMPLE_BANDS]
        err = max(abs(n - x) for n, x in zip(near, EXAMPLE_BANDS))
        if best is None or err < best[1]:
            best = (key, err, near, fr)
    return best


# ═════════════════════════════════════════════════════════════════════════════
# 4. Figures
# ═════════════════════════════════════════════════════════════════════════════

def draw_stage_ladder(ax, sp):
    """Hypnogram as a connected stepped ladder; top -> bottom Wake, N1, N2, N3, REM."""
    ax.set_yticks(STAGE_ROW_TICKS)
    ax.set_yticklabels(STAGE_ROW_LABELS)
    ax.set_ylim(-0.5, 4.5)
    ax.set_ylabel('Stage')
    t = np.asarray(sp['t_ep_hr'], float)
    codes = np.asarray(sp['codes'])
    n = min(len(t), len(codes))
    ys = np.array([STAGE_ROW.get(int(c), np.nan) for c in codes[:n]], float)
    ax.step(t[:n], ys, where='post', color='#2c3e50', lw=1.3)
    ax.grid(True, axis='y', alpha=0.15)


def fig8(fig8_data, channels, name):
    """Comb spectrogram(s) for S6N1 with the detected rungs, above the stage ladder."""
    label, sp, per_ch = fig8_data
    n = len(channels)
    fig, axes = plt.subplots(n + 1, 1, figsize=(15, 3.2 + 3.4 * n), sharex=True,
                             gridspec_kw={'height_ratios': [0.4 * max(1, n / 2)] + [1.0] * n})
    draw_stage_ladder(axes[0], sp)
    for ax, ch in zip(axes[1:], channels):
        f, t_hr, enh, active, episodes = per_ch[ch]
        ax.pcolormesh(t_hr, f, enh, shading='gouraud', cmap='magma',
                      vmin=0, vmax=np.percentile(enh, 99.5), rasterized=True)
        for ep in episodes:
            for fr, s0, s1 in ep['bands']:
                ax.plot([t_hr[s0], t_hr[s1]], [fr, fr], color='#00E5FF', lw=2.0, alpha=0.95)
        ax.set_ylim(0, FMAX)
        ax.set_ylabel(f'{ch}\nFrequency (Hz)')
    axes[-1].set_xlabel('Time (h)')
    n_ep = {ch: len(per_ch[ch][4]) for ch in channels}
    axes[0].set_title(f'{label}: harmonic-comb episodes (cyan: detected bands); episodes per '
                      'channel ' + ', '.join(f'{c} {k}' for c, k in n_ep.items()))
    return save(fig, name, STAGE)


def fig9(occupancy):
    """(a) stage occupancy around onset, (b) REM occupancy."""
    n_ev = int(occupancy.n_events.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    for st in STAGE_ORDER:
        ax.plot(occupancy.tau_min, occupancy[STAGE_LABELS[st]], color=STAGE_COLORS[st],
                lw=2, label=STAGE_LABELS[st])
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('Minutes relative to event onset')
    ax.set_ylabel('P(stage)')
    ax.set_title(f'Stage occupancy around onset ({n_ev} events)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False)
    ax.grid(alpha=0.15)
    ax.text(-0.12, 1.03, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold')

    ax = axes[1]
    ax.plot(occupancy.tau_min, occupancy['REM'], color=STAGE_COLORS[REM], lw=2.5)
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('Minutes relative to event onset')
    ax.set_ylabel('P(REM)')
    ax.set_title('REM occupancy around onset')
    ax.grid(alpha=0.15)
    ax.text(-0.12, 1.03, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold')
    fig.tight_layout()
    return save(fig, 'fig9_stage_occupancy', STAGE)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Numbers
# ═════════════════════════════════════════════════════════════════════════════

def _eq(computed, expected):
    """Status for paper values written in words ('Twenty-two', 'five of six')."""
    return 'MATCH' if computed == expected else 'DIFF'


def register(nb, ev, sm, occ, side, bands, best, n_ep_fig8):
    # ── Methods ¶160 ──
    nb.add('min_consec', 'Methods ¶160', 'consecutive integer-related rungs required', MIN_CONSEC,
           'at least three consecutive', status=_eq(MIN_CONSEC, 3), source='MIN_CONSEC')
    nb.add('rung_db', 'Methods ¶160', 'rung height above local floor', RUNG_DB,
           'at least 5 dB', unit='dB', source='RUNG_DB')
    nb.add('band_min', 'Methods ¶160', 'minimum band duration', MIN_BAND_SEC / 60,
           'at least 1.5 minutes', unit='min', source='MIN_BAND_SEC')
    nb.add('band_cover', 'Methods ¶160', 'band presence within its span', 100 * BAND_COVER,
           'at least 60%', unit='%', source='BAND_COVER')
    nb.add('occ_window', 'Methods ¶160', 'onset window for stage occupancy', WIN_MIN,
           '±30-minute onset window', unit='min', source='WIN_MIN')
    nb.add('null_draws', 'Methods ¶160', 'random-NREM null draws',
           f'{N_NULL} per session', '200 duration- and session-matched random NREM draws per event',
           status='DIFF', source='analyse',
           note='CORRECTION: 200 draws per SESSION (event durations cycled), not per event. '
                'Text should say per session (or the null be redrawn per event)')
    nb.add('min_episode', 'Methods ¶160', 'minimum episode length (not in Methods)',
           MIN_RUN_SEC / 60, unit='min', source='MIN_RUN_SEC',
           note=f'also: candidate cores >= {MIN_CORE * STEP_SEC / 60:.1f} min, gaps <= '
                f'{GAP_BRIDGE * STEP_SEC / 60:.0f} min bridged, f0 searched '
                f'{F0_LO}-{F0_HI} Hz; none is stated in Methods')

    # ── Results ¶219 ──
    nb.add('n_events', '§3.6 ¶219', 'harmonic-comb events (merged across channels)', len(ev),
           'Twenty-two events', status=_eq(len(ev), 22), source='analyse')
    nb.add('n_sessions', '§3.6 ¶219', 'sessions with an event', ev.session.nunique(),
           'nine sessions', status=_eq(ev.session.nunique(), 9), source='analyse')
    nb.add('n_subjects', '§3.6 ¶219', 'participants with an event', ev.subject.nunique(),
           'all six participants', status=_eq(ev.subject.nunique(), 6), source='analyse')
    vc = ev.dom_stage.value_counts()
    nb.add('n_n2', '§3.6 ¶219', 'events whose dominant stage is N2',
           f"{vc.get('N2', 0)} of {len(ev)}", '19 of 22', source='analyse')
    nb.add('pct_n2', '§3.6 ¶219', 'share of events in N2', 100 * vc.get('N2', 0) / len(ev),
           '(86%)', unit='%', source='analyse')
    nb.add('n_other', '§3.6 ¶219', 'events in N1 / N3 / Wake',
           f"{vc.get('N1', 0)} / {vc.get('N3', 0)} / {vc.get('Wake', 0)}",
           'one occurred in each of N1, N3, and wakefulness',
           status=_eq((vc.get('N1', 0), vc.get('N3', 0), vc.get('Wake', 0)), (1, 1, 1)),
           source='analyse')
    key, err, near, fr = best
    nb.add('example_bands', '§3.6 ¶219', 'example event bands (closest detected episode)',
           ', '.join(f'{x:.2f}' for x in near), '0.15, 0.28, 0.42, 0.68, and 0.95 Hz',
           unit='Hz', source='example_match',
           note=f'best episode {key[0]} {key[1]} #{key[2]}, worst mismatch {err:.3f} Hz; '
                f'its rungs: {", ".join(f"{x:.3f}" for x in fr)}. All rungs in ladder_bands.csv')

    # ── Results ¶220 ──
    p_n2 = float(occ.loc[np.isclose(occ.tau_min, 0), 'N2'].iloc[0])
    late = occ[(occ.tau_min > 0) & (occ.tau_min <= 10)]
    pk = late.loc[late.N2.idxmax()]
    nb.add('n2_onset', '§3.6 ¶220', 'P(N2) at event onset', p_n2, 'approximately 0.9',
           source='analyse',
           note=f'CORRECTION (table): P(N2) is {p_n2:.2f} at onset and peaks at {pk.N2:.2f} '
                f'{pk.tau_min:+.1f} min after it; ~0.9 holds 2.5-4 min after onset, not at '
                'onset. Values: ladder_onset_occupancy.csv')
    pre, post = ev.rem_frac_pre.mean(), ev.rem_frac_post.mean()
    npre, npost = sm.null_rem_frac_pre.mean(), sm.null_rem_frac_post.mean()
    nb.add('rem_pre', '§3.6 ¶220', 'REM share of the 30 min before onset (mean over events)',
           100 * pre, '5.9%', unit='%', source='analyse')
    nb.add('rem_pre_null', '§3.6 ¶220', 'same, random-NREM null (mean over sessions)',
           100 * npre, '1.7%', unit='%', source='analyse',
           note='null averaged over the 9 sessions with events, events averaged over 22')
    nb.add('rem_ratio', '§3.6 ¶220', 'pre-onset REM, events / null', pre / npre,
           'approximately 3.5 times', source='analyse',
           note='3.5 is 5.9/1.7 from rounded values; unrounded ratio shown')
    nb.add('rem_post', '§3.6 ¶220', 'REM share of the 30 min after offset', 100 * post,
           '0.7%', unit='%', source='analyse')
    nb.add('rem_post_null', '§3.6 ¶220', 'same, random-NREM null', 100 * npost, '3.1%',
           unit='%', source='analyse')
    nz = occ[(occ.tau_min < 0) & (occ.REM > 0)].tau_min
    nb.add('rem_window', '§3.6 ¶220', 'span before onset with any REM occupancy',
           f'{-nz.min():.0f}–{-nz.max():.1f}' if len(nz) else 'none',
           'approximately 30–8 min before onset', unit='min', source='analyse')
    nb.add('rem_med_before', '§3.6 ¶220', 'median minutes from preceding REM to onset',
           float(ev.rem_min_before.median()), '30 min', unit='min', source='analyse')
    nb.add('rem_med_after', '§3.6 ¶220', 'median minutes from offset to following REM',
           float(ev.rem_min_after.median()), '51 min', unit='min', source='analyse')
    n_before = int((side.rem_nearer == 'BEFORE').sum())
    nb.add('rem_side', '§3.6 ¶220', 'participants with REM nearer before than after',
           f'{n_before} of {len(side)}', 'five of six',
           status=_eq(n_before, 5), source='subject_rem_side')
    n1_late = occ[(occ.tau_min >= -10) & (occ.tau_min < 0)].N1.mean()
    n1_early = occ[(occ.tau_min >= -30) & (occ.tau_min < -10)].N1.mean()
    nb.add('n1_pre', '§3.6 ¶220', 'mean P(N1) in the 10 min before onset vs 30-10 min before',
           f'{n1_late:.2f} vs {n1_early:.2f}', 'N1 occupancy also increased during the 10 min '
           'preceding onset', status='MATCH' if n1_late > n1_early else 'DIFF',
           source='analyse', note='qualitative claim; values shown')
    q = ev.rem_min_before.dropna().quantile([0.25, 0.75])
    nb.add('rem_10_30', '§3.6 ¶224', 'minutes from preceding REM to onset, interquartile range',
           f'{q.iloc[0]:.0f}–{q.iloc[1]:.0f}', 'approximately 10–30 min after REM', unit='min',
           source='analyse', note=f'{ev.rem_min_before.notna().sum()} of {len(ev)} events have '
                                  'a preceding REM epoch')

    # ── Fig. 8 ──
    s6 = ev[ev.session == FIG8_SESSION]
    nb.add('fig8_events', 'Fig. 8 caption', f'{FIG8_SESSION} events',
           f'{n_ep_fig8} CH episodes; {len(s6)} merged events', 'Two events',
           status=_eq((n_ep_fig8, len(s6)), (2, 2)), source='analyse', note='caption says CH only; legacy figure stacks 3 channels. '
                                  'Both versions written')
    nb.add('fig8_stage', 'Fig. 8 caption', f'{FIG8_SESSION} event stages',
           ', '.join(s6.dom_stage), 'during consolidated N2 sleep',
           status='MATCH' if (s6.dom_stage == 'N2').all() else 'DIFF', source='analyse')


# ═════════════════════════════════════════════════════════════════════════════

def run():
    tab = TAB_DIR / STAGE
    tab.mkdir(parents=True, exist_ok=True)
    nb = Numbers(STAGE)

    ev, sm, long_df, bands, occ, fig8_data = analyse()
    ev.to_csv(tab / 'ladder_events.csv', index=False)
    sm.to_csv(tab / 'ladder_stage_summary.csv', index=False)
    long_df.to_csv(tab / 'harmonic_ladders_long.csv', index=False)
    bands.to_csv(tab / 'ladder_bands.csv', index=False)
    occ.to_csv(tab / 'ladder_onset_occupancy.csv', index=False)
    side = subject_rem_side(ev)
    side.to_csv(tab / 'ladder_rem_side_by_subject.csv', index=False)
    best = example_match(bands)

    # CORRECTION: the caption describes CH only; the legacy figure stacks all three.
    fig8(fig8_data, ['CH'], f'fig8_{FIG8_SESSION}_CH')
    fig8(fig8_data, CHANNELS, f'fig8_{FIG8_SESSION}_3ch')
    fig9(occ)

    register(nb, ev, sm, occ, side, bands, best, len(fig8_data[2]['CH'][4]))
    nb.save()


if __name__ == '__main__':
    run()
