"""
Harmonic-LADDER detection (distinct from the resp/cardiac ridges).

Reframing (per user): the ladder is NOT the respiratory or cardiac harmonic
series — those are reported by the ridge tracker.  The ladder is a separate,
temporally-STABLE phenomenon: a stack of FLAT, sustained spectral rungs at
integer-multiple frequencies (a candidate slow-wave-activity signature).  So it
must be built from flat persistent ridges, not a per-window spectral test, and
it is a single concept (not split resp/cardiac).

Method (per channel, kept separate — averaging channels diluted single-electrode
ladders):
  1. Track FLAT persistent ridges over 0.1-3 Hz with the shared ridge tracker
     (sleep_monitor.harmonics.detect_persistent_ridges: temporal continuity,
     gap-fill, fragment-merge, flatness metric).
  2. A window is a LADDER window when >= MIN_RUNGS of those concurrent ridges are
     FLAT (flatness >= FLAT_MIN) and their frequencies form an integer series
     f0, 2f0, 3f0, ... (sleep_monitor.harmonics.label_harmonic_ladder_windows).
  3. Keep only SUSTAINED contiguous ladder blocks (>= MIN_RUN_SEC).  The rungs
     drawn are the flat ridges themselves, so a ladder reads as flat horizontal
     lines, not a flickering per-window feature.

One figure per session; a stacked spectrogram panel per channel (CH/CLE/CRE)
with the flat ladder rungs overlaid + the stepped sleep-stage ladder.

Run:  python harmonic_ladder_overlay.py --all
Out:  writeup/figures/harmonics/ladders/ladder_<SESSION>.png
      reports/slow_wave/revamp/harmonic_ladders_long.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, find_peaks
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import STAGE_COLORS, STAGE_LABELS
from sleep_monitor.preprocessing import remove_acc_artifact

FIG_DIR = Path(__file__).resolve().parents[2] / 'writeup' / 'figures' / 'harmonics' / 'ladders'
REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'revamp'
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
FMAX = 3.0
WIN_SEC = 30.0
STEP_SEC = 15.0

# The ladder is a RICH, FLAT harmonic comb episode (distinct from the continuous
# resp/cardiac ridges).  Detected by a Harmonic-Product-Spectrum search per
# window: find the f0 whose integer multiples light up most above the local
# background; keep windows with a rich CONSECUTIVE comb; smooth f0 over time so
# the rungs are flat; keep only sustained blocks.
KMAX = 12
# The ladder is a rich, flat harmonic-comb EPISODE.  The discriminative feature
# (established by a dB sweep against the visible episodes) is the count of BRIGHT
# rungs: a rung is a peak >= RUNG_DB above the local spectral background.  The
# background subtraction is what makes RUNG_DB adaptive (dB above the local
# floor, not an absolute level); at 5 dB the episode combs (median ~5 rungs)
# separate cleanly from baseline noise (median 0).
F0_LO, F0_HI, F0_STEP = 0.15, 0.55, 0.006   # fundamental search grid (Hz)
RUNG_DB = 5.0            # a rung = peak >= this many dB above local background
RUNG_MIN_DB = 2.5        # draw/extend rungs down to this (episode edges are dimmer)
MIN_RUNGS = 3            # a comb needs >= this many rungs ...
MIN_CONSEC = 3           # ... including >= this many CONSECUTIVE from f0
EXTEND_RUNGS = 2         # hysteresis: grow the episode while >= this many rungs persist
MIN_CORE = 6             # a candidate run must be this many windows (~1.5 min)
                         # contiguous to count as a real core (kills noise blips)
GAP_BRIDGE = 16          # bridge dropouts up to ~4 min (heal short breaks)
MIN_RUN_SEC = 180.0      # a ladder is a sustained block >= 3 min
# Once an episode's time-extent is fixed, its RUNGS are the ACTUAL significant
# flat bands (peaks of the time-averaged episode spectrum), NOT forced onto a
# k*f0 grid — so quasi-harmonic / variable-spacing rungs land at their true
# frequencies and the rung count is determined automatically.
PEAK_DIST_HZ = 0.06      # minimum separation between distinct bands (Hz)
# RUNGS via a sensitive HORIZONTAL-BAND TRACKER (not episode-averaged peak-
# picking, which dilutes a band present in only part of the episode).  Per window
# we take LOW-threshold spectral peaks (sensitive) and LINK them across time;
# only bands that PERSIST survive (temporal stability, which rejects noise).  A
# band present for part of the episode shows as a shorter flat segment instead of
# vanishing.
BAND_DB = 3.0            # per-window peak height (dB above background)
BAND_PROM = 1.5          # ... minimum prominence (a real rung is a clear local peak)
BAND_JUMP = 0.035        # max freq step between linked windows (Hz) — keeps bands flat
BAND_GAP = 3             # bridge only brief dropouts (real rungs are near-continuous)
MIN_BAND_SEC = 90.0      # a band must persist >= 1.5 min (temporal-stability gate)
BAND_COVER = 0.6         # ... and be actually PRESENT in >= this fraction of its span
                         # (a real flat rung is consistent; noise flickers)
TSMOOTH = 7              # time-smoothing (windows) of the spectrogram before band
                         # peak-finding — suppresses transient noise, keeps rungs

# Sleep-stage ladder, top -> bottom = Wake, N1, N2, N3, REM (strict NREM depth,
# REM at the bottom).  Stage codes: 0=REM 1=N3 2=N2 3=N1 4=Wake.
_LADDER_Y = {4: 4, 3: 3, 2: 2, 1: 1, 0: 0}


def draw_stage_ladder(ax, sp):
    """Hypnogram as a connected stepped LADDER (staircase), not colour-coded
    dashes.  Top->bottom = Wake, N1, N2, N3, REM."""
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'], fontsize=7)
    ax.set_ylim(-0.5, 4.5)
    ax.set_ylabel('Stage', fontsize=9)
    if sp is None:
        return
    t = np.asarray(sp['t_ep_hr'], float)
    codes = np.asarray(sp['codes'])
    n = min(len(t), len(codes))
    ys = np.array([_LADDER_Y.get(int(c), np.nan) for c in codes[:n]], float)
    ax.step(t[:n], ys, where='post', color='#2c3e50', lw=1.3)
    ax.grid(True, axis='y', alpha=0.15)


def _sig(session, ch):
    acc = session.cap['acc_mag'].astype(np.float64)
    return remove_acc_artifact(session.cap[ch].astype(np.float64), acc, 0.05, 4.0)


def _stage_at(sp, t_hr):
    if sp is None:
        return -1
    idx = np.searchsorted(sp['t_ep_hr'], t_hr, side='right') - 1
    if 0 <= idx < len(sp['codes']):
        return int(sp['codes'][idx])
    return -1


def _enhance_spec(sig, fs):
    """Display/analysis spectrogram: per-column background-subtracted (dB above a
    frequency-smoothed background) so narrow rungs stand out at their true height
    regardless of local power level."""
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=int(WIN_SEC * fs),
                            noverlap=int((WIN_SEC - STEP_SEC) * fs))
    m = f <= FMAX
    f, Sxx = f[m], Sxx[m]
    db = 10 * np.log10(Sxx + 1e-20)
    dfq = f[1] - f[0]
    k = max(3, int(0.4 / dfq) | 1)
    return f, t / 3600.0, db - median_filter(db, size=(k, 1), mode='nearest')


RUNG_TOL = 0.05          # search radius for a rung peak near k*f0 (Hz)


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
    """Sensitive horizontal-band tracker over windows [lo, hi).

    Per window: low-threshold spectral peaks (catches faint bands).  Peaks are
    linked greedily across time within BAND_JUMP Hz; a band ends after a gap of
    > BAND_GAP windows.  Only bands lasting >= MIN_BAND_SEC survive.  Returns a
    list of (freq_median, start_idx, end_idx) — flat horizontal segments at their
    true frequencies (band present part of the episode -> shorter segment)."""
    df = freqs[1] - freqs[0]
    dist = max(1, int(round(PEAK_DIST_HZ / df)))
    # time-smooth the spectrogram first: a persistent rung reinforces, transient
    # noise peaks average down — so peak-finding returns mostly REAL flat bands
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
    """Bright-rung harmonic-comb ladder on one channel.  Returns
    (f, t_hr, enh, f0_final, conf[KMAX+1,n_win] bool, active[n_win] bool)."""
    f, t_hr, enh = _enhance_spec(_sig(session, ch), session.fs)
    n_win = enh.shape[1]
    f0_grid = np.arange(F0_LO, F0_HI + 1e-9, F0_STEP)

    # per window: richest BRIGHT comb (count of rungs >= RUNG_DB)
    best_cnt = np.zeros(n_win, int)
    for i in range(n_win):
        col = enh[:, i]
        best_cnt[i] = max((_comb_count(col, f, fc, RUNG_DB) for fc in f0_grid), default=0)
    is_cand = best_cnt >= MIN_RUNGS

    # remove short/isolated candidate runs (baseline noise blips) BEFORE bridging,
    # so scattered noise can't be stitched into a night-spanning block; only
    # genuine sustained cores survive.
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
    # now bridge dropouts BETWEEN sustained cores so a short break heals
    idx = np.where(is_cand)[0]
    if len(idx) >= 2:
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= GAP_BRIDGE:
                is_cand[a + 1:b] = True

    min_run = int(round(MIN_RUN_SEC / STEP_SEC))
    df = f[1] - f[0]
    dist = max(1, int(round(PEAK_DIST_HZ / df)))
    active = np.zeros(n_win, bool)
    episodes = []                    # each: {lo, hi, rungs (actual freqs)}
    i = 0
    while i < n_win:
        if is_cand[i]:
            j = i
            while j < n_win and is_cand[j]:
                j += 1
            if j - i >= min_run:
                # time-extent via the harmonic comb (worked well): fit f0, extend
                cols = [enh[:, w] for w in range(i, j)]
                f0b = max(f0_grid, key=lambda fc: sum(_comb_count(c, f, fc, RUNG_DB) for c in cols))
                lo, hi = i, j
                while lo > 0 and _comb_count(enh[:, lo - 1], f, f0b, RUNG_DB) >= EXTEND_RUNGS:
                    lo -= 1
                while hi < n_win and _comb_count(enh[:, hi], f, f0b, RUNG_DB) >= EXTEND_RUNGS:
                    hi += 1
                active[lo:hi] = True
                # RUNGS = sensitive horizontal-band tracker over the episode span:
                # catches faint and partial-duration bands (drawn as flat segments)
                # that the episode-averaged peak-pick used to dilute away.
                bands = track_bands(enh, f, lo, hi)
                episodes.append({'lo': lo, 'hi': hi, 'bands': bands})
                i = hi
            else:
                i = j
        else:
            i += 1
    return f, t_hr, enh, active, episodes


def overlay(session):
    sp = session.sleep_profile
    fig, axes = plt.subplots(len(CHANNELS) + 1, 1, figsize=(16, 11),
                             gridspec_kw={'height_ratios': [0.4] + [1.0] * len(CHANNELS)},
                             sharex=True)
    draw_stage_ladder(axes[0], sp)
    rows = []
    summary = []
    for ax, ch in zip(axes[1:], CHANNELS):
        f, t_hr, enh, active, episodes = detect_channel(session, ch)
        ax.pcolormesh(t_hr, f, enh, shading='gouraud', cmap='magma',
                      vmin=0, vmax=np.percentile(enh, 99.5), rasterized=True)
        # draw each detected band as a flat segment at its true frequency over the
        # part of the episode where it actually persists
        for ep in episodes:
            for fr, s0, s1 in ep['bands']:
                ax.plot([t_hr[s0], t_hr[s1]], [fr, fr], color='#00E5FF', lw=2.0, alpha=0.95)
        ax.set_ylim(0, FMAX)
        ax.set_ylabel(f'{ch}\nFreq (Hz)', fontsize=9)
        active_min = active.sum() * STEP_SEC / 60
        if episodes:
            longest = max(ep['hi'] - ep['lo'] for ep in episodes)
            longest_min = longest * STEP_SEC / 60
            stages = [_stage_at(sp, t_hr[i]) for i in np.where(active)[0]]
            stages = [s for s in stages if s >= 0]
            dom = STAGE_LABELS.get(max(set(stages), key=stages.count), '?') if stages else '?'
            n_rungs = [len(ep['bands']) for ep in episodes]
            med_n = int(np.median(n_rungs)) if n_rungs else 0
            f0s = [min(fr for fr, _, _ in ep['bands']) for ep in episodes if ep['bands']]
            med_f0 = float(np.median(f0s)) if f0s else np.nan
            summary.append(f'{ch}: {active_min:.0f}min ×{med_n} [{dom}]')
            rows.append(dict(session=session.label, subject=session.subject, channel=ch,
                             f0_hz=round(med_f0, 3), median_rungs=med_n,
                             active_min=round(active_min, 1),
                             longest_min=round(longest_min, 1), dominant_stage=dom))
        else:
            summary.append(f'{ch}: none')
    axes[-1].set_xlabel('Time (hr)', fontsize=10)
    axes[0].set_title(f'{session.label} — flat harmonic ladders  |  ' + '   '.join(summary),
                      fontsize=11)
    out = FIG_DIR / f'ladder_{session.label}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  {session.label}: " + ' | '.join(summary))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='S6N2')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    all_rows = []
    for i in range(12):
        s = load_session(i)
        if args.all or s.label == args.session:
            s.sleep_profile = load_sleep_profile(s)
            all_rows += overlay(s)
            if not args.all:
                break
    if args.all:
        pd.DataFrame(all_rows).to_csv(REPORT_DIR / 'harmonic_ladders_long.csv', index=False)
        print(f"\nwrote {REPORT_DIR / 'harmonic_ladders_long.csv'}")


if __name__ == '__main__':
    main()
