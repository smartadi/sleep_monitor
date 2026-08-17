"""
Ridge-detection tuning harness (overlay-only).

Purpose: hone the persistent-ridge DETECTION procedure by eye, on the
spectrogram+ridge overlay alone — no statistics, no stage figures yet.  One
session, one channel, three bands, fast to re-run.

The knobs that control sensitivity live in BANDS below:
  peak_prominence_frac : per-window peak threshold (higher = fewer peaks)
  min_prominence       : GATE — keep a ridge only if its median amplitude is
                         >= this multiple of the local spectral floor (the main
                         "only real ridges" lever)
  min_persistence_sec  : minimum ridge duration (higher = only long ridges)
  max_freq_jump        : continuity tolerance between windows
  smooth_windows       : temporal PSD smoothing (higher = cleaner peaks)

The spectrogram is displayed spectrally whitened (each frequency row minus its
own median dB) so a persistent ridge stands out from the 1/f background instead
of drowning in it.

Run:
  python ridge_overlay_tune.py                 # S1N1, CRE
  python ridge_overlay_tune.py --session S3N1 --channel CH
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import spectrogram, find_peaks
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import STAGE_LABELS, STAGE_ORDER, STAGE_COLORS
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import detect_persistent_ridges

FIG_DIR = Path(__file__).resolve().parents[2] / 'writeup' / 'figures' / 'harmonics' / 'ridges'
FIG_DIR.mkdir(parents=True, exist_ok=True)

BAND_COLOR = {'slow': '#00E5FF', 'resp': '#00E5FF', 'card': '#FF3B30'}
BAND_LABEL = {'slow': 'Slow (0-0.3 Hz)', 'resp': 'Respiratory (0.1-0.5 Hz)',
              'card': 'Cardiac (0.5-3.0 Hz)'}

# Connected stepped hypnogram LADDER; top -> bottom = Wake, N1, N2, N3, REM.
# Stage codes: 0=REM 1=N3 2=N2 3=N1 4=Wake.
_LADDER_Y = {4: 4, 3: 3, 2: 2, 1: 1, 0: 0}
_LADDER_TICKS = [0, 1, 2, 3, 4]
_LADDER_LABELS = ['REM', 'N3', 'N2', 'N1', 'Wake']


def draw_stage_ladder(ax, sp):
    """Hypnogram as a connected stepped ladder (staircase), Wake top / REM bottom."""
    ax.set_yticks(_LADDER_TICKS)
    ax.set_yticklabels(_LADDER_LABELS, fontsize=7)
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

# ── Detection config per band (edit these to tune sensitivity) ──────────────
# Starting point: STRICTER than the first pass — a prominence gate plus longer
# persistence so only the salient, sustained ridges survive.
# Physiological rhythms (respiration, heart rate) are ALWAYS present in a live
# subject, so their ridges should be near-continuous across the night — breaks
# are detection dropouts (mostly motion), not physiology.  For resp/card we
# therefore (a) do NOT motion-mask (the rhythm is still the dominant in-band
# peak through brief motion), (b) let the tracker coast across long gaps
# (max_gap_windows), and (c) stitch fragments across long separations
# (merge_gap_windows), then interpolate — yielding one long continuous ridge.
# The slow band DOES motion-mask (motion corrupts its 4-min windows).
# Respiration and heart rate are each ONE value at any instant, so they are
# tracked as a single constrained trace (strongest in-band peak per window,
# jumps clipped, gaps bridged) — see track_single_ridge.  This removes the
# multi-ridge artifacts (two cardiac bands at the same time; the ridge jumping
# to a subharmonic and "dropping" implausibly).  The slow band keeps the
# multi-ridge detector (several slow oscillations can genuinely coexist).
# Viterbi ridge tracker per rhythm.  `band` restricts the search to where the
# rhythm actually sits (cardiac ~1 Hz+, resp ~0.3-0.5 Hz for these subjects),
# `penalty` (log-amp per Hz) buys smoothness — higher = fewer wiggles, less
# band-hopping; `smooth_win` is a final median-smoothing kernel (windows).
TRACK = {
    'resp': dict(band=(0.25, 0.55), win_sec=30.0, penalty=20.0, smooth_win=13),
    # cardiac capped at 1.45 Hz (~87 bpm): sleeping HR rarely exceeds this, so
    # the path can't jump to brighter spurious/harmonic peaks above the band on
    # weak-cardiac nights (e.g. S2N1); higher penalty further suppresses spikes.
    'card': dict(band=(0.85, 1.45), win_sec=15.0, penalty=24.0, smooth_win=23),
}
BANDS = {
    'slow': dict(min_freq=0.0, max_freq=0.30, win_sec=240.0, step_sec=30.0,
                 welch_seg_sec=120.0, max_freq_jump=0.012, peak_prominence_frac=0.4,
                 smooth_windows=9, min_persistence_sec=1200.0, merge_gap_windows=30,
                 min_prominence=2.5, mask_motion=True),
}


def _sig(session, ch):
    acc = session.cap['acc_mag'].astype(np.float64)
    return remove_acc_artifact(session.cap[ch].astype(np.float64), acc, 0.05, 4.0)


VERBOSE = True


def detect(session, ch, band):
    bp = dict(BANDS[band])
    min_prom = bp.pop('min_prominence', 0.0)
    mask_motion = bp.pop('mask_motion', True)
    acc = session.cap['acc_mag'].astype(np.float64) if mask_motion else None
    rr = detect_persistent_ridges(_sig(session, ch), fs=session.fs, acc_mag=acc,
                                  fill_gaps=True, **bp)
    # GATE: keep only ridges whose median prominence clears the floor multiple
    kept = [r for r in rr['ridges']
            if np.isfinite(r.get('median_prominence', np.nan))
            and r['median_prominence'] >= min_prom]
    n_pre = len(rr['ridges'])
    kept.sort(key=lambda r: -r['duration_sec'])
    if VERBOSE:
        print(f"  [{band}] {n_pre} raw ridges -> {len(kept)} after prom>={min_prom}:")
        for r in kept:
            print(f"      f={r['median_freq']:.3f}Hz  dur={r['duration_sec']/60:5.0f}min  "
                  f"prom(med)={r.get('median_prominence', float('nan')):4.1f}  "
                  f"flat={r.get('flatness', float('nan')):.2f}  "
                  f"cov={r.get('coverage', float('nan')):.2f}")
    rr['ridges'] = kept
    return rr


def _enhance_spec(sig, fs, nperseg_sec, noverlap_sec, fmax, bg_hz=0.4):
    """Spectrogram with the smooth spectral background removed per time column.

    For each time slice we subtract a frequency-median-filtered version of the
    dB spectrum (kernel ~bg_hz wide).  This flattens the 1/f envelope and any
    broadband motion brightening while leaving NARROW peaks intact — so a
    persistent ridge stands out whether or not it is always on (unlike
    per-frequency-median whitening, which cancels always-on ridges)."""
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=int(nperseg_sec * fs),
                            noverlap=int(noverlap_sec * fs))
    m = f <= fmax
    f, Sxx = f[m], Sxx[m]
    db = 10 * np.log10(Sxx + 1e-20)
    dfq = f[1] - f[0] if len(f) > 1 else 1.0
    k = max(3, int(bg_hz / dfq) | 1)                 # odd kernel over ~bg_hz
    bg = median_filter(db, size=(k, 1), mode='nearest')
    return f, t, db - bg


def track_single_ridge(sig, fs, band, win_sec, penalty=6.0, smooth_win=7):
    """One smooth frequency trace for a single-rhythm band (respiration or heart
    rate) via a VITERBI ridge tracker.

    A greedy per-window peak picker hops between competing bands (a fundamental
    and its subharmonic) and jitters.  Viterbi instead finds the globally optimal
    path that maximises total spectral amplitude minus a smoothness penalty on
    frequency change, so it locks onto the single strongest CONTINUOUS band and
    stays smooth.  Emission is each column's log relative amplitude (PSD / column
    median) so a bright band anywhere in the search range is favoured; `penalty`
    (per Hz) sets how strongly jumps are discouraged.  The search `band` keeps it
    on the right rhythm (e.g. cardiac >= ~0.85 Hz, so it can't lock a subharmonic).
    """
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=int(win_sec * fs),
                            noverlap=int(win_sec * fs // 2))
    bm = (f >= band[0]) & (f <= band[1])
    fb, Sb = f[bm], Sxx[bm]
    nfreq, nt = Sb.shape
    if nfreq < 3 or nt < 3:
        return t / 3600.0, np.full(nt, np.nan), 0.0

    col_med = np.nanmedian(Sb, axis=0, keepdims=True) + 1e-30
    E = np.log(Sb / col_med + 1e-6)              # emission: log relative amplitude
    D = np.abs(fb[:, None] - fb[None, :])        # (nfreq, nfreq) freq-change cost

    score = np.full((nfreq, nt), -np.inf)
    back = np.zeros((nfreq, nt), int)
    score[:, 0] = E[:, 0]
    for i in range(1, nt):
        M = score[:, i - 1][None, :] - penalty * D   # (cur, prev)
        back[:, i] = np.argmax(M, axis=1)
        score[:, i] = E[:, i] + M[np.arange(nfreq), back[:, i]]

    path = np.zeros(nt, int)
    path[-1] = int(np.argmax(score[:, -1]))
    for i in range(nt - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    tr = fb[path].astype(float)
    if smooth_win >= 3:
        tr = median_filter(tr, size=smooth_win | 1, mode='nearest')

    # confidence: fraction of windows where the tracked band is clearly peaked
    conf = float(np.mean(Sb[path, np.arange(nt)] > 2.0 * col_med.ravel()))
    return t / 3600.0, tr, conf


def overlay(session, ch):
    sp = session.sleep_profile
    sig = _sig(session, ch)
    fs = session.fs

    # single continuous traces for the two physiological rhythms
    t_resp, resp_tr, resp_p = track_single_ridge(sig, fs, **TRACK['resp'])
    t_card, card_tr, card_p = track_single_ridge(sig, fs, **TRACK['card'])
    slow_rr = detect(session, ch, 'slow')

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.32, 1.0, 0.72], hspace=0.14)

    # stepped sleep-stage ladder
    ax0 = fig.add_subplot(gs[0])
    draw_stage_ladder(ax0, sp)
    n_slow = len(slow_rr['ridges'])
    ax0.set_title(f'{session.label} ({ch}) — resp/cardiac Viterbi traces '
                  f'(conf {resp_p:.0%}/{card_p:.0%}), {n_slow} slow ridges', fontsize=12)

    # 0-3 Hz enhanced spectrogram + single resp/cardiac traces
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    f, t, db = _enhance_spec(sig, fs, 30, 15, 3.0, bg_hz=0.4)
    vmax = np.percentile(db, 99.5)
    ax1.pcolormesh(t / 3600, f, db, shading='gouraud', cmap='magma',
                   vmin=0, vmax=vmax, rasterized=True)
    ax1.plot(t_resp, resp_tr, color=BAND_COLOR['resp'], lw=2.0, alpha=0.95, label='resp rate')
    ax1.plot(t_card, card_tr, color=BAND_COLOR['card'], lw=2.0, alpha=0.95, label='cardiac rate')
    ax1.set_ylim(0, 3.0); ax1.set_ylabel('Frequency (Hz)', fontsize=9)
    ax1.legend(loc='upper right', fontsize=8)

    # slow band 0-0.3 Hz enhanced spectrogram + slow ridges
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    fS, tS, dbS = _enhance_spec(sig, fs, 240, 210, 0.30, bg_hz=0.10)
    vmaxS = np.percentile(dbS, 99.5)
    ax2.pcolormesh(tS / 3600, fS, dbS, shading='gouraud', cmap='viridis',
                   vmin=0, vmax=vmaxS, rasterized=True)
    for r in slow_rr['ridges']:
        ax2.plot(slow_rr['t_hr'], r['freq_trace'], color=BAND_COLOR['slow'],
                 lw=2.0, alpha=0.95)
    ax2.set_ylim(0.0, 0.30); ax2.set_ylabel('Slow (Hz)', fontsize=9)
    ax2.set_xlabel('Time (hr)', fontsize=10)

    out = FIG_DIR / f'ridge_tune_{session.label}_{ch}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {session.label} {ch}: resp {resp_p:.0%} present / cardiac {card_p:.0%} present '
          f'/ {n_slow} slow ridges')
    return out


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='S1N1')
    ap.add_argument('--channel', default='CH')
    ap.add_argument('--all', action='store_true',
                    help='all 12 sessions x all 3 channels')
    args = ap.parse_args()

    if args.all:
        VERBOSE = False
        for i in range(12):
            s = load_session(i)
            s.sleep_profile = load_sleep_profile(s)
            for ch in ('CH', 'CLE', 'CRE'):
                overlay(s, ch)
        return

    found = False
    for i in range(12):
        s = load_session(i)
        if s.label == args.session:
            s.sleep_profile = load_sleep_profile(s)
            overlay(s, args.channel)
            found = True
            break
    if not found:
        print(f'session {args.session} not found')


if __name__ == '__main__':
    main()
