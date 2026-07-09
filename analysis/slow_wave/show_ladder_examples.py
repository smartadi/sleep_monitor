"""
Show real detected-ladder examples for visual tuning.

For a set of example windows, reconstructs the detected persistent-ridge ladder
and plots, per window:
  (left)  raw CH waveform for the 30 s window (see the non-sinusoidal shape)
  (right) fine Welch PSD (0-3 Hz) with:
            - grey ticks = every active persistent ridge (shows ridge density)
            - coloured dots = detected ladder members
            - dashed lines = integer-multiple guides k*f0

Lets you judge by eye whether each is a true harmonic ladder or a coincidental
integer-ratio pickup among many ridges.

Run:
  python show_ladder_examples.py                     # default curated set
  python show_ladder_examples.py --session S3N2 --t_hr 1.254   # one window
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.filters import bandpass
from sleep_monitor.harmonics import detect_persistent_ridges
from harmonic_rigor import best_ladder, DET_CFG, CHANNEL, WIN_SEC, RATIO_TOL, MIN_F0

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'harmonic_rigor'

# curated examples (tag, session, t_hr) — 3 rich ladders, 1 minimal, 2 non-ladders
DEFAULT_EXAMPLES = [
    ('rich, f0~0.25 (N2)',       'S6N2', 4.204),
    ('rich, f0~0.30 (N3)',       'S2N2', 4.871),
    ('rich, f0~0.45 (Wake)',     'S3N2', 7.071),
    ('minimal 3-member (N2)',    'S3N2', 1.254),
    ('multi non-harmonic (N1)',  'S3N2', 2.629),
    ('single tone (N2)',         'S2N2', 2.871),
]

_LABEL_TO_IDX = {m['label']: m['idx'] for m in SESSION_META}


def _members_for_window(active_freqs, active_amps):
    """Reconstruct ladder membership: (f0, members[(k,f,a)], quality)."""
    nm, q, f0, power, decay, m_amps = best_ladder(active_freqs, active_amps)
    if nm < 2 or not np.isfinite(f0):
        return None
    members = []
    for f, a in zip(active_freqs, active_amps):
        k = round(f / f0)
        if k >= 1 and abs(f / f0 - k) < RATIO_TOL:
            members.append((k, f, a))
    members.sort()
    return dict(f0=f0, n=nm, quality=q, decay=decay, members=members)


def _fine_psd(sig_win, fs):
    nperseg = min(len(sig_win), int(20 * fs))
    f, p = welch(sig_win, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                 nfft=4096, scaling='density')
    m = f <= 3.0
    return f[m], p[m]


def plot_examples(examples, out_path):
    # group by session so detection runs once each
    by_session = {}
    for tag, sess, t_hr in examples:
        by_session.setdefault(sess, []).append((tag, t_hr))

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.1 * n),
                             gridspec_kw={'width_ratios': [1, 1.3]})
    if n == 1:
        axes = axes.reshape(1, 2)
    row = 0
    row_order = []

    for sess, items in by_session.items():
        idx = _LABEL_TO_IDX[sess]
        session = load_session(idx)
        session.sleep_profile = load_sleep_profile(session)
        fs = session.fs
        sig = session.cap[CHANNEL].astype(np.float64)
        acc = session.cap['acc_mag'].astype(np.float64)
        rr = detect_persistent_ridges(sig, fs=fs, win_sec=WIN_SEC, step_sec=30.0,
                                      acc_mag=acc, **DET_CFG)
        t_hr_grid = rr['t_hr']

        for tag, t_hr in items:
            wi = int(np.argmin(np.abs(t_hr_grid - t_hr)))
            active = [(r['freq_trace'][wi], r['amp_trace'][wi]) for r in rr['ridges']
                      if np.isfinite(r['freq_trace'][wi])]
            afreqs = np.array([a[0] for a in active])
            aamps = np.array([a[1] for a in active])
            lad = _members_for_window(afreqs, aamps) if len(active) >= 2 else None

            # raw window
            s0 = int(round((t_hr_grid[wi] * 3600 - WIN_SEC / 2) * fs))
            s0 = max(0, s0)
            win = sig[s0:s0 + int(WIN_SEC * fs)]
            win_bp = bandpass(win, 0.1, 3.0, fs)
            tt = np.arange(len(win_bp)) / fs

            axl = axes[row, 0]
            axl.plot(tt, win_bp, color='#2C3E50', lw=0.8)
            axl.set_title(f'{tag}  —  {sess} @ {t_hr:.2f} hr', fontsize=10, loc='left')
            axl.set_ylabel('CH (0.1-3 Hz)')
            if row == n - 1:
                axl.set_xlabel('time (s)')

            # PSD
            f, p = _fine_psd(win, fs)
            axr = axes[row, 1]
            axr.semilogy(f, p + 1e-30, color='#7F8C8D', lw=1.0)
            # all active ridges as ticks
            ymin, ymax = axr.get_ylim()
            for fr in afreqs:
                if fr <= 3.0:
                    axr.axvline(fr, color='#BDC3C7', lw=0.6, alpha=0.7)
            # ladder overlay
            if lad is not None:
                f0 = lad['f0']
                for k in range(1, 9):
                    fk = k * f0
                    if fk <= 3.0:
                        axr.axvline(fk, color='#E67E22', ls='--', lw=1.0, alpha=0.6)
                for k, fr, a in lad['members']:
                    if fr <= 3.0:
                        pk = np.interp(fr, f, p)
                        axr.plot(fr, pk, 'o', color='#C0392B', ms=8)
                        axr.annotate(f'{k}f₀', (fr, pk), textcoords='offset points',
                                     xytext=(0, 6), fontsize=8, color='#C0392B',
                                     ha='center')
                axr.set_title(
                    f"f₀={f0:.3f} Hz | members={lad['n']} | quality={lad['quality']:.2f} "
                    f"| decay={lad['decay']:.2f} | {len(active)} active ridges",
                    fontsize=9, loc='left')
            else:
                axr.set_title(f'no ladder | {len(active)} active ridges',
                              fontsize=9, loc='left')
            axr.set_xlim(0, 3.0)
            axr.set_ylabel('PSD')
            if row == n - 1:
                axr.set_xlabel('frequency (Hz)')
            row_order.append((tag, sess, t_hr))
            row += 1

    fig.suptitle('Detected harmonic-ladder examples (CH channel) — orange dashed = k·f₀, '
                 'red dots = detected members, grey = all active ridges',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved -> {out_path}  ({row} examples)')
    return row_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=str, default=None)
    ap.add_argument('--t_hr', type=float, default=None)
    args = ap.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.session and args.t_hr is not None:
        examples = [(f'{args.session} @ {args.t_hr:.2f}hr', args.session, args.t_hr)]
        out = REPORT_DIR / f'ladder_example_{args.session}_{args.t_hr:.2f}.png'
    else:
        examples = DEFAULT_EXAMPLES
        out = REPORT_DIR / 'ladder_examples.png'
    plot_examples(examples, out)


if __name__ == '__main__':
    main()
