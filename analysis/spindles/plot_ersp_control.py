"""Plot the ERSP-control core spectra: spindle vs random-N2 null vs arousal,
and spindle split by arousal co-occurrence. Answers whether the CAP 0-3 Hz
bump is a real spindle signature or an artifact / arousal confound."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
CHANNELS = ['CH', 'CLE-CRE']
COND_STYLE = {
    'spindle':         ('#C0392B', '-',  2.2, 'spindles (all)'),
    'randN2':          ('gray',    '--', 1.5, 'random N2 (artifact null)'),
    'arousal':         ('#2980B9', '-',  1.8, 'scored arousals (N2)'),
    'spindle_arous':   ('#E67E22', '-',  1.8, 'spindles WITH arousal ±5s'),
    'spindle_noarous': ('#27AE60', '-',  1.8, 'spindles WITHOUT arousal'),
}


def sem(a):
    return np.nanstd(a, 0) / np.sqrt(a.shape[0])


def main():
    z = np.load(os.path.join(OUT, 'spindle_ersp_control.npz'))
    f = z['f']
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(7 * len(CHANNELS), 4.6), squeeze=False)
    axes = axes[0]
    for ax, ch in zip(axes, CHANNELS):
        for cond, (color, ls, lw, lab) in COND_STYLE.items():
            a = z[f'{ch}__{cond}']
            if a.ndim != 2 or a.shape[0] == 0:
                continue
            m, e = np.nanmean(a, 0), sem(a)
            ax.plot(f, m, color=color, ls=ls, lw=lw, label=f'{lab} (n={a.shape[0]})')
            if cond in ('spindle', 'randN2', 'spindle_noarous'):
                ax.fill_between(f, m - e, m + e, color=color, alpha=0.15)
        ax.axvspan(0.5, 3.0, color='gold', alpha=0.18, label='0.5–3 Hz')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlim(0, 20); ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Core (|t|<1 s) − baseline (dB)')
        ax.set_title(f'{ch}')
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle('ERSP control — is the CAP 0–3 Hz spindle bump real, an artifact, or an arousal confound?',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig_spindle_ersp_control.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p)


if __name__ == '__main__':
    main()
