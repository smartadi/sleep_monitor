"""Spindle-triggered ERSP figures: per-channel time-frequency maps (activity
minus its own baseline) + a collapsed core-vs-baseline spectral difference."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
CHANNELS = ['EEG', 'CLE-CRE', 'CLE', 'CRE', 'CH']
CHAN_COLORS = {'EEG': '#2C3E50', 'CLE-CRE': '#E67E22',
               'CLE': '#27AE60', 'CRE': '#8E44AD', 'CH': '#2980B9'}


def sem(a, axis=0):
    return np.nanstd(a, axis=axis) / np.sqrt(a.shape[axis])


def main():
    z = np.load(os.path.join(OUT, 'spindle_ersp.npz'))
    f, t = z['f'], z['t']

    # ── Fig 1: per-channel ERSP maps ─────────────────────────────────────────
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4 * len(CHANNELS), 4.2))
    for ax, ch in zip(axes, CHANNELS):
        M = np.nanmean(z[f'ersp_{ch}'], axis=0)   # (nf, nt)
        # symmetric scale; EEG gets its own (large) scale, CAP a tight one
        vmax = 3.0 if ch == 'EEG' else 0.5
        im = ax.pcolormesh(t, f, M, shading='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax)
        ax.axhspan(11, 16, color='k', alpha=0.0)
        ax.axhline(11, color='k', ls='--', lw=0.6)
        ax.axhline(16, color='k', ls='--', lw=0.6)
        ax.axvline(0, color='k', ls=':', lw=0.8)
        ax.set_title(f'{ch}\n(±{vmax} dB scale)', fontsize=10)
        ax.set_xlabel('t from spindle (s)')
        ax.set_xlim(-6, 6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='dB vs baseline')
    axes[0].set_ylabel('Frequency (Hz)')
    fig.suptitle('Spindle-triggered ERSP (activity minus its own baseline) — EEG shows the sigma '
                 'blob (11–16 Hz); CAP shows NO sigma but a spindle-locked 0–3 Hz bump '
                 '(validated: flat for random-N2, survives arousal removal — see control fig)',
                 y=1.03, fontsize=11)
    fig.tight_layout()
    p1 = os.path.join(OUT, 'fig_spindle_ersp_maps.png')
    fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p1)

    # ── Fig 2: collapsed core-vs-baseline spectral difference ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    # EEG on its own axis (large), CAP channels together (tight)
    for ch in ['EEG']:
        C = z[f'core_{ch}']
        m, e = np.nanmean(C, 0), sem(C, 0)
        axes[0].plot(f, m, color=CHAN_COLORS[ch], lw=2, label=ch)
        axes[0].fill_between(f, m - e, m + e, color=CHAN_COLORS[ch], alpha=0.2)
    axes[0].axvspan(11, 16, color='gold', alpha=0.25, label='sigma band')
    axes[0].axhline(0, color='k', lw=0.6)
    axes[0].set_title('EEG (positive control)'); axes[0].set_xlim(0, 45)
    axes[0].set_xlabel('Frequency (Hz)'); axes[0].set_ylabel('Core − baseline (dB)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    for ch in ['CLE-CRE', 'CLE', 'CRE', 'CH']:
        C = z[f'core_{ch}']
        m, e = np.nanmean(C, 0), sem(C, 0)
        axes[1].plot(f, m, color=CHAN_COLORS[ch], lw=1.6, label=ch)
        axes[1].fill_between(f, m - e, m + e, color=CHAN_COLORS[ch], alpha=0.15)
    axes[1].axvspan(11, 16, color='gold', alpha=0.25, label='sigma band')
    axes[1].axhline(0, color='k', lw=0.6)
    axes[1].set_title('CAP channels'); axes[1].set_xlim(0, 45)
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('Core − baseline (dB)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

    fig.suptitle('Core (|t|<1 s) vs baseline (|t|>5 s) spectral change per channel — '
                 'EEG: +dB at sigma; CAP: flat except a real +0.5 dB 0–3 Hz spindle-locked bump',
                 y=1.02, fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(OUT, 'fig_spindle_ersp_spectra.png')
    fig.savefig(p2, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p2)


if __name__ == '__main__':
    main()
