"""
Simple frequency-vs-dB (power spectral density) plot of the CAP signal, per session.

Standard frequency-analysis view: x = frequency (Hz), y = power in dB (10*log10 of
the full-night Welch PSD). One panel per session (4x3 grid) plus one standalone PNG
per session. Reuses the cached full-night PSDs in signal_characterization_cache.pkl
(no raw reload).

Channels: CLE-CRE differential (canonical, bold) with CH / CLE / CRE faint for context.
Respiratory (0.1-0.5 Hz) and cardiac (0.5-3.0 Hz) bands shaded.

Outputs:
  writeup/figures/signal_validation/psd_db_all_sessions.png     (4x3 grid)
  writeup/figures/signal_validation/psd_db/psd_db_<label>.png   (per session)
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'signal_characterization_cache.pkl'
OUT_GRID = HERE / 'psd_db_all_sessions.png'
OUT_DIR = HERE / 'psd_db'
OUT_DIR.mkdir(exist_ok=True)

FMAX = 20.0          # Hz shown (all CAP physiology + rolloff to the noise floor)
RESP = (0.1, 0.5)
CARD = (0.5, 3.0)
CH_STYLE = {
    'CLE-CRE': dict(color='#1f3a93', lw=1.6, alpha=0.95, zorder=5),
    'CH':      dict(color='#2980B9', lw=0.8, alpha=0.35, zorder=2),
    'CLE':     dict(color='#27AE60', lw=0.8, alpha=0.35, zorder=2),
    'CRE':     dict(color='#8E44AD', lw=0.8, alpha=0.35, zorder=2),
}


def to_db(psd):
    return 10.0 * np.log10(np.asarray(psd) + 1e-30)


def draw_panel(ax, entry, title):
    for ch in ('CH', 'CLE', 'CRE', 'CLE-CRE'):   # differential last = on top
        f, psd = entry[ch]
        f = np.asarray(f)
        m = f <= FMAX
        ax.plot(f[m], to_db(psd)[m], label=ch, **CH_STYLE[ch])
    ax.axvspan(*RESP, color='#4C72B0', alpha=0.10, lw=0)
    ax.axvspan(*CARD, color='#DD8452', alpha=0.10, lw=0)
    ax.set_xlim(0, FMAX)
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=10)


def main():
    d = pickle.load(open(CACHE, 'rb'))
    labels = d['labels']
    psd_by = d['psd_by_session']

    # ── grid ──
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    for i, lab in enumerate(labels):
        ax = axes[i]
        draw_panel(ax, psd_by[lab], lab)
        if i % 3 == 0:
            ax.set_ylabel('Power (dB)', fontsize=9)
        if i >= 9:
            ax.set_xlabel('Frequency (Hz)', fontsize=9)
    axes[0].legend(fontsize=7, loc='upper right', ncol=2)
    fig.suptitle('CAP signal power spectrum (full-night Welch PSD, dB) — per session',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_GRID, dpi=170)
    plt.close(fig)
    print('wrote', OUT_GRID)

    # ── one PNG per session ──
    for lab in labels:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        draw_panel(ax, psd_by[lab], f'{lab} — CAP power spectrum')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Power (dB)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f'psd_db_{lab}.png', dpi=160)
        plt.close(fig)
    print('wrote', len(labels), 'per-session PNGs ->', OUT_DIR)


if __name__ == '__main__':
    main()
