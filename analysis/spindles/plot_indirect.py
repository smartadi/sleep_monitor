"""Figures for the two indirect spindle tests (HR route + coherence route),
all CAP channels (CLE-CRE, CLE, CRE, CH)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
CAP_CHANNELS = ['CLE-CRE', 'CLE', 'CRE', 'CH']
CHAN_COLORS = {'CLE-CRE': '#E67E22', 'CLE': '#27AE60', 'CRE': '#8E44AD', 'CH': '#2980B9'}


def sem(a, axis=0):
    return np.nanstd(a, axis=axis) / np.sqrt(np.sum(np.isfinite(a), axis=axis))


def fig_hr():
    z = np.load(os.path.join(OUT, 'spindle_hr_triggered.npz'))
    tax = z['tax']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()

    # panel 0: ECG positive control
    ax = axes[0]
    m, e = np.nanmean(z['ecg'], 0), sem(z['ecg'], 0)
    nm, ne = np.nanmean(z['ecg_null'], 0), sem(z['ecg_null'], 0)
    ax.plot(tax, m, color='#C0392B', lw=2, label='at spindles')
    ax.fill_between(tax, m - e, m + e, color='#C0392B', alpha=0.25)
    ax.plot(tax, nm, color='gray', lw=1.2, ls='--', label='random N2 (null)')
    ax.fill_between(tax, nm - ne, nm + ne, color='gray', alpha=0.15)
    ax.axvline(0, color='k', ls=':', lw=1)
    ax.set_title('ECG heart rate (autonomic ground truth)')
    ax.set_ylabel('Δ heart rate (bpm)'); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # panels 1..4: each CAP channel's mask-derived HR
    for i, ch in enumerate(CAP_CHANNELS, start=1):
        ax = axes[i]
        cur, nul = z[f'cap_{ch}'], z[f'capnull_{ch}']
        m, e = np.nanmean(cur, 0), sem(cur, 0)
        nm, ne = np.nanmean(nul, 0), sem(nul, 0)
        ax.plot(tax, m, color=CHAN_COLORS[ch], lw=2, label='at spindles')
        ax.fill_between(tax, m - e, m + e, color=CHAN_COLORS[ch], alpha=0.25)
        ax.plot(tax, nm, color='gray', lw=1.2, ls='--', label='random N2 (null)')
        ax.fill_between(tax, nm - ne, nm + ne, color='gray', alpha=0.15)
        ax.axvline(0, color='k', ls=':', lw=1)
        ax.set_title(f'CAP-derived HR — {ch}')
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        if i >= 3:
            ax.set_xlabel('Time from spindle center (s)')
    axes[1].set_xlabel('Time from spindle center (s)')
    axes[3].set_ylabel('Δ heart rate (bpm)')
    axes[5].axis('off')

    fig.suptitle('#3 Spindle-triggered heart rate — clear biphasic autonomic beat in ECG; '
                 'no time-locked deflection in any CAP channel', y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(OUT, 'fig_spindle_hr_triggered.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p)


def fig_coh():
    z = np.load(os.path.join(OUT, 'spindle_coherence.npz'))
    f = z['freqs']
    df = pd.read_csv(os.path.join(OUT, 'spindle_coherence.csv'))
    band = f <= 25
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # left: spectra — anchor + every EEG-CAP pair
    ax = axes[0]
    a = z['cle_cre']
    m, e = np.nanmean(a, 0)[band], sem(a, 0)[band]
    ax.plot(f[band], m, color='k', lw=2, label='CLE vs CRE (same source — anchor)')
    ax.fill_between(f[band], m - e, m + e, color='k', alpha=0.15)
    for ch in CAP_CHANNELS:
        c = z[f'eeg_cap_{ch}']
        m = np.nanmean(c, 0)[band]
        ax.plot(f[band], m, color=CHAN_COLORS[ch], lw=1.5, label=f'EEG vs {ch}')
    ax.axvspan(11, 16, color='gold', alpha=0.25, label='sigma (spindle) band')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude-squared coherence')
    ax.set_title('Whole-night coherence spectrum'); ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25); ax.set_ylim(0, None)

    # right: per-session sigma-band coherence, all channels
    ax = axes[1]
    x = np.arange(len(df))
    w = 0.16
    ax.bar(x - 2 * w, df['cle_cre_coh_sigma'], w, color='k', label='CLE-CRE anchor')
    for j, ch in enumerate(CAP_CHANNELS):
        ax.bar(x + (j - 1) * w, df[f'eeg_cap_coh_sigma_{ch}'], w,
               color=CHAN_COLORS[ch], label=f'EEG-{ch}')
    ax.set_xticks(x); ax.set_xticklabels(df['session'], rotation=90, fontsize=7)
    ax.set_ylabel('Mean sigma-band coherence')
    ax.set_title('Sigma-band coherence per session')
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, axis='y')

    fig.suptitle('#2 EEG↔CAP sigma coherence ≈ 0 for every channel: no electrical pickup of cortical sigma',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig_spindle_coherence.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p)


if __name__ == '__main__':
    fig_hr()
    fig_coh()
