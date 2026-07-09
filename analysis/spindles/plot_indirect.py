"""Figures for the two indirect spindle tests (HR route + coherence route)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'outputs')


def sem(a, axis=0):
    return np.nanstd(a, axis=axis) / np.sqrt(np.sum(np.isfinite(a), axis=axis))


def fig_hr():
    z = np.load(os.path.join(OUT, 'spindle_hr_triggered.npz'))
    tax = z['tax']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    specs = [('ecg', 'ecg_null', 'ECG heart rate (autonomic ground truth)', '#C0392B'),
             ('cap', 'cap_null', 'CAP-derived heart rate (mask)', '#E67E22')]
    for ax, (key, nkey, title, color) in zip(axes, specs):
        cur, nul = z[key], z[nkey]
        m, e = np.nanmean(cur, 0), sem(cur, 0)
        nm, ne = np.nanmean(nul, 0), sem(nul, 0)
        ax.plot(tax, m, color=color, lw=2, label='at spindles')
        ax.fill_between(tax, m - e, m + e, color=color, alpha=0.25)
        ax.plot(tax, nm, color='gray', lw=1.2, ls='--', label='random N2 (null)')
        ax.fill_between(tax, nm - ne, nm + ne, color='gray', alpha=0.15)
        ax.axvline(0, color='k', ls=':', lw=1)
        ax.set_title(title); ax.set_xlabel('Time from spindle center (s)')
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    axes[0].set_ylabel('Δ heart rate (bpm)')
    fig.suptitle('#3 Spindle-triggered heart rate: a clear biphasic autonomic beat in ECG, '
                 'absent/inconsistent in the mask', y=1.02, fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig_spindle_hr_triggered.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p)


def fig_coh():
    z = np.load(os.path.join(OUT, 'spindle_coherence.npz'))
    f = z['freqs']
    df = pd.read_csv(os.path.join(OUT, 'spindle_coherence.csv'))
    band = f <= 25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for key, color, lab in [('cle_cre', '#27AE60', 'CLE vs CRE (same physical source — anchor)'),
                            ('eeg_cap', '#2C3E50', 'EEG vs CAP (electrical-leakage test)')]:
        c = z[key]
        m, e = np.nanmean(c, 0)[band], sem(c, 0)[band]
        ax.plot(f[band], m, color=color, lw=1.8, label=lab)
        ax.fill_between(f[band], m - e, m + e, color=color, alpha=0.2)
    ax.axvspan(11, 16, color='gold', alpha=0.25, label='sigma (spindle) band')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude-squared coherence')
    ax.set_title('Whole-night coherence spectrum'); ax.legend(fontsize=8)
    ax.grid(alpha=0.25); ax.set_ylim(0, None)

    ax = axes[1]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df['cle_cre_coh_sigma'], 0.4, color='#27AE60', label='CLE-CRE (anchor)')
    ax.bar(x + 0.2, df['eeg_cap_coh_sigma'], 0.4, color='#2C3E50', label='EEG-CAP')
    ax.set_xticks(x); ax.set_xticklabels(df['session'], rotation=90, fontsize=7)
    ax.set_ylabel('Mean sigma-band coherence'); ax.set_title('Sigma-band coherence per session')
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis='y')

    fig.suptitle('#2 EEG↔CAP sigma coherence ≈ 0: no electrical pickup of cortical sigma by the mask',
                 y=1.02, fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig_spindle_coherence.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p)


if __name__ == '__main__':
    fig_hr()
    fig_coh()
