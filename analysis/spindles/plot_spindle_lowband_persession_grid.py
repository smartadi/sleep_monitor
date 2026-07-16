"""
The most basic spindle finding, one panel per session: when a spindle occurs,
CAP low-frequency (0-3 Hz) power rises at the spindle onset.

Pure onset-triggered average of the CAP 0-3 Hz power (baseline-corrected) for
every N2 spindle, shown per session for all 12 recordings. Reads the cached
per-session triggered curves from `spindle_lowband_detection.npz` (no recompute).

Output: writeup/figures/spindles/fig_spindle_lowband_persession_grid.png

Usage:
  .venv/Scripts/python.exe -m analysis.spindles.plot_spindle_lowband_persession_grid
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
NPZ = os.path.join(HERE, 'outputs', 'spindle_lowband_detection.npz')
CSV = os.path.join(HERE, 'outputs', 'spindle_lowband_detection.csv')
FIG = os.path.join(HERE, '..', '..', 'writeup', 'figures', 'spindles',
                   'fig_spindle_lowband_persession_grid.png')

CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD', 'CLE-CRE': '#E67E22'}
CORE_HALF = 1.0

d = np.load(NPZ, allow_pickle=True)
t = d['t_axis']
core = np.abs(t) < CORE_HALF
df = pd.read_csv(CSV)
df = df[df.session != 'POOLED'].reset_index(drop=True)
labels = df.session.tolist()
nspin = df.n_spindles_N2.astype(int).tolist()

fig, axes = plt.subplots(4, 3, figsize=(15, 13),
                         gridspec_kw={'hspace': 0.45, 'wspace': 0.22},
                         sharex=True, sharey=True)
axes = axes.ravel()

for i, lab in enumerate(labels):
    ax = axes[i]
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.axvspan(-CORE_HALF, CORE_HALF, color='#2980B9', alpha=0.08)
    ax.axvline(0, color='k', lw=0.7, alpha=0.5)
    for ch in ['CLE', 'CRE', 'CLE-CRE']:
        ax.plot(t, d[f'trig_low_{ch}'][i], color=CH_COLORS[ch], lw=0.9, alpha=0.55)
    chc = d['trig_low_CH'][i]
    ax.plot(t, chc, color=CH_COLORS['CH'], lw=2.2)
    peak = chc[core].mean()
    ax.set_title(f'{lab}  (n={nspin[i]:,} spindles)   CH +{peak:.2f} dB',
                 fontsize=9.5, fontweight='bold')
    ax.set_xlim(t.min(), t.max())
    if i % 3 == 0:
        ax.set_ylabel('0–3 Hz power\n(dB vs baseline)', fontsize=8)
    if i >= 9:
        ax.set_xlabel('Time from spindle center (s)', fontsize=8)
    ax.tick_params(labelsize=7)

# one shared legend
handles = [plt.Line2D([], [], color=CH_COLORS[c], lw=2.2 if c == 'CH' else 0.9,
                      alpha=1 if c == 'CH' else 0.55, label=c)
           for c in ['CH', 'CLE', 'CRE', 'CLE-CRE']]
fig.legend(handles=handles, loc='upper center', fontsize=9, ncol=4,
           bbox_to_anchor=(0.5, 0.965), frameon=False)

fig.suptitle('What the CAP mask does at a sleep spindle — 0–3 Hz power rises at onset '
             '(onset-triggered average, all 12 sessions)',
             fontsize=13.5, fontweight='bold', y=1.0)
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('saved', FIG)
