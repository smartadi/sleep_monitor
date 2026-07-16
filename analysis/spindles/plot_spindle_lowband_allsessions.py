"""
All-sessions summary of the low-band (0-3 Hz) CAP spindle response.

Answers "is the 0-3 Hz power increase at spindle onset a real finding?" by
showing it across all 12 sessions, using the per-session onset-triggered curves
and per-session detection rates already computed by
`spindle_lowband_detection.py` (reads the cached .npz + .csv, no recompute).

Figure (writeup/figures/spindles/fig_spindle_lowband_allsessions.png):
  (A) All 12 sessions' CH low-band onset-triggered average curves (thin) with the
      grand mean +/- SD (bold) — the bump is present in every session.
  (B) Per-session CH onset bump (core |t|<1 s dB) and per-spindle detection rate,
      with the zero/chance references; 12/12 sessions positive.
  (C) Grand-mean onset-triggered curves for all four CAP channels (CH strongest),
      with the sigma-band negative control (per-spindle ~chance) annotated.

Usage:
  .venv/Scripts/python.exe -m analysis.spindles.plot_spindle_lowband_allsessions
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
                   'fig_spindle_lowband_allsessions.png')

CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD', 'CLE-CRE': '#E67E22'}
CORE_HALF = 1.0

d = np.load(NPZ, allow_pickle=True)
t = d['t_axis']
core = np.abs(t) < CORE_HALF
df = pd.read_csv(CSV)
df = df[df.session != 'POOLED'].reset_index(drop=True)
labels = df.session.tolist()
n_spindles = int(df.n_spindles_N2.sum())

fig, axes = plt.subplots(1, 3, figsize=(18, 5.6),
                         gridspec_kw={'wspace': 0.30, 'width_ratios': [1.3, 1.3, 1.1]})

# ---- (A) all 12 CH curves + grand mean ----
ax = axes[0]
trigCH = d['trig_low_CH']                     # (12, T)
gm = trigCH.mean(axis=0); gs = trigCH.std(axis=0)
ax.axhline(0, color='gray', lw=0.8, ls=':')
ax.axvspan(-CORE_HALF, CORE_HALF, color=CH_COLORS['CH'], alpha=0.10, label='onset core (±1 s)')
for i in range(trigCH.shape[0]):
    ax.plot(t, trigCH[i], color=CH_COLORS['CH'], lw=0.7, alpha=0.30)
ax.fill_between(t, gm - gs, gm + gs, color=CH_COLORS['CH'], alpha=0.20)
ax.plot(t, gm, color=CH_COLORS['CH'], lw=2.4, label='grand mean ± SD (12 sessions)')
ax.axvline(0, color='k', lw=0.8, alpha=0.6)
ax.set_title('(A) CH low-band (0–3 Hz) power at spindle onset\n'
             'every one of 12 sessions shows the bump', fontsize=10.5, fontweight='bold')
ax.set_xlabel('Time from spindle center (s)', fontsize=9)
ax.set_ylabel('CH 0–3 Hz power (dB vs own baseline)', fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(t.min(), t.max())

# ---- (B) per-session bump + detection rate ----
ax = axes[1]
peaks = trigCH[:, core].mean(axis=1)          # per-session onset core dB
x = np.arange(len(labels))
bars = ax.bar(x, peaks, color=CH_COLORS['CH'], alpha=0.85)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('CH onset bump (core dB)', fontsize=9, color=CH_COLORS['CH'])
ax.tick_params(axis='y', labelcolor=CH_COLORS['CH'])
ax.set_title(f'(B) Per-session onset bump & per-spindle detection rate\n'
             f'{int((peaks>0).sum())}/12 sessions positive '
             f'(median +{np.median(peaks):.2f} dB)', fontsize=10.5, fontweight='bold')
# detection rate on twin axis
ax2 = ax.twinx()
det = df.CH_low_detrate.to_numpy() * 100
null = df.CH_low_nullrate.to_numpy() * 100
ax2.plot(x, det, 'o-', color='#C0392B', lw=1.2, ms=5, label='detection rate')
ax2.plot(x, null, '--', color='gray', lw=1.0, label='chance (matched controls)')
ax2.set_ylabel('per-spindle detection rate (%)', fontsize=9, color='#C0392B')
ax2.tick_params(axis='y', labelcolor='#C0392B')
ax2.set_ylim(45, 60)
ax2.legend(fontsize=7.5, loc='upper right')

# ---- (C) grand-mean per channel ----
ax = axes[2]
ax.axhline(0, color='gray', lw=0.8, ls=':')
ax.axvspan(-CORE_HALF, CORE_HALF, color='gray', alpha=0.08)
for ch in ['CH', 'CLE', 'CRE', 'CLE-CRE']:
    g = d[f'trig_low_{ch}'].mean(axis=0)
    ax.plot(t, g, color=CH_COLORS[ch], lw=2.0 if ch == 'CH' else 1.3,
            label=f'{ch} (+{g[core].mean():.2f} dB)')
ax.axvline(0, color='k', lw=0.8, alpha=0.6)
sig_det = df.CH_sigma_detrate.mean() * 100
ax.set_title('(C) Grand-mean onset response by channel\n'
             f'CH strongest; sigma (11–16 Hz) stays at chance ({sig_det:.0f}%)',
             fontsize=10.5, fontweight='bold')
ax.set_xlabel('Time from spindle center (s)', fontsize=9)
ax.set_ylabel('0–3 Hz power (dB vs own baseline)', fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(t.min(), t.max())

fig.suptitle(f'Low-band (0–3 Hz) spindle response in the capacitive mask — all 12 sessions, '
             f'{n_spindles:,} N2 spindles',
             fontsize=13, fontweight='bold', y=1.02)
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('saved', FIG)
print('per-session CH onset bump (dB):', dict(zip(labels, np.round(peaks, 2))))
print('per-session CH detection rate (%):', dict(zip(labels, np.round(det, 1))))
