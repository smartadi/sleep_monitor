"""
Harmonic structure detection — S1N1 visualization.

Generates 3 figures:
1. Full-night harmonic traces overlaid on hypnogram
2. Harmonic feature distributions by sleep stage (boxplots)
3. Dominant channel breakdown (bar chart)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import (
    load_session, FS,
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER,
)
from sleep_monitor.harmonics import detect_harmonics_multichannel
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.viz import plot_hypnogram

PLOT_DIR = Path(__file__).resolve().parents[2] / 'notebooks' / 'plots' / 'harmonics'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_CODE_MAP = {4: 'Wake', 3: 'N1', 2: 'N2', 1: 'N3', 0: 'REM'}

# ── Load and preprocess ──────────────────────────────────────────────────────

print('Loading S1N1...')
s = load_session(0, with_profile=True)
acc_mag = s.cap['acc_mag']

signals = {}
for ch in ['CH', 'CLE', 'CRE']:
    signals[ch] = remove_acc_artifact(s.cap[ch], acc_mag, 0.05, 4.0)

print('Running harmonic detection...')
df = detect_harmonics_multichannel(signals, fs=FS, win_sec=30.0, step_sec=30.0, acc_mag=acc_mag)

# Assign sleep stage to each window
sp = s.sleep_profile
epoch_t_hr = sp['t_ep_hr']
codes = sp['codes']
stages = []
for _, row in df.iterrows():
    idx = np.argmin(np.abs(epoch_t_hr - row['t_hr']))
    stages.append(STAGE_CODE_MAP.get(int(codes[idx]), '?'))
df['stage'] = stages

valid = df[~df.motion_masked].copy()
print(f'  {len(df)} windows total, {len(valid)} valid (non-masked)')

# ── Figure 1: Full-night traces + hypnogram ──────────────────────────────────

print('Plotting full-night traces...')
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1.2]})

# Hypnogram
plot_hypnogram(sp, axes[0], title=f'{s.label} — Hypnogram')

# Stage background shading helper
def shade_stages(ax, t_hr, codes, alpha=0.08):
    for i in range(len(t_hr) - 1):
        c = int(codes[i])
        ax.axvspan(t_hr[i], t_hr[i + 1], color=STAGE_COLORS.get(c, '#AAA'), alpha=alpha)

# Harmonic energy ratio
ax = axes[1]
shade_stages(ax, epoch_t_hr, codes)
ax.plot(valid['t_hr'], valid['harmonic_energy_ratio'], color='#2C3E50', lw=0.8, alpha=0.9)
ax.axhline(valid['harmonic_energy_ratio'].median(), color='#E74C3C', ls='--', lw=0.7, alpha=0.6)
ax.set_ylabel('Harmonic\nenergy ratio', fontsize=8)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.2)
ax.set_title('Harmonic Energy Ratio (fraction of power in harmonic peaks)', fontsize=9)

# n_harmonics
ax = axes[2]
shade_stages(ax, epoch_t_hr, codes)
ax.plot(valid['t_hr'], valid['n_harmonics'], color='#8E44AD', lw=0.8, alpha=0.9)
ax.set_ylabel('n_harmonics', fontsize=8)
ax.set_ylim(-0.5, 7)
ax.grid(True, alpha=0.2)
ax.set_title('Confirmed Harmonic Count', fontsize=9)

# Cepstral prominence
ax = axes[3]
shade_stages(ax, epoch_t_hr, codes)
ax.plot(valid['t_hr'], valid['cep_prominence'], color='#27AE60', lw=0.8, alpha=0.9)
ax.axhline(valid['cep_prominence'].median(), color='#E74C3C', ls='--', lw=0.7, alpha=0.6)
ax.set_ylabel('Cepstral\nprominence', fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_title('Cepstral Prominence (harmonic vs broadband)', fontsize=9)

ax.set_xlabel('Time (hr)', fontsize=9)

# Stage legend
patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c]) for c in STAGE_ORDER]
fig.legend(handles=patches, loc='upper right', fontsize=7, ncol=5, framealpha=0.8)

fig.suptitle(f'{s.label} — Harmonic Structure Detection (30s windows, CH/CLE/CRE best-of)',
             fontsize=11, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOT_DIR / 'harmonics_fullnight_s1n1.png', dpi=200)
print(f'  Saved {PLOT_DIR / "harmonics_fullnight_s1n1.png"}')

# ── Figure 2: Stage boxplots ────────────────────────────────────────────────

print('Plotting stage boxplots...')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

features = [
    ('harmonic_energy_ratio', 'Harmonic Energy Ratio', '#2C3E50'),
    ('n_harmonics', 'Confirmed Harmonics Count', '#8E44AD'),
    ('cep_prominence', 'Cepstral Prominence', '#27AE60'),
    ('hps_score', 'HPS Score (log-domain)', '#E67E22'),
]

for ax, (col, title, color) in zip(axes.flat, features):
    data = [valid[valid.stage == st][col].dropna().values for st in STAGE_NAMES]
    bp = ax.boxplot(data, tick_labels=STAGE_NAMES, patch_artist=True, widths=0.6,
                    medianprops=dict(color='#E74C3C', lw=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, st in zip(bp['boxes'], STAGE_NAMES):
        code = {v: k for k, v in STAGE_CODE_MAP.items()}[st]
        patch.set_facecolor(STAGE_COLORS[code])
        patch.set_alpha(0.6)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(col, fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    counts = [len(d) for d in data]
    for i, n in enumerate(counts):
        ax.text(i + 1, ax.get_ylim()[0], f'n={n}', ha='center', va='bottom', fontsize=6, color='gray')

fig.suptitle(f'{s.label} — Harmonic Features by Sleep Stage', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOT_DIR / 'harmonics_stage_boxplots_s1n1.png', dpi=200)
print(f'  Saved {PLOT_DIR / "harmonics_stage_boxplots_s1n1.png"}')

# ── Figure 3: Dominant channel bar chart ─────────────────────────────────────

print('Plotting dominant channel breakdown...')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ch_colors = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}

# Overall
ax = axes[0]
counts = valid['dominant_channel'].value_counts()
bars = ax.bar(counts.index, counts.values, color=[ch_colors.get(c, '#999') for c in counts.index])
ax.set_title('Overall Dominant Channel', fontsize=10)
ax.set_ylabel('Window count', fontsize=9)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 5, str(v), ha='center', fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

# By stage
ax = axes[1]
stage_ch = pd.crosstab(valid['stage'], valid['dominant_channel'])
stage_ch = stage_ch.reindex(STAGE_NAMES).fillna(0)
stage_ch_pct = stage_ch.div(stage_ch.sum(axis=1), axis=0) * 100
stage_ch_pct[[c for c in ['CH', 'CLE', 'CRE'] if c in stage_ch_pct.columns]].plot(
    kind='bar', stacked=True, ax=ax,
    color=[ch_colors[c] for c in stage_ch_pct.columns if c in ch_colors])
ax.set_title('Dominant Channel by Stage (%)', fontsize=10)
ax.set_ylabel('Percentage', fontsize=9)
ax.set_xlabel('')
ax.set_xticklabels(STAGE_NAMES, rotation=0)
ax.legend(fontsize=8, title='Channel')
ax.grid(True, alpha=0.2, axis='y')

fig.suptitle(f'{s.label} — Which Channel Shows Strongest Harmonics?', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(PLOT_DIR / 'harmonics_dominant_channel_s1n1.png', dpi=200)
print(f'  Saved {PLOT_DIR / "harmonics_dominant_channel_s1n1.png"}')

plt.close('all')
print('Done.')
