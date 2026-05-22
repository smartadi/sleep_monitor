"""
Harmonic structure detection — all 12 sessions.

Stage 2: run harmonic detector across every session, aggregate statistics,
test whether N3 discrimination (observed in S1N1) generalises across subjects.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kruskal, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_all_sessions, FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.harmonics import detect_harmonics_multichannel
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.viz import plot_hypnogram

PLOT_DIR = Path(__file__).resolve().parents[2] / 'notebooks' / 'plots' / 'harmonics'
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / 'artifacts' / 'harmonics'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_CODE_MAP = {4: 'Wake', 3: 'N1', 2: 'N2', 1: 'N3', 0: 'REM'}
FEATURES = ['harmonic_energy_ratio', 'n_harmonics', 'cep_prominence', 'hps_score']

# ── Step 1: Load all sessions and compute harmonics ─────────────────────────

print('Loading all 12 sessions...')
sessions = load_all_sessions(with_sleep_profiles=True)

all_results = []
for s in sessions:
    print(f'  {s.label} ({s.subject}, {s.duration_hr:.1f} hr)...')

    acc_mag = s.cap['acc_mag']
    signals = {}
    for ch in ['CH', 'CLE', 'CRE']:
        signals[ch] = remove_acc_artifact(s.cap[ch], acc_mag, 0.05, 4.0)

    df = detect_harmonics_multichannel(
        signals, fs=FS, win_sec=30.0, step_sec=30.0, acc_mag=acc_mag,
    )

    sp = s.sleep_profile
    epoch_t_hr = sp['t_ep_hr']
    codes = sp['codes']
    stages = []
    for _, row in df.iterrows():
        idx = np.argmin(np.abs(epoch_t_hr - row['t_hr']))
        stages.append(STAGE_CODE_MAP.get(int(codes[idx]), '?'))
    df['stage'] = stages
    df['session'] = s.label
    df['subject'] = s.subject

    n_valid = (~df.motion_masked).sum()
    print(f'    {len(df)} windows, {n_valid} valid, '
          f'{(df.motion_masked).sum()} motion-masked')
    all_results.append(df)

full_df = pd.concat(all_results, ignore_index=True)
valid = full_df[~full_df.motion_masked].copy()
valid = valid[valid.stage != '?'].copy()

full_df.to_parquet(ARTIFACT_DIR / 'allsessions.parquet', index=False)
print(f'\nTotal: {len(full_df)} windows, {len(valid)} valid')
print(f'Saved {ARTIFACT_DIR / "allsessions.parquet"}')

# ── Step 2: Statistics ──────────────────────────────────────────────────────

print('\n' + '=' * 70)
print('CROSS-SESSION SUMMARY — harmonic features by sleep stage')
print('=' * 70)

summary_rows = []
for feat in FEATURES:
    for stage in STAGE_NAMES:
        vals = valid[valid.stage == stage][feat].dropna()
        summary_rows.append(dict(
            feature=feat, stage=stage, n=len(vals),
            mean=vals.mean(), median=vals.median(), std=vals.std(),
        ))
summary_df = pd.DataFrame(summary_rows)

for feat in FEATURES:
    print(f'\n  {feat}:')
    sub = summary_df[summary_df.feature == feat]
    for _, r in sub.iterrows():
        print(f'    {r.stage:>5}: n={int(r.n):>5}  mean={r["mean"]:.3f}  '
              f'median={r["median"]:.3f}  std={r["std"]:.3f}')

# Per-session × per-stage breakdown
print('\n' + '-' * 70)
print('PER-SESSION median harmonic_energy_ratio by stage')
print('-' * 70)
pivot = valid.pivot_table(
    values='harmonic_energy_ratio', index='session', columns='stage',
    aggfunc='median',
)
pivot = pivot.reindex(columns=STAGE_NAMES)
print(pivot.to_string(float_format='%.3f'))

# Kruskal-Wallis
print('\n' + '-' * 70)
print('KRUSKAL-WALLIS tests (5-stage comparison)')
print('-' * 70)
kw_results = []
for feat in FEATURES:
    groups = [valid[valid.stage == st][feat].dropna().values for st in STAGE_NAMES]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        print(f'  {feat:30s}  H={stat:8.2f}  p={p:.2e}')
        kw_results.append(dict(feature=feat, H=stat, p=p))
    else:
        print(f'  {feat:30s}  insufficient data')

# Mann-Whitney U: N3 vs each other stage
print('\n' + '-' * 70)
print('MANN-WHITNEY U: N3 vs each stage (Bonferroni-corrected)')
print('-' * 70)
n3_data = valid[valid.stage == 'N3']
mw_results = []
for feat in FEATURES:
    n3_vals = n3_data[feat].dropna().values
    if len(n3_vals) < 5:
        print(f'  {feat}: too few N3 samples ({len(n3_vals)})')
        continue
    print(f'  {feat}:')
    for other in ['Wake', 'N1', 'N2', 'REM']:
        other_vals = valid[valid.stage == other][feat].dropna().values
        if len(other_vals) < 5:
            continue
        u_stat, p_raw = mannwhitneyu(n3_vals, other_vals, alternative='two-sided')
        p_corr = min(p_raw * 4, 1.0)  # Bonferroni (4 comparisons)
        # rank-biserial effect size: r = 1 - 2U/(n1*n2)
        r_rb = 1 - (2 * u_stat) / (len(n3_vals) * len(other_vals))
        sig = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
        print(f'    N3 vs {other:>4}: U={u_stat:10.0f}  p_corr={p_corr:.3e}  '
              f'r_rb={r_rb:+.3f}  {sig}')
        mw_results.append(dict(
            feature=feat, comparison=f'N3 vs {other}',
            U=u_stat, p_raw=p_raw, p_corr=p_corr, r_rb=r_rb,
        ))

# Dominant channel
print('\n' + '-' * 70)
print('DOMINANT CHANNEL')
print('-' * 70)
ch_overall = valid['dominant_channel'].value_counts()
print('Overall:', ch_overall.to_dict())
ch_by_session = pd.crosstab(valid['session'], valid['dominant_channel'])
ch_by_session_pct = ch_by_session.div(ch_by_session.sum(axis=1), axis=0) * 100
print('\nBy session (%):\n', ch_by_session_pct.round(1).to_string())

# ── Step 3: Plots ───────────────────────────────────────────────────────────

session_labels = valid['session'].unique()
n_sessions = len(session_labels)

# --- Figure 1: Grid of full-night traces (harmonic_energy_ratio + stage shading)

print('\nPlotting full-night trace grid...')
fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=False, sharey=True)
axes = axes.flatten()

for i, sess_label in enumerate(sorted(session_labels)):
    ax = axes[i]
    sess = valid[valid.session == sess_label]
    s_obj = [s for s in sessions if s.label == sess_label][0]
    sp = s_obj.sleep_profile

    # stage shading
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                   color=STAGE_COLORS.get(c, '#AAA'), alpha=0.15)

    ax.plot(sess['t_hr'], sess['harmonic_energy_ratio'],
            color='#2C3E50', lw=0.6, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title(sess_label, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.15)
    if i >= 9:
        ax.set_xlabel('Time (hr)', fontsize=7)
    if i % 3 == 0:
        ax.set_ylabel('HER', fontsize=7)
    ax.tick_params(labelsize=6)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
           for c in STAGE_ORDER]
fig.legend(handles=patches, loc='upper right', fontsize=7, ncol=5, framealpha=0.8)
fig.suptitle('Harmonic Energy Ratio — All Sessions (30s windows, best-of CH/CLE/CRE)',
             fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(PLOT_DIR / 'fullnight_grid_allsessions.png', dpi=200)
print(f'  Saved fullnight_grid_allsessions.png')

# --- Figure 2: Cross-session stage boxplots (pooled)

print('Plotting cross-session stage boxplots...')
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
feat_labels = [
    ('harmonic_energy_ratio', 'Harmonic Energy Ratio'),
    ('n_harmonics', 'Confirmed Harmonics Count'),
    ('cep_prominence', 'Cepstral Prominence'),
    ('hps_score', 'HPS Score (log-domain)'),
]

for ax, (col, title) in zip(axes.flat, feat_labels):
    data = [valid[valid.stage == st][col].dropna().values for st in STAGE_NAMES]
    bp = ax.boxplot(data, tick_labels=STAGE_NAMES, patch_artist=True, widths=0.6,
                    medianprops=dict(color='#E74C3C', lw=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, st in zip(bp['boxes'], STAGE_NAMES):
        code = {v: k for k, v in STAGE_CODE_MAP.items()}[st]
        patch.set_facecolor(STAGE_COLORS[code])
        patch.set_alpha(0.6)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    for j, d in enumerate(data):
        ax.text(j + 1, ax.get_ylim()[0], f'n={len(d)}',
                ha='center', va='bottom', fontsize=6, color='gray')

fig.suptitle('Harmonic Features by Sleep Stage — All 12 Sessions Pooled', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOT_DIR / 'stage_boxplots_allsessions.png', dpi=200)
print(f'  Saved stage_boxplots_allsessions.png')

# --- Figure 3: Heatmap — median harmonic_energy_ratio per session × stage

print('Plotting per-subject heatmap...')
fig, ax = plt.subplots(figsize=(8, 6))
pivot_plot = pivot.copy()
im = ax.imshow(pivot_plot.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(len(STAGE_NAMES)))
ax.set_xticklabels(STAGE_NAMES)
ax.set_yticks(range(len(pivot_plot.index)))
ax.set_yticklabels(pivot_plot.index, fontsize=8)
for r in range(pivot_plot.shape[0]):
    for c in range(pivot_plot.shape[1]):
        v = pivot_plot.values[r, c]
        if np.isfinite(v):
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if v > 0.6 else 'black')
fig.colorbar(im, ax=ax, label='Median HER')
ax.set_title('Median Harmonic Energy Ratio — Session × Stage', fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'heatmap_her_session_stage.png', dpi=200)
print(f'  Saved heatmap_her_session_stage.png')

# --- Figure 4: Dominant channel stacked bar by session

print('Plotting dominant channel by session...')
fig, ax = plt.subplots(figsize=(10, 5))
ch_colors = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}
ch_pct = ch_by_session_pct.reindex(columns=['CH', 'CLE', 'CRE']).fillna(0)
ch_pct.plot(kind='bar', stacked=True, ax=ax,
            color=[ch_colors[c] for c in ch_pct.columns])
ax.set_ylabel('Percentage (%)', fontsize=10)
ax.set_xlabel('')
ax.set_xticklabels(ch_pct.index, rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=9, title='Channel')
ax.set_title('Dominant Channel by Session (%)', fontsize=11)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(PLOT_DIR / 'dominant_channel_allsessions.png', dpi=200)
print(f'  Saved dominant_channel_allsessions.png')

# --- Figure 5: Per-session N3-vs-rest effect size (rank-biserial)

print('Plotting N3 discrimination by session...')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (feat, title) in zip(axes.flat, feat_labels):
    r_rbs = []
    labels = []
    for sess_label in sorted(session_labels):
        sess = valid[valid.session == sess_label]
        n3 = sess[sess.stage == 'N3'][feat].dropna().values
        non_n3 = sess[sess.stage != 'N3'][feat].dropna().values
        if len(n3) >= 3 and len(non_n3) >= 3:
            u, _ = mannwhitneyu(n3, non_n3, alternative='two-sided')
            r_rb = 1 - (2 * u) / (len(n3) * len(non_n3))
            r_rbs.append(r_rb)
            labels.append(sess_label)

    colors = ['#27AE60' if r > 0 else '#E74C3C' for r in r_rbs]
    ax.barh(range(len(labels)), r_rbs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Rank-biserial r (N3 vs rest)', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2, axis='x')

fig.suptitle('N3 vs Non-N3 Discrimination — Per Session Effect Size', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOT_DIR / 'n3_effectsize_allsessions.png', dpi=200)
print(f'  Saved n3_effectsize_allsessions.png')

plt.close('all')

# ── Step 4: Console summary ────────────────────────────────────────────────

print('\n' + '=' * 70)
print('KEY FINDINGS')
print('=' * 70)

her_by_stage = valid.groupby('stage')['harmonic_energy_ratio'].median()
print(f'\nMedian HER across all sessions:')
for st in STAGE_NAMES:
    if st in her_by_stage:
        print(f'  {st:>5}: {her_by_stage[st]:.3f}')

n3_median = her_by_stage.get('N3', np.nan)
n1_median = her_by_stage.get('N1', np.nan)
n3_higher = (pivot['N3'] > pivot['N2']).sum() if 'N3' in pivot.columns and 'N2' in pivot.columns else 0
n_with_n3 = pivot['N3'].dropna().shape[0] if 'N3' in pivot.columns else 0
print(f'\nN3 median HER > N2 in {n3_higher}/{n_with_n3} sessions')

ch_winner = ch_overall.idxmax()
ch_pct_winner = ch_overall.max() / ch_overall.sum() * 100
print(f'Dominant channel: {ch_winner} ({ch_pct_winner:.0f}% of windows)')

print('\nDone. Run complete.')
