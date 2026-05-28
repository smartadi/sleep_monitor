"""
Thorax effort clusters on unsupervised UMAP projections.

Per session: bin thorax_rms into terciles (low/mid/high), run unsupervised
UMAP on pure-CAP features (no accelerometer), color by effort level.
Also show stage×effort cross-tab to see overlap.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import STAGE_ORDER

STAGE_NAMES = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_COLORS = {0: '#9B59B6', 1: '#2ECC71', 2: '#3498DB', 3: '#F39C12', 4: '#E74C3C'}

EFFORT_NAMES = {0: 'Low', 1: 'Mid', 2: 'High'}
EFFORT_COLORS = {0: '#2ECC71', 1: '#F1C40F', 2: '#E74C3C'}

REPORT_DIR = ROOT / 'reports' / 'projections'
META_COLS = {'t_hr', 'stage_code', 'stage_label', 'apnea_code', 'apnea_label', 'thorax_rms'}
ACC_COLS = {'acc_rms', 'acc_resp_power'}


def load_and_embed(label):
    df = pd.read_csv(REPORT_DIR / label / f'{label}_features.csv')
    df = df[df['stage_code'] >= 0].reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in META_COLS and c not in ACC_COLS]

    X_raw = df[feat_cols].values.copy().astype(np.float64)
    X_raw[~np.isfinite(X_raw)] = np.nan
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X_raw[:, j] = col

    X = StandardScaler().fit_transform(X_raw)

    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42)
    X_emb = reducer.fit_transform(X)

    return df, feat_cols, X_emb


def assign_effort(thorax_rms):
    t_low = np.percentile(thorax_rms, 33.3)
    t_high = np.percentile(thorax_rms, 66.7)
    effort = np.zeros(len(thorax_rms), dtype=int)
    effort[thorax_rms > t_high] = 2
    effort[(thorax_rms > t_low) & (thorax_rms <= t_high)] = 1
    return effort, t_low, t_high


def plot_effort(X_emb, df, effort, label, out_dir):
    t_hr = df['t_hr'].values
    stage_labels = df['stage_label'].values
    thorax_rms = df['thorax_rms'].values

    fig = go.Figure()
    for ec in [0, 1, 2]:
        mask = effort == ec
        if not mask.any():
            continue
        texts = [f't={t:.2f}h | {sl} | thor={tr:.1f}'
                 for t, sl, tr in zip(t_hr[mask], stage_labels[mask], thorax_rms[mask])]
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=3, color=EFFORT_COLORS[ec], opacity=0.6),
            name=f'{EFFORT_NAMES[ec]} (n={mask.sum()})',
            text=texts, hoverinfo='text+name',
        ))
    fig.update_layout(
        title=f'Unsupervised UMAP (no acc) — {label} — Thorax Effort',
        width=900, height=700,
        scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(str(out_dir / f'{label}_effort_clusters.html'))


def plot_stage_reference(X_emb, df, label, out_dir):
    stage_codes = df['stage_code'].values
    fig = go.Figure()
    for sc in STAGE_ORDER:
        mask = stage_codes == sc
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=2.5, color=STAGE_COLORS[sc], opacity=0.6),
            name=STAGE_NAMES[sc],
            hoverinfo='name',
        ))
    fig.update_layout(
        title=f'Unsupervised UMAP (no acc) — {label} — Stage reference',
        width=900, height=700,
        scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(str(out_dir / f'{label}_effort_stage_ref.html'))


def plot_effort_time(X_emb, df, effort, label, out_dir):
    """Trajectory colored by effort with time gradient for line."""
    t_hr = df['t_hr'].values
    stage_labels = df['stage_label'].values
    thorax_rms = df['thorax_rms'].values

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
        mode='lines',
        line=dict(color=t_hr, colorscale='Viridis', width=1.5),
        opacity=0.25, hoverinfo='skip', showlegend=False,
    ))
    for ec in [0, 1, 2]:
        mask = effort == ec
        if not mask.any():
            continue
        texts = [f't={t:.2f}h | {sl} | thor={tr:.1f}'
                 for t, sl, tr in zip(t_hr[mask], stage_labels[mask], thorax_rms[mask])]
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=3, color=EFFORT_COLORS[ec], opacity=0.7),
            name=f'{EFFORT_NAMES[ec]}',
            text=texts, hoverinfo='text+name',
        ))
    fig.update_layout(
        title=f'Unsupervised UMAP (no acc) — {label} — Effort Trajectory',
        width=900, height=700,
        scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(str(out_dir / f'{label}_effort_trajectory.html'))


def plot_crosstab(df, effort, label, out_dir):
    stage_codes = df['stage_code'].values
    present_stages = [sc for sc in STAGE_ORDER if (stage_codes == sc).sum() > 0]

    ct = np.zeros((3, len(present_stages)), dtype=int)
    for ei, ec in enumerate([0, 1, 2]):
        for si, sc in enumerate(present_stages):
            ct[ei, si] = int(((effort == ec) & (stage_codes == sc)).sum())

    ct_pct = ct / (ct.sum(axis=0, keepdims=True) + 1e-10) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Counts
    ax = axes[0]
    stage_names = [STAGE_NAMES[sc] for sc in present_stages]
    x = np.arange(len(present_stages))
    width = 0.25
    for ei, ec in enumerate([0, 1, 2]):
        ax.bar(x + (ei - 1) * width, ct[ei], width,
               label=EFFORT_NAMES[ec], color=EFFORT_COLORS[ec], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.set_ylabel('Count')
    ax.set_title(f'Stage × Effort (counts) — {label}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # Percentages within each stage
    ax = axes[1]
    bottom = np.zeros(len(present_stages))
    for ei, ec in enumerate([0, 1, 2]):
        ax.bar(x, ct_pct[ei], 0.6, bottom=bottom,
               label=EFFORT_NAMES[ec], color=EFFORT_COLORS[ec], alpha=0.8)
        for si in range(len(present_stages)):
            if ct_pct[ei, si] > 8:
                ax.text(x[si], bottom[si] + ct_pct[ei, si] / 2,
                        f'{ct_pct[ei, si]:.0f}%', ha='center', va='center', fontsize=7)
        bottom += ct_pct[ei]
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.set_ylabel('%')
    ax.set_title(f'Effort distribution within stage — {label}')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f'{label}_effort_crosstab.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return ct, ct_pct, present_stages


def run_session(label):
    print(f'\n{"=" * 50}')
    print(f'{label}')

    df, feat_cols, X_emb = load_and_embed(label)
    thorax_rms = df['thorax_rms'].values
    stage_codes = df['stage_code'].values

    effort, t_low, t_high = assign_effort(thorax_rms)
    print(f'  Terciles: low<{t_low:.1f}, mid<{t_high:.1f}, high>={t_high:.1f}')
    print(f'  Features: {len(feat_cols)} (no acc)')

    sil_effort = silhouette_score(X_emb, effort)
    sil_stage = silhouette_score(X_emb, stage_codes)
    print(f'  Silhouette (effort 3-class): {sil_effort:.3f}')
    print(f'  Silhouette (stage 5-class):  {sil_stage:.3f}')

    out_dir = REPORT_DIR / label
    plot_effort(X_emb, df, effort, label, out_dir)
    plot_stage_reference(X_emb, df, label, out_dir)
    plot_effort_time(X_emb, df, effort, label, out_dir)
    ct, ct_pct, present_stages = plot_crosstab(df, effort, label, out_dir)

    print(f'  Effort × Stage (%):')
    stage_names = [STAGE_NAMES[sc] for sc in present_stages]
    header = '         ' + '  '.join(f'{sn:>6s}' for sn in stage_names)
    print(f'  {header}')
    for ei in range(3):
        row = f'    {EFFORT_NAMES[ei]:5s}' + '  '.join(f'{ct_pct[ei, si]:5.0f}%' for si in range(len(present_stages)))
        print(f'  {row}')

    return {
        'label': label,
        'sil_effort': sil_effort,
        'sil_stage': sil_stage,
        't_low': t_low,
        't_high': t_high,
    }


if __name__ == '__main__':
    all_results = []
    for s in range(1, 7):
        for n in range(1, 3):
            label = f'S{s}N{n}'
            try:
                r = run_session(label)
                all_results.append(r)
            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback
                traceback.print_exc()

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  {"Sess":6s} {"sil_effort":>11s} {"sil_stage":>10s} {"t_low":>7s} {"t_high":>7s}')
    for r in all_results:
        print(f'  {r["label"]:6s} {r["sil_effort"]:+.3f}      {r["sil_stage"]:+.3f}     {r["t_low"]:7.1f} {r["t_high"]:7.1f}')

    effs = [r['sil_effort'] for r in all_results]
    stgs = [r['sil_stage'] for r in all_results]
    print(f'\n  Mean sil effort: {np.mean(effs):.3f}  |  Mean sil stage: {np.mean(stgs):.3f}')
    print(f'  Effort > stage in {sum(e > s for e, s in zip(effs, stgs))}/12 sessions')
