"""
Informed unsupervised projections.

Transfer knowledge from supervised UMAP to unsupervised methods:
1. KW-weighted: scale features by sqrt(Kruskal-Wallis H) before unsupervised UMAP
2. NCA-transformed: learn a linear projection via Neighborhood Components Analysis,
   then run unsupervised UMAP on the transformed space

Compare against baselines (raw unsupervised, supervised UMAP) across all 12 sessions.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kruskal

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import make_pipeline
import umap
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import STAGE_ORDER, APNEA_LABELS, APNEA_COLORS
STAGE_NAMES = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_COLORS = {0: '#9B59B6', 1: '#2ECC71', 2: '#3498DB', 3: '#F39C12', 4: '#E74C3C'}

REPORT_DIR = ROOT / 'reports' / 'projections'

META_COLS = {'t_hr', 'stage_code', 'stage_label', 'apnea_code', 'apnea_label', 'thorax_rms'}


def load_features(label):
    path = REPORT_DIR / label / f'{label}_features.csv'
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


def clean_and_scale(df, feat_cols):
    X_raw = df[feat_cols].values.copy().astype(np.float64)
    X_raw[~np.isfinite(X_raw)] = np.nan
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X_raw[:, j] = col
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, X_raw


def compute_kw_weights(X_raw, stage_codes, feat_cols):
    weights = np.ones(len(feat_cols))
    for j, fname in enumerate(feat_cols):
        present = [sc for sc in STAGE_ORDER if (stage_codes == sc).sum() > 0]
        groups = [X_raw[stage_codes == sc, j] for sc in present]
        if len(groups) >= 2:
            try:
                stat, _ = kruskal(*groups)
                weights[j] = stat
            except ValueError:
                weights[j] = 0.0
    weights = np.sqrt(np.maximum(weights, 0))
    weights /= (weights.max() + 1e-20)
    return weights


def plot_3d_stage(X_emb, stage_codes, title, filepath):
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
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
                      legend=dict(itemsizing='constant'))
    fig.write_html(str(filepath))


def plot_3d_time(X_emb, t_hr, stage_labels, title, filepath):
    texts = [f't={t:.2f}h | {sl}' for t, sl in zip(t_hr, stage_labels)]
    fig = go.Figure(data=[go.Scatter3d(
        x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
        mode='markers',
        marker=dict(size=2.5, color=t_hr, colorscale='Viridis',
                    opacity=0.6, colorbar=dict(title='Hours')),
        text=texts, hoverinfo='text',
    )])
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'))
    fig.write_html(str(filepath))


def plot_3d_thorax(X_emb, thorax_rms, title, filepath):
    fig = go.Figure(data=[go.Scatter3d(
        x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
        mode='markers',
        marker=dict(size=2.5, color=thorax_rms, colorscale='Plasma',
                    opacity=0.6, colorbar=dict(title='Thorax RMS')),
        hoverinfo='text',
        text=[f'thor={v:.2f}' for v in thorax_rms],
    )])
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'))
    fig.write_html(str(filepath))


def plot_trajectory(X_emb, t_hr, stage_codes, stage_labels, title, filepath):
    texts = [f't={t:.2f}h | {sl}' for t, sl in zip(t_hr, stage_labels)]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
        mode='lines',
        line=dict(color=t_hr, colorscale='Viridis', width=1.5),
        opacity=0.3, hoverinfo='skip', showlegend=False,
    ))
    for sc in STAGE_ORDER:
        mask = stage_codes == sc
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=3, color=STAGE_COLORS[sc], opacity=0.7),
            name=STAGE_NAMES[sc],
            text=[texts[j] for j in np.where(mask)[0]],
            hoverinfo='text+name',
        ))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
                      legend=dict(itemsizing='constant'))
    fig.write_html(str(filepath))


def run_session(label):
    print(f'\n{"=" * 60}')
    print(f'{label}')

    df, feat_cols = load_features(label)
    X_scaled, X_raw = clean_and_scale(df, feat_cols)
    stage_codes = df['stage_code'].values
    t_hr = df['t_hr'].values
    stage_labels = df['stage_label'].values
    thorax_rms = df['thorax_rms'].values

    out_dir = REPORT_DIR / label
    n_feat = len(feat_cols)

    # --- KW weights ---
    kw_weights = compute_kw_weights(X_raw, stage_codes, feat_cols)
    top_kw = sorted(zip(feat_cols, kw_weights), key=lambda x: x[1], reverse=True)
    print(f'  Top KW features: {", ".join(f"{n}({w:.2f})" for n, w in top_kw[:5])}')

    X_kw = X_scaled * kw_weights[np.newaxis, :]

    # --- NCA ---
    print(f'  Fitting NCA (n_components={min(n_feat - 1, 20)})...')
    n_nca = min(n_feat - 1, 20)
    nca = NeighborhoodComponentsAnalysis(n_components=n_nca, max_iter=200,
                                         random_state=42, verbose=0)
    nca.fit(X_scaled, stage_codes)
    X_nca = nca.transform(X_scaled)

    # Analyze NCA transformation — which original features contribute most?
    L = nca.components_  # (n_nca, n_feat) linear transformation
    feat_importance_nca = np.linalg.norm(L, axis=0)  # per-feature contribution
    feat_importance_nca /= (feat_importance_nca.max() + 1e-20)
    top_nca = sorted(zip(feat_cols, feat_importance_nca), key=lambda x: x[1], reverse=True)
    print(f'  Top NCA features: {", ".join(f"{n}({w:.2f})" for n, w in top_nca[:5])}')

    # --- Embeddings ---
    methods = {}

    # Baseline: raw unsupervised
    for nn in [15, 30]:
        reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
        methods[f'raw_nn{nn}'] = reducer.fit_transform(X_scaled)

    # Supervised baseline
    for nn in [15, 30]:
        reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
        methods[f'sup_nn{nn}'] = reducer.fit_transform(X_scaled, y=stage_codes)

    # KW-weighted unsupervised
    for nn in [15, 30]:
        reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
        methods[f'kw_nn{nn}'] = reducer.fit_transform(X_kw)

    # NCA-transformed unsupervised
    for nn in [15, 30]:
        reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
        methods[f'nca_nn{nn}'] = reducer.fit_transform(X_nca)

    # NCA + KW combined: weight NCA dimensions by stage variance
    nca_kw_weights = np.std(X_nca, axis=0)
    # Compute per-NCA-dimension stage discrimination
    nca_dim_kw = np.ones(X_nca.shape[1])
    for j in range(X_nca.shape[1]):
        present = [sc for sc in STAGE_ORDER if (stage_codes == sc).sum() > 0]
        groups = [X_nca[stage_codes == sc, j] for sc in present]
        if len(groups) >= 2:
            try:
                stat, _ = kruskal(*groups)
                nca_dim_kw[j] = np.sqrt(max(stat, 0))
            except ValueError:
                pass
    nca_dim_kw /= (nca_dim_kw.max() + 1e-20)
    X_nca_kw = X_nca * nca_dim_kw[np.newaxis, :]
    for nn in [15, 30]:
        reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
        methods[f'nca_kw_nn{nn}'] = reducer.fit_transform(X_nca_kw)

    # --- Silhouette comparison ---
    remap_3c = {4: 0, 3: 1, 2: 1, 1: 1, 0: 2}
    codes_3c = np.array([remap_3c[c] for c in stage_codes])

    sil_results = []
    for method_name, X_emb in methods.items():
        sil_5 = silhouette_score(X_emb, stage_codes)
        sil_3 = silhouette_score(X_emb, codes_3c)
        sil_samples_arr = silhouette_samples(X_emb, stage_codes)
        row = {'method': method_name, 'sil_5class': sil_5, 'sil_3class': sil_3}
        for sc in STAGE_ORDER:
            mask = stage_codes == sc
            if mask.any():
                row[STAGE_NAMES[sc]] = float(np.mean(sil_samples_arr[mask]))
        sil_results.append(row)

    sil_df = pd.DataFrame(sil_results).sort_values('sil_5class', ascending=False)
    print(f'\n  Silhouette comparison ({label}):')
    print('  ' + sil_df.to_string(index=False, float_format='{:.3f}'.format).replace('\n', '\n  '))

    sil_df.to_csv(out_dir / f'{label}_informed_silhouette.csv', index=False)

    # --- Improvement analysis ---
    raw_best = sil_df[sil_df['method'].str.startswith('raw_')]['sil_5class'].max()
    kw_best = sil_df[sil_df['method'].str.startswith('kw_')]['sil_5class'].max()
    nca_best = sil_df[sil_df['method'].str.startswith('nca_nn')]['sil_5class'].max()
    nca_kw_best = sil_df[sil_df['method'].str.startswith('nca_kw_')]['sil_5class'].max()
    sup_best = sil_df[sil_df['method'].str.startswith('sup_')]['sil_5class'].max()

    print(f'\n  Summary:')
    print(f'    Raw unsupervised:  {raw_best:+.3f}')
    print(f'    KW-weighted:       {kw_best:+.3f}  (delta={kw_best - raw_best:+.3f})')
    print(f'    NCA-transformed:   {nca_best:+.3f}  (delta={nca_best - raw_best:+.3f})')
    print(f'    NCA+KW combined:   {nca_kw_best:+.3f}  (delta={nca_kw_best - raw_best:+.3f})')
    print(f'    Supervised (ceil): {sup_best:+.3f}')

    # --- Save plots for best informed method ---
    best_informed_name = sil_df[~sil_df['method'].str.startswith('sup_')].iloc[0]['method']
    best_informed_emb = methods[best_informed_name]

    plot_3d_stage(best_informed_emb, stage_codes,
                  f'{best_informed_name} — {label} — Stage',
                  out_dir / f'{label}_{best_informed_name}_stage.html')
    plot_3d_time(best_informed_emb, t_hr, stage_labels,
                 f'{best_informed_name} — {label} — Time',
                 out_dir / f'{label}_{best_informed_name}_time.html')
    plot_3d_thorax(best_informed_emb, thorax_rms,
                   f'{best_informed_name} — {label} — Thorax',
                   out_dir / f'{label}_{best_informed_name}_thorax.html')
    plot_trajectory(best_informed_emb, t_hr, stage_codes, stage_labels,
                    f'{best_informed_name} — {label} — Trajectory',
                    out_dir / f'{label}_{best_informed_name}_trajectory.html')

    # Also save NCA feature importance
    importance_df = pd.DataFrame({
        'feature': feat_cols,
        'kw_weight': kw_weights,
        'nca_importance': feat_importance_nca,
    }).sort_values('nca_importance', ascending=False)
    importance_df.to_csv(out_dir / f'{label}_feature_importance.csv', index=False)

    # --- Feature importance comparison plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # KW weights
    ax = axes[0]
    sorted_kw = sorted(zip(feat_cols, kw_weights), key=lambda x: x[1], reverse=True)
    names_kw = [x[0] for x in sorted_kw[:15]]
    vals_kw = [x[1] for x in sorted_kw[:15]]
    ax.barh(range(len(names_kw)), vals_kw, color='#3498DB', alpha=0.8)
    ax.set_yticks(range(len(names_kw)))
    ax.set_yticklabels(names_kw, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f'KW weights (sqrt H, normalized)', fontsize=9)
    ax.set_xlabel('Weight')
    ax.grid(True, alpha=0.2, axis='x')

    # NCA importance
    ax = axes[1]
    sorted_nca = sorted(zip(feat_cols, feat_importance_nca), key=lambda x: x[1], reverse=True)
    names_nca = [x[0] for x in sorted_nca[:15]]
    vals_nca = [x[1] for x in sorted_nca[:15]]
    ax.barh(range(len(names_nca)), vals_nca, color='#E74C3C', alpha=0.8)
    ax.set_yticks(range(len(names_nca)))
    ax.set_yticklabels(names_nca, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f'NCA importance (||L_j||)', fontsize=9)
    ax.set_xlabel('Importance')
    ax.grid(True, alpha=0.2, axis='x')

    plt.suptitle(f'Feature Importance: KW vs NCA — {label}', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f'{label}_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'label': label,
        'raw': raw_best,
        'kw': kw_best,
        'nca': nca_best,
        'nca_kw': nca_kw_best,
        'sup': sup_best,
        'best_informed': best_informed_name,
        'kw_delta': kw_best - raw_best,
        'nca_delta': nca_best - raw_best,
        'nca_kw_delta': nca_kw_best - raw_best,
        'gap_closed_best': (max(kw_best, nca_best, nca_kw_best) - raw_best) / (sup_best - raw_best + 1e-10),
    }


# ── Cross-session summary plot ──────────────────────────────────────────────

def plot_summary(results):
    labels = [r['label'] for r in results]
    raw_vals = [r['raw'] for r in results]
    kw_vals = [r['kw'] for r in results]
    nca_vals = [r['nca'] for r in results]
    nca_kw_vals = [r['nca_kw'] for r in results]
    sup_vals = [r['sup'] for r in results]

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 2*width, raw_vals, width, label='Raw unsupervised', color='#95A5A6', alpha=0.8)
    ax.bar(x - width, kw_vals, width, label='KW-weighted', color='#3498DB', alpha=0.8)
    ax.bar(x, nca_vals, width, label='NCA-transformed', color='#E74C3C', alpha=0.8)
    ax.bar(x + width, nca_kw_vals, width, label='NCA+KW', color='#9B59B6', alpha=0.8)
    ax.bar(x + 2*width, sup_vals, width, label='Supervised (ceiling)', color='#2ECC71', alpha=0.8)

    ax.set_xlabel('Session')
    ax.set_ylabel('Silhouette (5-class)')
    ax.set_title('Informed Unsupervised vs Baselines — All 12 Sessions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(loc='upper left', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'informed_unsupervised_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Gap-closed plot
    fig, ax = plt.subplots(figsize=(14, 5))
    gap_closed = [r['gap_closed_best'] * 100 for r in results]
    colors = ['#2ECC71' if g > 50 else '#E67E22' if g > 25 else '#E74C3C' for g in gap_closed]
    ax.bar(x, gap_closed, color=colors, alpha=0.8)
    ax.set_xlabel('Session')
    ax.set_ylabel('% gap closed (raw → supervised)')
    ax.set_title('How much of the supervised advantage does informed unsupervised recover?')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.axhline(50, ls='--', color='gray', alpha=0.5, label='50%')
    ax.axhline(100, ls='--', color='green', alpha=0.3, label='100% (matches supervised)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'informed_gap_closed.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    session_labels = [f'S{s}N{n}' for s in range(1, 7) for n in range(1, 3)]
    all_results = []

    for label in session_labels:
        try:
            result = run_session(label)
            all_results.append(result)
        except Exception as e:
            print(f'\n  ERROR on {label}: {e}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 70)
    print('CROSS-SESSION COMPARISON')
    print('=' * 70)
    print(f'  {"Sess":6s} {"Raw":>7s} {"KW":>7s} {"NCA":>7s} {"NCA+KW":>7s} {"Sup":>7s} {"Gap%":>6s}  Best informed')
    for r in all_results:
        print(f'  {r["label"]:6s} {r["raw"]:+.3f} {r["kw"]:+.3f} {r["nca"]:+.3f} '
              f'{r["nca_kw"]:+.3f} {r["sup"]:+.3f} {r["gap_closed_best"]*100:5.1f}%  {r["best_informed"]}')

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(REPORT_DIR / 'informed_comparison_summary.csv', index=False)

    plot_summary(all_results)

    # Aggregate stats
    gaps = [r['gap_closed_best'] for r in all_results]
    print(f'\n  Gap closed: mean={np.mean(gaps)*100:.1f}%, median={np.median(gaps)*100:.1f}%')
    print(f'  KW avg delta: {np.mean([r["kw_delta"] for r in all_results]):+.3f}')
    print(f'  NCA avg delta: {np.mean([r["nca_delta"] for r in all_results]):+.3f}')
    print(f'  NCA+KW avg delta: {np.mean([r["nca_kw_delta"] for r in all_results]):+.3f}')
    print(f'\n  Summary plots saved to {REPORT_DIR}')
