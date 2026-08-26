"""
Phase 3: Clustering on supervised UMAP embeddings.

For each session:
  1. Load saved features CSV
  2. Recompute supervised UMAP (nn=30) embedding
  3. Fit GMM (k=3,4,5) and DBSCAN (multiple eps) on the 3D embedding
  4. Evaluate: ARI, NMI, confusion matrix vs PSG stage labels
  5. Save interactive 3D HTML, confusion matrix PNGs, summary CSV

Outputs → reports/projections/<session>/clustering/
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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix,
)
import umap
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.sessions import SESSION_META

REPORT_DIR = ROOT / 'reports' / 'projections'
STAGE_NAMES = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_NAME_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

CLUSTER_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

META_COLS = {'t_hr', 'stage_code', 'stage_label', 'apnea_code', 'apnea_label', 'thorax_rms'}


def load_features(label):
    csv_path = REPORT_DIR / label / f'{label}_features.csv'
    df = pd.read_csv(csv_path)
    df = df[df['stage_code'] >= 0].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


def scale_features(df, feat_cols):
    X_raw = df[feat_cols].values.copy()
    X_raw[~np.isfinite(X_raw)] = np.nan
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X_raw[:, j] = col
    scaler = StandardScaler()
    return scaler.fit_transform(X_raw), X_raw


def compute_supervised_umap(X, stage_codes, nn=30):
    reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
    return reducer.fit_transform(X, y=stage_codes)


def optimal_label_mapping(true_labels, cluster_labels):
    """Map each cluster to the PSG stage it overlaps most with (Hungarian-style greedy)."""
    from scipy.optimize import linear_sum_assignment

    true_set = sorted(set(true_labels))
    clust_set = sorted(set(cluster_labels))
    if -1 in clust_set:
        clust_set = [c for c in clust_set if c != -1]

    cost = np.zeros((len(clust_set), len(true_set)))
    for ci, c in enumerate(clust_set):
        for si, s in enumerate(true_set):
            cost[ci, si] = -np.sum((cluster_labels == c) & (true_labels == s))

    if len(clust_set) <= len(true_set):
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {}
        for ri, ci in zip(row_ind, col_ind):
            mapping[clust_set[ri]] = true_set[ci]
    else:
        col_ind, row_ind = linear_sum_assignment(cost.T)
        mapping = {}
        for ci, ri in zip(col_ind, row_ind):
            mapping[clust_set[ri]] = true_set[ci]
        for c in clust_set:
            if c not in mapping:
                best_s = true_set[np.argmax([np.sum((cluster_labels == c) & (true_labels == s)) for s in true_set])]
                mapping[c] = best_s

    mapped = np.full_like(cluster_labels, -1)
    for c, s in mapping.items():
        mapped[cluster_labels == c] = s
    if -1 in cluster_labels:
        mapped[cluster_labels == -1] = -1
    return mapped, mapping


def plot_clusters_3d(X_emb, cluster_labels, df, title, filepath, n_clusters):
    fig = go.Figure()
    texts = [f't={t:.2f}h | {sl}' for t, sl in zip(df['t_hr'], df['stage_label'])]

    unique_labels = sorted(set(cluster_labels))
    for cl in unique_labels:
        mask = cluster_labels == cl
        if not mask.any():
            continue
        if cl == -1:
            name = 'Noise'
            color = '#999999'
            opacity = 0.3
            size = 2
        else:
            name = f'Cluster {cl}'
            color = CLUSTER_COLORS[cl % len(CLUSTER_COLORS)]
            opacity = 0.6
            size = 3
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=size, color=color, opacity=opacity),
            name=f'{name} (n={mask.sum()})',
            text=[texts[j] for j in np.where(mask)[0]],
            hoverinfo='text+name',
        ))
    fig.update_layout(
        title=title, width=900, height=700,
        scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(str(filepath))


def plot_confusion_matrix(cm, row_labels, col_labels, title, filepath):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('PSG Stage', fontsize=10)
    ax.set_ylabel('Cluster', fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = 'white' if val > cm.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=9, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary_panel(X_emb, stage_codes, cluster_labels, mapped_labels, method_name,
                       label, filepath):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})

    for sc in STAGE_ORDER:
        mask = stage_codes == sc
        if not mask.any():
            continue
        axes[0].scatter(X_emb[mask, 0], X_emb[mask, 1], X_emb[mask, 2],
                        c=STAGE_COLORS[sc], s=4, alpha=0.4, label=STAGE_NAMES[sc],
                        edgecolors='none')
    axes[0].set_title('PSG Ground Truth', fontsize=9)

    unique_cl = sorted(set(cluster_labels))
    for cl in unique_cl:
        mask = cluster_labels == cl
        if cl == -1:
            c, lbl = '#999999', 'Noise'
        else:
            c, lbl = CLUSTER_COLORS[cl % len(CLUSTER_COLORS)], f'C{cl}'
        axes[1].scatter(X_emb[mask, 0], X_emb[mask, 1], X_emb[mask, 2],
                        c=c, s=4, alpha=0.4, label=lbl, edgecolors='none')
    axes[1].set_title(f'{method_name} Clusters', fontsize=9)

    for sc in STAGE_ORDER:
        mask = mapped_labels == sc
        if not mask.any():
            continue
        axes[2].scatter(X_emb[mask, 0], X_emb[mask, 1], X_emb[mask, 2],
                        c=STAGE_COLORS[sc], s=4, alpha=0.4, label=STAGE_NAMES[sc],
                        edgecolors='none')
    noise_mask = mapped_labels == -1
    if noise_mask.any():
        axes[2].scatter(X_emb[noise_mask, 0], X_emb[noise_mask, 1], X_emb[noise_mask, 2],
                        c='#999999', s=2, alpha=0.3, label='Noise', edgecolors='none')
    axes[2].set_title('Mapped to Stages', fontsize=9)

    for ax in axes:
        ax.set_xlabel('D1', fontsize=7)
        ax.set_ylabel('D2', fontsize=7)
        ax.set_zlabel('D3', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=135)

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=STAGE_COLORS[sc],
                      markersize=7, label=STAGE_NAMES[sc]) for sc in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    plt.suptitle(f'{label} — {method_name}', fontsize=11, y=1.05)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def select_dbscan_eps(X_emb, candidates=None):
    """Pick eps values based on k-distance heuristic."""
    from sklearn.neighbors import NearestNeighbors
    if candidates is not None:
        return candidates
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_emb)
    dists, _ = nn.kneighbors(X_emb)
    k_dists = np.sort(dists[:, -1])
    p25 = k_dists[int(len(k_dists) * 0.25)]
    p50 = k_dists[int(len(k_dists) * 0.50)]
    p75 = k_dists[int(len(k_dists) * 0.75)]
    return [round(p25, 3), round(p50, 3), round(p75, 3)]


def run_session_clustering(label):
    print(f'\n{"=" * 60}')
    print(f'Phase 3 Clustering: {label}')
    print('=' * 60)

    df, feat_cols = load_features(label)
    print(f'  {len(df)} windows, {len(feat_cols)} features')

    X, X_raw = scale_features(df, feat_cols)
    stage_codes = df['stage_code'].values

    print(f'  Computing supervised UMAP (nn=30)...')
    X_emb = compute_supervised_umap(X, stage_codes, nn=30)

    clust_dir = REPORT_DIR / label / 'clustering'
    clust_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # --- GMM ---
    for k in [3, 4, 5]:
        method_name = f'GMM_k{k}'
        print(f'  {method_name}...')
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              n_init=5, random_state=42)
        cluster_labels = gmm.fit_predict(X_emb)

        sil = silhouette_score(X_emb, cluster_labels)
        ari = adjusted_rand_score(stage_codes, cluster_labels)
        nmi = normalized_mutual_info_score(stage_codes, cluster_labels)

        mapped, mapping = optimal_label_mapping(stage_codes, cluster_labels)

        valid = mapped >= 0
        if valid.any():
            ari_mapped = adjusted_rand_score(stage_codes[valid], mapped[valid])
        else:
            ari_mapped = ari

        present_stages = sorted(set(stage_codes))
        present_clusters = sorted(set(cluster_labels))
        cm = confusion_matrix(stage_codes, cluster_labels,
                              labels=present_stages)

        row_labels_cm = [f'C{c}' for c in present_clusters]
        col_labels_cm = [STAGE_NAMES.get(s, str(s)) for s in present_stages]
        cm_clust_vs_stage = np.zeros((len(present_clusters), len(present_stages)), dtype=int)
        for ci, c in enumerate(present_clusters):
            for si, s in enumerate(present_stages):
                cm_clust_vs_stage[ci, si] = int(np.sum((cluster_labels == c) & (stage_codes == s)))

        plot_confusion_matrix(
            cm_clust_vs_stage, row_labels_cm, col_labels_cm,
            f'{label} — {method_name} vs PSG\nARI={ari:.3f}  NMI={nmi:.3f}  Sil={sil:.3f}',
            clust_dir / f'{label}_{method_name}_confusion.png',
        )
        plot_clusters_3d(
            X_emb, cluster_labels, df,
            f'{label} — {method_name} (k={k}, ARI={ari:.3f})',
            clust_dir / f'{label}_{method_name}_3d.html',
            n_clusters=k,
        )
        plot_summary_panel(
            X_emb, stage_codes, cluster_labels, mapped, method_name, label,
            clust_dir / f'{label}_{method_name}_panel.png',
        )

        bic = gmm.bic(X_emb)
        print(f'    ARI={ari:.3f}  NMI={nmi:.3f}  Sil={sil:.3f}  BIC={bic:.0f}')
        print(f'    Mapping: {{{", ".join(f"C{c}->{STAGE_NAMES.get(s,str(s))}" for c,s in sorted(mapping.items()))}}}')

        results.append({
            'session': label,
            'method': method_name,
            'n_clusters': k,
            'ARI': ari,
            'NMI': nmi,
            'silhouette': sil,
            'BIC': bic,
            'noise_pct': 0.0,
            'mapping': str(mapping),
        })

    # --- DBSCAN ---
    eps_values = select_dbscan_eps(X_emb)
    print(f'  DBSCAN eps candidates (k-dist 25/50/75th): {eps_values}')

    for eps in eps_values:
        method_name = f'DBSCAN_eps{eps}'
        print(f'  {method_name}...')
        db = DBSCAN(eps=eps, min_samples=5)
        cluster_labels = db.fit_predict(X_emb)

        n_clusters = len(set(cluster_labels) - {-1})
        noise_pct = 100.0 * (cluster_labels == -1).sum() / len(cluster_labels)

        if n_clusters < 2:
            print(f'    Only {n_clusters} cluster(s), {noise_pct:.1f}% noise — skipping metrics')
            results.append({
                'session': label,
                'method': method_name,
                'n_clusters': n_clusters,
                'ARI': np.nan,
                'NMI': np.nan,
                'silhouette': np.nan,
                'BIC': np.nan,
                'noise_pct': noise_pct,
                'mapping': '',
            })
            continue

        non_noise = cluster_labels >= 0
        sil = silhouette_score(X_emb[non_noise], cluster_labels[non_noise]) if non_noise.sum() > 1 else np.nan
        ari = adjusted_rand_score(stage_codes[non_noise], cluster_labels[non_noise]) if non_noise.sum() > 1 else np.nan
        nmi = normalized_mutual_info_score(stage_codes[non_noise], cluster_labels[non_noise]) if non_noise.sum() > 1 else np.nan

        mapped, mapping = optimal_label_mapping(stage_codes, cluster_labels)

        present_stages = sorted(set(stage_codes))
        present_clusters = sorted(set(cluster_labels))
        row_labels_cm = [f'C{c}' if c >= 0 else 'Noise' for c in present_clusters]
        col_labels_cm = [STAGE_NAMES.get(s, str(s)) for s in present_stages]
        cm_clust_vs_stage = np.zeros((len(present_clusters), len(present_stages)), dtype=int)
        for ci, c in enumerate(present_clusters):
            for si, s in enumerate(present_stages):
                cm_clust_vs_stage[ci, si] = int(np.sum((cluster_labels == c) & (stage_codes == s)))

        plot_confusion_matrix(
            cm_clust_vs_stage, row_labels_cm, col_labels_cm,
            f'{label} — {method_name}\nARI={ari:.3f}  NMI={nmi:.3f}  Noise={noise_pct:.1f}%',
            clust_dir / f'{label}_{method_name}_confusion.png',
        )
        plot_clusters_3d(
            X_emb, cluster_labels, df,
            f'{label} — {method_name} ({n_clusters} clusters, {noise_pct:.0f}% noise)',
            clust_dir / f'{label}_{method_name}_3d.html',
            n_clusters=n_clusters,
        )
        plot_summary_panel(
            X_emb, stage_codes, cluster_labels, mapped, method_name, label,
            clust_dir / f'{label}_{method_name}_panel.png',
        )

        print(f'    {n_clusters} clusters, {noise_pct:.1f}% noise  ARI={ari:.3f}  NMI={nmi:.3f}  Sil={sil:.3f}')
        print(f'    Mapping: {{{", ".join(f"C{c}->{STAGE_NAMES.get(s,str(s))}" for c,s in sorted(mapping.items()) if c >= 0)}}}')

        results.append({
            'session': label,
            'method': method_name,
            'n_clusters': n_clusters,
            'ARI': ari,
            'NMI': nmi,
            'silhouette': sil,
            'BIC': np.nan,
            'noise_pct': noise_pct,
            'mapping': str(mapping),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(clust_dir / f'{label}_clustering_results.csv', index=False)

    best = res_df.dropna(subset=['ARI']).sort_values('ARI', ascending=False).iloc[0] if not res_df.dropna(subset=['ARI']).empty else None
    if best is not None:
        print(f'\n  Best: {best["method"]}  ARI={best["ARI"]:.3f}  NMI={best["NMI"]:.3f}')

    print(f'  Output: {clust_dir}')
    return results


if __name__ == '__main__':
    all_results = []
    session_labels = [m['label'] for m in SESSION_META]

    for label in session_labels:
        try:
            results = run_session_clustering(label)
            all_results.extend(results)
        except Exception as e:
            print(f'\n  ERROR on {label}: {e}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 70)
    print('CROSS-SESSION CLUSTERING SUMMARY')
    print('=' * 70)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(REPORT_DIR / 'clustering_summary.csv', index=False)

    gmm5 = summary_df[summary_df['method'] == 'GMM_k5'].copy()
    if not gmm5.empty:
        print('\nGMM k=5 (matches 5 PSG stages):')
        for _, r in gmm5.iterrows():
            print(f'  {r["session"]:6s}  ARI={r["ARI"]:.3f}  NMI={r["NMI"]:.3f}  Sil={r["silhouette"]:.3f}')
        print(f'  Mean ARI={gmm5["ARI"].mean():.3f}  Mean NMI={gmm5["NMI"].mean():.3f}')

    gmm3 = summary_df[summary_df['method'] == 'GMM_k3'].copy()
    if not gmm3.empty:
        print('\nGMM k=3 (Wake/NREM/REM):')
        for _, r in gmm3.iterrows():
            print(f'  {r["session"]:6s}  ARI={r["ARI"]:.3f}  NMI={r["NMI"]:.3f}  Sil={r["silhouette"]:.3f}')
        print(f'  Mean ARI={gmm3["ARI"].mean():.3f}  Mean NMI={gmm3["NMI"].mean():.3f}')

    print(f'\nSummary saved to {REPORT_DIR / "clustering_summary.csv"}')
