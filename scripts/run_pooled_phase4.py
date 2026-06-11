"""
Phase 4: Pooled cross-session projection analysis.

1. Load all 12 sessions' feature CSVs
2. Subject-level z-score normalization (both nights share one scaler per subject)
3. Pooled PCA, UMAP, supervised UMAP, t-SNE on all windows
4. Color by: stage, subject, session, time-of-night
5. GMM k=4 on pooled supervised UMAP embedding (ARI, NMI vs PSG)
6. LOSO evaluation: for each held-out subject, fit UMAP+GMM on 5 subjects, project+predict held-out

Outputs → reports/projections/pooled/
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix,
)
from scipy.optimize import linear_sum_assignment
import umap
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.sessions import SESSION_META

REPORT_DIR = ROOT / 'reports' / 'projections' / 'pooled'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_DIR = ROOT / 'reports' / 'projections'
META_COLS = {'t_hr', 'stage_code', 'stage_label', 'apnea_code', 'apnea_label', 'thorax_rms'}

STAGE_NAMES = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_NAME_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

SESSION_LABELS = [f'S{(i//2)+1}N{(i%2)+1}' for i in range(12)]
SUBJECT_IDS = [f'S{i}' for i in range(1, 7)]
SUBJECT_COLORS = {
    'S1': '#1f77b4', 'S2': '#ff7f0e', 'S3': '#2ca02c',
    'S4': '#d62728', 'S5': '#9467bd', 'S6': '#8c564b',
}
SESSION_COLORS = {
    'S1N1': '#1f77b4', 'S1N2': '#aec7e8',
    'S2N1': '#ff7f0e', 'S2N2': '#ffbb78',
    'S3N1': '#2ca02c', 'S3N2': '#98df8a',
    'S4N1': '#d62728', 'S4N2': '#ff9896',
    'S5N1': '#9467bd', 'S5N2': '#c5b0d5',
    'S6N1': '#8c564b', 'S6N2': '#c49c94',
}

CLUSTER_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_features():
    """Load and concatenate feature CSVs from all 12 sessions."""
    frames = []
    for label in SESSION_LABELS:
        csv_path = FEAT_DIR / label / f'{label}_features.csv'
        if not csv_path.exists():
            print(f'  WARNING: {csv_path} not found, skipping')
            continue
        df = pd.read_csv(csv_path)
        df = df[df['stage_code'] >= 0].reset_index(drop=True)
        subject_num = int(label[1])
        df['session'] = label
        df['subject'] = f'S{subject_num}'
        frames.append(df)
        print(f'  {label}: {len(df)} windows')
    pooled = pd.concat(frames, ignore_index=True)
    print(f'  Total: {len(pooled)} windows from {len(frames)} sessions')
    return pooled


def get_feat_cols(df):
    """Return feature column names (exclude meta + session/subject)."""
    exclude = META_COLS | {'session', 'subject'}
    return [c for c in df.columns if c not in exclude]


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_per_subject(df, feat_cols):
    """Z-score normalize features per subject (both nights share one scaler)."""
    df_norm = df.copy()
    for subj in df['subject'].unique():
        mask = df['subject'] == subj
        X = df.loc[mask, feat_cols].values.copy()
        X[~np.isfinite(X)] = np.nan
        for j in range(X.shape[1]):
            col = X[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                med = np.nanmedian(col)
                col[nan_mask] = med if not np.isnan(med) else 0.0
                X[:, j] = col
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_norm.loc[mask, feat_cols] = X_scaled
    return df_norm


def global_scale(X_raw):
    """Clean NaN/Inf and standard-scale globally."""
    X = X_raw.copy()
    X[~np.isfinite(X)] = np.nan
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X[:, j] = col
    return StandardScaler().fit_transform(X)


# ── Projection methods ─────────────────────────────────────────────────────────

def run_pca(X, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    emb = pca.fit_transform(X)
    print(f'  PCA explained variance: {pca.explained_variance_ratio_[:3].sum():.1%}')
    return emb, pca


def run_umap(X, nn=30):
    reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
    return reducer.fit_transform(X)


def run_supervised_umap(X, y, nn=30):
    reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
    return reducer.fit_transform(X, y=y)


def run_tsne(X, perplexity=30):
    tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=1000,
                init='pca', random_state=42)
    return tsne.fit_transform(X)


# ── Clustering ─────────────────────────────────────────────────────────────────

def optimal_label_mapping(true_labels, cluster_labels):
    true_set = sorted(set(true_labels))
    clust_set = sorted(c for c in set(cluster_labels) if c != -1)

    cost = np.zeros((len(clust_set), len(true_set)))
    for ci, c in enumerate(clust_set):
        for si, s in enumerate(true_set):
            cost[ci, si] = -np.sum((cluster_labels == c) & (true_labels == s))

    if len(clust_set) <= len(true_set):
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {clust_set[ri]: true_set[ci] for ri, ci in zip(row_ind, col_ind)}
    else:
        col_ind, row_ind = linear_sum_assignment(cost.T)
        mapping = {clust_set[ri]: true_set[ci] for ci, ri in zip(col_ind, row_ind)}
        for c in clust_set:
            if c not in mapping:
                best_s = true_set[np.argmax([np.sum((cluster_labels == c) & (true_labels == s))
                                             for s in true_set])]
                mapping[c] = best_s

    mapped = np.full_like(cluster_labels, -1)
    for c, s in mapping.items():
        mapped[cluster_labels == c] = s
    if -1 in cluster_labels:
        mapped[cluster_labels == -1] = -1
    return mapped, mapping


def fit_gmm(X_emb, k=4):
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                           n_init=5, random_state=42)
    labels = gmm.fit_predict(X_emb)
    return labels, gmm


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_3d_categorical(X_emb, labels, color_map, legend_order, title, filepath,
                        hover_texts=None):
    """Interactive 3D scatter colored by a categorical variable."""
    fig = go.Figure()
    for lbl in legend_order:
        mask = labels == lbl
        if not mask.any():
            continue
        color = color_map.get(lbl, '#999999')
        kw = {}
        if hover_texts is not None:
            kw['text'] = [hover_texts[j] for j in np.where(mask)[0]]
            kw['hoverinfo'] = 'text+name'
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.5),
            name=f'{lbl} (n={mask.sum()})',
            **kw,
        ))
    fig.update_layout(
        title=title, width=1000, height=750,
        scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(str(filepath))


def plot_3d_continuous(X_emb, values, title, filepath, colorscale='Viridis',
                       hover_texts=None):
    """Interactive 3D scatter colored by a continuous variable."""
    fig = go.Figure()
    kw = {}
    if hover_texts is not None:
        kw['text'] = hover_texts
        kw['hoverinfo'] = 'text'
    fig.add_trace(go.Scatter3d(
        x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
        mode='markers',
        marker=dict(size=2, color=values, colorscale=colorscale,
                    opacity=0.5, colorbar=dict(title='value')),
        **kw,
    ))
    fig.update_layout(
        title=title, width=1000, height=750,
        scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
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
    ax.set_ylabel('Cluster / Predicted', fontsize=10)
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


def plot_static_panel(X_emb, stage_codes, subjects, method_name, filepath):
    """3-panel static plot: stages, subjects, time."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': '3d'})

    for sc in STAGE_ORDER:
        mask = stage_codes == sc
        if not mask.any():
            continue
        axes[0].scatter(X_emb[mask, 0], X_emb[mask, 1], X_emb[mask, 2],
                        c=STAGE_COLORS[sc], s=2, alpha=0.3, label=STAGE_NAMES[sc],
                        edgecolors='none')
    axes[0].set_title('By Sleep Stage', fontsize=10)
    axes[0].legend(fontsize=7, markerscale=3, loc='upper left')

    for subj in SUBJECT_IDS:
        mask = subjects == subj
        if not mask.any():
            continue
        axes[1].scatter(X_emb[mask, 0], X_emb[mask, 1], X_emb[mask, 2],
                        c=SUBJECT_COLORS[subj], s=2, alpha=0.3, label=subj,
                        edgecolors='none')
    axes[1].set_title('By Subject', fontsize=10)
    axes[1].legend(fontsize=7, markerscale=3, loc='upper left')

    sc = axes[2].scatter(X_emb[:, 0], X_emb[:, 1], X_emb[:, 2],
                         c=np.arange(len(X_emb)), cmap='viridis', s=2, alpha=0.3,
                         edgecolors='none')
    axes[2].set_title('By Window Index (temporal)', fontsize=10)

    fig.suptitle(f'Pooled {method_name} — All 12 Sessions (subject-normalized)', fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── LOSO cross-validation ─────────────────────────────────────────────────────

def loso_evaluation(df_norm, feat_cols, k=4):
    """
    Leave-one-subject-out: for each held-out subject,
    fit supervised UMAP + GMM on remaining 5 subjects,
    transform + predict held-out subject.
    """
    subjects = sorted(df_norm['subject'].unique())
    results = []

    for held_out in subjects:
        train_mask = df_norm['subject'] != held_out
        test_mask = df_norm['subject'] == held_out
        X_train = df_norm.loc[train_mask, feat_cols].values
        X_test = df_norm.loc[test_mask, feat_cols].values
        y_train = df_norm.loc[train_mask, 'stage_code'].values
        y_test = df_norm.loc[test_mask, 'stage_code'].values
        n_test = len(X_test)

        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42)
        emb_train = reducer.fit_transform(X_train, y=y_train)

        emb_test = reducer.transform(X_test)

        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               n_init=5, random_state=42)
        gmm.fit(emb_train)

        train_labels = gmm.predict(emb_train)
        mapped_train, mapping = optimal_label_mapping(y_train, train_labels)

        test_cluster = gmm.predict(emb_test)
        test_mapped = np.full_like(test_cluster, -1)
        for c, s in mapping.items():
            test_mapped[test_cluster == c] = s

        valid = test_mapped >= 0
        if valid.sum() > 0:
            ari = adjusted_rand_score(y_test[valid], test_mapped[valid])
            nmi = normalized_mutual_info_score(y_test[valid], test_mapped[valid])
            acc = np.mean(y_test[valid] == test_mapped[valid])
        else:
            ari = nmi = acc = 0.0

        sessions_held = sorted(df_norm.loc[test_mask, 'session'].unique())
        results.append({
            'held_out': held_out,
            'sessions': ', '.join(sessions_held),
            'n_windows': n_test,
            'ari': ari,
            'nmi': nmi,
            'accuracy': acc,
        })
        print(f'  LOSO {held_out}: ARI={ari:.3f}  NMI={nmi:.3f}  Acc={acc:.1%}  (n={n_test})')

    return pd.DataFrame(results)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Phase 4: Pooled Cross-Session Projection Analysis')
    print('=' * 60)

    # Load all feature CSVs
    print('\n1. Loading features...')
    df = load_all_features()
    feat_cols = get_feat_cols(df)
    print(f'   {len(feat_cols)} features: {feat_cols[:5]}...')

    # Subject-level normalization
    print('\n2. Subject-level z-score normalization...')
    df_norm = normalize_per_subject(df, feat_cols)
    X = df_norm[feat_cols].values
    stage_codes = df_norm['stage_code'].values.astype(int)
    subjects = df_norm['subject'].values
    sessions = df_norm['session'].values
    t_hr = df_norm['t_hr'].values

    hover = [f'{ses} t={t:.2f}h {sl}' for ses, t, sl
             in zip(sessions, t_hr, df_norm['stage_label'])]

    stage_color_map = {STAGE_NAMES[sc]: STAGE_COLORS[sc] for sc in STAGE_ORDER}
    stage_labels_arr = np.array([STAGE_NAMES[sc] for sc in stage_codes])

    # PCA
    print('\n3. Running PCA...')
    emb_pca, pca_model = run_pca(X)
    plot_3d_categorical(emb_pca, stage_labels_arr, stage_color_map, STAGE_NAME_ORDER,
                        'Pooled PCA — By Stage', REPORT_DIR / 'pooled_PCA_stage.html', hover)
    plot_3d_categorical(emb_pca, subjects, SUBJECT_COLORS, SUBJECT_IDS,
                        'Pooled PCA — By Subject', REPORT_DIR / 'pooled_PCA_subject.html', hover)
    plot_3d_continuous(emb_pca, t_hr, 'Pooled PCA — Time of Night',
                       REPORT_DIR / 'pooled_PCA_time.html', hover_texts=hover)
    plot_static_panel(emb_pca, stage_codes, subjects, 'PCA', REPORT_DIR / 'pooled_PCA_panel.png')

    # PCA loadings & variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(min(20, len(pca_model.explained_variance_ratio_))),
                pca_model.explained_variance_ratio_[:20])
    axes[0].set_xlabel('PC')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA Scree Plot (pooled)')
    loadings = pd.DataFrame(pca_model.components_[:3].T, index=feat_cols,
                            columns=['PC1', 'PC2', 'PC3'])
    top_feats = loadings.abs().sum(axis=1).nlargest(15).index
    loadings.loc[top_feats].plot.barh(ax=axes[1])
    axes[1].set_title('Top 15 Feature Loadings (PC1-3)')
    axes[1].set_xlabel('Loading')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'pooled_PCA_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # UMAP (unsupervised)
    print('\n4. Running UMAP (unsupervised, nn=30)...')
    emb_umap = run_umap(X, nn=30)
    sil_umap = silhouette_score(emb_umap, stage_codes)
    print(f'   Silhouette (5-class): {sil_umap:.3f}')
    plot_3d_categorical(emb_umap, stage_labels_arr, stage_color_map, STAGE_NAME_ORDER,
                        f'Pooled UMAP nn=30 — Stage (sil={sil_umap:.3f})',
                        REPORT_DIR / 'pooled_UMAP_stage.html', hover)
    plot_3d_categorical(emb_umap, subjects, SUBJECT_COLORS, SUBJECT_IDS,
                        'Pooled UMAP nn=30 — Subject',
                        REPORT_DIR / 'pooled_UMAP_subject.html', hover)
    plot_3d_categorical(emb_umap, sessions, SESSION_COLORS, SESSION_LABELS,
                        'Pooled UMAP nn=30 — Session',
                        REPORT_DIR / 'pooled_UMAP_session.html', hover)
    plot_static_panel(emb_umap, stage_codes, subjects, 'UMAP nn=30',
                      REPORT_DIR / 'pooled_UMAP_panel.png')

    # Supervised UMAP
    print('\n5. Running supervised UMAP (nn=30)...')
    emb_sup = run_supervised_umap(X, stage_codes, nn=30)
    sil_sup = silhouette_score(emb_sup, stage_codes)
    print(f'   Silhouette (5-class): {sil_sup:.3f}')
    plot_3d_categorical(emb_sup, stage_labels_arr, stage_color_map, STAGE_NAME_ORDER,
                        f'Pooled Supervised UMAP nn=30 — Stage (sil={sil_sup:.3f})',
                        REPORT_DIR / 'pooled_UMAP_sup_stage.html', hover)
    plot_3d_categorical(emb_sup, subjects, SUBJECT_COLORS, SUBJECT_IDS,
                        'Pooled Supervised UMAP nn=30 — Subject (batch effect check)',
                        REPORT_DIR / 'pooled_UMAP_sup_subject.html', hover)
    plot_3d_categorical(emb_sup, sessions, SESSION_COLORS, SESSION_LABELS,
                        'Pooled Supervised UMAP nn=30 — Session',
                        REPORT_DIR / 'pooled_UMAP_sup_session.html', hover)
    plot_static_panel(emb_sup, stage_codes, subjects, 'Supervised UMAP nn=30',
                      REPORT_DIR / 'pooled_UMAP_sup_panel.png')

    # t-SNE
    print('\n6. Running t-SNE (perplexity=30)...')
    emb_tsne = run_tsne(X, perplexity=30)
    sil_tsne = silhouette_score(emb_tsne, stage_codes)
    print(f'   Silhouette (5-class): {sil_tsne:.3f}')
    plot_3d_categorical(emb_tsne, stage_labels_arr, stage_color_map, STAGE_NAME_ORDER,
                        f'Pooled t-SNE p=30 — Stage (sil={sil_tsne:.3f})',
                        REPORT_DIR / 'pooled_tSNE_stage.html', hover)
    plot_3d_categorical(emb_tsne, subjects, SUBJECT_COLORS, SUBJECT_IDS,
                        'Pooled t-SNE p=30 — Subject',
                        REPORT_DIR / 'pooled_tSNE_subject.html', hover)
    plot_static_panel(emb_tsne, stage_codes, subjects, 't-SNE p=30',
                      REPORT_DIR / 'pooled_tSNE_panel.png')

    # GMM clustering on pooled supervised UMAP
    print('\n7. GMM clustering on pooled supervised UMAP...')
    for k in [3, 4, 5]:
        cl_labels, gmm = fit_gmm(emb_sup, k=k)
        mapped, mapping = optimal_label_mapping(stage_codes, cl_labels)
        ari = adjusted_rand_score(stage_codes, mapped)
        nmi = normalized_mutual_info_score(stage_codes, mapped)
        sil = silhouette_score(emb_sup, cl_labels)
        print(f'   GMM k={k}: ARI={ari:.3f}  NMI={nmi:.3f}  Sil={sil:.3f}')

        stage_set = sorted(set(stage_codes))
        stage_names_sorted = [STAGE_NAMES[s] for s in stage_set]
        cm = confusion_matrix(stage_codes, mapped, labels=stage_set)
        plot_confusion_matrix(cm, stage_names_sorted, stage_names_sorted,
                              f'Pooled GMM k={k} — ARI={ari:.3f} NMI={nmi:.3f}',
                              REPORT_DIR / f'pooled_GMM_k{k}_confusion.png')

        cat_labels = np.array([f'C{c}' for c in cl_labels])
        cat_colors = {f'C{i}': CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)}
        cat_order = [f'C{i}' for i in range(k)]
        plot_3d_categorical(emb_sup, cat_labels, cat_colors, cat_order,
                            f'Pooled GMM k={k} Clusters (ARI={ari:.3f})',
                            REPORT_DIR / f'pooled_GMM_k{k}_3d.html', hover)

    # LOSO cross-validation
    print('\n8. LOSO cross-validation (supervised UMAP + GMM k=4)...')
    loso_df = loso_evaluation(df_norm, feat_cols, k=4)
    loso_df.to_csv(REPORT_DIR / 'pooled_LOSO_results.csv', index=False)
    print(f'\n   Mean ARI:  {loso_df["ari"].mean():.3f} ± {loso_df["ari"].std():.3f}')
    print(f'   Mean NMI:  {loso_df["nmi"].mean():.3f} ± {loso_df["nmi"].std():.3f}')
    print(f'   Mean Acc:  {loso_df["accuracy"].mean():.1%} ± {loso_df["accuracy"].std():.1%}')

    # LOSO bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = range(len(loso_df))
    labels = loso_df['held_out']
    for ax, metric, title in zip(axes, ['ari', 'nmi', 'accuracy'],
                                  ['Adjusted Rand Index', 'Normalized Mutual Info', 'Accuracy']):
        bars = ax.bar(x, loso_df[metric], color=[SUBJECT_COLORS[s] for s in labels])
        ax.axhline(loso_df[metric].mean(), ls='--', color='k', alpha=0.5,
                    label=f'mean={loso_df[metric].mean():.3f}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend()
    fig.suptitle('LOSO Cross-Validation: Supervised UMAP + GMM k=4', fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'pooled_LOSO_barplot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Summary silhouette scores
    summary = pd.DataFrame({
        'method': ['PCA', 'UMAP_nn30', 'UMAP_sup_nn30', 'tSNE_p30'],
        'sil_5class': [silhouette_score(emb_pca, stage_codes), sil_umap, sil_sup, sil_tsne],
    })
    summary.to_csv(REPORT_DIR / 'pooled_silhouette_scores.csv', index=False)
    print('\n9. Silhouette scores (pooled):')
    for _, row in summary.iterrows():
        print(f'   {row["method"]}: {row["sil_5class"]:.3f}')

    # Batch effect quantification
    print('\n10. Batch effect check (subject silhouette on unsupervised UMAP)...')
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    subj_codes = le.fit_transform(subjects)
    sil_subj = silhouette_score(emb_umap, subj_codes)
    sil_subj_sup = silhouette_score(emb_sup, subj_codes)
    print(f'    Subject silhouette on UMAP:            {sil_subj:.3f}')
    print(f'    Subject silhouette on supervised UMAP: {sil_subj_sup:.3f}')
    print(f'    (lower = less batch effect, higher = subjects cluster separately)')

    print(f'\nAll outputs saved to {REPORT_DIR}/')
    print('Done.')


if __name__ == '__main__':
    main()
