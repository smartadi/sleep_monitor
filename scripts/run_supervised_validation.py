"""
Supervised UMAP train/test validation.

For each session:
  1. Split windows into train (labeled) and test (held-out)
  2. Fit supervised UMAP on train only
  3. Transform test data into the same embedding space
  4. Fit GMM on train embedding, predict test embedding
  5. Evaluate: silhouette on test, ARI/NMI/accuracy vs PSG labels
  6. Repeat for train fractions: 25%, 50%, 75%

Outputs → reports/projections/validation/
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

REPORT_DIR = ROOT / 'reports' / 'projections' / 'validation'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_DIR = ROOT / 'reports' / 'projections'
META_COLS = {'t_hr', 'stage_code', 'stage_label', 'apnea_code', 'apnea_label', 'thorax_rms'}

STAGE_NAMES = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_NAME_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']
SESSION_LABELS = [f'S{(i//2)+1}N{(i%2)+1}' for i in range(12)]

TRAIN_FRACTIONS = [0.25, 0.50, 0.75]


def load_features(label):
    csv_path = FEAT_DIR / label / f'{label}_features.csv'
    df = pd.read_csv(csv_path)
    df = df[df['stage_code'] >= 0].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


def scale_features(X_raw):
    X = X_raw.copy()
    X[~np.isfinite(X)] = np.nan
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X[:, j] = col
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


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
    return mapped, mapping


def plot_train_test_3d(emb_train, emb_test, y_train, y_test, title, filepath):
    """Interactive 3D showing train (solid) and test (transparent) colored by stage."""
    fig = go.Figure()
    for sc in STAGE_ORDER:
        mask_tr = y_train == sc
        mask_te = y_test == sc
        name = STAGE_NAMES[sc]
        color = STAGE_COLORS[sc]
        if mask_tr.any():
            fig.add_trace(go.Scatter3d(
                x=emb_train[mask_tr, 0], y=emb_train[mask_tr, 1], z=emb_train[mask_tr, 2],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.6),
                name=f'{name} train (n={mask_tr.sum()})',
            ))
        if mask_te.any():
            fig.add_trace(go.Scatter3d(
                x=emb_test[mask_te, 0], y=emb_test[mask_te, 1], z=emb_test[mask_te, 2],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.3,
                            symbol='diamond'),
                name=f'{name} test (n={mask_te.sum()})',
            ))
    fig.update_layout(
        title=title, width=1000, height=750,
        scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(str(filepath))


def plot_confusion(cm, title, filepath):
    stage_names = [STAGE_NAMES[s] for s in sorted(set(range(5)))]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, fontsize=9)
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names, fontsize=9)
    ax.set_xlabel('PSG Stage')
    ax.set_ylabel('Predicted')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = 'white' if val > cm.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_session_validation(label, df, feat_cols):
    """Run train/test validation for one session at multiple train fractions."""
    X_raw = df[feat_cols].values
    X_scaled, scaler = scale_features(X_raw)
    stage_codes = df['stage_code'].values.astype(int)
    n = len(df)

    session_dir = REPORT_DIR / label
    session_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)

    for frac in TRAIN_FRACTIONS:
        n_train = int(n * frac)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        y_train = stage_codes[train_idx]
        y_test = stage_codes[test_idx]

        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42)
        emb_train = reducer.fit_transform(X_train, y=y_train)
        emb_test = reducer.transform(X_test)

        # Silhouette on test embedding using true labels
        if len(set(y_test)) > 1:
            sil_test = silhouette_score(emb_test, y_test)
        else:
            sil_test = np.nan

        # GMM on train embedding → predict test
        gmm = GaussianMixture(n_components=4, covariance_type='full',
                               n_init=5, random_state=42)
        gmm.fit(emb_train)

        train_cl = gmm.predict(emb_train)
        _, mapping = optimal_label_mapping(y_train, train_cl)

        test_cl = gmm.predict(emb_test)
        test_mapped = np.full_like(test_cl, -1)
        for c, s in mapping.items():
            test_mapped[test_cl == c] = s

        valid = test_mapped >= 0
        if valid.sum() > 0:
            ari = adjusted_rand_score(y_test[valid], test_mapped[valid])
            nmi = normalized_mutual_info_score(y_test[valid], test_mapped[valid])
            acc = np.mean(y_test[valid] == test_mapped[valid])
        else:
            ari = nmi = acc = 0.0

        frac_pct = int(frac * 100)
        results.append({
            'session': label,
            'train_frac': frac,
            'n_train': n_train,
            'n_test': len(test_idx),
            'sil_test': sil_test,
            'ari_test': ari,
            'nmi_test': nmi,
            'acc_test': acc,
        })

        # 3D plot: train vs test
        plot_train_test_3d(
            emb_train, emb_test, y_train, y_test,
            f'{label} Supervised UMAP — {frac_pct}% train / {100-frac_pct}% test (test sil={sil_test:.3f}, ARI={ari:.3f})',
            session_dir / f'{label}_traintest_{frac_pct}pct.html',
        )

        # Confusion matrix on test set
        stage_set = sorted(set(stage_codes))
        cm = confusion_matrix(y_test[valid], test_mapped[valid], labels=stage_set)
        plot_confusion(
            cm,
            f'{label} Test Set — {frac_pct}% train (ARI={ari:.3f}, Acc={acc:.1%})',
            session_dir / f'{label}_confusion_{frac_pct}pct.png',
        )

        print(f'  {label} {frac_pct:3d}% train: sil_test={sil_test:.3f}  ARI={ari:.3f}  '
              f'NMI={nmi:.3f}  Acc={acc:.1%}  (n_test={len(test_idx)})')

    return results


def main():
    print('Supervised UMAP Train/Test Validation')
    print('=' * 60)

    all_results = []

    for label in SESSION_LABELS:
        csv_path = FEAT_DIR / label / f'{label}_features.csv'
        if not csv_path.exists():
            print(f'\n  WARNING: {label} features not found, skipping')
            continue
        print(f'\n{label}:')
        df, feat_cols = load_features(label)
        results = run_session_validation(label, df, feat_cols)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORT_DIR / 'validation_results.csv', index=False)

    # Summary table
    print('\n' + '=' * 80)
    print('SUMMARY: Mean metrics across all sessions')
    print('=' * 80)
    for frac in TRAIN_FRACTIONS:
        sub = results_df[results_df['train_frac'] == frac]
        print(f'\n  Train {int(frac*100)}%:')
        print(f'    Silhouette (test): {sub["sil_test"].mean():.3f} +/- {sub["sil_test"].std():.3f}')
        print(f'    ARI (test):        {sub["ari_test"].mean():.3f} +/- {sub["ari_test"].std():.3f}')
        print(f'    NMI (test):        {sub["nmi_test"].mean():.3f} +/- {sub["nmi_test"].std():.3f}')
        print(f'    Accuracy (test):   {sub["acc_test"].mean():.1%} +/- {sub["acc_test"].std():.1%}')

    # Summary heatmap: sessions x train fractions
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for ax, metric, title in zip(axes,
                                  ['sil_test', 'ari_test', 'acc_test'],
                                  ['Test Silhouette', 'Test ARI', 'Test Accuracy']):
        pivot = results_df.pivot(index='session', columns='train_frac', values=metric)
        pivot = pivot.reindex(SESSION_LABELS)
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0,
                        vmax=1 if metric == 'acc_test' else pivot.values.max())
        ax.set_xticks(range(len(TRAIN_FRACTIONS)))
        ax.set_xticklabels([f'{int(f*100)}%' for f in TRAIN_FRACTIONS])
        ax.set_yticks(range(len(SESSION_LABELS)))
        ax.set_yticklabels(SESSION_LABELS)
        ax.set_xlabel('Train Fraction')
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    fmt = f'{val:.1%}' if metric == 'acc_test' else f'{val:.3f}'
                    color = 'white' if val < pivot.values.max() * 0.4 else 'black'
                    ax.text(j, i, fmt, ha='center', va='center', fontsize=8, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Supervised UMAP Train/Test Validation — All Sessions', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'validation_summary_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Bar chart: per-session at 50% train
    sub50 = results_df[results_df['train_frac'] == 0.50].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title in zip(axes,
                                  ['sil_test', 'ari_test', 'acc_test'],
                                  ['Test Silhouette', 'Test ARI', 'Test Accuracy']):
        colors = ['#d62728' if v < 0.1 else '#2ca02c' if v > 0.4 else '#ff7f0e'
                  for v in sub50[metric]]
        ax.bar(range(len(sub50)), sub50[metric], color=colors)
        ax.set_xticks(range(len(sub50)))
        ax.set_xticklabels(sub50['session'], rotation=45, ha='right')
        ax.axhline(sub50[metric].mean(), ls='--', color='k', alpha=0.5,
                    label=f'mean={sub50[metric].mean():.3f}')
        ax.set_title(f'{title} (50% train)')
        ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'validation_50pct_barplot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\nAll outputs saved to {REPORT_DIR}/')
    print('Done.')


if __name__ == '__main__':
    main()
