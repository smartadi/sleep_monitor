"""
Supervised UMAP validation v2 — kNN + Random Forest.

For each session, 50/50 train/test split:
  A) kNN in supervised UMAP embedding space (k=5,10,20)
  B) kNN in raw feature space (baseline — no UMAP)
  C) Random Forest in raw feature space (proper supervised baseline)
  D) Random Forest in UMAP embedding (3D) — tests if UMAP adds value

This separates two questions:
  1. Can CAP features classify sleep stages within a session?
  2. Does supervised UMAP embedding improve or hurt classification?

Outputs → reports/projections/validation/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    classification_report, f1_score,
)
import umap
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import STAGE_COLORS, STAGE_ORDER
from sleep_monitor.sessions import SESSION_META

REPORT_DIR = ROOT / 'reports' / 'projections' / 'validation'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_DIR = ROOT / 'reports' / 'projections'
META_COLS = {'t_hr', 'stage_code', 'stage_label', 'apnea_code', 'apnea_label', 'thorax_rms'}

STAGE_NAMES = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_NAME_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']
SESSION_LABELS = [f'S{(i//2)+1}N{(i%2)+1}' for i in range(12)]


def load_features(label):
    csv_path = FEAT_DIR / label / f'{label}_features.csv'
    df = pd.read_csv(csv_path)
    df = df[df['stage_code'] >= 0].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


def clean_and_scale(X_raw):
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


def plot_confusion(cm, stage_names, title, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, fontsize=9)
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names, fontsize=9)
    ax.set_xlabel('True Stage')
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


def plot_train_test_3d(emb_train, emb_test, y_train, y_test, y_pred_test,
                       title, filepath):
    """3D plot: train points solid, test points colored by prediction correctness."""
    fig = go.Figure()
    for sc in STAGE_ORDER:
        mask_tr = y_train == sc
        if mask_tr.any():
            fig.add_trace(go.Scatter3d(
                x=emb_train[mask_tr, 0], y=emb_train[mask_tr, 1], z=emb_train[mask_tr, 2],
                mode='markers',
                marker=dict(size=2.5, color=STAGE_COLORS[sc], opacity=0.4),
                name=f'{STAGE_NAMES[sc]} train',
            ))
    correct = y_test == y_pred_test
    fig.add_trace(go.Scatter3d(
        x=emb_test[correct, 0], y=emb_test[correct, 1], z=emb_test[correct, 2],
        mode='markers',
        marker=dict(size=3, color='green', opacity=0.6, symbol='diamond'),
        name=f'Test correct (n={correct.sum()})',
    ))
    fig.add_trace(go.Scatter3d(
        x=emb_test[~correct, 0], y=emb_test[~correct, 1], z=emb_test[~correct, 2],
        mode='markers',
        marker=dict(size=3, color='red', opacity=0.6, symbol='x'),
        name=f'Test wrong (n=(~correct).sum())',
    ))
    fig.update_layout(
        title=title, width=1000, height=750,
        scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
        legend=dict(itemsizing='constant'),
    )
    fig.write_html(str(filepath))


def run_session(label):
    df, feat_cols = load_features(label)
    X_raw = df[feat_cols].values
    X_scaled, _ = clean_and_scale(X_raw)
    y = df['stage_code'].values.astype(int)
    n = len(df)

    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_train = n // 2
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    session_dir = REPORT_DIR / label
    session_dir.mkdir(parents=True, exist_ok=True)

    results = []
    stage_set = sorted(set(y))
    stage_names = [STAGE_NAMES[s] for s in stage_set]

    # A) kNN in supervised UMAP embedding
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42)
    emb_train = reducer.fit_transform(X_train, y=y_train)
    emb_test = reducer.transform(X_test)

    for k in [5, 10, 20]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(emb_train, y_train)
        y_pred = knn.predict(emb_test)
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results.append({
            'session': label, 'method': f'kNN_k{k}_UMAP',
            'accuracy': acc, 'kappa': kappa, 'f1_macro': f1,
            'n_train': n_train, 'n_test': len(test_idx),
        })

    # Save best kNN-UMAP confusion matrix + 3D plot
    knn_best = KNeighborsClassifier(n_neighbors=10)
    knn_best.fit(emb_train, y_train)
    y_pred_umap = knn_best.predict(emb_test)
    cm = confusion_matrix(y_test, y_pred_umap, labels=stage_set)
    plot_confusion(cm, stage_names,
                   f'{label} kNN-10 on UMAP — Acc={accuracy_score(y_test, y_pred_umap):.1%}',
                   session_dir / f'{label}_kNN_UMAP_confusion.png')
    plot_train_test_3d(emb_train, emb_test, y_train, y_test, y_pred_umap,
                       f'{label} kNN-10 on Supervised UMAP (Acc={accuracy_score(y_test, y_pred_umap):.1%})',
                       session_dir / f'{label}_kNN_UMAP_3d.html')

    # B) kNN in raw feature space (baseline)
    for k in [5, 10, 20]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results.append({
            'session': label, 'method': f'kNN_k{k}_features',
            'accuracy': acc, 'kappa': kappa, 'f1_macro': f1,
            'n_train': n_train, 'n_test': len(test_idx),
        })

    # C) Random Forest in feature space
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_rf)
    kappa = cohen_kappa_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)
    results.append({
        'session': label, 'method': 'RF_features',
        'accuracy': acc, 'kappa': kappa, 'f1_macro': f1,
        'n_train': n_train, 'n_test': len(test_idx),
    })
    cm = confusion_matrix(y_test, y_pred_rf, labels=stage_set)
    plot_confusion(cm, stage_names,
                   f'{label} Random Forest — Acc={acc:.1%}, Kappa={kappa:.3f}',
                   session_dir / f'{label}_RF_confusion.png')

    # D) Random Forest in UMAP embedding
    rf_umap = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf_umap.fit(emb_train, y_train)
    y_pred_rf_umap = rf_umap.predict(emb_test)
    acc_ru = accuracy_score(y_test, y_pred_rf_umap)
    kappa_ru = cohen_kappa_score(y_test, y_pred_rf_umap)
    f1_ru = f1_score(y_test, y_pred_rf_umap, average='macro', zero_division=0)
    results.append({
        'session': label, 'method': 'RF_UMAP',
        'accuracy': acc_ru, 'kappa': kappa_ru, 'f1_macro': f1_ru,
        'n_train': n_train, 'n_test': len(test_idx),
    })

    # Print summary for this session
    print(f'  kNN-10 UMAP:     Acc={accuracy_score(y_test, y_pred_umap):.1%}  '
          f'Kappa={cohen_kappa_score(y_test, y_pred_umap):.3f}')
    print(f'  kNN-10 features: Acc={accuracy_score(y_test, knn_best.predict(emb_test)):.1%}')
    knn_feat = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
    y_pred_feat = knn_feat.predict(X_test)
    print(f'  kNN-10 features: Acc={accuracy_score(y_test, y_pred_feat):.1%}  '
          f'Kappa={cohen_kappa_score(y_test, y_pred_feat):.3f}')
    print(f'  RF features:     Acc={acc:.1%}  Kappa={kappa:.3f}')
    print(f'  RF UMAP:         Acc={acc_ru:.1%}  Kappa={kappa_ru:.3f}')

    return results


def main():
    print('Supervised UMAP Validation v2 — kNN + RF Classifiers')
    print('=' * 60)

    all_results = []
    for label in SESSION_LABELS:
        csv_path = FEAT_DIR / label / f'{label}_features.csv'
        if not csv_path.exists():
            print(f'\n  WARNING: {label} not found, skipping')
            continue
        print(f'\n{label}:')
        results = run_session(label)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORT_DIR / 'validation_v2_results.csv', index=False)

    # Summary comparison table
    print('\n' + '=' * 80)
    print('SUMMARY: Mean test accuracy across all 12 sessions')
    print('=' * 80)
    methods_to_show = ['kNN_k10_UMAP', 'kNN_k10_features', 'RF_features', 'RF_UMAP']
    for method in methods_to_show:
        sub = results_df[results_df['method'] == method]
        print(f'  {method:20s}  Acc={sub["accuracy"].mean():.1%} +/- {sub["accuracy"].std():.1%}'
              f'  Kappa={sub["kappa"].mean():.3f} +/- {sub["kappa"].std():.3f}'
              f'  F1={sub["f1_macro"].mean():.3f} +/- {sub["f1_macro"].std():.3f}')

    # Grouped bar chart comparing methods
    fig, ax = plt.subplots(figsize=(16, 7))
    methods_plot = ['kNN_k10_features', 'kNN_k10_UMAP', 'RF_features', 'RF_UMAP']
    method_labels = ['kNN (features)', 'kNN (UMAP)', 'RF (features)', 'RF (UMAP)']
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
    n_sessions = len(SESSION_LABELS)
    width = 0.2
    x = np.arange(n_sessions)

    for i, (method, mlabel, color) in enumerate(zip(methods_plot, method_labels, colors)):
        sub = results_df[results_df['method'] == method].set_index('session')
        vals = [sub.loc[s, 'accuracy'] if s in sub.index else 0 for s in SESSION_LABELS]
        ax.bar(x + i * width, vals, width, label=mlabel, color=color)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(SESSION_LABELS, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1)
    ax.axhline(0.2, ls=':', color='gray', alpha=0.5, label='chance (5-class)')
    ax.legend(loc='upper right')
    ax.set_title('Per-Session Test Accuracy — Feature Space vs UMAP Embedding')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'validation_v2_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Kappa comparison
    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (method, mlabel, color) in enumerate(zip(methods_plot, method_labels, colors)):
        sub = results_df[results_df['method'] == method].set_index('session')
        vals = [sub.loc[s, 'kappa'] if s in sub.index else 0 for s in SESSION_LABELS]
        ax.bar(x + i * width, vals, width, label=mlabel, color=color)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(SESSION_LABELS, rotation=45, ha='right')
    ax.set_ylabel("Cohen's Kappa")
    ax.set_ylim(-0.1, 1)
    ax.axhline(0, ls=':', color='gray', alpha=0.5)
    ax.legend(loc='upper right')
    ax.set_title("Per-Session Cohen's Kappa — Feature Space vs UMAP Embedding")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'validation_v2_kappa.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\nAll outputs saved to {REPORT_DIR}/')
    print('Done.')


if __name__ == '__main__':
    main()
