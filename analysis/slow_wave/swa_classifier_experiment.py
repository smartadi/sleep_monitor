"""
CAP-SWA feature ablation + definition tuning (Workstream C, step 1 & 2).

Does the mechanical CAP-SWA score improve a LOSO N3 classifier, and how do the
0.60 threshold / geometric-mean weighting choices affect N3 discrimination?

Both feature matrices are already cached, so this runs in seconds with no raw
re-extraction:
  reports/slow_wave/sws_features.parquet           rich CAP features (detect_sws)
  reports/slow_wave/cap_swa/all_epoch_features.parquet   mechanical CAP-SWA features

PART A — LOSO ablation (step 1)
  Leave-one-subject-out, N3 vs non-N3, for nested feature sets drawn ENTIRELY from
  the self-consistent cap_swa table (same epoch/stage alignment as the score):
    swa_score_direct (composite used directly — no model, tie-free AUC)
    swa_score        (the composite through a linear/tree model)
    swa_subscores    (the 3 percentile sub-scores)
    mechanical       (sub-scores + raw slopes + motion)
  Reports pooled out-of-fold AUC, per-subject AUC spread, PR-AUC.
  Baselines for reference (published, CAP-only): ridge-only 0.51, full-CAP 0.56.

  NB: we deliberately do NOT fuse with the cached detect_sws feature matrix
  (reports/slow_wave/sws_features.parquet). That cache was extracted under an older
  epoch/stage alignment: ~43% of its per-epoch stage labels disagree with the current
  pipeline, so a join misaligns the score against the labels and destroys AUC. A valid
  fusion requires re-extracting the CAP features under the current alignment (flagged
  as follow-up in ANALYSIS_LOG).

PART B — threshold sweep (step 2a)
  Sweep the swa_score cut in [0.40, 0.85]; at each cut apply the sustained-bout
  rule per session and score the resulting candidate as an N3 detector
  (precision / recall / F1). AUC is threshold-free, reported once.

PART C — geometric-mean weighting sweep (step 2b)
  Weighted geomean of the 3 sub-scores over a weight-simplex grid; pooled N3 AUC
  and per-subject AUC spread per weighting. Equal weights vs best weighting.

Outputs -> reports/slow_wave/cap_swa/classifier/
  loso_ablation.csv           per-feature-set pooled + per-subject AUC/PR-AUC
  loso_ablation_folds.csv     per-subject AUC for every feature set
  threshold_sweep.csv         precision/recall/F1 vs swa_score threshold
  weighting_sweep.csv         pooled AUC + per-subject spread per weighting
  classifier_experiment.png   4-panel summary figure

Run:
  .venv/Scripts/python.exe analysis/slow_wave/swa_classifier_experiment.py
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
CAP_SWA_DIR = REPORT_DIR / 'cap_swa'
OUT_DIR = CAP_SWA_DIR / 'classifier'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N3_CODE = 1
MIN_SWA_EPOCHS = 4          # matches cap_swa_definition.py
RIDGE_BASELINE_AUC = 0.51   # paper_ridge_demo per-subject mean (published)
FULLCAP_BASELINE_AUC = 0.56 # detect_sws per-subject mean (published)

META = {'session', 'subject', 'epoch_idx', 't_hr', 'stage_code', 'stage_label',
        'swa_candidate'}

# mechanical CAP-SWA features carried over from cap_swa_definition
SWA_COMPOSITE = ['swa_score']
SWA_SUBSCORES = ['swa_s_dc', 'swa_s_thorax', 'swa_s_still']
SWA_RAW = ['dc_abs_slope', 'thorax_abs_slope', 'thorax_var', 'acc_rms']
MECHANICAL = SWA_COMPOSITE + SWA_SUBSCORES + SWA_RAW


# ── Data ─────────────────────────────────────────────────────────────────────

def load_capswa():
    """Load the self-consistent cap_swa table (score + labels share one alignment)."""
    swa = pd.read_parquet(CAP_SWA_DIR / 'all_epoch_features.parquet')
    scored = swa[swa['stage_code'] >= 0].reset_index(drop=True).copy()
    scored['is_n3'] = (scored['stage_code'] == N3_CODE).astype(int)
    return scored, swa


# ── LOSO ─────────────────────────────────────────────────────────────────────

def _build_model(kind, w_pos):
    if kind == 'logistic':
        # linear, continuous scores — fair to feature sets of any cardinality
        # (tree ensembles on 1-3 features tie their outputs and deflate AUC)
        return LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42)


def loso_probs(df, feature_cols, kind='logistic'):
    """Return out-of-fold P(N3) plus per-subject AUC/PR-AUC for one feature set."""
    subjects = sorted(df['subject'].unique())
    X = df[feature_cols].values.astype(np.float64)
    y = df['is_n3'].values
    oof = np.full(len(df), np.nan)

    n_pos, n_neg = y.sum(), len(y) - y.sum()
    w_pos = n_neg / n_pos if n_pos else 1.0

    per_subj = []
    for subj in subjects:
        te = (df['subject'] == subj).values
        tr = ~te
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y[tr], y[te]

        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        for j in range(Xtr.shape[1]):
            Xtr[np.isnan(Xtr[:, j]), j] = med[j]
            Xte[np.isnan(Xte[:, j]), j] = med[j]

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        clf = _build_model(kind, w_pos)
        if kind == 'gbm':
            clf.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, w_pos, 1.0))
        else:
            clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        oof[te] = p
        if 0 < yte.sum() < len(yte):
            per_subj.append(dict(subject=subj, n_test=int(len(yte)),
                                 n_n3=int(yte.sum()),
                                 auc=roc_auc_score(yte, p),
                                 pr_auc=average_precision_score(yte, p)))
    return oof, pd.DataFrame(per_subj)


def direct_score_auc(df, col):
    """Per-subject AUC of a single score used directly (no model, no ties)."""
    rows = []
    for subj, g in df.groupby('subject'):
        gy = g['is_n3'].values
        v = g[col].values
        m = np.isfinite(v)
        if 0 < gy[m].sum() < m.sum():
            rows.append(dict(subject=subj, feature_set=f'{col}_direct',
                             n_test=int(m.sum()), n_n3=int(gy[m].sum()),
                             auc=roc_auc_score(gy[m], v[m]),
                             pr_auc=average_precision_score(gy[m], v[m])))
    return pd.DataFrame(rows)


def run_ablation(df):
    sets = {
        'swa_score':     SWA_COMPOSITE,
        'swa_subscores': SWA_SUBSCORES,
        'mechanical':    MECHANICAL,
    }
    rows, fold_rows = [], []
    y = df['is_n3'].values

    # single-feature composite reported directly (the honest, tie-free number)
    direct = direct_score_auc(df, 'swa_score')
    fold_rows.append(direct)
    rows.append(dict(
        feature_set='swa_score_direct', model='direct', n_features=1,
        pooled_auc=roc_auc_score(y, df['swa_score']),
        pooled_pr_auc=average_precision_score(y, df['swa_score']),
        subj_auc_mean=direct['auc'].mean(), subj_auc_std=direct['auc'].std(),
        subj_auc_min=direct['auc'].min(), subj_auc_max=direct['auc'].max()))
    print(f"  {'swa_score_direct':16s} model=direct   pooled AUC="
          f"{roc_auc_score(y, df['swa_score']):.3f}  subj AUC={direct['auc'].mean():.3f}"
          f"±{direct['auc'].std():.3f} [{direct['auc'].min():.3f},{direct['auc'].max():.3f}]")

    for kind in ('logistic', 'gbm'):
        for name, cols in sets.items():
            cols = [c for c in cols if c in df.columns]
            oof, per = loso_probs(df, cols, kind=kind)
            m = np.isfinite(oof)
            pooled_auc = roc_auc_score(y[m], oof[m])
            pooled_pr = average_precision_score(y[m], oof[m])
            rows.append(dict(
                feature_set=name, model=kind, n_features=len(cols),
                pooled_auc=pooled_auc, pooled_pr_auc=pooled_pr,
                subj_auc_mean=per['auc'].mean(), subj_auc_std=per['auc'].std(),
                subj_auc_min=per['auc'].min(), subj_auc_max=per['auc'].max()))
            per = per.assign(feature_set=name, model=kind)
            fold_rows.append(per)
            print(f"  [{kind:8s}] {name:16s} nfeat={len(cols):2d}  pooled AUC="
                  f"{pooled_auc:.3f}  subj AUC={per['auc'].mean():.3f}"
                  f"±{per['auc'].std():.3f} [{per['auc'].min():.3f},{per['auc'].max():.3f}]")
    return pd.DataFrame(rows), pd.concat(fold_rows, ignore_index=True)


# ── Threshold sweep ──────────────────────────────────────────────────────────

def _sustained(above, min_len):
    """Run-length gate: keep only True-runs of length >= min_len."""
    out = np.zeros(len(above), dtype=bool)
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def threshold_sweep(swa_df, thresholds):
    """Precision/recall/F1 of the sustained swa candidate as an N3 detector."""
    df = swa_df[swa_df['stage_code'] >= 0].copy()
    y = (df['stage_code'] == N3_CODE).values
    rows = []
    for thr in thresholds:
        for sustain, lab in [(True, 'sustained'), (False, 'raw')]:
            pred = np.zeros(len(df), dtype=bool)
            for _, g in df.groupby('session'):
                idx = g.index.values
                above = (g['swa_score'].values >= thr)
                pred[np.searchsorted(df.index.values, idx)] = (
                    _sustained(above, MIN_SWA_EPOCHS) if sustain else above)
            tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y))
            fn = int(np.sum(~pred & y))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append(dict(threshold=round(float(thr), 3), mode=lab,
                             n_flagged=int(pred.sum()), precision=prec,
                             recall=rec, f1=f1))
    return pd.DataFrame(rows)


# ── Weighting sweep ──────────────────────────────────────────────────────────

def _weighted_geomean(df, w):
    w = np.asarray(w, dtype=float)
    s = np.vstack([df['swa_s_dc'].values, df['swa_s_thorax'].values,
                   df['swa_s_still'].values])
    with np.errstate(invalid='ignore', divide='ignore'):
        logs = np.log(np.clip(s, 1e-6, 1.0))
        gm = np.exp(np.sum(w[:, None] * logs, axis=0) / w.sum())
    gm[np.any(~np.isfinite(s), axis=0)] = np.nan
    return gm


def weighting_sweep(swa_df, step=0.25):
    df = swa_df[swa_df['stage_code'] >= 0].copy()
    y = (df['stage_code'] == N3_CODE).values
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    rows = []
    for w_dc, w_thx in product(grid, grid):
        w_still = 1.0 - w_dc - w_thx
        if w_still < -1e-9 or (w_dc + w_thx + max(w_still, 0)) < 1e-9:
            continue
        if w_still < 0:
            continue
        w = np.array([w_dc, w_thx, w_still])
        if w.sum() < 1e-9:
            continue
        score = _weighted_geomean(df, w)
        m = np.isfinite(score)
        if len(np.unique(y[m])) < 2:
            continue
        pooled = roc_auc_score(y[m], score[m])
        subj_aucs = []
        for subj, g in df.assign(_s=score).groupby('subject'):
            gm = g['_s'].values
            gy = (g['stage_code'] == N3_CODE).values
            ok = np.isfinite(gm)
            if 0 < gy[ok].sum() < ok.sum():
                subj_aucs.append(roc_auc_score(gy[ok], gm[ok]))
        rows.append(dict(w_dc=round(w_dc, 3), w_thorax=round(w_thx, 3),
                         w_still=round(float(w_still), 3), pooled_auc=pooled,
                         subj_auc_mean=float(np.mean(subj_aucs)),
                         subj_auc_std=float(np.std(subj_aucs)),
                         subj_auc_min=float(np.min(subj_aucs))))
    return pd.DataFrame(rows).sort_values('pooled_auc', ascending=False)


# ── Plot ─────────────────────────────────────────────────────────────────────

def make_figure(abl, folds, thr_df, wt_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('CAP-SWA classifier ablation & definition tuning (LOSO N3)',
                 fontsize=14, fontweight='bold')

    # A: ablation per-subject AUC (mean bar + per-subject dots)
    ax = axes[0, 0]
    order = ['swa_score_direct', 'swa_score', 'swa_subscores', 'mechanical']
    xs = np.arange(len(order))
    for i, fs in enumerate(order):
        if fs == 'swa_score_direct':
            sub = folds[folds['feature_set'] == 'swa_score_direct']['auc']
        else:
            sub = folds[(folds['feature_set'] == fs) &
                        (folds.get('model') == 'gbm')]['auc']
        ax.bar(i, sub.mean(), color='#16A085', alpha=0.8)
        ax.scatter(np.full(len(sub), i), sub, color='#2C3E50', s=22, zorder=3,
                   alpha=0.8)
    ax.axhline(RIDGE_BASELINE_AUC, color='#E74C3C', ls='--', lw=1,
               label=f'ridge baseline {RIDGE_BASELINE_AUC}')
    ax.axhline(FULLCAP_BASELINE_AUC, color='#E67E22', ls=':', lw=1,
               label=f'full-CAP baseline {FULLCAP_BASELINE_AUC}')
    ax.axhline(0.5, color='k', ls='-', lw=0.5, alpha=0.4)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('per-subject N3 AUC'); ax.set_ylim(0.35, 0.85)
    ax.set_title('A. LOSO ablation (bar=mean, dots=subjects)')
    ax.legend(fontsize=7)

    # B: threshold sweep
    ax = axes[0, 1]
    for mode, style in [('sustained', '-'), ('raw', '--')]:
        d = thr_df[thr_df['mode'] == mode].sort_values('threshold')
        ax.plot(d['threshold'], d['precision'], style, color='#E74C3C',
                label=f'precision ({mode})', lw=1.2)
        ax.plot(d['threshold'], d['recall'], style, color='#3498DB',
                label=f'recall ({mode})', lw=1.2)
        ax.plot(d['threshold'], d['f1'], style, color='#2ECC71',
                label=f'F1 ({mode})', lw=1.2)
    ax.axvline(0.60, color='k', ls=':', alpha=0.5, label='current cut 0.60')
    ax.set_xlabel('swa_score threshold'); ax.set_ylabel('score')
    ax.set_title('B. Threshold sweep (candidate as N3 detector)')
    ax.legend(fontsize=6, ncol=2)

    # C: weighting sweep — pooled AUC over the simplex
    ax = axes[1, 0]
    sc = ax.scatter(wt_df['w_dc'], wt_df['w_thorax'], c=wt_df['pooled_auc'],
                    cmap='viridis', s=140, marker='s')
    # mark equal weights
    eqrow = wt_df[(wt_df['w_dc'] == 0.25) & (wt_df['w_thorax'] == 0.25)]
    best = wt_df.iloc[0]
    ax.scatter([best['w_dc']], [best['w_thorax']], marker='*', s=260,
               edgecolor='r', facecolor='none', linewidth=2,
               label=f"best {best['pooled_auc']:.3f}")
    ax.set_xlabel('weight: slow-DC'); ax.set_ylabel('weight: slow-thorax')
    ax.set_title('C. Weighting sweep — pooled N3 AUC (w_still = 1-others)')
    plt.colorbar(sc, ax=ax, label='pooled AUC')
    ax.legend(fontsize=8, loc='upper right')

    # D: per-subject direct swa_score AUC — the consistency story (6/6 above chance)
    ax = axes[1, 1]
    d = folds[folds['feature_set'] == 'swa_score_direct'].set_index('subject')['auc']
    subs = sorted(d.index)
    xs = np.arange(len(subs))
    colors = ['#16A085' if d.loc[s] >= 0.5 else '#E74C3C' for s in subs]
    ax.bar(xs, d.loc[subs], color=colors, alpha=0.85)
    ax.axhline(0.5, color='k', lw=0.8, alpha=0.5)
    ax.axhline(d.mean(), color='#2C3E50', ls='--', lw=1,
               label=f'mean {d.mean():.3f}')
    for i, s in enumerate(subs):
        ax.text(i, d.loc[s] + 0.01, f'{d.loc[s]:.2f}', ha='center', fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(subs, fontsize=8)
    ax.set_ylabel('per-subject N3 AUC')
    ax.set_title('D. Direct swa_score per subject (self-consistent)')
    ax.legend(fontsize=8); ax.set_ylim(0.3, 0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / 'classifier_experiment.png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 64)
    print('CAP-SWA classifier experiment (step 1 & 2)')
    print('=' * 64)

    scored, swa = load_capswa()
    print(f"cap_swa (self-consistent): {len(scored)} scored epochs, "
          f"{scored['is_n3'].sum()} N3, {scored['subject'].nunique()} subjects\n")

    print('PART A — LOSO ablation (self-consistent cap_swa features only)')
    abl, folds = run_ablation(scored)
    abl.to_csv(OUT_DIR / 'loso_ablation.csv', index=False)
    folds.to_csv(OUT_DIR / 'loso_ablation_folds.csv', index=False)

    print('\nPART B — threshold sweep')
    thr_df = threshold_sweep(swa, np.arange(0.40, 0.851, 0.05))
    thr_df.to_csv(OUT_DIR / 'threshold_sweep.csv', index=False)
    best_f1 = thr_df[thr_df['mode'] == 'sustained'].sort_values('f1').iloc[-1]
    print(f"  best sustained F1={best_f1['f1']:.3f} at thr={best_f1['threshold']} "
          f"(prec={best_f1['precision']:.3f} rec={best_f1['recall']:.3f})")

    print('\nPART C — geometric-mean weighting sweep')
    wt_df = weighting_sweep(swa, step=0.25)
    wt_df.to_csv(OUT_DIR / 'weighting_sweep.csv', index=False)
    eq = wt_df[(wt_df['w_dc'] == 0.333) | ((wt_df['w_dc'] == 0.25))]
    best = wt_df.iloc[0]
    print(f"  best weighting w=(dc={best['w_dc']}, thx={best['w_thorax']}, "
          f"still={best['w_still']}) pooled AUC={best['pooled_auc']:.3f}")
    # equal weights (0.333 each) — nearest grid point at step 0.25 is (0.25,0.25,0.5);
    # report the true equal-weight geomean separately
    eq_score_auc = _equal_weight_auc(swa)
    print(f"  equal-weight (1/3,1/3,1/3) pooled AUC={eq_score_auc:.3f}")

    make_figure(abl, folds, thr_df, wt_df)
    print(f"\nsaved -> {OUT_DIR}")


def _equal_weight_auc(swa_df):
    df = swa_df[swa_df['stage_code'] >= 0]
    score = _weighted_geomean(df, [1/3, 1/3, 1/3])
    y = (df['stage_code'] == N3_CODE).values
    m = np.isfinite(score)
    return roc_auc_score(y[m], score[m])


if __name__ == '__main__':
    main()
