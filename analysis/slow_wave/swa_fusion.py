"""
CAP-SWA + full-CAP fusion (Workstream C, step-1 follow-up).

Prerequisite: reports/slow_wave/sws_features.parquet must be re-extracted under the
CURRENT sleep-profile alignment (re-run detect_sws.py first). The May-28 cache was
shifted by one 30 s epoch relative to the cap_swa table; this script refuses to run
if the two tables still disagree on stage labels.

Question: does adding the mechanical CAP-SWA features (swa_score + sub-scores + raw
slopes) to the full 30-feature CAP set lift the LOSO N3 per-subject AUC above the
full-CAP baseline?

Design:
  - inner-join sws_features (rich CAP) with cap_swa (mechanical) on (session, epoch_idx)
  - HARD CHECK: stage_code must agree on ≥99% of joined rows, else abort
  - LOSO leave-one-subject-out, N3 vs non-N3
  - primary model = logistic regression (a depth-3 GBM ties few-feature outputs and
    deflates AUC); GBM reported as secondary
  - feature sets: full_cap, full_cap+swa, and mechanical-only for reference

Out: reports/slow_wave/cap_swa/classifier/fusion_ablation.csv
     reports/slow_wave/cap_swa/classifier/fusion_folds.csv
     reports/slow_wave/cap_swa/classifier/fusion_ablation.png
Run: .venv/Scripts/python.exe analysis/slow_wave/swa_fusion.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'slow_wave'
CAP_SWA_DIR = REPORT_DIR / 'cap_swa'
OUT_DIR = CAP_SWA_DIR / 'classifier'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N3_CODE = 1
META = {'session', 'subject', 'epoch_idx', 't_hr', 'stage_code', 'stage_label',
        'stage_label_swa', 'swa_candidate', 'is_n3'}
SWA_FEATS = ['swa_score', 'swa_s_dc', 'swa_s_thorax', 'swa_s_still',
             'dc_abs_slope', 'thorax_abs_slope', 'thorax_var']


def load_and_check():
    rich = pd.read_parquet(REPORT_DIR / 'sws_features.parquet')
    swa = pd.read_parquet(CAP_SWA_DIR / 'all_epoch_features.parquet')
    swa_cols = ['session', 'epoch_idx', 'stage_code'] + \
               [c for c in SWA_FEATS if c in swa.columns]
    merged = rich.merge(swa[swa_cols], on=['session', 'epoch_idx'],
                        how='inner', suffixes=('', '_swa'))
    # alignment gate
    mism = (merged['stage_code'] != merged['stage_code_swa']).mean()
    print(f"alignment check: {len(merged)} joined rows, "
          f"stage-label disagreement = {mism*100:.1f}%")
    if mism > 0.01:
        raise SystemExit(
            f"ABORT: sws_features and cap_swa still disagree on {mism*100:.1f}% of "
            f"stage labels (>1%). Re-run detect_sws.py to refresh the cache under the "
            f"current sleep-profile alignment before fusing.")
    merged = merged.drop(columns=['stage_code_swa'])
    merged = merged[merged['stage_code'] >= 0].reset_index(drop=True)
    merged['is_n3'] = (merged['stage_code'] == N3_CODE).astype(int)
    return merged


def full_cap_cols(df):
    return [c for c in df.columns if c not in META and c not in SWA_FEATS
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]


def loso(df, cols, kind='logistic'):
    subs = sorted(df['subject'].unique())
    X = df[cols].values.astype(np.float64)
    y = df['is_n3'].values
    oof = np.full(len(df), np.nan)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    w_pos = n_neg / n_pos if n_pos else 1.0
    per = []
    for s in subs:
        te = (df['subject'] == s).values
        tr = ~te
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y[tr], y[te]
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        for j in range(Xtr.shape[1]):
            Xtr[np.isnan(Xtr[:, j]), j] = med[j]
            Xte[np.isnan(Xte[:, j]), j] = med[j]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        if kind == 'gbm':
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42)
            clf.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, w_pos, 1.0))
        else:
            clf = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        oof[te] = p
        if 0 < yte.sum() < len(yte):
            per.append(dict(subject=s, auc=roc_auc_score(yte, p),
                            pr_auc=average_precision_score(yte, p)))
    return oof, pd.DataFrame(per)


def main():
    df = load_and_check()
    print(f"fusion set: {len(df)} scored epochs, {df['is_n3'].sum()} N3, "
          f"{df['subject'].nunique()} subjects\n")
    fc = full_cap_cols(df)
    swa_present = [c for c in SWA_FEATS if c in df.columns]
    sets = {
        'full_cap':      fc,
        'mechanical':    swa_present,
        'full_cap+swa':  fc + swa_present,
    }
    y = df['is_n3'].values
    rows, folds = [], []
    for kind in ('logistic', 'gbm'):
        for name, cols in sets.items():
            oof, per = loso(df, cols, kind)
            m = np.isfinite(oof)
            rows.append(dict(model=kind, feature_set=name, n_features=len(cols),
                             pooled_auc=roc_auc_score(y[m], oof[m]),
                             subj_auc_mean=per['auc'].mean(),
                             subj_auc_std=per['auc'].std(),
                             subj_auc_min=per['auc'].min(),
                             subj_auc_max=per['auc'].max()))
            folds.append(per.assign(model=kind, feature_set=name))
            print(f"  [{kind:8s}] {name:14s} nfeat={len(cols):2d}  "
                  f"pooled={rows[-1]['pooled_auc']:.3f}  "
                  f"subj={per['auc'].mean():.3f}±{per['auc'].std():.3f} "
                  f"[{per['auc'].min():.3f},{per['auc'].max():.3f}]")
    abl = pd.DataFrame(rows)
    folds = pd.concat(folds, ignore_index=True)
    abl.to_csv(OUT_DIR / 'fusion_ablation.csv', index=False)
    folds.to_csv(OUT_DIR / 'fusion_folds.csv', index=False)

    # lift summary (logistic primary)
    lg = abl[abl['model'] == 'logistic'].set_index('feature_set')
    lift = lg.loc['full_cap+swa', 'subj_auc_mean'] - lg.loc['full_cap', 'subj_auc_mean']
    print(f"\nLOGISTIC per-subject AUC lift (full_cap+swa - full_cap): {lift:+.3f}")

    # figure: per-subject full_cap vs full_cap+swa (logistic)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lgf = folds[folds['model'] == 'logistic']
    base = lgf[lgf['feature_set'] == 'full_cap'].set_index('subject')['auc']
    plus = lgf[lgf['feature_set'] == 'full_cap+swa'].set_index('subject')['auc']
    subs = sorted(base.index)
    xs = np.arange(len(subs)); w = 0.38
    ax.bar(xs - w/2, base.loc[subs], w, color='#95A5A6', label='full_cap')
    ax.bar(xs + w/2, plus.loc[subs], w, color='#16A085', label='full_cap + CAP-SWA')
    ax.axhline(0.5, color='k', lw=0.6, alpha=0.5)
    ax.axhline(base.mean(), color='#7F8C8D', ls='--', lw=1,
               label=f'full_cap mean {base.mean():.3f}')
    ax.axhline(plus.mean(), color='#0E6655', ls='--', lw=1,
               label=f'+CAP-SWA mean {plus.mean():.3f}')
    ax.set_xticks(xs); ax.set_xticklabels(subs)
    ax.set_ylabel('LOSO N3 AUC (per subject)')
    ax.set_title('CAP-SWA + full-CAP fusion (aligned cache, logistic)')
    ax.legend(fontsize=8); ax.set_ylim(0.3, 0.85)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fusion_ablation.png', dpi=130, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"saved -> {OUT_DIR}")


if __name__ == '__main__':
    main()
