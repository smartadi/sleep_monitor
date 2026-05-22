"""
Motion-independent thorax prediction from CAP.

Approach: regress accel features -> thorax_resp_rms per session (Ridge),
take the residual as "motion-free thorax effort." Similarly residualize
each CAP feature. Then predict residual thorax from residual CAP.

This isolates direct CAP->thorax signal coupling from the shared
motion/position confound.

Output:
  artifacts/thorax_residual_results.csv
  notebooks/plots/thorax_analysis/residual_summary.png
  notebooks/plots/thorax_analysis/residual_per_session.png
  notebooks/plots/thorax_analysis/residual_feature_importance.png
  notebooks/plots/thorax_analysis/residual_timeseries_{session}.png  (x12)
  notebooks/plots/thorax_analysis/residual_variance_explained.png
"""

from __future__ import annotations
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'thorax_analysis'

TARGET = 'thorax_resp_rms'
ACCEL_COLS = ['movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg']

FEATURE_GROUP_COLORS = {
    'base': '#3498DB',
    'spectral': '#27AE60',
    'rate': '#E67E22',
    'cross': '#8E44AD',
    'context': '#E74C3C',
    'lag': '#95A5A6',
    'rolling': '#1ABC9C',
    'delta': '#F39C12',
}


def _feature_group(name):
    if any(k in name for k in ['_resp_rms', '_card_rms', '_raw_mean', '_raw_std']):
        if '_lag' in name:
            return 'lag'
        if '_roll' in name:
            return 'rolling'
        if '_delta' in name:
            return 'delta'
        return 'base'
    if any(k in name for k in ['spectral_entropy', 'power_ratio', 'resp_card_ratio', 'dom_resp_freq']):
        return 'spectral'
    if any(k in name for k in ['resp_rate_', 'card_rate_']):
        return 'rate'
    if any(k in name for k in ['coherence', 'phase_diff', 'rms_ratio']):
        return 'cross'
    if name in ['position_code', 'sin_t', 'cos_t', 'epoch_frac']:
        return 'context'
    if '_lag' in name:
        return 'lag'
    if '_roll' in name:
        return 'rolling'
    if '_delta' in name:
        return 'delta'
    return 'base'


def _metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'n': len(yt)}
    return {
        'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
        'mae': float(mean_absolute_error(yt, yp)),
        'r2': float(r2_score(yt, yp)),
        'n': int(len(yt)),
    }


def _split_train_test(df_sess, train_frac=0.7):
    n = len(df_sess)
    cut = int(n * train_frac)
    return df_sess.iloc[:cut], df_sess.iloc[cut:]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Residualize: remove motion from both target and CAP features
# ═══════════════════════════════════════════════════════════════════════════════

def residualize_per_session(df, target_col, feature_cols, accel_cols):
    """
    Per session: fit Ridge(accel -> col), replace col with residual.
    Returns df with residualized columns + per-session motion R2 on target.
    """
    df_out = df.copy()
    motion_r2 = {}

    for sess in sorted(df['session'].unique()):
        mask = df['session'] == sess
        idx = df.index[mask]
        X_acc = df.loc[idx, accel_cols].fillna(0).values

        # Residualize target
        y = df.loc[idx, target_col].values
        ok = np.isfinite(y)
        if ok.sum() < 20:
            motion_r2[sess] = np.nan
            continue
        model = Ridge(alpha=1.0)
        model.fit(X_acc[ok], y[ok])
        pred = model.predict(X_acc)
        residual = y - pred
        df_out.loc[idx, target_col] = residual
        motion_r2[sess] = float(r2_score(y[ok], pred[ok]))

        # Residualize each CAP feature
        for col in feature_cols:
            vals = df.loc[idx, col].values
            ok_f = np.isfinite(vals)
            if ok_f.sum() < 20:
                continue
            m = Ridge(alpha=1.0)
            m.fit(X_acc[ok_f], vals[ok_f])
            df_out.loc[idx, col] = vals - m.predict(X_acc)

    return df_out, motion_r2


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Add temporal features (lags, rolling, deltas)
# ═══════════════════════════════════════════════════════════════════════════════

CAP_CHANNELS = ['CLE', 'CRE', 'CH', 'CLE-CRE', 'avg']
LAG_COLS_KEY = [f'{ch}_resp_rms' for ch in CAP_CHANNELS] + ['movement_rms']
ROLLING_COLS = ['CLE-CRE_resp_rms', 'avg_resp_rms']
DELTA_COLS = ['CLE-CRE_resp_rms', 'avg_resp_rms', 'movement_rms']
N_LAGS = 3


def add_temporal_features(df):
    out = df.copy()
    for col in LAG_COLS_KEY:
        if col not in out.columns:
            continue
        for lag in range(1, N_LAGS + 1):
            out[f'{col}_lag{lag}'] = out.groupby('session')[col].shift(lag)
    for col in ROLLING_COLS:
        if col not in out.columns:
            continue
        for w in [3, 5]:
            out[f'{col}_roll{w}'] = out.groupby('session')[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean())
    for col in DELTA_COLS:
        if col not in out.columns:
            continue
        out[f'{col}_delta'] = out.groupby('session')[col].diff()
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Models on residualized data
# ═══════════════════════════════════════════════════════════════════════════════

def get_cap_features(df_columns):
    exclude = {'session', 'subject', 't_hr', 'stage_code', 'stage_label',
               TARGET, 'thorax_raw_mean', 'thorax_raw_std',
               'movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg',
               'position_code', 'sin_t', 'cos_t', 'epoch_frac'}
    return [c for c in df_columns if c not in exclude]


def run_xgb(df, features, model_name, df_raw_resid=None):
    import xgboost as xgb
    results = []
    predictions = {}
    importances_list = []
    sessions = sorted(df['session'].unique())

    for sess in sessions:
        ds = df[df['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        ds = ds.dropna(subset=[TARGET])
        train, test = _split_train_test(ds)
        if len(train) < 20 or len(test) < 10:
            continue
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        X_tr = train[features].fillna(0).values
        X_te = test[features].fillna(0).values
        model.fit(X_tr, train[TARGET].values,
                  eval_set=[(X_te, test[TARGET].values)], verbose=False)
        pred = model.predict(X_te)
        m = _metrics(test[TARGET].values, pred)
        m.update({'model': model_name, 'eval': 'within-session', 'session': sess})
        results.append(m)
        importances_list.append(model.feature_importances_)

        predictions[sess] = {
            't_hr': test['t_hr'].values,
            'actual': test[TARGET].values,
            'predicted': pred,
            'stage_code': test['stage_code'].values,
        }

    for hold in sessions:
        train_df = df[df['session'] != hold].dropna(subset=[TARGET])
        test_df = df[df['session'] == hold].sort_values('t_hr').dropna(subset=[TARGET])
        if len(train_df) < 50 or len(test_df) < 10:
            continue
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(train_df[features].fillna(0).values, train_df[TARGET].values, verbose=False)
        pred = model.predict(test_df[features].fillna(0).values)
        m = _metrics(test_df[TARGET].values, pred)
        m.update({'model': model_name, 'eval': 'LOSO', 'session': hold})
        results.append(m)

    avg_imp = np.mean(importances_list, axis=0) if importances_list else None
    return results, predictions, avg_imp


def run_ridge_baseline(df, features, model_name):
    from sklearn.preprocessing import StandardScaler
    results = []
    sessions = sorted(df['session'].unique())

    for sess in sessions:
        ds = df[df['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        ds = ds.dropna(subset=features + [TARGET])
        train, test = _split_train_test(ds)
        if len(train) < 20 or len(test) < 10:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[features].fillna(0).values)
        X_te = scaler.transform(test[features].fillna(0).values)
        model = Ridge(alpha=1.0)
        model.fit(X_tr, train[TARGET].values)
        pred = model.predict(X_te)
        m = _metrics(test[TARGET].values, pred)
        m.update({'model': model_name, 'eval': 'within-session', 'session': sess})
        results.append(m)

    for hold in sessions:
        train_df = df[df['session'] != hold].dropna(subset=features + [TARGET])
        test_df = df[df['session'] == hold].sort_values('t_hr').dropna(subset=features + [TARGET])
        if len(train_df) < 50 or len(test_df) < 10:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[features].fillna(0).values)
        X_te = scaler.transform(test_df[features].fillna(0).values)
        model = Ridge(alpha=1.0)
        model.fit(X_tr, train_df[TARGET].values)
        pred = model.predict(X_te)
        m = _metrics(test_df[TARGET].values, pred)
        m.update({'model': model_name, 'eval': 'LOSO', 'session': hold})
        results.append(m)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_variance_explained(motion_r2):
    sessions = sorted(motion_r2.keys())
    r2_vals = [motion_r2[s] for s in sessions]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(sessions))
    colors = ['#3498DB' if r >= 0 else '#E74C3C' for r in r2_vals]
    bars = ax.bar(x, r2_vals, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontsize=8, rotation=45)
    mean_r2 = np.nanmean(r2_vals)
    ax.axhline(mean_r2, color='red', ls='--', lw=1.2, label=f'Mean R2={mean_r2:.3f}')
    ax.set_ylabel('R2 (motion -> thorax_resp_rms)', fontsize=9)
    ax.set_title('Thorax variance explained by motion (per session)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.2)
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    fig.savefig(PLOT_DIR / 'residual_variance_explained.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved residual_variance_explained.png')


def plot_summary(res_df):
    models = sorted(res_df['model'].unique())
    colors_list = ['#3498DB', '#27AE60', '#E67E22', '#8E44AD', '#E74C3C']
    color_map = {m: colors_list[i % len(colors_list)] for i, m in enumerate(models)}
    evals = ['within-session', 'LOSO']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for col_idx, ev in enumerate(evals):
        sub = res_df[res_df['eval'] == ev]
        for row_idx, metric in enumerate(['rmse', 'r2']):
            ax = axes[row_idx, col_idx]
            x_pos = np.arange(len(models))
            vals, errs = [], []
            for mod in models:
                ms = sub[sub['model'] == mod][metric].dropna()
                vals.append(ms.median() if len(ms) > 0 else 0)
                errs.append(ms.std() if len(ms) > 1 else 0)
            bars = ax.bar(x_pos, vals, yerr=errs, capsize=3,
                          color=[color_map[m] for m in models], alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, fontsize=7, rotation=25, ha='right')
            ax.set_ylabel(metric.upper(), fontsize=9)
            ax.set_title(f'{metric.upper()} -- {ev}', fontsize=10)
            ax.grid(True, axis='y', alpha=0.2)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Motion-Residualized: CAP -> Thorax Prediction', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / 'residual_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved residual_summary.png')


def plot_per_session(res_df):
    models = sorted(res_df['model'].unique())
    colors_list = ['#3498DB', '#27AE60', '#E67E22', '#8E44AD', '#E74C3C']
    color_map = {m: colors_list[i % len(colors_list)] for i, m in enumerate(models)}
    sub = res_df[res_df['eval'] == 'within-session']
    sessions = sorted(sub['session'].unique())

    fig, ax = plt.subplots(figsize=(16, 6))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(sessions))

    for i, mod in enumerate(models):
        ms = sub[sub['model'] == mod]
        r2_vals = [ms[ms['session'] == s]['r2'].values[0]
                   if len(ms[ms['session'] == s]) > 0 else np.nan
                   for s in sessions]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, r2_vals, width=width, color=color_map[mod],
               alpha=0.8, label=mod)

    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontsize=8, rotation=45)
    ax.set_ylabel('R2 (on residuals)', fontsize=10)
    ax.set_title('Within-session R2 on motion-residualized thorax', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', ncol=len(models))
    ax.grid(True, axis='y', alpha=0.2)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    fig.savefig(PLOT_DIR / 'residual_per_session.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved residual_per_session.png')


def plot_feature_importance(avg_importance, feature_names):
    if avg_importance is None:
        return
    idx = np.argsort(avg_importance)[::-1][:30]
    names = [feature_names[i] for i in idx]
    vals = avg_importance[idx]
    groups = [_feature_group(n) for n in names]
    bar_colors = [FEATURE_GROUP_COLORS.get(g, '#999') for g in groups]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=bar_colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Gain', fontsize=9)
    ax.set_title('Top 30 Features — XGBoost on Residualized Data', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.2)

    from matplotlib.patches import Patch
    seen = {}
    for g in groups:
        if g not in seen:
            seen[g] = FEATURE_GROUP_COLORS.get(g, '#999')
    legend_handles = [Patch(facecolor=c, label=g) for g, c in seen.items()]
    ax.legend(handles=legend_handles, fontsize=7, loc='lower right')

    fig.savefig(PLOT_DIR / 'residual_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved residual_feature_importance.png')


def plot_timeseries(predictions, model_name):
    for sess, data in sorted(predictions.items()):
        fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.15})
        t = data['t_hr']
        actual = data['actual']
        predicted = data['predicted']
        stages = data['stage_code']
        residual = actual - predicted

        ok = np.isfinite(predicted) & np.isfinite(actual)
        r2 = r2_score(actual[ok], predicted[ok]) if ok.sum() >= 5 else np.nan
        mae = mean_absolute_error(actual[ok], predicted[ok]) if ok.sum() >= 5 else np.nan

        ax = axes[0]
        ax.plot(t, actual, color='#2C3E50', lw=0.8, alpha=0.9, label='Actual (residual)')
        ax.plot(t, predicted, color='#E67E22', lw=0.8, alpha=0.8, label=f'{model_name} pred')
        ax.set_ylabel('Residual thorax_resp_rms', fontsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title(f'{sess} — {model_name} on motion-residualized thorax  '
                     f'R2={r2:.3f}  MAE={mae:.3f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

        ax = axes[1]
        for code in STAGE_ORDER:
            mask = stages == code
            if mask.any():
                ax.bar(t[mask], residual[mask], width=0.008,
                       color=STAGE_COLORS[code], alpha=0.6, label=STAGE_LABELS[code])
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Pred error', fontsize=8)
        ax.set_xlabel('Time (hr)', fontsize=9)
        ax.legend(fontsize=6, loc='upper right', ncol=5)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

        fig.savefig(PLOT_DIR / f'residual_timeseries_{sess}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'  Saved {len(predictions)} timeseries plots')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('Motion-Residualized Thorax Prediction')
    print('=' * 60)

    # Load enhanced epochs
    pq_path = ART_DIR / 'thorax_caponly_epochs.parquet'
    print(f'\nLoading {pq_path}')
    df = pd.read_parquet(pq_path)
    print(f'  {len(df)} epochs, {df.session.nunique()} sessions')

    # Identify CAP feature columns (everything except metadata, target, accel, context)
    exclude_meta = {'session', 'subject', 't_hr', 'stage_code', 'stage_label'}
    exclude_target = {TARGET, 'thorax_raw_mean', 'thorax_raw_std'}
    exclude_accel = set(ACCEL_COLS)
    exclude_context = {'position_code', 'sin_t', 'cos_t', 'epoch_frac'}
    cap_feat_cols = [c for c in df.columns
                     if c not in exclude_meta | exclude_target | exclude_accel | exclude_context
                     and df[c].dtype in [np.float64, float]]

    print(f'  {len(cap_feat_cols)} CAP features to residualize')
    print(f'  Accel cols: {ACCEL_COLS}')

    # Z-score per session BEFORE residualizing (so Ridge operates on comparable scales)
    z_cols = cap_feat_cols + ACCEL_COLS + [TARGET]
    for col in z_cols:
        if col in df.columns:
            df[col] = df.groupby('session')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-12))

    # Step 1: Residualize
    print('\n-- Step 1: Residualizing motion from target + CAP features --')
    df_resid, motion_r2 = residualize_per_session(df, TARGET, cap_feat_cols, ACCEL_COLS)

    print('\n  Motion -> thorax R2 per session:')
    for sess in sorted(motion_r2.keys()):
        print(f'    {sess}: {motion_r2[sess]:.3f}')
    mean_mr2 = np.nanmean(list(motion_r2.values()))
    print(f'  Mean: {mean_mr2:.3f}')

    # Step 2: Add temporal features on residualized data
    print('\n-- Step 2: Adding temporal features --')
    df_resid = add_temporal_features(df_resid)

    # Get feature lists
    cap_feats_resid = get_cap_features(df_resid.columns.tolist())
    cap_feats_base = [c for c in cap_feat_cols if c in df_resid.columns]
    print(f'  Base CAP features: {len(cap_feats_base)}')
    print(f'  All CAP features (incl. temporal): {len(cap_feats_resid)}')

    all_results = []

    # Model A: Ridge on residualized base CAP features
    print('\n-- Model A: Ridge (residualized base CAP) --')
    ridge_res = run_ridge_baseline(df_resid, cap_feats_base, 'Ridge-Resid-Base')
    all_results.extend(ridge_res)
    ws = [r for r in ridge_res if r['eval'] == 'within-session']
    lo = [r for r in ridge_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    if lo:
        print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # Model B: XGBoost on residualized base CAP
    print('\n-- Model B: XGBoost (residualized base CAP) --')
    xgb_base_res, xgb_base_preds, _ = run_xgb(df_resid, cap_feats_base, 'XGB-Resid-Base')
    all_results.extend(xgb_base_res)
    ws = [r for r in xgb_base_res if r['eval'] == 'within-session']
    lo = [r for r in xgb_base_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    if lo:
        print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # Model C: XGBoost on residualized all CAP features
    print('\n-- Model C: XGBoost (residualized all CAP + temporal) --')
    xgb_all_res, xgb_all_preds, xgb_all_imp = run_xgb(
        df_resid, cap_feats_resid, 'XGB-Resid-All')
    all_results.extend(xgb_all_res)
    ws = [r for r in xgb_all_res if r['eval'] == 'within-session']
    lo = [r for r in xgb_all_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    if lo:
        print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # Model D: XGBoost on NON-residualized data, CAP-signal only (for comparison)
    print('\n-- Model D: XGBoost (original CAP signal, no residualization, no accel/context) --')
    df_orig = pd.read_parquet(pq_path)
    for col in z_cols:
        if col in df_orig.columns:
            df_orig[col] = df_orig.groupby('session')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-12))
    df_orig = add_temporal_features(df_orig)
    cap_feats_orig = get_cap_features(df_orig.columns.tolist())
    xgb_orig_res, _, _ = run_xgb(df_orig, cap_feats_orig, 'XGB-Orig-CAP-Only')
    all_results.extend(xgb_orig_res)
    ws = [r for r in xgb_orig_res if r['eval'] == 'within-session']
    lo = [r for r in xgb_orig_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    if lo:
        print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # Save
    res_df = pd.DataFrame(all_results)
    csv_path = ART_DIR / 'thorax_residual_results.csv'
    res_df.to_csv(csv_path, index=False)
    print(f'\nResults saved to {csv_path}')

    print('\n-- Summary (median across sessions) --')
    summary = res_df.groupby(['model', 'eval'])[['rmse', 'mae', 'r2']].median()
    print(summary.to_string())

    # Plots
    print('\n-- Generating plots --')
    plot_variance_explained(motion_r2)
    plot_summary(res_df)
    plot_per_session(res_df)
    plot_feature_importance(xgb_all_imp, cap_feats_resid)

    best_preds = xgb_all_preds if xgb_all_preds else xgb_base_preds
    best_name = 'XGB-Resid-All' if xgb_all_preds else 'XGB-Resid-Base'
    plot_timeseries(best_preds, best_name)

    print(f'\nAll outputs in {PLOT_DIR} and {ART_DIR}')


if __name__ == '__main__':
    main()
