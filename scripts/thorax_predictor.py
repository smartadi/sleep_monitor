"""
CAP -> Thorax resp RMS predictor: 4 model tiers.

Models:
  1. Ridge       — linear baseline, CAP features only
  2. ARIMAX      — AR on thorax + CAP exogenous features
  3. Stage-ARIMAX — separate ARIMAX per sleep stage
  4. XGBoost     — nonlinear with lagged thorax + CAP features

Evaluation:
  A. Within-session 70/30 temporal split (per-session fit quality)
  B. LOSO cross-validation (cross-subject generalization)

Target: thorax_resp_rms (per 30s epoch)

Output:
  artifacts/thorax_predictor_results.csv
  notebooks/plots/thorax_analysis/predictor_summary.png
  notebooks/plots/thorax_analysis/predictor_per_session.png
  notebooks/plots/thorax_analysis/predictor_timeseries_{label}.png  (x12)
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
import matplotlib.gridspec as gridspec

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT / "notebooks" / "plots" / "thorax_analysis"
ART_DIR = ROOT / "artifacts"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'thorax_resp_rms'

CAP_FEAT_COLS = []
for ch in ['CLE', 'CRE', 'CH', 'CLE-CRE', 'avg']:
    for suf in ['_raw_mean', '_raw_std', '_resp_rms', '_card_rms']:
        CAP_FEAT_COLS.append(f'{ch}{suf}')
ACCEL_COLS = ['movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg']
BASE_FEATURES = CAP_FEAT_COLS + ACCEL_COLS

N_LAGS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_lags(df, target_col, n_lags, feature_cols):
    """Add lagged target and lagged features to a session-level DataFrame."""
    out = df.copy()
    for lag in range(1, n_lags + 1):
        out[f'{target_col}_lag{lag}'] = out[target_col].shift(lag)
    for col in feature_cols:
        out[f'{col}_lag1'] = out[col].shift(1)
    return out.dropna().reset_index(drop=True)


def _zscore_per_session(df, cols):
    """Z-score columns within each session (removes DC drift)."""
    out = df.copy()
    for col in cols:
        out[col] = out.groupby('session')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12))
    return out


def _split_train_test(df_sess, train_frac=0.7):
    """Temporal 70/30 split within one session."""
    n = len(df_sess)
    cut = int(n * train_frac)
    return df_sess.iloc[:cut], df_sess.iloc[cut:]


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


# ---------------------------------------------------------------------------
# Model 1: Ridge regression (CAP features only)
# ---------------------------------------------------------------------------

def run_ridge(df, features):
    results = []
    sessions = sorted(df['session'].unique())

    # A. Within-session 70/30
    for sess in sessions:
        ds = df[df['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        train, test = _split_train_test(ds)
        if len(train) < 20 or len(test) < 10:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[features].values)
        X_te = scaler.transform(test[features].values)
        model = Ridge(alpha=1.0)
        model.fit(X_tr, train[TARGET].values)
        pred = model.predict(X_te)
        m = _metrics(test[TARGET].values, pred)
        m.update({'model': 'Ridge', 'eval': 'within-session', 'session': sess})
        results.append(m)

    # B. LOSO
    for hold in sessions:
        train_df = df[df['session'] != hold]
        test_df = df[df['session'] == hold].sort_values('t_hr')
        if len(train_df) < 50 or len(test_df) < 10:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[features].values)
        X_te = scaler.transform(test_df[features].values)
        model = Ridge(alpha=1.0)
        model.fit(X_tr, train_df[TARGET].values)
        pred = model.predict(X_te)
        m = _metrics(test_df[TARGET].values, pred)
        m.update({'model': 'Ridge', 'eval': 'LOSO', 'session': hold})
        results.append(m)

    return results


# ---------------------------------------------------------------------------
# Model 2: ARIMAX (AR on thorax + CAP exogenous)
# ---------------------------------------------------------------------------

def run_arimax(df, features, order=(2, 0, 0)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    results = []
    sessions = sorted(df['session'].unique())

    for sess in sessions:
        ds = df[df['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        train, test = _split_train_test(ds)
        if len(train) < 30 or len(test) < 10:
            continue

        y_tr = train[TARGET].values.astype(np.float64)
        X_tr = train[features].values.astype(np.float64)
        y_te = test[TARGET].values.astype(np.float64)
        X_te = test[features].values.astype(np.float64)

        try:
            model = SARIMAX(y_tr, exog=X_tr, order=order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=50)
            pred = fit.forecast(steps=len(y_te), exog=X_te)
            pred = np.asarray(pred, dtype=np.float64)
            m = _metrics(y_te, pred)
        except Exception:
            m = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'n': 0}

        m.update({'model': 'ARIMAX', 'eval': 'within-session', 'session': sess})
        results.append(m)

    return results


# ---------------------------------------------------------------------------
# Model 3: Stage-dependent ARIMAX
# ---------------------------------------------------------------------------

def run_stage_arimax(df, features, order=(2, 0, 0)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    results = []
    sessions = sorted(df['session'].unique())

    for sess in sessions:
        ds = df[df['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        train, test = _split_train_test(ds)
        if len(train) < 30 or len(test) < 10:
            continue

        pred_all = np.full(len(test), np.nan)

        stages_present = [c for c in STAGE_ORDER
                          if (train['stage_code'] == c).sum() >= 15
                          and (test['stage_code'] == c).sum() >= 3]

        stage_models = {}
        for code in stages_present:
            tr_s = train[train['stage_code'] == code]
            y_s = tr_s[TARGET].values.astype(np.float64)
            X_s = tr_s[features].values.astype(np.float64)
            try:
                mod = SARIMAX(y_s, exog=X_s, order=order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                stage_models[code] = mod.fit(disp=False, maxiter=50)
            except Exception:
                pass

        # Fallback: global model for stages without enough data
        y_tr_all = train[TARGET].values.astype(np.float64)
        X_tr_all = train[features].values.astype(np.float64)
        try:
            global_mod = SARIMAX(y_tr_all, exog=X_tr_all, order=order,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
            global_fit = global_mod.fit(disp=False, maxiter=50)
        except Exception:
            global_fit = None

        for code in STAGE_ORDER:
            te_mask = test['stage_code'] == code
            if te_mask.sum() == 0:
                continue
            te_s = test[te_mask]
            X_te = te_s[features].values.astype(np.float64)
            fit = stage_models.get(code, global_fit)
            if fit is None:
                continue
            try:
                p = fit.forecast(steps=len(X_te), exog=X_te)
                pred_all[te_mask.values] = np.asarray(p, dtype=np.float64)
            except Exception:
                pass

        m = _metrics(test[TARGET].values, pred_all)
        m.update({'model': 'Stage-ARIMAX', 'eval': 'within-session', 'session': sess})
        results.append(m)

    return results


# ---------------------------------------------------------------------------
# Model 4: XGBoost with lag features
# ---------------------------------------------------------------------------

def run_xgboost(df, base_features, df_raw=None):
    import xgboost as xgb
    results = []
    predictions = {}
    sessions = sorted(df['session'].unique())

    lag_cols = [f'{TARGET}_lag{i}' for i in range(1, N_LAGS + 1)]
    lag_feat_cols = [f'{c}_lag1' for c in base_features]
    all_feats = base_features + lag_cols + lag_feat_cols + ['stage_code']

    # A. Within-session 70/30
    for sess in sessions:
        ds = df[df['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        ds_lag = _add_lags(ds, TARGET, N_LAGS, base_features)
        train, test = _split_train_test(ds_lag)
        if len(train) < 20 or len(test) < 10:
            continue
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(train[all_feats].values, train[TARGET].values,
                  eval_set=[(test[all_feats].values, test[TARGET].values)],
                  verbose=False)
        pred = model.predict(test[all_feats].values)
        m = _metrics(test[TARGET].values, pred)
        m.update({'model': 'XGBoost', 'eval': 'within-session', 'session': sess})
        results.append(m)

        # Store raw (un-z-scored) thorax for background plotting
        raw_thorax = np.full(len(test), np.nan)
        if df_raw is not None:
            raw_sess = df_raw[df_raw['session'] == sess].sort_values('t_hr').reset_index(drop=True)
            raw_lag = _add_lags(raw_sess, TARGET, N_LAGS, base_features)
            _, raw_test = _split_train_test(raw_lag)
            if len(raw_test) == len(test):
                raw_thorax = raw_test[TARGET].values

        # Also get full-session raw thorax for context
        full_raw_t = np.array([])
        full_raw_thorax = np.array([])
        if df_raw is not None:
            raw_sess = df_raw[df_raw['session'] == sess].sort_values('t_hr')
            full_raw_t = raw_sess['t_hr'].values
            full_raw_thorax = raw_sess[TARGET].values

        predictions[sess] = {
            't_hr': test['t_hr'].values,
            'actual': test[TARGET].values,
            'predicted': pred,
            'stage_code': test['stage_code'].values,
            'raw_thorax': raw_thorax,
            'full_raw_t': full_raw_t,
            'full_raw_thorax': full_raw_thorax,
        }

    # B. LOSO
    for hold in sessions:
        train_frames = []
        for s in sessions:
            if s == hold:
                continue
            ds = df[df['session'] == s].sort_values('t_hr').reset_index(drop=True)
            train_frames.append(_add_lags(ds, TARGET, N_LAGS, base_features))
        train_df = pd.concat(train_frames, ignore_index=True)

        test_ds = df[df['session'] == hold].sort_values('t_hr').reset_index(drop=True)
        test_df = _add_lags(test_ds, TARGET, N_LAGS, base_features)
        if len(train_df) < 50 or len(test_df) < 10:
            continue
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(train_df[all_feats].values, train_df[TARGET].values,
                  verbose=False)
        pred = model.predict(test_df[all_feats].values)
        m = _metrics(test_df[TARGET].values, pred)
        m.update({'model': 'XGBoost', 'eval': 'LOSO', 'session': hold})
        results.append(m)

    return results, predictions


# ---------------------------------------------------------------------------
# Model 0: AR-only baseline (thorax lags only, no CAP)
# ---------------------------------------------------------------------------

def run_ar_baseline(df):
    """Pure autoregressive baseline -- predicts thorax from its own lags only."""
    results = []
    sessions = sorted(df['session'].unique())
    lag_cols = [f'{TARGET}_lag{i}' for i in range(1, N_LAGS + 1)]

    for sess in sessions:
        ds = df[df['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        ds_lag = _add_lags(ds, TARGET, N_LAGS, [])
        train, test = _split_train_test(ds_lag)
        if len(train) < 20 or len(test) < 10:
            continue
        model = Ridge(alpha=1.0)
        model.fit(train[lag_cols].values, train[TARGET].values)
        pred = model.predict(test[lag_cols].values)
        m = _metrics(test[TARGET].values, pred)
        m.update({'model': 'AR-only', 'eval': 'within-session', 'session': sess})
        results.append(m)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(res_df):
    """Bar chart comparing models on RMSE and R2, both eval modes."""
    models = ['AR-only', 'Ridge', 'ARIMAX', 'Stage-ARIMAX', 'XGBoost']
    evals = ['within-session', 'LOSO']
    colors = {'AR-only': '#95A5A6', 'Ridge': '#3498DB', 'ARIMAX': '#27AE60',
              'Stage-ARIMAX': '#8E44AD', 'XGBoost': '#E67E22'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for col_idx, ev in enumerate(evals):
        sub = res_df[res_df['eval'] == ev]
        if sub.empty:
            continue
        for row_idx, metric in enumerate(['rmse', 'r2']):
            ax = axes[row_idx, col_idx]
            x_pos = np.arange(len(models))
            vals = []
            errs = []
            for mod in models:
                ms = sub[sub['model'] == mod][metric].dropna()
                vals.append(ms.median() if len(ms) > 0 else 0)
                errs.append(ms.std() if len(ms) > 1 else 0)
            bars = ax.bar(x_pos, vals, yerr=errs, capsize=3,
                          color=[colors.get(m, '#999') for m in models], alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, fontsize=8, rotation=20, ha='right')
            ax.set_ylabel(metric.upper(), fontsize=9)
            ax.set_title(f'{metric.upper()} -- {ev}', fontsize=10)
            ax.grid(True, axis='y', alpha=0.2)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('CAP -> Thorax resp RMS: Model Comparison', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / 'predictor_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved predictor_summary.png')


def plot_per_session(res_df):
    """Per-session R2 for within-session eval, all models side-by-side."""
    models = ['AR-only', 'Ridge', 'ARIMAX', 'Stage-ARIMAX', 'XGBoost']
    colors = {'AR-only': '#95A5A6', 'Ridge': '#3498DB', 'ARIMAX': '#27AE60',
              'Stage-ARIMAX': '#8E44AD', 'XGBoost': '#E67E22'}
    sub = res_df[res_df['eval'] == 'within-session']
    sessions = sorted(sub['session'].unique())

    fig, ax = plt.subplots(figsize=(16, 6))
    n_models = len(models)
    width = 0.15
    x = np.arange(len(sessions))
    for i, mod in enumerate(models):
        ms = sub[sub['model'] == mod]
        r2_vals = [ms[ms['session'] == s]['r2'].values[0]
                   if len(ms[ms['session'] == s]) > 0 else np.nan
                   for s in sessions]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, r2_vals, width=width, color=colors[mod],
               alpha=0.8, label=mod)

    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontsize=8, rotation=45)
    ax.set_ylabel('R2', fontsize=10)
    ax.set_title('Within-session R2 by model and session', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', ncol=5)
    ax.grid(True, axis='y', alpha=0.2)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    fig.savefig(PLOT_DIR / 'predictor_per_session.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved predictor_per_session.png')


def plot_timeseries(predictions):
    """Overlay actual vs predicted thorax_resp_rms for XGBoost within-session,
    with raw thorax signal in the background for context."""
    for sess, data in sorted(predictions.items()):
        fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 3, 1], 'hspace': 0.15})

        t = data['t_hr']
        actual = data['actual']
        predicted = data['predicted']
        stages = data['stage_code']
        residual = actual - predicted
        full_t = data.get('full_raw_t', np.array([]))
        full_thorax = data.get('full_raw_thorax', np.array([]))

        ok = np.isfinite(predicted)
        r2 = r2_score(actual[ok], predicted[ok])
        mae = mean_absolute_error(actual[ok], predicted[ok])

        # Row 0: Raw thorax (full session) with test region highlighted
        ax = axes[0]
        if len(full_t) > 0:
            ax.plot(full_t, full_thorax, color='#95A5A6', lw=0.5, alpha=0.6)
            ax.axvspan(t[0], t[-1], color='#E67E22', alpha=0.08,
                       label='Test region (30%)')
            ax.plot(t, data.get('raw_thorax', actual),
                    color='#2C3E50', lw=0.7, alpha=0.9, label='Test raw')
        ax.set_ylabel('thorax_resp_rms\n(raw)', fontsize=8)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.set_title(f'{sess}  --  XGBoost within-session  '
                     f'R2={r2:.3f}  MAE={mae:.2f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

        # Row 1: z-scored actual vs predicted (main comparison)
        ax = axes[1]
        ax.plot(t, actual, color='#2C3E50', lw=0.8, alpha=0.9, label='Actual (z-scored)')
        ax.plot(t, predicted, color='#E67E22', lw=0.8, alpha=0.8, label='XGBoost pred')
        ax.set_ylabel('thorax_resp_rms\n(z-scored)', fontsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

        # Row 2: residuals colored by sleep stage
        ax = axes[2]
        for code in STAGE_ORDER:
            mask = stages == code
            if mask.any():
                ax.bar(t[mask], residual[mask], width=0.008,
                       color=STAGE_COLORS[code], alpha=0.6, label=STAGE_LABELS[code])
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Residual', fontsize=8)
        ax.set_xlabel('Time (hr)', fontsize=9)
        ax.legend(fontsize=6, loc='upper right', ncol=5)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

        fig.savefig(PLOT_DIR / f'predictor_timeseries_{sess}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'  Saved {len(predictions)} timeseries plots')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('CAP -> Thorax Predictor')
    print('=' * 60)

    pq_path = ART_DIR / 'thorax_cap_epochs.parquet'
    print(f'\nLoading {pq_path}')
    df_raw = pd.read_parquet(pq_path)
    print(f'  {len(df_raw)} epochs, {len(df_raw.session.unique())} sessions')

    # Z-score per session to remove DC drift
    z_cols = CAP_FEAT_COLS + [TARGET] + ACCEL_COLS
    df = _zscore_per_session(df_raw, z_cols)

    all_results = []

    # -- Model 0: AR-only baseline --
    print('\n-- Model 0: AR-only baseline --')
    ar_res = run_ar_baseline(df)
    all_results.extend(ar_res)
    ar_r2 = np.nanmedian([r['r2'] for r in ar_res])
    print(f'  Within-session median R2 = {ar_r2:.3f}')

    # -- Model 1: Ridge --
    print('\n-- Model 1: Ridge --')
    ridge_res = run_ridge(df, BASE_FEATURES)
    all_results.extend(ridge_res)
    ws = [r for r in ridge_res if r['eval'] == 'within-session']
    lo = [r for r in ridge_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # -- Model 2: ARIMAX --
    print('\n-- Model 2: ARIMAX (2,0,0) --')
    arimax_res = run_arimax(df, BASE_FEATURES, order=(2, 0, 0))
    all_results.extend(arimax_res)
    arimax_r2 = np.nanmedian([r['r2'] for r in arimax_res])
    print(f'  Within-session median R2 = {arimax_r2:.3f}')

    # -- Model 3: Stage-ARIMAX --
    print('\n-- Model 3: Stage-ARIMAX (2,0,0) --')
    stage_res = run_stage_arimax(df, BASE_FEATURES, order=(2, 0, 0))
    all_results.extend(stage_res)
    stage_r2 = np.nanmedian([r['r2'] for r in stage_res])
    print(f'  Within-session median R2 = {stage_r2:.3f}')

    # -- Model 4: XGBoost --
    print('\n-- Model 4: XGBoost + lags --')
    xgb_res, xgb_preds = run_xgboost(df, BASE_FEATURES)
    all_results.extend(xgb_res)
    ws = [r for r in xgb_res if r['eval'] == 'within-session']
    lo = [r for r in xgb_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # -- Save results --
    res_df = pd.DataFrame(all_results)
    csv_path = ART_DIR / 'thorax_predictor_results.csv'
    res_df.to_csv(csv_path, index=False)
    print(f'\nResults saved to {csv_path}')

    # -- Summary table --
    print('\n-- Summary (median across sessions) --')
    summary = res_df.groupby(['model', 'eval'])[['rmse', 'mae', 'r2']].median()
    print(summary.to_string())

    # -- Plots --
    print('\n-- Generating plots --')
    plot_summary(res_df)
    plot_per_session(res_df)
    plot_timeseries(xgb_preds)

    print(f'\nAll outputs in {PLOT_DIR} and {ART_DIR}')


if __name__ == '__main__':
    main()
