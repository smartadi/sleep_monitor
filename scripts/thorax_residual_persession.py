"""
Per-session slow-trend models on residualized (motion-removed) thorax data.

Tests whether simple models capture slow co-moving trends between residualized
thorax and CAP features. Key comparison: AR-only vs ARIMAX(cap) — does CAP
add predictive power beyond thorax's own autocorrelation?

Models (within-session 70/30 temporal split):
  M1: OLS with top 5 CAP features (by train-set correlation)
  M2: AR-only — SARIMAX(2,0,0) with no exogenous
  M3: ARIMAX(cap) — SARIMAX(2,0,0) with top 3 CAP exogenous
  M4: Smoothed Ridge — 5-epoch rolling mean on both sides

Output:
  artifacts/thorax_residual_persession.csv
  notebooks/plots/thorax_analysis/persession_trends_{session}.png     (x12)
  notebooks/plots/thorax_analysis/persession_predictions_{session}.png (x12)
  notebooks/plots/thorax_analysis/persession_summary.png
  notebooks/plots/thorax_analysis/persession_arimax_comparison.png
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

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'thorax_analysis'

TARGET = 'thorax_resp_rms'
ACCEL_COLS = ['movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg']
SMOOTH_WIN = 5
ARIMAX_ORDER = (2, 0, 0)
TRAIN_FRAC = 0.7

MODEL_COLORS = {
    'OLS-Top5': '#3498DB',
    'AR-only': '#E74C3C',
    'ARIMAX-cap': '#27AE60',
    'SmoothedRidge': '#8E44AD',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Reused helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


def _split_train_test(df_sess, train_frac=TRAIN_FRAC):
    n = len(df_sess)
    cut = int(n * train_frac)
    return df_sess.iloc[:cut], df_sess.iloc[cut:]


def residualize_per_session(df, target_col, feature_cols, accel_cols):
    df_out = df.copy()
    motion_r2 = {}
    for sess in sorted(df['session'].unique()):
        mask = df['session'] == sess
        idx = df.index[mask]
        X_acc = df.loc[idx, accel_cols].fillna(0).values
        y = df.loc[idx, target_col].values
        ok = np.isfinite(y)
        if ok.sum() < 20:
            motion_r2[sess] = np.nan
            continue
        model = Ridge(alpha=1.0)
        model.fit(X_acc[ok], y[ok])
        pred = model.predict(X_acc)
        df_out.loc[idx, target_col] = y - pred
        motion_r2[sess] = float(r2_score(y[ok], pred[ok]))
        for col in feature_cols:
            vals = df.loc[idx, col].values
            ok_f = np.isfinite(vals)
            if ok_f.sum() < 20:
                continue
            m = Ridge(alpha=1.0)
            m.fit(X_acc[ok_f], vals[ok_f])
            df_out.loc[idx, col] = vals - m.predict(X_acc)
    return df_out, motion_r2


def get_cap_features(df_columns):
    exclude = {'session', 'subject', 't_hr', 'stage_code', 'stage_label',
               TARGET, 'thorax_raw_mean', 'thorax_raw_std',
               'movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg',
               'position_code', 'sin_t', 'cos_t', 'epoch_frac'}
    return [c for c in df_columns if c not in exclude]


def _top_features_by_corr(train_df, cap_cols, target_col, n):
    corrs = {}
    for c in cap_cols:
        vals = train_df[[c, target_col]].dropna()
        if len(vals) < 10:
            continue
        corrs[c] = abs(vals[c].corr(vals[target_col]))
    ranked = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:n]]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Visualize slow trends
# ═══════════════════════════════════════════════════════════════════════════════

def step1_visualize_trends(df_resid, cap_cols):
    print('\n-- Step 1: Visualize slow trends --')
    sessions = sorted(df_resid['session'].unique())

    for sess in sessions:
        ds = df_resid[df_resid['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        if len(ds) < 30:
            continue

        # Top 3 CAP features by correlation with residualized thorax
        top3 = _top_features_by_corr(ds, cap_cols, TARGET, 3)
        if len(top3) < 1:
            continue

        # Smooth everything
        y_smooth = ds[TARGET].rolling(SMOOTH_WIN, min_periods=1, center=True).mean()
        cap_smooth = {}
        for feat in top3:
            cap_smooth[feat] = ds[feat].rolling(SMOOTH_WIN, min_periods=1, center=True).mean()

        fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.08})

        # Top panel: smoothed thorax + CAP
        ax = axes[0]
        t = ds['t_hr'].values
        ax.plot(t, y_smooth, color='#2C3E50', lw=1.5, label=f'thorax (resid, {SMOOTH_WIN}-ep smooth)')

        feat_colors = ['#E67E22', '#3498DB', '#27AE60']
        for i, feat in enumerate(top3):
            # Scale CAP feature to thorax range for visual comparison
            cs = cap_smooth[feat].values
            ys = y_smooth.values
            ok = np.isfinite(cs) & np.isfinite(ys)
            if ok.sum() < 10:
                continue
            scale = np.std(ys[ok]) / (np.std(cs[ok]) + 1e-12)
            offset = np.mean(ys[ok]) - scale * np.mean(cs[ok])
            cs_scaled = cs * scale + offset
            r_val = np.corrcoef(ys[ok], cs[ok])[0, 1]
            ax.plot(t, cs_scaled, color=feat_colors[i], lw=0.9, alpha=0.8,
                    label=f'{feat} (r={r_val:.2f})')

        ax.set_ylabel('Residualized signal (smoothed)', fontsize=9)
        ax.set_title(f'{sess} — Slow trends: residualized thorax vs top CAP features',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

        # Bottom panel: hypnogram
        ax = axes[1]
        stages = ds['stage_code'].values
        for code in STAGE_ORDER:
            mask = stages == code
            if mask.any():
                ax.bar(t[mask], np.ones(mask.sum()), width=t[1]-t[0] if len(t) > 1 else 0.01,
                       color=STAGE_COLORS[code], alpha=0.7, label=STAGE_LABELS[code])
        ax.set_yticks([])
        ax.set_xlabel('Time (hr)', fontsize=9)
        ax.legend(fontsize=6, loc='upper right', ncol=5)
        ax.tick_params(labelsize=7)

        fig.savefig(PLOT_DIR / f'persession_trends_{sess}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'  Saved {len(sessions)} trend plots')


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Per-session models
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_ols(train, test, cap_cols, n_features=5):
    top = _top_features_by_corr(train, cap_cols, TARGET, n_features)
    if len(top) < 1:
        return None, None, None
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[top].fillna(0).values)
    X_te = scaler.transform(test[top].fillna(0).values)
    model = LinearRegression()
    model.fit(X_tr, train[TARGET].values)
    pred = model.predict(X_te)
    m = _metrics(test[TARGET].values, pred)
    return m, pred, top


def _fit_ar_only(train, test):
    y_tr = train[TARGET].values.astype(np.float64)
    y_te = test[TARGET].values.astype(np.float64)
    try:
        model = SARIMAX(y_tr, order=ARIMAX_ORDER,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=100)
        pred = fit.forecast(steps=len(y_te))
        pred = np.asarray(pred, dtype=np.float64)
        m = _metrics(y_te, pred)
        m['aic'] = float(fit.aic)
        m['bic'] = float(fit.bic)
        return m, pred
    except Exception as e:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'n': 0,
                'aic': np.nan, 'bic': np.nan}, None


def _fit_arimax_cap(train, test, cap_cols, n_exog=3):
    top = _top_features_by_corr(train, cap_cols, TARGET, n_exog)
    if len(top) < 1:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'n': 0,
                'aic': np.nan, 'bic': np.nan}, None, top

    y_tr = train[TARGET].values.astype(np.float64)
    X_tr = train[top].fillna(0).values.astype(np.float64)
    y_te = test[TARGET].values.astype(np.float64)
    X_te = test[top].fillna(0).values.astype(np.float64)

    exog_pvals = {}
    try:
        model = SARIMAX(y_tr, exog=X_tr, order=ARIMAX_ORDER,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=100)
        pred = fit.forecast(steps=len(y_te), exog=X_te)
        pred = np.asarray(pred, dtype=np.float64)
        m = _metrics(y_te, pred)
        m['aic'] = float(fit.aic)
        m['bic'] = float(fit.bic)

        # Extract p-values for exogenous coefficients
        try:
            pvals = fit.pvalues
            for i, feat in enumerate(top):
                key = f'x{i+1}'
                if key in pvals.index:
                    exog_pvals[feat] = float(pvals[key])
        except Exception:
            pass

        m['exog_pvals'] = exog_pvals
        m['exog_features'] = top
        return m, pred, top
    except Exception:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'n': 0,
                'aic': np.nan, 'bic': np.nan, 'exog_pvals': {}, 'exog_features': top}, None, top


def _fit_smoothed_ridge(train, test, cap_cols):
    feats = [c for c in cap_cols if c in train.columns]
    if len(feats) < 1:
        return None, None

    # Smooth within train and test separately (no leakage)
    y_tr = train[TARGET].rolling(SMOOTH_WIN, min_periods=1).mean().values
    y_te = test[TARGET].rolling(SMOOTH_WIN, min_periods=1).mean().values
    X_tr_raw = train[feats].fillna(0)
    X_te_raw = test[feats].fillna(0)
    X_tr = X_tr_raw.rolling(SMOOTH_WIN, min_periods=1).mean().fillna(0).values
    X_te = X_te_raw.rolling(SMOOTH_WIN, min_periods=1).mean().fillna(0).values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # Metrics on smoothed target
    m = _metrics(y_te, pred)
    return m, pred


def step2_persession_models(df_resid, cap_cols):
    print('\n-- Step 2: Per-session models --')
    sessions = sorted(df_resid['session'].unique())
    all_results = []
    all_predictions = {}

    for sess in sessions:
        ds = df_resid[df_resid['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        ds = ds.dropna(subset=[TARGET])
        train, test = _split_train_test(ds)
        if len(train) < 30 or len(test) < 10:
            print(f'  {sess}: skipped (too few epochs)')
            continue

        preds_sess = {
            't_hr': test['t_hr'].values,
            'actual': test[TARGET].values,
            'stage_code': test['stage_code'].values,
        }

        # M1: OLS top 5
        m1, pred1, top5 = _fit_ols(train, test, cap_cols, n_features=5)
        if m1 is not None:
            m1.update({'model': 'OLS-Top5', 'session': sess, 'top_features': str(top5)})
            all_results.append(m1)
            preds_sess['OLS-Top5'] = pred1
            print(f'  {sess} OLS-Top5:       R2={m1["r2"]:.3f}  MAE={m1["mae"]:.3f}  feats={top5[:3]}')

        # M2: AR-only
        m2, pred2 = _fit_ar_only(train, test)
        m2.update({'model': 'AR-only', 'session': sess})
        all_results.append(m2)
        if pred2 is not None:
            preds_sess['AR-only'] = pred2
        print(f'  {sess} AR-only:        R2={m2["r2"]:.3f}  MAE={m2["mae"]:.3f}  AIC={m2.get("aic", np.nan):.1f}')

        # M3: ARIMAX(cap)
        m3, pred3, top3_exog = _fit_arimax_cap(train, test, cap_cols, n_exog=3)
        m3_clean = {k: v for k, v in m3.items() if k not in ('exog_pvals', 'exog_features')}
        m3_clean.update({'model': 'ARIMAX-cap', 'session': sess,
                         'exog_features': str(m3.get('exog_features', [])),
                         'exog_pvals': str(m3.get('exog_pvals', {}))})
        # Delta AIC/BIC
        if np.isfinite(m3.get('aic', np.nan)) and np.isfinite(m2.get('aic', np.nan)):
            m3_clean['delta_aic'] = m3['aic'] - m2['aic']
            m3_clean['delta_bic'] = m3['bic'] - m2['bic']
        all_results.append(m3_clean)
        if pred3 is not None:
            preds_sess['ARIMAX-cap'] = pred3
        daic = m3_clean.get('delta_aic', np.nan)
        print(f'  {sess} ARIMAX-cap:     R2={m3["r2"]:.3f}  MAE={m3["mae"]:.3f}  AIC={m3.get("aic", np.nan):.1f}  dAIC={daic:.1f}')

        # M4: Smoothed Ridge
        m4, pred4 = _fit_smoothed_ridge(train, test, cap_cols)
        if m4 is not None:
            m4.update({'model': 'SmoothedRidge', 'session': sess})
            all_results.append(m4)
            if pred4 is not None:
                preds_sess['SmoothedRidge'] = pred4
            print(f'  {sess} SmoothedRidge:  R2={m4["r2"]:.3f}  MAE={m4["mae"]:.3f}')

        all_predictions[sess] = preds_sess
        print()

    return pd.DataFrame(all_results), all_predictions


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_predictions(all_predictions):
    print('\n-- Plotting per-session predictions --')
    models = list(MODEL_COLORS.keys())

    for sess, data in sorted(all_predictions.items()):
        t = data['t_hr']
        actual = data['actual']
        stages = data['stage_code']

        fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.12})

        ax = axes[0]
        ax.plot(t, actual, color='#2C3E50', lw=1.0, alpha=0.9, label='Actual (residual)')

        for mod in models:
            if mod in data and data[mod] is not None:
                pred = data[mod]
                ok = np.isfinite(pred) & np.isfinite(actual)
                r2 = r2_score(actual[ok], pred[ok]) if ok.sum() >= 5 else np.nan
                ax.plot(t, pred, color=MODEL_COLORS[mod], lw=0.8, alpha=0.8,
                        label=f'{mod} (R2={r2:.3f})')

        ax.set_ylabel('Residual thorax_resp_rms', fontsize=9)
        ax.set_title(f'{sess} — Per-session model predictions on residualized thorax',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

        ax = axes[1]
        for code in STAGE_ORDER:
            mask = stages == code
            if mask.any():
                w = t[1] - t[0] if len(t) > 1 else 0.01
                ax.bar(t[mask], np.ones(mask.sum()), width=w,
                       color=STAGE_COLORS[code], alpha=0.7, label=STAGE_LABELS[code])
        ax.set_yticks([])
        ax.set_xlabel('Time (hr)', fontsize=9)
        ax.legend(fontsize=6, loc='upper right', ncol=5)
        ax.tick_params(labelsize=7)

        fig.savefig(PLOT_DIR / f'persession_predictions_{sess}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'  Saved {len(all_predictions)} prediction plots')


def plot_summary(res_df):
    print('\n-- Plotting summary --')
    models = [m for m in ['OLS-Top5', 'AR-only', 'ARIMAX-cap', 'SmoothedRidge']
              if m in res_df['model'].values]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(models))
    width = 0.6

    for ax_idx, metric in enumerate(['r2', 'rmse', 'mae']):
        ax = axes[ax_idx]
        medians, q25s, q75s = [], [], []
        for mod in models:
            vals = res_df[res_df['model'] == mod][metric].dropna()
            medians.append(vals.median() if len(vals) > 0 else 0)
            q25s.append(vals.quantile(0.25) if len(vals) > 0 else 0)
            q75s.append(vals.quantile(0.75) if len(vals) > 0 else 0)

        colors = [MODEL_COLORS.get(m, '#999') for m in models]
        yerr_low = [max(0, med - q25) for med, q25 in zip(medians, q25s)]
        yerr_high = [q75 - med for med, q75 in zip(medians, q75s)]
        bars = ax.bar(x, medians, width, yerr=[yerr_low, yerr_high], capsize=4,
                      color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8, rotation=20, ha='right')
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_title(f'{metric.upper()} (median, IQR)', fontsize=10)
        ax.grid(True, axis='y', alpha=0.2)
        if metric == 'r2':
            ax.axhline(0, color='black', lw=0.5, ls='--')
        for bar, v in zip(bars, medians):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Per-Session Models on Residualized Thorax — Within-Session Test Set',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(PLOT_DIR / 'persession_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved persession_summary.png')


def plot_arimax_comparison(res_df):
    print('\n-- Plotting ARIMAX comparison --')
    ar_rows = res_df[res_df['model'] == 'AR-only'].set_index('session')
    arimax_rows = res_df[res_df['model'] == 'ARIMAX-cap'].set_index('session')
    sessions = sorted(set(ar_rows.index) & set(arimax_rows.index))
    if not sessions:
        print('  No sessions with both AR and ARIMAX — skipping')
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'hspace': 0.25})
    x = np.arange(len(sessions))
    width = 0.35

    # Top: R2 comparison
    ax = axes[0]
    r2_ar = [ar_rows.loc[s, 'r2'] if s in ar_rows.index else np.nan for s in sessions]
    r2_arimax = [arimax_rows.loc[s, 'r2'] if s in arimax_rows.index else np.nan for s in sessions]
    ax.bar(x - width/2, r2_ar, width, color=MODEL_COLORS['AR-only'], alpha=0.85, label='AR-only')
    ax.bar(x + width/2, r2_arimax, width, color=MODEL_COLORS['ARIMAX-cap'], alpha=0.85, label='ARIMAX(cap)')
    ax.set_ylabel('R2', fontsize=10)
    ax.set_title('AR-only vs ARIMAX(cap): does CAP add signal?', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.2)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    for i, (a, b) in enumerate(zip(r2_ar, r2_arimax)):
        if np.isfinite(a):
            ax.text(x[i] - width/2, a, f'{a:.2f}', ha='center', va='bottom', fontsize=6)
        if np.isfinite(b):
            ax.text(x[i] + width/2, b, f'{b:.2f}', ha='center', va='bottom', fontsize=6)

    # Bottom: delta AIC
    ax = axes[1]
    daics = []
    for s in sessions:
        row = arimax_rows.loc[s]
        daic = row.get('delta_aic', np.nan)
        if isinstance(daic, pd.Series):
            daic = daic.iloc[0]
        daics.append(float(daic) if pd.notna(daic) else np.nan)

    colors = ['#27AE60' if (np.isfinite(d) and d < 0) else '#E74C3C' for d in daics]
    ax.bar(x, daics, width=0.6, color=colors, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(-2, color='#27AE60', lw=0.8, ls='--', alpha=0.5, label='dAIC=-2 (meaningful)')
    ax.set_ylabel('delta AIC (ARIMAX - AR)', fontsize=10)
    ax.set_xlabel('Session', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontsize=8, rotation=45)
    ax.set_title('AIC difference: negative = CAP exogenous helps', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.2)
    for i, d in enumerate(daics):
        if np.isfinite(d):
            ax.text(x[i], d, f'{d:.1f}', ha='center',
                    va='bottom' if d >= 0 else 'top', fontsize=7)

    fig.savefig(PLOT_DIR / 'persession_arimax_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved persession_arimax_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('Per-Session Slow-Trend Models on Residualized Data')
    print('=' * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    pq_path = ART_DIR / 'thorax_caponly_epochs.parquet'
    print(f'\nLoading {pq_path}')
    df = pd.read_parquet(pq_path)
    print(f'  {len(df)} epochs, {df.session.nunique()} sessions')

    # Identify CAP feature columns
    exclude = {'session', 'subject', 't_hr', 'stage_code', 'stage_label',
               TARGET, 'thorax_raw_mean', 'thorax_raw_std',
               'movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg',
               'position_code', 'sin_t', 'cos_t', 'epoch_frac'}
    cap_cols = [c for c in df.columns
                if c not in exclude and df[c].dtype in [np.float64, float]]
    print(f'  {len(cap_cols)} CAP features')

    # Z-score per session
    z_cols = cap_cols + ACCEL_COLS + [TARGET]
    for col in z_cols:
        if col in df.columns:
            df[col] = df.groupby('session')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-12))

    # Residualize motion
    print('\n-- Residualizing motion --')
    df_resid, motion_r2 = residualize_per_session(df, TARGET, cap_cols, ACCEL_COLS)
    mean_mr2 = np.nanmean(list(motion_r2.values()))
    print(f'  Mean motion R2: {mean_mr2:.3f}')

    # Step 1: Trend visualization
    step1_visualize_trends(df_resid, cap_cols)

    # Step 2: Models
    res_df, predictions = step2_persession_models(df_resid, cap_cols)

    # Save results
    csv_path = ART_DIR / 'thorax_residual_persession.csv'
    res_df.to_csv(csv_path, index=False)
    print(f'\nResults saved to {csv_path}')

    # Summary table
    print('\n-- Summary (median across sessions) --')
    summary = res_df.groupby('model')[['r2', 'rmse', 'mae']].agg(['median', 'mean', 'std'])
    print(summary.to_string())

    # Key comparison: AR vs ARIMAX
    ar_r2 = res_df[res_df['model'] == 'AR-only']['r2'].dropna()
    arimax_r2 = res_df[res_df['model'] == 'ARIMAX-cap']['r2'].dropna()
    print(f'\n-- Key comparison --')
    print(f'  AR-only    median R2: {ar_r2.median():.3f} (mean {ar_r2.mean():.3f})')
    print(f'  ARIMAX-cap median R2: {arimax_r2.median():.3f} (mean {arimax_r2.mean():.3f})')
    if 'delta_aic' in res_df.columns:
        daics = res_df[res_df['model'] == 'ARIMAX-cap']['delta_aic'].dropna()
        if len(daics) > 0:
            n_helps = (daics < -2).sum()
            print(f'  Sessions where CAP helps (dAIC < -2): {n_helps}/{len(daics)}')
            print(f'  Median delta AIC: {daics.median():.1f}')

    # Plots
    plot_predictions(predictions)
    plot_summary(res_df)
    plot_arimax_comparison(res_df)

    print(f'\nAll outputs in {PLOT_DIR} and {ART_DIR}')


if __name__ == '__main__':
    main()
