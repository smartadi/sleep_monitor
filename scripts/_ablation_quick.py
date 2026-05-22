"""Quick ablation: which feature groups predict thorax_resp_rms?"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

df_raw = pd.read_parquet('artifacts/thorax_caponly_epochs.parquet')

TARGET = 'thorax_resp_rms'

# Add lags for CAP signal features
cap_rms_cols = [c for c in df_raw.columns
                if ('_resp_rms' in c or '_card_rms' in c or '_raw_std' in c)
                and 'thorax' not in c]
for col in cap_rms_cols:
    for lag in [1, 2, 3]:
        df_raw[f'{col}_lag{lag}'] = df_raw.groupby('session')[col].shift(lag)

# Z-score per session
z_cols = [c for c in df_raw.columns
          if c not in {'session', 'subject', 't_hr', 'stage_code', 'stage_label',
                       'position_code', 'epoch_frac', 'sin_t', 'cos_t'}
          and df_raw[c].dtype in [np.float64, float]]
for col in z_cols:
    df_raw[col] = df_raw.groupby('session')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))

# Feature groups
cap_signal = [c for c in df_raw.columns
              if any(ch in c for ch in ['CLE', 'CRE', 'CH_', 'avg_', 'CLE-CRE'])
              and 'thorax' not in c and c not in {'session', 'subject'}]
# Also include spectral/rate/cross features that are CAP-derived
cap_derived = [c for c in df_raw.columns
               if any(k in c for k in ['resp_rate_', 'card_rate_', 'resp_spectral_',
                                        'dom_resp_freq', 'power_ratio', 'resp_card_ratio',
                                        'coherence', 'phase_diff', 'rms_ratio',
                                        'resp_rate_variability'])]
cap_all = list(set(cap_signal + cap_derived))

accel = ['movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg']
context = ['position_code', 'sin_t', 'cos_t', 'epoch_frac']

groups = {
    'CAP signal only':       [c for c in cap_all if c in df_raw.columns],
    'Accel only':            accel,
    'Context only':          context,
    'Accel + Context':       accel + context,
    'CAP + Accel (no ctx)':  [c for c in cap_all if c in df_raw.columns] + accel,
    'All features':          [c for c in cap_all if c in df_raw.columns] + accel + context,
}

sessions = sorted(df_raw['session'].unique())

print(f"{'Feature group':<25s} {'N_feat':>6s} {'WS R2':>8s} {'LOSO R2':>8s}")
print('-' * 55)

for name, feats in groups.items():
    feats = [f for f in feats if f in df_raw.columns]

    ws_r2 = []
    for sess in sessions:
        ds = df_raw[df_raw['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        ds = ds.dropna(subset=[TARGET])
        cut = int(len(ds) * 0.7)
        train, test = ds.iloc[:cut], ds.iloc[cut:]
        if len(train) < 20 or len(test) < 10:
            continue
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
        m.fit(train[feats].fillna(0).values, train[TARGET].values, verbose=False)
        pred = m.predict(test[feats].fillna(0).values)
        ok = np.isfinite(pred) & np.isfinite(test[TARGET].values)
        if ok.sum() >= 5:
            ws_r2.append(r2_score(test[TARGET].values[ok], pred[ok]))

    loso_r2 = []
    for hold in sessions:
        train_df = df_raw[df_raw['session'] != hold].dropna(subset=[TARGET])
        test_df = df_raw[df_raw['session'] == hold].sort_values('t_hr').dropna(subset=[TARGET])
        if len(train_df) < 50 or len(test_df) < 10:
            continue
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
        m.fit(train_df[feats].fillna(0).values, train_df[TARGET].values, verbose=False)
        pred = m.predict(test_df[feats].fillna(0).values)
        ok = np.isfinite(pred) & np.isfinite(test_df[TARGET].values)
        if ok.sum() >= 5:
            loso_r2.append(r2_score(test_df[TARGET].values[ok], pred[ok]))

    ws_med = np.nanmedian(ws_r2) if ws_r2 else float('nan')
    lo_med = np.nanmedian(loso_r2) if loso_r2 else float('nan')
    print(f"{name:<25s} {len(feats):>6d} {ws_med:>8.3f} {lo_med:>8.3f}")
