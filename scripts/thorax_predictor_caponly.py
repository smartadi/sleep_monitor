"""
CAP-only -> Thorax resp RMS predictor (no thorax lags, no PSG labels).

Tiers:
  0. Ridge       — CAP base features only (linear baseline)
  1. XGBoost-Base — base features + CAP lags (current minus thorax lags)
  2. XGBoost-Enh  — enhanced spectral/rate/cross-channel/context features
  3. XGBoost-Rec  — Tier 2 + recursive pseudo-lags (teacher-forced train)

Evaluation:
  A. Within-session 70/30 temporal split
  B. LOSO cross-validation

Target: thorax_resp_rms (per 30s epoch, z-scored per session)

Output:
  artifacts/thorax_caponly_epochs.parquet
  artifacts/thorax_caponly_results.csv
  notebooks/plots/thorax_analysis/caponly_*.png
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
from scipy.signal import welch, hilbert, coherence, csd
from scipy.stats import pearsonr

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, PSG_EPOCH_SEC,
)
from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.filters import bandpass
from sleep_monitor.motion import epoch_motion, classify_position
from sleep_monitor.rates import rate_peaks, rate_hilbert, rate_acf, rate_spectral

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'thorax_analysis'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'thorax_resp_rms'
EPOCH_SEC = 30.0
EPOCH_N = int(EPOCH_SEC * FS)
N_LAGS = 3

CAP_CHANNELS = ['CLE', 'CRE', 'CH', 'CLE-CRE', 'avg']
SPECTRAL_CHANNELS = ['CLE-CRE', 'CLE', 'CRE']

POSITION_MAP = {'supine': 0, 'left': 1, 'right': 2, 'prone': 3}


# ═══════════════════════════════════════════════════════════════════════════════
# Feature engineering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _welch_band_power(sig, fs, f_lo, f_hi, nperseg=None):
    if nperseg is None:
        nperseg = min(len(sig), int(fs * 4))
    if len(sig) < nperseg:
        return np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    df = freqs[1] - freqs[0]
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.nan
    return float(np.trapz(psd[mask], dx=df))


def _spectral_entropy(sig, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(len(sig), int(fs * 4))
    if len(sig) < nperseg:
        return np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    psd_norm = psd / (psd.sum() + 1e-20)
    psd_norm = psd_norm[psd_norm > 0]
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def _dom_freq(sig, fs, f_lo, f_hi, nperseg=None):
    if nperseg is None:
        nperseg = min(len(sig), int(fs * 4))
    if len(sig) < nperseg:
        return np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any() or psd[mask].max() == 0:
        return np.nan
    return float(freqs[mask][np.argmax(psd[mask])])


def _coherence_band(x, y, fs, f_lo, f_hi, nperseg=None):
    if nperseg is None:
        nperseg = min(min(len(x), len(y)), int(fs * 4))
    if len(x) < nperseg or len(y) < nperseg:
        return np.nan
    freqs, coh = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.nan
    return float(np.mean(coh[mask]))


def _phase_diff_band(x, y, fs, f_lo, f_hi, nperseg=None):
    if nperseg is None:
        nperseg = min(min(len(x), len(y)), int(fs * 4))
    if len(x) < nperseg or len(y) < nperseg:
        return np.nan
    freqs, Pxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.nan
    angles = np.angle(Pxy[mask])
    return float(np.mean(angles))


def _resp_rate_variability(seg, fs, f_lo, f_hi):
    if len(seg) < 64:
        return np.nan
    analytic = hilbert(seg.astype(np.float64))
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    valid = (inst_freq >= f_lo) & (inst_freq <= f_hi)
    if valid.sum() < 10:
        return np.nan
    return float(np.std(inst_freq[valid]))


def compute_spectral_features(seg_raw, fs, ch_prefix):
    row = {}
    total_power = _welch_band_power(seg_raw, fs, 0.1, 30.0)
    resp_power = _welch_band_power(seg_raw, fs, RESP_LO, RESP_HI)
    card_power = _welch_band_power(seg_raw, fs, CARD_LO, CARD_HI)

    row[f'{ch_prefix}_dom_resp_freq'] = _dom_freq(seg_raw, fs, RESP_LO, RESP_HI)
    row[f'{ch_prefix}_resp_spectral_entropy'] = _spectral_entropy(seg_raw, fs)

    if total_power and total_power > 0 and np.isfinite(total_power):
        row[f'{ch_prefix}_resp_power_ratio'] = resp_power / total_power if resp_power and np.isfinite(resp_power) else np.nan
        row[f'{ch_prefix}_card_power_ratio'] = card_power / total_power if card_power and np.isfinite(card_power) else np.nan
    else:
        row[f'{ch_prefix}_resp_power_ratio'] = np.nan
        row[f'{ch_prefix}_card_power_ratio'] = np.nan

    if resp_power and card_power and np.isfinite(resp_power) and np.isfinite(card_power) and card_power > 0:
        row[f'{ch_prefix}_resp_card_ratio'] = resp_power / card_power
    else:
        row[f'{ch_prefix}_resp_card_ratio'] = np.nan

    return row


def compute_rate_features(seg_resp_bp, seg_card_bp, fs):
    row = {}
    row['resp_rate_peaks'] = rate_peaks(seg_resp_bp, RESP_LO, RESP_HI, fs)
    row['resp_rate_hilbert'] = rate_hilbert(seg_resp_bp, RESP_LO, RESP_HI, fs)
    row['resp_rate_acf'] = rate_acf(seg_resp_bp, RESP_LO, RESP_HI, fs)
    row['card_rate_hilbert'] = rate_hilbert(seg_card_bp, CARD_LO, CARD_HI, fs)
    row['resp_rate_variability'] = _resp_rate_variability(seg_resp_bp, fs, RESP_LO, RESP_HI)
    return row


def compute_cross_channel_features(seg_cle_resp, seg_cre_resp, seg_cle_card, seg_cre_card, fs):
    row = {}
    cre_rms = np.sqrt(np.mean(seg_cre_resp ** 2))
    cle_rms = np.sqrt(np.mean(seg_cle_resp ** 2))
    row['cle_cre_resp_rms_ratio'] = cle_rms / (cre_rms + 1e-12)
    row['cle_cre_resp_coherence'] = _coherence_band(seg_cle_resp, seg_cre_resp, fs, RESP_LO, RESP_HI)
    row['cle_cre_card_coherence'] = _coherence_band(seg_cle_card, seg_cre_card, fs, CARD_LO, CARD_HI)
    row['cle_cre_resp_phase_diff'] = _phase_diff_band(seg_cle_resp, seg_cre_resp, fs, RESP_LO, RESP_HI)
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Enhanced epoch extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _stage_at_time(sp, t_hr):
    if sp is None:
        return -1, '?'
    idx = int(np.searchsorted(sp['t_ep_hr'], t_hr, side='right')) - 1
    idx = np.clip(idx, 0, len(sp['codes']) - 1)
    code = int(sp['codes'][idx])
    return code, STAGE_LABELS.get(code, '?')


def compute_session_epochs_enhanced(session):
    fs = session.fs
    n_samples = session.n_samples

    raw_cle = session.cap['CLE'].astype(np.float64)
    raw_cre = session.cap['CRE'].astype(np.float64)
    raw_ch = session.cap['CH'].astype(np.float64)
    raw_diff = raw_cle - raw_cre
    raw_avg = (raw_cle + raw_cre) / 2.0
    raw_caps = {'CLE': raw_cle, 'CRE': raw_cre, 'CH': raw_ch,
                'CLE-CRE': raw_diff, 'avg': raw_avg}
    acc_mag = session.cap['acc_mag'].astype(np.float64)

    resp_bp, card_bp = {}, {}
    for ch in CAP_CHANNELS:
        resp_bp[ch] = remove_acc_artifact(raw_caps[ch], acc_mag, RESP_LO, RESP_HI, fs)
        card_bp[ch] = remove_acc_artifact(raw_caps[ch], acc_mag, CARD_LO, CARD_HI, fs)

    thorax_raw = session.psg['Thorax'].astype(np.float64)
    thorax_bp = bandpass(thorax_raw, RESP_LO, RESP_HI, fs)

    motion = epoch_motion(session, epoch_sec=EPOCH_SEC)
    sp = session.sleep_profile
    dur_hr = n_samples / fs / 3600.0

    n_epochs = n_samples // EPOCH_N
    n_motion = len(motion['t_hr'])
    n_ep = min(n_epochs, n_motion)

    rows = []
    for i in range(n_ep):
        s, e = i * EPOCH_N, (i + 1) * EPOCH_N
        t_center_hr = float(np.mean(session.time_hr[s:e]))
        stage_code, stage_label = _stage_at_time(sp, t_center_hr)

        row = {
            'session': session.label,
            'subject': session.subject,
            't_hr': t_center_hr,
            'stage_code': stage_code,
            'stage_label': stage_label,
        }

        # Group A: base features
        for ch in CAP_CHANNELS:
            seg_raw = raw_caps[ch][s:e]
            row[f'{ch}_raw_mean'] = float(np.mean(seg_raw))
            row[f'{ch}_raw_std'] = float(np.std(seg_raw))
            row[f'{ch}_resp_rms'] = float(np.sqrt(np.mean(resp_bp[ch][s:e] ** 2)))
            row[f'{ch}_card_rms'] = float(np.sqrt(np.mean(card_bp[ch][s:e] ** 2)))

        # Thorax target
        seg_thorax_bp = thorax_bp[s:e]
        row['thorax_resp_rms'] = float(np.sqrt(np.mean(seg_thorax_bp ** 2)))

        # Accel
        row['movement_rms'] = float(motion['movement_rms'][i])
        row['movement_peak'] = float(motion['movement_peak'][i])
        row['roll_deg'] = float(motion['roll_deg'][i])
        row['pitch_deg'] = float(motion['pitch_deg'][i])

        # Group B: spectral features for 3 channels
        for ch in SPECTRAL_CHANNELS:
            row.update(compute_spectral_features(raw_caps[ch][s:e], fs, ch))

        # Group C: rate features on CLE-CRE
        row.update(compute_rate_features(resp_bp['CLE-CRE'][s:e], card_bp['CLE-CRE'][s:e], fs))

        # Group D: cross-channel features
        row.update(compute_cross_channel_features(
            resp_bp['CLE'][s:e], resp_bp['CRE'][s:e],
            card_bp['CLE'][s:e], card_bp['CRE'][s:e], fs))

        # Group E: context features
        pos_str = classify_position(
            np.array([motion['roll_deg'][i]]),
            np.array([motion['pitch_deg'][i]])
        )[0]
        row['position_code'] = POSITION_MAP.get(pos_str, 0)
        row['sin_t'] = float(np.sin(2 * np.pi * t_center_hr / dur_hr))
        row['cos_t'] = float(np.cos(2 * np.pi * t_center_hr / dur_hr))
        row['epoch_frac'] = i / max(n_ep - 1, 1)

        rows.append(row)

    return rows


def build_enhanced_epoch_table():
    all_rows = []
    for i in range(len(SESSION_META)):
        sess = load_session(i)
        sess.sleep_profile = load_sleep_profile(sess)
        print(f'  {sess.label}: extracting enhanced epochs...', flush=True)
        rows = compute_session_epochs_enhanced(sess)
        all_rows.extend(rows)
        print(f'    {len(rows)} epochs')
        del sess
    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Temporal features & feature group definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Base CAP features (Group A)
BASE_CAP_COLS = []
for ch in CAP_CHANNELS:
    for suf in ['_raw_mean', '_raw_std', '_resp_rms', '_card_rms']:
        BASE_CAP_COLS.append(f'{ch}{suf}')
ACCEL_COLS = ['movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg']
BASE_FEATURES = BASE_CAP_COLS + ACCEL_COLS

# Spectral features (Group B)
SPECTRAL_COLS = []
for ch in SPECTRAL_CHANNELS:
    for suf in ['_dom_resp_freq', '_resp_spectral_entropy', '_resp_power_ratio',
                '_card_power_ratio', '_resp_card_ratio']:
        SPECTRAL_COLS.append(f'{ch}{suf}')

# Rate features (Group C)
RATE_COLS = ['resp_rate_peaks', 'resp_rate_hilbert', 'resp_rate_acf',
             'card_rate_hilbert', 'resp_rate_variability']

# Cross-channel features (Group D)
CROSS_COLS = ['cle_cre_resp_rms_ratio', 'cle_cre_resp_coherence',
              'cle_cre_card_coherence', 'cle_cre_resp_phase_diff']

# Context features (Group E)
CONTEXT_COLS = ['position_code', 'sin_t', 'cos_t', 'epoch_frac']

ALL_EPOCH_FEATURES = BASE_FEATURES + SPECTRAL_COLS + RATE_COLS + CROSS_COLS + CONTEXT_COLS

# Columns to create lags for
LAG_COLS_KEY = [f'{ch}_resp_rms' for ch in CAP_CHANNELS] + ['movement_rms']
ROLLING_COLS = ['CLE-CRE_resp_rms', 'avg_resp_rms']
DELTA_COLS = ['CLE-CRE_resp_rms', 'avg_resp_rms', 'movement_rms']

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
    if name in BASE_FEATURES:
        return 'base'
    if name in SPECTRAL_COLS:
        return 'spectral'
    if name in RATE_COLS:
        return 'rate'
    if name in CROSS_COLS:
        return 'cross'
    if name in CONTEXT_COLS:
        return 'context'
    if '_lag' in name:
        return 'lag'
    if '_roll' in name:
        return 'rolling'
    if '_delta' in name:
        return 'delta'
    return 'base'


def add_temporal_features(df):
    out = df.copy()
    # Lags
    for col in LAG_COLS_KEY:
        for lag in range(1, N_LAGS + 1):
            out[f'{col}_lag{lag}'] = out.groupby('session')[col].shift(lag)
    # Rolling means
    for col in ROLLING_COLS:
        for w in [3, 5]:
            out[f'{col}_roll{w}'] = out.groupby('session')[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean())
    # Deltas
    for col in DELTA_COLS:
        out[f'{col}_delta'] = out.groupby('session')[col].diff()
    return out


def get_tier_features(tier, df_columns):
    if tier == 0:
        return [c for c in BASE_FEATURES if c in df_columns]

    if tier == 1:
        base = [c for c in BASE_FEATURES if c in df_columns]
        lags = [c for c in df_columns if '_lag' in c]
        return base + lags

    # Tier 2 and 3: all features except metadata and target
    exclude = {'session', 'subject', 't_hr', 'stage_code', 'stage_label',
               TARGET, 'thorax_raw_mean', 'thorax_raw_std'}
    return [c for c in df_columns if c not in exclude]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (reused from thorax_predictor.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _zscore_per_session(df, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out.groupby('session')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-12))
    return out


def _split_train_test(df_sess, train_frac=0.7):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

def run_ridge(df, features):
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
        m.update({'model': 'T0-Ridge', 'eval': 'within-session', 'session': sess})
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
        m.update({'model': 'T0-Ridge', 'eval': 'LOSO', 'session': hold})
        results.append(m)

    return results


def run_xgb_tier(df, tier_name, features, df_raw=None):
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
        m.update({'model': tier_name, 'eval': 'within-session', 'session': sess})
        results.append(m)

        importances_list.append(model.feature_importances_)

        raw_thorax = np.full(len(test), np.nan)
        if df_raw is not None:
            raw_sess = df_raw[df_raw['session'] == sess].sort_values('t_hr').reset_index(drop=True)
            raw_sess = raw_sess.dropna(subset=[TARGET])
            _, raw_test = _split_train_test(raw_sess)
            if len(raw_test) == len(test):
                raw_thorax = raw_test[TARGET].values

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
        m.update({'model': tier_name, 'eval': 'LOSO', 'session': hold})
        results.append(m)

    avg_importance = np.mean(importances_list, axis=0) if importances_list else None

    return results, predictions, avg_importance


def run_xgb_recursive(df, features, df_raw=None, n_pseudo_lags=2):
    import xgboost as xgb
    results = []
    predictions = {}
    sessions = sorted(df['session'].unique())

    pseudo_lag_names = [f'pseudo_thorax_lag{i}' for i in range(1, n_pseudo_lags + 1)]
    all_feats = features + pseudo_lag_names

    for sess in sessions:
        ds = df[df['session'] == sess].sort_values('t_hr').reset_index(drop=True)
        ds = ds.dropna(subset=[TARGET])

        # Add teacher-forced lags for training
        for lag in range(1, n_pseudo_lags + 1):
            ds[f'pseudo_thorax_lag{lag}'] = ds[TARGET].shift(lag)
        ds = ds.dropna(subset=pseudo_lag_names).reset_index(drop=True)

        train, test = _split_train_test(ds)
        if len(train) < 20 or len(test) < 10:
            continue

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(train[all_feats].fillna(0).values, train[TARGET].values,
                  eval_set=[(test[all_feats].fillna(0).values, test[TARGET].values)],
                  verbose=False)

        # Recursive prediction: use own predictions as lags
        test_r = test.copy()
        pred_buffer = list(train[TARGET].values[-(n_pseudo_lags):])
        preds = []
        for idx in range(len(test_r)):
            for lag in range(1, n_pseudo_lags + 1):
                buf_idx = len(pred_buffer) - lag
                test_r.iloc[idx, test_r.columns.get_loc(f'pseudo_thorax_lag{lag}')] = (
                    pred_buffer[buf_idx] if buf_idx >= 0 else 0.0)
            row_feats = test_r.iloc[idx:idx+1][all_feats].fillna(0).values
            p = float(model.predict(row_feats)[0])
            p = np.clip(p, -4.0, 4.0)
            preds.append(p)
            pred_buffer.append(p)

        pred = np.array(preds)
        m = _metrics(test[TARGET].values, pred)
        m.update({'model': 'T3-XGB-Rec', 'eval': 'within-session', 'session': sess})
        results.append(m)

        raw_thorax = np.full(len(test), np.nan)
        if df_raw is not None:
            raw_sess = df_raw[df_raw['session'] == sess].sort_values('t_hr').reset_index(drop=True)
            for lag in range(1, n_pseudo_lags + 1):
                raw_sess[f'pseudo_thorax_lag{lag}'] = raw_sess[TARGET].shift(lag)
            raw_sess = raw_sess.dropna(subset=pseudo_lag_names).reset_index(drop=True)
            _, raw_test = _split_train_test(raw_sess)
            if len(raw_test) == len(test):
                raw_thorax = raw_test[TARGET].values

        full_raw_t = np.array([])
        full_raw_thorax = np.array([])
        if df_raw is not None:
            raw_sess_full = df_raw[df_raw['session'] == sess].sort_values('t_hr')
            full_raw_t = raw_sess_full['t_hr'].values
            full_raw_thorax = raw_sess_full[TARGET].values

        predictions[sess] = {
            't_hr': test['t_hr'].values,
            'actual': test[TARGET].values,
            'predicted': pred,
            'stage_code': test['stage_code'].values,
            'raw_thorax': raw_thorax,
            'full_raw_t': full_raw_t,
            'full_raw_thorax': full_raw_thorax,
        }

    # LOSO with recursive prediction
    for hold in sessions:
        train_df = df[df['session'] != hold].dropna(subset=[TARGET]).copy()
        test_df = df[df['session'] == hold].sort_values('t_hr').dropna(subset=[TARGET]).copy()

        for lag in range(1, n_pseudo_lags + 1):
            train_df[f'pseudo_thorax_lag{lag}'] = train_df.groupby('session')[TARGET].shift(lag)
            test_df[f'pseudo_thorax_lag{lag}'] = test_df[TARGET].shift(lag)
        train_df = train_df.dropna(subset=pseudo_lag_names)
        test_df_full = test_df.dropna(subset=pseudo_lag_names).reset_index(drop=True)

        if len(train_df) < 50 or len(test_df_full) < 10:
            continue

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(train_df[all_feats].fillna(0).values, train_df[TARGET].values, verbose=False)

        # Recursive LOSO
        pred_buffer = [0.0] * n_pseudo_lags
        preds = []
        test_rec = test_df_full.copy()
        for idx in range(len(test_rec)):
            for lag in range(1, n_pseudo_lags + 1):
                buf_idx = len(pred_buffer) - lag
                test_rec.iloc[idx, test_rec.columns.get_loc(f'pseudo_thorax_lag{lag}')] = (
                    pred_buffer[buf_idx] if buf_idx >= 0 else 0.0)
            row_feats = test_rec.iloc[idx:idx+1][all_feats].fillna(0).values
            p = float(model.predict(row_feats)[0])
            p = np.clip(p, -4.0, 4.0)
            preds.append(p)
            pred_buffer.append(p)

        pred = np.array(preds)
        m = _metrics(test_df_full[TARGET].values, pred)
        m.update({'model': 'T3-XGB-Rec', 'eval': 'LOSO', 'session': hold})
        results.append(m)

    return results, predictions


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def load_reference_results():
    ref_path = ART_DIR / 'thorax_predictor_results.csv'
    if not ref_path.exists():
        return None
    return pd.read_csv(ref_path)


def plot_summary(res_df, ref_df):
    models = ['T0-Ridge', 'T1-XGB-Base', 'T2-XGB-Enh', 'T3-XGB-Rec']
    colors = {'T0-Ridge': '#3498DB', 'T1-XGB-Base': '#27AE60',
              'T2-XGB-Enh': '#E67E22', 'T3-XGB-Rec': '#8E44AD'}
    evals = ['within-session', 'LOSO']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ref_vals = {}
    if ref_df is not None:
        xgb_ref = ref_df[ref_df['model'] == 'XGBoost']
        for ev in evals:
            sub = xgb_ref[xgb_ref['eval'] == ev]
            ref_vals[(ev, 'rmse')] = sub['rmse'].median()
            ref_vals[(ev, 'r2')] = sub['r2'].median()

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

            ref_key = (ev, metric)
            if ref_key in ref_vals and np.isfinite(ref_vals[ref_key]):
                ax.axhline(ref_vals[ref_key], color='red', ls='--', lw=1.2,
                           label=f'Ref XGBoost ({ref_vals[ref_key]:.3f})')
                ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('CAP-Only -> Thorax resp RMS: Model Comparison', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / 'caponly_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved caponly_summary.png')


def plot_per_session(res_df, ref_df):
    models = ['T0-Ridge', 'T1-XGB-Base', 'T2-XGB-Enh', 'T3-XGB-Rec']
    colors = {'T0-Ridge': '#3498DB', 'T1-XGB-Base': '#27AE60',
              'T2-XGB-Enh': '#E67E22', 'T3-XGB-Rec': '#8E44AD'}
    sub = res_df[res_df['eval'] == 'within-session']
    sessions = sorted(sub['session'].unique())

    fig, ax = plt.subplots(figsize=(16, 6))
    n_models = len(models)
    width = 0.18
    x = np.arange(len(sessions))

    for i, mod in enumerate(models):
        ms = sub[sub['model'] == mod]
        r2_vals = [ms[ms['session'] == s]['r2'].values[0]
                   if len(ms[ms['session'] == s]) > 0 else np.nan
                   for s in sessions]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, r2_vals, width=width, color=colors[mod],
               alpha=0.8, label=mod)

    if ref_df is not None:
        ref_xgb = ref_df[(ref_df['model'] == 'XGBoost') & (ref_df['eval'] == 'within-session')]
        ref_r2 = [ref_xgb[ref_xgb['session'] == s]['r2'].values[0]
                  if len(ref_xgb[ref_xgb['session'] == s]) > 0 else np.nan
                  for s in sessions]
        ax.plot(x, ref_r2, 'rv--', markersize=6, lw=1.2, label='Ref XGBoost (w/ thorax lags)')

    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontsize=8, rotation=45)
    ax.set_ylabel('R2', fontsize=10)
    ax.set_title('Within-session R2 by model and session', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', ncol=5)
    ax.grid(True, axis='y', alpha=0.2)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    fig.savefig(PLOT_DIR / 'caponly_per_session.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved caponly_per_session.png')


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
    ax.set_title('Top 30 Features — XGBoost CAP-Enhanced (Tier 2)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.2)

    from matplotlib.patches import Patch
    seen = {}
    for g in groups:
        if g not in seen:
            seen[g] = FEATURE_GROUP_COLORS.get(g, '#999')
    legend_handles = [Patch(facecolor=c, label=g) for g, c in seen.items()]
    ax.legend(handles=legend_handles, fontsize=7, loc='lower right')

    fig.savefig(PLOT_DIR / 'caponly_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved caponly_feature_importance.png')


def plot_timeseries(predictions, tier_name):
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

        ok = np.isfinite(predicted) & np.isfinite(actual)
        r2 = r2_score(actual[ok], predicted[ok]) if ok.sum() >= 5 else np.nan
        mae = mean_absolute_error(actual[ok], predicted[ok]) if ok.sum() >= 5 else np.nan

        ax = axes[0]
        if len(full_t) > 0:
            ax.plot(full_t, full_thorax, color='#95A5A6', lw=0.5, alpha=0.6)
            ax.axvspan(t[0], t[-1], color='#E67E22', alpha=0.08, label='Test region (30%)')
            ax.plot(t, data.get('raw_thorax', actual),
                    color='#2C3E50', lw=0.7, alpha=0.9, label='Test raw')
        ax.set_ylabel('thorax_resp_rms\n(raw)', fontsize=8)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.set_title(f'{sess}  --  {tier_name} within-session  '
                     f'R2={r2:.3f}  MAE={mae:.3f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

        ax = axes[1]
        ax.plot(t, actual, color='#2C3E50', lw=0.8, alpha=0.9, label='Actual (z-scored)')
        ax.plot(t, predicted, color='#E67E22', lw=0.8, alpha=0.8, label=f'{tier_name} pred')
        ax.set_ylabel('thorax_resp_rms\n(z-scored)', fontsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

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

        fig.savefig(PLOT_DIR / f'caponly_timeseries_{sess}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'  Saved {len(predictions)} timeseries plots')


def plot_gap_analysis(res_df, ref_df):
    if ref_df is None:
        return
    ref_xgb = ref_df[(ref_df['model'] == 'XGBoost') & (ref_df['eval'] == 'within-session')]
    best_tier = 'T2-XGB-Enh'
    cap_only = res_df[(res_df['model'] == best_tier) & (res_df['eval'] == 'within-session')]

    sessions = sorted(set(ref_xgb['session']) & set(cap_only['session']))
    if not sessions:
        return

    ref_r2, cap_r2 = [], []
    for s in sessions:
        rr = ref_xgb[ref_xgb['session'] == s]['r2'].values
        cr = cap_only[cap_only['session'] == s]['r2'].values
        if len(rr) > 0 and len(cr) > 0:
            ref_r2.append(rr[0])
            cap_r2.append(cr[0])
        else:
            sessions = [x for x in sessions if x != s]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ref_r2, cap_r2, s=60, c='#E67E22', edgecolors='#2C3E50', zorder=5)
    for s, rx, cx in zip(sessions, ref_r2, cap_r2):
        ax.annotate(s, (rx, cx), fontsize=7, textcoords='offset points',
                    xytext=(5, 5))

    lims = [min(min(ref_r2), min(cap_r2)) - 0.05, max(max(ref_r2), max(cap_r2)) + 0.05]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='Parity')
    ax.set_xlabel('Original XGBoost R2 (with thorax lags)', fontsize=10)
    ax.set_ylabel(f'CAP-Only {best_tier} R2', fontsize=10)
    ax.set_title('Gap Analysis: CAP-Only vs Reference', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    fig.savefig(PLOT_DIR / 'caponly_gap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved caponly_gap_analysis.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('CAP-Only Thorax Predictor')
    print('=' * 60)

    # -- Phase 1: Feature extraction --
    pq_path = ART_DIR / 'thorax_caponly_epochs.parquet'
    if pq_path.exists():
        print(f'\nLoading cached {pq_path}')
        df_raw = pd.read_parquet(pq_path)
    else:
        print('\n-- Phase 1: Enhanced epoch extraction --')
        df_raw = build_enhanced_epoch_table()
        df_raw.to_parquet(pq_path, index=False)
        print(f'Saved {len(df_raw)} rows x {len(df_raw.columns)} cols to {pq_path}')

    print(f'  {len(df_raw)} epochs, {df_raw.session.nunique()} sessions, {len(df_raw.columns)} cols')

    # Add temporal features
    print('\n-- Adding temporal features --')
    df_raw = add_temporal_features(df_raw)
    print(f'  Now {len(df_raw.columns)} columns')

    # Z-score per session
    z_cols = [c for c in df_raw.columns
              if c not in {'session', 'subject', 't_hr', 'stage_code', 'stage_label',
                           'position_code', 'epoch_frac', 'sin_t', 'cos_t'}
              and df_raw[c].dtype in [np.float64, np.float32, float]]
    df = _zscore_per_session(df_raw, z_cols)

    # Load reference results
    ref_df = load_reference_results()
    all_results = []

    # -- Tier 0: Ridge --
    print('\n-- Tier 0: Ridge (CAP base only) --')
    t0_feats = get_tier_features(0, df.columns.tolist())
    print(f'  {len(t0_feats)} features')
    ridge_res = run_ridge(df, t0_feats)
    all_results.extend(ridge_res)
    ws = [r for r in ridge_res if r['eval'] == 'within-session']
    lo = [r for r in ridge_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    if lo:
        print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # -- Tier 1: XGBoost base --
    print('\n-- Tier 1: XGBoost (base + CAP lags) --')
    t1_feats = get_tier_features(1, df.columns.tolist())
    print(f'  {len(t1_feats)} features')
    t1_res, t1_preds, _ = run_xgb_tier(df, 'T1-XGB-Base', t1_feats, df_raw)
    all_results.extend(t1_res)
    ws = [r for r in t1_res if r['eval'] == 'within-session']
    lo = [r for r in t1_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    if lo:
        print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # -- Tier 2: XGBoost enhanced --
    print('\n-- Tier 2: XGBoost (all CAP-only features) --')
    t2_feats = get_tier_features(2, df.columns.tolist())
    print(f'  {len(t2_feats)} features')
    t2_res, t2_preds, t2_importance = run_xgb_tier(df, 'T2-XGB-Enh', t2_feats, df_raw)
    all_results.extend(t2_res)
    ws = [r for r in t2_res if r['eval'] == 'within-session']
    lo = [r for r in t2_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    if lo:
        print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # -- Tier 3: XGBoost recursive --
    print('\n-- Tier 3: XGBoost recursive (pseudo-lags) --')
    t3_res, t3_preds = run_xgb_recursive(df, t2_feats, df_raw, n_pseudo_lags=2)
    all_results.extend(t3_res)
    ws = [r for r in t3_res if r['eval'] == 'within-session']
    lo = [r for r in t3_res if r['eval'] == 'LOSO']
    print(f'  Within-session median R2 = {np.nanmedian([r["r2"] for r in ws]):.3f}')
    if lo:
        print(f'  LOSO median R2 = {np.nanmedian([r["r2"] for r in lo]):.3f}')

    # -- Save results --
    res_df = pd.DataFrame(all_results)
    csv_path = ART_DIR / 'thorax_caponly_results.csv'
    res_df.to_csv(csv_path, index=False)
    print(f'\nResults saved to {csv_path}')

    # -- Summary table --
    print('\n-- Summary (median across sessions) --')
    summary = res_df.groupby(['model', 'eval'])[['rmse', 'mae', 'r2']].median()
    print(summary.to_string())

    if ref_df is not None:
        ref_xgb = ref_df[ref_df['model'] == 'XGBoost']
        print('\n-- Reference (original XGBoost with thorax lags) --')
        ref_summary = ref_xgb.groupby('eval')[['rmse', 'mae', 'r2']].median()
        print(ref_summary.to_string())

    # -- Plots --
    print('\n-- Generating plots --')
    plot_summary(res_df, ref_df)
    plot_per_session(res_df, ref_df)
    plot_feature_importance(t2_importance, t2_feats)

    best_preds = t2_preds if t2_preds else t1_preds
    best_name = 'T2-XGB-Enh' if t2_preds else 'T1-XGB-Base'
    plot_timeseries(best_preds, best_name)
    plot_gap_analysis(res_df, ref_df)

    print(f'\nAll outputs in {PLOT_DIR} and {ART_DIR}')


if __name__ == '__main__':
    main()
