"""
sleep_monitor/classifier.py — rate-prediction classifiers.

Given per-window features (method estimates + quality features), learn to
predict the ground-truth rate. Trains and evaluates several models under
leave-one-subject-out (LOSO) cross-validation.

Design
------
Inputs per window (assembled by `build_dataset`):
    6 method estimates  : rate_{spectral,acf,hilbert,zerocross,peaks,envelope}_hz
                           (envelope only for cardiac; NaN elsewhere)
    method-is-nan flags : indicator columns for each method
    quality features    : snr_db, acf_prom, spec_conc, motion_db, rms, agreement_hz
    channel/preproc     : one-hot, so the model can learn channel preferences
Target:
    gt_rate_hz (same units as features). Rows with missing target are dropped.

Models (default set)
--------------------
ridge           : linear baseline with standardisation
rf              : RandomForestRegressor
hgb             : HistGradientBoostingRegressor
mlp             : MLPRegressor — small (64, 32)

Metrics (per fold, unit-scaled: Hz * 60 → br/min / BPM)
    mae, rmse, r, bias, p90_abs_err
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr


# ── Feature set ───────────────────────────────────────────────────────────────

METHOD_COLS = [
    'rate_spectral_hz', 'rate_acf_hz', 'rate_hilbert_hz',
    'rate_zerocross_hz', 'rate_peaks_hz', 'rate_envelope_hz',
]
QUALITY_COLS = [
    'snr_db', 'acf_prom', 'spec_conc', 'motion_db', 'rms', 'agreement_hz',
    'quality',
]


# ── Dataset assembly ──────────────────────────────────────────────────────────

def _subject_from_label(label: str) -> str:
    """'S3N2' -> 'S3'  (subject groups for LOSO)."""
    return label.split('N')[0]


def load_windows(windows_dir: Path, band: str,
                  preproc: Optional[str] = None,
                  channel: Optional[str] = None) -> pd.DataFrame:
    """
    Load all per-window parquet files for one band, optionally filtered to
    a specific channel and/or preprocessing.

    Filenames follow:  <band>_<channel>_<preproc>_w<win>__<session>.parquet
    """
    all_files = sorted(windows_dir.glob(f'{band}_*__*.parquet'))
    if not all_files:
        raise FileNotFoundError(f'No files matching "{band}_*" in {windows_dir}')

    frames = []
    for f in all_files:
        base_tag, _, _ = f.stem.partition('__')
        parts = base_tag.split('_')
        # ['resp', 'CLE-CRE', 'ols', 'w30']
        ch = parts[1] if len(parts) > 1 else ''
        pp = parts[2] if len(parts) > 2 else ''
        if channel is not None and ch != channel:
            continue
        if preproc is not None and pp != preproc:
            continue
        df = pd.read_parquet(f)
        df['base_channel'] = ch
        df['base_preproc'] = pp
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f'No files match band={band} channel={channel} preproc={preproc}')
    out = pd.concat(frames, ignore_index=True)
    out['subject'] = out['session'].map(_subject_from_label)
    return out


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    From a stacked windows DataFrame, assemble (X, y, groups, meta_df).

    X        : feature matrix (pandas DataFrame, preserves column names)
    y        : target rate in Hz
    groups   : subject label per row (for LOSO split)
    meta_df  : side information (session, t_s, band, base_channel, base_preproc)
    """
    keep = df['gt_rate_hz'].notna() & np.isfinite(df['gt_rate_hz'])
    sub  = df.loc[keep].reset_index(drop=True)
    if 'subject' not in sub.columns:
        sub['subject'] = sub['session'].map(_subject_from_label)

    method_cols = [c for c in METHOD_COLS if c in sub.columns]
    qual_cols   = [c for c in QUALITY_COLS if c in sub.columns]

    X = sub[method_cols + qual_cols].copy()
    for c in method_cols:
        X[c + '_isnan'] = (~np.isfinite(sub[c])).astype(float)
    # one-hot base_channel and base_preproc (if present)
    for cat in ('base_channel', 'base_preproc'):
        if cat in sub.columns:
            dummies = pd.get_dummies(sub[cat], prefix=cat, dtype=float)
            X = pd.concat([X, dummies], axis=1)

    y      = sub['gt_rate_hz'].to_numpy(dtype=float)
    groups = sub['subject'].to_numpy()
    meta   = sub[['session', 'subject', 't_s', 'band',
                   'base_channel', 'base_preproc']].copy()
    return X, y, groups, meta


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name:    str
    make_fn: callable
    needs_imputation: bool = True


def _make_ridge():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
        ('model',  Ridge(alpha=1.0)),
    ])


def _make_rf():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('model',  RandomForestRegressor(
            n_estimators=200, min_samples_leaf=4, n_jobs=-1, random_state=0)),
    ])


def _make_hgb():
    # HGB handles NaN natively — no imputer needed.
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=8,
        l2_regularization=1.0, random_state=0,
    )


def _make_mlp():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
        ('model',  MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu', solver='adam',
            max_iter=200, early_stopping=True,
            random_state=0)),
    ])


def default_models() -> List[ModelSpec]:
    return [
        ModelSpec('ridge', _make_ridge),
        ModelSpec('rf',    _make_rf),
        ModelSpec('hgb',   _make_hgb, needs_imputation=False),
        ModelSpec('mlp',   _make_mlp),
    ]


# ── Baseline: plain median of method estimates ───────────────────────────────

def _baseline_median(X: pd.DataFrame, y: np.ndarray,
                      f_lo: float, f_hi: float) -> np.ndarray:
    """
    Non-ML baseline: per-row median of in-band method estimates.
    Serves as the benchmark every model must beat.
    """
    method_cols = [c for c in METHOD_COLS if c in X.columns]
    mat = X[method_cols].to_numpy(dtype=float)
    mask = np.isfinite(mat) & (mat >= f_lo) & (mat <= f_hi)
    preds = np.full(mat.shape[0], np.nan)
    for i in range(mat.shape[0]):
        row = mat[i, mask[i]]
        if len(row) > 0:
            preds[i] = np.median(row)
    return preds


# ── Metrics ───────────────────────────────────────────────────────────────────

def _fold_metrics(pred: np.ndarray, ref: np.ndarray, scale: float = 60.0) -> dict:
    ok = np.isfinite(pred) & np.isfinite(ref)
    n = int(ok.sum())
    if n < 5:
        return dict(n=n, coverage=0.0, mae=np.nan, rmse=np.nan,
                    r=np.nan, bias=np.nan, p90_abs_err=np.nan)
    p = pred[ok] * scale
    r = ref[ok]  * scale
    err = p - r
    r_val = float(pearsonr(p, r)[0]) if n >= 3 else np.nan
    return dict(
        n          = n,
        coverage   = n / len(pred),
        mae        = float(np.mean(np.abs(err))),
        rmse       = float(np.sqrt(np.mean(err**2))),
        r          = r_val,
        bias       = float(np.mean(err)),
        p90_abs_err= float(np.quantile(np.abs(err), 0.90)),
    )


# ── LOSO driver ───────────────────────────────────────────────────────────────

def loso_evaluate(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
    models: Optional[List[ModelSpec]] = None,
    band: str = 'resp',
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Leave-One-Subject-Out cross-validation.

    Returns
    -------
    metrics_df : one row per (held_subject, model), with fold metrics.
    oof_preds  : {model_name: (N,) array of held-out predictions aligned with X}
    """
    from .config import RESP_LO, RESP_HI, CARD_LO, CARD_HI
    f_lo, f_hi = (RESP_LO, RESP_HI) if band == 'resp' else (CARD_LO, CARD_HI)

    if models is None:
        models = default_models()

    subjects = np.array(sorted(np.unique(groups)))
    rows: List[dict] = []
    oof = {m.name: np.full(len(y), np.nan) for m in models}
    oof['baseline_median'] = np.full(len(y), np.nan)

    for held in subjects:
        train_mask = groups != held
        test_mask  = groups == held
        Xtr, ytr = X.loc[train_mask], y[train_mask]
        Xte, yte = X.loc[test_mask],  y[test_mask]
        test_idx = np.where(test_mask)[0]

        # baseline
        base_pred = _baseline_median(Xte, yte, f_lo, f_hi)
        oof['baseline_median'][test_idx] = base_pred
        m = _fold_metrics(base_pred, yte)
        m.update(held=held, model='baseline_median', band=band,
                  n_train=int(train_mask.sum()))
        rows.append(m)

        # ML models
        for spec in models:
            est = spec.make_fn()
            est.fit(Xtr, ytr)
            pred = est.predict(Xte)
            # Band-clamp predictions (refuse to emit impossibilities)
            pred = np.where((pred >= f_lo) & (pred <= f_hi), pred, np.nan)
            oof[spec.name][test_idx] = pred
            m = _fold_metrics(pred, yte)
            m.update(held=held, model=spec.name, band=band,
                      n_train=int(train_mask.sum()))
            rows.append(m)

    return pd.DataFrame(rows), oof


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of fold metrics, per (band, model)."""
    agg = (metrics_df.groupby(['band', 'model'])[['mae', 'rmse', 'r', 'bias',
                                                       'coverage', 'p90_abs_err']]
              .agg(['mean', 'std'])
              .reset_index())
    agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
    return agg.sort_values(['band', 'mae_mean'])
