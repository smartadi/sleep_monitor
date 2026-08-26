#!/usr/bin/env python
"""
Validation study: compute per-epoch (30 s) CAP rate estimates vs PSG ground truth
for all 12 sessions using the best-performing estimators.

Respiratory : rate_peaks_scaled_resp  / calibrated k  (CLE-CRE, OLS)
Cardiac     : rate_hilbert_scaled_cardiac / calibrated k  (CLE-CRE, OLS)

Outputs
-------
artifacts/validation_windows.parquet   — one row per 30 s epoch per session
artifacts/validation_session.csv       — per-session summary metrics + k values
artifacts/validation_stage.csv         — per-stage summary metrics (pooled)

Usage
-----
    python scripts/run_validation.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, STAGE_LABELS,
)
from sleep_monitor.loader import load_all_sessions
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.ground_truth import gt_sliding_rates
from sleep_monitor.rates import (
    rate_peaks_scaled_resp,
    rate_hilbert_scaled_cardiac,
    calibrate_k_resp,
    calibrate_k_cardiac,
)

# ── Parameters ────────────────────────────────────────────────────────────────

WIN_SEC  = 30.0          # epoch length — matches PSG scoring
STEP_SEC = 30.0          # non-overlapping epochs
K_CAL_WIN = 30.0         # calibrate k at same window size
K_CAL_N   = 50           # number of random windows for k calibration
CHANNEL  = 'CLE-CRE'
PREPROC  = 'ols'
SEED     = 42
OUT_DIR  = ROOT / 'artifacts'


def assign_sleep_stage(t_hr: np.ndarray, profile: dict | None) -> np.ndarray:
    """Map each window centre time to a sleep stage code from the 30 s profile."""
    codes = np.full(len(t_hr), -1, dtype=np.int8)
    if profile is None:
        return codes
    ep_t = profile['t_ep_hr']
    ep_c = profile['codes']
    epoch_dur_hr = PSG_EPOCH_SEC / 3600.0
    for i, t in enumerate(t_hr):
        idx = int(t / epoch_dur_hr)
        if 0 <= idx < len(ep_c):
            codes[i] = ep_c[idx]
    return codes


def compute_session(session) -> pd.DataFrame:
    """Compute validation DataFrame for one session."""
    fs = session.fs
    label = session.label
    subject = session.subject

    # --- Calibrate k at 30 s windows ---
    k_resp = calibrate_k_resp(session, n_windows=K_CAL_N, win_s=K_CAL_WIN, seed=SEED)
    k_card = calibrate_k_cardiac(session, n_windows=K_CAL_N, win_s=K_CAL_WIN, seed=SEED)
    print(f"  {label}: k_resp={k_resp:.3f}, k_card={k_card:.3f}")

    # --- Prepare CAP signal (CLE-CRE, OLS artifact removal) ---
    raw_cap = session.cap['CLE'].astype(np.float64) - session.cap['CRE'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)

    sig_resp = remove_acc_artifact(raw_cap, acc, RESP_LO, RESP_HI, fs)
    sig_card = remove_acc_artifact(raw_cap, acc, CARD_LO, CARD_HI, fs)

    # --- GT from ECG R-peaks / Flow peaks ---
    gt = gt_sliding_rates(session, win_sec=WIN_SEC, step_sec=STEP_SEC)
    t_hr = gt['t_hr']
    gt_resp_hz = gt['resp_hz']
    gt_card_hz = gt['card_hz']

    # --- CAP rate per epoch ---
    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    n_samples = len(sig_resp)

    cap_resp_hz = np.full(len(t_hr), np.nan)
    cap_card_hz = np.full(len(t_hr), np.nan)

    for i, start in enumerate(range(0, n_samples - win_n + 1, step_n)):
        if i >= len(t_hr):
            break
        seg_resp = sig_resp[start:start + win_n]
        seg_card = sig_card[start:start + win_n]

        cap_resp_hz[i] = rate_peaks_scaled_resp(seg_resp, k=k_resp, fs=fs)
        cap_card_hz[i] = rate_hilbert_scaled_cardiac(seg_card, k=k_card, fs=fs)

    # --- Assign sleep stage ---
    stage_codes = assign_sleep_stage(t_hr, session.sleep_profile)
    stage_labels = [STAGE_LABELS.get(c, '?') for c in stage_codes]

    return pd.DataFrame({
        'session':      label,
        'subject':      subject,
        't_hr':         t_hr,
        'cap_resp_hz':  cap_resp_hz,
        'gt_resp_hz':   gt_resp_hz,
        'cap_card_hz':  cap_card_hz,
        'gt_card_hz':   gt_card_hz,
        'stage_code':   stage_codes,
        'stage':        stage_labels,
        'k_resp':       k_resp,
        'k_card':       k_card,
    })


def compute_metrics(pred: np.ndarray, ref: np.ndarray, scale: float = 60.0) -> dict:
    """MAE, RMSE, bias, Pearson r, p50, p90 on valid pairs, scaled to per-minute units."""
    ok = np.isfinite(pred) & np.isfinite(ref)
    n = int(ok.sum())
    if n < 3:
        return dict(n=n, mae=np.nan, rmse=np.nan, bias=np.nan,
                    r=np.nan, p50=np.nan, p90=np.nan, coverage=0.0)
    p, r_ = pred[ok] * scale, ref[ok] * scale
    err = p - r_
    r_val, _ = pearsonr(p, r_)
    return dict(
        n        = n,
        mae      = float(np.mean(np.abs(err))),
        rmse     = float(np.sqrt(np.mean(err ** 2))),
        bias     = float(np.mean(err)),
        r        = float(r_val),
        p50      = float(np.median(np.abs(err))),
        p90      = float(np.quantile(np.abs(err), 0.90)),
        coverage = float(n / len(pred)),
    )


def session_metrics(df: pd.DataFrame) -> dict:
    """Compute resp + cardiac metrics for one session."""
    resp = compute_metrics(df['cap_resp_hz'].values, df['gt_resp_hz'].values, scale=60.0)
    card = compute_metrics(df['cap_card_hz'].values, df['gt_card_hz'].values, scale=60.0)
    return {
        'session': df['session'].iloc[0],
        'subject': df['subject'].iloc[0],
        'k_resp':  df['k_resp'].iloc[0],
        'k_card':  df['k_card'].iloc[0],
        **{f'resp_{k}': v for k, v in resp.items()},
        **{f'card_{k}': v for k, v in card.items()},
    }


def stage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute resp + cardiac metrics stratified by sleep stage (pooled across sessions)."""
    rows = []
    for stage_name, grp in df.groupby('stage'):
        if stage_name == '?':
            continue
        resp = compute_metrics(grp['cap_resp_hz'].values, grp['gt_resp_hz'].values, scale=60.0)
        card = compute_metrics(grp['cap_card_hz'].values, grp['gt_card_hz'].values, scale=60.0)
        rows.append({
            'stage': stage_name,
            **{f'resp_{k}': v for k, v in resp.items()},
            **{f'card_{k}': v for k, v in card.items()},
        })
    stage_order = ['Wake', 'N1', 'N2', 'N3', 'REM']
    out = pd.DataFrame(rows)
    out['stage'] = pd.Categorical(out['stage'], categories=stage_order, ordered=True)
    return out.sort_values('stage').reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all 12 sessions with sleep profiles...")
    t0 = time.time()
    sessions = load_all_sessions(with_sleep_profiles=True)
    print(f"  Loaded {len(sessions)} sessions in {time.time() - t0:.1f}s\n")

    # --- Per-session computation ---
    frames = []
    for s in sessions:
        print(f"Processing {s.label}...")
        df = compute_session(s)
        frames.append(df)
        print(f"  {len(df)} epochs, "
              f"resp coverage={df['cap_resp_hz'].notna().mean():.1%}, "
              f"card coverage={df['cap_card_hz'].notna().mean():.1%}")

    all_windows = pd.concat(frames, ignore_index=True)

    # --- Per-session summary ---
    sess_rows = [session_metrics(g) for _, g in all_windows.groupby('session', sort=False)]
    sess_df = pd.DataFrame(sess_rows)

    # --- Add aggregate row ---
    agg_resp = compute_metrics(all_windows['cap_resp_hz'].values,
                               all_windows['gt_resp_hz'].values, scale=60.0)
    agg_card = compute_metrics(all_windows['cap_card_hz'].values,
                               all_windows['gt_card_hz'].values, scale=60.0)
    agg_row = {
        'session': 'ALL',
        'subject': 'ALL',
        'k_resp': all_windows['k_resp'].median(),
        'k_card': all_windows['k_card'].median(),
        **{f'resp_{k}': v for k, v in agg_resp.items()},
        **{f'card_{k}': v for k, v in agg_card.items()},
    }
    sess_df = pd.concat([sess_df, pd.DataFrame([agg_row])], ignore_index=True)

    # --- Per-stage summary ---
    stage_df = stage_metrics(all_windows)

    # --- Save ---
    all_windows.to_parquet(OUT_DIR / 'validation_windows.parquet', index=False)
    sess_df.to_csv(OUT_DIR / 'validation_session.csv', index=False, float_format='%.4f')
    stage_df.to_csv(OUT_DIR / 'validation_stage.csv', index=False, float_format='%.4f')

    print(f"\n{'='*70}")
    print("Per-session summary:")
    print(sess_df[['session', 'subject', 'k_resp', 'k_card',
                   'resp_mae', 'resp_r', 'card_mae', 'card_r']].to_string(index=False))
    print(f"\nPer-stage summary:")
    print(stage_df[['stage', 'resp_mae', 'resp_r', 'resp_n',
                    'card_mae', 'card_r', 'card_n']].to_string(index=False))

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  validation_windows.parquet  ({len(all_windows)} rows)")
    print(f"  validation_session.csv")
    print(f"  validation_stage.csv")


if __name__ == '__main__':
    main()
