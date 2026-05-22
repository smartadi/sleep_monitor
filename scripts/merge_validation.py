#!/usr/bin/env python
"""
Merge rate validation results with signal-level validation outputs.

Joins:
  artifacts/validation_windows.parquet  (CAP rates, GT rates, stage)
  artifacts/signal_validation.parquet   (coherence, xcorr, surrogate, spectral)

on (session, epoch_idx).

Output
------
artifacts/merged_validation.parquet  — unified analysis table with all columns

Usage
-----
    python scripts/merge_validation.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifacts'


def main():
    # --- Load both parquets ---
    rates_path = ART / 'validation_windows.parquet'
    signal_path = ART / 'signal_validation.parquet'

    if not rates_path.exists():
        raise FileNotFoundError(
            f"{rates_path} not found — run scripts/run_validation.py first")
    if not signal_path.exists():
        raise FileNotFoundError(
            f"{signal_path} not found — run scripts/signal_validation.py first")

    rates = pd.read_parquet(rates_path)
    signal = pd.read_parquet(signal_path)

    print(f"Rate validation:   {len(rates)} rows, columns: {list(rates.columns)}")
    print(f"Signal validation: {len(signal)} rows, columns: {list(signal.columns)}")

    # Both have the same 30 s epoch grid per session. Add epoch_idx to rates.
    rates['epoch_idx'] = rates.groupby('session').cumcount()

    # Merge on session + epoch_idx
    # Signal df is the authority for stage, apnea, motion — bring those from signal
    # Rates df has: cap_resp_hz, gt_resp_hz, cap_card_hz, gt_card_hz, k_resp, k_card
    rate_cols = ['session', 'epoch_idx', 'cap_resp_hz', 'gt_resp_hz',
                 'cap_card_hz', 'gt_card_hz', 'k_resp', 'k_card']
    merged = signal.merge(rates[rate_cols], on=['session', 'epoch_idx'], how='left')

    # Compute rate errors (in BPM / breaths-per-min)
    merged['resp_err_bpm'] = (merged['cap_resp_hz'] - merged['gt_resp_hz']) * 60.0
    merged['card_err_bpm'] = (merged['cap_card_hz'] - merged['gt_card_hz']) * 60.0
    merged['resp_abs_err_bpm'] = merged['resp_err_bpm'].abs()
    merged['card_abs_err_bpm'] = merged['card_err_bpm'].abs()

    # --- Save ---
    out_path = ART / 'merged_validation.parquet'
    merged.to_parquet(out_path, index=False)

    # --- Summary ---
    print(f"\nMerged: {len(merged)} rows, {len(merged.columns)} columns")
    print(f"Columns: {list(merged.columns)}")

    # Quick stats
    clean = merged[~merged['motion_flag']]
    print(f"\nClean epochs: {len(clean)}")
    for col in ['resp_abs_err_bpm', 'card_abs_err_bpm']:
        vals = clean[col].dropna()
        print(f"  {col}: median={vals.median():.2f}, mean={vals.mean():.2f} (n={len(vals)})")

    # Apnea breakdown
    print(f"\nApnea distribution:")
    print(merged['apnea_code'].value_counts().to_string())

    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
