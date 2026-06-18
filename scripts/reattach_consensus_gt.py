#!/usr/bin/env python
"""
Re-attach consensus respiratory GT to artifacts/mask_phase_a.parquet.

Replaces gt_hz for resp rows with the multi-signal consensus rate,
sampled at the IDENTICAL mask epoch times (exact join on session+t_hr,
no merge_asof). Cardiac rows are untouched.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ART = ROOT / 'artifacts'
MASK_PATH = ART / 'mask_phase_a.parquet'
GT_PATH = ART / 'consolidated_resp_gt.parquet'


def main():
    mask = pd.read_parquet(MASK_PATH)
    gt = pd.read_parquet(GT_PATH)

    n_resp = (mask['band'] == 'resp').sum()
    n_card = (mask['band'] == 'card').sum()
    print(f'Loaded mask_phase_a: {len(mask)} rows ({n_resp} resp, {n_card} card)')
    print(f'Loaded consensus GT: {len(gt)} rows')

    old_resp_gt = mask.loc[mask['band'] == 'resp', 'gt_hz'].copy()
    print(f'\nOld resp gt_hz: mean={old_resp_gt.mean():.6f}, nan={old_resp_gt.isna().sum()}')

    # Build lookup: (session, rounded_t_hr) -> consensus rate
    lookup = {}
    for _, row in gt.iterrows():
        key = (row['session'], round(row['t_hr'], 8))
        lookup[key] = row['rate_consensus']
    print(f'Consensus lookup: {len(lookup)} entries')

    # Replace gt_hz for resp rows via exact match
    resp_mask = mask['band'] == 'resp'
    new_gt = np.full(resp_mask.sum(), np.nan)
    matched = 0
    for i, (idx, row) in enumerate(mask[resp_mask].iterrows()):
        key = (row['session'], round(row['t_hr'], 8))
        val = lookup.get(key)
        if val is not None:
            new_gt[i] = val
            matched += 1

    # Unique epochs matched (each epoch has 5 channels)
    unique_epochs = matched // 5 if matched > 0 else 0
    total_unique = resp_mask.sum() // 5
    print(f'\nMatched: {matched}/{resp_mask.sum()} resp rows '
          f'({unique_epochs}/{total_unique} unique epochs)')

    unmatched = resp_mask.sum() - matched
    if unmatched > 0:
        print(f'WARNING: {unmatched} resp rows had no consensus match')

    mask.loc[resp_mask, 'gt_hz'] = new_gt

    # Verify cardiac untouched
    card_gt_before = mask.loc[mask['band'] == 'card', 'gt_hz']
    print(f'\nCard gt_hz (untouched): mean={card_gt_before.mean():.6f}, '
          f'nan={card_gt_before.isna().sum()}')

    new_resp_gt = mask.loc[resp_mask, 'gt_hz']
    print(f'New resp gt_hz: mean={new_resp_gt.mean():.6f}, nan={new_resp_gt.isna().sum()}')

    diff = np.abs(old_resp_gt.values - new_resp_gt.values)
    finite = np.isfinite(diff)
    print(f'Resp gt_hz change: median |delta|={np.median(diff[finite]):.6f} Hz '
          f'({np.median(diff[finite])*60:.2f} br/min), '
          f'max={np.max(diff[finite]):.6f} Hz')
    pct_changed = (diff[finite] > 1e-8).sum() / finite.sum() * 100
    print(f'  {pct_changed:.1f}% of resp epochs changed')

    # Save
    mask.to_parquet(MASK_PATH, index=False)
    print(f'\nSaved -> {MASK_PATH} ({len(mask)} rows)')


if __name__ == '__main__':
    main()
