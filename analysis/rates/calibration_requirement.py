"""
Does the rate pipeline actually need per-session k-calibration? And does channel
fusion help? Answers both from the cached paper-pipeline predictions
(artifacts/mask_phase_c.parquet) — no session reload needed.

Compares, per band, for each strategy:
  - per-session k (whole-night)      : rate_k_full_smooth   (the paper headline)
  - fixed POPULATION k (median of per-session k, no per-subject cal)
  - realistic first-10-min k          : rate_k_10min_smooth

Outputs -> analysis/rates/outputs/calibration_requirement.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)
PARQUET = 'artifacts/mask_phase_c.parquet'


def cmed(a, k=3):
    o = np.full(len(a), np.nan)
    for i in range(len(a)):
        w = a[max(0, i - k + 1):i + 1]
        w = w[np.isfinite(w)]
        if len(w):
            o[i] = np.median(w)
    return o


def per_session_median_mae(sub, col):
    maes = []
    for _, g in sub.groupby('session'):
        v = np.isfinite(g[col]) & np.isfinite(g.gt_hz)
        if v.sum() > 10:
            maes.append(np.median(np.abs(g[col].values[v] - g.gt_hz.values[v])) * 60)
    return float(np.median(maes)) if maes else np.nan


def fixed_pop_k_mae(sub, kpop):
    maes = []
    for _, g in sub.groupby('session'):
        pred = cmed(g.rate_raw.values / kpop)
        gt = g.gt_hz.values
        v = np.isfinite(pred) & np.isfinite(gt)
        if v.sum() > 10:
            maes.append(np.median(np.abs(pred[v] - gt[v])) * 60)
    return float(np.median(maes)) if maes else np.nan


def main():
    df = pd.read_parquet(PARQUET)
    rows = []
    for band, unit in [('resp', 'br/min'), ('card', 'BPM')]:
        b = df[(df.band == band) & df.gt_hz.notna()]
        for strat in sorted(b.strategy.unique()):
            sub = b[b.strategy == strat]
            ks = sub.groupby('session').k_full.first()
            kpop = float(ks.median())
            rows.append(dict(
                band=band, unit=unit, strategy=strat,
                mae_persession_k=per_session_median_mae(sub, 'rate_k_full_smooth'),
                mae_population_k=fixed_pop_k_mae(sub, kpop),
                mae_first10min_k=per_session_median_mae(sub, 'rate_k_10min_smooth'),
                k_pop=kpop, k_min=float(ks.min()), k_max=float(ks.max()),
            ))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'calibration_requirement.csv'), index=False)
    for band in ('resp', 'card'):
        print(f'=== {band} ===')
        print(res[res.band == band][['strategy', 'mae_persession_k', 'mae_population_k',
              'mae_first10min_k', 'k_pop', 'k_min', 'k_max']].to_string(index=False))
    print(f'\nSaved {OUT}/calibration_requirement.csv')


if __name__ == '__main__':
    main()
