#!/usr/bin/env python
"""
Corrected rate-detection tables for the manuscript (2026-08-06 audit).

Rebuilds Table 3 and Table S1 from artifacts/mask_phase_c.parquet — the same
checkpoint that produced the original numbers — under a single, non-degenerate
operational pipeline (agreement-gated five-channel fusion of loose peak
counting) and with the held-out calibration comparators the original tables
lacked:

  self-k        k fitted on the night being scored (what the paper reported)
  cross-night k k fitted on the subject's other night        [held out]
  population k  one k for the whole cohort                   [held out]
  no sensor     predict the cohort median rate for every epoch [no sensor input]

Outputs: reports/rates/mask/table3_corrected.csv
         reports/rates/mask/table_s1_corrected.csv
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PHASE_C = ROOT / 'artifacts' / 'mask_phase_c.parquet'
RPT = ROOT / 'reports' / 'rates' / 'mask'

PIPELINE = 'fused_agree'          # non-degenerate in both bands
RESPONSIVE = 'diff_peaks_loose'   # the responsive single-channel detector
K_LO, K_HI = 0.3, 5.0             # ratio clip used by run_mask_rate_detection.py


def per_session(band: str, strategy: str) -> pd.DataFrame:
    c = pd.read_parquet(PHASE_C)
    g0 = c[(c.band == band) & (c.strategy == strategy)].dropna(subset=['gt_hz'])

    D = {}
    for sess, g in g0.groupby('session'):
        g = g.sort_values('epoch')
        raw, gt = g.rate_raw.values * 60, g.gt_hz.values * 60
        m = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
        D[sess] = (raw[m], gt[m])

    def fit_k(raw, gt):
        ratios = raw / gt
        return float(np.median(ratios[(ratios > K_LO) & (ratios < K_HI)]))

    ks = {s: fit_k(*v) for s, v in D.items()}
    k_pop = float(np.median(list(ks.values())))
    ref_pop = float(np.median(np.concatenate([g for _, g in D.values()])))

    rows = []
    for s, (raw, gt) in D.items():
        other = s[:2] + ('N2' if s.endswith('N1') else 'N1')
        k_x = ks.get(other, np.nan)
        pred = raw / ks[s]
        rows.append(dict(
            session=s, k=ks[s], n=len(gt),
            epoch_self_k=np.median(np.abs(pred - gt)),
            epoch_population_k=np.median(np.abs(raw / k_pop - gt)),
            epoch_no_sensor=np.median(np.abs(ref_pop - gt)),
            night_self_k=abs(np.mean(pred) - np.mean(gt)),
            night_cross_night_k=abs(np.mean(raw / k_x) - np.mean(gt)),
            night_population_k=abs(np.mean(raw / k_pop) - np.mean(gt)),
            night_no_sensor=abs(ref_pop - np.mean(gt)),
            r_within=np.corrcoef(pred, gt)[0, 1] if np.std(pred) > 0 else np.nan,
            ref_sd=np.std(gt),
        ))
    d = pd.DataFrame(rows).sort_values('session').reset_index(drop=True)
    d.attrs['k_pop'] = k_pop
    d.attrs['ref_pop'] = ref_pop
    return d


def med_iqr(v: np.ndarray, dp: int = 2) -> str:
    return (f'{np.median(v):.{dp}f} [{np.percentile(v, 25):.{dp}f}'
            f'–{np.percentile(v, 75):.{dp}f}]')


def main() -> None:
    out_rows, s1 = [], {}
    for band, unit, dp in [('resp', 'br/min', 2), ('card', 'BPM', 2)]:
        d = per_session(band, PIPELINE)
        resp_det = per_session(band, RESPONSIVE)
        s1[band] = d
        col = lambda name: med_iqr(d[name].values, dp)
        out_rows.append({
            'band': band, 'unit': unit,
            'k_median_iqr': med_iqr(d.k.values),
            'k_range': f'{d.k.min():.2f}–{d.k.max():.2f}',
            'k_population': f'{d.attrs["k_pop"]:.2f}',
            'night_self_k': col('night_self_k'),
            'night_cross_night_k': col('night_cross_night_k'),
            'night_population_k': col('night_population_k'),
            'night_no_sensor': col('night_no_sensor'),
            'epoch_self_k': col('epoch_self_k'),
            'epoch_population_k': col('epoch_population_k'),
            'epoch_no_sensor': col('epoch_no_sensor'),
            'epoch_responsive_detector': med_iqr(resp_det.epoch_self_k.values),
            'r_within_median': f'{d.r_within.median():+.2f}',
            'nights_r_positive': f'{int((d.r_within > 0).sum())} / 12',
            'ref_sd_within_night': col('ref_sd'),
        })
        print(f'\n=== {band} ({unit}), pipeline={PIPELINE} ===')
        print(d.round(3).to_string(index=False))
        for k, v in out_rows[-1].items():
            print(f'  {k:28s} {v}')

    pd.DataFrame(out_rows).to_csv(RPT / 'table3_corrected.csv', index=False)

    merged = s1['resp'].merge(s1['card'], on='session', suffixes=('_resp', '_card'))
    keep = ['session'] + [c + s for s in ('_resp', '_card') for c in
                          ('k', 'night_self_k', 'night_cross_night_k',
                           'night_population_k', 'night_no_sensor',
                           'epoch_self_k', 'r_within')]
    merged[keep].round(3).to_csv(RPT / 'table_s1_corrected.csv', index=False)
    print(f'\nwrote {RPT / "table3_corrected.csv"} and {RPT / "table_s1_corrected.csv"}')


if __name__ == '__main__':
    main()
