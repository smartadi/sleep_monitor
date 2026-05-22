"""
scripts/train_classifier.py — train and evaluate rate-prediction classifiers.

Consumes the per-window parquets produced by scripts/sweep.py
(artifacts/sweep/windows/*.parquet) and runs LOSO CV for several models.

Usage
-----
    python scripts/train_classifier.py              # both bands, all data
    python scripts/train_classifier.py --band resp  # one band
    python scripts/train_classifier.py --channel CLE-CRE --preproc ols
                                                     # only one (channel, preproc) slice
Outputs
-------
    artifacts/classifier/metrics.parquet      per-fold metrics
    artifacts/classifier/summary.csv          mean ± std per model/band
    artifacts/classifier/oof_<band>.parquet   per-row out-of-fold predictions
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sleep_monitor.classifier import (
    load_windows, build_dataset, loso_evaluate, summarise, default_models,
)


ROOT    = Path(__file__).resolve().parent.parent
WINDOWS = ROOT / 'artifacts' / 'sweep' / 'windows'
OUT     = ROOT / 'artifacts' / 'classifier'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', choices=('resp', 'cardiac', 'both'), default='both')
    ap.add_argument('--channel', type=str, default=None,
                     help='restrict to one base channel')
    ap.add_argument('--preproc', type=str, default=None,
                     help='restrict to one preprocessing (none/ols/nlms)')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    bands = ('resp', 'cardiac') if args.band == 'both' else (args.band,)

    metrics_all = []
    for band in bands:
        print(f'\n=== {band.upper()} ===')
        df = load_windows(WINDOWS, band=band,
                           channel=args.channel, preproc=args.preproc)
        print(f'loaded {len(df):,} windows across {df["session"].nunique()} sessions')

        X, y, groups, meta = build_dataset(df)
        print(f'  dataset  X={X.shape}  y={y.shape}  subjects={len(np.unique(groups))}')

        metrics, oof = loso_evaluate(X, y, groups, models=default_models(), band=band)
        metrics_all.append(metrics)

        # dump out-of-fold predictions (one row per window, with targets and all model preds)
        oof_df = meta.copy()
        oof_df['y_gt_hz'] = y
        for name, arr in oof.items():
            oof_df[f'pred_{name}_hz'] = arr
        oof_df.to_parquet(OUT / f'oof_{band}.parquet', index=False)
        print(f'  wrote {OUT / f"oof_{band}.parquet"}')

    metrics = pd.concat(metrics_all, ignore_index=True)
    metrics.to_parquet(OUT / 'metrics.parquet', index=False)
    summary = summarise(metrics)
    summary.to_csv(OUT / 'summary.csv', index=False)
    print('\n── Summary (Hz * 60 → br/min or BPM) ──')
    print(summary.to_string(index=False))
    print(f'\nWrote {OUT / "metrics.parquet"} and {OUT / "summary.csv"}')


if __name__ == '__main__':
    main()
