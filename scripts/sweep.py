"""
scripts/sweep.py — fast grid search over (channel, preproc, estimator) per band.

Fast path: the signal-processing stage is O(channels × preprocs × sessions),
and each "base frame" is re-used across all estimators. Estimator selection
is derived post-hoc in O(rows).

Writes
------
artifacts/sweep/windows/<base_tag>__<session>.parquet   per-window feature matrix
artifacts/sweep/leaderboard.parquet                     one row per (session, cfg, gate)

The per-window parquets are the training data for the rate classifier phase —
their schema is stable across all configs.

Usage
-----
    python scripts/sweep.py                     # all 12 sessions, both bands
    python scripts/sweep.py --sessions 0 1      # restricted sessions
    python scripts/sweep.py --band resp         # one band only
"""

from __future__ import annotations
import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sleep_monitor import SESSION_META, load_session
from sleep_monitor.evaluate import (
    BaseKey, PipelineConfig, compute_base_windows, derive_rate,
    evaluate_pipeline, default_base_keys,
)


ARTIFACTS   = Path(__file__).resolve().parent.parent / 'artifacts' / 'sweep'
WINDOWS_DIR = ARTIFACTS / 'windows'


def _estimators_for(band: str) -> List[str]:
    base = ['spectral', 'acf', 'hilbert', 'zerocross', 'peaks',
             'median', 'trimmed', 'weighted']
    return base + (['envelope'] if band == 'cardiac' else [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, nargs='*', default=None)
    ap.add_argument('--band', choices=('resp', 'cardiac', 'both'), default='both')
    ap.add_argument('--gates', type=float, nargs='*',
                     default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument('--preprocs', type=str, nargs='*',
                     default=['none', 'ols', 'nlms'])
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

    sess_idx = args.sessions if args.sessions else list(range(len(SESSION_META)))
    bands    = ('resp', 'cardiac') if args.band == 'both' else (args.band,)

    rows: List[dict] = []
    for si in sess_idx:
        session = load_session(si)
        for band in bands:
            keys = [k for k in default_base_keys(band) if k.preproc in args.preprocs]
            ests = _estimators_for(band)
            print(f'\n[{session.label}] band={band}  '
                   f'base_keys={len(keys)}  estimators={len(ests)}  '
                   f'total_configs={len(keys)*len(ests)}', flush=True)

            for bi, key in enumerate(keys, 1):
                t0 = time.time()
                base = compute_base_windows(session, key)
                out_path = WINDOWS_DIR / f'{key.tag()}__{session.label}.parquet'
                base.to_parquet(out_path, index=False)
                dt_base = time.time() - t0

                # Derive all estimators from the cached base frame
                for est in ests:
                    rate_series = derive_rate(base, est, band=band)
                    df_eval = base.copy()
                    df_eval['rate_hz'] = rate_series
                    cfg = PipelineConfig(band=band, channel=key.channel,
                                          preproc=key.preproc, estimator=est,
                                          win_s=key.win_s, step_s=key.step_s)
                    for gate in args.gates:
                        m = evaluate_pipeline(df_eval, quality_gate=gate)
                        rows.append({
                            'session':      session.label,
                            'quality_gate': gate,
                            'tag':          cfg.tag(),
                            **asdict(cfg),
                            **m,
                        })
                print(f'  [{bi:2d}/{len(keys)}] {key.tag():40s}  '
                       f'base={dt_base:5.1f}s  total={time.time()-t0:5.1f}s')

    lb = pd.DataFrame(rows)
    out = ARTIFACTS / 'leaderboard.parquet'
    lb.to_parquet(out, index=False)
    print(f'\nWrote {out}   ({len(lb)} rows, {lb["tag"].nunique()} unique configs)')


if __name__ == '__main__':
    main()
