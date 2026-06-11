#!/usr/bin/env python
"""
Benchmark: adaptive_peaks vs existing rate estimators on real overnight data.

Runs all methods on one session (default S1N1), compares MAE/bias/Pearson r
against PSG ground truth for both respiratory and cardiac bands.
Outputs a summary table + per-method time-series overlay plot.

Usage:
    python scripts/benchmark_adaptive_peaks.py [--session 0] [--all]
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    METHOD_COLORS, METHOD_LABELS,
)
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader import load_all_sessions
from sleep_monitor.rates import estimate_rate, rate_adaptive_peaks
from sleep_monitor.ground_truth import gt_sliding_rates

import functools
print = functools.partial(print, flush=True)

PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'rate_analysis'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

WIN_SEC = 30.0
STEP_SEC = 30.0
METHODS = ['spectral', 'acf', 'hilbert', 'zerocross', 'peaks', 'adaptive_peaks']

BANDS = {
    'resp': (RESP_LO, RESP_HI, 'Thorax', 60.0),
    'card': (CARD_LO, CARD_HI, 'Pleth', 60.0),
}


def run_session(session, band_name: str):
    f_lo, f_hi, gt_chan, unit_scale = BANDS[band_name]
    fs = session.fs

    raw_cap = session.cap['CLE'].astype(np.float64) - session.cap['CRE'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)
    sig = remove_acc_artifact(raw_cap, acc, f_lo, f_hi, fs)

    gt_sig = bandpass(session.psg[gt_chan].astype(np.float64), f_lo, f_hi, fs)

    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    n_total = min(len(sig), len(gt_sig))

    rows = []
    for start in range(0, n_total - win_n + 1, step_n):
        seg = sig[start:start + win_n]
        seg_gt = gt_sig[start:start + win_n]
        t_center = (start + win_n / 2.0) / fs

        rates = estimate_rate(seg, f_lo, f_hi, fs)

        from sleep_monitor.rates import rate_acf
        gt_rate = rate_acf(seg_gt, f_lo, f_hi, fs, prominence=0.05)

        if not np.isfinite(gt_rate) or gt_rate <= 0:
            continue

        row = {'t_s': t_center, 'gt_hz': gt_rate}
        for m in METHODS:
            row[m] = rates.get(m, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(df, method):
    valid = df[['gt_hz', method]].dropna()
    if len(valid) < 10:
        return {'n': len(valid), 'mae': np.nan, 'bias': np.nan, 'r': np.nan}
    gt = valid['gt_hz'].values
    est = valid[method].values
    err = est - gt
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    r = np.corrcoef(gt, est)[0, 1] if np.std(gt) > 0 and np.std(est) > 0 else np.nan
    return {'n': len(valid), 'mae': mae, 'bias': bias, 'r': r}


def plot_timeseries(df, band_name, unit_scale, session_label, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax = axes[0]
    t_min = df['t_s'] / 60.0
    ax.plot(t_min, df['gt_hz'] * unit_scale, 'k-', lw=1.5, alpha=0.7, label='GT')
    for m in ['peaks', 'adaptive_peaks']:
        if m in df.columns:
            color = METHOD_COLORS.get(m, '#999')
            label = METHOD_LABELS.get(m, m)
            ax.plot(t_min, df[m] * unit_scale, '-', color=color, lw=0.8, alpha=0.7, label=label)
    ax.set_ylabel(f'{"Resp (br/min)" if band_name == "resp" else "Cardiac (BPM)"}')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{session_label} — {band_name} — peaks vs adaptive_peaks')

    ax = axes[1]
    for m in METHODS:
        if m in df.columns:
            err = (df[m] - df['gt_hz']) * unit_scale
            color = METHOD_COLORS.get(m, '#999')
            label = METHOD_LABELS.get(m, m)
            ax.plot(t_min, err, '-', color=color, lw=0.6, alpha=0.6, label=label)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('Error (est - GT)')
    ax.set_xlabel('Time (min)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, default=0, help='Session index (0-11)')
    parser.add_argument('--all', action='store_true', help='Run all 12 sessions')
    args = parser.parse_args()

    sessions = load_all_sessions()

    if args.all:
        indices = range(len(sessions))
    else:
        indices = [args.session]

    all_results = []

    for idx in indices:
        s = sessions[idx]
        label = f"S{s.meta['subject']}N{s.meta['night']}"
        print(f"\n{'='*60}")
        print(f"Session {idx}: {label}")
        print(f"{'='*60}")

        for band_name in ['resp', 'card']:
            f_lo, f_hi, gt_chan, unit_scale = BANDS[band_name]
            unit_label = 'br/min' if band_name == 'resp' else 'BPM'

            print(f"\n  {band_name.upper()} band ({f_lo}-{f_hi} Hz):")
            t0 = time.time()
            df = run_session(s, band_name)
            elapsed = time.time() - t0
            print(f"  {len(df)} windows, {elapsed:.1f}s")

            if len(df) < 10:
                print(f"  Too few windows, skipping.")
                continue

            print(f"\n  {'Method':<20s} {'MAE':>8s} {'Bias':>8s} {'r':>8s} {'N':>6s}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
            for m in METHODS:
                met = compute_metrics(df, m)
                mae_unit = met['mae'] * unit_scale if np.isfinite(met['mae']) else np.nan
                bias_unit = met['bias'] * unit_scale if np.isfinite(met['bias']) else np.nan
                print(f"  {METHOD_LABELS.get(m, m):<20s} {mae_unit:>8.2f} {bias_unit:>+8.2f} {met['r']:>8.3f} {met['n']:>6d}")

                all_results.append({
                    'session': label, 'band': band_name, 'method': m,
                    'mae_hz': met['mae'], 'bias_hz': met['bias'],
                    'mae_unit': mae_unit, 'bias_unit': bias_unit,
                    'r': met['r'], 'n': met['n'],
                })

            plot_path = PLOT_DIR / f'adaptive_peaks_benchmark_{label}_{band_name}.png'
            plot_timeseries(df, band_name, unit_scale, label, plot_path)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n\n{'='*60}")
        print("AGGREGATE RESULTS (mean across sessions)")
        print(f"{'='*60}")
        for band_name in ['resp', 'card']:
            unit_label = 'br/min' if band_name == 'resp' else 'BPM'
            sub = results_df[results_df['band'] == band_name]
            print(f"\n  {band_name.upper()} ({unit_label}):")
            print(f"  {'Method':<20s} {'MAE':>8s} {'Bias':>8s} {'r':>8s}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
            for m in METHODS:
                ms = sub[sub['method'] == m]
                print(f"  {METHOD_LABELS.get(m, m):<20s} "
                      f"{ms['mae_unit'].mean():>8.2f} "
                      f"{ms['bias_unit'].mean():>+8.2f} "
                      f"{ms['r'].mean():>8.3f}")

        csv_path = ROOT / 'artifacts' / 'adaptive_peaks_benchmark.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Results saved: {csv_path}")


if __name__ == '__main__':
    main()
