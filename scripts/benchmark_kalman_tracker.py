#!/usr/bin/env python
"""
Phase 1 benchmark: Kalman rate tracker fusing spectral + adaptive_peaks.

Compares raw per-window estimates vs Kalman-smoothed output against PSG GT
for both respiratory and cardiac bands across all 12 sessions.

Outputs to reports/rates/hybrid_phase1/:
  - Per-session time-series plots (raw vs Kalman vs GT)
  - Aggregate summary table + heatmap
  - Bland-Altman plots
  - Per-stage breakdown
  - CSV with all per-window results

Usage:
    python scripts/benchmark_kalman_tracker.py [--session 0] [--all]
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    METHOD_COLORS, METHOD_LABELS, STAGE_LABELS, STAGE_COLORS,
)
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader import load_all_sessions, load_sleep_profile
from sleep_monitor.rates import (
    estimate_rate, rate_acf, kalman_rate_track,
)

import functools
print = functools.partial(print, flush=True)

OUT_DIR = ROOT / 'reports' / 'rates' / 'hybrid_phase1'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = ROOT / 'artifacts'
ART_DIR.mkdir(parents=True, exist_ok=True)

WIN_SEC = 30.0
STEP_SEC = 30.0
PSG_EPOCH_SEC = 30.0

BANDS = {
    'resp': (RESP_LO, RESP_HI, 'Thorax', 60.0, 'br/min'),
    'card': (CARD_LO, CARD_HI, 'Pleth',  60.0, 'BPM'),
}

INPUT_METHODS = ['spectral', 'adaptive_peaks']
ALL_TRACES = INPUT_METHODS + ['kalman']

TRACE_COLORS = {
    'spectral':       METHOD_COLORS['spectral'],
    'adaptive_peaks': METHOD_COLORS['adaptive_peaks'],
    'kalman':         '#E74C3C',
}
TRACE_LABELS = {
    'spectral':       'Spectral',
    'adaptive_peaks': 'Adaptive peaks',
    'kalman':         'Kalman fused',
}

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


def get_stage_at_time(profile, t_sec):
    if profile is None:
        return -1
    t_hr = np.array(profile['t_ep_hr'])
    codes = np.array(profile['codes'])
    t_query_hr = t_sec / 3600.0
    idx = np.searchsorted(t_hr, t_query_hr, side='right') - 1
    if idx < 0 or idx >= len(codes):
        return -1
    return int(codes[idx])


def run_session(session, band_name):
    f_lo, f_hi, gt_chan, unit_scale, unit_label = BANDS[band_name]
    fs = session.fs

    raw_cap = session.cap['CLE'].astype(np.float64) - session.cap['CRE'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)
    sig = remove_acc_artifact(raw_cap, acc, f_lo, f_hi, fs)
    gt_sig = bandpass(session.psg[gt_chan].astype(np.float64), f_lo, f_hi, fs)

    profile = getattr(session, 'sleep_profile', None)

    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    n_total = min(len(sig), len(gt_sig))

    rows = []
    spectral_arr = []
    adaptive_arr = []

    for start in range(0, n_total - win_n + 1, step_n):
        seg = sig[start:start + win_n]
        seg_gt = gt_sig[start:start + win_n]
        t_center = (start + win_n / 2.0) / fs

        rates = estimate_rate(seg, f_lo, f_hi, fs)
        gt_rate = rate_acf(seg_gt, f_lo, f_hi, fs, prominence=0.05)

        stage_code = get_stage_at_time(profile, t_center)
        stage_name = STAGE_LABELS.get(stage_code, '?')

        spectral_arr.append(rates['spectral'])
        adaptive_arr.append(rates['adaptive_peaks'])

        row = {
            't_s': t_center,
            'gt_hz': gt_rate,
            'spectral': rates['spectral'],
            'adaptive_peaks': rates['adaptive_peaks'],
            'stage_code': stage_code,
            'stage': stage_name,
        }
        rows.append(row)

    estimates = {
        'spectral': np.array(spectral_arr),
        'adaptive_peaks': np.array(adaptive_arr),
    }
    kalman_out = kalman_rate_track(estimates, f_lo, f_hi, step_sec=STEP_SEC)

    for i, row in enumerate(rows):
        row['kalman'] = kalman_out[i]

    return pd.DataFrame(rows)


def compute_metrics(df, method):
    valid = df[['gt_hz', method]].dropna()
    valid = valid[np.isfinite(valid['gt_hz']) & (valid['gt_hz'] > 0)]
    if len(valid) < 10:
        return {'n': len(valid), 'mae': np.nan, 'bias': np.nan, 'r': np.nan, 'rmse': np.nan}
    gt = valid['gt_hz'].values
    est = valid[method].values
    err = est - gt
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    rmse = np.sqrt(np.mean(err ** 2))
    r = np.corrcoef(gt, est)[0, 1] if np.std(gt) > 0 and np.std(est) > 0 else np.nan
    return {'n': len(valid), 'mae': mae, 'bias': bias, 'r': r, 'rmse': rmse}


def plot_timeseries(df, band_name, session_label, out_path):
    f_lo, f_hi, _, unit_scale, unit_label = BANDS[band_name]
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 1]})

    t_min = df['t_s'] / 60.0

    # Panel 1: rate traces
    ax = axes[0]
    valid_gt = np.isfinite(df['gt_hz']) & (df['gt_hz'] > 0)
    ax.plot(t_min[valid_gt], df.loc[valid_gt, 'gt_hz'] * unit_scale,
            'k-', lw=1.2, alpha=0.5, label='GT (PSG)')
    for m in ALL_TRACES:
        if m in df.columns:
            ax.plot(t_min, df[m] * unit_scale, '-',
                    color=TRACE_COLORS[m], lw=0.9 if m != 'kalman' else 1.8,
                    alpha=0.4 if m != 'kalman' else 0.9,
                    label=TRACE_LABELS[m])
    ax.set_ylabel(f'Rate ({unit_label})')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{session_label} — {band_name} — Kalman tracker vs raw inputs')
    ax.set_ylim(f_lo * unit_scale * 0.8, f_hi * unit_scale * 1.05)

    # Panel 2: error traces
    ax = axes[1]
    for m in ALL_TRACES:
        if m in df.columns:
            err = (df[m] - df['gt_hz']) * unit_scale
            ax.plot(t_min, err, '-', color=TRACE_COLORS[m],
                    lw=0.7 if m != 'kalman' else 1.5,
                    alpha=0.4 if m != 'kalman' else 0.8,
                    label=TRACE_LABELS[m])
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel(f'Error ({unit_label})')
    ax.legend(loc='upper right', fontsize=7, ncol=3)

    # Panel 3: sleep stage
    ax = axes[2]
    stage_color_map = {
        'Wake': STAGE_COLORS[4], 'N1': STAGE_COLORS[3],
        'N2': STAGE_COLORS[2], 'N3': STAGE_COLORS[1], 'REM': STAGE_COLORS[0],
    }
    for i in range(len(t_min)):
        s = df.iloc[i]['stage']
        c = stage_color_map.get(s, '#AAAAAA')
        ax.axvspan(t_min.iloc[i] - 0.25, t_min.iloc[i] + 0.25,
                   color=c, alpha=0.7, linewidth=0)
    ax.set_yticks([])
    ax.set_xlabel('Time (min)')
    patches = [Patch(color=stage_color_map[s], label=s) for s in ['Wake', 'N1', 'N2', 'N3', 'REM']
               if s in stage_color_map]
    ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=5)
    ax.set_ylabel('Stage')

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bland_altman(df, method, band_name, session_label, out_path):
    _, _, _, unit_scale, unit_label = BANDS[band_name]
    valid = df[['gt_hz', method]].dropna()
    valid = valid[np.isfinite(valid['gt_hz']) & (valid['gt_hz'] > 0)]
    if len(valid) < 10:
        return

    gt = valid['gt_hz'].values * unit_scale
    est = valid[method].values * unit_scale
    mean_vals = (gt + est) / 2.0
    diff_vals = est - gt
    bias = np.mean(diff_vals)
    loa = 1.96 * np.std(diff_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_vals, diff_vals, s=4, alpha=0.3, color=TRACE_COLORS.get(method, '#333'))
    ax.axhline(bias, color='red', ls='-', lw=1, label=f'Bias: {bias:+.2f}')
    ax.axhline(bias + loa, color='grey', ls='--', lw=0.8, label=f'±1.96σ: ±{loa:.2f}')
    ax.axhline(bias - loa, color='grey', ls='--', lw=0.8)
    ax.set_xlabel(f'Mean of GT & estimate ({unit_label})')
    ax.set_ylabel(f'Estimate − GT ({unit_label})')
    ax.set_title(f'Bland-Altman: {TRACE_LABELS.get(method, method)} — {session_label} {band_name}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_aggregate_comparison(all_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, band_name in enumerate(['resp', 'card']):
        _, _, _, unit_scale, unit_label = BANDS[band_name]
        ax = axes[idx]
        sub = all_results[all_results['band'] == band_name]
        methods = ALL_TRACES
        x = np.arange(len(methods))
        means = [sub[sub['method'] == m]['mae_unit'].mean() for m in methods]
        stds = [sub[sub['method'] == m]['mae_unit'].std() for m in methods]
        colors = [TRACE_COLORS[m] for m in methods]
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([TRACE_LABELS[m] for m in methods], rotation=15, ha='right')
        ax.set_ylabel(f'MAE ({unit_label})')
        ax.set_title(f'{band_name.upper()} — MAE across 12 sessions')
        for bar, mean in zip(bars, means):
            if np.isfinite(mean):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_stage(all_windows, band_name, out_path):
    _, _, _, unit_scale, unit_label = BANDS[band_name]
    sub = all_windows[all_windows['band'] == band_name].copy()
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    methods = ALL_TRACES

    fig, ax = plt.subplots(figsize=(12, 6))
    n_stages = len(stages)
    n_methods = len(methods)
    width = 0.8 / n_methods
    offsets = np.arange(n_methods) - (n_methods - 1) / 2

    for j, m in enumerate(methods):
        maes = []
        for s in stages:
            ss = sub[(sub['stage'] == s) & sub[f'{m}_hz'].notna() &
                     sub['gt_hz'].notna() & (sub['gt_hz'] > 0)]
            if len(ss) >= 5:
                maes.append(np.mean(np.abs(ss[f'{m}_hz'] - ss['gt_hz'])) * unit_scale)
            else:
                maes.append(np.nan)
        x = np.arange(n_stages) + offsets[j] * width
        ax.bar(x, maes, width=width, color=TRACE_COLORS[m], alpha=0.8,
               label=TRACE_LABELS[m])

    ax.set_xticks(np.arange(n_stages))
    ax.set_xticklabels(stages)
    ax.set_ylabel(f'MAE ({unit_label})')
    ax.set_title(f'{band_name.upper()} — MAE by sleep stage (all sessions pooled)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_improvement_heatmap(all_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, band_name in enumerate(['resp', 'card']):
        _, _, _, unit_scale, unit_label = BANDS[band_name]
        ax = axes[idx]
        sub = all_results[all_results['band'] == band_name]
        sessions = sorted(sub['session'].unique())
        input_methods = INPUT_METHODS

        improvement = np.full((len(sessions), len(input_methods)), np.nan)
        for i, sess in enumerate(sessions):
            kalman_mae = sub[(sub['session'] == sess) & (sub['method'] == 'kalman')]['mae_unit'].values
            if len(kalman_mae) == 0:
                continue
            k_mae = kalman_mae[0]
            for j, m in enumerate(input_methods):
                m_mae = sub[(sub['session'] == sess) & (sub['method'] == m)]['mae_unit'].values
                if len(m_mae) > 0 and m_mae[0] > 0:
                    improvement[i, j] = (m_mae[0] - k_mae) / m_mae[0] * 100

        im = ax.imshow(improvement, cmap='RdYlGn', vmin=-20, vmax=50, aspect='auto')
        ax.set_xticks(range(len(input_methods)))
        ax.set_xticklabels([TRACE_LABELS[m] for m in input_methods])
        ax.set_yticks(range(len(sessions)))
        ax.set_yticklabels(sessions, fontsize=8)
        ax.set_title(f'{band_name.upper()} — Kalman MAE improvement (%)')
        for i in range(len(sessions)):
            for j in range(len(input_methods)):
                v = improvement[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:+.0f}%', ha='center', va='center', fontsize=7,
                            color='black' if abs(v) < 30 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, default=None)
    parser.add_argument('--all', action='store_true', default=True)
    args = parser.parse_args()

    sessions = load_all_sessions()
    for s in sessions:
        try:
            s.sleep_profile = load_sleep_profile(s)
        except Exception:
            s.sleep_profile = None

    if args.session is not None:
        indices = [args.session]
    else:
        indices = range(len(sessions))

    all_results = []
    all_windows = []

    for idx in indices:
        s = sessions[idx]
        label = f"S{s.meta['subject']}N{s.meta['night']}"
        print(f"\n{'='*60}")
        print(f"Session {idx}: {label}")
        print(f"{'='*60}")

        for band_name in ['resp', 'card']:
            f_lo, f_hi, gt_chan, unit_scale, unit_label = BANDS[band_name]

            t0 = time.time()
            df = run_session(s, band_name)
            elapsed = time.time() - t0
            print(f"\n  {band_name.upper()}: {len(df)} windows, {elapsed:.1f}s")

            if len(df) < 10:
                print(f"  Too few windows, skipping.")
                continue

            # Print metrics table
            print(f"\n  {'Method':<20s} {'MAE':>8s} {'RMSE':>8s} {'Bias':>8s} {'r':>8s} {'N':>6s}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
            for m in ALL_TRACES:
                met = compute_metrics(df, m)
                mae_u = met['mae'] * unit_scale if np.isfinite(met['mae']) else np.nan
                rmse_u = met['rmse'] * unit_scale if np.isfinite(met['rmse']) else np.nan
                bias_u = met['bias'] * unit_scale if np.isfinite(met['bias']) else np.nan
                marker = ' <<<' if m == 'kalman' else ''
                print(f"  {TRACE_LABELS.get(m, m):<20s} {mae_u:>8.2f} {rmse_u:>8.2f} {bias_u:>+8.2f} {met['r']:>8.3f} {met['n']:>6d}{marker}")

                all_results.append({
                    'session': label, 'band': band_name, 'method': m,
                    'mae_hz': met['mae'], 'rmse_hz': met['rmse'],
                    'bias_hz': met['bias'],
                    'mae_unit': mae_u, 'rmse_unit': rmse_u, 'bias_unit': bias_u,
                    'r': met['r'], 'n': met['n'],
                })

            # Per-window data for stage analysis
            for _, row in df.iterrows():
                wrow = {
                    'session': label, 'band': band_name,
                    'stage_code': row['stage_code'], 'stage': row['stage'],
                    'gt_hz': row['gt_hz'],
                }
                for m in ALL_TRACES:
                    wrow[f'{m}_hz'] = row.get(m, np.nan)
                all_windows.append(wrow)

            # Time-series plot
            ts_path = OUT_DIR / f'timeseries_{label}_{band_name}.png'
            plot_timeseries(df, band_name, label, ts_path)
            print(f"  Plot: {ts_path.name}")

            # Bland-Altman for Kalman
            ba_path = OUT_DIR / f'bland_altman_kalman_{label}_{band_name}.png'
            plot_bland_altman(df, 'kalman', band_name, label, ba_path)

    # Aggregate outputs
    if len(all_results) > 1:
        results_df = pd.DataFrame(all_results)
        windows_df = pd.DataFrame(all_windows)

        print(f"\n\n{'='*60}")
        print("AGGREGATE (mean across sessions)")
        print(f"{'='*60}")
        for band_name in ['resp', 'card']:
            _, _, _, unit_scale, unit_label = BANDS[band_name]
            sub = results_df[results_df['band'] == band_name]
            print(f"\n  {band_name.upper()} ({unit_label}):")
            print(f"  {'Method':<20s} {'MAE':>8s} {'RMSE':>8s} {'Bias':>8s} {'r':>8s}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for m in ALL_TRACES:
                ms = sub[sub['method'] == m]
                marker = ' <<<' if m == 'kalman' else ''
                print(f"  {TRACE_LABELS.get(m, m):<20s} "
                      f"{ms['mae_unit'].mean():>8.2f} "
                      f"{ms['rmse_unit'].mean():>8.2f} "
                      f"{ms['bias_unit'].mean():>+8.2f} "
                      f"{ms['r'].mean():>8.3f}{marker}")

        # Aggregate plots
        agg_path = OUT_DIR / 'aggregate_mae_comparison.png'
        plot_aggregate_comparison(results_df, agg_path)
        print(f"\n  Aggregate bar chart: {agg_path.name}")

        hm_path = OUT_DIR / 'improvement_heatmap.png'
        plot_improvement_heatmap(results_df, hm_path)
        print(f"  Improvement heatmap: {hm_path.name}")

        for band_name in ['resp', 'card']:
            stage_path = OUT_DIR / f'per_stage_mae_{band_name}.png'
            plot_per_stage(windows_df, band_name, stage_path)
            print(f"  Per-stage {band_name}: {stage_path.name}")

        # Save CSVs
        csv_path = OUT_DIR / 'kalman_tracker_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Results CSV: {csv_path.name}")

        win_csv = OUT_DIR / 'kalman_tracker_windows.csv'
        windows_df.to_csv(win_csv, index=False)
        print(f"  Windows CSV: {win_csv.name}")

    print(f"\n  All outputs in: {OUT_DIR}")


if __name__ == '__main__':
    main()
