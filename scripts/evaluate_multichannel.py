#!/usr/bin/env python
"""
Phase 2: Multi-channel Kalman fusion.

Runs spectral + adaptive_peaks → Kalman on each of 5 channels independently,
then fuses across channels using per-window quality-weighted averaging.

Compares:
  - Single-channel Kalman (CLE-CRE diff, the current default)
  - Multi-channel quality-weighted fusion
  - Oracle (best channel per window with hindsight)
  - Baseline /k methods for reference

Outputs to reports/rates/hybrid_phase2/:
  - Per-session time-series (multi-channel overlay)
  - Channel selection frequency heatmap
  - Aggregate comparison bar chart
  - Per-stage breakdown
  - CSV results
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    STAGE_LABELS, STAGE_COLORS,
)
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader import load_all_sessions, load_sleep_profile
from sleep_monitor.rates import (
    rate_spectral, rate_adaptive_peaks, rate_acf,
    kalman_rate_track,
)
from sleep_monitor.quality import window_features, combined_quality

import functools
print = functools.partial(print, flush=True)

OUT_DIR = ROOT / 'reports' / 'rates' / 'hybrid_phase2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN_SEC = 30.0
STEP_SEC = 30.0

BANDS = {
    'resp': (RESP_LO, RESP_HI, 'Thorax', 60.0, 'br/min'),
    'card': (CARD_LO, CARD_HI, 'Pleth',  60.0, 'BPM'),
}

CHANNELS = ['CLE', 'CRE', 'CH', 'avg', 'diff']
CHAN_COLORS = {
    'CLE': '#27AE60', 'CRE': '#8E44AD', 'CH': '#2980B9',
    'avg': '#E67E22', 'diff': '#E74C3C',
}

STAGE_NAME_MAP = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

PIPE_LABELS = {
    'single_diff':   'Single (CLE-CRE)',
    'multi_quality':  'Multi-ch quality-weighted',
    'multi_oracle':   'Oracle (best ch/window)',
}
PIPE_COLORS = {
    'single_diff':   '#E74C3C',
    'multi_quality':  '#2ECC71',
    'multi_oracle':   '#3498DB',
}

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


def prepare_channels(session):
    cle = session.cap['CLE'].astype(np.float64)
    cre = session.cap['CRE'].astype(np.float64)
    ch  = session.cap['CH'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)
    avg = (cle + cre) / 2.0
    diff = cle - cre
    return {'CLE': cle, 'CRE': cre, 'CH': ch, 'avg': avg, 'diff': diff}, acc


def get_stage_at_time(profile, t_sec):
    if profile is None:
        return -1
    t_hr = np.array(profile['t_ep_hr'])
    codes = np.array(profile['codes'])
    idx = np.searchsorted(t_hr, t_sec / 3600.0, side='right') - 1
    if idx < 0 or idx >= len(codes):
        return -1
    return int(codes[idx])


def run_session(session, band_name):
    f_lo, f_hi, gt_chan, unit_scale, unit_label = BANDS[band_name]
    fs = session.fs
    profile = getattr(session, 'sleep_profile', None)

    raw_channels, acc = prepare_channels(session)
    gt_sig = bandpass(session.psg[gt_chan].astype(np.float64), f_lo, f_hi, fs)

    # Preprocess each channel
    bp_channels = {}
    for ch_name, raw in raw_channels.items():
        n = min(len(raw), len(acc))
        bp_channels[ch_name] = remove_acc_artifact(raw[:n], acc[:n], f_lo, f_hi, fs)

    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    n_total = min(min(len(s) for s in bp_channels.values()), len(gt_sig))

    # Per-channel per-window: collect spectral + adaptive rates + quality
    per_chan_spectral = {ch: [] for ch in CHANNELS}
    per_chan_adaptive = {ch: [] for ch in CHANNELS}
    per_chan_quality  = {ch: [] for ch in CHANNELS}
    gt_rates = []
    t_centers = []
    stages = []

    acc_bp = acc[:n_total]

    for start in range(0, n_total - win_n + 1, step_n):
        t_center = (start + win_n / 2.0) / fs
        t_centers.append(t_center)

        seg_gt = gt_sig[start:start + win_n]
        gt_rate = rate_acf(seg_gt, f_lo, f_hi, fs, prominence=0.05)
        gt_rates.append(gt_rate)

        stage_code = get_stage_at_time(profile, t_center)
        stages.append(STAGE_NAME_MAP.get(stage_code, '?'))

        acc_win = acc_bp[start:start + win_n] if start + win_n <= len(acc_bp) else None

        for ch_name in CHANNELS:
            seg = bp_channels[ch_name][start:start + win_n]
            r_spec = rate_spectral(seg, f_lo, f_hi, fs)
            r_adap = rate_adaptive_peaks(seg, f_lo, f_hi, fs)
            per_chan_spectral[ch_name].append(r_spec)
            per_chan_adaptive[ch_name].append(r_adap)

            feats = window_features(seg, acc_win, f_lo, f_hi, fs,
                                    rates_hz={'spectral': r_spec, 'adaptive': r_adap})
            per_chan_quality[ch_name].append(combined_quality(feats))

    N = len(t_centers)
    gt_arr = np.array(gt_rates)

    # Run Kalman per channel
    per_chan_kalman = {}
    for ch_name in CHANNELS:
        estimates = {
            'spectral': np.array(per_chan_spectral[ch_name]),
            'adaptive_peaks': np.array(per_chan_adaptive[ch_name]),
        }
        per_chan_kalman[ch_name] = kalman_rate_track(estimates, f_lo, f_hi, step_sec=STEP_SEC)

    # Quality arrays
    quality_arr = {ch: np.array(per_chan_quality[ch]) for ch in CHANNELS}

    # Fusion strategies
    single_diff = per_chan_kalman['diff']

    # Quality-weighted fusion
    multi_quality = np.full(N, np.nan)
    best_chan = []
    for i in range(N):
        rates_i = []
        weights_i = []
        for ch_name in CHANNELS:
            r = per_chan_kalman[ch_name][i]
            q = quality_arr[ch_name][i]
            if np.isfinite(r) and f_lo <= r <= f_hi and np.isfinite(q) and q > 0:
                rates_i.append(r)
                weights_i.append(q)
        if rates_i:
            w = np.array(weights_i)
            multi_quality[i] = np.average(rates_i, weights=w)
            best_chan.append(CHANNELS[np.argmax(weights_i)])
        else:
            best_chan.append('none')

    # Oracle: best channel per window (lowest absolute error)
    multi_oracle = np.full(N, np.nan)
    oracle_chan = []
    for i in range(N):
        if not np.isfinite(gt_arr[i]) or gt_arr[i] <= 0:
            oracle_chan.append('none')
            continue
        best_err = np.inf
        best_r = np.nan
        best_c = 'none'
        for ch_name in CHANNELS:
            r = per_chan_kalman[ch_name][i]
            if np.isfinite(r):
                err = abs(r - gt_arr[i])
                if err < best_err:
                    best_err = err
                    best_r = r
                    best_c = ch_name
        multi_oracle[i] = best_r
        oracle_chan.append(best_c)

    # Build DataFrame
    rows = []
    for i in range(N):
        row = {
            't_s': t_centers[i], 'gt_hz': gt_arr[i], 'stage': stages[i],
            'single_diff': single_diff[i],
            'multi_quality': multi_quality[i],
            'multi_oracle': multi_oracle[i],
            'best_quality_chan': best_chan[i],
            'oracle_chan': oracle_chan[i],
        }
        for ch_name in CHANNELS:
            row[f'kalman_{ch_name}'] = per_chan_kalman[ch_name][i]
            row[f'quality_{ch_name}'] = quality_arr[ch_name][i]
        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(gt, est, unit_scale):
    valid = np.isfinite(gt) & np.isfinite(est) & (gt > 0)
    if valid.sum() < 10:
        return {'n': 0, 'mae': np.nan, 'rmse': np.nan, 'bias': np.nan, 'r': np.nan}
    g, e = gt[valid], est[valid]
    err = e - g
    return {
        'n': int(valid.sum()),
        'mae': np.mean(np.abs(err)) * unit_scale,
        'rmse': np.sqrt(np.mean(err ** 2)) * unit_scale,
        'bias': np.mean(err) * unit_scale,
        'r': np.corrcoef(g, e)[0, 1] if np.std(g) > 0 and np.std(e) > 0 else np.nan,
    }


def plot_timeseries(df, band_name, session_label, out_path):
    _, _, _, unit_scale, unit_label = BANDS[band_name]
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 1]})
    t_min = df['t_s'] / 60.0

    # Panel 1: per-channel Kalman + fused
    ax = axes[0]
    valid_gt = np.isfinite(df['gt_hz']) & (df['gt_hz'] > 0)
    ax.plot(t_min[valid_gt], df.loc[valid_gt, 'gt_hz'] * unit_scale,
            'k-', lw=1.2, alpha=0.4, label='GT')
    for ch in CHANNELS:
        ax.plot(t_min, df[f'kalman_{ch}'] * unit_scale, '-',
                color=CHAN_COLORS[ch], lw=0.4, alpha=0.25)
    ax.plot(t_min, df['multi_quality'] * unit_scale, '-',
            color=PIPE_COLORS['multi_quality'], lw=1.8, alpha=0.9, label='Multi-ch fused')
    ax.plot(t_min, df['single_diff'] * unit_scale, '-',
            color=PIPE_COLORS['single_diff'], lw=1.0, alpha=0.5, label='Single (diff)')
    ax.set_ylabel(f'Rate ({unit_label})')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{session_label} — {band_name} — multi-channel Kalman fusion')

    # Panel 2: per-channel quality
    ax = axes[1]
    for ch in CHANNELS:
        ax.plot(t_min, df[f'quality_{ch}'], '-', color=CHAN_COLORS[ch],
                lw=0.6, alpha=0.6, label=ch)
    ax.set_ylabel('Quality')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=7, ncol=5)

    # Panel 3: stage
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
    ax.set_ylabel('Stage')

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_channel_selection(all_windows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, band_name in enumerate(['resp', 'card']):
        ax = axes[idx]
        sub = all_windows[all_windows['band'] == band_name]

        # Quality-weighted best channel frequency
        counts_q = sub['best_quality_chan'].value_counts()
        # Oracle channel frequency
        counts_o = sub['oracle_chan'].value_counts()

        chans = CHANNELS
        x = np.arange(len(chans))
        w = 0.35
        q_vals = [counts_q.get(ch, 0) / len(sub) * 100 for ch in chans]
        o_vals = [counts_o.get(ch, 0) / len(sub) * 100 for ch in chans]

        ax.bar(x - w/2, q_vals, w, color=[CHAN_COLORS[c] for c in chans], alpha=0.7, label='Quality-weighted best')
        ax.bar(x + w/2, o_vals, w, color=[CHAN_COLORS[c] for c in chans], alpha=0.4, edgecolor='black', label='Oracle best')
        ax.set_xticks(x)
        ax.set_xticklabels(chans)
        ax.set_ylabel('% of windows')
        ax.set_title(f'{band_name.upper()} — channel selection frequency')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_aggregate(results_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pipes = ['single_diff', 'multi_quality', 'multi_oracle']

    for idx, band_name in enumerate(['resp', 'card']):
        _, _, _, unit_scale, unit_label = BANDS[band_name]
        ax = axes[idx]
        sub = results_df[results_df['band'] == band_name]

        means = [sub[sub['pipeline'] == p]['mae'].mean() for p in pipes]
        stds = [sub[sub['pipeline'] == p]['mae'].std() for p in pipes]
        colors = [PIPE_COLORS[p] for p in pipes]
        x = np.arange(len(pipes))
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([PIPE_LABELS[p] for p in pipes], rotation=10, ha='right', fontsize=9)
        ax.set_ylabel(f'MAE ({unit_label})')
        ax.set_title(f'{band_name.upper()} — single vs multi-channel (no k)')
        for bar, m in zip(bars, means):
            if np.isfinite(m):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{m:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_stage(all_windows, band_name, out_path):
    _, _, _, unit_scale, unit_label = BANDS[band_name]
    sub = all_windows[all_windows['band'] == band_name]
    pipes = ['single_diff', 'multi_quality']

    fig, ax = plt.subplots(figsize=(10, 5))
    n_stages = len(STAGE_ORDER)
    n_pipes = len(pipes)
    width = 0.8 / n_pipes
    offsets = np.arange(n_pipes) - (n_pipes - 1) / 2

    for j, pipe in enumerate(pipes):
        maes = []
        for s in STAGE_ORDER:
            ss = sub[(sub['stage'] == s) & sub[pipe].notna() &
                     sub['gt_hz'].notna() & (sub['gt_hz'] > 0)]
            if len(ss) >= 5:
                maes.append(np.mean(np.abs(ss[pipe] - ss['gt_hz'])) * unit_scale)
            else:
                maes.append(np.nan)
        x = np.arange(n_stages) + offsets[j] * width
        ax.bar(x, maes, width=width, color=PIPE_COLORS[pipe], alpha=0.8,
               label=PIPE_LABELS[pipe])

    ax.set_xticks(np.arange(n_stages))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel(f'MAE ({unit_label})')
    ax.set_title(f'{band_name.upper()} — per-stage: single vs multi-channel (no k)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    sessions = load_all_sessions()
    for s in sessions:
        try:
            s.sleep_profile = load_sleep_profile(s)
        except Exception:
            s.sleep_profile = None

    all_results = []
    all_windows = []

    for idx, s in enumerate(sessions):
        label = s.meta['label']
        print(f"\n{'='*60}")
        print(f"Session {idx}: {label}")
        print(f"{'='*60}")

        for band_name in ['resp', 'card']:
            _, _, _, unit_scale, unit_label = BANDS[band_name]
            t0 = time.time()
            df = run_session(s, band_name)
            elapsed = time.time() - t0
            print(f"\n  {band_name.upper()}: {len(df)} windows, {elapsed:.1f}s")

            if len(df) < 10:
                continue

            gt = df['gt_hz'].values
            print(f"  {'Pipeline':<30s} {'MAE':>8s} {'RMSE':>8s} {'Bias':>8s} {'r':>8s}")
            print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

            for pipe in ['single_diff', 'multi_quality', 'multi_oracle']:
                met = compute_metrics(gt, df[pipe].values, unit_scale)
                marker = ' <<<' if pipe == 'multi_quality' else ''
                print(f"  {PIPE_LABELS[pipe]:<30s} {met['mae']:>8.2f} {met['rmse']:>8.2f} "
                      f"{met['bias']:>+8.2f} {met['r']:>8.3f}{marker}")
                all_results.append({
                    'session': label, 'band': band_name, 'pipeline': pipe, **met,
                })

            # Per-channel Kalman MAE
            print(f"\n  Per-channel Kalman MAE ({unit_label}):")
            for ch in CHANNELS:
                met = compute_metrics(gt, df[f'kalman_{ch}'].values, unit_scale)
                print(f"    {ch:<6s}: {met['mae']:.2f}")

            # Window-level data
            for _, row in df.iterrows():
                wrow = {
                    'session': label, 'band': band_name,
                    'stage': row['stage'], 'gt_hz': row['gt_hz'],
                    'best_quality_chan': row['best_quality_chan'],
                    'oracle_chan': row['oracle_chan'],
                }
                for pipe in ['single_diff', 'multi_quality', 'multi_oracle']:
                    wrow[pipe] = row[pipe]
                all_windows.append(wrow)

            ts_path = OUT_DIR / f'timeseries_{label}_{band_name}.png'
            plot_timeseries(df, band_name, label, ts_path)

    results_df = pd.DataFrame(all_results)
    windows_df = pd.DataFrame(all_windows)

    # Aggregate summary
    print(f"\n\n{'='*60}")
    print("AGGREGATE (mean across 12 sessions, no k)")
    print(f"{'='*60}")
    for band_name in ['resp', 'card']:
        _, _, _, unit_scale, unit_label = BANDS[band_name]
        sub = results_df[results_df['band'] == band_name]
        print(f"\n  {band_name.upper()} ({unit_label}):")
        print(f"  {'Pipeline':<30s} {'MAE':>8s} {'RMSE':>8s} {'Bias':>8s} {'r':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for pipe in ['single_diff', 'multi_quality', 'multi_oracle']:
            ps = sub[sub['pipeline'] == pipe]
            print(f"  {PIPE_LABELS[pipe]:<30s} "
                  f"{ps['mae'].mean():>8.2f} {ps['rmse'].mean():>8.2f} "
                  f"{ps['bias'].mean():>+8.2f} {ps['r'].mean():>8.3f}")

    # Plots
    print(f"\nGenerating plots...")
    plot_aggregate(results_df, OUT_DIR / 'aggregate_comparison.png')
    plot_channel_selection(windows_df, OUT_DIR / 'channel_selection_frequency.png')
    for band_name in ['resp', 'card']:
        plot_per_stage(windows_df, band_name, OUT_DIR / f'per_stage_{band_name}.png')

    results_df.to_csv(OUT_DIR / 'multichannel_results.csv', index=False)
    windows_df.to_csv(OUT_DIR / 'multichannel_windows.csv', index=False)

    print(f"\n  All outputs in: {OUT_DIR}")
    print(f"  {len(list(OUT_DIR.glob('*.png')))} PNGs, 2 CSVs")


if __name__ == '__main__':
    main()
