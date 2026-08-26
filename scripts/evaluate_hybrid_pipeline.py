#!/usr/bin/env python
"""
Phase 3: Formal evaluation of the hybrid rate pipeline.

Compares:
  - Baseline k-scaled: hilbert/k (cardiac), peaks/k (resp)
  - Kalman pipeline: spectral+adaptive → Kalman → /k
  - Per-session k vs LOSO k (leave-one-subject-out)

Outputs to reports/rates/hybrid_phase3/:
  - Aggregate summary table
  - Per-session results CSV
  - Bland-Altman plots (aggregate + per-subject)
  - Per-stage MAE breakdown
  - Improvement heatmap (Kalman vs baselines)
  - Wilcoxon signed-rank test results
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
from scipy.stats import wilcoxon

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
    estimate_rate, rate_acf, kalman_rate_track,
)

import functools
print = functools.partial(print, flush=True)

OUT_DIR = ROOT / 'reports' / 'rates' / 'hybrid_phase3'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN_SEC = 30.0
STEP_SEC = 30.0

BANDS = {
    'resp': (RESP_LO, RESP_HI, 'Thorax', 60.0, 'br/min'),
    'card': (CARD_LO, CARD_HI, 'Pleth',  60.0, 'BPM'),
}

STAGE_NAME_MAP = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

PIPELINES = {
    'baseline_k':  'Baseline /k',
    'kalman_raw':  'Kalman (no k)',
    'kalman_k':    'Kalman /k (per-session)',
    'kalman_loso': 'Kalman /k (LOSO)',
}
PIPE_COLORS = {
    'baseline_k':  '#9B59B6',
    'kalman_raw':  '#E67E22',
    'kalman_k':    '#E74C3C',
    'kalman_loso': '#2ECC71',
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
    idx = np.searchsorted(t_hr, t_sec / 3600.0, side='right') - 1
    if idx < 0 or idx >= len(codes):
        return -1
    return int(codes[idx])


def compute_session_windows(session, band_name):
    """Compute per-window rates for all methods + Kalman on one session/band."""
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
    baseline_raw_arr = []

    for start in range(0, n_total - win_n + 1, step_n):
        seg = sig[start:start + win_n]
        seg_gt = gt_sig[start:start + win_n]
        t_center = (start + win_n / 2.0) / fs

        rates = estimate_rate(seg, f_lo, f_hi, fs)
        gt_rate = rate_acf(seg_gt, f_lo, f_hi, fs, prominence=0.05)

        stage_code = get_stage_at_time(profile, t_center)
        stage_name = STAGE_NAME_MAP.get(stage_code, '?')

        spectral_arr.append(rates['spectral'])
        adaptive_arr.append(rates['adaptive_peaks'])

        if band_name == 'card':
            baseline_raw_arr.append(rates['hilbert'])
        else:
            baseline_raw_arr.append(rates['peaks'])

        rows.append({
            't_s': t_center,
            'gt_hz': gt_rate,
            'spectral': rates['spectral'],
            'adaptive_peaks': rates['adaptive_peaks'],
            'baseline_raw': baseline_raw_arr[-1],
            'stage_code': stage_code,
            'stage': stage_name,
        })

    estimates = {
        'spectral': np.array(spectral_arr),
        'adaptive_peaks': np.array(adaptive_arr),
    }
    kalman_out = kalman_rate_track(estimates, f_lo, f_hi, step_sec=STEP_SEC)

    for i, row in enumerate(rows):
        row['kalman_raw'] = kalman_out[i]

    return pd.DataFrame(rows)


def calibrate_k_from_windows(df, rate_col='kalman_raw'):
    """Compute k = median(rate / gt) from valid windows."""
    valid = df[['gt_hz', rate_col]].dropna()
    valid = valid[(valid['gt_hz'] > 0) & np.isfinite(valid[rate_col]) & (valid[rate_col] > 0)]
    if len(valid) < 10:
        return np.nan
    ratios = valid[rate_col].values / valid['gt_hz'].values
    return float(np.median(ratios))


def compute_metrics(gt, est, unit_scale):
    valid = np.isfinite(gt) & np.isfinite(est) & (gt > 0)
    if valid.sum() < 10:
        return {'n': 0, 'mae': np.nan, 'rmse': np.nan, 'bias': np.nan, 'r': np.nan}
    g, e = gt[valid], est[valid]
    err = e - g
    mae = np.mean(np.abs(err)) * unit_scale
    rmse = np.sqrt(np.mean(err ** 2)) * unit_scale
    bias = np.mean(err) * unit_scale
    r = np.corrcoef(g, e)[0, 1] if np.std(g) > 0 and np.std(e) > 0 else np.nan
    return {'n': int(valid.sum()), 'mae': mae, 'rmse': rmse, 'bias': bias, 'r': r}


def plot_bland_altman_aggregate(windows_df, band_name, out_path):
    _, _, _, unit_scale, unit_label = BANDS[band_name]
    sub = windows_df[windows_df['band'] == band_name].copy()

    pipes = ['baseline_k', 'kalman_k', 'kalman_loso']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, pipe in zip(axes, pipes):
        valid = sub[['gt_hz', pipe]].dropna()
        valid = valid[(valid['gt_hz'] > 0) & np.isfinite(valid[pipe])]
        if len(valid) < 10:
            ax.set_title(f'{PIPELINES[pipe]}\n(insufficient data)')
            continue
        gt = valid['gt_hz'].values * unit_scale
        est = valid[pipe].values * unit_scale
        mean_v = (gt + est) / 2.0
        diff_v = est - gt
        bias = np.mean(diff_v)
        loa = 1.96 * np.std(diff_v)
        mae = np.mean(np.abs(diff_v))

        ax.scatter(mean_v, diff_v, s=2, alpha=0.15, color=PIPE_COLORS[pipe])
        ax.axhline(bias, color='red', ls='-', lw=1)
        ax.axhline(bias + loa, color='grey', ls='--', lw=0.7)
        ax.axhline(bias - loa, color='grey', ls='--', lw=0.7)
        ax.set_xlabel(f'Mean ({unit_label})')
        ax.set_title(f'{PIPELINES[pipe]}\nMAE={mae:.2f}, bias={bias:+.2f}, LoA=±{loa:.1f}')

    axes[0].set_ylabel(f'Est − GT ({unit_label})')
    fig.suptitle(f'{band_name.upper()} — Bland-Altman (all sessions pooled)', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_stage(windows_df, band_name, out_path):
    _, _, _, unit_scale, unit_label = BANDS[band_name]
    sub = windows_df[windows_df['band'] == band_name].copy()

    pipes = ['baseline_k', 'kalman_k', 'kalman_loso']
    fig, ax = plt.subplots(figsize=(12, 6))
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
        bars = ax.bar(x, maes, width=width, color=PIPE_COLORS[pipe], alpha=0.8,
                      label=PIPELINES[pipe])
        for bar, m in zip(bars, maes):
            if np.isfinite(m):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f'{m:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(np.arange(n_stages))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel(f'MAE ({unit_label})')
    ax.set_title(f'{band_name.upper()} — MAE by sleep stage (k-scaled, all sessions)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_session_comparison(results_df, band_name, out_path):
    _, _, _, unit_scale, unit_label = BANDS[band_name]
    sub = results_df[results_df['band'] == band_name].copy()
    sessions = sorted(sub['session'].unique())

    pipes = ['baseline_k', 'kalman_k', 'kalman_loso']
    fig, ax = plt.subplots(figsize=(14, 6))
    n_sess = len(sessions)
    n_pipes = len(pipes)
    width = 0.8 / n_pipes
    offsets = np.arange(n_pipes) - (n_pipes - 1) / 2

    for j, pipe in enumerate(pipes):
        maes = []
        for sess in sessions:
            row = sub[(sub['session'] == sess) & (sub['pipeline'] == pipe)]
            maes.append(row['mae'].values[0] if len(row) else np.nan)
        x = np.arange(n_sess) + offsets[j] * width
        ax.bar(x, maes, width=width, color=PIPE_COLORS[pipe], alpha=0.8,
               label=PIPELINES[pipe])

    ax.set_xticks(np.arange(n_sess))
    ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'MAE ({unit_label})')
    ax.set_title(f'{band_name.upper()} — Per-session MAE comparison')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_timeseries(df, band_name, session_label, out_path):
    _, _, _, unit_scale, unit_label = BANDS[band_name]
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    t_min = df['t_s'] / 60.0

    ax = axes[0]
    valid_gt = np.isfinite(df['gt_hz']) & (df['gt_hz'] > 0)
    ax.plot(t_min[valid_gt], df.loc[valid_gt, 'gt_hz'] * unit_scale,
            'k-', lw=1.2, alpha=0.5, label='GT')
    for pipe in ['baseline_k', 'kalman_loso']:
        if pipe in df.columns:
            ax.plot(t_min, df[pipe] * unit_scale, '-',
                    color=PIPE_COLORS[pipe],
                    lw=1.0 if pipe == 'baseline_k' else 1.6,
                    alpha=0.5 if pipe == 'baseline_k' else 0.85,
                    label=PIPELINES[pipe])
    ax.set_ylabel(f'Rate ({unit_label})')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{session_label} — {band_name} — baseline /k vs Kalman /k (LOSO)')

    ax = axes[1]
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
    patches = [Patch(color=stage_color_map[s], label=s) for s in STAGE_ORDER
               if s in stage_color_map]
    ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=5)

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

    # ── Step 1: compute per-window rates + per-session k for all pipelines ──
    session_dfs = {}
    k_values = {}

    for idx, s in enumerate(sessions):
        label = s.meta['label']
        subject = s.meta['subject']
        print(f"Computing {label}...", end=' ')
        t0 = time.time()

        for band_name in ['resp', 'card']:
            df = compute_session_windows(s, band_name)

            # Per-session k for baseline
            k_baseline = calibrate_k_from_windows(df, 'baseline_raw')
            # Per-session k for Kalman
            k_kalman = calibrate_k_from_windows(df, 'kalman_raw')

            df['baseline_k'] = df['baseline_raw'] / k_baseline if np.isfinite(k_baseline) and k_baseline > 0 else np.nan
            df['kalman_k'] = df['kalman_raw'] / k_kalman if np.isfinite(k_kalman) and k_kalman > 0 else np.nan

            key = (label, band_name)
            session_dfs[key] = df
            k_values[(label, band_name, 'baseline')] = k_baseline
            k_values[(label, band_name, 'kalman')] = k_kalman

        print(f"{time.time() - t0:.1f}s")

    # ── Step 2: LOSO k-calibration ──────────────────────────────────────────
    subjects = ['OS001', 'OS002', 'OS003', 'OS004', 'OS005', 'OS006']
    subject_sessions = defaultdict(list)
    for idx, s in enumerate(sessions):
        subject_sessions[s.meta['subject']].append(s.meta['label'])

    print("\nLOSO k-calibration:")
    for band_name in ['resp', 'card']:
        for test_subj in subjects:
            train_ks = []
            for subj, labels in subject_sessions.items():
                if subj == test_subj:
                    continue
                for lab in labels:
                    k = k_values.get((lab, band_name, 'kalman'), np.nan)
                    if np.isfinite(k):
                        train_ks.append(k)

            k_loso = float(np.median(train_ks)) if len(train_ks) >= 2 else np.nan

            for lab in subject_sessions[test_subj]:
                key = (lab, band_name)
                if key in session_dfs:
                    df = session_dfs[key]
                    if np.isfinite(k_loso) and k_loso > 0:
                        df['kalman_loso'] = df['kalman_raw'] / k_loso
                    else:
                        df['kalman_loso'] = np.nan
                    k_values[(lab, band_name, 'kalman_loso')] = k_loso

            print(f"  {band_name} test={test_subj}: k_loso={k_loso:.3f} (from {len(train_ks)} sessions)")

    # ── Step 3: compute metrics ─────────────────────────────────────────────
    all_results = []
    all_windows = []

    for (label, band_name), df in session_dfs.items():
        _, _, _, unit_scale, unit_label = BANDS[band_name]
        subject = label[:4]  # e.g. SOS0 → need full subject
        # Extract subject from session meta
        meta_match = [s.meta for s in sessions if s.meta['label'] == label]
        subject = meta_match[0]['subject'] if meta_match else label

        for pipe in ['baseline_k', 'kalman_raw', 'kalman_k', 'kalman_loso']:
            if pipe not in df.columns:
                continue
            gt = df['gt_hz'].values
            est = df[pipe].values
            met = compute_metrics(gt, est, unit_scale)
            all_results.append({
                'session': label, 'subject': subject, 'band': band_name,
                'pipeline': pipe, **met,
                'k': k_values.get((label, band_name,
                                   'kalman_loso' if pipe == 'kalman_loso' else
                                   'kalman' if 'kalman' in pipe else 'baseline'), np.nan),
            })

        # Window-level data
        for _, row in df.iterrows():
            wrow = {
                'session': label, 'subject': subject, 'band': band_name,
                'stage': row['stage'], 'gt_hz': row['gt_hz'],
            }
            for pipe in ['baseline_k', 'kalman_raw', 'kalman_k', 'kalman_loso']:
                wrow[pipe] = row.get(pipe, np.nan)
            all_windows.append(wrow)

    results_df = pd.DataFrame(all_results)
    windows_df = pd.DataFrame(all_windows)

    # ── Step 4: print summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (mean ± std across 12 sessions)")
    print(f"{'='*70}")

    for band_name in ['resp', 'card']:
        _, _, _, unit_scale, unit_label = BANDS[band_name]
        sub = results_df[results_df['band'] == band_name]
        print(f"\n  {band_name.upper()} ({unit_label}):")
        print(f"  {'Pipeline':<28s} {'MAE':>8s} {'RMSE':>8s} {'Bias':>9s} {'r':>8s} {'k':>8s}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
        for pipe in ['baseline_k', 'kalman_raw', 'kalman_k', 'kalman_loso']:
            ps = sub[sub['pipeline'] == pipe]
            if len(ps) == 0:
                continue
            k_str = f"{ps['k'].mean():.2f}" if np.isfinite(ps['k'].mean()) else "  —"
            print(f"  {PIPELINES[pipe]:<28s} "
                  f"{ps['mae'].mean():>8.2f} "
                  f"{ps['rmse'].mean():>8.2f} "
                  f"{ps['bias'].mean():>+9.2f} "
                  f"{ps['r'].mean():>8.3f} "
                  f"{k_str:>8s}")

    # ── Step 5: Wilcoxon tests ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STATISTICAL TESTS (paired Wilcoxon on per-session MAE)")
    print(f"{'='*70}")

    for band_name in ['resp', 'card']:
        _, _, _, unit_scale, unit_label = BANDS[band_name]
        sub = results_df[results_df['band'] == band_name]
        print(f"\n  {band_name.upper()}:")

        for test_pipe in ['kalman_k', 'kalman_loso']:
            baseline_maes = sub[sub['pipeline'] == 'baseline_k'].sort_values('session')['mae'].values
            test_maes = sub[sub['pipeline'] == test_pipe].sort_values('session')['mae'].values
            if len(baseline_maes) == len(test_maes) and len(baseline_maes) >= 5:
                diff = baseline_maes - test_maes
                try:
                    stat, p = wilcoxon(diff, alternative='greater')
                    wins = (diff > 0).sum()
                    losses = (diff < 0).sum()
                    print(f"  baseline vs {PIPELINES[test_pipe]}: "
                          f"W={stat:.0f}, p={p:.4f}, "
                          f"wins={wins}/losses={losses}/ties={len(diff)-wins-losses}")
                except Exception as e:
                    print(f"  baseline vs {PIPELINES[test_pipe]}: test failed ({e})")

    # ── Step 6: k summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("K-CALIBRATION SUMMARY")
    print(f"{'='*70}")

    for band_name in ['resp', 'card']:
        print(f"\n  {band_name.upper()}:")
        print(f"  {'Session':<12s} {'k_baseline':>12s} {'k_kalman':>12s} {'k_loso':>12s}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for idx, s in enumerate(sessions):
            lab = s.meta['label']
            kb = k_values.get((lab, band_name, 'baseline'), np.nan)
            kk = k_values.get((lab, band_name, 'kalman'), np.nan)
            kl = k_values.get((lab, band_name, 'kalman_loso'), np.nan)
            print(f"  {lab:<12s} {kb:>12.3f} {kk:>12.3f} {kl:>12.3f}")

    # ── Step 7: plots ───────────────────────────────────────────────────────
    print(f"\nGenerating plots...")

    for band_name in ['resp', 'card']:
        # Bland-Altman aggregate
        ba_path = OUT_DIR / f'bland_altman_aggregate_{band_name}.png'
        plot_bland_altman_aggregate(windows_df, band_name, ba_path)

        # Per-stage
        stage_path = OUT_DIR / f'per_stage_mae_{band_name}.png'
        plot_per_stage(windows_df, band_name, stage_path)

        # Per-session bar chart
        sess_path = OUT_DIR / f'session_comparison_{band_name}.png'
        plot_session_comparison(results_df, band_name, sess_path)

    # Per-session time-series (baseline vs kalman_loso)
    for (label, band_name), df in session_dfs.items():
        ts_path = OUT_DIR / f'timeseries_{label}_{band_name}.png'
        plot_timeseries(df, band_name, label, ts_path)

    # ── Step 8: save CSVs ───────────────────────────────────────────────────
    results_df.to_csv(OUT_DIR / 'evaluation_results.csv', index=False)
    windows_df.to_csv(OUT_DIR / 'evaluation_windows.csv', index=False)

    # k summary
    k_rows = []
    for (lab, band, kind), val in k_values.items():
        k_rows.append({'session': lab, 'band': band, 'type': kind, 'k': val})
    pd.DataFrame(k_rows).to_csv(OUT_DIR / 'k_calibration.csv', index=False)

    print(f"\n  All outputs in: {OUT_DIR}")
    print(f"  CSVs: evaluation_results.csv, evaluation_windows.csv, k_calibration.csv")
    print(f"  Plots: {len(list(OUT_DIR.glob('*.png')))} PNGs")


if __name__ == '__main__':
    main()
