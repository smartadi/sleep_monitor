#!/usr/bin/env python
"""
Best-of-both rate pipeline: Kalman (reactive) for resp, hilbert for cardiac.
Multi-channel quality-weighted fusion + heavy temporal smoothing + LOSO k-scaling.

Tunable parameters at the top — adjust SMOOTH_WIN, KALMAN_R_SCALE, etc.

Outputs to reports/rates/best_pipeline/:
  - Per-session time-series (resp + cardiac stacked with GT, stage bar)
  - Per-session Bland-Altman
  - Aggregate bar chart (MAE comparison)
  - Per-stage breakdown
  - CSV results
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

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
    rate_spectral, rate_adaptive_peaks, rate_acf, rate_hilbert,
    kalman_rate_track,
)
from sleep_monitor.quality import window_features, combined_quality

import functools
print = functools.partial(print, flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS — tweak these
# ══════════════════════════════════════════════════════════════════════════════

SMOOTH_WIN = 7          # temporal median filter width (epochs). 7 = 3.5 min @ 30s windows.
                        # increase for heavier smoothing (must be odd).

KALMAN_R_SCALE = 0.3    # multiply default R_base by this. <1 = more reactive (trusts
                        # instantaneous measurements more). 0.3 = ~3x more reactive.

KALMAN_Q_SCALE = 2.0    # multiply default Q by this. >1 = allows faster rate changes.

# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = ROOT / 'reports' / 'rates' / 'best_pipeline'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN_SEC = 30.0
STEP_SEC = 30.0

CHANNELS = ['CLE', 'CRE', 'CH', 'avg', 'diff']
CHAN_COLORS = {
    'CLE': '#27AE60', 'CRE': '#8E44AD', 'CH': '#2980B9',
    'avg': '#E67E22', 'diff': '#E74C3C',
}

STAGE_NAME_MAP = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

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


def temporal_smooth(arr, win=SMOOTH_WIN):
    if win <= 1:
        return arr
    out = arr.copy()
    valid = np.isfinite(arr)
    if valid.sum() < win:
        return out
    tmp = arr.copy()
    tmp[~valid] = np.nanmedian(arr)
    smoothed = median_filter(tmp, size=win, mode='reflect')
    out[valid] = smoothed[valid]
    return out


def reactive_kalman_track(estimates, f_lo, f_hi, step_sec=30.0):
    """Kalman with tuned reactivity: lower R, higher Q."""
    methods = list(estimates.keys())

    if f_hi <= 0.6:
        max_delta_hz = 2.0 / 60.0 * (step_sec / 30.0) * np.sqrt(KALMAN_Q_SCALE)
        R_base = {m: (2.5 / 60.0) ** 2 * KALMAN_R_SCALE for m in methods}
    else:
        max_delta_hz = 5.0 / 60.0 * (step_sec / 30.0) * np.sqrt(KALMAN_Q_SCALE)
        R_base = {m: (30.0 / 60.0) ** 2 * KALMAN_R_SCALE for m in methods}

    return kalman_rate_track(estimates, f_lo, f_hi,
                             step_sec=step_sec,
                             max_delta_hz=max_delta_hz,
                             R_base=R_base)


def run_session(session):
    """Run best pipeline on one session. Returns DataFrame with all windows."""
    fs = session.fs
    profile = getattr(session, 'sleep_profile', None)
    raw_channels, acc = prepare_channels(session)

    gt_resp_sig = bandpass(session.psg['Thorax'].astype(np.float64), RESP_LO, RESP_HI, fs)
    gt_card_sig = bandpass(session.psg['Pleth'].astype(np.float64), CARD_LO, CARD_HI, fs)

    # Preprocess each channel for both bands
    bp_resp = {}
    bp_card = {}
    for ch_name, raw in raw_channels.items():
        n = min(len(raw), len(acc))
        bp_resp[ch_name] = remove_acc_artifact(raw[:n], acc[:n], RESP_LO, RESP_HI, fs)
        bp_card[ch_name] = remove_acc_artifact(raw[:n], acc[:n], CARD_LO, CARD_HI, fs)

    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    n_total = min(
        min(len(s) for s in bp_resp.values()),
        min(len(s) for s in bp_card.values()),
        len(gt_resp_sig), len(gt_card_sig),
    )

    # Per-channel per-window raw estimates
    per_chan_resp_spec = {ch: [] for ch in CHANNELS}
    per_chan_resp_adap = {ch: [] for ch in CHANNELS}
    per_chan_card_hilb = {ch: [] for ch in CHANNELS}
    per_chan_resp_qual = {ch: [] for ch in CHANNELS}
    per_chan_card_qual = {ch: [] for ch in CHANNELS}
    gt_resp_rates = []
    gt_card_rates = []
    t_centers = []
    stages = []

    for start in range(0, n_total - win_n + 1, step_n):
        t_center = (start + win_n / 2.0) / fs
        t_centers.append(t_center)

        seg_gt_r = gt_resp_sig[start:start + win_n]
        seg_gt_c = gt_card_sig[start:start + win_n]
        gt_resp_rates.append(rate_acf(seg_gt_r, RESP_LO, RESP_HI, fs, prominence=0.05))
        gt_card_rates.append(rate_acf(seg_gt_c, CARD_LO, CARD_HI, fs, prominence=0.05))

        stage_code = get_stage_at_time(profile, t_center)
        stages.append(STAGE_NAME_MAP.get(stage_code, '?'))

        acc_win = acc[start:start + win_n] if start + win_n <= len(acc) else None

        for ch_name in CHANNELS:
            seg_r = bp_resp[ch_name][start:start + win_n]
            seg_c = bp_card[ch_name][start:start + win_n]

            # Resp: spectral + adaptive_peaks (for Kalman)
            r_spec = rate_spectral(seg_r, RESP_LO, RESP_HI, fs)
            r_adap = rate_adaptive_peaks(seg_r, RESP_LO, RESP_HI, fs)
            per_chan_resp_spec[ch_name].append(r_spec)
            per_chan_resp_adap[ch_name].append(r_adap)

            feats_r = window_features(seg_r, acc_win, RESP_LO, RESP_HI, fs,
                                      rates_hz={'spectral': r_spec, 'adaptive': r_adap})
            per_chan_resp_qual[ch_name].append(combined_quality(feats_r))

            # Cardiac: hilbert only
            r_hilb = rate_hilbert(seg_c, CARD_LO, CARD_HI, fs)
            per_chan_card_hilb[ch_name].append(r_hilb)

            feats_c = window_features(seg_c, acc_win, CARD_LO, CARD_HI, fs,
                                      rates_hz={'hilbert': r_hilb} if np.isfinite(r_hilb) else {})
            per_chan_card_qual[ch_name].append(combined_quality(feats_c))

    N = len(t_centers)
    gt_resp = np.array(gt_resp_rates)
    gt_card = np.array(gt_card_rates)

    # ── RESP: reactive Kalman per channel, then quality-weighted fusion ──
    per_chan_resp_kalman = {}
    for ch_name in CHANNELS:
        estimates = {
            'spectral': np.array(per_chan_resp_spec[ch_name]),
            'adaptive_peaks': np.array(per_chan_resp_adap[ch_name]),
        }
        per_chan_resp_kalman[ch_name] = reactive_kalman_track(
            estimates, RESP_LO, RESP_HI, step_sec=STEP_SEC)

    resp_quality = {ch: np.array(per_chan_resp_qual[ch]) for ch in CHANNELS}

    resp_fused = np.full(N, np.nan)
    for i in range(N):
        rates_i, weights_i = [], []
        for ch in CHANNELS:
            r = per_chan_resp_kalman[ch][i]
            q = resp_quality[ch][i]
            if np.isfinite(r) and RESP_LO <= r <= RESP_HI and np.isfinite(q) and q > 0:
                rates_i.append(r)
                weights_i.append(q)
        if rates_i:
            resp_fused[i] = np.average(rates_i, weights=weights_i)

    # ── CARDIAC: hilbert per channel, quality-weighted fusion (no Kalman) ──
    card_quality = {ch: np.array(per_chan_card_qual[ch]) for ch in CHANNELS}

    card_fused = np.full(N, np.nan)
    for i in range(N):
        rates_i, weights_i = [], []
        for ch in CHANNELS:
            r_arr = per_chan_card_hilb[ch]
            r = r_arr[i]
            q = card_quality[ch][i]
            if np.isfinite(r) and CARD_LO <= r <= CARD_HI and np.isfinite(q) and q > 0:
                rates_i.append(r)
                weights_i.append(q)
        if rates_i:
            card_fused[i] = np.average(rates_i, weights=weights_i)

    # ── Temporal smoothing ──
    resp_smooth = temporal_smooth(resp_fused, SMOOTH_WIN)
    card_smooth = temporal_smooth(card_fused, SMOOTH_WIN)

    # Build DataFrame
    rows = []
    for i in range(N):
        rows.append({
            't_s': t_centers[i],
            'stage': stages[i],
            'gt_resp_hz': gt_resp[i],
            'gt_card_hz': gt_card[i],
            'resp_fused_hz': resp_fused[i],
            'resp_smooth_hz': resp_smooth[i],
            'card_fused_hz': card_fused[i],
            'card_smooth_hz': card_smooth[i],
        })

    return pd.DataFrame(rows)


def calibrate_k_from_df(df, band):
    """Per-session k from paired windows: median(CAP_rate / GT_rate)."""
    if band == 'resp':
        cap = df['resp_smooth_hz'].values
        gt = df['gt_resp_hz'].values
    else:
        cap = df['card_smooth_hz'].values
        gt = df['gt_card_hz'].values

    valid = np.isfinite(cap) & np.isfinite(gt) & (gt > 0) & (cap > 0)
    if valid.sum() < 10:
        return np.nan
    ratios = cap[valid] / gt[valid]
    return float(np.median(ratios))


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


def plot_session(df, label, k_resp, k_card, out_dir):
    """Full-session time-series: resp + cardiac stacked, with stage bar."""
    t_min = df['t_s'].values / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 1]})

    # Resp
    ax = axes[0]
    v = np.isfinite(df['gt_resp_hz'].values) & (df['gt_resp_hz'].values > 0)
    ax.plot(t_min[v], df['gt_resp_hz'].values[v] * 60, 'k-', lw=1.0, alpha=0.4, label='GT')
    ax.plot(t_min, df['resp_fused_hz'].values * 60, '-', color='#BDC3C7', lw=0.5, alpha=0.4, label='Fused (raw)')

    resp_k = df['resp_smooth_hz'].values / k_resp if k_resp > 0 else df['resp_smooth_hz'].values
    ax.plot(t_min, resp_k * 60, '-', color='#E74C3C', lw=1.8, label=f'Kalman+smooth/k (k={k_resp:.2f})')
    ax.set_ylabel('Resp (br/min)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{label} — best pipeline (smooth={SMOOTH_WIN}, R×{KALMAN_R_SCALE}, Q×{KALMAN_Q_SCALE})')

    met = compute_metrics(df['gt_resp_hz'].values, resp_k, 60.0)
    ax.text(0.01, 0.95, f"MAE={met['mae']:.2f} br/min  r={met['r']:.3f}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Cardiac
    ax = axes[1]
    v = np.isfinite(df['gt_card_hz'].values) & (df['gt_card_hz'].values > 0)
    ax.plot(t_min[v], df['gt_card_hz'].values[v] * 60, 'k-', lw=1.0, alpha=0.4, label='GT')
    ax.plot(t_min, df['card_fused_hz'].values * 60, '-', color='#BDC3C7', lw=0.5, alpha=0.4, label='Fused (raw)')

    card_k = df['card_smooth_hz'].values / k_card if k_card > 0 else df['card_smooth_hz'].values
    ax.plot(t_min, card_k * 60, '-', color='#3498DB', lw=1.8, label=f'Hilbert+smooth/k (k={k_card:.2f})')
    ax.set_ylabel('Cardiac (BPM)')
    ax.legend(loc='upper right', fontsize=8)

    met = compute_metrics(df['gt_card_hz'].values, card_k, 60.0)
    ax.text(0.01, 0.95, f"MAE={met['mae']:.2f} BPM  r={met['r']:.3f}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Stage bar
    ax = axes[2]
    stage_color_map = {
        'Wake': STAGE_COLORS[4], 'N1': STAGE_COLORS[3],
        'N2': STAGE_COLORS[2], 'N3': STAGE_COLORS[1], 'REM': STAGE_COLORS[0],
    }
    for i in range(len(t_min)):
        s = df.iloc[i]['stage']
        c = stage_color_map.get(s, '#AAAAAA')
        ax.axvspan(t_min[i] - 0.25, t_min[i] + 0.25, color=c, alpha=0.7, linewidth=0)
    ax.set_yticks([])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Stage')

    plt.tight_layout()
    fig.savefig(out_dir / f'best_{label}.png')
    plt.close(fig)


def plot_bland_altman(df, label, k_resp, k_card, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (band, gt_col, est_arr, unit, ulabel) in enumerate([
        ('resp', 'gt_resp_hz',
         df['resp_smooth_hz'].values / k_resp if k_resp > 0 else df['resp_smooth_hz'].values,
         60.0, 'br/min'),
        ('card', 'gt_card_hz',
         df['card_smooth_hz'].values / k_card if k_card > 0 else df['card_smooth_hz'].values,
         60.0, 'BPM'),
    ]):
        ax = axes[idx]
        gt = df[gt_col].values
        v = np.isfinite(gt) & np.isfinite(est_arr) & (gt > 0)
        if v.sum() < 10:
            continue
        mean_val = (gt[v] + est_arr[v]) / 2.0 * unit
        diff_val = (est_arr[v] - gt[v]) * unit
        bias = np.mean(diff_val)
        sd = np.std(diff_val)

        ax.scatter(mean_val, diff_val, s=3, alpha=0.3, c='#34495E')
        ax.axhline(bias, color='red', lw=1.2, label=f'Bias={bias:.2f}')
        ax.axhline(bias + 1.96 * sd, color='red', lw=0.8, ls='--', label=f'+1.96σ={bias+1.96*sd:.2f}')
        ax.axhline(bias - 1.96 * sd, color='red', lw=0.8, ls='--', label=f'-1.96σ={bias-1.96*sd:.2f}')
        ax.set_xlabel(f'Mean ({ulabel})')
        ax.set_ylabel(f'Difference ({ulabel})')
        ax.set_title(f'{label} {band.upper()} Bland-Altman')
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / f'bland_altman_{label}.png')
    plt.close(fig)


def main():
    print(f"Best pipeline: smooth={SMOOTH_WIN}, R_scale={KALMAN_R_SCALE}, Q_scale={KALMAN_Q_SCALE}")
    print(f"Resp: reactive Kalman (spectral+adaptive) -> multi-ch fusion -> smooth -> k")
    print(f"Card: hilbert -> multi-ch fusion -> smooth -> k\n")

    sessions = load_all_sessions()
    for s in sessions:
        try:
            s.sleep_profile = load_sleep_profile(s)
        except Exception:
            s.sleep_profile = None

    # Phase 1: run all sessions, collect raw results
    session_dfs = {}
    for idx, s in enumerate(sessions):
        label = s.meta['label']
        t0 = time.time()
        print(f"[{idx+1:2d}/12] {label}...", end=' ')
        df = run_session(s)
        elapsed = time.time() - t0
        session_dfs[label] = df
        print(f"{len(df)} windows, {elapsed:.1f}s")

    # Phase 2: per-session k calibration
    k_values = {}
    for label, df in session_dfs.items():
        k_values[label] = {
            'k_resp': calibrate_k_from_df(df, 'resp'),
            'k_card': calibrate_k_from_df(df, 'card'),
        }

    # Phase 3: LOSO k calibration
    labels = list(session_dfs.keys())
    subjects = {}
    for lab in labels:
        subj = lab[:2]
        subjects.setdefault(subj, []).append(lab)

    loso_k = {}
    for held_out_subj, held_out_sessions in subjects.items():
        train_labels = [l for l in labels if l not in held_out_sessions]
        train_k_resp = [k_values[l]['k_resp'] for l in train_labels if np.isfinite(k_values[l]['k_resp'])]
        train_k_card = [k_values[l]['k_card'] for l in train_labels if np.isfinite(k_values[l]['k_card'])]
        k_r = float(np.median(train_k_resp)) if train_k_resp else np.nan
        k_c = float(np.median(train_k_card)) if train_k_card else np.nan
        for lab in held_out_sessions:
            loso_k[lab] = {'k_resp': k_r, 'k_card': k_c}

    # Phase 4: compute metrics and generate plots
    all_results = []
    all_windows = []

    print(f"\n{'='*80}")
    print(f"{'Session':<8s} | {'Resp MAE (br/min)':>18s} | {'Card MAE (BPM)':>18s} | {'k_resp':>7s} {'k_card':>7s}")
    print(f"{'':->8s}-+-{'':->18s}-+-{'':->18s}-+-{'':->7s}-{'':->7s}")

    for label, df in session_dfs.items():
        kr_ps = k_values[label]['k_resp']
        kc_ps = k_values[label]['k_card']
        kr_lo = loso_k[label]['k_resp']
        kc_lo = loso_k[label]['k_card']

        # Per-session k results (main)
        resp_k = df['resp_smooth_hz'].values / kr_ps if kr_ps > 0 else df['resp_smooth_hz'].values
        card_k = df['card_smooth_hz'].values / kc_ps if kc_ps > 0 else df['card_smooth_hz'].values
        met_r = compute_metrics(df['gt_resp_hz'].values, resp_k, 60.0)
        met_c = compute_metrics(df['gt_card_hz'].values, card_k, 60.0)

        print(f"{label:<8s} | {met_r['mae']:>18.2f} | {met_c['mae']:>18.2f} | {kr_ps:>7.3f} {kc_ps:>7.3f}")

        # LOSO k results
        resp_loso = df['resp_smooth_hz'].values / kr_lo if kr_lo > 0 else df['resp_smooth_hz'].values
        card_loso = df['card_smooth_hz'].values / kc_lo if kc_lo > 0 else df['card_smooth_hz'].values
        met_r_lo = compute_metrics(df['gt_resp_hz'].values, resp_loso, 60.0)
        met_c_lo = compute_metrics(df['gt_card_hz'].values, card_loso, 60.0)

        # No-k results
        met_r_nk = compute_metrics(df['gt_resp_hz'].values, df['resp_smooth_hz'].values, 60.0)
        met_c_nk = compute_metrics(df['gt_card_hz'].values, df['card_smooth_hz'].values, 60.0)

        for pipe, mr, mc in [
            ('no_k', met_r_nk, met_c_nk),
            ('per_session_k', met_r, met_c),
            ('loso_k', met_r_lo, met_c_lo),
        ]:
            all_results.append({
                'session': label, 'pipeline': pipe,
                'resp_mae': mr['mae'], 'resp_rmse': mr['rmse'],
                'resp_bias': mr['bias'], 'resp_r': mr['r'],
                'card_mae': mc['mae'], 'card_rmse': mc['rmse'],
                'card_bias': mc['bias'], 'card_r': mc['r'],
                'k_resp': kr_ps if pipe == 'per_session_k' else (kr_lo if pipe == 'loso_k' else np.nan),
                'k_card': kc_ps if pipe == 'per_session_k' else (kc_lo if pipe == 'loso_k' else np.nan),
            })

        # Window-level data for per-stage analysis
        for _, row in df.iterrows():
            all_windows.append({
                'session': label, 'stage': row['stage'],
                'gt_resp_hz': row['gt_resp_hz'], 'gt_card_hz': row['gt_card_hz'],
                'resp_k_hz': resp_k[_] if isinstance(_, int) else np.nan,
                'card_k_hz': card_k[_] if isinstance(_, int) else np.nan,
            })

        # Plots
        plot_session(df, label, kr_ps, kc_ps, OUT_DIR)
        plot_bland_altman(df, label, kr_ps, kc_ps, OUT_DIR)

    # Build window-level properly (iterrows gives index, not sequential int)
    all_windows = []
    for label, df in session_dfs.items():
        kr = k_values[label]['k_resp']
        kc = k_values[label]['k_card']
        resp_k = df['resp_smooth_hz'].values / kr if kr > 0 else df['resp_smooth_hz'].values
        card_k = df['card_smooth_hz'].values / kc if kc > 0 else df['card_smooth_hz'].values
        for i in range(len(df)):
            all_windows.append({
                'session': label, 'stage': df.iloc[i]['stage'],
                'gt_resp_hz': df.iloc[i]['gt_resp_hz'],
                'gt_card_hz': df.iloc[i]['gt_card_hz'],
                'resp_k_hz': resp_k[i],
                'card_k_hz': card_k[i],
            })

    results_df = pd.DataFrame(all_results)
    windows_df = pd.DataFrame(all_windows)

    # Aggregate summary
    print(f"\n{'='*80}")
    print(f"AGGREGATE (mean ± std across 12 sessions)")
    print(f"{'='*80}")
    for pipe in ['no_k', 'per_session_k', 'loso_k']:
        sub = results_df[results_df['pipeline'] == pipe]
        print(f"\n  {pipe}:")
        print(f"    Resp MAE: {sub['resp_mae'].mean():.2f} ± {sub['resp_mae'].std():.2f} br/min")
        print(f"    Card MAE: {sub['card_mae'].mean():.2f} ± {sub['card_mae'].std():.2f} BPM")
        print(f"    Resp r:   {sub['resp_r'].mean():.3f}    Card r: {sub['card_r'].mean():.3f}")

    # Aggregate bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pipes = ['no_k', 'per_session_k', 'loso_k']
    pipe_labels = {'no_k': 'No k', 'per_session_k': 'Per-session k', 'loso_k': 'LOSO k'}
    pipe_colors = {'no_k': '#95A5A6', 'per_session_k': '#E74C3C', 'loso_k': '#3498DB'}

    for idx, (metric, ylabel) in enumerate([('resp_mae', 'Resp MAE (br/min)'), ('card_mae', 'Card MAE (BPM)')]):
        ax = axes[idx]
        x = np.arange(len(pipes))
        means = [results_df[results_df['pipeline'] == p][metric].mean() for p in pipes]
        stds = [results_df[results_df['pipeline'] == p][metric].std() for p in pipes]
        colors = [pipe_colors[p] for p in pipes]
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([pipe_labels[p] for p in pipes])
        ax.set_ylabel(ylabel)
        ax.set_title(f'{"RESP" if idx == 0 else "CARD"} — best pipeline aggregate')
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{m:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'aggregate_comparison.png')
    plt.close(fig)

    # Per-stage analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (band, gt_col, est_col, unit, ulabel) in enumerate([
        ('resp', 'gt_resp_hz', 'resp_k_hz', 60.0, 'br/min'),
        ('card', 'gt_card_hz', 'card_k_hz', 60.0, 'BPM'),
    ]):
        ax = axes[idx]
        maes = []
        counts = []
        for s in STAGE_ORDER:
            ss = windows_df[(windows_df['stage'] == s)]
            v = np.isfinite(ss[gt_col]) & np.isfinite(ss[est_col]) & (ss[gt_col] > 0)
            if v.sum() >= 10:
                maes.append(np.mean(np.abs(ss.loc[v, est_col] - ss.loc[v, gt_col])) * unit)
                counts.append(v.sum())
            else:
                maes.append(0)
                counts.append(0)

        x = np.arange(len(STAGE_ORDER))
        stage_colors_list = [
            STAGE_COLORS[4], STAGE_COLORS[3], STAGE_COLORS[2],
            STAGE_COLORS[1], STAGE_COLORS[0],
        ]
        bars = ax.bar(x, maes, color=stage_colors_list, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}\n(n={c})' for s, c in zip(STAGE_ORDER, counts)], fontsize=8)
        ax.set_ylabel(f'MAE ({ulabel})')
        ax.set_title(f'{band.upper()} — per-stage MAE (per-session k)')
        for bar, m in zip(bars, maes):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f'{m:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'per_stage_mae.png')
    plt.close(fig)

    # Save CSVs
    results_df.to_csv(OUT_DIR / 'best_pipeline_results.csv', index=False)
    windows_df.to_csv(OUT_DIR / 'best_pipeline_windows.csv', index=False)

    n_plots = len(list(OUT_DIR.glob('*.png')))
    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"  {n_plots} PNGs, 2 CSVs")
    print(f"\nParams: SMOOTH_WIN={SMOOTH_WIN}, KALMAN_R_SCALE={KALMAN_R_SCALE}, KALMAN_Q_SCALE={KALMAN_Q_SCALE}")


if __name__ == '__main__':
    main()
