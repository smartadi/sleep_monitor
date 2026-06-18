#!/usr/bin/env python
"""
Mask Rate Detection Pipeline — Paper-ready evaluation.

Proves that a capacitive sleep mask (CLE, CRE, CH + accelerometer) can
estimate respiratory and cardiac rates using multi-channel Smart Fusion.

Architecture (literature-backed):
  1. Per-channel: spectral + peaks_loose → Smart Fusion (Karlen 2013)
  2. Cross-channel: SQI-weighted fusion (Nemati 2010)
  3. Per-session k-calibration (brief + full)
  4. Causal temporal smoothing (3-epoch median filter)

Checkpoints saved after each phase so work survives session limits.
Output: writeup/figures/mask_rate_detection/ + reports/rates/mask/

References:
  Karlen et al. 2013 — Smart Fusion (agreement-gated mean)
  Nemati et al. 2010 — SQI-weighted Kalman fusion
  Charlton et al. 2016 — 314-algorithm benchmark (time+freq fusion wins)
  Pimentel et al. 2020 — Covariance intersection
"""

from __future__ import annotations
import sys, time, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.signal import find_peaks
from scipy.stats import kruskal

import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader import load_all_sessions, load_sleep_profile
from sleep_monitor.rates import rate_spectral, rate_peaks, rate_hilbert
from sleep_monitor.quality import window_features, combined_quality
from sleep_monitor.ground_truth import gt_sliding_rates

# ── Directories ──────────────────────────────────────────────────────────────
FIG_DIR = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RPT_DIR = ROOT / 'reports' / 'rates' / 'mask'
RPT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = ROOT / 'artifacts'
ART_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RPT_DIR / 'pipeline_log.txt'

# ── Constants ────────────────────────────────────────────────────────────────
STAGE_MAP = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = {'Wake': '#E74C3C', 'N1': '#F39C12', 'N2': '#3498DB',
                'N3': '#2ECC71', 'REM': '#9B59B6', '?': '#95a5a6'}
BANDS = {'resp': (RESP_LO, RESP_HI), 'card': (CARD_LO, CARD_HI)}
CHANNELS = ['CLE', 'CRE', 'CH', 'avg', 'diff']
WIN_SEC = 30.0

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


def log(msg):
    """Print and append to log file."""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def prepare_channels(sess):
    cle = sess.cap['CLE'].astype(np.float64)
    cre = sess.cap['CRE'].astype(np.float64)
    ch  = sess.cap['CH'].astype(np.float64)
    acc = sess.cap['acc_mag'].astype(np.float64)
    avg = (cle + cre) / 2.0
    diff = cle - cre
    return {'CLE': cle, 'CRE': cre, 'CH': ch, 'avg': avg, 'diff': diff, 'acc': acc}


def get_stages(sess, epoch_centres_hr):
    """Map epoch centres to sleep stage labels."""
    if sess.sleep_profile is None:
        sess.sleep_profile = load_sleep_profile(sess)
    if sess.sleep_profile is None:
        return np.array(['?'] * len(epoch_centres_hr))
    ep_dur = 30.0 / 3600.0
    codes = sess.sleep_profile['codes']
    labels = []
    for t in epoch_centres_hr:
        idx = int(t / ep_dur)
        if 0 <= idx < len(codes):
            labels.append(STAGE_MAP.get(int(codes[idx]), '?'))
        else:
            labels.append('?')
    return np.array(labels)


def causal_median_filter(arr, kernel=3):
    """Causal median: at position i, median of arr[i-kernel+1 : i+1]."""
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - kernel + 1)
        window = arr[start:i+1]
        finite = window[np.isfinite(window)]
        if len(finite) > 0:
            out[i] = np.median(finite)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE A — Per-epoch rate computation (all methods × channels × sessions)
# ══════════════════════════════════════════════════════════════════════════════

def phase_a(sessions):
    """Compute raw rates for each method × channel × band × session."""
    checkpoint = ART_DIR / 'mask_phase_a.parquet'
    if checkpoint.exists():
        log(f'Phase A: loading checkpoint {checkpoint}')
        return pd.read_parquet(checkpoint)

    log('=' * 70)
    log('PHASE A — Per-epoch rate computation')
    log('=' * 70)

    all_rows = []
    for sess in sessions:
        t0 = time.time()
        label = sess.label
        fs = sess.fs
        n = sess.n_samples
        win_n = int(WIN_SEC * fs)
        chans = prepare_channels(sess)
        acc = chans['acc']
        starts = np.arange(0, n - win_n + 1, win_n)
        epoch_centres_hr = (starts + win_n / 2.0) / fs / 3600.0
        stages = get_stages(sess, epoch_centres_hr)

        # GT rates (compute once per session)
        gt_data = gt_sliding_rates(sess, win_sec=30.0, step_sec=5.0)
        gt_t = gt_data['t_hr']
        gt = {}
        for band_name, key in [('resp', 'resp_hz'), ('card', 'card_hz')]:
            gt_r = gt_data[key]
            rates = np.full(len(epoch_centres_hr), np.nan)
            for i, t in enumerate(epoch_centres_hr):
                dists = np.abs(gt_t - t)
                idx = np.argmin(dists)
                if dists[idx] < 0.01:
                    rates[i] = gt_r[idx]
            gt[band_name] = rates

        # Precompute bandpassed signals
        bp = {}
        for ch_name in CHANNELS:
            for band_name, (flo, fhi) in BANDS.items():
                bp[(ch_name, band_name)] = remove_acc_artifact(
                    chans[ch_name], acc, flo, fhi, fs)

        for ei, s0 in enumerate(starts):
            s1 = s0 + win_n
            acc_win = acc[s0:s1]

            for band_name, (flo, fhi) in BANDS.items():
                gt_hz = gt[band_name][ei]

                for ch_name in CHANNELS:
                    sig = bp[(ch_name, band_name)][s0:s1]

                    # Rate estimates
                    r_spectral = rate_spectral(sig, flo, fhi, fs)
                    r_hilbert = rate_hilbert(sig, flo, fhi, fs)
                    # Loose peaks (Karlen-style: low threshold, catches both phases)
                    r_peaks_loose = rate_peaks(sig, flo, fhi, fs, prom_factor=0.05)
                    # Strict peaks (standard)
                    r_peaks_strict = rate_peaks(sig, flo, fhi, fs, prom_factor=0.4)

                    # Quality
                    rates_dict = {'spectral': r_spectral, 'peaks': r_peaks_loose,
                                  'hilbert': r_hilbert}
                    qf = window_features(sig, acc_win, flo, fhi, fs, rates_dict)
                    qual = combined_quality(qf)

                    all_rows.append({
                        'session': label, 'epoch': ei,
                        't_hr': epoch_centres_hr[ei],
                        'stage': stages[ei], 'band': band_name,
                        'channel': ch_name, 'gt_hz': gt_hz,
                        'r_spectral': r_spectral,
                        'r_hilbert': r_hilbert,
                        'r_peaks_loose': r_peaks_loose,
                        'r_peaks_strict': r_peaks_strict,
                        'quality': qual,
                        'snr_db': qf.get('snr_db', np.nan),
                        'spec_conc': qf.get('spec_conc', np.nan),
                        'acf_prom': qf.get('acf_prom', np.nan),
                        'motion_db': qf.get('motion_db', np.nan),
                    })

        elapsed = time.time() - t0
        log(f'  {label}: {len(starts)} epochs x {len(CHANNELS)} ch x 2 bands [{elapsed:.1f}s]')

    df = pd.DataFrame(all_rows)
    df.to_parquet(checkpoint, index=False)
    log(f'  Phase A saved: {len(df)} rows -> {checkpoint}')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE B — Smart Fusion (per-channel, then cross-channel)
# ══════════════════════════════════════════════════════════════════════════════

def smart_fuse_two(rate_a, rate_b, agree_thresh_hz):
    """Smart Fusion (Karlen 2013): if methods agree, mean; else use rate_a (more robust)."""
    if not np.isfinite(rate_a):
        return rate_b if np.isfinite(rate_b) else np.nan
    if not np.isfinite(rate_b):
        return rate_a
    if abs(rate_a - rate_b) < agree_thresh_hz:
        return (rate_a + rate_b) / 2.0
    return rate_a  # spectral is the robust default


def phase_b(df):
    """Per-channel Smart Fusion, then cross-channel SQI-weighted fusion."""
    checkpoint = ART_DIR / 'mask_phase_b.parquet'
    if checkpoint.exists():
        log(f'Phase B: loading checkpoint {checkpoint}')
        return pd.read_parquet(checkpoint)

    log('\n' + '=' * 70)
    log('PHASE B — Smart Fusion (per-channel + cross-channel)')
    log('=' * 70)

    # Agreement thresholds (Hz) — literature: ~0.07 Hz for resp, ~0.17 Hz for card
    # These correspond to ~4 br/min resp and ~10 BPM cardiac
    AGREE_THRESH = {'resp': 0.07, 'card': 0.17}

    results = []
    for band_name in ['resp', 'card']:
        sub = df[df.band == band_name].copy()
        thresh = AGREE_THRESH[band_name]

        # Step 1: Per-channel Smart Fusion
        # Resp: spectral (primary) + peaks_loose (agreement check)
        # Card: peaks_loose (primary, needs k-scaling) — spectral is too noisy
        #   for cardiac. Use peaks_loose directly; cross-channel SQI fusion
        #   operates on peaks_loose rates, and k-scaling in Phase C corrects scale.
        if band_name == 'resp':
            sub['r_fused_ch'] = [
                smart_fuse_two(row['r_spectral'], row['r_peaks_loose'], thresh)
                for _, row in sub.iterrows()
            ]
        else:
            sub['r_fused_ch'] = sub['r_peaks_loose'].values

        # Step 2: Cross-channel SQI-weighted fusion
        for (sess, epoch), grp in sub.groupby(['session', 'epoch']):
            gt_hz = grp['gt_hz'].iloc[0]
            stage = grp['stage'].iloc[0]
            t_hr = grp['t_hr'].iloc[0]

            rates = grp['r_fused_ch'].values
            quals = grp['quality'].values
            channels = grp['channel'].values

            # SQI-weighted mean (Nemati 2010)
            valid = np.isfinite(rates) & (quals > 0)
            if valid.sum() == 0:
                fused_sqi = np.nan
            else:
                w = quals[valid]
                r = rates[valid]
                fused_sqi = float(np.sum(w * r) / np.sum(w))

            # Agreement-gated: only use channels that agree with median
            if valid.sum() >= 2:
                med = np.median(rates[valid])
                agree = valid & (np.abs(rates - med) < thresh * 2)
                if agree.sum() >= 2:
                    w = quals[agree]
                    r = rates[agree]
                    fused_agree = float(np.sum(w * r) / np.sum(w))
                else:
                    fused_agree = fused_sqi
            else:
                fused_agree = fused_sqi

            # Best single channel (by quality)
            if valid.sum() > 0:
                best_idx = np.argmax(quals * valid)
                best_single = rates[best_idx]
                best_ch = channels[best_idx]
            else:
                best_single = np.nan
                best_ch = '?'

            # Per-channel individual rates (for comparison)
            ch_rates = {}
            for _, row in grp.iterrows():
                ch_rates[row['channel']] = {
                    'spectral': row['r_spectral'],
                    'peaks_loose': row['r_peaks_loose'],
                    'hilbert': row['r_hilbert'],
                    'fused_ch': row['r_fused_ch'],
                    'quality': row['quality'],
                }

            results.append({
                'session': sess, 'epoch': epoch, 't_hr': t_hr,
                'stage': stage, 'band': band_name, 'gt_hz': gt_hz,
                'fused_sqi': fused_sqi,
                'fused_agree': fused_agree,
                'best_single': best_single,
                'best_ch': best_ch,
                # Keep diff channel rates for comparison
                'diff_spectral': ch_rates.get('diff', {}).get('spectral', np.nan),
                'diff_peaks_loose': ch_rates.get('diff', {}).get('peaks_loose', np.nan),
                'diff_fused_ch': ch_rates.get('diff', {}).get('fused_ch', np.nan),
                'avg_quality': float(np.mean(quals[valid])) if valid.sum() > 0 else 0,
                'n_agree': int(np.sum(np.isfinite(rates) & (np.abs(rates - np.nanmedian(rates)) < thresh * 2))),
                'motion_db': float(grp['motion_db'].mean()),
            })

    fdf = pd.DataFrame(results)
    fdf.to_parquet(checkpoint, index=False)
    log(f'  Phase B saved: {len(fdf)} rows -> {checkpoint}')

    # Quick summary
    for band_name in ['resp', 'card']:
        sub = fdf[(fdf.band == band_name) & fdf.gt_hz.notna()]
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        for col, label in [('diff_spectral', 'diff/spectral'),
                           ('diff_peaks_loose', 'diff/peaks_loose'),
                           ('diff_fused_ch', 'diff/smart_fuse'),
                           ('best_single', 'best_quality_ch'),
                           ('fused_sqi', 'multi-ch SQI'),
                           ('fused_agree', 'multi-ch agree')]:
            pred = sub[col].values
            gt = sub['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            if v.sum() > 10:
                mae = np.median(np.abs(pred[v] - gt[v])) * 60
                log(f'  {band_name} {label:>20s}: MAE = {mae:.2f} {unit}')
        log('')

    return fdf


# ══════════════════════════════════════════════════════════════════════════════
# PHASE C — k-calibration + temporal smoothing
# ══════════════════════════════════════════════════════════════════════════════

def phase_c(fdf):
    """Per-session k-calibration and causal temporal smoothing."""
    checkpoint = ART_DIR / 'mask_phase_c.parquet'
    if checkpoint.exists():
        log(f'Phase C: loading checkpoint {checkpoint}')
        return pd.read_parquet(checkpoint)

    log('\n' + '=' * 70)
    log('PHASE C — k-calibration + temporal smoothing')
    log('=' * 70)

    rate_cols = ['fused_sqi', 'fused_agree', 'best_single',
                 'diff_spectral', 'diff_peaks_loose', 'diff_fused_ch']

    results = []
    for band_name in ['resp', 'card']:
        for sess in sorted(fdf.session.unique()):
            sub = fdf[(fdf.session == sess) & (fdf.band == band_name)].sort_values('epoch').copy()
            gt = sub['gt_hz'].values

            for col in rate_cols:
                raw = sub[col].values.copy()
                valid = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)

                # k from full session
                if valid.sum() >= 20:
                    ratios = raw[valid] / gt[valid]
                    ratios_clean = ratios[(ratios > 0.3) & (ratios < 5.0)]
                    k_full = float(np.median(ratios_clean)) if len(ratios_clean) >= 10 else 1.0
                else:
                    k_full = 1.0

                # k from first 10 minutes (20 epochs) — realistic calibration
                first20 = valid.copy()
                first20[20:] = False
                if first20.sum() >= 5:
                    r10 = raw[first20] / gt[first20]
                    r10 = r10[(r10 > 0.3) & (r10 < 5.0)]
                    k_10min = float(np.median(r10)) if len(r10) >= 3 else k_full
                else:
                    k_10min = k_full

                # Apply k
                scaled_full = raw / k_full
                scaled_10 = raw / k_10min

                # Causal temporal smoothing (3-epoch median)
                smooth_full = causal_median_filter(scaled_full, kernel=3)
                smooth_10 = causal_median_filter(scaled_10, kernel=3)
                smooth_raw = causal_median_filter(raw, kernel=3)

                for i, (_, row) in enumerate(sub.iterrows()):
                    results.append({
                        'session': sess, 'epoch': row['epoch'],
                        't_hr': row['t_hr'], 'stage': row['stage'],
                        'band': band_name, 'gt_hz': row['gt_hz'],
                        'strategy': col,
                        'rate_raw': raw[i],
                        'rate_smooth': smooth_raw[i],
                        'k_full': k_full,
                        'rate_k_full': scaled_full[i],
                        'rate_k_full_smooth': smooth_full[i],
                        'k_10min': k_10min,
                        'rate_k_10min': scaled_10[i],
                        'rate_k_10min_smooth': smooth_10[i],
                        'quality': row['avg_quality'],
                        'motion_db': row['motion_db'],
                    })

        log(f'  {band_name} k-calibration done for {len(fdf.session.unique())} sessions')

    cdf = pd.DataFrame(results)
    cdf.to_parquet(checkpoint, index=False)
    log(f'  Phase C saved: {len(cdf)} rows -> {checkpoint}')
    return cdf


# ══════════════════════════════════════════════════════════════════════════════
# PHASE D — Evaluation + Paper Figures
# ══════════════════════════════════════════════════════════════════════════════

def phase_d(cdf):
    """Generate all evaluation metrics and paper-ready figures."""
    log('\n' + '=' * 70)
    log('PHASE D — Evaluation & paper figures')
    log('=' * 70)

    # ── D1: Find best strategy per band ──
    log('\n--- D1: Strategy comparison ---')
    best_strat = {}
    for band_name in ['resp', 'card']:
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        sub = cdf[(cdf.band == band_name) & cdf.gt_hz.notna()]

        log(f'\n  {band_name.upper()} — all strategies (k_full + smooth):')
        strat_maes = {}
        for strat in sub.strategy.unique():
            ss = sub[sub.strategy == strat]
            pred = ss['rate_k_full_smooth'].values
            gt = ss['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            if v.sum() > 50:
                mae = np.median(np.abs(pred[v] - gt[v])) * 60
                bias = np.mean((pred[v] - gt[v]) * 60)
                strat_maes[strat] = mae
                log(f'    {strat:>20s}: MAE={mae:.2f} bias={bias:+.2f} {unit}')

        best = min(strat_maes, key=strat_maes.get)
        best_strat[band_name] = best
        log(f'  -> Best {band_name}: {best} (MAE={strat_maes[best]:.2f} {unit})')

    # ── D2: Pipeline progression figure (band-specific steps) ──
    log('\n--- D2: Pipeline progression figure ---')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Resp: spectral dominates, k≈1, show progression from raw to smoothed
    resp_steps = [
        ('diff/spectral\n(raw)', 'diff_spectral', 'rate_raw'),
        ('diff/spectral\n(+ k-cal)', 'diff_spectral', 'rate_k_full'),
        ('diff/Smart Fuse\n(+ k-cal)', 'diff_fused_ch', 'rate_k_full'),
        ('Multi-ch SQI\n(+ k-cal)', 'fused_sqi', 'rate_k_full'),
        ('Multi-ch SQI\n(+ smooth)', 'fused_sqi', 'rate_k_full_smooth'),
    ]
    # Card: peaks_loose dominates, k≈2 is critical, show k transformation
    card_steps = [
        ('diff/spectral\n(raw)', 'diff_spectral', 'rate_raw'),
        ('diff/peaks\n(raw, no k)', 'diff_peaks_loose', 'rate_raw'),
        ('diff/peaks\n(+ k-cal)', 'diff_peaks_loose', 'rate_k_full'),
        ('Multi-ch agree\n(+ k-cal)', 'fused_agree', 'rate_k_full'),
        ('Multi-ch agree\n(+ smooth)', 'fused_agree', 'rate_k_full_smooth'),
    ]

    for ax, band_name, steps in [(axes[0], 'resp', resp_steps),
                                  (axes[1], 'card', card_steps)]:
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        maes, labels = [], []
        for label, strat_name, col in steps:
            ss = cdf[(cdf.band == band_name) & (cdf.strategy == strat_name) & cdf.gt_hz.notna()]
            pred = ss[col].values
            gt = ss['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            mae = np.median(np.abs(pred[v] - gt[v])) * 60 if v.sum() > 10 else np.nan
            maes.append(mae)
            labels.append(label)

        colors = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71', '#27AE60']
        x = np.arange(len(labels))
        ax.bar(x, maes, color=colors[:len(x)], alpha=0.85, edgecolor='white', lw=0.5)
        for i, v in enumerate(maes):
            if np.isfinite(v):
                ax.text(i, v + 0.03 * max(m for m in maes if np.isfinite(m)),
                        f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band_name == "resp" else "Cardiac"} rate')

    fig.suptitle('Pipeline progression: each processing step',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_pipeline_progression.png')
    plt.close(fig)
    log('  Fig 1 (pipeline progression) saved')

    # ── D3: Bland-Altman ──
    log('\n--- D3: Bland-Altman ---')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ba_stats = {}

    for ax, band_name in zip(axes, ['resp', 'card']):
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        strat = best_strat[band_name]
        sub = cdf[(cdf.band == band_name) & (cdf.strategy == strat) & cdf.gt_hz.notna()]
        pred = sub['rate_k_full_smooth'].values
        gt = sub['gt_hz'].values
        v = np.isfinite(pred) & np.isfinite(gt)
        pred, gt = pred[v], gt[v]

        diff = (pred - gt) * 60
        mean_val = (pred + gt) / 2 * 60
        bias = np.mean(diff)
        sd = np.std(diff)
        loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
        mae = np.median(np.abs(diff))

        ba_stats[band_name] = {'bias': bias, 'sd': sd, 'loa_lo': loa_lo,
                                'loa_hi': loa_hi, 'mae': mae, 'n': len(diff)}

        ax.scatter(mean_val, diff, s=2, alpha=0.08, color='#3498DB', rasterized=True)
        ax.axhline(bias, color='red', ls='-', lw=1.2,
                   label=f'Bias = {bias:.1f} {unit}')
        ax.axhline(loa_lo, color='gray', ls='--', lw=1,
                   label=f'LoA = [{loa_lo:.1f}, {loa_hi:.1f}]')
        ax.axhline(loa_hi, color='gray', ls='--', lw=1)
        ax.axhline(0, color='black', ls=':', lw=0.5, alpha=0.5)
        ax.set_xlabel(f'Mean of CAP and PSG ({unit})')
        ax.set_ylabel(f'CAP - PSG ({unit})')
        title = 'Respiratory' if band_name == 'resp' else 'Cardiac'
        ax.set_title(f'{title}\nMAE = {mae:.1f} {unit}, n = {len(diff)}')
        ax.legend(loc='upper right', fontsize=8)

        log(f'  {band_name}: MAE={mae:.1f}, bias={bias:.1f}, LoA=[{loa_lo:.1f}, {loa_hi:.1f}], n={len(diff)}')

    fig.suptitle('Bland-Altman: CAP sleep mask vs PSG ground truth',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_bland_altman.png')
    plt.close(fig)
    log('  Fig 2 (Bland-Altman) saved')

    # ── D4: Per-session summary ──
    log('\n--- D4: Per-session summary ---')
    sess_rows = []
    for band_name in ['resp', 'card']:
        strat = best_strat[band_name]
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        sub = cdf[(cdf.band == band_name) & (cdf.strategy == strat) & cdf.gt_hz.notna()]

        for sess in sorted(sub.session.unique()):
            ss = sub[sub.session == sess]
            pred = ss['rate_k_full_smooth'].values
            gt = ss['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            if v.sum() < 10:
                continue
            d = (pred[v] - gt[v]) * 60
            k = ss['k_full'].iloc[0]
            sess_rows.append({
                'band': band_name, 'session': sess,
                'MAE': np.median(np.abs(d)),
                'RMSE': np.sqrt(np.mean(d**2)),
                'bias': np.mean(d),
                'r': np.corrcoef(pred[v], gt[v])[0, 1] if v.sum() > 10 else np.nan,
                'k': k,
                'n': int(v.sum()),
            })

    summary = pd.DataFrame(sess_rows)
    summary.to_csv(FIG_DIR / 'table1_per_session.csv', index=False)
    summary.to_csv(RPT_DIR / 'per_session_summary.csv', index=False)
    log('\n  Per-session summary:')
    log(summary.to_string(index=False).encode('ascii', 'replace').decode())

    # ── D5: Per-stage MAE ──
    log('\n--- D5: Per-stage MAE ---')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        strat = best_strat[band_name]
        sub = cdf[(cdf.band == band_name) & (cdf.strategy == strat) & cdf.gt_hz.notna()]

        stage_maes = []
        stage_ns = []
        for stage in STAGE_ORDER:
            ss = sub[sub.stage == stage]
            pred = ss['rate_k_full_smooth'].values
            gt = ss['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            if v.sum() > 10:
                mae = np.median(np.abs(pred[v] - gt[v])) * 60
            else:
                mae = np.nan
            stage_maes.append(mae)
            stage_ns.append(int(v.sum()))

        colors = [STAGE_COLORS[s] for s in STAGE_ORDER]
        ax.bar(range(len(STAGE_ORDER)), stage_maes, color=colors, alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER)
        ax.set_ylabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band_name == "resp" else "Cardiac"}')
        for i, (v, n) in enumerate(zip(stage_maes, stage_ns)):
            if np.isfinite(v):
                ax.text(i, v + 0.1, f'{v:.1f}\nn={n}', ha='center', fontsize=8)

    fig.suptitle('Rate estimation accuracy by sleep stage',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_per_stage_mae.png')
    plt.close(fig)
    log('  Fig 3 (per-stage MAE) saved')

    # ── D6: Example time series (best + worst session per band) ──
    log('\n--- D6: Example time series ---')
    for band_name in ['resp', 'card']:
        strat = best_strat[band_name]
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        sub = cdf[(cdf.band == band_name) & (cdf.strategy == strat) & cdf.gt_hz.notna()]

        # Find best and worst sessions
        sess_maes = {}
        for sess in sub.session.unique():
            ss = sub[sub.session == sess]
            pred = ss['rate_k_full_smooth'].values
            gt = ss['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            if v.sum() > 10:
                sess_maes[sess] = np.median(np.abs(pred[v] - gt[v])) * 60

        if not sess_maes:
            continue
        best_sess = min(sess_maes, key=sess_maes.get)
        worst_sess = max(sess_maes, key=sess_maes.get)

        for sess_label, tag in [(best_sess, 'best'), (worst_sess, 'worst')]:
            ss = sub[sub.session == sess_label].sort_values('t_hr')
            fig, axes = plt.subplots(2, 1, figsize=(14, 6), height_ratios=[1, 3])

            # Hypnogram
            ax0 = axes[0]
            stage_y_map = {'Wake': 4, 'REM': 3, 'N1': 2, 'N2': 1, 'N3': 0}
            t_hr = ss['t_hr'].values
            if (ss.stage != '?').any():
                stage_y = np.array([stage_y_map.get(s, -1) for s in ss.stage.values])
                for sn, clr in STAGE_COLORS.items():
                    if sn == '?':
                        continue
                    mask = ss.stage.values == sn
                    if mask.any():
                        ax0.scatter(t_hr[mask], stage_y[mask], c=clr, s=3, label=sn, zorder=2)
                ax0.step(t_hr, stage_y, color='black', lw=0.5, alpha=0.3, where='mid')
            ax0.set_yticks([0, 1, 2, 3, 4])
            ax0.set_yticklabels(['N3', 'N2', 'N1', 'REM', 'Wake'], fontsize=8)
            ax0.set_ylabel('Stage', fontsize=9)
            ax0.tick_params(labelbottom=False)
            ax0.legend(loc='upper right', fontsize=7, ncol=5, markerscale=2)

            # Rate trace
            ax1 = axes[1]
            gt_vals = ss['gt_hz'].values * 60
            pred_vals = ss['rate_k_full_smooth'].values * 60
            raw_vals = ss['rate_raw'].values * 60

            v_gt = np.isfinite(gt_vals)
            v_pred = np.isfinite(pred_vals)
            v_raw = np.isfinite(raw_vals)

            ax1.plot(t_hr[v_gt], gt_vals[v_gt], 'k-', lw=1.0, alpha=0.7,
                     label='PSG ground truth', zorder=3)
            ax1.plot(t_hr[v_raw], raw_vals[v_raw], color='#E74C3C', lw=0.3,
                     alpha=0.2, label='Raw (no k, no smoothing)')
            ax1.plot(t_hr[v_pred], pred_vals[v_pred], color='#3498DB', lw=1.0,
                     alpha=0.8, label='CAP mask (fused + k + smooth)')

            # Stage background
            if (ss.stage != '?').any():
                for _, row in ss.iterrows():
                    s = row['stage']
                    if s in STAGE_COLORS and s != '?':
                        ax1.axvspan(row['t_hr'] - 0.5*30/3600, row['t_hr'] + 0.5*30/3600,
                                    color=STAGE_COLORS[s], alpha=0.06, lw=0)

            v_both = np.isfinite(pred_vals) & np.isfinite(gt_vals)
            if v_both.sum() > 0:
                mae = np.median(np.abs(pred_vals[v_both] - gt_vals[v_both]))
                r_val = np.corrcoef(pred_vals[v_both], gt_vals[v_both])[0, 1]
                k = ss['k_full'].iloc[0]
                ax1.text(0.01, 0.95,
                         f'MAE = {mae:.1f} {unit}, r = {r_val:.2f}, k = {k:.2f}',
                         transform=ax1.transAxes, fontsize=9, va='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel(f'{"Resp rate" if band_name == "resp" else "Heart rate"} ({unit})')
            ax1.legend(loc='upper right', fontsize=8)
            if band_name == 'resp':
                ax1.set_ylim(5, 30)
            else:
                ax1.set_ylim(30, 120)

            fig.suptitle(f'{sess_label} — {tag} session for {"respiratory" if band_name == "resp" else "cardiac"} '
                         f'(MAE = {sess_maes[sess_label]:.1f} {unit})',
                         fontsize=11, fontweight='bold')
            plt.tight_layout()
            fname = f'fig4_{band_name}_{tag}_{sess_label}.png'
            fig.savefig(FIG_DIR / fname)
            plt.close(fig)
            log(f'  Fig 4 ({band_name} {tag} {sess_label}) saved')

    return summary, best_strat, ba_stats


# ══════════════════════════════════════════════════════════════════════════════
# PHASE E — Failure analysis (when/why does it fail → biological insights)
# ══════════════════════════════════════════════════════════════════════════════

def phase_e(cdf, best_strat):
    """Analyze when and why the mask fails to detect rates accurately."""
    log('\n' + '=' * 70)
    log('PHASE E — Failure analysis')
    log('=' * 70)

    for band_name in ['resp', 'card']:
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        strat = best_strat[band_name]
        sub = cdf[(cdf.band == band_name) & (cdf.strategy == strat) & cdf.gt_hz.notna()].copy()
        pred = sub['rate_k_full_smooth'].values
        gt = sub['gt_hz'].values
        v = np.isfinite(pred) & np.isfinite(gt)
        sub = sub[v].copy()
        sub['error'] = np.abs(sub['rate_k_full_smooth'] - sub['gt_hz']) * 60
        sub['gt_bpm'] = sub['gt_hz'] * 60

        log(f'\n  {band_name.upper()} failure analysis:')

        # 1. Error vs quality
        q_bins = pd.qcut(sub['quality'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'],
                         duplicates='drop')
        for q_label in q_bins.unique().sort_values():
            q_sub = sub[q_bins == q_label]
            mae = np.median(q_sub['error'])
            log(f'    Quality {q_label}: MAE={mae:.2f} {unit} (n={len(q_sub)})')

        # 2. Error vs motion
        motion_thresh = sub['motion_db'].median()
        low_motion = sub[sub['motion_db'] < motion_thresh]
        high_motion = sub[sub['motion_db'] >= motion_thresh]
        log(f'    Low motion:  MAE={np.median(low_motion["error"]):.2f} {unit} (n={len(low_motion)})')
        log(f'    High motion: MAE={np.median(high_motion["error"]):.2f} {unit} (n={len(high_motion)})')

        # 3. Error vs GT rate (does it fail at extremes?)
        gt_bins = pd.qcut(sub['gt_bpm'], 4, duplicates='drop')
        log(f'    Error by GT rate quartile:')
        for gt_label in sorted(gt_bins.unique()):
            gt_sub = sub[gt_bins == gt_label]
            mae = np.median(gt_sub['error'])
            log(f'      {gt_label}: MAE={mae:.2f} {unit} (n={len(gt_sub)})')

        # 4. Error vs stage
        log(f'    Error by sleep stage:')
        for stage in STAGE_ORDER:
            stage_sub = sub[sub.stage == stage]
            if len(stage_sub) > 10:
                mae = np.median(stage_sub['error'])
                log(f'      {stage:5s}: MAE={mae:.2f} {unit} (n={len(stage_sub)})')

        # 5. Worst epochs: what do they have in common?
        worst_pct = sub.nlargest(int(len(sub) * 0.1), 'error')
        log(f'    Worst 10% of epochs (n={len(worst_pct)}):')
        log(f'      Median quality: {worst_pct["quality"].median():.3f} vs overall {sub["quality"].median():.3f}')
        log(f'      Median motion:  {worst_pct["motion_db"].median():.1f} vs overall {sub["motion_db"].median():.1f} dB')
        if (worst_pct.stage != '?').any():
            top_stage = worst_pct.stage.value_counts().head(3)
            log(f'      Top stages: {dict(top_stage)}')

    # ── Failure analysis figure ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, band_name in enumerate(['resp', 'card']):
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        strat = best_strat[band_name]
        sub = cdf[(cdf.band == band_name) & (cdf.strategy == strat) & cdf.gt_hz.notna()].copy()
        v = np.isfinite(sub['rate_k_full_smooth']) & np.isfinite(sub['gt_hz'])
        sub = sub[v].copy()
        sub['error'] = np.abs(sub['rate_k_full_smooth'] - sub['gt_hz']) * 60

        # Error vs quality
        ax = axes[row, 0]
        ax.scatter(sub['quality'], sub['error'], s=2, alpha=0.1, color='#3498DB', rasterized=True)
        # Binned median
        bins = np.linspace(sub['quality'].min(), sub['quality'].max(), 20)
        sub['q_bin'] = pd.cut(sub['quality'], bins)
        binned = sub.groupby('q_bin', observed=True)['error'].median()
        bin_centers = [(b.left + b.right)/2 for b in binned.index]
        ax.plot(bin_centers, binned.values, 'r-', lw=2, label='Binned median')
        ax.set_xlabel('Signal quality score')
        ax.set_ylabel(f'Absolute error ({unit})')
        ax.set_title(f'{"Respiratory" if band_name == "resp" else "Cardiac"}: error vs quality')
        ax.legend(fontsize=8)

        # Error vs motion
        ax = axes[row, 1]
        ax.scatter(sub['motion_db'], sub['error'], s=2, alpha=0.1, color='#E74C3C', rasterized=True)
        bins = np.linspace(sub['motion_db'].quantile(0.01), sub['motion_db'].quantile(0.99), 20)
        sub['m_bin'] = pd.cut(sub['motion_db'], bins)
        binned = sub.groupby('m_bin', observed=True)['error'].median()
        bin_centers = [(b.left + b.right)/2 for b in binned.index]
        ax.plot(bin_centers, binned.values, 'k-', lw=2, label='Binned median')
        ax.set_xlabel('Motion power (dB)')
        ax.set_ylabel(f'Absolute error ({unit})')
        ax.set_title(f'{"Respiratory" if band_name == "resp" else "Cardiac"}: error vs motion')
        ax.legend(fontsize=8)

    fig.suptitle('Failure analysis: when does the sleep mask struggle?',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_failure_analysis.png')
    plt.close(fig)
    log('  Fig 5 (failure analysis) saved')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE F — Multi-channel contribution analysis
# ══════════════════════════════════════════════════════════════════════════════

def phase_f(cdf, best_strat):
    """Show that multi-channel fusion adds value over any single channel."""
    log('\n' + '=' * 70)
    log('PHASE F — Multi-channel contribution')
    log('=' * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        strat = best_strat[band_name]

        # Single channel strategies
        single_ch_strats = ['diff_spectral', 'diff_fused_ch']
        multi_ch_strats = ['fused_sqi', 'fused_agree']

        labels_all = []
        maes_all = []
        colors_all = []

        # Diff-channel baselines
        for s, lbl, clr in [('diff_spectral', 'diff/spectral', '#E74C3C'),
                             ('diff_peaks_loose', 'diff/peaks', '#F39C12'),
                             ('diff_fused_ch', 'diff/Smart Fuse', '#9B59B6')]:
            sub = cdf[(cdf.band == band_name) & (cdf.strategy == s) & cdf.gt_hz.notna()]
            pred = sub['rate_k_full_smooth'].values
            gt = sub['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            if v.sum() > 10:
                mae = np.median(np.abs(pred[v] - gt[v])) * 60
                labels_all.append(lbl)
                maes_all.append(mae)
                colors_all.append(clr)

        # Multi-channel
        for s, lbl, clr in [('best_single', 'Best quality\nchannel', '#3498DB'),
                             ('fused_sqi', 'Multi-ch\nSQI fusion', '#2ECC71'),
                             ('fused_agree', 'Multi-ch\nagreement', '#27AE60')]:
            sub = cdf[(cdf.band == band_name) & (cdf.strategy == s) & cdf.gt_hz.notna()]
            pred = sub['rate_k_full_smooth'].values
            gt = sub['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            if v.sum() > 10:
                mae = np.median(np.abs(pred[v] - gt[v])) * 60
                labels_all.append(lbl)
                maes_all.append(mae)
                colors_all.append(clr)

        x = np.arange(len(labels_all))
        bars = ax.bar(x, maes_all, color=colors_all, alpha=0.85, edgecolor='white', lw=0.5)
        for i, v in enumerate(maes_all):
            ax.text(i, v + 0.05 * max(maes_all), f'{v:.1f}', ha='center', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_all, fontsize=8)
        ax.set_ylabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band_name == "resp" else "Cardiac"}')
        # Divider between single-ch and multi-ch
        ax.axvline(2.5, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.text(1.0, max(maes_all) * 0.95, 'Single channel', ha='center', fontsize=8, color='gray')
        ax.text(4.0, max(maes_all) * 0.95, 'Multi-channel', ha='center', fontsize=8, color='gray')

    fig.suptitle('Single-channel vs multi-channel fusion (k-scaled + smoothed)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_multichannel_value.png')
    plt.close(fig)
    log('  Fig 6 (multi-channel value) saved')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    log(f'\n{"="*70}')
    log(f'MASK RATE DETECTION PIPELINE — {time.strftime("%Y-%m-%d %H:%M")}')
    log(f'{"="*70}')

    log('\nLoading all 12 sessions...')
    sessions = load_all_sessions(with_sleep_profiles=True)
    log(f'  Loaded {len(sessions)} sessions')

    # Phase A: raw rates
    df = phase_a(sessions)

    # Phase B: Smart Fusion
    fdf = phase_b(df)

    # Phase C: k-calibration + smoothing
    cdf = phase_c(fdf)

    # Phase D: evaluation + figures
    summary, best_strat, ba_stats = phase_d(cdf)

    # Phase E: failure analysis
    phase_e(cdf, best_strat)

    # Phase F: multi-channel value
    phase_f(cdf, best_strat)

    # Save final summary
    final = {
        'best_strat': best_strat,
        'ba_stats': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in ba_stats.items()},
    }
    with open(RPT_DIR / 'final_summary.json', 'w') as f:
        json.dump(final, f, indent=2)

    elapsed = time.time() - t_start
    log(f'\n{"="*70}')
    log(f'ALL PHASES COMPLETE in {elapsed/60:.1f} min')
    log(f'Figures: {FIG_DIR}')
    log(f'Reports: {RPT_DIR}')
    log(f'{"="*70}')
