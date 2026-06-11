#!/usr/bin/env python
"""
Multi-channel fused rate pipeline — 6 phases.

Phase 1: Method benchmark (5 methods × 5 channels × 2 bands × 12 sessions, no k)
Phase 2: Channel confidence fusion (weighted + agreement-filtered)
Phase 3: CWT ridge tracker (cardiac, k-free)
Phase 4: Viterbi temporal smoothing (mandatory post-processing)
Phase 5: Combined evaluation (Bland-Altman, per-stage, comparison)
Phase 6: k-calibration across methods × channels, k-biomarker, session time series

Output: writeup/figures/rate_consolidation/  (paper-ready)
        reports/rates/                        (detailed)
        artifacts/rate_consolidation.parquet   (epoch-level data)
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.stats import kruskal, spearmanr

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader import load_all_sessions
from sleep_monitor.rates import (
    rate_spectral, rate_acf, rate_hilbert, rate_zerocross, rate_peaks,
    rate_envelope, estimate_rate, fuse_rates,
)
from sleep_monitor.rates_classical import rate_cwt, rate_stft_track
from sleep_monitor.quality import window_features, combined_quality
from sleep_monitor.ground_truth import gt_sliding_rates

FIG_DIR = ROOT / 'writeup' / 'figures' / 'rate_consolidation'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RPT_DIR = ROOT / 'reports' / 'rates'
RPT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = ROOT / 'artifacts'
ART_DIR.mkdir(parents=True, exist_ok=True)

STAGE_LABELS = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = {'Wake': '#E74C3C', 'N1': '#F39C12', 'N2': '#3498DB',
                'N3': '#2ECC71', 'REM': '#9B59B6'}

WIN_SEC = 30.0
STEP_SEC = 30.0

BANDS = {
    'resp': (RESP_LO, RESP_HI),
    'card': (CARD_LO, CARD_HI),
}
METHODS = ['spectral', 'acf', 'hilbert', 'zerocross', 'peaks']
CHANNELS = ['CLE', 'CRE', 'CH', 'avg', 'diff']

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_method_fn(method):
    return {
        'spectral': rate_spectral,
        'acf': rate_acf,
        'hilbert': rate_hilbert,
        'zerocross': lambda x, flo, fhi, fs=FS: rate_zerocross(x, fs),
        'peaks': rate_peaks,
        'envelope': rate_envelope,
    }[method]


def prepare_channels(session):
    """Prepare all channel signals (raw, pre-artifact-removal)."""
    cle = session.cap['CLE'].astype(np.float64)
    cre = session.cap['CRE'].astype(np.float64)
    ch  = session.cap['CH'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)
    avg = (cle + cre) / 2.0
    diff = cle - cre
    return {'CLE': cle, 'CRE': cre, 'CH': ch, 'avg': avg, 'diff': diff, 'acc': acc}


def bandpass_channel(sig, acc, f_lo, f_hi, fs):
    """Artifact removal + bandpass."""
    cleaned = remove_acc_artifact(sig, acc, f_lo, f_hi, fs)
    return cleaned


def get_gt_rate_per_epoch(session, band, epoch_centres_hr):
    """Get GT rate (Hz) at each epoch centre using sliding GT."""
    gt = gt_sliding_rates(session, win_sec=30.0, step_sec=5.0)
    if band == 'resp':
        gt_t, gt_r = gt['t_hr'], gt['resp_hz']
    else:
        gt_t, gt_r = gt['t_hr'], gt['card_hz']

    rates = np.full(len(epoch_centres_hr), np.nan)
    for i, t in enumerate(epoch_centres_hr):
        diffs = np.abs(gt_t - t)
        idx = np.argmin(diffs)
        if diffs[idx] < 0.01:
            rates[i] = gt_r[idx]
    return rates


def assign_stages(session, epoch_centres_hr):
    """Map epoch centres to sleep stage labels."""
    profile = session.sleep_profile
    codes = np.full(len(epoch_centres_hr), -1, dtype=int)
    if profile is None:
        return codes, np.array(['?'] * len(epoch_centres_hr))
    epoch_dur_hr = 30.0 / 3600.0
    ep_c = profile['codes']
    for i, t in enumerate(epoch_centres_hr):
        idx = int(t / epoch_dur_hr)
        if 0 <= idx < len(ep_c):
            codes[i] = ep_c[idx]
    labels = np.array([STAGE_LABELS.get(c, '?') for c in codes])
    return codes, labels


def rate_to_display(rate_hz, band):
    """Convert Hz to BPM (cardiac) or br/min (resp)."""
    return rate_hz * 60.0


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Method benchmark
# ══════════════════════════════════════════════════════════════════════════════

def phase1_benchmark(sessions):
    """Run all methods × channels × bands on all sessions. No k-scaling."""
    print('=' * 70)
    print('PHASE 1 — Method benchmark (no k)')
    print('=' * 70)

    all_rows = []
    for si, sess in enumerate(sessions):
        t0 = time.time()
        label = sess.label
        fs = sess.fs
        n = sess.n_samples
        win_n = int(WIN_SEC * fs)
        step_n = int(STEP_SEC * fs)

        chans = prepare_channels(sess)
        acc = chans['acc']
        starts = np.arange(0, n - win_n + 1, step_n)
        epoch_centres_hr = (starts + win_n / 2.0) / fs / 3600.0

        stage_codes, stage_labels = assign_stages(sess, epoch_centres_hr)

        # GT rates for both bands
        gt = {}
        for band_name, (flo, fhi) in BANDS.items():
            gt[band_name] = get_gt_rate_per_epoch(sess, band_name, epoch_centres_hr)

        # Per-channel bandpassed signals (precompute)
        bp_signals = {}
        for ch_name in CHANNELS:
            for band_name, (flo, fhi) in BANDS.items():
                bp_signals[(ch_name, band_name)] = bandpass_channel(
                    chans[ch_name], acc, flo, fhi, fs)

        for ei, s0 in enumerate(starts):
            s1 = s0 + win_n
            acc_win = acc[s0:s1]

            for band_name, (flo, fhi) in BANDS.items():
                gt_hz = gt[band_name][ei]

                for ch_name in CHANNELS:
                    sig_win = bp_signals[(ch_name, band_name)][s0:s1]

                    # Quality features
                    rates_dict = {}
                    for m in METHODS:
                        fn = get_method_fn(m)
                        try:
                            r = fn(sig_win, flo, fhi, fs)
                        except Exception:
                            r = np.nan
                        rates_dict[m] = r

                    qf = window_features(sig_win, acc_win, flo, fhi, fs, rates_dict)
                    qual = combined_quality(qf)

                    row = {
                        'session': label, 'epoch': ei,
                        't_hr': epoch_centres_hr[ei],
                        'stage_code': stage_codes[ei],
                        'stage': stage_labels[ei],
                        'band': band_name,
                        'channel': ch_name,
                        'gt_hz': gt_hz,
                        'quality': qual,
                        'snr_db': qf.get('snr_db', np.nan),
                        'spec_conc': qf.get('spec_conc', np.nan),
                        'acf_prom': qf.get('acf_prom', np.nan),
                    }
                    for m in METHODS:
                        row[f'rate_{m}'] = rates_dict[m]

                    all_rows.append(row)

        elapsed = time.time() - t0
        print(f'  {label}: {len(starts)} epochs x {len(CHANNELS)} channels x 2 bands  [{elapsed:.1f}s]')

    df = pd.DataFrame(all_rows)
    df.to_parquet(ART_DIR / 'rate_consolidation_phase1.parquet', index=False)
    print(f'\n  Phase 1 saved: {len(df)} rows -> artifacts/rate_consolidation_phase1.parquet')
    return df


def phase1_plots(df):
    """Heatmap of MAE per method × channel for each band."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        sub = df[(df.band == band_name) & (df.gt_hz.notna())]
        unit = 'br/min' if band_name == 'resp' else 'BPM'

        mae_grid = np.full((len(METHODS), len(CHANNELS)), np.nan)
        for mi, m in enumerate(METHODS):
            for ci, ch in enumerate(CHANNELS):
                chunk = sub[sub.channel == ch]
                pred = chunk[f'rate_{m}'].values
                gt = chunk['gt_hz'].values
                valid = np.isfinite(pred) & np.isfinite(gt)
                if valid.sum() > 50:
                    mae_grid[mi, ci] = np.nanmedian(np.abs(pred[valid] - gt[valid])) * 60.0

        im = ax.imshow(mae_grid, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(CHANNELS)))
        ax.set_xticklabels(CHANNELS, rotation=30, ha='right')
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS)
        ax.set_title(f'{"Respiratory" if band_name=="resp" else "Cardiac"} — median MAE ({unit})')

        for mi in range(len(METHODS)):
            for ci in range(len(CHANNELS)):
                v = mae_grid[mi, ci]
                if np.isfinite(v):
                    color = 'white' if v > np.nanpercentile(mae_grid[np.isfinite(mae_grid)], 60) else 'black'
                    ax.text(ci, mi, f'{v:.1f}', ha='center', va='center', fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label=f'MAE ({unit})')

    fig.suptitle('Phase 1 — Method x Channel benchmark (no k-scaling, all 12 sessions)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase1_method_channel_heatmap.png')
    fig.savefig(RPT_DIR / 'phase1_method_channel_heatmap.png')
    plt.close(fig)
    print('  Phase 1 heatmap saved')


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Channel confidence fusion
# ══════════════════════════════════════════════════════════════════════════════

def phase2_fusion(df):
    """Confidence-weighted and agreement-filtered channel fusion."""
    print('\n' + '=' * 70)
    print('PHASE 2 — Channel confidence fusion')
    print('=' * 70)

    results = []
    for band_name in ['resp', 'card']:
        sub = df[(df.band == band_name) & df.gt_hz.notna()]

        # Determine best single method from Phase 1
        best_method = None
        best_mae = np.inf
        for m in METHODS:
            pred = sub[f'rate_{m}'].values
            gt = sub['gt_hz'].values
            valid = np.isfinite(pred) & np.isfinite(gt)
            if valid.sum() > 100:
                mae = np.nanmedian(np.abs(pred[valid] - gt[valid]))
                if mae < best_mae:
                    best_mae = mae
                    best_method = m
        print(f'  {band_name}: best single method = {best_method} (MAE {best_mae*60:.2f})')

        # Group by session + epoch
        for (sess, epoch), grp in sub.groupby(['session', 'epoch']):
            gt_hz = grp.gt_hz.iloc[0]
            stage = grp.stage.iloc[0]
            t_hr = grp.t_hr.iloc[0]

            # Per-channel rates and quality
            ch_rates = {}
            ch_quality = {}
            ch_agreement = {}
            for _, row in grp.iterrows():
                ch = row['channel']
                ch_rates[ch] = row[f'rate_{best_method}']
                ch_quality[ch] = row['quality']
                # Cross-method agreement for this channel
                method_rates = {m: row[f'rate_{m}'] for m in METHODS}
                vals = [v for v in method_rates.values() if np.isfinite(v)]
                ch_agreement[ch] = np.std(vals) if len(vals) >= 2 else np.inf

            # Strategy A: Confidence-weighted fusion
            weights = []
            rates = []
            for ch in CHANNELS:
                r = ch_rates.get(ch, np.nan)
                q = ch_quality.get(ch, 0)
                if np.isfinite(r) and q > 0:
                    weights.append(q)
                    rates.append(r)
            if weights:
                w = np.array(weights)
                r = np.array(rates)
                fused_weighted = float(np.sum(w * r) / np.sum(w))
            else:
                fused_weighted = np.nan

            # Strategy B: Agreement-filtered then confidence-weighted
            flo, fhi = BANDS[band_name]
            agree_thresh = 0.15 if band_name == 'card' else 0.05
            weights_b = []
            rates_b = []
            for ch in CHANNELS:
                r = ch_rates.get(ch, np.nan)
                q = ch_quality.get(ch, 0)
                a = ch_agreement.get(ch, np.inf)
                if np.isfinite(r) and q > 0 and a < agree_thresh:
                    weights_b.append(q)
                    rates_b.append(r)
            if weights_b:
                w = np.array(weights_b)
                r = np.array(rates_b)
                fused_agreement = float(np.sum(w * r) / np.sum(w))
            else:
                fused_agreement = fused_weighted  # fallback

            # Best single fixed channel (from Phase 1)
            best_fixed_ch = None
            best_fixed_mae = np.inf
            for ch in CHANNELS:
                ch_sub = sub[sub.channel == ch]
                pred = ch_sub[f'rate_{best_method}'].values
                gt_arr = ch_sub['gt_hz'].values
                valid = np.isfinite(pred) & np.isfinite(gt_arr)
                if valid.sum() > 50:
                    m = np.nanmedian(np.abs(pred[valid] - gt_arr[valid]))
                    if m < best_fixed_mae:
                        best_fixed_mae = m
                        best_fixed_ch = ch

            # Oracle: best channel for this epoch
            oracle_rate = np.nan
            oracle_err = np.inf
            for ch in CHANNELS:
                r = ch_rates.get(ch, np.nan)
                if np.isfinite(r) and np.isfinite(gt_hz):
                    err = abs(r - gt_hz)
                    if err < oracle_err:
                        oracle_err = err
                        oracle_rate = r

            results.append({
                'session': sess, 'epoch': epoch, 't_hr': t_hr,
                'stage': stage, 'band': band_name, 'gt_hz': gt_hz,
                'best_fixed': ch_rates.get(best_fixed_ch, np.nan) if best_fixed_ch else np.nan,
                'best_fixed_ch': best_fixed_ch,
                'fused_weighted': fused_weighted,
                'fused_agreement': fused_agreement,
                'oracle': oracle_rate,
                'best_method': best_method,
            })

    fdf = pd.DataFrame(results)
    fdf.to_parquet(ART_DIR / 'rate_consolidation_phase2.parquet', index=False)
    print(f'  Phase 2 saved: {len(fdf)} rows')
    return fdf


def phase2_plots(fdf):
    """Bar chart: MAE comparison of fusion strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, band in zip(axes, ['resp', 'card']):
        sub = fdf[(fdf.band == band) & fdf.gt_hz.notna()]
        unit = 'br/min' if band == 'resp' else 'BPM'

        strategies = {
            'Best fixed\nchannel': 'best_fixed',
            'Confidence-\nweighted': 'fused_weighted',
            'Agreement-\nfiltered': 'fused_agreement',
            'Oracle\n(hindsight)': 'oracle',
        }

        maes = []
        labels = []
        for lbl, col in strategies.items():
            pred = sub[col].values
            gt = sub['gt_hz'].values
            valid = np.isfinite(pred) & np.isfinite(gt)
            if valid.sum() > 0:
                mae = np.median(np.abs(pred[valid] - gt[valid])) * 60.0
            else:
                mae = np.nan
            maes.append(mae)
            labels.append(lbl)

        colors = ['#95a5a6', '#3498DB', '#2ECC71', '#F39C12']
        x = np.arange(len(labels))
        bars = ax.bar(x, maes, color=colors, alpha=0.85, edgecolor='white', lw=0.5)
        for i, v in enumerate(maes):
            if np.isfinite(v):
                ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band=="resp" else "Cardiac"}')

    fig.suptitle('Phase 2 — Channel fusion strategies vs fixed channel and oracle',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase2_fusion_comparison.png')
    fig.savefig(RPT_DIR / 'phase2_fusion_comparison.png')
    plt.close(fig)
    print('  Phase 2 bar chart saved')


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — CWT ridge (cardiac focus)
# ══════════════════════════════════════════════════════════════════════════════

def phase3_cwt_ridge(df, sessions):
    """Run CWT on cardiac band for all channels, compare to base methods from Phase 1."""
    print('\n' + '=' * 70)
    print('PHASE 3 — CWT ridge tracker (cardiac)')
    print('=' * 70)

    flo, fhi = BANDS['card']

    # Compute CWT rates per session × channel (not done in Phase 1)
    cwt_rows = []
    for sess in sessions:
        label = sess.label
        fs = sess.fs
        n = sess.n_samples
        win_n = int(WIN_SEC * fs)
        step_n = int(STEP_SEC * fs)
        chans = prepare_channels(sess)
        acc = chans['acc']
        starts = np.arange(0, n - win_n + 1, step_n)

        for ch_name in CHANNELS:
            bp_sig = bandpass_channel(chans[ch_name], acc, flo, fhi, fs)
            for ei, s0 in enumerate(starts):
                sig_win = bp_sig[s0:s0 + win_n]
                try:
                    r = rate_cwt(sig_win, flo, fhi, fs, n_scales=32)
                except Exception:
                    r = np.nan
                cwt_rows.append({'session': label, 'epoch': ei, 'channel': ch_name, 'rate_cwt': r})
        print(f'  CWT {label} done')

    cwt_df = pd.DataFrame(cwt_rows)

    # Merge CWT with Phase 1 cardiac data
    card = df[(df.band == 'card') & df.gt_hz.notna()]
    merged = card.merge(cwt_df, on=['session', 'epoch', 'channel'], how='left')

    results = []
    methods_eval = ['hilbert', 'spectral', 'peaks', 'cwt']
    for ch in CHANNELS:
        ch_sub = merged[merged.channel == ch]
        for m in methods_eval:
            col = f'rate_{m}'
            if col not in ch_sub.columns:
                continue
            pred = ch_sub[col].values
            gt = ch_sub['gt_hz'].values
            valid = np.isfinite(pred) & np.isfinite(gt)
            if valid.sum() < 50:
                continue
            errors = (pred[valid] - gt[valid]) * 60.0
            results.append({
                'channel': ch, 'method': m,
                'MAE_BPM': np.median(np.abs(errors)),
                'bias_BPM': np.median(errors),
                'n': int(valid.sum()),
                'pct_valid': valid.mean() * 100,
            })

    rdf = pd.DataFrame(results)
    rdf.to_csv(RPT_DIR / 'phase3_cwt_cardiac_comparison.csv', index=False)
    print(rdf.to_string(index=False).encode('ascii', 'replace').decode())
    return rdf


def phase3_plots(rdf):
    """CWT vs other methods on cardiac band."""
    fig, ax = plt.subplots(figsize=(10, 5))

    methods_show = ['hilbert', 'cwt', 'spectral', 'peaks']
    method_colors = {'hilbert': '#9B59B6', 'cwt': '#E74C3C', 'spectral': '#3498DB', 'peaks': '#2ECC71'}

    x = np.arange(len(CHANNELS))
    width = 0.18
    for mi, m in enumerate(methods_show):
        vals = []
        for ch in CHANNELS:
            row = rdf[(rdf.channel == ch) & (rdf.method == m)]
            vals.append(row['MAE_BPM'].values[0] if len(row) > 0 else np.nan)
        offset = (mi - len(methods_show)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=method_colors[m], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=20, ha='right')
    ax.set_ylabel('Median MAE (BPM)')
    ax.set_title('Phase 3 — Cardiac rate: CWT ridge vs other methods (no k-scaling)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase3_cwt_cardiac.png')
    fig.savefig(RPT_DIR / 'phase3_cwt_cardiac.png')
    plt.close(fig)
    print('  Phase 3 cardiac comparison saved')


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Viterbi temporal smoothing
# ══════════════════════════════════════════════════════════════════════════════

def viterbi_smooth(rate_series, max_jump_hz_per_epoch):
    """
    Viterbi-style temporal smoothing on a 1-D rate time series.

    Finds the path through the observed rates that minimises
    sum of (deviation from observation) + (transition penalty for jumps).
    """
    rates = np.array(rate_series, dtype=float)
    n = len(rates)
    if n < 3:
        return rates.copy()

    # Build candidate set: at each epoch, consider the observed rate ± some offsets
    n_candidates = 21
    offsets = np.linspace(-max_jump_hz_per_epoch * 3, max_jump_hz_per_epoch * 3, n_candidates)

    # For NaN observations, use interpolated value
    valid_mask = np.isfinite(rates)
    if valid_mask.sum() < 2:
        return rates.copy()
    interp_rates = np.interp(np.arange(n),
                             np.where(valid_mask)[0],
                             rates[valid_mask])

    # Build grid
    candidates = np.zeros((n, n_candidates))
    for t in range(n):
        base = interp_rates[t] if not np.isfinite(rates[t]) else rates[t]
        candidates[t] = base + offsets

    # Viterbi DP
    obs_weight = 2.0
    cost = np.full((n, n_candidates), np.inf)
    back = np.zeros((n, n_candidates), dtype=int)

    for j in range(n_candidates):
        if np.isfinite(rates[0]):
            cost[0, j] = obs_weight * abs(candidates[0, j] - rates[0])
        else:
            cost[0, j] = 0

    for t in range(1, n):
        for j in range(n_candidates):
            obs_cost = obs_weight * abs(candidates[t, j] - rates[t]) if np.isfinite(rates[t]) else 0

            best_prev = np.inf
            best_k = 0
            for k in range(n_candidates):
                jump = abs(candidates[t, j] - candidates[t-1, k])
                trans_cost = (jump / max_jump_hz_per_epoch) ** 2 if jump > max_jump_hz_per_epoch else 0
                total = cost[t-1, k] + trans_cost
                if total < best_prev:
                    best_prev = total
                    best_k = k

            cost[t, j] = best_prev + obs_cost
            back[t, j] = best_k

    # Backtrace
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmin(cost[-1])
    for t in range(n-2, -1, -1):
        path[t] = back[t+1, path[t+1]]

    smoothed = np.array([candidates[t, path[t]] for t in range(n)])
    return smoothed


def phase4_viterbi(fdf):
    """Apply Viterbi smoothing to all fusion strategies."""
    print('\n' + '=' * 70)
    print('PHASE 4 — Viterbi temporal smoothing')
    print('=' * 70)

    results = []
    rate_cols = ['best_fixed', 'fused_weighted', 'fused_agreement', 'oracle']

    for band in ['resp', 'card']:
        max_jump = 2.0/60.0 if band == 'resp' else 5.0/60.0  # Hz per epoch

        for sess in fdf.session.unique():
            sub = fdf[(fdf.session == sess) & (fdf.band == band)].sort_values('epoch')
            if len(sub) < 5:
                continue

            for col in rate_cols:
                raw = sub[col].values.copy()
                smoothed = viterbi_smooth(raw, max_jump)

                for i, (_, row) in enumerate(sub.iterrows()):
                    results.append({
                        'session': row['session'], 'epoch': row['epoch'],
                        't_hr': row['t_hr'], 'stage': row['stage'],
                        'band': band, 'gt_hz': row['gt_hz'],
                        'strategy': col,
                        'rate_raw': raw[i],
                        'rate_smoothed': smoothed[i],
                    })

    vdf = pd.DataFrame(results)
    vdf.to_parquet(ART_DIR / 'rate_consolidation_phase4.parquet', index=False)
    print(f'  Phase 4 saved: {len(vdf)} rows')
    return vdf


def phase4_plots(vdf):
    """Before/after Viterbi: MAE comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, band in zip(axes, ['resp', 'card']):
        sub = vdf[(vdf.band == band) & vdf.gt_hz.notna()]
        unit = 'br/min' if band == 'resp' else 'BPM'

        strategies = ['best_fixed', 'fused_weighted', 'fused_agreement']
        strat_labels = ['Best fixed', 'Conf-weighted', 'Agree-filtered']

        raw_maes, smooth_maes = [], []
        for s in strategies:
            ss = sub[sub.strategy == s]
            valid_raw = np.isfinite(ss.rate_raw) & np.isfinite(ss.gt_hz)
            valid_sm = np.isfinite(ss.rate_smoothed) & np.isfinite(ss.gt_hz)
            raw_maes.append(np.median(np.abs(ss.rate_raw[valid_raw].values - ss.gt_hz[valid_raw].values)) * 60)
            smooth_maes.append(np.median(np.abs(ss.rate_smoothed[valid_sm].values - ss.gt_hz[valid_sm].values)) * 60)

        x = np.arange(len(strategies))
        width = 0.35
        ax.bar(x - width/2, raw_maes, width, label='Raw', color='#E74C3C', alpha=0.7)
        ax.bar(x + width/2, smooth_maes, width, label='+ Viterbi smoothing', color='#2ECC71', alpha=0.7)

        for i in range(len(strategies)):
            ax.text(i - width/2, raw_maes[i] + 0.1, f'{raw_maes[i]:.1f}', ha='center', fontsize=8)
            ax.text(i + width/2, smooth_maes[i] + 0.1, f'{smooth_maes[i]:.1f}', ha='center', fontsize=8)
            if raw_maes[i] > 0:
                pct = (raw_maes[i] - smooth_maes[i]) / raw_maes[i] * 100
                ax.text(i, max(raw_maes[i], smooth_maes[i]) + 0.5, f'-{pct:.0f}%',
                        ha='center', fontsize=8, color='green', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(strat_labels)
        ax.set_ylabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band=="resp" else "Cardiac"}')
        ax.legend()

    fig.suptitle('Phase 4 — Viterbi temporal smoothing reduces jitter',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase4_viterbi_improvement.png')
    fig.savefig(RPT_DIR / 'phase4_viterbi_improvement.png')
    plt.close(fig)
    print('  Phase 4 improvement chart saved')


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Combined evaluation
# ══════════════════════════════════════════════════════════════════════════════

def phase5_evaluation(vdf):
    """Bland-Altman, per-stage, per-session summary."""
    print('\n' + '=' * 70)
    print('PHASE 5 — Combined evaluation')
    print('=' * 70)

    # Use best fusion strategy + Viterbi
    best_strat = 'fused_weighted'

    # ── Bland-Altman ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, band in zip(axes, ['resp', 'card']):
        sub = vdf[(vdf.band == band) & (vdf.strategy == best_strat) & vdf.gt_hz.notna()]
        pred = sub.rate_smoothed.values
        gt = sub.gt_hz.values
        valid = np.isfinite(pred) & np.isfinite(gt)
        pred, gt = pred[valid], gt[valid]

        unit = 'br/min' if band == 'resp' else 'BPM'
        scale = 60.0
        diff = (pred - gt) * scale
        mean_val = (pred + gt) / 2 * scale

        bias = np.mean(diff)
        sd = np.std(diff)
        loa_lo, loa_hi = bias - 1.96*sd, bias + 1.96*sd

        ax.scatter(mean_val, diff, s=2, alpha=0.1, color='#3498DB', rasterized=True)
        ax.axhline(bias, color='red', ls='-', lw=1.2, label=f'Bias = {bias:.1f}')
        ax.axhline(loa_lo, color='gray', ls='--', lw=1, label=f'LoA = [{loa_lo:.1f}, {loa_hi:.1f}]')
        ax.axhline(loa_hi, color='gray', ls='--', lw=1)
        ax.axhline(0, color='black', ls=':', lw=0.5, alpha=0.5)
        ax.set_xlabel(f'Mean of CAP and PSG ({unit})')
        ax.set_ylabel(f'CAP - PSG ({unit})')
        ax.set_title(f'{"Respiratory" if band=="resp" else "Cardiac"}')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Phase 5 — Bland-Altman: fused multi-channel + Viterbi vs PSG',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase5_bland_altman.png')
    fig.savefig(RPT_DIR / 'phase5_bland_altman.png')
    plt.close(fig)
    print('  Bland-Altman saved')

    # ── Per-stage MAE ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, band in zip(axes, ['resp', 'card']):
        sub = vdf[(vdf.band == band) & (vdf.strategy == best_strat) & vdf.gt_hz.notna()]
        unit = 'br/min' if band == 'resp' else 'BPM'

        stage_maes = []
        stage_ns = []
        for stage in STAGE_ORDER:
            ss = sub[sub.stage == stage]
            valid = np.isfinite(ss.rate_smoothed) & np.isfinite(ss.gt_hz)
            if valid.sum() > 10:
                mae = np.median(np.abs(ss.rate_smoothed[valid].values - ss.gt_hz[valid].values)) * 60
            else:
                mae = np.nan
            stage_maes.append(mae)
            stage_ns.append(int(valid.sum()))

        colors = [STAGE_COLORS[s] for s in STAGE_ORDER]
        bars = ax.bar(range(len(STAGE_ORDER)), stage_maes, color=colors, alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER)
        ax.set_ylabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band=="resp" else "Cardiac"}')
        for i, (v, n) in enumerate(zip(stage_maes, stage_ns)):
            if np.isfinite(v):
                ax.text(i, v + 0.1, f'{v:.1f}\nn={n}', ha='center', fontsize=8)

    fig.suptitle('Phase 5 — Accuracy by sleep stage (fused + Viterbi)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase5_per_stage_mae.png')
    fig.savefig(RPT_DIR / 'phase5_per_stage_mae.png')
    plt.close(fig)
    print('  Per-stage MAE saved')

    # ── Per-session summary table ──
    rows = []
    for band in ['resp', 'card']:
        sub = vdf[(vdf.band == band) & (vdf.strategy == best_strat) & vdf.gt_hz.notna()]
        unit = 'br/min' if band == 'resp' else 'BPM'

        for sess in sorted(sub.session.unique()):
            ss = sub[sub.session == sess]
            valid = np.isfinite(ss.rate_smoothed) & np.isfinite(ss.gt_hz)
            pred = ss.rate_smoothed[valid].values
            gt = ss.gt_hz[valid].values
            diff = (pred - gt) * 60
            rows.append({
                'band': band, 'session': sess,
                'MAE': np.median(np.abs(diff)),
                'RMSE': np.sqrt(np.mean(diff**2)),
                'bias': np.mean(diff),
                'r': np.corrcoef(pred, gt)[0,1] if len(pred) > 10 else np.nan,
                'n_epochs': int(valid.sum()),
                'coverage': f'{valid.mean():.0%}',
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(FIG_DIR / 'phase5_per_session_summary.csv', index=False)
    summary.to_csv(RPT_DIR / 'phase5_per_session_summary.csv', index=False)
    print('\n  Per-session summary:')
    print(summary.to_string(index=False).encode('ascii', 'replace').decode())

    # ── Overall pipeline comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, band in zip(axes, ['resp', 'card']):
        sub_all = vdf[(vdf.band == band) & vdf.gt_hz.notna()]
        unit = 'br/min' if band == 'resp' else 'BPM'

        pipeline_maes = {}
        for strat in ['best_fixed', 'fused_weighted', 'fused_agreement']:
            for smooth_label, col in [('raw', 'rate_raw'), ('+ Viterbi', 'rate_smoothed')]:
                ss = sub_all[sub_all.strategy == strat]
                valid = np.isfinite(ss[col]) & np.isfinite(ss.gt_hz)
                mae = np.median(np.abs(ss[col][valid].values - ss.gt_hz[valid].values)) * 60
                key = f'{strat}\n{smooth_label}'
                pipeline_maes[key] = mae

        labels = list(pipeline_maes.keys())
        vals = list(pipeline_maes.values())
        colors_list = ['#E74C3C', '#27AE60'] * 3
        ax.barh(range(len(labels)), vals, color=colors_list, alpha=0.8, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band=="resp" else "Cardiac"}')
        for i, v in enumerate(vals):
            ax.text(v + 0.05, i, f'{v:.1f}', va='center', fontsize=8)
        ax.invert_yaxis()

    fig.suptitle('Phase 5 — Full pipeline comparison: all strategies x smoothing',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase5_pipeline_comparison.png')
    fig.savefig(RPT_DIR / 'phase5_pipeline_comparison.png')
    plt.close(fig)
    print('  Pipeline comparison saved')

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 — k-calibration, k-biomarker, session time series
# ══════════════════════════════════════════════════════════════════════════════

def phase6_k_calibration(df):
    """Compute per-session k for every method × channel × band, evaluate k-scaled accuracy."""
    print('\n' + '=' * 70)
    print('PHASE 6a — k-calibration across methods × channels')
    print('=' * 70)

    all_methods = METHODS
    k_rows = []
    scaled_rows = []

    for band_name in ['resp', 'card']:
        unit = 'br/min' if band_name == 'resp' else 'BPM'

        for sess in sorted(df.session.unique()):
            for ch in CHANNELS:
                sub = df[(df.session == sess) & (df.channel == ch) &
                         (df.band == band_name) & df.gt_hz.notna()]
                if len(sub) < 20:
                    continue

                for m in all_methods:
                    pred = sub[f'rate_{m}'].values
                    gt = sub['gt_hz'].values
                    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
                    if valid.sum() < 20:
                        continue

                    ratios = pred[valid] / gt[valid]
                    ratios = ratios[(ratios > 0.3) & (ratios < 5.0)]
                    if len(ratios) < 10:
                        continue
                    k = float(np.median(ratios))

                    # k-scaled rate
                    scaled = pred / k
                    errors = (scaled[valid] - gt[valid]) * 60.0
                    mae = np.median(np.abs(errors))
                    bias = np.mean(errors)
                    if valid.sum() > 10:
                        r_corr = np.corrcoef(scaled[valid], gt[valid])[0, 1]
                    else:
                        r_corr = np.nan

                    k_rows.append({
                        'band': band_name, 'session': sess,
                        'channel': ch, 'method': m,
                        'k': k, 'n_valid': int(valid.sum()),
                    })
                    scaled_rows.append({
                        'band': band_name, 'session': sess,
                        'channel': ch, 'method': m, 'k': k,
                        'MAE': mae, 'bias': bias, 'r': r_corr,
                    })

    kdf = pd.DataFrame(k_rows)
    sdf = pd.DataFrame(scaled_rows)
    kdf.to_csv(RPT_DIR / 'phase6_k_per_session.csv', index=False)
    sdf.to_csv(RPT_DIR / 'phase6_k_scaled_accuracy.csv', index=False)

    # Find best method × channel per band (lowest pooled MAE)
    best = {}
    for band_name in ['resp', 'card']:
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        sub = sdf[sdf.band == band_name]
        agg = sub.groupby(['method', 'channel']).agg(
            median_MAE=('MAE', 'median'),
            mean_MAE=('MAE', 'mean'),
            mean_r=('r', 'mean'),
            median_k=('k', 'median'),
        ).reset_index().sort_values('median_MAE')
        print(f'\n  {band_name.upper()} — top 5 method×channel (k-scaled, median MAE in {unit}):')
        print(agg.head(5).to_string(index=False).encode('ascii', 'replace').decode())
        best[band_name] = (agg.iloc[0]['method'], agg.iloc[0]['channel'])

    print(f'\n  Best resp: {best["resp"][0]} on {best["resp"][1]}')
    print(f'  Best card: {best["card"][0]} on {best["card"][1]}')

    return kdf, sdf, best


def phase6_k_heatmap(sdf):
    """Heatmap of k-scaled MAE per method × channel (mirrors Phase 1 heatmap)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        unit = 'br/min' if band_name == 'resp' else 'BPM'
        sub = sdf[sdf.band == band_name]

        mae_grid = np.full((len(METHODS), len(CHANNELS)), np.nan)
        for mi, m in enumerate(METHODS):
            for ci, ch in enumerate(CHANNELS):
                chunk = sub[(sub.method == m) & (sub.channel == ch)]
                if len(chunk) > 0:
                    mae_grid[mi, ci] = chunk['MAE'].median()

        im = ax.imshow(mae_grid, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(CHANNELS)))
        ax.set_xticklabels(CHANNELS, rotation=30, ha='right')
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS)
        ax.set_title(f'{"Respiratory" if band_name=="resp" else "Cardiac"} — median MAE ({unit}, k-scaled)')

        for mi in range(len(METHODS)):
            for ci in range(len(CHANNELS)):
                v = mae_grid[mi, ci]
                if np.isfinite(v):
                    color = 'white' if v > np.nanpercentile(mae_grid[np.isfinite(mae_grid)], 60) else 'black'
                    ax.text(ci, mi, f'{v:.1f}', ha='center', va='center', fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label=f'MAE ({unit})')

    fig.suptitle('Phase 6 — Method x Channel benchmark (k-scaled, all 12 sessions)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase6_k_scaled_heatmap.png')
    fig.savefig(RPT_DIR / 'phase6_k_scaled_heatmap.png')
    plt.close(fig)
    print('  Phase 6 k-scaled heatmap saved')


def phase6_k_biomarker(df, kdf, best):
    """k(t) biomarker analysis: stage dependence, within-night dynamics."""
    print('\n' + '=' * 70)
    print('PHASE 6b — k-biomarker analysis')
    print('=' * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        m, ch = best[band_name]
        unit = 'br/min' if band_name == 'resp' else 'BPM'

        sub = df[(df.band == band_name) & (df.channel == ch) &
                 df.gt_hz.notna() & (df.stage != '?')]
        pred = sub[f'rate_{m}'].values
        gt = sub['gt_hz'].values
        valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)

        k_per_window = np.full(len(sub), np.nan)
        k_per_window[valid] = pred[valid] / gt[valid]
        sub = sub.copy()
        sub['k_window'] = k_per_window

        stage_data = []
        stage_labels_plot = []
        for stage in STAGE_ORDER:
            vals = sub.loc[sub.stage == stage, 'k_window'].dropna()
            vals = vals[(vals > 0.3) & (vals < 5.0)]
            if len(vals) >= 10:
                stage_data.append(vals.values)
                stage_labels_plot.append(stage)

        if len(stage_data) >= 3:
            bp = ax.boxplot(stage_data, tick_labels=stage_labels_plot, patch_artist=True,
                           medianprops={'color': 'black', 'lw': 1.5},
                           flierprops={'markersize': 2, 'alpha': 0.3})
            for patch, sl in zip(bp['boxes'], stage_labels_plot):
                patch.set_facecolor(STAGE_COLORS.get(sl, '#aaa'))
                patch.set_alpha(0.7)

            # Kruskal-Wallis test
            try:
                h_stat, p_val = kruskal(*stage_data)
                ax.set_title(f'{"Resp" if band_name=="resp" else "Card"} k ({m}/{ch})\n'
                            f'KW H={h_stat:.0f}, p={p_val:.1e}')
            except Exception:
                ax.set_title(f'{"Resp" if band_name=="resp" else "Card"} k ({m}/{ch})')

            # Print medians
            print(f'\n  {band_name.upper()} k by stage ({m}/{ch}):')
            for sl, sd in zip(stage_labels_plot, stage_data):
                print(f'    {sl:5s}: median={np.median(sd):.3f}, IQR=[{np.percentile(sd,25):.3f}, {np.percentile(sd,75):.3f}], n={len(sd)}')

        ax.set_ylabel(f'k = rate_CAP / rate_GT')
        ax.axhline(1.0, color='gray', ls=':', lw=0.5)

    fig.suptitle('Phase 6 — k by sleep stage (best method × channel per band)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase6_k_by_stage.png')
    fig.savefig(RPT_DIR / 'phase6_k_by_stage.png')
    plt.close(fig)
    print('  k by stage boxplots saved')

    # ── k stability: night-to-night, cross-channel ──
    print('\n  k stability across sessions:')
    for band_name in ['resp', 'card']:
        m, ch = best[band_name]
        sub = kdf[(kdf.band == band_name) & (kdf.method == m) & (kdf.channel == ch)]
        if len(sub) > 0:
            k_vals = sub['k'].values
            print(f'    {band_name} ({m}/{ch}): median={np.median(k_vals):.3f}, '
                  f'range=[{np.min(k_vals):.3f}, {np.max(k_vals):.3f}], '
                  f'IQR=[{np.percentile(k_vals,25):.3f}, {np.percentile(k_vals,75):.3f}]')

    # ── k cross-channel comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, band_name in zip(axes, ['resp', 'card']):
        m, _ = best[band_name]
        sub = kdf[(kdf.band == band_name) & (kdf.method == m)]

        ch_data = []
        ch_labels = []
        for ch in CHANNELS:
            vals = sub.loc[sub.channel == ch, 'k'].values
            if len(vals) >= 3:
                ch_data.append(vals)
                ch_labels.append(ch)

        if ch_data:
            bp = ax.boxplot(ch_data, tick_labels=ch_labels, patch_artist=True,
                           medianprops={'color': 'black', 'lw': 1.5})
            for patch in bp['boxes']:
                patch.set_facecolor('#3498DB')
                patch.set_alpha(0.6)
        ax.set_ylabel('k (per-session)')
        ax.set_title(f'{"Resp" if band_name=="resp" else "Card"} k across channels ({m})')
        ax.axhline(1.0, color='gray', ls=':', lw=0.5)

    fig.suptitle('Phase 6 — k varies across channels (same method)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase6_k_cross_channel.png')
    fig.savefig(RPT_DIR / 'phase6_k_cross_channel.png')
    plt.close(fig)
    print('  k cross-channel boxplots saved')


def phase6_session_timeseries(df, kdf, best):
    """Per-session time series: GT, best no-k, best k-scaled, with hypnogram."""
    print('\n' + '=' * 70)
    print('PHASE 6c — Per-session time series')
    print('=' * 70)

    ts_dir = FIG_DIR / 'session_timeseries'
    ts_dir.mkdir(parents=True, exist_ok=True)

    sessions_list = sorted(df.session.unique())

    for sess_label in sessions_list:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), height_ratios=[1, 3, 3])

        sess_df = df[df.session == sess_label]

        # ── Hypnogram ──
        ax_hyp = axes[0]
        # Get unique epochs (one per time point, any channel)
        hyp = sess_df[sess_df.channel == CHANNELS[0]].drop_duplicates('epoch').sort_values('t_hr')
        stage_y_map = {'Wake': 4, 'REM': 3, 'N1': 2, 'N2': 1, 'N3': 0}

        if (hyp.stage != '?').any():
            t_hr = hyp.t_hr.values
            stage_y = np.array([stage_y_map.get(s, -1) for s in hyp.stage.values])
            for stage_name, color in STAGE_COLORS.items():
                mask = hyp.stage.values == stage_name
                if mask.any():
                    ax_hyp.scatter(t_hr[mask], stage_y[mask], c=color, s=3,
                                 label=stage_name, zorder=2)
            ax_hyp.step(t_hr, stage_y, color='black', lw=0.5, alpha=0.4, where='mid')

        ax_hyp.set_yticks([0, 1, 2, 3, 4])
        ax_hyp.set_yticklabels(['N3', 'N2', 'N1', 'REM', 'Wake'], fontsize=8)
        ax_hyp.set_xlim(hyp.t_hr.min(), hyp.t_hr.max())
        ax_hyp.set_ylabel('Stage', fontsize=9)
        ax_hyp.set_title(f'{sess_label} — Rate time series (GT vs estimated)', fontsize=11, fontweight='bold')
        ax_hyp.tick_params(labelbottom=False)
        ax_hyp.legend(loc='upper right', fontsize=7, ncol=5, markerscale=2)

        # ── Resp and Cardiac panels ──
        for ax, band_name in zip(axes[1:], ['resp', 'card']):
            m_best, ch_best = best[band_name]
            unit = 'br/min' if band_name == 'resp' else 'BPM'
            scale = 60.0

            # GT rate
            band_df = sess_df[(sess_df.band == band_name) & (sess_df.channel == ch_best)].sort_values('t_hr')
            t_hr = band_df.t_hr.values
            gt = band_df.gt_hz.values * scale

            # Raw (no k) rate — best method
            raw = band_df[f'rate_{m_best}'].values * scale

            # k-scaled rate
            k_row = kdf[(kdf.band == band_name) & (kdf.session == sess_label) &
                        (kdf.method == m_best) & (kdf.channel == ch_best)]
            if len(k_row) > 0:
                k_val = k_row.iloc[0]['k']
                scaled = raw / k_val
            else:
                k_val = np.nan
                scaled = np.full_like(raw, np.nan)

            # Viterbi smooth the k-scaled rate
            max_jump = 2.0 if band_name == 'resp' else 5.0
            scaled_smooth = viterbi_smooth(scaled / scale, max_jump / scale) * scale

            # Plot
            valid_gt = np.isfinite(gt)
            ax.plot(t_hr[valid_gt], gt[valid_gt], color='black', lw=1.0,
                    alpha=0.7, label='PSG ground truth', zorder=3)

            valid_raw = np.isfinite(raw)
            ax.plot(t_hr[valid_raw], raw[valid_raw], color='#E74C3C', lw=0.4,
                    alpha=0.3, label=f'{m_best}/{ch_best} raw (no k)')

            valid_sc = np.isfinite(scaled_smooth)
            ax.plot(t_hr[valid_sc], scaled_smooth[valid_sc], color='#3498DB', lw=1.0,
                    alpha=0.8, label=f'{m_best}/{ch_best} k-scaled+Viterbi (k={k_val:.2f})')

            # Stage background shading
            if (hyp.stage != '?').any():
                for _, row in hyp.iterrows():
                    s = row['stage']
                    if s in STAGE_COLORS:
                        ax.axvspan(row['t_hr'] - 0.5*30/3600, row['t_hr'] + 0.5*30/3600,
                                  color=STAGE_COLORS[s], alpha=0.06, lw=0)

            # Compute MAE for annotation
            both_valid = np.isfinite(scaled_smooth) & np.isfinite(gt)
            if both_valid.sum() > 0:
                mae = np.median(np.abs(scaled_smooth[both_valid] - gt[both_valid]))
                r_val = np.corrcoef(scaled_smooth[both_valid], gt[both_valid])[0, 1] if both_valid.sum() > 10 else np.nan
                ax.text(0.01, 0.95, f'MAE={mae:.1f} {unit}, r={r_val:.2f}, k={k_val:.2f}',
                       transform=ax.transAxes, fontsize=8, va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax.set_ylabel(f'{"Resp rate" if band_name=="resp" else "Heart rate"} ({unit})', fontsize=9)
            ax.set_xlim(t_hr.min(), t_hr.max())
            ax.legend(loc='upper right', fontsize=7)

            if band_name == 'resp':
                ax.set_ylim(5, 30)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_ylim(30, 120)
                ax.set_xlabel('Time (hours)', fontsize=9)

        plt.tight_layout()
        fig.savefig(ts_dir / f'{sess_label}_rate_timeseries.png')
        fig.savefig(RPT_DIR / f'{sess_label}_rate_timeseries.png')
        plt.close(fig)
        print(f'  {sess_label} time series saved')

    print(f'  All session time series saved to {ts_dir}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    print('Loading all 12 sessions...')
    sessions = load_all_sessions(with_sleep_profiles=True)
    print(f'  Loaded {len(sessions)} sessions\n')

    # Phase 1
    p1_path = ART_DIR / 'rate_consolidation_phase1.parquet'
    if p1_path.exists():
        print(f'Phase 1 data exists, loading from {p1_path}')
        df = pd.read_parquet(p1_path)
    else:
        df = phase1_benchmark(sessions)

    # Re-assign stages from loaded sleep profiles (fixes cached data with missing stages)
    from sleep_monitor.loader import load_sleep_profile
    sess_map = {s.label: s for s in sessions}
    epoch_dur_hr = 30.0 / 3600.0
    codes_arr = np.full(len(df), -1, dtype=int)
    n_profile_loaded = 0
    for sess_label, grp_idx in df.groupby('session').groups.items():
        sess = sess_map.get(sess_label)
        if sess is None:
            print(f'  WARNING: session {sess_label} not found in loaded sessions')
            continue
        # Ensure sleep profile is loaded
        if sess.sleep_profile is None:
            sess.sleep_profile = load_sleep_profile(sess)
        if sess.sleep_profile is None:
            print(f'  WARNING: no sleep profile for {sess_label}')
            continue
        n_profile_loaded += 1
        ep_c = sess.sleep_profile['codes']
        t_hrs = df.loc[grp_idx, 't_hr'].values
        idxs = (t_hrs / epoch_dur_hr).astype(int)
        valid = (idxs >= 0) & (idxs < len(ep_c))
        codes_arr[grp_idx[valid]] = ep_c[idxs[valid]]
    df['stage_code'] = codes_arr
    df['stage'] = df['stage_code'].map(STAGE_LABELS).fillna('?')
    n_valid = (df.stage != '?').sum()
    print(f'  Re-assigned stages: {n_valid}/{len(df)} epochs have valid stage labels '
          f'({n_profile_loaded}/12 profiles loaded)')

    phase1_plots(df)

    # Phase 2
    fdf = phase2_fusion(df)
    phase2_plots(fdf)

    # Phase 3
    rdf = phase3_cwt_ridge(df, sessions)
    phase3_plots(rdf)

    # Phase 4
    vdf = phase4_viterbi(fdf)
    phase4_plots(vdf)

    # Phase 5
    summary = phase5_evaluation(vdf)

    # Phase 6
    kdf, sdf, best = phase6_k_calibration(df)
    phase6_k_heatmap(sdf)
    phase6_k_biomarker(df, kdf, best)
    phase6_session_timeseries(df, kdf, best)

    elapsed = time.time() - t_start
    print(f'\n{"="*70}')
    print(f'ALL PHASES COMPLETE in {elapsed/60:.1f} min')
    print(f'Figures: {FIG_DIR}')
    print(f'Reports: {RPT_DIR}')
    print(f'{"="*70}')
