#!/usr/bin/env python
"""
Full-night rate accuracy analysis across 4 cap channels.

Channels: avg=(CLE+CRE)/2, diff=CLE-CRE, CLE, CRE
Estimators: resp=peaks_scaled/k, cardiac=hilbert_scaled/k
GT: Flow (resp), Pleth (cardiac)
Epochs: 30s non-overlapping, tagged with sleep stage, apnea, motion, electrode drift

Outputs
-------
artifacts/rate_accuracy.parquet          — per-epoch, per-channel metrics
artifacts/rate_accuracy_summary.csv      — aggregated MAE/RMSE/bias/r
notebooks/plots/rate_accuracy/fig1..fig8 — analysis plots

Usage
-----
    .venv\\Scripts\\python.exe scripts/rate_accuracy_analysis.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import spectrogram as sp_spectrogram

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    STAGE_LABELS, APNEA_LABELS, PSG_EPOCH_SEC,
)
from sleep_monitor.loader import load_all_sessions, load_sleep_profile
from sleep_monitor.preprocessing import preprocess_full
from sleep_monitor.ground_truth import gt_resp_rate, gt_heart_rate
from sleep_monitor.rates import (
    rate_peaks_scaled_resp, rate_hilbert_scaled_cardiac,
    rate_hilbert, rate_acf,
)

OUT_DIR  = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'rate_accuracy'

CHANNELS = ['avg', 'diff', 'CLE', 'CRE']
EPOCH_SEC = 30.0


# ── k calibration on arbitrary signal ────────────────────────────────────────

def calibrate_k_on_signal(cap_bp, gt_bp, band, fs,
                          n_windows=50, win_s=60.0, seed=42):
    """Median(raw_cap_rate / gt_rate) across random windows."""
    win_n = int(round(win_s * fs))
    total = len(cap_bp)
    if total < win_n * 2:
        return float('nan')

    rng = np.random.default_rng(seed)
    max_start = total - win_n - 1
    starts = sorted(rng.integers(0, max_start, size=n_windows).tolist())

    f_lo = RESP_LO if band == 'resp' else CARD_LO
    f_hi = RESP_HI if band == 'resp' else CARD_HI

    ratios = []
    for st in starts:
        seg_cap = cap_bp[st:st + win_n]
        seg_gt  = gt_bp[st:st + win_n]

        if band == 'resp':
            r_cap = rate_peaks_scaled_resp(seg_cap, k=1.0, fs=fs)
        else:
            r_cap = rate_hilbert(seg_cap, f_lo, f_hi, fs)

        r_gt = rate_acf(seg_gt, f_lo, f_hi, fs, prominence=0.05)

        if np.isfinite(r_cap) and np.isfinite(r_gt) and r_gt > 0 and r_cap > 0:
            ratios.append(r_cap / r_gt)

    if len(ratios) < 10:
        return float('nan')
    return float(np.median(ratios))


# ── GT with forced Pleth for cardiac ─────────────────────────────────────────

def gt_heart_rate_robust(session):
    """ECG primary, Pleth fallback with stricter peak detection."""
    from sleep_monitor.filters import bandpass
    from scipy.signal import find_peaks
    try:
        result = gt_heart_rate(session, fallback=False)
        return result
    except ValueError:
        pass
    # Pleth fallback with min_dist = 0.4s (150 BPM max) to avoid dicrotic notch
    pleth = session.psg.get('Pleth')
    if pleth is None:
        raise ValueError('No usable cardiac GT signal')
    fs = session.fs
    bp = bandpass(pleth.astype(np.float64), CARD_LO, CARD_HI, fs)
    min_dist = int(0.4 * fs)
    prom = 0.1 * np.std(bp)
    peaks, _ = find_peaks(bp, distance=min_dist, prominence=prom)
    from sleep_monitor.ground_truth import _quality_filter, _build_result
    peaks = _quality_filter(peaks, fs, CARD_LO, min(CARD_HI, 2.5))
    return _build_result(peaks, fs, 'Pleth', 'peak_detection_strict')


def peaks_to_epoch_rates(peak_times_s, epoch_starts_s, epoch_sec):
    """Convert peak times to per-epoch rates (Hz)."""
    rates = np.full(len(epoch_starts_s), np.nan)
    for i, s0 in enumerate(epoch_starts_s):
        s1 = s0 + epoch_sec
        in_win = peak_times_s[(peak_times_s >= s0) & (peak_times_s < s1)]
        if len(in_win) >= 2:
            rates[i] = (len(in_win) - 1) / (in_win[-1] - in_win[0])
    return rates


# ── Per-session computation ──────────────────────────────────────────────────

def compute_session(session):
    fs = session.fs
    n_samples = session.n_samples
    epoch_n = int(round(EPOCH_SEC * fs))

    # Preprocess all channels
    full_sigs, gt_sigs = preprocess_full(session, acc_removal=True)

    # Build channel dict: {channel_name: {'resp': array, 'card': array}}
    channels = {
        'avg':  {
            'resp': (full_sigs['CLE']['resp'] + full_sigs['CRE']['resp']) / 2.0,
            'card': (full_sigs['CLE']['card'] + full_sigs['CRE']['card']) / 2.0,
        },
        'diff': full_sigs['CLE-CRE'],
        'CLE':  full_sigs['CLE'],
        'CRE':  full_sigs['CRE'],
    }

    # GT bandpassed signals for k calibration
    from sleep_monitor.filters import bandpass
    gt_resp_bp = bandpass(session.psg['Flow'].astype(np.float64),
                          RESP_LO, RESP_HI, fs) if 'Flow' in session.psg else gt_sigs['thorax_bp']
    gt_card_bp = bandpass(session.psg['Pleth'].astype(np.float64),
                          CARD_LO, CARD_HI, fs)

    # Calibrate k per channel
    k_resp = {}
    k_card = {}
    for ch_name in CHANNELS:
        k_resp[ch_name] = calibrate_k_on_signal(
            channels[ch_name]['resp'], gt_resp_bp, 'resp', fs)
        k_card[ch_name] = calibrate_k_on_signal(
            channels[ch_name]['card'], gt_card_bp, 'cardiac', fs)

    print(f'    k_resp: ' + ', '.join(f'{ch}={k_resp[ch]:.2f}' for ch in CHANNELS))
    print(f'    k_card: ' + ', '.join(f'{ch}={k_card[ch]:.2f}' for ch in CHANNELS))

    # GT peak-level rates
    resp_gt = gt_resp_rate(session)
    card_gt = gt_heart_rate_robust(session)
    print(f'    GT resp: {resp_gt.signal_used} ({len(resp_gt.peak_indices)} peaks)')
    print(f'    GT card: {card_gt.signal_used} ({len(card_gt.peak_indices)} peaks)')

    # Epoch grid (non-overlapping 30s)
    epoch_starts_n = np.arange(0, n_samples - epoch_n + 1, epoch_n)
    epoch_starts_s = epoch_starts_n / fs
    epoch_centres_s = epoch_starts_s + EPOCH_SEC / 2.0
    epoch_centres_hr = epoch_centres_s / 3600.0
    n_epochs = len(epoch_starts_n)

    # GT rates per epoch
    resp_gt_hz = peaks_to_epoch_rates(resp_gt.peak_times_s, epoch_starts_s, EPOCH_SEC)
    card_gt_hz = peaks_to_epoch_rates(card_gt.peak_times_s, epoch_starts_s, EPOCH_SEC)

    # Sleep stage per epoch
    profile = session.sleep_profile or load_sleep_profile(session)
    if profile is not None:
        stage_codes = np.full(n_epochs, -1, dtype=int)
        for i, t in enumerate(epoch_centres_hr):
            diffs = np.abs(profile['t_ep_hr'] - t)
            idx = np.argmin(diffs)
            if diffs[idx] < PSG_EPOCH_SEC / 3600.0:
                stage_codes[i] = profile['codes'][idx]
        stages = [STAGE_LABELS.get(c, '?') for c in stage_codes]
    else:
        stage_codes = np.full(n_epochs, -1, dtype=int)
        stages = ['?'] * n_epochs

    # Apnea per epoch
    apnea_codes = session.apnea_at(epoch_centres_hr)
    apnea_labels = [APNEA_LABELS.get(int(c), 'Normal') for c in apnea_codes]

    # Motion and electrode drift per epoch
    acc_mag = session.cap['acc_mag'].astype(np.float64)
    cle_raw = session.cap['CLE'].astype(np.float64)
    cre_raw = session.cap['CRE'].astype(np.float64)

    acc_rms_arr = np.empty(n_epochs)
    cle_mean_arr = np.empty(n_epochs)
    cre_mean_arr = np.empty(n_epochs)

    for i, s0 in enumerate(epoch_starts_n):
        s1 = s0 + epoch_n
        acc_rms_arr[i] = np.sqrt(np.mean(acc_mag[s0:s1] ** 2))
        cle_mean_arr[i] = np.mean(cle_raw[s0:s1])
        cre_mean_arr[i] = np.mean(cre_raw[s0:s1])

    cle_mean_delta = np.concatenate([[0.0], np.diff(cle_mean_arr)])
    cre_mean_delta = np.concatenate([[0.0], np.diff(cre_mean_arr)])

    # Rate estimation per channel per epoch
    rows = []
    for ch_name in CHANNELS:
        ch_resp = channels[ch_name]['resp']
        ch_card = channels[ch_name]['card']
        kr = k_resp[ch_name]
        kc = k_card[ch_name]

        resp_cap_hz = np.empty(n_epochs)
        card_cap_hz = np.empty(n_epochs)

        for i, s0 in enumerate(epoch_starts_n):
            s1 = s0 + epoch_n
            resp_cap_hz[i] = rate_peaks_scaled_resp(ch_resp[s0:s1], kr, fs=fs)
            card_cap_hz[i] = rate_hilbert_scaled_cardiac(ch_card[s0:s1], kc, fs=fs)

        resp_err = resp_cap_hz - resp_gt_hz
        card_err = card_cap_hz - card_gt_hz

        for i in range(n_epochs):
            rows.append({
                'session': session.label,
                'subject': session.subject,
                't_hr': epoch_centres_hr[i],
                'stage': stages[i],
                'stage_code': stage_codes[i],
                'apnea': apnea_labels[i],
                'apnea_code': int(apnea_codes[i]),
                'acc_rms': acc_rms_arr[i],
                'cle_mean': cle_mean_arr[i],
                'cre_mean': cre_mean_arr[i],
                'cle_mean_delta': cle_mean_delta[i],
                'cre_mean_delta': cre_mean_delta[i],
                'channel': ch_name,
                'k_resp': kr,
                'k_cardiac': kc,
                'resp_gt_hz': resp_gt_hz[i],
                'resp_cap_hz': resp_cap_hz[i],
                'resp_err_hz': resp_err[i],
                'resp_abs_err_brpm': abs(resp_err[i]) * 60.0,
                'card_gt_hz': card_gt_hz[i],
                'card_cap_hz': card_cap_hz[i],
                'card_err_hz': card_err[i],
                'card_abs_err_bpm': abs(card_err[i]) * 60.0,
            })

    df = pd.DataFrame(rows)

    # Oracle best channel per epoch
    for band_prefix, err_col in [('resp', 'resp_abs_err_brpm'), ('card', 'card_abs_err_bpm')]:
        best_col = f'{band_prefix}_best_channel'
        pivot = df.pivot(index='t_hr', columns='channel', values=err_col)
        best = pivot.idxmin(axis=1)
        df[best_col] = df['t_hr'].map(best)

    return df


# ── Plotting ─────────────────────────────────────────────────────────────────

STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS_MAP = {'Wake': '#E74C3C', 'N1': '#F39C12', 'N2': '#3498DB',
                    'N3': '#2E86C1', 'REM': '#9B59B6', '?': '#AAAAAA'}
CH_COLORS = {'avg': '#2ECC71', 'diff': '#E67E22', 'CLE': '#3498DB', 'CRE': '#9B59B6'}


def fig1_overnight_rates(df, sessions):
    """Per-session full-night plots: hypnogram, apnea, spectrogram, resp & cardiac rates (all channels)."""
    APNEA_COLORS = {'Normal': '#2ECC71', 'Apnea': '#E74C3C', 'Hypopnea': '#E67E22'}

    for session in sessions:
        sess_label = session.label
        sdf_all = df[df.session == sess_label]
        sdf = sdf_all[sdf_all.channel == 'avg'].sort_values('t_hr')
        if len(sdf) == 0:
            continue

        t = sdf['t_hr'].values
        t_min, t_max = t.min(), t.max()

        fig, axes = plt.subplots(5, 1, figsize=(18, 14),
                                 gridspec_kw={'height_ratios': [1, 1, 3, 4, 4]})
        ax_hyp, ax_apn, ax_spec, ax_resp, ax_card = axes

        # Row 1: Hypnogram
        for i in range(len(sdf) - 1):
            color = STAGE_COLORS_MAP.get(sdf.iloc[i]['stage'], '#AAAAAA')
            ax_hyp.fill_between(
                [sdf.iloc[i]['t_hr'], sdf.iloc[i+1]['t_hr']],
                0, 1, color=color, alpha=0.7, linewidth=0)
        for st in STAGE_ORDER:
            ax_hyp.fill_between([], [], color=STAGE_COLORS_MAP[st], label=st)
        ax_hyp.legend(loc='upper right', fontsize=7, ncol=5, framealpha=0.8)
        ax_hyp.set_yticks([])
        ax_hyp.set_ylabel('Stage')
        ax_hyp.set_xlim(t_min, t_max)
        ax_hyp.set_title(f'{sess_label} — {session.subject}', fontsize=12, fontweight='bold')

        # Row 2: Apnea status
        for i in range(len(sdf) - 1):
            ap = sdf.iloc[i]['apnea']
            color = APNEA_COLORS.get(ap, '#2ECC71')
            alpha = 0.8 if ap != 'Normal' else 0.3
            ax_apn.fill_between(
                [sdf.iloc[i]['t_hr'], sdf.iloc[i+1]['t_hr']],
                0, 1, color=color, alpha=alpha, linewidth=0)
        for ap_label, ap_color in APNEA_COLORS.items():
            ax_apn.fill_between([], [], color=ap_color, label=ap_label)
        ax_apn.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.8)
        ax_apn.set_yticks([])
        ax_apn.set_ylabel('Apnea')
        ax_apn.set_xlim(t_min, t_max)

        # Row 3: Spectrogram (0-5 Hz on avg of CLE+CRE, after acc removal)
        fs = session.fs
        cle = session.cap['CLE'].astype(np.float64)
        cre = session.cap['CRE'].astype(np.float64)
        avg_raw = (cle + cre) / 2.0
        from sleep_monitor.preprocessing import remove_acc_artifact
        from sleep_monitor.filters import bandpass
        acc = session.cap['acc_mag'].astype(np.float64)
        avg_clean = avg_raw - np.mean(avg_raw)
        acc_bp05 = bandpass(acc, 0.05, 5.0, fs)
        beta = np.dot(acc_bp05, avg_clean) / (np.dot(acc_bp05, acc_bp05) + 1e-12)
        avg_clean = avg_clean - beta * acc_bp05

        nperseg = int(30 * fs)
        noverlap = int(20 * fs)
        f_sg, t_sg, Sxx = sp_spectrogram(avg_clean, fs=fs, nperseg=nperseg,
                                          noverlap=noverlap, window='hann')
        mask = f_sg <= 5.0
        f_sg = f_sg[mask]
        Sxx = Sxx[mask, :]
        t_sg_hr = t_sg / 3600.0

        ax_spec.pcolormesh(t_sg_hr, f_sg, 10 * np.log10(Sxx + 1e-20),
                           shading='gouraud', cmap='inferno', rasterized=True)
        ax_spec.set_ylabel('Freq (Hz)')
        ax_spec.set_ylim(0, 5)
        ax_spec.set_xlim(t_min, t_max)
        ax_spec.axhline(RESP_HI, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)
        ax_spec.axhline(CARD_LO, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)
        ax_spec.text(t_min + 0.02, 0.3, 'resp', color='cyan', fontsize=7, alpha=0.7)
        ax_spec.text(t_min + 0.02, CARD_LO + 0.1, 'cardiac', color='cyan', fontsize=7, alpha=0.7)

        # Row 4: Respiratory rates — all 4 channels + GT
        gt_t = sdf['t_hr'].values
        gt_resp = sdf['resp_gt_hz'].values * 60
        ax_resp.plot(gt_t, gt_resp, 'k-', linewidth=1.2, alpha=0.8, label='GT (Flow)', zorder=5)
        for ch in CHANNELS:
            ch_df = sdf_all[sdf_all.channel == ch].sort_values('t_hr')
            kr = ch_df['k_resp'].iloc[0]
            ax_resp.plot(ch_df['t_hr'].values, ch_df['resp_cap_hz'].values * 60,
                         '-', color=CH_COLORS[ch], linewidth=0.7, alpha=0.6,
                         label=f'{ch} (k={kr:.2f})')
        ax_resp.set_ylabel('Resp (br/min)')
        ax_resp.set_ylim(0, 40)
        ax_resp.legend(loc='upper right', fontsize=7, ncol=5, framealpha=0.8)
        ax_resp.grid(alpha=0.2)
        ax_resp.set_xlim(t_min, t_max)

        # Row 5: Cardiac rates — all 4 channels + GT
        gt_card = sdf['card_gt_hz'].values * 60
        ax_card.plot(gt_t, gt_card, 'k-', linewidth=1.2, alpha=0.8, label='GT', zorder=5)
        for ch in CHANNELS:
            ch_df = sdf_all[sdf_all.channel == ch].sort_values('t_hr')
            kc = ch_df['k_cardiac'].iloc[0]
            ax_card.plot(ch_df['t_hr'].values, ch_df['card_cap_hz'].values * 60,
                         '-', color=CH_COLORS[ch], linewidth=0.7, alpha=0.6,
                         label=f'{ch} (k={kc:.2f})')
        ax_card.set_ylabel('Cardiac (BPM)')
        ax_card.set_xlabel('Time (hours)')
        ax_card.set_ylim(30, 120)
        ax_card.legend(loc='upper right', fontsize=7, ncol=5, framealpha=0.8)
        ax_card.grid(alpha=0.2)
        ax_card.set_xlim(t_min, t_max)

        fig.tight_layout()
        out = PLOT_DIR / f'fig1_{sess_label}_overnight.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  {out.name}')


def fig2_error_by_stage(df):
    """Boxplots of absolute error by sleep stage."""
    avg_df = df[df.channel == 'avg']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, unit, title in [
        (axes[0], 'resp_abs_err_brpm', 'br/min', 'Respiratory'),
        (axes[1], 'card_abs_err_bpm', 'BPM', 'Cardiac'),
    ]:
        data, labels = [], []
        for st in STAGE_ORDER:
            vals = avg_df[avg_df.stage == st][col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(f'{st}\n(n={len(vals)})')
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        for patch, st in zip(bp['boxes'], STAGE_ORDER[:len(data)]):
            patch.set_facecolor(STAGE_COLORS_MAP.get(st, '#AAAAAA'))
            patch.set_alpha(0.7)
        for i, d in enumerate(data):
            ax.text(i + 1, np.median(d) + 0.1, f'{np.median(d):.1f}',
                    ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel(f'Absolute Error ({unit})')
        ax.set_title(f'{title} Error by Sleep Stage')
        ax.grid(alpha=0.2, axis='y')

    fig.suptitle('Figure 2: Rate Error by Sleep Stage (avg channel)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOT_DIR / 'fig2_error_by_stage.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')
    return out


def fig3_error_by_apnea(df):
    """Boxplots by apnea status."""
    avg_df = df[df.channel == 'avg']
    apnea_order = ['Normal', 'Apnea', 'Hypopnea']
    apnea_colors = {'Normal': '#2ECC71', 'Apnea': '#E74C3C', 'Hypopnea': '#E67E22'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, unit, title in [
        (axes[0], 'resp_abs_err_brpm', 'br/min', 'Respiratory'),
        (axes[1], 'card_abs_err_bpm', 'BPM', 'Cardiac'),
    ]:
        data, labels = [], []
        for ap in apnea_order:
            vals = avg_df[avg_df.apnea == ap][col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(f'{ap}\n(n={len(vals)})')
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        for patch, ap in zip(bp['boxes'], apnea_order[:len(data)]):
            patch.set_facecolor(apnea_colors.get(ap, '#AAAAAA'))
            patch.set_alpha(0.7)
        for i, d in enumerate(data):
            ax.text(i + 1, np.median(d) + 0.1, f'{np.median(d):.1f}',
                    ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel(f'Absolute Error ({unit})')
        ax.set_title(f'{title} Error by Apnea Status')
        ax.grid(alpha=0.2, axis='y')

    fig.suptitle('Figure 3: Rate Error by Apnea Status (avg channel)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOT_DIR / 'fig3_error_by_apnea.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')
    return out


def fig4_error_by_motion_and_drift(df):
    """2x2 grid: motion quartiles and mean-shift quartiles vs error."""
    avg_df = df[df.channel == 'avg'].copy()
    avg_df['mean_shift'] = avg_df['cle_mean_delta'].abs() + avg_df['cre_mean_delta'].abs()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col_idx, (err_col, unit, band_label) in enumerate([
        ('resp_abs_err_brpm', 'br/min', 'Respiratory'),
        ('card_abs_err_bpm', 'BPM', 'Cardiac'),
    ]):
        # Top row: motion
        ax = axes[0, col_idx]
        valid = avg_df[['acc_rms', err_col]].dropna()
        if len(valid) > 0:
            valid = valid.copy()
            valid['q'] = pd.qcut(valid['acc_rms'], 4, labels=['Q1\nlow', 'Q2', 'Q3', 'Q4\nhigh'], duplicates='drop')
            data = [g[err_col].values for _, g in valid.groupby('q', observed=True)]
            q_labels = [str(q) for q in valid['q'].cat.categories]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            ax.set_xticks(range(1, len(q_labels) + 1))
            ax.set_xticklabels(q_labels)
            for patch in bp['boxes']:
                patch.set_facecolor('#3498DB')
                patch.set_alpha(0.7)
            for i, d in enumerate(data):
                ax.text(i + 1, np.median(d) + 0.1, f'{np.median(d):.1f}',
                        ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel(f'Abs Error ({unit})')
        ax.set_title(f'{band_label} - Motion (acc RMS)')
        ax.grid(alpha=0.2, axis='y')

        # Bottom row: electrode drift
        ax = axes[1, col_idx]
        valid = avg_df[['mean_shift', err_col]].dropna()
        if len(valid) > 0:
            valid = valid.copy()
            valid['q'] = pd.qcut(valid['mean_shift'], 4, labels=['Q1\nlow', 'Q2', 'Q3', 'Q4\nhigh'], duplicates='drop')
            data = [g[err_col].values for _, g in valid.groupby('q', observed=True)]
            q_labels = [str(q) for q in valid['q'].cat.categories]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            ax.set_xticks(range(1, len(q_labels) + 1))
            ax.set_xticklabels(q_labels)
            for patch in bp['boxes']:
                patch.set_facecolor('#E67E22')
                patch.set_alpha(0.7)
            for i, d in enumerate(data):
                ax.text(i + 1, np.median(d) + 0.1, f'{np.median(d):.1f}',
                        ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel(f'Abs Error ({unit})')
        ax.set_xlabel('Electrode Drift Quartile')
        ax.set_title(f'{band_label} - Electrode Drift (|delta CLE| + |delta CRE|)')
        ax.grid(alpha=0.2, axis='y')

    fig.suptitle('Figure 4: Error vs Motion and Electrode Drift (avg channel)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOT_DIR / 'fig4_error_by_motion_and_drift.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')
    return out


def fig5_bland_altman(df):
    """Bland-Altman plots colored by stage."""
    avg_df = df[df.channel == 'avg']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, gt_col, cap_col, mult, unit, title in [
        (axes[0], 'resp_gt_hz', 'resp_cap_hz', 60, 'br/min', 'Respiratory'),
        (axes[1], 'card_gt_hz', 'card_cap_hz', 60, 'BPM', 'Cardiac'),
    ]:
        valid = avg_df[[gt_col, cap_col, 'stage']].dropna()
        if len(valid) == 0:
            continue
        gt = valid[gt_col].values * mult
        cap = valid[cap_col].values * mult
        mean_val = (gt + cap) / 2.0
        diff_val = cap - gt

        for st in STAGE_ORDER:
            mask = valid['stage'].values == st
            if mask.sum() > 0:
                ax.scatter(mean_val[mask], diff_val[mask], s=3, alpha=0.3,
                           color=STAGE_COLORS_MAP.get(st, '#AAAAAA'), label=st)

        bias = np.mean(diff_val)
        loa_upper = bias + 1.96 * np.std(diff_val)
        loa_lower = bias - 1.96 * np.std(diff_val)
        ax.axhline(bias, color='k', linewidth=1, linestyle='-', label=f'Bias={bias:.1f}')
        ax.axhline(loa_upper, color='k', linewidth=0.8, linestyle='--',
                   label=f'+1.96 SD={loa_upper:.1f}')
        ax.axhline(loa_lower, color='k', linewidth=0.8, linestyle='--',
                   label=f'-1.96 SD={loa_lower:.1f}')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

        ax.set_xlabel(f'Mean of GT and CAP ({unit})')
        ax.set_ylabel(f'CAP - GT ({unit})')
        ax.set_title(f'{title} Bland-Altman')
        ax.legend(fontsize=7, loc='upper left', markerscale=3)
        ax.grid(alpha=0.2)

    fig.suptitle('Figure 5: Bland-Altman Analysis (avg channel)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOT_DIR / 'fig5_bland_altman.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')
    return out


def fig6_per_session_summary(df):
    """MAE bar chart per session."""
    avg_df = df[df.channel == 'avg']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sessions = sorted(avg_df['session'].unique())
    x = np.arange(len(sessions))
    width = 0.6

    for ax, col, unit, title in [
        (axes[0], 'resp_abs_err_brpm', 'br/min', 'Respiratory MAE'),
        (axes[1], 'card_abs_err_bpm', 'BPM', 'Cardiac MAE'),
    ]:
        maes = [avg_df[avg_df.session == s][col].mean() for s in sessions]
        bars = ax.bar(x, maes, width, color='#3498DB', alpha=0.8)
        for bar, mae in zip(bars, maes):
            if not np.isnan(mae):
                ax.text(bar.get_x() + bar.get_width() / 2, mae + 0.1,
                        f'{mae:.1f}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=45, fontsize=8)
        ax.set_ylabel(f'MAE ({unit})')
        ax.set_title(title)
        ax.grid(alpha=0.2, axis='y')

    fig.suptitle('Figure 6: Per-Session Rate Accuracy (avg channel)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOT_DIR / 'fig6_per_session_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')
    return out


def fig7_channel_comparison(df):
    """Grouped bar: MAE per channel with oracle best line."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, unit, best_col, title in [
        (axes[0], 'resp_abs_err_brpm', 'br/min', 'resp_best_channel', 'Respiratory MAE'),
        (axes[1], 'card_abs_err_bpm', 'BPM', 'card_best_channel', 'Cardiac MAE'),
    ]:
        x = np.arange(len(CHANNELS))
        maes = []
        for ch in CHANNELS:
            maes.append(df[df.channel == ch][col].mean())

        bars = ax.bar(x, maes, 0.6,
                      color=[CH_COLORS[ch] for ch in CHANNELS], alpha=0.8)
        for bar, mae in zip(bars, maes):
            if not np.isnan(mae):
                ax.text(bar.get_x() + bar.get_width() / 2, mae + 0.1,
                        f'{mae:.1f}', ha='center', fontsize=9, fontweight='bold')

        # Oracle best: pick the best channel's error for each epoch
        oracle_errors = []
        for t, grp in df.groupby('t_hr'):
            best_row = grp.loc[grp[col].idxmin()]
            oracle_errors.append(best_row[col])
        oracle_mae = np.nanmean(oracle_errors)
        ax.axhline(oracle_mae, color='k', linewidth=1.5, linestyle='--',
                   label=f'Oracle best={oracle_mae:.1f}')

        ax.set_xticks(x)
        ax.set_xticklabels(CHANNELS)
        ax.set_ylabel(f'MAE ({unit})')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2, axis='y')

    fig.suptitle('Figure 7: Channel Comparison (all sessions)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOT_DIR / 'fig7_channel_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')
    return out


def fig8_best_channel_distribution(df):
    """Stacked bar: how often each channel is best, by stage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, best_col, title in [
        (axes[0], 'resp_best_channel', 'Respiratory'),
        (axes[1], 'card_best_channel', 'Cardiac'),
    ]:
        # Use only one channel row per epoch (avoid quadruple counting)
        epoch_df = df[df.channel == 'avg'][['stage', best_col]].dropna()
        stages_present = [st for st in STAGE_ORDER if st in epoch_df['stage'].values]

        bottom = np.zeros(len(stages_present))
        for ch in CHANNELS:
            counts = []
            for st in stages_present:
                sub = epoch_df[epoch_df.stage == st]
                counts.append((sub[best_col] == ch).sum() / len(sub) * 100 if len(sub) > 0 else 0)
            ax.bar(range(len(stages_present)), counts, bottom=bottom, width=0.6,
                   color=CH_COLORS[ch], alpha=0.8, label=ch)
            bottom += np.array(counts)

        ax.set_xticks(range(len(stages_present)))
        n_per_stage = [len(epoch_df[epoch_df.stage == st]) for st in stages_present]
        ax.set_xticklabels([f'{st}\n(n={n})' for st, n in zip(stages_present, n_per_stage)])
        ax.set_ylabel('% epochs where channel is best')
        ax.set_title(f'{title} Best Channel Distribution')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.2, axis='y')

    fig.suptitle('Figure 8: Oracle Best Channel by Sleep Stage', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOT_DIR / 'fig8_best_channel_distribution.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    sessions = load_all_sessions(with_sleep_profiles=True, with_apnea=True)

    all_dfs = []
    for session in sessions:
        t0 = time.time()
        print(f'  {session.label}...')
        df = compute_session(session)
        all_dfs.append(df)
        print(f'    {len(df)} rows in {time.time()-t0:.1f}s')

    full = pd.concat(all_dfs, ignore_index=True)

    # Save per-epoch data
    pq_path = OUT_DIR / 'rate_accuracy.parquet'
    full.to_parquet(pq_path, index=False)
    print(f'\nSaved {pq_path} ({len(full)} rows)')

    # Summary table
    summary_rows = []
    for ch in CHANNELS:
        ch_df = full[full.channel == ch]
        for sess in sorted(full['session'].unique()):
            sdf = ch_df[ch_df.session == sess]
            valid_resp = sdf[['resp_gt_hz', 'resp_cap_hz']].dropna()
            valid_card = sdf[['card_gt_hz', 'card_cap_hz']].dropna()

            resp_r = float(pearsonr(valid_resp['resp_gt_hz'], valid_resp['resp_cap_hz'])[0]) if len(valid_resp) >= 5 else np.nan
            card_r = float(pearsonr(valid_card['card_gt_hz'], valid_card['card_cap_hz'])[0]) if len(valid_card) >= 5 else np.nan

            summary_rows.append({
                'session': sess,
                'channel': ch,
                'resp_mae_brpm': sdf['resp_abs_err_brpm'].mean(),
                'resp_bias_brpm': sdf['resp_err_hz'].mean() * 60,
                'resp_rmse_brpm': np.sqrt((sdf['resp_err_hz'] ** 2).mean()) * 60,
                'resp_r': resp_r,
                'card_mae_bpm': sdf['card_abs_err_bpm'].mean(),
                'card_bias_bpm': sdf['card_err_hz'].mean() * 60,
                'card_rmse_bpm': np.sqrt((sdf['card_err_hz'] ** 2).mean()) * 60,
                'card_r': card_r,
                'k_resp': sdf['k_resp'].iloc[0] if len(sdf) > 0 else np.nan,
                'k_cardiac': sdf['k_cardiac'].iloc[0] if len(sdf) > 0 else np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    csv_path = OUT_DIR / 'rate_accuracy_summary.csv'
    summary.to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')

    # Print highlights
    print('\n=== Overall MAE (avg channel) ===')
    avg_df = full[full.channel == 'avg']
    print(f'  Resp:    {avg_df["resp_abs_err_brpm"].mean():.2f} br/min')
    print(f'  Cardiac: {avg_df["card_abs_err_bpm"].mean():.2f} BPM')

    print('\n=== MAE by channel ===')
    for ch in CHANNELS:
        ch_df = full[full.channel == ch]
        print(f'  {ch:5s}: resp={ch_df["resp_abs_err_brpm"].mean():.2f} br/min, '
              f'card={ch_df["card_abs_err_bpm"].mean():.2f} BPM')

    print('\n=== MAE by stage (avg channel) ===')
    for st in STAGE_ORDER:
        sdf = avg_df[avg_df.stage == st]
        if len(sdf) > 0:
            print(f'  {st:5s} (n={len(sdf):4d}): '
                  f'resp={sdf["resp_abs_err_brpm"].mean():.2f}, '
                  f'card={sdf["card_abs_err_bpm"].mean():.2f}')

    print('\n=== MAE by apnea (avg channel) ===')
    for ap in ['Normal', 'Apnea', 'Hypopnea']:
        sdf = avg_df[avg_df.apnea == ap]
        if len(sdf) > 0:
            print(f'  {ap:10s} (n={len(sdf):4d}): '
                  f'resp={sdf["resp_abs_err_brpm"].mean():.2f}, '
                  f'card={sdf["card_abs_err_bpm"].mean():.2f}')

    # Plots
    print('\nGenerating figures...')
    fig1_overnight_rates(full, sessions)
    fig2_error_by_stage(full)
    fig3_error_by_apnea(full)
    fig4_error_by_motion_and_drift(full)
    fig5_bland_altman(full)
    fig6_per_session_summary(full)
    fig7_channel_comparison(full)
    fig8_best_channel_distribution(full)

    print('\nDone.')


if __name__ == '__main__':
    main()
