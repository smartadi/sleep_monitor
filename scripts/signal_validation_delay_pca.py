#!/usr/bin/env python
"""
Delay-embedding PCA for cap sensor channel combination.

Standard PCA on 2 z-scored channels degenerates to avg or diff because the
eigenvectors of [[1,r],[r,1]] are always [1,1] and [1,-1]. Delay embedding
lifts [CLE, CRE] into a higher-dimensional space where PCA can find
intermediate mixing angles, potentially recovering more of the physiological
signal.

Sweeps tau and n_delays parameters, evaluates each configuration via coherence
analysis against PSG ground truth, and compares to fixed methods + canonical
coherence upper bound.

Outputs
-------
artifacts/delay_pca_validation.parquet  — per-epoch, per-config metrics
artifacts/delay_pca_sweep_summary.csv   — one row per (band, tau, n_delays)
notebooks/plots/validation_report/delay_pca_*.png — diagnostic plots

Usage
-----
    .venv\\Scripts\\python.exe scripts/signal_validation_delay_pca.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, STAGE_LABELS, PSG_EPOCH_SEC
from sleep_monitor.filters import bandpass
from sleep_monitor.loader import load_all_sessions
from sleep_monitor.preprocessing import remove_acc_artifact

# Reuse analysis helpers from enhanced validation
from scripts.signal_validation_enhanced import (
    spectral_peak_and_snr, coherence_at_frequency, band_coherence_mean,
    _get_nperseg_psd, _get_nperseg_coh,
    compute_motion_mask, assign_sleep_stage,
    RESP_BAND, CARD_BAND, WIN_SEC, STEP_SEC,
    RESP_FREQ_TOL, CARD_FREQ_TOL,
)

OUT_DIR  = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'validation_report'

# ── Sweep configurations ────────────────────────────────────────────────────
RESP_TAUS = [50, 75, 100, 125, 150]       # samples (0.5s to 1.5s)
CARD_TAUS = [10, 15, 20, 25, 30]          # samples (0.1s to 0.3s)
N_DELAYS_LIST = [5, 8, 10, 12, 15]        # per channel
MAX_EMBED_SPAN = 1500                     # max samples consumed = 50% of epoch


def build_configs():
    """Build list of (band, tau, n_delays) configs, skipping those that
    consume more than 50% of the epoch."""
    configs = []
    for tau in RESP_TAUS:
        for nd in N_DELAYS_LIST:
            span = (nd - 1) * tau
            if span <= MAX_EMBED_SPAN:
                configs.append(('resp', tau, nd))
    for tau in CARD_TAUS:
        for nd in N_DELAYS_LIST:
            span = (nd - 1) * tau
            if span <= MAX_EMBED_SPAN:
                configs.append(('card', tau, nd))
    return configs


def delay_embed_pca(cle_bp, cre_bp, tau, n_delays, n_pcs=3):
    """Delay-embed [CLE, CRE], z-score per column, SVD -> top PCs.

    Returns
    -------
    pcs : (N, n_pcs) array of PC signals
    info : dict with singular_values, var_explained, effective weights
    """
    L = len(cle_bp)
    N = L - (n_delays - 1) * tau
    if N < 100:
        return None, None

    d = 2 * n_delays
    X = np.empty((N, d))
    for j in range(n_delays):
        offset = j * tau
        X[:, j]            = cle_bp[offset : offset + N]
        X[:, j + n_delays] = cre_bp[offset : offset + N]

    X -= X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X /= stds

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n_pcs = min(n_pcs, len(S))
    pcs = U[:, :n_pcs] * S[:n_pcs]

    var_total = (S ** 2).sum()
    var_explained = (S[:n_pcs] ** 2) / var_total if var_total > 0 else np.zeros(n_pcs)

    w_cle_norm = np.linalg.norm(Vt[0, :n_delays])
    w_cre_norm = np.linalg.norm(Vt[0, n_delays:])
    mixing_angle = np.degrees(np.arctan2(w_cre_norm, w_cle_norm))

    info = {
        'singular_values': S,
        'var_explained': var_explained,
        'w_cle_norm': w_cle_norm,
        'w_cre_norm': w_cre_norm,
        'mixing_angle_deg': mixing_angle,
        'pc1_weights': Vt[0],
    }
    return pcs, info


def compute_session(session, configs):
    """Run delay-PCA sweep for one session."""
    fs = session.fs
    label = session.label
    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    n_samples = session.n_samples

    cap_cle = session.cap['CLE'].astype(np.float64)
    cap_cre = session.cap['CRE'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)

    gt_flow = session.psg['Flow'].astype(np.float64)
    gt_ecg  = session.psg['ECG'].astype(np.float64)

    starts = np.arange(0, n_samples - win_n + 1, step_n)
    n_epochs = len(starts)
    t_hr = (starts + win_n / 2.0) / fs / 3600.0

    motion_flag = compute_motion_mask(acc, fs, win_n, step_n)
    stage_codes = assign_sleep_stage(t_hr, session.sleep_profile)
    apnea_codes = session.apnea_at(t_hr)

    gt_flow_bp = bandpass(gt_flow, RESP_BAND[0], RESP_BAND[1], fs)
    gt_ecg_bp  = bandpass(gt_ecg,  CARD_BAND[0], CARD_BAND[1], fs)

    cle_resp_bp = remove_acc_artifact(cap_cle, acc, RESP_BAND[0], RESP_BAND[1], fs)
    cre_resp_bp = remove_acc_artifact(cap_cre, acc, RESP_BAND[0], RESP_BAND[1], fs)
    cle_card_bp = remove_acc_artifact(cap_cle, acc, CARD_BAND[0], CARD_BAND[1], fs)
    cre_card_bp = remove_acc_artifact(cap_cre, acc, CARD_BAND[0], CARD_BAND[1], fs)

    rows = []

    for band_name, tau, n_delays in configs:
        if band_name == 'resp':
            cle_bp, cre_bp = cle_resp_bp, cre_resp_bp
            gt_bp = gt_flow_bp
            band = RESP_BAND
            freq_tol = RESP_FREQ_TOL
        else:
            cle_bp, cre_bp = cle_card_bp, cre_card_bp
            gt_bp = gt_ecg_bp
            band = CARD_BAND
            freq_tol = CARD_FREQ_TOL

        embed_span = (n_delays - 1) * tau
        config_label = f't{tau}_d{n_delays}'

        for i, s0 in enumerate(starts):
            s1 = s0 + win_n
            cle_seg = cle_bp[s0:s1]
            cre_seg = cre_bp[s0:s1]
            gt_seg  = gt_bp[s0:s1]

            pcs, info = delay_embed_pca(cle_seg, cre_seg, tau, n_delays)
            if pcs is None:
                continue

            pc1 = pcs[:, 0]
            gt_trunc = gt_seg[embed_span:]

            # Spectral peak + SNR
            mask_freq, mask_snr, _ = spectral_peak_and_snr(pc1, fs, band)
            gt_freq, _, _ = spectral_peak_and_snr(gt_trunc, fs, band)

            freq_err = abs(mask_freq - gt_freq) if not (np.isnan(mask_freq) or np.isnan(gt_freq)) else np.nan
            freq_match = 1.0 if (not np.isnan(freq_err) and freq_err <= freq_tol) else 0.0

            # Coherence
            coh_at_peak = coherence_at_frequency(pc1, gt_trunc, fs, gt_freq, band)
            coh_band = band_coherence_mean(pc1, gt_trunc, fs, band)

            # PC2 coherence (diagnostic)
            pc2_coh = np.nan
            if pcs.shape[1] >= 2:
                pc2_coh = coherence_at_frequency(pcs[:, 1], gt_trunc, fs, gt_freq, band)

            rows.append({
                'session': label,
                'band': band_name,
                'tau': tau,
                'n_delays': n_delays,
                'config': config_label,
                'epoch_idx': i,
                't_hr': t_hr[i],
                'stage_code': stage_codes[i],
                'stage': STAGE_LABELS.get(int(stage_codes[i]), '?'),
                'apnea_code': apnea_codes[i],
                'motion_flag': motion_flag[i],
                'mask_freq': mask_freq,
                'gt_freq': gt_freq,
                'freq_err': freq_err,
                'freq_match': freq_match,
                'mask_snr': mask_snr,
                'coh_at_peak': coh_at_peak,
                'coh_band': coh_band,
                'pc2_coh_at_peak': pc2_coh,
                'var_explained_pc1': info['var_explained'][0],
                'var_explained_pc2': info['var_explained'][1] if len(info['var_explained']) > 1 else np.nan,
                'mixing_angle_deg': info['mixing_angle_deg'],
                'w_cle_norm': info['w_cle_norm'],
                'w_cre_norm': info['w_cre_norm'],
                'usable_samples': len(pc1),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary + comparison
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_summary(df):
    """One row per (band, tau, n_delays) with aggregated metrics."""
    grp = df.groupby(['band', 'tau', 'n_delays'])
    summary = grp.agg(
        n_epochs=('epoch_idx', 'count'),
        freq_match_rate=('freq_match', 'mean'),
        coh_at_peak_mean=('coh_at_peak', 'mean'),
        coh_band_mean=('coh_band', 'mean'),
        snr_median=('mask_snr', 'median'),
        pc2_coh_mean=('pc2_coh_at_peak', 'mean'),
        var_expl_pc1_mean=('var_explained_pc1', 'mean'),
        mixing_angle_mean=('mixing_angle_deg', 'mean'),
        mixing_angle_std=('mixing_angle_deg', 'std'),
    ).reset_index()
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sweep_heatmap(summary):
    """Heatmap: tau vs n_delays, colored by coh_at_peak_mean."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        sub = summary[summary.band == band_name]
        if sub.empty:
            continue
        pivot = sub.pivot(index='n_delays', columns='tau', values='coh_at_peak_mean')
        im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                       cmap='viridis', vmin=0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('tau (samples)')
        ax.set_ylabel('n_delays per channel')
        ax.set_title(f'{band_name.upper()}: mean coh at GT peak')

        for ri in range(len(pivot.index)):
            for ci in range(len(pivot.columns)):
                v = pivot.values[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f'{v:.3f}', ha='center', va='center',
                            fontsize=8, color='white' if v < 0.4 else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Delay-PCA Sweep: Coherence at GT Peak', fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / 'delay_pca_sweep_heatmap.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


def plot_vs_methods(summary, enhanced_path, canon_path):
    """Bar chart: best delay-PCA vs fixed methods vs canonical."""
    enh = pd.read_csv(enhanced_path)
    canon = pd.read_parquet(canon_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        sub = summary[summary.band == band_name]
        if sub.empty:
            continue
        best_row = sub.loc[sub['coh_at_peak_mean'].idxmax()]
        best_label = f"dePCA t{int(best_row.tau)}_d{int(best_row.n_delays)}"
        best_coh = best_row['coh_at_peak_mean']

        enh_sub = enh[enh.band == band_name]
        methods = []
        cohs = []
        colors = []
        for ch in ['diff', 'avg', 'cle', 'cre', 'pca']:
            row = enh_sub[enh_sub.channel == ch]
            if not row.empty:
                methods.append(ch)
                cohs.append(row.coh_at_peak_mean.values[0])
                colors.append('#7FB3D8')

        methods.append(best_label)
        cohs.append(best_coh)
        colors.append('#2ECC71')

        canon_col = f'canon_{band_name}_coh_at_peak'
        canon_val = canon[canon_col].mean()
        methods.append('canonical')
        cohs.append(canon_val)
        colors.append('#E74C3C')

        bars = ax.bar(methods, cohs, color=colors)
        ax.set_ylabel('Coherence at GT peak')
        ax.set_title(f'{band_name.upper()}')
        ax.set_ylim(0, max(cohs) * 1.15)
        for bar, v in zip(bars, cohs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('Delay-PCA vs Fixed Methods vs Canonical Upper Bound', fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / 'delay_pca_vs_methods.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


def plot_mixing_angles(df):
    """Histogram of PC1 mixing angles across epochs for best config per band."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        sub = df[df.band == band_name]
        if sub.empty:
            continue
        grp = sub.groupby(['tau', 'n_delays'])['coh_at_peak'].mean()
        best_idx = grp.idxmax()
        best_sub = sub[(sub.tau == best_idx[0]) & (sub.n_delays == best_idx[1])]

        angles = best_sub['mixing_angle_deg'].dropna()
        ax.hist(angles, bins=50, color='#3498DB', edgecolor='white', alpha=0.8)
        ax.axvline(45, color='green', ls='--', lw=1.5, label='avg (45 deg)')
        ax.axvline(0, color='red', ls='--', lw=1.5, label='CLE only (0 deg)')
        ax.axvline(90, color='orange', ls='--', lw=1.5, label='CRE only (90 deg)')
        ax.set_xlabel('Mixing angle (degrees)')
        ax.set_ylabel('Count')
        ax.set_title(f'{band_name.upper()} (tau={best_idx[0]}, d={best_idx[1]})')
        ax.legend(fontsize=8)

    fig.suptitle('Delay-PCA PC1 Mixing Angle Distribution', fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / 'delay_pca_mixing_angles.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


def plot_variance_explained(df):
    """Distribution of PC1/PC2 variance explained for best config per band."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, band_name in zip(axes, ['resp', 'card']):
        sub = df[df.band == band_name]
        if sub.empty:
            continue
        grp = sub.groupby(['tau', 'n_delays'])['coh_at_peak'].mean()
        best_idx = grp.idxmax()
        best_sub = sub[(sub.tau == best_idx[0]) & (sub.n_delays == best_idx[1])]

        ve1 = best_sub['var_explained_pc1'].dropna() * 100
        ve2 = best_sub['var_explained_pc2'].dropna() * 100

        ax.hist(ve1, bins=50, color='#E67E22', edgecolor='white', alpha=0.8, label='PC1')
        ax.hist(ve2, bins=50, color='#3498DB', edgecolor='white', alpha=0.6, label='PC2')
        ax.set_xlabel('Variance explained (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{band_name.upper()} (tau={best_idx[0]}, d={best_idx[1]})')
        ax.legend()

    fig.suptitle('Delay-PCA Variance Explained by PC1 vs PC2', fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / 'delay_pca_variance_explained.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    print(f'Sweep configurations: {len(configs)}')
    for band_name in ['resp', 'card']:
        band_cfgs = [(b, t, d) for b, t, d in configs if b == band_name]
        print(f'  {band_name}: {len(band_cfgs)} configs')
        for b, t, d in band_cfgs:
            span = (d - 1) * t
            print(f'    tau={t:4d}  n_delays={d:2d}  dims={2*d:3d}  '
                  f'span={span:5d} ({span/FS:.1f}s)  '
                  f'usable={3000-span:5d} ({(3000-span)/FS:.1f}s)')

    sessions = load_all_sessions()
    all_dfs = []

    for session in sessions:
        t0 = time.time()
        print(f'\nProcessing {session.label}...')
        df = compute_session(session, configs)
        all_dfs.append(df)
        elapsed = time.time() - t0
        print(f'  {len(df)} rows in {elapsed:.1f}s')

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Save
    pq_path = OUT_DIR / 'delay_pca_validation.parquet'
    full_df.to_parquet(pq_path)
    print(f'\nSaved {pq_path} ({len(full_df)} rows)')

    summary = sweep_summary(full_df)
    csv_path = OUT_DIR / 'delay_pca_sweep_summary.csv'
    summary.to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')

    # Print best configs
    print('\n=== Best configurations ===')
    for band_name in ['resp', 'card']:
        sub = summary[summary.band == band_name]
        if sub.empty:
            continue
        best = sub.loc[sub['coh_at_peak_mean'].idxmax()]
        print(f'{band_name}: tau={int(best.tau)}, n_delays={int(best.n_delays)}, '
              f'coh_at_peak={best.coh_at_peak_mean:.4f}, '
              f'freq_match={best.freq_match_rate:.4f}, '
              f'mixing_angle={best.mixing_angle_mean:.1f} +/- {best.mixing_angle_std:.1f} deg')

    # Plots
    print('\nGenerating plots...')
    plot_sweep_heatmap(summary)

    enhanced_csv = OUT_DIR / 'channel_comparison_summary.csv'
    canon_pq = OUT_DIR / 'canonical_coherence.parquet'
    if enhanced_csv.exists() and canon_pq.exists():
        plot_vs_methods(summary, enhanced_csv, canon_pq)

    plot_mixing_angles(full_df)
    plot_variance_explained(full_df)

    print('\nDone.')


if __name__ == '__main__':
    main()
