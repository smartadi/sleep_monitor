#!/usr/bin/env python
"""
Generate all figures from the signal validation reporting checklist.

Reads: artifacts/merged_validation.parquet
Saves: notebooks/plots/validation_report/*.png

Figures produced:
  1. coherence_spectrogram_resp.png       — coherence over time (resp band)
  2. coherence_spectrogram_card.png       — coherence over time (cardiac band)
  3. coherence_boxplot_stage.png          — per-epoch coherence by sleep stage
  4. coherence_boxplot_apnea.png          — per-epoch coherence by apnea type
  5. surrogate_histogram.png             — real r vs surrogate distribution
  6. waveform_r_distribution.png         — xcorr r and phase lag distributions
  7. spectral_peak_scatter.png           — mask vs GT dominant frequency
  8. spectral_peak_timeseries.png        — overnight spectral peak overlay
  9. bland_altman_resp.png               — Bland-Altman respiratory rate
  10. bland_altman_card.png              — Bland-Altman cardiac rate
  11. bland_altman_by_stage.png          — Bland-Altman per stage
  12. apnea_stratified_summary.png       — coherence/r comparison normal vs apnea
  13. lr_consistency.png                 — Left vs Right channel agreement

Usage
-----
    python scripts/plot_validation_report.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'validation_report'

STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = {'Wake': '#E74C3C', 'N1': '#F39C12', 'N2': '#3498DB',
                'N3': '#2ECC71', 'REM': '#9B59B6'}
APNEA_LABELS = {0: 'Normal', 1: 'Apnea', 2: 'Hypopnea'}
APNEA_COLORS = {0: '#2ECC71', 1: '#E74C3C', 2: '#E67E22'}


def load_data() -> pd.DataFrame:
    path = ART / 'merged_validation.parquet'
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run merge_validation.py first")
    df = pd.read_parquet(path)
    df['apnea_label'] = df['apnea_code'].map(APNEA_LABELS)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Figure generators
# ═══════════════════════════════════════════════════════════════════════════════

def plot_coherence_by_stage(df: pd.DataFrame):
    """Box plots of per-epoch coherence grouped by sleep stage."""
    clean = df[~df['motion_flag']].copy()
    clean['stage'] = pd.Categorical(clean['stage'], categories=STAGE_ORDER, ordered=True)
    clean = clean[clean['stage'].notna() & (clean['stage'] != '?')]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Respiratory
    data_resp = [clean[clean['stage'] == s]['coh_diff_flow_resp'].dropna()
                 for s in STAGE_ORDER]
    bp = axes[0].boxplot(data_resp, labels=STAGE_ORDER, patch_artist=True)
    for patch, stage in zip(bp['boxes'], STAGE_ORDER):
        patch.set_facecolor(STAGE_COLORS[stage])
        patch.set_alpha(0.6)
    axes[0].set_ylabel('Mean coherence')
    axes[0].set_title('Respiratory: Mask vs Flow')
    axes[0].set_ylim(0, 1)

    # Cardiac
    data_card = [clean[clean['stage'] == s]['coh_diff_ecg_card'].dropna()
                 for s in STAGE_ORDER]
    bp = axes[1].boxplot(data_card, labels=STAGE_ORDER, patch_artist=True)
    for patch, stage in zip(bp['boxes'], STAGE_ORDER):
        patch.set_facecolor(STAGE_COLORS[stage])
        patch.set_alpha(0.6)
    axes[1].set_ylabel('Mean coherence')
    axes[1].set_title('Cardiac: Mask vs ECG')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'coherence_boxplot_stage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  coherence_boxplot_stage.png")


def plot_coherence_by_apnea(df: pd.DataFrame):
    """Box plots of per-epoch coherence grouped by apnea type."""
    clean = df[~df['motion_flag']].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = ['Normal', 'Apnea', 'Hypopnea']
    codes = [0, 1, 2]

    for ax, col, title in [
        (axes[0], 'coh_diff_flow_resp', 'Respiratory: Mask vs Flow'),
        (axes[1], 'coh_diff_ecg_card', 'Cardiac: Mask vs ECG'),
    ]:
        data = [clean[clean['apnea_code'] == c][col].dropna() for c in codes]
        # Only plot groups with data
        valid = [(d, l, c) for d, l, c in zip(data, labels, codes) if len(d) > 0]
        if not valid:
            continue
        bp = ax.boxplot([v[0] for v in valid],
                        labels=[v[1] for v in valid], patch_artist=True)
        for patch, (_, _, c) in zip(bp['boxes'], valid):
            patch.set_facecolor(APNEA_COLORS[c])
            patch.set_alpha(0.6)
        ax.set_ylabel('Mean coherence')
        ax.set_title(title)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'coherence_boxplot_apnea.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  coherence_boxplot_apnea.png")


def plot_surrogate_histogram(df: pd.DataFrame):
    """Histogram of surrogate p-values and significance rate."""
    clean = df[~df['motion_flag']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, band in [
        (axes[0], 'surr_resp_p', 'Respiratory'),
        (axes[1], 'surr_card_p', 'Cardiac'),
    ]:
        vals = clean[col].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(0.05, color='red', linestyle='--', linewidth=1.5, label='p=0.05')
        sig_pct = (vals < 0.05).mean() * 100
        ax.set_xlabel('Surrogate p-value')
        ax.set_ylabel('Count')
        ax.set_title(f'{band}: {sig_pct:.1f}% significant (p<0.05)')
        ax.legend()

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'surrogate_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  surrogate_histogram.png")


def plot_waveform_r_distribution(df: pd.DataFrame):
    """Distribution of waveform cross-correlation r and phase lag."""
    clean = df[~df['motion_flag']]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: r distributions
    for ax, col, title in [
        (axes[0, 0], 'xcorr_resp_r', 'Respiratory xcorr r'),
        (axes[0, 1], 'xcorr_card_r', 'Cardiac xcorr r'),
    ]:
        vals = clean[col].dropna()
        ax.hist(vals, bins=50, color='teal', alpha=0.7, edgecolor='white')
        ax.axvline(vals.median(), color='red', linestyle='--',
                   label=f'median={vals.median():.3f}')
        ax.set_xlabel('Cross-correlation r')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()

    # Bottom row: phase lag distributions (in ms)
    fs = 100.0
    for ax, col, title in [
        (axes[1, 0], 'xcorr_resp_lag', 'Respiratory phase lag'),
        (axes[1, 1], 'xcorr_card_lag', 'Cardiac phase lag'),
    ]:
        vals = clean[col].dropna() / fs * 1000  # convert samples to ms
        ax.hist(vals, bins=50, color='coral', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(vals.median(), color='red', linestyle='--',
                   label=f'median={vals.median():.1f} ms')
        ax.set_xlabel('Phase lag (ms)')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'waveform_r_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  waveform_r_distribution.png")


def plot_spectral_peak_scatter(df: pd.DataFrame):
    """Scatter: mask spectral peak freq vs GT freq, coloured by stage."""
    clean = df[~df['motion_flag']].copy()
    clean['stage'] = pd.Categorical(clean['stage'], categories=STAGE_ORDER, ordered=True)
    clean = clean[clean['stage'].notna() & (clean['stage'] != '?')]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Respiratory
    ax = axes[0]
    for stage in STAGE_ORDER:
        sub = clean[clean['stage'] == stage]
        ax.scatter(sub['gt_flow_peak_hz'], sub['mask_resp_peak_hz'],
                   c=STAGE_COLORS[stage], alpha=0.3, s=10, label=stage)
    lim = (0.08, 0.55)
    ax.plot(lim, lim, 'k--', linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('GT Flow peak freq (Hz)')
    ax.set_ylabel('Mask peak freq (Hz)')
    ax.set_title('Respiratory spectral peak alignment')
    ax.legend(fontsize=8, markerscale=2)

    # Cardiac
    ax = axes[1]
    for stage in STAGE_ORDER:
        sub = clean[clean['stage'] == stage]
        ax.scatter(sub['gt_ecg_peak_hz'], sub['mask_card_peak_hz'],
                   c=STAGE_COLORS[stage], alpha=0.3, s=10, label=stage)
    lim = (0.6, 4.2)
    ax.plot(lim, lim, 'k--', linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('GT ECG peak freq (Hz)')
    ax.set_ylabel('Mask peak freq (Hz)')
    ax.set_title('Cardiac spectral peak alignment')
    ax.legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'spectral_peak_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  spectral_peak_scatter.png")


def plot_spectral_peak_timeseries(df: pd.DataFrame):
    """Overnight time series: mask spectral peak overlaid on GT rate (one session)."""
    sessions = df['session'].unique()
    # Pick a representative session with good coverage
    sess = sessions[0] if len(sessions) > 0 else None
    if sess is None:
        return

    sub = df[df['session'] == sess].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Respiratory
    ax = axes[0]
    ax.plot(sub['t_hr'], sub['gt_flow_peak_hz'] * 60, 'k-', alpha=0.7,
            linewidth=0.8, label='GT Flow')
    ax.plot(sub['t_hr'], sub['mask_resp_peak_hz'] * 60, 'b-', alpha=0.5,
            linewidth=0.8, label='Mask')
    ax.set_ylabel('Resp rate (br/min)')
    ax.set_title(f'{sess} — Spectral peak frequency overnight')
    ax.legend(loc='upper right')
    ax.set_ylim(5, 35)

    # Cardiac
    ax = axes[1]
    ax.plot(sub['t_hr'], sub['gt_ecg_peak_hz'] * 60, 'k-', alpha=0.7,
            linewidth=0.8, label='GT ECG')
    ax.plot(sub['t_hr'], sub['mask_card_peak_hz'] * 60, 'r-', alpha=0.5,
            linewidth=0.8, label='Mask')
    ax.set_ylabel('Heart rate (BPM)')
    ax.set_xlabel('Time (hours)')
    ax.legend(loc='upper right')
    ax.set_ylim(30, 120)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'spectral_peak_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  spectral_peak_timeseries.png")


def plot_bland_altman(df: pd.DataFrame):
    """Bland-Altman plots for respiratory and cardiac rates."""
    clean = df[~df['motion_flag']].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pred_col, ref_col, title, unit in [
        (axes[0], 'cap_resp_hz', 'gt_resp_hz', 'Respiratory Rate', 'br/min'),
        (axes[1], 'cap_card_hz', 'gt_card_hz', 'Cardiac Rate', 'BPM'),
    ]:
        ok = clean[pred_col].notna() & clean[ref_col].notna()
        sub = clean[ok]
        pred = sub[pred_col].values * 60.0
        ref = sub[ref_col].values * 60.0
        mean_val = (pred + ref) / 2
        diff_val = pred - ref

        bias = np.mean(diff_val)
        sd = np.std(diff_val)
        loa_upper = bias + 1.96 * sd
        loa_lower = bias - 1.96 * sd

        ax.scatter(mean_val, diff_val, alpha=0.15, s=5, c='steelblue')
        ax.axhline(bias, color='red', linestyle='-', linewidth=1,
                   label=f'Bias: {bias:.2f}')
        ax.axhline(loa_upper, color='red', linestyle='--', linewidth=0.8,
                   label=f'+1.96 SD: {loa_upper:.2f}')
        ax.axhline(loa_lower, color='red', linestyle='--', linewidth=0.8,
                   label=f'-1.96 SD: {loa_lower:.2f}')
        ax.set_xlabel(f'Mean ({unit})')
        ax.set_ylabel(f'Difference (CAP - GT, {unit})')
        ax.set_title(f'Bland-Altman: {title}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'bland_altman.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  bland_altman.png")


def plot_bland_altman_by_stage(df: pd.DataFrame):
    """Bland-Altman stratified by sleep stage (2x5 grid)."""
    clean = df[~df['motion_flag']].copy()
    clean['stage'] = pd.Categorical(clean['stage'], categories=STAGE_ORDER, ordered=True)
    clean = clean[clean['stage'].notna() & (clean['stage'] != '?')]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for row, (pred_col, ref_col, band_label) in enumerate([
        ('cap_resp_hz', 'gt_resp_hz', 'Resp'),
        ('cap_card_hz', 'gt_card_hz', 'Cardiac'),
    ]):
        for col_idx, stage in enumerate(STAGE_ORDER):
            ax = axes[row, col_idx]
            sub = clean[(clean['stage'] == stage) &
                        clean[pred_col].notna() & clean[ref_col].notna()]
            if len(sub) < 5:
                ax.set_title(f'{stage} (n<5)')
                continue
            pred = sub[pred_col].values * 60.0
            ref = sub[ref_col].values * 60.0
            mean_val = (pred + ref) / 2
            diff_val = pred - ref
            bias = np.mean(diff_val)
            sd = np.std(diff_val)

            ax.scatter(mean_val, diff_val, alpha=0.2, s=5,
                       c=STAGE_COLORS[stage])
            ax.axhline(bias, color='red', linestyle='-', linewidth=0.8)
            ax.axhline(bias + 1.96 * sd, color='red', linestyle='--', linewidth=0.6)
            ax.axhline(bias - 1.96 * sd, color='red', linestyle='--', linewidth=0.6)
            ax.set_title(f'{band_label} — {stage}\nbias={bias:.1f} LoA=[{bias-1.96*sd:.1f},{bias+1.96*sd:.1f}]',
                         fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('Diff (CAP-GT)')

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'bland_altman_by_stage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  bland_altman_by_stage.png")


def plot_apnea_stratified(df: pd.DataFrame):
    """Compare coherence and waveform r between normal, apnea, and hypopnea epochs."""
    clean = df[~df['motion_flag']].copy()

    metrics = ['coh_diff_flow_resp', 'coh_diff_ecg_card', 'xcorr_resp_r', 'xcorr_card_r']
    labels = ['Resp coherence', 'Cardiac coherence', 'Resp xcorr r', 'Cardiac xcorr r']
    apnea_types = ['Normal', 'Apnea', 'Hypopnea']
    codes = [0, 1, 2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, metric, label in zip(axes, metrics, labels):
        data = []
        tick_labels = []
        colors = []
        for code, atype in zip(codes, apnea_types):
            vals = clean[clean['apnea_code'] == code][metric].dropna()
            if len(vals) > 0:
                data.append(vals)
                tick_labels.append(f'{atype}\n(n={len(vals)})')
                colors.append(APNEA_COLORS[code])

        if data:
            bp = ax.boxplot(data, labels=tick_labels, patch_artist=True)
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
        ax.set_title(label)
        ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'apnea_stratified_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  apnea_stratified_summary.png")


def plot_lr_consistency(df: pd.DataFrame):
    """Left vs Right channel consistency distributions."""
    clean = df[~df['motion_flag']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title in [
        (axes[0], 'lr_resp_r', 'Respiratory: L vs R correlation'),
        (axes[1], 'lr_card_r', 'Cardiac: L vs R correlation'),
    ]:
        vals = clean[col].dropna()
        ax.hist(vals, bins=50, color='mediumpurple', alpha=0.7, edgecolor='white')
        ax.axvline(vals.median(), color='red', linestyle='--',
                   label=f'median={vals.median():.3f}')
        ax.set_xlabel('Cross-correlation r')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'lr_consistency.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  lr_consistency.png")


def plot_coherence_spectrogram(df: pd.DataFrame):
    """Coherence over time (one session), like a spectrogram-style strip plot."""
    sessions = df['session'].unique()
    sess = sessions[0] if len(sessions) > 0 else None
    if sess is None:
        return

    sub = df[df['session'] == sess].copy()
    t = sub['t_hr'].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Sleep stage strip
    ax = axes[0]
    stage_map = {'Wake': 4, 'N1': 3, 'N2': 2, 'N3': 1, 'REM': 0}
    numeric_stage = sub['stage'].map(stage_map).fillna(-1).values
    ax.scatter(t, numeric_stage, c=[STAGE_COLORS.get(s, '#AAAAAA')
                                     for s in sub['stage']], s=8, marker='|')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'])
    ax.set_title(f'{sess} — Sleep stages + coherence over time')
    ax.set_ylabel('Stage')

    # Respiratory coherence
    ax = axes[1]
    ax.plot(t, sub['coh_diff_flow_resp'], 'b-', alpha=0.7, linewidth=0.8)
    ax.fill_between(t, 0, sub['coh_diff_flow_resp'], alpha=0.2, color='blue')
    ax.set_ylabel('Coherence')
    ax.set_title('Respiratory: Mask(CLE-CRE) vs Flow')
    ax.set_ylim(0, 1)

    # Cardiac coherence
    ax = axes[2]
    ax.plot(t, sub['coh_diff_ecg_card'], 'r-', alpha=0.7, linewidth=0.8)
    ax.fill_between(t, 0, sub['coh_diff_ecg_card'], alpha=0.2, color='red')
    ax.set_ylabel('Coherence')
    ax.set_xlabel('Time (hours)')
    ax.set_title('Cardiac: Mask(CLE-CRE) vs ECG')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'coherence_spectrogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  coherence_spectrogram.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading merged validation data...")
    df = load_data()
    print(f"  {len(df)} epochs, {df['session'].nunique()} sessions\n")
    print("Generating figures:")

    plot_coherence_spectrogram(df)
    plot_coherence_by_stage(df)
    plot_coherence_by_apnea(df)
    plot_surrogate_histogram(df)
    plot_waveform_r_distribution(df)
    plot_spectral_peak_scatter(df)
    plot_spectral_peak_timeseries(df)
    plot_bland_altman(df)
    plot_bland_altman_by_stage(df)
    plot_apnea_stratified(df)
    plot_lr_consistency(df)

    print(f"\nAll figures saved to: {PLOT_DIR}/")


if __name__ == '__main__':
    main()
