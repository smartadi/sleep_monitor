#!/usr/bin/env python
"""
Test 4 improvements for cardiac coherence:
  1. GT reference: ECG vs Pleth
  2. Band: wide (0.7-4.0) vs narrow (0.8-2.0)
  3. Acc artifact removal: on vs off
  4. Epoch length: 30s vs 60s

Factorial design: 2x2x2x2 = 16 conditions.
Uses avg channel combination (best fixed method for cardiac).

Outputs
-------
artifacts/cardiac_coherence_test.csv — summary table (16 rows)
notebooks/plots/validation_report/cardiac_improvement_test.png

Usage
-----
    .venv\\Scripts\\python.exe scripts/cardiac_coherence_test.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path
from itertools import product

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
from scripts.signal_validation_enhanced import (
    spectral_peak_and_snr, coherence_at_frequency, band_coherence_mean,
    _get_nperseg_psd, _get_nperseg_coh,
)

OUT_DIR  = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'validation_report'

# ── Conditions ───────────────────────────────────────────────────────────────
GT_REFS      = ['ECG', 'Pleth']
BANDS        = {'wide': (0.7, 4.0), 'narrow': (0.8, 2.0)}
ACC_REMOVAL  = [True, False]
EPOCH_SECS   = [30, 60]

FREQ_TOL = 0.10  # Hz


def compute_session(session, conditions):
    """Run all conditions for one session."""
    fs = session.fs
    label = session.label
    n_samples = session.n_samples

    cap_cle = session.cap['CLE'].astype(np.float64)
    cap_cre = session.cap['CRE'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)
    gt_ecg  = session.psg['ECG'].astype(np.float64)
    gt_pleth = session.psg['Pleth'].astype(np.float64)

    rows = []

    for gt_name, band_name, acc_rm, epoch_sec in conditions:
        band = BANDS[band_name]
        win_n = int(round(epoch_sec * fs))
        step_n = win_n

        # GT signal
        gt_raw = gt_ecg if gt_name == 'ECG' else gt_pleth
        gt_bp = bandpass(gt_raw, band[0], band[1], fs)

        # Cap signal: avg of CLE and CRE
        cap_avg = (cap_cle + cap_cre) / 2.0

        if acc_rm:
            cap_bp = remove_acc_artifact(cap_avg, acc, band[0], band[1], fs)
        else:
            cap_bp = bandpass(cap_avg, band[0], band[1], fs)

        # Epoch grid
        starts = np.arange(0, n_samples - win_n + 1, step_n)
        n_epochs = len(starts)

        epoch_coh_peak = []
        epoch_coh_band = []
        epoch_freq_match = []
        epoch_snr = []

        for s0 in starts:
            s1 = s0 + win_n
            m_seg = cap_bp[s0:s1]
            g_seg = gt_bp[s0:s1]

            mask_freq, mask_snr, _ = spectral_peak_and_snr(m_seg, fs, band)
            gt_freq, _, _ = spectral_peak_and_snr(g_seg, fs, band)

            freq_err = abs(mask_freq - gt_freq) if not (np.isnan(mask_freq) or np.isnan(gt_freq)) else np.nan
            freq_match = 1.0 if (not np.isnan(freq_err) and freq_err <= FREQ_TOL) else 0.0

            coh_peak = coherence_at_frequency(m_seg, g_seg, fs, gt_freq, band)
            coh_band_val = band_coherence_mean(m_seg, g_seg, fs, band)

            epoch_coh_peak.append(coh_peak)
            epoch_coh_band.append(coh_band_val)
            epoch_freq_match.append(freq_match)
            if not np.isnan(mask_snr):
                epoch_snr.append(mask_snr)

        rows.append({
            'session': label,
            'gt_ref': gt_name,
            'band': band_name,
            'acc_removal': acc_rm,
            'epoch_sec': epoch_sec,
            'n_epochs': n_epochs,
            'coh_at_peak': np.nanmean(epoch_coh_peak),
            'coh_band': np.nanmean(epoch_coh_band),
            'freq_match': np.nanmean(epoch_freq_match),
            'snr_median': np.median(epoch_snr) if epoch_snr else np.nan,
        })

    return pd.DataFrame(rows)


def plot_results(summary):
    """Main effects plot for all 4 factors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    factors = [
        ('gt_ref', 'GT Reference', ['ECG', 'Pleth']),
        ('band', 'Frequency Band', ['wide', 'narrow']),
        ('acc_removal', 'Acc Removal', [True, False]),
        ('epoch_sec', 'Epoch Length', [30, 60]),
    ]

    for idx, (factor, flabel, levels) in enumerate(factors):
        ax = axes[idx // 2, idx % 2]
        vals = []
        labels = []
        for level in levels:
            sub = summary[summary[factor] == level]
            vals.append(sub['coh_at_peak'].values)
            labels.append(str(level))

        positions = [0, 1]
        bp = ax.boxplot(vals, positions=positions, widths=0.6, patch_artist=True)
        colors = ['#3498DB', '#E74C3C']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, v in enumerate(vals):
            ax.text(i, np.median(v) + 0.005, f'{np.median(v):.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel(flabel)
        ax.set_ylabel('Coherence at GT peak')
        ax.grid(alpha=0.3, axis='y')

    fig.suptitle('Cardiac Coherence: 4-Factor Improvement Test (avg channel)', fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / 'cardiac_improvement_test.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


def plot_interaction(summary):
    """Heatmap of GT x Band interaction (the two most likely factors)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes, ['coh_at_peak', 'freq_match'],
                                  ['Coherence at GT peak', 'Freq match rate']):
        # Average over acc_removal and epoch_sec
        grp = summary.groupby(['gt_ref', 'band'])[metric].mean().reset_index()
        pivot = grp.pivot(index='gt_ref', columns='band', values=metric)

        im = ax.imshow(pivot.values, aspect='auto', cmap='YlGn',
                        vmin=pivot.values.min() * 0.9, vmax=pivot.values.max() * 1.05)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('Band')
        ax.set_ylabel('GT Reference')
        ax.set_title(title)

        for ri in range(len(pivot.index)):
            for ci in range(len(pivot.columns)):
                v = pivot.values[ri, ci]
                ax.text(ci, ri, f'{v:.3f}', ha='center', va='center',
                        fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('GT Reference x Band Interaction (averaged over acc/epoch factors)', fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / 'cardiac_gt_band_interaction.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


def plot_full_factorial(summary):
    """All 16 conditions ranked by coherence."""
    grp = summary.groupby(['gt_ref', 'band', 'acc_removal', 'epoch_sec']).agg(
        coh=('coh_at_peak', 'mean'),
        fmatch=('freq_match', 'mean'),
        coh_band=('coh_band', 'mean'),
    ).reset_index().sort_values('coh', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [f"{r.gt_ref}|{r.band}|acc={'Y' if r.acc_removal else 'N'}|{r.epoch_sec}s"
              for _, r in grp.iterrows()]
    colors = ['#E74C3C' if r.gt_ref == 'ECG' else '#3498DB' for _, r in grp.iterrows()]

    bars = ax.barh(range(len(grp)), grp['coh'].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(grp)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean coherence at GT peak')
    ax.set_title('All 16 Conditions Ranked by Cardiac Coherence')

    for i, (_, r) in enumerate(grp.iterrows()):
        ax.text(r.coh + 0.002, i, f'{r.coh:.3f} (fm={r.fmatch:.3f})',
                va='center', fontsize=8)

    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color='#E74C3C', alpha=0.8, label='ECG ref'),
        plt.Rectangle((0,0),1,1, color='#3498DB', alpha=0.8, label='Pleth ref'),
    ], loc='lower right')

    fig.tight_layout()
    out = PLOT_DIR / 'cardiac_all_conditions.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    conditions = list(product(GT_REFS, BANDS.keys(), ACC_REMOVAL, EPOCH_SECS))
    print(f'Testing {len(conditions)} conditions across all sessions')

    sessions = load_all_sessions()
    all_dfs = []

    for session in sessions:
        t0 = time.time()
        print(f'  {session.label}...', end=' ', flush=True)
        df = compute_session(session, conditions)
        all_dfs.append(df)
        print(f'{time.time()-t0:.1f}s')

    full = pd.concat(all_dfs, ignore_index=True)

    csv_path = OUT_DIR / 'cardiac_coherence_test.csv'
    full.to_csv(csv_path, index=False)
    print(f'\nSaved {csv_path}')

    # Print summary
    print('\n=== Main effects (mean coherence at GT peak) ===')
    for factor in ['gt_ref', 'band', 'acc_removal', 'epoch_sec']:
        print(f'\n{factor}:')
        grp = full.groupby(factor)['coh_at_peak'].mean()
        for level, val in grp.items():
            fmatch = full[full[factor] == level]['freq_match'].mean()
            print(f'  {str(level):8s}: coh={val:.4f}, freq_match={fmatch:.4f}')

    print('\n=== Best condition ===')
    grp = full.groupby(['gt_ref', 'band', 'acc_removal', 'epoch_sec']).agg(
        coh=('coh_at_peak', 'mean'),
        fmatch=('freq_match', 'mean'),
    ).sort_values('coh', ascending=False)
    best = grp.iloc[0]
    print(f'{grp.index[0]}: coh={best.coh:.4f}, freq_match={best.fmatch:.4f}')

    print('\n=== Current baseline (ECG, wide, acc=True, 30s) ===')
    baseline = full[(full.gt_ref=='ECG') & (full.band=='wide') &
                    (full.acc_removal==True) & (full.epoch_sec==30)]
    print(f'coh={baseline.coh_at_peak.mean():.4f}, freq_match={baseline.freq_match.mean():.4f}')

    # Plots
    print('\nGenerating plots...')
    plot_results(full)
    plot_interaction(full)
    plot_full_factorial(full)

    print('\nDone.')


if __name__ == '__main__':
    main()
