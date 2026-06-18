#!/usr/bin/env python
"""
Full-session rate traces for ALL 12 sessions (resp + cardiac), from the final
mask pipeline (artifacts/mask_phase_c.parquet). Each panel: PSG ground truth vs
CAP mask prediction (best strategy, k-scaled + causally smoothed), annotated with
median MAE (mean-level agreement) and within-session Pearson r (tracking).

Shows the honest story: predictions hug each session's mean rate (low MAE) but do
not follow within-session variation (r ~ 0).

Output: writeup/figures/mask_rate_detection/fig14_all_sessions_resp.png
        writeup/figures/mask_rate_detection/fig15_all_sessions_card.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifacts'
FIG = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'

STAGE_COLORS = {'Wake': '#E74C3C', 'N1': '#F39C12', 'N2': '#3498DB',
                'N3': '#2ECC71', 'REM': '#9B59B6'}
plt.rcParams.update({'font.size': 9, 'figure.dpi': 130, 'savefig.dpi': 180,
                     'savefig.bbox': 'tight'})


def wcorr(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 20 or np.std(a[v]) < 1e-9 or np.std(b[v]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[v], b[v])[0, 1])


def main():
    d = pd.read_parquet(ART / 'mask_phase_c.parquet')
    sessions = sorted(d.session.unique())

    for band, unit, ylim, n_fig in [('resp', 'br/min', (5, 32), 14),
                                     ('card', 'BPM', (35, 110), 15)]:
        bsub = d[(d.band == band) & d.gt_hz.notna()]
        # pick best strategy by pooled median MAE on k_full_smooth
        best, best_mae = None, np.inf
        for strat in bsub.strategy.unique():
            s = bsub[bsub.strategy == strat]
            pred = s['rate_k_full_smooth'].values * 60
            gt = s['gt_hz'].values * 60
            v = np.isfinite(pred) & np.isfinite(gt)
            mae = np.median(np.abs(pred[v] - gt[v]))
            if mae < best_mae:
                best_mae, best = mae, strat

        fig, axes = plt.subplots(4, 3, figsize=(16, 11))
        axes = axes.flatten()
        for ax, sess in zip(axes, sessions):
            s = bsub[(bsub.strategy == best) & (bsub.session == sess)].sort_values('t_hr')
            t = s['t_hr'].values
            gt = s['gt_hz'].values * 60
            pred = s['rate_k_full_smooth'].values * 60
            # stage background
            for _, row in s.iterrows():
                c = STAGE_COLORS.get(row['stage'])
                if c:
                    ax.axvspan(row['t_hr'] - 30/7200, row['t_hr'] + 30/7200,
                               color=c, alpha=0.05, lw=0)
            ax.plot(t, gt, color='black', lw=0.8, alpha=0.8, label='PSG (truth)')
            ax.plot(t, pred, color='#C0392B' if band == 'resp' else '#1F77B4',
                    lw=0.9, alpha=0.85, label='CAP mask')
            v = np.isfinite(pred) & np.isfinite(gt)
            mae = np.median(np.abs(pred[v] - gt[v])) if v.sum() else np.nan
            r = wcorr(pred, gt)
            ax.set_title(f'{sess}   MAE={mae:.1f} {unit}   r={r:+.2f}', fontsize=9)
            ax.set_ylim(*ylim)
            ax.set_xlabel('Time (h)', fontsize=8)
            ax.set_ylabel(unit, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].legend(loc='upper right', fontsize=7)
        # stage legend
        handles = [plt.Line2D([0], [0], marker='s', ls='', color=c, label=s, alpha=0.5)
                   for s, c in STAGE_COLORS.items()]
        fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=8,
                   bbox_to_anchor=(0.5, 1.005))
        fig.suptitle(f'{"Respiratory" if band=="resp" else "Cardiac"} rate — all 12 '
                     f'sessions  (strategy: {best}, k-scaled + smoothed)\n'
                     f'Low MAE = good mean-level agreement;  r ~ 0 = no within-session '
                     f'tracking', fontsize=12, fontweight='bold', y=1.04)
        plt.tight_layout()
        fig.savefig(FIG / f'fig{n_fig}_all_sessions_{band}.png')
        plt.close(fig)
        print(f'fig{n_fig}_all_sessions_{band}.png  (best strategy={best})')


if __name__ == '__main__':
    main()
