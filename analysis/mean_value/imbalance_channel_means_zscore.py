"""
Per-session channel slow-mean traces, z-scored, over the hypnogram.

Instead of the raw a.u. scale (dominated by a per-session DC offset), we z-score
each channel WITHIN each session so the *shape* of the slow mean is comparable
across sessions and across channels. Robust z (median / MAD) so single-epoch
motion spikes don't set the scale.

Channels (slow-moving mean = vlf_* column, the "mean value" the hypothesis is about):
    CLE        left temple sensor baseline
    CRE        right temple sensor baseline
    CLE-CRE    L-R difference  = directional "imbalance"   (emphasised)
    CLE+CRE    L+R sum         = common-mode baseline

Outputs
    per session : notebooks/plots/mean_value/chanz_<SESSION>.png   (12 files)
    z grid imbalance : notebooks/plots/mean_value/imbalance_zscore_all_sessions_grid.png

Reads reports/mean_value/mean_value_epochs.csv.

Usage:
    .venv/Scripts/python.exe analysis/mean_value/imbalance_channel_means_zscore.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'mean_value'
REPORT_DIR = ROOT / 'reports' / 'mean_value'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_CODES = {0, 1, 2, 3}
ZCLIP = 6.0   # clip display to +/-6 robust-sigma

# channel -> (vlf column, colour, line width, label)
CHANNELS = [
    ('vlf_CLE',     '#27AE60', 1.1, 'CLE (left)'),
    ('vlf_CRE',     '#8E44AD', 1.1, 'CRE (right)'),
    ('vlf_CLE+CRE', '#7F8C8D', 1.1, 'CLE+CRE (common)'),
    ('vlf_CLE-CRE', '#E67E22', 1.8, 'CLE-CRE (imbalance)'),
]


def robust_z(x):
    """(x - median) / (1.4826*MAD), computed over finite values. Spike-resistant."""
    x = np.asarray(x, float)
    good = np.isfinite(x)
    med = np.median(x[good])
    mad = np.median(np.abs(x[good] - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def _bands(ax, t, codes):
    for j in range(len(t) - 1):
        ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                   alpha=0.16, lw=0)


def fig_session(g, label):
    g = g.sort_values('epoch')
    t = g['t_hr'].to_numpy()
    codes = g['stage_code'].to_numpy()
    motion = g['motion'].to_numpy() if 'motion' in g else np.zeros(len(g), bool)

    fig, ax = plt.subplots(figsize=(14, 4.2))
    _bands(ax, t, codes)
    for col, color, lw, lbl in CHANNELS:
        if col not in g:
            continue
        z = np.clip(robust_z(g[col].to_numpy()), -ZCLIP, ZCLIP)
        ax.plot(t, z, lw=lw, color=color, label=lbl, alpha=0.9,
                zorder=4 if 'imbalance' in lbl else 3)
    ax.axhline(0, color='#2C3E50', ls='--', lw=0.8, zorder=2)
    if motion.any():
        ax.plot(t[motion], np.full(motion.sum(), ZCLIP), '|', color='k',
                ms=5, alpha=0.5, zorder=5)
    ax.set_ylim(-ZCLIP - 0.5, ZCLIP + 0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('channel slow-mean  (robust z, per session)')
    ax.set_title(f'{label} — channel slow-means (z-scored) vs hypnogram',
                 fontsize=12, fontweight='bold')
    stage_handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
                     for k in STAGE_ORDER]
    ch_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=ch_handles + stage_handles, loc='upper right', ncol=4,
              fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.12)
    out = PLOT_DIR / f'chanz_{label}.png'
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches='tight'); plt.close(fig)
    return out


def fig_imbalance_grid(df, sessions):
    fig, axes = plt.subplots(6, 2, figsize=(20, 20), sharey=True)
    for ax, lbl in zip(axes.ravel(), sessions):
        g = df[df['session'] == lbl].sort_values('epoch')
        t = g['t_hr'].to_numpy()
        codes = g['stage_code'].to_numpy()
        _bands(ax, t, codes)
        z = np.clip(robust_z(g['vlf_CLE-CRE'].to_numpy()), -ZCLIP, ZCLIP)
        med = 0.0
        ax.plot(t, z, lw=1.1, color='#E67E22', zorder=3)
        ax.axhline(med, color='#2C3E50', ls='--', lw=0.8, zorder=2)
        ax.fill_between(t, med, z, where=(z >= med), color='#C0392B', alpha=0.22,
                        interpolate=True, zorder=1)
        ax.fill_between(t, med, z, where=(z < med), color='#2980B9', alpha=0.22,
                        interpolate=True, zorder=1)
        ax.set_ylim(-ZCLIP - 0.5, ZCLIP + 0.5)
        ax.set_title(lbl, fontsize=10, fontweight='bold')
        ax.set_ylabel('imbalance (z)', fontsize=8)
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
               for k in STAGE_ORDER]
    handles += [plt.Line2D([], [], color='#E67E22', label='imbalance (CLE-CRE), z'),
                mpatches.Patch(color='#C0392B', alpha=0.4, label='above median'),
                mpatches.Patch(color='#2980B9', alpha=0.4, label='below median')]
    fig.legend(handles=handles, loc='upper center', ncol=8, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle('Capacitance imbalance (CLE-CRE) z-scored per session — all 12 sessions',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = PLOT_DIR / 'imbalance_zscore_all_sessions_grid.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    return out


def main():
    csv = REPORT_DIR / 'mean_value_epochs.csv'
    if not csv.exists():
        sys.exit(f'Missing {csv}. Run mean_value_vs_stage.py first.')
    df = pd.read_csv(csv)
    sessions = sorted(df['session'].unique())
    print(f'{len(sessions)} sessions')

    print('\nPer-session channel-mean (z-scored) figures:')
    for lbl in sessions:
        g = df[df['session'] == lbl]
        out = fig_session(g, lbl)
        print(f'  {lbl:5s} -> {out}')

    grid = fig_imbalance_grid(df, sessions)
    print(f'\nZ-scored imbalance grid -> {grid}')
    print(f'\nAll figures saved under: {PLOT_DIR}')


if __name__ == '__main__':
    main()
