"""
Full-session directional-imbalance figures for ALL 12 sessions.

imbalance(t) = vlf_CLE-CRE = the slowly-moving L-R mean ("capacitance imbalance").
For each session we draw the whole-night imbalance trace over a colour-banded
hypnogram, shaded above/below the session median (left- vs right-dominant),
with high-motion epochs marked. Also emits one combined 6x2 grid overview.

Reads reports/mean_value/mean_value_epochs.csv.

Usage:
    .venv/Scripts/python.exe analysis/mean_value/imbalance_full_session_figs.py
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

IMBALANCE_COL = 'vlf_CLE-CRE'
SLEEP_CODES = {0, 1, 2, 3}


# common y-range for the shared-scale grid (keeps S6 real swings + S5N1 positive,
# clips only the narrow artifact spikes below the floor).
COMMON_YLIM = (-760.0, 430.0)


def _draw(ax, g, label, legend=False, ylim=None):
    g = g.sort_values('epoch')
    t = g['t_hr'].to_numpy()
    f = g[IMBALANCE_COL].to_numpy()
    codes = g['stage_code'].to_numpy()
    motion = g['motion'].to_numpy() if 'motion' in g else np.zeros(len(g), bool)

    sl = np.isin(codes, list(SLEEP_CODES))
    med = np.nanmedian(f[sl]) if sl.any() else np.nanmedian(f)

    # hypnogram colour bands
    for j in range(len(t) - 1):
        ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                   alpha=0.16, lw=0)

    fp = f
    if ylim is not None:
        fp = np.clip(f, ylim[0], ylim[1])         # clip trace to shared range
    ax.plot(t, fp, lw=1.1, color='#E67E22', zorder=3)
    ax.axhline(med, color='#2C3E50', ls='--', lw=0.9, zorder=2)
    ax.fill_between(t, med, fp, where=(fp >= med), color='#C0392B', alpha=0.22,
                    interpolate=True, zorder=1)
    ax.fill_between(t, med, fp, where=(fp < med), color='#2980B9', alpha=0.22,
                    interpolate=True, zorder=1)
    # mark epochs clipped by the shared floor/ceiling
    if ylim is not None:
        below = f < ylim[0]
        if below.any():
            ax.plot(t[below], np.full(below.sum(), ylim[0]), marker='v', ls='',
                    color='#7f0000', ms=5, zorder=5)
        above = f > ylim[1]
        if above.any():
            ax.plot(t[above], np.full(above.sum(), ylim[1]), marker='^', ls='',
                    color='#7f0000', ms=5, zorder=5)
    # motion ticks along the top
    if motion.any():
        ytop = ylim[1] if ylim is not None else np.nanmax(f)
        ax.plot(t[motion], np.full(motion.sum(), ytop), '|', color='k',
                ms=5, alpha=0.5, zorder=4)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.12)

    if legend:
        handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
                   for k in STAGE_ORDER]
        handles += [plt.Line2D([], [], color='#E67E22', label='imbalance (CLE-CRE)'),
                    plt.Line2D([], [], color='#2C3E50', ls='--', label='session median'),
                    mpatches.Patch(color='#C0392B', alpha=0.4, label='left-dominant'),
                    mpatches.Patch(color='#2980B9', alpha=0.4, label='right-dominant'),
                    plt.Line2D([], [], color='k', marker='|', ls='', label='motion')]
        ax.legend(handles=handles, loc='upper right', ncol=4, fontsize=6.5,
                  framealpha=0.9)


def main():
    csv = REPORT_DIR / 'mean_value_epochs.csv'
    if not csv.exists():
        sys.exit(f'Missing {csv}. Run mean_value_vs_stage.py first.')
    df = pd.read_csv(csv)
    sessions = sorted(df['session'].unique())
    print(f'{len(sessions)} sessions: {sessions}')

    # ── individual full-session traces ──
    for lbl in sessions:
        g = df[df['session'] == lbl]
        fig, ax = plt.subplots(figsize=(14, 3.6))
        _draw(ax, g, f'{lbl} — capacitance imbalance (L-R slow mean) vs hypnogram',
              legend=True)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('imbalance  (a.u.)')
        out = PLOT_DIR / f'imbalance_full_{lbl}.png'
        fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches='tight')
        plt.close(fig)
        print(f'  {lbl}: {out.name}')

    # ── combined 6x2 grid ──
    fig, axes = plt.subplots(6, 2, figsize=(20, 20), sharex=False)
    for ax, lbl in zip(axes.ravel(), sessions):
        g = df[df['session'] == lbl]
        _draw(ax, g, lbl, legend=False)
        ax.set_ylabel('imbalance (a.u.)', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    # one shared legend on the figure
    handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
               for k in STAGE_ORDER]
    handles += [plt.Line2D([], [], color='#E67E22', label='imbalance (CLE-CRE)'),
                plt.Line2D([], [], color='#2C3E50', ls='--', label='session median'),
                mpatches.Patch(color='#C0392B', alpha=0.4, label='left-dominant'),
                mpatches.Patch(color='#2980B9', alpha=0.4, label='right-dominant'),
                plt.Line2D([], [], color='k', marker='|', ls='', label='motion')]
    fig.legend(handles=handles, loc='upper center', ncol=9, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle('Full-session capacitance imbalance (L-R slow mean) — all 12 sessions '
                 '(per-panel autoscale)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = PLOT_DIR / 'imbalance_full_all_sessions_grid.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'\nAutoscale grid -> {out}')

    # ── combined 6x2 grid, COMMON y-scale ──
    fig, axes = plt.subplots(6, 2, figsize=(20, 20), sharex=False, sharey=True)
    for ax, lbl in zip(axes.ravel(), sessions):
        g = df[df['session'] == lbl]
        _draw(ax, g, lbl, legend=False, ylim=COMMON_YLIM)
        ax.set_ylabel('imbalance (a.u.)', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    handles2 = list(handles) + [
        plt.Line2D([], [], color='#7f0000', marker='v', ls='',
                   label='clipped (off-scale artifact)')]
    fig.legend(handles=handles2, loc='upper center', ncol=9, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle('Full-session capacitance imbalance (L-R slow mean) — all 12 sessions '
                 '(COMMON y-scale, artifact spikes clipped)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    out2 = PLOT_DIR / 'imbalance_full_all_sessions_grid_commonY.png'
    fig.savefig(out2, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Common-Y grid -> {out2}')


if __name__ == '__main__':
    main()
