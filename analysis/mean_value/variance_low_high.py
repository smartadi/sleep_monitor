"""
Both tails of the capacitive variance, on all three channels.

high_variance_zones.py asks where the TOP decile of epoch variance falls, on CH
only, and finds Wake and REM over-represented and N3 almost empty. Two things
were left open:

1. The same question on CLE and CRE, the single-ended channels, so the result
   is not a property of the CH configuration alone.
2. The other tail. If high variance marks arousal and almost never lands in
   N3, the quiet epochs -- the BOTTOM decile -- are where N3 should be, if
   anywhere. That is the low-variance question.

Same quantity as before: per 30 s epoch, variance of the <10 Hz signal; a
decile is taken within each recording, so every night is on its own scale.
Motion epochs (top decile of accelerometer SD) are removed for the headline
figure, because a moving subject is high-variance and not asleep.

Descriptive and per-night only: one enrichment (observed / expected occupancy)
per night and stage, the median of the twelve drawn as a bar, and a count of
how many nights go each way. Nothing pooled, no p-values.

Reads   reports/mean_value/high_variance_epochs.parquet   (high_variance_zones.py)
Writes  reports/mean_value/variance_tails_enrichment.csv
        writeup/figures/mean_value/variance_tails_enrichment.png
        writeup/figures/mean_value/variance_tails_traces_{CH,CLE,CRE}.png

Usage
    .venv/Scripts/python.exe analysis/mean_value/variance_low_high.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec   # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sleep_monitor.config import STAGE_LADDER, STAGE_COLORS, STAGE_ORDER  # noqa: E402

SRC = ROOT / 'reports' / 'mean_value' / 'high_variance_epochs.parquet'
OUT = ROOT / 'reports' / 'mean_value'
FIG = ROOT / 'writeup' / 'figures' / 'mean_value'

CHANNELS = ['CH', 'CLE', 'CRE']
STAGES = list(STAGE_LADDER)                     # Wake, N1, N2, N3, REM
STAGE_COLOR = {n: STAGE_COLORS[c] for n, c in zip(STAGE_LADDER, STAGE_ORDER)}
YPOS = {s: -i for i, s in enumerate(STAGES)}
HI_Q, LO_Q = 0.90, 0.10
DEAD_FF2 = 1e-4          # epoch variance below this = dead channel, dropped
HI_COLOR, LO_COLOR = '#C0392B', '#1E8449'

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False})


def tails(d):
    # 14 epochs of S4N1 (mostly the final Wake minutes) have variance ~1e-20:
    # the sensor stopped, it did not go quiet. Left in, they fill the bottom
    # decile of that night with "low variance" that is a dead channel.
    dead = (d[['var_' + c for c in CHANNELS]] < DEAD_FF2).any(axis=1)
    d = d[~dead].copy()
    for ch in CHANNELS:
        v = d.groupby('session')['var_' + ch]
        d['hi_' + ch] = d['var_' + ch] > v.transform(lambda x: x.quantile(HI_Q))
        d['lo_' + ch] = d['var_' + ch] < v.transform(lambda x: x.quantile(LO_Q))
    return d


def enrichment(d):
    rows = []
    for subset, sub in (('all epochs', d), ('motion-free', d[~d.motion])):
        for ch in CHANNELS:
            for tail in ('hi', 'lo'):
                for sess, g in sub.groupby('session'):
                    sel = g[g[f'{tail}_{ch}']]
                    if len(sel) < 10:
                        continue
                    for st in STAGES:
                        exp = (g.stage == st).mean()
                        if exp < 0.01:
                            continue
                        obs = (sel.stage == st).mean()
                        rows.append(dict(subset=subset, channel=ch, tail=tail,
                                         session=sess, stage=st, n_tail=len(sel),
                                         observed=obs, expected=exp,
                                         enrichment=obs / exp))
    return pd.DataFrame(rows)


def fig_enrichment(e):
    sub = e[e.subset == 'motion-free']
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    floor = 0.03                           # zero enrichment drawn at the floor
    for r, tail in enumerate(('hi', 'lo')):
        for c, ch in enumerate(CHANNELS):
            ax = axes[r, c]
            g = sub[(sub['tail'] == tail) & (sub.channel == ch)]
            for i, st in enumerate(STAGES):
                v = g[g.stage == st].enrichment.to_numpy()
                if not len(v):
                    continue
                x = i + np.random.default_rng(i).uniform(-0.17, 0.17, len(v))
                ax.scatter(x, np.maximum(v, floor), s=34, color=STAGE_COLOR[st],
                           edgecolor='k', lw=0.4, alpha=0.85, zorder=3)
                ax.hlines(max(np.median(v), floor), i - 0.32, i + 0.32,
                          color='#2C3E50', lw=3, zorder=4)
                up = int((v > 1).sum())
                ax.annotate(f'{up}/{len(v)}', (i, 9.0), ha='center', fontsize=11,
                            color='#2C3E50')
            ax.axhline(1, color='k', ls='--', lw=1)
            ax.set_yscale('log')
            ax.set_ylim(floor * 0.8, 14)
            ax.set_xticks(range(len(STAGES)))
            ax.set_xticklabels(STAGES)
            ax.grid(axis='y', alpha=0.25, which='both')
            name = 'top decile (high variance)' if tail == 'hi' else \
                'bottom decile (low variance)'
            ax.set_title(f'{ch} — {name}', loc='left',
                         color=HI_COLOR if tail == 'hi' else LO_COLOR)
        axes[r, 0].set_ylabel('enrichment\n(observed / expected)')
    fig.text(0.5, 0.005, 'Motion-free epochs. One point per night; bar = median of '
             'the nights; k/n above each stage = nights enriched (>1) of nights with that stage. '
             'Points at the floor had none.', ha='center', fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = FIG / 'variance_tails_enrichment.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_traces(d, ch):
    """Both tails on the variance trace, two columns of six, hypnogram above."""
    sessions = sorted(d.session.unique())
    ncols, per_col = 2, 6
    fig = plt.figure(figsize=(27, 1.75 * per_col))
    gs = gridspec.GridSpec(2 * per_col, ncols, figure=fig,
                           height_ratios=[0.42, 1.0] * per_col, hspace=0.0, wspace=0.13)
    tmax = d.t_hr.max()
    for i, sname in enumerate(sessions):
        col, row = divmod(i, per_col)
        g = d[d.session == sname].sort_values('t_hr')
        v = g['var_' + ch].to_numpy(float)
        hi_t, lo_t = np.quantile(v, HI_Q), np.quantile(v, LO_Q)

        axh = fig.add_subplot(gs[2 * row, col])
        y = g.stage.map(YPOS).to_numpy(float)
        axh.step(g.t_hr, y, where='mid', lw=1.0, color='#2C3E50')
        for st in STAGES:
            m = (g.stage == st).to_numpy()
            axh.plot(g.t_hr[m], np.full(m.sum(), YPOS[st]), '|',
                     color=STAGE_COLOR[st], ms=4, mew=2.0)
        axh.set_yticks([YPOS[s] for s in STAGES])
        axh.set_yticklabels(STAGES, fontsize=7)
        axh.set_ylim(-4.6, 0.6)
        axh.set_xlim(0, tmax)
        axh.set_xticks([])
        axh.tick_params(axis='y', length=0)
        for side in ('top', 'right', 'bottom'):
            axh.spines[side].set_visible(False)
        axh.set_ylabel(sname, rotation=0, ha='right', va='center', fontsize=11,
                       labelpad=28, fontweight='bold')

        axv = fig.add_subplot(gs[2 * row + 1, col])
        axv.plot(g.t_hr, v, lw=0.7, color='#34495E', alpha=0.9, zorder=2)
        axv.axhline(hi_t, color=HI_COLOR, ls='--', lw=0.9)
        axv.axhline(lo_t, color=LO_COLOR, ls='--', lw=0.9)
        still = ~g.motion.to_numpy()
        for tail_m, colr in ((v > hi_t, HI_COLOR), (v < lo_t, LO_COLOR)):
            axv.scatter(g.t_hr[tail_m & still], v[tail_m & still], s=14, color=colr,
                        edgecolor='none', zorder=5)
            axv.scatter(g.t_hr[tail_m & ~still], v[tail_m & ~still], s=12,
                        color='#BDC3C7', edgecolor='none', zorder=4)
        axv.set_yscale('log')
        pos = v[v > 0]
        if len(pos):
            axv.set_ylim(np.percentile(pos, 1) / 3.0, pos.max() * 2.0)
        axv.set_xlim(0, tmax)
        axv.set_ylabel('var (fF$^2$)', fontsize=9)
        axv.tick_params(labelsize=8)
        axv.grid(alpha=0.2, lw=0.4)
        if row < per_col - 1:
            axv.set_xticklabels([])
        else:
            axv.set_xlabel('time (hours)', fontsize=12)
    handles = [plt.Line2D([], [], marker='o', ls='none', ms=6, color=HI_COLOR,
                          label='top decile, still'),
               plt.Line2D([], [], marker='o', ls='none', ms=6, color=LO_COLOR,
                          label='bottom decile, still'),
               plt.Line2D([], [], marker='o', ls='none', ms=6, color='#BDC3C7',
                          label='either tail, moving')]
    fig.legend(handles=handles, fontsize=12, ncol=3, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5, 0.93))
    out = FIG / f'variance_tails_traces_{ch}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    d = tails(pd.read_parquet(SRC))
    e = enrichment(d)
    e.to_csv(OUT / 'variance_tails_enrichment.csv', index=False)
    print('  ->', fig_enrichment(e).name)
    for ch in CHANNELS:
        print('  ->', fig_traces(d, ch).name)
    sub = e[e.subset == 'motion-free']
    print('\nmotion-free enrichment, median over nights [nights >1 / n]:')
    for tail in ('hi', 'lo'):
        for ch in CHANNELS:
            parts = []
            for st in STAGES:
                v = sub[(sub['tail'] == tail) & (sub.channel == ch) & (sub.stage == st)].enrichment
                parts.append(f'{st} {v.median():4.2f} [{(v > 1).sum()}/{len(v)}]')
            print(f'  {tail} {ch:3s}  ' + '  '.join(parts))


if __name__ == '__main__':
    main()
