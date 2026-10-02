"""
Section 5 of the rate supplement: one k applied per epoch, and how far a
per-epoch k would have to move.

Replaces an earlier figure that drew every recording's running k on one time
axis and took a median across them. That was wrong: the twelve recordings are
six participants on two nights each, independent of one another, and nothing
lines up across them at "hour 3". Nights are reported separately here.

Two things are shown.

First, a whole night. The per-epoch peak count is converted to a rate with the
single k for that recording, smoothed over five epochs, and drawn against the
PSG reference on the same axis. This is what the pipeline actually outputs for a
night, so a reader can see directly how much of the reference it follows.

Second, what one k costs. A per-epoch k can be computed from the reference --
k(t) = count(t) / reference(t) -- which is not an estimator, because it uses
the answer, but it does say how much the ratio moves within a recording. Its
median and interquartile range are reported for every recording and channel.
A wide range means one fixed k cannot be right for most of the night.

Reads   artifacts/rate_rerun_phase_a.parquet
Writes  reports/rates/k_per_epoch_spread.csv
        writeup/figures/rate_supp/fig_rate_fullnight.png
        writeup/figures/rate_supp/fig_k_per_epoch.png

Usage
-----
    .venv/Scripts/python.exe analysis/rates/supp_rate_epoch_k.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sleep_monitor.sessions import SESSION_META      # noqa: E402

SRC = ROOT / 'artifacts' / 'rate_rerun_phase_a.parquet'
FIG = ROOT / 'writeup' / 'figures' / 'rate_supp'
TAB = ROOT / 'reports' / 'rates'
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
CH_COLORS = {'CH': '#1f4e79', 'CLE': '#2e9e5b', 'CRE': '#8e44ad'}
BANDS = [('resp', 'Breathing', 'breaths/min'),
         ('card', 'Heart rate', 'beats/min')]
CLIP = (0.3, 5.0)
SMOOTH_EPOCHS = 5          # 5 x 30 s = 2.5 min

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 15, 'legend.fontsize': 14,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.dpi': 200,
})


def load():
    subj = {m['label']: m['subject'] for m in SESSION_META}
    d = pd.read_parquet(SRC)
    d = d[d.channel.isin(CHANNELS)].copy()
    d['subject'] = d.session.map(subj)
    return d


def smooth(s):
    return s.rolling(SMOOTH_EPOCHS, center=True, min_periods=2).median()


def fig_fullnight(d, channel='CRE'):
    """One night: the pipeline's output against the reference."""
    # the recording with the most usable epochs, so the figure is not an
    # argument about coverage
    cov = (d[(d.channel == channel) & d.peaks_loose.notna()]
           .groupby('session').size())
    sess = cov.idxmax()
    g = d[(d.session == sess) & (d.channel == channel)]

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.4), sharex=True)
    for ax, (band, title, unit) in zip(axes, BANDS):
        b = g[g.band == band].sort_values('t_hr')
        k = (b.peaks_loose / b.gt_hz).clip(*CLIP).median()
        est = smooth(b.peaks_loose / k) * 60.0
        ref = smooth(b.gt_hz) * 60.0
        ax.plot(b.t_hr, ref, color='#111111', lw=2.6, label='PSG reference')
        ax.plot(b.t_hr, est, color=CH_COLORS[channel], lw=2.0, alpha=0.9,
                label=f'SEC, peaks ÷ k  (k = {k:.2f})')
        ax.set_ylabel(f'{title}\n({unit})')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', ncol=2)
        err = float(np.nanmedian(np.abs(est - ref)))
        ax.annotate(f'median |difference| = {err:.2f} {unit}',
                    (0.012, 0.06), xycoords='axes fraction', fontsize=15,
                    color='#444444')
    axes[-1].set_xlabel('hours into the recording')
    fig.suptitle(f'One whole night, {sess}, channel {channel}\n'
                 'Both traces smoothed over 5 epochs (2.5 min); a single k per '
                 'band for the whole recording',
                 fontsize=20, fontweight='bold', x=0.015, ha='left', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p = FIG / 'fig_rate_fullnight.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p.name, f'({sess}, {channel})')
    return sess


def fig_k_per_epoch(d):
    """Per-epoch k, reported per recording and channel -- never pooled."""
    sessions = sorted(d.session.unique())
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(16.5, 9.6), sharex=True)
    for ax, (band, title, _) in zip(axes, BANDS):
        for xi, sess in enumerate(sessions):
            for ci, ch in enumerate(CHANNELS):
                g = d[(d.session == sess) & (d.channel == ch)
                      & (d.band == band)].sort_values('t_hr')
                ke = smooth((g.peaks_loose / g.gt_hz).clip(*CLIP)).dropna()
                if len(ke) < 20:
                    continue
                q1, med_, q3 = ke.quantile([.25, .5, .75])
                x = xi + (ci - 1) * 0.26
                ax.vlines(x, q1, q3, color=CH_COLORS[ch], lw=5.0, alpha=0.75)
                ax.plot(x, med_, 'o', ms=8, color=CH_COLORS[ch],
                        markeredgecolor='white', markeredgewidth=1.2, zorder=4)
                rows.append(dict(session=sess, subject=g.subject.iloc[0],
                                 band=band, channel=ch, k_median=med_,
                                 k_q1=q1, k_q3=q3, iqr=q3 - q1,
                                 n_epochs=int(len(ke))))
        ax.axhline(1.0, color='#999999', ls=':', lw=1.6)
        ax.set_ylabel('per-epoch k')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
    axes[-1].set_xticks(range(len(sessions)))
    axes[-1].set_xticklabels(sessions, rotation=45, ha='right')
    axes[-1].set_xlabel('recording   (two nights per participant)')
    handles = [plt.Line2D([], [], color=CH_COLORS[c], lw=5, label=c)
               for c in CHANNELS]
    axes[0].legend(handles=handles, loc='upper right', ncol=3)
    fig.suptitle('How much does k move within a recording?\n'
                 'Point is the median per-epoch k, bar is its interquartile '
                 'range · each recording stands alone, nothing is pooled '
                 'across nights',
                 fontsize=20, fontweight='bold', x=0.015, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    p = FIG / 'fig_k_per_epoch.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p.name)

    t = pd.DataFrame(rows)
    t.to_csv(TAB / 'k_per_epoch_spread.csv', index=False)
    print('\nper-epoch k, interquartile width (median across the 12 recordings):')
    for band, title, _ in BANDS:
        s = t[t.band == band].groupby('channel')['iqr'].median()
        print(f'  {title:11s} ' + '   '.join(f'{c} {s[c]:.2f}' for c in CHANNELS))
    return t


def main():
    d = load()
    fig_fullnight(d)
    fig_k_per_epoch(d)
    print(f'\n-> {TAB / "k_per_epoch_spread.csv"}')


if __name__ == '__main__':
    main()
