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

# from analysis/rates/outputs/k_vs_age_per_subject.csv
AGE = {'OS006': 25, 'OS003': 37, 'OS004': 54,
       'OS005': 55, 'OS001': 61, 'OS002': 66}

plt.rcParams.update({
    'font.size': 19, 'axes.titlesize': 21, 'axes.labelsize': 19,
    'xtick.labelsize': 17, 'ytick.labelsize': 18, 'legend.fontsize': 17,
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


def fig_fullnight(d):
    """One night, all three channels: the pipeline's output against the
    reference.

    A row per channel rather than three traces on one axis -- the estimates are
    noisy enough that overlaying them hides exactly the thing the figure is for,
    which is how much of the reference each channel follows.
    """
    # the recording with the most usable epochs, so the figure is not an
    # argument about coverage
    cov = (d[(d.channel == 'CRE') & d.peaks_loose.notna()]
           .groupby('session').size())
    sess = cov.idxmax()
    g = d[d.session == sess]

    fig, axes = plt.subplots(len(CHANNELS), len(BANDS), sharex=True,
                             figsize=(16.0, 11.0))
    for r, ch in enumerate(CHANNELS):
        for c, (band, title, unit) in enumerate(BANDS):
            ax = axes[r, c]
            b = g[(g.channel == ch) & (g.band == band)].sort_values('t_hr')
            k = (b.peaks_loose / b.gt_hz).clip(*CLIP).median()
            est = smooth(b.peaks_loose / k) * 60.0
            ref = smooth(b.gt_hz) * 60.0
            ax.plot(b.t_hr, ref, color='#111111', lw=2.4,
                    label='PSG reference')
            ax.plot(b.t_hr, est, color=CH_COLORS[ch], lw=1.9, alpha=0.9,
                    label=f'SEC {ch}  ÷ k = {k:.2f}')
            err = float(np.nanmedian(np.abs(est - ref)))
            ax.annotate(f'median |difference| = {err:.2f} {unit}',
                        (0.012, 0.055), xycoords='axes fraction', fontsize=13.5,
                        color='#444444')
            ax.grid(alpha=0.25)
            ax.legend(loc='upper right', ncol=2, fontsize=12.5)
            if r == 0:
                ax.set_title(f'{title}  ({unit})', loc='left',
                             fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{ch}\n({unit})')
            else:
                ax.set_ylabel(f'({unit})')
    for ax in axes[-1]:
        ax.set_xlabel('hours into the recording')
    fig.suptitle(f'One whole night, {sess}, all three channels\n'
                 'Both traces smoothed over 5 epochs (2.5 min); a single k per '
                 'channel and band for the whole recording',
                 fontsize=20, fontweight='bold', x=0.015, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = FIG / 'fig_rate_fullnight.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p.name, f'({sess}, {", ".join(CHANNELS)})')
    return sess


def fig_allnight_counts_and_k(d, sess=None):
    """What is counted, and the ratio it implies, across one night.

    Top row is the raw count -- peaks per minute, undivided -- against the
    reference, so the gap between them IS k, visible rather than asserted. The
    cardiac counts run at about twice the reference and the respiratory counts
    somewhat above it, which is the whole content of the calibration.

    Bottom row is that ratio epoch by epoch, with each channel's whole-night k
    drawn flat across it. How far the trace departs from its own flat line is
    what a single k per recording gets wrong.
    """
    if sess is None:
        sess = (d[(d.channel == 'CRE') & d.peaks_loose.notna()]
                .groupby('session').size().idxmax())
    g = d[d.session == sess]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0), sharex=True)
    for c, (band, title, unit) in enumerate(BANDS):
        top, bot = axes[0, c], axes[1, c]
        ref = None
        for ch in CHANNELS:
            b = g[(g.channel == ch) & (g.band == band)].sort_values('t_hr')
            if ref is None:
                ref = (b.t_hr.to_numpy(), (smooth(b.gt_hz) * 60.0).to_numpy())
            top.plot(b.t_hr, smooth(b.peaks_loose) * 60.0, lw=1.7,
                     color=CH_COLORS[ch], alpha=0.9, label=ch)

            ke_raw = (b.peaks_loose / b.gt_hz).clip(*CLIP)
            bot.plot(b.t_hr, smooth(ke_raw), lw=1.7, color=CH_COLORS[ch],
                     alpha=0.9, label=ch)
            bot.axhline(ke_raw.median(), color=CH_COLORS[ch], lw=1.6, ls='--',
                        alpha=0.85)

        top.plot(ref[0], ref[1], color='#111111', lw=2.8,
                 label='PSG reference', zorder=5)
        top.set_title(f'{title}', loc='left', fontweight='bold')
        top.set_ylabel(f'peaks counted per minute\n({unit} for the reference)')
        top.grid(alpha=0.25)
        top.legend(loc='upper right', ncol=4, fontsize=12.5)

        bot.axhline(1.0, color='#999999', ls=':', lw=1.6)
        bot.set_ylabel('per-epoch k\n(count ÷ reference)')
        bot.set_xlabel('hours into the recording')
        bot.grid(alpha=0.25)
        bot.legend(loc='upper right', ncol=3, fontsize=12.5)

    # kept short on each line: a long single-line suptitle is wider than the
    # canvas, and bbox_inches='tight' then grows the figure sideways to fit it
    fig.suptitle(f'What is counted, and the ratio it implies — {sess}\n'
                 'Top: raw peak count vs reference, undivided — the gap is k\n'
                 'Bottom: that ratio per epoch; dashed = the night’s single k',
                 fontsize=17, fontweight='bold', x=0.015, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = FIG / 'fig_rate_counts_and_k.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p.name, f'({sess})')


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
    subj_of = dict(zip(d.session, d.subject))
    axes[-1].set_xticks(range(len(sessions)))
    axes[-1].set_xticklabels(
        [f'{s}\n{AGE.get(subj_of.get(s), "?")} y' for s in sessions],
        rotation=0, ha='center')
    axes[-1].set_xlabel('recording   (two nights per participant, with age)')
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
    sess = fig_fullnight(d)
    fig_allnight_counts_and_k(d, sess)      # same night, so they read together
    fig_k_per_epoch(d)
    print(f'\n-> {TAB / "k_per_epoch_spread.csv"}')


if __name__ == '__main__':
    main()
