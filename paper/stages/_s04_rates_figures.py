"""Figures S7–S12 for stage s04_rates (the supplementary rate section).

Drawing only: every number drawn is computed in s04_rates.py and passed in. Large
bold type throughout, as the legacy supplement figures used at the reviewer's
request.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from seclib import SESSION_META
from seclib.figures import save
from stages.s04_rates import CHANNELS, k_clipped, night_traces, smooth

BIG = {'font.size': 21, 'axes.titlesize': 23, 'axes.labelsize': 21,
       'xtick.labelsize': 19, 'ytick.labelsize': 19, 'legend.fontsize': 19,
       'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
       'axes.spines.top': False, 'axes.spines.right': False}
CH_COLORS = {'CH': '#1f4e79', 'CLE': '#2e9e5b', 'CRE': '#8e44ad'}
SUBJ_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#d62728']
BANDS_LBL = [('resp', 'Breathing', 'breaths/min'), ('card', 'Heart rate', 'beats/min')]
BLUE, GREY, RED, GREEN = '#2e75b6', '#6b6b6b', '#c0392b', '#1b7a43'


def fig_s7_pipeline(ex, stage):
    """Fig. S7: one minute of S2N1 CRE through the pipeline, divided by S2N1's own k."""
    with plt.rc_context(BIG):
        fig, ax = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True,
                               gridspec_kw={'height_ratios': [1, 1.25]})
        ax[0].plot(ex['t'], ex['raw'], color=GREY, lw=1.4)
        ax[0].set_ylabel('CRE, raw\n(fF)')
        ax[0].set_title('Step 1-2.  One channel (CRE), band-passed to the '
                        'respiratory band 0.1–0.5 Hz', loc='left')
        ax[1].plot(ex['t'], ex['bp'], color=BLUE, lw=2.2)
        ax[1].plot(ex['t'][ex['pk']], ex['bp'][ex['pk']], 'v', ms=11, color=RED, zorder=5)
        ax[1].axhline(0, color=GREY, lw=0.8)
        ax[1].set_ylabel('CRE, filtered\n(fF)')
        ax[1].set_xlabel('seconds')
        ax[1].set_title('Step 3.  Count peaks — loose detector (prominence 0.05σ)',
                        loc='left')
        # CORRECTION (1): the recording's own k, not the cohort median
        txt = (f"Step 4–5.   {ex['n']} peaks  →  {ex['raw_rate']:.1f} per min  ÷  "
               f"k = {ex['k_own']:.2f}  →  {ex['rate_own']:.1f} breaths/min\n"
               f"k is {ex['session']}'s own CRE k; "
               f"PSG reference over this minute {ex['ref']:.1f} breaths/min")
        fig.text(0.5, 0.015, txt, ha='center', va='bottom',
                 fontsize=18, color='#1b2a41',
                 bbox=dict(boxstyle='round,pad=0.5', fc='#eef3f9', ec='#b9c8da'))
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        save(fig, 'fig_S7_rate_pipeline', stage)


def fig_s8_k_by_channel(k, age, stage):
    """Fig. S8: k per night, channel, detector and band; one point per night."""
    methods = [('peaks_loose', 'Peak counting'), ('spectral', 'Spectral peak')]
    with plt.rc_context({**BIG, 'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 22,
                         'xtick.labelsize': 21, 'ytick.labelsize': 21}):
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.6), sharex=True)
        subjects = sorted(k.subject.unique(), key=lambda s: age.get(s, 999))
        cmap = dict(zip(subjects, SUBJ_COLORS))
        for r, (band, band_lbl, _) in enumerate(BANDS_LBL):
            for c, (meth, meth_lbl) in enumerate(methods):
                ax = axes[r, c]
                sub = k[(k.band == band) & (k.method == meth)]
                for xi, ch in enumerate(CHANNELS):
                    s = sub[sub.channel == ch]
                    jit = np.linspace(-0.22, 0.22, len(s))
                    ax.scatter(xi + jit, s.k, s=95, c=[cmap[x] for x in s.subject],
                               edgecolor='white', linewidth=1.0, zorder=3)
                    med = s.k.median()
                    ax.plot([xi - 0.27, xi + 0.27], [med] * 2, color='black', lw=3.0, zorder=4)
                    ax.annotate(f'{med:.2f}', (xi + 0.29, med), ha='left', va='center',
                                fontsize=16, fontweight='bold', zorder=5)
                ax.axhline(1.0, color='#999999', lw=1.6, ls=':', zorder=1)
                ax.set_xticks(range(len(CHANNELS)))
                ax.set_xticklabels(CHANNELS)
                ax.set_xlim(-0.55, len(CHANNELS) - 1 + 0.92)
                ax.set_ylabel('k' if c == 0 else '')
                ax.set_title(f'{band_lbl}  —  {meth_lbl}', loc='left')
                ax.grid(axis='y', alpha=0.25)
                lo, hi = sub.k.min(), sub.k.max()
                pad = 0.18 * max(hi - lo, 0.4)
                ax.set_ylim(min(lo, 0.85) - pad, hi + pad * 1.6)
        handles = [plt.Line2D([], [], marker='o', ls='', ms=13, color=cmap[s],
                              label=f'{age[s]} y') for s in subjects]
        fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
                   bbox_to_anchor=(0.5, -0.012), title='participant age', title_fontsize=17)
        fig.tight_layout(rect=(0, 0.075, 1, 1))
        save(fig, 'fig_S8_k_by_channel', stage)


def fig_s9_counts_and_k(d, sess, stage):
    """Fig. S9: raw counts per minute against the reference, and the implied per-epoch k."""
    g = d[d.session == sess]
    with plt.rc_context({**BIG, 'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 22,
                         'xtick.labelsize': 20, 'ytick.labelsize': 21}):
        fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.0), sharex=True)
        for c, (band, title, unit) in enumerate(BANDS_LBL):
            top, bot = axes[0, c], axes[1, c]
            ref = None
            for ch in CHANNELS:
                b = g[(g.channel == ch) & (g.band == band)].sort_values('t_hr')
                if ref is None:
                    ref = (b.t_hr.to_numpy(), (smooth(b.gt_hz) * 60.0).to_numpy())
                top.plot(b.t_hr, smooth(b.peaks_loose) * 60.0, lw=1.7, color=CH_COLORS[ch], alpha=0.9, label=ch)
                ke_raw = k_clipped(b.peaks_loose, b.gt_hz)
                bot.plot(b.t_hr, smooth(ke_raw), lw=1.7, color=CH_COLORS[ch], alpha=0.9,
                         label=ch)
                bot.axhline(ke_raw.median(), color=CH_COLORS[ch], lw=1.6, ls='--', alpha=0.85)
            top.plot(ref[0], ref[1], color='#111111', lw=2.8, label='PSG reference', zorder=5)
            top.set_title(title, loc='left')
            top.set_ylabel(f'peaks per minute\n(reference: {unit})')
            top.grid(alpha=0.25)
            top.legend(loc='upper right', ncol=4, fontsize=12.5)
            bot.axhline(1.0, color='#999999', ls=':', lw=1.6)
            bot.set_ylabel('per-epoch k\n(count ÷ reference)')
            bot.set_xlabel('hours into the recording')
            bot.grid(alpha=0.25)
            bot.legend(loc='upper right', ncol=3, fontsize=12.5)
        fig.tight_layout()
        save(fig, 'fig_S9_counts_and_k', stage)



def fig_s10_allsessions(d, age, stage):
    """Fig. S10: every night, both bands, each channel divided by its own k."""
    subj = {m['label']: m['subject'] for m in SESSION_META}
    sessions = sorted(d.session.unique())
    with plt.rc_context({**BIG, 'font.size': 18, 'axes.titlesize': 19, 'axes.labelsize': 18,
                         'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16}):
        fig, axes = plt.subplots(len(sessions), 2, figsize=(17.0, 2.35 * len(sessions)))
        for r, sess in enumerate(sessions):
            for c, (band, title, unit) in enumerate(BANDS_LBL):
                ax = axes[r, c]
                notes = []
                for i, ch in enumerate(CHANNELS):
                    b, _, est, ref = night_traces(d, sess, band, ch)
                    if i == 0:
                        ax.plot(b.t_hr, ref, color='#111111', lw=2.2, zorder=5,
                                label='PSG reference')
                    ax.plot(b.t_hr, est, color=CH_COLORS[ch], lw=1.3, alpha=0.85, label=ch)
                    notes.append(f'{ch} {np.nanmedian(np.abs(est - ref)):.2f}')
                ax.annotate('  '.join(notes), (0.012, 0.045), xycoords='axes fraction',
                            fontsize=11, color='#555555')
                ax.grid(alpha=0.22)
                ax.set_ylabel(f'{sess}  ({age[subj[sess]]} y)\n{unit}' if c == 0 else unit)
                if r == 0:
                    ax.set_title(title, loc='left')
                    ax.legend(loc='upper right', ncol=4, fontsize=11)
                if r == len(sessions) - 1:
                    ax.set_xlabel('hours into the recording')
        fig.tight_layout()
        save(fig, 'fig_S10_rate_allsessions', stage)


def fig_s11_calibration(per_session, stage):
    """Fig. S11: night-level error under each calibration against the no-sensor baseline."""
    strategies = [('night_self', 'Same night\n(calibrated on the\nnight being reported)'),
                  ('night_cross', 'The same person,\nother night'),
                  ('night_pop', 'Everyone else\n(population)')]
    with plt.rc_context(BIG):
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0))
        for ax, (band, title, unit) in zip(axes, BANDS_LBL):
            p = per_session[per_session.band == band]
            vals = [float(np.median(p[k])) for k, _ in strategies]
            base = float(np.median(p.night_nosensor))
            xs = np.arange(len(vals))
            ax.bar(xs, vals, 0.62, color=[GREEN if v < base else RED for v in vals],
                   edgecolor='black', lw=0.8, zorder=3)
            for x, v in zip(xs, vals):
                near = abs(v - base) < 0.15 * base      # inside the bar, clear of the line
                ax.annotate(f'{v:.2f}', (x, v), xytext=(0, -8 if near else 7),
                            textcoords='offset points', ha='center',
                            va='top' if near else 'bottom', fontsize=17, fontweight='bold',
                            color='white' if near else 'black')
            ax.axhline(base, color='black', lw=2.4, ls='--', zorder=4)
            ax.annotate(f'no sensor at all: {base:.2f}', (0.99, base),
                        xycoords=('axes fraction', 'data'), xytext=(0, 8),
                        textcoords='offset points', ha='right', va='bottom', fontsize=16,
                        fontweight='bold')
            ax.set_xticks(xs)
            ax.set_xticklabels([lbl for _, lbl in strategies], fontsize=14)
            ax.set_ylabel(f'error in the night average\n({unit})')
            ax.set_title(title, loc='left')
            ax.set_ylim(0, max(max(vals), base) * 1.32)
            ax.grid(axis='y', alpha=0.25, zorder=0)
        fig.tight_layout()
        save(fig, 'fig_S11_rate_calibration', stage)


def fig_s12_k_per_epoch(spread, age, stage):
    """Fig. S12: median per-epoch k and its IQR, per night and channel -- never pooled."""
    sessions = sorted(spread.session.unique())
    subj_of = dict(zip(spread.session, spread.subject))
    with plt.rc_context({**BIG, 'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 22,
                         'xtick.labelsize': 20, 'ytick.labelsize': 21}):
        fig, axes = plt.subplots(2, 1, figsize=(16.5, 9.6), sharex=True)
        for ax, (band, title, _) in zip(axes, BANDS_LBL):
            for xi, sess in enumerate(sessions):
                for ci, ch in enumerate(CHANNELS):
                    row = spread[(spread.session == sess) & (spread.channel == ch)
                                 & (spread.band == band)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    x = xi + (ci - 1) * 0.26
                    ax.vlines(x, row.k_q1, row.k_q3, color=CH_COLORS[ch], lw=5.0, alpha=0.75)
                    ax.plot(x, row.k_median, 'o', ms=8, color=CH_COLORS[ch],
                            markeredgecolor='white', markeredgewidth=1.2, zorder=4)
            ax.axhline(1.0, color='#999999', ls=':', lw=1.6)
            ax.set_ylabel('per-epoch k')
            ax.set_title(title, loc='left')
            ax.grid(axis='y', alpha=0.25)
        axes[-1].set_xticks(range(len(sessions)))
        axes[-1].set_xticklabels([f'{s}\n{age[subj_of[s]]} y' for s in sessions],
                                 rotation=0, ha='center')
        axes[-1].set_xlabel('recording   (two nights per participant, with age)')
        handles = [plt.Line2D([], [], color=CH_COLORS[c], lw=5, label=c) for c in CHANNELS]
        axes[0].legend(handles=handles, loc='upper right', ncol=3)
        fig.tight_layout()
        save(fig, 'fig_S12_k_per_epoch', stage)
