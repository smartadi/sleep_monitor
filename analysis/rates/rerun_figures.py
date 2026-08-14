#!/usr/bin/env python
"""
Paper figures for the recomputed rate analysis (2026-08-14).

Three figures, all built from reports/rates/rerun/ and the per-epoch parquet:

  fig_rate_calibration.png   main text — error under four calibration regimes,
                             both bands, night- and epoch-level, paired by night
  fig_rate_estimators.png    supplementary — accuracy against variability, which
                             is what exposes a degenerate estimator
  fig_rate_within_night.png  supplementary — within-night correlation per night
                             against a circular-shift null

Form follows the data's job. The calibration comparison is a repeated measure —
the same twelve nights under four conditions — so it is drawn as a paired dot
plot with per-night connecting lines, not as bars of group means, which would
hide that eight of twelve nights move one way and four the other. The estimator
panel is a scatter of error against variability because neither axis alone
separates a real estimator from a constant.

Colour does one job: identity of the calibration regime. Three categorical hues
(slots 1-3 of the reference palette, the subset validated for all-pairs use) for
the three sensor-based regimes, and neutral grey for the no-sensor comparator,
which is a reference rather than a series.

Run from the repo root:  python analysis/rates/rerun_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RPT = ROOT / 'reports' / 'rates' / 'rerun'
ART = ROOT / 'artifacts' / 'rate_rerun_phase_a.parquet'
FIG = ROOT / 'writeup' / 'figures' / 'rate_rerun'
FIG.mkdir(parents=True, exist_ok=True)

# reference palette, categorical slots 1-3 (validated for all-pairs use) + neutrals
C_SELF, C_CROSS, C_POP = '#2a78d6', '#eb6834', '#1baf7a'
C_NULL, C_INK, C_MUTED, C_FAINT = '#8a8880', '#0b0b0b', '#52514e', '#d8d6cf'
UNIT = {'resp': 'br/min', 'card': 'BPM'}
BAND_NAME = {'resp': 'Respiratory', 'card': 'Cardiac'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': C_INK, 'text.color': C_INK,
    'xtick.color': C_MUTED, 'ytick.color': C_MUTED,
    'axes.edgecolor': C_MUTED,
    'legend.frameon': False,
    'figure.dpi': 200, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
})

MM = 1 / 25.4


def panel_letter(ax, letter, dx=-0.13, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='left', color=C_INK)


# ─────────────────────────────────────────────── 1. calibration regimes

def fig_calibration(d: pd.DataFrame) -> None:
    regimes = [('self', 'k from\nsame night', C_SELF),
               ('cross', 'k from\nother night', C_CROSS),
               ('pop', 'population k\n(held out)', C_POP),
               ('nosensor', 'no sensor', C_NULL)]

    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 115 * MM))
    for row, band in enumerate(['resp', 'card']):
        g = d[d.band == band]
        for col, level in enumerate(['night', 'epoch']):
            ax = axes[row, col]
            cols = [f'{level}_{key}' for key, _, _ in regimes]
            vals = g[cols].values                      # nights x regimes
            xs = np.arange(len(regimes))

            # per-night paths: the repeated measure is the point of the figure
            for r in vals:
                ax.plot(xs, r, '-', color=C_FAINT, lw=0.5, zorder=1,
                        solid_capstyle='round')
            for j, (key, _, colour) in enumerate(regimes):
                v = vals[:, j]
                v = v[np.isfinite(v)]
                jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.012
                ax.plot(np.full(len(v), xs[j]) + jit, v, 'o', ms=3.4,
                        mfc=colour, mec='white', mew=0.5, zorder=3)
                ax.plot([xs[j] - 0.26, xs[j] + 0.26], [np.median(v)] * 2,
                        '-', color=colour, lw=1.8, zorder=4, solid_capstyle='round')

            # the no-sensor median as the line every sensor regime must beat
            base = np.nanmedian(vals[:, 3])
            ax.axhline(base, color=C_NULL, lw=0.7, ls=(0, (3, 2)), zorder=2)

            ax.set_xticks(xs)
            ax.set_xticklabels([lab for _, lab, _ in regimes], color=C_MUTED)
            ax.set_xlim(-0.55, len(regimes) - 0.45)
            ax.set_ylabel(f'|error|  ({UNIT[band]})')
            ax.set_title(f'{BAND_NAME[band]} — '
                         + ('night mean rate' if level == 'night'
                            else 'per-epoch rate'), color=C_INK, pad=6)
            panel_letter(ax, 'ABCD'[row * 2 + col])

            # The cardiac error spans 1-57 BPM because the two S6 nights are
            # coupling outliers. On a linear axis they flatten the other ten
            # nights onto the floor, so that row is drawn in log.
            finite = vals[np.isfinite(vals)]
            if band == 'card':
                ax.set_yscale('log')
                ax.set_ylim(0.6 * finite.min(), 1.9 * finite.max())
                ax.set_ylabel(f'|error|  ({UNIT[band]}, log)')
            else:
                ax.set_ylim(0, 1.12 * finite.max())

            # label the comparator once per panel, clear of the marks
            ax.text(0.015, base, ' no sensor', transform=ax.get_yaxis_transform(),
                    va='bottom', ha='left', fontsize=6, color=C_NULL)

    fig.text(0.5, -0.015,
             'Each point is one of 12 recordings; grey paths connect the same night '
             'across regimes. Bars are medians; the dashed line is the no-sensor median.',
             ha='center', fontsize=6.5, color=C_MUTED)
    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    out = FIG / 'fig_rate_calibration.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  {out}')


# ─────────────────────────────────────────── 2. accuracy against variability

def fig_estimators(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 78 * MM))
    marks = {'spectral': ('X', '#e34948', 'spectral (4 s Welch)'),
             'spectral_interp': ('s', C_SELF, 'spectral (full window)'),
             'peaks_loose': ('o', C_POP, 'peaks, loose'),
             'peaks_strict': ('^', C_CROSS, 'peaks, strict'),
             'hilbert': ('D', '#4a3aa7', 'Hilbert')}

    for ax, band in zip(axes, ['resp', 'card']):
        t = tab[tab.band == band]
        ref_sd = t.ref_sd.median()
        ax.axvspan(0, 0.5, color=C_FAINT, alpha=0.45, lw=0, zorder=0)
        ax.axvline(1.0, color=C_MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for est, (mk, colour, label) in marks.items():
            s = t[t.estimator == est]
            if not len(s):
                continue
            ax.plot(s.raw_sd / ref_sd, s.epoch_self, mk, ms=4.2, mfc=colour,
                    mec='white', mew=0.5, label=label, zorder=3, ls='none')
        ax.set_xlabel('estimator SD ÷ reference SD')
        ax.set_ylabel(f'per-epoch |error|  ({UNIT[band]})')
        ax.set_title(f'{BAND_NAME[band]}', color=C_INK, pad=6)
        ax.set_xlim(left=-0.05)
        ax.set_ylim(bottom=0)
        panel_letter(ax, 'AB'[list(['resp', 'card']).index(band)])

    axes[0].annotate('returns a constant:\nlowest error, no information',
                     xy=(0.02, 0.95), xytext=(0.55, 2.6), fontsize=6.5,
                     color='#e34948',
                     arrowprops=dict(arrowstyle='->', color='#e34948', lw=0.7,
                                     shrinkA=0, shrinkB=3))
    axes[0].text(0.25, 0.03, 'less variable\nthan physiology', fontsize=6,
                 color=C_MUTED, ha='center', transform=axes[0].get_xaxis_transform())
    # lower right is the empty quadrant in the cardiac panel; upper right holds
    # the high-variance spectral points
    axes[1].legend(loc='lower right', handletextpad=0.3, borderpad=0.2,
                   labelspacing=0.35)
    fig.text(0.5, -0.02,
             'Each marker is one estimator on one channel. An estimator that varies '
             'far less than the reference cannot be tracking it, whatever its error.',
             ha='center', fontsize=6.5, color=C_MUTED)
    fig.tight_layout(w_pad=2.6)
    out = FIG / 'fig_rate_estimators.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  {out}')


# ──────────────────────────────────────────── 3. within-night correlation

def within_night_null(df, band, channel, est, n_iter=200, seed=0):
    """Observed within-night r per night, and a circular-shift null."""
    rng = np.random.default_rng(seed)
    g0 = df[(df.band == band) & (df.channel == channel)].dropna(subset=['gt_hz'])
    obs, null = {}, []
    for s, g in g0.groupby('session'):
        g = g.sort_values('epoch')
        raw, gt = g[est].values * 60.0, g.gt_hz.values * 60.0
        m = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
        raw, gt = raw[m], gt[m]
        if len(gt) < 20 or np.std(raw) < 1e-9:
            continue
        obs[s] = np.corrcoef(raw, gt)[0, 1]
        for _ in range(n_iter):
            k = int(rng.integers(1, len(raw)))
            null.append(np.corrcoef(np.roll(raw, k), gt)[0, 1])
    return obs, np.asarray(null)


def fig_within_night(df, choice) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 72 * MM))
    above = {}
    for ax, band in zip(axes, ['resp', 'card']):
        est, ch = choice[band]
        obs, null = within_night_null(df, band, ch, est)
        lo, hi = np.percentile(null, [2.5, 97.5])
        names = sorted(obs)
        vals = np.array([obs[s] for s in names])
        ys = np.arange(len(names))
        above[band] = (int((vals > hi).sum()), len(vals),
                       [n for n, v in zip(names, vals) if v > hi])

        ax.axvspan(lo, hi, color=C_FAINT, alpha=0.55, lw=0, zorder=0)
        ax.axvline(0, color=C_MUTED, lw=0.6, zorder=1)
        inside = (vals >= lo) & (vals <= hi)
        ax.barh(ys, vals, height=0.55, zorder=2, linewidth=0,
                color=np.where(inside, C_NULL, C_SELF))
        ax.set_yticks(ys)
        ax.set_yticklabels(names)
        ax.set_ylim(len(names) - 0.4, -1.3)          # headroom for the null label
        ax.set_xlabel('within-night r  (estimate vs reference)')
        ax.set_xlim(-0.75, 0.75)
        ax.set_title(f'{BAND_NAME[band]} — {est} on {ch}', color=C_INK, pad=10)
        ax.text(0.5 * (lo + hi), -1.05, '95% of circular-shift null',
                ha='center', va='center', fontsize=6, color=C_MUTED)
        panel_letter(ax, 'AB'[list(['resp', 'card']).index(band)], dx=-0.20)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    nr, tr, _ = above['resp']
    nc, tc, who = above['card']
    note = (f'{nr}/{tr} respiratory and {nc}/{tc} cardiac nights exceed the null '
            f'on the positive side'
            + (f' ({", ".join(who)})' if who else '')
            + '. At 12 nights the null alone produces at least one such night '
              'about a quarter of the time.')
    fig.text(0.5, -0.03, note, ha='center', fontsize=6.5, color=C_MUTED)
    fig.tight_layout(w_pad=3.0)
    out = FIG / 'fig_rate_within_night.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  {out}')


def main() -> None:
    d = pd.read_csv(RPT / 'per_session.csv')
    tab = pd.read_csv(RPT / 'estimator_table.csv')
    choice = {r.band: (r.estimator, r.channel)
              for r in pd.read_csv(RPT / 'operational_choice.csv').itertuples()}
    df = pd.read_parquet(ART)

    print('figures ->')
    fig_calibration(d)
    fig_estimators(tab)
    fig_within_night(df, choice)


if __name__ == '__main__':
    main()
