#!/usr/bin/env python
"""
Full-session rate traces, all twelve recordings (2026-08-14).

One panel per recording, whole night, showing the PSG reference against three
estimators after per-session k-scaling, so all four are on the same axis and the
question "does this estimator follow the reference?" is answerable by eye:

  reference        PSG consensus (respiration) / ECG R-peaks (cardiac)
  spectral 4 s     the published estimator — Welch nperseg fixed at 4 s
  spectral window  the same method with the whole window as one segment
  peaks loose      prominence-thresholded peak counting, the operational choice

These are the plots the accuracy tables cannot show. A per-epoch error of
0.91 br/min looks excellent in a table; here the same estimator is a horizontal
line through every night.

Outputs to writeup/figures/rate_rerun/:
  fig_sessions_resp.png
  fig_sessions_card.png

Run from the repo root:  python analysis/rates/rerun_session_plots.py
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

ART = ROOT / 'artifacts' / 'rate_rerun_phase_a.parquet'
FIG = ROOT / 'writeup' / 'figures' / 'rate_rerun'
FIG.mkdir(parents=True, exist_ok=True)

C_REF, C_OLD, C_NEW, C_PK = '#0b0b0b', '#e34948', '#2a78d6', '#1baf7a'
C_INK, C_MUTED, C_FAINT = '#0b0b0b', '#52514e', '#d8d6cf'
UNIT = {'resp': 'br/min', 'card': 'BPM'}
BAND_NAME = {'resp': 'Respiratory', 'card': 'Cardiac'}
YLIM = {'resp': (5, 32), 'card': (35, 125)}
CHANNEL = 'CRE'
K_LO, K_HI = 0.3, 5.0

# Emphasis differs by band because the failure differs by band. In respiration
# the published estimator is a flat line that would hide under the reference, so
# it is drawn last, dashed and heavy. In the cardiac band the same estimator is
# not flat but quantized to 0.25 Hz steps, so drawn heavy it fills the panel and
# hides everything; there the operational estimator carries the comparison and
# both spectral variants go faint and behind.
#            column             label                            colour a    lw   ls          z
SERIES = {
    'resp': [('spectral_interp', 'spectral, full window',        C_NEW, .40, 0.5, '-',        2),
             ('peaks_loose',     'peaks, loose (operational)',   C_PK,  .80, 0.5, '-',        3),
             ('spectral',        'spectral, 4 s Welch (published)', C_OLD, 1., 1.3, (0, (4, 2)), 6)],
    'card': [('spectral',        'spectral, 4 s Welch (published)', C_OLD, .22, 0.4, '-',     1),
             ('spectral_interp', 'spectral, full window',        C_NEW, .22, 0.4, '-',        2),
             ('peaks_loose',     'peaks, loose (operational)',   C_PK,  .95, 0.7, '-',        5)],
}
LEGEND_ORDER = ['PSG reference', 'spectral, 4 s Welch (published)',
                'peaks, loose (operational)', 'spectral, full window']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': C_INK, 'text.color': C_INK,
    'xtick.color': C_MUTED, 'ytick.color': C_MUTED, 'axes.edgecolor': C_MUTED,
    'legend.frameon': False,
    'figure.dpi': 200, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
})
MM = 1 / 25.4


def fit_k(raw, gt):
    r = raw / gt
    r = r[(r > K_LO) & (r < K_HI) & np.isfinite(r)]
    return float(np.median(r)) if len(r) >= 10 else np.nan


def plot_band(df: pd.DataFrame, band: str) -> None:
    g0 = df[(df.band == band) & (df.channel == CHANNEL)].dropna(subset=['gt_hz'])
    sessions = sorted(g0.session.unique())
    fig, axes = plt.subplots(4, 3, figsize=(183 * MM, 168 * MM), sharey=True)

    for ax, sess in zip(axes.ravel(), sessions):
        g = g0[g0.session == sess].sort_values('epoch')
        t = g.t_hr.values
        gt = g.gt_hz.values * 60.0
        ax.plot(t, gt, '-', color=C_REF, lw=0.9, zorder=4, label='PSG reference')

        for col, label, colour, alpha, lw, ls, z in SERIES[band]:
            raw = g[col].values * 60.0
            m = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
            if m.sum() < 20:
                continue
            k = fit_k(raw[m], gt[m])
            ax.plot(t[m], raw[m] / k, ls=ls, color=colour, lw=lw, alpha=alpha,
                    zorder=z, label=label, solid_capstyle='butt')

        ax.set_title(sess, color=C_INK, pad=3, loc='left', fontsize=7.5)
        ax.set_ylim(*YLIM[band])
        ax.set_xlim(0, max(t.max(), 1))
        ax.text(0.985, 0.04, f'ref SD {np.std(gt):.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6, color=C_MUTED, zorder=8,
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.2))
        ax.grid(axis='y', color=C_FAINT, lw=0.4, zorder=0)
        ax.set_axisbelow(True)

    for ax in axes[-1]:
        ax.set_xlabel('time (h)')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'rate ({UNIT[band]})')

    h0, l0 = axes[0, 0].get_legend_handles_labels()
    order = [l0.index(x) for x in LEGEND_ORDER if x in l0]
    handles, labels = [h0[i] for i in order], [l0[i] for i in order]
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.035), handlelength=1.6, columnspacing=1.6)
    fig.suptitle(f'{BAND_NAME[band]} rate across the whole night, all twelve '
                 f'recordings ({CHANNEL} channel, each estimator k-scaled to its own night)',
                 fontsize=8.5, y=1.002)
    fig.text(0.5, -0.055,
             'Every estimator is divided by a scale factor fitted on the same night, '
             'which is the most favourable case. SD is of the plotted trace.',
             ha='center', fontsize=6.5, color=C_MUTED)
    fig.tight_layout(w_pad=1.6, h_pad=1.9)
    out = FIG / f'fig_sessions_{band}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  {out}')


def main() -> None:
    df = pd.read_parquet(ART)
    print('figures ->')
    for band in ['resp', 'card']:
        plot_band(df, band)
    plot_representative(df)




# ─────────────────────────────────────────── compact main-text version

def plot_representative(df: pd.DataFrame, session: str = 'S1N1') -> None:
    """One night, both bands — the main-text 'what the sensor produces' figure.

    The twelve-panel versions carry the cohort; this carries the reader.
    """
    fig, axes = plt.subplots(2, 1, figsize=(183 * MM, 88 * MM), sharex=True)
    for ax, band in zip(axes, ['resp', 'card']):
        g = (df[(df.band == band) & (df.channel == CHANNEL) & (df.session == session)]
             .dropna(subset=['gt_hz']).sort_values('epoch'))
        t = g.t_hr.values
        gt = g.gt_hz.values * 60.0
        raw = g.peaks_loose.values * 60.0
        m = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
        k = fit_k(raw[m], gt[m])
        ax.plot(t[m], raw[m] / k, '-', color=C_PK, lw=0.55, alpha=0.9, zorder=2,
                label='capacitive mask (peak counting, per-session k)')
        ax.plot(t, gt, '-', color=C_REF, lw=1.0, zorder=3, label='PSG reference')
        ax.set_ylim(*YLIM[band])
        ax.set_ylabel(f'{BAND_NAME[band].lower()} rate\n({UNIT[band]})')
        ax.grid(axis='y', color=C_FAINT, lw=0.4)
        ax.set_axisbelow(True)
        err = np.median(np.abs(raw[m] / k - gt[m]))
        ax.text(0.995, 0.94, f'k = {k:.2f}   median |error| {err:.2f} {UNIT[band]}',
                transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
                color=C_MUTED)
        ax.text(-0.075, 1.02, 'AB'['resp card'.split().index(band)], transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='bottom', color=C_INK)
    axes[1].set_xlabel('time (h)')
    axes[0].legend(loc='lower left', ncol=2, handlelength=1.6, borderpad=0.2)
    fig.suptitle(f'Respiratory and cardiac rate across one night ({session})',
                 fontsize=8.5, y=0.995)
    fig.tight_layout(h_pad=1.4)
    out = FIG / 'fig_representative_night.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  {out}')


if __name__ == '__main__':
    main()
