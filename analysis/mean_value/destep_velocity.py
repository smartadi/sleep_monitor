"""
Slow-trend velocity of the de-stepped CLE-CRE and CH.

diff_motion_regressed.py removes the motion jumps from CLE-CRE and CH (by
zeroing the derivative inside head movements and integrating back). This adds
the speed of the slow trend of that motion-removed trace, in fF per hour, as a
fourth panel under the same night.

The velocity is the slope of a straight line fitted to the de-stepped trace over
the trailing TREND_MIN minutes, recomputed at every 10-s block:

    v(t) = OLS slope of y over [t - TREND_MIN, t]          (fF/h)

Why a fitted slope and not a difference. The first version took a 2-minute
backward difference of the 5-minute causal median. That is a derivative of a
short smoother, so it followed every small wobble and the slow drift -- the thing
the marker is for -- was buried under it. A least-squares slope over a long
window is itself the smoother: every block in the window contributes, so noise
averages out and only the trend survives. It is trailing, so it cannot
anticipate an event; it lags it by about half the window.

Blocks inside a head movement are left out of each fit. The de-stepping holds
the trace flat across a movement, and fitting through those flat stretches
would pull every slope that spans one towards zero.

The old velocity marker (imbalance_marker.py) differentiated the RAW
differential, so every re-seat of the mask was a spike that dominated it. Here
the jumps are removed first.

Writes  writeup/figures/imbalance/fig_destep_velocity_{S}.png
        writeup/figures/imbalance/fig_destep_velocity_allsessions.png
        reports/mean_value/destep_velocity.csv   (per-night summary)

Usage
    .venv/Scripts/python.exe analysis/mean_value/destep_velocity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import diff_motion_regressed as dmr   # noqa: E402  (also sets the rcParams)
from sleep_monitor.sessions import SESSION_META   # noqa: E402

TREND_MIN = 30.0         # trailing window of the fitted slope
DIFF_COLOR, CH_COLOR = dmr.SM_COLOR, '#1F618D'


def trend_velocity(y, moving):
    """Trailing least-squares slope of y over TREND_MIN, in fF/h.

    slope = cov(t, y) / var(t) over the window, from pandas rolling moments;
    movement blocks are NaN so they drop out of every fit. A window needs at
    least half its blocks to give a value.
    """
    n = int(round(TREND_MIN * 60 / dmr.BLOCK_S))
    t_h = pd.Series(np.arange(len(y)) * dmr.BLOCK_S / 3600.0)
    yy = pd.Series(np.where(moving, np.nan, y))
    tt = t_h.where(yy.notna())
    cov = tt.rolling(n, min_periods=n // 2).cov(yy)
    var = tt.rolling(n, min_periods=n // 2).var()
    return (cov / var).to_numpy()


def _robust_lim(v, k=6.0):
    w = v[np.isfinite(v)]
    if not w.size:
        return 1.0
    m = np.median(w)
    s = 1.4826 * np.median(np.abs(w - m))
    return max(k * s, np.percentile(np.abs(w), 99), 1e-3)


def _vel_panel(ax, r, fontsize=13, legend=False):
    t = r['t']
    for a, b in dmr._spans(r['moving']):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color='#F2C9C0', lw=0, zorder=0)
    ax.plot(t, r['v_diff'], lw=1.6, color=DIFF_COLOR, label='trend velocity CLE−CRE')
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.0)
    lim = _robust_lim(r['v_diff'])
    ax.set_ylim(-lim, lim)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(t, r['v_ch'], lw=1.6, color=CH_COLOR, alpha=0.85, label='trend velocity CH')
    lim2 = _robust_lim(r['v_ch'])
    ax2.set_ylim(-lim2, lim2)
    ax2.set_ylabel('CH (fF/h)', color=CH_COLOR, fontsize=fontsize)
    ax2.tick_params(axis='y', labelcolor=CH_COLOR, labelsize=fontsize - 2)
    ax2.spines['top'].set_visible(False)
    if legend:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=fontsize - 1, ncol=2)


def draw_one(r):
    fig, axes = plt.subplots(4, 1, figsize=(15.5, 10.6), sharex=True,
                             gridspec_kw={'height_ratios': [0.38, 0.6, 1.15, 0.9]})
    lad, hd, tr, vl = axes
    t = r['t']
    dmr._ladder(lad, t, r['codes'], fontsize=13)
    lad.set_title(f"{r['label']}   —   {100 * r['pct_moving']:.0f}% of blocks "
                  f"inside a head movement", loc='left', fontsize=19)
    hd.plot(t, r['turn'], lw=2.4, color='#1B7A43')
    hd.axhline(0, color='#2C3E50', ls=':', lw=1.2)
    hd.set_ylabel('head turn\n(deg)')
    hd.annotate('+ left   − right', (0.995, 0.06), xycoords='axes fraction',
                ha='right', fontsize=13, color='#1B7A43')
    hd.grid(alpha=0.2)
    dmr._trace(tr, r, legend=True)
    tr.set_ylabel('CLE−CRE\n(fF)')
    _vel_panel(vl, r, legend=True)
    vl.set_ylabel(f'{TREND_MIN:.0f}-min trend\nvelocity (fF/h)')
    vl.set_xlabel('Time (hours)')
    fig.tight_layout()
    out = dmr.FIG / f"fig_destep_velocity_{r['label']}.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def draw_all(rs):
    nrow, ncol = 6, 2
    fig = plt.figure(figsize=(21.0, 3.1 * nrow))
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[0.26, 1.0] * nrow,
                          hspace=0.30, wspace=0.26)
    for k, r in enumerate(rs):
        c, rw = k % ncol, k // ncol
        lad = fig.add_subplot(gs[rw * 2, c])
        ax = fig.add_subplot(gs[rw * 2 + 1, c])
        dmr._ladder(lad, r['t'], r['codes'], fontsize=10)
        lad.set_title(f"{r['label']}   ({100 * r['pct_moving']:.0f}% moving)",
                      loc='left', fontsize=16)
        _vel_panel(ax, r, fontsize=11, legend=(k == 0))
        ax.set_ylabel('CLE−CRE\n(fF/h)', fontsize=12)
        if rw == nrow - 1:
            ax.set_xlabel('Time (hours)', fontsize=13)
        else:
            ax.tick_params(labelbottom=False)
    out = dmr.FIG / 'fig_destep_velocity_allsessions.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    rs, rows = [], []
    for meta in SESSION_META:
        r = dmr.one_session(meta)
        r['ch_sm'] = dmr._smooth(r['ch_destep'])
        r['v_diff'] = trend_velocity(r['d_destep'], r['moving'])
        r['v_ch'] = trend_velocity(r['ch_destep'], r['moving'])
        rs.append(r)
        still = ~r['moving']
        row = dict(session=r['label'], pct_moving=100 * r['pct_moving'])
        for key, nm in (('v_diff', 'diff'), ('v_ch', 'CH')):
            v = r[key][still & np.isfinite(r[key])]
            row[f'{nm}_median_abs_fF_per_h'] = float(np.median(np.abs(v)))
            row[f'{nm}_p95_abs_fF_per_h'] = float(np.percentile(np.abs(v), 95))
        ok = np.isfinite(r['v_diff']) & np.isfinite(r['v_ch']) & still
        row['corr_vdiff_vch'] = float(np.corrcoef(r['v_diff'][ok], r['v_ch'][ok])[0, 1])
        rows.append(row)
        print(f"  {r['label']}: |v| median CLE−CRE {row['diff_median_abs_fF_per_h']:.3f}, "
              f"CH {row['CH_median_abs_fF_per_h']:.3f} fF/h; "
              f"r(v_diff, v_CH) {row['corr_vdiff_vch']:+.2f} -> {draw_one(r).name}")
    pd.DataFrame(rows).to_csv(dmr.TAB / 'destep_velocity.csv', index=False)
    print('  sheet ->', draw_all(rs).name)


if __name__ == '__main__':
    main()
