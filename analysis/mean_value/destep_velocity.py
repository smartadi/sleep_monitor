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

Layout: each signal (CLE-CRE, then CH) on its own panel with its velocity on a
separate panel directly below it, so the two never share an axis.

Writes  writeup/figures/imbalance/fig_destep_velocity_{S}.png
        writeup/figures/imbalance/fig_destep_velocity_allsessions_{CLE-CRE,CH}.png
        ..._supp.png versions of both sheets and of S4N2, without the movement
        shading, for the supplement
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


SIGNALS = {
    # key: (label, de-stepped trace, causal median, velocity, colour)
    'CLE-CRE': ('CLE−CRE', 'd_destep', 'sm_destep', 'v_diff', DIFF_COLOR),
    'CH': ('CH', 'ch_destep', 'ch_sm', 'v_ch', CH_COLOR),
}


SHADE_MOTION = True     # False for the supplement versions: plain white panels


def _movement(ax, r):
    if not SHADE_MOTION:
        return
    t = r['t']
    for a, b in dmr._spans(r['moving']):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color='#F2C9C0', lw=0, zorder=0)


def _signal_panel(ax, r, key, fontsize=13, legend=False):
    """The motion-removed trace and its causal median (raw CLE−CRE behind it)."""
    name, y_key, sm_key, _, color = SIGNALS[key]
    t = r['t']
    _movement(ax, r)
    if key == 'CLE-CRE':
        ax.plot(t, r['d'], lw=0.8, color='#C2C8D0', label='raw')
    ax.plot(t, r[y_key], lw=1.1, color='#5B6B7F', alpha=0.85, label='jumps removed')
    ax.plot(t, r[sm_key], lw=2.6, color=color, label='5-min causal median')
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.0)
    v = r[y_key][np.isfinite(r[y_key])]
    m = np.median(v)
    half = max(6 * 1.4826 * np.median(np.abs(v - m)),
               4 * np.nanstd(r[sm_key]), 1e-3)
    ax.set_ylim(m - half, m + half)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(f'{name}\n(fF)', fontsize=fontsize)
    ax.grid(alpha=0.2)
    if legend:
        ax.legend(loc='upper right', fontsize=fontsize - 2, ncol=3)


def _velocity_panel(ax, r, key, fontsize=13):
    """That trace's slow-trend velocity, on its own axis directly below it."""
    name, _, _, v_key, color = SIGNALS[key]
    t = r['t']
    _movement(ax, r)
    v = r[v_key]
    ax.fill_between(t, 0, v, where=np.isfinite(v) & (v > 0), color=color,
                    alpha=0.18, lw=0)
    ax.fill_between(t, 0, v, where=np.isfinite(v) & (v < 0), color=color,
                    alpha=0.08, lw=0)
    ax.plot(t, v, lw=1.8, color=color)
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.0)
    lim = _robust_lim(v)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(f'{name} velocity\n(fF/h)', fontsize=fontsize)
    ax.grid(alpha=0.2)


def draw_one(r, suffix=''):
    """One night: stages, head turn, then each signal with its velocity below it."""
    fig, axes = plt.subplots(
        6, 1, figsize=(15.5, 14.5), sharex=True,
        gridspec_kw={'height_ratios': [0.38, 0.55, 1.0, 0.62, 1.0, 0.62]})
    lad, hd, s1, v1, s2, v2 = axes
    t = r['t']
    dmr._ladder(lad, t, r['codes'], fontsize=13)
    shade = '(shaded)' if SHADE_MOTION else ''
    lad.set_title(f"{r['label']}   —   {100 * r['pct_moving']:.0f}% of blocks inside a "
                  f"head movement {shade};  velocity = {TREND_MIN:.0f}-min trailing slope",
                  loc='left', fontsize=17)
    hd.plot(t, r['turn'], lw=2.4, color='#1B7A43')
    hd.axhline(0, color='#2C3E50', ls=':', lw=1.2)
    hd.set_ylabel('head turn\n(deg)')
    hd.annotate('+ left   − right', (0.995, 0.06), xycoords='axes fraction',
                ha='right', fontsize=13, color='#1B7A43')
    hd.grid(alpha=0.2)
    _signal_panel(s1, r, 'CLE-CRE', legend=True)
    _velocity_panel(v1, r, 'CLE-CRE')
    _signal_panel(s2, r, 'CH', legend=True)
    _velocity_panel(v2, r, 'CH')
    v2.set_xlabel('Time (hours)')
    fig.tight_layout()
    out = dmr.FIG / f"fig_destep_velocity_{r['label']}{suffix}.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def draw_all(rs, key, suffix=''):
    """All twelve nights for one signal: stages, trace, velocity below it."""
    nrow, ncol = 6, 2
    fig = plt.figure(figsize=(21.0, 4.8 * nrow))
    # a thin empty row after each night keeps its title off the panel above
    gs = fig.add_gridspec(nrow * 4, ncol, height_ratios=[0.24, 0.8, 0.6, 0.32] * nrow,
                          hspace=0.12, wspace=0.22)
    for k, r in enumerate(rs):
        c, rw = k % ncol, k // ncol
        lad = fig.add_subplot(gs[rw * 4, c])
        sig = fig.add_subplot(gs[rw * 4 + 1, c], sharex=lad)
        vel = fig.add_subplot(gs[rw * 4 + 2, c], sharex=lad)
        dmr._ladder(lad, r['t'], r['codes'], fontsize=9)
        lad.set_title(f"{r['label']}" + (f"   ({100 * r['pct_moving']:.0f}% moving)"
                                          if SHADE_MOTION else ''),
                      loc='left', fontsize=15)
        _signal_panel(sig, r, key, fontsize=11, legend=(k == 0))
        sig.tick_params(labelbottom=False)
        _velocity_panel(vel, r, key, fontsize=11)
        if rw == nrow - 1:
            vel.set_xlabel('Time (hours)', fontsize=13)
        else:
            vel.tick_params(labelbottom=False)
    out = dmr.FIG / f'fig_destep_velocity_allsessions_{key}{suffix}.png'
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
    for key in SIGNALS:
        print('  sheet ->', draw_all(rs, key).name)

    # supplement versions: no movement shading, and the title says nothing about it
    global SHADE_MOTION
    SHADE_MOTION = False
    for key in SIGNALS:
        print('  supplement ->', draw_all(rs, key, suffix='_supp').name)
    ex = next(r for r in rs if r['label'] == 'S4N2')
    print('  supplement ->', draw_one(ex, suffix='_supp').name)
    SHADE_MOTION = True


if __name__ == '__main__':
    main()
