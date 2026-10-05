"""
Supplement section S5 figures, drawn at print size.

The analysis scripts draw their figures for slides (13-21 in wide); placed at the
supplement's 6 in text width their type shrinks to 4-6 pt. This redraws the five
S5 figures in the shared print style of the supplementary rate figures
(analysis/rates/_supp_style.py: 9 in wide, ~8-9 pt on the page, no titles,
panel letters, white background, no head-movement shading). Nothing is
recomputed differently -- the numbers come from the same functions:

    S5 fig 1  variance tails by stage      variance_low_high.py (its CSV)
    S5 fig 2  one night in full (S4N2)     diff_motion_regressed + destep_velocity
    S5 fig 3  CLE-CRE, all twelve nights   idem
    S5 fig 4  CH, all twelve nights        idem
    S5 fig 5  velocity at REM onset        rem_onset_velocity.py

Writes  writeup/figures/supp_s5/*.png

Usage
    .venv/Scripts/python.exe analysis/mean_value/supp_s5_figures.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'rates'))
sys.path.insert(0, str(ROOT))

import diff_motion_regressed as dmr          # noqa: E402
import destep_velocity as dv                 # noqa: E402
import rem_onset_velocity as rov             # noqa: E402
import _supp_style as st                     # noqa: E402
from sleep_monitor.config import STAGE_COLORS, STAGE_LABELS, STAGE_ORDER  # noqa: E402
from sleep_monitor.sessions import SESSION_META                            # noqa: E402

st.apply()       # after the imports above, which set their own rcParams

OUT = ROOT / 'writeup' / 'figures' / 'supp_s5'
OUT.mkdir(parents=True, exist_ok=True)
REP = ROOT / 'reports' / 'mean_value'
DIFF_C, CH_C, GREY = '#B9380B', '#1F618D', '#5B6B7F'
STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLOR = {STAGE_LABELS[c]: STAGE_COLORS[c] for c in STAGE_ORDER}


def save(fig, name):
    p = OUT / f'{name}.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p.relative_to(ROOT))
    return p


# ── fig 1: variance tails ────────────────────────────────────────────────────

def fig_variance():
    e = pd.read_csv(REP / 'variance_tails_enrichment.csv')
    e = e[e.subset == 'motion-free']
    floor = 0.03
    fig, axes = plt.subplots(2, 3, figsize=(st.WIDTH_IN, 5.8), sharey=True,
                             sharex=True)
    for r, tail in enumerate(('hi', 'lo')):
        for c, ch in enumerate(('CH', 'CLE', 'CRE')):
            ax = axes[r, c]
            g = e[(e['tail'] == tail) & (e.channel == ch)]
            for i, s in enumerate(STAGES):
                v = g[g.stage == s].enrichment.to_numpy()
                if not len(v):
                    continue
                x = i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
                ax.scatter(x, np.maximum(v, floor), s=14, color=STAGE_COLOR[s],
                           edgecolor='k', lw=0.3, zorder=3)
                ax.hlines(max(np.median(v), floor), i - 0.3, i + 0.3,
                          color='#2C3E50', lw=1.8, zorder=4)
                ax.text(i, 9.5, f'{int((v > 1).sum())}/{len(v)}', ha='center',
                        fontsize=8.5, color='#2C3E50')
            ax.axhline(1, color='k', ls='--', lw=0.8)
            ax.set_yscale('log')
            ax.set_ylim(floor * 0.8, 15)
            ax.set_xticks(range(len(STAGES)))
            ax.set_xticklabels(STAGES, fontsize=10.5)
            ax.grid(axis='y', alpha=0.25, which='major')
            x0 = -0.16 if c == 0 else -0.06
            st.letter(ax, 'abcdef'[3 * r + c], x=x0)
            ax.text(x0 + 0.09, 1.045, ch, transform=ax.transAxes, fontsize=12,
                    va='bottom', color='#333333')
        axes[r, 0].set_ylabel('observed / expected')
    fig.tight_layout(w_pad=0.6)
    return save(fig, 'figS5_1_variance_tails')


# ── figs 2-4: movement-corrected traces and trend velocity ───────────────────

def _band(ax, t, codes):
    """Hypnogram as one coloured band (the stage legend is drawn once)."""
    for c in STAGE_ORDER:
        m = codes == c
        ax.fill_between(t, 0, 1, where=m, color=STAGE_COLORS[c], lw=0, step='post')
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(labelbottom=False, bottom=False)
    for s in ('left', 'top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)


def _trace(ax, r, key, raw=False):
    name, y_key, sm_key, _, color = dv.SIGNALS[key]
    t = r['t']
    if raw:
        ax.plot(t, r['d'], lw=0.5, color='#C2C8D0', label='raw')
    ax.plot(t, r[y_key], lw=0.6, color=GREY, alpha=0.9, label='steps removed')
    ax.plot(t, r[sm_key], lw=1.4, color=color, label='5-min running median')
    ax.axhline(0, color='#2C3E50', ls=':', lw=0.7)
    v = r[y_key][np.isfinite(r[y_key])]
    m = np.median(v)
    half = max(6 * 1.4826 * np.median(np.abs(v - m)), 4 * np.nanstd(r[sm_key]), 1e-3)
    ax.set_ylim(m - half, m + half)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.2)


def _vel(ax, r, key):
    name, _, _, v_key, color = dv.SIGNALS[key]
    t, v = r['t'], r[v_key]
    ax.fill_between(t, 0, v, where=np.isfinite(v), color=color, alpha=0.15, lw=0)
    ax.plot(t, v, lw=1.1, color=color)
    ax.axhline(0, color='#2C3E50', ls=':', lw=0.7)
    lim = dv._robust_lim(v)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.2)


def _stage_legend(fig, y):
    h = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOR[s]) for s in STAGES]
    fig.legend(h, STAGES, loc='lower center', bbox_to_anchor=(0.5, y), ncol=5,
               frameon=False, fontsize=10, handlelength=1.2, columnspacing=1.2)


def fig_one_night(r):
    fig, axes = plt.subplots(
        6, 1, figsize=(st.WIDTH_IN, 9.2), sharex=True,
        gridspec_kw={'height_ratios': [0.18, 0.55, 1.0, 0.62, 1.0, 0.62]})
    band, hd, s1, v1, s2, v2 = axes
    t = r['t']
    _band(band, t, np.asarray(r['codes']))
    hd.plot(t, r['turn'], lw=1.2, color='#1B7A43')
    hd.axhline(0, color='#2C3E50', ls=':', lw=0.7)
    hd.set_ylabel('head turn (°)')
    hd.grid(alpha=0.2)
    _trace(s1, r, 'CLE-CRE', raw=True)
    s1.set_ylabel('CLE−CRE (fF)')
    s1.legend(loc='lower right', bbox_to_anchor=(1.0, 0.98), fontsize=9, ncol=3,
              frameon=False)
    _vel(v1, r, 'CLE-CRE')
    v1.set_ylabel('velocity (fF/h)')
    _trace(s2, r, 'CH')
    s2.set_ylabel('CH (fF)')
    _vel(v2, r, 'CH')
    v2.set_ylabel('velocity (fF/h)')
    v2.set_xlabel('time (h)')
    for ax, l in zip(axes, 'abcdef'):
        st.letter(ax, l, x=-0.13, y=0.2 if ax is band else 1.04)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.5)
    _stage_legend(fig, 1.0)
    return save(fig, f'figS5_2_one_night_{r["label"]}')


def fig_all_nights(rs, key):
    nrow, ncol = 6, 2
    fig = plt.figure(figsize=(st.WIDTH_IN, 12.6))
    gs = fig.add_gridspec(nrow * 4, ncol, height_ratios=[0.13, 0.75, 0.55, 0.30] * nrow,
                          hspace=0.08, wspace=0.28)
    name = dv.SIGNALS[key][0]
    for k, r in enumerate(rs):
        c, rw = k % ncol, k // ncol
        b = fig.add_subplot(gs[rw * 4, c])
        s = fig.add_subplot(gs[rw * 4 + 1, c], sharex=b)
        v = fig.add_subplot(gs[rw * 4 + 2, c], sharex=b)
        _band(b, r['t'], np.asarray(r['codes']))
        b.text(0.0, 1.25, r['label'], transform=b.transAxes, fontsize=11,
               fontweight='bold', va='bottom')
        _trace(s, r, key)
        _vel(v, r, key)
        s.tick_params(labelbottom=False, labelsize=9)
        v.tick_params(labelsize=9)
        s.set_ylabel(f'{name}\n(fF)', fontsize=9.5)
        v.set_ylabel('fF/h', fontsize=9.5)
        if rw == nrow - 1:
            v.set_xlabel('time (h)')
        else:
            v.tick_params(labelbottom=False)
    _stage_legend(fig, 0.905)
    return save(fig, f'figS5_{3 if key == "CLE-CRE" else 4}_all_nights_{key}')


# ── fig 5: velocity at REM onset ────────────────────────────────────────────

def fig_rem(rs):
    w = int(round(rov.WIN_MIN * 60 / dmr.BLOCK_S))
    lag = np.arange(-w, w + 1) * dmr.BLOCK_S / 60.0
    rng = np.random.default_rng(0)       # same draws as rem_onset_velocity.main
    curves = {k: {} for k in rov.SIG}
    null = {k: {} for k in rov.SIG}
    for r in rs:
        codes = np.asarray(r['codes'])
        on = rov.rem_onsets(codes)
        if not on:
            continue
        nt = rov.null_times(codes, on, len(on) * rov.NULL_DRAWS, rng)
        for key, (field, _) in rov.SIG.items():
            v = rov.centred_slope(r[field], r['moving'])
            curves[key].setdefault(r['label'][:2], []).append(rov.windows(v, on, w))
            null[key].setdefault(r['label'][:2], []).append(rov.windows(v, nt, w))
    sj = pd.read_csv(REP / 'rem_onset_velocity_subjects.csv')

    fig, axes = plt.subplots(2, 2, figsize=(st.WIDTH_IN, 6.4),
                             gridspec_kw={'width_ratios': [1.55, 1]})
    for row, (key, (_, color)) in enumerate(rov.SIG.items()):
        ax = axes[row, 0]
        subs = sorted(curves[key])
        sm = np.array([np.nanmean(np.vstack(curves[key][s]), axis=0) for s in subs])
        nm = np.array([np.nanmean(np.vstack(null[key][s]), axis=0) for s in subs])
        for y in sm:
            ax.plot(lag, y, color=color, lw=0.6, alpha=0.35)
        m = np.nanmean(sm, axis=0)
        se = np.nanstd(sm, axis=0) / np.sqrt(len(sm))
        ax.fill_between(lag, m - se, m + se, color=color, alpha=0.18, lw=0)
        ax.plot(lag, m, color=color, lw=1.8, label='REM onset')
        ax.plot(lag, np.nanmean(nm, axis=0), color='#7F8C8D', lw=1.2, ls='--',
                label='random NREM times')
        ax.axvline(0, color='k', lw=0.8)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_xlim(-rov.WIN_MIN, rov.WIN_MIN)
        ax.set_ylabel(f'{key} velocity (fF/h)')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc='upper left', frameon=False)
        if row == 1:
            ax.set_xlabel('time from REM onset (min)')
        st.letter(ax, 'ab'[row], x=-0.15)

        ax2 = axes[row, 1]
        s = sj[sj.signal == key].reset_index(drop=True)
        y = np.arange(len(s))
        ax2.hlines(y, s.null_p05, s.null_p95, color='#BDC3C7', lw=4)
        ax2.plot(s.null_median, y, '|', color='#7F8C8D', ms=9, mew=1.5)
        ax2.plot(s.response_fF_h, y, 'o', color=color, ms=5.5)
        ax2.axvline(0, color='k', lw=0.5, ls=':')
        ax2.set_yticks(y)
        ax2.set_yticklabels([f'{a} ({b})' for a, b in zip(s.subject, s.n_events)])
        ax2.invert_yaxis()
        ax2.grid(alpha=0.2, axis='x')
        if row == 1:
            ax2.set_xlabel('onset response (fF/h)')
        st.letter(ax2, 'cd'[row], x=-0.30)
    fig.tight_layout()
    return save(fig, 'figS5_5_rem_onset')


def main():
    fig_variance()
    with contextlib.redirect_stdout(io.StringIO()):
        rs = [dmr.one_session(m) for m in SESSION_META]
    for r in rs:
        r['ch_sm'] = dmr._smooth(r['ch_destep'])
        r['v_diff'] = dv.trend_velocity(r['d_destep'], r['moving'])
        r['v_ch'] = dv.trend_velocity(r['ch_destep'], r['moving'])
    fig_one_night(next(r for r in rs if r['label'] == 'S4N2'))
    fig_all_nights(rs, 'CLE-CRE')
    fig_all_nights(rs, 'CH')
    fig_rem(rs)


if __name__ == '__main__':
    main()
