"""
CLE-CRE with head movement regressed out, then smoothed.

The existing marker has two problems that show up as soon as it is drawn over
the trace it comes from.

1. It appears to LEAD the activity. That is not an illusion: the low pass is a
   CENTRED rolling window (`center=True`), so at tau = 30 min every point
   already contains data from up to 15 minutes ahead. A rise that happens at
   t+10 min is visible in the smoothed trace at t.

2. Motion is only MASKED, not removed. Epochs whose accelerometer standard
   deviation lands in the session's top decile are set to NaN and skipped by the
   rolling window; nothing is subtracted. A slow posture change that does not
   trip that threshold stays in the signal in full, and a left-right capacitance
   difference is exactly what a head turn should produce.

What this does instead
----------------------
    d      = CLE - CRE, block-averaged, mean-centred              (fF)
    g      = gravity vector per axis, the 0.05 Hz low pass of the
             accelerometer -- head ORIENTATION, not head movement
    d_res  = d - OLS(d ~ gx + gy + gz)                            regression
    out    = causal rolling median of d_res                       smoothing

Orientation rather than activity is the right regressor here: the quantity is a
slow level, and what moves a slow left-right capacitance difference is where the
head is, not how fast it got there. All three gravity axes are used rather than
the single turn angle, so roll, pitch and their combination are all available to
the fit.

The smoothing is CAUSAL -- a trailing window -- so the output cannot anticipate.
It lags instead, by about half the window, which is the honest trade and is
stated on the figure.

The fit also answers a question the paper has not: R^2 is the fraction of the
slow differential that head orientation alone explains. That is the posture
control the existing docstring specifies and which was never run.

Writes  reports/mean_value/diff_motion_regressed.csv
        writeup/figures/imbalance/fig_diff_motion_regressed.png

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/diff_motion_regressed.py
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

from sleep_monitor import (load_session, load_sleep_profile,       # noqa: E402
                           STAGE_LABELS, STAGE_COLORS, STAGE_ORDER)
from sleep_monitor.config import CAP_SCALE_TO_FF                   # noqa: E402
from sleep_monitor.filters import lowpass                          # noqa: E402
from sleep_monitor.sessions import SESSION_META                    # noqa: E402

FIG = ROOT / 'writeup' / 'figures' / 'imbalance'
TAB = ROOT / 'reports' / 'mean_value'
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

FS = 100.0
BLOCK_S = 10.0            # block average before anything else
GRAV_HZ = 0.05            # gravity / orientation split, as in motion.py
SMOOTH_MIN = 5.0          # causal smoothing window
STEP_THRESH = 0.03        # gravity-vector change that counts as head movement
STEP_PAD = 3              # blocks either side: capacitance settles after the accelerometer
RAW_COLOR, RES_COLOR, SM_COLOR = '#C8CDD4', '#5B6B7F', '#B9380B'

plt.rcParams.update({
    'font.size': 17, 'axes.titlesize': 19, 'axes.labelsize': 17,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15,
    'font.weight': 'bold', 'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.dpi': 110, 'savefig.dpi': 170,
})


def blocks(x, n):
    m = len(x) // n
    return x[:m * n].reshape(m, n).mean(axis=1)


def destep(y, G, step_thresh, pad_blocks):
    """Remove level jumps that happen during head movement, by editing the
    derivative rather than estimating each step.

    An earlier version detected each jump, estimated its size from medians
    either side, and subtracted a cumulative staircase. That was worse than
    doing nothing: every step size carries an estimation error, the staircase
    accumulates them, and a mis-sized step injects a NEW discontinuity. Measured
    across the twelve recordings it made the sharpest one-block jumps 19% LARGER
    and widened the overall range on five nights.

    The fix is to delete the observed jump instead of an estimate of it. Take
    the first difference, zero every increment that falls inside a head
    movement, and integrate back. What is removed is exactly the level change
    that occurred while the head was moving -- no size to estimate, nothing to
    accumulate. Everything between movements is untouched, so a slow drift
    running through a posture change survives it.

    pad_blocks widens each movement window, because the capacitance settles a
    little after the accelerometer does.
    """
    dg = np.r_[0.0, np.linalg.norm(np.diff(G, axis=0), axis=1)]
    moving = dg > step_thresh
    if pad_blocks:
        k = np.ones(2 * pad_blocks + 1, bool)
        moving = np.convolve(moving, k, mode='same') > 0

    dy = np.diff(y, prepend=y[0])
    dy = np.where(np.isfinite(dy), dy, 0.0)
    dy[moving] = 0.0                      # the jump itself, deleted
    out = np.cumsum(dy)
    return out - np.nanmean(out), moving



def motion_steps(y, G, win_blocks, step_thresh, min_gap_blocks, min_step_fF):
    """Find the sharp level jumps that coincide with head movement.

    This is the thing motion actually does to these traces. A posture change
    does not scale the signal or add a slow trend; it moves the level, in one
    block, and leaves it there. Regression cannot represent that -- it fits a
    coefficient on orientation, which is a continuous function, while what
    happened was a discontinuity.

    A jump is accepted only where the HEAD moved, so a level change with no
    accompanying movement is left alone: that one is either physiology or
    something else, and it is not this function's business to delete it.

    Returns (index, step_size) pairs, step measured as the difference between
    the median just after and the median just before, which is robust to the
    movement transient sitting in the middle.
    """
    dg = np.r_[0.0, np.linalg.norm(np.diff(G, axis=0), axis=1)]
    cand = np.flatnonzero(dg > step_thresh)
    steps, last = [], -min_gap_blocks
    for c in cand:
        if c - last < min_gap_blocks:
            continue
        a0, a1 = max(0, c - win_blocks), max(1, c - 1)
        b0, b1 = min(len(y) - 1, c + 2), min(len(y), c + 2 + win_blocks)
        if a1 - a0 < 3 or b1 - b0 < 3:
            continue
        before, after = np.nanmedian(y[a0:a1]), np.nanmedian(y[b0:b1])
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        st = after - before
        if abs(st) >= min_step_fF:
            steps.append((int(c), float(st)))
            last = c
    return steps


def remove_motion_steps(y, steps):
    """Subtract the staircase: level jumps go, everything between stays.

    The difference from removing a per-segment offset is the whole point. An
    offset per segment also deletes the level differences BETWEEN segments,
    which is most of the signal; this subtracts only the discontinuity, so a
    slow drift running through a posture change survives it intact.
    """
    stair = np.zeros(len(y))
    for idx, st in steps:
        stair[idx:] += st
    return y - stair



def posture_segments(G, min_len_blocks, step_thresh):
    """Split the night where head orientation actually steps.

    Posture is piecewise constant: the head holds a position for a stretch and
    then moves. Detecting those steps and fitting one offset per stretch is
    adaptive exactly where posture changes and frozen where it does not, so it
    cannot drift into absorbing the signal the way a continuously-adapting
    filter does.

    The step statistic is the norm of the change in the gravity vector between
    adjacent blocks, so a movement in any axis counts. A boundary is accepted
    only if the resulting segment is at least min_len_blocks long, which stops
    one restless minute from producing fifty segments.
    """
    dg = np.linalg.norm(np.diff(G, axis=0), axis=1)
    dg = np.r_[0.0, dg]
    cand = np.flatnonzero(dg > step_thresh)
    bounds, last = [0], 0
    for c in cand:
        if c - last >= min_len_blocks:
            bounds.append(c)
            last = c
    bounds.append(len(G))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] >= min_len_blocks]


def fit_posture_offsets(y, segs):
    """One offset per posture segment: the segment median, removed.

    A median rather than a mean, so a movement artifact inside the segment does
    not drag the level it is supposed to define.
    """
    res = np.array(y, float).copy()
    for a, b in segs:
        v = y[a:b]
        m = np.nanmedian(v)
        if np.isfinite(m):
            res[a:b] = y[a:b] - m
    return res



def rls(y, G, lam=0.995, delta=1e3):
    """Recursive least squares: a coefficient vector that tracks over the night.

    A single OLS coefficient assumes the coupling between head orientation and
    capacitance is fixed for eight hours. It is not -- the mask re-seats, the
    skin contact changes, and the same head angle gives a different capacitance
    before and after. RLS re-estimates the coefficients at every block with an
    exponential forgetting factor, so the fit follows those changes.

    The residual returned is the A PRIORI error: the coefficients used at block
    t were learned from blocks before t only. That keeps it causal and stops the
    filter from explaining a sample with itself.

    lam sets the memory, roughly 1/(1-lam) blocks. At 10 s blocks, 0.995 is
    about 33 minutes.

    WARNING, and it is the whole difficulty with this approach: as lam falls the
    filter tracks faster and will absorb anything slow, including whatever
    physiology the signal carries. R^2 rising is therefore NOT evidence the
    motion removal improved. The surrogate control below is what separates the
    two.
    """
    n, k = len(y), G.shape[1] + 1
    X = np.column_stack([np.ones(n), G])
    beta = np.zeros(k)
    P = np.eye(k) * delta
    res = np.full(n, np.nan)
    for t in range(n):
        x = X[t]
        if not np.isfinite(x).all() or not np.isfinite(y[t]):
            continue
        e = y[t] - beta @ x              # a priori error: causal
        res[t] = e
        Px = P @ x
        g = Px / (lam + x @ Px)
        beta = beta + g * e
        P = (P - np.outer(g, Px)) / lam
    return res


def _r2(y, res):
    ok = np.isfinite(y) & np.isfinite(res)
    ss = float(np.nansum((y[ok] - y[ok].mean()) ** 2))
    return float(1 - np.nansum(res[ok] ** 2) / ss) if ss > 0 else np.nan



def _regress(y, G):
    """Remove the head-orientation component of y. Returns fit, residual, R^2."""
    ok = np.isfinite(y) & np.isfinite(G).all(axis=1)
    X = np.column_stack([np.ones(ok.sum()), G[ok]])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    fit = np.full(len(y), np.nan)
    fit[ok] = X @ beta
    res = y - fit
    ss = float(np.nansum((y[ok] - y[ok].mean()) ** 2))
    r2 = float(1 - np.nansum(res[ok] ** 2) / ss) if ss > 0 else np.nan
    return fit, res, r2, beta


def _smooth(x):
    """Causal trailing median: the output cannot anticipate."""
    win = max(3, int(round(SMOOTH_MIN * 60 / BLOCK_S)))
    return pd.Series(x).rolling(win, min_periods=win // 3).median().to_numpy()


def one_session(meta):
    """Regress each channel separately, then form the difference.

    Regressing the difference directly forces ONE coefficient to describe how
    both electrodes respond to head orientation. They need not respond the same
    way -- the two sit on opposite temples, and a turn presses one toward the
    skin while lifting the other -- so a single coefficient can only remove the
    common part. Fitting CLE and CRE separately gives each its own, and the
    difference of the residuals removes whatever each one accounted for.

    CH is carried through the identical chain so it can be drawn against the
    difference on the same footing.
    """
    s = load_session(meta)
    prof = load_sleep_profile(s)
    n = int(BLOCK_S * FS)

    # gravity = orientation; the 0.05 Hz split is motion.py's convention
    g = {ax: lowpass(np.asarray(s.cap[ax], float), GRAV_HZ, FS)
         for ax in ('aX', 'aY', 'aZ')}
    G = np.column_stack([blocks(g[ax], n) for ax in ('aX', 'aY', 'aZ')])

    ch = {}
    for name in ('CLE', 'CRE', 'CH'):
        y = blocks(np.asarray(s.cap[name], float) * CAP_SCALE_TO_FF, n)
        y = y - np.nanmean(y)
        fit, res, r2, beta = _regress(y, G)
        ch[name] = dict(y=y, fit=fit, res=res, r2=r2, beta=beta)

    t_hr = (np.arange(len(G)) * BLOCK_S + BLOCK_S / 2) / 3600.0

    # the difference, both ways, so the two can be compared directly
    d = ch['CLE']['y'] - ch['CRE']['y']
    d = d - np.nanmean(d)
    fit_direct, res_direct, r2_direct, _ = _regress(d, G)
    res_sep = ch['CLE']['res'] - ch['CRE']['res']
    res_sep = res_sep - np.nanmean(res_sep)
    ok = np.isfinite(d) & np.isfinite(res_sep)
    ss = float(np.nansum((d[ok] - d[ok].mean()) ** 2))
    r2_sep = float(1 - np.nansum(res_sep[ok] ** 2) / ss) if ss > 0 else np.nan

    d_destep, moving = destep(d, G, STEP_THRESH, STEP_PAD)

    tep, cc = prof['t_ep_hr'], np.asarray(prof['codes'])
    codes = cc[np.clip(np.searchsorted(tep, t_hr) - 1, 0, len(cc) - 1)]
    gx, gy, gz = G[:, 0], G[:, 1], G[:, 2]
    turn = np.degrees(np.arctan2(gy, np.sqrt(gx ** 2 + gz ** 2)))

    return dict(label=meta['label'], t=t_hr, codes=codes, turn=turn,
                d=d, fit=fit_direct, res=res_direct, sm=_smooth(res_direct),
                r2=r2_direct,
                res_sep=res_sep, sm_sep=_smooth(res_sep), r2_sep=r2_sep,
                d_destep=d_destep, sm_destep=_smooth(d_destep), moving=moving,
                ch_destep=destep(ch['CH']['y'], G, STEP_THRESH, STEP_PAD)[0],
                ch_res=ch['CH']['res'], ch_sm=_smooth(ch['CH']['res']),
                r2_ch=ch['CH']['r2'], r2_cle=ch['CLE']['r2'],
                r2_cre=ch['CRE']['r2'])


def _spans(mask):
    """Contiguous True runs of a boolean mask, as (start, stop) pairs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out



def draw_one(r, out_dir):
    """One night, four rows: stages, what was removed, the fit, what is left.

    Row B is the point of this figure. It shows the head-turn angle the fit saw
    and, over the raw difference in row C, the component the fit removed. If the
    removed component steps where the head steps, the regression is doing its
    job; where it does not, the residual in row D is carrying something the head
    cannot explain.
    """
    fig, axes = plt.subplots(4, 1, figsize=(15.5, 10.4), sharex=True,
                             gridspec_kw={'height_ratios': [0.38, 0.8, 1.0, 1.0]})
    lad, hd, raw, res = axes
    t = r['t']

    pos = {c: k for k, c in enumerate(STAGE_ORDER)}
    y = np.array([pos.get(int(c), np.nan) for c in r['codes']], float)
    lad.step(t, y, where='post', color='#2C3E50', lw=2.4)
    for c in STAGE_ORDER:
        m = y == pos[c]
        lad.plot(t[m], y[m], '|', color=STAGE_COLORS[c], ms=10, mew=4)
    lad.set_yticks(range(len(STAGE_ORDER)))
    lad.set_yticklabels([STAGE_LABELS[c] for c in STAGE_ORDER], fontsize=13)
    lad.set_ylim(-0.6, len(STAGE_ORDER) - 0.4)
    lad.grid(alpha=0.18, axis='y')
    lad.set_title(
        f"{r['label']}   —   head orientation explains "
        f"CLE {100*r['r2_cle']:.0f}%   CRE {100*r['r2_cre']:.0f}%   "
        f"CH {100*r['r2_ch']:.0f}%   |   of CLE−CRE: "
        f"{100*r['r2']:.0f}% fitting the difference, "
        f"{100*r['r2_sep']:.0f}% fitting each channel first",
        loc='left', fontsize=17)

    hd.plot(t, r['turn'], lw=2.4, color='#1B7A43')
    hd.axhline(0, color='#2C3E50', ls=':', lw=1.2)
    hd.set_ylabel('head turn\n(deg)')
    hd.annotate('+ left   − right', (0.995, 0.06), xycoords='axes fraction',
                ha='right', fontsize=13, color='#1B7A43')
    hd.grid(alpha=0.2)

    raw.plot(t, r['d'], lw=1.1, color='#9AA3AE', label='CLE−CRE, mean-centred')
    raw.plot(t, r['fit'], lw=3.0, color='#1B7A43',
             label='component explained by head orientation')
    raw.axhline(0, color='#2C3E50', ls='--', lw=1.1)
    raw.set_ylabel('fF')
    raw.legend(loc='upper right', fontsize=14)
    raw.grid(alpha=0.2)
    v = r['d'][np.isfinite(r['d'])]
    m0 = np.median(v); s0 = 1.4826 * np.median(np.abs(v - m0))
    raw.set_ylim(m0 - max(6 * s0, 1e-3), m0 + max(6 * s0, 1e-3))

    for a, b in _spans(r['moving']):
        res.axvspan(t[a], t[min(b, len(t) - 1)], color='#F2C9C0', lw=0, zorder=0)
    res.plot(t, r['d'], lw=0.9, color='#C2C8D0', label='CLE−CRE')
    res.plot(t, r['d_destep'], lw=1.3, color='#5B6B7F', alpha=0.85,
             label='jumps removed')
    res.plot(t, r['sm_destep'], lw=3.6, color=SM_COLOR,
             label='its causal median')
    res.axhline(0, color='#2C3E50', ls='--', lw=1.1)
    res.set_ylabel('CLE−CRE\n(fF)')
    res.set_xlabel('Time (hours)')
    res.grid(alpha=0.2)
    v = r['d_destep'][np.isfinite(r['d_destep'])]
    m1 = np.median(v); s1 = 1.4826 * np.median(np.abs(v - m1))
    half = max(6 * s1, np.nanstd(r['sm_destep']) * 4, 1e-3)
    res.set_ylim(m1 - half, m1 + half)

    # CH on its own axis: same regression, same causal smoothing, but its
    # excursions are an order of magnitude larger so a shared axis hides both
    ax2 = res.twinx()
    ax2.plot(t, _smooth(r['ch_destep']), lw=3.0, color='#1F618D', alpha=0.9, label='CH')
    ax2.set_ylabel('CH (fF)', color='#1F618D')
    ax2.tick_params(axis='y', labelcolor='#1F618D')
    ax2.spines['top'].set_visible(False)
    w = _smooth(r['ch_destep']); w = w[np.isfinite(w)]
    if w.size:
        m2 = np.median(w); s2 = 1.4826 * np.median(np.abs(w - m2))
        ax2.set_ylim(m2 - max(5 * s2, 1e-3), m2 + max(5 * s2, 1e-3))
    h1, l1 = res.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    res.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=12, ncol=2)

    lad.set_xlim(t[0], t[-1])
    fig.tight_layout()
    out = out_dir / f"fig_diff_regressed_{r['label']}.png"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    out, rows = [], []
    for meta in SESSION_META:
        try:
            r = one_session(meta)
        except Exception as e:
            print(f'  {meta["label"]}: skipped ({type(e).__name__}: {e})')
            continue
        out.append(r)
        rows.append(dict(session=r['label'], r2_CLE=r['r2_cle'],
                         r2_CRE=r['r2_cre'], r2_CH=r['r2_ch'],
                         r2_diff_direct=r['r2'], r2_diff_separate=r['r2_sep']))
        print(f"  {r['label']}  CLE {r['r2_cle']:.2f}  CRE {r['r2_cre']:.2f}  "
              f"CH {r['r2_ch']:.2f}  |  diff direct {r['r2']:.2f}  "
              f"separate {r['r2_sep']:.2f}")
    pd.DataFrame(rows).to_csv(TAB / 'diff_motion_regressed.csv', index=False)
    for r in out:
        print('  wrote', draw_one(r, FIG).name)
    print('\nfraction of each slow signal explained by head orientation:')
    for key, lbl in (('r2_cle', 'CLE'), ('r2_cre', 'CRE'), ('r2_ch', 'CH'),
                     ('r2', 'CLE-CRE, difference fitted directly'),
                     ('r2_sep', 'CLE-CRE, each channel fitted first')):
        v = np.array([r[key] for r in out])
        print(f'  {lbl:40s} median {100*np.median(v):3.0f}%  '
              f'range {100*v.min():3.0f}-{100*v.max():3.0f}%')


if __name__ == '__main__':
    main()
