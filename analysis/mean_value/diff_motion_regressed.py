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

    tep, cc = prof['t_ep_hr'], np.asarray(prof['codes'])
    codes = cc[np.clip(np.searchsorted(tep, t_hr) - 1, 0, len(cc) - 1)]
    gx, gy, gz = G[:, 0], G[:, 1], G[:, 2]
    turn = np.degrees(np.arctan2(gy, np.sqrt(gx ** 2 + gz ** 2)))

    return dict(label=meta['label'], t=t_hr, codes=codes, turn=turn,
                d=d, fit=fit_direct, res=res_direct, sm=_smooth(res_direct),
                r2=r2_direct,
                res_sep=res_sep, sm_sep=_smooth(res_sep), r2_sep=r2_sep,
                ch_res=ch['CH']['res'], ch_sm=_smooth(ch['CH']['res']),
                r2_ch=ch['CH']['r2'], r2_cle=ch['CLE']['r2'],
                r2_cre=ch['CRE']['r2'])


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

    res.plot(t, r['res'], lw=0.9, color='#C2C8D0',
             label='residual, difference fitted directly')
    res.plot(t, r['sm'], lw=2.2, color='#8A94A3', ls='--',
             label='its causal median')
    res.plot(t, r['sm_sep'], lw=3.6, color=SM_COLOR,
             label='each channel fitted first, then subtracted')
    res.axhline(0, color='#2C3E50', ls='--', lw=1.1)
    res.set_ylabel('CLE−CRE\n(fF)')
    res.set_xlabel('Time (hours)')
    res.grid(alpha=0.2)
    v = r['res_sep'][np.isfinite(r['res_sep'])]
    m1 = np.median(v); s1 = 1.4826 * np.median(np.abs(v - m1))
    half = max(6 * s1, np.nanstd(r['sm_sep']) * 4, 1e-3)
    res.set_ylim(m1 - half, m1 + half)

    # CH on its own axis: same regression, same causal smoothing, but its
    # excursions are an order of magnitude larger so a shared axis hides both
    ax2 = res.twinx()
    ax2.plot(t, r['ch_sm'], lw=3.0, color='#1F618D', alpha=0.9, label='CH')
    ax2.set_ylabel('CH (fF)', color='#1F618D')
    ax2.tick_params(axis='y', labelcolor='#1F618D')
    ax2.spines['top'].set_visible(False)
    w = r['ch_sm'][np.isfinite(r['ch_sm'])]
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
