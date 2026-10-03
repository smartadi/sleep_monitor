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


def one_session(meta):
    s = load_session(meta)
    prof = load_sleep_profile(s)
    n = int(BLOCK_S * FS)

    cle = np.asarray(s.cap['CLE'], float) * CAP_SCALE_TO_FF
    cre = np.asarray(s.cap['CRE'], float) * CAP_SCALE_TO_FF
    d_full = cle - cre
    # gravity = orientation; the 0.05 Hz split is motion.py's convention
    g = {ax: lowpass(np.asarray(s.cap[ax], float), GRAV_HZ, FS)
         for ax in ('aX', 'aY', 'aZ')}

    d = blocks(d_full, n)
    G = np.column_stack([blocks(g[ax], n) for ax in ('aX', 'aY', 'aZ')])
    t_hr = (np.arange(len(d)) * BLOCK_S + BLOCK_S / 2) / 3600.0
    d = d - np.nanmean(d)

    ok = np.isfinite(d) & np.isfinite(G).all(axis=1)
    X = np.column_stack([np.ones(ok.sum()), G[ok]])
    beta, *_ = np.linalg.lstsq(X, d[ok], rcond=None)
    fit = np.full(len(d), np.nan)
    fit[ok] = X @ beta
    res = d - fit
    ss_tot = float(np.nansum((d[ok] - d[ok].mean()) ** 2))
    r2 = float(1 - np.nansum(res[ok] ** 2) / ss_tot) if ss_tot > 0 else np.nan

    # causal smoothing: trailing window, so the output cannot anticipate
    win = max(3, int(round(SMOOTH_MIN * 60 / BLOCK_S)))
    sm = pd.Series(res).rolling(win, min_periods=win // 3).median().to_numpy()

    codes = np.full(len(d), -1)
    tep, cc = prof['t_ep_hr'], np.asarray(prof['codes'])
    j = np.clip(np.searchsorted(tep, t_hr) - 1, 0, len(cc) - 1)
    codes = cc[j]
    # head turn in degrees, from the same gravity vector the fit used:
    # positive = subject's left (motion.py convention)
    gx, gy, gz = G[:, 0], G[:, 1], G[:, 2]
    turn = np.degrees(np.arctan2(gy, np.sqrt(gx ** 2 + gz ** 2)))
    return dict(label=meta['label'], t=t_hr, d=d, res=res, sm=sm, fit=fit,
                turn=turn, codes=codes, r2=r2, beta=beta)


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
    lad.set_title(f"{r['label']}   —   head orientation explains "
                  f"{100*r['r2']:.0f}% of the slow CLE−CRE",
                  loc='left', fontsize=20)

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

    res.plot(t, r['res'], lw=1.0, color='#5B6B7F', alpha=0.8,
             label='residual, head orientation removed')
    res.plot(t, r['sm'], lw=3.4, color=SM_COLOR,
             label=f'causal {SMOOTH_MIN:.0f}-min median')
    res.axhline(0, color='#2C3E50', ls='--', lw=1.1)
    res.set_ylabel('fF')
    res.set_xlabel('Time (hours)')
    res.legend(loc='upper right', fontsize=14)
    res.grid(alpha=0.2)
    v = r['res'][np.isfinite(r['res'])]
    m1 = np.median(v); s1 = 1.4826 * np.median(np.abs(v - m1))
    half = max(6 * s1, np.nanstd(r['sm']) * 4, 1e-3)
    res.set_ylim(m1 - half, m1 + half)

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
        rows.append(dict(session=r['label'], r2_head_orientation=r['r2'],
                         beta_aX=r['beta'][1], beta_aY=r['beta'][2],
                         beta_aZ=r['beta'][3]))
        print(f"  {r['label']}  R2 = {r['r2']:.3f}")
    pd.DataFrame(rows).to_csv(TAB / 'diff_motion_regressed.csv', index=False)
    for r in out:
        print('  wrote', draw_one(r, FIG).name)
    r2 = np.array([r['r2'] for r in out])
    print(f'\nhead orientation explains median {100*np.median(r2):.0f}% '
          f'of the slow CLE-CRE (range {100*r2.min():.0f}-{100*r2.max():.0f}%)')


if __name__ == '__main__':
    main()
