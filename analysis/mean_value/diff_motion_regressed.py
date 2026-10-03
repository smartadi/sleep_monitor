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
    return dict(label=meta['label'], t=t_hr, d=d, res=res, sm=sm,
                codes=codes, r2=r2, beta=beta)


def draw(sessions):
    nrow = len(sessions)
    fig, axes = plt.subplots(nrow * 2, 1, figsize=(15.5, 3.6 * nrow),
                             gridspec_kw={'height_ratios': [0.34, 1.0] * nrow})
    for i, r in enumerate(sessions):
        lad, ax = axes[2 * i], axes[2 * i + 1]
        # stage ladder on top, no background shading anywhere
        pos = {c: k for k, c in enumerate(STAGE_ORDER)}
        y = np.array([pos.get(int(c), np.nan) for c in r['codes']], float)
        lad.step(r['t'], y, where='post', color='#2C3E50', lw=2.2)
        for c in STAGE_ORDER:
            m = y == pos[c]
            lad.plot(r['t'][m], y[m], '|', color=STAGE_COLORS[c], ms=9, mew=4)
        lad.set_yticks(range(len(STAGE_ORDER)))
        lad.set_yticklabels([STAGE_LABELS[c] for c in STAGE_ORDER], fontsize=12)
        lad.set_ylim(-0.6, len(STAGE_ORDER) - 0.4)
        lad.set_xlim(r['t'][0], r['t'][-1])
        lad.tick_params(labelbottom=False)
        lad.grid(alpha=0.18, axis='y')
        lad.set_title(f"{r['label']}   head orientation explains "
                      f"{100*r['r2']:.0f}% of the slow difference",
                      loc='left', fontsize=17)

        ax.plot(r['t'], r['d'], lw=0.9, color=RAW_COLOR, label='CLE−CRE, mean-centred')
        ax.plot(r['t'], r['res'], lw=1.0, color=RES_COLOR, alpha=0.75,
                label='after regressing out head orientation')
        ax.plot(r['t'], r['sm'], lw=3.2, color=SM_COLOR,
                label=f'causal {SMOOTH_MIN:.0f}-min median')
        ax.axhline(0, color='#2C3E50', ls='--', lw=1.1)
        ax.set_ylabel('fF')
        ax.set_xlim(r['t'][0], r['t'][-1])
        # robust limits: a single re-seat spike otherwise sets the axis and
        # flattens the whole night into a line
        v = r['res'][np.isfinite(r['res'])]
        med = np.median(v)
        sd = 1.4826 * np.median(np.abs(v - med))
        half = max(6 * sd, np.nanstd(r['sm']) * 4, 1e-3)
        ax.set_ylim(med - half, med + half)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=13)
    axes[-1].set_xlabel('Time (hours)')
    fig.tight_layout()
    p = FIG / 'fig_diff_motion_regressed.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p.name)


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
    draw(out)
    r2 = np.array([r['r2'] for r in out])
    print(f'\nhead orientation explains median {100*np.median(r2):.0f}% '
          f'of the slow CLE-CRE (range {100*r2.min():.0f}-{100*r2.max():.0f}%)')


if __name__ == '__main__':
    main()
