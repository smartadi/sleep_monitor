#!/usr/bin/env python
"""
k versus subject age, on the recomputed pipeline (2026-08-14).

Third pass on this analysis, and the k series changes each time, so state which
one this is: k here is the **session-wide** factor from today's rerun — the
median of estimate/reference over every valid epoch of the night, ratios clipped
to 0.3-5.0, for loose peak counting on CRE, the operational estimator. One
scalar per night, two nights per subject, n = 6 subjects.

  k_age_prior.py   older per-band single-channel k (respiratory k from the
                   degenerate 4 s spectral estimator, so its "k" was
                   15 / median(reference) — an age trend in k was an age trend
                   in median breathing rate)
  k_age_fused.py   the smart-fused pipeline
  this file        loose peak counting, the estimator that actually varies

Because the respiratory estimator now has k ~ 1.18 rather than ~1.0, the
"no calibration" comparator (k = 1) is a straw man: it is wrong by 18% before
age is considered. The honest comparator is the best constant prior — one k for
everyone — and that is the one the figure emphasises.

Inputs : reports/rates/rerun/per_session.csv          (k per session)
         analysis/rates/outputs/k_vs_age_per_subject.csv  (demographics)
Outputs: analysis/rates/outputs/k_age_rerun.csv
         writeup/figures/rate_rerun/fig_k_vs_age.png

Run from the repo root:  python analysis/rates/k_age_rerun.py
"""
from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SRC = ROOT / 'reports' / 'rates' / 'rerun' / 'per_session.csv'
DEMO = ROOT / 'analysis' / 'rates' / 'outputs' / 'k_vs_age_per_subject.csv'
OUT = ROOT / 'analysis' / 'rates' / 'outputs' / 'k_age_rerun.csv'
FIG = ROOT / 'writeup' / 'figures' / 'rate_rerun' / 'fig_k_vs_age.png'

BAND_NAME = {'resp': 'Respiratory', 'card': 'Cardiac'}
C_M, C_F = '#2a78d6', '#eb6834'
C_AGE, C_POP, C_NONE = '#2a78d6', '#1baf7a', '#c8c6bf'
C_INK, C_MUTED, C_FAINT = '#0b0b0b', '#52514e', '#d8d6cf'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': C_INK, 'text.color': C_INK,
    'xtick.color': C_MUTED, 'ytick.color': C_MUTED, 'axes.edgecolor': C_MUTED,
    'legend.frameon': False,
    'figure.dpi': 200, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
})
MM = 1 / 25.4


def exact_p(x, y) -> float:
    """Two-sided permutation p over all 720 orderings. n = 6."""
    rho = spearmanr(x, y).statistic
    y = np.asarray(y)
    null = [spearmanr(x, y[list(p)]).statistic for p in permutations(range(len(y)))]
    return float(np.mean(np.abs(null) >= abs(rho) - 1e-12))


def loo_rho(sub) -> tuple[float, float]:
    r = [spearmanr(sub.drop(i).age, sub.drop(i).k_mean).statistic for i in sub.index]
    return min(r), max(r)


def per_subject() -> pd.DataFrame:
    ses = pd.read_csv(SRC)
    ses['subj'] = ses.session.str[:2]
    demo = (pd.read_csv(DEMO)[['band', 'subj', 'age', 'sex', 'psqi']]
            .drop_duplicates(subset=['band', 'subj']))
    rows = []
    for band, g in ses.groupby('band'):
        t = g.groupby('subj').k.agg(k_mean='mean', k_lo='min', k_hi='max').reset_index()
        t['band'] = band
        t['dk'] = t.k_hi - t.k_lo
        rows.append(t)
    per = pd.concat(rows, ignore_index=True).merge(demo, on=['band', 'subj'])
    return per.sort_values(['band', 'age']).reset_index(drop=True)


def loso(sub) -> pd.DataFrame:
    rows = []
    for i, row in sub.iterrows():
        tr = sub.drop(i)
        m, c = np.polyfit(tr.age, tr.k_mean, 1)
        rows.append(dict(subj=row.subj, age=row.age, k_true=row.k_mean,
                         k_age=m * row.age + c, k_pop=tr.k_mean.mean(), k_none=1.0))
    p = pd.DataFrame(rows)
    for s in ('age', 'pop', 'none'):
        p[f'err_{s}'] = (p[f'k_{s}'] - p.k_true).abs()
    return p


def main() -> None:
    per = per_subject()
    stats, preds = [], {}
    for band in ('resp', 'card'):
        sub = per[per.band == band].reset_index(drop=True)
        pr = loso(sub)
        preds[band] = pr
        rho = spearmanr(sub.age, sub.k_mean).statistic
        lo, hi = loo_rho(sub)
        stats.append(dict(band=band, n=len(sub), k_median=sub.k_mean.median(),
                          k_min=sub.k_mean.min(), k_max=sub.k_mean.max(),
                          rho=rho, p_exact=exact_p(sub.age.values, sub.k_mean.values),
                          loo_rho_min=lo, loo_rho_max=hi, sign_stable=bool(lo * hi > 0),
                          dk_median=sub.dk.median(), dk_max=sub.dk.max(),
                          mae_age=pr.err_age.mean(), mae_pop=pr.err_pop.mean(),
                          mae_none=pr.err_none.mean()))
    st = pd.DataFrame(stats)
    st.to_csv(OUT, index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 40):
        print(st.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM))
    for ax, band in zip(axes, ('resp', 'card')):
        sub = per[per.band == band].reset_index(drop=True)
        for _, r in sub.iterrows():
            ax.plot([r.age, r.age], [r.k_lo, r.k_hi], '-', color=C_FAINT, lw=1.4,
                    zorder=1, solid_capstyle='round')
            ax.plot(r.age, r.k_mean, 'o' if r.sex == 'M' else 's', ms=4.6,
                    mfc=C_M if r.sex == 'M' else C_F, mec='white', mew=0.6, zorder=3)
            ax.annotate(r.subj, (r.age, r.k_mean), textcoords='offset points',
                        xytext=(7, 1.5), fontsize=6, color=C_MUTED)
        m, c = np.polyfit(sub.age, sub.k_mean, 1)
        xs = np.array([sub.age.min() - 3, sub.age.max() + 3])
        # least-squares guide only; neither band's correlation is significant
        ax.plot(xs, m * xs + c, ':', color=C_FAINT, lw=1.0, zorder=2)
        rho = spearmanr(sub.age, sub.k_mean).statistic
        p = exact_p(sub.age.values, sub.k_mean.values)
        lo, hi = loo_rho(sub)
        ax.set_title(f'{BAND_NAME[band]} k\n'
                     rf'$\rho$ = {rho:+.2f}, exact p = {p:.3f}' + '\n'
                     rf'leave-one-out $\rho$ {lo:+.2f} to {hi:+.2f}',
                     fontsize=7, color=C_INK, pad=5)
        ax.set_xlabel('age (years)')
        ax.set_ylabel(f'k  ({BAND_NAME[band].lower()})')
        ax.grid(axis='y', color=C_FAINT, lw=0.4)
        ax.set_axisbelow(True)
        ax.text(-0.26, 1.16, 'AB'[('resp', 'card').index(band)], transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', color=C_INK)

    ax = axes[2]
    pr = preds['resp']
    xs, w = np.arange(len(pr)), 0.27
    for off, (col, colour, lab) in enumerate([
            ('err_age', C_AGE, 'age prior'),
            ('err_pop', C_POP, 'best constant'),
            ('err_none', C_NONE, 'no calibration (k = 1)')]):
        ax.bar(xs + (off - 1) * w, pr[col], w, color=colour, label=lab, lw=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(pr.subj)
    ax.set_xlabel('held-out subject')
    ax.set_ylabel('|predicted k − true k|')
    ax.set_title('Respiratory k predicted for a held-out subject\n'
                 f'age prior {pr.err_age.mean():.3f} vs constant {pr.err_pop.mean():.3f} — the prior is worse',
                 fontsize=7.2, color=C_INK, pad=5)
    ax.legend(handlelength=1.1, borderpad=0.2, labelspacing=0.3)
    ax.grid(axis='y', color=C_FAINT, lw=0.4)
    ax.set_axisbelow(True)
    ax.text(-0.26, 1.16, 'C', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', color=C_INK)

    fig.text(0.5, -0.10,
             'k is the whole-night median of estimate ÷ reference for loose peak counting on CRE — '
             'one value per night, bars span each subject\'s two nights. Circles male, squares female.',
             ha='center', fontsize=6.5, color=C_MUTED)
    fig.tight_layout(w_pad=3.4)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    print(f'figure -> {FIG}')


if __name__ == '__main__':
    main()
