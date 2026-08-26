#!/usr/bin/env python
"""
Per-window k as a SECONDARY biomarker (cache-only, no raw reprocessing).

Motivation (see CONTINUATION_RATE_DETECTION.md): the paper's rate figures use a
single static per-session k (median peaks/GT). A recurring question is whether a
*per-window* k(t) carries physiological information that could be used as a marker
for OTHER analyses (staging, autonomic state) even though it is NOT the primary
rate detector.

The honest difficulty already established:
  - within-session corr(k_gt, rate) ~= -0.83  => naive per-epoch k is mostly 1/rate.
  - pooled stage stats (old p=1e-130) are inflated by between-session mean-matching.

So this script:
  1. Reproduces the rate confound for the supervised k_gt = peaks_loose / gt.
  2. Defines a GT-FREE, deployable marker  M = peaks_loose / hilbert
     ("morphological multiplicity": how many local pulse maxima per dominant
     oscillation -> proxy for biphasic / dicrotic waveform complexity).
  3. Tests whether M holds information, using WITHIN-session / LOSO statistics only:
       (a) rate deconfound check      : within-session corr(M, rate) ~ 0
       (b) temporal structure         : lag-1 autocorr vs epoch-shuffle null
       (c) sleep-stage discrimination : within-session Kruskal-Wallis + direction
                                        consistency (vs the inflated pooled p)
       (d) subject reproducibility    : night-1 vs night-2 mean M, ICC
       (e) incremental predictive use : LOSO AUC for N3 / REM one-vs-rest from M
  4. Emits a marker table other analyses can join on, plus a summary figure.

Outputs:
  reports/rates/mask/k_biomarker_perwindow.csv     (per (session,epoch) marker)
  reports/rates/mask/k_biomarker_stats.csv         (all test results)
  writeup/figures/mask_rate_detection/fig_k_biomarker.png

Run: C:/Users/adity/anaconda3/python.exe scripts/analyze_k_biomarker_perwindow.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import functools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ART = ROOT / 'artifacts'
RPT = ROOT / 'reports' / 'rates' / 'mask'
FIG = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'
RPT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLOR = {'Wake': '#7f8c8d', 'N1': '#f39c12', 'N2': '#3498db',
               'N3': '#2c3e50', 'REM': '#e74c3c'}
CLIP = (0.3, 5.0)


def clip_ratio(a, b):
    r = np.where((np.isfinite(a)) & (np.isfinite(b)) & (b > 0), a / b, np.nan)
    return np.where((r > CLIP[0]) & (r < CLIP[1]), r, np.nan)


def within_session_corr(df, xcol, ycol):
    """Mean +/- sd of per-session Pearson r (Fisher-z averaged)."""
    zs, rs = [], []
    for _, g in df.groupby('session'):
        x, y = g[xcol].values, g[ycol].values
        v = np.isfinite(x) & np.isfinite(y)
        if v.sum() < 30:
            continue
        r = np.corrcoef(x[v], y[v])[0, 1]
        if np.isfinite(r):
            rs.append(r)
            zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
    if not zs:
        return np.nan, np.nan, 0
    rbar = np.tanh(np.mean(zs))
    return float(rbar), float(np.std(rs)), len(rs)


def lag1_autocorr(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return np.nan
    return float(np.sum(x[:-1] * x[1:]) / denom)


def shuffle_null_autocorr(x, n=200, rng=None):
    """Distribution of lag-1 autocorr under epoch shuffling (destroys time order)."""
    rng = rng or np.random.default_rng(0)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan, np.nan
    vals = []
    for _ in range(n):
        vals.append(lag1_autocorr(rng.permutation(x)))
    vals = np.array(vals)
    return float(np.nanmean(vals)), float(np.nanpercentile(vals, 95))


def icc1(groups):
    """One-way ICC(1) from a list of arrays (each = one subject's night means)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan
    k = np.array([len(g) for g in groups])
    if not np.all(k == k[0]):
        m = k.min()
        groups = [g[:m] for g in groups]
        k = m
    else:
        k = k[0]
    n = len(groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    means = np.array([g.mean() for g in groups])
    ssb = k * np.sum((means - grand) ** 2)
    ssw = np.sum([np.sum((g - g.mean()) ** 2) for g in groups])
    msb = ssb / (n - 1)
    msw = ssw / (n * (k - 1))
    denom = msb + (k - 1) * msw
    return float((msb - msw) / denom) if denom != 0 else np.nan


def build_marker(df, band, channel='diff'):
    b = df[(df.band == band) & (df.channel == channel)].copy()
    b = b.sort_values(['session', 'epoch']).reset_index(drop=True)
    b['M'] = clip_ratio(b['r_peaks_loose'].values, b['r_hilbert'].values)
    b['k_gt'] = clip_ratio(b['r_peaks_loose'].values, b['gt_hz'].values)
    b['rate'] = b['gt_hz'].values
    # rate-deconfounded supervised residual: regress k_gt on 1/rate per session
    resid = np.full(len(b), np.nan)
    for sess, g in b.groupby('session'):
        idx = g.index.values
        y = g['k_gt'].values
        xr = 1.0 / g['rate'].values
        v = np.isfinite(y) & np.isfinite(xr)
        if v.sum() >= 20:
            A = np.vstack([xr[v], np.ones(v.sum())]).T
            coef, *_ = np.linalg.lstsq(A, y[v], rcond=None)
            pred = coef[0] * xr + coef[1]
            resid[idx] = y - pred
    b['k_resid'] = resid
    return b


def subject_of(sess):
    return sess.split('N')[0]


def analyze_band(df, band, rng):
    unit = 'br/min' if band == 'resp' else 'BPM'
    b = build_marker(df, band, 'diff')
    staged = b[b.stage.isin(STAGE_ORDER)].copy()
    rows = []

    def rec(test, **kw):
        d = {'band': band, 'test': test}
        d.update(kw)
        rows.append(d)
        return d

    print('\n' + '=' * 72)
    print(f'{band.upper()}  (diff channel)  unit={unit}')
    print('=' * 72)

    # (a) rate confound / deconfound
    r_kgt, s_kgt, n1 = within_session_corr(b, 'k_gt', 'rate')
    r_m, s_m, _ = within_session_corr(b, 'M', 'rate')
    r_res, s_res, _ = within_session_corr(b, 'k_resid', 'rate')
    print(f'  within-session corr(k_gt , rate) = {r_kgt:+.3f} +/- {s_kgt:.3f}  (confound)')
    print(f'  within-session corr(M    , rate) = {r_m:+.3f} +/- {s_m:.3f}  (GT-free marker)')
    print(f'  within-session corr(kres , rate) = {r_res:+.3f} +/- {s_res:.3f}  (deconfounded)')
    rec('corr_kgt_rate', r=r_kgt, sd=s_kgt, n_sess=n1)
    rec('corr_M_rate', r=r_m, sd=s_m)
    rec('corr_kresid_rate', r=r_res, sd=s_res)

    # (b) temporal structure of M
    ac, null_ac, null_hi = [], [], []
    for sess, g in b.groupby('session'):
        x = g['M'].values
        a = lag1_autocorr(x)
        nm, nhi = shuffle_null_autocorr(x, n=200, rng=rng)
        if np.isfinite(a):
            ac.append(a); null_ac.append(nm); null_hi.append(nhi)
    ac = np.array(ac)
    frac_above = float(np.mean(ac > np.array(null_hi))) if len(ac) else np.nan
    print(f'  M lag-1 autocorr = {np.nanmean(ac):.3f} (null {np.nanmean(null_ac):+.3f}); '
          f'{frac_above*100:.0f}% of sessions exceed shuffle-95%')
    rec('M_autocorr', mean=float(np.nanmean(ac)), null_mean=float(np.nanmean(null_ac)),
        frac_sessions_above_null=frac_above, n_sess=len(ac))

    # (c) stage discrimination: pooled (inflated) vs within-session
    grp = [staged[staged.stage == s]['M'].dropna().values for s in STAGE_ORDER]
    grp = [g for g in grp if len(g) > 5]
    H, p_pool = stats.kruskal(*grp) if len(grp) >= 2 else (np.nan, np.nan)
    n_sig, n_tot, etas, medians = 0, 0, [], {s: [] for s in STAGE_ORDER}
    for sess, g in staged.groupby('session'):
        sub = [g[g.stage == s]['M'].dropna().values for s in STAGE_ORDER]
        present = [s for s, a in zip(STAGE_ORDER, sub) if len(a) > 5]
        sub = [a for a in sub if len(a) > 5]
        if len(sub) < 2:
            continue
        n_tot += 1
        Hs, ps = stats.kruskal(*sub)
        N = sum(len(a) for a in sub)
        eta = (Hs - len(sub) + 1) / (N - len(sub)) if N > len(sub) else np.nan
        etas.append(eta)
        if ps < 0.05:
            n_sig += 1
        for s, a in zip(present, sub):
            medians[s].append(np.median(a))
    print(f'  stage KW pooled  p = {p_pool:.2e}   (INFLATED by pooling)')
    print(f'  stage KW within  : {n_sig}/{n_tot} sessions p<0.05, median eta^2 = {np.nanmedian(etas):.3f}')
    stage_med = {s: (float(np.mean(v)) if v else np.nan) for s, v in medians.items()}
    print('  per-stage mean-of-session-medians (M): ' +
          '  '.join(f'{s}={stage_med[s]:.3f}' for s in STAGE_ORDER if np.isfinite(stage_med[s])))
    # direction consistency via Friedman (sessions x stages) on the 3 always-present stages
    core = ['N1', 'N2', 'N3', 'REM']
    mat = []
    for sess, g in staged.groupby('session'):
        row = [np.median(g[g.stage == s]['M'].dropna().values) if (g.stage == s).sum() > 5 else np.nan
               for s in core]
        if all(np.isfinite(row)):
            mat.append(row)
    fried_p = np.nan
    if len(mat) >= 3:
        fried_stat, fried_p = stats.friedmanchisquare(*np.array(mat).T)
    print(f'  stage direction consistency (Friedman over {len(mat)} sessions, {core}) p = {fried_p:.2e}')
    rec('stage_pooled_kw', p=float(p_pool))
    rec('stage_within_kw', n_sig=n_sig, n_sess=n_tot, median_eta2=float(np.nanmedian(etas)),
        friedman_p=float(fried_p), **{f'M_{s}': stage_med[s] for s in STAGE_ORDER})

    # (d) reproducibility across nights
    sess_mean = b.groupby('session')['M'].median()
    subs = {}
    for sess, m in sess_mean.items():
        subs.setdefault(subject_of(sess), []).append(m)
    icc = icc1(list(subs.values()))
    n1v = [v[0] for v in subs.values() if len(v) == 2]
    n2v = [v[1] for v in subs.values() if len(v) == 2]
    r_nights = np.corrcoef(n1v, n2v)[0, 1] if len(n1v) >= 3 else np.nan
    print(f'  subject reproducibility: ICC(1) = {icc:.3f}; night1-vs-night2 r = {r_nights:+.3f} '
          f'(n={len(n1v)} subjects)')
    rec('reproducibility', icc1=float(icc), night_r=float(r_nights), n_subj=len(n1v))

    # (e) incremental predictive value (LOSO one-vs-rest AUC from M features)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    feat = staged.copy()
    # local context features of the marker
    feat['M_r5'] = feat.groupby('session')['M'].transform(
        lambda s: s.rolling(11, center=True, min_periods=3).mean())
    feat['M_sd5'] = feat.groupby('session')['M'].transform(
        lambda s: s.rolling(11, center=True, min_periods=3).std())
    Xcols = ['M', 'M_r5', 'M_sd5']
    feat = feat.dropna(subset=Xcols)
    auc_res = {}
    for target in ['N3', 'REM']:
        aucs = []
        for held in feat.session.unique():
            tr = feat[feat.session != held]
            te = feat[feat.session == held]
            ytr = (tr.stage == target).astype(int)
            yte = (te.stage == target).astype(int)
            if yte.nunique() < 2 or ytr.nunique() < 2:
                continue
            clf = LogisticRegression(max_iter=500, class_weight='balanced')
            clf.fit(tr[Xcols], ytr)
            pr = clf.predict_proba(te[Xcols])[:, 1]
            aucs.append(roc_auc_score(yte, pr))
        auc_res[target] = (float(np.mean(aucs)), float(np.std(aucs)), len(aucs))
        print(f'  LOSO AUC ({target} vs rest) from M-features = '
              f'{auc_res[target][0]:.3f} +/- {auc_res[target][1]:.3f} (n={auc_res[target][2]})')
    rec('loso_auc', **{f'{t}_auc': auc_res[t][0] for t in auc_res},
        **{f'{t}_sd': auc_res[t][1] for t in auc_res})

    return b, pd.DataFrame(rows), stage_med, ac, np.array(null_hi)


def make_figure(cardb, card_stage_med, card_ac, card_null, resb):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # A: confound scatter k_gt vs rate (cardiac)
    ax = fig.add_subplot(gs[0, 0])
    s = cardb.sample(min(3000, len(cardb)), random_state=1)
    ax.scatter(s['rate'] * 60, s['k_gt'], s=4, alpha=0.25, color='#e74c3c')
    ax.set_xlabel('True HR (BPM)'); ax.set_ylabel('k_gt = peaks/GT')
    ax.set_title('A. Supervised k is ~1/rate (confounded)', fontsize=10, fontweight='bold')

    # B: GT-free marker M vs rate (deconfounded)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(s['rate'] * 60, s['M'], s=4, alpha=0.25, color='#2ecc71')
    ax.set_xlabel('True HR (BPM)'); ax.set_ylabel('M = peaks/hilbert')
    ax.set_title('B. GT-free marker M is rate-flat', fontsize=10, fontweight='bold')

    # C: M by stage (within-session z-scored, pooled for display)
    ax = fig.add_subplot(gs[0, 2])
    zparts = []
    for sess, g in cardb[cardb.stage.isin(STAGE_ORDER)].groupby('session'):
        z = (g['M'] - g['M'].mean()) / (g['M'].std() + 1e-9)
        zparts.append(pd.DataFrame({'stage': g['stage'].values, 'z': z.values}))
    zdf = pd.concat(zparts)
    data = [zdf[zdf.stage == st]['z'].dropna().values for st in STAGE_ORDER]
    bp = ax.boxplot(data, tick_labels=STAGE_ORDER, showfliers=False, patch_artist=True)
    for patch, st in zip(bp['boxes'], STAGE_ORDER):
        patch.set_facecolor(STAGE_COLOR[st]); patch.set_alpha(0.6)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_ylabel('within-session z(M)')
    ax.set_title('C. Marker vs sleep stage', fontsize=10, fontweight='bold')

    # D: example session M(t) trace + stage bands
    ax = fig.add_subplot(gs[1, :2])
    ex = 'S1N1'
    g = cardb[cardb.session == ex].sort_values('t_hr')
    if len(g) == 0:
        g = cardb[cardb.session == cardb.session.iloc[0]].sort_values('t_hr')
        ex = g.session.iloc[0]
    ax.plot(g['t_hr'], g['M'], color='#16a085', lw=0.7, label='M(t)')
    ax.plot(g['t_hr'], g['M'].rolling(11, center=True, min_periods=3).mean(),
            color='#c0392b', lw=1.6, label='M(t) smoothed')
    for st in STAGE_ORDER:
        m = g.stage == st
        if m.any():
            ax.scatter(g['t_hr'][m], np.full(m.sum(), g['M'].min() - 0.05),
                       c=STAGE_COLOR[st], s=8, marker='s')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('M')
    ax.set_title(f'D. Marker time course with sleep stages ({ex}); '
                 'square row = hypnogram', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # E: autocorr M vs shuffle null
    ax = fig.add_subplot(gs[1, 2])
    xpos = np.arange(len(card_ac))
    ax.bar(xpos - 0.2, card_ac, width=0.4, color='#16a085', label='M lag-1 AC')
    ax.bar(xpos + 0.2, card_null, width=0.4, color='#bdc3c7', label='shuffle 95%')
    ax.set_xlabel('session index'); ax.set_ylabel('lag-1 autocorr')
    ax.set_title('E. Temporal structure > noise', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    fig.suptitle('Per-window k as a secondary biomarker (cardiac, diff channel): '
                 'GT-free morphological marker M = peaks/hilbert',
                 fontsize=13, fontweight='bold', y=0.995)
    out = FIG / 'fig_k_biomarker.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved: {out}')


def main():
    df = pd.read_parquet(ART / 'mask_phase_a.parquet')
    print(f'Loaded cache: {df.shape}')
    rng = np.random.default_rng(42)

    cardb, cstats, cmed, cac, cnull = analyze_band(df, 'card', rng)
    respb, rstats, rmed, rac, rnull = analyze_band(df, 'resp', rng)

    allstats = pd.concat([cstats, rstats], ignore_index=True)
    allstats.to_csv(RPT / 'k_biomarker_stats.csv', index=False)

    # marker export table (both bands, diff channel) for downstream joins
    keep = ['session', 'epoch', 't_hr', 'stage', 'rate', 'M', 'k_gt', 'k_resid']
    marker = pd.concat([
        cardb[keep].assign(band='card'),
        respb[keep].assign(band='resp'),
    ], ignore_index=True)
    marker.to_csv(RPT / 'k_biomarker_perwindow.csv', index=False)
    print(f'\nMarker table: {RPT / "k_biomarker_perwindow.csv"}  ({len(marker)} rows)')
    print(f'Stats table : {RPT / "k_biomarker_stats.csv"}')

    make_figure(cardb, cmed, cac, cnull, respb)
    print('\nDONE.')


if __name__ == '__main__':
    main()
