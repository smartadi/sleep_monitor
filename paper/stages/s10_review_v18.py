"""Review of manuscript V18 (2026-11-06): the analyses the review asked for.

    R1  delta-burst response with arousal-coincident onsets removed   (V18 §3.5; review Major 5)
    R2  variance by stage with eye-movement epochs removed           (V18 §3.8; Major 4)
    R3  participant-level statistics (n = 6) for Fig. 10-11 / §3.9   (Major 2)
    R4  N3 ridge power relative to the in-band spectral level        (V18 §3.3; Major 1)
    R5  K-complex and delta-burst latencies without S4/S5            (V18 §3.5-3.6; Major 7)
    +   Fig. 8 (K-complex response) drawn from causal envelopes      (V18 Fig. 8; edit point 66)

Section labels in the numbers refer to V18 (writeup/edits/REVIEW_V18_2026-11-06.md and
V18_EDIT_LIST_2026-11-06.md). Every number here is NEW: V18 does not state it yet.

Reads   outputs/tables/s08_delta_kcomplex/{arousal_control,arousal_control_counts,
            kcomplex_latencies}.csv
        outputs/tables/s09_reproducibility/{per_night_features.csv, reproducibility*.csv,
            high_variance_epochs_legacy_grid.parquet}
        _cache/s05_ridges/band_ridge_epochs.parquet
        the sessions themselves (PSG EOG for R2, SEC for R4), via seclib
Writes  outputs/tables/s10_review_v18/*.csv, outputs/figures/s10_review_v18/*.png,
        outputs/numbers/s10_review_v18.csv
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr

from seclib import CACHE_DIR, FS, TAB_DIR, Numbers, iter_sessions
from seclib.figures import save
from seclib.filters import bandpass
from seclib.preprocessing import remove_acc_artifact

STAGE = 's10_review_v18'
REQUIRES = ['s05_ridges', 's08_delta_kcomplex', 's09_reproducibility']

OUT = TAB_DIR / STAGE
SINGLE_ANCHOR = ('S4', 'S5')          # one sync anchor only (notebook: "second anchor
                                      # is not available"), drift uncorrected
STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']
VAR_CHANNELS = ['CH', 'CLE', 'CRE']
HI_Q, LO_Q = 0.90, 0.10
DEAD_FF2 = 1e-4                       # dead-channel epochs, as variance_low_high.py
EOG_BAND = (0.3, 5.0)                 # Hz, eye-movement band for the EOG control
EM_DROP = 0.20                        # drop the night's top 20 % eye-movement epochs
N_DRAWS = 200                         # count-matching draws (as s05)


# ═════════════════════════════════════════════════════════════════════════════
# R1  Delta-burst response without arousal-coincident onsets
# ═════════════════════════════════════════════════════════════════════════════
def r1_arousal_free():
    src = TAB_DIR / 's08_delta_kcomplex'
    ar = pd.read_csv(src / 'arousal_control.csv')
    counts = pd.read_csv(src / 'arousal_control_counts.csv')
    ok = ar.dropna(subset=['peak_z'])
    # paired: recordings where BOTH subsets were analysable
    both = ok.groupby('session').subset.nunique()
    paired = ok[ok.session.isin(both[both == 2].index)]
    subj = (paired.groupby(['subset', 'subject', 'channel', 'band'], as_index=False)
            [['peak_z', 'null_peak_z']].mean())
    wide = subj.pivot_table(index=['subject', 'channel', 'band'], columns='subset',
                            values=['peak_z', 'null_peak_z']).reset_index()
    wide.columns = ['_'.join(c).strip('_') for c in wide.columns]
    rows = []
    for (ch, b), g in wide.groupby(['channel', 'band']):
        rows.append(dict(channel=ch, band=b, n_subj=len(g),
                         peak_all=g.peak_z_all.mean(), peak_af=g.peak_z_arousal_free.mean(),
                         null_af=g.null_peak_z_arousal_free.mean(),
                         n_subj_weaker=int((g.peak_z_arousal_free < g.peak_z_all).sum()),
                         n_subj_af_above_null=int((g.peak_z_arousal_free
                                                   > g.null_peak_z_arousal_free).sum())))
    summ = pd.DataFrame(rows)
    summ['ratio_af_all'] = summ.peak_af / summ.peak_all
    wide.to_csv(OUT / 'r1_arousal_free_per_subject.csv', index=False)
    summ.to_csv(OUT / 'r1_arousal_free_summary.csv', index=False)
    return summ, wide, counts, paired


# ═════════════════════════════════════════════════════════════════════════════
# R2  Variance by stage, eye-movement control
# ═════════════════════════════════════════════════════════════════════════════
def eog_epochs():
    """Per legacy-grid epoch (floor(N/3000) windows from sample 0, as s09's legacy grid):
    anti-phase EOG power, -cov(EOGl, EOGr) in 0.3-5 Hz. Conjugate eye movements drive the
    two periocular electrodes in opposite directions; frontal EEG (slow waves) drives them
    together, so plain EOG power would rise in N3 and removing it would remove N3."""
    rows = []
    n = int(30.0 * FS)
    for s in iter_sessions(profile=False):
        L = bandpass(np.nan_to_num(s.psg['EOGl'].astype(np.float64)), *EOG_BAND, fs=s.fs)
        R = bandpass(np.nan_to_num(s.psg['EOGr'].astype(np.float64)), *EOG_BAND, fs=s.fs)
        m = min(len(L), len(R)) // n
        L, R = L[: m * n].reshape(m, n), R[: m * n].reshape(m, n)
        Lc, Rc = L - L.mean(1, keepdims=True), R - R.mean(1, keepdims=True)
        cov = (Lc * Rc).mean(1)
        vl, vr = Lc.var(1), Rc.var(1)
        rows.append(pd.DataFrame({'session': s.label, 'epoch': np.arange(m),
                                  'em': np.clip(-cov, 0, None),
                                  'eog_lr_r': cov / np.sqrt(vl * vr + 1e-30)}))
        print(f'  EOG {s.label}: {m} epochs')
    return pd.concat(rows, ignore_index=True)


def tail_flags(d):
    """variance_low_high.tails: deciles within each night over all non-dead epochs
    (motion included), fixed before any subset is taken."""
    d = d.copy()
    for ch in VAR_CHANNELS:
        v = d.groupby('session')['var_' + ch]
        d['hi_' + ch] = d['var_' + ch] > v.transform(lambda x: x.quantile(HI_Q))
        d['lo_' + ch] = d['var_' + ch] < v.transform(lambda x: x.quantile(LO_Q))
    return d


def tails_enrichment(d, tag):
    """variance_low_high.enrichment on a subset: the whole-night decile flags are kept,
    enrichment = observed / expected stage share within the subset, per night."""
    rows = []
    for ch in VAR_CHANNELS:
        for sess, g in d.groupby('session'):
            for tail in ('hi', 'lo'):
                tsub = g[g[f'{tail}_{ch}']]
                if len(tsub) < 10:
                    continue
                for st in STAGES:
                    exp = (g.stage == st).mean()
                    if exp < 0.01:
                        continue
                    rows.append(dict(subset=tag, channel=ch, tail=tail, session=sess, stage=st,
                                     enrichment=(tsub.stage == st).mean() / exp))
    return pd.DataFrame(rows)


def tails_summary(e):
    rows = []
    for (subset, ch), g in e.groupby(['subset', 'channel']):
        def pick(tail, st):
            return g[(g['tail'] == tail) & (g.stage == st)].enrichment
        lo_n3, hi_w, hi_n3 = pick('lo', 'N3'), pick('hi', 'Wake'), pick('hi', 'N3')
        rows.append(dict(subset=subset, channel=ch,
                         lo_N3_median=lo_n3.median(), lo_N3_up=int((lo_n3 > 1).sum()),
                         lo_N3_n=len(lo_n3),
                         hi_Wake_median=hi_w.median(), hi_Wake_up=int((hi_w > 1).sum()),
                         hi_Wake_n=len(hi_w), hi_N3_median=hi_n3.median()))
    return pd.DataFrame(rows)


def stage_regression(d):
    """log10 variance on stage dummies (N2 reference) + night fixed effects, with and
    without log10(1 + anti-phase EOG power) as a covariate."""
    X = pd.get_dummies(d.stage)[['Wake', 'N1', 'N3', 'REM']].astype(float).to_numpy()
    S = pd.get_dummies(d.session, drop_first=True).astype(float).to_numpy()
    lem = np.log10(d.em.to_numpy() + 1.0)[:, None]
    rows = []
    for ch in VAR_CHANNELS:
        y = np.log10(d['var_' + ch].to_numpy())
        for tag, cols in (('stage', [X, S]), ('stage + eye movement', [X, S, lem])):
            M = np.column_stack([np.ones(len(d))] + cols)
            b = np.linalg.lstsq(M, y, rcond=None)[0]
            rows.append(dict(channel=ch, model=tag, Wake=b[1], N1=b[2], N3=b[3], REM=b[4],
                             eye_movement=b[-1] if 'eye' in tag else np.nan))
    return pd.DataFrame(rows)


def r2_eye_movements():
    var = pd.read_parquet(TAB_DIR / 's09_reproducibility'
                          / 'high_variance_epochs_legacy_grid.parquet')
    eog = eog_epochs()
    dead = (var[['var_' + c for c in VAR_CHANNELS]] < DEAD_FF2).any(axis=1)
    d = tail_flags(var[~dead])
    n_all = len(d)
    d = d.merge(eog, on=['session', 'epoch'], how='inner')
    assert len(d) == n_all, f'EOG epochs missing: {n_all - len(d)}'
    base = d[~d.motion.astype(bool)]
    thr = base.groupby('session').em.transform(lambda v: v.quantile(1 - EM_DROP))
    no_em = base[base.em <= thr]
    e = pd.concat([tails_enrichment(base, 'motion-free (V18)'),
                   tails_enrichment(no_em, f'motion-free, top {int(EM_DROP * 100)}% '
                                           'eye-movement epochs removed')], ignore_index=True)
    summ = tails_summary(e)
    reg = stage_regression(base)
    em_stage = (base.assign(lem=np.log10(base.em + 1))
                .groupby(['session', 'stage']).lem.median().unstack()[STAGES])
    lr_stage = base.groupby(['session', 'stage']).eog_lr_r.median().unstack()[STAGES]
    e.to_csv(OUT / 'r2_variance_tails_enrichment.csv', index=False)
    summ.to_csv(OUT / 'r2_variance_tails_summary.csv', index=False)
    reg.to_csv(OUT / 'r2_stage_regression.csv', index=False)
    em_stage.to_csv(OUT / 'r2_eye_movement_power_by_stage.csv')
    lr_stage.to_csv(OUT / 'r2_eog_left_right_corr_by_stage.csv')
    fig_r2(e)
    return summ, reg, em_stage, lr_stage


def fig_r2(e):
    import matplotlib.pyplot as plt
    subsets = list(dict.fromkeys(e.subset))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for r, tail in enumerate(('hi', 'lo')):
        for c, ch in enumerate(VAR_CHANNELS):
            ax = axes[r, c]
            for k, sub in enumerate(subsets):
                g = e[(e.subset == sub) & (e['tail'] == tail) & (e.channel == ch)]
                for i, st in enumerate(STAGES):
                    v = np.maximum(g[g.stage == st].enrichment.to_numpy(), 0.03)
                    x = i + (k - 0.5) * 0.36
                    ax.scatter(np.full(len(v), x), v, s=14, alpha=0.6,
                               color=('#7F8C8D', '#C0392B')[k])
                    if len(v):
                        ax.hlines(np.median(v), x - 0.15, x + 0.15, lw=3,
                                  color=('#2C3E50', '#922B21')[k])
            ax.axhline(1, ls='--', color='k', lw=1)
            ax.set_yscale('log')
            ax.set_xticks(range(len(STAGES)))
            ax.set_xticklabels(STAGES)
            ax.set_title(f'{ch}: {"top" if tail == "hi" else "bottom"} decile', loc='left')
        axes[r, 0].set_ylabel('enrichment (observed / expected)')
    fig.text(0.5, 0.01, 'grey: motion-free epochs (V18 §3.8); red: also without the night\'s '
             'top 20% eye-movement epochs (anti-phase EOG). One point per night; bar = median.',
             ha='center')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, 'r2_variance_tails_eye_movement_control', STAGE)


# ═════════════════════════════════════════════════════════════════════════════
# R3  Participant-level statistics
# ═════════════════════════════════════════════════════════════════════════════
def exact_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    null = [spearmanr(x, y[list(p)]).statistic for p in itertools.permutations(range(len(y)))]
    return float(rho), float(np.mean(np.abs(null) >= abs(rho) - 1e-12))


def icc(a, b):
    """ICC(3,1) consistency and ICC(2,1) absolute agreement for two nights."""
    Y = np.c_[a, b].astype(float)
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return ((msr - mse) / (msr + (k - 1) * mse),
            (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


FIG10 = [('dur_above_pct', 'Fig. 10a', '% of night with variance > 10 fF²'),
         ('pct_above_10x_median', 'Fig. 10b', '% of night > 10× night median variance'),
         ('night_median_fF2', 'Fig. 10c', 'median variance')]

ASSOC = [  # x, y, V18 location, what
    ('idx_respiratory', 'impulses_per_hour', 'Fig. 11a / §3.9', 'impulse frequency vs respiratory arousal index'),
    ('arousal_index', 'impulses_per_hour', '§3.9', 'impulse frequency vs total arousal index'),
    ('idx_spontaneous', 'impulses_per_hour', '§3.9', 'impulse frequency vs spontaneous arousal index'),
    ('psqi', 'impulses_per_hour', '§3.9 / Fig. S14c', 'impulse frequency vs PSQI'),
    ('psqi', 'osc_pct', 'Fig. 11b', 'LFO ridge duration (hand-measured) vs PSQI'),
    ('psqi', 'dur_above_pct', 'Fig. 11c', '% of night with variance > 10 fF² vs PSQI'),
    ('psqi', 'night_median_fF2', '§3.9', 'median variance vs PSQI'),
    ('age', 'area_per_hour_fF', '§3.9', 'mean SEC area vs age'),
    ('sws_pct_tst', 'area_per_hour_fF', 'Discussion ¶1', 'mean SEC area vs PSG N3 % of TST'),
]


def r3_participant_stats():
    night = pd.read_csv(TAB_DIR / 's09_reproducibility' / 'per_night_features.csv')
    subj = night.groupby('subject').mean(numeric_only=True)
    rep = []
    for key, where, what in FIG10:
        w = night.pivot(index='subject', columns='night', values=key)
        r, p = pearsonr(w[1], w[2])
        rho, pe = exact_spearman(w[1], w[2])
        i31, i21 = icc(w[1].to_numpy(), w[2].to_numpy())
        rep.append(dict(feature=key, where=where, what=what, n=len(w), pearson_r=r,
                        pearson_p=p, spearman_rho=rho, spearman_exact_p=pe, icc31=i31,
                        icc21=i21, night2_higher=int((w[2] > w[1]).sum())))
    rep = pd.DataFrame(rep)
    assoc = []
    for x, y, where, what in ASSOC:
        if x not in night or y not in night:
            continue
        ok = night[[x, y]].dropna()
        r12, p12 = pearsonr(ok[x], ok[y])
        oks = subj[[x, y]].dropna()
        r6, p6 = pearsonr(oks[x], oks[y])
        rho, pe = exact_spearman(oks[x], oks[y])
        assoc.append(dict(x=x, y=y, where=where, what=what, n_nights=len(ok), r_nights=r12,
                          p_nights=p12, n_subj=len(oks), r_subj=r6, p_subj=p6,
                          rho_subj=rho, p_rho_subj_exact=pe))
    assoc = pd.DataFrame(assoc)
    rep.to_csv(OUT / 'r3_reproducibility_n6.csv', index=False)
    assoc.to_csv(OUT / 'r3_associations_n12_vs_n6.csv', index=False)
    legacy = pd.read_csv(TAB_DIR / 's09_reproducibility' / 'reproducibility_legacy_grid.csv')
    return rep, assoc, legacy


# ═════════════════════════════════════════════════════════════════════════════
# R4  N3 ridge power relative to the in-band spectral level
# ═════════════════════════════════════════════════════════════════════════════
R4_BANDS = {'resp': dict(lo=0.1, hi=0.5, seg=30.0), 'card': dict(lo=0.5, hi=3.0, seg=8.0)}
R4_SMOOTH = 7          # s05 SMOOTH_WINDOWS: the ridge amplitude is read off a 7-window
                       # median-smoothed PSD, so the in-band level is smoothed the same way


def band_level(sig, fs, t_hr, lo, hi, seg):
    """Mean in-band PSD per 30-s window (windows centred on t_hr), temporally median-
    smoothed over R4_SMOOTH windows like the detector's PSD."""
    win = int(30.0 * fs)
    nper = int(seg * fs)
    lvl = np.full(len(t_hr), np.nan)
    for i, t in enumerate(t_hr):
        s0 = int(round(t * 3600.0 * fs - win / 2))
        if s0 < 0 or s0 + win > len(sig):
            continue
        f, p = welch(sig[s0:s0 + win], fs=fs, nperseg=nper, noverlap=nper // 2)
        m = (f >= lo) & (f <= hi)
        lvl[i] = p[m].mean()
    half = R4_SMOOTH // 2
    return np.array([np.nanmedian(lvl[max(0, i - half):i + half + 1])
                     for i in range(len(lvl))])


def r4_relative_ridge_power():
    ep = pd.read_parquet(CACHE_DIR / 's05_ridges' / 'band_ridge_epochs.parquet')
    ep = ep[ep.band.isin(R4_BANDS)].copy()
    ep['band_level'] = np.nan
    for s in iter_sessions(profile=False):
        acc = s.cap['acc_mag']
        for ch in VAR_CHANNELS:
            sig = remove_acc_artifact(s.cap[ch], acc, 0.05, 4.0).astype(np.float64)
            for band, b in R4_BANDS.items():
                m = (ep.session == s.label) & (ep.channel == ch) & (ep.band == band)
                ep.loc[m, 'band_level'] = band_level(sig, s.fs, ep.loc[m, 't_hr'].to_numpy(),
                                                     b['lo'], b['hi'], b['seg'])
        print(f'  in-band level {s.label}')
    ep['rel_ridge_power'] = ep.total_ridge_power / ep.band_level
    ep['log_band_level'] = np.log10(ep.band_level)
    clean = ep[~ep.motion_masked.astype(bool)]
    rows = []
    for feature in ('total_ridge_power', 'rel_ridge_power', 'band_level'):
        for (band, ch), g in clean.groupby(['band', 'channel']):
            rng = np.random.default_rng(0)
            d_mean, d_med = [], []
            for subj, sg in g.groupby('subject'):
                use = sg if feature == 'band_level' else sg[sg.total_ridge_power.notna()]
                n3 = use.loc[use.stage_label == 'N3', feature].dropna().to_numpy()
                oth = use.loc[use.stage_label.isin(['N1', 'N2', 'REM', 'Wake']),
                              feature].dropna().to_numpy()
                if len(n3) < 10 or len(oth) < len(n3):
                    continue
                dm, dd = [], []
                for _ in range(N_DRAWS):
                    pick = rng.choice(oth, len(n3), replace=False)
                    dm.append(n3.mean() - pick.mean())
                    dd.append(np.median(n3) - np.median(pick))
                d_mean.append(np.mean(dm))
                d_med.append(np.mean(dd))
            rows.append(dict(feature=feature, band=band, channel=ch, n_subj=len(d_mean),
                             n3_lower_mean=int(sum(v < 0 for v in d_mean)),
                             n3_lower_median=int(sum(v < 0 for v in d_med))))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'r4_n3_ridge_power_count_matched.csv', index=False)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# R5  Latencies without the single-anchor nights
# ═════════════════════════════════════════════════════════════════════════════
def r5_latencies(paired_ar):
    kl = pd.read_csv(TAB_DIR / 's08_delta_kcomplex' / 'kcomplex_latencies.csv')
    excl = kl.session.str[:2].isin(SINGLE_ANCHOR)
    rows = []
    for est in ('zerophase', 'causal'):
        for ch in ('CLE', 'CRE', 'CH'):
            col = f'{ch}_lat_s_{est}'
            if col not in kl:
                continue
            rows.append(dict(event='K-complex', estimator=est, channel=ch, band='low',
                             n_all=int(kl[col].notna().sum()), median_all=kl[col].median(),
                             n_two_anchor=int(kl.loc[~excl, col].notna().sum()),
                             median_two_anchor=kl.loc[~excl, col].median()))
    a = paired_ar[(paired_ar.subset == 'all')].dropna(subset=['peak_lat_s'])
    ex = a.session.str[:2].isin(SINGLE_ANCHOR)
    for (ch, b), g in a.groupby(['channel', 'band']):
        gx = g[~ex.loc[g.index]]
        rows.append(dict(event='delta burst', estimator='causal', channel=ch, band=b,
                         n_all=len(g), median_all=g.peak_lat_s.median(),
                         n_two_anchor=len(gx), median_two_anchor=gx.peak_lat_s.median()))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'r5_latencies_two_anchor_nights.csv', index=False)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 8 (V18) with causal envelopes and a causal null
# ═════════════════════════════════════════════════════════════════════════════
def fig8_causal():
    import stages.s08_delta_kcomplex as S8
    from stages._s08_delta_kcomplex_figures import fig12_kcomplex
    _, _, marks, _ = S8.kcomplex_marks()
    rng = np.random.default_rng(11)
    lags = np.arange(-S8.KC_PRE, S8.KC_POST) / S8.AFS
    ev = {c: [] for c in S8.CHANNELS}
    null = {c: [] for c in S8.CHANNELS}
    eeg, rows = [], []
    w = (lags >= S8.KC_PEAK_WIN[0]) & (lags <= S8.KC_PEAK_WIN[1])
    for lab in S8.LABELS:
        k = np.round(marks[lab] * S8.AFS).astype(int)
        if k.size == 0:
            continue
        envs, eeg_delta, nrem, motion = S8.load_causal(lab)
        k = k[(k >= S8.KC_PRE) & (k + S8.KC_POST < len(eeg_delta))]
        if k.size == 0:
            continue
        nulls = S8.random_nrem_centers(nrem, motion, k, S8.KC_PRE, S8.KC_POST,
                                       max(k.size * 20, 200), rng)
        st = S8.peri_stack(S8.zscore_on(eeg_delta, nrem), k, S8.KC_PRE, S8.KC_POST)
        if st is not None:
            eeg.append(S8.baselined(st))
        row = {'session': lab, 'n_kcomplex': int(k.size)}
        for ch in S8.CHANNELS:
            tr = S8.lowband_trace(envs, ch, nrem)
            st = S8.baselined(S8.peri_stack(tr, k, S8.KC_PRE, S8.KC_POST))
            ev[ch].append(st)
            if nulls.size:
                ns = S8.peri_stack(tr, nulls, S8.KC_PRE, S8.KC_POST)
                if ns is not None:
                    null[ch].append(S8.baselined(ns))
            m = st.mean(axis=0)
            row[f'{ch}_peak_z'] = float(m[w].max())
            row[f'{ch}_peak_lat_s'] = float(lags[w][np.argmax(m[w])])
        rows.append(row)
    cat = lambda d: {c: (np.vstack(v) if v else np.zeros((0, len(lags)))) for c, v in d.items()}
    ev, null = cat(ev), cat(null)
    eeg = np.vstack(eeg) if eeg else np.zeros((0, len(lags)))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / 'fig8_causal_per_recording.csv', index=False)
    save(fig12_kcomplex(lags, ev, null, eeg, tab, S8.KC_PEAK_WIN, rng),
         'fig8_kcomplex_response_causal', STAGE)
    med = {ch: float(np.median(lags[w][np.argmax(ev[ch][:, w], axis=1)]))
           for ch in S8.CHANNELS if len(ev[ch])}
    nullpk = {ch: float(null[ch].mean(0)[w].max()) for ch in S8.CHANNELS if len(null[ch])}
    evpk = {ch: float(ev[ch].mean(0)[w].max()) for ch in S8.CHANNELS if len(ev[ch])}
    return med, evpk, nullpk, int(sum(len(v) for v in ev.values()) / len(S8.CHANNELS))


# ═════════════════════════════════════════════════════════════════════════════
# Numbers
# ═════════════════════════════════════════════════════════════════════════════
def register(nb, r1, r1_counts, r2, reg, rep, assoc, legacy, r4, r5, f8):
    # R1
    an = r1_counts[r1_counts.analysed]
    nb.add('r1_onsets_arousal_free', 'V18 §3.5', 'arousal-free delta onsets / all onsets '
           '(recordings with >= MIN_ONSETS)', f'{int(an.n_arousal_free.sum())} / '
           f'{int(an.n_onsets.sum())}', source='arousal_control_counts')
    nb.add('r1_peak_range_all', 'V18 §3.5', 'subject-mean post-onset peak, all onsets, '
           'range over 9 channel-band combinations (paired recordings)',
           f'{r1.peak_all.min():.2f}–{r1.peak_all.max():.2f}', unit='z')
    nb.add('r1_peak_range_af', 'V18 §3.5', 'same, arousal-free onsets only',
           f'{r1.peak_af.min():.2f}–{r1.peak_af.max():.2f}', unit='z')
    nb.add('r1_ratio_range', 'V18 §3.5', 'arousal-free / all peak ratio, range over 9 combos',
           f'{r1.ratio_af_all.min():.2f}–{r1.ratio_af_all.max():.2f}')
    nb.add('r1_null_af', 'V18 §3.5', 'random-NREM null peak (arousal-free run), max over combos',
           float(r1.null_af.max()), unit='z')
    nb.add('r1_subj_weaker', 'V18 §3.5', 'participants whose response weakens without '
           'arousals, range over combos (of n)',
           f'{r1.n_subj_weaker.min()}–{r1.n_subj_weaker.max()} of {r1.n_subj.max()}')
    nb.add('r1_subj_above_null', 'V18 §3.5', 'participants whose arousal-free response '
           'exceeds the null, range over combos (of n)',
           f'{r1.n_subj_af_above_null.min()}–{r1.n_subj_af_above_null.max()} of '
           f'{r1.n_subj.max()}')
    # R2
    for _, r in r2.iterrows():
        tag = 'V18' if r.subset.startswith('motion-free (') else 'EOG'
        nb.add(f'r2_lo_n3_{tag}_{r.channel}', 'V18 §3.8',
               f'bottom-decile enrichment in N3, {r.channel}, {r.subset}: median (nights > 1)',
               f'{r.lo_N3_median:.2f} ({r.lo_N3_up}/{r.lo_N3_n})')
        nb.add(f'r2_hi_w_{tag}_{r.channel}', 'V18 §3.8',
               f'top-decile enrichment in Wake, {r.channel}, {r.subset}: median (nights > 1)',
               f'{r.hi_Wake_median:.2f} ({r.hi_Wake_up}/{r.hi_Wake_n})')
        nb.add(f'r2_hi_n3_{tag}_{r.channel}', 'V18 §3.8',
               f'top-decile enrichment in N3, {r.channel}, {r.subset}: median',
               float(r.hi_N3_median))
    for _, r in reg.iterrows():
        nb.add(f'r2_reg_{r.channel}_{"em" if "eye" in r.model else "base"}', 'V18 §3.8',
               f'log10 variance vs N2, {r.channel}, {r.model}: Wake / N1 / N3 / REM',
               f'{r.Wake:+.2f} / {r.N1:+.2f} / {r.N3:+.2f} / {r.REM:+.2f}')
    # R3
    for _, r in rep.iterrows():
        nb.add(f'r3_{r.feature}', f'V18 {r["where"]}', f'{r.what}: Pearson r, p (n = 6); '
               'exact Spearman rho, p; ICC(2,1); nights with night 2 higher',
               f'r {r.pearson_r:.2f}, p {r.pearson_p:.3f}; rho {r.spearman_rho:.2f}, '
               f'p {r.spearman_exact_p:.3f}; ICC {r.icc21:.2f}; {r.night2_higher}/6')
    for _, r in legacy.iterrows():
        nb.add(f'r3_legacy_{r.feature}', 'V18 Fig. 10', f'{r.feature} on the legacy grid: '
               'Pearson r, p (n = 6)', f'r {r.r:.2f}, p {r.p:.3f}')
    for _, r in assoc.iterrows():
        nb.add(f'r3_{r.y}_{r.x}', f'V18 {r["where"]}', f'{r.what}: nights (n = '
               f'{r.n_nights}) vs participants (n = {r.n_subj}), Pearson; exact Spearman (n = 6)',
               f'n12 r {r.r_nights:.2f} p {r.p_nights:.3f} | n6 r {r.r_subj:.2f} '
               f'p {r.p_subj:.3f} | rho {r.rho_subj:.2f} p {r.p_rho_subj_exact:.3f}')
    # R4
    for _, r in r4.iterrows():
        nb.add(f'r4_{r.feature}_{r.band}_{r.channel}', 'V18 §3.3',
               f'{r.feature}, {r.band}, {r.channel}: participants lower in N3 '
               '(count-matched, motion-free), by mean / by median',
               f'{r.n3_lower_mean}/{r.n_subj} / {r.n3_lower_median}/{r.n_subj}')
    # R5
    for _, r in r5.iterrows():
        nb.add(f'r5_{r.event}_{r.estimator}_{r.channel}_{r.band}'.replace(' ', '_'),
               'V18 §3.5-3.6', f'{r.event} median peak latency, {r.estimator}, {r.channel} '
               f'{r.band}: all nights (n) vs without S4/S5 (n)',
               f'{r.median_all:.2f} ({r.n_all}) vs {r.median_two_anchor:.2f} '
               f'({r.n_two_anchor})', unit='s')
    # Fig. 8 causal
    med, evpk, nullpk, n = f8
    nb.add('f8_causal_latency', 'V18 §3.6 / Fig. 8', 'K-complex median per-event latency, '
           'causal: CLE / CRE / CH', ' / '.join(f'{med[c]:.1f}' for c in ('CLE', 'CRE', 'CH')),
           unit='s')
    nb.add('f8_causal_peak_vs_null', 'V18 Fig. 8', 'causal grand-mean peak vs causal null: '
           'CLE / CRE / CH', ' / '.join(f'{evpk[c]:.2f} vs {nullpk[c]:.2f}'
                                        for c in ('CLE', 'CRE', 'CH')), unit='z')
    nb.add('f8_n_events', 'V18 Fig. 8', 'K-complex events in the causal figure', n)


def fig_s6(channel='CRE'):
    """R7: Fig. S6 panels (one per night) in the Fig. 5 layout, LFO from the current
    low-band detector (_s05_lowband), saved as figS6_<night>_<channel>."""
    import stages.s05_ridges as S5
    from stages import _s05_lowband as LB
    for s in iter_sessions():
        out = S5.fig5(LB.night(s.cap[channel], s.fs), s=s, channel=channel, stage=STAGE)
        out.replace(out.with_name(f'figS6_{s.label}_{channel}.png'))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    nb = Numbers(STAGE)
    print('  R1 arousal-free delta bursts')
    r1, _, r1_counts, paired = r1_arousal_free()
    print('  R2 eye-movement control')
    r2, reg, _, _ = r2_eye_movements()
    print('  R3 participant-level statistics')
    rep, assoc, legacy = r3_participant_stats()
    print('  R4 relative ridge power')
    r4 = r4_relative_ridge_power()
    print('  R5 latencies without S4/S5')
    r5 = r5_latencies(paired)
    print('  Fig. 8, causal')
    f8 = fig8_causal()
    print('  Fig. S6, current low-band detector')
    fig_s6()
    register(nb, r1, r1_counts, r2, reg, rep, assoc, legacy, r4, r5, f8)
    nb.save()


if __name__ == '__main__':
    run()
