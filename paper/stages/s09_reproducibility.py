"""
Night-to-night reproducibility and exploratory associations of SEC variance features (Results §3.7-3.8).

Manuscript items
    §3.7 ¶230   R² night 1 vs night 2 (n = 6 participants) for three features:
                % of night with CLE-CRE variance > 10 fF² (text: "absolute signal
                amplitude"), % of night > 10x the nightly median, median variance
    Fig 10      the three night-1 vs night-2 scatters with fit line, R² and n
    §3.8 ¶233   the feature list and the figure reference
    §3.8 ¶238   impulse frequency vs total / respiratory / spontaneous arousal index;
                impulse frequency vs PSQI; oscillation duration vs mean SEC area;
                two-night mean oscillation duration vs PSQI; % of night > 10 fF² vs
                PSQI; median variance vs PSQI; mean SEC area vs age
    Fig 11      (a) impulse frequency vs respiratory arousal index, (b) two-night mean
                oscillation duration vs PSQI, (c) % of night > 10 fF² vs PSQI
    ¶248, ¶259  "mean SEC area showed a moderate association with PSG-measured SWS
                duration"
    tables      TAB_DIR/s09_reproducibility/: high_variance_epochs.parquet (shared grid),
                high_variance_epochs_legacy_grid.parquet, prof_metrics_per_night.csv,
                prof_metrics_threshold_policy.csv (+ _legacy_grid versions),
                arousal_counts.csv, psg_sws.csv, reproducibility.csv, associations.csv

Definitions (CLE-CRE throughout, as in the legacy code)
    variance          per 30-s epoch, variance of the channel after a 10-Hz zero-phase
                      Butterworth low pass (order 3); unscored epochs dropped
    % > 10 fF²        share of a night's epochs whose variance exceeds an ABSOLUTE
                      10 fF², the same threshold for every night
    % > 10x median    share above 10 x that night's median variance
    median variance   the night's median epoch variance, fF²
    impulse frequency contiguous runs of epochs above 10 fF², per hour of recording
                      (epochs x 30 s), "events/h"
    mean SEC area     mean |CLE-CRE level - session mean| over the night's epochs, fF
                      (= the integral of |d| divided by the hours; the text's
                      "fF·h/h"); d_fF of s02_overnight
    arousal index     PSG-scored cortical arousals with onset in scored sleep, per hour
                      of sleep (TST = sleep epochs x 30 s); subtypes from the scorer's
                      exact labels (SpO2 arousals counted as respiratory)
    PSG SWS %         N3 epochs / sleep epochs x 100 of the PSG hypnogram (also given
                      per hour of recording)
    oscillation       EXTERNAL. Hand-measured duration of low-frequency SEC oscillation
    duration          episodes, as % of recording time, from the co-author workbook
                      (writeup/review/Overnight_sleep_subject_list_V3.xlsx, sheet
                      "SWS % calculation", column Z = sum of ruler lengths / ratio /
                      Table-1 duration x 100). The workbook and the manuscript call it
                      "SWS duration" / "SWS time/TST"; it is not derived from the PSG
                      and no code in the repository computes it. Hard-coded below.
    PSQI, age         Table 1 of the manuscript; hard-coded below.

Statistics: Pearson r and R² = r², with n stated (n = 6 participants when one value per
participant, n = 12 nights otherwise -- two nights of the same participant are not
independent). Descriptive; p-values only in the registry notes.

Ported from:
    analysis/mean_value/high_variance_zones.py   epoch_reduce, build (the variance table)
    analysis/mean_value/prof_metrics.py          runs_above, metric1, metric23, POLICIES,
                                                 policy_table, the per-night table of main()
    analysis/swa_validation/arousal_index.py     LABEL_GROUP, classify, stage_at, build
    co-author Excel workbook (sheets "reproducibility", "EEG arousal", "Cortical and SWS",
    "SWS %", "SWS % calculation")               Fig 10, Fig 11 and every §3.7-3.8 R/R²

Changes from the legacy code:
    * The legacy variance table used its own grid: floor(N / 3000) back-to-back 30-s
      windows from the first sample, staged by the profile epoch under each centre
      (9,227 scored epochs). It is reproduced exactly (legacy_grid tables) and then
      replaced by the shared s01 grid (PSG profile epochs, 9,221 scored), whose motion
      flag is s01's. Every number below is on the shared grid; the legacy-grid value
      is in the note where it differs.
    * CORRECTION: Fig 10 and every §3.7-3.8 R/R² were computed in Excel from values
      read off a plot ("Estimated from graph", threshold_policy.png). They are now
      computed from the exact per-night values: R² 0.63 / 0.71 / 0.68 become 0.648 /
      0.717 / 0.859 on the legacy grid and 0.637 / 0.710 / 0.855 on the shared grid.
      The grid switch also moves impulse frequency vs respiratory index 0.502 -> 0.514.
    * CORRECTION: Fig 10a and ¶230 call the first feature "absolute (signal) amplitude".
      It is the % of the night with CLE-CRE variance above an absolute 10 fF²; the
      panel is labelled so. The same applies to Fig 11c and ¶233.
    * CORRECTION: oscillation duration vs mean SEC area, R = 0.68, came from a workbook
      column ("Cortical and SWS"!I) holding the durations in night-1-then-night-2
      order beside area values in session order. Correctly paired R = 0.48.
    * CORRECTION: signs. Spontaneous arousal index (-0.48, text "positive 0.48"),
      PSQI vs impulse frequency (-0.57, text 0.56), PSQI vs median variance (-0.55
      exact; -0.48 from the graph-read values; text 0.48), age vs mean SEC area
      (-0.63, text 0.63).
    * CORRECTION: R² = 0.10 for the total arousal index has no source; the exact value
      is 0.006 (the limb index gives 0.09).
    * CORRECTION: ¶248/¶259 say mean SEC area is associated with "PSG-measured SWS
      duration". The association in the workbook is with the hand-measured oscillation
      duration. PSG SWS % (N3 / TST) is computed here from the hypnograms and its
      association with mean SEC area registered separately.
    * CORRECTION (numbers only): ¶230 cites "Fig. 11" for Fig 10, ¶233 cites "Fig. 12"
      for Fig 11, and ¶238 cites "Fig. Sxx".
    * PSQI and age come from Table 1 (hard-coded) instead of
      analysis/rates/outputs/k_vs_age_per_subject.csv.
    * Dropped: the threshold sweep (2-50 fF²), stage-enrichment of high-variance epochs,
      the CAP-detected arousal columns of arousal_counts.csv, and every legacy figure
      (method, metric1-3, results_table, threshold_policy, arousal_*) -- none is in the
      paper. Fig 10 and Fig 11 are drawn here instead of in Excel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from seclib import CACHE_DIR, TAB_DIR, iter_sessions
from seclib.config import CAP_UNIT, CAP_UNIT_SQ
from seclib.figures import save
from seclib.filters import lowpass
from seclib.loader import load_arousals, load_autonomic_arousals
from seclib.numbers import Numbers

STAGE = 's09_reproducibility'
REQUIRES = ['s01_recordings', 's02_overnight']

EPOCH_SEC = 30.0
EPOCH_MIN = EPOCH_SEC / 60.0
EPOCH_HR = EPOCH_SEC / 3600.0
LP_HZ = 10.0
VAR_CHANNELS = ['CH', 'CLE', 'CRE', 'CLE-CRE']
CH = 'CLE-CRE'                          # the channel every per-night metric uses
TOP_DECILE = 0.90
HEADLINE = 10.0                         # fF², absolute threshold
STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']
CODE2STAGE = {4: 'Wake', 3: 'N1', 2: 'N2', 1: 'N3', 0: 'REM'}
SLEEP_CODES = {0, 1, 2, 3}

# Table 1 of the manuscript (V11, "Recording sessions and demographics"). Not in the
# data files; the only per-participant clinical data the paper uses.
PARTICIPANTS = {
    'S1': {'age': 61, 'sex': 'F', 'psqi': 9},
    'S2': {'age': 66, 'sex': 'M', 'psqi': 4},
    'S3': {'age': 37, 'sex': 'M', 'psqi': 9},
    'S4': {'age': 54, 'sex': 'M', 'psqi': 8},
    'S5': {'age': 55, 'sex': 'F', 'psqi': 6},
    'S6': {'age': 25, 'sex': 'M', 'psqi': 6},
}

# EXTERNAL. Hand-measured low-frequency oscillation duration, % of recording time:
# writeup/review/Overnight_sleep_subject_list_V3.xlsx, sheet "SWS % calculation",
# column Z (ruler lengths L:Q / ratio K, summed, / duration J x 100). Called "SWS %" /
# "SWS time/TST" in the workbook and "SWS duration" in ¶233; it is not PSG staging.
OSC_DURATION_PCT = {
    'S1N1': 0.0, 'S1N2': 30.471821756225431, 'S2N1': 49.482535575679172,
    'S2N2': 50.020142339196994, 'S3N1': 39.026629935720841, 'S3N2': 25.074232926426919,
    'S4N1': 56.957928802588995, 'S4N2': 45.514950166112953, 'S5N1': 18.664800186648005,
    'S5N2': 70.872890295358644, 'S6N1': 37.445802128498215, 'S6N2': 78.520095821133879,
}
# The workbook's "Cortical and SWS"!I column, which the R = 0.68 was computed from: the
# same durations listed night 1 of S1..S6, then night 2 of S1..S6, against area values
# in session order. Kept only to show where 0.68 came from.
OSC_MISPAIRED_ORDER = ['S1N1', 'S2N1', 'S3N1', 'S4N1', 'S5N1', 'S6N1',
                       'S1N2', 'S2N2', 'S3N2', 'S4N2', 'S5N2', 'S6N2']

# scorer label -> arousal subtype (exact labels; SpO2 arousals are desaturation-related)
LABEL_GROUP = {
    'Arousal': 'spontaneous',
    'Respiratory Arousal': 'respiratory',
    'SpO2 Arousal': 'respiratory',
    'LM Arousal': 'limb',
    'PLM Arousal': 'limb',
    'Cardiac Arousal': 'cardiac',
}
AROUSAL_TYPES = ['spontaneous', 'respiratory', 'limb', 'cardiac']

SUBJ_COLORS = {f'S{k + 1}': plt.cm.tab10(k) for k in range(6)}
NIGHT_MARKER = {1: 'o', 2: 's'}


# ── per-epoch variance ───────────────────────────────────────────────────────

def lowpassed(s):
    cap = {c: s.cap[c].astype(np.float64) for c in ('CH', 'CLE', 'CRE')}
    cap['CLE-CRE'] = cap['CLE'] - cap['CRE']
    return {c: lowpass(v, LP_HZ, s.fs) for c, v in cap.items()}


def epoch_reduce(x, n, fn):
    m = len(x) // n
    return fn(x[: m * n].reshape(-1, n), axis=1)


def variance_legacy_grid(s, lp):
    """The legacy grid: floor(N/3000) windows from the first sample, staged by centre."""
    n = int(EPOCH_SEC * s.fs)
    nep = min(len(v) for v in lp.values()) // n
    sp = s.sleep_profile
    codes = np.full(nep, -1)
    t_ep = (np.arange(nep) * EPOCH_SEC + EPOCH_SEC / 2) / 3600.0
    prof_t = np.asarray(sp['t_ep_hr'], float)
    prof_c = np.asarray(sp['codes'], int)
    j = np.searchsorted(prof_t, t_ep) - 1
    ok = (j >= 0) & (j < len(prof_c))
    codes[ok] = prof_c[j[ok]]
    rec = {'session': s.label, 'subject': s.label[:2], 'epoch': np.arange(nep),
           't_hr': t_ep, 'stage': [CODE2STAGE.get(c, '?') for c in codes],
           'acc_sd': epoch_reduce(s.cap['acc_mag'].astype(np.float64), n, np.std)[:nep]}
    for ch in VAR_CHANNELS:
        rec['var_' + ch] = epoch_reduce(lp[ch], n, np.var)[:nep]
    return pd.DataFrame(rec)


def variance_shared_grid(s, lp, ep):
    """The same variance on the shared s01 epoch grid (sample indices i0, i1)."""
    rec = {'session': s.label, 'subject': s.label[:2], 'epoch': ep.epoch.to_numpy(),
           't_hr': ep.t_hr.to_numpy(), 'stage': ep.stage.to_numpy(),
           'acc_sd': ep.acc_sd.to_numpy(), 'motion': ep.motion.to_numpy()}
    for ch in VAR_CHANNELS:
        rec['var_' + ch] = np.array([lp[ch][a:b].var() for a, b in zip(ep.i0, ep.i1)])
    return pd.DataFrame(rec)


def finish_variance(d, own_motion):
    """Drop unscored epochs, flag motion (legacy: per-session top decile) and top-decile
    variance per channel."""
    d = d[d.stage.isin(STAGES)].copy()
    if own_motion:
        d['motion'] = d.groupby('session').acc_sd.transform(
            lambda v: v > v.quantile(TOP_DECILE))
    for ch in VAR_CHANNELS:
        d['hi_' + ch] = d.groupby('session')['var_' + ch].transform(
            lambda v: v > v.quantile(TOP_DECILE))
    return d.reset_index(drop=True)


# ── per-night metrics ────────────────────────────────────────────────────────

def runs_above(mask):
    """Start and end indices of each contiguous True run -- one run = one impulse."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], m.view(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def metric1(imb):
    """Absolute area under the mean-referenced CLE-CRE curve, per night."""
    rows = []
    for sess, g in imb.groupby('session'):
        d = g['d_fF'].to_numpy(float)
        d = d[np.isfinite(d)]
        rows.append({'session': sess, 'hours': len(d) * EPOCH_HR,
                     'area_fF_h': float(np.abs(d).sum() * EPOCH_HR),
                     'area_per_hour_fF': float(np.abs(d).mean())})
    return pd.DataFrame(rows)


def metric23(var, thr, motion_free=False):
    """Duration above threshold and impulse count, per night."""
    rows = []
    for sess, g in var.groupby('session'):
        g = g.sort_values('epoch')
        if motion_free:
            g = g[~g.motion]
        v = g['var_' + CH].to_numpy(float)
        hours = len(v) * EPOCH_HR
        above = v > thr
        runs = runs_above(above)
        lens = np.array([b - a for a, b in runs], float) * EPOCH_MIN
        rows.append({'session': sess, 'hours': hours,
                     'dur_above_min': float(above.sum() * EPOCH_MIN),
                     'dur_above_pct': float(100.0 * above.mean()) if len(v) else np.nan,
                     'n_impulses': len(runs),
                     'impulses_per_hour': float(len(runs) / hours) if hours else np.nan,
                     'median_impulse_min': float(np.median(lens)) if len(lens) else 0.0})
    return pd.DataFrame(rows)


POLICIES = [
    ('absolute %g fF$^2$' % HEADLINE, lambda v: HEADLINE),
    ('per-night 90th pct', lambda v: float(np.percentile(v, 90))),
    ('10x night median', lambda v: 10.0 * float(np.median(v))),
]


def policy_table(var):
    """Every night under each threshold policy."""
    rows = []
    for name, fn in POLICIES:
        for sess, g in var.groupby('session'):
            v = g.sort_values('epoch')['var_' + CH].to_numpy(float)
            thr = fn(v)
            above = v > thr
            rows.append({'policy': name, 'session': sess, 'subject': sess[:2],
                         'night': int(sess[-1]), 'threshold_fF2': thr,
                         'dur_above_pct': 100.0 * above.mean(),
                         'impulses_per_hour': len(runs_above(above)) / (len(v) * EPOCH_HR),
                         'night_median_fF2': float(np.median(v))})
    return pd.DataFrame(rows)


def per_night(imb, var):
    """The legacy prof_metrics_per_night table, at the 10 fF² headline threshold."""
    m1 = metric1(imb)
    head = metric23(var, HEADLINE).drop(columns='hours')
    mf = metric23(var, HEADLINE, motion_free=True)
    head['dur_above_pct_motionfree'] = mf['dur_above_pct'].to_numpy()
    head['impulses_per_hour_motionfree'] = mf['impulses_per_hour'].to_numpy()
    t = m1.merge(head, on='session')
    t['age'] = t.session.str[:2].map(lambda s_: PARTICIPANTS[s_]['age'])
    t['psqi'] = t.session.str[:2].map(lambda s_: PARTICIPANTS[s_]['psqi'])
    return t


# ── PSG: arousals and SWS ────────────────────────────────────────────────────

def stage_at(t_hr, prof):
    """Scored stage code at each event onset."""
    pt = np.asarray(prof['t_ep_hr'], float)
    pc = np.asarray(prof['codes'], int)
    j = np.searchsorted(pt, np.asarray(t_hr, float)) - 1
    out = np.full(len(j), -1)
    ok = (j >= 0) & (j < len(pc))
    out[ok] = pc[j[ok]]
    return out


def arousal_row(s):
    """EEG-scored arousal counts and indices per hour of sleep, one recording."""
    prof = s.sleep_profile
    codes = np.asarray(prof['codes'], int)
    rec_hr = float(s.time_hr[-1])
    tst_hr = float(np.isin(codes, list(SLEEP_CODES)).sum() * EPOCH_HR)
    rec = {'session': s.label, 'subject': s.label[:2], 'recording_hr': rec_hr,
           'tst_hr': tst_hr, 'sleep_efficiency_pct': 100.0 * tst_hr / rec_hr}
    ar = load_arousals(s)
    if ar is None:
        rec.update(n_arousals=np.nan, arousal_index=np.nan)
    else:
        st = stage_at(ar['start_hr'], prof)
        in_sleep = np.isin(st, list(SLEEP_CODES))
        unknown = sorted({t for t in ar['types'] if t.strip() not in LABEL_GROUP})
        if unknown:
            print(f'  {s.label}: unknown arousal labels {unknown} -> "other"')
        kinds = np.array([LABEL_GROUP.get(t.strip(), 'other') for t in ar['types']])
        rec['n_arousals'] = int(in_sleep.sum())
        rec['n_arousals_all'] = int(len(st))
        rec['arousal_index'] = rec['n_arousals'] / tst_hr
        rec['per_recording_hour'] = rec['n_arousals'] / rec_hr
        rec['median_duration_s'] = (float(np.median(ar['duration_s'][in_sleep]))
                                    if in_sleep.any() else np.nan)
        for k in AROUSAL_TYPES + ['other']:
            n = int(((kinds == k) & in_sleep).sum())
            rec['n_' + k] = n
            rec['idx_' + k] = n / tst_hr
        types = np.array(ar['types'])
        for lab in LABEL_GROUP:
            rec['raw_' + lab.replace(' ', '_')] = int(((types == lab) & in_sleep).sum())
        for code, name in CODE2STAGE.items():
            if code in SLEEP_CODES:
                rec['n_in_' + name] = int((st == code).sum())
    auto = load_autonomic_arousals(s)
    if auto is not None:
        n_auto = int(np.isin(stage_at(auto['start_hr'], prof), list(SLEEP_CODES)).sum())
        rec['n_autonomic'] = n_auto
        rec['autonomic_index'] = n_auto / tst_hr
    return rec


def psg_sws_row(s):
    """PSG slow-wave sleep (N3) from the hypnogram: % of TST and % of recording."""
    codes = np.asarray(s.sleep_profile['codes'], int)
    n3 = int((codes == 1).sum())
    n_sleep = int(np.isin(codes, list(SLEEP_CODES)).sum())
    rec_hr = float(s.time_hr[-1])
    return {'session': s.label, 'n3_epochs': n3, 'sleep_epochs': n_sleep,
            'n3_hr': n3 * EPOCH_HR, 'sws_pct_tst': 100.0 * n3 / n_sleep,
            'sws_pct_recording': 100.0 * n3 * EPOCH_HR / rec_hr}


# ── statistics ───────────────────────────────────────────────────────────────

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    r, p = pearsonr(x[ok], y[ok])
    return {'r': float(r), 'r2': float(r * r), 'n': int(ok.sum()), 'p': float(p)}


def reproducibility(night):
    """Night 1 vs night 2 per participant for the three Fig 10 features."""
    rows = []
    for key, label in FIG10:
        w = night.pivot(index='subject', columns='night', values=key)
        c = corr(w[1], w[2])
        slope, icpt = np.polyfit(w[1], w[2], 1)
        rows.append({'feature': key, 'label': label, **c, 'slope': slope,
                     'intercept': icpt})
    return pd.DataFrame(rows)


def associations(night, subj):
    """Every §3.8 / Discussion association, with its n."""
    pairs = [
        # id, x, y, table, what
        ('imp_total', 'arousal_index', 'impulses_per_hour', night,
         'impulse frequency vs total arousal index'),
        ('imp_resp', 'idx_respiratory', 'impulses_per_hour', night,
         'impulse frequency vs respiratory arousal index'),
        ('imp_spont', 'idx_spontaneous', 'impulses_per_hour', night,
         'impulse frequency vs spontaneous arousal index'),
        ('imp_limb', 'idx_limb', 'impulses_per_hour', night,
         'impulse frequency vs limb arousal index'),
        ('imp_cardiac', 'idx_cardiac', 'impulses_per_hour', night,
         'impulse frequency vs cardiac arousal index'),
        ('imp_autonomic', 'autonomic_index', 'impulses_per_hour', night,
         'impulse frequency vs autonomic arousal index'),
        ('imp_psqi', 'psqi', 'impulses_per_hour', night, 'impulse frequency vs PSQI'),
        ('osc_area', 'area_per_hour_fF', 'osc_pct', night,
         'oscillation duration (EXTERNAL) vs mean SEC area, correctly paired'),
        ('osc_area_mispaired', 'area_per_hour_fF', 'osc_pct_mispaired', night,
         'oscillation duration vs mean SEC area, workbook pairing (night-1-then-2)'),
        ('osc2_psqi', 'psqi', 'osc_pct', subj,
         'two-night mean oscillation duration (EXTERNAL) vs PSQI'),
        ('d10_psqi', 'psqi', 'dur_above_pct', night, '% of night > 10 fF² vs PSQI'),
        ('medvar_psqi', 'psqi', 'night_median_fF2', night, 'median variance vs PSQI'),
        ('area_age', 'age', 'area_per_hour_fF', night, 'mean SEC area vs age'),
        ('area_sws', 'sws_pct_tst', 'area_per_hour_fF', night,
         'mean SEC area vs PSG SWS (N3 % of TST)'),
        ('area_sws_rec', 'sws_pct_recording', 'area_per_hour_fF', night,
         'mean SEC area vs PSG SWS (N3 % of recording)'),
        ('area2_sws2', 'sws_pct_tst', 'area_per_hour_fF', subj,
         'two-night mean SEC area vs two-night mean PSG SWS % of TST'),
        ('osc_sws', 'sws_pct_tst', 'osc_pct', night,
         'oscillation duration (EXTERNAL) vs PSG SWS % of TST'),
        ('sws2_psqi', 'psqi', 'sws_pct_tst', subj, 'two-night mean PSG SWS % vs PSQI'),
    ]
    rows = []
    for pid, x, y, tab, what in pairs:
        rows.append({'id': pid, 'x': x, 'y': y, 'what': what,
                     'unit': 'participant' if tab is subj else 'night',
                     **corr(tab[x], tab[y])})
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────

FIG10 = [
    ('dur_above_pct', f'% of night with variance > {HEADLINE:g} {CAP_UNIT_SQ}'),
    ('pct_above_10x_median', '% of night > 10× nightly median variance'),
    ('night_median_fF2', f'median variance ({CAP_UNIT_SQ})'),
]


def _fit(ax, x, y, ls=':'):
    x, y = np.asarray(x, float), np.asarray(y, float)
    k, c = np.polyfit(x, y, 1)
    pad = 0.06 * (x.max() - x.min())
    xx = np.array([x.min() - pad, x.max() + pad])
    ax.plot(xx, k * xx + c, ls=ls, lw=1.8, color='#2C3E50', zorder=2)


def _points(ax, df, x, y, by_night=True):
    for _, r in df.iterrows():
        ax.scatter(r[x], r[y], s=70, color=SUBJ_COLORS[r.subject],
                   marker=NIGHT_MARKER[int(r.night)] if by_night else 'o',
                   edgecolor='white', lw=0.8, zorder=4)


def _stat(ax, text, loc='upper left'):
    vert, horiz = loc.split()
    x, ha = (0.04, 'left') if horiz == 'left' else (0.96, 'right')
    y, va = (0.96, 'top') if vert == 'upper' else (0.04, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBBBBB', lw=0.6))


def _subject_legend(fig, nights=False, y=1.0):
    h = [plt.Line2D([], [], ls='', marker='o', ms=9, color=c, label=s_)
         for s_, c in SUBJ_COLORS.items()]
    if nights:
        h += [plt.Line2D([], [], ls='', marker=NIGHT_MARKER[n], ms=8, color='#777',
                         label=f'night {n}') for n in (1, 2)]
    fig.legend(handles=h, loc='lower center', ncol=len(h), bbox_to_anchor=(0.5, y),
               frameon=False, fontsize=11)


def fig10(night, rep):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (letter, (key, label)) in zip(axes, zip('abc', FIG10)):
        w = night.pivot(index='subject', columns='night', values=key)
        for subj, r in w.iterrows():
            ax.scatter(r[1], r[2], s=80, color=SUBJ_COLORS[subj], edgecolor='white',
                       lw=0.8, zorder=4)
        _fit(ax, w[1], w[2])
        lo = min(w.min().min(), 0) if key != 'night_median_fF2' else w.min().min() * 0.9
        hi = w.max().max() * 1.08
        ax.plot([lo, hi], [lo, hi], ls='--', lw=0.8, color='#BBBBBB', zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        rr = rep.set_index('feature').loc[key]
        _stat(ax, f'R² = {rr.r2:.3f}\nn = {int(rr.n)} participants', 'lower right')
        ax.set_title(f'({letter})  {label}', loc='left', fontsize=12.5)
        ax.set_xlabel('night 1')
        ax.set_ylabel('night 2')
        ax.grid(alpha=0.2)
    _subject_legend(fig, y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, 'fig10_reproducibility', STAGE)


def fig11(night, subj, assoc):
    a = assoc.set_index('id')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    ax = axes[0]
    _points(ax, night, 'idx_respiratory', 'impulses_per_hour')
    _fit(ax, night.idx_respiratory, night.impulses_per_hour)
    r = a.loc['imp_resp']
    _stat(ax, f'R = {r.r:+.2f}  (R² = {r.r2:.2f})\nn = {int(r.n)} nights', 'upper left')
    ax.set_xlabel('respiratory arousal index (/h of sleep)')
    ax.set_ylabel('impulse frequency (events/h)')
    ax.set_title('(a)  impulse frequency vs respiratory arousals', loc='left',
                 fontsize=12.5)

    ax = axes[1]
    for _, rw in subj.iterrows():
        ax.scatter(rw.psqi, rw.osc_pct, s=80, color=SUBJ_COLORS[rw.subject],
                   edgecolor='white', lw=0.8, zorder=4)
    _fit(ax, subj.psqi, subj.osc_pct)
    r = a.loc['osc2_psqi']
    _stat(ax, f'R = {r.r:+.2f}  (R² = {r.r2:.2f})\nn = {int(r.n)} participants',
          'lower left')
    ax.set_xlabel('PSQI')
    ax.set_ylabel('oscillation duration, mean of 2 nights\n(% of recording; hand-measured)')
    ax.set_title('(b)  low-frequency oscillation duration vs PSQI', loc='left',
                 fontsize=12.5)

    ax = axes[2]
    _points(ax, night, 'psqi', 'dur_above_pct')
    _fit(ax, night.psqi, night.dur_above_pct)
    r = a.loc['d10_psqi']
    _stat(ax, f'R = {r.r:+.2f}  (R² = {r.r2:.2f})\nn = {int(r.n)} nights', 'upper right')
    ax.set_xlabel('PSQI')
    ax.set_ylabel(f'% of night with variance > {HEADLINE:g} {CAP_UNIT_SQ}')
    ax.set_title(f'(c)  time above {HEADLINE:g} {CAP_UNIT_SQ} vs PSQI', loc='left',
                 fontsize=12.5)
    for ax in axes[1:]:
        ax.set_xlim(3, 10)
    for ax in axes:
        ax.grid(alpha=0.2)
    _subject_legend(fig, nights=True, y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, 'fig11_associations', STAGE)


# ── run ──────────────────────────────────────────────────────────────────────

def run():
    grid = pd.read_parquet(CACHE_DIR / 's01_recordings' / 'epochs.parquet')
    imb = pd.read_parquet(CACHE_DIR / 's02_overnight' / 'imbalance_epochs.parquet')
    out_t = TAB_DIR / STAGE
    out_t.mkdir(parents=True, exist_ok=True)

    leg, shared, arous, sws = [], [], [], []
    for s in iter_sessions():
        lp = lowpassed(s)
        leg.append(variance_legacy_grid(s, lp))
        ep = grid[grid.session == s.label].sort_values('epoch').reset_index(drop=True)
        shared.append(variance_shared_grid(s, lp, ep))
        arous.append(arousal_row(s))
        sws.append(psg_sws_row(s))
        print(f'  {s.label}: legacy grid {len(leg[-1])}, shared grid {len(ep)} epochs')

    var_leg = finish_variance(pd.concat(leg, ignore_index=True), own_motion=True)
    var = finish_variance(pd.concat(shared, ignore_index=True), own_motion=False)
    var_leg.to_parquet(out_t / 'high_variance_epochs_legacy_grid.parquet', index=False)
    var.to_parquet(out_t / 'high_variance_epochs.parquet', index=False)

    tabs = {}
    for tag, v in [('_legacy_grid', var_leg), ('', var)]:
        t = per_night(imb, v)
        pt = policy_table(v)
        t.to_csv(out_t / f'prof_metrics_per_night{tag}.csv', index=False)
        pt.to_csv(out_t / f'prof_metrics_threshold_policy{tag}.csv', index=False)
        tabs[tag] = (t, pt)
    ar = pd.DataFrame(arous)
    ar['age'] = ar.subject.map(lambda s_: PARTICIPANTS[s_]['age'])
    ar['psqi'] = ar.subject.map(lambda s_: PARTICIPANTS[s_]['psqi'])
    ar.to_csv(out_t / 'arousal_counts.csv', index=False)
    sw = pd.DataFrame(sws)
    sw['osc_pct_EXTERNAL'] = sw.session.map(OSC_DURATION_PCT)
    sw.to_csv(out_t / 'psg_sws.csv', index=False)

    results = {}
    for tag, (t, pt) in tabs.items():
        night = night_table(t, pt, ar, sw)
        subj = night.groupby('subject', as_index=False).mean(numeric_only=True)
        rep = reproducibility(night)
        assoc = associations(night, subj)
        results[tag] = (night, subj, rep, assoc)
    night, subj, rep, assoc = results['']
    night.to_csv(out_t / 'per_night_features.csv', index=False)
    rep.to_csv(out_t / 'reproducibility.csv', index=False)
    assoc.to_csv(out_t / 'associations.csv', index=False)
    results['_legacy_grid'][2].to_csv(out_t / 'reproducibility_legacy_grid.csv', index=False)
    results['_legacy_grid'][3].to_csv(out_t / 'associations_legacy_grid.csv', index=False)

    fig10(night, rep)
    fig11(night, subj, assoc)
    register_numbers(results, var, var_leg)


def night_table(t, pt, ar, sw):
    """One row per night: every feature Fig 10 / Fig 11 / §3.8 uses."""
    pol = pt.pivot(index='session', columns='policy', values='dur_above_pct')
    med = pt.groupby('session').night_median_fF2.first()
    n = t[['session', 'area_per_hour_fF', 'dur_above_pct', 'impulses_per_hour',
           'age', 'psqi']].copy()
    n['subject'] = n.session.str[:2]
    n['night'] = n.session.str[-1].astype(int)
    n['pct_above_10x_median'] = n.session.map(pol['10x night median'])
    n['night_median_fF2'] = n.session.map(med)
    n = n.merge(ar[['session', 'arousal_index', 'idx_spontaneous', 'idx_respiratory',
                    'idx_limb', 'idx_cardiac', 'autonomic_index']], on='session')
    n = n.merge(sw[['session', 'sws_pct_tst', 'sws_pct_recording']], on='session')
    n['osc_pct'] = n.session.map(OSC_DURATION_PCT)
    # the workbook's mis-paired column, aligned to sessions in order
    n = n.sort_values('session').reset_index(drop=True)
    n['osc_pct_mispaired'] = [OSC_DURATION_PCT[k] for k in OSC_MISPAIRED_ORDER]
    return n


def register_numbers(results, var, var_leg):
    nb = Numbers(STAGE)
    night, subj, rep, assoc = results['']
    rep_l, assoc_l = results['_legacy_grid'][2], results['_legacy_grid'][3]
    a, al = assoc.set_index('id'), assoc_l.set_index('id')
    rp, rpl = rep.set_index('feature'), rep_l.set_index('feature')
    grid_note = (f'shared s01 grid ({len(var):,} scored epochs); legacy rate grid '
                 f'({len(var_leg):,} epochs) gives ')

    # §3.7 / Fig 10
    for key, pv, fig_pv, what in [
            ('dur_above_pct', '0.63', '0.6306',
             '% of night with CLE-CRE variance > 10 fF² (text: "absolute signal amplitude")'),
            ('pct_above_10x_median', '0.71', '0.7097', '% of night > 10x nightly median'),
            ('night_median_fF2', '0.68', '0.6805', 'median variance')]:
        r = rp.loc[key]
        p_note = f'p = {r.p:.3f} (n = 6)'
        nb.add(f'rep_r2_{key}', '§3.7 ¶230', f'R² night 1 vs night 2, {what}', r.r2, pv,
               source='reproducibility',
               note=f'CORRECTION: paper value from graph-read Excel values; {grid_note}'
                    f'{rpl.loc[key].r2:.3f}. n = 6 participants; {p_note}')
        nb.add(f'fig10_r2_{key}', 'Fig 10', f'R² annotation, {what}', r.r2, fig_pv,
               source='reproducibility',
               note='workbook "reproducibility" sheet, values "Estimated from graph" '
                    '(threshold_policy.png); see checks/s09_reproducibility_vs_legacy.out')
        nb.add(f'rep_n_{key}', 'Fig 10', f'participants per panel, {what}', int(r.n),
               source='reproducibility')
    nb.add('rep_significant', '§3.7 ¶230', '"Significant positive relationships" (n = 6)',
           '; '.join(f'p = {rp.loc[k].p:.3f}' for k, _ in FIG10),
           'Significant positive relationships', status='DIFF', source='reproducibility',
           note='n = 6 participants; '
                + ', '.join(f'{k} p={rp.loc[k].p:.3f}' for k, _ in FIG10)
                + '. Describe as R² with n rather than "significant"')
    nb.add('fig10a_label', 'Fig 10 caption, ¶230', 'feature (a)',
           f'% of night with CLE-CRE variance > {HEADLINE:g} fF²', 'absolute amplitude',
           status='DIFF', source='prof_metrics.metric23',
           note='CORRECTION: an absolute 10 fF² threshold on 30-s variance, not an amplitude')
    nb.add('fig_ref_230', '§3.7 ¶230', 'figure cited for reproducibility', 'Fig. 10',
           'Fig. 11', status='DIFF', note='cross-reference error')

    # §3.8
    nb.add('fig_ref_233', '§3.8 ¶233', 'figure cited for the associations', 'Fig. 11',
           'Fig. 12', status='DIFF', note='cross-reference error')
    nb.add('feat_sws_233', '§3.8 ¶233', '"SWS duration (% of the night)"',
           'hand-measured low-frequency oscillation duration, % of recording',
           'SWS duration (% of the night)', status='EXTERNAL',
           source='workbook "SWS % calculation" col Z',
           note='not PSG SWS: sum of ruler-measured oscillation episodes / Table-1 duration. '
                'PSG N3 % of TST per night: '
                + ', '.join(f'{r.session} {r.sws_pct_tst:.1f}'
                            for _, r in night.iterrows()))
    nb.add('feat_amp_233', '§3.8 ¶233', '"absolute SEC amplitude above 10 fF²"',
           f'% of night with CLE-CRE 30-s variance > {HEADLINE:g} fF²',
           'absolute SEC amplitude above 10 fF²', status='DIFF',
           note='CORRECTION: variance threshold (fF²), not amplitude')
    nb.add('feat_area_233', '§3.8 ¶233', 'mean SEC area unit',
           'fF (mean |CLE-CRE - session mean|)', 'fF·h/h', status='MATCH',
           note='fF·h/h = fF; area_per_hour_fF of prof_metrics.metric1')

    def addr(rid, aid, what, pv, sq=False, status=None, note=''):
        r = a.loc[aid]
        val = r.r2 if sq else r.r
        lv = al.loc[aid].r2 if sq else al.loc[aid].r
        extra = f'n = {int(r.n)} {r.unit}s; p = {r.p:.3f}'
        if abs(lv - val) > 5e-4:
            extra += f'; {grid_note}{lv:+.3f}'
        nb.add(rid, '§3.8 ¶238', what, float(val), pv, source='associations',
               status=status, note=(note + '. ' if note else '') + extra)

    addr('r2_imp_total', 'imp_total', 'R², impulse frequency vs total arousal index', '0.10',
         sq=True, note=f'CORRECTION: untraced in the workbook; R = {a.loc["imp_total"].r:+.3f}. '
                       f'Limb index gives R² = {a.loc["imp_limb"].r2:.3f}, autonomic '
                       f'{a.loc["imp_autonomic"].r2:.3f}, cardiac {a.loc["imp_cardiac"].r2:.3f}')
    addr('r_imp_resp', 'imp_resp', 'R, impulse frequency vs respiratory arousal index', '0.50')
    addr('r_imp_spont', 'imp_spont', 'R, impulse frequency vs spontaneous arousal index',
         '0.48', note='CORRECTION: negative; text calls both "positive associations"')
    addr('r_imp_psqi', 'imp_psqi', 'R, impulse frequency vs PSQI', '0.56',
         note='CORRECTION: sign (text says "decreased" but prints 0.56) and value')
    addr('r_osc_area', 'osc_area', 'R, oscillation duration vs mean SEC area', '0.68',
         note='CORRECTION: 0.68 is the workbook\'s mis-paired column (night-1-then-night-2 '
              f'durations against session-ordered area): reproduced here as '
              f'{a.loc["osc_area_mispaired"].r:+.3f}. Correctly paired as computed. '
              'Durations are EXTERNAL (hand-measured)')
    addr('r_osc2_psqi', 'osc2_psqi', 'R, two-night mean oscillation duration vs PSQI',
         '-0.68', status='EXTERNAL',
         note='input durations are hand-measured in the workbook (no code); computed from '
              'the hard-coded values. PSG N3 % instead gives '
              f'R = {a.loc["sws2_psqi"].r:+.3f} (n = 6)')
    addr('r_d10_psqi', 'd10_psqi', 'R, % of night > 10 fF² vs PSQI', '-0.57')
    addr('r_medvar_psqi', 'medvar_psqi', 'R, median variance vs PSQI', '0.48',
         note='CORRECTION: sign; -0.48 is what the graph-read workbook values give, '
              'the exact medians give the computed value')
    addr('r_area_age', 'area_age', 'R, mean SEC area vs age', '0.63',
         note='CORRECTION: sign (text says "decreased")')
    nb.add('sxx_238', '§3.8 ¶238', 'supporting figure for the arousal subtypes', 'none',
           'Fig. Sxx', status='DIFF', note='placeholder never filled')
    nb.add('fig11a_n', 'Fig 11', 'panel (a) n', f'{int(a.loc["imp_resp"].n)} nights',
           source='associations', note='two nights per participant: not independent')
    nb.add('fig11b_n', 'Fig 11', 'panel (b) n', f'{int(a.loc["osc2_psqi"].n)} participants',
           source='associations')
    nb.add('fig11c_n', 'Fig 11', 'panel (c) n', f'{int(a.loc["d10_psqi"].n)} nights',
           source='associations')
    nb.add('fig11b_label', 'Fig 11 caption', 'panel (b) quantity',
           'hand-measured low-frequency oscillation duration (% of recording)',
           'mean SWS duration across two nights', status='EXTERNAL',
           note='not PSG SWS; see feat_sws_233')
    nb.add('fig11c_label', 'Fig 11 caption', 'panel (c) quantity',
           f'% of night with CLE-CRE variance > {HEADLINE:g} fF²',
           'SEC absolute amplitude above 10 fF²', status='DIFF',
           note='CORRECTION: variance threshold, not amplitude')

    # PSG SWS, Discussion
    s_ = a.loc['area_sws']
    nb.add('r_area_psg_sws', 'Discussion ¶248', 'mean SEC area vs PSG-measured SWS duration '
           '(N3 % of TST)', f'R = {s_.r:+.2f} (n = {int(s_.n)} nights)',
           'moderate association', status='DIFF', source='associations',
           note='CORRECTION: the workbook association (0.68) is with the hand-measured '
                'oscillation duration, mis-paired; correctly paired '
                f'{a.loc["osc_area"].r:+.2f}. Against PSG N3: % of recording '
                f'R = {a.loc["area_sws_rec"].r:+.2f}; two-night means '
                f'R = {a.loc["area2_sws2"].r:+.2f} (n = 6). Oscillation duration vs PSG '
                f'N3 % R = {a.loc["osc_sws"].r:+.2f} (n = 12)')
    nb.add('r_area_psg_sws_259', 'Conclusion ¶259', 'association between mean SEC area and '
           'PSG-defined SWS duration', f'R = {s_.r:+.2f} (n = {int(s_.n)} nights)',
           'the association', status='DIFF', source='associations',
           note='same claim as ¶248')
    nb.add('psg_sws_range', 'new', 'PSG N3 % of TST, range over 12 nights',
           f'{night.sws_pct_tst.min():.1f} to {night.sws_pct_tst.max():.1f}',
           unit='%', source='psg_sws')
    nb.save()


if __name__ == '__main__':
    run()
