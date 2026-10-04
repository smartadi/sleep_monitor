"""Persistent spectral ridges: respiratory/cardiac ridge features by stage, Viterbi traces, the low band, Fig. 5.

Manuscript items
    Methods ¶153      Viterbi respiratory / cardiac trajectories and tracker confidence
    Methods ¶158-159  persistent ridges in the respiratory (0.1-0.5 Hz) and cardiac
                      (0.5-3.0 Hz) bands, per-epoch features, N3-vs-rest contrasts,
                      motion exclusion, count matching; the low band
    Results ¶190      dominant-ridge channel; ridge frequencies
    Results ¶194      recurrent low-frequency ridges (replaced, see corrections)
    Results ¶195, Discussion ¶248
                      "reduced respiratory ridge power during N3 ... after controlling
                      for motion and unequal stage duration"
    Fig. 5            S4N2 CRE: stage ladder; 0-3 Hz background-corrected spectrogram
                      with Viterbi traces; smoothed low band with its gated ridge
    Supp (low band)   all twelve nights, low band with gated ridge
    Tables            ridge_epoch_counts, band_ridge_stage_summary (CRE),
                      ridge_stage_all_channels, ridge_robustness,
                      ridge_power_count_matched, ridge_dominance, viterbi_nights,
                      lowband_nights, lowband_episodes, lowband_stage_occupancy

Ported from
    analysis/slow_wave/band_ridge_analysis.py   BANDS, ridge_epoch_features,
                                                align_sleep_stages, stage_summary
    analysis/slow_wave/recompute_stage_summary.py  (the tie-aware stage_summary)
    analysis/slow_wave/ridge_stage_all_channels.py  directions
    analysis/rates/reviewer_pass_analyses.py    m6_m7_ridges
    analysis/slow_wave/ridge_overlay_tune.py    TRACK, track_single_ridge,
                                                _enhance_spec, draw_stage_ladder, _sig
    analysis/slow_wave/ridge_lowband_smooth.py  low band (-> _s05_lowband.py), figure, sheet

Changes from the legacy code
    1. Fig. 5 is drawn for S4N2 CRE from the current code. The manuscript's file is a
       stale render of ridge_overlay_tune.py (old low band, no gated ridge).
    2. The low band is ridge_lowband_smooth.py, not ridge_overlay_tune.py's 'slow'
       band. The old "67 ridges, median 6, 2-8 per night, none in S1N1, 0.05-0.08 Hz"
       came from a low band built on the motion-cancelled signal, whose 0.05 Hz
       band-pass corner is a constant hump at 0.05-0.08 Hz after 1/f detrending;
       those values are registered as DIFF against the new result (ridge % of night,
       episodes, frequencies).
    3. Count matching existed only for the ridge COUNT (m6_m7_ridges). The power claim
       of ¶195/¶248 had no duration control. Count matching is now also run for total
       respiratory and cardiac ridge power: per participant, motion-free epochs, 200
       draws of as many non-N3 epochs as there are N3 epochs, N3 statistic minus the
       draw's statistic (mean and median), averaged over draws; the participant-level
       direction count per channel is the reported result.
    4. "CRE contained the dominant ridge in 9 of 12" came from an older detector
       (run_ridge_overlay.py harmonic score). The dominant channel is recomputed on
       the current detector with a stated rule (see `dominance`).
    5. The Viterbi emission is log(PSD / column-median PSD) inside the search band of
       a plain spectrogram, and confidence counts windows where the tracked bin
       exceeds 2x the column median. Methods ¶153 describes the display spectrogram
       (frequency-median-filtered background subtracted) as if it were the tracker's
       input. Code kept; the difference is registered as a note.
    Also kept but flagged: band_ridge_stage_summary compares N3 with N1+N2+REM
    (Wake excluded), whereas ridge_stage_all_channels and the robustness tables
    compare N3 with Wake+N1+N2+REM, as Methods ¶159 says ("all other stages").
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import spectrogram
from scipy.stats import kruskal, mannwhitneyu

from seclib import CACHE_DIR, FS, STAGE_LABELS, STAGE_ORDER, TAB_DIR, iter_sessions, get_session
from seclib.config import STAGE_ROW, STAGE_ROW_TICKS, STAGE_ROW_LABELS
from seclib.figures import save
from seclib.harmonics import detect_persistent_ridges
from seclib.numbers import Numbers
from seclib.preprocessing import remove_acc_artifact
from stages import _s05_lowband as LB

STAGE = 's05_ridges'
REQUIRES = []

CHANNELS = ['CH', 'CLE', 'CRE']
FIG5_SESSION, FIG5_CHANNEL = 'S4N2', 'CRE'

# ── Persistent-ridge detector, per band (band_ridge_analysis.py) ─────────────
# Respiratory needs fine frequency resolution (single 30 s periodogram,
# df ~ 0.033 Hz) since the band is only 0.4 Hz wide; cardiac uses 8 s Welch
# segments.
BANDS = {
    'resp': dict(min_freq=0.1, max_freq=0.5, welch_seg_sec=30.0,
                 max_freq_jump=0.05, peak_prominence_frac=0.3),
    'card': dict(min_freq=0.5, max_freq=3.0, welch_seg_sec=8.0,
                 max_freq_jump=0.10, peak_prominence_frac=0.5),
}
WIN_SEC = 30.0
STEP_SEC = 30.0
SMOOTH_WINDOWS = 7
MIN_PERSIST_SEC = 300.0
MAX_GAP_WINDOWS = 5
POOL_CH = {'resp': 'CRE', 'card': 'CRE'}   # channel of the pooled stage summary

FEATURES = ['ridge_present', 'n_ridges', 'min_ridge_freq',
            'total_ridge_power', 'freq_spread', 'n_groups_active']

# ── Viterbi trackers for the two always-present rhythms (ridge_overlay_tune.py)
# `band` restricts the search, `penalty` (log-amplitude per Hz) buys smoothness,
# `smooth_win` is a final median kernel (windows). Cardiac is capped at 1.45 Hz
# so the path cannot jump to brighter harmonic peaks on weak-cardiac nights.
TRACK = {
    'resp': dict(band=(0.25, 0.55), win_sec=30.0, penalty=20.0, smooth_win=13),
    'card': dict(band=(0.85, 1.45), win_sec=15.0, penalty=24.0, smooth_win=23),
}
CONF_RATIO = 2.0          # tracked bin > 2x column median counts as 'confident'
TRACE_COLOR = {'resp': '#00E5FF', 'card': '#FF3B30'}
RIDGE_COLOR = '#FF2D55'

N_DRAWS = 200             # count-matching draws per participant


# ═════════════════════════════════════════════════════════════════════════════
# 1. Signals
# ═════════════════════════════════════════════════════════════════════════════

def rate_band_signal(session, ch):
    """The rate-band channel: OLS accelerometer regression, 0.05-4 Hz (as in the rate pipeline)."""
    acc = session.cap['acc_mag'].astype(np.float64)
    return remove_acc_artifact(session.cap[ch].astype(np.float64), acc, 0.05, 4.0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Persistent ridges per 30-s epoch (Methods ¶158)
# ═════════════════════════════════════════════════════════════════════════════

def ridge_epoch_features(rr: dict, ch: str, band: str) -> pd.DataFrame:
    """Per-epoch summary of band-limited ridge structure."""
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    groups = rr['harmonic_groups']
    n_win = len(t_hr)

    n_ridges = np.zeros(n_win, dtype=int)
    n_groups_active = np.zeros(n_win, dtype=int)
    min_ridge_freq = np.full(n_win, np.nan)
    mean_ridge_freq = np.full(n_win, np.nan)
    freq_spread = np.full(n_win, np.nan)
    total_ridge_power = np.full(n_win, np.nan)
    max_prominence = np.full(n_win, np.nan)

    for i in range(n_win):
        freqs_i, amps_i, proms_i = [], [], []
        for ridge in ridges:
            f = ridge['freq_trace'][i]
            if np.isfinite(f):
                freqs_i.append(f)
                amps_i.append(ridge['amp_trace'][i])
                pr = ridge.get('prominence_trace')
                if pr is not None and np.isfinite(pr[i]):
                    proms_i.append(pr[i])
        n_ridges[i] = len(freqs_i)
        if freqs_i:
            fa = np.array(freqs_i)
            min_ridge_freq[i] = fa.min()
            mean_ridge_freq[i] = fa.mean()
            freq_spread[i] = fa.std() if len(fa) > 1 else 0.0
            total_ridge_power[i] = float(np.sum(amps_i))
        if proms_i:
            max_prominence[i] = float(np.max(proms_i))

        grp_count = 0
        for grp in groups:
            active = sum(
                1 for mi in grp['harmonic_idxs']
                if mi < len(ridges) and np.isfinite(ridges[mi]['freq_trace'][i])
            )
            if active >= 2:
                grp_count += 1
        n_groups_active[i] = grp_count

    return pd.DataFrame({
        't_hr': t_hr,
        'band': band,
        'channel': ch,
        'motion_masked': rr['motion_mask'],
        'n_ridges': n_ridges,
        'n_groups_active': n_groups_active,
        'min_ridge_freq': min_ridge_freq,
        'mean_ridge_freq': mean_ridge_freq,
        'freq_spread': freq_spread,
        'total_ridge_power': total_ridge_power,
        'max_prominence': max_prominence,
        'ridge_present': (n_ridges > 0).astype(int),
    })


def align_sleep_stages(df: pd.DataFrame, sp: dict) -> pd.DataFrame:
    if sp is None:
        df['stage_code'] = -1
        df['stage_label'] = '?'
        return df
    t_ep = sp['t_ep_hr']
    codes = sp['codes']
    scodes = []
    for t in df['t_hr']:
        idx = np.searchsorted(t_ep, t, side='right') - 1
        idx = np.clip(idx, 0, len(codes) - 1)
        scodes.append(int(codes[idx]))
    df['stage_code'] = scodes
    df['stage_label'] = [STAGE_LABELS.get(c, '?') for c in scodes]
    return df


def ridge_epochs():
    """Run the persistent-ridge detector on 12 nights x 3 channels x 2 bands.

    Returns (per-epoch features, per-ridge list) and caches both.
    """
    all_dfs, ridge_rows = [], []
    for s in iter_sessions():
        acc_mag = s.cap['acc_mag']
        for ch in CHANNELS:
            sig = remove_acc_artifact(s.cap[ch], acc_mag, 0.05, 4.0)
            for band, bp in BANDS.items():
                rr = detect_persistent_ridges(
                    sig, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC,
                    min_freq=bp['min_freq'], max_freq=bp['max_freq'],
                    smooth_windows=SMOOTH_WINDOWS,
                    min_persistence_sec=MIN_PERSIST_SEC,
                    max_freq_jump=bp['max_freq_jump'],
                    peak_prominence_frac=bp['peak_prominence_frac'],
                    welch_seg_sec=bp['welch_seg_sec'],
                    max_gap_windows=MAX_GAP_WINDOWS,
                    acc_mag=acc_mag,
                )
                df = ridge_epoch_features(rr, ch, band)
                df = align_sleep_stages(df, s.sleep_profile)
                df['session'] = s.label
                df['subject'] = s.subject
                all_dfs.append(df)
                for r in rr['ridges']:
                    ridge_rows.append(dict(
                        session=s.label, subject=s.subject, channel=ch, band=band,
                        median_freq=r['median_freq'], duration_min=r['duration_sec'] / 60,
                        median_prominence=r.get('median_prominence', np.nan)))
        print(f'  ridges {s.label}')
    epochs = pd.concat(all_dfs, ignore_index=True)
    ridges = pd.DataFrame(ridge_rows)
    out = CACHE_DIR / STAGE
    out.mkdir(parents=True, exist_ok=True)
    epochs.to_parquet(out / 'band_ridge_epochs.parquet', index=False)
    ridges.to_csv(out / 'band_ridges.csv', index=False)
    return epochs, ridges


# ═════════════════════════════════════════════════════════════════════════════
# 3. Stage contrasts (Methods ¶159). Participant direction counts are the unit;
#    pooled Kruskal-Wallis / Mann-Whitney are descriptive only (epochs are not
#    independent).
# ═════════════════════════════════════════════════════════════════════════════

def stage_summary(all_epochs: pd.DataFrame) -> pd.DataFrame:
    """CRE, motion-free: pooled KW/MWU (descriptive) and per-participant N3 vs N1+N2+REM.

    Per-subject direction: these features are small integer counts, so medians
    tie often; a tie is its own category, never folded into a direction. The
    mean is recorded beside the median.
    """
    rows = []
    for band in BANDS:
        pool = all_epochs[
            (all_epochs['band'] == band)
            & (all_epochs['channel'] == POOL_CH[band])
            & (~all_epochs['motion_masked'])
            & (all_epochs['stage_code'] >= 0)
        ].copy()
        pool_ex_wake = pool[pool['stage_code'] != 4].copy()
        pool_ex_wake['is_N3'] = pool_ex_wake['stage_code'] == 1
        subjects = sorted(pool['subject'].unique())
        for feat in FEATURES:
            groups = []
            for sc in STAGE_ORDER:
                vals = pool.loc[pool['stage_code'] == sc, feat].dropna()
                if len(vals) > 0:
                    groups.append(vals.values)
            kw_p = kruskal(*groups)[1] if len(groups) >= 2 else np.nan

            n3 = pool_ex_wake.loc[pool_ex_wake['is_N3'], feat].dropna()
            oth = pool_ex_wake.loc[~pool_ex_wake['is_N3'], feat].dropna()
            if len(n3) > 5 and len(oth) > 5:
                mwu_p = mannwhitneyu(n3, oth, alternative='two-sided')[1]
            else:
                mwu_p = np.nan
            dirs, mean_dirs = [], []
            for subj in subjects:
                sv = pool_ex_wake[pool_ex_wake['subject'] == subj]
                a = sv.loc[sv['is_N3'], feat].dropna()
                b = sv.loc[~sv['is_N3'], feat].dropna()
                if len(a) > 3 and len(b) > 3:
                    am, bm = a.median(), b.median()
                    dirs.append('N3=' if am == bm else ('N3>' if am > bm else 'N3<'))
                    mean_dirs.append('N3>' if a.mean() > b.mean() else 'N3<')
                else:
                    dirs.append('?')
                    mean_dirs.append('?')
            rows.append(dict(
                band=band, channel=POOL_CH[band], feature=feat,
                n3_median=float(n3.median()) if len(n3) else np.nan,
                other_median=float(oth.median()) if len(oth) else np.nan,
                kw_p=kw_p, mwu_p=mwu_p,
                n_subj_N3_up=dirs.count('N3>'), n_subj_N3_dn=dirs.count('N3<'),
                n_subj_tied=dirs.count('N3='),
                n_subj_N3_dn_by_mean=mean_dirs.count('N3<'),
                directions=','.join(dirs),
                directions_by_mean=','.join(mean_dirs),
            ))
    return pd.DataFrame(rows)


def directions(df, feature, agg):
    """Per-subject N3 vs Wake+N1+N2+REM direction, using `agg` within each side."""
    out = []
    for subj, g in df.groupby('subject'):
        n3 = g.loc[g['stage_label'] == 'N3', feature].dropna()
        rest = g.loc[g['stage_label'].isin(['Wake', 'N1', 'N2', 'REM']), feature].dropna()
        if len(n3) < 10 or len(rest) < 10:
            out.append((subj, np.nan, np.nan, '?'))
            continue
        a, b = agg(n3), agg(rest)
        if not np.isfinite(a) or not np.isfinite(b):
            sym = '?'
        elif np.isclose(a, b):
            sym = '='
        else:
            sym = '<' if a < b else '>'
        out.append((subj, float(a), float(b), sym))
    return out


def all_channel_directions(epochs):
    """Every band x channel x feature, motion-free, by median and by mean."""
    clean = epochs[epochs['motion_masked'] == 0]
    rows = []
    for (band, ch), g in clean.groupby(['band', 'channel']):
        for feat in FEATURES:
            for label, agg in (('median', np.median), ('mean', np.mean)):
                d = directions(g, feat, agg)
                syms = [s for _, _, _, s in d]
                rows.append(dict(
                    band=band, channel=ch, feature=feat, statistic=label,
                    n3_lower=syms.count('<'), n3_higher=syms.count('>'),
                    tied=syms.count('='), undetermined=syms.count('?'),
                    directions=','.join(f'{sub}:{s}' for sub, _, _, s in d),
                ))
    return pd.DataFrame(rows).sort_values(['band', 'feature', 'statistic', 'channel'])


def ridge_robustness(epochs):
    """Mean active ridges per epoch, N3 minus rest, per participant: all epochs,
    motion-free, and motion-free count-matched (m6_m7_ridges)."""
    d = epochs[epochs.stage_label.notna()]
    rows = []
    for (band, ch), g in d.groupby(['band', 'channel']):
        for label, sub in (('all epochs', g),
                           ('motion-free', g[~g.motion_masked.astype(bool)])):
            per_subj = []
            for subj, s in sub.groupby('subject'):
                n3 = s[s.stage_label == 'N3'].n_ridges
                oth = s[s.stage_label.isin(['N1', 'N2', 'REM', 'Wake'])].n_ridges
                if len(n3) < 20 or len(oth) < 20:
                    continue
                per_subj.append(n3.mean() - oth.mean())
            if not per_subj:
                continue
            rows.append({'band': band, 'channel': ch, 'subset': label,
                         'n_subj': len(per_subj),
                         'n3_lower': int(sum(1 for v in per_subj if v < 0)),
                         'median_delta': float(np.median(per_subj))})

            # count-matched: draw as many non-N3 epochs as N3, per subject
            if label == 'motion-free':
                rng = np.random.default_rng(0)
                matched = []
                for subj, s in sub.groupby('subject'):
                    n3 = s[s.stage_label == 'N3'].n_ridges.values
                    oth = s[s.stage_label.isin(['N1', 'N2', 'REM', 'Wake'])].n_ridges.values
                    if len(n3) < 20 or len(oth) < len(n3):
                        continue
                    draws = [n3.mean() - rng.choice(oth, len(n3), replace=False).mean()
                             for _ in range(N_DRAWS)]
                    matched.append(float(np.mean(draws)))
                if matched:
                    rows.append({'band': band, 'channel': ch, 'subset': 'count-matched',
                                 'n_subj': len(matched),
                                 'n3_lower': int(sum(1 for v in matched if v < 0)),
                                 'median_delta': float(np.median(matched))})
    return pd.DataFrame(rows).sort_values(['band', 'channel', 'subset'])


def power_count_matched(epochs, feature='total_ridge_power'):
    # CORRECTION: ¶195/¶248 claim the N3 ridge-POWER reduction survives "controlling
    # for motion and unequal stage duration", but count matching was only ever run
    # on n_ridges. Here it is run on ridge power: per participant, motion-free
    # epochs with a ridge (power is undefined without one), N3 vs Wake+N1+N2+REM,
    # 200 draws of len(N3) non-N3 epochs without replacement; the participant's
    # effect is the draw-averaged N3-minus-matched difference of means and of
    # medians. Unmatched (all non-N3 epochs) contrasts are kept beside it.
    clean = epochs[~epochs.motion_masked.astype(bool)]
    rows = []
    for (band, ch), g in clean.groupby(['band', 'channel']):
        rng = np.random.default_rng(0)
        for subj, s in g.groupby('subject'):
            n3 = s.loc[s.stage_label == 'N3', feature].dropna().values
            oth = s.loc[s.stage_label.isin(['N1', 'N2', 'REM', 'Wake']), feature].dropna().values
            row = dict(band=band, channel=ch, subject=subj, feature=feature,
                       n_n3=len(n3), n_other=len(oth))
            if len(n3) < 10 or len(oth) < len(n3):
                rows.append(row)
                continue
            dm, dmed = [], []
            for _ in range(N_DRAWS):
                pick = rng.choice(oth, len(n3), replace=False)
                dm.append(n3.mean() - pick.mean())
                dmed.append(np.median(n3) - np.median(pick))
            row.update(delta_mean_unmatched=n3.mean() - oth.mean(),
                       delta_median_unmatched=np.median(n3) - np.median(oth),
                       delta_mean_matched=float(np.mean(dm)),
                       delta_median_matched=float(np.mean(dmed)),
                       frac_draws_n3_lower_mean=float(np.mean(np.array(dm) < 0)))
            rows.append(row)
    per = pd.DataFrame(rows)
    summ = []
    for (band, ch), g in per.groupby(['band', 'channel']):
        ok = g.dropna(subset=['delta_mean_matched'])
        summ.append(dict(
            band=band, channel=ch, feature=feature, n_subj=len(ok),
            n3_lower_mean_unmatched=int((ok.delta_mean_unmatched < 0).sum()),
            n3_lower_median_unmatched=int((ok.delta_median_unmatched < 0).sum()),
            n3_lower_mean_matched=int((ok.delta_mean_matched < 0).sum()),
            n3_lower_median_matched=int((ok.delta_median_matched < 0).sum())))
    return per, pd.DataFrame(summ)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Which channel carries the dominant ridge (¶158, ¶190)
# ═════════════════════════════════════════════════════════════════════════════

def dominance(epochs):
    # CORRECTION: "CRE contained the dominant ridge in 9 of 12" came from an older
    # detector (run_ridge_overlay.py, harmonic score). Rule on the current
    # detector: in each night and band, the dominant channel is the one whose
    # persistent ridges stand highest above their own local spectral floor --
    # the median, over motion-free epochs that hold a ridge, of the epoch's
    # largest ridge prominence (peak / local floor; scale-free, so CH's larger
    # absolute capacitance does not decide it). Ridge coverage (fraction of
    # motion-free epochs with a ridge) is tabulated beside it.
    clean = epochs[~epochs.motion_masked.astype(bool)]
    g = (clean.groupby(['band', 'session', 'channel'])
         .agg(median_prominence=('max_prominence', 'median'),
              coverage=('ridge_present', 'mean'))
         .reset_index())
    rows = []
    for (band, sess), d in g.groupby(['band', 'session']):
        row = dict(band=band, session=sess)
        for _, r in d.iterrows():
            row[f'prom_{r.channel}'] = r.median_prominence
            row[f'cov_{r.channel}'] = r.coverage
        row['dominant_by_prominence'] = d.loc[d.median_prominence.idxmax(), 'channel']
        row['dominant_by_coverage'] = d.loc[d.coverage.idxmax(), 'channel']
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Viterbi respiratory / cardiac trajectories (Methods ¶153, Fig. 5)
# ═════════════════════════════════════════════════════════════════════════════

def track_single_ridge(sig, fs, band, win_sec, penalty=6.0, smooth_win=7):
    """One smooth frequency trace for a single-rhythm band, by Viterbi.

    Finds the path that maximises total emission minus `penalty` x |frequency
    change| (per Hz). Emission is each column's log relative amplitude
    (PSD / column median) inside the search band.
    """
    # CORRECTION (note only): Methods ¶153 says the path runs through a
    # background-corrected spectrogram (frequency-median-filtered spectrum
    # subtracted per column). That is the display (enhance_spec); the tracker's
    # emission is log(PSD / in-band column median) of the plain spectrogram, and
    # confidence is the fraction of windows where the tracked bin exceeds
    # CONF_RATIO x the column median.
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=int(win_sec * fs),
                            noverlap=int(win_sec * fs // 2))
    bm = (f >= band[0]) & (f <= band[1])
    fb, Sb = f[bm], Sxx[bm]
    nfreq, nt = Sb.shape

    col_med = np.nanmedian(Sb, axis=0, keepdims=True) + 1e-30
    E = np.log(Sb / col_med + 1e-6)              # emission: log relative amplitude
    D = np.abs(fb[:, None] - fb[None, :])        # (nfreq, nfreq) freq-change cost

    score = np.full((nfreq, nt), -np.inf)
    back = np.zeros((nfreq, nt), int)
    score[:, 0] = E[:, 0]
    for i in range(1, nt):
        M = score[:, i - 1][None, :] - penalty * D   # (cur, prev)
        back[:, i] = np.argmax(M, axis=1)
        score[:, i] = E[:, i] + M[np.arange(nfreq), back[:, i]]

    path = np.zeros(nt, int)
    path[-1] = int(np.argmax(score[:, -1]))
    for i in range(nt - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    tr = fb[path].astype(float)
    if smooth_win >= 3:
        tr = median_filter(tr, size=smooth_win | 1, mode='nearest')

    conf = float(np.mean(Sb[path, np.arange(nt)] > CONF_RATIO * col_med.ravel()))
    return t / 3600.0, tr, conf


def enhance_spec(sig, fs, nperseg_sec=30, noverlap_sec=15, fmax=3.0, bg_hz=0.4):
    """Display spectrogram: per column, the dB spectrum minus its frequency-median
    filtered version (kernel ~bg_hz), which flattens 1/f and broadband motion
    brightening but leaves narrow peaks."""
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=int(nperseg_sec * fs),
                            noverlap=int(noverlap_sec * fs))
    m = f <= fmax
    f, Sxx = f[m], Sxx[m]
    db = 10 * np.log10(Sxx + 1e-20)
    dfq = f[1] - f[0] if len(f) > 1 else 1.0
    k = max(3, int(bg_hz / dfq) | 1)                 # odd kernel over ~bg_hz
    bg = median_filter(db, size=(k, 1), mode='nearest')
    return f, t, db - bg


def viterbi_nights(channel='CRE'):
    """Viterbi traces for every night on one channel: night median frequency and confidence."""
    rows = []
    for s in iter_sessions(profile=False):
        sig = rate_band_signal(s, channel)
        row = dict(session=s.label, channel=channel)
        for band, kw in TRACK.items():
            _, tr, conf = track_single_ridge(sig, s.fs, **kw)
            row[f'{band}_median_hz'] = float(np.median(tr))
            row[f'{band}_conf'] = conf
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Low band (¶158, ¶194): see _s05_lowband.py
# ═════════════════════════════════════════════════════════════════════════════

def lowband_all(channel='CRE'):
    nights, rows, eps, occ = [], [], [], []
    for s in iter_sessions():
        r = LB.night(s.cap[channel], s.fs)
        r.update(label=s.label, prof=s.sleep_profile)
        keep = r['keep']
        f_med = float(np.median(r['fr'][keep])) if keep.any() else np.nan
        r['f_med'] = f_med
        rows.append(dict(session=s.label, channel=channel, gate_db=r['gate_db'],
                         pct_night=100 * keep.mean(), n_episodes=len(r['episodes']),
                         f_median_hz=f_med,
                         longest_min=max([e['dur_min'] for e in r['episodes']], default=0.0)))
        for e in r['episodes']:
            eps.append(dict(session=s.label, channel=channel, **e))
        # stage under each gated ridge column (is the low band restricted to N3?)
        sp = s.sleep_profile
        idx = np.clip(np.searchsorted(sp['t_ep_hr'], r['t'], side='right') - 1,
                      0, len(sp['codes']) - 1)
        st = np.asarray(sp['codes'])[idx]
        for code in STAGE_ORDER:
            occ.append(dict(session=s.label, stage=STAGE_LABELS[code],
                            ridge_cols=int(np.sum(keep & (st == code))),
                            night_cols=int(np.sum(st == code))))
        nights.append(r)
        print(f'  low band {s.label}: ridge {100 * keep.mean():.0f}% of night, '
              f'{len(r["episodes"])} episodes, median {f_med:.3f} Hz, gate {r["gate_db"]:.2f} dB')
    return nights, pd.DataFrame(rows), pd.DataFrame(eps), pd.DataFrame(occ)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Figures
# ═════════════════════════════════════════════════════════════════════════════

def draw_stage_ladder(ax, sp, fontsize=12):
    """Hypnogram as a connected stepped ladder, Wake top / REM bottom (depth order)."""
    ax.set_yticks(STAGE_ROW_TICKS)
    ax.set_yticklabels(STAGE_ROW_LABELS, fontsize=fontsize)
    ax.set_ylim(-0.5, 4.5)
    if sp is None:
        return
    t = np.asarray(sp['t_ep_hr'], float)
    codes = np.asarray(sp['codes'])
    n = min(len(t), len(codes))
    ys = np.array([STAGE_ROW.get(int(c), np.nan) for c in codes[:n]], float)
    ax.step(t[:n], ys, where='post', color='#2c3e50', lw=1.3)
    ax.grid(True, axis='y', alpha=0.15)


def fig5(lowband_night):
    """Fig. 5: S4N2 CRE -- stage ladder, 0-3 Hz spectrogram + Viterbi traces, low band."""
    # CORRECTION: regenerated for S4N2 from the current code; the manuscript's
    # Fig. 5 is a stale render with the old (filter-artifact) low band.
    s = get_session(FIG5_SESSION)
    sig = rate_band_signal(s, FIG5_CHANNEL)
    t_resp, resp_tr, resp_p = track_single_ridge(sig, s.fs, **TRACK['resp'])
    t_card, card_tr, card_p = track_single_ridge(sig, s.fs, **TRACK['card'])
    r = lowband_night

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.32, 1.0, 0.72], hspace=0.12)
    ax0 = fig.add_subplot(gs[0])
    draw_stage_ladder(ax0, s.sleep_profile)
    ax0.set_ylabel('Stage')
    ax0.tick_params(labelbottom=False)
    ax0.set_title(f'{FIG5_SESSION} {FIG5_CHANNEL}: Viterbi tracker confidence respiratory '
                  f'{resp_p:.0%}, cardiac {card_p:.0%}; low-band ridge on '
                  f'{100 * r["keep"].mean():.0f}% of the night', fontsize=13)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    f1, t1, db1 = enhance_spec(sig, s.fs)
    ax1.pcolormesh(t1 / 3600, f1, db1, shading='gouraud', cmap='magma',
                   vmin=0, vmax=np.percentile(db1, 99.5), rasterized=True)
    ax1.plot(t_resp, resp_tr, color=TRACE_COLOR['resp'], lw=2.0, label='respiratory')
    ax1.plot(t_card, card_tr, color=TRACE_COLOR['card'], lw=2.0, label='cardiac')
    ax1.set_ylim(0, 3.0)
    ax1.set_ylabel('Frequency (Hz)')
    ax1.tick_params(labelbottom=False)
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    lo, hi = np.percentile(r['db'], [5, 99.5])
    ax2.pcolormesh(r['t'], r['f'], r['db'], shading='gouraud', cmap='viridis',
                   vmin=lo, vmax=hi, rasterized=True)
    ax2.plot(r['t'], np.where(r['keep'], np.nan, r['fr']), color='white', lw=1.0,
             alpha=0.45, ls=':', label='Viterbi path (below gate)')
    ax2.plot(r['t'], np.where(r['keep'], r['fr'], np.nan), color=RIDGE_COLOR, lw=2.6,
             label='gated low-band ridge')
    ax2.set_ylim(0.0, LB.FMAX)
    ax2.set_ylabel('Low band (Hz)')
    ax2.set_xlabel('Time (h)')
    ax2.legend(loc='upper right', fontsize=9)
    ax0.set_xlim(0, max(t1[-1] / 3600, r['t'][-1]))
    return save(fig, 'fig5_S4N2_CRE', STAGE)


def lowband_sheet(nights, channel='CRE'):
    """All twelve low-band panels, three columns by four rows."""
    ncol, nrow = 3, 4
    fig = plt.figure(figsize=(24, 17))
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[0.25, 1.0] * nrow,
                          hspace=0.40, wspace=0.12)
    for k, r in enumerate(nights):
        c, rw = k % ncol, k // ncol
        a0 = fig.add_subplot(gs[rw * 2, c])
        a1 = fig.add_subplot(gs[rw * 2 + 1, c], sharex=a0)
        draw_stage_ladder(a0, r['prof'], fontsize=8)
        a0.tick_params(labelbottom=False, labelsize=8)
        a0.set_title(f"{r['label']}: ridge {100 * r['keep'].mean():.0f}% of night, "
                     f"median {r['f_med']:.3f} Hz", loc='left', fontsize=13)
        lo, hi = np.percentile(r['db'], [5, 99.5])
        a1.pcolormesh(r['t'], r['f'], r['db'], shading='gouraud', cmap='viridis',
                      vmin=lo, vmax=hi, rasterized=True)
        a1.plot(r['t'], np.where(r['keep'], r['fr'], np.nan), color=RIDGE_COLOR, lw=2.4)
        a1.set_ylim(0, LB.FMAX)
        a1.set_xlim(0, 9)
        if c == 0:
            a1.set_ylabel('Hz')
        a1.tick_params(labelbottom=(rw == nrow - 1))
        if rw == nrow - 1:
            a1.set_xlabel('Time (h)')
    fig.suptitle(f'Low band (0-0.3 Hz), {channel}, all twelve nights: smoothed, '
                 '1/f-detrended spectrogram with the shuffle-gated ridge', fontsize=16, y=0.91)
    return save(fig, f'lowband_allsessions_{channel}', STAGE)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Numbers
# ═════════════════════════════════════════════════════════════════════════════

def _rng(x, fmt='{:.2f}'):
    return f'{fmt.format(np.nanmin(x))}–{fmt.format(np.nanmax(x))}'


def register(nb, epochs, ridges, summary, alldir, robust, pw_summ, dom, vit,
             lb_rows, lb_eps, lb_occ, fig5_conf):
    # ── Methods ¶153: Viterbi ──
    nb.add('vit_resp_band', 'Methods ¶153', 'Viterbi respiratory search band',
           f"{TRACK['resp']['band'][0]}-{TRACK['resp']['band'][1]}", '0.25-0.55 Hz',
           unit='Hz', source='TRACK')
    nb.add('vit_card_band', 'Methods ¶153', 'Viterbi cardiac search band',
           f"{TRACK['card']['band'][0]}-{TRACK['card']['band'][1]}", '0.85-1.45 Hz',
           unit='Hz', source='TRACK')
    nb.add('vit_method', 'Methods ¶153', 'what the Viterbi path runs on',
           'emission log(PSD / in-band column median) of a plain spectrogram (resp 30 s, '
           'cardiac 15 s windows, 50% overlap); penalty 20 / 24 per Hz; median smoothing '
           '13 / 23 windows', paper_value='a frequency-median-filtered spectrum was subtracted',
           status='DIFF', source='track_single_ridge',
           note='the median-filter background subtraction is the display spectrogram '
                '(enhance_spec), not the tracker input; text should describe the emission')
    nb.add('vit_conf_rule', 'Methods ¶153', 'tracker confidence criterion',
           f'tracked bin > {CONF_RATIO:g}x column median', paper_value='prespecified detection criterion',
           status='DIFF', source='track_single_ridge',
           note='Methods does not state the criterion (2x the in-band column median, i.e. 3 dB)')
    nb.add('fig5_conf_resp', 'Fig. 5', 'S4N2 CRE respiratory tracker confidence',
           100 * fig5_conf[0], unit='%', source='track_single_ridge')
    nb.add('fig5_conf_card', 'Fig. 5', 'S4N2 CRE cardiac tracker confidence',
           100 * fig5_conf[1], unit='%', source='track_single_ridge')
    nb.add('vit_conf_resp_range', '§3.3', 'CRE respiratory tracker confidence, 12 nights',
           _rng(100 * vit.resp_conf, '{:.0f}'), unit='%', source='viterbi_nights')
    nb.add('vit_conf_card_range', '§3.3', 'CRE cardiac tracker confidence, 12 nights',
           _rng(100 * vit.card_conf, '{:.0f}'), unit='%', source='viterbi_nights')

    # ── Methods ¶158: detector settings ──
    nb.add('ridge_min_dur', 'Methods ¶158', 'minimum ridge duration', MIN_PERSIST_SEC / 60,
           'at least 5 minutes', unit='min', source='BANDS / MIN_PERSIST_SEC')
    nb.add('ridge_win', 'Methods ¶158', 'spectrogram window', WIN_SEC,
           '30-second spectrogram windows', unit='s', source='WIN_SEC')
    nb.add('ridge_resp_band', 'Methods ¶158', 'respiratory ridge band',
           f"{BANDS['resp']['min_freq']}-{BANDS['resp']['max_freq']}", '(0.1-0.5 Hz)',
           unit='Hz', source='BANDS')
    nb.add('ridge_card_band', 'Methods ¶158', 'cardiac ridge band',
           f"{BANDS['card']['min_freq']}-{BANDS['card']['max_freq']}", '(0.5-3.0 Hz)',
           unit='Hz', source='BANDS')
    nb.add('ridge_card_seg', 'Methods ¶158', 'cardiac Welch segment',
           BANDS['card']['welch_seg_sec'], '8-second segments', unit='s', source='BANDS')
    nb.add('lb_band', 'Methods ¶158', 'low band (map range)', f'{LB.FMIN}~{LB.FMAX}',
           '(0.01~0.3 Hz)', unit='Hz', source='_s05_lowband',
           note=f'the ridge itself is searched in {LB.RIDGE_BAND[0]}-{LB.RIDGE_BAND[1]} Hz; '
                f'signal band-passed {LB.HP_HZ}-0.5 Hz, no accelerometer regression')
    nb.add('lb_window', 'Methods ¶158', 'low-band spectral window / step',
           f'{LB.WIN_S / 60:.0f}-minute Welch windows (4-min segments), step '
           f'{LB.STEP_S / 60:.1f} minutes',
           '5-minute windows with 50% overlap, yielding spectral estimates every 2.5 minutes',
           source='_s05_lowband',
           note='old ridge_overlay_tune.py low band also differed: 5-min windows stepped 30 s '
                '(90% overlap). Methods should describe the new low band')
    nb.add('lb_gate', 'Methods ¶158', 'low-band ridge gate (not in Methods)',
           f'>= gate dB above in-band median for >= {LB.MIN_RUN_MIN:.0f} min; gate from '
           f'{LB.N_SURR} 10-min block shuffles, <= {100 * LB.SURR_FALSE:.0f}% of night',
           source='shuffle_gate', note='Methods should state the gate and the null')

    # ── dominance (¶158, ¶190) ──
    for band in ('resp', 'card'):
        d = dom[dom.band == band]
        n_cre = int((d.dominant_by_prominence == 'CRE').sum())
        counts = d.dominant_by_prominence.value_counts().to_dict()
        note = ('rule: per night, channel with the highest median (motion-free epochs with a '
                'ridge) of the epoch maximum ridge prominence (peak / local floor). '
                f'Winners: {counts}. By ridge coverage instead: '
                f"{d.dominant_by_coverage.value_counts().to_dict()}. Old 9/12 came from "
                'run_ridge_overlay.py harmonic score (older detector)')
        if band == 'resp':
            nb.add('dominant_cre_158', 'Methods ¶158', 'nights where CRE holds the dominant '
                   'respiratory ridge', f'{n_cre} of 12', '9 of 12 recordings',
                   source='dominance', note=note)
            nb.add('dominant_cre_190', '§3.3 ¶190', 'nights where CRE holds the dominant '
                   'respiratory ridge', f'{n_cre} of 12', '9 of 12 sessions',
                   source='dominance', note=note)
        else:
            nb.add('dominant_cre_card', '§3.3', 'nights where CRE holds the dominant cardiac '
                   'ridge', f'{n_cre} of 12', source='dominance', note=note)

    # ── ridge frequencies (¶190) ──
    clean = epochs[~epochs.motion_masked.astype(bool) & (epochs.channel == 'CRE')]
    night_f = clean.groupby(['band', 'session']).mean_ridge_freq.median()
    nb.add('resp_ridge_freq', '§3.3 ¶190', 'respiratory ridge frequency, CRE, range of night '
           'medians (mean active-ridge frequency per epoch)', _rng(night_f['resp']),
           '0.20–0.25 Hz', unit='Hz', source='ridge_epoch_features',
           note=f'Viterbi trace night medians (search floor 0.25 Hz): '
                f'{_rng(vit.resp_median_hz, "{:.3f}")} Hz')
    nb.add('card_ridge_freq', '§3.3 ¶190', 'cardiac ridge frequency, CRE, range of night '
           'medians', _rng(night_f['card']), 'between 0.9 and 1.8 Hz', unit='Hz',
           source='ridge_epoch_features',
           note=f'Viterbi cardiac trace night medians (capped 1.45 Hz): '
                f'{_rng(vit.card_median_hz, "{:.3f}")} Hz')
    nb.add('infra_freq', '§3.3 ¶190', 'infra-slow (low-band) ridge frequency, CRE night '
           'medians', _rng(lb_rows.f_median_hz, '{:.3f}'), 'around 0.1 Hz', unit='Hz',
           source='lowband_all', note='no common frequency; several nights sit at the '
           '0.02 Hz search edge')
    n_resp = ridges[(ridges.band == 'resp') & (ridges.channel == 'CRE')].groupby('session').size()
    nb.add('n_resp_ridges_cre', '§3.3', 'persistent respiratory ridges per night, CRE',
           _rng(n_resp.values, '{:.0f}'), source='ridge_epochs')

    # ── ¶194: the low band (CORRECTION: old values were a filter artifact) ──
    art = ('old value came from the low band of ridge_overlay_tune.py, built on the '
           'motion-cancelled signal: its 0.05 Hz band-pass corner is a constant hump at '
           '0.05-0.08 Hz after 1/f detrending (filter artifact). Replaced by '
           'ridge_lowband_smooth.py (channel itself, 0.005-0.5 Hz, shuffle-gated ridge)')
    s1 = lb_rows.set_index('session').loc['S1N1']
    nb.add('lb_s1n1', '§3.3 ¶194', 'S1N1 low-band ridge',
           f"{s1.n_episodes} episodes, {s1.pct_night:.1f}% of night",
           'no corresponding slow-frequency ridge was detected during the first night of '
           'Subject 1', status='DIFF', source='lowband_all', note=art)
    nb.add('lb_per_night', '§3.3 ¶194', 'low-band ridge episodes per night',
           _rng(lb_rows.n_episodes, '{:.0f}'), '2–8 ridges were detected per night',
           source='lowband_all', note=art)
    nb.add('lb_total', '§3.3 ¶194', 'low-band ridge episodes, total',
           int(lb_rows.n_episodes.sum()), '67 total', source='lowband_all', note=art)
    nb.add('lb_median', '§3.3 ¶194', 'low-band ridge episodes per night, median',
           float(lb_rows.n_episodes.median()), 'median, 6', source='lowband_all', note=art)
    nb.add('lb_conc', '§3.3 ¶194', 'low-band ridge frequency, episode medians',
           _rng(lb_eps.f_med_hz, '{:.2f}'), 'concentrated near 0.05–0.08 Hz',
           unit='Hz', source='lowband_all', note=art)
    nb.add('lb_below012', '§3.3 ¶194', 'episodes below 0.12 Hz',
           f'{int((lb_eps.f_med_hz < 0.12).sum())} of {len(lb_eps)}',
           'Most occurred below 0.12 Hz', status='MATCH' if (lb_eps.f_med_hz < 0.12).mean() > 0.5
           else 'DIFF', source='lowband_all', note='qualitative statement; count shown')
    nb.add('lb_max_f', '§3.3 ¶194', 'highest episode frequency', float(lb_eps.f_med_hz.max()),
           'extended to approximately 0.30 Hz', unit='Hz', source='lowband_all',
           note=f'ridge search band ends at {LB.RIDGE_BAND[1]} Hz')
    nb.add('lb_longest', '§3.3 ¶194', 'longest episode', float(lb_eps.dur_min.max()),
           'approximately 1 hour or longer', unit='min', source='lowband_all',
           note=f'episodes >= 50 min: {int((lb_eps.dur_min >= 50).sum())}; none reaches 60 min '
                '("several ... 1 hour or longer" does not hold)', status='DIFF')
    nb.add('lb_pct_night', '§3.3 ¶194', 'share of the night with a gated low-band ridge, CRE',
           f'{_rng(lb_rows.pct_night, "{:.0f}")} (median {lb_rows.pct_night.median():.0f})',
           unit='%', source='lowband_all', note='null ceiling 5% by construction of the gate')
    nb.add('lb_n_episodes', '§3.3 ¶194', 'low-band episodes, nights with >= 1',
           f'{int((lb_rows.n_episodes > 0).sum())} of 12', source='lowband_all')
    occ = lb_occ.groupby('stage')[['ridge_cols', 'night_cols']].sum()
    share = (100 * occ.ridge_cols / occ.ridge_cols.sum()).round(0)
    nb.add('lb_stages', '§3.3 ¶194', 'stage of gated low-band ridge time, pooled, %',
           ', '.join(f'{st} {share[st]:.0f}' for st in ['Wake', 'N1', 'N2', 'N3', 'REM']),
           'occurred episodically across different sleep stages and were not restricted to N3',
           status='MATCH' if share.get('N3', 0) < 50 else 'DIFF', source='lowband_all',
           note='qualitative claim; per-night occupancy in lowband_stage_occupancy.csv')
    s4 = lb_rows.set_index('session').loc['S4N2']
    nb.add('fig5_lowband', 'Fig. 5 caption', 'S4N2 CRE low band',
           f'ridge {s4.pct_night:.0f}% of night at {s4.f_median_hz:.3f} Hz',
           'persistent power concentrated between 0.05 and 0.25 Hz', status='DIFF',
           source='fig5', note=art)

    # ── ¶195 / ¶248: N3 respiratory ridge power ──
    for ch in CHANNELS:
        r = pw_summ[(pw_summ.band == 'resp') & (pw_summ.channel == ch)].iloc[0]
        ad = alldir[(alldir.band == 'resp') & (alldir.channel == ch)
                    & (alldir.feature == 'total_ridge_power')].set_index('statistic')
        val = (f"count-matched {r.n3_lower_mean_matched}/{r.n_subj} lower by mean, "
               f"{r.n3_lower_median_matched}/{r.n_subj} by median; motion-free unmatched "
               f"{ad.loc['mean', 'n3_lower']}/6 by mean, {ad.loc['median', 'n3_lower']}/6 by median")
        most = r.n3_lower_mean_matched >= 4 and r.n3_lower_median_matched >= 4
        nb.add(f'resp_power_n3_{ch}', '§3.3 ¶195 / ¶248',
               f'participants with lower respiratory ridge power in N3, {ch}', val,
               'Respiratory ridge power was lower during N3 in most participants ... after '
               'controlling for motion and unequal stage duration',
               status='MATCH' if most else 'DIFF', source='power_count_matched',
               note='CORRECTION: count matching newly applied to power (200 draws, within '
                    'participant, N3 vs all other stages); MATCH = >= 4/6 on both statistics')
    for ch in CHANNELS:
        r = pw_summ[(pw_summ.band == 'card') & (pw_summ.channel == ch)].iloc[0]
        nb.add(f'card_power_n3_{ch}', '§3.3', f'participants with lower cardiac ridge power in '
               f'N3, {ch}, count-matched', f'{r.n3_lower_mean_matched}/{r.n_subj} by mean, '
               f'{r.n3_lower_median_matched}/{r.n_subj} by median', source='power_count_matched')
    for ch in CHANNELS:
        rr = robust[(robust.band == 'resp') & (robust.channel == ch)
                    & (robust.subset == 'count-matched')].iloc[0]
        nb.add(f'resp_count_n3_{ch}', '§3.3', f'participants with fewer respiratory ridges in '
               f'N3, {ch}, count-matched', f'{rr.n3_lower}/{rr.n_subj}', source='ridge_robustness',
               note='Discussion ¶248: "ridge counts varied among participants"')
    cre = summary[(summary.band == 'resp') & (summary.feature == 'total_ridge_power')].iloc[0]
    nb.add('resp_power_kw', '§3.3', 'pooled KW p, CRE respiratory ridge power by stage '
           '(DESCRIPTIVE ONLY, epochs not independent)', f'{cre.kw_p:.1e}', source='stage_summary',
           note='not inferential; participant directions are the evidence')
    nb.add('resp_power_mwu', '§3.3', 'pooled MWU p, CRE respiratory ridge power N3 vs '
           'N1+N2+REM (DESCRIPTIVE ONLY)', f'{cre.mwu_p:.1e}', source='stage_summary',
           note='not inferential')
    nb.add('count_match_draws', 'Methods ¶159', 'count-matching draws per participant', N_DRAWS,
           source='ridge_robustness / power_count_matched',
           note='Methods ¶159 does not state the number of draws')


# ═════════════════════════════════════════════════════════════════════════════

def run():
    tab = TAB_DIR / STAGE
    tab.mkdir(parents=True, exist_ok=True)
    nb = Numbers(STAGE)

    epochs, ridges = ridge_epochs()
    ridges.groupby(['band', 'channel', 'session']).size().rename('n_ridges') \
        .reset_index().to_csv(tab / 'ridge_counts.csv', index=False)

    summary = stage_summary(epochs)
    summary.to_csv(tab / 'band_ridge_stage_summary.csv', index=False)
    alldir = all_channel_directions(epochs)
    alldir.to_csv(tab / 'ridge_stage_all_channels.csv', index=False)
    robust = ridge_robustness(epochs)
    robust.to_csv(tab / 'ridge_robustness.csv', index=False)
    pw_frames, pw_summs = [], []
    for feat in ('total_ridge_power', 'n_ridges'):
        per, summ = power_count_matched(epochs, feat)
        pw_frames.append(per)
        pw_summs.append(summ)
    pd.concat(pw_frames).to_csv(tab / 'ridge_count_matched_per_participant.csv', index=False)
    pd.concat(pw_summs).to_csv(tab / 'ridge_count_matched_summary.csv', index=False)
    pw_summ = pw_summs[0]
    print(pw_summ.to_string(index=False))
    dom = dominance(epochs)
    dom.to_csv(tab / 'ridge_dominance.csv', index=False)

    vit = viterbi_nights('CRE')
    vit.to_csv(tab / 'viterbi_nights_CRE.csv', index=False)

    nights, lb_rows, lb_eps, lb_occ = lowband_all('CRE')
    lb_rows.to_csv(tab / 'lowband_nights.csv', index=False)
    lb_eps.to_csv(tab / 'lowband_episodes.csv', index=False)
    lb_occ.to_csv(tab / 'lowband_stage_occupancy.csv', index=False)

    fig5(next(r for r in nights if r['label'] == FIG5_SESSION))
    lowband_sheet(nights, 'CRE')
    v = vit.set_index('session').loc[FIG5_SESSION]

    register(nb, epochs, ridges, summary, alldir, robust, pw_summ, dom, vit,
             lb_rows, lb_eps, lb_occ, (v.resp_conf, v.card_conf))
    nb.save()


if __name__ == '__main__':
    run()
