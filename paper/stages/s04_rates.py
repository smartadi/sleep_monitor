"""Respiratory and cardiac rate from loose peak counting, per-recording k, and held-out calibration.

Manuscript items
    Discussion ¶253     night-level error 0.24 br/min / 1.56 BPM; epoch-level 1.79 / 3.41;
                        cardiac k 1.96 (1.77–2.01); 2.02 peaks per cardiac cycle; resp k 1.18
    Discussion ¶254     nightly mean respiratory reference range; constant predictor 1.20
    Conclusion ¶259     0.24 br/min / 1.56 BPM
    Supp ¶222–245       the rate section: Fig. S7 (worked example), S8 (k by channel and
                        detector), the cardiac-reference check against ECG R-peaks, S9 (raw
                        counts and per-epoch k), S10 (all twelve recordings), S11 (calibration
                        regimes against a no-sensor baseline), S12 (per-epoch k spread)
    Tables              epochs (per-epoch estimates + PSG reference), per_session,
                        heldout_table, k_by_channel, ref_check, peaks_per_beat,
                        k_per_epoch_spread, allsessions_error, worked_example

The pipeline: one channel, band-passed with the OLS motion canceller to 0.1–0.5 Hz
(respiration) or 0.5–3.0 Hz (cardiac), peaks counted in each non-overlapping 30-s
epoch with `seclib.rates.rate_peaks(prom_factor=0.05)`, divided by one k per
recording. The operational channel is CRE in both bands. k = median over the
night of estimate ÷ PSG reference. The spectral estimator (`rate_spectral`) is
computed only because Fig. S8 shows it; it is degenerate in the respiratory band
(returns 0.25 Hz = 15 br/min in nearly every epoch), which the supplement states.

Ported from:
    analysis/rates/rerun_rate_detection.py   prepare_channels, phase_a, fit_k, pairs,
                                             evaluate, med_iqr, the held-out summary in main
    analysis/rates/supp_k_by_channel.py      k_table, figure                      (Fig. S8)
    analysis/rates/ref_sanity_check.py       main                (cardiac reference vs ECG)
    analysis/rates/supp_rate_pipeline.py     loose_peaks, worked_example, fig_pipeline,
                                             fig_result                      (Fig. S7, S11)
    analysis/rates/supp_rate_epoch_k.py      smooth, fig_allnight_counts_and_k,
                                             fig_k_per_epoch                 (Fig. S9, S12)
    analysis/rates/supp_rate_allsessions.py  main                                 (Fig. S10)
    analysis/rates/peaks_per_beat.py         bp, cap_peaks, asleep_mask_samples, main

Changes from the legacy code:
    - Only what the paper reports is computed. Dropped: the avg and CLE−CRE channels,
      the spectral_interp / peaks_strict / hilbert estimators, the per-window quality
      score, the PSG stage column and the estimator-selection step that picked
      peaks_loose on CRE (the check confirms the legacy selection is exactly that, so
      it is fixed here as the operational choice), fig_rate_fullnight (not in the
      supplement), and the CLE−CRE rows of peaks_per_beat.
    - One pass over the twelve nights does all session-level work (per-epoch rates,
      the ECG R-peak check, peaks per beat, the worked example); the legacy scripts
      each reloaded every night.
    - The respiratory reference is the consensus product of stage s03_signal (built by
      the ported scripts/build_consolidated_resp_gt.py), passed to
      seclib.ground_truth explicitly because its default path points inside paper/.
    - ref_check: the legacy CSV predates the script's own 'ECG unusable' flag (it kept
      ratios of 262 and 879 for S5N1 and S6N2); the port applies the flag as the
      script wrote it. It also reports the median reference over sleep epochs, which
      is what the supplement says for S6N2.
    - peaks_per_beat: k_reported came from the superseded June pipeline
      (reports/rates/mask/per_session_summary.csv); it is now this stage's CRE k.
    - CORRECTION (1) Fig. S7 divided the worked example by the cohort k 1.18 while the
      text says k is one number per recording. The example now divides by S2N1's own
      CRE respiratory k (1.03) and says so; the cohort-k version is registered too.
    - CORRECTION (2) "R-peak-triggered averaging yielded 2.02 peaks per cardiac cycle":
      the number is a peak COUNT over the asleep night divided by the ECG R-peak count,
      not a triggered average. Registered with that note. The legacy median also
      dropped S4N1 by hand (a 7,700 fF glitch inflates σ, so the 0.05σ prominence
      admits 113 peaks all night); the exclusion is now a stated rule, < 0.5 peaks
      per beat = detector failure.
    - CORRECTION (3) the nightly-mean respiratory reference range (¶254, 14.4–16.8
      br/min) came from the old pipeline; the rerun range is registered. ¶254 cites a
      "Table 3" that V11 does not have.
    - CORRECTION (4) the verified-reference cardiac k (1.98, 1.81–2.28, eight
      recordings) was typed into the supplement build; it is computed here from the
      ECG check.
    - CORRECTION (5) the spectral respiratory k (0.96) is kept and registered with
      the fraction of epochs at the single 0.25 Hz bin, so the degeneracy the
      supplement describes is a computed fact.
    - CORRECTION (6) the cardiac reference. gt_heart_rate takes ECG R-peaks and falls back
      to Pleth when the ECG detector raises; the exception is swallowed. In the legacy
      rerun (artifacts/rate_rerun_phase_a.parquet) the fallback fired on all twelve
      nights -- its reference equals the Pleth-only reference to the last bit -- while
      the same ECG detector worked in ref_sanity_check.py and peaks_per_beat.py, and
      works today on every night with a usable ECG (the cause is not reproducible).
      The Pleth detector over-counts on S2N1 (+36%) and S6N1 (+29%): that is the
      "reference runs high" caveat of ¶229. This stage uses gt_heart_rate as written,
      so the reference is ECG on ten nights and Pleth on S5N1 and S6N2. Every cardiac
      number moves (night error 1.56 -> 1.19 BPM, k 1.96 -> 2.00); the check
      reproduces the legacy values exactly when the Pleth fallback is forced.
    - The respiratory reference is rebuilt by s03_signal, which keeps the legacy Flow
      detector path (the same silent neurokit2 fallback, see s03's change 6); it equals
      the legacy artifact in all but 69 of 55,856 grid points, and the respiratory
      numbers reproduce the paper.
    - The per-epoch estimates match today's legacy code bit for bit; the stored legacy
      parquet differs from it in 92 of 55,914 peak counts (0.16%, 71 of them S4N1) and
      7 spectral values, moving the held-out IQR bounds in the second decimal. That is
      the stored artifact, not the port (see the check).

Two definitions of k exist and are both kept, because both are quoted: the held-out
evaluation (main text 1.18 / 1.96) DROPS ratios outside 0.3–5.0 and needs ≥10
epochs (`fit_k`); Figs. S8–S12 CLIP ratios to 0.3–5.0 (`k_clipped`), which is what
the supplement text describes. They differ in the third decimal.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

import seclib
from seclib import ground_truth
from seclib.config import RESP_LO, RESP_HI, CARD_LO, CARD_HI
from seclib.filters import bandpass
from seclib.numbers import Numbers
from seclib.preprocessing import remove_acc_artifact
from seclib.rates import rate_peaks, rate_spectral

STAGE = 's04_rates'
REQUIRES = ['s03_signal']

CACHE = seclib.CACHE_DIR / STAGE
TAB = seclib.TAB_DIR / STAGE
# respiratory reference: multi-sensor consensus, gated, from stage s03_signal
RESP_GT = seclib.CACHE_DIR / 's03_signal' / 'consolidated_resp_gt.parquet'

BANDS = {'resp': (RESP_LO, RESP_HI), 'card': (CARD_LO, CARD_HI)}
UNIT = {'resp': 'br/min', 'card': 'BPM'}
CHANNELS = ['CH', 'CLE', 'CRE']
OPERATIONAL = ('peaks_loose', 'CRE')      # estimator, channel -- both bands
WIN_SEC = 30.0
PROM_LOOSE = 0.05                         # rate_peaks prominence, in σ of the epoch
K_LO, K_HI = 0.3, 5.0
SMOOTH_EPOCHS = 5                         # 5 x 30 s, for the whole-night figures

# participant ages (years), from the study's subject list
AGE = {'OS006': 25, 'OS003': 37, 'OS004': 54, 'OS005': 55, 'OS001': 61, 'OS002': 66}

# Fig. S7: one motion-free minute of S2N1, CRE, respiratory band
EXAMPLE = dict(session='S2N1', minutes_in=120.0, span_s=60.0)
EXAMPLE_MIN_DIST_S = 0.4                  # the figure's own detector (see numbers)

# peaks per heartbeat: cardiac band on the raw channel, as peaks_per_beat.py
PPB_ORDER, PPB_MIN_DIST_S = 4, 0.25
PPB_FAIL = 0.5                            # below this the detector has failed


# ─────────────────────────────────────────────────────── per-epoch estimates

def prepare_channels(sess):
    return {'CLE': sess.cap['CLE'].astype(np.float64),
            'CRE': sess.cap['CRE'].astype(np.float64),
            'CH': sess.cap['CH'].astype(np.float64),
            'acc': sess.cap['acc_mag'].astype(np.float64)}


def epoch_rates(sess) -> pd.DataFrame:
    """Every 30-s epoch, band and channel: spectral and loose-peak rate, PSG reference (Hz)."""
    fs = sess.fs
    win_n = int(WIN_SEC * fs)
    chans = prepare_channels(sess)
    acc = chans['acc']
    starts = np.arange(0, sess.n_samples - win_n + 1, win_n)
    centres = (starts + win_n / 2.0) / fs / 3600.0

    # PSG reference on its 30 s / 5 s grid, matched to each epoch centre
    gt_data = ground_truth.gt_sliding_rates(sess, win_sec=30.0, step_sec=5.0)
    gt_t = gt_data['t_hr']
    gt = {}
    for band, key in [('resp', 'resp_hz'), ('card', 'card_hz')]:
        r = np.full(len(centres), np.nan)
        for i, t in enumerate(centres):
            dists = np.abs(gt_t - t)
            j = int(np.argmin(dists))
            if dists[j] < 0.01:
                r[i] = gt_data[key][j]
        gt[band] = r

    bp = {(c, b): remove_acc_artifact(chans[c], acc, lo, hi, fs)
          for c in CHANNELS for b, (lo, hi) in BANDS.items()}

    rows = []
    for ei, s0 in enumerate(starts):
        s1 = s0 + win_n
        for band, (lo, hi) in BANDS.items():
            for ch in CHANNELS:
                sig = bp[(ch, band)][s0:s1]
                rows.append(dict(
                    session=sess.label, epoch=ei, t_hr=centres[ei], band=band,
                    channel=ch, gt_hz=gt[band][ei],
                    spectral=rate_spectral(sig, lo, hi, fs),
                    peaks_loose=rate_peaks(sig, lo, hi, fs, prom_factor=PROM_LOOSE)))
    out = pd.DataFrame(rows)
    out.attrs['card_source'] = gt_data['card_gt'].signal_used     # 'ECG' or 'Pleth'
    return out


# ──────────────────────────────────────────── ECG R-peaks: reference check, peaks per beat

def ecg_rpeaks(sess):
    """Raw R-peak indices, or (None, reason) when the ECG cannot be used."""
    ecg = sess.psg.get('ECG')
    if ecg is None:
        return None, 'no ECG channel'
    try:
        return ground_truth._ecg_rpeaks(np.asarray(ecg, float), sess.fs), ''
    except Exception as e:                       # noqa: BLE001
        return None, type(e).__name__


def pleth_reference_bpm(sess):
    """Median epoch rate of the Pleth fallback alone -- the cardiac reference the legacy
    rerun used on every night (see CORRECTION 6)."""
    fs = sess.fs
    pk = ground_truth._fallback_peaks(sess.psg['Pleth'], CARD_LO, CARD_HI, fs)
    pk = ground_truth._quality_filter(pk, fs, CARD_LO, CARD_HI)
    win_n = int(WIN_SEC * fs)
    starts = np.arange(0, sess.n_samples - win_n + 1, win_n)
    rate = ground_truth._peaks_to_sliding_rate(pk / fs, (starts + win_n / 2.0) / fs, WIN_SEC)
    return float(np.nanmedian(rate) * 60)


def ecg_rate(rp, fs):
    """Median R-R rate; a detector that fails returns nonsense rather than raising."""
    bpm = 60.0 / (np.median(np.diff(rp)) / fs)
    return bpm, ('' if 30 < bpm < 200 else 'ECG unusable')


def asleep_mask_samples(n, prof, fs):
    """Per-sample boolean: inside a scored, non-Wake 30-s epoch."""
    m = np.zeros(n, bool)
    if prof is None:
        return ~m
    for t0, c in zip(prof['t_ep_hr'], prof['codes']):
        if c in (4, -1):
            continue
        a = max(0, int(t0 * 3600 * fs))
        b = min(n, int((t0 * 3600 + 30.0) * fs))
        if b > a:
            m[a:b] = True
    return m


def peaks_per_beat(sess, rp):
    """Cardiac-band CRE peaks counted over the asleep night ÷ ECG R-peaks over the same span.

    A count ratio, not a triggered average: band-pass the raw channel (4th-order
    0.5–3 Hz), 5-sample moving average, peaks ≥0.25 s apart with prominence 0.05σ.
    """
    fs = sess.fs
    if rp is None or len(rp) < 200:
        return None
    asleep = asleep_mask_samples(len(sess.time_hr), sess.sleep_profile, fs)
    rp = rp[asleep[np.clip(rp, 0, len(asleep) - 1)]]
    if len(rp) < 200:
        return None
    b, a = butter(PPB_ORDER, [CARD_LO / (fs / 2), CARD_HI / (fs / 2)], btype='band')
    sig = filtfilt(b, a, sess.cap['CRE'].astype(np.float64))
    sm = np.convolve(sig, np.ones(5) / 5, mode='same')
    pk, _ = find_peaks(sm, distance=int(PPB_MIN_DIST_S * fs),
                       prominence=PROM_LOOSE * np.std(sm))
    pk = pk[asleep[np.clip(pk, 0, len(asleep) - 1)]]
    return dict(session=sess.label, channel='CRE', cap_peaks=len(pk),
                ecg_beats=len(rp), peaks_per_beat=len(pk) / len(rp))


# ──────────────────────────────────────────────────────────── Fig. S7 worked example

def worked_example_segment(sess):
    """The S2N1 minute: raw CRE, band-passed CRE, and the figure's peaks in it."""
    fs = sess.fs
    raw = np.asarray(sess.cap['CRE'], dtype=float)
    i0 = int(EXAMPLE['minutes_in'] * 60 * fs)
    i1 = i0 + int(EXAMPLE['span_s'] * fs)
    seg_bp = bandpass(raw, RESP_LO, RESP_HI, fs)[i0:i1]
    pk = find_peaks(seg_bp, distance=max(1, int(EXAMPLE_MIN_DIST_S * fs)),
                    prominence=PROM_LOOSE * np.std(seg_bp))[0]
    n = len(pk)
    raw_rate = (n - 1) / ((pk[-1] - pk[0]) / fs) * 60 if n >= 2 else np.nan
    return dict(t=np.arange(len(seg_bp)) / fs, raw=raw[i0:i1], bp=seg_bp, pk=pk,
                n=n, raw_rate=raw_rate, t0_hr=i0 / fs / 3600, t1_hr=i1 / fs / 3600)


# ─────────────────────────────────────────────────────── calibration and evaluation

def fit_k(raw, gt):
    """Session k for the held-out evaluation: median ratio, out-of-range ratios dropped."""
    r = raw / gt
    r = r[(r > K_LO) & (r < K_HI) & np.isfinite(r)]
    return float(np.median(r)) if len(r) >= 10 else np.nan


def k_clipped(est, ref):
    """Session k as Figs. S8–S12 compute it: median ratio, ratios clipped to 0.3–5.0."""
    return (est / ref).clip(K_LO, K_HI)


def pairs(df, band, channel, est):
    g0 = df[(df.band == band) & (df.channel == channel)].dropna(subset=['gt_hz'])
    out = {}
    for s, g in g0.groupby('session'):
        g = g.sort_values('epoch')
        raw, gt = g[est].values * 60.0, g.gt_hz.values * 60.0
        m = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
        if m.sum() >= 20:
            out[s] = (raw[m], gt[m])
    return out


def evaluate(df, band, channel, est) -> pd.DataFrame:
    """Per-session accuracy under four calibration regimes.

    self        k fitted on the night being scored
    cross       k from the subject's other night                    [held out]
    pop         median k of the other subjects (leave-one-subject-out) [held out]
    nosensor    predict the other subjects' median reference every epoch
    """
    D = pairs(df, band, channel, est)
    ks = {s: fit_k(r, g) for s, (r, g) in D.items()}
    rows = []
    for s, (raw, gt) in D.items():
        subj = s[:2]
        other = subj + ('N2' if s.endswith('N1') else 'N1')
        loso_k = [v for t, v in ks.items() if not t.startswith(subj) and np.isfinite(v)]
        loso_ref = [g for t, (_, g) in D.items() if not t.startswith(subj)]
        k_pop = float(np.median(loso_k)) if loso_k else np.nan
        ref_pop = float(np.median(np.concatenate(loso_ref))) if loso_ref else np.nan
        k_self, k_x = ks[s], ks.get(other, np.nan)

        def night(pred):
            return abs(np.mean(pred) - np.mean(gt))

        def epoch(pred):
            return float(np.median(np.abs(pred - gt)))

        rows.append(dict(
            session=s, subject=subj, n=len(gt), k=k_self, k_cross=k_x, k_pop=k_pop,
            ref_mean=float(np.mean(gt)), ref_sd=float(np.std(gt)),
            night_self=night(raw / k_self),
            night_cross=night(raw / k_x) if np.isfinite(k_x) else np.nan,
            night_pop=night(raw / k_pop) if np.isfinite(k_pop) else np.nan,
            night_nosensor=abs(ref_pop - np.mean(gt)),
            epoch_self=epoch(raw / k_self),
            epoch_cross=epoch(raw / k_x) if np.isfinite(k_x) else np.nan,
            epoch_pop=epoch(raw / k_pop) if np.isfinite(k_pop) else np.nan,
            epoch_nosensor=epoch(np.full_like(gt, ref_pop)),
            r_within=(np.corrcoef(raw, gt)[0, 1] if np.std(raw) > 1e-9 else np.nan),
            raw_sd=float(np.std(raw)),
        ))
    return pd.DataFrame(rows).sort_values('session').reset_index(drop=True)


def med_iqr(v, dp=2):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return 'n/a'
    return f'{np.median(v):.{dp}f} [{np.percentile(v, 25):.{dp}f}–{np.percentile(v, 75):.{dp}f}]'


def heldout_summary(per_session) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        d = per_session[per_session.band == band]
        row = {'band': band, 'unit': UNIT[band],
               'estimator': OPERATIONAL[0], 'channel': OPERATIONAL[1]}
        for col in ['self', 'cross', 'pop', 'nosensor']:
            row[f'night_{col}'] = med_iqr(d[f'night_{col}'])
            row[f'epoch_{col}'] = med_iqr(d[f'epoch_{col}'])
        row['k_median'] = med_iqr(d.k)
        row['r_within'] = f'{d.r_within.median():+.3f}'
        row['nights_r_pos'] = f'{int((d.r_within > 0).sum())}/{len(d)}'
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────── k tables behind Figs. S8–S12

def subject_of():
    return {m['label']: m['subject'] for m in seclib.SESSION_META}


def k_table(d) -> pd.DataFrame:
    """k per (band, method, channel, session) -- Fig. S8."""
    rows = []
    for meth in ('peaks_loose', 'spectral'):
        t = d.assign(k=k_clipped(d[meth], d.gt_hz)).dropna(subset=['k'])
        g = (t.groupby(['band', 'channel', 'session'])['k']
             .agg(k='median', n_epochs='size').reset_index())
        g['method'] = meth
        rows.append(g)
    k = pd.concat(rows, ignore_index=True)
    k['subject'] = k.session.map(subject_of())
    k['night'] = k.session.str[-1].astype(int)
    return k.sort_values(['band', 'method', 'channel', 'session']).reset_index(drop=True)


def smooth(s):
    return s.rolling(SMOOTH_EPOCHS, center=True, min_periods=2).median()


def k_per_epoch_spread(d) -> pd.DataFrame:
    """Per-epoch k (count ÷ reference, smoothed), its median and IQR per night -- Fig. S12."""
    rows = []
    subj = subject_of()
    for band in BANDS:
        for sess in sorted(d.session.unique()):
            for ch in CHANNELS:
                g = d[(d.session == sess) & (d.channel == ch)
                      & (d.band == band)].sort_values('t_hr')
                ke = smooth(k_clipped(g.peaks_loose, g.gt_hz)).dropna()
                if len(ke) < 20:
                    continue
                q1, med_, q3 = ke.quantile([.25, .5, .75])
                rows.append(dict(session=sess, subject=subj[sess], band=band, channel=ch,
                                 k_median=med_, k_q1=q1, k_q3=q3, iqr=q3 - q1,
                                 n_epochs=int(len(ke))))
    return pd.DataFrame(rows)


def night_traces(d, sess, band, ch):
    """One night, one channel: smoothed estimate (÷ its own k) and smoothed reference, per min."""
    b = d[(d.session == sess) & (d.channel == ch) & (d.band == band)].sort_values('t_hr')
    k = k_clipped(b.peaks_loose, b.gt_hz).median()
    est = smooth(b.peaks_loose / k * 60.0)
    ref = smooth(b.gt_hz * 60.0)
    return b, float(k), est, ref


def allsessions_error(d) -> pd.DataFrame:
    """Median |smoothed estimate − smoothed reference| per night and channel -- Fig. S10."""
    subj = subject_of()
    rows = []
    for sess in sorted(d.session.unique()):
        for band in BANDS:
            for ch in CHANNELS:
                _, k, est, ref = night_traces(d, sess, band, ch)
                rows.append(dict(session=sess, subject=subj[sess], age=AGE[subj[sess]],
                                 band=band, channel=ch, k=k,
                                 median_abs_err=float(np.nanmedian(np.abs(est - ref)))))
    return pd.DataFrame(rows)


def counts_night(d):
    """The night drawn in Fig. S9: most CRE epochs with a peak-count estimate."""
    return (d[(d.channel == 'CRE') & d.peaks_loose.notna()]
            .groupby('session').size().idxmax())


# ───────────────────────────────────────────────────────────────────────── run

def compute(resp_gt=RESP_GT):
    """One pass over the twelve nights; returns the per-epoch table and the ECG-side tables."""
    if not resp_gt.exists():
        raise FileNotFoundError(f'{resp_gt} missing: run stage s03_signal first')
    ground_truth._CONSENSUS_PARQUET = resp_gt
    ground_truth._CONSENSUS_CACHE.clear()

    epochs, ecg_rows, ppb_rows, profiles, example = [], [], [], {}, None
    for sess in seclib.iter_sessions():
        profiles[sess.label] = sess.sleep_profile
        ep = epoch_rates(sess)
        epochs.append(ep)
        rp, why = ecg_rpeaks(sess)
        bpm, note = (np.nan, why) if rp is None else ecg_rate(rp, sess.fs)
        ecg_rows.append(dict(session=sess.label, card_ref_source=ep.attrs['card_source'],
                             ecg_bpm=bpm, pleth_ref_bpm=pleth_reference_bpm(sess), note=note))
        ppb = peaks_per_beat(sess, rp)
        if ppb is not None:
            ppb_rows.append(ppb)
        if sess.label == EXAMPLE['session']:
            example = worked_example_segment(sess)
        print(f'  {sess.label}: {ep.epoch.nunique()} epochs', flush=True)
    return (pd.concat(epochs, ignore_index=True), pd.DataFrame(ecg_rows),
            pd.DataFrame(ppb_rows), profiles, example)


def ref_check(d, ecg, profiles_asleep):
    """Cardiac reference (CRE rows) against the median ECG R-R rate of the same night.

    ratio        this stage's reference ÷ ECG rate
    pleth_ratio  Pleth-fallback reference ÷ ECG rate: the legacy check (¶229), whose
                 reference was the Pleth fallback on every night
    """
    card = d[(d.band == 'card') & (d.channel == 'CRE')]
    t = ecg.copy()
    t['ref_bpm'] = t.session.map(card.groupby('session').gt_hz.median() * 60)
    t['ref_bpm_asleep'] = t.session.map(profiles_asleep)
    t['ratio'] = (t.ref_bpm / t.ecg_bpm).round(2)
    t['pleth_ratio'] = (t.pleth_ref_bpm / t.ecg_bpm).round(2)
    t.loc[t.note != '', ['ratio', 'pleth_ratio']] = np.nan
    t['verified'] = (t.ratio > 0.9) & (t.ratio < 1.15)
    t['pleth_verified'] = (t.pleth_ratio > 0.9) & (t.pleth_ratio < 1.15)
    return t


def asleep_reference(d, profiles):
    """Median cardiac reference over epochs scored as sleep (stage codes 0–3)."""
    out = {}
    card = d[(d.band == 'card') & (d.channel == 'CRE')]
    for label, prof in profiles.items():
        g = card[card.session == label]
        if prof is None:
            out[label] = np.nan
            continue
        idx = (g.t_hr.values / (30.0 / 3600.0)).astype(int)
        ok = (idx >= 0) & (idx < len(prof['codes']))
        codes = np.full(len(g), -1)
        codes[ok] = prof['codes'][idx[ok]]
        out[label] = float(np.nanmedian(g.gt_hz.values[(codes >= 0) & (codes <= 3)]) * 60)
    return out


def s9_stats(d, sess):
    """What ¶233 says about the Fig. S9 night, in numbers."""
    g = d[d.session == sess]
    out = {}
    card = g[g.band == 'card']
    counts = np.concatenate([(smooth(card[card.channel == ch].sort_values('t_hr').peaks_loose)
                              * 60.0).to_numpy() for ch in CHANNELS])
    ref = smooth(card[card.channel == 'CRE'].sort_values('t_hr').gt_hz) * 60.0
    out['card_count'] = float(np.nanmedian(counts))
    out['card_ref'] = tuple(np.nanpercentile(ref, [5, 95]))
    rs = []                    # how closely the three channels' per-epoch k move together
    for band in BANDS:
        tr = pd.DataFrame({ch: smooth(k_clipped(b.peaks_loose, b.gt_hz)).to_numpy()
                           for ch in CHANNELS
                           for b in [g[(g.band == band) & (g.channel == ch)].sort_values('t_hr')]})
        rs.append(tr.corr().values[np.triu_indices(3, 1)].min())
    out['k_r'] = float(min(rs))
    return out


def assemble(d, ecg, ppb, profiles, ex) -> dict:
    """Everything the tables, figures and numbers need, from the one pass over the nights."""
    per_session = pd.concat(
        [evaluate(d, band, OPERATIONAL[1], OPERATIONAL[0]).assign(band=band)
         for band in BANDS], ignore_index=True)
    per_session = per_session[['band'] + [c for c in per_session if c != 'band']]

    ppb = ppb.copy()
    ppb['k_card_cre'] = ppb.session.map(
        per_session[per_session.band == 'card'].set_index('session').k)
    # CORRECTION (2): the detector-failure exclusion behind the quoted median, as a rule
    ppb['detector_failed'] = ppb.peaks_per_beat < PPB_FAIL

    # Fig. S7 worked example: the recording's own k (CORRECTION 1) and the cohort k
    resp = per_session[per_session.band == 'resp']
    k_own = float(resp.set_index('session').loc[EXAMPLE['session'], 'k'])
    k_cohort = float(np.round(resp.k.median(), 2))
    g = d[(d.session == EXAMPLE['session']) & (d.band == 'resp') & (d.channel == 'CRE')]
    win = g[(g.t_hr > ex['t0_hr']) & (g.t_hr < ex['t1_hr'])]
    ex = dict(ex, session=EXAMPLE['session'], k_own=k_own, k_cohort=k_cohort,
              rate_own=ex['raw_rate'] / k_own, rate_cohort=ex['raw_rate'] / k_cohort,
              ref=float(win.gt_hz.mean() * 60),
              operational_raw=float(win.peaks_loose.mean() * 60))

    s9_night = counts_night(d)
    return dict(d=d, per_session=per_session, held=heldout_summary(per_session),
                kt=k_table(d), spread=k_per_epoch_spread(d), alls=allsessions_error(d),
                refc=ref_check(d, ecg, asleep_reference(d, profiles)), ppb=ppb, ex=ex,
                s9_night=s9_night, s9=s9_stats(d, s9_night))


def run():
    from stages._s04_rates_figures import (fig_s7_pipeline, fig_s8_k_by_channel,
                                           fig_s9_counts_and_k, fig_s10_allsessions,
                                           fig_s11_calibration, fig_s12_k_per_epoch)
    for p in (CACHE, TAB):
        p.mkdir(parents=True, exist_ok=True)

    d, ecg, ppb, profiles, ex = compute()
    d.to_parquet(CACHE / 'epochs.parquet', index=False)
    P = assemble(d, ecg, ppb, profiles, ex)

    example_tab = pd.DataFrame([{k: v for k, v in P['ex'].items() if np.isscalar(v)}])
    for name, t in [('per_session', P['per_session']), ('heldout_table', P['held']),
                    ('k_by_channel', P['kt']), ('ref_check', P['refc']),
                    ('peaks_per_beat', P['ppb']), ('k_per_epoch_spread', P['spread']),
                    ('allsessions_error', P['alls']), ('worked_example', example_tab)]:
        t.to_csv(TAB / f'{name}.csv', index=False)

    fig_s7_pipeline(P['ex'], STAGE)
    fig_s8_k_by_channel(P['kt'], AGE, STAGE)
    fig_s9_counts_and_k(d, P['s9_night'], STAGE)
    fig_s10_allsessions(d, AGE, STAGE)
    fig_s11_calibration(P['per_session'], STAGE)
    fig_s12_k_per_epoch(P['spread'], AGE, STAGE)

    register_numbers(P).save()


# ─────────────────────────────────────────────────────────────────── numbers

def _rng(v, dp=2, sep='–'):
    return f'{np.min(v):.{dp}f}{sep}{np.max(v):.{dp}f}'


def approx(computed, stated, rel=0.10):
    """Status for a number the text gives only approximately ("near", "about")."""
    return 'MATCH' if abs(computed - stated) <= rel * abs(stated) else 'DIFF'


def register_numbers(P) -> Numbers:
    """Every rate number V11 and the supplement quote, plus the ones the corrections add."""
    d, ps, kt, refc, ppb, spread, ex, s9_night, s9 = (
        P[k] for k in ('d', 'per_session', 'kt', 'refc', 'ppb', 'spread', 'ex', 's9_night', 's9'))
    nb = Numbers(STAGE)
    R, C = ps[ps.band == 'resp'], ps[ps.band == 'card']
    med = lambda s: float(np.median(s))                       # noqa: E731
    ev = 'evaluate'

    # ── main text ──
    for sec in ('Discussion ¶253', 'Conclusion ¶259'):
        tag = sec.split()[0][:4].lower()
        nb.add(f'night_err_resp_{tag}', sec, 'night-level |error|, respiratory, self-k, median of 12',
               med(R.night_self), '0.24 breaths/min', 'br/min', ev)
        nb.add(f'night_err_card_{tag}', sec, 'night-level |error|, cardiac, self-k, median of 12',
               med(C.night_self), '1.56 beats/min', 'BPM', ev)
    nb.add('epoch_err_resp', 'Discussion ¶253', 'epoch-level median |error|, respiratory, self-k',
           med(R.epoch_self), '1.79 breaths/min', 'br/min', ev)
    nb.add('epoch_err_card', 'Discussion ¶253', 'epoch-level median |error|, cardiac, self-k',
           med(C.epoch_self), '3.41 beats/min', 'BPM', ev)
    nb.add('k_card', 'Discussion ¶253', 'cardiac k, median (IQR) of 12, CRE',
           f'{med(C.k):.2f} ({np.percentile(C.k, 25):.2f}–{np.percentile(C.k, 75):.2f})',
           '1.96; interquartile range, 1.77–2.01', '', 'fit_k',
           note='k = median of estimate/reference with ratios outside 0.3–5.0 dropped')
    nb.add('k_resp', 'Discussion ¶253', 'respiratory k, median of 12, CRE', med(R.k),
           '1.18', '', 'fit_k')
    good = ppb[~ppb.detector_failed]
    nb.add('peaks_per_beat', 'Discussion ¶253', 'cardiac-band CRE peaks per ECG beat, asleep, median',
           float(good.peaks_per_beat.median()), '2.02 peaks per cardiac cycle', 'peaks/beat',
           'peaks_per_beat',
           note=(f'CORRECTION: a peak COUNT ratio (CRE peaks ÷ ECG R-peaks over the asleep '
                 f'night), not R-peak-triggered averaging as ¶253 says. n = {len(good)} '
                 f'nights, range {_rng(good.peaks_per_beat)}; '
                 f'{", ".join(ppb[ppb.detector_failed].session)} excluded (< {PPB_FAIL} '
                 f'peaks/beat: detector failure from a glitch-inflated σ); '
                 f'{12 - len(ppb)} nights without usable ECG.'))
    nb.add('resp_ref_range', 'Discussion ¶254', 'range of nightly mean respiratory reference, 12 nights',
           _rng(R.ref_mean), '14.4–16.8 breaths/min', 'br/min', ev,
           note='CORRECTION: 14.4–16.8 is from the superseded pipeline; this is the rerun '
                'reference (consensus of Flow/Thorax/Abdomen/RIP). ¶254 cites "Table 3", '
                'which does not exist in V11.')
    nb.add('nosensor_resp', 'Discussion ¶254', 'constant (no-sensor) predictor, night-level, respiratory',
           med(R.night_nosensor), '1.20 breaths/min', 'br/min', ev,
           note='predicts the leave-one-subject-out cohort median rate. "(Table 3)" does not exist in V11.')

    # ── supplement: pipeline description ──
    nb.add('prominence', 'Supp ¶222', 'loose detector prominence', PROM_LOOSE, '0.05σ', 'σ',
           'rate_peaks')
    nb.add('min_spacing', 'Supp ¶222', 'loose detector minimum peak spacing',
           f'{int(0.9 * 100 / RESP_HI) / 100:.1f} s (resp), {int(0.9 * 100 / CARD_HI) / 100:.1f} s (card)',
           '0.4 s', 's', 'seclib.rates.rate_peaks',
           note='the operational detector spaces peaks by 0.9/f_hi (1.8 s resp, 0.3 s card) '
                'after a moving average; 0.4 s and no smoothing is only the Fig. S7 '
                'illustration detector. The text should state the operational values.')

    # ── Fig. S7 worked example ──
    nb.add('s7_peaks', 'Supp ¶224 Fig. S7', 'peaks in the 60-s S2N1 example', ex['n'],
           '20 peaks', '', 'worked_example_segment')
    nb.add('s7_raw_rate', 'Supp ¶224 Fig. S7', 'raw count rate in the example', ex['raw_rate'],
           '19.6 per minute', 'per min', 'worked_example_segment')
    nb.add('s7_k_cohort', 'Supp ¶224 Fig. S7', 'k used by the legacy figure (cohort median)',
           ex['k_cohort'], 'k = 1.18', '', 'fit_k')
    nb.add('s7_rate_cohort', 'Supp ¶224 Fig. S7', 'example rate ÷ cohort k', ex['rate_cohort'],
           '16.6 breaths per minute', 'br/min', 'worked_example_segment',
           note='CORRECTION (1): the text says k is one number per recording but the '
                'figure divided by the cohort median; the figure now uses S2N1\'s own k.')
    nb.add('s7_k_own', 'Supp ¶224 Fig. S7', 'S2N1 own CRE respiratory k', ex['k_own'], None, '',
           'fit_k', note='CORRECTION (1): the k the figure now divides by')
    nb.add('s7_rate_own', 'Supp ¶224 Fig. S7', 'example rate ÷ S2N1 own k', ex['rate_own'], None,
           'br/min', 'worked_example_segment',
           note='what ¶224 should say if the figure keeps one k per recording')
    nb.add('s7_operational', 'Supp ¶224 Fig. S7', 'operational detector (rate_peaks, motion-cancelled) raw rate, same minute',
           ex['operational_raw'], None, 'per min', 'epoch_rates',
           note='mean of the two 30-s epochs; for comparison with the illustration detector')
    nb.add('s7_ref', 'Supp ¶224 Fig. S7', 'PSG reference over the example minute', ex['ref'], None,
           'br/min', 'epoch_rates', note='mean of the two 30-s epochs covering the minute')

    # ── Fig. S8 k by channel ──
    def kmed(band, meth):
        s = kt[(kt.band == band) & (kt.method == meth)]
        return ', '.join(f'{s[s.channel == c].k.median():.2f}' for c in CHANNELS)
    nb.add('s8_k_resp_peaks', 'Supp ¶225', 'respiratory k, peak counting, CH, CLE, CRE',
           kmed('resp', 'peaks_loose'), '1.04 on CH, 1.14 on CLE and 1.18 on CRE', '', 'k_table',
           note='ratios clipped to 0.3–5.0, as the text says')
    nb.add('s8_k_card_peaks', 'Supp ¶225', 'cardiac k, peak counting, CH, CLE, CRE',
           kmed('card', 'peaks_loose'), '1.93, 1.95 and 1.96', '', 'k_table')
    # CORRECTION (5): the degenerate spectral respiratory k, with the degeneracy computed
    rs = d[d.band == 'resp']
    frac15 = float(np.mean(np.isclose(rs.spectral.dropna(), 0.25)))
    ksp = kmed('resp', 'spectral')
    nb.add('s8_k_resp_spectral', 'Supp ¶228', 'respiratory k, spectral (identical on CH, CLE, CRE?)',
           ksp.split(', ')[0] if len(set(ksp.split(', '))) == 1 else ksp, '0.96', '', 'k_table',
           note=f'paper: "identical on all three channels, 0.96". rate_spectral returns '
                f'0.25 Hz (15 br/min) in {100 * frac15:.2f}% of respiratory epochs, so k = '
                f'15/median reference.')
    nb.add('s8_spectral_resp_const', 'Supp ¶228', 'share of respiratory epochs where rate_spectral = 15 br/min',
           100 * frac15, None, '%', 'rate_spectral',
           note='"returns nearly the same value in every epoch"')
    nb.add('s8_k_card_spectral', 'Supp ¶228', 'cardiac k, spectral, CH, CLE, CRE ("lands near 1")',
           kmed('card', 'spectral'), None, '', 'k_table')
    iqr = lambda s: s.quantile(.75) - s.quantile(.25)       # noqa: E731
    ratios = []
    for c in CHANNELS:
        sp = kt[(kt.band == 'card') & (kt.method == 'spectral') & (kt.channel == c)].k
        pk = kt[(kt.band == 'card') & (kt.method == 'peaks_loose') & (kt.channel == c)].k
        ratios.append(iqr(sp) / iqr(pk))
    nb.add('s8_spread_ratio', 'Supp ¶228', 'cardiac k IQR, spectral ÷ peak counting, CH, CLE, CRE',
           ', '.join(f'{r:.1f}' for r in ratios), 'two to three times', '×', 'k_table',
           status='MATCH' if all(1.95 <= r <= 3.05 for r in ratios) else 'DIFF',
           note='paper value is in words; status set by whether every ratio is in 2–3')

    # ── cardiac reference vs ECG (¶229) ──
    # The legacy check compared the Pleth-fallback reference with ECG; the stage's own
    # reference is ECG-derived wherever the ECG is usable (CORRECTION 6), so both are kept.
    rc = refc.set_index('session')
    pver = refc[refc.pleth_verified]
    nb.add('ref_verified_n', 'Supp ¶229', 'recordings whose (Pleth-fallback) cardiac reference agrees with ECG, 0.90–1.15',
           len(pver), 'eight of the twelve', 'recordings', 'ref_check',
           status='MATCH' if len(pver) == 8 else 'DIFF',
           note=f'paper value in words. With the ECG-first reference '
                f'{int(refc.verified.sum())} of 12 agree; only '
                f'{", ".join(refc[refc.card_ref_source == "Pleth"].session)} still use Pleth.')
    for s_, pv in [('S2N1', '36%'), ('S6N1', '29%')]:
        nb.add(f'ref_high_{s_}', 'Supp ¶229', f'{s_} Pleth-fallback reference above the ECG rate',
               100 * (float(rc.loc[s_, 'pleth_ratio']) - 1), pv, '%', 'ref_check',
               note=f'CORRECTION (6): the excess is the Pleth detector over-counting; the '
                    f'stage reference on {s_} is ECG ({rc.loc[s_, "ref_bpm"]:.1f} vs ECG '
                    f'{rc.loc[s_, "ecg_bpm"]:.1f} BPM), so this caveat no longer applies')
    unusable = refc[refc.note != ''].session.tolist()
    nb.add('ref_unusable', 'Supp ¶229', 'recordings with unusable ECG', ', '.join(unusable),
           'S5N1 or S6N2', '', 'ref_check',
           status='MATCH' if unusable == ['S5N1', 'S6N2'] else 'DIFF')
    s6 = rc.loc['S6N2']
    nb.add('ref_S6N2', 'Supp ¶229', 'S6N2 cardiac reference, median during sleep',
           float(s6.ref_bpm_asleep), '129 beats/min', 'BPM', 'ref_check',
           note=f'whole-night median {s6.ref_bpm:.1f} (the legacy figure was the whole-night '
                f'median); Pleth fallback, ECG unusable')
    # CORRECTION (4): computed, not typed in
    kcre = kt[(kt.band == 'card') & (kt.method == 'peaks_loose') & (kt.channel == 'CRE')]
    kv = kcre[kcre.session.isin(pver.session)].k
    kv_all = kcre[kcre.session.isin(refc[refc.verified].session)].k
    nb.add('k_card_verified', 'Supp ¶229', 'cardiac k (CRE), median and range, over the eight recordings the legacy check verified',
           f'{kv.median():.2f}, range {_rng(kv)}', '1.98, range 1.81–2.28', '', 'k_table',
           note=f'CORRECTION (4): was hard-coded in the supplement build. Computed with the '
                f'ECG-first reference; with the legacy references it is exactly 1.98, '
                f'1.81–2.28 (checks/s04_rates_vs_legacy). Over all {len(kv_all)} '
                f'ECG-referenced nights: {kv_all.median():.2f}, {_rng(kv_all)}.')
    low3 = kcre.nsmallest(3, 'k').session.tolist()
    nb.add('k_card_low3', 'Supp ¶229', 'three lowest cardiac k (CRE)', ', '.join(sorted(low3)),
           None, '', 'k_table', note='text: "exactly the three that sit below the rest" '
           '(S2N1, S6N1, S6N2) -- true only under the Pleth-fallback reference')

    # ── Fig. S9 ──
    nb.add('s9_night', 'Supp ¶232 Fig. S9', 'night drawn in Figs. S9', s9_night, None, '',
           'counts_night', note='most CRE epochs with an estimate')
    nb.add('s9_card_count', 'Supp ¶233', 'cardiac raw count, S9 night, median of the smoothed traces (CH, CLE, CRE)',
           s9['card_count'], 'near 120 per minute', 'per min', 'fig_s9_counts_and_k',
           status=approx(s9['card_count'], 120),
           note='paper value is approximate ("near"): MATCH within 10%')
    lo, hi = s9['card_ref']
    nb.add('s9_card_ref', 'Supp ¶233', 'cardiac reference, S9 night, 5th–95th percentile of the smoothed trace',
           f'{lo:.1f}–{hi:.1f}', 'between about 55 and 80', 'BPM', 'fig_s9_counts_and_k',
           status='MATCH' if approx(lo, 55) == approx(hi, 80) == 'MATCH' else 'DIFF',
           note='paper value is approximate ("about"): MATCH within 10%')
    nb.add('s9_k_channel_r', 'Supp ¶233', 'per-epoch k, S9 night: lowest pairwise r between channels',
           s9['k_r'], None, '', 'fig_s9_counts_and_k',
           note='"the three channels move together almost exactly"')

    # ── within-night tracking (¶234: "shown rather than tabulated") ──
    for b, s in (('resp', R), ('card', C)):
        nb.add(f'r_within_{b}', 'Supp ¶234', f'within-night r, estimate vs reference, {b}, median',
               float(s.r_within.median()), None, '', ev,
               note=f'{int((s.r_within > 0).sum())}/12 nights positive')

    # ── Fig. S11 ──
    pv = {'resp': ('0.24', '0.57', '0.94', '1.20'), 'card': (None, '3.77', '3.19', '2.76')}
    for b, s in (('resp', R), ('card', C)):
        for col, p in zip(('self', 'cross', 'pop', 'nosensor'), pv[b]):
            if p is None:
                continue
            nb.add(f's11_{b}_{col}', 'Supp ¶240 Fig. S11', f'night-level error, {b}, {col} k',
                   med(s[f'night_{col}']), p, UNIT[b], ev)
    worse = {b: {c: med(s[f'epoch_{c}']) >= med(s.epoch_nosensor)
                 for c in ('self', 'cross', 'pop')} for b, s in (('resp', R), ('card', C))}
    nb.add('s11_epoch_baseline', 'Supp ¶240', 'epoch-level error vs no-sensor baseline (self/cross/pop vs no sensor)',
           '; '.join(f'{b} ' + '/'.join(f'{med(s["epoch_" + c]):.2f}' for c in ('self', 'cross', 'pop'))
                     + f' vs {med(s.epoch_nosensor):.2f}' for b, s in (('resp', R), ('card', C))),
           'Epoch by epoch neither band beats the baseline', UNIT['resp'] + ' / ' + UNIT['card'], ev,
           status='MATCH' if all(all(v.values()) for v in worse.values()) else 'DIFF',
           note='cardiac self-k beats the baseline epoch by epoch; the statement holds only '
                'for the two transferable calibrations' if not worse['card']['self'] else '')

    # ── Fig. S12 ──
    for b, pv_ in (('resp', '0.18–0.20'), ('card', '0.19–0.24')):
        m = spread[spread.band == b].groupby('channel').iqr.median()
        nb.add(f's12_iqr_{b}', 'Supp ¶241', f'per-epoch k IQR width, {b}: range over CH/CLE/CRE of the per-night median',
               _rng(m.values), pv_, '', 'k_per_epoch_spread',
               note=', '.join(f'{c} {m[c]:.2f}' for c in CHANNELS))

    # ── ¶244 ages ──
    kc = kt[(kt.band == 'card') & (kt.method == 'peaks_loose') & (kt.channel == 'CRE')].copy()
    kc['age'] = kc.subject.map(AGE)
    far = kc.assign(dev=(kc.k - 2).abs()).nlargest(2, 'dev')
    nb.add('s_age_far', 'Supp ¶244', 'two recordings with cardiac k furthest from 2 (age)',
           ', '.join(f'{r.session} ({r.age} y)' for r in far.itertuples()), None, '', 'k_table',
           note='text: they belong to the youngest (25 y) and the oldest (66 y) participant')

    # The two reference corrections move every reference-dependent number; say so on
    # each one that no longer matches. Under the legacy references the port reproduces
    # the legacy values (paper/checks/s04_rates_vs_legacy.out.txt, pass A).
    ref_note = {
        'card': 'CORRECTION (6): cardiac reference = ECG R-peaks (Pleth only on S5N1, S6N2); '
                'the legacy rerun used the Pleth fallback on all 12 nights.',
        'resp': 'Respiratory reference = s03_signal consensus, which differs from the legacy '
                'artifact (its Flow-rate column differs in most epochs).'}
    card_ids = {'s8_spread_ratio', 's11_epoch_baseline', 's_age_far', 'k_card_verified'}
    for r in nb.rows:
        if r['status'] != 'DIFF':
            continue
        band = 'card' if ('card' in r['id'] or r['id'] in card_ids) else \
               'resp' if ('resp' in r['id'] or r['id'].startswith('s7_')) else None
        if band:
            r['note'] = (r['note'] + ' ' if r['note'] else '') + ref_note[band] + \
                ' Legacy-reference value: see checks/s04_rates_vs_legacy.out.txt.'
    return nb


if __name__ == '__main__':
    run()
