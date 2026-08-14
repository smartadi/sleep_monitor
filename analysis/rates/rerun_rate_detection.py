#!/usr/bin/env python
"""
Rate detection, recomputed (2026-08-14).

Two things were wrong with the pipeline behind the manuscript numbers, both
found in code review:

  1. `rate_spectral` fixes the Welch segment at 4 s, so Δf = 0.25 Hz at
     fs = 100. The 0.1–0.5 Hz respiratory band holds two usable bins and the
     estimator returned 15.0 br/min in 99.98% of epochs — a constant. In the
     cardiac band the same Δf quantizes the output to 15 BPM steps.
  2. The "non-degenerate" fused pipeline adopted as the fix does not fix it.
     For respiration `fused_agree` is a smart fusion of that constant with
     peaks_loose, and its pooled SD is 0.46 br/min against a reference SD of
     ~1.6 — it varies, but far less than the physiology it is meant to track.

This recomputes every per-epoch estimate with `rate_spectral_interp` (full-window
periodogram, zero-padded, parabolically interpolated) alongside the original
estimator, so old and new are compared on identical windows, and then scores
every estimator under **held-out calibration** with a no-sensor comparator —
the test the original analysis never ran.

  self-k        k fitted on the night being scored (what the paper reported)
  cross-night   k fitted on the subject's other night          [held out]
  population    one k for the cohort, leave-one-subject-out    [held out]
  no-sensor     predict the cohort median rate every epoch     [no sensor input]

Outputs
  artifacts/rate_rerun_phase_a.parquet     per-epoch estimates, all channels
  reports/rates/rerun/estimator_table.csv  accuracy by estimator x channel
  reports/rates/rerun/heldout_table.csv    the four calibration regimes
  reports/rates/rerun/per_session.csv      per-night values for the operational choice

Run from the repo root:  python analysis/rates/rerun_rate_detection.py [--sessions N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import RESP_LO, RESP_HI, CARD_LO, CARD_HI      # noqa: E402
from sleep_monitor.ground_truth import gt_sliding_rates                   # noqa: E402
from sleep_monitor.loader import load_all_sessions, load_sleep_profile    # noqa: E402
from sleep_monitor.preprocessing import remove_acc_artifact               # noqa: E402
from sleep_monitor.quality import combined_quality, window_features       # noqa: E402
from sleep_monitor.rates import (rate_hilbert, rate_peaks, rate_spectral, # noqa: E402
                                 rate_spectral_interp)

ART = ROOT / 'artifacts' / 'rate_rerun_phase_a.parquet'
RPT = ROOT / 'reports' / 'rates' / 'rerun'
RPT.mkdir(parents=True, exist_ok=True)

BANDS = {'resp': (RESP_LO, RESP_HI), 'card': (CARD_LO, CARD_HI)}
CHANNELS = ['CLE', 'CRE', 'CH', 'avg', 'diff']
WIN_SEC = 30.0
STAGE_MAP = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
UNIT = {'resp': 'br/min', 'card': 'BPM'}
ESTIMATORS = ['spectral', 'spectral_interp', 'peaks_loose', 'peaks_strict', 'hilbert']
K_LO, K_HI = 0.3, 5.0


# ────────────────────────────────────────────────────────────── per-epoch pass

def prepare_channels(sess):
    cle = sess.cap['CLE'].astype(np.float64)
    cre = sess.cap['CRE'].astype(np.float64)
    return {'CLE': cle, 'CRE': cre, 'CH': sess.cap['CH'].astype(np.float64),
            'avg': (cle + cre) / 2.0, 'diff': cle - cre,
            'acc': sess.cap['acc_mag'].astype(np.float64)}


def stages_for(sess, centres_hr):
    if sess.sleep_profile is None:
        sess.sleep_profile = load_sleep_profile(sess)
    if sess.sleep_profile is None:
        return np.array(['?'] * len(centres_hr))
    codes, ep = sess.sleep_profile['codes'], 30.0 / 3600.0
    out = []
    for t in centres_hr:
        i = int(t / ep)
        out.append(STAGE_MAP.get(int(codes[i]), '?') if 0 <= i < len(codes) else '?')
    return np.array(out)


def phase_a(sessions) -> pd.DataFrame:
    rows = []
    for sess in sessions:
        t0, fs = time.time(), sess.fs
        win_n = int(WIN_SEC * fs)
        chans = prepare_channels(sess)
        acc = chans['acc']
        starts = np.arange(0, sess.n_samples - win_n + 1, win_n)
        centres = (starts + win_n / 2.0) / fs / 3600.0
        stages = stages_for(sess, centres)

        gt_data = gt_sliding_rates(sess, win_sec=30.0, step_sec=5.0)
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

        for ei, s0 in enumerate(starts):
            s1 = s0 + win_n
            acc_win = acc[s0:s1]
            for band, (lo, hi) in BANDS.items():
                for ch in CHANNELS:
                    sig = bp[(ch, band)][s0:s1]
                    r_sp = rate_spectral(sig, lo, hi, fs)
                    r_si = rate_spectral_interp(sig, lo, hi, fs)
                    r_pl = rate_peaks(sig, lo, hi, fs, prom_factor=0.05)
                    r_ps = rate_peaks(sig, lo, hi, fs, prom_factor=0.4)
                    r_hi = rate_hilbert(sig, lo, hi, fs)
                    qf = window_features(sig, acc_win, lo, hi, fs,
                                         {'spectral': r_si, 'peaks': r_pl, 'hilbert': r_hi})
                    rows.append(dict(
                        session=sess.label, epoch=ei, t_hr=centres[ei],
                        stage=stages[ei], band=band, channel=ch, gt_hz=gt[band][ei],
                        spectral=r_sp, spectral_interp=r_si, peaks_loose=r_pl,
                        peaks_strict=r_ps, hilbert=r_hi,
                        quality=combined_quality(qf)))
        print(f'  {sess.label}: {len(starts)} epochs in {time.time() - t0:.1f}s', flush=True)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────── evaluation

def fit_k(raw, gt):
    r = raw / gt
    r = r[(r > K_LO) & (r < K_HI) & np.isfinite(r)]
    return float(np.median(r)) if len(r) >= 10 else np.nan


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
    """Per-session accuracy under four calibration regimes."""
    D = pairs(df, band, channel, est)
    if len(D) < 2:
        return pd.DataFrame()
    ks = {s: fit_k(r, g) for s, (r, g) in D.items()}
    rows = []
    for s, (raw, gt) in D.items():
        subj = s[:2]
        other = subj + ('N2' if s.endswith('N1') else 'N1')
        # leave-one-SUBJECT-out population k and reference, so nothing from this
        # subject informs the held-out predictors
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, default=None,
                    help='limit to the first N sessions (smoke test)')
    ap.add_argument('--force', action='store_true', help='recompute phase A')
    args = ap.parse_args()

    if ART.exists() and not args.force:
        print(f'loading {ART}')
        df = pd.read_parquet(ART)
    else:
        idx = list(range(args.sessions)) if args.sessions else None
        print('loading sessions...')
        sessions = load_all_sessions(indices=idx)
        print('phase A — per-epoch estimates')
        df = phase_a(sessions)
        df.to_parquet(ART, index=False)
        print(f'wrote {ART}  ({len(df):,} rows)')

    # ---- estimator x channel comparison, self-k (the optimistic regime) -------
    rows = []
    for band in BANDS:
        for est in ESTIMATORS:
            for ch in CHANNELS:
                d = evaluate(df, band, ch, est)
                if d.empty:
                    continue
                rows.append(dict(
                    band=band, estimator=est, channel=ch,
                    epoch_self=np.median(d.epoch_self), night_self=np.median(d.night_self),
                    epoch_pop=np.median(d.epoch_pop), night_cross=np.median(d.night_cross),
                    r_within=np.median(d.r_within), raw_sd=np.median(d.raw_sd),
                    ref_sd=np.median(d.ref_sd)))
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise SystemExit('no session pairs to evaluate — the held-out regimes need '
                         'at least two sessions; rerun without --sessions')
    tab.to_csv(RPT / 'estimator_table.csv', index=False)
    for band in BANDS:
        print(f'\n=== {band} ({UNIT[band]}) — estimator x channel, self-k ===')
        t = tab[tab.band == band].sort_values('epoch_self')
        print(t.round(3).to_string(index=False))

    # ---- held-out calibration for the best non-degenerate choice per band ----
    best, held = {}, []
    for band in BANDS:
        t = tab[(tab.band == band) & (tab.estimator != 'spectral')]
        # require the estimator to actually vary: raw SD at least half the
        # reference SD, so a near-constant cannot win on error alone
        ok = t[t.raw_sd >= 0.5 * t.ref_sd]
        pick = (ok if len(ok) else t).sort_values('epoch_self').iloc[0]
        best[band] = (pick.estimator, pick.channel)
        d = evaluate(df, band, pick.channel, pick.estimator)
        d.insert(0, 'band', band)
        held.append(d)
        print(f'\n=== {band}: {pick.estimator} on {pick.channel} — calibration regimes ===')
        for lab, col in [('self-k (same night)', 'self'), ('cross-night k', 'cross'),
                         ('population k (LOSO)', 'pop'), ('no sensor', 'nosensor')]:
            print(f'  {lab:24s} night {med_iqr(d["night_" + col]):>22s}'
                  f'   epoch {med_iqr(d["epoch_" + col]):>22s}')
        print(f'  within-night r  median {d.r_within.median():+.3f}'
              f'   positive in {(d.r_within > 0).sum()}/{len(d)} nights')
        print(f'  raw SD {d.raw_sd.median():.2f} vs reference SD {d.ref_sd.median():.2f}')

    out = pd.concat(held, ignore_index=True)
    out.to_csv(RPT / 'per_session.csv', index=False)
    pd.DataFrame([{'band': b, 'estimator': e, 'channel': c}
                  for b, (e, c) in best.items()]).to_csv(RPT / 'operational_choice.csv', index=False)

    summary = []
    for band in BANDS:
        d = out[out.band == band]
        row = {'band': band, 'unit': UNIT[band],
               'estimator': best[band][0], 'channel': best[band][1]}
        for col in ['self', 'cross', 'pop', 'nosensor']:
            row[f'night_{col}'] = med_iqr(d[f'night_{col}'])
            row[f'epoch_{col}'] = med_iqr(d[f'epoch_{col}'])
        row['k_median'] = med_iqr(d.k)
        row['r_within'] = f'{d.r_within.median():+.3f}'
        row['nights_r_pos'] = f'{int((d.r_within > 0).sum())}/{len(d)}'
        summary.append(row)
    pd.DataFrame(summary).to_csv(RPT / 'heldout_table.csv', index=False)
    print(f'\nwrote {RPT}/estimator_table.csv, heldout_table.csv, per_session.csv')


if __name__ == '__main__':
    main()
