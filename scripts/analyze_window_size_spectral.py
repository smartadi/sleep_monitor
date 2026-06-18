#!/usr/bin/env python
"""
Window-size / spectral-resolution analysis for the resp 'spectral wins' result.

MOTIVATION: cache showed r_spectral(resp) = 0.25 Hz in 9317/9319 epochs — a
CONSTANT predictor (corr with GT = 0.000), an artifact of rate_spectral using
nperseg = max(64, fs*4) = 400 samples -> df = 0.25 Hz, which collapses the whole
resp band (0.1-0.5 Hz) to one bin. So 'spectral wins resp' is a resolution
artifact: it just predicts the population-mean rate. MAE is misleading; we must
also report tracking (correlation with GT).

This script reprocesses (resp + card, diff channel) with a HIGH-RESOLUTION
spectral estimator (full-window periodogram + parabolic interpolation) across
window sizes, comparing spectral vs peaks on BOTH MAE and correlation.

Filtering is done ONCE per session/band (the expensive step); window sweep reuses
the filtered signal. Outputs:
  reports/rates/mask/window_size_sweep.csv
  writeup/figures/mask_rate_detection/fig10_window_size_resp.png
  writeup/figures/mask_rate_detection/fig11_window_size_card.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader import load_all_sessions
from sleep_monitor.ground_truth import gt_sliding_rates

RPT = ROOT / 'reports' / 'rates' / 'mask'
FIG = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'
RPT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BANDS = {'resp': (RESP_LO, RESP_HI), 'card': (CARD_LO, CARD_HI)}
WINDOWS = [15, 20, 30, 45, 60, 90, 120]  # seconds
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})


def spectral_lowres(x, flo, fhi, fs):
    """Current estimator: nperseg=400 (4 s) -> df=0.25 Hz."""
    nperseg = min(len(x), max(64, int(fs * 4)))
    if len(x) < nperseg:
        return np.nan
    f, p = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    m = (f >= flo) & (f <= fhi)
    if not m.any():
        return np.nan
    return float(f[m][np.argmax(p[m])])


def spectral_hires(x, flo, fhi, fs):
    """Full-window periodogram (df=fs/n) + parabolic interpolation around peak."""
    n = len(x)
    if n < 64:
        return np.nan
    # 2 averaged segments for mild variance reduction while keeping resolution high
    nperseg = n
    f, p = welch(x, fs=fs, nperseg=nperseg, noverlap=0)
    m = (f >= flo) & (f <= fhi)
    if not m.any():
        return np.nan
    band_idx = np.where(m)[0]
    local = band_idx[np.argmax(p[band_idx])]
    if 0 < local < len(p) - 1:
        a, b, c = p[local - 1], p[local], p[local + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        df = f[1] - f[0]
        return float(f[local] + delta * df)
    return float(f[local])


def peaks_loose(x, flo, fhi, fs):
    min_dist = max(1, int(0.9 * fs / fhi))
    sw = max(3, min_dist // 4)
    xs = np.convolve(x.astype(np.float64), np.ones(sw) / sw, mode='same')
    pks, _ = find_peaks(xs, distance=min_dist, prominence=0.05 * np.std(xs))
    if len(pks) < 2:
        return np.nan
    dur = (pks[-1] - pks[0]) / fs
    return (len(pks) - 1) / dur if dur > 0 else np.nan


def per_session_k(raw, gt):
    v = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
    if v.sum() < 20:
        return 1.0
    r = raw[v] / gt[v]
    r = r[(r > 0.3) & (r < 5.0)]
    return float(np.median(r)) if len(r) >= 10 else 1.0


def evaluate(pred, gt):
    v = np.isfinite(pred) & np.isfinite(gt)
    if v.sum() < 20:
        return np.nan, np.nan
    mae = float(np.median(np.abs(pred[v] - gt[v])) * 60)
    if np.std(pred[v]) < 1e-9 or np.std(gt[v]) < 1e-9:
        r = 0.0
    else:
        r = float(np.corrcoef(pred[v], gt[v])[0, 1])
    return mae, r


def main():
    sessions = load_all_sessions(with_sleep_profiles=False)
    print(f'Loaded {len(sessions)} sessions')

    rows = []
    for band, (flo, fhi) in BANDS.items():
        gtkey = 'resp_hz' if band == 'resp' else 'card_hz'
        # Pre-filter diff channel once per session for this band
        filtered = {}
        for sess in sessions:
            fs = sess.fs
            cle = sess.cap['CLE'].astype(np.float64)
            cre = sess.cap['CRE'].astype(np.float64)
            acc = sess.cap['acc_mag'].astype(np.float64)
            diff = cle - cre
            filtered[sess.label] = remove_acc_artifact(diff, acc, flo, fhi, fs)
        print(f'  {band}: filtered diff channel for {len(sessions)} sessions')

        for W in WINDOWS:
            # accumulate across sessions, applying per-session k
            agg = {m: ([], []) for m in ['lowres', 'hires', 'peaks']}
            for sess in sessions:
                fs = sess.fs
                win_n = int(W * fs)
                sig = filtered[sess.label]
                n = len(sig)
                starts = np.arange(0, n - win_n + 1, win_n)
                centres_hr = (starts + win_n / 2.0) / fs / 3600.0
                gt_data = gt_sliding_rates(sess, win_sec=float(W), step_sec=5.0)
                gt_t = gt_data['t_hr']
                gt_r_full = gt_data[gtkey]
                # match GT to epoch centres
                gt = np.full(len(starts), np.nan)
                for i, t in enumerate(centres_hr):
                    j = np.argmin(np.abs(gt_t - t))
                    if abs(gt_t[j] - t) < 0.02:
                        gt[i] = gt_r_full[j]

                est = {m: np.full(len(starts), np.nan) for m in ['lowres', 'hires', 'peaks']}
                for i, s0 in enumerate(starts):
                    seg = sig[s0:s0 + win_n]
                    est['lowres'][i] = spectral_lowres(seg, flo, fhi, fs)
                    est['hires'][i] = spectral_hires(seg, flo, fhi, fs)
                    est['peaks'][i] = peaks_loose(seg, flo, fhi, fs)

                for m in est:
                    k = per_session_k(est[m], gt)
                    agg[m][0].append(est[m] / k)
                    agg[m][1].append(gt)

            for m in agg:
                pred = np.concatenate(agg[m][0])
                gt = np.concatenate(agg[m][1])
                mae, r_pooled = evaluate(pred, gt)
                # WITHIN-session correlation (honest tracking metric): per-session
                # Pearson r, averaged. Avoids between-session mean-matching artifact
                # injected by per-session k-normalisation.
                within = []
                for ps, gs in zip(agg[m][0], agg[m][1]):
                    v = np.isfinite(ps) & np.isfinite(gs)
                    if v.sum() >= 20 and np.std(ps[v]) > 1e-9 and np.std(gs[v]) > 1e-9:
                        within.append(np.corrcoef(ps[v], gs[v])[0, 1])
                r_within = float(np.mean(within)) if within else np.nan
                rows.append({'band': band, 'window_s': W, 'method': m,
                             'MAE': mae, 'corr_pooled': r_pooled,
                             'corr_within': r_within})
            print(f'    W={W:3d}s done')

    res = pd.DataFrame(rows)
    res.to_csv(RPT / 'window_size_sweep.csv', index=False)
    print('\nResults:')
    for band in ['resp', 'card']:
        unit = 'br/min' if band == 'resp' else 'BPM'
        print(f'\n  {band.upper()} ({unit}):')
        piv_mae = res[res.band == band].pivot(index='window_s', columns='method', values='MAE')
        piv_pool = res[res.band == band].pivot(index='window_s', columns='method', values='corr_pooled')
        piv_within = res[res.band == band].pivot(index='window_s', columns='method', values='corr_within')
        print('  MAE:')
        print(piv_mae.round(2).to_string())
        print('  Correlation POOLED (inflated by between-session mean-matching):')
        print(piv_pool.round(3).to_string())
        print('  Correlation WITHIN-session (honest tracking ability):')
        print(piv_within.round(3).to_string())

    # Figures
    for band in ['resp', 'card']:
        unit = 'br/min' if band == 'resp' else 'BPM'
        sub = res[res.band == band]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        colors = {'lowres': '#E74C3C', 'hires': '#2ECC71', 'peaks': '#F39C12'}
        names = {'lowres': 'spectral (current, df=0.25Hz)',
                 'hires': 'spectral hi-res (full window)',
                 'peaks': 'peaks (loose)'}
        for m in ['lowres', 'hires', 'peaks']:
            d = sub[sub.method == m].sort_values('window_s')
            ax1.plot(d.window_s, d.MAE, 'o-', color=colors[m], label=names[m])
            ax2.plot(d.window_s, d['corr_within'], 'o-', color=colors[m], label=names[m])
        ax1.set_xlabel('Window size (s)'); ax1.set_ylabel(f'Median MAE ({unit})')
        ax1.set_title('Accuracy (MAE) — lower better'); ax1.legend(fontsize=8)
        ax2.set_xlabel('Window size (s)'); ax2.set_ylabel('Within-session Pearson r with GT')
        ax2.set_title('Tracking (within-session corr) — higher better'); ax2.legend(fontsize=8)
        ax2.axhline(0, color='gray', lw=0.5, ls=':')
        fig.suptitle(f'{"Respiratory" if band=="resp" else "Cardiac"}: window size vs '
                     f'spectral resolution\n(MAE alone is misleading — low-res spectral '
                     f'is a near-constant predictor)',
                     fontsize=11, fontweight='bold', y=1.04)
        plt.tight_layout()
        n = 10 if band == 'resp' else 11
        fig.savefig(FIG / f'fig{n}_window_size_{band}.png')
        plt.close(fig)
        print(f'  Fig {n} ({band}) saved')

    print('\nDONE.')


if __name__ == '__main__':
    main()
