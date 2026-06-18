#!/usr/bin/env python
"""
Last-chance cardiac tracking test: CWT ridge + continuous Viterbi ridge tracker.

The windowed estimators (peaks/hilbert/spectral) all fail to track within-session
HR (r ~ 0; see diagnose_cardiac_tracking.py). This tests the two methods designed
for tracking:
  (A) rate_cwt per 30s epoch (within-window ridge) — sleep_monitor.rates_classical
  (B) a CONTINUOUS night-long Viterbi ridge over the cardiac STFT with a
      frequency-continuity penalty (the proper temporal tracker).

For each: align to GT on the 30s epoch grid, per-session k, and report
WITHIN-session Pearson r vs SMOOTHED GT (the honest tracking metric). Compares to
the peaks baseline. If both ~0, within-session cardiac tracking is genuinely not
recoverable and the paper reframes to mean-rate + signal-limited tracking.

Output: reports/rates/mask/cwt_ridge_tracking.csv
        writeup/figures/mask_rate_detection/fig13_cwt_ridge_tracking.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft

import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, CARD_LO, CARD_HI
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader import load_all_sessions
from sleep_monitor.ground_truth import gt_sliding_rates
from sleep_monitor.rates_classical import rate_cwt

RPT = ROOT / 'reports' / 'rates' / 'mask'
FIG = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'
RPT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ['diff', 'avg', 'CRE']
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})


def per_session_k(raw, gt):
    v = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
    if v.sum() < 20:
        return 1.0
    r = raw[v] / gt[v]
    r = r[(r > 0.3) & (r < 5.0)]
    return float(np.median(r)) if len(r) >= 10 else 1.0


def roll_med(x, k=5):
    out = np.full_like(x, np.nan, dtype=float)
    h = k // 2
    for i in range(len(x)):
        seg = x[max(0, i - h):i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def wcorr(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 20 or np.std(a[v]) < 1e-9 or np.std(b[v]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[v], b[v])[0, 1])


def continuous_viterbi_ridge(sig, fs, flo, fhi, win_s=15.0, hop_s=5.0,
                             jump_penalty=40.0):
    """Track the dominant cardiac frequency over the whole night with a
    Viterbi path that penalises large frame-to-frame frequency jumps.
    Returns (t_centers_hr, ridge_freq_hz)."""
    nperseg = int(win_s * fs)
    hop = int(hop_s * fs)
    noverlap = nperseg - hop
    f, t, Z = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   boundary=None, padded=False)
    P = np.abs(Z) ** 2
    band = (f >= flo) & (f <= fhi)
    fb = f[band]
    Pb = P[band, :]  # (n_freq, n_frames)
    nf, nt = Pb.shape
    if nt < 5 or nf < 3:
        return np.array([]), np.array([])
    # log-power emission; normalise per frame
    eps = 1e-12
    logP = np.log(Pb + eps)
    logP -= logP.max(axis=0, keepdims=True)
    # transition penalty matrix (freq-bin distance squared, scaled)
    df = fb[1] - fb[0]
    idx = np.arange(nf)
    trans = jump_penalty * (df * (idx[:, None] - idx[None, :])) ** 2
    # Viterbi (maximise logP - penalty)
    score = logP[:, 0].copy()
    back = np.zeros((nf, nt), dtype=int)
    for tt in range(1, nt):
        prev = score[:, None] - trans  # (from, to)
        best_from = np.argmax(prev, axis=0)
        score = prev[best_from, np.arange(nf)] + logP[:, tt]
        back[:, tt] = best_from
    path = np.zeros(nt, dtype=int)
    path[-1] = np.argmax(score)
    for tt in range(nt - 1, 0, -1):
        path[tt - 1] = back[path[tt], tt]
    ridge = fb[path]
    t_hr = t / 3600.0
    return t_hr, ridge


def main():
    sessions = load_all_sessions(with_sleep_profiles=False)
    flo, fhi = CARD_LO, CARD_HI
    win_n = int(30 * FS)

    rows = []
    # accumulate within-session r per method/channel
    agg = {}
    for sess in sessions:
        fs = sess.fs
        cle = sess.cap['CLE'].astype(np.float64)
        cre = sess.cap['CRE'].astype(np.float64)
        ch = sess.cap['CH'].astype(np.float64)
        acc = sess.cap['acc_mag'].astype(np.float64)
        chan_sig = {'diff': cle - cre, 'avg': (cle + cre) / 2.0, 'CRE': cre}

        n = sess.n_samples
        starts = np.arange(0, n - win_n + 1, win_n)
        centres_hr = (starts + win_n / 2.0) / fs / 3600.0
        gt_data = gt_sliding_rates(sess, win_sec=30.0, step_sec=5.0)
        gt_t = gt_data['t_hr']; gt_r = gt_data['card_hz']
        gt = np.full(len(starts), np.nan)
        for i, tc in enumerate(centres_hr):
            j = np.argmin(np.abs(gt_t - tc))
            if abs(gt_t[j] - tc) < 0.02:
                gt[i] = gt_r[j]
        gt_s = roll_med(gt, 5)

        for chn in CHANNELS:
            filt = remove_acc_artifact(chan_sig[chn], acc, flo, fhi, fs)

            # (A) rate_cwt per epoch
            cwt_ep = np.full(len(starts), np.nan)
            for i, s0 in enumerate(starts):
                cwt_ep[i] = rate_cwt(filt[s0:s0 + win_n], flo, fhi, fs, n_scales=24)
            k = per_session_k(cwt_ep, gt)
            r_cwt = wcorr(roll_med(cwt_ep / k, 5), gt_s)

            # (B) continuous Viterbi ridge -> sample at epoch centres
            tr_hr, ridge = continuous_viterbi_ridge(filt, fs, flo, fhi)
            vit_ep = np.full(len(starts), np.nan)
            if len(tr_hr):
                for i, tc in enumerate(centres_hr):
                    j = np.argmin(np.abs(tr_hr - tc))
                    vit_ep[i] = ridge[j]
            kv = per_session_k(vit_ep, gt)
            r_vit = wcorr(roll_med(vit_ep / kv, 5), gt_s)

            agg.setdefault(chn, {'cwt': [], 'viterbi': []})
            agg[chn]['cwt'].append(r_cwt)
            agg[chn]['viterbi'].append(r_vit)
            rows.append({'session': sess.label, 'channel': chn,
                         'r_cwt_epoch': r_cwt, 'r_viterbi_ridge': r_vit})
        print(f'  {sess.label} done')

    res = pd.DataFrame(rows)
    res.to_csv(RPT / 'cwt_ridge_tracking.csv', index=False)

    print('\nWITHIN-session r vs smoothed GT (cardiac):')
    print(f'{"channel":>8s} | rate_cwt(epoch)  Viterbi-ridge(continuous)')
    print('-' * 55)
    summary = {}
    for chn in CHANNELS:
        rc = np.nanmean(agg[chn]['cwt'])
        rv = np.nanmean(agg[chn]['viterbi'])
        summary[chn] = (rc, rv)
        print(f'{chn:>8s} |      {rc:+.3f}           {rv:+.3f}')
    print('\nReference (from diagnose_cardiac_tracking.py): peaks/hilbert/spectral '
          'all within-session r ~ 0 (best -0.008).')

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CHANNELS)); w = 0.35
    rc = [summary[c][0] for c in CHANNELS]
    rv = [summary[c][1] for c in CHANNELS]
    ax.bar(x - w/2, rc, w, label='rate_cwt (per-epoch ridge)', color='#E74C3C', alpha=0.85)
    ax.bar(x + w/2, rv, w, label='continuous Viterbi ridge', color='#2ECC71', alpha=0.85)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
    ax.set_ylabel('Mean within-session Pearson r vs smoothed GT')
    ax.set_title('Last-chance cardiac tracking: CWT & continuous ridge\n'
                 '(windowed peaks/hilbert/spectral all ~0)', fontweight='bold', fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / 'fig13_cwt_ridge_tracking.png')
    plt.close(fig)
    print('\nFig 13 saved.')


if __name__ == '__main__':
    main()
