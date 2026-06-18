#!/usr/bin/env python
"""
Consolidated respiratory ground truth from MULTIPLE independent PSG signals
(Flow + Thorax + Abdomen + RIPSum), on the standard 30s/5s sliding grid.

Motivation: the single-signal Flow GT has estimation jitter that we were
(wrongly) lumping into "noise". Flow and RIPSum (thorax+abdomen) are physically
INDEPENDENT sensors. If they agree on the within-session rate VARIATION, that
variation is real physiology, not noise — and their consensus is a cleaner GT.

This builds a PARALLEL, versioned GT (does NOT overwrite ground_truth.py / caches).
After validation we decide whether to flip it canonical.

Literature basis:
  - RIPSum (thorax+abdomen) is the AASM airflow surrogate when flow is unreliable.
  - Flow drops to 0 during obstructive AND central apnea; RIP effort continues in
    obstructive, stops in central -> apnea epochs are labelled, not forced to a rate.
  - Thoraco-abdominal asynchrony -> global polarity correction before summing.

Outputs:
  artifacts/consolidated_resp_gt.parquet      (per-epoch multi-signal + consensus)
  reports/rates/mask/gt_cross_signal_agreement.csv
  writeup/figures/mask_rate_detection/fig16_gt_cross_signal_agreement.png
  writeup/figures/mask_rate_detection/fig17_gt_consensus_example.png
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, RESP_LO, RESP_HI
from sleep_monitor.filters import bandpass
from sleep_monitor.loader import load_all_sessions
from scipy.signal import find_peaks

ART = ROOT / 'artifacts'
RPT = ROOT / 'reports' / 'rates' / 'mask'
FIG = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'
for d in (ART, RPT, FIG):
    d.mkdir(parents=True, exist_ok=True)

SIGNALS = ['Flow', 'Thorax', 'Abdomen', 'RIPSum']
AGREE_TOL_HZ = 2.0 / 60.0   # 2 br/min: signals "agree" within this


def flow_breaths(flow, fs):
    """neurokit2 respiratory peaks; fallback to bandpass+find_peaks."""
    try:
        import neurokit2 as nk
        clean = nk.rsp_clean(flow, sampling_rate=int(fs))
        info = nk.rsp_findpeaks(clean, sampling_rate=int(fs))
        pk = np.array(info['RSP_Peaks'], dtype=int)
        if len(pk) >= 10:
            return pk
    except Exception:
        pass
    return rip_breaths(flow, fs)


def rip_breaths(sig, fs):
    """Bandpass + peak detection for RIP / effort signals."""
    bp = bandpass(sig.astype(np.float64), RESP_LO, RESP_HI, fs)
    min_dist = int(fs / RESP_HI * 0.6)
    peaks, _ = find_peaks(bp, distance=min_dist, prominence=0.05 * np.std(bp))
    return peaks


def quality_filter(pk, fs):
    if len(pk) < 2:
        return pk
    iv = np.diff(pk) / fs
    good = (iv >= 1.0 / RESP_HI) & (iv <= 1.0 / RESP_LO)
    keep = np.ones(len(pk), dtype=bool)
    keep[1:][~good] = False
    return pk[keep]


def sliding_rate(pk_times, centres_s, win_sec=30.0):
    half = win_sec / 2.0
    out = np.full(len(centres_s), np.nan)
    for i, tc in enumerate(centres_s):
        w = pk_times[(pk_times >= tc - half) & (pk_times <= tc + half)]
        if len(w) >= 2:
            out[i] = (len(w) - 1) / (w[-1] - w[0])
    return out


def roll_med(x, k=5):
    o = np.full_like(x, np.nan, float); h = k // 2
    for i in range(len(x)):
        s = x[max(0, i - h):i + h + 1]; s = s[np.isfinite(s)]
        if len(s):
            o[i] = np.median(s)
    return o


def wcorr(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 20 or np.std(a[v]) < 1e-9 or np.std(b[v]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[v], b[v])[0, 1])


def main():
    sessions = load_all_sessions(with_sleep_profiles=False, with_apnea=True)
    rows = []
    agree_rows = []

    for sess in sessions:
        fs = sess.fs
        psg = sess.psg
        flow = psg['Flow'].astype(np.float64)
        thx = psg['Thorax'].astype(np.float64)
        abd = psg['Abdomen'].astype(np.float64)

        # phase-aware RIPSum: global polarity correction
        thx_bp = bandpass(thx, RESP_LO, RESP_HI, fs)
        abd_bp = bandpass(abd, RESP_LO, RESP_HI, fs)
        sign = 1.0 if np.corrcoef(thx_bp, abd_bp)[0, 1] >= 0 else -1.0
        ripsum = thx_bp + sign * abd_bp

        breaths = {
            'Flow': quality_filter(flow_breaths(flow, fs), fs),
            'Thorax': quality_filter(rip_breaths(thx, fs), fs),
            'Abdomen': quality_filter(rip_breaths(abd, fs), fs),
            'RIPSum': quality_filter(rip_breaths(ripsum, fs), fs),
        }

        n = sess.n_samples
        win_n = int(30 * fs); step_n = int(5 * fs)
        centres_s = np.array([(s + win_n / 2.0) / fs
                              for s in range(0, n - win_n + 1, step_n)])
        t_hr = centres_s / 3600.0

        rates = {sig: sliding_rate(breaths[sig] / fs, centres_s) for sig in SIGNALS}
        apnea = sess.apnea_at(t_hr) if sess.apnea_events is not None else np.zeros(len(t_hr))

        # Per-session signal-quality gate: drop any reference signal whose rate is
        # net anti-correlated with the median of the others (handles e.g. S3 Thorax
        # polarity/paradox/contamination). Always keep >=2 best signals.
        allM = np.vstack([rates[s] for s in SIGNALS])
        keep_sigs = []
        sig_corr = {}
        for j, s in enumerate(SIGNALS):
            others = np.delete(allM, j, axis=0)
            med_others = np.nanmedian(others, axis=0)
            sig_corr[s] = wcorr(rates[s], med_others)
        ranked = sorted(SIGNALS, key=lambda s: (sig_corr[s] if np.isfinite(sig_corr[s]) else -1), reverse=True)
        for s in SIGNALS:
            c = sig_corr[s]
            if np.isfinite(c) and c >= 0.10:
                keep_sigs.append(s)
        if len(keep_sigs) < 2:
            keep_sigs = ranked[:2]   # fall back to 2 best
        dropped = [s for s in SIGNALS if s not in keep_sigs]

        # consensus: median of KEPT signals; agreement count within tol
        M = np.vstack([rates[s] for s in keep_sigs])         # (>=2, K)
        consensus = np.nanmedian(M, axis=0)
        n_agree = np.array([
            np.sum(np.abs(M[:, i] - consensus[i]) <= AGREE_TOL_HZ)
            for i in range(M.shape[1])
        ])
        spread = np.nanstd(M, axis=0)
        confidence = np.clip(1.0 - spread / AGREE_TOL_HZ, 0, 1)

        for i in range(len(t_hr)):
            rows.append({
                'session': sess.label, 't_hr': t_hr[i],
                'rate_flow': rates['Flow'][i], 'rate_thorax': rates['Thorax'][i],
                'rate_abdomen': rates['Abdomen'][i], 'rate_ripsum': rates['RIPSum'][i],
                'rate_consensus': consensus[i], 'n_agree': int(n_agree[i]),
                'confidence': float(confidence[i]), 'apnea': int(apnea[i]),
            })

        # cross-signal agreement (the "variation is real" test): within-session r
        # on raw and on fluctuations (detrended via subtracting rolling median)
        fr, rr = rates['Flow'], rates['RIPSum']
        agree_rows.append({
            'session': sess.label,
            'r_flow_ripsum': wcorr(fr, rr),
            'r_flow_thorax': wcorr(fr, rates['Thorax']),
            'r_flow_abdomen': wcorr(fr, rates['Abdomen']),
            'r_flow_ripsum_smooth': wcorr(roll_med(fr), roll_med(rr)),
            'r_flow_ripsum_fluct': wcorr(fr - roll_med(fr, 9), rr - roll_med(rr, 9)),
            'mean_n_agree': float(np.nanmean(n_agree)),
            'pct_apnea': float(np.mean(apnea > 0) * 100),
            'dropped': ','.join(dropped) if dropped else '-',
        })
        print(f'  {sess.label}: r(Flow,RIPSum)={agree_rows[-1]["r_flow_ripsum"]:+.3f}  '
              f'kept={keep_sigs}  dropped={dropped if dropped else "none"}')

    df = pd.DataFrame(rows)
    df.to_parquet(ART / 'consolidated_resp_gt.parquet', index=False)
    adf = pd.DataFrame(agree_rows)
    adf.to_csv(RPT / 'gt_cross_signal_agreement.csv', index=False)

    print('\n=== CROSS-SIGNAL AGREEMENT (is within-session variation real?) ===')
    print(adf.to_string(index=False))
    print('\nAggregate (mean across sessions):')
    for c in ['r_flow_ripsum', 'r_flow_ripsum_smooth', 'r_flow_ripsum_fluct',
              'r_flow_thorax', 'r_flow_abdomen', 'mean_n_agree']:
        print(f'  {c:>24s}: {adf[c].mean():+.3f}')

    # ── Figures ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(adf))
    ax.bar(x - 0.2, adf['r_flow_ripsum'], 0.4, label='Flow vs RIPSum (raw rate)', color='#3498DB')
    ax.bar(x + 0.2, adf['r_flow_ripsum_fluct'], 0.4, label='Flow vs RIPSum (fluctuations only)', color='#2ECC71')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(adf['session'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Within-session Pearson r')
    ax.set_title('Cross-signal agreement: do two INDEPENDENT sensors (Flow vs RIPSum)\n'
                 'agree on within-session rate variation? (high => variation is real, not noise)',
                 fontsize=10, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / 'fig16_gt_cross_signal_agreement.png')
    plt.close(fig)
    print('\nFig 16 saved.')

    # example consensus trace (session with most epochs)
    ex = df.groupby('session').size().idxmax()
    s = df[df.session == ex].sort_values('t_hr')
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(s.t_hr, s.rate_flow * 60, lw=0.6, alpha=0.5, label='Flow', color='#E74C3C')
    ax.plot(s.t_hr, s.rate_ripsum * 60, lw=0.6, alpha=0.5, label='RIPSum', color='#3498DB')
    ax.plot(s.t_hr, s.rate_consensus * 60, lw=1.0, alpha=0.9, label='Consensus', color='black')
    ap = s[s.apnea > 0]
    ax.scatter(ap.t_hr, np.full(len(ap), 6), s=4, color='orange', label='apnea/hypopnea epoch')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Resp rate (br/min)'); ax.set_ylim(5, 30)
    ax.set_title(f'{ex}: multi-signal consensus respiratory GT', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / 'fig17_gt_consensus_example.png')
    plt.close(fig)
    print('Fig 17 saved.')
    print(f'\nSaved {len(df)} epoch rows -> artifacts/consolidated_resp_gt.parquet')


if __name__ == '__main__':
    main()
