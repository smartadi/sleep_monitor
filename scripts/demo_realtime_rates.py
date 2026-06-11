#!/usr/bin/env python
"""
Phase 4: Real-time streaming rate demo.

Simulates real-time processing of a CAP recording, producing respiratory
and cardiac rate estimates epoch-by-epoch with a rolling Kalman filter.

The demo reads one session's data and processes it in 30-second chunks,
printing a live-updating rate display. It also saves a summary plot
showing the full-night tracked rates vs GT.

Usage:
    python scripts/demo_realtime_rates.py [--session 0] [--speed 10]

    --session : session index (0-11, default 0)
    --speed   : playback speed multiplier (default 10 = 10x real-time)
                Use 0 for maximum speed (no delay).
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    STAGE_LABELS,
)
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader import load_all_sessions, load_sleep_profile
from sleep_monitor.rates import (
    rate_spectral, rate_adaptive_peaks, rate_acf,
)

import functools
print = functools.partial(print, flush=True)

OUT_DIR = ROOT / 'reports' / 'rates' / 'hybrid_phase4'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN_SEC = 30.0


class KalmanState:
    """Lightweight scalar Kalman filter for streaming use."""

    def __init__(self, f_lo: float, f_hi: float, step_sec: float = 30.0,
                 max_delta_hz: float | None = None):
        self.f_lo = f_lo
        self.f_hi = f_hi
        self.x = (f_lo + f_hi) / 2.0
        self.P = ((f_hi - f_lo) / 2.0) ** 2

        if max_delta_hz is None:
            if f_hi <= 0.6:
                max_delta_hz = 2.0 / 60.0 * (step_sec / 30.0)
            else:
                max_delta_hz = 5.0 / 60.0 * (step_sec / 30.0)
        self.Q = max_delta_hz ** 2

        if f_hi <= 0.6:
            self.R_base = (2.5 / 60.0) ** 2
        else:
            self.R_base = (30.0 / 60.0) ** 2

    def update(self, observations: list[float]) -> float:
        """Push one epoch's observations and return the filtered rate."""
        x_pred = self.x
        P_pred = self.P + self.Q

        for z in observations:
            if np.isfinite(z) and self.f_lo <= z <= self.f_hi:
                S = P_pred + self.R_base
                K = P_pred / S
                x_pred = x_pred + K * (z - x_pred)
                P_pred = (1.0 - K) * P_pred

        self.x = np.clip(x_pred, self.f_lo, self.f_hi)
        self.P = P_pred
        return self.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, default=0)
    parser.add_argument('--speed', type=float, default=10.0)
    args = parser.parse_args()

    sessions = load_all_sessions()
    s = sessions[args.session]
    label = s.meta['label']

    try:
        s.sleep_profile = load_sleep_profile(s)
    except Exception:
        s.sleep_profile = None

    profile = s.sleep_profile
    fs = s.fs

    # Preprocess
    raw_cap = s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    acc = s.cap['acc_mag'].astype(np.float64)

    sig_resp = remove_acc_artifact(raw_cap, acc, RESP_LO, RESP_HI, fs)
    sig_card = remove_acc_artifact(raw_cap, acc, CARD_LO, CARD_HI, fs)
    gt_resp = bandpass(s.psg['Thorax'].astype(np.float64), RESP_LO, RESP_HI, fs)
    gt_card = bandpass(s.psg['Pleth'].astype(np.float64), CARD_LO, CARD_HI, fs)

    win_n = int(round(WIN_SEC * fs))
    n_total = min(len(sig_resp), len(sig_card), len(gt_resp), len(gt_card))
    n_epochs = (n_total - win_n) // win_n + 1
    delay = WIN_SEC / args.speed if args.speed > 0 else 0

    # Init Kalman states
    kf_resp = KalmanState(RESP_LO, RESP_HI, WIN_SEC)
    kf_card = KalmanState(CARD_LO, CARD_HI, WIN_SEC)

    # Storage
    t_arr = []
    resp_kalman, card_kalman = [], []
    resp_gt_arr, card_gt_arr = [], []
    stages_arr = []

    print(f"\n  Session: {label} ({n_total / fs / 3600:.1f} hr, {n_epochs} epochs)")
    print(f"  Speed: {'max' if args.speed == 0 else f'{args.speed}x'}")
    print(f"\n  {'Epoch':>5s}  {'Time':>8s}  {'Resp':>10s}  {'GT_R':>8s}  {'Card':>10s}  {'GT_C':>8s}  {'Stage':>6s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*6}")

    t_start = time.time()

    for epoch in range(n_epochs):
        start = epoch * win_n
        seg_resp = sig_resp[start:start + win_n]
        seg_card = sig_card[start:start + win_n]
        seg_gt_r = gt_resp[start:start + win_n]
        seg_gt_c = gt_card[start:start + win_n]

        t_center = (start + win_n / 2.0) / fs
        t_arr.append(t_center)

        # Compute observations
        obs_resp = [
            rate_spectral(seg_resp, RESP_LO, RESP_HI, fs),
            rate_adaptive_peaks(seg_resp, RESP_LO, RESP_HI, fs),
        ]
        obs_card = [
            rate_spectral(seg_card, CARD_LO, CARD_HI, fs),
            rate_adaptive_peaks(seg_card, CARD_LO, CARD_HI, fs),
        ]

        # Kalman update
        r_resp = kf_resp.update(obs_resp)
        r_card = kf_card.update(obs_card)
        resp_kalman.append(r_resp)
        card_kalman.append(r_card)

        # GT
        gt_r = rate_acf(seg_gt_r, RESP_LO, RESP_HI, fs, prominence=0.05)
        gt_c = rate_acf(seg_gt_c, CARD_LO, CARD_HI, fs, prominence=0.05)
        resp_gt_arr.append(gt_r)
        card_gt_arr.append(gt_c)

        # Stage
        stage = '?'
        if profile is not None:
            t_hr = np.array(profile['t_ep_hr'])
            codes = np.array(profile['codes'])
            si = np.searchsorted(t_hr, t_center / 3600.0, side='right') - 1
            if 0 <= si < len(codes):
                stage = STAGE_LABELS.get(int(codes[si]), '?')
        stages_arr.append(stage)

        # Display
        resp_bpm = r_resp * 60.0
        card_bpm = r_card * 60.0
        gt_r_bpm = gt_r * 60.0 if np.isfinite(gt_r) else float('nan')
        gt_c_bpm = gt_c * 60.0 if np.isfinite(gt_c) else float('nan')

        print(f"  {epoch:5d}  {t_center/60:7.1f}m  {resp_bpm:7.1f} b/m  {gt_r_bpm:6.1f}  "
              f"{card_bpm:7.1f} BPM  {gt_c_bpm:6.1f}  {stage:>6s}")

        if delay > 0:
            time.sleep(delay)

    elapsed = time.time() - t_start
    print(f"\n  Processed {n_epochs} epochs in {elapsed:.1f}s "
          f"({n_epochs * WIN_SEC / elapsed:.0f}x real-time)")

    # Summary plot
    t_min = np.array(t_arr) / 60.0
    resp_k = np.array(resp_kalman) * 60.0
    card_k = np.array(card_kalman) * 60.0
    resp_gt = np.array(resp_gt_arr) * 60.0
    card_gt = np.array(card_gt_arr) * 60.0

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax = axes[0]
    valid = np.isfinite(resp_gt)
    ax.plot(t_min[valid], resp_gt[valid], 'k-', lw=0.8, alpha=0.5, label='GT')
    ax.plot(t_min, resp_k, '-', color='#E74C3C', lw=1.5, label='Kalman (streaming)')
    ax.set_ylabel('Resp (br/min)')
    ax.legend(fontsize=9)
    ax.set_title(f'{label} — streaming Kalman rate tracker')

    ax = axes[1]
    valid = np.isfinite(card_gt)
    ax.plot(t_min[valid], card_gt[valid], 'k-', lw=0.8, alpha=0.5, label='GT')
    ax.plot(t_min, card_k, '-', color='#E74C3C', lw=1.5, label='Kalman (streaming)')
    ax.set_ylabel('Cardiac (BPM)')
    ax.set_xlabel('Time (min)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = OUT_DIR / f'streaming_demo_{label}.png'
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")

    # Quick metrics
    for name, est, gt in [('Resp', resp_k, resp_gt), ('Card', card_k, card_gt)]:
        v = np.isfinite(gt) & np.isfinite(est) & (gt > 0)
        if v.sum() > 10:
            mae = np.mean(np.abs(est[v] - gt[v]))
            print(f"  {name} MAE: {mae:.2f} {'br/min' if name == 'Resp' else 'BPM'}")


if __name__ == '__main__':
    main()
