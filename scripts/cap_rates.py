"""
cap_rates.py — CLI for heart-rate & respiratory-rate analysis.

Run from the repo root:
  python scripts/cap_rates.py --subject OS001 --night 2 --start 2.0 --mode inspect
  python scripts/cap_rates.py --subject OS001 --night 2 --mode rates
  python scripts/cap_rates.py --subject OS001 --night 2 --mode metrics
  python scripts/cap_rates.py --subject OS001 --night 2 --start 2.0 --mode all

Available methods: peaks, zerocross, acf, spectral, hilbert
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from sleep_monitor import (
    find_meta, load_session,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI, FS,
    CAP_CHANS, METHOD_NAMES, METHOD_COLORS, METHOD_LABELS, GT_COLOR,
    preprocess_window, preprocess_full,
    sliding_rates, accuracy_metrics,
    plot_window_inspection,
)


# ── Mode: inspect ──────────────────────────────────────────────────────────────

def mode_inspect(session, start_hr: float, win_hr: float = 1/60,
                 acc_removal: bool = True, method: str = 'peaks') -> None:
    win = preprocess_window(session, start_hr, win_hr, acc_removal)
    fig = plot_window_inspection(win, win['t_s'], session, start_hr, method)
    plt.show()


# ── Mode: rates ────────────────────────────────────────────────────────────────

def mode_rates(session, acc_removal: bool = True,
               win_sec: float = 20.0, step_sec: float = 1.0) -> None:
    m    = session.meta
    t_hr = session.time_hr
    full, gt = preprocess_full(session, acc_removal)

    print(f'Sliding window (win={win_sec}s, step={step_sec}s)...')
    slide, slide_t = {}, {}
    for ch in CAP_CHANS:
        t_s, rates = sliding_rates(full[ch]['resp'], RESP_LO, RESP_HI, FS, win_sec, step_sec)
        slide[ch]  = rates
        slide_t[ch] = t_s / 3600

    gt_t_s, gt_rates = sliding_rates(gt['thorax_bp'], RESP_LO, RESP_HI, FS, win_sec, step_sec)
    gt_t_hr = gt_t_s / 3600

    fig, axes = plt.subplots(len(CAP_CHANS), 1,
                              figsize=(16, 3 * len(CAP_CHANS)),
                              sharex=True, gridspec_kw={'hspace': 0.4})
    for ax, ch in zip(axes, CAP_CHANS):
        from sleep_monitor import plot_rates_vs_gt
        plot_rates_vs_gt(slide_t[ch], slide[ch],
                         gt_t_hr, gt_rates['acf'],
                         band='resp', ax=ax, channel=ch)
    axes[-1].set_xlabel('Time (hr)', fontsize=8)
    plt.suptitle(f"{m['label']} {m['subject']} — Respiratory Rates", fontsize=11)
    plt.tight_layout()
    plt.show()


# ── Mode: metrics ──────────────────────────────────────────────────────────────

def mode_metrics(session, acc_removal: bool = True,
                 win_sec: float = 20.0, step_sec: float = 1.0) -> None:
    full, gt = preprocess_full(session, acc_removal)

    gt_t_s, gt_rates = sliding_rates(gt['thorax_bp'], RESP_LO, RESP_HI, FS, win_sec, step_sec)
    gt_t_hr = gt_t_s / 3600
    gt_hz   = gt_rates['acf']

    print(f'\n{"Channel":<12} {"Method":<12} {"n":>6} {"MAE":>8} {"RMSE":>8} {"r":>7} {"Bias":>8}')
    print('-' * 62)
    for ch in CAP_CHANS:
        t_s, rates = sliding_rates(full[ch]['resp'], RESP_LO, RESP_HI, FS, win_sec, step_sec)
        t_hr = t_s / 3600
        for method in METHOD_NAMES:
            m = accuracy_metrics(gt_t_hr, gt_hz, t_hr, rates[method])
            if m['n'] == 0:
                continue
            mae_bpm  = m['mae']  * 60
            rmse_bpm = m['rmse'] * 60
            bias_bpm = m['bias'] * 60
            print(f"{ch:<12} {method:<12} {m['n']:>6} {mae_bpm:>8.2f} {rmse_bpm:>8.2f} "
                  f"{m['r']:>7.3f} {bias_bpm:>8.2f}")
        print()


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='CAP respiratory/cardiac rate analysis')
    p.add_argument('--subject',  required=True, help='e.g. OS001')
    p.add_argument('--night',    required=True, type=int, choices=[1, 2])
    p.add_argument('--mode',     default='inspect',
                   choices=['inspect', 'rates', 'metrics', 'all'])
    p.add_argument('--method',   default='peaks', choices=METHOD_NAMES)
    p.add_argument('--start',    type=float, default=2.0,
                   help='Inspection window start (hours)')
    p.add_argument('--win-min',  type=float, default=1.0,
                   help='Inspection window length (minutes)')
    p.add_argument('--win-sec',  type=float, default=20.0,
                   help='Sliding window length (seconds)')
    p.add_argument('--step-sec', type=float, default=1.0,
                   help='Sliding window step (seconds)')
    p.add_argument('--no-acc-removal', action='store_true',
                   help='Disable accelerometer artifact removal')
    return p.parse_args()


def main():
    args    = parse_args()
    meta    = find_meta(args.subject, args.night)
    session = load_session(meta)
    acc_rem = not args.no_acc_removal
    win_hr  = args.win_min / 60.0

    if args.mode in ('inspect', 'all'):
        mode_inspect(session, args.start, win_hr, acc_rem, args.method)
    if args.mode in ('rates', 'all'):
        mode_rates(session, acc_rem, args.win_sec, args.step_sec)
    if args.mode in ('metrics', 'all'):
        mode_metrics(session, acc_rem, args.win_sec, args.step_sec)


if __name__ == '__main__':
    main()
