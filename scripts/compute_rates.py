"""
Compute sliding-window rate estimates for all sessions and save metrics.

GT sources:
    Respiratory : Flow (nasal airflow) peak detection via neurokit2
    Cardiac     : ECG R-peak detection via neurokit2 (Pan-Tompkins)

Outputs:
    artifacts/rates/metrics.parquet   — accuracy metrics per session/channel/band/method
    artifacts/rates/windows/          — per-session rate time series
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import (
    load_all_sessions,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI, FS,
    CAP_CHANS, METHOD_NAMES,
    preprocess_full, sliding_rates, accuracy_metrics,
    gt_sliding_rates,
)


def compute_session_rates(session, win_sec, step_sec, acc_removal):
    full, _ = preprocess_full(session, acc_removal)

    gt = gt_sliding_rates(session, win_sec=win_sec, step_sec=step_sec)
    gt_t_hr = gt['t_hr']
    gt_resp_hz = gt['resp_hz']
    gt_card_hz = gt['card_hz']

    print(f'  GT resp: {gt["resp_gt"].signal_used} ({gt["resp_gt"].method}), '
          f'{len(gt["resp_gt"].peak_indices)} peaks')
    print(f'  GT card: {gt["card_gt"].signal_used} ({gt["card_gt"].method}), '
          f'{len(gt["card_gt"].peak_indices)} peaks')

    metric_rows = []
    window_frames = []

    for band, sig_key, gt_hz, flo, fhi in [
        ('resp', 'resp', gt_resp_hz, RESP_LO, RESP_HI),
        ('card', 'card', gt_card_hz, CARD_LO, CARD_HI),
    ]:
        for ch in CAP_CHANS:
            t_s, rates = sliding_rates(
                full[ch][sig_key], flo, fhi, FS, win_sec, step_sec)
            t_hr = t_s / 3600

            gt_interp = np.interp(
                t_hr,
                gt_t_hr[~np.isnan(gt_hz)],
                gt_hz[~np.isnan(gt_hz)],
                left=np.nan, right=np.nan)

            wdf = pd.DataFrame({'t_hr': t_hr, 'gt_hz': gt_interp})
            wdf['session'] = session.label
            wdf['band'] = band
            wdf['channel'] = ch
            for m in METHOD_NAMES:
                wdf[m] = rates[m]
            window_frames.append(wdf)

            for method in METHOD_NAMES:
                m = accuracy_metrics(gt_t_hr, gt_hz, t_hr, rates[method])
                metric_rows.append({
                    'session': session.label,
                    'subject': session.subject,
                    'channel': ch,
                    'band': band,
                    'method': method,
                    'gt_signal': gt['resp_gt'].signal_used if band == 'resp'
                                 else gt['card_gt'].signal_used,
                    **m,
                })

    return metric_rows, window_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--win', type=float, default=20.0, help='Window length (s)')
    parser.add_argument('--step', type=float, default=2.0, help='Step size (s)')
    parser.add_argument('--fast', action='store_true',
                        help='Use win=30s step=10s for faster runs')
    parser.add_argument('--no-acc-removal', action='store_true')
    args = parser.parse_args()

    win_sec = 30.0 if args.fast else args.win
    step_sec = 10.0 if args.fast else args.step
    acc_removal = not args.no_acc_removal

    out_dir = ROOT / 'artifacts' / 'rates'
    win_dir = out_dir / 'windows'
    win_dir.mkdir(parents=True, exist_ok=True)

    print(f'Win={win_sec}s  Step={step_sec}s  acc_removal={acc_removal}')

    all_metrics = []
    for session in load_all_sessions():
        print(f'\n── {session.label} ──')
        metrics, windows = compute_session_rates(
            session, win_sec, step_sec, acc_removal)
        all_metrics.extend(metrics)

        wdf = pd.concat(windows, ignore_index=True)
        wdf.to_parquet(win_dir / f'{session.label}.parquet', index=False)

    df = pd.DataFrame(all_metrics)
    df.to_parquet(out_dir / 'metrics.parquet', index=False)

    print(f'\nWrote {len(df)} metric rows to {out_dir / "metrics.parquet"}')
    print(f'Per-session windows in {win_dir}/')

    for band in ['resp', 'card']:
        unit = 'br/min' if band == 'resp' else 'BPM'
        sub = df[df['band'] == band]
        tbl = (sub.groupby(['channel', 'method'])['mae']
               .mean().unstack() * 60).round(2)
        print(f'\n── {band.upper()} Mean MAE ({unit}) ──')
        print(tbl.to_string())


if __name__ == '__main__':
    main()
