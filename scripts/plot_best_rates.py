"""
Plot the two best rate detection methods (raw + scaled) against GT
for all 6 subjects (Night 1), with sleep stage overlay.

Best methods:
    Respiratory : rate_peaks_scaled_resp  (loose peaks / k)
    Cardiac     : rate_hilbert_scaled_cardiac  (Hilbert inst. freq / k)

Outputs PNG files to artifacts/plots/best_rates/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import (
    load_session, load_sleep_profile,
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    STAGE_LABELS, STAGE_COLORS,
    gt_sliding_rates,
    calibrate_k_cardiac, calibrate_k_resp,
    rate_hilbert, rate_peaks_scaled_resp,
)
from sleep_monitor.preprocessing import preprocess_full


def sliding_raw_and_scaled(signal, band, fs, k, win_sec, step_sec):
    """Compute raw and k-scaled rates in sliding windows."""
    win_n = int(round(win_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))

    t_list, raw_list, scaled_list = [], [], []

    if band == 'resp':
        f_lo, f_hi = RESP_LO, RESP_HI
    else:
        f_lo, f_hi = CARD_LO, CARD_HI

    for start in range(0, len(signal) - win_n + 1, step_n):
        seg = signal[start:start + win_n]
        t_list.append((start + win_n / 2.0) / fs)

        if band == 'card':
            raw_hz = rate_hilbert(seg, f_lo, f_hi, fs)
            scaled_hz = raw_hz / k if (np.isfinite(raw_hz) and k > 0) else np.nan
        else:
            # raw = loose peaks without k correction (k=1)
            raw_hz = rate_peaks_scaled_resp(seg, k=1.0, fs=fs)
            scaled_hz = rate_peaks_scaled_resp(seg, k=k, fs=fs)

        raw_list.append(raw_hz)
        scaled_list.append(scaled_hz)

    return (np.array(t_list),
            np.array(raw_list),
            np.array(scaled_list))


def paint_stages(ax, profile, t_max_hr):
    """Paint sleep stage background colours onto an axes."""
    if profile is None:
        return
    t_ep = profile['t_ep_hr']
    codes = profile['codes']
    epoch_hr = 30.0 / 3600.0

    for i, code in enumerate(codes):
        t0 = t_ep[i]
        t1 = t0 + epoch_hr
        if t0 > t_max_hr:
            break
        colour = STAGE_COLORS.get(int(code), '#AAAAAA')
        ax.axvspan(t0, t1, alpha=0.15, color=colour, linewidth=0)


def stage_legend_handles():
    return [Patch(facecolor=STAGE_COLORS[c], alpha=0.3,
                  label=STAGE_LABELS[c])
            for c in [4, 3, 2, 1, 0]]


def plot_session(session, out_dir, win_sec=30.0, step_sec=5.0):
    """Generate a 3-panel figure for one session."""
    label = session.label

    print(f'  Calibrating k...', end=' ', flush=True)
    k_card = calibrate_k_cardiac(session)
    k_resp = calibrate_k_resp(session)
    print(f'k_card={k_card:.2f}, k_resp={k_resp:.2f}')

    print(f'  Preprocessing...', end=' ', flush=True)
    full, _ = preprocess_full(session, acc_removal=True)
    print('done')

    print(f'  Computing GT...', end=' ', flush=True)
    gt = gt_sliding_rates(session, win_sec=win_sec, step_sec=step_sec)
    gt_t_hr = gt['t_hr']
    gt_resp_bpm = gt['resp_hz'] * 60
    gt_card_bpm = gt['card_hz'] * 60
    print('done')

    sig_card = full['CLE-CRE']['card']
    sig_resp = full['CLE-CRE']['resp']

    print(f'  Computing CAP rates...', end=' ', flush=True)
    t_card, raw_card, sc_card = sliding_raw_and_scaled(
        sig_card, 'card', FS, k_card, win_sec, step_sec)
    t_resp, raw_resp, sc_resp = sliding_raw_and_scaled(
        sig_resp, 'resp', FS, k_resp, win_sec, step_sec)
    print('done')

    # Convert to BPM / br-per-min
    t_card_hr = t_card / 3600
    t_resp_hr = t_resp / 3600
    raw_card_bpm = raw_card * 60
    sc_card_bpm = sc_card * 60
    raw_resp_bpm = raw_resp * 60
    sc_resp_bpm = sc_resp * 60

    profile = session.sleep_profile
    t_max_hr = float(session.time_hr[-1])

    # Compute MAE for scaled rates
    def _mae(gt_t, gt_v, est_t, est_v):
        valid_gt = ~np.isnan(gt_v)
        if valid_gt.sum() < 10:
            return np.nan
        interp = np.interp(est_t, gt_t[valid_gt], gt_v[valid_gt],
                           left=np.nan, right=np.nan)
        valid = np.isfinite(interp) & np.isfinite(est_v)
        if valid.sum() < 10:
            return np.nan
        return float(np.mean(np.abs(est_v[valid] - interp[valid])))

    mae_card = _mae(gt_t_hr, gt_card_bpm, t_card_hr, sc_card_bpm)
    mae_resp = _mae(gt_t_hr, gt_resp_bpm, t_resp_hr, sc_resp_bpm)

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 3, 3], hspace=0.08,
                  left=0.06, right=0.94, top=0.93, bottom=0.06)

    # --- Panel 0: Sleep stages (hypnogram) ---
    ax0 = fig.add_subplot(gs[0])
    if profile is not None:
        t_ep = profile['t_ep_hr']
        codes = profile['codes']
        ax0.step(t_ep, codes, where='post', color='#2C3E50', linewidth=1.2)
        ax0.set_yticks([0, 1, 2, 3, 4])
        ax0.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'], fontsize=8)
        ax0.set_ylim(-0.5, 4.5)
        ax0.invert_yaxis()
        paint_stages(ax0, profile, t_max_hr)
    else:
        ax0.text(0.5, 0.5, 'No sleep profile available',
                 transform=ax0.transAxes, ha='center', va='center')
    ax0.set_xlim(0, t_max_hr)
    ax0.set_ylabel('Stage', fontsize=9)
    ax0.tick_params(labelbottom=False)
    ax0.set_title(f'{label} ({session.subject})  —  '
                  f'k_card={k_card:.2f}  k_resp={k_resp:.2f}',
                  fontsize=12, fontweight='bold')

    # --- Panel 1: Respiratory rate ---
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    paint_stages(ax1, profile, t_max_hr)
    ax1.plot(gt_t_hr, gt_resp_bpm, color='#2C3E50', linewidth=1.0,
             alpha=0.8, label='GT (Flow peaks)')
    ax1.plot(t_resp_hr, raw_resp_bpm, color='#E74C3C', linewidth=0.5,
             alpha=0.4, label='Raw loose peaks')
    ax1.plot(t_resp_hr, sc_resp_bpm, color='#3498DB', linewidth=0.8,
             alpha=0.8, label=f'Scaled /k={k_resp:.2f}  (MAE={mae_resp:.1f})')
    ax1.set_ylim(0, 40)
    ax1.set_ylabel('Resp rate (br/min)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    ax1.tick_params(labelbottom=False)
    ax1.grid(axis='y', alpha=0.3)

    # --- Panel 2: Cardiac rate ---
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    paint_stages(ax2, profile, t_max_hr)
    ax2.plot(gt_t_hr, gt_card_bpm, color='#2C3E50', linewidth=1.0,
             alpha=0.8, label='GT (ECG R-peaks)')
    ax2.plot(t_card_hr, raw_card_bpm, color='#E74C3C', linewidth=0.5,
             alpha=0.4, label='Raw Hilbert')
    ax2.plot(t_card_hr, sc_card_bpm, color='#3498DB', linewidth=0.8,
             alpha=0.8, label=f'Scaled /k={k_card:.2f}  (MAE={mae_card:.1f})')
    ax2.set_ylim(30, 120)
    ax2.set_ylabel('Heart rate (BPM)', fontsize=10)
    ax2.set_xlabel('Time (hours)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8, ncol=3)
    ax2.grid(axis='y', alpha=0.3)

    fig.savefig(out_dir / f'{label}_rates.png', dpi=150)
    plt.close(fig)
    print(f'  Saved {label}_rates.png  '
          f'(resp MAE={mae_resp:.1f} br/min, card MAE={mae_card:.1f} BPM)')


def main():
    out_dir = ROOT / 'artifacts' / 'plots' / 'best_rates'
    out_dir.mkdir(parents=True, exist_ok=True)

    win_sec = 30.0
    step_sec = 5.0

    # Night 1 for each of 6 subjects
    session_indices = [0, 2, 4, 6, 8, 10]

    for idx in session_indices:
        session = load_session(idx, with_profile=True)
        print(f'\n== {session.label} ==')
        plot_session(session, out_dir, win_sec, step_sec)

    print(f'\nAll plots saved to {out_dir}/')


if __name__ == '__main__':
    main()
