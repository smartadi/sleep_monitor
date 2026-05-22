"""
Plot best rate detection methods on the validation dataset.

Validation dataset: 6 subjects, ~12.5 min each, controlled posture phases.
Channels: Cvl, Cvr, Cbl, Cbr (cap), Pleth, Thorax (PSG subset).
No ECG/Flow — GT from Pleth (cardiac) and Thorax (resp) peak detection.

Outputs PNGs to notebooks/plots/validation/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    bandpass, rate_hilbert, rate_peaks_scaled_resp,
    rate_acf,
)
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.loader_validation import (
    load_validation_session, load_all_validation_sessions,
    PHASE_COLORS, PHASE_LABELS,
)


# ── GT from Pleth and Thorax (no ECG/Flow available) ─────────────────────────

def gt_cardiac_from_pleth(pleth: np.ndarray, fs: float = FS):
    """Detect cardiac peaks from Pleth signal and return peak times."""
    pleth_bp = bandpass(pleth, CARD_LO, CARD_HI, fs)
    min_dist = max(1, int(0.3 * fs))  # min 0.3s between beats (~200 BPM max)
    pks, _ = find_peaks(pleth_bp, distance=min_dist,
                        prominence=0.15 * np.std(pleth_bp))
    if len(pks) < 2:
        return pks, np.array([]), np.array([])
    intervals = np.diff(pks) / fs
    # Quality filter: 30-200 BPM
    rate_hz = 1.0 / intervals
    valid = (rate_hz >= 0.5) & (rate_hz <= 3.33)
    bad = np.where(~valid)[0]
    keep = np.ones(len(pks), dtype=bool)
    for b in bad:
        keep[b + 1] = False
    pks = pks[keep]
    intervals = np.diff(pks) / fs
    rate_hz = 1.0 / intervals
    return pks, intervals, rate_hz


def gt_resp_from_thorax(thorax: np.ndarray, fs: float = FS):
    """Detect respiratory peaks from Thorax signal and return peak times."""
    thorax_bp = bandpass(thorax, RESP_LO, RESP_HI, fs)
    min_dist = max(1, int(1.5 * fs))  # min 1.5s between breaths (~40 br/min max)
    pks, _ = find_peaks(thorax_bp, distance=min_dist,
                        prominence=0.1 * np.std(thorax_bp))
    if len(pks) < 2:
        return pks, np.array([]), np.array([])
    intervals = np.diff(pks) / fs
    rate_hz = 1.0 / intervals
    # Quality filter: 4-40 br/min
    valid = (rate_hz >= 0.067) & (rate_hz <= 0.667)
    bad = np.where(~valid)[0]
    keep = np.ones(len(pks), dtype=bool)
    for b in bad:
        keep[b + 1] = False
    pks = pks[keep]
    intervals = np.diff(pks) / fs
    rate_hz = 1.0 / intervals
    return pks, intervals, rate_hz


def peaks_to_sliding_rate(peak_times_s, centres_s, win_sec):
    """Convert peak times to sliding-window average rate."""
    rates = np.full(len(centres_s), np.nan)
    half = win_sec / 2.0
    for i, c in enumerate(centres_s):
        mask = (peak_times_s >= c - half) & (peak_times_s <= c + half)
        n_peaks = mask.sum()
        if n_peaks >= 2:
            span = peak_times_s[mask][-1] - peak_times_s[mask][0]
            if span > 0:
                rates[i] = (n_peaks - 1) / span
    return rates


# ── k calibration for validation sessions ─────────────────────────────────────

def calibrate_k_validation(cap_sig, acc_mag, gt_sig, band, fs=FS,
                           n_windows=30, win_s=30.0, seed=42):
    """
    Calibrate k from the validation session.
    Shorter sessions so use 30s windows, fewer samples.
    """
    if band == 'card':
        f_lo, f_hi = CARD_LO, CARD_HI
    else:
        f_lo, f_hi = RESP_LO, RESP_HI

    sig = remove_acc_artifact(cap_sig, acc_mag, f_lo, f_hi, fs)
    gt_bp = bandpass(gt_sig, f_lo, f_hi, fs)

    win_n = int(round(win_s * fs))
    total = len(sig)
    if total < win_n * 2:
        return 1.0

    rng = np.random.default_rng(seed)
    max_start = total - win_n - 1
    n_actual = min(n_windows, max_start // (win_n // 2))
    starts = sorted(rng.integers(0, max_start, size=max(n_actual, 10)).tolist())

    ratios = []
    for st in starts:
        seg = sig[st:st + win_n]
        seg_gt = gt_bp[st:st + win_n]

        if band == 'card':
            r_cap = rate_hilbert(seg, f_lo, f_hi, fs)
        else:
            r_cap = rate_peaks_scaled_resp(seg, k=1.0, fs=fs)

        r_gt = rate_acf(seg_gt, f_lo, f_hi, fs, prominence=0.05)

        if np.isfinite(r_cap) and np.isfinite(r_gt) and r_gt > 0 and r_cap > 0:
            ratios.append(r_cap / r_gt)

    if len(ratios) < 5:
        return 1.0
    return float(np.median(ratios))


# ── Sliding-window rate computation ───────────────────────────────────────────

def sliding_raw_and_scaled(signal, band, fs, k, win_sec, step_sec):
    """Compute raw and k-scaled rates in sliding windows."""
    win_n = int(round(win_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))

    if band == 'resp':
        f_lo, f_hi = RESP_LO, RESP_HI
    else:
        f_lo, f_hi = CARD_LO, CARD_HI

    t_list, raw_list, scaled_list = [], [], []

    for start in range(0, len(signal) - win_n + 1, step_n):
        seg = signal[start:start + win_n]
        t_list.append((start + win_n / 2.0) / fs)

        if band == 'card':
            raw_hz = rate_hilbert(seg, f_lo, f_hi, fs)
            scaled_hz = raw_hz / k if (np.isfinite(raw_hz) and k > 0) else np.nan
        else:
            raw_hz = rate_peaks_scaled_resp(seg, k=1.0, fs=fs)
            scaled_hz = rate_peaks_scaled_resp(seg, k=k, fs=fs)

        raw_list.append(raw_hz)
        scaled_list.append(scaled_hz)

    return np.array(t_list), np.array(raw_list), np.array(scaled_list)


# ── Plotting ──────────────────────────────────────────────────────────────────

def paint_phases(ax, segments, t_max_s):
    """Paint phase background colours."""
    for seg in segments:
        colour = PHASE_COLORS.get(seg['phase'], '#AAAAAA')
        ax.axvspan(seg['start_s'], seg['end_s'], alpha=0.15,
                   color=colour, linewidth=0)


def phase_legend_handles(segments):
    """Create legend patches for unique phases in order of appearance."""
    seen = []
    for seg in segments:
        if seg['phase'] not in seen:
            seen.append(seg['phase'])
    return [Patch(facecolor=PHASE_COLORS.get(p, '#AAA'), alpha=0.35,
                  label=PHASE_LABELS.get(p, p))
            for p in seen]


def plot_validation_session(session, out_dir, win_sec=15.0, step_sec=2.0):
    """Generate a 3-panel figure for one validation session."""
    label = session.label
    subject = session.subject
    fs = session.fs
    t_max_s = session.time_s[-1]

    # Pick best differential channel
    cap_diff = session.cap['Cvl-Cvr']
    acc_mag = session.cap['acc_mag']

    # GT
    print(f'  Computing GT...', end=' ', flush=True)
    card_pks, card_ipi, card_rate_hz = gt_cardiac_from_pleth(session.psg['Pleth'], fs)
    resp_pks, resp_ipi, resp_rate_hz = gt_resp_from_thorax(session.psg['Thorax'], fs)

    card_peak_times = card_pks / fs
    resp_peak_times = resp_pks / fs

    # Sliding GT
    centres_s = np.arange(win_sec / 2, t_max_s - win_sec / 2 + 0.1, step_sec)
    gt_card_hz = peaks_to_sliding_rate(card_peak_times, centres_s, win_sec)
    gt_resp_hz = peaks_to_sliding_rate(resp_peak_times, centres_s, win_sec)
    gt_card_bpm = gt_card_hz * 60
    gt_resp_bpm = gt_resp_hz * 60
    print(f'{len(card_pks)} card peaks, {len(resp_pks)} resp peaks')

    # Calibrate k
    print(f'  Calibrating k...', end=' ', flush=True)
    k_card = calibrate_k_validation(
        cap_diff, acc_mag, session.psg['Pleth'], 'card', fs)
    k_resp = calibrate_k_validation(
        cap_diff, acc_mag, session.psg['Thorax'], 'resp', fs)
    print(f'k_card={k_card:.2f}, k_resp={k_resp:.2f}')

    # Preprocess CAP
    sig_card = remove_acc_artifact(cap_diff, acc_mag, CARD_LO, CARD_HI, fs)
    sig_resp = remove_acc_artifact(cap_diff, acc_mag, RESP_LO, RESP_HI, fs)

    # Sliding rates
    print(f'  Computing CAP rates...', end=' ', flush=True)
    t_card, raw_card, sc_card = sliding_raw_and_scaled(
        sig_card, 'card', fs, k_card, win_sec, step_sec)
    t_resp, raw_resp, sc_resp = sliding_raw_and_scaled(
        sig_resp, 'resp', fs, k_resp, win_sec, step_sec)
    print('done')

    raw_card_bpm = raw_card * 60
    sc_card_bpm = sc_card * 60
    raw_resp_bpm = raw_resp * 60
    sc_resp_bpm = sc_resp * 60

    # MAE
    def _mae(gt_t, gt_v, est_t, est_v):
        valid_gt = ~np.isnan(gt_v)
        if valid_gt.sum() < 5:
            return np.nan
        interp = np.interp(est_t, gt_t[valid_gt], gt_v[valid_gt],
                           left=np.nan, right=np.nan)
        valid = np.isfinite(interp) & np.isfinite(est_v)
        if valid.sum() < 5:
            return np.nan
        return float(np.mean(np.abs(est_v[valid] - interp[valid])))

    mae_card = _mae(centres_s, gt_card_bpm, t_card, sc_card_bpm)
    mae_resp = _mae(centres_s, gt_resp_bpm, t_resp, sc_resp_bpm)

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 3, 3], hspace=0.08,
                  left=0.06, right=0.94, top=0.93, bottom=0.06)

    # --- Panel 0: Phase timeline ---
    ax0 = fig.add_subplot(gs[0])
    for seg in session.phase_segments:
        colour = PHASE_COLORS.get(seg['phase'], '#AAAAAA')
        mid = (seg['start_s'] + seg['end_s']) / 2
        ax0.axvspan(seg['start_s'], seg['end_s'], alpha=0.4,
                    color=colour, linewidth=0.5, edgecolor='white')
        duration = seg['end_s'] - seg['start_s']
        if duration > 15:
            short = PHASE_LABELS.get(seg['phase'], seg['phase'])
            ax0.text(mid, 0.5, short, ha='center', va='center',
                     fontsize=7, fontweight='bold', transform=ax0.get_xaxis_transform())
    ax0.set_xlim(0, t_max_s)
    ax0.set_yticks([])
    ax0.set_ylabel('Phase', fontsize=9)
    ax0.tick_params(labelbottom=False)
    ax0.set_title(f'{label} ({subject})  --  '
                  f'k_card={k_card:.2f}  k_resp={k_resp:.2f}',
                  fontsize=12, fontweight='bold')

    handles = phase_legend_handles(session.phase_segments)
    ax0.legend(handles=handles, loc='upper right', fontsize=6,
               ncol=min(len(handles), 5), framealpha=0.8)

    # --- Panel 1: Respiratory rate ---
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    paint_phases(ax1, session.phase_segments, t_max_s)
    ax1.plot(centres_s, gt_resp_bpm, color='#2C3E50', linewidth=1.0,
             alpha=0.8, label='GT (Thorax peaks)')
    ax1.plot(t_resp, raw_resp_bpm, color='#E74C3C', linewidth=0.6,
             alpha=0.4, label='Raw loose peaks')
    mae_resp_str = f'{mae_resp:.1f}' if np.isfinite(mae_resp) else 'N/A'
    ax1.plot(t_resp, sc_resp_bpm, color='#3498DB', linewidth=0.9,
             alpha=0.8, label=f'Scaled /k={k_resp:.2f}  (MAE={mae_resp_str})')
    ax1.set_ylim(0, 40)
    ax1.set_ylabel('Resp rate (br/min)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    ax1.tick_params(labelbottom=False)
    ax1.grid(axis='y', alpha=0.3)

    # --- Panel 2: Cardiac rate ---
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    paint_phases(ax2, session.phase_segments, t_max_s)
    ax2.plot(centres_s, gt_card_bpm, color='#2C3E50', linewidth=1.0,
             alpha=0.8, label='GT (Pleth peaks)')
    ax2.plot(t_card, raw_card_bpm, color='#E74C3C', linewidth=0.6,
             alpha=0.4, label='Raw Hilbert')
    mae_card_str = f'{mae_card:.1f}' if np.isfinite(mae_card) else 'N/A'
    ax2.plot(t_card, sc_card_bpm, color='#3498DB', linewidth=0.9,
             alpha=0.8, label=f'Scaled /k={k_card:.2f}  (MAE={mae_card_str})')
    ax2.set_ylim(30, 150)
    ax2.set_ylabel('Heart rate (BPM)', fontsize=10)
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8, ncol=3)
    ax2.grid(axis='y', alpha=0.3)

    fig.savefig(out_dir / f'{label}_{subject}_rates.png', dpi=150)
    plt.close(fig)
    print(f'  Saved {label}_{subject}_rates.png  '
          f'(resp MAE={mae_resp_str} br/min, card MAE={mae_card_str} BPM)')


def main():
    out_dir = ROOT / 'notebooks' / 'plots' / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    win_sec = 15.0
    step_sec = 2.0

    for idx in range(6):
        session = load_validation_session(idx)
        print(f'\n== {session.label} ({session.subject}) ==')
        plot_validation_session(session, out_dir, win_sec, step_sec)

    print(f'\nAll plots saved to {out_dir}/')


if __name__ == '__main__':
    main()
