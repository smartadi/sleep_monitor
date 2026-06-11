"""
Post-movement settling event visualization.

Detects head movements from accelerometer, extracts +-15 min windows around each,
and plots stacked panels showing what happens as the body settles:
  - Accelerometer magnitude (the trigger)
  - Raw CAP (CLE-CRE) with slow DC mean overlay
  - Thorax envelope (rolling RMS)
  - CAP cardiac rate vs ECG heart rate (PSG)
  - CAP respiratory rate vs Flow/Thorax rate (PSG)
  - Pleth (PPG) amplitude envelope
  - Sleep stage bar (N3 highlighted)

Events are color-coded by whether N3 follows the movement.

Output: reports/slow_wave/settling_events_<session>.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, STAGE_LABELS, STAGE_COLORS,
)
from sleep_monitor.filters import bandpass, lowpass
from sleep_monitor.rates import rate_acf
from sleep_monitor.sessions import SESSION_META

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_MIN = 15  # minutes before/after movement event
WIN_SAMP = int(WINDOW_MIN * 60 * FS)


# ── Movement event detection ─────────────────────────────────────────────────

def detect_movement_events(acc_mag, fs=FS, min_gap_sec=120, threshold_mad=4.0):
    """
    Detect head movement events as peaks in accelerometer RMS.

    Uses a sliding 5s RMS, thresholds at median + threshold_mad * MAD,
    then merges events closer than min_gap_sec.
    """
    win = int(5.0 * fs)
    n = len(acc_mag)
    acc = acc_mag.astype(np.float64)

    # Sliding RMS (AC component)
    rms = np.zeros(n)
    for i in range(0, n - win, win // 2):
        chunk = acc[i:i + win]
        rms[i:i + win] = np.maximum(rms[i:i + win],
                                     np.sqrt(np.mean((chunk - np.mean(chunk)) ** 2)))

    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-12
    threshold = med + threshold_mad * mad

    above = rms > threshold
    events = []
    in_event = False
    start = 0
    for i in range(n):
        if above[i] and not in_event:
            start = i
            in_event = True
        elif not above[i] and in_event:
            peak_idx = start + np.argmax(rms[start:i])
            events.append(peak_idx)
            in_event = False
    if in_event:
        peak_idx = start + np.argmax(rms[start:])
        events.append(peak_idx)

    # Merge events closer than min_gap
    if len(events) < 2:
        return np.array(events, dtype=int), rms, threshold
    merged = [events[0]]
    min_gap = int(min_gap_sec * fs)
    for e in events[1:]:
        if e - merged[-1] > min_gap:
            merged.append(e)
        else:
            if rms[e] > rms[merged[-1]]:
                merged[-1] = e

    return np.array(merged, dtype=int), rms, threshold


# ── Rate computation in a window ─────────────────────────────────────────────

def sliding_rate_in_window(sig, f_lo, f_hi, fs=FS, win_sec=30.0, step_sec=10.0):
    """Compute ACF-based rate on a sliding window within a segment."""
    bp = bandpass(sig.astype(np.float64), f_lo, f_hi, fs)
    win_n = int(win_sec * fs)
    step_n = int(step_sec * fs)
    n = len(bp)
    times = []
    rates = []
    for s0 in range(0, n - win_n + 1, step_n):
        chunk = bp[s0:s0 + win_n]
        r = rate_acf(chunk, f_lo, f_hi, fs)
        times.append((s0 + win_n / 2) / fs)
        rates.append(r)
    return np.array(times), np.array(rates)


def psg_rate_in_window(peak_times_s, t_start_s, t_end_s, win_sec=30.0, step_sec=10.0):
    """Compute PSG ground truth rate from pre-detected peaks."""
    half = win_sec / 2.0
    times = []
    rates = []
    tc = t_start_s + half
    while tc + half <= t_end_s:
        in_win = peak_times_s[(peak_times_s >= tc - half) & (peak_times_s <= tc + half)]
        if len(in_win) >= 2:
            rates.append((len(in_win) - 1) / (in_win[-1] - in_win[0]))
        else:
            rates.append(np.nan)
        times.append(tc - t_start_s)  # relative to window start
        tc += step_sec
    return np.array(times), np.array(rates)


# ── Stage lookup ─────────────────────────────────────────────────────────────

def get_stage_at_time(sleep_profile, t_hr):
    """Return stage code at a given time (hours)."""
    if sleep_profile is None:
        return -1
    t_ep = sleep_profile['t_ep_hr']
    codes = sleep_profile['codes']
    idx = np.searchsorted(t_ep, t_hr, side='right') - 1
    if 0 <= idx < len(codes):
        return int(codes[idx])
    return -1


def get_stage_series(sleep_profile, t_hr_start, t_hr_end, fs=FS):
    """Return per-sample stage codes for a time range."""
    if sleep_profile is None:
        n = int((t_hr_end - t_hr_start) * 3600 * fs)
        return np.full(n, -1, dtype=np.int8)
    t_ep = sleep_profile['t_ep_hr']
    codes = sleep_profile['codes']
    n = int((t_hr_end - t_hr_start) * 3600 * fs)
    result = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        t = t_hr_start + i / (fs * 3600)
        idx = np.searchsorted(t_ep, t, side='right') - 1
        if 0 <= idx < len(codes):
            result[i] = codes[idx]
    return result


def get_stage_blocks(sleep_profile, t_hr_start, t_hr_end):
    """Return list of (start_min, end_min, stage_code) blocks relative to window start."""
    if sleep_profile is None:
        return []
    t_ep = sleep_profile['t_ep_hr']
    codes = sleep_profile['codes']
    epoch_hr = PSG_EPOCH_SEC / 3600.0
    blocks = []
    for i in range(len(codes)):
        ep_start = t_ep[i]
        ep_end = ep_start + epoch_hr
        if ep_end < t_hr_start or ep_start > t_hr_end:
            continue
        s = max(0, (ep_start - t_hr_start) * 60)
        e = min((t_hr_end - t_hr_start) * 60, (ep_end - t_hr_start) * 60)
        blocks.append((s, e, int(codes[i])))
    return blocks


# ── What stage follows a movement event? ─────────────────────────────────────

def stage_after_movement(sleep_profile, event_t_hr, lookahead_min=10):
    """Determine the dominant sleep stage in the window after a movement."""
    if sleep_profile is None:
        return -1, 'unknown'
    t_ep = sleep_profile['t_ep_hr']
    codes = sleep_profile['codes']
    lookahead_hr = lookahead_min / 60.0

    mask = (t_ep >= event_t_hr) & (t_ep <= event_t_hr + lookahead_hr)
    post_codes = codes[mask]
    if len(post_codes) == 0:
        return -1, 'unknown'

    # Check if any N3 epochs in the post window
    has_n3 = np.any(post_codes == 1)
    n3_frac = np.mean(post_codes == 1)

    if n3_frac > 0.3:
        return 1, f'N3 ({n3_frac:.0%})'
    # Dominant stage
    unique, counts = np.unique(post_codes[post_codes >= 0], return_counts=True)
    if len(unique) == 0:
        return -1, 'unknown'
    dom = unique[np.argmax(counts)]
    return int(dom), STAGE_LABELS.get(int(dom), '?')


# ── Main plotting ─────────────────────────────────────────────────────────────

def plot_settling_event(session, event_idx, event_num, total_events,
                        stage_after, stage_label,
                        card_gt=None, resp_gt=None):
    """Plot one settling event with all panels."""
    fs = session.fs
    n = session.n_samples

    lo = max(0, event_idx - WIN_SAMP)
    hi = min(n, event_idx + WIN_SAMP)
    sl = slice(lo, hi)

    t_s_abs_start = lo / fs
    t_s_abs_end = hi / fs
    t_hr_start = lo / fs / 3600
    t_hr_end = hi / fs / 3600

    # Time axis in minutes relative to movement event
    t_min = (np.arange(hi - lo) - (event_idx - lo)) / fs / 60.0

    # Signals
    acc = session.cap['acc_mag'][sl].astype(np.float64)
    cle = session.cap['CLE'][sl].astype(np.float64)
    cre = session.cap['CRE'][sl].astype(np.float64)
    diff = cle - cre
    ch = session.cap['CH'][sl].astype(np.float64)

    thorax = session.psg['Thorax'][sl].astype(np.float64)
    pleth = session.psg['Pleth'][sl].astype(np.float64)

    # DC mean: lowpass at 0.01 Hz
    diff_dc = lowpass(diff, 0.02, fs)

    # Thorax envelope: rolling RMS (5s window)
    env_win = int(5 * fs)
    thorax_env = np.array([
        np.sqrt(np.mean(thorax[max(0, i - env_win // 2):i + env_win // 2] ** 2))
        for i in range(0, len(thorax), env_win // 4)
    ])
    t_thorax_env = np.linspace(t_min[0], t_min[-1], len(thorax_env))

    # Pleth envelope
    pleth_bp = bandpass(pleth, CARD_LO, CARD_HI, fs)
    pleth_env = np.array([
        np.sqrt(np.mean(pleth_bp[max(0, i - env_win // 2):i + env_win // 2] ** 2))
        for i in range(0, len(pleth_bp), env_win // 4)
    ])
    t_pleth_env = np.linspace(t_min[0], t_min[-1], len(pleth_env))

    # CAP rates
    seg_diff = diff.copy()
    t_cap_resp, cap_resp_rate = sliding_rate_in_window(seg_diff, RESP_LO, RESP_HI, fs)
    t_cap_card, cap_card_rate = sliding_rate_in_window(seg_diff, CARD_LO, CARD_HI, fs)
    # Convert to minutes relative to event
    event_offset_s = (event_idx - lo) / fs
    t_cap_resp_min = (t_cap_resp - event_offset_s) / 60
    t_cap_card_min = (t_cap_card - event_offset_s) / 60

    # PSG rates
    t_psg_resp = t_psg_card = np.array([])
    psg_resp_rate = psg_card_rate = np.array([])
    if resp_gt is not None:
        t_psg_resp, psg_resp_rate = psg_rate_in_window(
            resp_gt.peak_times_s, t_s_abs_start, t_s_abs_end)
        t_psg_resp = (t_psg_resp - event_offset_s) / 60
    if card_gt is not None:
        t_psg_card, psg_card_rate = psg_rate_in_window(
            card_gt.peak_times_s, t_s_abs_start, t_s_abs_end)
        t_psg_card = (t_psg_card - event_offset_s) / 60

    # Sleep stage blocks
    stage_blocks = get_stage_blocks(session.sleep_profile, t_hr_start, t_hr_end)

    # ── Plot ──
    fig, axes = plt.subplots(7, 1, figsize=(18, 20), sharex=True,
                              gridspec_kw={'height_ratios': [0.5, 1, 1, 1, 1, 1, 1]})

    event_t_hr = event_idx / fs / 3600
    is_n3 = stage_after == 1
    title_color = '#2ECC71' if is_n3 else '#E74C3C'
    fig.suptitle(
        f'{session.label} - Movement Event {event_num}/{total_events} '
        f'(t={event_t_hr:.2f} hr) -- Post-settling: {stage_label}',
        fontsize=14, color=title_color, fontweight='bold',
    )

    # Panel 0: Sleep stages
    ax = axes[0]
    for s_start, s_end, s_code in stage_blocks:
        s_start_rel = s_start - WINDOW_MIN
        s_end_rel = s_end - WINDOW_MIN
        color = STAGE_COLORS.get(s_code, '#AAAAAA')
        alpha = 0.8 if s_code == 1 else 0.4
        ax.axvspan(s_start_rel, s_end_rel, color=color, alpha=alpha)
        mid = (s_start_rel + s_end_rel) / 2
        label = STAGE_LABELS.get(s_code, '?')
        if s_end_rel - s_start_rel > 0.3:
            ax.text(mid, 0.5, label, ha='center', va='center', fontsize=7,
                    fontweight='bold' if s_code == 1 else 'normal')
    ax.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_ylabel('Stage')
    ax.set_yticks([])
    ax.set_xlim(-WINDOW_MIN, WINDOW_MIN)

    # Panel 1: Accelerometer
    ax = axes[1]
    # Downsample for plotting
    ds = 10
    ax.plot(t_min[::ds], acc[::ds], color='#34495E', linewidth=0.3, alpha=0.6)
    # Rolling RMS overlay
    acc_rms_win = int(2 * fs)
    acc_rms = np.array([
        np.sqrt(np.mean((acc[max(0, i - acc_rms_win):i + acc_rms_win] -
                         np.mean(acc[max(0, i - acc_rms_win):i + acc_rms_win])) ** 2))
        for i in range(0, len(acc), acc_rms_win // 2)
    ])
    t_acc_rms = np.linspace(t_min[0], t_min[-1], len(acc_rms))
    ax.plot(t_acc_rms, acc_rms, color='#E74C3C', linewidth=1.5, label='RMS (2s)')
    ax.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax.set_ylabel('Accel')
    ax.legend(fontsize=8, loc='upper right')

    # Panel 2: CAP CLE-CRE with DC mean overlay
    ax = axes[2]
    ax.plot(t_min[::ds], diff[::ds], color='#BDC3C7', linewidth=0.2, alpha=0.5)
    ax.plot(t_min[::ds], diff_dc[::ds], color='#E67E22', linewidth=2, label='DC mean (0.02 Hz LP)')
    ax.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax.set_ylabel('CLE-CRE')
    ax.legend(fontsize=8, loc='upper right')

    # Panel 3: Thorax envelope
    ax = axes[3]
    ax.plot(t_min[::ds], thorax[::ds], color='#BDC3C7', linewidth=0.2, alpha=0.4)
    ax.plot(t_thorax_env, thorax_env, color='#8E44AD', linewidth=1.5, label='Thorax RMS (5s)')
    ax.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax.set_ylabel('Thorax')
    ax.legend(fontsize=8, loc='upper right')

    # Panel 4: Respiratory rates (CAP vs PSG)
    ax = axes[4]
    valid_cap = np.isfinite(cap_resp_rate)
    ax.plot(t_cap_resp_min[valid_cap], cap_resp_rate[valid_cap] * 60,
            color='#E67E22', linewidth=1.5, label='CAP resp (br/min)', alpha=0.8)
    if len(psg_resp_rate) > 0:
        valid_psg = np.isfinite(psg_resp_rate)
        ax.plot(t_psg_resp[valid_psg], psg_resp_rate[valid_psg] * 60,
                color='#2980B9', linewidth=1.5, label='PSG resp (br/min)', alpha=0.8)
    ax.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax.set_ylabel('Resp Rate\n(br/min)')
    ax.set_ylim(5, 30)
    ax.legend(fontsize=8, loc='upper right')

    # Panel 5: Cardiac rates (CAP vs PSG)
    ax = axes[5]
    valid_cap = np.isfinite(cap_card_rate)
    ax.plot(t_cap_card_min[valid_cap], cap_card_rate[valid_cap] * 60,
            color='#E67E22', linewidth=1.5, label='CAP cardiac (BPM)', alpha=0.8)
    if len(psg_card_rate) > 0:
        valid_psg = np.isfinite(psg_card_rate)
        ax.plot(t_psg_card[valid_psg], psg_card_rate[valid_psg] * 60,
                color='#C0392B', linewidth=1.5, label='ECG HR (BPM)', alpha=0.8)
    ax.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax.set_ylabel('Heart Rate\n(BPM)')
    ax.set_ylim(30, 120)
    ax.legend(fontsize=8, loc='upper right')

    # Panel 6: Pleth (PPG) envelope
    ax = axes[6]
    ax.plot(t_pleth_env, pleth_env, color='#1ABC9C', linewidth=1.5, label='PPG envelope')
    ax.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax.set_ylabel('PPG\namplitude')
    ax.set_xlabel('Time relative to movement (minutes)')
    ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _post_movement_thorax_rms(session, event_idx, post_sec=300):
    """Compute thorax RMS in the post-movement window (default 5 min)."""
    fs = session.fs
    lo = event_idx
    hi = min(session.n_samples, event_idx + int(post_sec * fs))
    thorax = session.psg['Thorax'][lo:hi].astype(np.float64)
    return float(np.sqrt(np.mean(thorax ** 2)))


def process_session(session_idx, max_events=None):
    """Process one session: detect movements, plot ALL events sorted by thorax RMS."""
    session = load_session(session_idx)
    session.sleep_profile = load_sleep_profile(session)

    print(f"\n{'='*60}")
    print(f"Processing {session.label}")
    print(f"{'='*60}")

    acc = session.cap['acc_mag'].astype(np.float64)
    events, rms, threshold = detect_movement_events(acc)
    print(f"  Detected {len(events)} movement events (threshold={threshold:.4f})")

    margin = WIN_SAMP
    events = events[(events > margin) & (events < session.n_samples - margin)]
    print(f"  {len(events)} events after margin filtering")

    if len(events) == 0:
        print("  No valid events found")
        return

    # PSG ground truth rates
    print("  Computing PSG ground truth rates...")
    from sleep_monitor.ground_truth import gt_heart_rate, gt_resp_rate
    try:
        card_gt = gt_heart_rate(session)
        print(f"    Cardiac GT: {len(card_gt.peak_indices)} peaks from {card_gt.signal_used}")
    except Exception as e:
        print(f"    Cardiac GT failed: {e}")
        card_gt = None
    try:
        resp_gt = gt_resp_rate(session)
        print(f"    Resp GT: {len(resp_gt.peak_indices)} peaks from {resp_gt.signal_used}")
    except Exception as e:
        print(f"    Resp GT failed: {e}")
        resp_gt = None

    # Compute post-movement thorax RMS for each event and sort
    print("  Computing post-movement thorax RMS for sorting...")
    event_info = []
    for ev in events:
        t_hr = ev / session.fs / 3600
        stage_code, stage_label = stage_after_movement(session.sleep_profile, t_hr)
        thorax_rms = _post_movement_thorax_rms(session, ev)
        event_info.append((ev, stage_code, stage_label, thorax_rms))

    # Sort by thorax RMS (lowest first = most stable thorax)
    event_info.sort(key=lambda x: x[3])

    if max_events is not None:
        event_info = event_info[:max_events]

    print(f"  Plotting {len(event_info)} events (ordered by thorax RMS, low->high)...")

    # Save to subfolder per session
    session_dir = REPORT_DIR / 'settling_events' / session.label
    session_dir.mkdir(parents=True, exist_ok=True)

    for i, (ev, stage_code, stage_label, thorax_rms) in enumerate(event_info):
        fig = plot_settling_event(
            session, ev, i + 1, len(event_info),
            stage_code, stage_label,
            card_gt=card_gt, resp_gt=resp_gt,
        )
        out = session_dir / f'event_{i+1:03d}_thoraxRMS_{thorax_rms:.1f}.png'
        fig.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    {i+1}/{len(event_info)} saved (thorax RMS={thorax_rms:.1f})")

    print(f"  Done: {len(event_info)} plots saved to {session_dir}")


def main():
    print("Post-Movement Settling Event Visualization")
    print("All events, ordered by post-movement thorax RMS (low = stable)")
    print("=" * 60)

    for idx in range(12):
        try:
            process_session(idx)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
