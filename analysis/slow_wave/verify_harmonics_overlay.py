"""
Harmonic detection verification — overlay detected peaks on spectrograms + PSDs.

For a chosen session, plot:
  Figure 1: Spectrogram per channel with OLD per-window detections (cyan dots).
  Figure 2: Spectrogram per channel with NEW persistent ridges (coloured lines).
  Figure 3: Grid of PSD windows with detected peaks marked.
  Figure 4: All channels side-by-side at peak-harmonic times.
  Figure 5: Smoothed vs raw PSD comparison at selected times.

Usage:
    python verify_harmonics_overlay.py              # defaults to S1N1
    python verify_harmonics_overlay.py S3N2          # specific session
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import welch, spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_all_sessions, FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import _explicit_harmonics, _hps, _cepstral
from sleep_monitor.harmonics import detect_persistent_ridges

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE', 'acc_mag']
CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD', 'acc_mag': '#E67E22'}
STAGE_CODE_MAP = {4: 'Wake', 3: 'N1', 2: 'N2', 1: 'N3', 0: 'REM'}

# Detector parameters (must match what run_harmonic_allsessions.py uses)
WIN_SEC = 30.0
STEP_SEC = 30.0
F0_RANGE = (0.1, 0.8)
MAX_HARMONICS = 6
F_TOLERANCE = 0.05
WELCH_SEG_SEC = 8.0
MIN_PROMINENCE = 0.1
SPEC_MAX_FREQ = 5.0


def load_session_by_label(label):
    sessions = load_all_sessions(with_sleep_profiles=True)
    for s in sessions:
        if s.label == label:
            return s
    raise ValueError(f'Session {label} not found. Available: {[s.label for s in sessions]}')


def prepare_signals(session):
    acc_mag = session.cap['acc_mag']
    signals = {}
    for ch in CHANNELS:
        if ch == 'acc_mag':
            signals[ch] = acc_mag.astype(np.float64)
        else:
            signals[ch] = remove_acc_artifact(session.cap[ch], acc_mag, 0.05, 4.0)
    return signals, acc_mag


def run_detector_per_channel(signals, acc_mag):
    """Run harmonic detector on each channel, returning per-window PSD + detections."""
    win_n = int(WIN_SEC * FS)
    step_n = int(STEP_SEC * FS)
    nperseg = min(int(WELCH_SEG_SEC * FS), win_n)

    results = {}
    for ch in CHANNELS:
        sig = signals[ch]
        n = len(sig)
        starts = np.arange(0, n - win_n + 1, step_n)

        # Motion mask
        acc = acc_mag.astype(np.float64)
        motion_rms = np.array([
            np.sqrt(np.mean((acc[s0:s0+win_n] - np.mean(acc[s0:s0+win_n]))**2))
            for s0 in starts
        ])
        med = np.median(motion_rms)
        mad = np.median(np.abs(motion_rms - med)) + 1e-12
        motion_mask = motion_rms > (med + 3.0 * mad)

        windows = []
        for i, s0 in enumerate(starts):
            t_hr = (s0 + win_n / 2) / FS / 3600.0

            if motion_mask[i]:
                windows.append(dict(
                    t_hr=t_hr, motion=True,
                    freqs=None, psd=None,
                    f0=np.nan, harmonics=[], n_harmonics=0,
                    her=np.nan, hps_f0=np.nan, cep_f0=np.nan,
                ))
                continue

            chunk = sig[s0:s0+win_n].astype(np.float64)
            freqs, psd = welch(chunk, fs=FS, nperseg=nperseg,
                               noverlap=nperseg // 2, scaling='density')

            explicit = _explicit_harmonics(
                psd, freqs, F0_RANGE,
                max_harmonics=MAX_HARMONICS,
                f_tolerance=F_TOLERANCE,
                min_prominence=MIN_PROMINENCE,
            )
            hps_f0, _ = _hps(psd, freqs, F0_RANGE)
            cep_f0, _ = _cepstral(psd, freqs, F0_RANGE)

            # Build list of confirmed harmonic frequencies
            f0 = explicit['f0_hz']
            harm_freqs = [f0]
            for k_idx, amp in enumerate(explicit['per_harmonic_amps'][1:], start=2):
                if amp > 0:
                    harm_freqs.append(k_idx * f0)

            windows.append(dict(
                t_hr=t_hr, motion=False,
                freqs=freqs, psd=psd,
                f0=f0, harmonics=harm_freqs,
                n_harmonics=explicit['n_harmonics'],
                her=explicit['harmonic_energy_ratio'],
                hps_f0=hps_f0, cep_f0=cep_f0,
            ))

        results[ch] = windows
    return results


def figure1_spectrogram_overlay(session, signals, detections, acc_mag):
    """Spectrogram per channel with detected harmonics overlaid."""
    sp = session.sleep_profile
    fig, axes = plt.subplots(len(CHANNELS) + 1, 1, figsize=(16, 3 + 3*len(CHANNELS)),
                             gridspec_kw={'height_ratios': [1] + [3]*len(CHANNELS)},
                             sharex=True)

    # Row 0: hypnogram bar
    ax_hyp = axes[0]
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax_hyp.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j+1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.6)
    ax_hyp.set_yticks([])
    ax_hyp.set_ylabel('Stage', fontsize=8)
    patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
               for c in STAGE_ORDER]
    ax_hyp.legend(handles=patches, loc='upper right', fontsize=6, ncol=5)
    ax_hyp.set_title(f'{session.label} — Spectrogram + Detected Harmonics', fontsize=12)

    for idx, ch in enumerate(CHANNELS):
        ax = axes[idx + 1]
        sig = signals[ch]

        # Compute spectrogram
        nperseg_spec = int(10.0 * FS)
        f, t, Sxx = spectrogram(sig.astype(np.float64), fs=FS,
                                nperseg=nperseg_spec,
                                noverlap=nperseg_spec // 2,
                                scaling='density')
        t_hr = t / 3600.0

        # Limit to SPEC_MAX_FREQ
        f_mask = f <= SPEC_MAX_FREQ
        ax.pcolormesh(t_hr, f[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-30),
                      shading='gouraud', cmap='inferno', vmin=-40, vmax=20)

        # Overlay detected harmonics
        wins = detections[ch]
        for w in wins:
            if w['motion'] or len(w['harmonics']) == 0:
                continue
            for h_freq in w['harmonics']:
                if h_freq <= SPEC_MAX_FREQ:
                    ax.plot(w['t_hr'], h_freq, 'o',
                            color='cyan', markersize=2.5, alpha=0.7,
                            markeredgewidth=0)

        # Overlay f0 track as a line
        valid_wins = [w for w in wins if not w['motion'] and np.isfinite(w['f0'])]
        if valid_wins:
            t_hrs = [w['t_hr'] for w in valid_wins]
            f0s = [w['f0'] for w in valid_wins]
            ax.plot(t_hrs, f0s, '-', color='lime', lw=0.8, alpha=0.6, label='f0 track')

        ax.set_ylabel(f'{ch}\nFreq (Hz)', fontsize=8)
        ax.set_ylim(0, SPEC_MAX_FREQ)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

    axes[-1].set_xlabel('Time (hr)', fontsize=9)
    fig.tight_layout()
    return fig


def figure2_psd_windows(session, detections, n_examples=12):
    """Grid of PSD plots for selected windows with harmonic peaks marked."""
    sp = session.sleep_profile

    # Pick windows spread across the night, preferring high-HER windows
    # from different stages, plus a few low-HER for contrast
    all_wins = []
    for ch in CHANNELS:
        for w in detections[ch]:
            if not w['motion'] and np.isfinite(w['her']):
                all_wins.append((ch, w))

    if not all_wins:
        print('  No valid windows found!')
        return None

    # Sort by HER descending, pick top windows spread across time
    all_wins.sort(key=lambda x: x[1]['her'], reverse=True)

    # Take top HER windows but ensure time diversity
    selected = []
    used_times = set()
    # First pass: high HER, one per ~30min slot
    for ch, w in all_wins:
        slot = round(w['t_hr'] * 2)  # 30-min slots
        key = (ch, slot)
        if key not in used_times and len(selected) < n_examples * 2:
            selected.append((ch, w))
            used_times.add(key)

    # Take top n_examples//2 (high HER) + bottom n_examples//2 (low HER)
    high_her = selected[:n_examples // 2]
    # Get low-HER windows
    low_wins = [(ch, w) for ch, w in all_wins if w['her'] < 0.3 and not w['motion']]
    low_wins.sort(key=lambda x: x[1]['t_hr'])
    low_her = low_wins[::max(1, len(low_wins) // (n_examples // 2))][:n_examples // 2]

    picks = high_her + low_her
    picks.sort(key=lambda x: x[1]['t_hr'])

    if len(picks) == 0:
        print('  No suitable windows found for PSD grid!')
        return None

    n_cols = 4
    n_rows = (len(picks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (ch, w) in enumerate(picks):
        ax = axes[i // n_cols, i % n_cols]
        freqs = w['freqs']
        psd = w['psd']

        # Find sleep stage at this time
        ep_idx = np.argmin(np.abs(sp['t_ep_hr'] - w['t_hr']))
        stage = STAGE_CODE_MAP.get(int(sp['codes'][ep_idx]), '?')

        # Plot PSD
        f_mask = freqs <= SPEC_MAX_FREQ
        ax.semilogy(freqs[f_mask], psd[f_mask], color=CH_COLORS[ch], lw=1.0, alpha=0.9)

        # Mark detected harmonics
        for h_idx, h_freq in enumerate(w['harmonics']):
            if h_freq <= SPEC_MAX_FREQ:
                # Find nearest PSD value
                f_idx = np.argmin(np.abs(freqs - h_freq))
                label = f'f0={h_freq:.2f}' if h_idx == 0 else f'{h_idx+1}×f0'
                ax.axvline(h_freq, color='red', lw=0.8, alpha=0.6, ls='--')
                ax.plot(freqs[f_idx], psd[f_idx], 'rv', markersize=6, alpha=0.8)
                ax.text(h_freq, ax.get_ylim()[1] * 0.5, label,
                        fontsize=5, rotation=90, va='top', ha='right', color='red')

        # Mark HPS and cepstral f0 estimates for comparison
        if np.isfinite(w['hps_f0']) and w['hps_f0'] <= SPEC_MAX_FREQ:
            ax.axvline(w['hps_f0'], color='blue', lw=0.6, alpha=0.4, ls=':')
        if np.isfinite(w['cep_f0']) and w['cep_f0'] <= SPEC_MAX_FREQ:
            ax.axvline(w['cep_f0'], color='green', lw=0.6, alpha=0.4, ls=':')

        ax.set_title(f'{ch} t={w["t_hr"]:.2f}h  {stage}\n'
                     f'HER={w["her"]:.2f}  n_harm={w["n_harmonics"]}',
                     fontsize=7, fontweight='bold')
        ax.set_xlim(0, SPEC_MAX_FREQ)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if i % n_cols == 0:
            ax.set_ylabel('PSD', fontsize=7)
        if i >= len(picks) - n_cols:
            ax.set_xlabel('Freq (Hz)', fontsize=7)

    # Hide unused axes
    for j in range(len(picks), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', ls='--', lw=1, label='Detected harmonic'),
        plt.Line2D([0], [0], color='blue', ls=':', lw=1, label='HPS f0'),
        plt.Line2D([0], [0], color='green', ls=':', lw=1, label='Cepstral f0'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=3)
    fig.suptitle(f'{session.label} — PSD Windows with Detected Harmonics '
                 f'(red ▼ = confirmed peaks)', fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def figure3_psd_allchannels_panel(session, detections, n_times=6):
    """For selected time points, show all 3 channels side-by-side so user can
    compare which channel shows harmonics and whether they agree."""
    sp = session.sleep_profile

    # Pick time points with highest HER across any channel
    time_her = {}
    for ch in CHANNELS:
        for w in detections[ch]:
            if not w['motion'] and np.isfinite(w['her']):
                t = round(w['t_hr'], 4)
                if t not in time_her or w['her'] > time_her[t]:
                    time_her[t] = w['her']

    sorted_times = sorted(time_her.keys(), key=lambda t: time_her[t], reverse=True)

    # Pick n_times spread across the night (not all from same cluster)
    selected_times = []
    for t in sorted_times:
        if all(abs(t - st) > 0.25 for st in selected_times):
            selected_times.append(t)
        if len(selected_times) >= n_times:
            break
    selected_times.sort()

    fig, axes = plt.subplots(len(selected_times), len(CHANNELS),
                             figsize=(15, 3 * len(selected_times)),
                             squeeze=False)

    for row, target_t in enumerate(selected_times):
        ep_idx = np.argmin(np.abs(sp['t_ep_hr'] - target_t))
        stage = STAGE_CODE_MAP.get(int(sp['codes'][ep_idx]), '?')

        for col, ch in enumerate(CHANNELS):
            ax = axes[row, col]
            # Find closest window
            wins = detections[ch]
            valid = [(abs(w['t_hr'] - target_t), w) for w in wins if not w['motion']]
            if not valid:
                ax.text(0.5, 0.5, 'motion', transform=ax.transAxes,
                        ha='center', fontsize=10, color='gray')
                continue
            _, w = min(valid, key=lambda x: x[0])

            freqs = w['freqs']
            psd = w['psd']
            f_mask = freqs <= SPEC_MAX_FREQ

            ax.semilogy(freqs[f_mask], psd[f_mask], color=CH_COLORS[ch], lw=1.2)

            for h_idx, h_freq in enumerate(w['harmonics']):
                if h_freq <= SPEC_MAX_FREQ:
                    f_idx = np.argmin(np.abs(freqs - h_freq))
                    ax.axvline(h_freq, color='red', lw=0.8, alpha=0.5, ls='--')
                    ax.plot(freqs[f_idx], psd[f_idx], 'rv', markersize=5)

            ax.set_xlim(0, SPEC_MAX_FREQ)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(ch, fontsize=10, fontweight='bold', color=CH_COLORS[ch])
            if col == 0:
                ax.set_ylabel(f't={target_t:.2f}h\n{stage}', fontsize=8, fontweight='bold')
                info = f'HER={w["her"]:.2f} n={w["n_harmonics"]}'
                ax.text(0.98, 0.95, info, transform=ax.transAxes,
                        fontsize=6, ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            else:
                info = f'HER={w["her"]:.2f} n={w["n_harmonics"]}'
                ax.text(0.98, 0.95, info, transform=ax.transAxes,
                        fontsize=6, ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    fig.suptitle(f'{session.label} — All Channels Side-by-Side at Peak-Harmonic Times',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def figure_ridges_spectrogram(session, signals, ridge_results, acc_mag):
    """Spectrogram per channel with persistent ridges overlaid as coloured lines."""
    sp = session.sleep_profile
    fig, axes = plt.subplots(len(CHANNELS) + 1, 1, figsize=(16, 3 + 3*len(CHANNELS)),
                             gridspec_kw={'height_ratios': [1] + [3]*len(CHANNELS)},
                             sharex=True)

    # Hypnogram bar
    ax_hyp = axes[0]
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax_hyp.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j+1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.6)
    ax_hyp.set_yticks([])
    ax_hyp.set_ylabel('Stage', fontsize=8)
    patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
               for c in STAGE_ORDER]
    ax_hyp.legend(handles=patches, loc='upper right', fontsize=6, ncol=5)
    ax_hyp.set_title(f'{session.label} — Persistent Ridges on Spectrogram '
                     f'(min {MIN_PERSIST_SEC}s, smoothed over {SMOOTH_WINDOWS} windows)',
                     fontsize=12)

    ridge_cmap = plt.cm.tab20
    for idx, ch in enumerate(CHANNELS):
        ax = axes[idx + 1]
        sig = signals[ch]

        # Spectrogram
        nperseg_spec = int(10.0 * FS)
        f, t, Sxx = spectrogram(sig.astype(np.float64), fs=FS,
                                nperseg=nperseg_spec,
                                noverlap=nperseg_spec // 2,
                                scaling='density')
        t_hr_spec = t / 3600.0
        f_mask = f <= SPEC_MAX_FREQ
        ax.pcolormesh(t_hr_spec, f[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-30),
                      shading='gouraud', cmap='inferno', vmin=-40, vmax=20)

        # Overlay persistent ridges
        rr = ridge_results[ch]
        ridges = rr['ridges']
        t_hr = rr['t_hr']
        for ri, ridge in enumerate(ridges):
            color = ridge_cmap(ri % 20)
            valid = ~np.isnan(ridge['freq_trace'])
            ax.plot(t_hr[valid], ridge['freq_trace'][valid], '-',
                    color=color, lw=1.8, alpha=0.85)
            # Label at midpoint
            mid = np.where(valid)[0]
            if len(mid) > 0:
                mi = mid[len(mid) // 2]
                ax.text(t_hr[mi], ridge['freq_trace'][mi] + 0.08,
                        f'{ridge["median_freq"]:.2f}Hz ({ridge["duration_sec"]/60:.0f}m)',
                        fontsize=5, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5))

        # Mark harmonic groups with brackets
        groups = rr['harmonic_groups']
        for gi, grp in enumerate(groups):
            member_freqs = [ridges[mi]['median_freq'] for mi in grp['harmonic_idxs']]
            label = f'f0={grp["f0_median"]:.2f} ({len(grp["harmonic_idxs"])} ridges)'
            ax.text(0.01, 0.97 - gi * 0.06, label,
                    transform=ax.transAxes, fontsize=6, color='white',
                    fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#333', alpha=0.7))

        ax.set_ylabel(f'{ch}\nFreq (Hz)', fontsize=8)
        ax.set_ylim(0, SPEC_MAX_FREQ)
        ax.tick_params(labelsize=7)

        n_ridges = len(ridges)
        n_groups = len(groups)
        ax.text(0.99, 0.97, f'{n_ridges} ridges, {n_groups} harmonic groups',
                transform=ax.transAxes, fontsize=7, ha='right', va='top',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='#333', alpha=0.7))

    axes[-1].set_xlabel('Time (hr)', fontsize=9)
    fig.tight_layout()
    return fig


def figure_smoothed_vs_raw_psd(session, ridge_results, n_times=6):
    """Side-by-side raw vs smoothed PSD at selected times, with ridges marked."""
    sp = session.sleep_profile

    # Pick times where ridges are most active
    ch = 'CH'
    rr = ridge_results[ch]
    t_hr = rr['t_hr']
    ridges = rr['ridges']

    ridge_count = np.zeros(len(t_hr))
    for ridge in ridges:
        valid = ~np.isnan(ridge['freq_trace'])
        ridge_count[valid] += 1

    # Pick windows with most concurrent ridges, spread across time
    order = np.argsort(-ridge_count)
    selected = []
    for idx in order:
        if rr['motion_mask'][idx]:
            continue
        if all(abs(t_hr[idx] - t_hr[s]) > 0.3 for s in selected):
            selected.append(idx)
        if len(selected) >= n_times:
            break
    selected.sort()

    if not selected:
        return None

    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 3 * len(selected)),
                             squeeze=False)

    for row, wi in enumerate(selected):
        ep_idx = np.argmin(np.abs(sp['t_ep_hr'] - t_hr[wi]))
        stage = STAGE_CODE_MAP.get(int(sp['codes'][ep_idx]), '?')
        freqs = rr['freqs']

        for col, (psd_arr, title) in enumerate([
            (rr['psds'][wi], 'Raw PSD'),
            (rr['psds_smooth'][wi], f'Smoothed PSD ({SMOOTH_WINDOWS}-win median)'),
        ]):
            ax = axes[row, col]
            if np.all(np.isnan(psd_arr)):
                ax.text(0.5, 0.5, 'NaN', transform=ax.transAxes, ha='center')
                continue

            ax.semilogy(freqs, psd_arr, color=CH_COLORS[ch], lw=1.0)

            # Mark active ridges at this window
            for ri, ridge in enumerate(ridges):
                f_val = ridge['freq_trace'][wi]
                if np.isfinite(f_val):
                    f_idx = np.argmin(np.abs(freqs - f_val))
                    ax.axvline(f_val, color='red', lw=0.8, alpha=0.5, ls='--')
                    ax.plot(freqs[f_idx], psd_arr[f_idx], 'rv', markersize=6)
                    ax.text(f_val + 0.03, psd_arr[f_idx],
                            f'{f_val:.2f}Hz', fontsize=5, color='red')

            ax.set_xlim(0, SPEC_MAX_FREQ)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f't={t_hr[wi]:.2f}h  {stage}\n'
                              f'{int(ridge_count[wi])} ridges',
                              fontsize=7, fontweight='bold')

    fig.suptitle(f'{session.label} CH — Raw vs Smoothed PSD with Persistent Ridges',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── Tunable constants for persistent ridge detection ──────────────────────
SMOOTH_WINDOWS = 7
MIN_PERSIST_SEC = 300.0
MAX_FREQ_JUMP = 0.08

# ── Main ──────────────────────────────────────────────────────────────────────

def process_one_session(session):
    """Run all 5 figures for a single session."""
    label = session.label
    print(f'\n{"="*60}')
    print(f'{label} ({session.subject}, {session.duration_hr:.1f} hr)')
    print(f'{"="*60}')

    print('  Preparing signals (accel artifact removal)...')
    signals, acc_mag = prepare_signals(session)

    # ── Old per-window detector ──
    print('  Running per-window harmonic detector...')
    detections = run_detector_per_channel(signals, acc_mag)

    for ch in CHANNELS:
        valid = [w for w in detections[ch] if not w['motion']]
        hers = [w['her'] for w in valid if np.isfinite(w['her'])]
        print(f'    {ch}: {len(valid)} valid windows, '
              f'median HER={np.median(hers):.3f}, '
              f'mean n_harmonics={np.mean([w["n_harmonics"] for w in valid]):.1f}')

    # ── New persistent ridge detector ──
    print('  Running persistent ridge detector...')
    ridge_results = {}
    for ch in CHANNELS:
        rr = detect_persistent_ridges(
            signals[ch], fs=FS,
            win_sec=WIN_SEC, step_sec=STEP_SEC,
            max_freq=SPEC_MAX_FREQ,
            smooth_windows=SMOOTH_WINDOWS,
            min_persistence_sec=MIN_PERSIST_SEC,
            max_freq_jump=MAX_FREQ_JUMP,
            peak_prominence_frac=0.5,
            max_gap_windows=5,
            acc_mag=None if ch == 'acc_mag' else acc_mag,
        )
        ridge_results[ch] = rr
        n_r = len(rr['ridges'])
        n_g = len(rr['harmonic_groups'])
        print(f'    {ch}: {n_r} ridges, {n_g} harmonic groups')

    sess_dir = REPORT_DIR / 'harmonics_overlay' / label
    sess_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Spectrogram + old per-window detections (cyan dots) ──
    print('  Fig 1: spectrogram + per-window detections...')
    fig1 = figure1_spectrogram_overlay(session, signals, detections, acc_mag)
    p1 = sess_dir / f'{label}_spectrogram_overlay.png'
    fig1.savefig(p1, dpi=200, bbox_inches='tight')
    plt.close(fig1)

    # ── Figure 2: Spectrogram + persistent ridges (coloured lines) ──
    print('  Fig 2: spectrogram + persistent ridges...')
    fig2 = figure_ridges_spectrogram(session, signals, ridge_results, acc_mag)
    p2 = sess_dir / f'{label}_ridges_overlay.png'
    fig2.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # ── Figure 3: PSD windows with detected peaks ──
    print('  Fig 3: PSD windows...')
    fig3 = figure2_psd_windows(session, detections)
    if fig3:
        p3 = sess_dir / f'{label}_psd_windows.png'
        fig3.savefig(p3, dpi=200, bbox_inches='tight')
        plt.close(fig3)

    # ── Figure 4: All channels side-by-side ──
    print('  Fig 4: channels side-by-side...')
    fig4 = figure3_psd_allchannels_panel(session, detections)
    if fig4:
        p4 = sess_dir / f'{label}_channels_sidebyside.png'
        fig4.savefig(p4, dpi=200, bbox_inches='tight')
        plt.close(fig4)

    # ── Figure 5: Smoothed vs raw PSD comparison ──
    print('  Fig 5: smoothed vs raw PSD...')
    fig5 = figure_smoothed_vs_raw_psd(session, ridge_results)
    if fig5:
        p5 = sess_dir / f'{label}_smoothed_vs_raw.png'
        fig5.savefig(p5, dpi=200, bbox_inches='tight')
        plt.close(fig5)

    plt.close('all')
    print(f'  Done -> {sess_dir}')


if __name__ == '__main__':
    from sleep_monitor import load_all_sessions

    if len(sys.argv) > 1:
        labels = sys.argv[1:]
    else:
        labels = None  # all sessions

    print('Loading sessions...')
    sessions = load_all_sessions(with_sleep_profiles=True)

    if labels is not None:
        sessions = [s for s in sessions if s.label in labels]
        if not sessions:
            print(f'No sessions matched {labels}')
            sys.exit(1)

    for session in sessions:
        process_one_session(session)

    print(f'\nAll done. {len(sessions)} sessions processed.')
