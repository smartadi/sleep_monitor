"""
Spectral peak tracker — auto-detect all spectral peaks per window, track
persistent ridges across time, overlay on CAP spectrograms for all sessions.

Unlike the harmonic detector (Stage 1-2) which assumed integer-multiple
relationships, this finds ALL significant peaks and lets structure emerge
from the data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.signal import welch, find_peaks, spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_all_sessions, FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.viz import plot_hypnogram

PLOT_DIR = Path(__file__).resolve().parents[2] / 'notebooks' / 'plots' / 'harmonics'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CAP_CHS = ['CH', 'CLE', 'CRE']
CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}
F_MAX = 5.0          # Hz — max frequency to display / search
WIN_SEC = 30.0       # sliding window
STEP_SEC = 30.0      # step (non-overlapping epochs)
WELCH_SEG = 10.0     # Welch sub-segment for smoother PSD
MIN_PROM_FRAC = 0.15 # peak must be ≥ 15% of max PSD in window
RIDGE_TOL_HZ = 0.08  # max freq drift between adjacent windows for ridge linking
MIN_RIDGE_LEN = 4    # minimum windows a ridge must span to be plotted


# ── Peak detection per window ───────────────────────────────────────────────

def detect_peaks_window(psd, freqs, f_max, min_prom_frac, min_distance_hz=0.08):
    """Find all significant spectral peaks in [0, f_max] Hz."""
    f_mask = freqs <= f_max
    psd_r = psd[f_mask]
    freqs_r = freqs[f_mask]

    if len(psd_r) < 5:
        return np.array([]), np.array([])

    df = freqs_r[1] - freqs_r[0]
    min_dist_bins = max(1, int(min_distance_hz / df))
    prom_thresh = min_prom_frac * np.max(psd_r)

    idxs, props = find_peaks(psd_r, prominence=prom_thresh,
                             distance=min_dist_bins)

    if len(idxs) == 0:
        return np.array([]), np.array([])

    return freqs_r[idxs], psd_r[idxs]


# ── Ridge tracker ───────────────────────────────────────────────────────────

def track_ridges(all_peaks, t_centers, tol_hz, min_len):
    """
    Link detected peaks across time into ridges (persistent frequency tracks).

    Parameters
    ----------
    all_peaks : list of arrays, each element = peak frequencies for that window
    t_centers : array of window centre times (hours)
    tol_hz    : max frequency jump between adjacent windows
    min_len   : minimum number of windows for a ridge to be kept

    Returns
    -------
    list of dicts: {t_hr: [], freq_hz: [], amp: []} for each ridge
    """
    active_ridges = []    # each: {t_hr: [], freq_hz: [], last_freq: float}
    finished_ridges = []

    for wi, (peaks_f, t_hr) in enumerate(zip(all_peaks, t_centers)):
        used = set()

        # try to extend existing ridges
        for ridge in active_ridges:
            if len(peaks_f) == 0:
                continue
            diffs = np.abs(peaks_f - ridge['last_freq'])
            best_idx = np.argmin(diffs)
            if diffs[best_idx] <= tol_hz and best_idx not in used:
                ridge['t_hr'].append(float(t_hr))
                ridge['freq_hz'].append(float(peaks_f[best_idx]))
                ridge['last_freq'] = float(peaks_f[best_idx])
                used.add(best_idx)
            # if no match, ridge will be closed below

        # close ridges that weren't extended
        still_active = []
        for ridge in active_ridges:
            if len(ridge['t_hr']) > 0 and ridge['t_hr'][-1] == float(t_hr):
                still_active.append(ridge)
            else:
                finished_ridges.append(ridge)
        active_ridges = still_active

        # start new ridges from unmatched peaks
        for pi in range(len(peaks_f)):
            if pi not in used:
                active_ridges.append({
                    't_hr': [float(t_hr)],
                    'freq_hz': [float(peaks_f[pi])],
                    'last_freq': float(peaks_f[pi]),
                })

    finished_ridges.extend(active_ridges)
    return [r for r in finished_ridges if len(r['t_hr']) >= min_len]


# ── Process one session ─────────────────────────────────────────────────────

def process_session(s):
    """Run peak detection + ridge tracking on all 3 CAP channels."""
    acc_mag = s.cap['acc_mag']
    win_n = int(WIN_SEC * FS)
    step_n = int(STEP_SEC * FS)

    results = {}
    for ch in CAP_CHS:
        sig = remove_acc_artifact(s.cap[ch], acc_mag, 0.05, 4.0)
        n = len(sig)
        starts = np.arange(0, n - win_n + 1, step_n)
        nperseg = min(int(WELCH_SEG * FS), win_n)

        t_centers = (starts + win_n / 2) / FS / 3600.0
        all_peaks_f = []
        all_peaks_a = []

        for s0 in starts:
            chunk = sig[s0:s0 + win_n].astype(np.float64)
            freqs, psd = welch(chunk, fs=FS, nperseg=nperseg,
                               noverlap=nperseg // 2, scaling='density')
            pf, pa = detect_peaks_window(psd, freqs, F_MAX, MIN_PROM_FRAC)
            all_peaks_f.append(pf)
            all_peaks_a.append(pa)

        ridges = track_ridges(all_peaks_f, t_centers, RIDGE_TOL_HZ, MIN_RIDGE_LEN)

        # build spectrogram for plotting
        f_spec, t_spec, Sxx = spectrogram(
            sig.astype(np.float64), fs=FS,
            nperseg=int(WELCH_SEG * FS),
            noverlap=int(WELCH_SEG * FS) // 2,
            scaling='density',
        )
        f_mask = f_spec <= F_MAX
        f_spec = f_spec[f_mask]
        Sxx = Sxx[f_mask, :]
        t_spec_hr = t_spec / 3600.0

        results[ch] = dict(
            t_centers=t_centers,
            all_peaks_f=all_peaks_f,
            all_peaks_a=all_peaks_a,
            ridges=ridges,
            f_spec=f_spec, t_spec_hr=t_spec_hr, Sxx=Sxx,
        )
    return results


# ── Plot one session ────────────────────────────────────────────────────────

def plot_session(s, results, out_path):
    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True,
                             gridspec_kw={'height_ratios': [0.6, 1.5, 1.5, 1.5, 1.2]})

    # Row 0: Hypnogram
    sp = s.sleep_profile
    plot_hypnogram(sp, axes[0], title='')
    axes[0].set_ylabel('Stage', fontsize=8)
    axes[0].set_title(f'{s.label} ({s.subject})', fontsize=10, fontweight='bold', loc='left')

    # Rows 1-3: Spectrograms + detected peaks
    for i, ch in enumerate(CAP_CHS):
        ax = axes[i + 1]
        r = results[ch]

        # spectrogram
        Sxx_db = 10 * np.log10(r['Sxx'] + 1e-30)
        vmin, vmax = np.percentile(Sxx_db, [5, 95])
        ax.pcolormesh(r['t_spec_hr'], r['f_spec'], Sxx_db,
                      shading='gouraud', cmap='inferno',
                      vmin=vmin, vmax=vmax, rasterized=True)

        # scatter all detected peaks
        for wi, (pf, t_hr) in enumerate(zip(r['all_peaks_f'], r['t_centers'])):
            if len(pf) > 0:
                ax.scatter(np.full(len(pf), t_hr), pf,
                           s=4, c='cyan', alpha=0.4, linewidths=0, zorder=3)

        # overlay ridges as coloured lines
        cmap_ridge = plt.cm.spring
        n_ridges = max(len(r['ridges']), 1)
        for ri, ridge in enumerate(r['ridges']):
            color = cmap_ridge(ri / n_ridges)
            ax.plot(ridge['t_hr'], ridge['freq_hz'],
                    color=color, lw=1.8, alpha=0.9, zorder=4)

        ax.set_ylim(0, F_MAX)
        ax.set_ylabel(f'{ch}\nFreq (Hz)', fontsize=8)
        ax.text(0.005, 0.92, f'{ch}  ({len(r["ridges"])} ridges)',
                transform=ax.transAxes, fontsize=8, fontweight='bold',
                color='white', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

    # Row 4: Ridge frequency traces — all channels overlaid
    ax = axes[4]
    # stage shading
    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.12)

    for ch in CAP_CHS:
        r = results[ch]
        for ridge in r['ridges']:
            ax.plot(ridge['t_hr'], ridge['freq_hz'],
                    color=CH_COLORS[ch], lw=1.2, alpha=0.7)

    ax.set_ylim(0, F_MAX)
    ax.set_ylabel('Ridge freq (Hz)', fontsize=8)
    ax.set_xlabel('Time (hr)', fontsize=9)
    ax.set_title('Persistent spectral ridges — all channels', fontsize=9)
    ax.grid(True, alpha=0.2)

    # channel legend
    ch_patches = [mpatches.Patch(color=CH_COLORS[c], label=c) for c in CAP_CHS]
    stage_patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
                     for c in STAGE_ORDER]
    ax.legend(handles=ch_patches + stage_patches, fontsize=6, ncol=8,
              loc='upper right', framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading all sessions...')
    sessions = load_all_sessions(with_sleep_profiles=True)

    for s in sessions:
        print(f'\n{s.label} ({s.subject}, {s.duration_hr:.1f} hr)')
        print('  Computing peaks + ridges...')
        results = process_session(s)

        for ch in CAP_CHS:
            n_r = len(results[ch]['ridges'])
            n_p = sum(len(p) for p in results[ch]['all_peaks_f'])
            print(f'    {ch}: {n_p} peaks detected, {n_r} ridges (>={MIN_RIDGE_LEN} windows)')

        out = PLOT_DIR / f'spectrogram_peaks_{s.label.lower()}.png'
        print(f'  Plotting -> {out.name}')
        plot_session(s, results, out)

    print('\nDone.')
