"""
Harmonic structure detection in CAP sensor signals.

Three methods to detect and characterize harmonic ladders (fundamental + integer
multiples) in sliding windows:

1. Harmonic Product Spectrum (HPS) — fast f0 detection via downsampled PSD product
2. Cepstral analysis — harmonic vs broadband discrimination via cepstral peak
3. Explicit f0 + harmonic counting — interpretable per-harmonic amplitudes

All methods operate on Welch PSD and return per-window feature DataFrames.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

from .config import FS


# ── Method 1: Harmonic Product Spectrum ──────────────────────────────────────

def _hps(
    psd: np.ndarray,
    freqs: np.ndarray,
    f0_range: Tuple[float, float],
    n_downsample: int = 5,
) -> Tuple[float, float]:
    """
    Harmonic Product Spectrum in log domain (sum of log-downsampled PSDs).

    Returns (f0_hz, hps_score) where hps_score = peak - median of the log-sum.
    """
    log_psd = np.log(psd + 1e-30)
    log_sum = log_psd.copy()
    for k in range(2, n_downsample + 1):
        decimated = log_psd[::k]
        n = min(len(log_sum), len(decimated))
        log_sum = log_sum[:n] + decimated[:n]

    f_mask = (freqs[:len(log_sum)] >= f0_range[0]) & (freqs[:len(log_sum)] <= f0_range[1])
    if not np.any(f_mask):
        return np.nan, 0.0

    sum_in_range = log_sum[f_mask]
    freqs_in_range = freqs[:len(log_sum)][f_mask]

    idx = np.argmax(sum_in_range)
    f0 = freqs_in_range[idx]
    score = sum_in_range[idx] - np.median(log_sum)

    return float(f0), float(score)


# ── Method 2: Cepstral analysis ─────────────────────────────────────────────

def _cepstral(
    psd: np.ndarray,
    freqs: np.ndarray,
    f0_range: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Cepstral analysis: IFFT of log-PSD → peak in quefrency domain.

    Returns (f0_hz, cepstral_prominence).
    """
    log_psd = np.log(psd + 1e-30)
    cepstrum = np.fft.irfft(log_psd)

    df = freqs[1] - freqs[0]
    if df <= 0:
        return np.nan, 0.0

    n_cep = len(cepstrum)
    quefrency = np.arange(n_cep) / (n_cep * df)

    q_lo = 1.0 / f0_range[1] if f0_range[1] > 0 else 0
    q_hi = 1.0 / f0_range[0] if f0_range[0] > 0 else len(cepstrum)
    q_mask = (quefrency >= q_lo) & (quefrency <= q_hi)

    if not np.any(q_mask):
        return np.nan, 0.0

    ceps_region = np.abs(cepstrum[q_mask])
    q_region = quefrency[q_mask]

    idx = np.argmax(ceps_region)
    q_peak = q_region[idx]
    f0 = 1.0 / q_peak if q_peak > 0 else np.nan

    floor = np.median(np.abs(cepstrum)) + 1e-30
    prominence = ceps_region[idx] / floor

    return float(f0), float(prominence)


# ── Method 3: Explicit f0 + harmonic counting ───────────────────────────────

def _explicit_harmonics(
    psd: np.ndarray,
    freqs: np.ndarray,
    f0_range: Tuple[float, float],
    max_harmonics: int = 6,
    f_tolerance: float = 0.05,
    min_prominence: float = 0.1,
) -> dict:
    """
    Find dominant peak in f0_range, then check for integer harmonics.

    Returns dict with: f0_hz, n_harmonics, harmonic_energy_ratio,
    harmonic_decay_rate, per_harmonic_amps (list).
    """
    df = freqs[1] - freqs[0]
    f_mask = (freqs >= f0_range[0]) & (freqs <= f0_range[1])

    if not np.any(f_mask):
        return dict(f0_hz=np.nan, n_harmonics=0, harmonic_energy_ratio=0.0,
                    harmonic_decay_rate=np.nan, per_harmonic_amps=[])

    psd_in_range = psd[f_mask]
    freqs_in_range = freqs[f_mask]

    peaks, props = find_peaks(psd_in_range, prominence=min_prominence * np.max(psd_in_range))

    if len(peaks) == 0:
        idx = np.argmax(psd_in_range)
        f0 = freqs_in_range[idx]
        a0 = psd_in_range[idx]
    else:
        best = peaks[np.argmax(psd_in_range[peaks])]
        f0 = freqs_in_range[best]
        a0 = psd_in_range[best]

    harmonic_power = a0 * df
    amps = [float(a0)]
    n_harmonics = 0

    for k in range(2, max_harmonics + 2):
        fk = k * f0
        idx_lo = np.searchsorted(freqs, fk - f_tolerance)
        idx_hi = np.searchsorted(freqs, fk + f_tolerance)
        if idx_lo >= len(freqs) or idx_hi <= idx_lo:
            break

        region = psd[idx_lo:idx_hi]
        ak = np.max(region)
        local_med = np.median(psd[max(0, idx_lo - 5):min(len(psd), idx_hi + 5)])

        if ak > local_med * 1.5:
            n_harmonics += 1
            harmonic_power += ak * df
            amps.append(float(ak))
        else:
            amps.append(0.0)

    total_power = np.trapz(psd, dx=df) + 1e-30
    energy_ratio = harmonic_power / total_power

    log_amps = np.log(np.array(amps) + 1e-30)
    nonzero = log_amps[np.array(amps) > 0]
    if len(nonzero) >= 2:
        ks = np.arange(len(nonzero))
        decay_rate = float(np.polyfit(ks, nonzero, 1)[0])
    else:
        decay_rate = np.nan

    return dict(
        f0_hz=float(f0),
        n_harmonics=n_harmonics,
        harmonic_energy_ratio=float(energy_ratio),
        harmonic_decay_rate=decay_rate,
        per_harmonic_amps=amps,
    )


# ── Persistent ridge tracking ──────────────────────────────────────────────

def detect_persistent_ridges(
    sig: np.ndarray,
    fs: float = FS,
    win_sec: float = 30.0,
    step_sec: float = 30.0,
    max_freq: float = 5.0,
    smooth_windows: int = 5,
    min_persistence_sec: float = 120.0,
    max_freq_jump: float = 0.08,
    peak_prominence_frac: float = 0.15,
    welch_seg_sec: float = 8.0,
    max_gap_windows: int = 2,
    acc_mag: Optional[np.ndarray] = None,
    motion_thresh_mad: float = 3.0,
) -> dict:
    """
    Detect spectral ridges that persist over time, then group into harmonic sets.

    Unlike per-window detection, this enforces temporal continuity: a ridge is a
    spectral peak that stays within max_freq_jump Hz between consecutive windows
    and persists for at least min_persistence_sec.

    Parameters
    ----------
    sig                 : 1-D signal array
    fs                  : sampling rate (Hz)
    win_sec             : analysis window length (seconds)
    step_sec            : step between windows (seconds)
    max_freq            : upper frequency limit (Hz)
    smooth_windows      : median-smooth PSDs over this many adjacent windows
    min_persistence_sec : minimum ridge duration to keep (seconds)
    max_freq_jump       : max Hz change between consecutive windows for continuity
    peak_prominence_frac: peak prominence threshold as fraction of local PSD median
    welch_seg_sec       : Welch sub-segment length (seconds)
    max_gap_windows     : allow ridge to survive this many missing windows
    acc_mag             : accelerometer magnitude for motion masking
    motion_thresh_mad   : motion threshold in MAD units

    Returns
    -------
    dict with keys:
        't_hr'      : array of window centre times (hours)
        'freqs'     : frequency axis of PSDs
        'psds'      : 2-D array (n_windows x n_freqs) — raw PSDs
        'psds_smooth': 2-D array — temporally smoothed PSDs
        'motion_mask': boolean array
        'ridges'    : list of dicts, each with:
                        'freq_trace'  : array of frequency per window (NaN where absent)
                        'amp_trace'   : array of PSD amplitude per window
                        'start_idx'   : first window index
                        'end_idx'     : last window index
                        'duration_sec': total duration
                        'median_freq' : median frequency of ridge
                        'label'       : string label
        'harmonic_groups': list of dicts, each with:
                        'fundamental_idx' : ridge index of fundamental
                        'harmonic_idxs'   : list of ridge indices forming the group
                        'f0_median'       : median fundamental frequency
    """
    win_n = int(win_sec * fs)
    step_n = int(step_sec * fs)
    nperseg = min(int(welch_seg_sec * fs), win_n)
    n = len(sig)

    starts = np.arange(0, n - win_n + 1, step_n)
    n_win = len(starts)

    # ── Motion mask ──
    motion_mask = np.zeros(n_win, dtype=bool)
    if acc_mag is not None:
        acc = acc_mag.astype(np.float64)
        motion_rms = np.array([
            np.sqrt(np.mean((acc[s0:s0+win_n] - np.mean(acc[s0:s0+win_n]))**2))
            for s0 in starts
        ])
        med = np.median(motion_rms)
        mad = np.median(np.abs(motion_rms - med)) + 1e-12
        motion_mask = motion_rms > (med + motion_thresh_mad * mad)

    # ── Step 1: Compute PSDs ──
    t_hr = np.array([(s0 + win_n / 2) / fs / 3600.0 for s0 in starts])
    sample_freqs, sample_psd = welch(
        sig[:win_n].astype(np.float64), fs=fs, nperseg=nperseg,
        noverlap=nperseg // 2, scaling='density',
    )
    f_mask = sample_freqs <= max_freq
    freqs = sample_freqs[f_mask]
    n_f = len(freqs)

    psds = np.full((n_win, n_f), np.nan)
    for i, s0 in enumerate(starts):
        if motion_mask[i]:
            continue
        chunk = sig[s0:s0+win_n].astype(np.float64)
        _, psd = welch(chunk, fs=fs, nperseg=nperseg,
                       noverlap=nperseg // 2, scaling='density')
        psds[i] = psd[f_mask]

    # ── Step 2: Temporal median smoothing ──
    half = smooth_windows // 2
    psds_smooth = np.full_like(psds, np.nan)
    for i in range(n_win):
        lo = max(0, i - half)
        hi = min(n_win, i + half + 1)
        block = psds[lo:hi]
        valid_rows = ~np.all(np.isnan(block), axis=1)
        if valid_rows.sum() >= 2:
            psds_smooth[i] = np.nanmedian(block[valid_rows], axis=0)
        elif valid_rows.sum() == 1:
            psds_smooth[i] = block[valid_rows][0]

    # ── Step 3: Find peaks in smoothed PSDs per window ──
    peaks_per_window = []
    for i in range(n_win):
        if np.all(np.isnan(psds_smooth[i])):
            peaks_per_window.append([])
            continue
        psd_s = psds_smooth[i]
        local_med = np.nanmedian(psd_s) + 1e-30
        prom_thresh = peak_prominence_frac * local_med
        peak_idxs, props = find_peaks(psd_s, prominence=prom_thresh)
        peak_list = [(freqs[pi], psd_s[pi]) for pi in peak_idxs]
        peaks_per_window.append(peak_list)

    # ── Step 4: Track ridges with continuity constraint ──
    active_ridges = []    # each: {'freq_trace': [...], 'amp_trace': [...],
                          #        'last_freq': float, 'gap': int, 'start_idx': int}
    finished_ridges = []

    for i in range(n_win):
        peaks = list(peaks_per_window[i])
        matched_peak_idxs = set()

        for ridge in active_ridges:
            best_dist = max_freq_jump + 1
            best_pi = -1
            for pi, (f, a) in enumerate(peaks):
                if pi in matched_peak_idxs:
                    continue
                dist = abs(f - ridge['last_freq'])
                if dist < best_dist:
                    best_dist = dist
                    best_pi = pi

            if best_pi >= 0 and best_dist <= max_freq_jump:
                f, a = peaks[best_pi]
                ridge['freq_trace'][i] = f
                ridge['amp_trace'][i] = a
                ridge['last_freq'] = f
                ridge['gap'] = 0
                ridge['end_idx'] = i
                matched_peak_idxs.add(best_pi)
            else:
                ridge['gap'] += 1

        # Terminate ridges that exceeded gap tolerance
        still_active = []
        for ridge in active_ridges:
            if ridge['gap'] > max_gap_windows:
                finished_ridges.append(ridge)
            else:
                still_active.append(ridge)
        active_ridges = still_active

        # Start new ridges from unmatched peaks
        for pi, (f, a) in enumerate(peaks):
            if pi not in matched_peak_idxs:
                freq_trace = np.full(n_win, np.nan)
                amp_trace = np.full(n_win, np.nan)
                freq_trace[i] = f
                amp_trace[i] = a
                active_ridges.append({
                    'freq_trace': freq_trace,
                    'amp_trace': amp_trace,
                    'last_freq': f,
                    'gap': 0,
                    'start_idx': i,
                    'end_idx': i,
                })

    finished_ridges.extend(active_ridges)

    # ── Step 5: Filter by minimum persistence and frequency ──
    min_windows = max(1, int(min_persistence_sec / step_sec))
    df_freq = freqs[1] - freqs[0] if len(freqs) > 1 else 0.1
    min_freq = 2 * df_freq  # skip lowest 2 bins (DC leakage)
    ridges = []
    for r in finished_ridges:
        n_present = np.sum(~np.isnan(r['freq_trace']))
        duration = (r['end_idx'] - r['start_idx'] + 1) * step_sec
        median_freq = float(np.nanmedian(r['freq_trace']))
        if (n_present >= min_windows and duration >= min_persistence_sec
                and median_freq >= min_freq):
            ridges.append({
                'freq_trace': r['freq_trace'],
                'amp_trace': r['amp_trace'],
                'start_idx': r['start_idx'],
                'end_idx': r['end_idx'],
                'duration_sec': duration,
                'median_freq': median_freq,
                'n_present': int(n_present),
                'label': f'{median_freq:.2f}Hz',
            })

    ridges.sort(key=lambda r: r['median_freq'])

    # ── Step 5b: Merge fragmented ridges ──
    ridges = _merge_ridge_fragments(ridges, n_win, step_sec,
                                    max_freq_jump, max_gap_windows * 3)

    # ── Step 6: Group into harmonic sets ──
    harmonic_groups = _find_harmonic_groups(ridges, t_hr)

    return {
        't_hr': t_hr,
        'freqs': freqs,
        'psds': psds,
        'psds_smooth': psds_smooth,
        'motion_mask': motion_mask,
        'ridges': ridges,
        'harmonic_groups': harmonic_groups,
    }


def _merge_ridge_fragments(
    ridges: list,
    n_win: int,
    step_sec: float,
    max_freq_jump: float,
    merge_gap_windows: int,
) -> list:
    """
    Merge ridge fragments that end and restart at similar frequencies.

    After greedy tracking, a single physical ridge often gets split into
    fragments when it briefly dips below the prominence threshold.  This
    pass stitches them back together if the gap is small and the frequency
    difference at the boundary is within tolerance.
    """
    if len(ridges) < 2:
        return ridges

    ridges = sorted(ridges, key=lambda r: (r['start_idx'], r['median_freq']))
    merged = [ridges[0]]

    for cand in ridges[1:]:
        did_merge = False
        for mi, base in enumerate(merged):
            gap = cand['start_idx'] - base['end_idx']
            if gap < 1 or gap > merge_gap_windows:
                continue
            freq_base = base['freq_trace'][base['end_idx']]
            freq_cand = cand['freq_trace'][cand['start_idx']]
            if np.isnan(freq_base) or np.isnan(freq_cand):
                continue
            if abs(freq_base - freq_cand) > max_freq_jump * 2:
                continue

            new_freq = base['freq_trace'].copy()
            new_amp = base['amp_trace'].copy()
            mask_c = ~np.isnan(cand['freq_trace'])
            new_freq[mask_c] = cand['freq_trace'][mask_c]
            new_amp[mask_c] = cand['amp_trace'][mask_c]

            n_present = int(np.sum(~np.isnan(new_freq)))
            end_idx = max(base['end_idx'], cand['end_idx'])
            start_idx = min(base['start_idx'], cand['start_idx'])
            duration = (end_idx - start_idx + 1) * step_sec
            median_freq = float(np.nanmedian(new_freq))

            merged[mi] = {
                'freq_trace': new_freq,
                'amp_trace': new_amp,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration_sec': duration,
                'median_freq': median_freq,
                'n_present': n_present,
                'label': f'{median_freq:.2f}Hz',
            }
            did_merge = True
            break

        if not did_merge:
            merged.append(cand)

    merged.sort(key=lambda r: r['median_freq'])
    return merged


def compute_harmonic_score(
    rr: dict,
    ratio_tol: float = 0.12,
    min_f0: float = 0.1,
) -> dict:
    """
    Continuous per-window harmonic strength score from persistent ridges.

    For each window, collects all active ridges and scores how well they form
    integer-ratio ladders.  Unlike the binary ``label_harmonic_ladder_windows``,
    this returns a continuous score in [0, 1] that encodes:

    * **ratio_quality** — how close active ridge pairs are to exact integer
      ratios (1.0 = perfect, decays with deviation)
    * **n_harmonics** — how many ridges participate in the best ladder
    * **power** — total PSD amplitude of ladder members (log-scaled, normalised)

    The composite score is: ``ratio_quality × log2(n_harmonics) / log2(6) × power_norm``
    clamped to [0, 1].

    Parameters
    ----------
    rr        : output dict from ``detect_persistent_ridges()``
    ratio_tol : tolerance for integer ratio check
    min_f0    : ignore fundamentals below this frequency

    Returns
    -------
    dict with keys (all arrays of length n_windows):
        'harmonic_score'  : float [0, 1] composite score
        'ratio_quality'   : float [0, 1] how close to perfect integer ratios
        'n_ladder'        : int   number of ridges in best ladder
        'ladder_f0'       : float fundamental of best ladder (NaN if none)
        'ladder_power'    : float total amplitude of ladder members
        'ladder_freqs'    : list  frequencies of ladder members per window
    """
    ridges = rr['ridges']
    n_win = len(rr['t_hr'])

    score = np.zeros(n_win)
    rq = np.zeros(n_win)
    n_lad = np.zeros(n_win, dtype=int)
    lad_f0 = np.full(n_win, np.nan)
    lad_power = np.zeros(n_win)
    lad_freqs = [[] for _ in range(n_win)]

    for i in range(n_win):
        active = []
        for ri, ridge in enumerate(ridges):
            f = ridge['freq_trace'][i]
            a = ridge['amp_trace'][i]
            if np.isfinite(f) and f >= min_f0:
                active.append((ri, f, a))

        if len(active) < 2:
            continue

        active.sort(key=lambda x: x[1])

        best_score = 0.0
        best_rq = 0.0
        best_n = 0
        best_f0 = np.nan
        best_pwr = 0.0
        best_fqs = []

        for ai in range(len(active)):
            _, f0, a0 = active[ai]
            if f0 < min_f0:
                continue

            members = [(f0, a0)]
            deviations = []

            for aj in range(len(active)):
                if aj == ai:
                    continue
                _, fj, aj_amp = active[aj]
                ratio = fj / f0
                nearest_int = round(ratio)
                dev = abs(ratio - nearest_int)
                if nearest_int >= 2 and dev < ratio_tol:
                    members.append((fj, aj_amp))
                    deviations.append(dev)

            if len(members) < 2:
                continue

            quality = 1.0 - np.mean(deviations) / ratio_tol if deviations else 0.0
            n_harm = len(members)
            power = sum(m[1] for m in members)
            harm_factor = min(np.log2(max(n_harm, 1)) / np.log2(6), 1.0)
            s = quality * harm_factor

            if s > best_score:
                best_score = s
                best_rq = quality
                best_n = n_harm
                best_f0 = f0
                best_pwr = power
                best_fqs = [m[0] for m in members]

        score[i] = best_score
        rq[i] = best_rq
        n_lad[i] = best_n
        lad_f0[i] = best_f0
        lad_power[i] = best_pwr
        lad_freqs[i] = best_fqs

    # Normalise power to [0, 1] across session
    valid_pwr = lad_power[lad_power > 0]
    if len(valid_pwr) > 0:
        p95 = np.percentile(valid_pwr, 95)
        if p95 > 0:
            power_norm = np.clip(lad_power / p95, 0, 1)
            score = score * power_norm

    return {
        'harmonic_score': np.clip(score, 0, 1),
        'ratio_quality': rq,
        'n_ladder': n_lad,
        'ladder_f0': lad_f0,
        'ladder_power': lad_power,
        'ladder_freqs': lad_freqs,
    }


def _find_harmonic_groups(ridges: list, t_hr: np.ndarray,
                          ratio_tol: float = 0.12) -> list:
    """
    Among persistent ridges, find sets where frequencies form integer ratios.

    For each pair of concurrent ridges, check if freq_high / freq_low is close
    to an integer (2, 3, 4, ...). Build groups bottom-up from lowest frequency.
    """
    if len(ridges) < 2:
        return []

    n_ridges = len(ridges)
    groups = []
    used = set()

    for i in range(n_ridges):
        if i in used:
            continue
        f_i = ridges[i]['median_freq']
        if f_i < 0.05:
            continue

        members = [i]
        for j in range(i + 1, n_ridges):
            if j in used:
                continue
            f_j = ridges[j]['median_freq']
            ratio = f_j / f_i
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) < ratio_tol:
                # Check temporal overlap
                overlap_start = max(ridges[i]['start_idx'], ridges[j]['start_idx'])
                overlap_end = min(ridges[i]['end_idx'], ridges[j]['end_idx'])
                if overlap_end > overlap_start:
                    members.append(j)

        if len(members) >= 2:
            for m in members:
                used.add(m)
            groups.append({
                'fundamental_idx': i,
                'harmonic_idxs': members,
                'f0_median': f_i,
            })

    return groups


# ── Concurrent-ridge harmonic ladder labeling ─────────────────────────────

def label_harmonic_ladder_windows(
    rr: dict,
    ratio_tol: float = 0.12,
    min_harmonics: int = 2,
    min_f0: float = 0.1,
) -> dict:
    """
    Per-window harmonic ladder labeling from persistent ridge output.

    At each window, collects all active persistent ridges and checks whether
    any subset forms an integer-ratio ladder (f0, 2*f0, 3*f0, ...).  Because
    the input ridges are already persistence-filtered and temporally smoothed,
    only stable spectral features participate — no short-lived noise.

    Parameters
    ----------
    rr           : output dict from detect_persistent_ridges()
    ratio_tol    : tolerance for integer ratio check (|f_j/f_i - round(...)| < tol)
    min_harmonics: minimum ladder members to label a window (2 = fundamental + one harmonic)
    min_f0       : ignore candidate fundamentals below this frequency

    Returns
    -------
    dict with keys:
        'is_ladder'       : bool array (n_windows,) — True if harmonic ladder active
        'ladder_f0'       : float array — fundamental freq of best ladder (NaN if none)
        'ladder_n'        : int array — number of ladder members
        'ladder_power'    : float array — total PSD amplitude of ladder members
        'ladder_members'  : list of lists — ridge indices forming the ladder per window
        'ladder_freqs'    : list of lists — frequencies of ladder members per window
    """
    ridges = rr['ridges']
    n_win = len(rr['t_hr'])

    is_ladder = np.zeros(n_win, dtype=bool)
    ladder_f0 = np.full(n_win, np.nan)
    ladder_n = np.zeros(n_win, dtype=int)
    ladder_power = np.full(n_win, np.nan)
    ladder_members = [[] for _ in range(n_win)]
    ladder_freqs = [[] for _ in range(n_win)]

    for i in range(n_win):
        active = []
        for ri, ridge in enumerate(ridges):
            f = ridge['freq_trace'][i]
            a = ridge['amp_trace'][i]
            if np.isfinite(f) and f >= min_f0:
                active.append((ri, f, a))

        if len(active) < min_harmonics:
            continue

        active.sort(key=lambda x: x[1])

        best_ladder = []
        best_f0 = np.nan
        best_power = 0.0

        for ai in range(len(active)):
            ri_0, f0, a0 = active[ai]
            if f0 < min_f0:
                continue

            members = [(ri_0, f0, a0, 1)]  # (ridge_idx, freq, amp, harmonic_number)

            for aj in range(len(active)):
                if aj == ai:
                    continue
                ri_j, f_j, a_j = active[aj]
                ratio = f_j / f0
                nearest_int = round(ratio)
                if nearest_int >= 2 and abs(ratio - nearest_int) < ratio_tol:
                    members.append((ri_j, f_j, a_j, nearest_int))

            if len(members) >= min_harmonics and len(members) > len(best_ladder):
                best_ladder = members
                best_f0 = f0
                best_power = sum(m[2] for m in members)

        if len(best_ladder) >= min_harmonics:
            is_ladder[i] = True
            ladder_f0[i] = best_f0
            ladder_n[i] = len(best_ladder)
            ladder_power[i] = best_power
            ladder_members[i] = [m[0] for m in best_ladder]
            ladder_freqs[i] = [m[1] for m in best_ladder]

    return {
        'is_ladder': is_ladder,
        'ladder_f0': ladder_f0,
        'ladder_n': ladder_n,
        'ladder_power': ladder_power,
        'ladder_members': ladder_members,
        'ladder_freqs': ladder_freqs,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def detect_harmonics(
    sig: np.ndarray,
    fs: float = FS,
    win_sec: float = 30.0,
    step_sec: float = 10.0,
    f0_range: Tuple[float, float] = (0.1, 0.8),
    max_harmonics: int = 6,
    f_tolerance: float = 0.05,
    welch_seg_sec: float = 8.0,
    min_prominence: float = 0.1,
    acc_mag: Optional[np.ndarray] = None,
    motion_thresh_mad: float = 3.0,
) -> pd.DataFrame:
    """
    Sliding-window harmonic structure detection using three methods.

    Parameters
    ----------
    sig             : 1-D signal array
    fs              : sampling rate (Hz)
    win_sec         : window length in seconds
    step_sec        : step size in seconds
    f0_range        : (f_lo, f_hi) Hz — search range for fundamental
    max_harmonics   : maximum number of harmonics to look for above f0
    f_tolerance     : Hz tolerance for confirming a harmonic peak
    welch_seg_sec   : Welch sub-segment length in seconds
    min_prominence  : minimum peak prominence as fraction of max PSD
    acc_mag         : accelerometer magnitude (same length as sig) for motion gating
    motion_thresh_mad : motion threshold in MAD units

    Returns
    -------
    DataFrame with columns:
        t_s                     : window centre time (seconds)
        t_hr                    : window centre time (hours)
        hps_f0_hz               : f0 from HPS method
        hps_score               : HPS peak/median ratio
        cep_f0_hz               : f0 from cepstral method
        cep_prominence          : cepstral peak prominence
        f0_hz                   : f0 from explicit method (primary)
        n_harmonics             : count of confirmed harmonics
        harmonic_energy_ratio   : fraction of power in harmonic peaks
        harmonic_decay_rate     : slope of log(amplitude) vs harmonic number
        motion_masked           : True if window was masked for motion
    """
    win_n = int(win_sec * fs)
    step_n = int(step_sec * fs)
    nperseg = min(int(welch_seg_sec * fs), win_n)
    n = len(sig)

    starts = np.arange(0, n - win_n + 1, step_n)
    k = len(starts)

    motion_mask = np.zeros(k, dtype=bool)
    if acc_mag is not None:
        acc = acc_mag.astype(np.float64)
        motion_rms = np.empty(k)
        for i, s0 in enumerate(starts):
            chunk = acc[s0:s0 + win_n]
            motion_rms[i] = np.sqrt(np.mean((chunk - np.mean(chunk)) ** 2))
        med = np.median(motion_rms)
        mad = np.median(np.abs(motion_rms - med)) + 1e-12
        motion_mask = motion_rms > (med + motion_thresh_mad * mad)

    records = []
    for i, s0 in enumerate(starts):
        t_s = (s0 + win_n / 2) / fs
        t_hr = t_s / 3600.0

        if motion_mask[i]:
            records.append(dict(
                t_s=t_s, t_hr=t_hr,
                hps_f0_hz=np.nan, hps_score=np.nan,
                cep_f0_hz=np.nan, cep_prominence=np.nan,
                f0_hz=np.nan, n_harmonics=np.nan,
                harmonic_energy_ratio=np.nan, harmonic_decay_rate=np.nan,
                motion_masked=True,
            ))
            continue

        chunk = sig[s0:s0 + win_n].astype(np.float64)
        freqs, psd = welch(chunk, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2, scaling='density')

        hps_f0, hps_score = _hps(psd, freqs, f0_range)
        cep_f0, cep_prom = _cepstral(psd, freqs, f0_range)
        explicit = _explicit_harmonics(psd, freqs, f0_range,
                                       max_harmonics=max_harmonics,
                                       f_tolerance=f_tolerance,
                                       min_prominence=min_prominence)

        records.append(dict(
            t_s=t_s, t_hr=t_hr,
            hps_f0_hz=hps_f0, hps_score=hps_score,
            cep_f0_hz=cep_f0, cep_prominence=cep_prom,
            f0_hz=explicit['f0_hz'],
            n_harmonics=explicit['n_harmonics'],
            harmonic_energy_ratio=explicit['harmonic_energy_ratio'],
            harmonic_decay_rate=explicit['harmonic_decay_rate'],
            motion_masked=False,
        ))

    return pd.DataFrame(records)


def detect_harmonics_multichannel(
    signals: dict[str, np.ndarray],
    fs: float = FS,
    **kwargs,
) -> pd.DataFrame:
    """
    Run detect_harmonics on multiple channels, pick the dominant one per window.

    Parameters
    ----------
    signals : {channel_name: signal_array} — e.g. {'CH': ..., 'CLE': ..., 'CRE': ...}
    fs      : sampling rate
    **kwargs: forwarded to detect_harmonics

    Returns
    -------
    DataFrame with all columns from the best channel per window, plus:
        dominant_channel : which channel had highest harmonic_energy_ratio
    """
    channel_dfs = {}
    for name, sig in signals.items():
        df = detect_harmonics(sig, fs=fs, **kwargs)
        channel_dfs[name] = df

    ref_name = list(signals.keys())[0]
    ref_df = channel_dfs[ref_name]
    n_windows = len(ref_df)

    best_rows = []
    for i in range(n_windows):
        best_ch = None
        best_ratio = -1.0
        for name, df in channel_dfs.items():
            ratio = df.iloc[i]['harmonic_energy_ratio']
            if np.isfinite(ratio) and ratio > best_ratio:
                best_ratio = ratio
                best_ch = name

        row = channel_dfs[best_ch or ref_name].iloc[i].to_dict()
        row['dominant_channel'] = best_ch or ''
        best_rows.append(row)

    return pd.DataFrame(best_rows)
