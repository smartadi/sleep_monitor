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
