"""
Moving-window frequency band power ratio analysis for CAP sensor signals.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import welch

from .config import FS, EEG_BANDS


def compute_band_power_ratios(
    sig: np.ndarray,
    fs: float = FS,
    win_sec: float = 30.0,
    step_sec: float = 10.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 30.0),
    welch_seg_sec: float = 4.0,
    acc_mag: Optional[np.ndarray] = None,
    motion_thresh_mad: float = 3.0,
) -> Dict[str, np.ndarray]:
    """
    Sliding-window band power ratios via Welch PSD.

    Parameters
    ----------
    sig              : 1-D signal array
    fs               : sampling rate (Hz)
    win_sec          : window length in seconds
    step_sec         : step size in seconds
    bands            : {name: (f_lo, f_hi)} — defaults to EEG_BANDS
    total_range      : (f_lo, f_hi) for total power denominator
    welch_seg_sec    : Welch sub-segment length in seconds
    acc_mag          : accelerometer magnitude array (same length as sig).
                       If provided, high-motion windows are set to NaN.
    motion_thresh_mad: motion threshold in MAD units above median RMS

    Returns
    -------
    dict with keys:
        t_hr           : (K,) window centre times in hours
        <band_name>    : (K,) ratio arrays for each band
        total_power    : (K,) total power in total_range
        motion_mask    : (K,) bool — True where window was masked for motion
    """
    if bands is None:
        bands = EEG_BANDS

    win_n = int(win_sec * fs)
    step_n = int(step_sec * fs)
    nperseg = int(welch_seg_sec * fs)
    n = len(sig)

    starts = np.arange(0, n - win_n + 1, step_n)
    k = len(starts)

    t_hr = (starts + win_n / 2) / fs / 3600.0

    # Pre-compute per-window motion RMS for gating
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

    ratios = {name: np.empty(k) for name in bands}
    abs_powers = {name: np.empty(k) for name in bands}
    total_power = np.empty(k)

    for i, s0 in enumerate(starts):
        if motion_mask[i]:
            total_power[i] = np.nan
            for name in bands:
                ratios[name][i] = np.nan
                abs_powers[name][i] = np.nan
            continue

        chunk = sig[s0:s0 + win_n].astype(np.float64)
        freqs, psd = welch(chunk, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2, scaling='density')
        df = freqs[1] - freqs[0]

        t_mask = (freqs >= total_range[0]) & (freqs <= total_range[1])
        tp = np.trapz(psd[t_mask], dx=df)
        total_power[i] = tp

        for name, (flo, fhi) in bands.items():
            mask = (freqs >= flo) & (freqs <= fhi)
            bp = np.trapz(psd[mask], dx=df)
            ratios[name][i] = bp / tp if tp > 0 else 0.0
            abs_powers[name][i] = bp

    abs_dict = {f'{name}_abs': arr for name, arr in abs_powers.items()}
    return {'t_hr': t_hr, 'total_power': total_power,
            'motion_mask': motion_mask, **ratios, **abs_dict}
