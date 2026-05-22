"""
Signal filtering utilities: bandpass, lowpass, detrend, outlier clipping.
All functions operate on 1-D numpy arrays and return float64 arrays.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from .config import FS


def bandpass(x: np.ndarray, f_lo: float, f_hi: float,
             fs: float = FS, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    return filtfilt(b, a, x.astype(np.float64))


def lowpass(x: np.ndarray, f_hi: float,
            fs: float = FS, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, f_hi / nyq, btype='low')
    return filtfilt(b, a, x.astype(np.float64))


def highpass(x: np.ndarray, f_lo: float,
             fs: float = FS, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth highpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, f_lo / nyq, btype='high')
    return filtfilt(b, a, x.astype(np.float64))


def detrend_segment(sig: np.ndarray, win_ms: float = 500.0,
                    order: int = 2, fs: float = FS) -> np.ndarray:
    """
    Subtract a running polynomial trend from the signal.
    Uses overlapping windows of length win_ms milliseconds.
    """
    win_n = max(order + 2, int(win_ms / 1000.0 * fs))
    out   = sig.astype(np.float64).copy()
    t     = np.arange(win_n)
    half  = win_n // 2
    for start in range(0, len(sig) - win_n + 1, half):
        seg = sig[start:start + win_n].astype(np.float64)
        out[start:start + win_n] -= np.polyval(np.polyfit(t, seg, order), t)
    return out


def outlier_clip(sig: np.ndarray, n_std: float = 4.0) -> np.ndarray:
    """Clip samples beyond ±n_std standard deviations of the mean."""
    mu = np.nanmean(sig)
    sd = np.nanstd(sig)
    return np.clip(sig.astype(np.float64), mu - n_std * sd, mu + n_std * sd)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Uniform (rectangular) moving average."""
    return np.convolve(x.astype(np.float64), np.ones(win) / win, mode='same')


def rolling_zscore(x: np.ndarray, win: int) -> np.ndarray:
    """
    Robust z-score normalisation using a rolling window.
    Falls back to global z-score for short signals.
    """
    x = x.astype(np.float64)
    if len(x) < win:
        sd = np.std(x) + 1e-12
        return (x - np.mean(x)) / sd
    out = np.empty_like(x)
    half = win // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        seg = x[lo:hi]
        out[i] = (x[i] - np.mean(seg)) / (np.std(seg) + 1e-12)
    return out
