"""
Per-window quality scoring.

A quality score summarises how trustworthy a rate estimate from one window is.
It is used to (a) gate the output stream so garbage windows emit NaN, and
(b) serve as features for the rate-classifier phase.

Public API
----------
window_features(sig_band, acc_mag_win, f_lo, f_hi, fs, rates=None) -> dict
    One row of features for one window.
combined_quality(features) -> float in [0, 1]
    Collapse the feature dict into a single scalar quality score.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from scipy.signal import welch, find_peaks

from .config import FS


# ── Individual quality metrics ────────────────────────────────────────────────

def inband_snr(sig: np.ndarray, f_lo: float, f_hi: float, fs: float = FS) -> float:
    """
    SNR = in-band power / out-of-band power (within Nyquist).
    Returns dB. Higher = cleaner periodic content in the target band.
    """
    if len(sig) < 16:
        return np.nan
    nperseg = min(len(sig), max(64, int(fs * 4)))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    nyq = fs / 2.0
    in_band  = (freqs >= f_lo) & (freqs <= f_hi)
    out_band = (freqs > 0) & (freqs <= nyq) & ~in_band
    p_in  = float(psd[in_band].sum())
    p_out = float(psd[out_band].sum()) + 1e-20
    if p_in <= 0:
        return -np.inf
    return 10.0 * np.log10(p_in / p_out)


def acf_prominence(sig: np.ndarray, f_lo: float, f_hi: float,
                    fs: float = FS) -> float:
    """
    Prominence of the dominant ACF peak in the lag range implied by [f_lo, f_hi].
    Higher = stronger periodicity at the expected rate.
    """
    x = sig.astype(np.float64) - sig.mean()
    n = len(x)
    if n < 16:
        return np.nan
    lag_min = max(1, int(np.floor(fs / f_hi)))
    lag_max = min(n - 1, int(np.ceil(fs / f_lo)))
    if lag_min >= lag_max:
        return np.nan
    nfft = 1
    while nfft < 2 * n - 1:
        nfft <<= 1
    X = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(X * np.conj(X), n=nfft)[:n].real
    acf /= (acf[0] + 1e-12)
    _, props = find_peaks(acf[lag_min:lag_max + 1], prominence=0.0)
    if not len(props.get('prominences', [])):
        return 0.0
    return float(np.max(props['prominences']))


def spectral_concentration(sig: np.ndarray, f_lo: float, f_hi: float,
                             fs: float = FS) -> float:
    """
    Fraction of in-band power at the peak frequency bin ± 1 bin.
    1.0 = pure tone, 0 = flat noise.
    """
    if len(sig) < 16:
        return np.nan
    nperseg = min(len(sig), max(64, int(fs * 4)))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.nan
    band_psd = psd[mask]
    k = int(np.argmax(band_psd))
    lo, hi = max(0, k - 1), min(len(band_psd), k + 2)
    peak_pw = float(band_psd[lo:hi].sum())
    total   = float(band_psd.sum()) + 1e-20
    return peak_pw / total


def motion_power(acc_mag_win: np.ndarray, f_lo: float, f_hi: float,
                  fs: float = FS) -> float:
    """
    Power in the accelerometer magnitude within the target rate band.
    High values indicate motion that can mimic or destroy the signal.
    Returned in dB relative to 1.
    """
    if acc_mag_win is None or len(acc_mag_win) < 16:
        return np.nan
    nperseg = min(len(acc_mag_win), max(64, int(fs * 4)))
    freqs, psd = welch(acc_mag_win - acc_mag_win.mean(), fs=fs,
                         nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    p = float(psd[mask].sum()) + 1e-20
    return 10.0 * np.log10(p)


def method_agreement(rates_hz: Dict[str, float]) -> float:
    """
    Std-dev of rates across methods (Hz). Low = methods agree = higher confidence.
    Returns NaN if fewer than 2 finite rates.
    """
    vals = np.array([v for v in rates_hz.values()
                      if v is not None and np.isfinite(v)])
    if len(vals) < 2:
        return np.nan
    return float(np.std(vals))


# ── Per-window feature vector ─────────────────────────────────────────────────

def window_features(
    sig_band: np.ndarray,
    acc_mag_win: Optional[np.ndarray],
    f_lo: float,
    f_hi: float,
    fs: float = FS,
    rates_hz: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute the quality feature vector for one window.

    Parameters
    ----------
    sig_band    : band-filtered signal window
    acc_mag_win : accelerometer magnitude for the same window (or None)
    f_lo, f_hi  : band edges
    fs          : sampling rate
    rates_hz    : optional {method: rate_Hz} to include agreement

    Returns
    -------
    dict with keys: snr_db, acf_prom, spec_conc, motion_db, agreement_hz,
                    rms (signal RMS), n (window length in samples)
    """
    out = {
        'snr_db':    inband_snr(sig_band, f_lo, f_hi, fs),
        'acf_prom':  acf_prominence(sig_band, f_lo, f_hi, fs),
        'spec_conc': spectral_concentration(sig_band, f_lo, f_hi, fs),
        'motion_db': motion_power(acc_mag_win, f_lo, f_hi, fs)
                      if acc_mag_win is not None else np.nan,
        'rms':       float(np.sqrt(np.mean(sig_band.astype(np.float64)**2))),
        'n':         int(len(sig_band)),
    }
    out['agreement_hz'] = method_agreement(rates_hz) if rates_hz else np.nan
    return out


# ── Single scalar score ───────────────────────────────────────────────────────

def combined_quality(features: Dict[str, float]) -> float:
    """
    Collapse window_features() output to a single scalar in [0, 1].

    Composition (heuristic, calibrated later):
      + SNR (6..20 dB       -> 0..1)
      + spectral concentration (raw)
      + ACF prominence (clipped 0..0.6 -> 0..1)
      - motion (−10..+10 dB -> 1..0)
      - method disagreement (0..0.2 Hz -> 1..0) if present
    """
    def _ramp(x, lo, hi):
        if not np.isfinite(x):
            return 0.5
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    snr      = _ramp(features.get('snr_db',    np.nan),  6.0, 20.0)
    conc     = _ramp(features.get('spec_conc', np.nan),  0.05, 0.5)
    prom     = _ramp(features.get('acf_prom',  np.nan),  0.0,  0.6)
    motion   = 1.0 - _ramp(features.get('motion_db', np.nan), -10.0, 10.0)
    agree_hz = features.get('agreement_hz', np.nan)
    agree    = 1.0 - _ramp(agree_hz, 0.0, 0.2) if np.isfinite(agree_hz) else 0.7

    weights = np.array([0.30, 0.20, 0.20, 0.20, 0.10])
    vals    = np.array([snr,  conc, prom, motion, agree])
    return float(np.clip((weights * vals).sum(), 0.0, 1.0))
