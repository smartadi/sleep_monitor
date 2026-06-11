"""
SWA Validation — Shared Processing Pipeline (Step 1)

Implements the Lucey et al. 2019 spectral pipeline identically for
both contact EEG and capacitive EEG signals.

Functions
---------
bandpass_fir       : Zero-phase FIR bandpass (least-squares design)
compute_epoch_psd  : 6-sec epoch Welch PSD
compute_band_powers: Band powers from PSD
reject_artifacts   : Artifact gating on 20-30 Hz + accelerometer
process_signal     : Full pipeline: filter → epoch → PSD → band powers → artifact reject
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
from scipy.signal import firwin, filtfilt, welch

# ── Band definitions ──────────────────────────────────────────────────────────
SWA_BANDS = {
    'swa_total': (1.0, 4.5),
    'swa_1_2':   (1.0, 2.0),
    'swa_2_3':   (2.0, 3.0),
    'swa_3_4':   (3.0, 4.0),
    'emg':       (20.0, 30.0),
}

EPOCH_SEC = 6.0
FILT_LO = 0.5
FILT_HI = 40.0
FIR_NTAPS = 331   # odd, ~3.3 s at 100 Hz → good transition band


def bandpass_fir(sig: np.ndarray, fs: float,
                 lo: float = FILT_LO, hi: float = FILT_HI,
                 ntaps: int = FIR_NTAPS) -> np.ndarray:
    coeffs = firwin(ntaps, [lo, hi], pass_zero=False, fs=fs)
    return filtfilt(coeffs, 1.0, sig).astype(np.float64)


def compute_epoch_psd(sig_filtered: np.ndarray, fs: float,
                      epoch_sec: float = EPOCH_SEC
                      ) -> Tuple[np.ndarray, np.ndarray, int]:
    epoch_samp = int(epoch_sec * fs)
    n_epochs = len(sig_filtered) // epoch_samp
    sig_trimmed = sig_filtered[:n_epochs * epoch_samp].reshape(n_epochs, epoch_samp)

    freqs, psd = welch(sig_trimmed, fs=fs, window='hamming',
                       nperseg=epoch_samp, noverlap=0, axis=1)
    return freqs, psd, n_epochs


def compute_band_powers(freqs: np.ndarray, psd: np.ndarray,
                        bands: Dict[str, Tuple[float, float]] | None = None
                        ) -> Dict[str, np.ndarray]:
    if bands is None:
        bands = SWA_BANDS
    powers = {}
    df = freqs[1] - freqs[0]
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        powers[name] = np.sum(psd[:, mask], axis=1) * df
    total_mask = (freqs >= FILT_LO) & (freqs <= FILT_HI)
    powers['total_0.5_40'] = np.sum(psd[:, total_mask], axis=1) * df
    return powers


def compute_relative_powers(powers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    total = powers['total_0.5_40']
    total_safe = np.where(total > 0, total, 1e-20)
    rel = {}
    for name, vals in powers.items():
        if name != 'total_0.5_40':
            rel[f'{name}_rel'] = vals / total_safe
    return rel


def reject_artifacts(powers: Dict[str, np.ndarray],
                     acc_epochs: np.ndarray | None = None,
                     emg_pctile: float = 97.5,
                     acc_threshold: float = 0.15
                     ) -> np.ndarray:
    emg = powers['emg']
    thresh = np.percentile(emg, emg_pctile)
    bad = emg > thresh

    if acc_epochs is not None:
        bad |= acc_epochs > acc_threshold

    return bad


def epoch_accelerometer(acc_mag: np.ndarray, fs: float,
                        epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    epoch_samp = int(epoch_sec * fs)
    n_epochs = len(acc_mag) // epoch_samp
    acc_trimmed = acc_mag[:n_epochs * epoch_samp].reshape(n_epochs, epoch_samp)
    baseline = np.median(acc_trimmed)
    return np.std(acc_trimmed - baseline, axis=1)


def process_signal(sig: np.ndarray, fs: float,
                   acc_mag: np.ndarray | None = None
                   ) -> Dict:
    sig_filt = bandpass_fir(sig.astype(np.float64), fs)
    freqs, psd, n_epochs = compute_epoch_psd(sig_filt, fs)
    powers = compute_band_powers(freqs, psd)
    rel_powers = compute_relative_powers(powers)

    acc_ep = None
    if acc_mag is not None:
        acc_ep = epoch_accelerometer(acc_mag.astype(np.float64), fs)
        acc_ep = acc_ep[:n_epochs]

    bad = reject_artifacts(powers, acc_ep)

    return {
        'freqs': freqs,
        'psd': psd,
        'powers': powers,
        'rel_powers': rel_powers,
        'artifact_mask': bad,
        'acc_epochs': acc_ep,
        'n_epochs': n_epochs,
        'artifact_pct': bad.sum() / n_epochs * 100,
    }


def align_stages_to_epochs(t_ep_hr: np.ndarray, codes: np.ndarray,
                            n_epochs: int, fs: float,
                            epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    stage_per_epoch = np.full(n_epochs, -1, dtype=np.int8)
    epoch_centers_hr = (np.arange(n_epochs) * epoch_sec + epoch_sec / 2) / 3600.0

    for i, t_hr in enumerate(epoch_centers_hr):
        diffs = np.abs(t_ep_hr - t_hr)
        nearest = np.argmin(diffs)
        if diffs[nearest] < 30.0 / 3600.0:
            stage_per_epoch[i] = codes[nearest]

    return stage_per_epoch
