"""
Signal preprocessing: accelerometer artifact removal, window extraction,
full-session bandpass preprocessing.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from .config import RESP_LO, RESP_HI, CARD_LO, CARD_HI, FS
from .filters import bandpass
from .sessions import SleepSession


# ── Artifact removal ───────────────────────────────────────────────────────────

def remove_acc_artifact(cap_sig: np.ndarray, acc_mag: np.ndarray,
                         f_lo: float, f_hi: float, fs: float = FS) -> np.ndarray:
    """
    OLS-regress bandpassed accelerometer magnitude out of a CAP channel.

    Both signals are bandpassed to the same band before regression, so only
    motion energy in that band is removed.
    """
    cap_bp = bandpass(cap_sig, f_lo, f_hi, fs)
    acc_bp = bandpass(acc_mag, f_lo, f_hi, fs)
    beta   = np.dot(acc_bp, cap_bp) / (np.dot(acc_bp, acc_bp) + 1e-12)
    return cap_bp - beta * acc_bp


def remove_acc_artifact_nlms(cap_sig: np.ndarray, acc_mag: np.ndarray,
                              f_lo: float, f_hi: float,
                              fs: float = FS,
                              taps: int = 16, mu: float = 0.05) -> np.ndarray:
    """
    Normalised LMS adaptive cancellation of accelerometer noise.

    Unlike OLS (one stationary coupling coefficient), NLMS tracks a time-varying
    FIR coupling between accelerometer and CAP channel. More robust when posture
    changes or the coupling drifts across the night.

    Returns the residual (CAP after subtracting the adaptive ACC prediction),
    bandpassed to [f_lo, f_hi].
    """
    MIN_SAMPLES = 50  # filtfilt needs padlen ≈ 21; use a generous margin
    if len(cap_sig) < MIN_SAMPLES:
        return np.zeros(len(cap_sig), dtype=np.float64)
    cap_bp = bandpass(cap_sig, f_lo, f_hi, fs).astype(np.float64)
    acc_bp = bandpass(acc_mag, f_lo, f_hi, fs).astype(np.float64)
    n = len(cap_bp)
    if n <= taps:
        return cap_bp
    # Normalise ACC for stable step size
    a_std = np.std(acc_bp) + 1e-12
    acc_n = acc_bp / a_std
    w = np.zeros(taps)
    y = np.zeros(n)
    for i in range(taps, n):
        x = acc_n[i - taps:i][::-1]
        pred = float(np.dot(w, x))
        e = cap_bp[i] - pred
        norm = float(np.dot(x, x)) + 1e-6
        w += (mu / norm) * e * x
        y[i] = e
    # First `taps` samples were un-filtered; pad with bandpass copy
    y[:taps] = cap_bp[:taps]
    return y


# ── Internal helper ────────────────────────────────────────────────────────────

def _bandpass_channels(
    raw_ch: Dict[str, np.ndarray],
    acc: np.ndarray,
    acc_removal: bool,
    fs: float = FS,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply resp+cardiac bandpass (with optional accelerometer removal) to each channel.

    Returns
    -------
    {channel: {'resp': array, 'card': array}}
    """
    sigs: Dict[str, Dict[str, np.ndarray]] = {}
    for ch, sig in raw_ch.items():
        if acc_removal:
            sigs[ch] = {
                'resp': remove_acc_artifact(sig, acc, RESP_LO, RESP_HI, fs),
                'card': remove_acc_artifact(sig, acc, CARD_LO, CARD_HI, fs),
            }
        else:
            sigs[ch] = {
                'resp': bandpass(sig, RESP_LO, RESP_HI, fs),
                'card': bandpass(sig, CARD_LO, CARD_HI, fs),
            }
    return sigs


# ── Public preprocessing functions ────────────────────────────────────────────

def preprocess_window(
    session: SleepSession,
    start_hr: float,
    win_hr: float,
    acc_removal: bool = True,
) -> dict:
    """
    Extract a time window from a session and apply bandpass filtering.

    Parameters
    ----------
    session     : SleepSession
    start_hr    : window start in hours from recording start
    win_hr      : window duration in hours
    acc_removal : if True, regress out accelerometer artifact

    Returns
    -------
    dict with keys:
        t_s           : (N,) seconds within the window
        idx           : (N,) sample indices into session arrays
        raw           : {ch: raw_signal} for CH, CLE, CRE, CLE-CRE
        sigs          : {ch: {'resp': ..., 'card': ...}} bandpassed signals
        gt_resp       : bandpassed PSG Thorax (resp band)
        gt_card       : bandpassed PSG Pleth (cardiac band)
        gt_thorax_raw : raw PSG Thorax
        gt_pleth_raw  : raw PSG Pleth
    """
    t = session.time_hr
    end_hr = start_hr + win_hr
    mask = (t >= start_hr) & (t <= end_hr)
    if not mask.any():
        raise ValueError(f'No samples in [{start_hr:.3f}, {end_hr:.3f}] hr')

    idx = np.where(mask)[0]
    t_s = (t[idx] - t[idx[0]]) * 3600.0

    cap, psg = session.cap, session.psg
    acc = cap['acc_mag'][idx].astype(np.float64)

    raw_ch: Dict[str, np.ndarray] = {
        'CH':  cap['CH'][idx].astype(np.float64),
        'CLE': cap['CLE'][idx].astype(np.float64),
        'CRE': cap['CRE'][idx].astype(np.float64),
    }
    raw_ch['CLE-CRE'] = raw_ch['CLE'] - raw_ch['CRE']

    return dict(
        t_s           = t_s,
        idx           = idx,
        raw           = raw_ch,
        sigs          = _bandpass_channels(raw_ch, acc, acc_removal, session.fs),
        gt_resp       = bandpass(psg['Thorax'][idx], RESP_LO, RESP_HI, session.fs),
        gt_card       = bandpass(psg['Pleth'][idx],  CARD_LO, CARD_HI, session.fs),
        gt_thorax_raw = psg['Thorax'][idx].astype(np.float64),
        gt_pleth_raw  = psg['Pleth'][idx].astype(np.float64),
    )


def preprocess_full(
    session: SleepSession,
    acc_removal: bool = True,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Bandpass-filter the entire session for whole-night sliding-window analysis.

    Parameters
    ----------
    session     : SleepSession
    acc_removal : if True, regress out accelerometer artifact

    Returns
    -------
    (full_sigs, gt_sigs)
        full_sigs : {ch: {'resp': array, 'card': array}}
        gt_sigs   : {'thorax_bp': array, 'pleth_bp': array,
                     'flow_bp': array, 'ecg': array}
    """
    cap, psg = session.cap, session.psg
    acc = cap['acc_mag'].astype(np.float64)
    print('Preprocessing...', end=' ', flush=True)

    raw_ch: Dict[str, np.ndarray] = {
        'CH':  cap['CH'].astype(np.float64),
        'CLE': cap['CLE'].astype(np.float64),
        'CRE': cap['CRE'].astype(np.float64),
    }
    raw_ch['CLE-CRE'] = raw_ch['CLE'] - raw_ch['CRE']

    full = _bandpass_channels(raw_ch, acc, acc_removal, session.fs)
    gt = {
        'thorax_bp': bandpass(psg['Thorax'].astype(np.float64), RESP_LO, RESP_HI, session.fs),
        'pleth_bp':  bandpass(psg['Pleth'].astype(np.float64),  CARD_LO, CARD_HI, session.fs),
        'flow_bp':   bandpass(psg['Flow'].astype(np.float64),   RESP_LO, RESP_HI, session.fs),
        'ecg':       psg['ECG'].astype(np.float64),
    }
    print('done')
    return full, gt
