"""
Ground-truth rate extraction from PSG reference signals.

Cardiac GT : ECG R-peak detection (neurokit2 Pan-Tompkins variant)
Respiratory GT : Multi-signal consensus (Flow+Thorax+Abdomen+RIPSum),
                 fallback to Flow peak detection, then Thorax
Fallbacks      : Pleth for cardiac, Thorax for respiratory

Public API
----------
gt_heart_rate(session, ...)          -> GTResult  (beat-level HR from ECG)
gt_resp_rate(session, ...)           -> GTResult  (breath-level BR from Flow)
gt_resp_rate_consensus(label, ...)   -> dict      (consensus resp rate from parquet)
gt_sliding_rates(session, ..., resp_method='consensus')
                                     -> dict with t_hr, resp_hz, card_hz arrays
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI
from .filters import bandpass
from .sessions import SleepSession

_CONSENSUS_PARQUET = (
    Path(__file__).resolve().parent.parent / 'artifacts' / 'consolidated_resp_gt.parquet'
)
_CONSENSUS_CACHE: dict = {}


@dataclass
class GTResult:
    peak_indices: np.ndarray
    peak_times_s: np.ndarray
    intervals_s: np.ndarray
    instant_rate_hz: np.ndarray
    instant_rate_bpm: np.ndarray
    signal_used: str
    method: str


def _ecg_rpeaks(ecg: np.ndarray, fs: float = FS) -> np.ndarray:
    import neurokit2 as nk
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=int(fs))
    info = nk.ecg_findpeaks(ecg_clean, sampling_rate=int(fs))
    return np.array(info['ECG_R_Peaks'], dtype=int)


def _flow_peaks(flow: np.ndarray, fs: float = FS) -> np.ndarray:
    import neurokit2 as nk
    rsp_clean = nk.rsp_clean(flow, sampling_rate=int(fs))
    info = nk.rsp_findpeaks(rsp_clean, sampling_rate=int(fs))
    return np.array(info['RSP_Peaks'], dtype=int)


def _fallback_peaks(signal: np.ndarray, f_lo: float, f_hi: float,
                    fs: float = FS) -> np.ndarray:
    """ACF-guided peak detection on a bandpassed signal (Thorax or Pleth)."""
    bp = bandpass(signal.astype(np.float64), f_lo, f_hi, fs)
    min_dist = int(fs / f_hi * 0.6)
    prom = 0.05 * np.std(bp)
    peaks, _ = find_peaks(bp, distance=min_dist, prominence=prom)
    return peaks


def _quality_filter(peak_indices: np.ndarray, fs: float,
                    rate_lo_hz: float, rate_hi_hz: float) -> np.ndarray:
    """Remove peaks that produce physiologically impossible intervals."""
    if len(peak_indices) < 2:
        return peak_indices
    intervals = np.diff(peak_indices) / fs
    min_interval = 1.0 / rate_hi_hz
    max_interval = 1.0 / rate_lo_hz
    good = (intervals >= min_interval) & (intervals <= max_interval)
    keep = np.ones(len(peak_indices), dtype=bool)
    for i in range(len(good)):
        if not good[i]:
            keep[i + 1] = False
    return peak_indices[keep]


def _build_result(peaks: np.ndarray, fs: float,
                  signal_name: str, method: str) -> GTResult:
    peak_times = peaks / fs
    intervals = np.diff(peak_times)
    instant_hz = 1.0 / intervals
    return GTResult(
        peak_indices=peaks,
        peak_times_s=peak_times,
        intervals_s=intervals,
        instant_rate_hz=instant_hz,
        instant_rate_bpm=instant_hz * 60.0,
        signal_used=signal_name,
        method=method,
    )


def gt_heart_rate(
    session: SleepSession,
    fallback: bool = True,
) -> GTResult:
    """
    Extract beat-level heart rate from ECG (primary) or Pleth (fallback).

    Uses neurokit2 Pan-Tompkins variant for R-peak detection on ECG.
    Falls back to peak detection on bandpassed Pleth if ECG fails.
    """
    ecg = session.psg.get('ECG')
    if ecg is not None:
        try:
            peaks = _ecg_rpeaks(ecg.astype(np.float64), session.fs)
            peaks = _quality_filter(peaks, session.fs,
                                    CARD_LO, CARD_HI)
            if len(peaks) >= 10:
                return _build_result(peaks, session.fs, 'ECG', 'pan_tompkins')
        except Exception:
            pass

    if fallback:
        pleth = session.psg.get('Pleth')
        if pleth is not None:
            peaks = _fallback_peaks(pleth, CARD_LO, CARD_HI, session.fs)
            peaks = _quality_filter(peaks, session.fs, CARD_LO, CARD_HI)
            return _build_result(peaks, session.fs, 'Pleth', 'peak_detection')

    raise ValueError('No usable cardiac GT signal found')


def gt_resp_rate(
    session: SleepSession,
    fallback: bool = True,
) -> GTResult:
    """
    Extract breath-level respiratory rate from Flow (primary) or Thorax (fallback).

    Uses neurokit2 respiratory peak detection on nasal airflow.
    Falls back to peak detection on bandpassed Thorax if Flow fails.
    """
    flow = session.psg.get('Flow')
    if flow is not None:
        try:
            peaks = _flow_peaks(flow.astype(np.float64), session.fs)
            peaks = _quality_filter(peaks, session.fs,
                                    RESP_LO, RESP_HI)
            if len(peaks) >= 10:
                return _build_result(peaks, session.fs, 'Flow', 'neurokit2')
        except Exception:
            pass

    if fallback:
        thorax = session.psg.get('Thorax')
        if thorax is not None:
            peaks = _fallback_peaks(thorax, RESP_LO, RESP_HI, session.fs)
            peaks = _quality_filter(peaks, session.fs, RESP_LO, RESP_HI)
            return _build_result(peaks, session.fs, 'Thorax', 'peak_detection')

    raise ValueError('No usable respiratory GT signal found')


def gt_resp_rate_consensus(
    session_label: str,
    t_hr: Optional[np.ndarray] = None,
) -> dict:
    """
    Load consensus respiratory GT from the consolidated multi-signal parquet.

    The consensus is the per-epoch median of quality-gated PSG signals
    (Flow + Thorax + Abdomen + RIPSum), built by build_consolidated_resp_gt.py.

    Parameters
    ----------
    session_label : e.g. 'S1N1'
    t_hr : if given, exact-sample consensus at these times (must lie on the
           5 s grid).  If None, return the full 5 s grid.

    Returns
    -------
    dict with 't_hr', 'resp_hz' (consensus rate in Hz), 'method'='consensus'
    """
    if 'df' not in _CONSENSUS_CACHE:
        if not _CONSENSUS_PARQUET.exists():
            raise FileNotFoundError(
                f'Consensus resp GT not found: {_CONSENSUS_PARQUET}'
            )
        _CONSENSUS_CACHE['df'] = pd.read_parquet(_CONSENSUS_PARQUET)

    sess_df = _CONSENSUS_CACHE['df']
    sess_df = sess_df[sess_df['session'] == session_label]
    if len(sess_df) == 0:
        raise ValueError(f'No consensus GT for session {session_label}')

    if t_hr is None:
        return {
            't_hr': sess_df['t_hr'].values,
            'resp_hz': sess_df['rate_consensus'].values,
            'method': 'consensus',
        }

    lookup = dict(zip(
        np.round(sess_df['t_hr'].values, 8),
        sess_df['rate_consensus'].values,
    ))
    rates = np.array([lookup.get(round(t, 8), np.nan) for t in t_hr])
    return {
        't_hr': t_hr,
        'resp_hz': rates,
        'method': 'consensus',
    }


def gt_sliding_rates(
    session: SleepSession,
    win_sec: float = 30.0,
    step_sec: float = 5.0,
    resp_method: str = 'consensus',
) -> dict:
    """
    Compute GT respiratory and cardiac rates on a sliding window grid.

    Parameters
    ----------
    resp_method : 'consensus' (default) uses the multi-signal consolidated GT
        from ``artifacts/consolidated_resp_gt.parquet``.  Falls back to
        Flow peak detection if the parquet is missing or the session is not
        in it.  Pass 'flow' to force legacy Flow-only (then Thorax) GT.

    Returns dict with:
        t_hr        : (K,) window centre times in hours
        resp_hz     : (K,) respiratory rate in Hz
        card_hz     : (K,) cardiac rate in Hz
        resp_gt     : GTResult for respiratory (None when consensus is used)
        card_gt     : GTResult for cardiac
    """
    card_gt = gt_heart_rate(session)

    win_n = int(round(win_sec * session.fs))
    step_n = max(1, int(round(step_sec * session.fs)))
    n_samples = session.n_samples

    centres_s = []
    for start in range(0, n_samples - win_n + 1, step_n):
        centres_s.append((start + win_n / 2.0) / session.fs)
    centres_s = np.array(centres_s)
    t_hr = centres_s / 3600.0

    resp_gt = None
    if resp_method == 'consensus':
        try:
            cons = gt_resp_rate_consensus(session.label, t_hr)
            resp_hz = cons['resp_hz']
        except (FileNotFoundError, ValueError):
            resp_gt = gt_resp_rate(session)
            resp_hz = _peaks_to_sliding_rate(
                resp_gt.peak_times_s, centres_s, win_sec)
    else:
        resp_gt = gt_resp_rate(session)
        resp_hz = _peaks_to_sliding_rate(
            resp_gt.peak_times_s, centres_s, win_sec)

    card_hz = _peaks_to_sliding_rate(
        card_gt.peak_times_s, centres_s, win_sec)

    return {
        't_hr': t_hr,
        'resp_hz': resp_hz,
        'card_hz': card_hz,
        'resp_gt': resp_gt,
        'card_gt': card_gt,
    }


def _peaks_to_sliding_rate(
    peak_times_s: np.ndarray,
    centres_s: np.ndarray,
    win_sec: float,
) -> np.ndarray:
    """Convert peak times to sliding-window average rate."""
    half = win_sec / 2.0
    rates = np.full(len(centres_s), np.nan)
    for i, tc in enumerate(centres_s):
        lo, hi = tc - half, tc + half
        in_win = peak_times_s[(peak_times_s >= lo) & (peak_times_s <= hi)]
        if len(in_win) >= 2:
            rates[i] = (len(in_win) - 1) / (in_win[-1] - in_win[0])
    return rates
