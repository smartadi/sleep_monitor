"""
Head orientation and movement extraction from 3-axis accelerometer.

Roll and pitch are derived from the gravity vector (LP-filtered accelerometer).
Yaw is not observable without a gyroscope or magnetometer.
"""

from __future__ import annotations
from typing import Dict

import numpy as np

from .config import FS
from .filters import lowpass, highpass
from .sessions import SleepSession


def head_orientation(
    aX: np.ndarray,
    aY: np.ndarray,
    aZ: np.ndarray,
    fs: float = FS,
    lp_cutoff: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Compute roll and pitch angles from accelerometer gravity vector.

    Returns dict with keys: roll_deg, pitch_deg (both in degrees).
    """
    gX = lowpass(aX.astype(np.float64), lp_cutoff, fs)
    gY = lowpass(aY.astype(np.float64), lp_cutoff, fs)
    gZ = lowpass(aZ.astype(np.float64), lp_cutoff, fs)

    # Both use sqrt() denominator to clamp output to [-90°, 90°]
    # and avoid wrap-around / gimbal lock singularities.
    roll = np.arctan2(gY, np.sqrt(gX**2 + gZ**2))
    pitch = np.arctan2(-gX, np.sqrt(gY**2 + gZ**2))

    return {
        'roll_deg': np.degrees(roll),
        'pitch_deg': np.degrees(pitch),
        'gX': gX, 'gY': gY, 'gZ': gZ,
    }


def dynamic_acceleration(
    aX: np.ndarray,
    aY: np.ndarray,
    aZ: np.ndarray,
    fs: float = FS,
    hp_cutoff: float = 0.5,
) -> np.ndarray:
    """HP-filtered accelerometer magnitude (gravity removed)."""
    dX = highpass(aX.astype(np.float64), hp_cutoff, fs)
    dY = highpass(aY.astype(np.float64), hp_cutoff, fs)
    dZ = highpass(aZ.astype(np.float64), hp_cutoff, fs)
    return np.sqrt(dX**2 + dY**2 + dZ**2)


def epoch_motion(
    session: SleepSession,
    epoch_sec: float = 30.0,
    hp_cutoff: float = 0.5,
    lp_cutoff: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Compute per-epoch head orientation and movement metrics.

    Returns dict with arrays of length n_epochs:
        t_hr          : epoch centre time (hours)
        roll_deg      : mean roll angle in epoch
        pitch_deg     : mean pitch angle in epoch
        movement_rms  : RMS of dynamic acceleration in epoch
        movement_peak : peak dynamic acceleration in epoch
    """
    cap = session.cap
    orient = head_orientation(cap['aX'], cap['aY'], cap['aZ'], session.fs, lp_cutoff)
    dyn_acc = dynamic_acceleration(cap['aX'], cap['aY'], cap['aZ'], session.fs, hp_cutoff)

    epoch_n = int(epoch_sec * session.fs)
    n_epochs = len(session.time_hr) // epoch_n

    t_hr = np.empty(n_epochs)
    roll = np.empty(n_epochs)
    pitch = np.empty(n_epochs)
    move_rms = np.empty(n_epochs)
    move_peak = np.empty(n_epochs)

    for i in range(n_epochs):
        s = i * epoch_n
        e = s + epoch_n
        t_hr[i] = np.mean(session.time_hr[s:e])
        roll[i] = np.mean(orient['roll_deg'][s:e])
        pitch[i] = np.mean(orient['pitch_deg'][s:e])
        seg = dyn_acc[s:e]
        move_rms[i] = np.sqrt(np.mean(seg**2))
        move_peak[i] = np.max(np.abs(seg))

    return {
        't_hr': t_hr,
        'roll_deg': roll,
        'pitch_deg': pitch,
        'movement_rms': move_rms,
        'movement_peak': move_peak,
    }


def epoch_cap_stats(
    session: SleepSession,
    epoch_sec: float = 30.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute per-epoch mean and std of raw cap channels (CH, CLE, CRE).

    Returns {channel: {'mean': array, 'std': array}} plus 't_hr'.
    """
    epoch_n = int(epoch_sec * session.fs)
    n_epochs = len(session.time_hr) // epoch_n

    result: Dict[str, Dict[str, np.ndarray]] = {}
    t_hr = np.empty(n_epochs)

    for ch in ('CLE', 'CRE', 'CH'):
        sig = session.cap[ch].astype(np.float64)
        means = np.empty(n_epochs)
        stds = np.empty(n_epochs)
        for i in range(n_epochs):
            s = i * epoch_n
            e = s + epoch_n
            seg = sig[s:e]
            means[i] = np.mean(seg)
            stds[i] = np.std(seg)
            if ch == 'CLE' and i < len(t_hr):
                t_hr[i] = np.mean(session.time_hr[s:e])
        result[ch] = {'mean': means, 'std': stds}

    result['t_hr'] = t_hr  # type: ignore[assignment]
    return result


def classify_position(roll_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    """
    Classify head position from roll/pitch angles.

    Returns string array: 'supine', 'left', 'right', 'prone'.
    Thresholds are approximate — mounting orientation may need calibration.
    """
    pos = np.full(len(roll_deg), 'supine', dtype='U8')
    pos[roll_deg < -45] = 'left'
    pos[roll_deg > 45] = 'right'
    pos[(np.abs(pitch_deg) > 45) & (np.abs(roll_deg) <= 45)] = 'prone'
    return pos
