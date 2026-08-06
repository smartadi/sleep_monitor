"""
Head orientation and movement extraction from 3-axis accelerometer.

Angles are derived from the gravity vector (LP-filtered accelerometer). Yaw
about the gravity axis is not observable without a gyroscope or magnetometer.

Axis convention (established empirically against the PSG-scored body-position
channel in analysis/mean_value/head_angle_validate.py, all 12 sessions):

    +Z  anterior   — out of the face;  gZ ≈ +1 g when supine
    +Y  to the subject's LEFT — gY ≈ +0.9 g lying on the left side,
                                gY ≈ −0.9 g lying on the right side
    +X  caudal     — down the body;  gX ≈ +0.9 g when upright

The accelerometer records in units of g: |a| sits at 1.02–1.04 g with a
coefficient of variation under 2 % across every session, i.e. the DC vector
really is gravity and tilt angles from it are physically meaningful.

Preferred angles (head_angle):
    turn_deg  = atan2(gY, gZ)      full ±180°: 0 supine, +90 left, −90 right,
                                   ±180 prone. This is the head-rotation angle
                                   the L−R capacitance difference responds to.
    elev_deg  = asin(gX / |g|)     head-axis elevation: ≈0–30° recumbent,
                                   ≈90° upright.
    tilt_deg  = angle to +Z        total deviation from face-up.

head_orientation() below keeps the older ±90°-clamped roll/pitch pair for
backwards compatibility, but it cannot separate supine from prone (it discards
the sign of gZ) and should not be used for new work — see head_angle().
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np

from .config import FS
from .filters import lowpass, highpass
from .sessions import SleepSession

# Gravity must be extracted well below the respiratory band (0.1–0.5 Hz),
# otherwise breathing motion leaks straight into the "static" angle.
GRAVITY_LP_HZ = 0.05


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


def fit_accel_calibration(
    aX: np.ndarray, aY: np.ndarray, aZ: np.ndarray, max_samples: int = 200_000,
) -> Optional[dict]:
    """
    Least-squares ellipsoid (axis-aligned) fit of per-axis bias and gain.

    Solves [a², a] · θ = 1 so that ((a - bias) / gain) lands on the unit sphere.

    A head that spends the night supine samples only a small cap of the sphere,
    which makes the fit ill-conditioned, so the result carries a `coverage`
    figure (fraction of the 3 axes' sign-space actually visited, via the spread
    of the unit gravity direction). Callers should ignore the fit when coverage
    is low. Returns None if the system is singular.

    Note the angles from head_angle() are ratios of axes and are therefore
    exactly invariant to an isotropic gain error — only per-axis BIAS moves
    them, which is what this fit is for.
    """
    A = np.column_stack([aX, aY, aZ]).astype(np.float64)
    good = np.isfinite(A).all(axis=1)
    A = A[good]
    if len(A) < 1000:
        return None
    if len(A) > max_samples:                     # uniform thin, keeps coverage
        A = A[:: int(np.ceil(len(A) / max_samples))]

    M = np.column_stack([A ** 2, A])
    try:
        theta, *_ = np.linalg.lstsq(M, np.ones(len(M)), rcond=None)
    except np.linalg.LinAlgError:
        return None
    p, q = theta[:3], theta[3:]
    if np.any(p <= 0):
        return None
    bias = -q / (2 * p)
    k = 1.0 + np.sum(q ** 2 / (4 * p))
    if k <= 0:
        return None
    gain = np.sqrt(k / p)

    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    n = A[nrm[:, 0] > 1e-6] / nrm[nrm[:, 0] > 1e-6]
    coverage = float(np.mean(np.std(n, axis=0)) * np.sqrt(3))   # 0 = one pose
    resid = np.linalg.norm((A - bias) / gain, axis=1) - 1.0
    return {
        'bias': bias, 'gain': gain, 'coverage': coverage,
        'resid_rms': float(np.sqrt(np.mean(resid ** 2))),
        'cond': float(np.linalg.cond(M)),
    }


def head_angle(
    aX: np.ndarray,
    aY: np.ndarray,
    aZ: np.ndarray,
    fs: float = FS,
    lp_cutoff: float = GRAVITY_LP_HZ,
    calib: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Head angles from the gravity vector, in the validated axis convention.

    Parameters
    ----------
    lp_cutoff : gravity-extraction cutoff, Hz. Default 0.05 Hz sits an octave
                below the respiratory band so breathing does not modulate the
                "static" angle (the legacy 0.5 Hz cutoff does not).
    calib     : optional dict from fit_accel_calibration(); applies per-axis
                bias/gain before the angles are formed.

    Returns
    -------
    dict with turn_deg (±180, 0 = supine, + = left), elev_deg (head-axis
    elevation, 90 = upright), tilt_deg (total angle from face-up), the gravity
    components gX/gY/gZ and gmag (|g|, the validity check — it must stay ≈1 g).
    """
    g = np.column_stack([
        lowpass(np.asarray(aX, np.float64), lp_cutoff, fs),
        lowpass(np.asarray(aY, np.float64), lp_cutoff, fs),
        lowpass(np.asarray(aZ, np.float64), lp_cutoff, fs),
    ])
    if calib is not None:
        g = (g - calib['bias']) / calib['gain']
    gX, gY, gZ = g[:, 0], g[:, 1], g[:, 2]
    gmag = np.linalg.norm(g, axis=1)

    turn = np.degrees(np.arctan2(gY, gZ))
    elev = np.degrees(np.arcsin(np.clip(gX / np.maximum(gmag, 1e-9), -1, 1)))
    tilt = np.degrees(np.arccos(np.clip(gZ / np.maximum(gmag, 1e-9), -1, 1)))
    return {'turn_deg': turn, 'elev_deg': elev, 'tilt_deg': tilt,
            'gX': gX, 'gY': gY, 'gZ': gZ, 'gmag': gmag}


def classify_head_position(turn_deg: np.ndarray, elev_deg: np.ndarray,
                           upright_elev: float = 55.0) -> np.ndarray:
    """
    Label head position from the validated angles, in PSG vocabulary.

    Boundaries at ±45° / ±135° of turn, upright taken first because a seated
    subject's turn angle is not meaningful. Sign follows the PSG convention
    verified in head_angle_validate.py: positive turn = subject's LEFT.
    """
    turn = np.asarray(turn_deg, float)
    pos = np.full(len(turn), 'Supine', dtype=object)
    at = np.abs(turn)
    pos[(at > 45) & (at <= 135) & (turn > 0)] = 'Left'
    pos[(at > 45) & (at <= 135) & (turn < 0)] = 'Right'
    pos[at > 135] = 'Prone'
    pos[np.asarray(elev_deg, float) > upright_elev] = 'Upright'
    return pos


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
    DEPRECATED — use classify_head_position() with head_angle() instead.

    Classify head position from the legacy clamped roll/pitch pair.

    The left/right sign here was inverted relative to the PSG-scored position
    channel (gY > 0, i.e. roll > 0, is the subject's LEFT — see the axis
    convention at the top of this module); that is corrected below. The 'prone'
    rule remains unsound: the clamped roll/pitch discard the sign of gZ, so
    prone is not separable from supine on these two angles at all.

    Returns string array: 'supine', 'left', 'right', 'prone'.
    """
    pos = np.full(len(roll_deg), 'supine', dtype='U8')
    pos[roll_deg > 45] = 'left'
    pos[roll_deg < -45] = 'right'
    pos[(np.abs(pitch_deg) > 45) & (np.abs(roll_deg) <= 45)] = 'prone'
    return pos
