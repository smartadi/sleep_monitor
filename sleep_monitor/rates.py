"""
Rate estimators for respiratory and cardiac rates from bandpassed signals.

Six methods are implemented:
  spectral        — Welch PSD peak frequency
  acf             — ACF dominant lag (parabolic interpolation)
  hilbert         — Hilbert instantaneous frequency median
  zerocross       — Upward zero-crossing rate
  peaks           — Peak-counting with prominence threshold
  adaptive_peaks  — Spectral-guided, amplitude-adaptive peak detector

Public API
----------
estimate_rate(x, f_lo, f_hi)          -> dict {method: Hz}
sliding_rates(signal, f_lo, f_hi)     -> (t_s, {method: rates_Hz})
peaks_by_method(sig, f_lo, f_hi, m)   -> peak indices
detect_peaks(sig, f_lo, f_hi)         -> peak indices
zerocross_indices(x)                  -> crossing indices
"""

from __future__ import annotations
import numpy as np
from scipy.signal import welch, find_peaks, hilbert

from .config import FS, METHOD_NAMES, CARD_LO, CARD_HI, RESP_LO, RESP_HI


# ── Scalar rate estimators ─────────────────────────────────────────────────────

def rate_spectral(x: np.ndarray, f_lo: float, f_hi: float, fs: float = FS) -> float:
    """Peak frequency from Welch power spectral density."""
    nperseg = min(len(x), max(64, int(fs * 4)))
    if len(x) < nperseg:
        return np.nan
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.nan
    return float(freqs[mask][np.argmax(psd[mask])])


def rate_acf(x: np.ndarray, f_lo: float, f_hi: float,
             fs: float = FS, prominence: float = 0.10) -> float:
    """
    ACF-based rate: FFT-autocorrelation, find dominant lag, parabolic interpolation.
    Returns rate in Hz, or np.nan on failure.
    """
    x = x.astype(np.float64) - x.mean()
    n = len(x)
    lag_min = max(1, int(np.floor(fs / f_hi)))
    lag_max = min(n - 1, int(np.ceil(fs / f_lo)))
    if lag_min >= lag_max or n < lag_min + 2:
        return np.nan
    # FFT-based ACF (O(n log n))
    nfft = 1
    while nfft < 2 * n - 1:
        nfft <<= 1
    X   = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(X * np.conj(X), n=nfft)[:n].real
    acf /= (acf[0] + 1e-12)
    peaks, props = find_peaks(acf[lag_min:lag_max + 1], prominence=prominence)
    if not len(peaks):
        return np.nan
    k = peaks[np.argmax(props['prominences'])] + lag_min
    # Sub-sample parabolic interpolation
    if 0 < k < n - 1:
        a0, a1, a2 = acf[k - 1], acf[k], acf[k + 1]
        d = a0 - 2 * a1 + a2
        delta = np.clip(0.5 * (a0 - a2) / (d + 1e-12) if abs(d) > 1e-12 else 0.0, -0.5, 0.5)
        period_s = (k + delta) / fs
    else:
        period_s = k / fs
    return 1.0 / period_s if period_s > 0 else np.nan


def rate_hilbert(x: np.ndarray, f_lo: float, f_hi: float, fs: float = FS) -> float:
    """Median instantaneous frequency from Hilbert transform, weighted by amplitude."""
    analytic  = hilbert(x.astype(np.float64))
    phase     = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    amplitude = np.abs(analytic)[:-1]
    valid = (
        (inst_freq >= f_lo) &
        (inst_freq <= f_hi) &
        (amplitude >= np.percentile(amplitude, 25))
    )
    return float(np.median(inst_freq[valid])) if valid.sum() >= 10 else np.nan


def rate_zerocross(x: np.ndarray, fs: float = FS) -> float:
    """Rate from upward zero-crossing intervals (sub-sample interpolated)."""
    x = x.astype(np.float64)
    signs = np.sign(x)
    signs[signs == 0] = 1
    cross_raw = np.where(np.diff(signs) > 0)[0]
    if len(cross_raw) < 2:
        return np.nan
    t_cross = [
        (i + (-x[i] / (x[i + 1] - x[i] + 1e-12))) / fs
        for i in cross_raw if i + 1 < len(x)
    ]
    if len(t_cross) < 2:
        return np.nan
    t_cross = np.array(t_cross)
    return (len(t_cross) - 1) / (t_cross[-1] - t_cross[0])


def rate_peaks(x: np.ndarray, f_lo: float, f_hi: float,
               fs: float = FS, prom_factor: float = 0.4) -> float:
    """Rate from peak counting with prominence threshold and mild pre-smoothing."""
    min_dist   = max(1, int(0.9 * fs / f_hi))
    smooth_win = max(3, min_dist // 4)
    x_sm       = np.convolve(x.astype(np.float64), np.ones(smooth_win) / smooth_win, mode='same')
    pks, _     = find_peaks(x_sm, distance=min_dist, prominence=prom_factor * np.std(x_sm))
    if len(pks) < 2:
        return np.nan
    duration = (pks[-1] - pks[0]) / fs
    return (len(pks) - 1) / duration if duration > 0 else np.nan


# ── Envelope-based estimator (primarily for cardiac / BCG signals) ────────────

def rate_envelope(x: np.ndarray, f_lo: float, f_hi: float,
                  fs: float = FS, env_lo: float = 0.5,
                  env_hi: float = 3.0) -> float:
    """
    Teager-Kaiser energy envelope + ACF rate.

    Mask-based cardiac signals are BCG-like: short pulses whose repetition rate
    is the heart rate, but whose individual samples aren't a clean sinusoid.
    The Teager-Kaiser operator emphasises instantaneous energy, giving a
    slowly-varying envelope whose fundamental period IS the heart rate.
    """
    x = x.astype(np.float64)
    if len(x) < 16:
        return np.nan
    # Teager-Kaiser: psi[n] = x[n]^2 - x[n-1]*x[n+1]
    tk = x[1:-1] ** 2 - x[:-2] * x[2:]
    tk = np.concatenate(([tk[0]], tk, [tk[-1]]))
    # Smooth to an envelope (lowpass via moving average ~1/f_hi seconds)
    win = max(3, int(round(fs / max(f_hi, 1e-3))))
    kernel = np.ones(win) / win
    env = np.convolve(tk, kernel, mode='same')
    env = env - env.mean()
    # Run ACF on the envelope, bounded by [env_lo, env_hi]
    return rate_acf(env, env_lo, env_hi, fs)


# ── Adaptive peak detector ────────────────────────────────────────────────────

def rate_adaptive_peaks(x: np.ndarray, f_lo: float, f_hi: float,
                        fs: float = FS,
                        ipi_cv_max: float = 0.40) -> float:
    """
    Spectral-guided, amplitude-adaptive peak detector.

    Improvements over rate_peaks:
      1. Uses spectral peak to set expected rate → min_distance adapts to the
         actual dominant frequency, not just the band ceiling.
      2. Prominence threshold tracks the local amplitude envelope (rolling MAD)
         instead of a fixed fraction of global std.
      3. Inter-peak-interval (IPI) validation rejects estimates where the CV
         of detected intervals exceeds ipi_cv_max.

    Returns rate in Hz, or np.nan on failure.
    """
    x = x.astype(np.float64)
    N = len(x)
    if N < int(fs / f_lo):
        return np.nan

    # Step 1: spectral guidance — expected rate from Welch PSD
    f_expected = rate_spectral(x, f_lo, f_hi, fs)
    if not np.isfinite(f_expected) or f_expected <= 0:
        f_expected = (f_lo + f_hi) / 2.0

    expected_period_s = 1.0 / f_expected
    min_dist = max(1, int(0.6 * expected_period_s * fs))

    # Step 2: amplitude-adaptive prominence via rolling MAD
    env = np.abs(x)
    mad_win = max(3, int(expected_period_s * fs))
    kernel = np.ones(mad_win) / mad_win
    local_mean = np.convolve(env, kernel, mode='same')
    local_mad = np.convolve(np.abs(env - local_mean), kernel, mode='same')
    med_mad = np.median(local_mad)
    if med_mad < 1e-12:
        med_mad = np.std(x) * 0.3
    prom_threshold = 0.5 * med_mad

    # Step 3: mild smoothing then peak detection
    smooth_win = max(3, min_dist // 4)
    x_sm = np.convolve(x, np.ones(smooth_win) / smooth_win, mode='same')
    pks, props = find_peaks(x_sm, distance=min_dist, prominence=prom_threshold)

    if len(pks) < 2:
        return np.nan

    # Step 4: IPI validation — reject if too irregular
    ipis = np.diff(pks) / fs
    ipi_mean = ipis.mean()
    if ipi_mean <= 0:
        return np.nan
    ipi_cv = ipis.std() / ipi_mean
    if ipi_cv > ipi_cv_max:
        # Fallback: keep only IPIs within 1.5 MAD of the median
        ipi_med = np.median(ipis)
        ipi_mad = np.median(np.abs(ipis - ipi_med))
        if ipi_mad < 1e-6:
            ipi_mad = 0.1 * ipi_med
        good = np.abs(ipis - ipi_med) <= 1.5 * ipi_mad
        if good.sum() < 2:
            return np.nan
        ipi_mean = ipis[good].mean()

    rate_hz = 1.0 / ipi_mean
    if rate_hz < f_lo or rate_hz > f_hi:
        return np.nan
    return float(rate_hz)


# ── Multi-method dispatcher ────────────────────────────────────────────────────

def estimate_rate(x: np.ndarray, f_lo: float, f_hi: float, fs: float = FS,
                   include_envelope: bool = False) -> dict:
    """
    Run rate estimators; return {method: rate_Hz}.

    Parameters
    ----------
    include_envelope : if True, also include Teager-Kaiser envelope ACF estimate.
        Typically enabled for cardiac band, disabled for respiratory.
    """
    out = {
        'spectral':        rate_spectral(x,  f_lo, f_hi, fs),
        'acf':             rate_acf(x,       f_lo, f_hi, fs),
        'hilbert':         rate_hilbert(x,   f_lo, f_hi, fs),
        'zerocross':       rate_zerocross(x, fs),
        'peaks':           rate_peaks(x,     f_lo, f_hi, fs),
        'adaptive_peaks':  rate_adaptive_peaks(x, f_lo, f_hi, fs),
    }
    if include_envelope:
        out['envelope'] = rate_envelope(x, f_lo, f_hi, fs)
    return out


# ── Fusion across methods ─────────────────────────────────────────────────────

def fuse_rates(rates_hz: dict, f_lo: float, f_hi: float,
                how: str = 'median') -> float:
    """
    Combine multiple method estimates into one rate.

    Methods
    -------
    'median'    : plain median of finite values inside [f_lo, f_hi]
    'trimmed'   : median after dropping the extreme low + high values
    'weighted'  : inverse-distance-from-median weighted mean (robust)
    """
    vals = np.array([v for v in rates_hz.values()
                      if v is not None and np.isfinite(v)
                      and f_lo <= v <= f_hi])
    if len(vals) == 0:
        return np.nan
    if len(vals) == 1 or how == 'median':
        return float(np.median(vals))
    if how == 'trimmed' and len(vals) >= 3:
        trimmed = np.sort(vals)[1:-1]
        return float(np.median(trimmed))
    if how == 'weighted':
        med = np.median(vals)
        d = np.abs(vals - med) + 1e-6
        w = 1.0 / d
        return float(np.sum(w * vals) / np.sum(w))
    return float(np.median(vals))


# ── Peak / crossing index functions ───────────────────────────────────────────

def detect_peaks(sig: np.ndarray, f_lo: float, f_hi: float,
                 fs: float = FS, prom_factor: float = 0.4) -> np.ndarray:
    """Return peak sample indices using prominence-based detection."""
    dist       = max(1, int(0.9 * fs / f_hi))
    smooth_win = max(3, dist // 4)
    sig_sm     = np.convolve(sig.astype(np.float64), np.ones(smooth_win) / smooth_win, mode='same')
    pks, _     = find_peaks(sig_sm, distance=dist, prominence=prom_factor * np.std(sig_sm))
    return pks


def zerocross_indices(x: np.ndarray) -> np.ndarray:
    """Return sample indices of upward zero crossings (sub-sample interpolated → nearest int)."""
    x = x.astype(np.float64)
    signs = np.sign(x)
    signs[signs == 0] = 1
    cross_raw = np.where(np.diff(signs) > 0)[0]
    result = [
        int(round(i + (-x[i] / (x[i + 1] - x[i] + 1e-12))))
        for i in cross_raw if i + 1 < len(x)
    ]
    return np.array(result, dtype=int)


def peaks_by_method(sig: np.ndarray, f_lo: float, f_hi: float, method: str,
                    fs: float = FS, prom_factor: float = 0.4) -> np.ndarray:
    """
    Return peak/crossing indices using the chosen method.

    For rate-based methods (acf, spectral, hilbert), the estimated period
    constrains the min-distance of find_peaks so detected positions are
    real signal maxima consistent with the rate estimate.
    """
    if method == 'peaks':
        return detect_peaks(sig, f_lo, f_hi, fs, prom_factor)
    if method == 'zerocross':
        return zerocross_indices(sig)
    # rate-estimated methods: derive min_dist from estimated period
    rate_fn = {'acf': rate_acf, 'spectral': rate_spectral, 'hilbert': rate_hilbert}[method]
    rate_hz = rate_fn(sig, f_lo, f_hi, fs)
    if np.isnan(rate_hz) or rate_hz <= 0:
        return detect_peaks(sig, f_lo, f_hi, fs, prom_factor)   # fallback
    min_dist = max(1, int(0.85 / rate_hz * fs))
    pks, _   = find_peaks(sig, distance=min_dist, prominence=prom_factor * np.std(sig))
    return pks


# ── Scaled estimators (per-session k removes the systematic overcount) ────────
#
# Both bands' CLE-CRE bandpass reliably over-counts the true rate by a stable
# per-session factor:
#   - resp    : ~1.3× (inhale + exhale produce two bumps on many breaths)
#   - cardiac : ~1.7× (systolic + dicrotic-like bump on each heart cycle)
# Learn k per session with calibrate_k_*(), then divide the raw estimate by k.
# See notebooks/ANALYSIS_LOG.md (2026-04-16) and peak_ratio_method_writeup.md.

def rate_hilbert_scaled_cardiac(x: np.ndarray, k: float,
                                 fs: float = FS,
                                 f_lo: float = CARD_LO,
                                 f_hi: float = CARD_HI) -> float:
    """Cardiac rate from Hilbert inst. frequency, divided by a calibrated k.

    Raw rate_hilbert on CLE-CRE cardiac bandpass over-counts by ~1.5–1.9×
    per session. A scalar k learned from paired CAP+Pleth windows removes
    the overcount. Cross-12-session median k = 1.67 (range [1.48, 1.93]).

    Parameters
    ----------
    x  : 1-D bandpassed cardiac signal
    k  : per-session scaling factor (use calibrate_k_cardiac(session) to fit)

    Returns
    -------
    rate in Hz, or np.nan if hilbert fails or k <= 0.
    """
    if k is None or not np.isfinite(k) or k <= 0:
        return np.nan
    r = rate_hilbert(x, f_lo, f_hi, fs)
    return r / k if np.isfinite(r) else np.nan


def rate_peaks_scaled_resp(x: np.ndarray, k: float,
                            fs: float = FS,
                            prom_factor: float = 0.05,
                            min_dist_s: float = 0.4) -> float:
    """Respiratory rate from loose peak counting, divided by a calibrated k.

    Default rate_peaks uses a strict detector (prom=0.4σ, min_dist=1.8 s)
    that is sensitive to breath-to-breath variation. This scaled variant
    uses a loose detector (prom=0.05σ, min_dist=0.4 s) that consistently
    catches both inhale and exhale bumps, then divides by a session-learned
    k (typical range [1.2, 1.6], cross-session median ~1.3).

    Parameters
    ----------
    x           : 1-D bandpassed resp signal
    k           : per-session scaling factor (use calibrate_k_resp(session))
    prom_factor : loose-detector prominence (default 0.05)
    min_dist_s  : loose-detector min peak distance in seconds (default 0.4)

    Returns
    -------
    rate in Hz, or np.nan if detection fails or k <= 0.
    """
    if k is None or not np.isfinite(k) or k <= 0:
        return np.nan
    x = x.astype(np.float64)
    if len(x) < 16:
        return np.nan
    min_dist   = max(1, int(round(min_dist_s * fs)))
    smooth_win = max(3, min_dist // 4)
    x_sm = np.convolve(x, np.ones(smooth_win) / smooth_win, mode='same')
    pks, _ = find_peaks(x_sm, distance=min_dist,
                         prominence=prom_factor * np.std(x_sm))
    if len(pks) < 2:
        return np.nan
    duration = len(x) / fs
    return (len(pks) / k) / duration


# ── Per-session k calibration ─────────────────────────────────────────────────

def _calibrate_k(session, band: str, *,
                  n_windows: int = 50,
                  win_s: float = 60.0,
                  seed: int = 42) -> float:
    """Learn median(rate_cap_raw / rate_gt) across random 1-min windows.

    Shared core for calibrate_k_cardiac and calibrate_k_resp.
    """
    # Local imports avoid any risk of import cycle if rates.py is imported
    # before preprocessing/filters at package init.
    from .preprocessing import remove_acc_artifact
    from .filters import bandpass

    if band == 'cardiac':
        f_lo, f_hi = CARD_LO, CARD_HI
        gt_channel = 'Pleth'
        cap_fn = lambda seg, fs: rate_hilbert(seg, f_lo, f_hi, fs)
    elif band == 'resp':
        f_lo, f_hi = RESP_LO, RESP_HI
        gt_channel = 'Thorax'
        # k=1.0 makes rate_peaks_scaled_resp return the un-divided loose rate
        cap_fn = lambda seg, fs: rate_peaks_scaled_resp(seg, k=1.0, fs=fs)
    else:
        raise ValueError(f"band must be 'cardiac' or 'resp', got {band!r}")

    fs = session.fs
    raw_cap = (session.cap['CLE'].astype(np.float64)
                - session.cap['CRE'].astype(np.float64))
    acc = session.cap['acc_mag'].astype(np.float64)
    sig = remove_acc_artifact(raw_cap, acc, f_lo, f_hi, fs)
    gt  = bandpass(session.psg[gt_channel].astype(np.float64), f_lo, f_hi, fs)

    win_n = int(round(win_s * fs))
    total = len(sig)
    if total < win_n * 2:
        return float('nan')

    rng = np.random.default_rng(seed)
    max_start = total - win_n - 1
    starts = sorted(rng.integers(0, max_start, size=n_windows).tolist())

    ratios = []
    for st in starts:
        seg    = sig[st:st + win_n]
        seg_gt = gt [st:st + win_n]
        r_cap = cap_fn(seg, fs)
        r_gt  = rate_acf(seg_gt, f_lo, f_hi, fs, prominence=0.05)
        if np.isfinite(r_cap) and np.isfinite(r_gt) and r_gt > 0:
            ratios.append(r_cap / r_gt)

    if len(ratios) < 10:
        return float('nan')
    return float(np.median(ratios))


def calibrate_k_cardiac(session, n_windows: int = 50,
                         win_s: float = 60.0, seed: int = 42) -> float:
    """Fit the cardiac scaling factor `k` for rate_hilbert_scaled_cardiac.

    Returns the median of rate_hilbert(CLE-CRE cardiac bp) / rate_acf(Pleth bp)
    across N random `win_s`-second windows. Cross-session median is 1.67
    (12-session range [1.48, 1.93]). Calibrate per-night when possible —
    night-to-night |Δk| up to 0.19 observed on some subjects.

    Returns
    -------
    k : float, or np.nan if fewer than 10 windows produced valid ratios.
    """
    return _calibrate_k(session, 'cardiac',
                        n_windows=n_windows, win_s=win_s, seed=seed)


def calibrate_k_resp(session, n_windows: int = 50,
                      win_s: float = 60.0, seed: int = 42) -> float:
    """Fit the respiratory scaling factor `k` for rate_peaks_scaled_resp.

    Returns the median of loose-peak-rate(CLE-CRE resp bp) / rate_acf(Thorax bp)
    across N random `win_s`-second windows. Cross-session median is ~1.3
    (12-session range [1.18, 1.61]; clusters by subject). Calibrate per-night
    when possible.

    Returns
    -------
    k : float, or np.nan if fewer than 10 windows produced valid ratios.
    """
    return _calibrate_k(session, 'resp',
                        n_windows=n_windows, win_s=win_s, seed=seed)


# ── Sliding-window rate estimation ─────────────────────────────────────────────

def sliding_rates(
    signal: np.ndarray,
    f_lo: float,
    f_hi: float,
    fs: float = FS,
    win_sec: float = 20.0,
    step_sec: float = 1.0,
) -> tuple:
    """
    Sliding-window rate estimation using all six methods.

    Parameters
    ----------
    signal   : 1-D bandpassed signal
    f_lo     : lower frequency bound (Hz)
    f_hi     : upper frequency bound (Hz)
    fs       : sampling rate
    win_sec  : window length in seconds
    step_sec : step size in seconds

    Returns
    -------
    (t_s, rates)
        t_s   : (K,) array of window centre times in seconds
        rates : {method: (K,) array of rates in Hz}
    """
    win_n  = int(round(win_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))
    t_list = []
    r_list: dict = {m: [] for m in METHOD_NAMES}

    for start in range(0, len(signal) - win_n + 1, step_n):
        seg = signal[start:start + win_n]
        t_list.append((start + win_n / 2.0) / fs)
        res = estimate_rate(seg, f_lo, f_hi, fs)
        for m in METHOD_NAMES:
            r_list[m].append(res[m])

    t_s = np.array(t_list)
    rates = {m: np.array(r_list[m]) for m in METHOD_NAMES}
    return t_s, rates


# ── Kalman rate tracker ──────────────────────────────────────────────────────

def kalman_rate_track(
    estimates: dict,
    f_lo: float,
    f_hi: float,
    step_sec: float = 30.0,
    max_delta_hz: float | None = None,
    R_base: dict | None = None,
) -> np.ndarray:
    """
    Kalman-filter fusion of multiple per-window rate estimates.

    Fuses two or more method time series into a single smooth rate track
    with physiological rate-of-change constraints.

    Parameters
    ----------
    estimates : {method_name: (K,) array of rate_Hz}.
        NaN entries are skipped (infinite measurement noise).
    f_lo, f_hi : frequency band bounds (Hz). Used to clamp output and
        initialise the state when no observations are available.
    step_sec : time between consecutive windows (seconds).
    max_delta_hz : maximum rate change per step (Hz). Controls process
        noise Q. Default: 2 br/min per 30s for resp, 5 BPM per 30s for
        cardiac, auto-selected from f_hi.
    R_base : {method_name: measurement_variance_Hz2}. Default derived
        from Phase 0 benchmark MAEs.

    Returns
    -------
    (K,) array of Kalman-smoothed rates in Hz. NaN where no observations
    were available and the prediction drifted outside the band.
    """
    methods = list(estimates.keys())
    K = len(next(iter(estimates.values())))

    if max_delta_hz is None:
        if f_hi <= 0.6:
            max_delta_hz = 2.0 / 60.0 * (step_sec / 30.0)
        else:
            max_delta_hz = 5.0 / 60.0 * (step_sec / 30.0)

    Q = max_delta_hz ** 2

    if R_base is None:
        if f_hi <= 0.6:
            R_base = {m: (2.5 / 60.0) ** 2 for m in methods}
        else:
            R_base = {m: (30.0 / 60.0) ** 2 for m in methods}

    f_mid = (f_lo + f_hi) / 2.0
    x = f_mid
    P = ((f_hi - f_lo) / 2.0) ** 2

    out = np.full(K, np.nan)

    for k in range(K):
        x_pred = x
        P_pred = P + Q

        obs = []
        R_obs = []
        for m in methods:
            z = estimates[m][k]
            if np.isfinite(z) and f_lo <= z <= f_hi:
                obs.append(z)
                R_obs.append(R_base[m])

        if len(obs) == 0:
            x = x_pred
            P = P_pred
            if f_lo <= x <= f_hi:
                out[k] = x
            continue

        for z_i, r_i in zip(obs, R_obs):
            S = P_pred + r_i
            K_gain = P_pred / S
            x_pred = x_pred + K_gain * (z_i - x_pred)
            P_pred = (1.0 - K_gain) * P_pred

        x = np.clip(x_pred, f_lo, f_hi)
        P = P_pred
        out[k] = x

    return out
