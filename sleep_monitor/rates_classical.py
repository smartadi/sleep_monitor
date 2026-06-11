"""
Advanced classical rate estimators beyond the base five.

Methods
-------
rate_vmd       -- Variational Mode Decomposition dominant mode frequency
rate_cwt       -- Continuous Wavelet Transform ridge tracking
rate_stft_track -- STFT spectrogram with Viterbi-style peak tracking
rate_music     -- MUSIC pseudo-spectrum super-resolution

Each returns a scalar rate in Hz (or np.nan on failure), matching the
interface of the base estimators in rates.py.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import welch, find_peaks, stft as scipy_stft

from .config import FS


# ── VMD (Variational Mode Decomposition) ─────────────────────────────────────

def _vmd(signal: np.ndarray, K: int = 4, alpha: float = 2000.0,
         tau: float = 0.0, tol: float = 1e-7, max_iter: int = 200):
    """
    Variational Mode Decomposition.

    Decomposes signal into K band-limited intrinsic modes by solving a
    constrained optimization in the frequency domain.  Each mode has a
    learned center frequency.

    Parameters
    ----------
    signal   : 1-D real input
    K        : number of modes to extract
    alpha    : bandwidth constraint (higher = narrower modes)
    tau      : noise-tolerance (0 = exact reconstruction)
    tol      : convergence threshold on mode energy change
    max_iter : max ADMM iterations

    Returns
    -------
    u : (K, N) array of modes in time domain
    omega : (K,) center frequencies in normalised units [0, 0.5]
    """
    N = len(signal)
    half = N // 2

    # Mirror-extend to reduce boundary effects
    f_mirror = np.concatenate([signal[half-1::-1], signal, signal[-1:-half-1:-1]])
    T = len(f_mirror)

    # Frequency axis in normalised units [0, 0.5] for positive side
    freqs = np.arange(T) / T - 0.5

    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T // 2] = 0

    # Initialise center frequencies uniformly in [0, 0.5]
    omega_k = np.array([(0.5 / (K + 1)) * (k + 1) for k in range(K)])

    u_hat_k = np.zeros((T, K), dtype=complex)
    u_hat_k_prev = np.zeros((T, K), dtype=complex)
    lambda_hat = np.zeros(T, dtype=complex)

    for n in range(max_iter):
        u_hat_k_prev[:] = u_hat_k

        for k in range(K):
            sum_uk = np.sum(u_hat_k, axis=1) - u_hat_k[:, k]

            u_hat_k[:, k] = (
                (f_hat_plus - sum_uk + lambda_hat / 2.0)
                / (1.0 + alpha * (freqs - omega_k[k]) ** 2)
            )

            # Update center frequency (weighted mean of positive freqs)
            pos = T // 2
            power = np.abs(u_hat_k[pos:, k]) ** 2
            freq_pos = np.abs(freqs[pos:])
            denom = np.sum(power) + 1e-20
            omega_k[k] = np.dot(freq_pos, power) / denom

        lambda_hat += tau * (f_hat_plus - np.sum(u_hat_k, axis=1))

        uDiff = np.sum(np.abs(u_hat_k - u_hat_k_prev) ** 2) / T
        if uDiff < tol:
            break

    # Reconstruct modes
    u = np.zeros((K, N))
    for k in range(K):
        u_k = np.fft.ifftshift(u_hat_k[:, k])
        u_k = np.real(np.fft.ifft(u_k))
        u[k] = u_k[half:half + N]

    return u, omega_k


def rate_vmd(x: np.ndarray, f_lo: float, f_hi: float,
             fs: float = FS, K: int = 4, alpha: float = 2000.0) -> float:
    """
    Rate from Variational Mode Decomposition.

    Decomposes signal into K modes with learned center frequencies.
    Returns the center frequency of the mode whose energy is highest
    within the target band.
    """
    x = x.astype(np.float64)
    if len(x) < 64:
        return np.nan
    try:
        u, omega_norm = _vmd(x, K=K, alpha=alpha)
    except Exception:
        return np.nan

    omega_hz = omega_norm * fs

    best_freq = np.nan
    best_power = -1.0
    for k in range(K):
        if f_lo <= omega_hz[k] <= f_hi:
            power = np.sum(u[k] ** 2)
            if power > best_power:
                best_power = power
                best_freq = omega_hz[k]

    return float(best_freq)


# ── CWT Ridge Tracking ───────────────────────────────────────────────────────

def _morlet_cwt(x: np.ndarray, scales: np.ndarray, omega0: float = 6.0):
    """
    Continuous Wavelet Transform with Morlet wavelet.

    Returns complex CWT coefficients (n_scales, n_samples).
    """
    N = len(x)
    x = x - x.mean()
    x_hat = np.fft.fft(x)
    freqs_fft = np.fft.fftfreq(N) * 2 * np.pi

    cwt_out = np.zeros((len(scales), N), dtype=complex)
    for i, s in enumerate(scales):
        # Morlet wavelet in frequency domain
        norm = (np.pi ** -0.25) * np.sqrt(s)
        psi_hat = norm * np.exp(-0.5 * (s * freqs_fft - omega0) ** 2)
        psi_hat *= (s * freqs_fft > 0).astype(float)
        cwt_out[i] = np.fft.ifft(x_hat * np.conj(psi_hat))

    return cwt_out


def rate_cwt(x: np.ndarray, f_lo: float, f_hi: float,
             fs: float = FS, n_scales: int = 64, omega0: float = 6.0) -> float:
    """
    Rate from CWT ridge (maximum-amplitude scale at each time step).

    Uses a Morlet wavelet. The ridge is the scale with maximum amplitude
    across the window; the rate is the median frequency along the ridge,
    weighted by amplitude.
    """
    x = x.astype(np.float64)
    N = len(x)
    if N < 64:
        return np.nan

    # Build scales corresponding to f_lo..f_hi
    f_hi_safe = min(f_hi, fs / 2.0 - 0.01)
    f_lo_safe = max(f_lo, 0.01)
    scales = np.logspace(
        np.log10(omega0 / (2 * np.pi * f_hi_safe) * fs),
        np.log10(omega0 / (2 * np.pi * f_lo_safe) * fs),
        n_scales,
    )
    freqs_cwt = omega0 * fs / (2 * np.pi * scales)

    try:
        cwt_coeffs = _morlet_cwt(x, scales, omega0)
    except Exception:
        return np.nan

    power = np.abs(cwt_coeffs) ** 2

    # Ridge: at each time step, pick the scale with max power
    ridge_idx = np.argmax(power, axis=0)
    ridge_freq = freqs_cwt[ridge_idx]
    ridge_amp = np.array([power[ridge_idx[t], t] for t in range(N)])

    # Only keep frequencies in-band
    valid = (ridge_freq >= f_lo) & (ridge_freq <= f_hi) & (ridge_amp > 0)
    if valid.sum() < N * 0.3:
        return np.nan

    # Amplitude-weighted median
    f_valid = ridge_freq[valid]
    a_valid = ridge_amp[valid]
    sorted_idx = np.argsort(f_valid)
    f_sorted = f_valid[sorted_idx]
    a_sorted = a_valid[sorted_idx]
    cumw = np.cumsum(a_sorted)
    median_idx = np.searchsorted(cumw, cumw[-1] / 2.0)
    median_idx = min(median_idx, len(f_sorted) - 1)

    return float(f_sorted[median_idx])


# ── STFT + Viterbi-style Peak Tracking ────────────────────────────────────────

def rate_stft_track(x: np.ndarray, f_lo: float, f_hi: float,
                    fs: float = FS, nperseg: int = 0,
                    max_jump_hz: float = 0.15) -> float:
    """
    Rate from STFT spectrogram with continuity-constrained peak tracking.

    Builds an STFT, then finds the optimal frequency track through the
    spectrogram that maximises power while penalising large frequency
    jumps between frames (Viterbi-style dynamic programming).

    Parameters
    ----------
    max_jump_hz : maximum allowed frequency change per STFT frame.
        Acts as a physiological rate-of-change constraint.
    """
    x = x.astype(np.float64)
    N = len(x)
    if N < 64:
        return np.nan

    if nperseg <= 0:
        nperseg = min(N, max(64, int(fs * 4)))
    noverlap = nperseg * 3 // 4

    try:
        f, t_stft, Zxx = scipy_stft(x, fs=fs, nperseg=nperseg,
                                     noverlap=noverlap, boundary=None)
    except Exception:
        return np.nan

    power = np.abs(Zxx) ** 2
    band_mask = (f >= f_lo) & (f <= f_hi)
    if not band_mask.any():
        return np.nan

    band_f = f[band_mask]
    band_power = power[band_mask, :]
    n_freq, n_frames = band_power.shape

    if n_frames < 2 or n_freq < 2:
        return np.nan

    df = band_f[1] - band_f[0] if len(band_f) > 1 else 1.0
    max_jump_bins = max(1, int(np.ceil(max_jump_hz / df)))

    # Viterbi DP: cost[frame, freq_bin] = max cumulative log-power
    log_power = np.log(band_power + 1e-20)
    cost = np.full((n_frames, n_freq), -np.inf)
    back = np.zeros((n_frames, n_freq), dtype=int)

    cost[0, :] = log_power[:, 0]

    for t in range(1, n_frames):
        for j in range(n_freq):
            lo = max(0, j - max_jump_bins)
            hi = min(n_freq, j + max_jump_bins + 1)
            prev = cost[t - 1, lo:hi]
            best_local = np.argmax(prev)
            cost[t, j] = prev[best_local] + log_power[j, t]
            back[t, j] = lo + best_local

    # Traceback
    path = np.zeros(n_frames, dtype=int)
    path[-1] = np.argmax(cost[-1, :])
    for t in range(n_frames - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    tracked_freqs = band_f[path]
    return float(np.median(tracked_freqs))


# ── MUSIC Pseudo-Spectrum ─────────────────────────────────────────────────────

def rate_music(x: np.ndarray, f_lo: float, f_hi: float,
               fs: float = FS, n_signals: int = 2,
               n_scan: int = 500) -> float:
    """
    Rate from MUSIC (Multiple Signal Classification) pseudo-spectrum.

    Super-resolution spectral estimator that resolves closely-spaced
    frequencies better than Welch PSD.  Requires choosing the number
    of signal components (n_signals); uses eigendecomposition of the
    data covariance matrix.

    Parameters
    ----------
    n_signals : assumed number of sinusoidal components in the signal
    n_scan    : number of frequency bins to scan in [f_lo, f_hi]
    """
    x = x.astype(np.float64) - np.mean(x)
    N = len(x)
    # Correlation matrix dimension
    M = min(N // 3, max(16, int(fs * 2)))
    if M < n_signals + 2 or N < M + 1:
        return np.nan

    # Build Toeplitz-style data matrix
    n_snapshots = N - M + 1
    X = np.zeros((M, n_snapshots))
    for i in range(n_snapshots):
        X[:, i] = x[i:i + M]

    R = X @ X.T / n_snapshots

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(R)
    except np.linalg.LinAlgError:
        return np.nan

    # Noise subspace: eigenvectors for the smallest eigenvalues
    idx = np.argsort(eigenvalues)
    noise_vecs = eigenvectors[:, idx[:M - n_signals]]

    # Scan frequencies
    scan_freqs = np.linspace(f_lo, f_hi, n_scan)
    pseudo_spectrum = np.zeros(n_scan)

    for i, freq in enumerate(scan_freqs):
        a = np.exp(1j * 2 * np.pi * freq * np.arange(M) / fs)
        noise_proj = noise_vecs.conj().T @ a
        denom = np.real(np.dot(noise_proj.conj(), noise_proj))
        pseudo_spectrum[i] = 1.0 / (denom + 1e-20)

    peak_idx = np.argmax(pseudo_spectrum)
    peak_freq = scan_freqs[peak_idx]

    # Parabolic interpolation around peak
    if 0 < peak_idx < n_scan - 1:
        a0 = pseudo_spectrum[peak_idx - 1]
        a1 = pseudo_spectrum[peak_idx]
        a2 = pseudo_spectrum[peak_idx + 1]
        d = a0 - 2 * a1 + a2
        if abs(d) > 1e-12:
            delta = 0.5 * (a0 - a2) / d
            df = scan_freqs[1] - scan_freqs[0]
            peak_freq += delta * df

    return float(peak_freq)


# ── Multi-method dispatcher for advanced methods ─────────────────────────────

ADVANCED_METHOD_NAMES = ['vmd', 'cwt', 'stft_track', 'music']

_ESTIMATORS = {
    'vmd':        rate_vmd,
    'cwt':        rate_cwt,
    'stft_track': rate_stft_track,
    'music':      rate_music,
}


def estimate_rate_advanced(x: np.ndarray, f_lo: float, f_hi: float,
                           fs: float = FS,
                           methods: list | None = None) -> dict:
    """
    Run advanced rate estimators; return {method: rate_Hz}.

    Parameters
    ----------
    methods : list of method names to run. Default: all four.
    """
    if methods is None:
        methods = ADVANCED_METHOD_NAMES
    out = {}
    for m in methods:
        fn = _ESTIMATORS[m]
        out[m] = fn(x, f_lo, f_hi, fs)
    return out


def sliding_rates_advanced(
    signal: np.ndarray,
    f_lo: float,
    f_hi: float,
    fs: float = FS,
    win_sec: float = 20.0,
    step_sec: float = 2.0,
    methods: list | None = None,
) -> tuple:
    """
    Sliding-window rate estimation using advanced methods.

    Same interface as rates.sliding_rates but uses VMD/CWT/STFT/MUSIC.
    """
    if methods is None:
        methods = ADVANCED_METHOD_NAMES
    win_n = int(round(win_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))
    t_list = []
    r_list: dict = {m: [] for m in methods}

    for start in range(0, len(signal) - win_n + 1, step_n):
        seg = signal[start:start + win_n]
        t_list.append((start + win_n / 2.0) / fs)
        res = estimate_rate_advanced(seg, f_lo, f_hi, fs, methods=methods)
        for m in methods:
            r_list[m].append(res[m])

    t_s = np.array(t_list)
    rates = {m: np.array(r_list[m]) for m in methods}
    return t_s, rates
