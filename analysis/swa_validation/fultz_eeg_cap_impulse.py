"""
Fultz-style EEG->CAP lead-lag / impulse-response analysis.

Motivation
----------
Fultz et al. (Science 2019) found that the amplitude envelope of slow-delta EEG
(0.2-4 Hz) does NOT correlate with CSF inflow at zero lag, but LEADS it by ~6.4 s
via a best-fit impulse response. Our prior SWA-validation tested only *zero-lag*
spectral agreement (CAP delta power vs EEG delta power -> r~0.015) and therefore
could not have detected a delayed, downstream coupling.

This script asks the Fultz question of our data: does the slow-delta EEG envelope
LEAD a slow oscillation in the capacitive temple sensor (CAP), the way it leads
CSF? The CAP slow signal plays the role of CSF/BOLD.

Method (per session, NREM only)
-------------------------------
1. EEG drive        : bandpass 0.2-4 Hz -> Hilbert envelope -> smooth.
2. CAP slow target  : several candidate CAP channels, bandpassed to the slow
                      oscillation band (0.01-0.1 Hz, Fultz CSF ~0.05 Hz).
3. Restrict to contiguous NREM (N2+N3), low-motion segments.
4. Cross-correlation over +-LAG_MAX s, with circular-shift null (95% band).
   Positive lag = EEG leads CAP (the Fultz sign convention).
5. Best-fit impulse response EEG_env -> CAP via ridge (Tikhonov) regression.
6. Peak-locked average of EEG envelope around slow-CAP peaks.

Usage
-----
    py analysis/swa_validation/fultz_eeg_cap_impulse.py            # default S2N2
    py analysis/swa_validation/fultz_eeg_cap_impulse.py --session 3
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, decimate, find_peaks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.config import FS, STAGE_LABELS

CACHE_DIR = Path(r'C:\Users\adity\AppData\Local\Temp\claude'
                 r'\C--Users-adity-Documents-sleep-monitor-code'
                 r'\86e66664-fe54-44f7-82d6-f69a07979595\scratchpad\sess_cache')

# ── Parameters ────────────────────────────────────────────────────────────────
EEG_LO, EEG_HI = 0.2, 4.0        # Fultz slow-delta band
SLOW_LO, SLOW_HI = 0.01, 0.1     # slow oscillation band (CSF ~0.05 Hz)
ENV_SMOOTH_S = 2.0               # envelope smoothing (s)
ANALYSIS_FS = 5.0                # common rate for lag analysis (Hz)
LAG_MAX_S = 40.0                 # +- lag window (s)
NREM_CODES = (1, 2)             # 1=N3, 2=N2  (see config.STAGE_LABELS)
MIN_SEG_S = 120.0               # minimum contiguous NREM segment (s)
IR_LEN_S = 30.0                 # impulse-response length (s)
IR_RIDGE = 1.0                  # ridge penalty (relative)
N_SHUFFLE = 500                 # circular-shift null iterations
PEAK_WIN_S = 40.0               # peak-locked averaging half-window (s)

CAP_CANDIDATES = ['CLE', 'CRE', 'CH', 'CLE-CRE']


# ── Filters ───────────────────────────────────────────────────────────────────
def _bandpass(sig, fs, lo, hi, order=4):
    """Zero-phase Butterworth bandpass (SOS). Cheap even for very low cutoffs."""
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, sig.astype(np.float64))


def _smooth(sig, fs, win_s):
    n = max(1, int(fs * win_s))
    k = np.ones(n) / n
    return np.convolve(sig, k, mode='same')


def _zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def _decimate_to(sig, fs_in, fs_out):
    q = int(round(fs_in / fs_out))
    if q <= 1:
        return sig.astype(np.float64), fs_in
    # decimate in stages if q large
    out = sig.astype(np.float64)
    while q > 13:
        out = decimate(out, 10, ftype='fir', zero_phase=True)
        q = int(round((fs_in / 10) / fs_out))
        fs_in = fs_in / 10
    out = decimate(out, q, ftype='fir', zero_phase=True)
    return out, fs_in / q


from types import SimpleNamespace


def load_cached(idx):
    """Load a session, caching the few channels + sleep profile we need as .npz."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f'sess_{idx}.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        cap = {k: z[f'cap_{k}'] for k in ('CLE', 'CRE', 'CH')}
        psg = {'EEG': z['EEG']}
        prof = {'t_ep_hr': z['t_ep_hr'], 'codes': z['codes']}
        return SimpleNamespace(label=str(z['label']), cap=cap, psg=psg, profile=prof)
    sess = load_session(idx)
    prof = load_sleep_profile(sess)
    if prof is None:
        raise RuntimeError('No sleep profile for this session.')
    np.savez_compressed(
        cache, label=sess.label,
        EEG=sess.psg['EEG'].astype(np.float32),
        cap_CLE=sess.cap['CLE'].astype(np.float32),
        cap_CRE=sess.cap['CRE'].astype(np.float32),
        cap_CH=sess.cap['CH'].astype(np.float32),
        t_ep_hr=np.asarray(prof['t_ep_hr']), codes=np.asarray(prof['codes']),
    )
    return SimpleNamespace(
        label=sess.label,
        cap={k: sess.cap[k] for k in ('CLE', 'CRE', 'CH')},
        psg={'EEG': sess.psg['EEG']},
        profile={'t_ep_hr': np.asarray(prof['t_ep_hr']), 'codes': np.asarray(prof['codes'])},
    )


# ── NREM segmentation ─────────────────────────────────────────────────────────
def nrem_segments(sess, n_samples, fs):
    """Contiguous NREM runs (>= MIN_SEG_S) as (start, stop) sample indices at `fs`."""
    prof = sess.profile
    t_ep_hr = prof['t_ep_hr']
    codes = np.asarray(prof['codes'])
    epoch_samp = int(30.0 * fs)
    is_nrem = np.zeros(n_samples, dtype=bool)
    for t_hr, c in zip(t_ep_hr, codes):
        if c in NREM_CODES:
            s = int(t_hr * 3600.0 * fs)
            e = min(s + epoch_samp, n_samples)
            if 0 <= s < n_samples:
                is_nrem[s:e] = True
    segs = []
    idx = np.flatnonzero(is_nrem)
    if len(idx) == 0:
        return segs, is_nrem
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    stops = np.concatenate([idx[breaks], [idx[-1]]])
    for s, e in zip(starts, stops):
        if (e - s) / fs >= MIN_SEG_S:
            segs.append((int(s), int(e) + 1))
    return segs, is_nrem


# ── CAP channel extraction ────────────────────────────────────────────────────
def cap_channel(sess, name):
    if name == 'CLE-CRE':
        cle = sess.cap['CLE'].astype(np.float64)
        cre = sess.cap['CRE'].astype(np.float64)
        # OLS differential: CLE - beta*CRE (project default channel)
        beta = np.dot(cle - cle.mean(), cre - cre.mean()) / np.dot(cre - cre.mean(), cre - cre.mean())
        return cle - beta * cre
    return sess.cap[name].astype(np.float64)


# ── Cross-correlation with circular-shift null (FFT-based) ────────────────────
def _xcorr_core(d_z, t_z, lag_max):
    """FFT cross-correlation. Returns xc over lags -lag_max..+lag_max where
    xc[L] = mean_n d[n] * t[n+L].  Positive lag => d (drive) LEADS t (target)."""
    n = len(d_z)
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    D = np.fft.rfft(d_z, nf)
    T = np.fft.rfft(t_z, nf)
    cc = np.fft.irfft(np.conj(D) * T, nf) / n
    xc = np.concatenate([cc[nf - lag_max:], cc[:lag_max + 1]])  # lags -L..+L
    return xc


def xcorr_lagged(drive, target, fs, lag_max_s):
    lag_max = int(lag_max_s * fs)
    d = _zscore(drive); t = _zscore(target)
    xc = _xcorr_core(d, t, lag_max)
    lags = np.arange(-lag_max, lag_max + 1) / fs
    return lags, xc


def shuffle_null(drive, target, fs, lag_max_s, n_iter, rng):
    """Circular-shift null: max |xc| over the lag window for shifted targets."""
    lag_max = int(lag_max_s * fs)
    n = len(drive)
    d = _zscore(drive); t = _zscore(target)
    peak_null = np.empty(n_iter)
    for k in range(n_iter):
        shift = int(rng.integers(lag_max, n - lag_max))
        xc = _xcorr_core(d, np.roll(t, shift), lag_max)
        peak_null[k] = np.max(np.abs(xc))
    return peak_null


# ── Impulse response (ridge deconvolution) ────────────────────────────────────
def impulse_response(drive, target, fs, ir_len_s, ridge):
    """Fit target(t) = sum_k h[k] drive(t-k). Causal (k>=0). Ridge-regularized."""
    L = int(ir_len_s * fs)
    d = _zscore(drive)
    t = _zscore(target)
    n = len(d)
    # design matrix of lagged drive
    X = np.zeros((n - L, L))
    for k in range(L):
        X[:, k] = d[L - k - 1: n - k - 1]
    y = t[L:]
    A = X.T @ X
    lam = ridge * np.trace(A) / L
    h = np.linalg.solve(A + lam * np.eye(L), X.T @ y)
    pred = X @ h
    r = np.corrcoef(pred, y)[0, 1]
    lags = np.arange(L) / fs
    return lags, h, r


# ── Peak-locked averaging ─────────────────────────────────────────────────────
def peak_locked(drive, target, fs, win_s):
    w = int(win_s * fs)
    t = _zscore(target)
    d = _zscore(drive)
    peaks, _ = find_peaks(t, distance=int(10 * fs), prominence=0.5)
    peaks = peaks[(peaks > w) & (peaks < len(t) - w)]
    if len(peaks) == 0:
        return None, None, 0
    stack = np.stack([d[p - w:p + w] for p in peaks])
    tax = np.arange(-w, w) / fs
    return tax, stack, len(peaks)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(session_idx, outdir):
    rng = np.random.default_rng(0)
    sess = load_cached(session_idx)
    label = sess.label

    # ── EEG drive: 0.2-4 Hz envelope, computed at 100 Hz then decimated ─────────
    eeg = sess.psg['EEG'].astype(np.float64)
    eeg_bp = _bandpass(eeg, FS, EEG_LO, EEG_HI, order=4)
    eeg_env_full = _smooth(np.abs(hilbert(eeg_bp)), FS, ENV_SMOOTH_S)
    eeg_env, fs_a = _decimate_to(eeg_env_full, FS, ANALYSIS_FS)   # continuous, analysis rate
    n_a = len(eeg_env)

    # ── NREM runs at the analysis rate ──────────────────────────────────────────
    segs, is_nrem = nrem_segments(sess, n_a, fs_a)
    nrem_min = is_nrem.sum() / fs_a / 60.0
    print(f'{label}: analysis fs={fs_a:.2f} Hz, {len(segs)} NREM runs >= {MIN_SEG_S:.0f}s, '
          f'{nrem_min:.0f} min NREM total')
    if not segs:
        print('  No usable NREM segments — aborting.')
        return

    results = {}
    for cap_name in CAP_CANDIDATES:
        # Decimate the raw CAP to analysis rate, then slow-bandpass CONTINUOUSLY
        # (filtering per short segment would be dominated by edge transients).
        cap_a, _ = _decimate_to(cap_channel(sess, cap_name), FS, ANALYSIS_FS)
        cap_slow = _bandpass(cap_a, fs_a, SLOW_LO, SLOW_HI, order=2)

        drive_parts, targ_parts = [], []
        for s, e in segs:
            m = min(e, n_a, len(cap_slow)) - s
            drive_parts.append(_zscore(eeg_env[s:s + m]))
            targ_parts.append(_zscore(cap_slow[s:s + m]))
        drive = np.concatenate(drive_parts)
        target = np.concatenate(targ_parts)

        lags, xc = xcorr_lagged(drive, target, ANALYSIS_FS, LAG_MAX_S)
        peak_i = np.argmax(np.abs(xc))
        peak_lag = lags[peak_i]
        peak_r = xc[peak_i]

        null = shuffle_null(drive, target, ANALYSIS_FS, LAG_MAX_S, N_SHUFFLE, rng)
        thr = np.percentile(null, 95)
        sig = np.abs(peak_r) > thr

        ir_lags, ir_h, ir_r = impulse_response(drive, target, ANALYSIS_FS, IR_LEN_S, IR_RIDGE)
        tax, stack, n_peaks = peak_locked(drive, target, ANALYSIS_FS, PEAK_WIN_S)

        results[cap_name] = dict(
            lags=lags, xc=xc, peak_lag=peak_lag, peak_r=peak_r,
            null_thr=thr, sig=sig, ir_lags=ir_lags, ir_h=ir_h, ir_r=ir_r,
            tax=tax, stack=stack, n_peaks=n_peaks, fs=ANALYSIS_FS,
        )
        star = '  ***' if sig else ''
        print(f'  {cap_name:8}: peak |r|={abs(peak_r):.3f} at lag {peak_lag:+.1f}s '
              f'(EEG {"leads" if peak_lag>0 else "lags"} CAP)  '
              f'null95={thr:.3f}{star}   IR r={ir_r:.3f}  n_peaks={n_peaks}')

    _plot(sess, results, eeg_env, is_nrem, fs_a, outdir)
    return results


def _plot(sess, results, eeg_env, is_nrem, fs_a, outdir):
    label = sess.label
    fig, axes = plt.subplots(4, len(CAP_CANDIDATES), figsize=(4.6 * len(CAP_CANDIDATES), 13))
    fig.suptitle(f'{label}: slow-delta EEG (0.2-4 Hz) envelope vs slow CAP (0.01-0.1 Hz)\n'
                 f'Positive lag = EEG leads CAP  (Fultz-style)', fontsize=13)

    for j, cap_name in enumerate(CAP_CANDIDATES):
        R = results[cap_name]
        # Row 0: cross-correlation
        ax = axes[0, j]
        ax.plot(R['lags'], R['xc'], color='#2980B9')
        ax.axhline(R['null_thr'], ls='--', color='gray', lw=0.8)
        ax.axhline(-R['null_thr'], ls='--', color='gray', lw=0.8)
        ax.axvline(0, color='k', lw=0.6)
        ax.axvline(R['peak_lag'], color='#C0392B', lw=1.0,
                   label=f"{R['peak_lag']:+.1f}s, r={R['peak_r']:.2f}")
        ax.set_title(f'{cap_name}  xcorr')
        ax.set_xlabel('lag (s)  [+ = EEG leads]')
        ax.set_ylabel('corr')
        ax.legend(fontsize=8)

        # Row 1: impulse response
        ax = axes[1, j]
        ax.plot(R['ir_lags'], R['ir_h'], color='#8E44AD')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(f'impulse response (r={R["ir_r"]:.2f})')
        ax.set_xlabel('lag (s)')
        ax.set_ylabel('h')

        # Row 2: peak-locked average
        ax = axes[2, j]
        if R['stack'] is not None:
            m = R['stack'].mean(0)
            sem = R['stack'].std(0) / np.sqrt(len(R['stack']))
            ax.plot(R['tax'], m, color='#27AE60')
            ax.fill_between(R['tax'], m - sem, m + sem, color='#27AE60', alpha=0.25)
            ax.axvline(0, color='#C0392B', lw=1.0)
            ax.set_title(f'EEG env @ CAP peaks (n={R["n_peaks"]})')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('time from CAP peak (s)')
        ax.set_ylabel('EEG env (z)')

    # Row 3: spectrograms of EEG env and one CAP slow signal (context, per project rule)
    from scipy.signal import spectrogram
    nrem_idx = np.flatnonzero(is_nrem)
    s0, s1 = nrem_idx[0], nrem_idx[-1]
    nper = int(120 * fs_a)
    ax = axes[3, 0]
    f, t, Sxx = spectrogram(eeg_env[s0:s1], fs=fs_a, nperseg=nper, noverlap=nper // 2)
    fm = f <= 0.15
    ax.pcolormesh(t / 60, f[fm], 10 * np.log10(Sxx[fm] + 1e-12), shading='auto', cmap='magma')
    ax.set_title('EEG-envelope spectrogram (NREM span)')
    ax.set_xlabel('min'); ax.set_ylabel('Hz')

    cap_a, _ = _decimate_to(cap_channel(sess, 'CLE'), FS, ANALYSIS_FS)
    cap_slow = _bandpass(cap_a, fs_a, SLOW_LO, SLOW_HI, order=2)
    ax = axes[3, 1]
    f, t, Sxx = spectrogram(cap_slow[s0:s1], fs=fs_a, nperseg=nper, noverlap=nper // 2)
    ax.pcolormesh(t / 60, f[fm], 10 * np.log10(Sxx[fm] + 1e-12), shading='auto', cmap='viridis')
    ax.set_title('CAP-slow (CLE) spectrogram')
    ax.set_xlabel('min'); ax.set_ylabel('Hz')

    # example time series in remaining row-3 cells
    for j, cap_name in zip(range(2, len(CAP_CANDIDATES)), CAP_CANDIDATES[2:]):
        ax = axes[3, j]
        e_seg = min(s0 + int(300 * fs_a), s1)
        tt = np.arange(s0, e_seg) / fs_a / 60
        capj_a, _ = _decimate_to(cap_channel(sess, cap_name), FS, ANALYSIS_FS)
        capj_slow = _bandpass(capj_a, fs_a, SLOW_LO, SLOW_HI, order=2)
        ax.plot(tt, _zscore(eeg_env[s0:e_seg]), color='#2980B9', lw=0.7, label='EEG env')
        ax.plot(tt, _zscore(capj_slow[s0:e_seg]), color='#E67E22', lw=0.7, label=f'{cap_name} slow')
        ax.set_title(f'{cap_name}: 5-min example')
        ax.set_xlabel('min'); ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(outdir) / f'fultz_eeg_cap_{label}.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  saved {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=int, default=3, help='session index (default 3 = S2N2)')
    ap.add_argument('--outdir', default='analysis/swa_validation/outputs')
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    run(args.session, args.outdir)
