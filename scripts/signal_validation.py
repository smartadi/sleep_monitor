#!/usr/bin/env python
"""
Signal-level validation: prove the mask signal contains cardiac and respiratory
information at the waveform level, before any rate derivation.

Per-epoch (30 s) analyses for all 12 sessions:
  Level 2 — Spectral peak alignment (mask vs GT dominant frequency)
  Level 3 — Magnitude-squared coherence (mask vs GT)
  Level 4 — Waveform cross-correlation + surrogate test

Also: Left vs Right capacitor consistency check.

Outputs
-------
artifacts/signal_validation.parquet  — one row per epoch per session

Usage
-----
    python scripts/signal_validation.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, coherence as sig_coherence, correlate
from scipy.fft import rfft, irfft

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, RESP_LO, RESP_HI, STAGE_LABELS, PSG_EPOCH_SEC
from sleep_monitor.filters import bandpass
from sleep_monitor.loader import load_all_sessions
from sleep_monitor.preprocessing import remove_acc_artifact

# ── Validation frequency bands (per pathway doc) ────────────────────────────
RESP_BAND = (0.1, 0.5)    # Hz
CARD_BAND = (0.7, 4.0)    # Hz

# ── Parameters ──────────────────────────────────────────────────────────────
WIN_SEC   = 30.0
STEP_SEC  = 30.0
N_SURR    = 200           # surrogate iterations per epoch
SEED      = 42
OUT_DIR   = ROOT / 'artifacts'

# Motion threshold (accelerometer RMS, MAD units above median)
MOTION_MAD_THRESH = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def spectral_peak_freq(sig: np.ndarray, fs: float, band: tuple) -> tuple:
    """
    Dominant frequency and spectral SNR within a frequency band.

    Returns (peak_freq_hz, snr_db). SNR = peak power / mean noise floor in band.
    """
    nperseg = min(len(sig), int(fs * 4))
    if len(sig) < nperseg or nperseg < 16:
        return np.nan, np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any() or psd[mask].sum() == 0:
        return np.nan, np.nan
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    peak_idx = np.argmax(psd_band)
    peak_freq = freqs_band[peak_idx]
    peak_power = psd_band[peak_idx]
    # SNR: peak / mean of non-peak bins
    noise_mask = np.ones(len(psd_band), dtype=bool)
    lo = max(0, peak_idx - 1)
    hi = min(len(psd_band), peak_idx + 2)
    noise_mask[lo:hi] = False
    if noise_mask.sum() > 0:
        noise_floor = psd_band[noise_mask].mean()
        snr = 10 * np.log10(peak_power / (noise_floor + 1e-30))
    else:
        snr = np.nan
    return peak_freq, snr


def band_coherence(sig1: np.ndarray, sig2: np.ndarray, fs: float,
                   band: tuple) -> float:
    """Mean magnitude-squared coherence within a frequency band."""
    nperseg = min(len(sig1), int(fs * 4))
    if len(sig1) < nperseg or nperseg < 16:
        return np.nan
    freqs, coh = sig_coherence(sig1, sig2, fs=fs, nperseg=nperseg,
                               noverlap=nperseg // 2)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return np.nan
    return float(np.mean(coh[mask]))


def waveform_xcorr(sig1: np.ndarray, sig2: np.ndarray) -> tuple:
    """
    Normalised cross-correlation between two signals (same length).

    Returns (peak_r, lag_samples).
    """
    n = len(sig1)
    if n < 10:
        return np.nan, np.nan
    s1 = (sig1 - sig1.mean()) / (sig1.std() + 1e-12)
    s2 = (sig2 - sig2.mean()) / (sig2.std() + 1e-12)
    # Only check lags within ±0.5 seconds (avoid spurious distant peaks)
    max_lag = min(n // 2, int(FS * 0.5))
    xcorr = correlate(s1, s2, mode='full') / n
    mid = n - 1
    xcorr_window = xcorr[mid - max_lag:mid + max_lag + 1]
    peak_idx = np.argmax(np.abs(xcorr_window))
    peak_r = xcorr_window[peak_idx]
    lag = peak_idx - max_lag
    return float(peak_r), int(lag)


def phase_randomize(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomize a signal (preserves power spectrum, destroys phase)."""
    n = len(sig)
    X = rfft(sig)
    phases = rng.uniform(0, 2 * np.pi, len(X))
    phases[0] = 0  # DC component stays real
    if n % 2 == 0:
        phases[-1] = 0  # Nyquist stays real for even-length
    X_rand = np.abs(X) * np.exp(1j * phases)
    return irfft(X_rand, n=n).real


def surrogate_test(mask_sig: np.ndarray, gt_sig: np.ndarray,
                   n_surr: int, rng: np.random.Generator) -> tuple:
    """
    Surrogate test: compare real cross-correlation against phase-randomized GT.

    Returns (real_r, surr_mean, surr_std, p_value).
    """
    real_r, real_lag = waveform_xcorr(mask_sig, gt_sig)
    if np.isnan(real_r):
        return np.nan, np.nan, np.nan, np.nan

    surr_rs = np.empty(n_surr)
    for i in range(n_surr):
        gt_surr = phase_randomize(gt_sig, rng)
        r, _ = waveform_xcorr(mask_sig, gt_surr)
        surr_rs[i] = abs(r)

    p_value = float(np.mean(surr_rs >= abs(real_r)))
    return real_r, float(surr_rs.mean()), float(surr_rs.std()), p_value


def compute_motion_mask(acc_mag: np.ndarray, fs: float,
                        win_n: int, step_n: int) -> np.ndarray:
    """Compute per-epoch motion flag using accelerometer RMS + MAD threshold."""
    n = len(acc_mag)
    starts = np.arange(0, n - win_n + 1, step_n)
    k = len(starts)
    motion_rms = np.empty(k)
    for i, s0 in enumerate(starts):
        chunk = acc_mag[s0:s0 + win_n]
        motion_rms[i] = np.sqrt(np.mean((chunk - chunk.mean()) ** 2))
    med = np.median(motion_rms)
    mad = np.median(np.abs(motion_rms - med)) + 1e-12
    return motion_rms > (med + MOTION_MAD_THRESH * mad)


def assign_sleep_stage(t_hr: np.ndarray, profile: dict | None) -> np.ndarray:
    """Map each epoch centre time to a sleep stage code."""
    codes = np.full(len(t_hr), -1, dtype=np.int8)
    if profile is None:
        return codes
    epoch_dur_hr = PSG_EPOCH_SEC / 3600.0
    ep_c = profile['codes']
    for i, t in enumerate(t_hr):
        idx = int(t / epoch_dur_hr)
        if 0 <= idx < len(ep_c):
            codes[i] = ep_c[idx]
    return codes


# ═══════════════════════════════════════════════════════════════════════════════
# Main per-session computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_session(session, rng: np.random.Generator) -> pd.DataFrame:
    """Run all signal validation analyses for one session."""
    fs = session.fs
    label = session.label
    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    n_samples = session.n_samples

    # --- Prepare signals ---
    cap_cle = session.cap['CLE'].astype(np.float64)
    cap_cre = session.cap['CRE'].astype(np.float64)
    cap_diff = cap_cle - cap_cre
    acc = session.cap['acc_mag'].astype(np.float64)

    # GT signals
    gt_flow = session.psg['Flow'].astype(np.float64)
    gt_thorax = session.psg['Thorax'].astype(np.float64)
    gt_ecg = session.psg['ECG'].astype(np.float64)

    # Preprocess: accelerometer artifact removal for mask channels
    mask_resp_diff = remove_acc_artifact(cap_diff, acc, RESP_BAND[0], RESP_BAND[1], fs)
    mask_card_diff = remove_acc_artifact(cap_diff, acc, CARD_BAND[0], CARD_BAND[1], fs)
    mask_resp_cle = remove_acc_artifact(cap_cle, acc, RESP_BAND[0], RESP_BAND[1], fs)
    mask_resp_cre = remove_acc_artifact(cap_cre, acc, RESP_BAND[0], RESP_BAND[1], fs)
    mask_card_cle = remove_acc_artifact(cap_cle, acc, CARD_BAND[0], CARD_BAND[1], fs)
    mask_card_cre = remove_acc_artifact(cap_cre, acc, CARD_BAND[0], CARD_BAND[1], fs)

    # Bandpass GT signals into validation bands
    gt_flow_bp = bandpass(gt_flow, RESP_BAND[0], RESP_BAND[1], fs)
    gt_thorax_bp = bandpass(gt_thorax, RESP_BAND[0], RESP_BAND[1], fs)
    gt_ecg_bp = bandpass(gt_ecg, CARD_BAND[0], CARD_BAND[1], fs)

    # --- Epoch grid ---
    starts = np.arange(0, n_samples - win_n + 1, step_n)
    n_epochs = len(starts)
    t_hr = (starts + win_n / 2.0) / fs / 3600.0

    # --- Motion mask ---
    motion_flag = compute_motion_mask(acc, fs, win_n, step_n)

    # --- Sleep stage ---
    stage_codes = assign_sleep_stage(t_hr, session.sleep_profile)

    # --- Apnea codes ---
    apnea_codes = session.apnea_at(t_hr)

    # --- Pre-allocate output arrays ---
    cols = {
        # Spectral peak alignment
        'mask_resp_peak_hz': np.full(n_epochs, np.nan),
        'gt_flow_peak_hz': np.full(n_epochs, np.nan),
        'gt_thorax_peak_hz': np.full(n_epochs, np.nan),
        'mask_card_peak_hz': np.full(n_epochs, np.nan),
        'gt_ecg_peak_hz': np.full(n_epochs, np.nan),
        'mask_resp_snr_db': np.full(n_epochs, np.nan),
        'mask_card_snr_db': np.full(n_epochs, np.nan),
        # Coherence — respiratory
        'coh_diff_flow_resp': np.full(n_epochs, np.nan),
        'coh_diff_thorax_resp': np.full(n_epochs, np.nan),
        'coh_cle_flow_resp': np.full(n_epochs, np.nan),
        'coh_cre_flow_resp': np.full(n_epochs, np.nan),
        # Coherence — cardiac
        'coh_diff_ecg_card': np.full(n_epochs, np.nan),
        'coh_cle_ecg_card': np.full(n_epochs, np.nan),
        'coh_cre_ecg_card': np.full(n_epochs, np.nan),
        # Waveform cross-correlation
        'xcorr_resp_r': np.full(n_epochs, np.nan),
        'xcorr_resp_lag': np.full(n_epochs, np.nan),
        'xcorr_card_r': np.full(n_epochs, np.nan),
        'xcorr_card_lag': np.full(n_epochs, np.nan),
        # Surrogate test
        'surr_resp_real_r': np.full(n_epochs, np.nan),
        'surr_resp_p': np.full(n_epochs, np.nan),
        'surr_card_real_r': np.full(n_epochs, np.nan),
        'surr_card_p': np.full(n_epochs, np.nan),
        # Left vs Right consistency
        'lr_resp_r': np.full(n_epochs, np.nan),
        'lr_card_r': np.full(n_epochs, np.nan),
    }

    # --- Per-epoch computation ---
    for i, s0 in enumerate(starts):
        s1 = s0 + win_n

        # Mask channel segments (already bandpassed)
        m_resp = mask_resp_diff[s0:s1]
        m_card = mask_card_diff[s0:s1]
        m_resp_cle = mask_resp_cle[s0:s1]
        m_resp_cre = mask_resp_cre[s0:s1]
        m_card_cle = mask_card_cle[s0:s1]
        m_card_cre = mask_card_cre[s0:s1]

        # GT segments (bandpassed)
        g_flow = gt_flow_bp[s0:s1]
        g_thorax = gt_thorax_bp[s0:s1]
        g_ecg = gt_ecg_bp[s0:s1]

        # --- Level 2: Spectral peak alignment ---
        cols['mask_resp_peak_hz'][i], cols['mask_resp_snr_db'][i] = \
            spectral_peak_freq(m_resp, fs, RESP_BAND)
        cols['gt_flow_peak_hz'][i], _ = spectral_peak_freq(g_flow, fs, RESP_BAND)
        cols['gt_thorax_peak_hz'][i], _ = spectral_peak_freq(g_thorax, fs, RESP_BAND)
        cols['mask_card_peak_hz'][i], cols['mask_card_snr_db'][i] = \
            spectral_peak_freq(m_card, fs, CARD_BAND)
        cols['gt_ecg_peak_hz'][i], _ = spectral_peak_freq(g_ecg, fs, CARD_BAND)

        # --- Level 3: Coherence ---
        cols['coh_diff_flow_resp'][i] = band_coherence(m_resp, g_flow, fs, RESP_BAND)
        cols['coh_diff_thorax_resp'][i] = band_coherence(m_resp, g_thorax, fs, RESP_BAND)
        cols['coh_cle_flow_resp'][i] = band_coherence(m_resp_cle, g_flow, fs, RESP_BAND)
        cols['coh_cre_flow_resp'][i] = band_coherence(m_resp_cre, g_flow, fs, RESP_BAND)
        cols['coh_diff_ecg_card'][i] = band_coherence(m_card, g_ecg, fs, CARD_BAND)
        cols['coh_cle_ecg_card'][i] = band_coherence(m_card_cle, g_ecg, fs, CARD_BAND)
        cols['coh_cre_ecg_card'][i] = band_coherence(m_card_cre, g_ecg, fs, CARD_BAND)

        # --- Level 4: Waveform cross-correlation ---
        cols['xcorr_resp_r'][i], cols['xcorr_resp_lag'][i] = \
            waveform_xcorr(m_resp, g_flow)
        cols['xcorr_card_r'][i], cols['xcorr_card_lag'][i] = \
            waveform_xcorr(m_card, g_ecg)

        # --- Surrogate test (skip motion-contaminated epochs for speed) ---
        if not motion_flag[i]:
            r_r, _, _, p_r = surrogate_test(m_resp, g_flow, N_SURR, rng)
            cols['surr_resp_real_r'][i] = r_r
            cols['surr_resp_p'][i] = p_r

            r_c, _, _, p_c = surrogate_test(m_card, g_ecg, N_SURR, rng)
            cols['surr_card_real_r'][i] = r_c
            cols['surr_card_p'][i] = p_c

        # --- Left vs Right consistency ---
        cols['lr_resp_r'][i], _ = waveform_xcorr(m_resp_cle, m_resp_cre)
        cols['lr_card_r'][i], _ = waveform_xcorr(m_card_cle, m_card_cre)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        'session': label,
        'epoch_idx': np.arange(n_epochs),
        't_hr': t_hr,
        'stage_code': stage_codes,
        'stage': [STAGE_LABELS.get(c, '?') for c in stage_codes],
        'apnea_code': apnea_codes,
        'motion_flag': motion_flag[:n_epochs],
        **cols,
    })
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading all 12 sessions with sleep profiles + apnea events...")
    t0 = time.time()
    sessions = load_all_sessions(with_sleep_profiles=True, with_apnea=True)
    print(f"  Loaded {len(sessions)} sessions in {time.time() - t0:.1f}s\n")

    frames = []
    for s in sessions:
        print(f"Processing {s.label}...", flush=True)
        ts = time.time()
        df = compute_session(s, rng)
        elapsed = time.time() - ts
        n_clean = (~df['motion_flag']).sum()
        print(f"  {len(df)} epochs ({n_clean} clean), "
              f"resp coh mean={df['coh_diff_flow_resp'].mean():.3f}, "
              f"card coh mean={df['coh_diff_ecg_card'].mean():.3f}, "
              f"{elapsed:.1f}s")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # --- Save ---
    out_path = OUT_DIR / 'signal_validation.parquet'
    all_df.to_parquet(out_path, index=False)

    # --- Summary ---
    clean = all_df[~all_df['motion_flag']]
    print(f"\n{'='*70}")
    print(f"Total epochs: {len(all_df)} ({len(clean)} clean)")
    print(f"\nClean-epoch summary (mean ± std):")
    for col in ['coh_diff_flow_resp', 'coh_diff_ecg_card',
                'xcorr_resp_r', 'xcorr_card_r']:
        vals = clean[col].dropna()
        print(f"  {col:25s}: {vals.mean():.3f} ± {vals.std():.3f}  (n={len(vals)})")

    # Surrogate significance rate
    for band, col in [('resp', 'surr_resp_p'), ('card', 'surr_card_p')]:
        vals = clean[col].dropna()
        sig_pct = (vals < 0.05).mean() * 100
        print(f"  surrogate sig rate ({band:4s}): {sig_pct:.1f}%  (n={len(vals)})")

    print(f"\nSaved: {out_path} ({len(all_df)} rows)")


if __name__ == '__main__':
    main()
