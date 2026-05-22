#!/usr/bin/env python
"""
Enhanced signal validation — stronger tests for confirming CAP contains GT information.

Five analyses:
  1. Spectral peak frequency agreement (tolerance-based match rate)
  2. Peak-frequency coherence (coherence at GT's dominant freq, not band-averaged)
  3. Frequency tracking correlation (do mask & GT peak freqs covary across the night?)
  4. SNR-gated re-analysis (only evaluate high-SNR epochs)
  5. Canonical coherence (optimal 2-channel fusion upper bound)

Additionally: compare 5 L/R combination strategies to find which yields
strongest signal-level evidence:
  - CLE-CRE (difference)
  - (CLE+CRE)/2 (average)
  - CLE only
  - CRE only
  - PCA first component

And compare all single-channel results against canonical coherence (the
theoretical upper bound achievable by any linear combination of [CLE, CRE]).

Outputs
-------
artifacts/signal_validation_enhanced.parquet  — per-epoch metrics for all channel combos
artifacts/canonical_coherence.parquet         — per-epoch canonical coherence (2-channel)
artifacts/channel_comparison_summary.csv      — which combination is best

Usage
-----
    .venv\Scripts\python.exe scripts/signal_validation_enhanced.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, coherence as sig_coherence, csd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, STAGE_LABELS, PSG_EPOCH_SEC
from sleep_monitor.filters import bandpass
from sleep_monitor.loader import load_all_sessions
from sleep_monitor.preprocessing import remove_acc_artifact

# ── Frequency bands (per pathway doc) ────────────────────────────────────────
RESP_BAND = (0.1, 0.5)
CARD_BAND = (0.7, 4.0)

# ── Parameters ───────────────────────────────────────────────────────────────
WIN_SEC  = 30.0
STEP_SEC = 30.0
OUT_DIR  = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'validation_report'

# Spectral peak agreement tolerance (Hz)
# At 30s window with nperseg=30s, resolution = 1/30 = 0.033 Hz
# Allow ±1 bin for resp, ±2 bins for cardiac
RESP_FREQ_TOL = 0.05    # ~1.5 bins at 0.033 Hz resolution
CARD_FREQ_TOL = 0.10    # wider for cardiac (broader peaks, harmonics)

# SNR threshold for gated analysis (dB)
SNR_THRESH_DB = 3.0

# Welch segment lengths — key fix: use long segments for resp to get fine resolution
RESP_NPERSEG_SEC = 30.0   # full window → 0.033 Hz resolution in 0.1-0.5 Hz band
CARD_NPERSEG_SEC = 4.0    # short segments OK for cardiac (0.7-4.0 Hz is wide)

# Motion threshold
MOTION_MAD_THRESH = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Channel combination strategies
# ═══════════════════════════════════════════════════════════════════════════════

def combine_channels(cle: np.ndarray, cre: np.ndarray, method: str) -> np.ndarray:
    """Combine L and R cap channels using the specified strategy."""
    if method == 'diff':
        return cle - cre
    elif method == 'avg':
        return (cle + cre) / 2.0
    elif method == 'cle':
        return cle.copy()
    elif method == 'cre':
        return cre.copy()
    elif method == 'pca':
        # PCA on z-scored channels so neither dominates by variance alone
        cle_z = (cle - cle.mean()) / (cle.std() + 1e-12)
        cre_z = (cre - cre.mean()) / (cre.std() + 1e-12)
        X = np.column_stack([cle_z, cre_z])
        cov = X.T @ X / (len(X) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1_weights = eigvecs[:, -1]
        return X @ pc1_weights
    else:
        raise ValueError(f"Unknown method: {method}")


CHANNEL_METHODS = ['diff', 'avg', 'cle', 'cre', 'pca']


# ═══════════════════════════════════════════════════════════════════════════════
# Core analysis functions
# ═══════════════════════════════════════════════════════════════════════════════

def _get_nperseg_psd(fs: float, band: tuple, sig_len: int) -> int:
    """Welch segment length for PSD (peak detection, SNR). Maximise freq resolution."""
    if band[1] <= 0.5:
        nperseg = min(sig_len, int(fs * RESP_NPERSEG_SEC))
    else:
        nperseg = min(sig_len, int(fs * CARD_NPERSEG_SEC))
    return max(nperseg, 16)


def _get_nperseg_coh(fs: float, band: tuple, sig_len: int) -> int:
    """
    Welch segment length for COHERENCE. Needs multiple segments for meaningful
    estimates — coherence with 1 segment is trivially 1.0.

    Respiratory: 10 s segments → 5 segments in 30 s window, 0.1 Hz resolution
    Cardiac: 4 s segments → ~12 segments, 0.25 Hz resolution
    """
    if band[1] <= 0.5:
        nperseg = min(sig_len, int(fs * 10.0))  # 10 s → 5 segments
    else:
        nperseg = min(sig_len, int(fs * CARD_NPERSEG_SEC))
    return max(nperseg, 16)


def spectral_peak_and_snr(sig: np.ndarray, fs: float, band: tuple):
    """
    Return (peak_freq, snr_db, peak_power).

    SNR is computed as peak power relative to the median PSD across the full
    band (excluding ±2 bins around peak). For respiratory with few bins, we
    use the full spectrum outside the band as noise reference.
    """
    nperseg = _get_nperseg_psd(fs, band, len(sig))
    if len(sig) < nperseg:
        return np.nan, np.nan, np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not band_mask.any() or psd[band_mask].sum() == 0:
        return np.nan, np.nan, np.nan

    psd_band = psd[band_mask]
    freqs_band = freqs[band_mask]
    peak_idx = np.argmax(psd_band)
    peak_freq = freqs_band[peak_idx]
    peak_power = psd_band[peak_idx]

    # SNR: peak relative to noise floor
    # Try in-band noise first (exclude ±2 bins around peak)
    noise_mask = np.ones(len(psd_band), dtype=bool)
    lo = max(0, peak_idx - 2)
    hi = min(len(psd_band), peak_idx + 3)
    noise_mask[lo:hi] = False

    if noise_mask.sum() >= 3:
        noise_floor = np.median(psd_band[noise_mask])
    else:
        # Not enough in-band noise bins (narrow band) — use out-of-band noise
        # Take PSD from 0.5-5 Hz excluding the target band
        oob_mask = (freqs >= 0.5) & (freqs <= 5.0) & ~band_mask
        if oob_mask.sum() >= 3:
            noise_floor = np.median(psd[oob_mask])
        else:
            noise_floor = np.median(psd[psd > 0])

    snr = 10 * np.log10(peak_power / (noise_floor + 1e-30))
    return peak_freq, snr, peak_power


def coherence_at_frequency(sig1: np.ndarray, sig2: np.ndarray, fs: float,
                           target_freq: float, band: tuple) -> float:
    """Coherence value at a specific target frequency (nearest bin)."""
    nperseg = _get_nperseg_coh(fs, band, len(sig1))
    if len(sig1) < nperseg or np.isnan(target_freq):
        return np.nan
    freqs, coh = sig_coherence(sig1, sig2, fs=fs, nperseg=nperseg,
                               noverlap=nperseg // 2)
    idx = np.argmin(np.abs(freqs - target_freq))
    return float(coh[idx])


def band_coherence_mean(sig1: np.ndarray, sig2: np.ndarray, fs: float,
                        band: tuple) -> float:
    """Mean coherence within band."""
    nperseg = _get_nperseg_coh(fs, band, len(sig1))
    if len(sig1) < nperseg:
        return np.nan
    freqs, coh = sig_coherence(sig1, sig2, fs=fs, nperseg=nperseg,
                               noverlap=nperseg // 2)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return np.nan
    return float(np.mean(coh[mask]))


def canonical_coherence(x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
                        fs: float, band: tuple) -> dict:
    """
    Canonical coherence: find the linear combination of [x1, x2] that maximises
    coherence with y at each frequency, then summarise within band.

    At each frequency f:
        S_xx(f) = 2x2 cross-spectral matrix of [x1, x2]
        S_xy(f) = 2x1 cross-spectrum of [x1, x2] with y
        S_yy(f) = auto-spectrum of y
        canonical_coh(f) = S_xy' @ inv(S_xx) @ S_xy / S_yy

    Returns dict with:
        canon_coh_at_peak : canonical coherence at GT's dominant frequency
        canon_coh_band    : mean canonical coherence within band
        optimal_weights   : [w1, w2] at the GT peak frequency (how to mix L/R)
        canon_coh_spectrum: (freqs_in_band, coh_values) for plotting
    """
    nperseg = _get_nperseg_coh(fs, band, len(x1))
    if len(x1) < nperseg:
        return {'canon_coh_at_peak': np.nan, 'canon_coh_band': np.nan,
                'optimal_w1': np.nan, 'optimal_w2': np.nan}

    noverlap = nperseg // 2

    # Cross-spectral densities (complex)
    freqs, S_x1x1 = welch(x1, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, S_x2x2 = welch(x2, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, S_yy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, S_x1x2 = csd(x1, x2, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, S_x1y = csd(x1, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, S_x2y = csd(x2, y, fs=fs, nperseg=nperseg, noverlap=noverlap)

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not band_mask.any():
        return {'canon_coh_at_peak': np.nan, 'canon_coh_band': np.nan,
                'optimal_w1': np.nan, 'optimal_w2': np.nan}

    # Compute canonical coherence at each frequency in band
    canon_coh = np.full(len(freqs), np.nan)
    optimal_weights = np.full((len(freqs), 2), np.nan)

    for fi in np.where(band_mask)[0]:
        # 2x2 input cross-spectral matrix
        Sxx = np.array([
            [S_x1x1[fi].real, S_x1x2[fi]],
            [np.conj(S_x1x2[fi]), S_x2x2[fi].real]
        ], dtype=complex)

        # 2x1 input-output cross-spectrum
        Sxy = np.array([S_x1y[fi], S_x2y[fi]], dtype=complex)

        Syy = S_yy[fi].real

        if Syy < 1e-30:
            continue

        # Regularise Sxx for numerical stability
        Sxx += np.eye(2) * 1e-12 * np.trace(Sxx).real

        try:
            Sxx_inv = np.linalg.inv(Sxx)
        except np.linalg.LinAlgError:
            continue

        # Canonical coherence = Sxy' @ Sxx_inv @ Sxy / Syy
        cc = (np.conj(Sxy) @ Sxx_inv @ Sxy / Syy).real
        canon_coh[fi] = min(max(cc, 0.0), 1.0)  # clamp [0, 1]

        # Optimal weight vector: w = Sxx_inv @ Sxy (unnormalised)
        w = Sxx_inv @ Sxy
        w_norm = w / (np.abs(w).max() + 1e-12)  # normalise for interpretability
        optimal_weights[fi] = [w_norm[0].real, w_norm[1].real]

    # Find GT peak frequency for "at peak" value
    _, S_yy_psd = welch(y, fs=fs, nperseg=_get_nperseg_psd(fs, band, len(y)),
                        noverlap=_get_nperseg_psd(fs, band, len(y)) // 2)
    freqs_psd, _ = welch(y, fs=fs, nperseg=_get_nperseg_psd(fs, band, len(y)),
                         noverlap=_get_nperseg_psd(fs, band, len(y)) // 2)
    psd_band_mask = (freqs_psd >= band[0]) & (freqs_psd <= band[1])
    if psd_band_mask.any() and S_yy_psd[psd_band_mask].sum() > 0:
        gt_peak_freq = freqs_psd[psd_band_mask][np.argmax(S_yy_psd[psd_band_mask])]
        # Find nearest coherence freq bin
        peak_fi = np.argmin(np.abs(freqs - gt_peak_freq))
        coh_at_peak = canon_coh[peak_fi] if not np.isnan(canon_coh[peak_fi]) else np.nan
        w_at_peak = optimal_weights[peak_fi]
    else:
        coh_at_peak = np.nan
        w_at_peak = [np.nan, np.nan]

    # Band mean (excluding NaN)
    valid = canon_coh[band_mask]
    coh_band = float(np.nanmean(valid)) if np.any(~np.isnan(valid)) else np.nan

    return {
        'canon_coh_at_peak': coh_at_peak,
        'canon_coh_band': coh_band,
        'optimal_w1': w_at_peak[0],
        'optimal_w2': w_at_peak[1],
    }


def compute_motion_mask(acc_mag: np.ndarray, fs: float,
                        win_n: int, step_n: int) -> np.ndarray:
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


def assign_sleep_stage(t_hr: np.ndarray, profile) -> np.ndarray:
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
# Per-session computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_session(session) -> pd.DataFrame:
    fs = session.fs
    label = session.label
    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    n_samples = session.n_samples

    # Raw channels
    cap_cle = session.cap['CLE'].astype(np.float64)
    cap_cre = session.cap['CRE'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)

    # GT signals
    gt_flow = session.psg['Flow'].astype(np.float64)
    gt_ecg = session.psg['ECG'].astype(np.float64)

    # Epoch grid
    starts = np.arange(0, n_samples - win_n + 1, step_n)
    n_epochs = len(starts)
    t_hr = (starts + win_n / 2.0) / fs / 3600.0

    # Motion, stage, apnea
    motion_flag = compute_motion_mask(acc, fs, win_n, step_n)
    stage_codes = assign_sleep_stage(t_hr, session.sleep_profile)
    apnea_codes = session.apnea_at(t_hr)

    # Bandpass GT signals (full-length, then slice per epoch)
    gt_flow_bp = bandpass(gt_flow, RESP_BAND[0], RESP_BAND[1], fs)
    gt_ecg_bp = bandpass(gt_ecg, CARD_BAND[0], CARD_BAND[1], fs)

    # --- Canonical coherence: bandpass CLE and CRE independently ---
    cle_resp_bp = remove_acc_artifact(cap_cle, acc, RESP_BAND[0], RESP_BAND[1], fs)
    cre_resp_bp = remove_acc_artifact(cap_cre, acc, RESP_BAND[0], RESP_BAND[1], fs)
    cle_card_bp = remove_acc_artifact(cap_cle, acc, CARD_BAND[0], CARD_BAND[1], fs)
    cre_card_bp = remove_acc_artifact(cap_cre, acc, CARD_BAND[0], CARD_BAND[1], fs)

    canon_rows = []
    for i, s0 in enumerate(starts):
        s1 = s0 + win_n
        cc_resp = canonical_coherence(
            cle_resp_bp[s0:s1], cre_resp_bp[s0:s1], gt_flow_bp[s0:s1],
            fs, RESP_BAND)
        cc_card = canonical_coherence(
            cle_card_bp[s0:s1], cre_card_bp[s0:s1], gt_ecg_bp[s0:s1],
            fs, CARD_BAND)
        canon_rows.append({
            'session': label,
            'epoch_idx': i,
            't_hr': t_hr[i],
            'stage_code': stage_codes[i],
            'stage': STAGE_LABELS.get(int(stage_codes[i]), '?'),
            'apnea_code': apnea_codes[i],
            'motion_flag': motion_flag[i],
            'canon_resp_coh_at_peak': cc_resp['canon_coh_at_peak'],
            'canon_resp_coh_band': cc_resp['canon_coh_band'],
            'canon_resp_w_cle': cc_resp['optimal_w1'],
            'canon_resp_w_cre': cc_resp['optimal_w2'],
            'canon_card_coh_at_peak': cc_card['canon_coh_at_peak'],
            'canon_card_coh_band': cc_card['canon_coh_band'],
            'canon_card_w_cle': cc_card['optimal_w1'],
            'canon_card_w_cre': cc_card['optimal_w2'],
        })

    canon_df = pd.DataFrame(canon_rows)

    # --- For each channel combination, bandpass and compute per-epoch ---
    rows = []

    for ch_method in CHANNEL_METHODS:
        # Combine raw channels
        combined_raw = combine_channels(cap_cle, cap_cre, ch_method)

        # Artifact removal + bandpass
        cap_resp = remove_acc_artifact(combined_raw, acc, RESP_BAND[0], RESP_BAND[1], fs)
        cap_card = remove_acc_artifact(combined_raw, acc, CARD_BAND[0], CARD_BAND[1], fs)

        for i, s0 in enumerate(starts):
            s1 = s0 + win_n
            m_resp = cap_resp[s0:s1]
            m_card = cap_card[s0:s1]
            g_flow = gt_flow_bp[s0:s1]
            g_ecg = gt_ecg_bp[s0:s1]

            # --- 1. Spectral peak frequency ---
            mask_resp_freq, mask_resp_snr, _ = spectral_peak_and_snr(m_resp, fs, RESP_BAND)
            gt_resp_freq, _, _ = spectral_peak_and_snr(g_flow, fs, RESP_BAND)
            mask_card_freq, mask_card_snr, _ = spectral_peak_and_snr(m_card, fs, CARD_BAND)
            gt_card_freq, _, _ = spectral_peak_and_snr(g_ecg, fs, CARD_BAND)

            # Frequency agreement
            resp_freq_err = abs(mask_resp_freq - gt_resp_freq) if not (np.isnan(mask_resp_freq) or np.isnan(gt_resp_freq)) else np.nan
            card_freq_err = abs(mask_card_freq - gt_card_freq) if not (np.isnan(mask_card_freq) or np.isnan(gt_card_freq)) else np.nan
            resp_freq_match = 1.0 if (not np.isnan(resp_freq_err) and resp_freq_err <= RESP_FREQ_TOL) else 0.0
            card_freq_match = 1.0 if (not np.isnan(card_freq_err) and card_freq_err <= CARD_FREQ_TOL) else 0.0

            # --- 2. Peak-frequency coherence ---
            coh_at_resp_peak = coherence_at_frequency(m_resp, g_flow, fs, gt_resp_freq, RESP_BAND)
            coh_at_card_peak = coherence_at_frequency(m_card, g_ecg, fs, gt_card_freq, CARD_BAND)

            # --- Band-averaged coherence (for comparison) ---
            coh_band_resp = band_coherence_mean(m_resp, g_flow, fs, RESP_BAND)
            coh_band_card = band_coherence_mean(m_card, g_ecg, fs, CARD_BAND)

            rows.append({
                'session': label,
                'channel': ch_method,
                'epoch_idx': i,
                't_hr': t_hr[i],
                'stage_code': stage_codes[i],
                'stage': STAGE_LABELS.get(int(stage_codes[i]), '?'),
                'apnea_code': apnea_codes[i],
                'motion_flag': motion_flag[i],
                # Spectral peaks
                'mask_resp_freq': mask_resp_freq,
                'gt_resp_freq': gt_resp_freq,
                'mask_card_freq': mask_card_freq,
                'gt_card_freq': gt_card_freq,
                'resp_freq_err': resp_freq_err,
                'card_freq_err': card_freq_err,
                'resp_freq_match': resp_freq_match,
                'card_freq_match': card_freq_match,
                # SNR
                'mask_resp_snr': mask_resp_snr,
                'mask_card_snr': mask_card_snr,
                # Coherence at peak freq
                'coh_at_resp_peak': coh_at_resp_peak,
                'coh_at_card_peak': coh_at_card_peak,
                # Band-averaged coherence
                'coh_band_resp': coh_band_resp,
                'coh_band_card': coh_band_card,
            })

    return pd.DataFrame(rows), canon_df


# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing: frequency tracking + SNR gating
# ═══════════════════════════════════════════════════════════════════════════════

def compute_freq_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 3: Correlate mask peak freq time series with GT peak freq time series.
    Per session, per channel method.
    """
    results = []
    for (sess, ch), grp in df.groupby(['session', 'channel']):
        clean = grp[~grp['motion_flag']].sort_values('t_hr')

        for band, mask_col, gt_col in [
            ('resp', 'mask_resp_freq', 'gt_resp_freq'),
            ('card', 'mask_card_freq', 'gt_card_freq'),
        ]:
            ok = clean[mask_col].notna() & clean[gt_col].notna()
            if ok.sum() < 10:
                r_val = np.nan
            else:
                r_val, _ = pearsonr(clean.loc[ok, mask_col], clean.loc[ok, gt_col])

            results.append({
                'session': sess,
                'channel': ch,
                'band': band,
                'freq_tracking_r': r_val,
                'n_epochs': int(ok.sum()),
            })

    return pd.DataFrame(results)


def compute_snr_gated_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 4: Only evaluate epochs where mask SNR > threshold.
    Compare metrics between high-SNR and all epochs.
    """
    results = []
    for (sess, ch), grp in df.groupby(['session', 'channel']):
        clean = grp[~grp['motion_flag']]

        for band, snr_col, freq_match_col, coh_peak_col, coh_band_col in [
            ('resp', 'mask_resp_snr', 'resp_freq_match', 'coh_at_resp_peak', 'coh_band_resp'),
            ('card', 'mask_card_snr', 'card_freq_match', 'coh_at_card_peak', 'coh_band_card'),
        ]:
            # All clean epochs
            all_n = len(clean)
            all_match = clean[freq_match_col].mean() if all_n > 0 else np.nan
            all_coh_peak = clean[coh_peak_col].mean() if all_n > 0 else np.nan

            # High-SNR epochs only
            hi_snr = clean[clean[snr_col] > SNR_THRESH_DB]
            hi_n = len(hi_snr)
            hi_match = hi_snr[freq_match_col].mean() if hi_n > 0 else np.nan
            hi_coh_peak = hi_snr[coh_peak_col].mean() if hi_n > 0 else np.nan

            results.append({
                'session': sess,
                'channel': ch,
                'band': band,
                'all_n': all_n,
                'all_freq_match_rate': all_match,
                'all_coh_at_peak': all_coh_peak,
                'hi_snr_n': hi_n,
                'hi_snr_frac': hi_n / all_n if all_n > 0 else 0,
                'hi_snr_freq_match_rate': hi_match,
                'hi_snr_coh_at_peak': hi_coh_peak,
            })

    return pd.DataFrame(results)


def channel_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across all sessions: which channel combo is best?"""
    clean = df[~df['motion_flag']]
    rows = []
    for ch in CHANNEL_METHODS:
        sub = clean[clean['channel'] == ch]
        for band, freq_match_col, coh_peak_col, coh_band_col, snr_col in [
            ('resp', 'resp_freq_match', 'coh_at_resp_peak', 'coh_band_resp', 'mask_resp_snr'),
            ('card', 'card_freq_match', 'coh_at_card_peak', 'coh_band_card', 'mask_card_snr'),
        ]:
            hi_snr = sub[sub[snr_col] > SNR_THRESH_DB]
            rows.append({
                'channel': ch,
                'band': band,
                'n_epochs': len(sub),
                'freq_match_rate': sub[freq_match_col].mean(),
                'coh_at_peak_mean': sub[coh_peak_col].mean(),
                'coh_band_mean': sub[coh_band_col].mean(),
                'snr_median': sub[snr_col].median(),
                'hi_snr_n': len(hi_snr),
                'hi_snr_freq_match': hi_snr[freq_match_col].mean() if len(hi_snr) > 0 else np.nan,
                'hi_snr_coh_at_peak': hi_snr[coh_peak_col].mean() if len(hi_snr) > 0 else np.nan,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_channel_comparison(summary: pd.DataFrame, canon_df: pd.DataFrame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cc_clean = canon_df[~canon_df['motion_flag']]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, band in enumerate(['resp', 'card']):
        sub = summary[summary['band'] == band]
        channels = list(sub['channel'].values) + ['canonical']
        x = np.arange(len(channels))

        # Get canonical values
        canon_peak_col = f'canon_{band}_coh_at_peak'
        canon_band_col = f'canon_{band}_coh_band'
        canon_coh_peak = cc_clean[canon_peak_col].mean()
        canon_coh_band = cc_clean[canon_band_col].mean()

        # Freq match rate (canonical doesn't have this — use NaN)
        ax = axes[row, 0]
        freq_vals = list(sub['freq_match_rate']) + [np.nan]
        colors = ['steelblue'] * 5 + ['gold']
        bars = ax.bar(x[:-1], freq_vals[:-1], color='steelblue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(channels, fontsize=8)
        ax.set_ylabel('Freq match rate')
        ax.set_title(f'{band.title()}: Spectral peak agreement')
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, freq_vals[:-1]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', fontsize=9)
        ax.text(x[-1], 0.05, 'N/A', ha='center', fontsize=9, color='gray')

        # Coherence at peak vs band average — include canonical
        ax = axes[row, 1]
        w = 0.35
        peak_vals = list(sub['coh_at_peak_mean']) + [canon_coh_peak]
        band_vals = list(sub['coh_band_mean']) + [canon_coh_band]
        bar_colors_peak = ['teal'] * 5 + ['darkgreen']
        bar_colors_band = ['coral'] * 5 + ['orangered']
        bars1 = ax.bar(x - w/2, peak_vals, w, color=bar_colors_peak, alpha=0.7)
        bars2 = ax.bar(x + w/2, band_vals, w, color=bar_colors_band, alpha=0.7)
        # Add value labels on canonical bar
        ax.text(x[-1] - w/2, peak_vals[-1] + 0.02,
                f'{peak_vals[-1]:.2f}', ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(channels, fontsize=8)
        ax.set_ylabel('Coherence')
        ax.set_title(f'{band.title()}: Coherence (incl. canonical upper bound)')
        ax.legend([bars1[0], bars2[0], bars1[-1]],
                  ['At GT peak', 'Band average', 'Canonical'], fontsize=8)
        ax.set_ylim(0, 1)

        # High-SNR gated improvement
        ax = axes[row, 2]
        snr_all = list(sub['freq_match_rate'])
        snr_hi = list(sub['hi_snr_freq_match'])
        ax.bar(np.arange(5) - w/2, snr_all, w, label='All epochs', color='gray', alpha=0.5)
        ax.bar(np.arange(5) + w/2, snr_hi, w, label='High-SNR only', color='green', alpha=0.7)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(list(sub['channel'].values), fontsize=8)
        ax.set_ylabel('Freq match rate')
        ax.set_title(f'{band.title()}: SNR gating effect')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'channel_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: channel_comparison.png")


def plot_freq_tracking(tracking_df: pd.DataFrame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, band in zip(axes, ['resp', 'card']):
        sub = tracking_df[tracking_df['band'] == band]
        channels = CHANNEL_METHODS
        data = []
        for ch in channels:
            vals = sub[sub['channel'] == ch]['freq_tracking_r'].dropna()
            data.append(vals)
        bp = ax.boxplot(data, labels=channels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightskyblue')
            patch.set_alpha(0.7)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Freq tracking r (per session)')
        ax.set_title(f'{band.title()}: Mask-GT frequency covariation')
        ax.set_ylim(-1, 1)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'freq_tracking_by_channel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: freq_tracking_by_channel.png")


def plot_canonical_weights(canon_df: pd.DataFrame):
    """Plot distribution of optimal CLE/CRE weights from canonical coherence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    clean = canon_df[~canon_df['motion_flag']]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, (band, w1_col, w2_col, coh_col) in enumerate([
        ('Respiratory', 'canon_resp_w_cle', 'canon_resp_w_cre', 'canon_resp_coh_at_peak'),
        ('Cardiac', 'canon_card_w_cle', 'canon_card_w_cre', 'canon_card_coh_at_peak'),
    ]):
        # Weight scatter: w_CLE vs w_CRE, colored by coherence
        ax = axes[row, 0]
        ok = clean[w1_col].notna() & clean[w2_col].notna() & clean[coh_col].notna()
        sc = ax.scatter(clean.loc[ok, w1_col], clean.loc[ok, w2_col],
                        c=clean.loc[ok, coh_col], cmap='viridis',
                        s=5, alpha=0.4, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label='Canonical coherence')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        # Mark diff (1,-1) and avg (1,1) directions
        ax.plot(1, -1, 'r^', markersize=10, label='diff (1,-1)')
        ax.plot(1, 1, 'bs', markersize=10, label='avg (1,1)')
        ax.set_xlabel('w_CLE')
        ax.set_ylabel('w_CRE')
        ax.set_title(f'{band}: Optimal mixing weights')
        ax.legend(fontsize=8)

        # Weight ratio histogram
        ax = axes[row, 1]
        ratio = clean.loc[ok, w2_col] / (clean.loc[ok, w1_col].abs() + 1e-12)
        ratio_clipped = ratio.clip(-5, 5)
        ax.hist(ratio_clipped, bins=50, color='mediumpurple', alpha=0.7, edgecolor='white')
        ax.axvline(1, color='blue', linestyle='--', linewidth=1, label='avg (ratio=1)')
        ax.axvline(-1, color='red', linestyle='--', linewidth=1, label='diff (ratio=-1)')
        ax.axvline(ratio_clipped.median(), color='black', linestyle='-', linewidth=1.5,
                   label=f'median={ratio_clipped.median():.2f}')
        ax.set_xlabel('w_CRE / |w_CLE|')
        ax.set_ylabel('Count')
        ax.set_title(f'{band}: Weight ratio distribution')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / 'canonical_weights.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: canonical_weights.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all 12 sessions with sleep profiles + apnea events...")
    t0 = time.time()
    sessions = load_all_sessions(with_sleep_profiles=True, with_apnea=True)
    print(f"  Loaded {len(sessions)} sessions in {time.time() - t0:.1f}s\n")

    # --- Per-session computation ---
    frames = []
    canon_frames = []
    for s in sessions:
        print(f"Processing {s.label} (5 channel combos + canonical)...", flush=True)
        ts = time.time()
        df, canon_df = compute_session(s)
        elapsed = time.time() - ts
        n_ep = len(df) // 5
        c_clean = canon_df[~canon_df['motion_flag']]
        print(f"  {n_ep} epochs, "
              f"canon resp coh={c_clean['canon_resp_coh_at_peak'].mean():.3f}, "
              f"canon card coh={c_clean['canon_card_coh_at_peak'].mean():.3f}, "
              f"{elapsed:.1f}s")
        frames.append(df)
        canon_frames.append(canon_df)

    all_df = pd.concat(frames, ignore_index=True)
    all_canon = pd.concat(canon_frames, ignore_index=True)

    # --- Save epoch-level data ---
    all_df.to_parquet(OUT_DIR / 'signal_validation_enhanced.parquet', index=False)
    all_canon.to_parquet(OUT_DIR / 'canonical_coherence.parquet', index=False)
    print(f"\nSaved: signal_validation_enhanced.parquet ({len(all_df)} rows)")
    print(f"Saved: canonical_coherence.parquet ({len(all_canon)} rows)")

    # --- Analysis 3: Frequency tracking ---
    print("\nComputing frequency tracking correlations...")
    tracking_df = compute_freq_tracking(all_df)
    tracking_df.to_csv(OUT_DIR / 'freq_tracking.csv', index=False, float_format='%.4f')

    # --- Analysis 4: SNR-gated summary ---
    print("Computing SNR-gated summaries...")
    snr_gated = compute_snr_gated_summary(all_df)
    snr_gated.to_csv(OUT_DIR / 'snr_gated_summary.csv', index=False, float_format='%.4f')

    # --- Channel comparison ---
    print("Computing channel comparison...")
    ch_summary = channel_comparison_summary(all_df)
    ch_summary.to_csv(OUT_DIR / 'channel_comparison_summary.csv', index=False, float_format='%.4f')

    # --- Print results ---
    print(f"\n{'='*70}")
    print("CHANNEL COMPARISON (clean epochs, all sessions pooled)")
    print(f"{'='*70}")
    print(ch_summary.to_string(index=False))

    print(f"\n{'='*70}")
    print("FREQUENCY TRACKING (r: correlation of peak-freq time series)")
    print(f"{'='*70}")
    for band in ['resp', 'card']:
        print(f"\n  {band.upper()}:")
        sub = tracking_df[tracking_df['band'] == band]
        for ch in CHANNEL_METHODS:
            vals = sub[sub['channel'] == ch]['freq_tracking_r'].dropna()
            print(f"    {ch:5s}: mean r = {vals.mean():.3f}, "
                  f"median = {vals.median():.3f}, n_sessions = {len(vals)}")

    # --- Canonical coherence summary ---
    print(f"\n{'='*70}")
    print("CANONICAL COHERENCE (upper bound, 2-channel [CLE, CRE])")
    print(f"{'='*70}")
    cc_clean = all_canon[~all_canon['motion_flag']]
    for band, peak_col, band_col, w1_col, w2_col in [
        ('Resp', 'canon_resp_coh_at_peak', 'canon_resp_coh_band',
         'canon_resp_w_cle', 'canon_resp_w_cre'),
        ('Card', 'canon_card_coh_at_peak', 'canon_card_coh_band',
         'canon_card_w_cle', 'canon_card_w_cre'),
    ]:
        vals_peak = cc_clean[peak_col].dropna()
        vals_band = cc_clean[band_col].dropna()
        w1 = cc_clean[w1_col].dropna()
        w2 = cc_clean[w2_col].dropna()
        print(f"\n  {band}:")
        print(f"    Coh at GT peak : {vals_peak.mean():.3f} +/- {vals_peak.std():.3f}  (n={len(vals_peak)})")
        print(f"    Coh band mean  : {vals_band.mean():.3f} +/- {vals_band.std():.3f}")
        print(f"    Optimal w_CLE  : {w1.mean():.3f} +/- {w1.std():.3f}")
        print(f"    Optimal w_CRE  : {w2.mean():.3f} +/- {w2.std():.3f}")

    # Compare canonical vs best single-channel
    print(f"\n{'='*70}")
    print("CANONICAL vs SINGLE-CHANNEL (coherence at GT peak, clean epochs)")
    print(f"{'='*70}")
    for band, canon_col in [('resp', 'canon_resp_coh_at_peak'),
                             ('card', 'canon_card_coh_at_peak')]:
        canon_val = cc_clean[canon_col].mean()
        best_ch = ch_summary[ch_summary['band'] == band].sort_values(
            'coh_at_peak_mean', ascending=False).iloc[0]
        print(f"  {band.upper()}: canonical={canon_val:.3f}  "
              f"best_single({best_ch['channel']})={best_ch['coh_at_peak_mean']:.3f}  "
              f"gain={canon_val - best_ch['coh_at_peak_mean']:+.3f}")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_channel_comparison(ch_summary, all_canon)
    plot_freq_tracking(tracking_df)
    plot_canonical_weights(all_canon)

    print("\nDone.")


if __name__ == '__main__':
    main()
