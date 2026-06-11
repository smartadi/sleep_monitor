"""
Epoch-level CAP feature extraction for sleep stage analysis.

Two representations:
1. extract_epoch_psd  — raw Welch PSD per channel, concatenated (~450 dims).
   No hand-picked features; lets PCA/t-SNE/UMAP discover structure.
2. extract_epoch_features — ~33 hand-crafted features per epoch.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import welch, coherence
from scipy.stats import kurtosis as sp_kurtosis

from .config import (
    FS, EEG_BANDS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, STAGE_LABELS,
)
from .filters import bandpass
from .rates import rate_acf
from .sessions import SleepSession


# ── Welch PSD extraction (raw spectral representation) ───────────────────────

def extract_epoch_psd(
    session: SleepSession,
    f_max: float = 30.0,
    welch_seg_sec: float = 4.0,
    channels: Optional[List[str]] = None,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract Welch PSD per epoch per CAP channel — no hand-picked features.

    Parameters
    ----------
    session       : SleepSession with sleep_profile loaded
    f_max         : upper frequency limit (Hz); bins above this are dropped
    welch_seg_sec : Welch sub-segment length in seconds
    channels      : which CAP channels to include (default: ['CLE', 'CRE', 'CH'])
    normalize     : if True, normalize each channel's PSD to unit area before
                    log-transform — removes overall amplitude, keeps spectral shape

    Returns
    -------
    X      : (n_epochs, n_channels * n_freq_bins) — concatenated PSD matrix
    freqs  : (n_freq_bins,) — frequency axis (shared across channels)
    labels : (n_epochs,) int8 — PSG stage codes
    meta   : DataFrame with session/subject/epoch_idx/t_hr/stage_code/stage_label
    """
    if session.sleep_profile is None:
        raise ValueError(f'{session.label}: no sleep profile loaded')
    if channels is None:
        channels = ['CLE', 'CRE', 'CH']

    sp = session.sleep_profile
    n_epochs = len(sp['codes'])
    fs = session.fs
    epoch_n = int(PSG_EPOCH_SEC * fs)
    nperseg = min(epoch_n, int(welch_seg_sec * fs))
    t_hr = session.time_hr

    cap = session.cap
    sigs = {}
    for ch in channels:
        if ch == 'CLE-CRE':
            sigs[ch] = cap['CLE'].astype(np.float64) - cap['CRE'].astype(np.float64)
        else:
            sigs[ch] = cap[ch].astype(np.float64)

    psd_rows: List[np.ndarray] = []
    meta_rows: List[dict] = []
    freqs_out = None

    for ei in range(n_epochs):
        t_start = sp['t_ep_hr'][ei]
        t_end = t_start + PSG_EPOCH_SEC / 3600.0
        mask = (t_hr >= t_start) & (t_hr < t_end)
        if mask.sum() < epoch_n * 0.5:
            continue

        idx = np.where(mask)[0]
        ch_psds = []
        for ch in channels:
            seg = sigs[ch][idx]
            seg = seg - seg.mean()
            f, pxx = welch(seg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                           scaling='density')
            f_mask = f <= f_max
            pxx_cut = pxx[f_mask]
            if normalize:
                pxx_cut = pxx_cut / (pxx_cut.sum() + 1e-20)
            pxx_cut = np.log10(pxx_cut + 1e-20)
            ch_psds.append(pxx_cut)
            if freqs_out is None:
                freqs_out = f[f_mask]

        psd_rows.append(np.concatenate(ch_psds))
        meta_rows.append({
            'session': session.label,
            'subject': session.subject,
            'epoch_idx': ei,
            't_hr': float(t_start),
            'stage_code': int(sp['codes'][ei]),
            'stage_label': sp['labels'][ei],
        })

    X = np.array(psd_rows, dtype=np.float64)
    meta = pd.DataFrame(meta_rows)
    labels = meta['stage_code'].values.astype(np.int8)
    return X, freqs_out, labels, meta


def _welch_band_powers(
    sig: np.ndarray,
    fs: float,
    bands: Dict[str, Tuple[float, float]],
    total_range: Tuple[float, float] = (0.5, 30.0),
    nperseg: int = 400,
) -> Dict[str, float]:
    if len(sig) < nperseg:
        out = {name: np.nan for name in bands}
        out['total_power'] = np.nan
        return out

    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                       scaling='density')
    df = freqs[1] - freqs[0]

    t_mask = (freqs >= total_range[0]) & (freqs <= total_range[1])
    tp = float(np.trapz(psd[t_mask], dx=df))

    out = {}
    for name, (flo, fhi) in bands.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        bp = float(np.trapz(psd[mask], dx=df))
        out[name] = bp / tp if tp > 0 else 0.0
    out['total_power'] = tp
    return out


def _spectral_entropy(sig: np.ndarray, fs: float, nperseg: int = 400) -> float:
    if len(sig) < nperseg:
        return np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    psd_norm = psd / (psd.sum() + 1e-20)
    psd_norm = psd_norm[psd_norm > 0]
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def _zero_crossing_rate(sig: np.ndarray, fs: float) -> float:
    if len(sig) < 2:
        return np.nan
    zc = np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0)
    return float(zc) / (len(sig) / fs)


def _coherence_band(
    x: np.ndarray, y: np.ndarray,
    fs: float, f_lo: float, f_hi: float,
    nperseg: int = 400,
) -> float:
    if len(x) < nperseg or len(y) < nperseg:
        return np.nan
    freqs, coh = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.nan
    return float(np.mean(coh[mask]))


def extract_epoch_features(
    session: SleepSession,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Extract per-epoch features from CAP channels for sleep stage analysis.

    Parameters
    ----------
    session : SleepSession with sleep_profile loaded
    bands   : frequency bands for power ratios (default: EEG_BANDS)

    Returns
    -------
    DataFrame with one row per epoch, columns = features + metadata.
    """
    if session.sleep_profile is None:
        raise ValueError(f'{session.label}: no sleep profile loaded')
    if bands is None:
        bands = EEG_BANDS

    sp = session.sleep_profile
    n_epochs = len(sp['codes'])
    fs = session.fs
    epoch_n = int(PSG_EPOCH_SEC * fs)
    nperseg = min(epoch_n, int(4.0 * fs))

    cap = session.cap
    cle = cap['CLE'].astype(np.float64)
    cre = cap['CRE'].astype(np.float64)
    ch = cap['CH'].astype(np.float64)
    diff = cle - cre
    acc = cap['acc_mag'].astype(np.float64)
    t_hr = session.time_hr

    channels = {'CLE': cle, 'CRE': cre, 'CH': ch, 'CLE-CRE': diff}

    rows: List[dict] = []

    for ei in range(n_epochs):
        t_ep_start = sp['t_ep_hr'][ei]
        t_ep_end = t_ep_start + PSG_EPOCH_SEC / 3600.0

        mask = (t_hr >= t_ep_start) & (t_hr < t_ep_end)
        if mask.sum() < epoch_n * 0.5:
            continue

        idx = np.where(mask)[0]
        row: dict = {
            'session': session.label,
            'subject': session.subject,
            'epoch_idx': ei,
            't_hr': float(t_ep_start),
            'stage_code': int(sp['codes'][ei]),
            'stage_label': sp['labels'][ei],
        }

        # Per-channel band power ratios + RMS
        for ch_name, ch_sig in channels.items():
            seg = ch_sig[idx]
            bp = _welch_band_powers(seg, fs, bands, nperseg=nperseg)
            prefix = ch_name.replace('-', '_')
            for bname, val in bp.items():
                row[f'{prefix}_{bname}'] = val
            row[f'{prefix}_rms'] = float(np.sqrt(np.mean(seg ** 2)))

        # CLE-CRE differential stats
        seg_diff = diff[idx]
        row['diff_spectral_entropy'] = _spectral_entropy(seg_diff, fs, nperseg)
        row['diff_zcr'] = _zero_crossing_rate(seg_diff, fs)
        row['diff_kurtosis'] = float(sp_kurtosis(seg_diff, fisher=True))

        # CLE-CRE coherence in resp and cardiac bands
        seg_cle = cle[idx]
        seg_cre = cre[idx]
        row['coh_resp'] = _coherence_band(seg_cle, seg_cre, fs,
                                          RESP_LO, RESP_HI, nperseg)
        row['coh_card'] = _coherence_band(seg_cle, seg_cre, fs,
                                          CARD_LO, CARD_HI, nperseg)

        # Rate features from CLE-CRE via ACF
        bp_resp = bandpass(seg_diff, RESP_LO, RESP_HI, fs)
        bp_card = bandpass(seg_diff, CARD_LO, CARD_HI, fs)
        row['resp_rate_hz'] = rate_acf(bp_resp, RESP_LO, RESP_HI, fs)
        row['card_rate_hz'] = rate_acf(bp_card, CARD_LO, CARD_HI, fs)

        # Motion features
        seg_acc = acc[idx]
        row['acc_rms'] = float(np.sqrt(np.mean((seg_acc - np.mean(seg_acc)) ** 2)))
        acc_bp_resp = bandpass(seg_acc, RESP_LO, RESP_HI, fs)
        row['acc_resp_power'] = float(np.mean(acc_bp_resp ** 2))

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of numeric feature columns (exclude metadata)."""
    meta = {'session', 'subject', 'epoch_idx', 't_hr', 'stage_code', 'stage_label'}
    return [c for c in df.columns if c not in meta and df[c].dtype in (np.float64, np.float32)]
