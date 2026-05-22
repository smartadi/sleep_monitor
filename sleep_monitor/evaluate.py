"""
Evaluation harness: PipelineConfig + run_pipeline + evaluate_pipeline.

A pipeline is:
    (channel, preproc, estimator, fusion) applied to one band (resp or cardiac).

run_pipeline produces one row per sliding window:
    t_s, rate_hz, quality, gt_rate_hz, rate_<method>_hz for each method,
    + feature columns used later by the rate classifier.

evaluate_pipeline collapses those rows into a metrics dict (MAE, RMSE, r,
bias, coverage) at a chosen quality gate.

Design note
-----------
The same per-window DataFrame that drives evaluation will be the feature
matrix for the classifier phase. Keep schema stable: one row per window,
columns are features + ground-truth label.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from .config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI
from .filters import bandpass
from .preprocessing import remove_acc_artifact, remove_acc_artifact_nlms
from .rates import estimate_rate, fuse_rates, rate_acf
from .quality import window_features, combined_quality
from .sessions import SleepSession


# ── Configuration ─────────────────────────────────────────────────────────────

_CHANNEL_CHOICES = ('CH', 'CLE', 'CRE', 'CLE-CRE', 'fused')
_PREPROC_CHOICES = ('none', 'ols', 'nlms')
_ESTIMATOR_CHOICES = (
    'spectral', 'acf', 'hilbert', 'zerocross', 'peaks', 'envelope',
    'median', 'trimmed', 'weighted',   # fusion variants
)
_BAND_CHOICES = ('resp', 'cardiac')


@dataclass(frozen=True)
class BaseKey:
    """Identifies the signal-processing part of a pipeline — shared across estimators."""
    band:    str
    channel: str = 'CLE-CRE'
    preproc: str = 'ols'
    win_s:   float = 30.0
    step_s:  float = 5.0

    @property
    def band_edges(self) -> Tuple[float, float]:
        return (RESP_LO, RESP_HI) if self.band == 'resp' else (CARD_LO, CARD_HI)

    def tag(self) -> str:
        return f'{self.band}_{self.channel}_{self.preproc}_w{int(self.win_s)}'


@dataclass(frozen=True)
class PipelineConfig:
    """One configuration to evaluate.

    Attributes
    ----------
    band       : 'resp' or 'cardiac'
    channel    : 'CH', 'CLE', 'CRE', 'CLE-CRE', or 'fused' (quality-weighted avg)
    preproc    : 'none' | 'ols' | 'nlms'  (accelerometer removal strategy)
    estimator  : one estimator name, or a fusion rule ('median'|'trimmed'|'weighted')
    win_s      : sliding-window length (seconds)
    step_s     : sliding-window step (seconds)
    """
    band:      str
    channel:   str = 'CLE-CRE'
    preproc:   str = 'ols'
    estimator: str = 'acf'
    win_s:     float = 30.0
    step_s:    float = 5.0

    def __post_init__(self) -> None:
        if self.band not in _BAND_CHOICES:
            raise ValueError(f'band must be in {_BAND_CHOICES}')
        if self.channel not in _CHANNEL_CHOICES:
            raise ValueError(f'channel must be in {_CHANNEL_CHOICES}')
        if self.preproc not in _PREPROC_CHOICES:
            raise ValueError(f'preproc must be in {_PREPROC_CHOICES}')
        if self.estimator not in _ESTIMATOR_CHOICES:
            raise ValueError(f'estimator must be in {_ESTIMATOR_CHOICES}')

    @property
    def band_edges(self) -> Tuple[float, float]:
        return (RESP_LO, RESP_HI) if self.band == 'resp' else (CARD_LO, CARD_HI)

    def tag(self) -> str:
        return (f"{self.band}_{self.channel}_{self.preproc}_{self.estimator}"
                f"_w{int(self.win_s)}")


# ── Channel preparation ───────────────────────────────────────────────────────

def _single_channel_bandpass(sig_raw: np.ndarray, acc_mag: np.ndarray,
                              f_lo: float, f_hi: float,
                              preproc: str, fs: float) -> np.ndarray:
    """Return the bandpassed channel after the chosen preprocessing."""
    if preproc == 'ols':
        return remove_acc_artifact(sig_raw, acc_mag, f_lo, f_hi, fs)
    if preproc == 'nlms':
        return remove_acc_artifact_nlms(sig_raw, acc_mag, f_lo, f_hi, fs)
    return bandpass(sig_raw, f_lo, f_hi, fs)


def _prepare_channel(session: SleepSession, cfg: PipelineConfig) -> np.ndarray:
    """
    Return a single 1-D band-filtered signal for the configured channel.

    For 'fused', we bandpass each of CH/CLE/CRE/CLE-CRE and later combine
    them per window by quality-weighted mean.
    """
    f_lo, f_hi = cfg.band_edges
    cap  = session.cap
    acc  = cap['acc_mag'].astype(np.float64)
    fs   = session.fs

    def raw(ch: str) -> np.ndarray:
        if ch == 'CLE-CRE':
            return cap['CLE'].astype(np.float64) - cap['CRE'].astype(np.float64)
        return cap[ch].astype(np.float64)

    if cfg.channel != 'fused':
        return _single_channel_bandpass(raw(cfg.channel), acc,
                                          f_lo, f_hi, cfg.preproc, fs)

    # fused: stack channels as rows; fuse per window inside the loop
    # CH is dropped — redundant with CLE/CRE and noisier
    channels = ('CLE', 'CRE', 'CLE-CRE')
    stacked = np.stack([
        _single_channel_bandpass(raw(ch), acc, f_lo, f_hi, cfg.preproc, fs)
        for ch in channels
    ], axis=0)
    return stacked    # shape (3, N)


# ── Ground-truth reference ────────────────────────────────────────────────────

def _gt_rate_series(session: SleepSession, cfg: PipelineConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window GT rate derived by ACF on the PSG reference channel.

    Uses Thorax for resp, Pleth for cardiac.
    """
    f_lo, f_hi = cfg.band_edges
    ref = session.psg['Thorax' if cfg.band == 'resp' else 'Pleth'].astype(np.float64)
    ref_bp = bandpass(ref, f_lo, f_hi, session.fs)

    win_n  = int(round(cfg.win_s * session.fs))
    step_n = max(1, int(round(cfg.step_s * session.fs)))
    t_list, r_list = [], []
    for start in range(0, len(ref_bp) - win_n + 1, step_n):
        seg = ref_bp[start:start + win_n]
        r = rate_acf(seg, f_lo, f_hi, session.fs)
        t_list.append((start + win_n / 2.0) / session.fs)
        r_list.append(r)
    return np.array(t_list), np.array(r_list)


# ── Main pipeline runner ──────────────────────────────────────────────────────

def run_pipeline(session: SleepSession, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Execute one PipelineConfig on one session; return per-window DataFrame.

    Columns
    -------
    t_s               window centre time (seconds from recording start)
    rate_hz           chosen pipeline rate (band-clamped; NaN if invalid)
    quality           combined quality score in [0, 1]
    gt_rate_hz        PSG reference rate for the same window
    rate_spectral/... per-method rates (NaN for missing)
    snr_db, acf_prom, spec_conc, motion_db, rms, n, agreement_hz
                      quality features (for future classifier)
    session           session label
    band              'resp' or 'cardiac'
    """
    f_lo, f_hi  = cfg.band_edges
    fs           = session.fs
    sig          = _prepare_channel(session, cfg)
    acc_mag      = session.cap['acc_mag'].astype(np.float64)
    win_n        = int(round(cfg.win_s * fs))
    step_n       = max(1, int(round(cfg.step_s * fs)))

    is_fused     = (sig.ndim == 2)
    total_len    = sig.shape[-1]
    use_envelope = cfg.band == 'cardiac'

    # Ground truth on the same grid
    gt_t, gt_hz = _gt_rate_series(session, cfg)
    valid_gt = ~np.isnan(gt_hz)
    if valid_gt.sum() >= 2:
        gt_interp = interp1d(gt_t[valid_gt], gt_hz[valid_gt],
                              kind='linear', bounds_error=False, fill_value=np.nan)
    else:
        gt_interp = lambda _t: np.nan   # noqa: E731

    rows: List[dict] = []
    for start in range(0, total_len - win_n + 1, step_n):
        t_c = (start + win_n / 2.0) / fs
        acc_win = acc_mag[start:start + win_n]

        if is_fused:
            # Estimate per-channel, then fuse by quality-weighted average.
            per_ch_rates: List[float] = []
            per_ch_quality: List[float] = []
            per_method_rates: Dict[str, List[float]] = {}
            per_ch_signals: List[np.ndarray] = []
            for ch_idx in range(sig.shape[0]):
                seg = sig[ch_idx, start:start + win_n]
                per_ch_signals.append(seg)
                rates = estimate_rate(seg, f_lo, f_hi, fs, include_envelope=use_envelope)
                # estimator selection within each channel
                if cfg.estimator in ('median', 'trimmed', 'weighted'):
                    r = fuse_rates(rates, f_lo, f_hi, how=cfg.estimator)
                else:
                    r = rates.get(cfg.estimator, np.nan)
                per_ch_rates.append(r)
                feat = window_features(seg, acc_win, f_lo, f_hi, fs, rates_hz=rates)
                per_ch_quality.append(combined_quality(feat))
                for m, v in rates.items():
                    per_method_rates.setdefault(m, []).append(v)
            qv = np.array(per_ch_quality)
            rv = np.array(per_ch_rates)
            ok = np.isfinite(rv) & (qv > 0)
            if ok.any():
                w = qv[ok] + 1e-6
                rate_val = float(np.sum(w * rv[ok]) / np.sum(w))
            else:
                rate_val = np.nan
            # For reporting: use the best-quality channel's features
            best_ch = int(np.argmax(qv)) if np.isfinite(qv).any() else 0
            feat = window_features(per_ch_signals[best_ch], acc_win,
                                     f_lo, f_hi, fs,
                                     rates_hz={m: vs[best_ch]
                                               for m, vs in per_method_rates.items()})
            q_val = combined_quality(feat)
            method_rates = {m: float(np.nanmedian(vs))
                             for m, vs in per_method_rates.items()}
        else:
            seg = sig[start:start + win_n]
            rates = estimate_rate(seg, f_lo, f_hi, fs, include_envelope=use_envelope)
            if cfg.estimator in ('median', 'trimmed', 'weighted'):
                rate_val = fuse_rates(rates, f_lo, f_hi, how=cfg.estimator)
            else:
                rate_val = rates.get(cfg.estimator, np.nan)
            feat = window_features(seg, acc_win, f_lo, f_hi, fs, rates_hz=rates)
            q_val = combined_quality(feat)
            method_rates = rates

        # Clamp to band — anything outside physiology is spurious
        if np.isfinite(rate_val) and not (f_lo <= rate_val <= f_hi):
            rate_val = np.nan

        row = {
            'session':    session.label,
            'band':       cfg.band,
            't_s':        t_c,
            'rate_hz':    rate_val,
            'quality':    q_val,
            'gt_rate_hz': float(gt_interp(t_c)),
            **{f'rate_{m}_hz': method_rates.get(m, np.nan)
                for m in ('spectral','acf','hilbert','zerocross','peaks','envelope')},
            **feat,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ── Metrics on pipeline output ────────────────────────────────────────────────

def evaluate_pipeline(df: pd.DataFrame,
                       quality_gate: float = 0.0,
                       scale: float = 60.0) -> dict:
    """
    Compute summary metrics from a run_pipeline() DataFrame.

    Parameters
    ----------
    quality_gate : drop windows with quality < this threshold
    scale        : 60.0 converts Hz -> br/min or BPM

    Returns
    -------
    dict with keys:
        n_total, n_used, coverage,
        mae, rmse, r, bias, p50_abs_err, p90_abs_err
    """
    n_total = len(df)
    if n_total == 0:
        return dict(n_total=0, n_used=0, coverage=0.0,
                    mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan,
                    p50_abs_err=np.nan, p90_abs_err=np.nan)

    ok = (
        df['quality'].fillna(0).to_numpy() >= quality_gate
    ) & np.isfinite(df['rate_hz'].to_numpy()) & np.isfinite(df['gt_rate_hz'].to_numpy())
    n_used   = int(ok.sum())
    coverage = float(n_used / n_total) if n_total else 0.0
    if n_used < 5:
        return dict(n_total=n_total, n_used=n_used, coverage=coverage,
                    mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan,
                    p50_abs_err=np.nan, p90_abs_err=np.nan)

    pred = df.loc[ok, 'rate_hz'].to_numpy() * scale
    ref  = df.loc[ok, 'gt_rate_hz'].to_numpy() * scale
    err  = pred - ref
    r_val = float(pearsonr(pred, ref)[0]) if n_used >= 3 else np.nan
    return dict(
        n_total     = n_total,
        n_used      = n_used,
        coverage    = coverage,
        mae         = float(np.mean(np.abs(err))),
        rmse        = float(np.sqrt(np.mean(err**2))),
        r           = r_val,
        bias        = float(np.mean(err)),
        p50_abs_err = float(np.median(np.abs(err))),
        p90_abs_err = float(np.quantile(np.abs(err), 0.90)),
    )


# ── Fast-path: compute base windows once, derive all estimators from it ───────

def compute_base_windows(session: SleepSession, key: BaseKey) -> pd.DataFrame:
    """
    Compute the per-window table for one (channel, preproc) — all estimators
    share this. The rate_hz column is intentionally omitted here; derive it
    with `derive_rate(base_df, estimator)`.

    This is the expensive step: everything signal-processing-related happens
    once per (channel, preproc), not once per estimator.
    """
    cfg = PipelineConfig(band=key.band, channel=key.channel, preproc=key.preproc,
                          estimator='acf', win_s=key.win_s, step_s=key.step_s)
    df = run_pipeline(session, cfg)
    return df.drop(columns=['rate_hz'])


def derive_rate(base_df: pd.DataFrame, estimator: str,
                 band: str) -> pd.Series:
    """
    Derive a `rate_hz` series from a base windows DataFrame for one estimator.

    Supported
    ---------
    direct methods : 'spectral', 'acf', 'hilbert', 'zerocross', 'peaks', 'envelope'
    fusion rules   : 'median', 'trimmed', 'weighted'
    """
    f_lo, f_hi = (RESP_LO, RESP_HI) if band == 'resp' else (CARD_LO, CARD_HI)
    method_cols = [c for c in
                    ['rate_spectral_hz', 'rate_acf_hz', 'rate_hilbert_hz',
                      'rate_zerocross_hz', 'rate_peaks_hz', 'rate_envelope_hz']
                    if c in base_df.columns]

    if estimator in ('spectral', 'acf', 'hilbert', 'zerocross', 'peaks', 'envelope'):
        col = f'rate_{estimator}_hz'
        if col not in base_df.columns:
            return pd.Series(np.nan, index=base_df.index)
        vals = base_df[col].to_numpy(dtype=float).copy()
    else:
        # fusion across methods per row
        mat = base_df[method_cols].to_numpy(dtype=float)
        mask = np.isfinite(mat) & (mat >= f_lo) & (mat <= f_hi)
        vals = np.full(mat.shape[0], np.nan)
        for i in range(mat.shape[0]):
            row = mat[i, mask[i]]
            if len(row) == 0:
                continue
            if len(row) == 1 or estimator == 'median':
                vals[i] = np.median(row)
            elif estimator == 'trimmed' and len(row) >= 3:
                trimmed = np.sort(row)[1:-1]
                vals[i] = np.median(trimmed)
            elif estimator == 'weighted':
                med = np.median(row)
                d = np.abs(row - med) + 1e-6
                w = 1.0 / d
                vals[i] = float(np.sum(w * row) / np.sum(w))
            else:
                vals[i] = np.median(row)

    # Clamp to band
    out_of_band = np.isfinite(vals) & ((vals < f_lo) | (vals > f_hi))
    vals[out_of_band] = np.nan
    return pd.Series(vals, index=base_df.index, name='rate_hz')


# ── Grid generation helper ────────────────────────────────────────────────────

def default_grid(band: str) -> List[PipelineConfig]:
    """
    Enumerate the default search grid for one band.

    channels × preproc × estimators  (win_s fixed to 30s, step_s 5s).
    CH is excluded — noisier and redundant with CLE/CRE fusion.
    """
    channels   = ('CLE', 'CRE', 'CLE-CRE', 'fused')
    preprocs   = ('none', 'ols', 'nlms')
    estimators = ['spectral', 'acf', 'hilbert', 'zerocross', 'peaks',
                   'median', 'trimmed', 'weighted']
    if band == 'cardiac':
        estimators = estimators + ['envelope']
    return [PipelineConfig(band=band, channel=c, preproc=p, estimator=e)
             for c in channels for p in preprocs for e in estimators]


def default_base_keys(band: str) -> List[BaseKey]:
    """Enumerate unique (channel, preproc) pairs whose signal processing is shared."""
    return [BaseKey(band=band, channel=c, preproc=p)
             for c in ('CLE', 'CRE', 'CLE-CRE', 'fused')
             for p in ('none', 'ols', 'nlms')]


# ── Multi-session convenience ─────────────────────────────────────────────────

def evaluate_on_sessions(sessions: List[SleepSession],
                          cfg: PipelineConfig,
                          quality_gate: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run `cfg` on multiple sessions and aggregate metrics.

    Returns
    -------
    (metrics_df, windows_df)
        metrics_df : one row per session with evaluate_pipeline() metrics + config tag
        windows_df : vertically concatenated run_pipeline() outputs with session label
    """
    metrics_rows, win_frames = [], []
    for s in sessions:
        w = run_pipeline(s, cfg)
        m = evaluate_pipeline(w, quality_gate=quality_gate)
        m['session'] = s.label
        m['tag']     = cfg.tag()
        m.update(asdict(cfg))
        metrics_rows.append(m)
        win_frames.append(w)
    return pd.DataFrame(metrics_rows), pd.concat(win_frames, ignore_index=True)
