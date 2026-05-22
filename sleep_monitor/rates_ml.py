"""
ML-driven rate estimators.

Models
------
RateFusionModel    -- Learned fusion: takes classical rate estimates + quality
                     features as input, predicts a single fused rate.
                     Lightweight (XGBoost or MLP), works with 12 sessions.

DenoisingAutoencoder -- Self-supervised convolutional autoencoder that learns
                       to clean noisy CAP signals. No GT labels needed for
                       training. Classical rate extraction on the cleaned
                       output.

SpectrogramCNN     -- 2D CNN on STFT spectrograms -> rate regression.
                     Optionally uses ImageNet-pretrained backbone.

Training uses leave-one-subject-out (LOSO) cross-validation to avoid
leaking overnight structure. All models are torch-free by default
(numpy + sklearn/xgboost), with optional PyTorch for the CNN and
autoencoder when available.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

from .config import FS, METHOD_NAMES, RESP_LO, RESP_HI, CARD_LO, CARD_HI


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Learned Fusion  (XGBoost / sklearn -- no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════════

FUSION_FEATURE_COLS = [
    # base rate estimates (Hz)
    'spectral', 'acf', 'hilbert', 'zerocross', 'peaks',
    # quality features
    'snr_db', 'acf_prom', 'spec_conc', 'motion_db', 'agreement_hz',
    # inter-method stats
    'rate_median', 'rate_std', 'rate_range',
]


def build_fusion_features(rates_hz: Dict[str, float],
                          quality: Dict[str, float]) -> np.ndarray:
    """
    Build a feature vector for the fusion model from one window's
    rate estimates and quality features.

    Returns 1-D array of length len(FUSION_FEATURE_COLS).
    """
    vals = np.array([v for v in rates_hz.values()
                     if v is not None and np.isfinite(v)])
    rate_median = float(np.median(vals)) if len(vals) else np.nan
    rate_std = float(np.std(vals)) if len(vals) >= 2 else np.nan
    rate_range = float(np.ptp(vals)) if len(vals) >= 2 else np.nan

    row = {
        **{m: rates_hz.get(m, np.nan) for m in METHOD_NAMES},
        'snr_db':       quality.get('snr_db', np.nan),
        'acf_prom':     quality.get('acf_prom', np.nan),
        'spec_conc':    quality.get('spec_conc', np.nan),
        'motion_db':    quality.get('motion_db', np.nan),
        'agreement_hz': quality.get('agreement_hz', np.nan),
        'rate_median':  rate_median,
        'rate_std':     rate_std,
        'rate_range':   rate_range,
    }
    return np.array([row[c] for c in FUSION_FEATURE_COLS], dtype=np.float64)


@dataclass
class FusionDataset:
    """Holds features + targets for fusion model training."""
    X: np.ndarray          # (N, n_features)
    y: np.ndarray          # (N,) target rate in Hz
    subjects: np.ndarray   # (N,) subject IDs for LOSO splits
    sessions: np.ndarray   # (N,) session labels
    bands: np.ndarray      # (N,) 'resp' or 'card'


def build_fusion_dataset(
    window_dir: Path,
    metrics_path: Path,
    band: str = 'resp',
    channel: str = 'CLE-CRE',
) -> FusionDataset:
    """
    Build fusion dataset from pre-computed window parquet files.

    Expects artifacts from compute_rates.py:
      - window_dir/*.parquet  with columns: t_hr, gt_hz, session, band, channel,
                                            spectral, acf, hilbert, zerocross, peaks
      - metrics_path          for subject mapping

    Parameters
    ----------
    band    : 'resp' or 'card'
    channel : CAP channel to use
    """
    import pandas as pd

    metrics = pd.read_parquet(metrics_path)
    session_subject = dict(zip(metrics['session'], metrics['subject']))

    frames = []
    for pq in sorted(window_dir.glob('*.parquet')):
        df = pd.read_parquet(pq)
        df = df[(df['band'] == band) & (df['channel'] == channel)]
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise ValueError(f'No window data found for band={band}, channel={channel}')

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=['gt_hz'])

    rate_cols = [c for c in METHOD_NAMES if c in data.columns]
    rate_vals = data[rate_cols].values

    # Compute inter-method stats
    rate_median = np.nanmedian(rate_vals, axis=1)
    rate_std = np.nanstd(rate_vals, axis=1)
    rate_range = np.nanmax(rate_vals, axis=1) - np.nanmin(rate_vals, axis=1)

    # Quality features are not in window files -- fill with NaN for now,
    # to be augmented by a dedicated quality-extraction step.
    n = len(data)
    X = np.column_stack([
        rate_vals,
        np.full(n, np.nan),  # snr_db
        np.full(n, np.nan),  # acf_prom
        np.full(n, np.nan),  # spec_conc
        np.full(n, np.nan),  # motion_db
        np.full(n, np.nan),  # agreement_hz
        rate_median,
        rate_std,
        rate_range,
    ])

    subjects = np.array([session_subject.get(s, s) for s in data['session']])

    return FusionDataset(
        X=X,
        y=data['gt_hz'].values,
        subjects=subjects,
        sessions=data['session'].values,
        bands=np.full(n, band),
    )


def train_fusion_model(dataset: FusionDataset, model_type: str = 'xgb'):
    """
    Train a fusion model with LOSO cross-validation.

    Parameters
    ----------
    model_type : 'xgb' (XGBoost), 'rf' (RandomForest), or 'mlp' (sklearn MLP)

    Returns
    -------
    (model, cv_results)
        model      : fitted on full dataset
        cv_results : list of dicts with 'subject', 'mae', 'rmse', 'n'
    """
    X, y = dataset.X.copy(), dataset.y.copy()
    subjects = dataset.subjects

    # Replace NaN with column median for tree models
    from numpy import nanmedian
    col_medians = np.array([nanmedian(X[:, j]) for j in range(X.shape[1])])
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    mask = ~np.isfinite(X)
    for j in range(X.shape[1]):
        X[mask[:, j], j] = col_medians[j]

    def _make_model():
        if model_type == 'xgb':
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, verbosity=0,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=42,
                )
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=42,
            )
        elif model_type == 'mlp':
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500,
                early_stopping=True, random_state=42,
            )
        else:
            raise ValueError(f'Unknown model_type: {model_type}')

    # LOSO CV
    unique_subjects = np.unique(subjects)
    cv_results = []
    for subj in unique_subjects:
        test_mask = subjects == subj
        train_mask = ~test_mask
        if train_mask.sum() < 10 or test_mask.sum() < 5:
            continue
        model = _make_model()
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        errors = np.abs(pred - y[test_mask])
        cv_results.append({
            'subject': subj,
            'mae': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'n': int(test_mask.sum()),
        })

    # Fit on all data
    full_model = _make_model()
    full_model.fit(X, y)

    return full_model, cv_results


def predict_fused_rate(model, rates_hz: Dict[str, float],
                       quality: Dict[str, float]) -> float:
    """Predict a single fused rate from a trained fusion model."""
    feat = build_fusion_features(rates_hz, quality).reshape(1, -1)
    mask = ~np.isfinite(feat)
    if mask.any():
        feat[mask] = 0.0
    return float(model.predict(feat)[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Self-Supervised Denoising Autoencoder  (numpy for inference, torch for training)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AutoencoderConfig:
    """Configuration for the denoising autoencoder."""
    win_samples: int = 2000       # 20s at 100 Hz
    latent_dim: int = 64
    n_filters: List[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_size: int = 15
    noise_std: float = 0.3
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 32


def _extract_windows(signal: np.ndarray, win_n: int,
                     step_n: int) -> np.ndarray:
    """Slice signal into overlapping windows. Returns (N, win_n)."""
    starts = np.arange(0, len(signal) - win_n + 1, step_n)
    windows = np.zeros((len(starts), win_n))
    for i, s in enumerate(starts):
        w = signal[s:s + win_n].astype(np.float64)
        std = np.std(w) + 1e-12
        windows[i] = (w - np.mean(w)) / std
    return windows


def prepare_autoencoder_data(
    sessions,
    band: str = 'resp',
    channel: str = 'CLE-CRE',
    win_sec: float = 20.0,
    step_sec: float = 5.0,
    acc_removal: bool = True,
) -> np.ndarray:
    """
    Extract normalised windows from all sessions for autoencoder training.

    No GT labels needed -- this is self-supervised.

    Parameters
    ----------
    sessions : list of SleepSession objects
    band     : 'resp' or 'card'
    channel  : which CAP channel to use

    Returns
    -------
    windows : (N, win_samples) array of normalised signal windows
    """
    from .preprocessing import preprocess_full

    f_lo = RESP_LO if band == 'resp' else CARD_LO
    f_hi = RESP_HI if band == 'resp' else CARD_HI
    fs = FS

    win_n = int(round(win_sec * fs))
    step_n = int(round(step_sec * fs))

    all_windows = []
    for session in sessions:
        full, _ = preprocess_full(session, acc_removal)
        sig = full[channel][band]
        windows = _extract_windows(sig, win_n, step_n)
        all_windows.append(windows)

    return np.concatenate(all_windows, axis=0)


def train_autoencoder(windows: np.ndarray,
                      cfg: AutoencoderConfig | None = None):
    """
    Train a 1D convolutional denoising autoencoder.

    The model learns to reconstruct clean windows from noise-corrupted
    versions. At inference, passing a noisy CAP window through the
    trained encoder-decoder produces a denoised version suitable for
    classical rate extraction.

    Requires PyTorch. Returns (model, losses).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            'PyTorch required for autoencoder training. '
            'Install with: pip install torch'
        )

    if cfg is None:
        cfg = AutoencoderConfig()

    class Conv1dAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            enc_layers = []
            in_ch = 1
            for nf in cfg.n_filters:
                enc_layers.extend([
                    nn.Conv1d(in_ch, nf, cfg.kernel_size, padding=cfg.kernel_size // 2),
                    nn.BatchNorm1d(nf),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ])
                in_ch = nf
            self.encoder = nn.Sequential(*enc_layers)

            # Decoder
            dec_layers = []
            for i, nf in enumerate(reversed(cfg.n_filters)):
                out_ch = cfg.n_filters[-(i + 2)] if i < len(cfg.n_filters) - 1 else 1
                dec_layers.extend([
                    nn.ConvTranspose1d(nf, out_ch, cfg.kernel_size,
                                       padding=cfg.kernel_size // 2),
                    nn.Upsample(scale_factor=2),
                ])
                if i < len(cfg.n_filters) - 1:
                    dec_layers.append(nn.ReLU())
            self.decoder = nn.Sequential(*dec_layers)

        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            # Trim/pad to match input length
            if out.shape[-1] > x.shape[-1]:
                out = out[:, :, :x.shape[-1]]
            elif out.shape[-1] < x.shape[-1]:
                pad = x.shape[-1] - out.shape[-1]
                out = nn.functional.pad(out, (0, pad))
            return out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv1dAutoencoder().to(device)

    # Prepare data
    X = torch.FloatTensor(windows).unsqueeze(1)  # (N, 1, win_samples)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            # Add noise
            noise = torch.randn_like(batch) * cfg.noise_std
            noisy = batch + noise

            recon = model(noisy)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))

    return model, losses


def denoise_signal(model, signal: np.ndarray,
                   win_sec: float = 20.0,
                   step_sec: float = 5.0,
                   fs: float = FS) -> np.ndarray:
    """
    Apply trained autoencoder to denoise a full signal.

    Uses overlapping windows with Hann blending for smooth output.
    Returns denoised signal of same length as input.
    """
    try:
        import torch
    except ImportError:
        raise ImportError('PyTorch required for autoencoder inference.')

    signal = signal.astype(np.float64)
    win_n = int(round(win_sec * fs))
    step_n = int(round(step_sec * fs))
    N = len(signal)

    if N < win_n:
        return signal.copy()

    device = next(model.parameters()).device
    model.eval()

    output = np.zeros(N)
    weights = np.zeros(N)
    hann = np.hanning(win_n)

    with torch.no_grad():
        for start in range(0, N - win_n + 1, step_n):
            seg = signal[start:start + win_n]
            mu, std = seg.mean(), seg.std() + 1e-12
            seg_norm = (seg - mu) / std

            x = torch.FloatTensor(seg_norm).unsqueeze(0).unsqueeze(0).to(device)
            recon = model(x).cpu().numpy().squeeze()

            # Un-normalise
            recon = recon[:win_n] * std + mu

            output[start:start + win_n] += recon * hann
            weights[start:start + win_n] += hann

    # Fill any uncovered edges
    mask = weights > 0
    output[mask] /= weights[mask]
    output[~mask] = signal[~mask]

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Spectrogram CNN  (PyTorch, optional ImageNet transfer)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectrogramCNNConfig:
    """Configuration for the spectrogram-based CNN rate estimator."""
    win_samples: int = 2000       # 20s at 100 Hz
    nperseg_stft: int = 256
    n_freq_bins: int = 64
    n_time_bins: int = 32
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 32
    pretrained: bool = False


def _signal_to_spectrogram(x: np.ndarray, f_lo: float, f_hi: float,
                           fs: float = FS,
                           nperseg: int = 256) -> np.ndarray:
    """
    Convert a 1-D signal window to a 2-D log-power spectrogram
    cropped to the target frequency band.

    Returns (n_freq, n_time) array.
    """
    from scipy.signal import stft as scipy_stft

    noverlap = nperseg * 3 // 4
    f, t, Zxx = scipy_stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                            boundary=None)
    power = np.abs(Zxx) ** 2
    band_mask = (f >= f_lo) & (f <= f_hi)
    spec = np.log(power[band_mask, :] + 1e-10)

    # Normalise to zero mean, unit variance
    mu, std = spec.mean(), spec.std() + 1e-12
    return (spec - mu) / std


def prepare_spectrogram_dataset(
    sessions,
    band: str = 'resp',
    channel: str = 'CLE-CRE',
    win_sec: float = 20.0,
    step_sec: float = 5.0,
    acc_removal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (spectrograms, gt_rates, subjects) arrays for CNN training.

    Returns
    -------
    specs    : list of 2-D spectrograms
    rates    : (N,) GT rates in Hz
    subjects : (N,) subject IDs
    """
    from .preprocessing import preprocess_full
    from .ground_truth import gt_sliding_rates

    f_lo = RESP_LO if band == 'resp' else CARD_LO
    f_hi = RESP_HI if band == 'resp' else CARD_HI
    fs = FS
    win_n = int(round(win_sec * fs))
    step_n = int(round(step_sec * fs))

    all_specs = []
    all_rates = []
    all_subjects = []

    for session in sessions:
        full, _ = preprocess_full(session, acc_removal)
        sig = full[channel][band]
        gt = gt_sliding_rates(session, win_sec=win_sec, step_sec=step_sec)
        gt_hz = gt['resp_hz'] if band == 'resp' else gt['card_hz']
        gt_t_hr = gt['t_hr']

        for start in range(0, len(sig) - win_n + 1, step_n):
            seg = sig[start:start + win_n]
            t_hr = (start + win_n / 2.0) / fs / 3600.0

            # Find closest GT rate
            gt_valid = ~np.isnan(gt_hz)
            if not gt_valid.any():
                continue
            idx = np.argmin(np.abs(gt_t_hr[gt_valid] - t_hr))
            rate = gt_hz[gt_valid][idx]
            if not np.isfinite(rate):
                continue

            spec = _signal_to_spectrogram(seg, f_lo, f_hi, fs)
            all_specs.append(spec)
            all_rates.append(rate)
            all_subjects.append(session.subject)

    rates = np.array(all_rates)
    subjects = np.array(all_subjects)
    return all_specs, rates, subjects


def train_spectrogram_cnn(specs: list, rates: np.ndarray,
                          subjects: np.ndarray,
                          cfg: SpectrogramCNNConfig | None = None):
    """
    Train a small 2D CNN on spectrograms -> rate regression with LOSO CV.

    Requires PyTorch. Returns (model, cv_results, losses).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            'PyTorch required for SpectrogramCNN. '
            'Install with: pip install torch'
        )

    if cfg is None:
        cfg = SpectrogramCNNConfig()

    # Pad/resize spectrograms to uniform size
    max_f = max(s.shape[0] for s in specs)
    max_t = max(s.shape[1] for s in specs)
    X = np.zeros((len(specs), 1, max_f, max_t), dtype=np.float32)
    for i, s in enumerate(specs):
        X[i, 0, :s.shape[0], :s.shape[1]] = s

    class SmallCNN(nn.Module):
        def __init__(self, h, w):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.regressor(self.features(x)).squeeze(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LOSO CV
    unique_subjects = np.unique(subjects)
    cv_results = []
    for subj in unique_subjects:
        test_mask = subjects == subj
        train_mask = ~test_mask
        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        X_train = torch.FloatTensor(X[train_mask]).to(device)
        y_train = torch.FloatTensor(rates[train_mask]).to(device)
        X_test = torch.FloatTensor(X[test_mask]).to(device)
        y_test = rates[test_mask]

        model = SmallCNN(max_f, max_t).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(X_train, y_train)
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

        model.train()
        for epoch in range(cfg.epochs):
            for xb, yb in loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_test).cpu().numpy()

        errors = np.abs(pred - y_test)
        cv_results.append({
            'subject': subj,
            'mae': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'n': int(test_mask.sum()),
        })

    # Train final model on all data
    X_all = torch.FloatTensor(X).to(device)
    y_all = torch.FloatTensor(rates).to(device)
    final_model = SmallCNN(max_f, max_t).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    train_ds = TensorDataset(X_all, y_all)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    losses = []
    final_model.train()
    for epoch in range(cfg.epochs):
        ep_loss = 0.0
        nb = 0
        for xb, yb in loader:
            pred = final_model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1
        losses.append(ep_loss / max(nb, 1))

    return final_model, cv_results, losses


# ═══════════════════════════════════════════════════════════════════════════════
#  Method catalog  (central registry of all available methods)
# ═══════════════════════════════════════════════════════════════════════════════

METHOD_CATALOG = {
    # ── Base classical (rates.py) ──
    'spectral': {
        'type': 'classical',
        'module': 'rates',
        'description': 'Welch PSD peak frequency',
        'input': 'bandpassed window',
        'complexity': 'O(N log N)',
    },
    'acf': {
        'type': 'classical',
        'module': 'rates',
        'description': 'FFT-autocorrelation dominant lag with parabolic interpolation',
        'input': 'bandpassed window',
        'complexity': 'O(N log N)',
    },
    'hilbert': {
        'type': 'classical',
        'module': 'rates',
        'description': 'Hilbert instantaneous frequency (amplitude-weighted median)',
        'input': 'bandpassed window',
        'complexity': 'O(N log N)',
    },
    'zerocross': {
        'type': 'classical',
        'module': 'rates',
        'description': 'Upward zero-crossing rate with sub-sample interpolation',
        'input': 'bandpassed window',
        'complexity': 'O(N)',
    },
    'peaks': {
        'type': 'classical',
        'module': 'rates',
        'description': 'Peak counting with prominence threshold and smoothing',
        'input': 'bandpassed window',
        'complexity': 'O(N)',
    },
    'envelope': {
        'type': 'classical',
        'module': 'rates',
        'description': 'Teager-Kaiser energy envelope + ACF (for BCG-like cardiac)',
        'input': 'bandpassed window',
        'complexity': 'O(N log N)',
    },
    # ── Scaled variants (rates.py) ──
    'hilbert_scaled_cardiac': {
        'type': 'classical',
        'module': 'rates',
        'description': 'Hilbert inst. freq. / per-session k (cardiac)',
        'input': 'bandpassed window + calibrated k',
        'complexity': 'O(N log N)',
    },
    'peaks_scaled_resp': {
        'type': 'classical',
        'module': 'rates',
        'description': 'Loose peak counting / per-session k (respiratory)',
        'input': 'bandpassed window + calibrated k',
        'complexity': 'O(N)',
    },
    # ── Advanced classical (rates_classical.py) ──
    'vmd': {
        'type': 'classical_advanced',
        'module': 'rates_classical',
        'description': 'Variational Mode Decomposition -- learned band-limited modes',
        'input': 'bandpassed window',
        'complexity': 'O(K * N * max_iter)',
        'reference': 'Dragomiretskiy & Zosso 2014',
    },
    'cwt': {
        'type': 'classical_advanced',
        'module': 'rates_classical',
        'description': 'Continuous Wavelet Transform ridge tracking (Morlet)',
        'input': 'bandpassed window',
        'complexity': 'O(S * N log N)',
        'reference': 'Addison 2005',
    },
    'stft_track': {
        'type': 'classical_advanced',
        'module': 'rates_classical',
        'description': 'STFT spectrogram + Viterbi DP peak tracking with continuity constraint',
        'input': 'bandpassed window',
        'complexity': 'O(N log N + F * T)',
    },
    'music': {
        'type': 'classical_advanced',
        'module': 'rates_classical',
        'description': 'MUSIC pseudo-spectrum (subspace super-resolution)',
        'input': 'bandpassed window',
        'complexity': 'O(M^3 + n_scan * M)',
        'reference': 'Schmidt 1986',
    },
    # ── ML methods (rates_ml.py) ──
    'fusion_xgb': {
        'type': 'ml',
        'module': 'rates_ml',
        'description': 'Learned fusion of classical estimates + quality features (XGBoost)',
        'input': '5 rate estimates + quality features',
        'requires': 'xgboost or sklearn',
        'training': 'LOSO CV, works with 12 sessions',
    },
    'fusion_rf': {
        'type': 'ml',
        'module': 'rates_ml',
        'description': 'Learned fusion via Random Forest',
        'input': '5 rate estimates + quality features',
        'requires': 'sklearn',
        'training': 'LOSO CV, works with 12 sessions',
    },
    'fusion_mlp': {
        'type': 'ml',
        'module': 'rates_ml',
        'description': 'Learned fusion via MLP regressor',
        'input': '5 rate estimates + quality features',
        'requires': 'sklearn',
        'training': 'LOSO CV, works with 12 sessions',
    },
    'denoising_ae': {
        'type': 'ml',
        'module': 'rates_ml',
        'description': 'Self-supervised 1D conv autoencoder denoising + classical extraction',
        'input': 'raw bandpassed window',
        'requires': 'torch',
        'training': 'Self-supervised (no GT labels), all sessions',
    },
    'spectrogram_cnn': {
        'type': 'ml',
        'module': 'rates_ml',
        'description': '2D CNN on STFT spectrograms -> rate regression',
        'input': 'STFT spectrogram image',
        'requires': 'torch',
        'training': 'LOSO CV, benefits from data augmentation',
    },
}


def list_methods(type_filter: str | None = None) -> list:
    """
    List available rate detection methods.

    Parameters
    ----------
    type_filter : 'classical', 'classical_advanced', 'ml', or None for all
    """
    methods = []
    for name, info in METHOD_CATALOG.items():
        if type_filter and info['type'] != type_filter:
            continue
        methods.append({'name': name, **info})
    return methods


def print_method_catalog():
    """Print a formatted summary of all available methods."""
    for mtype in ['classical', 'classical_advanced', 'ml']:
        methods = list_methods(mtype)
        if not methods:
            continue
        label = {
            'classical': 'Base Classical',
            'classical_advanced': 'Advanced Classical',
            'ml': 'Machine Learning',
        }[mtype]
        print(f'\n{"="*60}')
        print(f' {label} Methods')
        print(f'{"="*60}')
        for m in methods:
            print(f"\n  {m['name']}")
            print(f"    {m['description']}")
            if 'reference' in m:
                print(f"    ref: {m['reference']}")
            if 'requires' in m:
                print(f"    requires: {m['requires']}")
