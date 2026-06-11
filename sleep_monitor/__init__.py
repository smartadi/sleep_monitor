"""
sleep_monitor — capacitive sleep sensor analysis package.

Quick start
-----------
>>> from sleep_monitor import load_session, SESSION_META, list_sessions
>>> list_sessions()
>>> session = load_session(0)          # load S1N1
>>> session = load_session(0, with_profile=True)   # also load PSG staging
"""

from .config import (
    FS, BASE_DIR, PSG_BASE_DIR,
    CAP_CHANNELS, PSG_CHANNELS, ALL_SIG_COLS, CAP_CHANS,
    CAP_COLORS, GT_COLOR,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    EEG_BANDS, BAND_COLORS,
    DELTA_SUB_BANDS, DELTA_SUB_COLORS,
    METHOD_NAMES, METHOD_LABELS, METHOD_COLORS,
    ADVANCED_METHOD_NAMES, ADVANCED_METHOD_LABELS, ADVANCED_METHOD_COLORS,
    ML_METHOD_NAMES, ALL_METHOD_NAMES, ALL_METHOD_LABELS, ALL_METHOD_COLORS,
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER,
    APNEA_CODES, APNEA_LABELS, APNEA_COLORS,
)
from .sessions import SESSION_META, SleepSession, find_meta, list_sessions
from .loader import load_session as _load_session, load_sleep_profile, load_apnea_events, load_all_sessions
from .filters import bandpass, lowpass, highpass, detrend_segment, outlier_clip, moving_average, rolling_zscore
from .preprocessing import (
    remove_acc_artifact, remove_acc_artifact_nlms,
    preprocess_window, preprocess_full,
)
from .rates import (
    rate_spectral, rate_acf, rate_hilbert, rate_zerocross, rate_peaks, rate_envelope,
    rate_hilbert_scaled_cardiac, rate_peaks_scaled_resp,
    calibrate_k_cardiac, calibrate_k_resp,
    estimate_rate, fuse_rates, detect_peaks, zerocross_indices,
    peaks_by_method, sliding_rates,
)
from .quality import (
    window_features, combined_quality,
    inband_snr, acf_prominence, spectral_concentration, motion_power, method_agreement,
)
from .evaluate import (
    PipelineConfig, BaseKey, run_pipeline, evaluate_pipeline,
    compute_base_windows, derive_rate,
    default_grid, default_base_keys, evaluate_on_sessions,
)
from .classifier import (
    load_windows, build_dataset, default_models,
    loso_evaluate, summarise,
    METHOD_COLS, QUALITY_COLS,
)
from .metrics import accuracy_metrics, metrics_table, summary_by_method
from .morphology import (
    MorphCfg, ClusterEvent,
    preprocess_diff, run_pipeline,
    acf_rates_from_cap, compute_rate_divisor,
    events_to_rates, band_events_to_rates,
    gt_event_rates, gt_event_times_peaks,
    bland_altman, event_summary,
)
from .rates_classical import (
    rate_vmd, rate_cwt, rate_stft_track, rate_music,
    estimate_rate_advanced, sliding_rates_advanced,
)
from .rates_ml import (
    build_fusion_features, FusionDataset, build_fusion_dataset,
    train_fusion_model, predict_fused_rate,
    AutoencoderConfig, prepare_autoencoder_data,
    train_autoencoder, denoise_signal,
    SpectrogramCNNConfig, prepare_spectrogram_dataset, train_spectrogram_cnn,
    METHOD_CATALOG, list_methods, print_method_catalog,
    FUSION_FEATURE_COLS,
)
from .spectral import compute_band_power_ratios
from .harmonics import detect_harmonics, detect_harmonics_multichannel
from .staging import extract_epoch_psd, extract_epoch_features, get_feature_columns
from .ground_truth import (
    GTResult, gt_heart_rate, gt_resp_rate, gt_sliding_rates,
)
from .motion import (
    head_orientation, dynamic_acceleration,
    epoch_motion, epoch_cap_stats, classify_position,
)
from .viz import (
    plot_hypnogram, plot_session_overview, plot_all_sessions_grid,
    plot_rates_vs_gt, plot_window_inspection, plot_eeg_spectrogram,
)


def load_session(idx_or_meta, dtype=None, with_profile: bool = False, with_apnea: bool = False):
    """
    Convenience wrapper: load a session and optionally attach its sleep profile / apnea events.

    Parameters
    ----------
    idx_or_meta  : int (0-11) or meta dict
    dtype        : numpy dtype for signal arrays (default float32)
    with_profile : if True, also load the PSG sleep staging profile
    with_apnea   : if True, also load the PSG apnea/hypopnea events
    """
    kw = {} if dtype is None else {'dtype': dtype}
    s = _load_session(idx_or_meta, **kw)
    if with_profile:
        s.sleep_profile = load_sleep_profile(s)
    if with_apnea:
        s.apnea_events = load_apnea_events(s)
    return s


__all__ = [
    # config
    'FS', 'BASE_DIR', 'PSG_BASE_DIR',
    'CAP_CHANNELS', 'PSG_CHANNELS', 'ALL_SIG_COLS', 'CAP_CHANS',
    'CAP_COLORS', 'GT_COLOR',
    'RESP_LO', 'RESP_HI', 'CARD_LO', 'CARD_HI',
    'EEG_BANDS', 'BAND_COLORS',
    'METHOD_NAMES', 'METHOD_LABELS', 'METHOD_COLORS',
    'ADVANCED_METHOD_NAMES', 'ADVANCED_METHOD_LABELS', 'ADVANCED_METHOD_COLORS',
    'ML_METHOD_NAMES', 'ALL_METHOD_NAMES', 'ALL_METHOD_LABELS', 'ALL_METHOD_COLORS',
    'STAGE_LABELS', 'STAGE_COLORS', 'STAGE_ORDER',
    'APNEA_CODES', 'APNEA_LABELS', 'APNEA_COLORS',
    # sessions
    'SESSION_META', 'SleepSession', 'find_meta', 'list_sessions',
    # loading
    'load_session', 'load_sleep_profile', 'load_apnea_events', 'load_all_sessions',
    # filters
    'bandpass', 'lowpass', 'highpass', 'detrend_segment',
    'outlier_clip', 'moving_average', 'rolling_zscore',
    # preprocessing
    'remove_acc_artifact', 'remove_acc_artifact_nlms',
    'preprocess_window', 'preprocess_full',
    # rates (base)
    'rate_spectral', 'rate_acf', 'rate_hilbert', 'rate_zerocross', 'rate_peaks', 'rate_envelope',
    'rate_hilbert_scaled_cardiac', 'rate_peaks_scaled_resp',
    'calibrate_k_cardiac', 'calibrate_k_resp',
    'estimate_rate', 'fuse_rates', 'detect_peaks', 'zerocross_indices',
    'peaks_by_method', 'sliding_rates',
    # rates (advanced classical)
    'rate_vmd', 'rate_cwt', 'rate_stft_track', 'rate_music',
    'estimate_rate_advanced', 'sliding_rates_advanced',
    # rates (ML)
    'build_fusion_features', 'FusionDataset', 'build_fusion_dataset',
    'train_fusion_model', 'predict_fused_rate',
    'AutoencoderConfig', 'prepare_autoencoder_data',
    'train_autoencoder', 'denoise_signal',
    'SpectrogramCNNConfig', 'prepare_spectrogram_dataset', 'train_spectrogram_cnn',
    'METHOD_CATALOG', 'list_methods', 'print_method_catalog',
    'FUSION_FEATURE_COLS',
    # quality
    'window_features', 'combined_quality',
    'inband_snr', 'acf_prominence', 'spectral_concentration', 'motion_power', 'method_agreement',
    # evaluate
    'PipelineConfig', 'BaseKey', 'run_pipeline', 'evaluate_pipeline',
    'compute_base_windows', 'derive_rate',
    'default_grid', 'default_base_keys', 'evaluate_on_sessions',
    # classifier
    'load_windows', 'build_dataset', 'default_models',
    'loso_evaluate', 'summarise',
    'METHOD_COLS', 'QUALITY_COLS',
    # spectral
    'compute_band_power_ratios',
    # harmonics
    'detect_harmonics', 'detect_harmonics_multichannel',
    # staging
    'extract_epoch_psd', 'extract_epoch_features', 'get_feature_columns',
    # ground truth
    'GTResult', 'gt_heart_rate', 'gt_resp_rate', 'gt_sliding_rates',
    # metrics
    'accuracy_metrics', 'metrics_table', 'summary_by_method',
    # motion
    'head_orientation', 'dynamic_acceleration',
    'epoch_motion', 'epoch_cap_stats', 'classify_position',
    # viz
    'plot_hypnogram', 'plot_session_overview', 'plot_all_sessions_grid',
    'plot_rates_vs_gt', 'plot_window_inspection', 'plot_eeg_spectrogram',
]
