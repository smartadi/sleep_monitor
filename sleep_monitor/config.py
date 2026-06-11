"""
Central configuration: data paths, channel lists, frequency bands, visual constants.
All notebooks and scripts import from here — change paths in one place only.
"""

import os
from pathlib import Path

# ── Data directories ───────────────────────────────────────────────────────────
# Set SLEEP_DATA_DIR to the parent folder containing the two dataset directories.
# Example: SLEEP_DATA_DIR=C:\Users\you\Documents\sleep monitor
_data_root = Path(os.environ.get('SLEEP_DATA_DIR', r'C:\Users\adity\Documents\sleep monitor'))

BASE_DIR = (
    _data_root
    / 'overnight_6subject_pelthupdate_030526'
    / 'overnight_6subject_pelthupdate_030526'
)
PSG_BASE_DIR = (
    _data_root
    / 'overnight_6subject_complete_032626'
    / 'overnight_6subject_complete_032626'
)

# ── Sampling rate ──────────────────────────────────────────────────────────────
FS = 100.0  # Hz

# ── Channel definitions ────────────────────────────────────────────────────────
CAP_CHANNELS = ['CH', 'CLE', 'CRE', 'aX', 'aY', 'aZ']
PSG_CHANNELS = ['EEG', 'EOGl', 'EOGr', 'ECG', 'Flow', 'Pleth', 'Thorax', 'Abdomen']
ALL_SIG_COLS = CAP_CHANNELS + PSG_CHANNELS

CAP_CHANS  = ['CH', 'CLE', 'CRE', 'CLE-CRE']
CAP_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD', 'CLE-CRE': '#E67E22'}
GT_COLOR   = '#2C3E50'

# ── Frequency bands ────────────────────────────────────────────────────────────
RESP_LO, RESP_HI = 0.1, 0.5   # Hz  (~6–30 br/min)
CARD_LO, CARD_HI = 0.5, 3.0   # Hz  (~30–180 BPM)

EEG_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
}
BAND_COLORS = {
    'delta': '#C0392B', 'theta': '#E67E22',
    'alpha': '#27AE60', 'beta':  '#2980B9',
}

DELTA_SUB_BANDS = {
    'infra_slow': (0.0, 0.5),
    'SO':         (0.5, 1.0),
    'delta_low':  (1.0, 2.0),
    'delta_high': (2.0, 4.0),
}
DELTA_SUB_COLORS = {
    'infra_slow': '#1ABC9C',
    'SO':         '#8E44AD',
    'delta_low':  '#C0392B',
    'delta_high': '#E67E22',
}

# ── Rate estimation methods ────────────────────────────────────────────────────
METHOD_NAMES  = ['spectral', 'acf', 'hilbert', 'zerocross', 'peaks']
METHOD_LABELS = {
    'spectral':  'Spectral peak',
    'acf':       'ACF',
    'hilbert':   'Hilbert inst. freq.',
    'zerocross': 'Zero-crossing',
    'peaks':     'Peak counting',
}
METHOD_COLORS = {
    'spectral':  '#3498DB',
    'acf':       '#E74C3C',
    'hilbert':   '#27AE60',
    'zerocross': '#9B59B6',
    'peaks':     '#E67E22',
}

# ── Advanced rate estimation methods ──────────────────────────────────────────
ADVANCED_METHOD_NAMES = ['vmd', 'cwt', 'stft_track', 'music']
ADVANCED_METHOD_LABELS = {
    'vmd':        'VMD',
    'cwt':        'CWT ridge',
    'stft_track': 'STFT+Viterbi',
    'music':      'MUSIC',
}
ADVANCED_METHOD_COLORS = {
    'vmd':        '#1ABC9C',
    'cwt':        '#F1C40F',
    'stft_track': '#E91E63',
    'music':      '#795548',
}

# ── ML method names ───────────────────────────────────────────────────────────
ML_METHOD_NAMES = ['fusion_xgb', 'fusion_rf', 'fusion_mlp',
                   'denoising_ae', 'spectrogram_cnn']

# ── All methods combined ──────────────────────────────────────────────────────
ALL_METHOD_NAMES = METHOD_NAMES + ADVANCED_METHOD_NAMES + ML_METHOD_NAMES
ALL_METHOD_LABELS = {**METHOD_LABELS, **ADVANCED_METHOD_LABELS}
ALL_METHOD_COLORS = {**METHOD_COLORS, **ADVANCED_METHOD_COLORS}

# ── Sleep stage labels & colors ────────────────────────────────────────────────
STAGE_LABELS = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake', -1: '?'}
STAGE_COLORS = {
    0: '#9B59B6', 1: '#2ECC71', 2: '#3498DB',
    3: '#F39C12', 4: '#E74C3C', -1: '#AAAAAA',
}
STAGE_ORDER = [0, 1, 2, 3, 4]

# Stage codes used in PSG Sleep Profile text files
PSG_STAGE_MAP = {
    'wake': 4, 'stage 1': 3, 'stage 2': 2, 'stage 3': 1,
    'rem': 0, 'artifact': 4, 'a': -1,
}
PSG_EPOCH_SEC = 30.0  # PSG scoring epoch length

# ── Apnea event labels & colors ──────────────────────────────────────────────
APNEA_CODES  = {
    'apnea': 1, 'hypopnea': 2,
    'obstructive apnea': 1, 'mixed apnea': 1, 'central apnea': 1,
}
APNEA_LABELS = {0: 'Normal', 1: 'Apnea', 2: 'Hypopnea'}
APNEA_COLORS = {0: '#2ECC71', 1: '#E74C3C', 2: '#E67E22'}
