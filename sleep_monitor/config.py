"""
Central configuration: data paths, channel lists, frequency bands, visual constants.
All notebooks and scripts import from here — change paths in one place only.
"""

from pathlib import Path

# ── Data directories ───────────────────────────────────────────────────────────
BASE_DIR = Path(
    r'C:\Users\adity\Documents\sleep monitor'
    r'\overnight_6subject_pelthupdate_030526'
    r'\overnight_6subject_pelthupdate_030526'
)
PSG_BASE_DIR = Path(
    r'C:\Users\adity\Documents\sleep monitor'
    r'\overnight_6subject_complete_032626'
    r'\overnight_6subject_complete_032626'
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

# ── Capacitance units ─────────────────────────────────────────────────────────
# The CAP columns (CH, CLE, CRE) in the overnight sync CSVs are true capacitance
# readings from the mask's capacitance-to-digital front end, recorded in
# FEMTOFARADS (fF). Use CAP_UNIT / CAP_UNIT_SQ for axis labels so no script has
# to hard-code "a.u." again.
#
# Provenance / evidence for the fF scale (NOT a stated figure — see caveats):
#   * On-head values are CLE ~1960-2050, CRE ~1620-2350, i.e. ~2 pF, comfortably
#     inside the front end's +/-15 pF range and the right order for this sensor:
#     the source paper reports a bare-sensor C0 of 739.7 +/- 20.9 fF and a
#     proximity swing of 22.1 -> 345.7 fF from 30 mm to 2 mm.
#   * The no-subject baseline recording ('baseline noise/SM2_33.txt') carries the
#     same channels at almost exactly 1/1000 the magnitude (CLE 1.83, CRE 2.18,
#     CH -0.914) -> that file is in pF, these are in fF.
#   * CH is negative, so it is a signed difference channel, not an absolute
#     capacitance (see CH_IS_INTERHEMISPHERIC_DIFF below).
#
# CAVEATS (flagged rather than buried):
#   * Neither source paper states the units of the recorded files. Bench numbers
#     are quoted in fF but the nap-study Methods use "a.u.". The 1:1 scale below
#     is inferred from the evidence above, not read off a datasheet.
#   * The papers name the readout IC as TI FDC1004 ("sub-femtofarad resolution",
#     2.4 V @ 25 kHz excitation). "FTD001" appears nowhere in either paper.
#   * No per-LSB figure or full-scale range is given in either paper.
# If a calibration sheet later gives a different scale, change CAP_SCALE_TO_FF
# only -- every downstream label and statistic follows from it.
CAP_SCALE_TO_FF = 1.0        # multiply a raw CAP column by this to get fF
CAP_UNIT        = 'fF'       # axis-label unit for a mean / level
CAP_UNIT_SQ     = 'fF²'      # axis-label unit for a variance
CAP_UNIT_RATE   = 'fF/hr'    # axis-label unit for a drift rate

# CH is documented in the source paper as the INTERHEMISPHERIC difference
# ("the difference between left and right hemispheres", positive => higher
# regional ICP on the left), from a four-sensor / three-signal front end. It is
# NOT stated to equal CLE - CRE, and empirically it does not: see
# analysis/mean_value/ch_vs_diff.py.
CH_IS_INTERHEMISPHERIC_DIFF = True

# Capacitance -> intracranial pressure sensitivity, for converting a capacitance
# excursion into an approximate pressure excursion. PORCINE, CLE, respiratory
# band; the source paper reports linearity only over <1 mm displacement and
# 0-8.2 mmHg (r = 0.92), and explicitly warns the sensitivity depends on
# cerebrovascular/tissue geometry. Treat any mmHg figure derived from it as a
# rough scale, not a calibrated pressure, and never as absolute ICP -- only as
# a CHANGE in pressure.
ICP_SENS_FF_PER_MMHG = 4.79

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
METHOD_NAMES  = ['spectral', 'acf', 'hilbert', 'zerocross', 'peaks', 'adaptive_peaks']
METHOD_LABELS = {
    'spectral':       'Spectral peak',
    'acf':            'ACF',
    'hilbert':        'Hilbert inst. freq.',
    'zerocross':      'Zero-crossing',
    'peaks':          'Peak counting',
    'adaptive_peaks': 'Adaptive peaks',
}
METHOD_COLORS = {
    'spectral':       '#3498DB',
    'acf':            '#E74C3C',
    'hilbert':        '#27AE60',
    'zerocross':      '#9B59B6',
    'peaks':          '#E67E22',
    'adaptive_peaks': '#16A085',
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
