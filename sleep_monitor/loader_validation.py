"""
Loader for the validation dataset (combinedDataAnalyses_041626).

Short (~12.5 min) controlled-posture sessions with 4 cap channels
(Cvl, Cvr, Cbl, Cbr) and PSG subset (Pleth, Thorax, EEG, Puls, MAP).

No ECG or Flow — GT falls back to Pleth (cardiac) and Thorax (respiratory).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .config import FS

VALIDATION_DIR = Path(
    r'C:\Users\adity\Documents\sleep monitor\combinedDataAnalyses_041626'
)

CAP_CHANNELS_VAL = ['Cvl', 'Cvr', 'Cbl', 'Cbr']

PHASE_COLORS = {
    'layDownRest':   '#3498DB',
    'sit90DegRest':  '#E74C3C',
    'degree0':       '#2ECC71',
    'degree30':      '#F39C12',
    'degree90':      '#9B59B6',
    'turnLeft':      '#1ABC9C',
    'turnMiddle':    '#E67E22',
    'turnRight':     '#795548',
    'valsavaMild':   '#C0392B',
    'valsavaHigh':   '#8E44AD',
}

PHASE_LABELS = {
    'layDownRest':   'Lay down',
    'sit90DegRest':  'Sit 90°',
    'degree0':       '0°',
    'degree30':      '30°',
    'degree90':      '90°',
    'turnLeft':      'Turn L',
    'turnMiddle':    'Turn M',
    'turnRight':     'Turn R',
    'valsavaMild':   'Valsalva mild',
    'valsavaHigh':   'Valsalva high',
}


@dataclass
class ValidationSession:
    label: str
    subject: str
    time_s: np.ndarray
    time_hr: np.ndarray
    cap: Dict[str, np.ndarray]
    psg: Dict[str, np.ndarray]
    phases: np.ndarray          # string array of experimentMode per sample
    phase_segments: List[dict]  # [{phase, start_s, end_s, start_idx, end_idx}]
    fs: float = FS


def load_validation_session(subject_idx: int) -> ValidationSession:
    """
    Load one validation session by index (0-5).
    """
    files = sorted(VALIDATION_DIR.glob('S000* - ICP - *.txt'))
    if subject_idx >= len(files):
        raise IndexError(f'Only {len(files)} validation files, got index {subject_idx}')

    path = files[subject_idx]
    label = f'V{subject_idx + 1}'
    subject = path.name.split(' - ')[0]

    print(f'Loading {label} ({subject}) from {path.name}...', end=' ', flush=True)

    df = pd.read_csv(path, sep='\t')

    # Time axis
    t_ms = pd.to_datetime(df['time'])
    t_s = ((t_ms - t_ms.iloc[0]).dt.total_seconds()).values.astype(np.float64)
    t_hr = t_s / 3600.0

    # CAP channels
    cap = {}
    for ch in CAP_CHANNELS_VAL:
        cap[ch] = df[ch].values.astype(np.float64)
    cap['Cvl-Cvr'] = cap['Cvl'] - cap['Cvr']
    cap['Cbl-Cbr'] = cap['Cbl'] - cap['Cbr']
    cap['acc_mag'] = np.sqrt(
        df['aX'].values.astype(np.float64)**2 +
        df['aY'].values.astype(np.float64)**2 +
        df['aZ'].values.astype(np.float64)**2
    )

    # PSG channels — map to names the GT functions expect
    psg = {
        'Pleth':  df['Pleth'].values.astype(np.float64),
        'Thorax': df['Thorax'].values.astype(np.float64),
        'EEG':    df['EEG'].values.astype(np.float64),
    }
    # Provide empty arrays for channels GT functions may try to access
    n = len(df)
    psg['ECG'] = np.zeros(n, dtype=np.float64)
    psg['Flow'] = np.zeros(n, dtype=np.float64)

    # Phases
    phases = df['experimentMode'].values

    # Build phase segments
    segments = []
    current = phases[0]
    seg_start = 0
    for i in range(1, len(phases)):
        if phases[i] != current:
            segments.append({
                'phase': current,
                'start_s': t_s[seg_start],
                'end_s': t_s[i - 1],
                'start_idx': seg_start,
                'end_idx': i - 1,
            })
            current = phases[i]
            seg_start = i
    segments.append({
        'phase': current,
        'start_s': t_s[seg_start],
        'end_s': t_s[-1],
        'start_idx': seg_start,
        'end_idx': len(phases) - 1,
    })

    print(f'{t_s[-1]:.0f}s  ({n:,} samples, {len(segments)} phases)')

    return ValidationSession(
        label=label, subject=subject,
        time_s=t_s, time_hr=t_hr,
        cap=cap, psg=psg,
        phases=phases, phase_segments=segments,
        fs=FS,
    )


def load_all_validation_sessions() -> List[ValidationSession]:
    files = sorted(VALIDATION_DIR.glob('S000* - ICP - *.txt'))
    return [load_validation_session(i) for i in range(len(files))]
