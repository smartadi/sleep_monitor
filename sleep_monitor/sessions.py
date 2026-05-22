"""
Session registry and SleepSession dataclass.

SESSION_META is the authoritative list of all 12 recordings.
SleepSession is the data container passed through the entire pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .config import BASE_DIR, PSG_BASE_DIR


# ── Path helpers ───────────────────────────────────────────────────────────────

def _csv_path(sid: str, ini: str, d: str, var: str = '') -> Path:
    tag = '_1point_sync' if var == '1point' else ''
    return (BASE_DIR / f'{sid} - {ini}' / d / f'Sync_{d}'
            / f'SleepMask_PSG_100Hz{tag}_combined_{d}.csv.gz')


def _psg_dir(sid: str, ini: str, d: str) -> Path:
    return PSG_BASE_DIR / f'{sid} - {ini}' / d


# ── Raw session table ──────────────────────────────────────────────────────────
# (subject_id, initials, date, sync_variant)
_SESSIONS = [
    ('OS001', 'KJK', '09-17-2024', ''),       # S1N1
    ('OS001', 'KJK', '09-18-2024', ''),       # S1N2
    ('OS002', 'LDI', '09-19-2024', ''),       # S2N1
    ('OS002', 'LDI', '09-20-2024', ''),       # S2N2
    ('OS003', 'LCW', '12-18-2025', ''),       # S3N1
    ('OS003', 'LCW', '12-19-2025', ''),       # S3N2
    ('OS004', 'CJH', '12-25-2025', '1point'), # S4N1
    ('OS004', 'CJH', '12-26-2025', '1point'), # S4N2
    ('OS005', 'CJY', '01-03-2026', '1point'), # S5N1
    ('OS005', 'CJY', '12-27-2025', '1point'), # S5N2
    ('OS006', 'SK',  '01-14-2026', ''),       # S6N1
    ('OS006', 'SK',  '01-15-2026', ''),       # S6N2
]

SESSION_META: List[dict] = [
    {
        'idx':      i,
        'subject':  sid,
        'initials': ini,
        'night':    (i % 2) + 1,
        'label':    f'S{(i // 2) + 1}N{(i % 2) + 1}',
        'date':     d,
        'csv':      _csv_path(sid, ini, d, var),
        'psg_dir':  _psg_dir(sid, ini, d),
    }
    for i, (sid, ini, d, var) in enumerate(_SESSIONS)
]


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class SleepSession:
    """
    Container for one overnight sleep recording.

    Attributes
    ----------
    meta          : metadata dict from SESSION_META
    time_ms       : (N,) float32 — milliseconds from recording start
    time_hr       : (N,) float32 — hours from recording start
    time_start    : pd.Timestamp or None — absolute wall-clock start
    cap           : dict of (N,) arrays — CH, CLE, CRE, aX, aY, aZ, acc_mag
    psg           : dict of (N,) arrays — EEG, EOGl, EOGr, ECG, Flow, Pleth, Thorax, Abdomen
    fs            : sampling rate in Hz (default 100.0)
    sleep_profile : optional dict loaded by load_sleep_profile()
                    keys: t_ep_hr, labels, codes
    """
    meta:          dict
    time_ms:       np.ndarray
    time_hr:       np.ndarray
    time_start:    object                        # pd.Timestamp | None
    cap:           Dict[str, np.ndarray]
    psg:           Dict[str, np.ndarray]
    fs:            float = 100.0
    sleep_profile: Optional[dict] = field(default=None, repr=False)
    apnea_events:  Optional[dict] = field(default=None, repr=False)

    # ── convenience properties ─────────────────────────────────────────────────

    @property
    def label(self) -> str:
        return self.meta['label']

    @property
    def subject(self) -> str:
        return self.meta['subject']

    @property
    def duration_hr(self) -> float:
        return float(self.time_hr[-1])

    @property
    def n_samples(self) -> int:
        return len(self.time_hr)

    def apnea_at(self, t_hr: np.ndarray) -> np.ndarray:
        """Return apnea code (0=Normal, 1=Apnea, 2=Hypopnea) for each time in t_hr."""
        codes = np.zeros(len(t_hr), dtype=np.int8)
        if self.apnea_events is None:
            return codes
        ev = self.apnea_events
        for s, e, c in zip(ev['start_hr'], ev['end_hr'], ev['codes']):
            mask = (t_hr >= s) & (t_hr <= e)
            np.maximum(codes, np.where(mask, c, 0).astype(np.int8), out=codes)
        return codes

    def __repr__(self) -> str:
        sp = ' +PSG' if self.sleep_profile is not None else ''
        ap = ' +Apnea' if self.apnea_events is not None else ''
        return (f"SleepSession({self.label} {self.subject}-{self.meta['initials']}"
                f" {self.meta['date']} {self.duration_hr:.2f}hr {self.n_samples:,}samp{sp}{ap})")


# ── Lookup helpers ─────────────────────────────────────────────────────────────

def find_meta(subject: str, night: int) -> dict:
    """Return SESSION_META entry for the given subject and night (1 or 2)."""
    matches = [m for m in SESSION_META if m['subject'] == subject and m['night'] == night]
    if not matches:
        available = sorted({m['subject'] for m in SESSION_META})
        raise ValueError(
            f"No session for subject={subject!r} night={night}. "
            f"Available subjects: {available}"
        )
    return matches[0]


def list_sessions(check_files: bool = True) -> None:
    """Print a summary of all sessions with optional file existence check."""
    for m in SESSION_META:
        if check_files:
            status = '✓' if m['csv'].exists() else '✗'
            print(f"  [{m['idx']:2d}] {m['label']}  {m['subject']}-{m['initials']}"
                  f"  {m['date']}  csv:{status}")
        else:
            print(f"  [{m['idx']:2d}] {m['label']}  {m['subject']}-{m['initials']}"
                  f"  {m['date']}")
