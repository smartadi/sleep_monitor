"""
Session loading with an on-disk cache.

Parsing one overnight .csv.gz takes several seconds and every stage needs all
twelve, so the parsed arrays are kept under paper/_cache/sessions/ as
uncompressed .npz (about 2 GB for the cohort). The cache is a pure copy of what
`loader.load_session` returns -- delete it at any time and it is rebuilt.

Writes are atomic (temp file + os.replace), so stages run in parallel cannot
read a half-written cache file.
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .loader import load_session, load_sleep_profile
from .sessions import SESSION_META, SleepSession

LABELS = [m['label'] for m in SESSION_META]
_SESS_CACHE = CACHE_DIR / 'sessions'
_CACHE_VERSION = 1


def _meta(label: str) -> dict:
    for m in SESSION_META:
        if m['label'] == label:
            return m
    raise KeyError(f'unknown session {label!r}; expected one of {LABELS}')


def _save(path, s: SleepSession) -> None:
    arrays = {'time_ms': s.time_ms, 'time_hr': s.time_hr,
              'fs': np.array(s.fs), 'version': np.array(_CACHE_VERSION),
              'time_start': np.array('' if s.time_start is None else str(s.time_start))}
    arrays.update({f'cap__{k}': v for k, v in s.cap.items()})
    arrays.update({f'psg__{k}': v for k, v in s.psg.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.npz.tmp')
    os.close(fd)
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _load(path, meta: dict) -> Optional[SleepSession]:
    try:
        z = np.load(path, allow_pickle=False)
        if int(z['version']) != _CACHE_VERSION:
            return None
        ts = str(z['time_start'])
        return SleepSession(
            meta=meta, time_ms=z['time_ms'], time_hr=z['time_hr'],
            time_start=pd.Timestamp(ts) if ts else None,
            cap={k[5:]: z[k] for k in z.files if k.startswith('cap__')},
            psg={k[5:]: z[k] for k in z.files if k.startswith('psg__')},
            fs=float(z['fs']))
    except Exception:
        return None


def get_session(label: str, profile: bool = True) -> SleepSession:
    """One recording by label ('S1N1'...'S6N2'), with its PSG sleep profile."""
    meta = _meta(label)
    path = _SESS_CACHE / f'{label}.npz'
    s = _load(path, meta) if path.exists() else None
    if s is None:
        s = load_session(meta)
        _save(path, s)
    if profile:
        s.sleep_profile = load_sleep_profile(s)
    return s


def iter_sessions(labels=None, profile: bool = True) -> Iterator[SleepSession]:
    """All twelve recordings in order, one at a time (memory stays at one night)."""
    for lab in (labels or LABELS):
        yield get_session(lab, profile=profile)
