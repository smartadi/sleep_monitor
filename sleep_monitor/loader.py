"""
Data loading: CSV.GZ recordings and PSG Sleep Profile text files.
"""

from __future__ import annotations
import glob
import re
from typing import List, Optional
import numpy as np
import pandas as pd

from .config import CAP_CHANNELS, PSG_CHANNELS, ALL_SIG_COLS, FS, PSG_STAGE_MAP, PSG_EPOCH_SEC, STAGE_LABELS, APNEA_CODES
from .sessions import SESSION_META, SleepSession


# ── Session loader ─────────────────────────────────────────────────────────────

def load_session(idx_or_meta, dtype=np.float32) -> SleepSession:
    """
    Load one overnight session from its CSV.GZ file.

    Parameters
    ----------
    idx_or_meta : int or dict
        Session index 0-11, or a meta dict from SESSION_META.
    dtype : numpy dtype
        Storage precision for signal arrays (float32 saves ~50% vs float64).

    Returns
    -------
    SleepSession
    """
    meta = SESSION_META[idx_or_meta] if isinstance(idx_or_meta, int) else idx_or_meta
    print(
        f"Loading {meta['label']} ({meta['subject']}-{meta['initials']} {meta['date']})...",
        end=' ', flush=True,
    )

    # Try reading with wall-clock column (timeSM); fall back gracefully if absent
    try:
        df = pd.read_csv(
            meta['csv'], compression='gzip',
            dtype={c: np.float32 for c in ALL_SIG_COLS + ['timeMS']},
            usecols=['timeSM', 'timeMS'] + ALL_SIG_COLS,
        )
        time_start = pd.to_datetime(df['timeSM'].iloc[0])
    except (ValueError, KeyError):
        df = pd.read_csv(
            meta['csv'], compression='gzip',
            dtype={c: np.float32 for c in ALL_SIG_COLS + ['timeMS']},
            usecols=['timeMS'] + ALL_SIG_COLS,
        )
        time_start = None

    t_ms = df['timeMS'].to_numpy(dtype=dtype)
    t_ms -= t_ms[0]
    t_hr = (t_ms / 3_600_000.0).astype(dtype)

    cap = {ch: df[ch].to_numpy(dtype=dtype) for ch in CAP_CHANNELS}
    cap['acc_mag'] = np.sqrt(
        cap['aX'].astype(np.float64)**2 +
        cap['aY'].astype(np.float64)**2 +
        cap['aZ'].astype(np.float64)**2
    ).astype(dtype)
    psg = {ch: df[ch].to_numpy(dtype=dtype) for ch in PSG_CHANNELS}

    print(f'{t_hr[-1]:.2f} hr  ({len(t_hr):,} samples)')
    return SleepSession(
        meta=meta, time_ms=t_ms, time_hr=t_hr, time_start=time_start,
        cap=cap, psg=psg, fs=FS,
    )


def load_all_sessions(
    indices: Optional[List[int]] = None,
    dtype=np.float32,
    with_sleep_profiles: bool = False,
    with_apnea: bool = False,
) -> List[SleepSession]:
    """
    Load multiple sessions.

    Parameters
    ----------
    indices            : list of session indices (default: all 12)
    dtype              : storage dtype
    with_sleep_profiles: also load PSG sleep stage profiles
    with_apnea         : also load PSG apnea/hypopnea events
    """
    if indices is None:
        indices = list(range(len(SESSION_META)))
    sessions = []
    for i in indices:
        s = load_session(i, dtype=dtype)
        if with_sleep_profiles:
            s.sleep_profile = load_sleep_profile(s)
        if with_apnea:
            s.apnea_events = load_apnea_events(s)
        sessions.append(s)
    return sessions


# ── PSG sleep profile loader ───────────────────────────────────────────────────

_DATA_LINE_RE = re.compile(r'^(\d{2}):(\d{2}):(\d{2}),(\d{3});\s*(.+)$')
_APNEA_LINE_RE = re.compile(
    r'^(\d{2}:\d{2}:\d{2}),(\d{3})-(\d{2}:\d{2}:\d{2}),(\d{3});\s*(\d+);(.+)$'
)


def _tod_sec_to_session_hr(tod_sec: np.ndarray, session: SleepSession) -> np.ndarray:
    """
    Convert PSG time-of-day (seconds since midnight) to hours from CSV start.

    Same wall-clock alignment used by load_sleep_profile: without it a time-of-day
    stamp is only accidentally correct for recordings that begin near midnight.
    """
    if session.time_start is None:
        return tod_sec / 3600.0
    ts = session.time_start
    if getattr(ts, 'tz', None) is not None:
        ts = ts.tz_localize(None) if hasattr(ts, 'tz_localize') else ts.replace(tzinfo=None)
    start_tod = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6
    off = np.asarray(tod_sec, float) - start_tod
    off = np.where(off < -43200, off + 86400, off)
    off = np.where(off > 43200, off - 86400, off)
    return off / 3600.0


def load_sleep_profile(session: SleepSession) -> Optional[dict]:
    """
    Parse the PSG Sleep Profile text file for the given session.

    Aligns epochs to the CSV recording's wall-clock time axis using the
    ``timeSM`` column (session.time_start) and the Sleep Profile's absolute
    timestamps.  Epochs that fall before or after the CSV recording are
    dropped.

    Returns
    -------
    dict with keys:
        t_ep_hr : (M,) float array — epoch start time in hours from recording start
        labels  : list[str]        — stage label strings ('REM', 'N3', …)
        codes   : (M,) int8 array  — numeric stage codes (0=REM, 1=N3, …, 4=Wake)
    Returns None if no file is found.
    """
    psg_dir = session.meta.get('psg_dir')
    if psg_dir is None:
        return None

    pattern = str(psg_dir / 'PSG_analysis_*' / 'Sleep Profile*.txt')
    matches = glob.glob(pattern)
    matches = [m for m in matches if 'reliability' not in m.lower()]
    if not matches:
        return None

    codes: List[int] = []
    epoch_tod_sec: List[float] = []
    with open(matches[0], 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for line in lines:
        m = _DATA_LINE_RE.match(line.strip())
        if not m:
            continue
        h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        label = m.group(5).strip().lower()
        code = -1
        for k, v in PSG_STAGE_MAP.items():
            if k == 'a':
                if label == 'a':
                    code = v
                    break
            elif k in label:
                code = v
                break
        codes.append(code)
        epoch_tod_sec.append(h * 3600 + mi * 60 + s + ms / 1000.0)

    if not codes:
        return None

    epoch_tod = np.array(epoch_tod_sec)

    # Wall-clock alignment: convert absolute epoch times to hours from CSV start.
    if session.time_start is not None:
        ts = session.time_start
        if hasattr(ts, 'tz') and ts.tz is not None:
            ts = ts.tz_localize(None) if hasattr(ts, 'tz_localize') else ts.replace(tzinfo=None)
        csv_start_tod = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6

        # Handle midnight crossing: if epoch time < csv_start by > 12 h,
        # the epoch is on the next calendar day.
        offset = epoch_tod - csv_start_tod
        offset[offset < -43200] += 86400  # epoch on next day
        offset[offset > 43200] -= 86400   # csv on next day

        t_ep_hr = offset / 3600.0
    else:
        t_ep_hr = np.arange(len(codes)) * PSG_EPOCH_SEC / 3600.0

    # Keep only epochs within the CSV recording window.
    session_dur_hr = float(session.time_hr[-1])
    keep = (t_ep_hr >= -PSG_EPOCH_SEC / 3600.0) & (t_ep_hr <= session_dur_hr)
    t_ep_hr = t_ep_hr[keep]
    codes = [c for c, k in zip(codes, keep) if k]

    return {
        't_ep_hr': t_ep_hr,
        'labels':  [STAGE_LABELS.get(c, '?') for c in codes],
        'codes':   np.array(codes, dtype=np.int8),
    }


# ── PSG apnea event loader ───────────────────────────────────────────────────

def _hms_to_hr(hh_mm_ss: str, ms: str) -> float:
    h, m, s = (int(x) for x in hh_mm_ss.split(':'))
    return h + m / 60.0 + (s + int(ms) / 1000.0) / 3600.0


def _parse_flow_file(path: str) -> List[dict]:
    """Parse one Flow *.txt file into a list of respiratory event dicts."""
    events = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _APNEA_LINE_RE.match(line.strip())
            if not m:
                continue
            start_hr = _hms_to_hr(m.group(1), m.group(2))
            end_hr   = _hms_to_hr(m.group(3), m.group(4))
            dur_s    = int(m.group(5))
            etype    = m.group(6).strip().lower()
            code     = APNEA_CODES.get(etype, 0)
            if code == 0:
                continue  # skip non-respiratory events (e.g. "body event")
            events.append({
                'start_hr': start_hr, 'end_hr': end_hr,
                'duration_s': dur_s, 'type': etype, 'code': code,
            })
    return events


def load_apnea_events(session: SleepSession) -> Optional[dict]:
    """
    Parse PSG Flow file for apnea/hypopnea events.

    Returns
    -------
    dict with keys:
        start_hr   : (M,) float — event start in hours from recording start
        end_hr     : (M,) float — event end in hours from recording start
        duration_s : (M,) float — event duration in seconds
        types      : list[str]  — event type (e.g. 'obstructive apnea', 'hypopnea')
        codes      : (M,) int8  — 1=Apnea, 2=Hypopnea
    Returns None if no Flow file is found.
    """
    psg_dir = session.meta.get('psg_dir')
    if psg_dir is None:
        return None

    pattern = str(psg_dir / 'PSG_analysis_*' / 'Flow*.txt')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None

    all_events: List[dict] = []
    for path in matches:
        all_events.extend(_parse_flow_file(path))

    if not all_events:
        return None

    all_events.sort(key=lambda e: e['start_hr'])

    # Merge overlapping events, keeping the higher-severity code (Apnea > Hypopnea)
    merged = [all_events[0].copy()]
    for ev in all_events[1:]:
        prev = merged[-1]
        if ev['start_hr'] <= prev['end_hr']:
            prev['end_hr'] = max(prev['end_hr'], ev['end_hr'])
            prev['duration_s'] = round(
                (prev['end_hr'] - prev['start_hr']) * 3600.0
            )
            if ev['code'] < prev['code']:  # 1 (Apnea) beats 2 (Hypopnea)
                prev['code'] = ev['code']
                prev['type'] = ev['type']
        else:
            merged.append(ev.copy())

    session_dur_hr = float(session.time_hr[-1])
    merged = [e for e in merged if e['start_hr'] <= session_dur_hr]

    return {
        'start_hr':   np.array([e['start_hr'] for e in merged], dtype=np.float64),
        'end_hr':     np.array([e['end_hr'] for e in merged], dtype=np.float64),
        'duration_s': np.array([e['duration_s'] for e in merged], dtype=np.float32),
        'types':      [e['type'] for e in merged],
        'codes':      np.array([e['code'] for e in merged], dtype=np.int8),
    }


# ── PSG arousal / body-position loaders ──────────────────────────────────────
#
# Both file families live beside the Sleep Profile in PSG_analysis_*/ and both
# stamp time-of-day, so they go through _tod_sec_to_session_hr rather than being
# read as hours-from-start.

_INTERVAL_RE = re.compile(
    r'^(\d{2}):(\d{2}):(\d{2}),(\d{3})-(\d{2}):(\d{2}):(\d{2}),(\d{3});'
    r'\s*([\d.]+);\s*(.+)$'
)


def _find_psg_file(session: SleepSession, stem: str) -> Optional[str]:
    psg_dir = session.meta.get('psg_dir')
    if psg_dir is None:
        return None
    matches = sorted(glob.glob(str(psg_dir / 'PSG_analysis_*' / f'{stem}*.txt')))
    return matches[0] if matches else None


def load_interval_events(session: SleepSession, stem: str) -> Optional[dict]:
    """
    Parse a PSG interval-event file ('hh:mm:ss,mmm-hh:mm:ss,mmm; dur; label').

    Used for 'Classification Arousal' and 'Autonomic arousals'.

    Returns dict with start_hr, end_hr, duration_s (arrays) and types (list of
    label strings), all clipped to the CSV recording window. None if no file.
    """
    path = _find_psg_file(session, stem)
    if path is None:
        return None

    starts, ends, durs, types = [], [], [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _INTERVAL_RE.match(line.strip())
            if not m:
                continue
            g = m.groups()
            starts.append(int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1e3)
            ends.append(int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1e3)
            durs.append(float(g[8]))
            types.append(g[9].strip())
    if not starts:
        return None

    s_hr = _tod_sec_to_session_hr(np.array(starts), session)
    e_hr = s_hr + np.array(durs) / 3600.0     # duration is authoritative; avoids
                                              # a midnight wrap splitting an event
    dur_hr = float(session.time_hr[-1])
    keep = (e_hr >= 0) & (s_hr <= dur_hr)
    return {
        'start_hr':   s_hr[keep],
        'end_hr':     e_hr[keep],
        'duration_s': np.array(durs)[keep],
        'types':      [t for t, k in zip(types, keep) if k],
    }


def load_arousals(session: SleepSession) -> Optional[dict]:
    """EEG-scored arousals ('Classification Arousal'): Arousal / Respiratory / LM."""
    return load_interval_events(session, 'Classification Arousal')


def load_autonomic_arousals(session: SleepSession) -> Optional[dict]:
    """Pleth-derived autonomic arousals — the non-cortical arousal channel."""
    return load_interval_events(session, 'Autonomic arousals')


def load_position(session: SleepSession) -> Optional[dict]:
    """
    Parse the PSG 'Position' file — scored body position, the independent
    reference for accelerometer-derived head angle.

    The file is a step signal: each line marks the time a new position begins.

    Returns dict with t_hr (change times, hours from CSV start) and labels
    (Supine / Left / Right / Prone / Upright), clipped to the recording.
    """
    path = _find_psg_file(session, 'Position')
    if path is None:
        return None

    tod, labels = [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _DATA_LINE_RE.match(line.strip())
            if not m:
                continue
            tod.append(int(m.group(1)) * 3600 + int(m.group(2)) * 60
                       + int(m.group(3)) + int(m.group(4)) / 1e3)
            labels.append(m.group(5).strip())
    if not tod:
        return None

    t_hr = _tod_sec_to_session_hr(np.array(tod), session)
    order = np.argsort(t_hr)
    t_hr = t_hr[order]
    labels = [labels[i] for i in order]
    dur_hr = float(session.time_hr[-1])
    keep = (t_hr <= dur_hr)
    return {
        't_hr':   t_hr[keep],
        'labels': [l for l, k in zip(labels, keep) if k],
    }


def position_at(pos: dict, t_hr: np.ndarray) -> np.ndarray:
    """Sample a step-signal position record onto arbitrary times (string array)."""
    if pos is None or len(pos['t_hr']) == 0:
        return np.full(len(t_hr), 'Unknown', dtype=object)
    idx = np.searchsorted(pos['t_hr'], np.asarray(t_hr, float), side='right') - 1
    lab = np.array(pos['labels'], dtype=object)
    out = np.where(idx >= 0, lab[np.clip(idx, 0, len(lab) - 1)], 'Unknown')
    return out.astype(object)
