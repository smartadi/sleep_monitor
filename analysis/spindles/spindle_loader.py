"""
Spindle event loader — parse PSG-scored sleep-spindle annotation files and
align them to the CAP recording's time axis.

Two annotation files are exported by the PSG software per session:
  Spindle  K - *.txt      : one line per spindle, value = duration (ms)
  Spindle frequency - *.txt : one line per spindle, value = intra-spindle freq (Hz)

Line format (identical to the apnea Flow files):
    HH:MM:SS,mmm-HH:MM:SS,mmm; <value>;<label>

Events are timestamped as wall-clock time-of-day; we convert them to hours from
the CAP recording start using session.time_start, with the same midnight-crossing
handling as loader.load_sleep_profile().
"""
from __future__ import annotations
import glob
import re
from typing import Optional

import numpy as np

# start-end wall clock, numeric value (int or float), label
_SPINDLE_RE = re.compile(
    r'^(\d{2}):(\d{2}):(\d{2}),(\d{3})-(\d{2}):(\d{2}):(\d{2}),(\d{3});\s*'
    r'([0-9.]+);(.+)$'
)


def _tod_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _find_spindle_file(psg_dir, kind: str) -> Optional[str]:
    """kind = 'K' (duration) or 'frequency'."""
    stem = 'Spindle  K' if kind == 'K' else 'Spindle frequency'
    pattern = str(psg_dir / 'PSG_analysis_*' / f'{stem}*.txt')
    matches = [m for m in glob.glob(pattern) if 'MACOSX' not in m]
    return sorted(matches)[0] if matches else None


def load_spindles(session) -> Optional[dict]:
    """
    Parse the spindle annotation for one session and align to CAP time axis.

    Prefers the 'Spindle  K' (duration) file; falls back to the frequency file
    for timing when the duration file is absent (S4N2).

    Returns
    -------
    dict with keys (all np arrays, one entry per spindle within the CAP window):
        start_hr, end_hr, center_hr : hours from CAP recording start
        duration_s                  : spindle duration (s)
        freq_hz                     : intra-spindle frequency (Hz) or NaN
    Returns None if no annotation file is found.
    """
    psg_dir = session.meta.get('psg_dir')
    if psg_dir is None:
        return None

    kfile = _find_spindle_file(psg_dir, 'K')
    ffile = _find_spindle_file(psg_dir, 'frequency')
    src = kfile or ffile
    if src is None:
        return None

    # Build a freq lookup keyed by start-tod string (if freq file exists).
    freq_by_start: dict = {}
    if ffile is not None:
        with open(ffile, 'r', encoding='latin-1') as fh:
            for line in fh:
                mt = _SPINDLE_RE.match(line.strip())
                if not mt:
                    continue
                key = (mt.group(1), mt.group(2), mt.group(3), mt.group(4))
                freq_by_start[key] = float(mt.group(9))

    starts, ends, durs, freqs = [], [], [], []
    with open(src, 'r', encoding='latin-1') as fh:
        for line in fh:
            mt = _SPINDLE_RE.match(line.strip())
            if not mt:
                continue
            s_tod = _tod_sec(mt.group(1), mt.group(2), mt.group(3), mt.group(4))
            e_tod = _tod_sec(mt.group(5), mt.group(6), mt.group(7), mt.group(8))
            starts.append(s_tod)
            ends.append(e_tod)
            if src == kfile:
                durs.append(float(mt.group(9)) / 1000.0)  # ms -> s
            else:
                durs.append(e_tod - s_tod)
            key = (mt.group(1), mt.group(2), mt.group(3), mt.group(4))
            freqs.append(freq_by_start.get(key, np.nan))

    if not starts:
        return None

    starts = np.array(starts)
    ends = np.array(ends)
    durs = np.array(durs)
    freqs = np.array(freqs)

    # Wall-clock alignment to CAP start (same logic as load_sleep_profile).
    if session.time_start is None:
        return None
    ts = session.time_start
    if hasattr(ts, 'tz') and ts.tz is not None:
        ts = ts.tz_localize(None) if hasattr(ts, 'tz_localize') else ts.replace(tzinfo=None)
    csv_start_tod = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6

    def to_hr(tod):
        off = tod - csv_start_tod
        off = np.where(off < -43200, off + 86400, off)
        off = np.where(off > 43200, off - 86400, off)
        return off / 3600.0

    start_hr = to_hr(starts)
    end_hr = to_hr(ends)
    center_hr = 0.5 * (start_hr + end_hr)

    dur_hr = float(session.time_hr[-1])
    keep = (center_hr >= 0.0) & (center_hr <= dur_hr)

    return {
        'start_hr':   start_hr[keep],
        'end_hr':     end_hr[keep],
        'center_hr':  center_hr[keep],
        'duration_s': durs[keep],
        'freq_hz':    freqs[keep],
        'n_total':    int(len(starts)),
        'n_in_window': int(keep.sum()),
        'source':     'K' if src == kfile else 'frequency',
    }
