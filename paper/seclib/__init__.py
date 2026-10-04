"""
seclib — the library the paper code runs on.

A frozen copy of the parts of `sleep_monitor` the paper uses (vendored
2026-10-03), plus three helpers written for the paper:

    data      cached session loading, so twelve raw .csv.gz parses happen once
    numbers   the registry every stage reports its quoted numbers to
    figures   one figure style and one save function

Nothing here imports `sleep_monitor`; `paper/` runs on its own.
"""

from .config import (FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, STAGE_LADDER,
                     RESP_LO, RESP_HI, CARD_LO, CARD_HI, OUT_DIR, FIG_DIR, TAB_DIR,
                     NUM_DIR, CACHE_DIR, DATA_ROOT)
from .sessions import SESSION_META, SleepSession
from .data import LABELS, get_session, iter_sessions
from .numbers import Numbers

__all__ = ['FS', 'STAGE_LABELS', 'STAGE_COLORS', 'STAGE_ORDER', 'STAGE_LADDER',
           'RESP_LO', 'RESP_HI', 'CARD_LO', 'CARD_HI', 'OUT_DIR', 'FIG_DIR',
           'TAB_DIR', 'NUM_DIR', 'CACHE_DIR', 'DATA_ROOT', 'SESSION_META',
           'SleepSession', 'LABELS', 'get_session', 'iter_sessions', 'Numbers']
