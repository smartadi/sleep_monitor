"""
Stage-transition responses on every channel and every transition the data supports.

stage_event_locked.py reports CH and CLE-CRE only (the two signals of the
slow-trend marker) and its transitions are N3 / REM / Wake. This reruns the
identical procedure -- same windows, same stability and movement exclusions,
same matched-random-time null -- on CH, CLE, CRE and CLE-CRE, and adds the two
most frequent transitions in these hypnograms, N1 -> N2 and N2 -> N1.

Every channel is the movement-corrected (de-stepped) 10-s series.

Writes  reports/mean_value/stage_event_locked_all.csv
        reports/mean_value/stage_event_locked_all_summary.csv

Usage
    .venv/Scripts/python.exe analysis/mean_value/stage_event_locked_all.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import diff_motion_regressed as dmr          # noqa: E402
import stage_event_locked as sel             # noqa: E402
from sleep_monitor import load_session       # noqa: E402
from sleep_monitor.filters import lowpass    # noqa: E402
from sleep_monitor.sessions import SESSION_META   # noqa: E402

TAB = ROOT / 'reports' / 'mean_value'

# codes: 4 Wake, 3 N1, 2 N2, 1 N3, 0 REM
TRANSITIONS = dict(sel.TRANSITIONS)
TRANSITIONS.update({
    'N1 -> N2': lambda a, b: a == 3 and b == 2,
    'N2 -> N1': lambda a, b: a == 2 and b == 3,
})
CHANS = [('CH', 'ch_destep', '#1F618D'), ('CLE', 'cle_destep', '#27AE60'),
         ('CRE', 'cre_destep', '#8E44AD'), ('CLE-CRE', 'd_destep', '#B9380B')]


def add_single_channels(r, meta):
    """De-step CLE and CRE on their own, exactly as one_session does the difference."""
    with contextlib.redirect_stdout(io.StringIO()):
        s = load_session(meta)
    fs = s.fs
    n = int(dmr.BLOCK_S * fs)
    G = np.column_stack([dmr.blocks(lowpass(np.asarray(s.cap[a], float), dmr.GRAV_HZ, fs), n)
                         for a in ('aX', 'aY', 'aZ')])
    for ch in ('CLE', 'CRE'):
        y = dmr.blocks(np.asarray(s.cap[ch], float), n)
        y = y - np.nanmean(y)
        r[f'{ch.lower()}_destep'] = dmr.destep(y, G, dmr.STEP_THRESH, dmr.STEP_PAD)[0]
    return r


def sign_p(k, n):
    """Two-sided sign test: probability of a split at least this uneven by chance."""
    if n == 0:
        return np.nan
    m = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(m, n + 1)) / 2 ** n)


def main():
    cache = []
    for meta in SESSION_META:
        with contextlib.redirect_stdout(io.StringIO()):
            r = dmr.one_session(meta)
        cache.append(add_single_channels(r, meta))
        print('  loaded', meta['label'])

    sel.CHANS = CHANS
    cfg = dict(sel.SCALES['transitions'], events=TRANSITIONS)
    with contextlib.redirect_stdout(io.StringIO()):
        counts, df, _ = sel.run_scale('transitions', cfg, cache)
    df.to_csv(TAB / 'stage_event_locked_all.csv', index=False)

    rows = []
    for (ev, ch), g in df.groupby(['event', 'channel']):
        g = g.dropna(subset=['response_fF'])
        dn, up = int((g.response_fF < 0).sum()), int((g.response_fF > 0).sum())
        rows.append(dict(event=ev, channel=ch, n_nights=len(g),
                         n_events=int(g.n_events.sum()), fall=dn, rise=up,
                         median_fF=g.response_fF.median(),
                         sign_p=sign_p(up, dn + up)))
    sm = pd.DataFrame(rows)
    ev_n = counts.groupby('event')[['all', 'stage_held', 'movement_free']].sum()
    sm = sm.merge(ev_n, left_on='event', right_index=True)
    sm.to_csv(TAB / 'stage_event_locked_all_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(sm.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
