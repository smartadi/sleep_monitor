"""
Does the overnight drift survive de-stepping?

On the original marker the left-right difference ended the night left-dominant
in 11 of 12 recordings, having started right-dominant in most: a consistent
overnight drift. That was computed before motion was handled properly, and the
obvious worry is that the drift accumulates THROUGH the posture changes -- a
series of movement-coupled steps, each nudging the level the same way -- rather
than developing continuously between them.

De-stepping separates those two. It deletes the level change that happens while
the head is moving and leaves everything else intact, so:

    drift that survives   -> develops between movements, not through them
    drift that disappears -> was carried by the steps, i.e. by posture

Both are computed here on the same recordings so the comparison is direct, and
the trend is measured two ways: the first-third/last-third contrast used
originally, and a straight least-squares slope over the night.

Writes  reports/mean_value/overnight_drift_test.csv

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/overnight_drift_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import importlib.util
spec = importlib.util.spec_from_file_location(
    'dmr', ROOT / 'analysis' / 'mean_value' / 'diff_motion_regressed.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from sleep_monitor import load_session                       # noqa: E402
from sleep_monitor.filters import lowpass                    # noqa: E402
from sleep_monitor.config import CAP_SCALE_TO_FF             # noqa: E402
from sleep_monitor.sessions import SESSION_META              # noqa: E402

OUT = ROOT / 'reports' / 'mean_value' / 'overnight_drift_test.csv'


def thirds(t, y):
    """Median of the first and last third, and the least-squares slope."""
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 30:
        return np.nan, np.nan, np.nan
    n = len(y) // 3
    slope = float(np.polyfit(t, y, 1)[0])          # fF per hour
    return float(np.median(y[:n])), float(np.median(y[-n:])), slope


def main():
    rows = []
    for meta in SESSION_META:
        s = load_session(meta)
        n = int(m.BLOCK_S * m.FS)
        g = {ax: lowpass(np.asarray(s.cap[ax], float), m.GRAV_HZ, m.FS)
             for ax in ('aX', 'aY', 'aZ')}
        G = np.column_stack([m.blocks(g[a], n) for a in ('aX', 'aY', 'aZ')])
        cle = m.blocks(np.asarray(s.cap['CLE'], float) * CAP_SCALE_TO_FF, n)
        cre = m.blocks(np.asarray(s.cap['CRE'], float) * CAP_SCALE_TO_FF, n)
        d = cle - cre
        d = d - np.nanmean(d)
        dd, moving = m.destep(d, G, m.STEP_THRESH, m.STEP_PAD)
        t = (np.arange(len(d)) * m.BLOCK_S + m.BLOCK_S / 2) / 3600.0

        f_r, l_r, sl_r = thirds(t, d)
        f_d, l_d, sl_d = thirds(t, dd)
        rows.append(dict(session=meta['label'],
                         raw_first=f_r, raw_last=l_r, raw_slope=sl_r,
                         destep_first=f_d, destep_last=l_d, destep_slope=sl_d))
        print(f"  {meta['label']}  raw {f_r:+8.1f} -> {l_r:+8.1f} "
              f"(slope {sl_r:+7.2f})   destepped {f_d:+8.1f} -> {l_d:+8.1f} "
              f"(slope {sl_d:+7.2f})")

    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    print('\n' + '=' * 64)
    for tag in ('raw', 'destep'):
        f, l, sl = t[f'{tag}_first'], t[f'{tag}_last'], t[f'{tag}_slope']
        print(f"\n{tag.upper()}")
        print(f"  last third above the night mean : {(l > 0).sum()}/12")
        print(f"  first third above               : {(f > 0).sum()}/12")
        print(f"  sign flips first -> last        : {(np.sign(f) != np.sign(l)).sum()}/12")
        print(f"  rises across the night (last>first): {(l > f).sum()}/12")
        print(f"  positive slope                  : {(sl > 0).sum()}/12   "
              f"median {sl.median():+.2f} fF/h")


if __name__ == '__main__':
    main()
