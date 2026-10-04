"""
s02_overnight against the legacy outputs it replaces (reports/mean_value/).

    imbalance_epochs.parquet   vs imbalance_epochs.csv          (every column)
    session_levels.csv         vs mean_centred_session_means.csv
    imbalance_session.csv      vs imbalance_session.csv         (columns kept)
    imbalance_burden.csv       vs imbalance_burden.csv
    ch_vs_clecre_sessions.csv  vs ch_vs_clecre_sessions.csv     (columns kept)

Writes the report next to this file as results/s02_overnight_vs_legacy.txt.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
sys.path.insert(0, str(PAPER))
from seclib import CACHE_DIR   # noqa: E402

STAGE = 's02_overnight'
LEG = REPO / 'reports' / 'mean_value'
lines = []


def say(msg=''):
    print(msg)
    lines.append(msg)


def compare(new, old, keys, label, rtol=1e-7, atol=1e-7):
    cols = [c for c in new.columns if c in old.columns and c not in keys]
    missing = [c for c in old.columns if c not in new.columns]
    m = new.merge(old, on=keys, how='outer', suffixes=('_new', '_old'), indicator=True)
    say(f'{label}: new {len(new)} rows, legacy {len(old)}, only-new '
        f'{int((m._merge == "left_only").sum())}, only-legacy '
        f'{int((m._merge == "right_only").sum())}; legacy columns not ported: {missing or "none"}')
    m = m[m._merge == 'both']
    for c in cols:
        a, b = m[f'{c}_new'], m[f'{c}_old']
        if a.dtype == object or b.dtype == object or a.dtype == bool or b.dtype == bool:
            bad = int((a.astype(str) != b.astype(str)).sum())
            say(f'   {c:26s} mismatches {bad:5d}')
            continue
        a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
        nn = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=atol) | nn
        d = np.nanmax(np.abs(np.where(nn, 0, a - b)))
        say(f'   {c:26s} mismatches {int((~close).sum()):5d}   max |diff| {d:.3g}')


ep = pd.read_parquet(CACHE_DIR / STAGE / 'imbalance_epochs.parquet')
old = pd.read_csv(LEG / 'imbalance_epochs.csv')
ep['k'] = ep.groupby('session').cumcount()
old['k'] = old.groupby('session').cumcount()
compare(ep, old, ['session', 'k'], 'imbalance_epochs')
say()
for new, leg in [('session_levels', 'mean_centred_session_means'),
                 ('imbalance_session', 'imbalance_session'),
                 ('imbalance_burden', 'imbalance_burden'),
                 ('ch_vs_clecre_sessions', 'ch_vs_clecre_sessions')]:
    compare(pd.read_csv(CACHE_DIR / STAGE / f'{new}.csv'), pd.read_csv(LEG / f'{leg}.csv'),
            ['session'], f'{new} vs {leg}.csv')
    say()

(Path(__file__).parent / 'results' / (Path(__file__).stem + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf8')
