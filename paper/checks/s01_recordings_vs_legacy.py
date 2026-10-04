"""
s01_recordings against the legacy outputs it replaces.

    epochs.parquet               vs reports/mean_value/mean_value_epochs.csv
                                    (epoch, t_hr, stage_code, acc_std, motion)
    table1 epochs_rate_grid      vs reports/rates/mask/per_session_summary.csv n
                                    (max over the resp / card rows = Table 1)
                                 and reports/mean_value/head_angle_epochs.csv row counts
    motion_cancel (CLE-CRE rows) vs analysis/rates/outputs/motion_cancel_validation.csv

Writes the report next to this file as results/s01_recordings_vs_legacy.txt.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
sys.path.insert(0, str(PAPER))
from seclib import CACHE_DIR, TAB_DIR   # noqa: E402

STAGE = 's01_recordings'
lines = []


def say(msg=''):
    print(msg)
    lines.append(msg)


def compare(new, old, keys, cols, label, rtol=1e-9, atol=1e-9):
    m = new.merge(old, on=keys, how='outer', suffixes=('_new', '_old'), indicator=True)
    say(f'{label}: new {len(new)} rows, legacy {len(old)} rows, '
        f'only-new {int((m._merge == "left_only").sum())}, '
        f'only-legacy {int((m._merge == "right_only").sum())}')
    m = m[m._merge == 'both']
    for c in cols:
        a, b = m[f'{c}_new'], m[f'{c}_old']
        if a.dtype == bool or b.dtype == bool or a.dtype == object:
            bad = (a.astype(str) != b.astype(str)).sum()
            say(f'   {c:28s} mismatches {bad}')
        else:
            a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
            both_nan = np.isnan(a) & np.isnan(b)
            close = np.isclose(a, b, rtol=rtol, atol=atol) | both_nan
            d = np.nanmax(np.abs(a - b)) if (~both_nan).any() else 0.0
            say(f'   {c:28s} mismatches {int((~close).sum()):5d}   max |diff| {d:.3g}')


ep = pd.read_parquet(CACHE_DIR / STAGE / 'epochs.parquet').rename(columns={'acc_sd': 'acc_std'})
mv = pd.read_csv(REPO / 'reports/mean_value/mean_value_epochs.csv')
compare(ep, mv, ['session', 'epoch'], ['t_hr', 'stage_code', 'acc_std', 'motion'],
        'epoch grid vs mean_value_epochs.csv')

t1 = pd.read_csv(TAB_DIR / STAGE / 'table1_recordings.csv')
ps = pd.read_csv(REPO / 'reports/rates/mask/per_session_summary.csv')
leg = ps.pivot(index='session', columns='band', values='n')
leg['table1'] = leg.max(axis=1)
ha = pd.read_csv(REPO / 'reports/mean_value/head_angle_epochs.csv').groupby('session').size()
say('\nTable 1 epochs: rate grid (new) | legacy n resp, card, max | head-angle grid rows')
for _, r in t1.iterrows():
    L = leg.loc[r.session]
    say(f'   {r.session}  {r.epochs_rate_grid:5d} | {int(L.resp):5d} {int(L.card):5d} '
        f'{int(L.table1):5d} | {ha[r.session]:5d}   profile grid {r.epochs_profile_grid}')
say(f'   totals: new {t1.epochs_rate_grid.sum()}, legacy Table 1 {int(leg.table1.sum())}, '
    f'head-angle grid {ha.sum()}, profile grid {t1.epochs_profile_grid.sum()}')

cv = pd.read_csv(TAB_DIR / STAGE / 'motion_cancel_validation.csv')
old = pd.read_csv(REPO / 'analysis/rates/outputs/motion_cancel_validation.csv')
say()
compare(cv[cv.channel == 'CLE-CRE'], old, ['session', 'band'],
        ['frac_var_removed', 'corr_removed_vs_motion', 'coh_ref_before', 'coh_ref_after'],
        'canceller (CLE-CRE) vs motion_cancel_validation.csv', rtol=1e-6, atol=1e-12)

(Path(__file__).parent / 'results' / (Path(__file__).stem + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf8')
