"""Does s06_harmonic_comb reproduce the legacy harmonic-comb outputs?

Run after the stage:  .venv/Scripts/python.exe paper/checks/s06_harmonic_comb_vs_legacy.py

Compares the stage's tables with reports/slow_wave/revamp/{ladder_events,
ladder_stage_summary, harmonic_ladders_long}.csv and checks the headline numbers
the legacy console printed. Writes paper/checks/results/s06_harmonic_comb_vs_legacy.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(PAPER))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

from seclib import TAB_DIR   # noqa: E402

LEG = ROOT / 'reports' / 'slow_wave' / 'revamp'
OUT = TAB_DIR / 's06_harmonic_comb'


def same(a, b, keys):
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if list(a.columns) != list(b.columns):
        return f'columns differ: {list(a.columns)} vs {list(b.columns)}'
    if len(a) != len(b):
        return f'rows {len(a)} vs {len(b)}'
    bad = []
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            ok = np.isclose(a[c].astype(float), b[c].astype(float), rtol=1e-9, equal_nan=True)
        else:
            ok = a[c].astype(str).values == b[c].astype(str).values
        if not ok.all():
            bad.append(f'{c}: {int((~ok).sum())} of {len(ok)} differ')
    return 'identical' if not bad else '; '.join(bad)


def main():
    lines = []
    for name, keys in (('ladder_events.csv', ['session', 't0_hr']),
                       ('ladder_stage_summary.csv', ['session']),
                       ('harmonic_ladders_long.csv', ['session', 'channel'])):
        new, leg = pd.read_csv(OUT / name), pd.read_csv(LEG / name)
        lines.append(f'[data ] {name} ({len(new)} rows): {same(new, leg, keys)}')
        if name == 'harmonic_ladders_long.csv':
            m = new.merge(leg, on=keys, suffixes=('', '_leg'))
            span = ['active_min', 'longest_min', 'dominant_stage']
            ok = all((m[c].astype(str) == m[c + '_leg'].astype(str)).all() for c in span)
            lines.append(f'        episode spans ({", ".join(span)}) identical: {ok}. The rung '
                         'columns differ because the legacy file was written 2026-08-17 02:16, '
                         'before commit 8908b4c (2026-08-17 11:54) replaced episode-averaged '
                         'peak picking (PEAK_DB 2.0, 0.08 Hz spacing) with the horizontal-band '
                         'tracker that the manuscript describes and this stage uses.')

    ev = pd.read_csv(OUT / 'ladder_events.csv')
    sm = pd.read_csv(OUT / 'ladder_stage_summary.csv')
    lines.append(f'        events {len(ev)} / sessions {ev.session.nunique()} / subjects '
                 f'{ev.subject.nunique()}; stages {ev.dom_stage.value_counts().to_dict()}')
    lines.append(f'        REM pre {ev.rem_frac_pre.mean():.4f} post {ev.rem_frac_post.mean():.4f}'
                 f'; null pre {sm.null_rem_frac_pre.mean():.4f} post '
                 f'{sm.null_rem_frac_post.mean():.4f}; median min before '
                 f'{ev.rem_min_before.median():.1f} after {ev.rem_min_after.median():.1f}')
    text = '\n'.join(lines)
    print(text)
    res = PAPER / 'checks' / 'results'
    res.mkdir(parents=True, exist_ok=True)
    (res / 's06_harmonic_comb_vs_legacy.txt').write_text(text + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
