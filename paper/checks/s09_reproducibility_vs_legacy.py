"""
s09_reproducibility against the legacy outputs it replaces, and the co-author workbook.

    high_variance_epochs_legacy_grid.parquet  vs reports/mean_value/high_variance_epochs.parquet
    prof_metrics_per_night_legacy_grid.csv    vs reports/mean_value/prof_metrics_per_night.csv
    prof_metrics_threshold_policy_legacy_grid vs reports/mean_value/prof_metrics_threshold_policy.csv
    arousal_counts.csv                        vs reports/psg/arousal_counts.csv
    legacy grid -> shared s01 grid            effect on the per-night metrics and every R/R²
    workbook writeup/review/Overnight_sleep_subject_list_V3.xlsx: the paper's R/R² recomputed
    from its own cells (graph-read values, the mis-paired column), and its values against
    the exact ones

Writes the report next to this file as results/s09_reproducibility_vs_legacy.txt.
"""

import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
sys.path.insert(0, str(PAPER))
from seclib import TAB_DIR   # noqa: E402

STAGE = 's09_reproducibility'
NEW = TAB_DIR / STAGE
LEG = REPO / 'reports'
XLSX = REPO / 'writeup' / 'review' / 'Overnight_sleep_subject_list_V3.xlsx'
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
            say(f'   {c:30s} mismatches {bad:5d}')
            continue
        a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
        nn = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=atol) | nn
        d = np.nanmax(np.abs(np.where(nn, 0, a - b))) if len(a) else 0.0
        say(f'   {c:30s} mismatches {int((~close).sum()):5d}   max |diff| {d:.3g}')


# ── 1. legacy tables ─────────────────────────────────────────────────────────
hv = pd.read_parquet(NEW / 'high_variance_epochs_legacy_grid.parquet')
compare(hv, pd.read_parquet(LEG / 'mean_value' / 'high_variance_epochs.parquet'),
        ['session', 'epoch'], 'high_variance_epochs (legacy grid)')
say()
compare(pd.read_csv(NEW / 'prof_metrics_per_night_legacy_grid.csv'),
        pd.read_csv(LEG / 'mean_value' / 'prof_metrics_per_night.csv'), ['session'],
        'prof_metrics_per_night (legacy grid)')
say()
pn = pd.read_csv(NEW / 'prof_metrics_threshold_policy_legacy_grid.csv')
po = pd.read_csv(LEG / 'mean_value' / 'prof_metrics_threshold_policy.csv')
compare(pn, po, ['policy', 'session'], 'prof_metrics_threshold_policy (legacy grid)')
say()
compare(pd.read_csv(NEW / 'arousal_counts.csv'), pd.read_csv(LEG / 'psg' / 'arousal_counts.csv'),
        ['session'], 'arousal_counts')
say()

# ── 2. legacy grid -> shared grid ────────────────────────────────────────────
say('legacy rate grid -> shared s01 grid')
a = pd.read_csv(NEW / 'prof_metrics_per_night_legacy_grid.csv').set_index('session')
b = pd.read_csv(NEW / 'prof_metrics_per_night.csv').set_index('session')
for c in ['dur_above_pct', 'n_impulses', 'impulses_per_hour', 'dur_above_pct_motionfree']:
    d = (b[c] - a[c])
    say(f'   {c:28s} max |change| {d.abs().max():.3f}  (nights changed: '
        f'{int((d.abs() > 1e-9).sum())})')
pa = pd.read_csv(NEW / 'prof_metrics_threshold_policy_legacy_grid.csv')
pb = pd.read_csv(NEW / 'prof_metrics_threshold_policy.csv')
d = (pb.set_index(['policy', 'session']).night_median_fF2
     - pa.set_index(['policy', 'session']).night_median_fF2)
say(f'   {"night_median_fF2":28s} max |change| {d.abs().max():.4f}')
ra = pd.read_csv(NEW / 'reproducibility_legacy_grid.csv').set_index('feature')
rb = pd.read_csv(NEW / 'reproducibility.csv').set_index('feature')
for f in ra.index:
    say(f'   R² {f:25s} legacy grid {ra.loc[f, "r2"]:.4f}  shared grid {rb.loc[f, "r2"]:.4f}')
aa = pd.read_csv(NEW / 'associations_legacy_grid.csv').set_index('id')
ab = pd.read_csv(NEW / 'associations.csv').set_index('id')
for i in ab.index:
    say(f'   R  {i:25s} legacy grid {aa.loc[i, "r"]:+.4f}  shared grid {ab.loc[i, "r"]:+.4f}'
        f'  n={ab.loc[i, "n"]}')
say()

# ── 3. the workbook ──────────────────────────────────────────────────────────
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def read_xlsx(path):
    """{sheet: {cell: value}} using only the standard library (no openpyxl here)."""
    z = zipfile.ZipFile(path)
    ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
            ss.append(''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])))
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    rels = {x.get('Id'): x.get('Target')
            for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    out = {}
    for s in wb.find('m:sheets', NS):
        tgt = rels[s.get('{%s}id' % NS['r'])].lstrip('/').replace('xl/', '')
        cells = {}
        for c in ET.fromstring(z.read('xl/' + tgt)).iter('{%s}c' % NS['m']):
            v = c.find('m:v', NS)
            if v is None:
                continue
            val = ss[int(v.text)] if c.get('t') == 's' else v.text
            try:
                val = float(val)
            except (TypeError, ValueError):
                pass
            cells[c.get('r')] = val
        out[s.get('name')] = cells
    return out


def col(cells, letter, r0, r1):
    return [cells.get(f'{letter}{r}') for r in range(r0, r1 + 1)]


r = lambda x, y: float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])

if not XLSX.exists():
    say(f'workbook not found: {XLSX}')
else:
    wb = read_xlsx(XLSX)
    rep = wb['reproducibility']
    sess = col(rep, 'D', 3, 14)
    say('workbook "reproducibility" (values "Estimated from graph") -> R² night 1 vs 2')
    exact = pd.read_csv(LEG / 'mean_value' / 'prof_metrics_threshold_policy.csv')
    for letter, name, pol in [('F', 'Absolute 10 fF² (% night)', 'absolute'),
                              ('H', '10x night median (% night)', '10x'),
                              ('E', 'Median variance', None)]:
        v = pd.Series(col(rep, letter, 3, 14), index=sess)
        n1, n2 = v[[s for s in sess if s.endswith('1')]], v[[s for s in sess if s.endswith('2')]]
        ex = exact[exact.policy.str.startswith(pol or 'absolute')].set_index('session')
        ex = ex.night_median_fF2 if pol is None else ex.dur_above_pct
        say(f'   {name:30s} workbook R² {r(n1, n2) ** 2:.4f}   exact R² '
            f'{r(ex[n1.index], ex[n2.index]) ** 2:.4f}   max |workbook - exact| '
            f'{np.max(np.abs(v - ex[v.index])):.3f}')
    say()

    say('workbook "Cortical and SWS": column I ("SWS time/TST") vs J ("area/hour")')
    cs = wb['Cortical and SWS']
    I, J, H = col(cs, 'I', 3, 14), col(cs, 'J', 3, 14), col(cs, 'H', 3, 14)
    calc = wb['SWS % calculation']
    z = dict(zip(col(calc, 'A', 2, 13), col(calc, 'Z', 2, 13)))
    say(f'   SWS % calculation!Z (session order): {[round(z[k], 2) for k in sorted(z)]}')
    say(f'   Cortical and SWS!I (as listed)      : {[round(x, 2) for x in I]}')
    say(f'   Cortical and SWS!H PSQI (as listed) : {[int(x) for x in H]}')
    say(f'   Cortical and SWS!J area (as listed) : {J}')
    say(f'   R(I, J) as listed = {r(I, J):+.4f}   (paper 0.68; workbook G34 = sqrt(0.4608) '
        f'= {wb["SWS %"].get("G34", float("nan")):.4f})')
    say(f'   R(Z, J) correctly paired by session = {r([z[k] for k in sorted(z)], J):+.4f}')
    say('   -> column I is night 1 of S1..S6 then night 2 of S1..S6; J is session order.')
    s2 = wb['SWS %']
    avg = col(s2, 'B', 24, 29)
    ps = col(s2, 'C', 24, 29)
    say(f'   SWS %!B24:B29 two-night mean vs PSQI: R = {r(avg, ps):+.4f}  (n = 6; paper -0.68)')
    say()

    say('workbook "EEG arousal": the §3.8 R values from its own cells (session order)')
    ea = wb['EEG arousal']
    E = {k: col(ea, k, 4, 15) for k in 'BCDHIJKLOQRST'}
    for what, x, y in [('impulses vs total index', 'O', 'H'), ('impulses vs resp', 'O', 'J'),
                       ('impulses vs spont', 'O', 'I'), ('impulses vs limb', 'O', 'K'),
                       ('impulses vs PSQI', 'O', 'D'), ('abs10 % vs PSQI', 'R', 'D'),
                       ('median var vs PSQI', 'Q', 'D'), ('area/hour vs age', 'T', 'C'),
                       ('"SWS time/TST" (col S) vs area/hour', 'S', 'T')]:
        rr = r(E[x], E[y])
        say(f'   {what:38s} R = {rr:+.4f}  R² = {rr * rr:.4f}  (n = 12)')
    say(f'   EEG arousal!S equals Cortical and SWS!I (mis-paired): '
        f'{np.allclose(E["S"], I)}')

(Path(__file__).parent / 'results' / (Path(__file__).stem + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf8')
