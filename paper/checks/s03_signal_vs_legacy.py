"""s03_signal against the legacy outputs it replaces.

Run after `python paper/run_all.py --only s03_signal`. Compares
    snr.csv                    <- writeup/figures/signal_validation/inband_snr_summary.csv
    snr_split.csv              <- writeup/figures/signal_validation/inband_snr_split_summary.csv
    band_fractions.csv (legacy definition, CLE-CRE)
                               <- writeup/figures/signal_validation/signal_characterization_summary.csv
    coherence.csv              <- reports/rates/coupling/cap_psg_coherence.csv
    gt_quality_gate.csv        <- reports/rates/mask/gt_quality_gate.csv
    gt_cross_signal_agreement.csv <- reports/rates/mask/gt_cross_signal_agreement.csv
    consolidated_resp_gt.parquet  <- artifacts/consolidated_resp_gt.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
sys.path.insert(0, str(PAPER))
from seclib import CACHE_DIR, TAB_DIR   # noqa: E402

NEW = TAB_DIR / 's03_signal'
SV = REPO / 'writeup' / 'figures' / 'signal_validation'
results = []


def report(name, ok, detail=''):
    results.append((name, ok))
    print(f'[{"MATCH" if ok else "DIFF "}] {name}  {detail}')


def max_abs(a, b):
    return float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# SNR, pooled (legacy rounds to 2 dp; so do we)
old, new = pd.read_csv(SV / 'inband_snr_summary.csv'), pd.read_csv(NEW / 'snr.csv')
d = max_abs(old[['CH', 'CLE', 'CRE']], new[['CH', 'CLE', 'CRE']])
report('SNR pooled 0.1-3 Hz (36 values)', d == 0, f'max |diff| {d:g} dB')

# SNR split + unworn control (legacy has no 'pooled' control row)
old = pd.read_csv(SV / 'inband_snr_split_summary.csv')
new = pd.read_csv(NEW / 'snr_split.csv')
m = old.merge(new, on=['recording', 'band'], suffixes=('_o', '_n'))
d = max(max_abs(m[c + '_o'], m[c + '_n']) for c in ['CH', 'CLE', 'CRE'])
report(f'SNR split resp/card + unworn ({len(m)} rows x 3)', len(m) == len(old) and d == 0,
       f'max |diff| {d:g} dB')

# band fractions, legacy definition (CLE-CRE, 0.05-10 Hz denominator), rounded to 4 dp
old = pd.read_csv(SV / 'signal_characterization_summary.csv').set_index('session')
new = pd.read_csv(NEW / 'band_fractions.csv')
new = new[new.channel == 'CLE-CRE'].set_index('session')
dr = max_abs(old.resp_pow_frac, new.resp_frac_legacy.round(4).reindex(old.index))
dc = max_abs(old.card_pow_frac, new.card_frac_legacy.round(4).reindex(old.index))
report('band fractions, legacy definition (12 nights)', max(dr, dc) == 0,
       f'max |diff| resp {dr:g}, card {dc:g}')

# coherence
old = pd.read_csv(REPO / 'reports/rates/coupling/cap_psg_coherence.csv')
new = pd.read_csv(NEW / 'coherence.csv')
k = ['session', 'cap', 'psg', 'band']
m = old.merge(new, on=k, suffixes=('_o', '_n'))
d = max(max_abs(m.coh_mean_o, m.coh_mean_n), max_abs(m.coh_peak_o, m.coh_peak_n))
report(f'coherence ({len(m)}/{len(old)} rows)',
       len(m) == len(old) == len(new) and d < 1e-9 and (m.n_blocks_o == m.n_blocks_n).all(),
       f'max |diff| {d:.2e}')

# respiratory reference: per-epoch rates, gate, agreement
old = pd.read_parquet(REPO / 'artifacts/consolidated_resp_gt.parquet')
new = pd.read_parquet(CACHE_DIR / 's03_signal' / 'consolidated_resp_gt.parquet')
same_grid = len(old) == len(new) and np.allclose(old.t_hr, new.t_hr)
report('resp reference epoch grid', same_grid, f'{len(new)} epochs')
for c in ['rate_flow', 'rate_thorax', 'rate_abdomen', 'rate_ripsum', 'rate_consensus',
          'apnea']:
    eq = np.isclose(old[c], new[c], atol=1e-6, equal_nan=True)
    report(f'  {c}', eq.mean() > 0.995,
           f'{eq.mean() * 100:.2f}% epochs within 1e-6 Hz; max |diff| {max_abs(old[c], new[c]):.2e}')

for fname, src in (('gt_quality_gate.csv', 'reports/rates/mask/gt_quality_gate.csv'),
                   ('gt_cross_signal_agreement.csv',
                    'reports/rates/mask/gt_cross_signal_agreement.csv')):
    o, n = pd.read_csv(REPO / src), pd.read_csv(NEW / fname)
    cols = [c for c in o.columns if c.startswith('r_')]
    d = max_abs(o[cols], n[cols])
    report(f'{fname} ({len(cols)} r columns)', d < 0.005 and (o.dropped == n.dropped).all(),
           f'max |diff| {d:.3f}; dropped equal: {(o.dropped == n.dropped).all()}')

print(f'\n{sum(ok for _, ok in results)}/{len(results)} checks match')
