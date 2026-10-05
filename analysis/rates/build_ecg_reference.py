"""
The rate table with the cardiac reference taken from ECG wherever ECG is usable.

artifacts/rate_rerun_phase_a.parquet was built when the reference code silently
fell back from ECG to the pulse oximeter (Pleth) on every night (the fallback
was a bare `except: pass`). Checked against ECG R-peaks, Pleth agrees on eight
nights, reads 36% high on S2N1 and 29% high on S6N1, and is the only option on
S5N1 and S6N2, where the ECG channel is unusable (paper/outputs/tables/s04_rates/
ref_check.csv).

The paper pipeline (paper/stages/s04_rates.py) recomputes the per-epoch
reference ECG-first. This copies its cardiac reference into the legacy table,
row for row by (session, epoch); every other column is untouched (the
respiratory reference is the same in both, 99.8% of epochs identical).

Writes  artifacts/rate_rerun_phase_a_ecgref.parquet

Usage
    .venv/Scripts/python.exe paper/run_all.py --only s03_signal s04_rates   (if its cache is missing)
    .venv/Scripts/python.exe analysis/rates/build_ecg_reference.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEG = ROOT / 'artifacts' / 'rate_rerun_phase_a.parquet'
NEW = ROOT / 'paper' / '_cache' / 's04_rates' / 'epochs.parquet'
OUT = ROOT / 'artifacts' / 'rate_rerun_phase_a_ecgref.parquet'


def main():
    a = pd.read_parquet(LEG)
    ref = (pd.read_parquet(NEW).query("band == 'card'")
           .groupby(['session', 'epoch']).gt_hz.first().rename('ecg_ref'))
    a = a.merge(ref, left_on=['session', 'epoch'], right_index=True, how='left')
    card = a.band == 'card'
    changed = card & ~(a.gt_hz.round(9) == a.ecg_ref.round(9))
    a.loc[card, 'gt_hz'] = a.loc[card, 'ecg_ref']
    a = a.drop(columns='ecg_ref')
    a.to_parquet(OUT, index=False)
    print(f'wrote {OUT.name}: cardiac reference replaced in {int(changed.sum())} of '
          f'{int(card.sum())} cardiac rows')


if __name__ == '__main__':
    main()
