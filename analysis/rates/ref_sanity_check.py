"""
Is the PSG cardiac reference right? Check it against ECG R-peaks directly.

The per-channel k work reported cardiac k near 2 on most recordings but 1.64 on
S2N1, 1.36 on S6N1 and 0.97 on S6N2, and that spread was being read as a
property of the sensor -- a participant whose pulse waveform shows one
deflection per beat rather than two.

k is count divided by reference, so an inflated reference reads as a deflated k.
This checks the reference the only way available: against R-peaks detected on
the raw ECG of the same recording. Where the two agree the reference is sound;
where they do not, k cannot be interpreted until the reference is fixed.

Writes  reports/rates/ref_sanity_check.csv

Usage
-----
    .venv/Scripts/python.exe analysis/rates/ref_sanity_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sleep_monitor.loader import load_session          # noqa: E402
from sleep_monitor.sessions import SESSION_META        # noqa: E402
from sleep_monitor.ground_truth import _ecg_rpeaks     # noqa: E402

FS = 100.0
OUT = ROOT / 'reports' / 'rates' / 'ref_sanity_check.csv'


def main():
    d = pd.read_parquet(ROOT / 'artifacts' / 'rate_rerun_phase_a.parquet')
    ref = d[(d.band == 'card') & (d.channel == 'CRE')] \
        .groupby('session').gt_hz.median() * 60
    rows = []
    for m in SESSION_META:
        lab = m['label']
        try:
            ecg = load_session(m).psg.get('ECG')
            if ecg is None:
                rows.append((lab, np.nan, float(ref[lab]), 'no ECG channel'))
                continue
            rp = _ecg_rpeaks(np.asarray(ecg, float), FS)
            bpm = 60.0 / (np.median(np.diff(rp)) / FS)
            # a detector that fails returns a nonsense rate rather than raising
            note = '' if 30 < bpm < 200 else 'ECG unusable'
            rows.append((lab, bpm, float(ref[lab]), note))
        except Exception as e:
            rows.append((lab, np.nan, float(ref[lab]), type(e).__name__))
    t = pd.DataFrame(rows, columns=['session', 'ecg_bpm',
                                    'parquet_ref_bpm', 'note'])
    t['ratio'] = (t.parquet_ref_bpm / t.ecg_bpm).round(2)
    t.loc[t.note != '', 'ratio'] = np.nan
    t.to_csv(OUT, index=False)
    print(t.round(1).to_string(index=False))
    good = t[(t.ratio > 0.9) & (t.ratio < 1.15)]
    print(f'\nreference verified on {len(good)} of {len(t)} recordings')


if __name__ == '__main__':
    main()
