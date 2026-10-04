"""Does s07_spindles reproduce the legacy spindle outputs?

Runs the stage's analysis in legacy mode (K-complex marks NOT filtered, as the
legacy loader did) and compares it with the files the legacy scripts wrote:

    analysis/spindles/outputs/spindle_lowband_detection.csv  (+ .npz)
    analysis/spindles/outputs/spindle_per_session.csv        (N2 spindle counts only)
    analysis/spindles/outputs/spindle_ersp.csv
    analysis/spindles/outputs/spindle_ersp_control.csv

The legacy 0.5-3 Hz 'lowc' columns are not ported (unused by the paper) and are
skipped. The result is printed and kept in results/s07_spindles_vs_legacy.txt.

    .venv/Scripts/python.exe paper/checks/s07_spindles_vs_legacy.py
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
from stages import s07_spindles as st   # noqa: E402

LEG = PAPER.parent / 'analysis' / 'spindles' / 'outputs'
TOL = 1e-9


def compare_frames(name, legacy, port, keys):
    legacy = legacy.set_index(keys)
    port = port.set_index(keys).reindex(legacy.index)
    cols = [c for c in legacy.columns if c in port.columns]
    skipped = [c for c in legacy.columns if c not in port.columns]
    worst, bad = 0.0, []
    for c in cols:
        a, b = legacy[c], port[c]
        if pd.api.types.is_numeric_dtype(a):
            d = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
            nan_mismatch = bool((a.isna() != b.isna()).any())
            worst = max(worst, d)
            if d > TOL or nan_mismatch:
                bad.append((c, d))
        elif not (a.astype(str) == b.astype(str)).all():
            bad.append((c, 'text differs'))
    verdict = 'MATCH' if not bad else 'DIFF'
    print(f'{name:34s} {verdict:5s} {len(cols)} columns x {len(legacy)} rows, '
          f'max |diff| {worst:.2e}' + (f'; skipped {len(skipped)} lowc columns'
                                        if skipped else ''))
    for c, d in bad[:20]:
        print(f'    {c}: {d}')
    return not bad


def main():
    res = st.analyse(drop_kcomplex=False)
    ok = []

    leg = pd.read_csv(LEG / 'spindle_lowband_detection.csv')
    port = pd.concat([res['per_session'].drop(columns='n_controls'), res['pooled']],
                     ignore_index=True)
    ok.append(compare_frames('spindle_lowband_detection.csv', leg, port, ['session']))

    z = np.load(LEG / 'spindle_lowband_detection.npz')
    worst = max(float(np.max(np.abs(z[k] - res['arrays'][k]))) for k in z.files)
    same_shape = all(z[k].shape == res['arrays'][k].shape for k in z.files)
    print(f'{"spindle_lowband_detection.npz":34s} '
          f'{"MATCH" if same_shape and worst <= TOL else "DIFF":5s} '
          f'{len(z.files)} arrays, max |diff| {worst:.2e}')
    ok.append(same_shape and worst <= TOL)

    leg = pd.read_csv(LEG / 'spindle_per_session.csv')
    leg = leg[leg.channel == 'EEG'][['session', 'n_spindles_N2']]
    ok.append(compare_frames('spindle_per_session.csv (counts)', leg,
                             res['per_session'][['session', 'n_spindles_N2']], ['session']))

    ok.append(compare_frames('spindle_ersp.csv', pd.read_csv(LEG / 'spindle_ersp.csv'),
                             res['ersp'], ['session', 'channel']))
    ok.append(compare_frames('spindle_ersp_control.csv',
                             pd.read_csv(LEG / 'spindle_ersp_control.csv'),
                             res['control'], ['channel', 'cond']))
    print('\nALL MATCH' if all(ok) else '\nSOME DIFF')


if __name__ == '__main__':
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()
    print(out)
    (Path(__file__).parent / 'results' / (Path(__file__).stem + '.txt')).write_text(out, encoding='utf8')
