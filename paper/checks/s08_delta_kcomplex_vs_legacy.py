"""Does s08_delta_kcomplex reproduce the legacy analysis/delta_onset/ outputs?

Run after the stage (it reads the stage's tables and cached features):

    .venv/Scripts/python.exe paper/checks/s08_delta_kcomplex_vs_legacy.py

Compared
    delta_onsets_summary.csv (q30 rows), delta_onsets_<rec>_q30.npz onset samples
    response_consistency_q30.csv, lowband_precursor_check_q30.csv
    precursor_summary_q30.csv  -- replayed with the legacy random-draw order
                                  (xcorr for all 12 -> q15 onsets -> q30 onsets)
    arousal_control_q30.csv
    kcomplex_marks.csv, kcomplex_criteria.csv, kcomplex_cap_response.csv
    the Fig. 12b medians printed on the legacy figure (3.6 / 3.7 / 4.3 s)

The result is written next to this file as results/s08_delta_kcomplex_vs_legacy.txt.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))

from seclib import LABELS, TAB_DIR                                   # noqa: E402
from stages import s08_delta_kcomplex as S                           # noqa: E402
from stages._s08_delta_kcomplex_signal import (AFS, KEYS, LAG_MAX_S,  # noqa: E402
                                               detect_onsets, detector_features,
                                               nrem_concat)

LEGACY = PAPER.parent / 'analysis' / 'delta_onset' / 'outputs'
NEW = TAB_DIR / S.STAGE


def compare(name, old, new, keys, tol=0.0):
    """Row-aligned comparison of every shared column; prints max |diff| per column."""
    old = old.sort_values(keys).reset_index(drop=True)
    new = new.sort_values(keys).reset_index(drop=True)
    ok = len(old) == len(new) and (old[keys].astype(str).values ==
                                  new[keys].astype(str).values).all()
    if not ok:
        print(f'[FAIL] {name}: rows differ ({len(old)} legacy vs {len(new)} port)')
        return False
    worst = []
    for c in [c for c in old.columns if c in new.columns and c not in keys]:
        a, b = old[c], new[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            d = np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))) \
                if len(a) else 0.0
            same_nan = (a.isna() == b.isna()).all()
            if d > tol or not same_nan:
                ok = False
            worst.append(f'{c}={d:.2g}')
        elif not (a.astype(str) == b.astype(str)).all():
            ok = False
            worst.append(f'{c}=MISMATCH')
    print(f'[{"OK" if ok else "FAIL"}] {name}  ({len(old)} rows, tol {tol:g})  '
          f'max|diff|: {", ".join(worst)}')
    return ok


def main():
    res = {}

    # 1. onsets
    leg = pd.read_csv(LEGACY / 'delta_onsets_summary.csv')
    leg = leg[leg.quiet_pre_s == 30]
    res['summary'] = compare('delta_onsets_summary (q30)', leg,
                             pd.read_csv(NEW / 'delta_onsets_summary.csv'), ['session'])
    ev = pd.read_csv(NEW / 'delta_onsets.csv')
    bad = []
    for lab in LABELS:
        old = np.load(LEGACY / f'delta_onsets_{lab}_q30.npz')['onset_samp'].astype(int)
        new = ev.loc[ev.session == lab, 'onset_samp'].to_numpy(int)
        if not np.array_equal(old, new):
            bad.append(lab)
    res['onset_samples'] = not bad
    print(f'[{"OK" if not bad else "FAIL"}] onset samples, 12 recordings, exact'
          + (f'  mismatched: {bad}' if bad else f'  ({len(ev)} onsets identical)'))

    # 2. causal / zero-phase peri-onset tables (float32 feature cache -> tol 1e-3 on
    #    values the legacy CSV rounds to 3 decimals)
    for f_old, f_new, keys in (
            ('response_consistency_q30.csv', 'response_consistency.csv',
             ['channel', 'band_hz']),
            ('lowband_precursor_check_q30.csv', 'lowband_precursor_check.csv',
             ['channel', 'band_hz', 'estimator'])):
        res[f_new] = compare(f_new, pd.read_csv(LEGACY / f_old).drop(columns='tag'),
                             pd.read_csv(NEW / f_new), keys, tol=1e-3)

    # 3. precursor summary, replaying the legacy draw order: legacy nrem_xcorr drew
    #    150 circular shifts per channel x band per recording before any AUC null
    rng = np.random.default_rng(0)
    xc = S.xcorr_pass(LABELS, S.load_zerophase)
    lag_max = int(LAG_MAX_S * AFS)
    for lab in LABELS:
        nrem = S.load_zerophase(lab)[2]
        d = nrem_concat(np.zeros(len(nrem)), nrem, lag_max)
        for key in KEYS:
            if xc[lab][key] is not None:
                rng.integers(lag_max, len(d) - lag_max, size=150)
    q15 = {}
    for lab in LABELS:
        F = detector_features(lab)
        on, _ = detect_onsets(F['env'], F['codes'], F['motion'], quiet_pre_s=15.0)
        q15[lab] = on['onset_samp'].to_numpy() if len(on) else np.array([], int)
        old15 = np.load(LEGACY / f'delta_onsets_{lab}_q15.npz')['onset_samp'].astype(int)
        assert np.array_equal(old15, q15[lab]), f'q15 onsets differ in {lab}'
    S.onset_pass(LABELS, S.to_20hz(q15), S.load_zerophase, xc, rng)
    q30 = {lab: ev.loc[ev.session == lab, 'onset_samp'].to_numpy(int) for lab in LABELS}
    replay = S.precursor_summary(
        S.onset_pass(LABELS, S.to_20hz(q30), S.load_zerophase, xc, rng), 'zerophase')
    leg = pd.read_csv(LEGACY / 'precursor_summary_q30.csv').drop(columns='tag')
    res['precursor_replay'] = compare('precursor_summary, legacy draw order', leg,
                                      replay.drop(columns='estimator'),
                                      ['channel', 'band_hz'])
    stage = pd.read_csv(NEW / 'precursor_summary.csv')
    stage = stage[stage.estimator == 'zerophase'].drop(columns='estimator')
    print('  (info) stage zero-phase table vs legacy, stage draw order (no q15 pass):')
    compare('precursor_summary, stage draw order', leg, stage, ['channel', 'band_hz'],
            tol=np.inf)

    # 4. arousal control
    res['arousal'] = compare('arousal_control', pd.read_csv(LEGACY / 'arousal_control_q30.csv'),
                             pd.read_csv(NEW / 'arousal_control.csv'),
                             ['session', 'subset', 'channel', 'band'], tol=1e-4)

    # 5. K-complexes
    res['kc_marks'] = compare('kcomplex_marks', pd.read_csv(LEGACY / 'kcomplex_marks.csv'),
                              pd.read_csv(NEW / 'kcomplex_marks.csv'), ['session'])
    old = pd.read_csv(LEGACY / 'kcomplex_criteria.csv')
    new = pd.read_csv(NEW / 'kcomplex_criteria.csv')
    old['i'] = old.groupby('session').cumcount()
    new['i'] = new.groupby('session').cumcount()
    res['kc_criteria'] = compare('kcomplex_criteria', old, new, ['session', 'i'], tol=1e-9)
    old = pd.read_csv(LEGACY / 'kcomplex_cap_response.csv')
    new = pd.read_csv(NEW / 'kcomplex_cap_response.csv')[old.columns]
    res['kc_response'] = compare('kcomplex_cap_response (zero-phase)', old, new,
                                 ['session'], tol=1e-9)
    lat = pd.read_csv(NEW / 'kcomplex_latencies.csv')
    meds = [round(float(np.median(lat[f'{c}_lat_s_zerophase'])), 1) for c in ('CLE', 'CRE', 'CH')]
    res['kc_medians'] = meds == [3.6, 3.7, 4.3]
    print(f'[{"OK" if res["kc_medians"] else "FAIL"}] Fig. 12b medians CLE/CRE/CH '
          f'{meds} vs legacy figure [3.6, 3.7, 4.3]')

    print(f'\n{sum(res.values())}/{len(res)} checks pass')
    return res


if __name__ == '__main__':
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()
    print(out)
    (Path(__file__).parent / 'results' / (Path(__file__).stem + '.txt')).write_text(out, encoding='utf8')
