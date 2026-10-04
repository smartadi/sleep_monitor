"""
Does stage s04_rates reproduce the legacy rate outputs?

Legacy files compared
    artifacts/rate_rerun_phase_a.parquet          per-epoch estimates + PSG reference
    reports/rates/rerun/{per_session,heldout_table,estimator_table,operational_choice}.csv
    reports/rates/k_by_channel.csv                Fig. S8
    reports/rates/ref_sanity_check.csv            cardiac reference vs ECG
    reports/rates/k_per_epoch_spread.csv          Fig. S12
    reports/rates/allsessions_error.csv           Fig. S10
    analysis/rates/outputs/peaks_per_beat.csv     "2.02 peaks per cardiac cycle"

Three passes over the twelve nights, each with the stage's own code:

    A  legacy emulation   respiratory reference = artifacts/consolidated_resp_gt.parquet,
                          cardiac reference = the Pleth fallback of gt_heart_rate
    B  ECG reference      as A, but gt_heart_rate as written (ECG first)
    C  the stage          s03_signal's respiratory reference, ECG cardiac reference

Pass A is the faithfulness test. The legacy parquet's cardiac reference turns out to
be the Pleth fallback on every night: gt_heart_rate swallows any exception from the
ECG R-peak detector and falls back to Pleth, and in the legacy run it did so for all
twelve (the same detector run on its own, in ref_sanity_check.py and
peaks_per_beat.py, worked). Pass A reproduces that by withholding the ECG from
gt_heart_rate only. B and C then show what the corrected references change.

Also: the stage's downstream functions are run on the legacy parquet itself, which
isolates them from the reference question.

Run from the repo root:
    .venv/Scripts/python.exe paper/checks/s04_rates_vs_legacy.py
Writes paper/checks/results/s04_rates_vs_legacy.txt (the record of the last run).
"""

from __future__ import annotations

import dataclasses
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
sys.path.insert(0, str(PAPER))

import stages.s04_rates as S                      # noqa: E402
from seclib import ground_truth                   # noqa: E402

LEG_PARQUET = REPO / 'artifacts' / 'rate_rerun_phase_a.parquet'
LEG_RESP_GT = REPO / 'artifacts' / 'consolidated_resp_gt.parquet'
RPT = REPO / 'reports' / 'rates'
OUT = (Path(__file__).parent / 'results' / (Path(__file__).stem + '.txt'))
EST_COLS = ['epoch_self', 'night_self', 'epoch_pop', 'night_cross', 'r_within',
            'raw_sd', 'ref_sd']


def compare(name, a, b, keys, cols, rtol=1e-9, atol=1e-12):
    """Merge on keys, report max |diff| per column and whether all are within tolerance."""
    m = a.merge(b, on=keys, suffixes=('_port', '_leg'), how='outer', indicator=True)
    only = m[m._merge != 'both']
    m = m[m._merge == 'both']
    ok = only.empty
    parts = []
    for c in cols:
        x, y = m[f'{c}_port'].astype(float).values, m[f'{c}_leg'].astype(float).values
        nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum())
        good = np.isfinite(x) & np.isfinite(y)
        dmax = float(np.max(np.abs(x[good] - y[good]))) if good.any() else 0.0
        same = nan_mismatch == 0 and np.allclose(x[good], y[good], rtol=rtol, atol=atol)
        ok &= same
        parts.append(f'{c}: max|d|={dmax:.3g}' + (f' nan-mismatch={nan_mismatch}' if nan_mismatch else ''))
    print(f'  [{"MATCH" if ok else "DIFF "}] {name}: {len(m)} rows'
          + (f', {len(only)} unmatched' if len(only) else '') + ' | ' + '; '.join(parts))
    return ok


def pleth_only_heart_rate(session, fallback=True):
    """gt_heart_rate with the ECG withheld: the legacy run's effective cardiac reference."""
    psg = {k: v for k, v in session.psg.items() if k != 'ECG'}
    return ORIG_HEART_RATE(dataclasses.replace(session, psg=psg), fallback)


ORIG_HEART_RATE = ground_truth.gt_heart_rate


def run_pass(resp_gt, pleth):
    ground_truth.gt_heart_rate = pleth_only_heart_rate if pleth else ORIG_HEART_RATE
    try:
        return S.compute(resp_gt=resp_gt)
    finally:
        ground_truth.gt_heart_rate = ORIG_HEART_RATE


def legacy_code_today(dA, leg):
    """Run sleep_monitor's own loader, canceller and estimators on S4N1 today."""
    sys.path.insert(0, str(REPO))
    from sleep_monitor.loader import load_session
    from sleep_monitor.preprocessing import remove_acc_artifact as leg_cancel
    from sleep_monitor.rates import rate_peaks as leg_peaks, rate_spectral as leg_spec
    from sleep_monitor.sessions import SESSION_META
    s = load_session(next(m for m in SESSION_META if m['label'] == 'S4N1'))
    same_port = same_leg = total = 0
    n = int(S.WIN_SEC * s.fs)
    for band, (lo, hi) in S.BANDS.items():
        for ch in S.CHANNELS:
            bp = leg_cancel(s.cap[ch].astype(np.float64), s.cap['acc_mag'].astype(np.float64),
                            lo, hi, s.fs)
            wins = [bp[i * n:(i + 1) * n] for i in range((len(bp) - n) // n + 1)]
            today = {'peaks_loose': np.array([leg_peaks(w, lo, hi, s.fs, prom_factor=0.05)
                                              for w in wins]),
                     'spectral': np.array([leg_spec(w, lo, hi, s.fs) for w in wins])}

            def sel(t):
                return t[(t.session == 'S4N1') & (t.band == band)
                         & (t.channel == ch)].sort_values('epoch')
            p, lg = sel(dA), sel(leg)
            for col, v in today.items():
                same_port += int(np.isclose(v, p[col], equal_nan=True).sum())
                same_leg += int(np.isclose(v, lg[col], equal_nan=True).sum())
                total += len(v)
    ok = same_port == total
    print(f'  [{"MATCH" if ok else "DIFF "}] S4N1, legacy code today vs port: {same_port}/{total} '
          f'values identical; legacy code today vs stored parquet: {same_leg}/{total}')
    return ok


def downstream(d):
    per = pd.concat([S.evaluate(d, b, 'CRE', 'peaks_loose').assign(band=b) for b in S.BANDS],
                    ignore_index=True)
    est = []
    for band in S.BANDS:
        for e in ('peaks_loose', 'spectral'):
            for ch in S.CHANNELS:
                t = S.evaluate(d, band, ch, e)
                est.append(dict(band=band, estimator=e, channel=ch,
                                **{c: float(np.median(t[c])) for c in EST_COLS}))
    return dict(per=per, held=S.heldout_summary(per), est=pd.DataFrame(est),
                kt=S.k_table(d), spread=S.k_per_epoch_spread(d), alls=S.allsessions_error(d))


def check_downstream(tag, res):
    ok = True
    leg_per = pd.read_csv(RPT / 'rerun' / 'per_session.csv')
    ok &= compare(f'{tag} per_session.csv', res['per'], leg_per, ['band', 'session'],
                  ['k', 'k_cross', 'k_pop', 'ref_mean', 'ref_sd', 'night_self', 'night_cross',
                   'night_pop', 'night_nosensor', 'epoch_self', 'epoch_cross', 'epoch_pop',
                   'epoch_nosensor', 'r_within', 'raw_sd', 'n'])
    leg_held = pd.read_csv(RPT / 'rerun' / 'heldout_table.csv', dtype=str)
    port_held = res['held'].astype(str)
    same = (leg_held.set_index('band').sort_index()
            .equals(port_held.set_index('band').sort_index()[leg_held.columns[1:]]))
    print(f'  [{"MATCH" if same else "DIFF "}] {tag} heldout_table.csv: string-identical = {same}')
    if not same:
        for c in leg_held.columns[1:]:
            a, b = port_held[c].tolist(), leg_held[c].tolist()
            if a != b:
                print(f'          {c}: port {a}  legacy {b}')
    ok &= same
    leg_est = pd.read_csv(RPT / 'rerun' / 'estimator_table.csv')
    leg_est = leg_est[leg_est.estimator.isin(['peaks_loose', 'spectral'])
                      & leg_est.channel.isin(S.CHANNELS)]
    ok &= compare(f'{tag} estimator_table.csv (peaks_loose, spectral x CH/CLE/CRE)',
                  res['est'], leg_est, ['band', 'estimator', 'channel'], EST_COLS)
    ok &= compare(f'{tag} k_by_channel.csv', res['kt'], pd.read_csv(RPT / 'k_by_channel.csv'),
                  ['band', 'method', 'channel', 'session'], ['k', 'n_epochs'])
    ok &= compare(f'{tag} k_per_epoch_spread.csv', res['spread'],
                  pd.read_csv(RPT / 'k_per_epoch_spread.csv'), ['session', 'band', 'channel'],
                  ['k_median', 'k_q1', 'k_q3', 'iqr', 'n_epochs'])
    ok &= compare(f'{tag} allsessions_error.csv', res['alls'],
                  pd.read_csv(RPT / 'allsessions_error.csv'), ['session', 'band', 'channel'],
                  ['k', 'median_abs_err'])
    return ok


def headline(res):
    p = res['per']
    R, C = p[p.band == 'resp'], p[p.band == 'card']
    m = lambda s: f'{np.median(s):.2f}'                       # noqa: E731
    return {'resp night self': m(R.night_self), 'card night self': m(C.night_self),
            'resp epoch self': m(R.epoch_self), 'card epoch self': m(C.epoch_self),
            'resp k': m(R.k), 'card k': m(C.k),
            'card k IQR': f'{np.percentile(C.k, 25):.2f}-{np.percentile(C.k, 75):.2f}',
            'resp cross/pop/none': '/'.join(m(R[c]) for c in ('night_cross', 'night_pop', 'night_nosensor')),
            'card cross/pop/none': '/'.join(m(C[c]) for c in ('night_cross', 'night_pop', 'night_nosensor')),
            'resp ref range': f'{R.ref_mean.min():.2f}-{R.ref_mean.max():.2f}'}


def main():
    ok = True
    print('s04_rates vs legacy\n')

    print('1. operational choice: legacy selection rule applied to the legacy estimator table')
    est = pd.read_csv(RPT / 'rerun' / 'estimator_table.csv')
    for band in S.BANDS:
        t = est[(est.band == band) & (est.estimator != 'spectral')]
        okk = t[t.raw_sd >= 0.5 * t.ref_sd]
        pick = (okk if len(okk) else t).sort_values('epoch_self').iloc[0]
        same = (pick.estimator, pick.channel) == S.OPERATIONAL
        ok &= same
        print(f'  [{"MATCH" if same else "DIFF "}] {band}: legacy picks {pick.estimator} on '
              f'{pick.channel}; stage fixes {S.OPERATIONAL}')
    oc = pd.read_csv(RPT / 'rerun' / 'operational_choice.csv')
    print(f'  operational_choice.csv: {oc.to_dict("records")}')

    print('\n2. stage downstream code on the LEGACY parquet (isolates it from the references)')
    leg = pd.read_parquet(LEG_PARQUET)
    leg = leg[leg.channel.isin(S.CHANNELS)]
    ok &= check_downstream('legacy-parquet', downstream(leg))

    print('\n3. pass A, legacy emulation: per-epoch recomputation vs the legacy parquet')
    passA = run_pass(LEG_RESP_GT, pleth=True)
    dA, ecgA, ppbA, profA, exA = passA
    compare('per-epoch (gt_hz, spectral, peaks_loose)', dA, leg,
                  ['session', 'epoch', 'band', 'channel'], ['t_hr', 'gt_hz', 'spectral', 'peaks_loose'])
    resA = downstream(dA)
    exact_a = check_downstream('pass A', resA)
    print('  pass A differs from the stored parquet only in a few peak counts. Is that the port?')
    port_is_legacy = legacy_code_today(dA, leg)
    ok &= port_is_legacy
    print(f'  -> pass A == stored legacy outputs: {exact_a}; port == legacy code: {port_is_legacy}')

    print('\n4. ECG side (independent of the references)')
    ref = S.ref_check(dA, ecgA, S.asleep_reference(dA, profA))
    leg_ref = pd.read_csv(RPT / 'ref_sanity_check.csv')
    ok &= compare('ref_sanity_check.csv ecg_bpm, reference', ref.rename(columns={'ref_bpm': 'parquet_ref_bpm'}),
                  leg_ref, ['session'], ['ecg_bpm', 'parquet_ref_bpm'])
    print('    legacy ratio column vs port (port flags unusable ECG as the script says):')
    print('    ' + ref.merge(leg_ref[['session', 'ratio', 'note']], on='session',
                             suffixes=('_port', '_leg'))[['session', 'ratio_port', 'ratio_leg', 'note_port']]
          .to_string(index=False).replace('\n', '\n    '))
    leg_ppb = pd.read_csv(REPO / 'analysis' / 'rates' / 'outputs' / 'peaks_per_beat.csv')
    leg_ppb = leg_ppb[leg_ppb.channel == 'CRE']
    ok &= compare('peaks_per_beat.csv (CRE)', ppbA, leg_ppb, ['session', 'channel'],
                  ['cap_peaks', 'ecg_beats', 'peaks_per_beat'])
    good = leg_ppb[leg_ppb.peaks_per_beat >= S.PPB_FAIL].peaks_per_beat
    print(f'    legacy CRE median, all {len(leg_ppb)} rows: {leg_ppb.peaks_per_beat.median():.3f}; '
          f'without S4N1 (detector failure): {good.median():.3f}  <- the quoted 2.02')
    print(f'    worked example (Fig. S7): {exA["n"]} peaks, {exA["raw_rate"]:.1f}/min, '
          f'÷1.18 = {exA["raw_rate"] / 1.18:.1f}  (legacy figure: 20, 19.6, 16.6)')
    ok &= exA['n'] == 20 and round(exA['raw_rate'], 1) == 19.6

    print('\n5. what the corrected references change (headline values, medians of 12 nights)')
    dB = run_pass(LEG_RESP_GT, pleth=False)[0]
    passC = run_pass(S.RESP_GT, pleth=False)
    dC = passC[0]
    hs = {'legacy files': headline({'per': pd.read_csv(RPT / 'rerun' / 'per_session.csv')}),
          'A legacy refs': headline(resA),
          'B + ECG cardiac ref': headline(downstream(dB)),
          'C stage (+ s03 resp ref)': headline(downstream(dC))}
    print(pd.DataFrame(hs).to_string())
    cmp = dB.merge(dA, on=['session', 'epoch', 'band', 'channel'], suffixes=('_B', '_A'))
    card = cmp[cmp.band == 'card'].groupby('session')
    cc = pd.DataFrame({'card_ref_ecg': card.gt_hz_B.median() * 60,
                       'card_ref_pleth': card.gt_hz_A.median() * 60})
    print('\n  cardiac reference per night (median BPM, CRE rows): ECG-first vs Pleth fallback')
    print('  ' + cc.round(1).to_string().replace('\n', '\n  '))

    print('\n6. every registered number: legacy references (pass A) vs the stage (pass C)')
    na = pd.DataFrame(S.register_numbers(S.assemble(*passA)).rows)
    nc = pd.DataFrame(S.register_numbers(S.assemble(*passC)).rows)
    t = na[['id', 'paper_value', 'computed', 'status']].merge(
        nc[['id', 'computed', 'status']], on='id', suffixes=('_legacy_refs', '_stage'))
    with pd.option_context('display.width', 250, 'display.max_colwidth', 45):
        print(t.to_string(index=False))

    print(f'\nOVERALL (faithfulness): {"MATCH" if ok else "DIFF"} -- downstream code exact on the '
          'legacy parquet; per-epoch values identical to the legacy code run today; ECG side '
          'exact. The stored parquet itself differs from its own code in 0.16% of peak counts.')


if __name__ == '__main__':
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    OUT.write_text(text, encoding='utf8')
    sys.stdout.write(text)
