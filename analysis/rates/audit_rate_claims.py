#!/usr/bin/env python
"""
Audit of the manuscript's rate-detection claims (2026-08-06).

Recomputes every headline number in §4.2 / Table 3 / §4.2.1 from the cached
per-window predictions, and adds the held-out calibration tests the manuscript
never ran. Findings written up in writeup/edits/RATE_AUDIT_2026-08-06.md.

Run:  python analysis/rates/audit_rate_claims.py
"""
from __future__ import annotations
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PHASE_A = ROOT / 'artifacts' / 'mask_phase_a.parquet'
RPT = ROOT / 'reports' / 'rates' / 'mask'
OUT = ROOT / 'analysis' / 'rates' / 'outputs'

AGE = {'S1': 61, 'S2': 66, 'S3': 37, 'S4': 54, 'S5': 55, 'S6': 25}


def load(band: str, channel: str, est: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-session (raw estimate, reference) pairs in per-minute units."""
    a = pd.read_parquet(PHASE_A)
    g0 = a[(a.band == band) & (a.channel == channel)].dropna(subset=['gt_hz'])
    out = {}
    for sess, g in g0.groupby('session'):
        gt = g.gt_hz.values * 60.0
        raw = g[est].values * 60.0
        m = ~np.isnan(raw) & ~np.isnan(gt)
        out[sess] = (raw[m], gt[m])
    return out


def check_spectral_degeneracy() -> None:
    """A.1 — the respiratory spectral estimator returns a single value."""
    a = pd.read_parquet(PHASE_A)
    r = a[(a.band == 'resp') & (a.channel == 'diff')]
    vals = (r.r_spectral * 60).value_counts(dropna=False)
    frac = vals.iloc[0] / vals.sum()
    print(f'  modal output {vals.index[0]:.1f} br/min in {vals.iloc[0]}/{vals.sum()} epochs ({frac:.4%})')
    nconst = sum((g.r_spectral * 60).nunique() == 1 for _, g in r.groupby('session'))
    print(f'  sessions with zero within-session variance: {nconst}/12')

    # k_resp is identically 15 / median(reference)
    D = load('resp', 'diff', 'r_spectral')
    dev = [abs(np.median(raw / gt) - 15.0 / np.median(gt)) for raw, gt in D.values()]
    print(f'  max |k - 15/median(ref)| across sessions: {max(dev):.2e}')


def constant_baselines(band: str, channel: str, est: str, unit: str) -> pd.DataFrame:
    """A.2 — held-out calibration and no-sensor comparators."""
    D = load(band, channel, est)
    ks = {s: np.median(r / g) for s, (r, g) in D.items()}
    k_pop = np.median(list(ks.values()))
    ref_pop = np.median(np.concatenate([g for _, g in D.values()]))

    rows = []
    for s, (r, g) in D.items():
        other = s[:2] + ('N2' if s.endswith('N1') else 'N1')
        k_x = ks.get(other, np.nan)
        rows.append(dict(
            session=s,
            night_self_k=abs(np.mean(r / ks[s]) - np.mean(g)),
            night_cross_night_k=abs(np.mean(r / k_x) - np.mean(g)),
            night_population_k=abs(np.mean(r / k_pop) - np.mean(g)),
            night_no_sensor=abs(ref_pop - np.mean(g)),
            epoch_self_k=np.median(np.abs(r / ks[s] - g)),
            epoch_cross_night_k=np.median(np.abs(r / k_x - g)),
            epoch_population_k=np.median(np.abs(r / k_pop - g)),
            epoch_no_sensor=np.median(np.abs(ref_pop - g)),
            best_constant=np.median(np.abs(np.median(g) - g)),
        ))
    d = pd.DataFrame(rows)
    print(f'  k_population = {k_pop:.3f}, population median reference = {ref_pop:.2f} {unit}')
    print(d.drop(columns=['session']).median().round(3).to_string().replace('\n', '\n  '))
    return d


def k_stability() -> None:
    """B — the '<=0.04' agreement between 50-window k and whole-night k."""
    ps = pd.read_csv(RPT / 'per_session_summary.csv')
    for band, ch, est in [('resp', 'diff', 'r_spectral'), ('card', 'diff', 'r_peaks_loose')]:
        D = load(band, ch, est)
        rows = []
        for s, (r, g) in D.items():
            rep = ps[(ps.band == band) & (ps.session == s)].k.values
            rows.append(dict(session=s, k_whole=np.median(r / g),
                             k_50win=rep[0] if len(rep) else np.nan))
        d = pd.DataFrame(rows)
        d['delta'] = (d.k_whole - d.k_50win).abs()
        bad = d[d.delta > 0.04]
        print(f'  {band}: max delta {d.delta.max():.4f}, '
              f'{len(bad)}/12 exceed 0.04'
              + (f' ({", ".join(bad.session)})' if len(bad) else ''))


def k_vs_age_exact() -> None:
    """B — exact permutation p for the k-age correlation, both k aggregations."""
    per_subj = pd.read_csv(OUT / 'k_vs_age_per_subject.csv')
    resp = per_subj[per_subj.band == 'resp'].sort_values('age')
    variants = {'50-window k (as published)': resp.k_mean.values}

    D = load('resp', 'diff', 'r_spectral')
    ks = pd.Series({s: np.median(r / g) for s, (r, g) in D.items()})
    whole = ks.groupby(ks.index.str[:2]).mean()
    variants['whole-night k'] = whole.reindex(resp.subj.values).values

    ages = resp.age.values
    for name, k in variants.items():
        rho = stats.spearmanr(k, ages).statistic
        cnt = sum(1 for p in permutations(k)
                  if abs(stats.spearmanr(p, ages).statistic) >= abs(rho) - 1e-12)
        print(f'  {name:28s} rho={rho:+.3f}  exact p={cnt}/720={cnt / 720:.4f}  '
              f'(t-approx p={stats.spearmanr(k, ages).pvalue:.4f})')


def table3_provenance() -> None:
    """B — which pipeline each Table 3 cell actually came from."""
    for band, ch, est, unit in [('resp', 'diff', 'r_spectral', 'br/min'),
                                ('card', 'CRE', 'r_peaks_loose', 'BPM')]:
        D = load(band, ch, est)
        mae = np.array([np.median(np.abs(r / np.median(r / g) - g)) for r, g in D.values()])
        print(f'  {band} ({ch}/{est}, whole-night k, unsmoothed): '
              f'median {np.median(mae):.2f} IQR [{np.percentile(mae, 25):.2f}-'
              f'{np.percentile(mae, 75):.2f}] range {mae.min():.2f}-{mae.max():.2f} {unit}')
    ps = pd.read_csv(RPT / 'per_session_summary.csv')
    for band in ['resp', 'card']:
        v = ps[ps.band == band].MAE.values
        print(f'  {band} (per_session_summary.csv, 50-window k, smoothed):  '
              f'median {np.median(v):.2f} IQR [{np.percentile(v, 25):.2f}-'
              f'{np.percentile(v, 75):.2f}] range {v.min():.2f}-{v.max():.2f}')
        k = ps[ps.band == band].k.values
        print(f'  {band} k: median {np.median(k):.2f} '
              f'IQR [{np.percentile(k, 25):.2f}-{np.percentile(k, 75):.2f}] '
              f'range {k.min():.2f}-{k.max():.2f}')


if __name__ == '__main__':
    print('\n[A.1] respiratory spectral estimator degeneracy')
    check_spectral_degeneracy()

    print('\n[A.2] held-out calibration vs no-sensor baselines - CARDIAC (CRE/peaks_loose)')
    constant_baselines('card', 'CRE', 'r_peaks_loose', 'BPM')
    print('\n[A.2] held-out calibration vs no-sensor baselines - RESPIRATORY (diff/spectral)')
    constant_baselines('resp', 'diff', 'r_spectral', 'br/min')

    print('\n[B] Table 3 provenance')
    table3_provenance()

    print('\n[B] k stability claim (50-window vs whole-night, paper claims <=0.04)')
    k_stability()

    print('\n[B] k vs age, exact permutation test')
    k_vs_age_exact()
