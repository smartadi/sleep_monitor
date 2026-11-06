"""
Per-channel N3-vs-rest ridge summary, on subject-level directions.

§3.3 claims "the same directional trend was also observed in CLE and CH,
indicating that it was not specific to the post hoc-selected CRE channel",
but reports/slow_wave/band_ridge_stage_summary.csv only emits CRE, so the
claim has no number behind it.  The per-epoch parquet already holds all three
channels; this just surfaces them.

Subject-level direction counts are the inferential unit (pooled epochs are not
independent -- see LOGGING_POLICY / the stats review).  Medians are primary,
means are carried alongside because the two disagree for some features and the
manuscript has quoted both.

Outputs -> reports/slow_wave/ridge_stage_all_channels.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports' / 'slow_wave'

FEATURES = ['ridge_present', 'n_ridges', 'min_ridge_freq',
            'total_ridge_power', 'freq_spread', 'n_groups_active']


def directions(df, feature, agg):
    """Per-subject N3 vs rest direction, using `agg` as the within-stage statistic."""
    out = []
    for subj, g in df.groupby('subject'):
        n3 = g.loc[g['stage_label'] == 'N3', feature].dropna()
        rest = g.loc[g['stage_label'].isin(['Wake', 'N1', 'N2', 'REM']), feature].dropna()
        if len(n3) < 10 or len(rest) < 10:
            out.append((subj, np.nan, np.nan, '?'))
            continue
        a, b = agg(n3), agg(rest)
        if not np.isfinite(a) or not np.isfinite(b):
            sym = '?'
        elif np.isclose(a, b):
            sym = '='
        else:
            sym = '<' if a < b else '>'
        out.append((subj, float(a), float(b), sym))
    return out


def main():
    src = REPORTS / 'band_ridge_epochs.parquet'
    df = pd.read_parquet(src)
    print(f'{src.name}: {len(df):,} epoch rows, '
          f'bands={sorted(df["band"].unique())}, channels={sorted(df["channel"].unique())}')

    clean = df[df['motion_masked'] == 0] if 'motion_masked' in df else df
    print(f'motion-free epochs: {len(clean):,} of {len(df):,}')

    rows = []
    for (band, ch), g in clean.groupby(['band', 'channel']):
        for feat in FEATURES:
            if feat not in g:
                continue
            for label, agg in (('median', np.median), ('mean', np.mean)):
                d = directions(g, feat, agg)
                syms = [s for _, _, _, s in d]
                rows.append(dict(
                    band=band, channel=ch, feature=feat, statistic=label,
                    n3_lower=syms.count('<'), n3_higher=syms.count('>'),
                    tied=syms.count('='), undetermined=syms.count('?'),
                    directions=','.join(f'{sub}:{s}' for sub, _, _, s in d),
                ))

    out = pd.DataFrame(rows).sort_values(['band', 'feature', 'statistic', 'channel'])
    dest = REPORTS / 'ridge_stage_all_channels.csv'
    out.to_csv(dest, index=False)

    # headline: the manuscript's claim is about respiratory ridge POWER
    print('\nRespiratory ridge power, N3 vs rest, subjects with N3 lower (of 6):')
    hl = out[(out.band == 'resp') & (out.feature == 'total_ridge_power')]
    for _, r in hl.iterrows():
        print(f'  {r.channel:<4} by {r.statistic:<6} {r.n3_lower}/6 lower, '
              f'{r.n3_higher}/6 higher, {r.tied} tied   [{r.directions}]')

    print('\nRespiratory ridge COUNT (the manuscript says no consistent pattern):')
    hl = out[(out.band == 'resp') & (out.feature == 'n_ridges')]
    for _, r in hl.iterrows():
        print(f'  {r.channel:<4} by {r.statistic:<6} {r.n3_lower}/6 lower, '
              f'{r.n3_higher}/6 higher, {r.tied} tied')

    print(f'\nwrote -> {dest}')


if __name__ == '__main__':
    main()
