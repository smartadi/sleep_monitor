"""
Is the calibration factor k a subject-level physiological trait that varies with age?

k is the per-session ratio (CAP peak-count estimate / PSG rate). k_resp ~ 1 (one
displacement per breath); k_card ~ 2 (biphasic pulse: systolic peak + dicrotic notch).
Hypothesis: dicrotic-notch prominence falls with arterial stiffening / age, so cardiac k
should drift toward 1 with age. Respiratory k has no obvious age mechanism (control test).

Unit of analysis = subject (n=6): age is per-subject, so k is averaged over the two
nights. Night-to-night |dk| is reported as a within-subject reliability check (a trait
must be reproducible before it can be a biomarker).

Outputs -> analysis/rates/outputs/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K_CSV = 'reports/rates/mask/per_session_summary.csv'
OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)

# Demographics from manuscript Table 1 (subject-level).
DEMO = {
    'S1': dict(subject='OS001', age=61, sex='F', psqi=9),
    'S2': dict(subject='OS002', age=66, sex='M', psqi=4),
    'S3': dict(subject='OS003', age=37, sex='M', psqi=9),
    'S4': dict(subject='OS004', age=54, sex='M', psqi=8),
    'S5': dict(subject='OS005', age=55, sex='F', psqi=6),
    'S6': dict(subject='OS006', age=25, sex='M', psqi=6),
}


def subj_of(session):   # 'S3N2' -> 'S3'
    return session.split('N')[0]


def corr_line(x, y, label):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = spearmanr(x, y)
    r, pr = pearsonr(x, y)
    return f'{label:26s} Spearman rho={rho:+.3f} p={p:.3f} | Pearson r={r:+.3f} p={pr:.3f}'


def main():
    df = pd.read_csv(K_CSV)
    df['subj'] = df['session'].map(subj_of)
    for c in ('age', 'psqi'):
        df[c] = df['subj'].map(lambda s: DEMO[s][c])
    df['sex'] = df['subj'].map(lambda s: DEMO[s]['sex'])

    # per-subject aggregation per band
    print('=== Per-subject k (mean of 2 nights) and night-to-night spread ===')
    rows = []
    for band in ('resp', 'card'):
        b = df[df.band == band]
        g = b.groupby('subj')
        for subj, sub in g:
            kk = sub['k'].values
            rows.append(dict(band=band, subj=subj, age=DEMO[subj]['age'],
                             sex=DEMO[subj]['sex'], psqi=DEMO[subj]['psqi'],
                             k_mean=kk.mean(), k_n1=kk[0], k_n2=kk[-1],
                             dk=abs(kk[0] - kk[-1])))
    ps = pd.DataFrame(rows).sort_values(['band', 'age'])
    ps.to_csv(os.path.join(OUT, 'k_vs_age_per_subject.csv'), index=False)
    for band in ('resp', 'card'):
        sub = ps[ps.band == band].sort_values('age')
        print(f'\n-- {band} --')
        print(sub[['subj', 'age', 'sex', 'psqi', 'k_mean', 'dk']]
              .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        print(f'   night-to-night |dk|: median={sub.dk.median():.3f} max={sub.dk.max():.3f}')

    # between-subject correlations (n=6)
    print('\n=== Between-subject correlations (unit = subject, n=6) ===')
    stat_rows = []
    for band in ('resp', 'card'):
        sub = ps[ps.band == band]
        print(f'\n-- {band} k --')
        for xvar in ('age', 'psqi'):
            print('  ' + corr_line(sub[xvar], sub['k_mean'], f'k vs {xvar}'))
            rho, p = spearmanr(sub[xvar], sub['k_mean'])
            stat_rows.append(dict(band=band, x=xvar, n=len(sub),
                                  spearman_rho=rho, spearman_p=p))
        # sex descriptive
        m = sub[sub.sex == 'M']['k_mean'].values
        f = sub[sub.sex == 'F']['k_mean'].values
        print(f'  k by sex: M={m.mean():.3f} (n={len(m)})  F={f.mean():.3f} (n={len(f)})')
        # cardiac: sensitivity dropping S6 coupling outlier
        if band == 'card':
            s = sub[sub.subj != 'S6']
            print('  [sensitivity, drop S6 outlier] ' +
                  corr_line(s['age'], s['k_mean'], 'k vs age (n=5)'))
    pd.DataFrame(stat_rows).to_csv(os.path.join(OUT, 'k_vs_age_stats.csv'), index=False)

    # ── figure: k vs age (resp + cardiac) ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, band, ylab in zip(axes, ('resp', 'card'),
                              ('Respiratory k', 'Cardiac k')):
        sub = ps[ps.band == band].sort_values('age')
        ages = sub['age'].values
        kmean = sub['k_mean'].values
        err = np.abs(sub[['k_n1', 'k_n2']].values - kmean[:, None]).max(1)
        colors = ['#C0392B' if s == 'F' else '#2980B9' for s in sub['sex']]
        ax.errorbar(ages, kmean, yerr=err, fmt='none', ecolor='gray',
                    elinewidth=1, capsize=3, zorder=1)
        ax.scatter(ages, kmean, c=colors, s=70, zorder=3, edgecolor='k', lw=0.5)
        for a, k, s in zip(ages, kmean, sub['subj']):
            ax.annotate(s, (a, k), textcoords='offset points', xytext=(6, 4), fontsize=8)
        # regression line + rho
        rho, p = spearmanr(ages, kmean)
        z = np.polyfit(ages, kmean, 1)
        xs = np.linspace(ages.min() - 2, ages.max() + 2, 50)
        ax.plot(xs, np.polyval(z, xs), 'k--', lw=1, alpha=0.6)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(ylab)
        ax.set_title(f'{ylab} vs age  (Spearman rho={rho:+.2f}, p={p:.2f}, n=6)')
        ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], marker='o', ls='', color='#2980B9', label='M'),
                            Line2D([], [], marker='o', ls='', color='#C0392B', label='F')],
                   fontsize=8, loc='best')
    fig.suptitle('Calibration factor k vs subject age (per-subject; bars = night-to-night range)',
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_k_vs_age.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()
