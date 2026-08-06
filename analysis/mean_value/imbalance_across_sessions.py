"""
Report capacitance imbalance (L-R slow mean) across sessions, grouped by subject.

No sleep-quality target here — just: what is the imbalance operating point on each
night, and how consistent is it night-to-night within a subject?

imbalance_bias  = median of imbalance(t) = median(vlf_CLE-CRE) over the sleep period (a.u.)
imbalance_std   = night-to-night modulation magnitude (context whisker)

Reads reports/mean_value/imbalance_session.csv (per-session table already
produced by imbalance_vs_quality.py).

Usage:
    .venv/Scripts/python.exe analysis/mean_value/imbalance_across_sessions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'mean_value'
REPORT_DIR = ROOT / 'reports' / 'mean_value'
CSV = REPORT_DIR / 'imbalance_session.csv'


def main():
    if not CSV.exists():
        sys.exit(f'Missing {CSV}. Run imbalance_vs_quality.py first.')
    df = pd.read_csv(CSV).sort_values(['subject', 'night']).reset_index(drop=True)

    subs = sorted(df['subject'].unique())
    cmap = plt.get_cmap('tab10')
    color = {s: cmap(i) for i, s in enumerate(subs)}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color='#888', lw=1.0, ls='-', zorder=1)

    for i, sub in enumerate(subs):
        g = df[df['subject'] == sub].sort_values('night')
        xs = [i - 0.15, i + 0.15][:len(g)]
        ys = g['imbalance_bias'].to_numpy()
        # connect the two nights
        ax.plot(xs, ys, '-', color=color[sub], lw=1.5, alpha=0.6, zorder=2)
        for x, (_, r) in zip(xs, g.iterrows()):
            ax.errorbar(x, r['imbalance_bias'], yerr=r['imbalance_std'], fmt='o', ms=11,
                        color=color[sub], ecolor=color[sub], elinewidth=1.2,
                        capsize=4, alpha=0.9, zorder=3, markeredgecolor='k',
                        markeredgewidth=0.6)
            ax.annotate(f"N{int(r['night'])}", (x, r['imbalance_bias']), fontsize=7,
                        xytext=(0, 12), textcoords='offset points', ha='center')

    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels(subs)
    ax.set_xlabel('subject')
    ax.set_ylabel('imbalance_bias  =  median(CLE-CRE) over sleep  (a.u.)')
    ax.set_title('Capacitance imbalance operating point across sessions, per subject\n'
                 'point = night median;  whisker = within-night imbalance std;  '
                 'line joins the two nights',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')
    # annotate the sign convention
    ax.text(0.005, 0.985,
            'imbalance > 0  →  left-dominant\nimbalance < 0  →  right-dominant',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#ccc'))
    out = PLOT_DIR / 'imbalance_across_sessions_by_subject.png'
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)

    # ── console table ──
    print('=' * 60)
    print('Capacitance imbalance across sessions (per subject)')
    print('=' * 60)
    tbl = df[['subject', 'session', 'night', 'imbalance_bias', 'imbalance_std',
              'imbalance_range', 'n_clean_epochs']].copy()
    tbl['imbalance_bias'] = tbl['imbalance_bias'].round(1)
    tbl['imbalance_std'] = tbl['imbalance_std'].round(1)
    tbl['imbalance_range'] = tbl['imbalance_range'].round(1)
    print(tbl.to_string(index=False))

    # night-to-night consistency per subject
    print('\nNight-to-night imbalance_bias per subject:')
    for sub, g in df.groupby('subject'):
        if g['night'].nunique() < 2:
            continue
        g = g.sort_values('night')
        b = g['imbalance_bias'].to_numpy()
        same = '(same sign)' if np.sign(b[0]) == np.sign(b[1]) else '(SIGN FLIP)'
        print(f'  {sub}: N1={b[0]:+8.1f}  N2={b[1]:+8.1f}  '
              f'delta={b[1]-b[0]:+8.1f}  {same}')

    print(f'\nFigure -> {out}')


if __name__ == '__main__':
    main()
