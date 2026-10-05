"""
All twelve recordings on one sheet: what the detector reports against the
reference, every channel, both bands.

This replaces the two blocks of twelve separate panels the supplement carried
(one block for respiration, one for cardiac). Those were twenty-four images for
a comparison a reader wants to make across recordings, which a single sheet
makes possible.

A row per recording, a column per band. In each panel the PSG reference is the
black trace and the three SEC channels are the coloured ones, each converted
from its own peak count by that channel and recording's own k, so every trace
is the pipeline's actual output rather than a rescaled version of the same
thing. Both are smoothed over five 30-second epochs.

Each panel prints the median absolute difference per channel, so the reader can
compare recordings without measuring off the page.

Reads   artifacts/rate_rerun_phase_a.parquet
Writes  reports/rates/allsessions_error.csv
        writeup/figures/rate_supp/fig_rate_allsessions.png

Usage
-----
    .venv/Scripts/python.exe analysis/rates/supp_rate_allsessions.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sleep_monitor.sessions import SESSION_META      # noqa: E402

# cardiac reference ECG-first (build_ecg_reference.py); RATE_SRC=legacy for the
# original Pleth-fallback table the V3 supplement was drawn from
SRC = ROOT / 'artifacts' / ('rate_rerun_phase_a.parquet'
                            if os.environ.get('RATE_SRC') == 'legacy'
                            else 'rate_rerun_phase_a_ecgref.parquet')
FIG = ROOT / 'writeup' / 'figures' / 'rate_supp'
TAB = ROOT / 'reports' / 'rates'
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
CH_COLORS = {'CH': '#1f4e79', 'CLE': '#2e9e5b', 'CRE': '#8e44ad'}
BANDS = [('resp', 'Breathing', 'breaths/min'),
         ('card', 'Heart rate', 'beats/min')]
CLIP = (0.3, 5.0)
SMOOTH = 5

AGE = {'OS006': 25, 'OS003': 37, 'OS004': 54,
       'OS005': 55, 'OS001': 61, 'OS002': 66}

# print style shared by every supplementary rate figure (see _supp_style.py)
import _supp_style   # noqa: E402
_supp_style.apply()


def main():
    subj = {m['label']: m['subject'] for m in SESSION_META}
    d = pd.read_parquet(SRC)
    d = d[d.channel.isin(CHANNELS)].copy()
    sessions = sorted(d.session.unique())

    fig, axes = plt.subplots(len(sessions), len(BANDS),
                             figsize=(_supp_style.WIDTH_IN, 1.12 * len(sessions)),
                             sharex='col')
    rows = []
    for r, sess in enumerate(sessions):
        for c, (band, title, unit) in enumerate(BANDS):
            ax = axes[r, c]
            ref_drawn = False
            notes = []
            for ch in CHANNELS:
                b = d[(d.session == sess) & (d.channel == ch)
                      & (d.band == band)].sort_values('t_hr')
                if b.empty:
                    continue
                k = (b.peaks_loose / b.gt_hz).clip(*CLIP).median()
                est = (b.peaks_loose / k * 60.0).rolling(
                    SMOOTH, center=True, min_periods=2).median()
                ref = (b.gt_hz * 60.0).rolling(
                    SMOOTH, center=True, min_periods=2).median()
                if not ref_drawn:
                    ax.plot(b.t_hr, ref, color='#111111', lw=1.3, zorder=5,
                            label='PSG')
                    ref_drawn = True
                ax.plot(b.t_hr, est, color=CH_COLORS[ch], lw=0.7, alpha=0.85,
                        label=ch)
                err = float(np.nanmedian(np.abs(est - ref)))
                notes.append(f'{ch} {err:.2f}')
                rows.append(dict(session=sess, subject=subj[sess],
                                 age=AGE.get(subj[sess]), band=band,
                                 channel=ch, k=float(k), median_abs_err=err))
            ax.annotate('  '.join(notes), (0.012, 0.04),
                        xycoords='axes fraction', fontsize=8.5,
                        color='#222222', zorder=10,
                        bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.6))
            ax.grid(alpha=0.22)
            ax.tick_params(labelsize=10)
            if c == 0:
                ax.set_ylabel(f'{sess} ({AGE.get(subj[sess])} y)', fontsize=10.5)
            if r == 0:
                # column a = breathing (breaths/min), b = heart rate
                # (beats/min): stated in the caption
                _supp_style.letter(ax, 'ab'[c], x=-0.14, y=1.08)
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4,
                          fontsize=9.5, frameon=False, handlelength=1.4)
            if r == len(sessions) - 1:
                ax.set_xlabel('time (h)')

    fig.tight_layout(h_pad=0.4)
    p = FIG / 'fig_rate_allsessions.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)

    t = pd.DataFrame(rows)
    t.to_csv(TAB / 'allsessions_error.csv', index=False)
    print(f'wrote {p.name}')
    print('\nmedian absolute difference, median across the 12 recordings:')
    for band, title, unit in BANDS:
        s = t[t.band == band].groupby('channel')['median_abs_err'].median()
        print(f'  {title:11s} ' +
              '   '.join(f'{c} {s[c]:.2f}' for c in CHANNELS) + f'  {unit}')


if __name__ == '__main__':
    main()
