"""
The scaling factor k, per night, per subject, per channel, for both methods.

The rate story in one quantity. Run a detector on a filtered SEC channel, count
what it finds, compare that with the PSG reference, and the ratio is k -- how
many things the sensor reports per real physiological cycle. k is not a
correction bolted on afterwards; it is the measurement of what the waveform
actually contains.

Two detectors, reported the same way so they can be compared directly:

    peak counting   loose peak detector on the band-passed channel
    spectral        peak of the Welch spectrum in the band

and both for respiration and for cardiac activity, on all three raw channels
(CH, CLE, CRE), for each of the twelve nights.

What the numbers say
--------------------
Peak counting gives a stable k that means something physical. Respiration lands
just above 1 (CH 1.04, CLE 1.14, CRE 1.18): roughly one detected peak per
breath, with CH closest to exactly one. Cardiac lands just under 2 on every
channel (1.93-1.96): two deflections per heartbeat, which matches the
R-peak-triggered average of 2.02 and a biphasic systolic/dicrotic waveform.

Spectral gives k near 1 for cardiac (0.94-1.30), so it is finding the
fundamental rather than a harmonic -- but its spread across nights is two to
three times wider than peak counting's, which is the same instability that
makes its epoch-level error five times larger.

For respiration the spectral k is identical on all three channels (0.963, with
identical spread). That is not a coincidence and not a result: the respiratory
spectral estimator returns the same value in almost every epoch, so its k is
just that constant divided by the reference. It is reported here because the
degeneracy is visible in exactly this figure, and a reader comparing methods
should see why that column is not usable.

Reads   artifacts/rate_rerun_phase_a.parquet   (per epoch, per channel, both
        bands, every estimator, with the PSG reference)
Writes  reports/rates/k_by_channel.csv
        writeup/figures/rate_supp/fig_k_by_channel.png

Usage
-----
    .venv/Scripts/python.exe analysis/rates/supp_k_by_channel.py
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
OUT_FIG = ROOT / 'writeup' / 'figures' / 'rate_supp'
OUT_TAB = ROOT / 'reports' / 'rates'
for p in (OUT_FIG, OUT_TAB):
    p.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
METHODS = [('peaks_loose', 'Peak counting'), ('spectral', 'Spectral peak')]
BANDS = [('resp', 'Breathing'), ('card', 'Heart rate')]
CLIP = (0.3, 5.0)          # the deployed clip on individual epoch ratios

# one colour per participant, so a reader can see night-to-night pairing.
# Ordered by age below, so the colour ramp carries the age ordering too.
SUBJ_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#d62728']

# from analysis/rates/outputs/k_vs_age_per_subject.csv
AGE = {'OS006': 25, 'OS003': 37, 'OS004': 54,
       'OS005': 55, 'OS001': 61, 'OS002': 66}

# print style shared by every supplementary rate figure (see _supp_style.py)
import _supp_style   # noqa: E402
_supp_style.apply()


def k_table() -> pd.DataFrame:
    """k per (band, method, channel, session), with its subject."""
    subj = {m['label']: m['subject'] for m in SESSION_META}
    d = pd.read_parquet(SRC)
    d = d[d.channel.isin(CHANNELS)]
    rows = []
    for meth, _ in METHODS:
        ratio = (d[meth] / d.gt_hz).clip(*CLIP)
        t = d.assign(k=ratio).dropna(subset=['k'])
        g = (t.groupby(['band', 'channel', 'session'])['k']
             .agg(k='median', n_epochs='size').reset_index())
        g['method'] = meth
        rows.append(g)
    k = pd.concat(rows, ignore_index=True)
    k['subject'] = k.session.map(subj)
    k['night'] = k.session.str[-1].astype(int)
    return k.sort_values(['band', 'method', 'channel', 'session'])


def figure(k: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(_supp_style.WIDTH_IN, 6.8), sharex=True)
    # ordered by age, so the legend reads as an age ladder
    subjects = sorted(k.subject.unique(), key=lambda s: AGE.get(s, 999))
    cmap = dict(zip(subjects, SUBJ_COLORS))

    for r, (band, band_lbl) in enumerate(BANDS):
        for c, (meth, meth_lbl) in enumerate(METHODS):
            ax = axes[r, c]
            sub = k[(k.band == band) & (k.method == meth)]
            for xi, ch in enumerate(CHANNELS):
                s = sub[sub.channel == ch]
                jit = np.linspace(-0.22, 0.22, len(s))
                ax.scatter(xi + jit, s.k, s=34,
                           c=[cmap[x] for x in s.subject],
                           edgecolor='white', linewidth=0.6, zorder=3)
                med = s.k.median()
                ax.plot([xi - 0.28, xi + 0.28], [med] * 2, color='black',
                        lw=2.0, zorder=4)
                # beside the bar, not above it: the jitter cloud spans +/-0.22
                # and a centred label lands inside it
                ax.annotate(f'{med:.2f}', (xi + 0.31, med), ha='left',
                            va='center', fontsize=9.5, zorder=5)
            ax.axhline(1.0, color='#999999', lw=1.0, ls=':', zorder=1)
            ax.set_xticks(range(len(CHANNELS)))
            ax.set_xticklabels(CHANNELS)
            ax.set_xlim(-0.55, len(CHANNELS) - 1 + 0.92)
            ax.set_ylabel('k' if c == 0 else '')   # long label clips at this type size
            # panel content (band, detector) is stated in the caption
            _supp_style.letter(ax, 'abcd'[2 * r + c], x=-0.16)
            ax.grid(axis='y', alpha=0.25)
            lo, hi = sub.k.min(), sub.k.max()
            pad = 0.18 * max(hi - lo, 0.4)
            ax.set_ylim(min(lo, 0.85) - pad, hi + pad * 1.6)

    handles = [plt.Line2D([], [], marker='o', ls='', ms=7, color=cmap[s],
                          label=f'{AGE[s]} y')
               for s in subjects]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.005), title='participant age',
               title_fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = OUT_FIG / 'fig_k_by_channel.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p.name)


def main():
    k = k_table()
    k.to_csv(OUT_TAB / 'k_by_channel.csv', index=False)
    figure(k)

    pd.set_option('display.width', 200)
    print('\nk, median across the 12 nights (IQR in brackets)\n')
    print(f"{'band':6s} {'method':13s} " +
          '  '.join(f'{c:>16s}' for c in CHANNELS))
    for band, _ in BANDS:
        for meth, _ in METHODS:
            cells = []
            for ch in CHANNELS:
                s = k[(k.band == band) & (k.method == meth)
                      & (k.channel == ch)].k
                cells.append(f'{s.median():.2f} [{s.quantile(.25):.2f}'
                             f'–{s.quantile(.75):.2f}]')
            print(f'{band:6s} {meth:13s} ' + '  '.join(f'{c:>16s}' for c in cells))
    print(f'\n-> {OUT_TAB / "k_by_channel.csv"}')


if __name__ == '__main__':
    main()
