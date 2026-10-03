"""
Supplementary rate figures: the pipeline we actually run, and what it achieves.

The rate section as written described seven estimators, four calibration
strategies, Bland-Altman limits, phase-randomised surrogates and a 56-feature
gradient-boosted model. Only one of those is the pipeline in use. This script
reports that one, in two plain figures with large type, for the supplement.

The pipeline, in full
---------------------
    1  take one channel, CRE
    2  band-pass it to the band of interest
         respiratory 0.1-0.5 Hz, cardiac 0.5-3.0 Hz
    3  count peaks in each 30 s epoch with a LOOSE detector
         prominence 0.05 sigma, minimum spacing 0.4 s
    4  divide the count by k, one number per recording
    5  that is the rate

k exists because the capacitive waveform carries more than one deflection per
physiological cycle -- roughly one extra bump per breath and two per heartbeat.
It is the median of (raw count / PSG reference) over that recording's valid
epochs, with individual ratios clipped to 0.3-5.0. Measured: k = 1.18 for
respiration and 1.96 for cardiac activity.

What the figures say
--------------------
Figure 1 works the pipeline through a single epoch, so a reader can see what is
being counted and what k does.

Figure 2 is the honest result, and it is mostly a negative one. Night-average
respiratory rate beats a no-sensor predictor under every calibration, including
the two that transfer between recordings. Nothing else does: epoch-level
respiratory error is worse than no sensor at all, and cardiac error is worse
than no sensor under both transferable calibrations. The bar to beat is drawn
as a line so this cannot be missed.

Numbers come from reports/rates/rerun/heldout_table.csv, which the 2026-08-14
rerun produced; nothing is recomputed here except the worked example.

Outputs
-------
    writeup/figures/rate_supp/fig_rate_pipeline.png
    writeup/figures/rate_supp/fig_rate_result.png

The per-epoch and whole-night views live in supp_rate_epoch_k.py; an earlier
running-k figure was removed because it drew twelve independent recordings on
one time axis and took a median across them, which they do not support.

Usage
-----
    .venv/Scripts/python.exe analysis/rates/supp_rate_pipeline.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
from scipy.signal import find_peaks   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor import load_session, load_sleep_profile   # noqa: E402
from sleep_monitor.filters import bandpass                   # noqa: E402
from sleep_monitor.sessions import SESSION_META              # noqa: E402

OUT = ROOT / 'writeup' / 'figures' / 'rate_supp'
OUT.mkdir(parents=True, exist_ok=True)
HELDOUT = ROOT / 'reports' / 'rates' / 'rerun' / 'heldout_table.csv'

FS = 100.0
RESP = (0.1, 0.5)
EPOCH_S = 30.0
PROM = 0.05          # loose detector, as deployed
MIN_DIST_S = 0.4
K_RESP = 1.18        # measured per-recording median
K_CARD = 1.96

# large type throughout: these are supplementary figures meant to be read
# quickly, and the reviewer asked for bigger fonts
plt.rcParams.update({
    'font.size': 21, 'axes.titlesize': 23, 'axes.labelsize': 21,
    'xtick.labelsize': 19, 'ytick.labelsize': 19, 'legend.fontsize': 19,
    'font.weight': 'bold', 'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.dpi': 200,
})

BLUE, GREY, RED, GREEN = '#2e75b6', '#6b6b6b', '#c0392b', '#1b7a43'


def loose_peaks(x, fs=FS):
    """The deployed detector: loose prominence, short minimum spacing."""
    d = max(1, int(MIN_DIST_S * fs))
    return find_peaks(x, distance=d, prominence=PROM * np.std(x))[0]


def worked_example(session_label='S2N1', minutes_in=120.0, span_s=60.0):
    """One motion-free stretch, carried through every step of the pipeline."""
    meta = next(m for m in SESSION_META if m['label'] == session_label)
    s = load_session(meta)
    raw = np.asarray(s.cap['CRE'], dtype=float)
    i0 = int(minutes_in * 60 * FS)
    i1 = i0 + int(span_s * FS)
    seg_raw = raw[i0:i1]
    seg_bp = bandpass(raw, *RESP, FS)[i0:i1]
    pk = loose_peaks(seg_bp)
    t = np.arange(len(seg_bp)) / FS

    n_peaks = len(pk)
    dur = (pk[-1] - pk[0]) / FS if n_peaks >= 2 else np.nan
    raw_rate = (n_peaks - 1) / dur * 60 if n_peaks >= 2 else np.nan
    return dict(t=t, raw=seg_raw, bp=seg_bp, pk=pk, n=n_peaks,
                raw_rate=raw_rate, rate=raw_rate / K_RESP, label=session_label)


def fig_pipeline(ex):
    fig, ax = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1.25]})

    ax[0].plot(ex['t'], ex['raw'], color=GREY, lw=1.4)
    ax[0].set_ylabel('CRE, raw\n(fF)')
    ax[0].set_title('Step 1-2.  One channel (CRE), band-passed to the '
                    'respiratory band 0.1–0.5 Hz', loc='left')

    ax[1].plot(ex['t'], ex['bp'], color=BLUE, lw=2.2)
    ax[1].plot(ex['t'][ex['pk']], ex['bp'][ex['pk']], 'v', ms=11,
               color=RED, zorder=5)
    ax[1].axhline(0, color=GREY, lw=0.8)
    ax[1].set_ylabel('CRE, filtered\n(a.u.)')
    ax[1].set_xlabel('seconds')
    ax[1].set_title('Step 3.  Count peaks — loose detector '
                    '(prominence 0.05σ, minimum spacing 0.4 s)', loc='left')

    txt = (f"Step 4–5.   {ex['n']} peaks  →  "
           f"{ex['raw_rate']:.1f} per min  ÷  k = {K_RESP}"
           f"  →  {ex['rate']:.1f} breaths/min")
    ax[1].annotate(txt, (0.5, -0.42), xycoords='axes fraction', ha='center',
                   va='top', fontsize=19, color='#1b2a41',
                   bbox=dict(boxstyle='round,pad=0.5', fc='#eef3f9',
                             ec='#b9c8da'))

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    p = OUT / 'fig_rate_pipeline.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p.name)


def _median(cell):
    """'0.24 [0.14-0.34]' -> 0.24"""
    return float(re.match(r'\s*([-\d.]+)', str(cell)).group(1))


def fig_result():
    h = pd.read_csv(HELDOUT).set_index('band')
    strategies = [('night_self', 'Same night\n(calibrated on the\nnight being reported)'),
                  ('night_cross', 'The same person,\nother night'),
                  ('night_pop', 'Everyone else\n(population)')]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0))
    for ax, (band, title, unit) in zip(
            axes, [('resp', 'Breathing rate', 'breaths/min'),
                   ('card', 'Heart rate', 'beats/min')]):
        vals = [_median(h.loc[band, k]) for k, _ in strategies]
        base = _median(h.loc[band, 'night_nosensor'])
        cols = [GREEN if v < base else RED for v in vals]
        xs = np.arange(len(vals))
        ax.bar(xs, vals, 0.62, color=cols, edgecolor='black', lw=0.8, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f'{v:.2f}', (x, v), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=17,
                        fontweight='bold')
        ax.axhline(base, color='black', lw=2.4, ls='--', zorder=4)
        ax.annotate(f'no sensor at all: {base:.2f}', (0.99, base),
                    xycoords=('axes fraction', 'data'), xytext=(0, 8),
                    textcoords='offset points', ha='right', va='bottom',
                    fontsize=16, fontweight='bold')
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in strategies], fontsize=14)
        ax.set_ylabel(f'error in the night average\n({unit})')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylim(0, max(max(vals), base) * 1.32)
        ax.grid(axis='y', alpha=0.25, zorder=0)

    fig.tight_layout(rect=(0, 0, 1, 1))
    p = OUT / 'fig_rate_result.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p.name)


if __name__ == '__main__':
    fig_result()                       # no session load needed
    fig_pipeline(worked_example())
