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
    'font.size': 17, 'axes.titlesize': 19, 'axes.labelsize': 17,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15,
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

    fig.suptitle('How a rate is produced   '
                 f'({ex["label"]}, 60 s)', fontsize=21, fontweight='bold',
                 x=0.015, ha='left', y=0.985)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
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

    fig.suptitle('Does the sensor beat having no sensor?\n'
                 'Green beats the dashed line; red does not',
                 fontsize=21, fontweight='bold', x=0.015, ha='left', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    p = OUT / 'fig_rate_result.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p.name)


def fig_running_k(win_min=30.0):
    """k recomputed continuously, to show whether it could be learned live.

    The deployed k is one number per recording: the median of
    (raw count / PSG reference) over that recording's valid epochs, individual
    ratios clipped to 0.3-5.0. Running it in a trailing window instead asks the
    question a real-time device would face -- does k settle quickly, and does it
    then stay put, or does it wander enough that a value learned early is wrong
    later?

    Same definition as the deployed one, just over a trailing window.
    """
    d = pd.read_parquet(ROOT / 'artifacts' / 'rate_rerun_phase_a.parquet')
    d = d[d.channel == 'CRE']                    # the operational channel
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), sharex=True)

    for ax, (band, title, k_night) in zip(
            axes, [('resp', 'Breathing', K_RESP), ('card', 'Heart rate', K_CARD)]):
        b = d[d.band == band]
        # a shared time axis, so the recordings can be averaged without
        # depending on where each one happens to have epochs
        grid_t = np.arange(0.0, 9.0, 0.05)
        curves = []
        for sess, g in b.groupby('session'):
            g = g.sort_values('t_hr')
            ratio = (g.peaks_loose / g.gt_hz).clip(0.3, 5.0)
            ok = np.isfinite(ratio).to_numpy()
            t, r = g.t_hr.to_numpy()[ok], ratio.to_numpy()[ok]
            if len(r) < 50:
                continue
            run = (pd.Series(r, index=pd.to_timedelta(t, unit='h'))
                   .rolling(f'{int(win_min)}min').median().to_numpy())
            ax.plot(t, run, color=BLUE, lw=1.3, alpha=0.45)
            on_grid = np.interp(grid_t, t, run, left=np.nan, right=np.nan)
            curves.append(on_grid)
        if curves:
            stack = np.vstack(curves)
            n_at_t = np.sum(np.isfinite(stack), axis=0)
            with np.errstate(invalid='ignore'):
                med = np.nanmedian(stack, axis=0)
            # late in the night only one or two recordings are still running,
            # and a "median" of one recording is just that recording
            med[n_at_t < 4] = np.nan
            ax.plot(grid_t, med, color='#13305a', lw=3.6,
                    label='median of the recordings', zorder=5)
        ax.axhline(k_night, color=RED, lw=2.6, ls='--', zorder=4,
                   label=f'k used in the paper = {k_night}')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlabel('hours into the night')
        ax.set_ylabel('k  (peaks counted per real cycle)')
        ax.set_ylim(0.8, 3.0 if band == 'card' else 1.8)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right')

    fig.suptitle('Could k be learned live?   k recomputed in a '
                 f'{int(win_min)}-minute trailing window\n'
                 'One faint line per recording; the same definition the paper '
                 'uses, just rolling', fontsize=20, fontweight='bold',
                 x=0.015, ha='left', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    p = OUT / 'fig_rate_running_k.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p.name)

    # the number that decides whether live calibration is plausible
    for band, k_night in [('resp', K_RESP), ('card', K_CARD)]:
        b = d[d.band == band]
        spread = []
        for sess, g in b.groupby('session'):
            ratio = (g.sort_values('t_hr').peaks_loose / g.sort_values('t_hr').gt_hz).clip(0.3, 5.0)
            r = ratio[np.isfinite(ratio)].to_numpy()
            if len(r) < 50:
                continue
            run = pd.Series(r).rolling(int(win_min * 2)).median().dropna()
            spread.append(run.max() - run.min())
        print(f'  {band}: within-night swing of running k, median across '
              f'recordings = {np.median(spread):.2f} (whole-night k = {k_night})')


if __name__ == '__main__':
    fig_result()                       # no session load needed
    fig_running_k()
    fig_pipeline(worked_example())
