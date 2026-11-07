"""Figures for s08_delta_kcomplex: Fig. 7 (delta-burst onsets) and Fig. 12 (K-complexes).

Both are drawn at their printed size (SLEEP two-column width) in the paper style,
`seclib.style`. The callers save them with `style.save`.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator

from seclib import style
from seclib.config import CAP_COLORS

from stages._s08_delta_kcomplex_signal import BANDS, CHANNELS

BAND_COLORS = {'0-0.5': '#1ABC9C', '0.5-1': '#8E44AD', '1-3': '#E67E22'}
# One look per role, in both figures: the EEG event at time 0, the random-NREM null,
# and the EEG delta envelope (positive control, Fig. 12a).
EVENT_STYLE = dict(color='0.15', lw=0.75, zorder=1.5)
NULL_STYLE = dict(color='0.3', lw=0.9, ls=(0, (1, 1.2)))
EEG_COLOR = '0.55'
TIME_TICK_S, TIME_LABEL_S = 5, 15        # tick every 5 s, label every 15 s


def _num(v):
    """A tick value as printed: shortest form, true minus sign, no '−0'."""
    return format(0.0 if abs(v) < 1e-12 else v, 'g').replace('-', '−')


def _step(lo, hi, nbins):
    t = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return t[1] - t[0]


def _ticks(lo, hi, nbins=5):
    """Nice ticks from the last one at or below `lo` to the first at or above `hi`
    (a sliver under 1 % of a step is let go), so each axis line can end on its end
    ticks with the data inside."""
    step = _step(lo, hi, nbins)
    start = np.floor(lo / step + 0.01) * step
    stop = np.ceil(hi / step - 0.01) * step
    return np.round(np.arange(start, stop + step / 2, step), 10)


def _set_ticks(ax, which, ticks, sparse=False):
    """Ticks, labels and limits on the end ticks. `sparse` labels only the two ends
    and zero; the other ticks stay as unlabeled marks."""
    labels = [_num(t) if not sparse or i in (0, len(ticks) - 1) or abs(t) < 1e-12 else ''
              for i, t in enumerate(ticks)]
    (ax.set_xticks if which == 'x' else ax.set_yticks)(ticks, labels)
    (ax.set_xlim if which == 'x' else ax.set_ylim)(ticks[0], ticks[-1])


def _time_axis(ax, t0, t1):
    """Time axis from t0 to t1 (s): a tick every TIME_TICK_S, labels every TIME_LABEL_S."""
    s = TIME_TICK_S
    ticks = np.arange(np.floor(t0 / s + 0.01) * s, np.ceil(t1 / s - 0.01) * s + s / 2, s)
    labels = [_num(t) if t % TIME_LABEL_S == 0 else '' for t in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_xlim(ticks[0], ticks[-1])


def _zero_aligned_ticks(lo1, hi1, lo2, hi2, nbins=4):
    """Ticks for a twin-y pair: each axis keeps its own nice step, both get the same
    number of steps below and above zero, so the zeros and the tick rows line up."""
    s1, s2 = _step(lo1, hi1, nbins), _step(lo2, hi2, nbins)
    below = max(0, np.ceil(-lo1 / s1 - 0.01), np.ceil(-lo2 / s2 - 0.01))
    above = max(0, np.ceil(hi1 / s1 - 0.01), np.ceil(hi2 / s2 - 0.01))
    k = np.arange(-below, above + 1)
    return np.round(k * s1, 10), np.round(k * s2, 10)


@style.paper
def fig7_causal_grid(tax, curves, nulls, n_subj, n_events):
    """3x3 peri-onset grid, strictly causal estimator (lowband_precursor_check.run).

    curves, nulls: {(ch, band): (n_subj, T)} per-subject means.
    """
    fig, axes = style.subplots(len(BANDS), len(CHANNELS), style.COL2_MM, 104,
                               layout='constrained', sharex=True)
    for i, bn in enumerate(BANDS):
        for j, ch in enumerate(CHANNELS):
            ax = axes[i, j]
            C, N = curves[(ch, bn)], nulls[(ch, bn)]
            m = np.nanmean(C, 0)
            sem = np.nanstd(C, 0) / np.sqrt(C.shape[0])
            nm = np.nanmean(N, 0)
            ax.plot(tax, m, color=BAND_COLORS[bn], lw=1.0, zorder=3)
            ax.fill_between(tax, m - sem, m + sem, color=BAND_COLORS[bn], alpha=0.25, lw=0,
                            zorder=2)
            ax.plot(tax, nm, zorder=2.5, **NULL_STYLE)
            ax.axvline(0, **EVENT_STYLE)
            style.reference_line(ax, y=0)
            lo = np.nanmin([np.nanmin(m - sem), np.nanmin(nm), 0.0])
            hi = np.nanmax([np.nanmax(m + sem), np.nanmax(nm), 0.0])
            _set_ticks(ax, 'y', _ticks(lo, hi, nbins=6), sparse=True)
            if i == 0:
                ax.set_title(ch, loc='center')
            if j == 0:
                ax.set_ylabel(f'{bn.replace("-", "–")} Hz power (z)')
    _time_axis(axes[-1, 0], tax[0], tax[-1])          # x is shared by every panel
    axes[-1, len(CHANNELS) // 2].set_xlabel('Time from delta-burst onset (s)')
    for ax in axes.flat:
        ax.spines[['left', 'bottom']].set_position(('outward', 3))   # corner labels apart
        style.spines_to_ticks(ax)
    fig.align_ylabels(axes[:, 0])
    means = tuple(plt.Line2D([], [], color=BAND_COLORS[b], lw=1.0) for b in BANDS)
    fig.legend([means, plt.Line2D([], [], **NULL_STYLE)],
               [f'Mean ± SEM across {n_subj} participants ({n_events} onsets)',
                'Count-matched random-NREM control'],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, handlelength=3.2,
               loc='outside upper center', ncol=2, columnspacing=2.5)
    return fig


@style.paper
def fig12_kcomplex(lags, ev, null, eeg, table, peak_win, rng):
    """Three-panel K-complex figure (kcomplex_cap_response.figure)."""
    ch_color = {c: CAP_COLORS[c] for c in CHANNELS}
    fig = style.figure(style.COL2_MM, 66, layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.05])
    w = (lags >= peak_win[0]) & (lags <= peak_win[1])

    # (a) event-triggered average; EEG delta (positive control) on the right axis
    ax = fig.add_subplot(gs[0, 0])
    handles = []
    lo, hi = 0.0, 0.0
    for ch in CHANNELS:
        st = ev[ch]
        if not len(st):
            continue
        m, se = st.mean(0), st.std(0) / np.sqrt(len(st))
        handles += ax.plot(lags, m, lw=1.0, color=ch_color[ch], label=ch, zorder=4)
        ax.fill_between(lags, m - se, m + se, color=ch_color[ch], alpha=0.18, lw=0, zorder=3)
        lo, hi = min(lo, (m - se).min()), max(hi, (m + se).max())
    nl = null[CHANNELS[-1]]
    if len(nl):
        nm = nl.mean(0)
        handles += ax.plot(lags, nm, label=f'Random-NREM\nnull ({CHANNELS[-1]})', zorder=2,
                           **NULL_STYLE)
        lo, hi = min(lo, nm.min()), max(hi, nm.max())
    ax.axvline(0, **EVENT_STYLE)
    style.reference_line(ax, y=0)
    _time_axis(ax, lags[0], lags[-1])
    if len(eeg):
        axt = ax.twinx()
        em = eeg.mean(0)
        handles += axt.plot(lags, em, color=EEG_COLOR, lw=1.0, zorder=1,
                            label='EEG delta\n(positive control)')
        yl, yr = _zero_aligned_ticks(lo, hi, min(em.min(), 0.0), max(em.max(), 0.0))
        _set_ticks(axt, 'y', yr)
        axt.set_ylabel('EEG delta (z)', color=EEG_COLOR)
        axt.tick_params(axis='y', colors=EEG_COLOR)
        axt.spines[['left', 'bottom', 'top']].set_visible(False)
        axt.spines['right'].set_visible(True)
        axt.spines['right'].set_color(EEG_COLOR)
        axt.spines['right'].set_position(('outward', 3))
        style.spines_to_ticks(axt, sides=('right',))
        ax.set_zorder(axt.get_zorder() + 1)       # SEC traces over the EEG trace
        ax.patch.set_visible(False)
    else:
        yl = _ticks(lo, hi, nbins=4)
    _set_ticks(ax, 'y', yl)
    ax.set_xlabel('Time from scored K-complex (s)')
    ax.set_ylabel('Low-frequency SEC power (z, baselined)')
    ax.legend(handles=handles, loc='upper left', handlelength=1.4)
    ax.spines[['left', 'bottom']].set_position(('outward', 3))       # corner labels apart
    style.spines_to_ticks(ax)
    style.panel_label(ax, 'a')

    # (b) per-event peak latency: one row per channel
    ax = fig.add_subplot(gs[0, 1])
    ax.yaxis.update_units(np.array(CHANNELS))          # a categorical axis
    for i, ch in enumerate(CHANNELS):
        st = ev[ch]
        if not len(st):
            continue
        lat = lags[w][np.argmax(st[:, w], axis=1)]
        # jitter only separates the dots; it is kept inside the search window
        x = np.clip(lat + rng.normal(0, 0.06, lat.size), *peak_win)
        y = np.full(lat.size, i) + rng.normal(0, 0.07, lat.size)
        ax.scatter(x, y, s=5, alpha=0.5, color=ch_color[ch], edgecolors='none',
                   clip_on=False, zorder=3)
        med = float(np.median(lat))
        ax.plot([med, med], [i - 0.32, i + 0.32], color=ch_color[ch], lw=1.5,
                solid_capstyle='butt', zorder=4)
        ax.text(med, i + 0.36, f'Median {med:.1f} s', ha='center', va='bottom',
                fontsize=6, color=ch_color[ch])
    ax.axvline(0, **EVENT_STYLE)
    ax.text(0.2, -0.62, 'Electrical pickup would sit here', fontsize=6,
            color=EVENT_STYLE['color'], ha='left', va='center')
    ax.set_yticks(range(len(CHANNELS)), CHANNELS)
    for lab, ch in zip(ax.get_yticklabels(), CHANNELS):
        lab.set_color(ch_color[ch])
    ax.set_ylim(-0.85, len(CHANNELS) - 0.5)
    _set_ticks(ax, 'x', _ticks(*peak_win, nbins=5))
    ax.set_xlabel('Latency of the peak response (s)')
    ax.spines[['left', 'bottom']].set_position(('outward', 3))   # time-0 line stays visible
    style.spines_to_ticks(ax, sides=('bottom',))
    style.panel_label(ax, 'b')

    # (c) per recording
    ax = fig.add_subplot(gs[0, 2])
    t = table.dropna(subset=[f'{CHANNELS[0]}_peak_z'])
    x = np.arange(len(t))
    names = [f'{r.session} ({r.n_kcomplex})' for r in t.itertuples()]
    ax.xaxis.update_units(np.array(names))                 # a categorical axis
    for i, ch in enumerate(CHANNELS):
        ax.bar(x + (i - 1) * 0.27, t[f'{ch}_peak_z'], width=0.26, color=ch_color[ch],
               label=ch, lw=0)
    peaks = t[[f'{ch}_peak_z' for ch in CHANNELS]].to_numpy()
    yt = _ticks(min(np.nanmin(peaks), 0.0), max(np.nanmax(peaks), 0.0), nbins=4)
    _set_ticks(ax, 'y', yt)
    if yt[0] < 0:
        style.reference_line(ax, y=0)
    ax.set_xticks(x, names, rotation=90)
    ax.tick_params(axis='x', direction='out')    # category ticks stay out of the bars
    ax.set_xlim(-0.7, len(t) - 0.3)              # first bars clear the y ticks
    ax.set_xlabel('Recording (number of K-complexes)')
    ax.set_ylabel('Peak response (z)')
    style.spines_to_ticks(ax, sides=('left',))
    style.panel_label(ax, 'c')

    # no figure title: the manuscript caption carries it; channel key in panel a only
    return fig
