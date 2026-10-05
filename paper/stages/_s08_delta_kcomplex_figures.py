"""Figures for s08_delta_kcomplex: Fig. 7 (delta-burst onsets) and Fig. 12 (K-complexes)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from seclib.config import CAP_COLORS

from stages._s08_delta_kcomplex_signal import BANDS, CHANNELS

BAND_COLORS = {'0-0.5': '#1ABC9C', '0.5-1': '#8E44AD', '1-3': '#E67E22'}
ONSET_COLOR = '#27AE60'
EVENT_COLOR = '#C0392B'


def fig7_causal_grid(tax, curves, nulls, n_subj, n_events):
    """3x3 peri-onset grid, strictly causal estimator (lowband_precursor_check.run).

    curves, nulls: {(ch, band): (n_subj, T)} per-subject means.
    """
    fig, axes = plt.subplots(len(BANDS), len(CHANNELS), figsize=(13.5, 9.6), layout="constrained",
                             sharex=True)
    for i, bn in enumerate(BANDS):
        for j, ch in enumerate(CHANNELS):
            ax = axes[i, j]
            C, N = curves[(ch, bn)], nulls[(ch, bn)]
            m = np.nanmean(C, 0)
            sem = np.nanstd(C, 0) / np.sqrt(C.shape[0])
            ax.plot(tax, m, color=BAND_COLORS[bn], lw=1.8)
            ax.fill_between(tax, m - sem, m + sem, color=BAND_COLORS[bn], alpha=0.25, lw=0)
            ax.plot(tax, np.nanmean(N, 0), color='#888888', ls='--', lw=1.1)
            ax.axvline(0, color=ONSET_COLOR, lw=1.3)
            ax.axhline(0, color='k', lw=0.5)
            ax.grid(True, alpha=0.15)
            if i == 0:
                ax.set_title(ch, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{bn} Hz\nband power (z)')
            if i == len(BANDS) - 1:
                ax.set_xlabel('Time from delta-burst onset (s)')
    fig.suptitle('SEC band power at EEG delta-burst onset, strictly causal estimator\n'
                 f'mean ± SEM across {n_subj} participants ({n_events} onsets); '
                 'grey dashed = count-matched random-NREM control', fontsize=13)
    return fig


def fig12_kcomplex(lags, ev, null, eeg, table, peak_win, rng):
    """Three-panel K-complex figure (kcomplex_cap_response.figure)."""
    ch_color = {c: CAP_COLORS[c] for c in CHANNELS}
    fig = plt.figure(figsize=(17, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.05], wspace=0.34)
    w = (lags >= peak_win[0]) & (lags <= peak_win[1])

    # (a) event-triggered average
    ax = fig.add_subplot(gs[0, 0])
    axt = ax.twinx()
    axt.spines['right'].set_visible(True)
    if len(eeg):
        axt.plot(lags, eeg.mean(0), color='#95A5A6', lw=1.8, ls='--', zorder=1)
        axt.set_ylabel('EEG delta (z), positive control', color='#7F8C8D', fontsize=11)
        axt.tick_params(axis='y', labelcolor='#7F8C8D')
    for ch in CHANNELS:
        st = ev[ch]
        if not len(st):
            continue
        m, se = st.mean(0), st.std(0) / np.sqrt(len(st))
        ax.plot(lags, m, lw=2.3, color=ch_color[ch], label=ch, zorder=4)
        ax.fill_between(lags, m - se, m + se, color=ch_color[ch], alpha=0.18, lw=0, zorder=3)
    nl = null[CHANNELS[-1]]
    if len(nl):
        ax.plot(lags, nl.mean(0), lw=1.6, color='#555555', ls=':',
                label='random-NREM null', zorder=2)
    ax.axvline(0, color=EVENT_COLOR, lw=1.5, ls='--')
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xlabel('Time from scored K-complex (s)')
    ax.set_ylabel('Low-frequency SEC power (z, baselined)')
    ax.set_title('a', loc='left', fontweight='bold', fontsize=15)
    ax.legend(loc='upper left', framealpha=0.92)
    ax.grid(True, alpha=0.15)

    # (b) per-event peak latency
    ax = fig.add_subplot(gs[0, 1])
    for i, ch in enumerate(CHANNELS):
        st = ev[ch]
        if not len(st):
            continue
        lat = lags[w][np.argmax(st[:, w], axis=1)]
        ax.scatter(lat + rng.normal(0, 0.06, lat.size),
                   np.full(lat.size, i) + rng.normal(0, 0.07, lat.size),
                   s=16, alpha=0.4, color=ch_color[ch], edgecolors='none')
        med = float(np.median(lat))
        ax.plot([med, med], [i - 0.32, i + 0.32], color=ch_color[ch], lw=3.2)
        ax.text(med, i + 0.42, f'median {med:.1f} s', ha='center', fontsize=11,
                color=ch_color[ch], fontweight='bold')
    ax.axvline(0, color=EVENT_COLOR, lw=1.5, ls='--')
    ax.text(0.22, -0.52, 'electrical pickup would sit here', fontsize=10.5,
            color=EVENT_COLOR, va='center')
    ax.set_yticks(range(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS)
    ax.set_ylim(-0.75, 2.85)
    ax.set_xlim(peak_win)
    ax.set_xlabel('Latency of the peak response (s)')
    ax.set_title('b', loc='left', fontweight='bold', fontsize=15)
    ax.grid(True, axis='x', alpha=0.15)

    # (c) per recording
    ax = fig.add_subplot(gs[0, 2])
    t = table.dropna(subset=[f'{CHANNELS[0]}_peak_z'])
    x = np.arange(len(t))
    for i, ch in enumerate(CHANNELS):
        ax.bar(x + (i - 1) * 0.27, t[f'{ch}_peak_z'], width=0.26, color=ch_color[ch],
               label=ch, alpha=0.9)
    ax.axhline(0, color='#333', lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r.session}  n={r.n_kcomplex}' for r in t.itertuples()],
                       fontsize=9.5, rotation=60, ha='right')
    ax.set_ylabel('Peak response (z)')
    ax.set_title('c', loc='left', fontweight='bold', fontsize=15)
    ax.legend(framealpha=0.92)
    ax.grid(True, axis='y', alpha=0.15)

    # no figure title: the manuscript caption carries it
    return fig
