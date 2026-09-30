"""
K-complex -> SEC response, reported the way the spindle result is reported.

The spindle section triggers on technologist-scored spindles. The direct
analogue for K-complexes is now available: the PSG export's `Spindle  K`
channel carries scored K-Complex marks as well as Spindle marks. This script
uses those marks — 50 across the cohort — as an INDEPENDENT trigger, i.e. one
that owes nothing to our own EEG envelope detector, and asks the same three
questions the spindle figure asks.

    A  event-triggered average of low-frequency SEC power at the K-complex,
       per channel, against a count-matched random-NREM null
    B  per-event peak response and its LATENCY
    C  per-session response, so a pooled average cannot hide a split cohort

Why latency carries the argument here, and not frequency. For spindles the
discriminating measurement is a frequency one: sigma (11-16 Hz) sits far outside
the capacitive band, so "no sigma in SEC" cleanly rules out electrical pickup.
A K-complex's own frequency is 0.5-4 Hz, which OVERLAPS the SEC bands, so that
test is unavailable. What separates the two explanations instead is timing:
electrical pickup would appear at zero lag, a mechanical/hemodynamic response
some seconds later (Fultz et al. 2019 put EEG ahead of the CSF response by ~6 s).

n = 50 scored marks over 11 sessions is small and S5N1 has none, so this is
reported per-session and descriptively. The 340-event N2 delta-onset proxy in
analysis/prof_requests_sep2026/kcomplex_comparison.py carries the statistics;
this is the version with an independent trigger.

Usage
-----
    .venv/Scripts/python.exe analysis/delta_onset/kcomplex_cap_response.py

Outputs
-------
    analysis/delta_onset/outputs/fig_kcomplex_cap_response.png
    analysis/delta_onset/outputs/kcomplex_cap_response.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from sleep_monitor.loader import load_session, load_interval_events
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import CAP_COLORS, FS
import delta_cap_precursor as P

OUT = HERE / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

AFS = P.ANALYSIS_FS                     # 20 Hz analysis grid
PRE_S, POST_S = 15.0, 15.0
PRE, POST = int(PRE_S * AFS), int(POST_S * AFS)
BASE_END = PRE - int(2 * AFS)           # baseline = -15 s .. -2 s
CHANS = P.CHANNELS                      # CLE, CRE, CH
BANDS = list(P.BANDS.keys())            # 0-0.5, 0.5-1, 1-3
PEAK_WIN = (0.0, 10.0)                  # where a response peak is looked for (s)
RNG = np.random.default_rng(11)

CH_COLOR = {c: CAP_COLORS[c] for c in CHANS}
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10.5,
    'axes.linewidth': 0.9, 'axes.edgecolor': '#333333',
    'font.family': 'DejaVu Sans', 'figure.dpi': 200,
})


def scored_kcomplexes_20hz(meta):
    """Scored K-Complex marks for one session, as indices on the 20 Hz grid."""
    s = load_session(meta['idx'])
    ev = load_interval_events(s, 'Spindle  K')
    if ev is None:
        return np.array([], int)
    types = np.array([t.strip().lower() for t in ev['types']])
    k_hr = np.asarray(ev['start_hr'])[np.char.startswith(types, 'k')]
    return np.round(k_hr * 3600.0 * AFS).astype(int)


def lowband(cap_envs, ch, nrem):
    """One low-frequency SEC trace per channel: the mean of the three band
    envelopes after each is standardised over NREM. Stands in for the spindle
    figure's 0-3 Hz power."""
    z = [P._zscore_on(cap_envs[f'cap_{ch}_{b}'].astype(np.float64), nrem) for b in BANDS]
    return np.mean(z, axis=0)


def baselined(stack):
    return stack - stack[:, :BASE_END].mean(axis=1, keepdims=True)


def collect():
    lags = np.arange(-PRE, POST) / AFS
    ev_stacks = {c: [] for c in CHANS}
    null_stacks = {c: [] for c in CHANS}
    eeg_stacks = []
    rows = []

    for meta in SESSION_META:
        lab = meta['label']
        k = scored_kcomplexes_20hz(meta)
        if k.size == 0:
            print(f'  {lab}: no scored K-complexes')
            continue
        _, cap_envs, eeg_delta, nrem, motion = P.load_features(meta['idx'])
        n = len(eeg_delta)
        k = k[(k >= PRE) & (k + POST < n)]
        if k.size == 0:
            continue
        nulls = P.random_nrem_centers(nrem, motion, k, PRE, POST, max(k.size * 20, 200), RNG)

        st = P.peri_stack(P._zscore_on(eeg_delta, nrem), k, PRE, POST)
        if st is not None:
            eeg_stacks.append(baselined(st))

        row = {'session': lab, 'n_kcomplex': int(k.size)}
        for ch in CHANS:
            tr = lowband(cap_envs, ch, nrem)
            st = P.peri_stack(tr, k, PRE, POST)
            if st is None:
                continue
            st = baselined(st)
            ev_stacks[ch].append(st)
            if nulls.size:
                ns = P.peri_stack(tr, nulls, PRE, POST)
                if ns is not None:
                    null_stacks[ch].append(baselined(ns))
            m = st.mean(axis=0)
            w = (lags >= PEAK_WIN[0]) & (lags <= PEAK_WIN[1])
            row[f'{ch}_peak_z'] = float(m[w].max())
            row[f'{ch}_peak_lat_s'] = float(lags[w][np.argmax(m[w])])
        rows.append(row)
        print(f'  {lab}: {k.size} scored K-complexes')

    cat = lambda d: {c: (np.vstack(v) if v else np.zeros((0, PRE + POST))) for c, v in d.items()}
    eeg = np.vstack(eeg_stacks) if eeg_stacks else np.zeros((0, PRE + POST))
    return lags, cat(ev_stacks), cat(null_stacks), eeg, pd.DataFrame(rows)


def figure(lags, ev, null, eeg, table):
    fig = plt.figure(figsize=(17, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.05], wspace=0.30)
    w = (lags >= PEAK_WIN[0]) & (lags <= PEAK_WIN[1])
    n_ev = len(ev[CHANS[0]])

    # ── A: event-triggered average ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    axt = ax.twinx()
    if len(eeg):
        m = eeg.mean(0)
        axt.plot(lags, m, color='#95A5A6', lw=1.8, ls='--', zorder=1)
        axt.set_ylabel('EEG delta (z) — positive control', color='#7F8C8D', fontsize=11)
        axt.tick_params(axis='y', labelcolor='#7F8C8D')
    for ch in CHANS:
        st = ev[ch]
        if not len(st):
            continue
        m, se = st.mean(0), st.std(0) / np.sqrt(len(st))
        ax.plot(lags, m, lw=2.3, color=CH_COLOR[ch], label=ch, zorder=4)
        ax.fill_between(lags, m - se, m + se, color=CH_COLOR[ch], alpha=0.18, lw=0, zorder=3)
    nl = null[CHANS[-1]]
    if len(nl):
        ax.plot(lags, nl.mean(0), lw=1.6, color='#555555', ls=':',
                label='random-NREM null', zorder=2)
    ax.axvline(0, color='#C0392B', lw=1.5, ls='--')
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xlabel('Time from scored K-complex (s)')
    ax.set_ylabel('Low-frequency SEC power (z, baselined)')
    ax.set_title(f'A   SEC response at the K-complex  (n={n_ev} events)',
                 loc='left', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.92)
    ax.grid(True, alpha=0.15)

    # ── B: peak latency ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for i, ch in enumerate(CHANS):
        st = ev[ch]
        if not len(st):
            continue
        lat = lags[w][np.argmax(st[:, w], axis=1)]
        ax.scatter(lat + RNG.normal(0, 0.06, lat.size), np.full(lat.size, i)
                   + RNG.normal(0, 0.07, lat.size), s=16, alpha=0.4,
                   color=CH_COLOR[ch], edgecolors='none')
        med = float(np.median(lat))
        ax.plot([med, med], [i - 0.32, i + 0.32], color=CH_COLOR[ch], lw=3.2)
        ax.text(med, i + 0.42, f'median {med:.1f} s', ha='center', fontsize=11,
                color=CH_COLOR[ch], fontweight='bold')
    ax.axvline(0, color='#C0392B', lw=1.5, ls='--')
    ax.text(0.22, -0.52, 'electrical pickup would sit here', fontsize=10.5,
            color='#C0392B', va='center')
    ax.set_yticks(range(len(CHANS)))
    ax.set_yticklabels(CHANS)
    ax.set_ylim(-0.75, 2.85)
    ax.set_xlim(PEAK_WIN)
    ax.set_xlabel('Latency of the peak response (s)')
    ax.set_title('B   Response follows the event', loc='left', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.15)

    # ── C: per session ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    t = table.dropna(subset=[f'{CHANS[0]}_peak_z'])
    x = np.arange(len(t))
    for i, ch in enumerate(CHANS):
        ax.bar(x + (i - 1) * 0.27, t[f'{ch}_peak_z'], width=0.26,
               color=CH_COLOR[ch], label=ch, alpha=0.9)
    ax.axhline(0, color='#333', lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r.session}  n={r.n_kcomplex}' for r in t.itertuples()],
                       fontsize=9.5, rotation=60, ha='right')
    ax.set_ylabel('Peak response (z)')
    ax.set_title('C   Per session (note how few events)', loc='left',
                 fontweight='bold')
    ax.legend(framealpha=0.92)
    ax.grid(True, axis='y', alpha=0.15)

    fig.suptitle('K-complex → SEC: a low-frequency response that FOLLOWS the '
                 'cortical event  (PSG-scored K-complexes, independent trigger)',
                 fontsize=15, fontweight='bold')
    p = OUT / 'fig_kcomplex_cap_response.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ->', p.name)


def main():
    print('K-complex -> SEC response, triggered on PSG-scored K-complexes:')
    lags, ev, null, eeg, table = collect()
    w = (lags >= PEAK_WIN[0]) & (lags <= PEAK_WIN[1])
    print('\npooled, per channel:')
    for ch in CHANS:
        st, nl = ev[ch], null[ch]
        if not len(st):
            continue
        m = st.mean(0)
        pk = m[w].max(); lat = lags[w][np.argmax(m[w])]
        nullpk = nl.mean(0)[w].max() if len(nl) else np.nan
        print(f'  {ch:4}  peak {pk:+.2f} z at {lat:+.1f} s   '
              f'(random-NREM null peak {nullpk:+.2f} z)   n={len(st)} events')
    table.to_csv(OUT / 'kcomplex_cap_response.csv', index=False)
    print('  -> kcomplex_cap_response.csv')
    figure(lags, ev, null, eeg, table)


if __name__ == '__main__':
    main()
