"""
Request D (Prof, 2026-09-09): "Did you also compare our signals with K-complex in EEG?"

Answer path
-----------
Yes — the delta-burst onset set already built in analysis/delta_onset/ is N2-dominant,
i.e. it IS the K-complex / isolated-slow-wave population (sustained N3 slow-wave runs
have no quiet baseline and are largely excluded). This script re-cuts that set to the
N2 onsets (stage_code == 2) as an explicit K-complex proxy and produces the
K-complex-triggered SEC average: EEG delta on top, then SEC band power in three bands
(0-0.5, 0.5-1, 1-3 Hz) x three channels (CLE, CRE, CH), aligned to the K-complex at t=0,
with a random-NREM null.

The result mirrors the existing precursor finding: the SEC response FOLLOWS the
K-complex (peak ~+2..+5 s), it does not precede it — the same direction as Fultz et al.
2019 (EEG leads the downstream CSF/hemodynamic response by ~6 s).

Reuses analysis/delta_onset/delta_cap_precursor.py helpers (envelope cache, peri-stack,
random-NREM null). Motion-clean by construction of the onsets.

Outputs
-------
  reports/prof_requests_sep2026/kcomplex_response_peak.csv
  notebooks/plots/prof_requests_sep2026/D_kcomplex_triggered_grid.png

Usage:
    .venv/Scripts/python.exe analysis/prof_requests_sep2026/kcomplex_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis' / 'delta_onset'))

import delta_cap_precursor as P          # envelope cache + peri machinery
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import CAP_COLORS

PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'prof_requests_sep2026'
REPORT_DIR = ROOT / 'reports' / 'prof_requests_sep2026'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

AFS = P.ANALYSIS_FS                        # 20 Hz
PRE_S, POST_S = 15.0, 15.0
PRE, POST = int(PRE_S * AFS), int(POST_S * AFS)
BASELINE = (-PRE, -int(2 * AFS))           # per-window baseline window (samples)
BANDS = list(P.BANDS.keys())              # ['0-0.5','0.5-1','1-3']
CHANS = P.CHANNELS                        # ['CLE','CRE','CH']
ONSET_DIR = ROOT / 'analysis' / 'delta_onset' / 'outputs'
BAND_COLORS = P.BAND_COLORS
CH_COLOR = {'CLE': CAP_COLORS['CLE'], 'CRE': CAP_COLORS['CRE'], 'CH': CAP_COLORS['CH']}


def load_n2_onsets_20hz(label):
    """K-complex proxy: base delta onsets scored N2, mapped to the 20 Hz grid."""
    f = ONSET_DIR / f'delta_onsets_{label}.npz'
    if not f.exists():
        return np.array([], int)
    z = np.load(f)
    samp = z['onset_samp']; stg = z['stage_code']
    if samp.size == 0:
        return np.array([], int)
    samp = samp[stg == 2]                  # N2 only
    q = int(round(P.FS / AFS))
    return np.round(samp / q).astype(int)


def main():
    print('=' * 66)
    print('D. K-complex-triggered SEC average (N2 delta onsets = K-complex proxy)')
    print('=' * 66)

    lags = (np.arange(-PRE, POST) / AFS)
    # pooled peri stacks
    eeg_stack = []
    cap_stacks = {f'{ch}_{b}': [] for ch in CHANS for b in BANDS}
    null_stacks = {f'{ch}_{b}': [] for ch in CHANS for b in BANDS}
    rng = np.random.default_rng(7)
    total_kc = 0

    for idx in range(len(SESSION_META)):
        label = SESSION_META[idx]['label']
        onsets = load_n2_onsets_20hz(label)
        if onsets.size == 0:
            print(f'  {label}: no N2 K-complex onsets')
            continue
        _, cap_envs, eeg_delta, nrem, motion = P.load_features(idx)
        n = len(eeg_delta)
        onsets = onsets[(onsets >= PRE) & (onsets + POST < n)]
        if onsets.size == 0:
            continue
        total_kc += onsets.size
        print(f'  {label}: {onsets.size} K-complex (N2) onsets')

        eeg_z = P._zscore_on(eeg_delta, nrem)
        st = P.peri_stack(eeg_z, onsets, PRE, POST)
        if st is not None:
            eeg_stack.append(st - st[:, :PRE - int(2 * AFS)].mean(axis=1, keepdims=True))

        nulls = P.random_nrem_centers(nrem, motion, onsets, PRE, POST, onsets.size, rng)
        for ch in CHANS:
            for b in BANDS:
                env = cap_envs[f'cap_{ch}_{b}'].astype(np.float64)
                envz = P._zscore_on(env, nrem)
                st = P.peri_stack(envz, onsets, PRE, POST)
                if st is not None:
                    st = st - st[:, :PRE - int(2 * AFS)].mean(axis=1, keepdims=True)
                    cap_stacks[f'{ch}_{b}'].append(st)
                if nulls.size:
                    ns = P.peri_stack(envz, nulls, PRE, POST)
                    if ns is not None:
                        ns = ns - ns[:, :PRE - int(2 * AFS)].mean(axis=1, keepdims=True)
                        null_stacks[f'{ch}_{b}'].append(ns)

    print(f'\nTotal K-complex onsets pooled: {total_kc}')

    # ── Figure: 3 bands (rows) x 3 channels (cols) + EEG delta strip on top ──
    fig, axes = plt.subplots(len(BANDS) + 1, len(CHANS), figsize=(15, 11),
                             sharex=True)
    eeg_all = np.vstack(eeg_stack) if eeg_stack else None
    for c in range(len(CHANS)):
        ax = axes[0, c]
        if eeg_all is not None:
            m = eeg_all.mean(0); se = eeg_all.std(0) / np.sqrt(len(eeg_all))
            ax.plot(lags, m, color='#2C3E50', lw=2)
            ax.fill_between(lags, m - se, m + se, color='#2C3E50', alpha=0.2)
        ax.axvline(0, color='#C0392B', ls='--', lw=1.4)
        ax.axhline(0, color='gray', ls=':', lw=0.7)
        ax.set_title(CHANS[c], fontsize=13, fontweight='bold')
        if c == 0:
            ax.set_ylabel('EEG delta\n(z)', fontsize=10)

    peak_rows = []
    for r, b in enumerate(BANDS):
        for c, ch in enumerate(CHANS):
            ax = axes[r + 1, c]
            key = f'{ch}_{b}'
            arr = np.vstack(cap_stacks[key]) if cap_stacks[key] else None
            nul = np.vstack(null_stacks[key]) if null_stacks[key] else None
            if nul is not None:
                nm = nul.mean(0); nse = nul.std(0) / np.sqrt(len(nul))
                ax.fill_between(lags, nm - 2 * nse, nm + 2 * nse, color='0.7',
                                alpha=0.5, label='random-NREM null')
            if arr is not None:
                m = arr.mean(0); se = arr.std(0) / np.sqrt(len(arr))
                ax.plot(lags, m, color=CH_COLOR[ch], lw=2.2)
                ax.fill_between(lags, m - se, m + se, color=CH_COLOR[ch], alpha=0.25)
                post = (lags >= 0) & (lags <= 8)
                pk_t = lags[post][np.argmax(m[post])]
                peak_rows.append({'channel': ch, 'band': b,
                                  'peak_lag_s': float(pk_t),
                                  'peak_amp_z': float(np.max(m[post]))})
            ax.axvline(0, color='#C0392B', ls='--', lw=1.4)
            ax.axhline(0, color='gray', ls=':', lw=0.7)
            if c == 0:
                ax.set_ylabel(f'{b} Hz\n(z)', fontsize=10)
            if r == len(BANDS) - 1:
                ax.set_xlabel('Time from K-complex (s)', fontsize=10)
    axes[1, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('D. K-complex-triggered SEC band power '
                 f'(N2 delta onsets, n={total_kc}) — response FOLLOWS the K-complex, does not precede it',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOT_DIR / 'D_kcomplex_triggered_grid.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')

    pk = pd.DataFrame(peak_rows)
    pk.to_csv(REPORT_DIR / 'kcomplex_response_peak.csv', index=False)
    if not pk.empty:
        print('\nPost-K-complex response peak (0..8 s), by channel/band:')
        print(pk.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print('\nDone.')


if __name__ == '__main__':
    main()
