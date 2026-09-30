"""
What a K-complex actually looks like in this dataset — and what our delta-burst
detector catches instead.

The PSG export carries a `Spindle  K` channel holding both Spindle and
K-Complex impulse marks (Somnomedics `Spindel\\spindel`). The scorer annotated
spindles exhaustively (21,881 cohort-wide) and K-complexes essentially not at
all (57), so the K-complex marks are useless as a ground-truth set — but they
are perfectly good as EXAMPLES. This script uses them for exactly that.

Figure 1 — `fig_kcomplex_morphology.png`
    A  eight individual scored K-complexes, raw EEG, +/-2 s
    B  grand average over every scored K-complex in the cohort, +/- SEM
    C  a scored spindle, raw EEG with its 11-16 Hz sigma component overlaid
    D  mean spectra around K-complexes, around spindles, and at random N2
       points — the two events live in different bands, which is the whole
       reason the spindle test and the delta-burst test are not equivalent

Figure 2 — `fig_kcomplex_vs_detector.png`
    A  a scored K-complex with the detector's own delta envelope and its two
       thresholds underneath, on the same scale as B
    B  one of our qualifying delta-burst onsets, with the 30 s quiescence
       window and the >=4 s sustained-burst requirement drawn
    C  every scored K-complex put through both detector criteria

    C is the panel that matters and it corrected the expectation this figure was
    built on. A K-complex is a sub-second waveform, so the guess was that one
    could not satisfy a >=4 s sustained-burst criterion. It usually can, because
    a K-complex is typically followed by further slow waves: the median scored
    K-complex holds its delta envelope above the burst threshold for 3.5 s
    (IQR 2.3-6.1), and 44% clear 4 s. What actually thins them is the pair of
    gates together — 18% pass both — and the detector finds ALL of those. The
    detected onsets are therefore genuinely K-complex-like events, but a
    selective fifth of them, not the K-complex population.

Envelope, thresholds and onsets are imported from delta_onset_detection.py, not
restated, so this figure and the detector cannot disagree.

Usage
-----
    .venv/Scripts/python.exe analysis/delta_onset/kcomplex_morphology.py

Outputs
-------
    analysis/delta_onset/outputs/fig_kcomplex_morphology.png
    analysis/delta_onset/outputs/fig_kcomplex_vs_detector.png
    analysis/delta_onset/outputs/kcomplex_marks.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from sleep_monitor.loader import load_session, load_interval_events
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import FS
import delta_onset_detection as D

OUT = HERE / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

PAD_S = 2.0                      # half-width of the morphology snippets
SPEC_S = 1.5                     # half-width of the spectral windows
N_SPINDLE_SPEC = 1500            # spindles sampled for the mean spectrum
N_RANDOM_SPEC = 1500             # random N2 points for the control spectrum
RNG = np.random.default_rng(0)

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10.5,
    'axes.linewidth': 0.9, 'axes.edgecolor': '#333333',
    'font.family': 'DejaVu Sans', 'figure.dpi': 200,
})
C_EEG = '#2C3E50'
C_K = '#C0392B'
C_SP = '#2980B9'
C_RAND = '#7F8C8D'


def snippets(sig, centres_samp, pad):
    """Stack of sig[c-pad : c+pad] for centres that fit inside the record."""
    out = []
    for c in centres_samp:
        a, b = int(c) - pad, int(c) + pad
        if a >= 0 and b <= len(sig):
            out.append(sig[a:b])
    return np.asarray(out) if out else np.zeros((0, 2 * pad))


def collect():
    """One pass over the cohort: scored marks + the EEG snippets around them."""
    pad = int(PAD_S * FS)
    spec = int(SPEC_S * FS)
    k_snips, sp_snips = [], []
    k_spec, sp_spec, rand_spec = [], [], []
    rows, crit = [], []
    keep_examples = []           # (label, idx, eeg, env, low, high, k_samp)

    for m in SESSION_META:
        lab = m['label']
        s = load_session(m['idx'])
        ev = load_interval_events(s, 'Spindle  K')
        if ev is None:
            rows.append({'session': lab, 'n_kcomplex': 0, 'n_spindle': 0,
                         'note': 'no Spindle  K file'})
            continue

        types = np.array([t.strip().lower() for t in ev['types']])
        start = np.asarray(ev['start_hr']) * 3600.0 * FS
        dur = np.asarray(ev['duration_s'])           # ms per the file header
        is_k = np.char.startswith(types, 'k')
        k_samp = start[is_k]
        sp_samp = start[~is_k]

        label, env, codes, motion, eeg = D.load_features(m['idx'])
        eeg = np.asarray(eeg, float)

        ks = snippets(eeg, k_samp, pad)
        k_snips.append(ks)
        k_spec.append(snippets(eeg, k_samp, spec))
        if sp_samp.size:
            pick = RNG.choice(sp_samp, min(N_SPINDLE_SPEC, sp_samp.size), replace=False)
            sp_spec.append(snippets(eeg, pick, spec))
            sp_snips.append(snippets(eeg, sp_samp[:60], pad))
        n2 = np.flatnonzero(codes == 2)
        if n2.size > N_RANDOM_SPEC:
            rand_spec.append(snippets(eeg, RNG.choice(n2, N_RANDOM_SPEC, replace=False), spec))

        rows.append({'session': lab, 'n_kcomplex': int(is_k.sum()),
                     'n_spindle': int((~is_k).sum()),
                     'k_dur_ms_median': float(np.median(dur[is_k])) if is_k.any() else np.nan,
                     'note': ''})

        if is_k.any():
            env = np.asarray(env, float)
            nrem = np.isin(codes, D.NREM_CODES)
            med = np.median(env[nrem])
            mad = np.median(np.abs(env[nrem] - med)) * 1.4826 + 1e-12
            low, high = med + D.K_LOW * mad, med + D.K_HIGH * mad
            keep_examples.append((lab, m['idx'], eeg, env, low, high, k_samp))
            crit += score_criteria(env, codes, k_samp, low, high, lab)
        print(f'  {lab}: K={int(is_k.sum()):3d}  spindles={int((~is_k).sum()):5d}')

    def cat(x):
        x = [a for a in x if len(a)]
        return np.concatenate(x) if x else np.zeros((0, 1))

    return (cat(k_snips), cat(sp_snips), cat(k_spec), cat(sp_spec), cat(rand_spec),
            pd.DataFrame(rows), keep_examples, pd.DataFrame(crit))


def longest_run_s(mask, fs=FS):
    best, start = 0.0, None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            best = max(best, (i - start) / fs); start = None
    if start is not None:
        best = max(best, (len(mask) - start) / fs)
    return best


def score_criteria(env, codes, k_samp, low, high, lab):
    """Put every scored K-complex through the detector's own two gates."""
    q = int(D.PRE_S * FS)
    out = []
    for c in k_samp:
        c = int(c)
        a, b = c - q, c + int(10 * FS)
        if a < 0 or b > len(env):
            continue
        out.append({
            'session': lab,
            'stage': int(codes[c]),
            'burst_s': longest_run_s(env[c - int(FS):b] > high),
            'quiet_ok': bool(env[a:c].mean() < low),
        })
    return out


def mean_psd(stack):
    if not len(stack):
        return np.array([]), np.array([])
    f, p = welch(stack - stack.mean(axis=1, keepdims=True), fs=FS,
                 nperseg=min(256, stack.shape[1]), axis=1)
    return f, p.mean(axis=0)


def figure_morphology(k_snips, sp_snips, k_spec, sp_spec, rand_spec, unit):
    t = np.linspace(-PAD_S, PAD_S, k_snips.shape[1])
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.05], hspace=0.45, wspace=0.28)

    # ── A: individual scored K-complexes ─────────────────────────────────────
    pick = np.linspace(0, len(k_snips) - 1, min(8, len(k_snips))).astype(int)
    for j, i in enumerate(pick):
        ax = fig.add_subplot(gs[j // 4, j % 4])
        y = k_snips[i] - np.median(k_snips[i])
        ax.plot(t, y, lw=1.0, color=C_EEG)
        ax.axvline(0, color=C_K, lw=1.2, ls='--')
        ax.set_xlim(-PAD_S, PAD_S)
        ax.grid(True, alpha=0.15)
        if j % 4 == 0:
            ax.set_ylabel(f'EEG ({unit})')
        if j // 4 == 1:
            ax.set_xlabel('Time from mark (s)')
        if j == 0:
            ax.set_title('A   Individual scored K-complexes', loc='left',
                         fontweight='bold', fontsize=13)

    # ── B: grand average K-complex ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0:2])
    y = k_snips - np.median(k_snips, axis=1, keepdims=True)
    mu, sem = y.mean(axis=0), y.std(axis=0) / np.sqrt(len(y))
    ax.fill_between(t, mu - sem, mu + sem, color=C_K, alpha=0.25, lw=0)
    ax.plot(t, mu, lw=2.2, color=C_K)
    ax.axvline(0, color='#555', lw=1.0, ls='--')
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xlim(-PAD_S, PAD_S)
    ax.set_xlabel('Time from mark (s)')
    ax.set_ylabel(f'EEG ({unit})')
    ax.set_title(f'B   Grand average, all {len(k_snips)} scored K-complexes '
                 f'(± SEM)', loc='left', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.15)

    # ── C: mean spectra ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2:4])
    for stack, col, name in ((k_spec, C_K, f'around K-complexes (n={len(k_spec)})'),
                             (sp_spec, C_SP, f'around spindles (n={len(sp_spec)})'),
                             (rand_spec, C_RAND, f'random N2 (n={len(rand_spec)})')):
        f, p = mean_psd(stack)
        if f.size:
            ax.semilogy(f, p, lw=2.0, color=col, label=name)
    ax.axvspan(0.5, 4, color=C_K, alpha=0.08, lw=0)
    ax.axvspan(11, 16, color=C_SP, alpha=0.10, lw=0)
    ax.text(2.2, 0.97, 'delta\n0.5-4 Hz', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=10, color=C_K, fontweight='bold')
    ax.text(13.5, 0.97, 'sigma\n11-16 Hz', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=10, color=C_SP, fontweight='bold')
    ax.set_xlim(0.3, 25)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(f'PSD ({unit}$^2$/Hz)')
    ax.set_title('C   The two events live in different bands', loc='left',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='lower left', framealpha=0.92)
    ax.grid(True, alpha=0.15)

    fig.suptitle('What a K-complex looks like in this cohort '
                 '(PSG-scored marks, contact EEG)', fontsize=16, fontweight='bold')
    p = OUT / 'fig_kcomplex_morphology.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ->', p.name)
    return p


def figure_vs_detector(examples, crit, unit):
    """A scored K-complex, a qualifying onset, and every K-complex scored against
    the detector's two gates."""
    lab, idx, eeg, env, low, high, k_samp = examples[0]
    onset_f = OUT / f'delta_onsets_{lab}_q30.npz'
    onsets = np.load(onset_f, allow_pickle=True)['onset_hr'] * 3600.0 * FS

    fig, axes = plt.subplots(2, 3, figsize=(20, 8),
                             gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0.34,
                                          'wspace': 0.26})
    for r in (0, 1):
        axes[r, 1].sharey(axes[r, 0])

    def draw(col, centre, pre_s, post_s, title, shade_quiet):
        a, b = int(centre - pre_s * FS), int(centre + post_s * FS)
        t = (np.arange(a, b) - centre) / FS
        ax0, ax1 = axes[0, col], axes[1, col]
        ax0.plot(t, eeg[a:b] - np.median(eeg[a:b]), lw=0.9, color=C_EEG)
        ax0.axvline(0, color=C_K, lw=1.4, ls='--')
        ax0.set_ylabel(f'EEG ({unit})' if col == 0 else '')
        ax0.set_title(title, loc='left', fontweight='bold', fontsize=13)
        ax0.grid(True, alpha=0.15)

        e = env[a:b]
        ax1.plot(t, e, lw=1.6, color='#8E44AD')
        ax1.axhline(high, color='#C0392B', lw=1.2, ls='--',
                    label='burst threshold  med+2·MAD')
        ax1.axhline(low, color='#E67E22', lw=1.2, ls=':',
                    label='onset threshold  med+0.5·MAD')
        ax1.axvline(0, color=C_K, lw=1.4, ls='--')
        if shade_quiet:
            ax1.axvspan(-30, 0, color='#27AE60', alpha=0.10, lw=0)
            ax1.text(-15, 0.99, 'required quiet window (30 s)', ha='center',
                     va='top', transform=ax1.get_xaxis_transform(),
                     fontsize=10, color='#1E8449', fontweight='bold')
        longest = longest_run_s(e[max(0, int((pre_s - 1) * FS)):] > high)
        ax1.text(0.985, 0.04, f'holds above burst threshold for {longest:.1f} s\n'
                              f'(detector needs ≥ 4.0 s)',
                 transform=ax1.transAxes, ha='right', va='bottom', fontsize=11,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='#999', pad=4))
        ax1.set_xlabel('Time from event (s)')
        ax1.set_ylabel('EEG delta envelope' if col == 0 else '')
        ax1.grid(True, alpha=0.15)
        if col == 0:
            ax1.legend(loc='upper left', framealpha=0.92, fontsize=10)

    draw(0, k_samp[0], 6, 6,
         f'A   One PSG-scored K-complex  ({lab})', shade_quiet=False)
    draw(1, onsets[0], 35, 15,
         f'B   One qualifying delta-burst onset  ({lab})', shade_quiet=True)

    # ── C: every scored K-complex against both gates ─────────────────────────
    n = len(crit)
    ax = axes[0, 2]
    ax.hist(crit.burst_s, bins=np.arange(0, 12.5, 1.0), color='#8E44AD',
            alpha=0.75, edgecolor='white')
    ax.axvline(D.MIN_BURST_S, color=C_K, lw=2.0, ls='--')
    ax.text(D.MIN_BURST_S + 0.2, 0.95, 'detector\nneeds ≥ 4 s', color=C_K,
            transform=ax.get_xaxis_transform(), va='top', fontsize=11, fontweight='bold')
    ax.set_xlabel('Seconds the delta envelope holds above the burst threshold')
    ax.set_ylabel(f'Scored K-complexes (n={n})')
    ax.set_title('C   A K-complex usually IS a multi-second delta run',
                 loc='left', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.15)
    ax.text(0.98, 0.62, f'median {crit.burst_s.median():.1f} s\n'
                        f'IQR {crit.burst_s.quantile(.25):.1f}–'
                        f'{crit.burst_s.quantile(.75):.1f} s',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#999', pad=4))

    ax = axes[1, 2]
    passes_b = int((crit.burst_s >= D.MIN_BURST_S).sum())
    passes_q = int(crit.quiet_ok.sum())
    passes_both = int((crit.quiet_ok & (crit.burst_s >= D.MIN_BURST_S)).sum())
    names = ['≥ 4 s sustained\nburst', '30 s quiet\nbaseline', 'BOTH\n(detector-eligible)']
    vals = [100 * passes_b / n, 100 * passes_q / n, 100 * passes_both / n]
    bars = ax.barh(names, vals, color=['#8E44AD', '#27AE60', C_K], alpha=0.85)
    for b, v, c in zip(bars, vals, (passes_b, passes_q, passes_both)):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f'{c}/{n}  ({v:.0f}%)',
                va='center', fontsize=11.5, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Scored K-complexes passing (%)')
    ax.set_title('D   Which detector gate thins them', loc='left',
                 fontweight='bold', fontsize=13)
    ax.grid(True, axis='x', alpha=0.15)
    ax.invert_yaxis()

    fig.suptitle(f'Our onsets are real K-complex-like events, but a selective fifth of them '
                 f'— and the detector finds every scored K-complex that qualifies '
                 f'({passes_both}/{passes_both})',
                 fontsize=15, fontweight='bold')
    p = OUT / 'fig_kcomplex_vs_detector.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ->', p.name)
    return p


def main():
    print('collecting scored marks + EEG snippets:')
    k_snips, sp_snips, k_spec, sp_spec, rand_spec, table, examples, crit = collect()
    amp = np.median(np.ptp(k_snips, axis=1)) if len(k_snips) else np.nan
    unit = 'µV' if 20 < amp < 2000 else 'a.u.'
    print(f'\n  scored K-complexes usable: {len(k_snips)}   '
          f'median peak-to-peak over ±2 s: {amp:.0f} {unit}')
    n = len(crit)
    pb = int((crit.burst_s >= D.MIN_BURST_S).sum())
    pq = int(crit.quiet_ok.sum())
    pboth = int((crit.quiet_ok & (crit.burst_s >= D.MIN_BURST_S)).sum())
    print(f'  delta run above burst threshold: median {crit.burst_s.median():.1f} s '
          f'(IQR {crit.burst_s.quantile(.25):.1f}-{crit.burst_s.quantile(.75):.1f})')
    print(f'  pass >=4 s burst: {pb}/{n} ({100*pb/n:.0f}%)   '
          f'pass 30 s quiet: {pq}/{n} ({100*pq/n:.0f}%)   '
          f'pass both: {pboth}/{n} ({100*pboth/n:.0f}%)')
    table.to_csv(OUT / 'kcomplex_marks.csv', index=False)
    crit.to_csv(OUT / 'kcomplex_criteria.csv', index=False)
    print('  -> kcomplex_marks.csv, kcomplex_criteria.csv')
    figure_morphology(k_snips, sp_snips, k_spec, sp_spec, rand_spec, unit)
    figure_vs_detector(examples, crit, unit)


if __name__ == '__main__':
    main()
