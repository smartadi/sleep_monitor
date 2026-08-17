"""
Sweep band definitions and smoothing timescales to find where
CAP low-band power anti-correlates with EEG delta.

Dimensions:
- Low/high split frequency: 1.0, 1.5, 2.0, 2.5, 3.0 Hz
- High-band ceiling: 3, 4, 5, 6, 8 Hz
- Smoothing window: 1, 2, 5, 10, 20, 30 min
- All 12 sessions

Output: heatmaps of mean Spearman r across sessions.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor.loader import load_session
from sleep_monitor.sessions import SESSION_META
from swa_pipeline import bandpass_fir, EPOCH_SEC

OUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = [1.0, 1.5, 2.0, 2.5, 3.0]
CEILINGS = [3, 4, 5, 6, 8]
SMOOTH_MINS = [1, 2, 5, 10, 20, 30]
EPOCH_PER_MIN = 60.0 / EPOCH_SEC  # 10 epochs per minute


def compute_power(sig, fs, lo, hi, epoch_sec=EPOCH_SEC):
    sig_filt = bandpass_fir(sig, fs)
    epoch_samp = int(epoch_sec * fs)
    n_ep = len(sig_filt) // epoch_samp
    sig_trim = sig_filt[:n_ep * epoch_samp].reshape(n_ep, epoch_samp)
    freqs, psd = welch(sig_trim, fs=fs, window='hamming',
                       nperseg=epoch_samp, noverlap=0, axis=1)
    df = freqs[1] - freqs[0]
    mask = (freqs >= lo) & (freqs <= hi)
    return np.sum(psd[:, mask], axis=1) * df


def run_sweep():
    # Preload all sessions
    sessions = []
    for idx in range(12):
        m = SESSION_META[idx]
        s = load_session(idx)
        fs = s.fs
        eeg = s.psg['EEG'].astype(np.float64)
        diff = (s.cap['CLE'] - s.cap['CRE']).astype(np.float64)
        min_len = min(len(eeg), len(diff))
        eeg_delta = compute_power(eeg[:min_len], fs, 1.0, 4.5)
        sessions.append({
            'label': m['label'],
            'fs': fs,
            'diff': diff[:min_len],
            'eeg_delta': eeg_delta,
            'min_len': min_len,
        })
        print(f'  Loaded {m["label"]}: {len(eeg_delta)} epochs')

    # For each (split, ceiling), precompute CAP low and high power
    # Then for each smoothing window, compute correlation
    # Results: dict[(split, ceiling, smooth_min)] -> list of 12 r values
    results = {}

    for split in SPLITS:
        for ceil_hz in CEILINGS:
            if ceil_hz <= split:
                continue
            print(f'\n  Band: 0.5-{split} / {split}-{ceil_hz} Hz')

            per_session_powers = []
            for sess in sessions:
                cap_low = compute_power(sess['diff'], sess['fs'], 0.5, split)
                cap_high = compute_power(sess['diff'], sess['fs'], split, ceil_hz)
                n = min(len(cap_low), len(cap_high), len(sess['eeg_delta']))
                ratio = cap_low[:n] / np.maximum(cap_high[:n], 1e-20)
                per_session_powers.append({
                    'eeg_delta': sess['eeg_delta'][:n],
                    'cap_low': cap_low[:n],
                    'cap_ratio': ratio,
                    'label': sess['label'],
                })

            for sm in SMOOTH_MINS:
                sm_epochs = max(1, int(sm * EPOCH_PER_MIN))
                rs = []
                for sp in per_session_powers:
                    eeg_s = uniform_filter1d(sp['eeg_delta'], sm_epochs)
                    # Test both raw low power and ratio
                    cap_s = uniform_filter1d(sp['cap_low'], sm_epochs)
                    ratio_s = uniform_filter1d(sp['cap_ratio'], sm_epochs)

                    # Use log for better distribution
                    log_eeg = np.log10(np.maximum(eeg_s, 1e-20))
                    log_cap = np.log10(np.maximum(cap_s, 1e-20))
                    log_ratio = np.log10(np.maximum(ratio_s, 1e-20))

                    valid = np.isfinite(log_eeg) & np.isfinite(log_cap)
                    if valid.sum() > 20:
                        r_low, _ = spearmanr(log_eeg[valid], log_cap[valid])
                        r_ratio, _ = spearmanr(log_eeg[valid], log_ratio[valid])
                    else:
                        r_low, r_ratio = np.nan, np.nan
                    rs.append({'r_low': r_low, 'r_ratio': r_ratio,
                               'label': sp['label']})

                mean_r_low = np.nanmean([r['r_low'] for r in rs])
                mean_r_ratio = np.nanmean([r['r_ratio'] for r in rs])
                results[(split, ceil_hz, sm)] = {
                    'mean_r_low': mean_r_low,
                    'mean_r_ratio': mean_r_ratio,
                    'per_session': rs,
                }

    # Print summary table
    print('\n' + '='*100)
    print('SUMMARY: Mean Spearman r (EEG delta vs CAP low power) across 12 sessions')
    print('='*100)
    print(f'{"Split":>6} {"Ceil":>5} | ' + ' '.join(f'{sm:>6}min' for sm in SMOOTH_MINS))
    print('-' * 100)
    for split in SPLITS:
        for ceil_hz in CEILINGS:
            if ceil_hz <= split:
                continue
            vals = []
            for sm in SMOOTH_MINS:
                key = (split, ceil_hz, sm)
                if key in results:
                    vals.append(results[key]['mean_r_low'])
                else:
                    vals.append(np.nan)
            print(f'{split:>5.1f} {ceil_hz:>5} | ' +
                  ' '.join(f'{v:>+7.3f}' if not np.isnan(v) else '    nan' for v in vals))

    print('\n' + '='*100)
    print('SUMMARY: Mean Spearman r (EEG delta vs CAP low/high RATIO) across 12 sessions')
    print('='*100)
    print(f'{"Split":>6} {"Ceil":>5} | ' + ' '.join(f'{sm:>6}min' for sm in SMOOTH_MINS))
    print('-' * 100)
    for split in SPLITS:
        for ceil_hz in CEILINGS:
            if ceil_hz <= split:
                continue
            vals = []
            for sm in SMOOTH_MINS:
                key = (split, ceil_hz, sm)
                if key in results:
                    vals.append(results[key]['mean_r_ratio'])
                else:
                    vals.append(np.nan)
            print(f'{split:>5.1f} {ceil_hz:>5} | ' +
                  ' '.join(f'{v:>+7.3f}' if not np.isnan(v) else '    nan' for v in vals))

    # Find the best (most negative) combination
    best_key = min(results.keys(),
                   key=lambda k: results[k]['mean_r_low'] if not np.isnan(results[k]['mean_r_low']) else 0)
    best = results[best_key]
    print(f'\nBest anti-correlation (CAP low): split={best_key[0]}, ceil={best_key[1]}, '
          f'smooth={best_key[2]}min, r={best["mean_r_low"]:+.3f}')

    best_key_r = min(results.keys(),
                     key=lambda k: results[k]['mean_r_ratio'] if not np.isnan(results[k]['mean_r_ratio']) else 0)
    best_r = results[best_key_r]
    print(f'Best anti-correlation (ratio):   split={best_key_r[0]}, ceil={best_key_r[1]}, '
          f'smooth={best_key_r[2]}min, r={best_r["mean_r_ratio"]:+.3f}')

    # Per-session detail for the best
    print(f'\nPer-session detail for best (CAP low, split={best_key[0]}, ceil={best_key[1]}, smooth={best_key[2]}min):')
    for sr in best['per_session']:
        print(f'  {sr["label"]}: r_low={sr["r_low"]:+.3f}  r_ratio={sr["r_ratio"]:+.3f}')

    # Plot heatmaps
    plot_heatmaps(results)

    return results


def plot_heatmaps(results):
    # One heatmap per ceiling, x=smooth, y=split
    for metric, title_prefix in [('mean_r_low', 'EEG delta vs CAP low power'),
                                  ('mean_r_ratio', 'EEG delta vs CAP low/high ratio')]:
        fig, axes = plt.subplots(1, len(CEILINGS), figsize=(5 * len(CEILINGS), 5),
                                 sharey=True)
        fig.suptitle(f'Spearman r: {title_prefix}\n(mean across 12 sessions, negative = anti-correlated)',
                     fontsize=14, fontweight='bold')

        for ci, ceil_hz in enumerate(CEILINGS):
            ax = axes[ci]
            valid_splits = [s for s in SPLITS if s < ceil_hz]
            mat = np.full((len(valid_splits), len(SMOOTH_MINS)), np.nan)
            for si, split in enumerate(valid_splits):
                for smi, sm in enumerate(SMOOTH_MINS):
                    key = (split, ceil_hz, sm)
                    if key in results:
                        mat[si, smi] = results[key][metric]

            vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.1)
            im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           aspect='auto', origin='lower')
            ax.set_xticks(range(len(SMOOTH_MINS)))
            ax.set_xticklabels([f'{s}' for s in SMOOTH_MINS], fontsize=10)
            ax.set_xlabel('Smooth (min)')
            ax.set_yticks(range(len(valid_splits)))
            ax.set_yticklabels([f'{s:.1f}' for s in valid_splits], fontsize=10)
            if ci == 0:
                ax.set_ylabel('Low/High split (Hz)')
            ax.set_title(f'Ceiling = {ceil_hz} Hz', fontsize=12)

            for si in range(mat.shape[0]):
                for smi in range(mat.shape[1]):
                    if not np.isnan(mat[si, smi]):
                        color = 'white' if abs(mat[si, smi]) > vmax * 0.6 else 'black'
                        ax.text(smi, si, f'{mat[si, smi]:+.2f}',
                                ha='center', va='center', fontsize=8, color=color)

            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        suffix = 'low_power' if 'low' in metric else 'ratio'
        out = OUT_DIR / f'sweep_heatmap_{suffix}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {out}')


if __name__ == '__main__':
    run_sweep()
