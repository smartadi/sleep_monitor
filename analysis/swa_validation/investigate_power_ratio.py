"""
Investigate: does EEG delta power dip when cap low-band power jumps?

Compute per 6-sec epoch:
- EEG delta power (1-4.5 Hz)
- CAP low-band power (0.5-2 Hz) — where resp/cardiac fundamentals live
- CAP high-band power (2-8 Hz) — harmonics and other content
- CAP low/high ratio

Check correlation between EEG delta and CAP features, especially the ratio.
Report per-session and per-subject.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from swa_pipeline import bandpass_fir, EPOCH_SEC

OUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {
    'eeg_delta': (1.0, 4.5),
    'cap_low':   (0.5, 2.0),
    'cap_high':  (2.0, 8.0),
    'cap_delta': (1.0, 4.5),
}


def compute_band_powers(sig, fs, epoch_sec=EPOCH_SEC):
    sig_filt = bandpass_fir(sig, fs)
    epoch_samp = int(epoch_sec * fs)
    n_ep = len(sig_filt) // epoch_samp
    sig_trim = sig_filt[:n_ep * epoch_samp].reshape(n_ep, epoch_samp)
    freqs, psd = welch(sig_trim, fs=fs, window='hamming',
                       nperseg=epoch_samp, noverlap=0, axis=1)
    df = freqs[1] - freqs[0]

    powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        powers[name] = np.sum(psd[:, mask], axis=1) * df

    t_min = (np.arange(n_ep) * epoch_sec + epoch_sec / 2) / 60.0
    return t_min, powers


def analyze_session(idx):
    m = SESSION_META[idx]
    s = load_session(idx)
    fs = s.fs

    eeg = s.psg['EEG'].astype(np.float64)
    diff = (s.cap['CLE'] - s.cap['CRE']).astype(np.float64)

    # Ensure same length
    min_len = min(len(eeg), len(diff))
    eeg = eeg[:min_len]
    diff = diff[:min_len]

    t_eeg, p_eeg = compute_band_powers(eeg, fs)
    t_cap, p_cap = compute_band_powers(diff, fs)

    n_ep = min(len(t_eeg), len(t_cap))
    t = t_eeg[:n_ep]
    eeg_delta = p_eeg['eeg_delta'][:n_ep]
    cap_low = p_cap['cap_low'][:n_ep]
    cap_high = p_cap['cap_high'][:n_ep]
    cap_delta = p_cap['cap_delta'][:n_ep]

    cap_ratio = cap_low / np.maximum(cap_high, 1e-20)
    cap_high_low_ratio = cap_high / np.maximum(cap_low, 1e-20)

    # Log-transform for better distribution
    log_eeg = np.log10(np.maximum(eeg_delta, 1e-20))
    log_cap_low = np.log10(np.maximum(cap_low, 1e-20))
    log_cap_high = np.log10(np.maximum(cap_high, 1e-20))
    log_ratio = np.log10(np.maximum(cap_ratio, 1e-20))
    log_h2l = np.log10(np.maximum(cap_high_low_ratio, 1e-20))

    # Remove any inf/nan
    valid = np.isfinite(log_eeg) & np.isfinite(log_cap_low) & np.isfinite(log_cap_high) & np.isfinite(log_ratio)

    results = {}
    for name, cap_feat in [('cap_low', log_cap_low), ('cap_high', log_cap_high),
                            ('cap_low/high', log_ratio), ('cap_high/low', log_h2l),
                            ('cap_delta', np.log10(np.maximum(cap_delta[:n_ep], 1e-20)))]:
        v = valid & np.isfinite(cap_feat)
        if v.sum() > 10:
            r_p, p_p = pearsonr(log_eeg[v], cap_feat[v])
            r_s, p_s = spearmanr(log_eeg[v], cap_feat[v])
            results[name] = {'pearson_r': r_p, 'pearson_p': p_p,
                             'spearman_r': r_s, 'spearman_p': p_s}
        else:
            results[name] = {'pearson_r': np.nan, 'pearson_p': np.nan,
                             'spearman_r': np.nan, 'spearman_p': np.nan}

    # Load sleep profile for stage-specific analysis
    sp = load_sleep_profile(s)
    stage_results = None
    if sp is not None:
        labels = sp['labels']
        t_hr = sp['t_ep_hr']
        # Map 6-sec epochs to 30-sec stage labels
        stage_per_epoch = []
        for i in range(n_ep):
            t_sec = t[i] * 60
            ep30_idx = int(t_sec / 30)
            if ep30_idx < len(labels):
                stage_per_epoch.append(str(labels[ep30_idx]))
            else:
                stage_per_epoch.append('?')
        stage_per_epoch = np.array(stage_per_epoch)

        stage_results = {}
        for stage in ['N3', 'N2', 'N1', 'REM', 'Wake', 'W']:
            mask_st = stage_per_epoch == stage
            if mask_st.sum() > 10:
                v = mask_st & valid
                stage_results[stage] = {
                    'n_epochs': int(mask_st.sum()),
                    'mean_eeg_delta': float(np.mean(eeg_delta[mask_st])),
                    'mean_cap_low': float(np.mean(cap_low[mask_st])),
                    'mean_cap_high': float(np.mean(cap_high[mask_st])),
                    'mean_ratio': float(np.mean(cap_ratio[mask_st])),
                }

    return {
        'label': m['label'],
        'subject': m['subject'],
        'n_epochs': n_ep,
        'correlations': results,
        'stage_results': stage_results,
        'timeseries': {
            't': t, 'eeg_delta': eeg_delta,
            'cap_low': cap_low, 'cap_high': cap_high,
            'cap_ratio': cap_ratio,
        }
    }


def main():
    print('='*80)
    print('EEG delta vs CAP band power investigation')
    print('='*80)

    all_results = []
    for idx in range(12):
        r = analyze_session(idx)
        all_results.append(r)
        print(f'\n--- {r["label"]} ({r["subject"]}) — {r["n_epochs"]} epochs ---')
        for feat, corr in r['correlations'].items():
            sig = '***' if abs(corr['spearman_p']) < 0.001 else '**' if abs(corr['spearman_p']) < 0.01 else '*' if abs(corr['spearman_p']) < 0.05 else ''
            print(f'  EEG_delta vs {feat:15s}: r_s={corr["spearman_r"]:+.3f} {sig:3s}  r_p={corr["pearson_r"]:+.3f}')

        if r['stage_results']:
            print(f'  Stage breakdown:')
            for st, sr in r['stage_results'].items():
                print(f'    {st:4s}: n={sr["n_epochs"]:5d}  eeg_d={sr["mean_eeg_delta"]:.1f}  cap_lo={sr["mean_cap_low"]:.4f}  cap_hi={sr["mean_cap_high"]:.4f}  ratio={sr["mean_ratio"]:.3f}')

    # Summary across sessions
    print('\n' + '='*80)
    print('SUMMARY — mean correlation across all 12 sessions')
    print('='*80)
    for feat in ['cap_low', 'cap_high', 'cap_low/high', 'cap_high/low', 'cap_delta']:
        rs = [r['correlations'][feat]['spearman_r'] for r in all_results if not np.isnan(r['correlations'][feat]['spearman_r'])]
        rp = [r['correlations'][feat]['pearson_r'] for r in all_results if not np.isnan(r['correlations'][feat]['pearson_r'])]
        print(f'  EEG_delta vs {feat:15s}: r_s={np.mean(rs):+.3f} ± {np.std(rs):.3f}  r_p={np.mean(rp):+.3f} ± {np.std(rp):.3f}  (n={len(rs)})')

    # Plot for best session
    plot_diagnostic_ratio(all_results)

    return all_results


def plot_diagnostic_ratio(all_results):
    """Scatter + time-series for the session with strongest ratio correlation."""
    # Find session with strongest absolute spearman for cap_low/high
    best_idx = max(range(len(all_results)),
                   key=lambda i: abs(all_results[i]['correlations']['cap_low/high']['spearman_r']))
    r = all_results[best_idx]
    ts = r['timeseries']

    fig, axes = plt.subplots(4, 1, figsize=(24, 16),
                             gridspec_kw={'height_ratios': [1.5, 1, 1, 1]})
    fig.suptitle(f'{r["label"]} — EEG delta vs CAP power ratio diagnostic',
                 fontsize=16, fontweight='bold')

    # Smoothing for visual clarity
    from scipy.ndimage import uniform_filter1d
    smooth_n = 10  # 60 sec smoothing (10 x 6s epochs)

    t = ts['t']
    eeg_s = uniform_filter1d(ts['eeg_delta'], smooth_n)
    cap_lo_s = uniform_filter1d(ts['cap_low'], smooth_n)
    cap_hi_s = uniform_filter1d(ts['cap_high'], smooth_n)
    ratio_s = uniform_filter1d(ts['cap_ratio'], smooth_n)

    # Row 0: overlay EEG delta and cap ratio (dual y-axis)
    ax = axes[0]
    ax.fill_between(t, eeg_s, alpha=0.3, color='#1565C0', label='EEG delta (1-4.5 Hz)')
    ax.plot(t, eeg_s, color='#1565C0', lw=0.8)
    ax.set_ylabel('EEG Delta Power', color='#1565C0', fontsize=12)
    ax.tick_params(axis='y', labelcolor='#1565C0')
    ax2 = ax.twinx()
    ax2.plot(t, ratio_s, color='#E65100', lw=0.8, label='CAP low/high ratio')
    ax2.set_ylabel('CAP Low/High Ratio', color='#E65100', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#E65100')
    ax.set_xlim(t[0], t[-1])
    ax.set_title('EEG delta power vs CAP low/high power ratio (60s smoothed)', fontsize=13)
    ax.tick_params(labelbottom=False)

    # Row 1: EEG delta
    ax = axes[1]
    ax.fill_between(t, eeg_s, alpha=0.4, color='#1565C0')
    ax.plot(t, eeg_s, color='#1565C0', lw=0.5)
    ax.set_ylabel('EEG Delta\n1–4.5 Hz', fontsize=11)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelbottom=False)

    # Row 2: CAP low
    ax = axes[2]
    ax.fill_between(t, cap_lo_s, alpha=0.4, color='#2E7D32')
    ax.plot(t, cap_lo_s, color='#2E7D32', lw=0.5)
    ax.set_ylabel('CAP Low\n0.5–2 Hz', fontsize=11)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelbottom=False)

    # Row 3: CAP high
    ax = axes[3]
    ax.fill_between(t, cap_hi_s, alpha=0.4, color='#C62828')
    ax.plot(t, cap_hi_s, color='#C62828', lw=0.5)
    ax.set_ylabel('CAP High\n2–8 Hz', fontsize=11)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time (minutes)', fontsize=12)

    plt.tight_layout()
    out = OUT_DIR / 'power_ratio_diagnostic.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {out}')

    # Scatter plot: all sessions
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle('EEG delta vs CAP low/high ratio — all 12 sessions (log-log, 60s smoothed)',
                 fontsize=15, fontweight='bold')
    for i, r in enumerate(all_results):
        ax = axes[i // 4, i % 4]
        ts = r['timeseries']
        from scipy.ndimage import uniform_filter1d
        eeg_s = uniform_filter1d(ts['eeg_delta'], 10)
        ratio_s = uniform_filter1d(ts['cap_ratio'], 10)
        valid = (eeg_s > 0) & (ratio_s > 0)
        ax.scatter(np.log10(ratio_s[valid]), np.log10(eeg_s[valid]),
                   s=1, alpha=0.3, c='#1565C0')
        corr = r['correlations']['cap_low/high']
        ax.set_title(f'{r["label"]}\nr_s={corr["spearman_r"]:+.3f}', fontsize=11)
        ax.set_xlabel('log CAP low/high')
        ax.set_ylabel('log EEG delta')
    plt.tight_layout()
    out2 = OUT_DIR / 'power_ratio_scatter_all.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out2}')


if __name__ == '__main__':
    main()
