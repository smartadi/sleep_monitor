"""
Diagnostic plot: EEG delta + CAP low/high ratio overlaid on spectrograms.

Layout:
1. Hypnogram
2. EEG delta power (1-4.5 Hz)
3. CAP low/high power ratio (0.5-2 Hz / 2-8 Hz)  [CLE-CRE]
4. Dual overlay: EEG delta (blue) vs CAP ratio (cyan) on shared axes
5. EEG spectrogram with delta power overlay (cyan)
6. CLE-CRE spectrogram with ratio overlay (cyan)
7. CLE spectrogram with ratio overlay (cyan)
8. CRE spectrogram with ratio overlay (cyan)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as sp_spectrogram, welch
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from swa_pipeline import bandpass_fir, EPOCH_SEC

OUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FMAX = 8.0
SMOOTH_N = 10  # 60s smoothing

# High-contrast colors
CLR_DELTA = '#FFD700'       # gold — EEG delta on spectrograms (pops against inferno)
CLR_RATIO = '#00FFFF'       # cyan — CAP ratio on spectrograms
CLR_DELTA_PANEL = '#1565C0' # blue — EEG delta standalone panel
CLR_RATIO_PANEL = '#E65100' # orange — CAP ratio standalone panel
# Comparison row: distinct per-channel ratio colors
CLR_COMP_EEG  = '#FFD700'   # gold — EEG delta
CLR_COMP_DIFF = '#00FFFF'   # cyan — CLE-CRE
CLR_COMP_CLE  = '#FF69B4'   # hot pink — CLE
CLR_COMP_CRE  = '#FF6633'   # orange-red — CRE


def smooth_spectrogram(sig, fs):
    nperseg = min(2048, len(sig) // 4)
    noverlap = int(nperseg * 0.875)
    f, t, Sxx = sp_spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                window='hamming', mode='psd')
    return f, t, Sxx


def compute_band_power(sig, fs, lo, hi, epoch_sec=EPOCH_SEC):
    sig_filt = bandpass_fir(sig, fs)
    epoch_samp = int(epoch_sec * fs)
    n_ep = len(sig_filt) // epoch_samp
    sig_trim = sig_filt[:n_ep * epoch_samp].reshape(n_ep, epoch_samp)
    freqs, psd = welch(sig_trim, fs=fs, window='hamming',
                       nperseg=epoch_samp, noverlap=0, axis=1)
    df = freqs[1] - freqs[0]
    mask = (freqs >= lo) & (freqs <= hi)
    power = np.sum(psd[:, mask], axis=1) * df
    t_min = (np.arange(n_ep) * epoch_sec + epoch_sec / 2) / 60.0
    return t_min, power


def plot_session(idx, t_start_min=None, t_end_min=None):
    m = SESSION_META[idx]
    s = load_session(idx)
    fs = s.fs

    eeg = s.psg['EEG'].astype(np.float64)
    cle = s.cap['CLE'].astype(np.float64)
    cre = s.cap['CRE'].astype(np.float64)
    diff = cle - cre

    # Band powers from EEG
    t_ep, eeg_delta = compute_band_power(eeg, fs, 1.0, 4.5)

    # Band powers from each cap channel
    cap_channels = {
        'CLE-CRE': diff,
        'CLE': cle,
        'CRE': cre,
    }
    cap_ratios = {}
    for ch_name, sig in cap_channels.items():
        min_len = min(len(sig), len(eeg))
        _, cap_low = compute_band_power(sig[:min_len], fs, 0.5, 2.0)
        _, cap_high = compute_band_power(sig[:min_len], fs, 2.0, 8.0)
        n = min(len(cap_low), len(cap_high), len(t_ep))
        ratio = cap_low[:n] / np.maximum(cap_high[:n], 1e-20)
        cap_ratios[ch_name] = ratio

    n_ep = min(len(t_ep), *[len(r) for r in cap_ratios.values()])
    t_ep = t_ep[:n_ep]
    eeg_delta = eeg_delta[:n_ep]
    for k in cap_ratios:
        cap_ratios[k] = cap_ratios[k][:n_ep]

    if t_start_min is None:
        t_start_min = 0
    if t_end_min is None:
        t_end_min = t_ep[-1]

    # Smoothed versions for overlay
    eeg_delta_s = uniform_filter1d(eeg_delta, SMOOTH_N)
    ratio_s = {k: uniform_filter1d(v, SMOOTH_N) for k, v in cap_ratios.items()}

    # Clip ratio for display (cap outliers from motion)
    for k in ratio_s:
        p99 = np.percentile(ratio_s[k], 99)
        ratio_s[k] = np.clip(ratio_s[k], 0, p99 * 1.5)

    # Normalize for overlay comparison
    vis = (t_ep >= t_start_min) & (t_ep <= t_end_min)
    def znorm(x):
        v = x[vis]
        return (x - np.mean(v)) / max(np.std(v), 1e-20)

    eeg_delta_z = znorm(eeg_delta_s)
    ratio_z = {k: znorm(v) for k, v in ratio_s.items()}

    # Hypnogram
    sp = load_sleep_profile(s)
    has_hyp = sp is not None and len(sp.get('labels', [])) > 0

    nrows = 8 if has_hyp else 7
    ratios_h = [0.5, 0.9, 0.9, 1.2, 2.2, 2.2, 2.2, 2.2] if has_hyp else [0.9, 0.9, 1.2, 2.2, 2.2, 2.2, 2.2]

    fig, axes = plt.subplots(nrows, 1, figsize=(30, nrows * 2.8),
                             gridspec_kw={'height_ratios': ratios_h})
    title_str = f'{m["label"]}  ({m["subject"]} {m["date"]})'
    if t_start_min > 0 or t_end_min < t_ep[-1]:
        title_str += f'  —  {t_start_min:.0f}–{t_end_min:.0f} min'
    fig.suptitle(title_str, fontsize=18, fontweight='bold', y=0.998)

    ax_idx = 0

    # ── Row: Hypnogram ──
    if has_hyp:
        ax = axes[ax_idx]; ax_idx += 1
        stage_map = {'W': 0, 'Wake': 0, 'R': 1, 'REM': 1,
                     'N1': 2, 'N2': 3, 'N3': 4}
        labels = sp['labels']
        t_hr = sp['t_ep_hr']
        stage_t = [t * 60 for t in t_hr]
        stage_vals = [stage_map.get(str(lbl), 0) for lbl in labels]

        ax.step(stage_t, stage_vals, where='post', color='#333333', lw=1.5)
        for i in range(len(stage_vals)):
            if stage_vals[i] == 4:
                dt = (stage_t[i+1] - stage_t[i]) if i + 1 < len(stage_t) else 0.5
                ax.axvspan(stage_t[i], stage_t[i] + dt, alpha=0.3, color='#2196F3', lw=0)
        ax.set_yticks(range(5))
        ax.set_yticklabels(['W', 'R', 'N1', 'N2', 'N3'], fontsize=12)
        ax.set_ylabel('Stage', fontsize=13)
        ax.invert_yaxis()
        ax.set_xlim(t_start_min, t_end_min)
        ax.tick_params(labelbottom=False, labelsize=11)
        ax.grid(axis='x', alpha=0.3)

    # ── Row: EEG delta power ──
    ax = axes[ax_idx]; ax_idx += 1
    ax.fill_between(t_ep[vis], eeg_delta_s[vis], alpha=0.3, color=CLR_DELTA_PANEL)
    ax.plot(t_ep[vis], eeg_delta_s[vis], color=CLR_DELTA_PANEL, lw=0.8)
    ax.set_ylabel('EEG Delta\n1–4.5 Hz', fontsize=13, color=CLR_DELTA_PANEL)
    ax.set_xlim(t_start_min, t_end_min)
    ax.tick_params(labelbottom=False, labelsize=11)
    ax.grid(axis='x', alpha=0.3)
    if vis.sum() > 0:
        p95 = np.percentile(eeg_delta_s[vis], 95)
        ax.set_ylim(0, p95 * 1.3)

    # ── Row: CAP low/high ratio (CLE-CRE) ──
    ax = axes[ax_idx]; ax_idx += 1
    r_vis = ratio_s['CLE-CRE']
    ax.fill_between(t_ep[vis], r_vis[vis], alpha=0.3, color=CLR_RATIO_PANEL)
    ax.plot(t_ep[vis], r_vis[vis], color=CLR_RATIO_PANEL, lw=0.8)
    ax.set_ylabel('CAP Ratio\nLow/High\n(0.5-2 / 2-8 Hz)', fontsize=13, color=CLR_RATIO_PANEL)
    ax.set_xlim(t_start_min, t_end_min)
    ax.tick_params(labelbottom=False, labelsize=11)
    ax.grid(axis='x', alpha=0.3)
    if vis.sum() > 0:
        p95 = np.percentile(r_vis[vis], 95)
        ax.set_ylim(0, p95 * 1.3)

    # ── Row: Multi-trace comparison — z-scored EEG delta vs all CAP ratios ──
    ax = axes[ax_idx]; ax_idx += 1
    ax.set_facecolor('#1a1a2e')
    ax.plot(t_ep[vis], eeg_delta_z[vis], color=CLR_COMP_EEG, lw=1.2,
            alpha=0.95, label='EEG delta')
    ax.plot(t_ep[vis], ratio_z['CLE-CRE'][vis], color=CLR_COMP_DIFF, lw=1.0,
            alpha=0.85, label='CLE−CRE ratio')
    ax.plot(t_ep[vis], ratio_z['CLE'][vis], color=CLR_COMP_CLE, lw=0.8,
            alpha=0.7, label='CLE ratio')
    ax.plot(t_ep[vis], ratio_z['CRE'][vis], color=CLR_COMP_CRE, lw=0.8,
            alpha=0.7, label='CRE ratio')
    ax.set_ylabel('Z-scored\ncomparison', fontsize=13)
    ax.set_xlim(t_start_min, t_end_min)
    ax.tick_params(labelbottom=False, labelsize=11, colors='white')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.85,
              facecolor='#2a2a3e', edgecolor='white', labelcolor='white')
    ax.grid(axis='x', alpha=0.2, color='white')
    ax.grid(axis='y', alpha=0.15, color='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.yaxis.label.set_color('white')
    if vis.sum() > 0:
        all_z = np.concatenate([eeg_delta_z[vis], ratio_z['CLE-CRE'][vis],
                                ratio_z['CLE'][vis], ratio_z['CRE'][vis]])
        ymax = np.percentile(np.abs(all_z), 98)
        ax.set_ylim(-ymax * 1.1, ymax * 1.1)

    # ── Rows: Spectrograms with overlays ──
    spec_channels = [
        ('EEG (contact)', eeg, eeg_delta_s, CLR_DELTA, 'Delta'),
        ('CLE − CRE (cap diff)', diff, ratio_s['CLE-CRE'], CLR_RATIO, 'Ratio'),
        ('CLE (cap left)', cle, ratio_s.get('CLE', None), CLR_RATIO, 'Ratio'),
        ('CRE (cap right)', cre, ratio_s.get('CRE', None), CLR_RATIO, 'Ratio'),
    ]

    for i, (label, sig, overlay, overlay_color, overlay_label) in enumerate(spec_channels):
        ax = axes[ax_idx]; ax_idx += 1
        f, t, Sxx = smooth_spectrogram(sig, fs)
        fmask = f <= FMAX
        t_min_spec = t / 60.0

        log_Sxx = 10 * np.log10(np.maximum(Sxx[fmask], 1e-20))
        t_vis_spec = (t_min_spec >= t_start_min) & (t_min_spec <= t_end_min)
        vis_data = log_Sxx[:, t_vis_spec] if t_vis_spec.any() else log_Sxx
        vmin = np.percentile(vis_data, 5)
        vmax = np.percentile(vis_data, 95)

        ax.pcolormesh(t_min_spec, f[fmask], log_Sxx, shading='gouraud',
                      cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_ylabel(f'{label}\nFreq (Hz)', fontsize=13)
        ax.set_ylim(0, FMAX)
        ax.set_xlim(t_start_min, t_end_min)

        ax.axhline(1.0, color='white', ls='--', lw=0.5, alpha=0.4)
        ax.axhline(4.5, color='white', ls='--', lw=0.5, alpha=0.4)

        if overlay is not None:
            ax2 = ax.twinx()
            ax2.plot(t_ep[vis], overlay[vis], color=overlay_color,
                     lw=1.4, alpha=0.9)
            ax2.set_ylabel(overlay_label, fontsize=11, color=overlay_color)
            ax2.tick_params(axis='y', labelcolor=overlay_color, labelsize=9)
            if vis.sum() > 0:
                p95 = np.percentile(overlay[vis], 97)
                ax2.set_ylim(0, p95 * 1.5)

        if i < len(spec_channels) - 1:
            ax.tick_params(labelbottom=False, labelsize=11)
        else:
            ax.set_xlabel('Time (minutes)', fontsize=13)
            ax.tick_params(labelsize=11)

        ax.grid(axis='x', alpha=0.15, color='white')

    plt.tight_layout(h_pad=0.4)
    suffix = ''
    if t_start_min > 0 or t_end_min < t_ep[-1]:
        suffix = f'_{t_start_min:.0f}_{t_end_min:.0f}min'
    out_path = OUT_DIR / f'diagnostic_ratio_{m["label"]}{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_path}')
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, required=True)
    parser.add_argument('--start', type=float, default=None)
    parser.add_argument('--end', type=float, default=None)
    args = parser.parse_args()
    plot_session(args.session, args.start, args.end)
