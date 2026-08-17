"""
Diagnostic plot: EEG delta power + hypnogram over sleep mask spectrograms.

Layout (top to bottom):
1. Hypnogram (PSG sleep stages)
2. EEG delta power (1-4.5 Hz) time series
3. EEG spectrogram (0-8 Hz) — reference
4. CLE-CRE spectrogram — primary cap channel
5. CLE spectrogram
6. CRE spectrogram
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import spectrogram as sp_spectrogram, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from swa_pipeline import bandpass_fir, EPOCH_SEC

OUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FMAX = 8.0
DELTA_BAND = (1.0, 4.5)


def smooth_spectrogram(sig, fs):
    nperseg = min(2048, len(sig) // 4)
    noverlap = int(nperseg * 0.875)
    f, t, Sxx = sp_spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                window='hamming', mode='psd')
    return f, t, Sxx


def compute_delta_power(sig, fs, epoch_sec=EPOCH_SEC):
    sig_filt = bandpass_fir(sig, fs)
    epoch_samp = int(epoch_sec * fs)
    n_ep = len(sig_filt) // epoch_samp
    sig_trim = sig_filt[:n_ep * epoch_samp].reshape(n_ep, epoch_samp)
    freqs, psd = welch(sig_trim, fs=fs, window='hamming',
                       nperseg=epoch_samp, noverlap=0, axis=1)
    df = freqs[1] - freqs[0]
    mask = (freqs >= DELTA_BAND[0]) & (freqs <= DELTA_BAND[1])
    delta = np.sum(psd[:, mask], axis=1) * df
    t_min = (np.arange(n_ep) * epoch_sec + epoch_sec / 2) / 60.0
    return t_min, delta


def plot_session(idx):
    m = SESSION_META[idx]
    s = load_session(idx)
    fs = s.fs

    eeg = s.psg['EEG'].astype(np.float64)
    cle = s.cap['CLE'].astype(np.float64)
    cre = s.cap['CRE'].astype(np.float64)
    diff = cle - cre

    t_delta, delta_power = compute_delta_power(eeg, fs)

    # Hypnogram from sleep profile
    sp = load_sleep_profile(s)
    has_hyp = sp is not None and len(sp.get('labels', [])) > 0

    nrows = 6 if has_hyp else 5
    ratios = [0.6, 1, 2, 2, 2, 2] if has_hyp else [1, 2, 2, 2, 2]

    fig, axes = plt.subplots(nrows, 1, figsize=(24, nrows * 2.8),
                             gridspec_kw={'height_ratios': ratios})
    fig.suptitle(f'{m["label"]}  ({m["subject"]} {m["date"]})',
                 fontsize=16, fontweight='bold', y=0.995)

    ax_idx = 0

    # Hypnogram
    if has_hyp:
        ax = axes[ax_idx]; ax_idx += 1
        stage_map = {'W': 0, 'Wake': 0, 'R': 1, 'REM': 1,
                     'N1': 2, 'N2': 3, 'N3': 4}
        stage_labels_y = ['W', 'R', 'N1', 'N2', 'N3']
        labels = sp['labels']
        t_hr = sp['t_ep_hr']
        stage_t = [t * 60 for t in t_hr]  # hours → minutes
        stage_vals = [stage_map.get(str(lbl), 0) for lbl in labels]

        ax.step(stage_t, stage_vals, where='post', color='#333333', lw=1.2)
        for i in range(len(stage_vals)):
            if stage_vals[i] == 4:
                dt = (stage_t[i+1] - stage_t[i]) if i + 1 < len(stage_t) else 0.5
                ax.axvspan(stage_t[i], stage_t[i] + dt, alpha=0.25, color='#2196F3', lw=0)
        ax.set_yticks(range(5))
        ax.set_yticklabels(stage_labels_y, fontsize=10)
        ax.set_ylabel('Stage', fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, t_delta[-1])
        ax.tick_params(labelbottom=False)
        ax.grid(axis='x', alpha=0.3)

    # EEG delta power
    ax = axes[ax_idx]; ax_idx += 1
    ax.fill_between(t_delta, delta_power, alpha=0.4, color='#1565C0')
    ax.plot(t_delta, delta_power, color='#1565C0', lw=0.5)
    ax.set_ylabel('EEG Delta\n1–4.5 Hz\n(µV²/Hz)', fontsize=10)
    ax.set_xlim(0, t_delta[-1])
    ax.tick_params(labelbottom=False)
    ax.grid(axis='x', alpha=0.3)
    p95 = np.percentile(delta_power, 95)
    ax.set_ylim(0, p95 * 1.3)

    # Spectrograms: EEG, CLE-CRE, CLE, CRE
    channel_data = [
        ('EEG (contact)', eeg, '#E65100'),
        ('CLE − CRE (cap diff)', diff, '#E65100'),
        ('CLE (cap left)', cle, '#E65100'),
        ('CRE (cap right)', cre, '#E65100'),
    ]

    for i, (label, sig, _) in enumerate(channel_data):
        ax = axes[ax_idx]; ax_idx += 1
        f, t, Sxx = smooth_spectrogram(sig, fs)
        fmask = f <= FMAX
        t_min = t / 60.0

        log_Sxx = 10 * np.log10(np.maximum(Sxx[fmask], 1e-20))
        vmin = np.percentile(log_Sxx, 5)
        vmax = np.percentile(log_Sxx, 95)

        im = ax.pcolormesh(t_min, f[fmask], log_Sxx, shading='gouraud',
                           cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_ylabel(f'{label}\nFreq (Hz)', fontsize=10)
        ax.set_ylim(0, FMAX)
        ax.set_xlim(0, t_delta[-1])

        # delta band markers
        ax.axhline(1.0, color='white', ls='--', lw=0.5, alpha=0.5)
        ax.axhline(4.5, color='white', ls='--', lw=0.5, alpha=0.5)

        if i < len(channel_data) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time (minutes)', fontsize=11)

        ax.grid(axis='x', alpha=0.15, color='white')

    plt.tight_layout(h_pad=0.4)
    out_path = OUT_DIR / f'diagnostic_{m["label"]}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_path}')
    return out_path


def plot_session_zoomed(idx, t_start_min=0, t_end_min=None):
    """Same as plot_session but zoomed to a time window."""
    m = SESSION_META[idx]
    s = load_session(idx)
    fs = s.fs

    eeg = s.psg['EEG'].astype(np.float64)
    cle = s.cap['CLE'].astype(np.float64)
    cre = s.cap['CRE'].astype(np.float64)
    diff = cle - cre

    t_delta, delta_power = compute_delta_power(eeg, fs)

    sp = load_sleep_profile(s)
    has_hyp = sp is not None and len(sp.get('labels', [])) > 0

    if t_end_min is None:
        t_end_min = t_delta[-1]

    nrows = 6 if has_hyp else 5
    ratios = [0.8, 1.2, 2.5, 2.5, 2.5, 2.5] if has_hyp else [1.2, 2.5, 2.5, 2.5, 2.5]

    fig, axes = plt.subplots(nrows, 1, figsize=(28, nrows * 3.2),
                             gridspec_kw={'height_ratios': ratios})
    fig.suptitle(f'{m["label"]}  ({m["subject"]} {m["date"]})  —  {t_start_min:.0f}–{t_end_min:.0f} min',
                 fontsize=18, fontweight='bold', y=0.995)

    ax_idx = 0

    if has_hyp:
        ax = axes[ax_idx]; ax_idx += 1
        stage_map = {'W': 0, 'Wake': 0, 'R': 1, 'REM': 1,
                     'N1': 2, 'N2': 3, 'N3': 4}
        stage_labels_y = ['W', 'R', 'N1', 'N2', 'N3']
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
        ax.set_yticklabels(stage_labels_y, fontsize=13)
        ax.set_ylabel('Stage', fontsize=13)
        ax.invert_yaxis()
        ax.set_xlim(t_start_min, t_end_min)
        ax.tick_params(labelbottom=False, labelsize=11)
        ax.grid(axis='x', alpha=0.3)

    # EEG delta power
    ax = axes[ax_idx]; ax_idx += 1
    mask_t = (t_delta >= t_start_min) & (t_delta <= t_end_min)
    ax.fill_between(t_delta[mask_t], delta_power[mask_t], alpha=0.4, color='#1565C0')
    ax.plot(t_delta[mask_t], delta_power[mask_t], color='#1565C0', lw=0.6)
    ax.set_ylabel('EEG Delta\n1–4.5 Hz\n(µV²/Hz)', fontsize=13)
    ax.set_xlim(t_start_min, t_end_min)
    ax.tick_params(labelbottom=False, labelsize=11)
    ax.grid(axis='x', alpha=0.3)
    visible_delta = delta_power[mask_t]
    if len(visible_delta) > 0:
        p95 = np.percentile(visible_delta, 95)
        ax.set_ylim(0, p95 * 1.3)

    channel_data = [
        ('EEG (contact)', eeg),
        ('CLE − CRE (cap diff)', diff),
        ('CLE (cap left)', cle),
        ('CRE (cap right)', cre),
    ]

    for i, (label, sig) in enumerate(channel_data):
        ax = axes[ax_idx]; ax_idx += 1
        f, t, Sxx = smooth_spectrogram(sig, fs)
        fmask = f <= FMAX
        t_min = t / 60.0

        log_Sxx = 10 * np.log10(np.maximum(Sxx[fmask], 1e-20))
        t_vis = (t_min >= t_start_min) & (t_min <= t_end_min)
        vis_data = log_Sxx[:, t_vis] if t_vis.any() else log_Sxx
        vmin = np.percentile(vis_data, 5)
        vmax = np.percentile(vis_data, 95)

        im = ax.pcolormesh(t_min, f[fmask], log_Sxx, shading='gouraud',
                           cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_ylabel(f'{label}\nFreq (Hz)', fontsize=13)
        ax.set_ylim(0, FMAX)
        ax.set_xlim(t_start_min, t_end_min)

        ax.axhline(1.0, color='white', ls='--', lw=0.7, alpha=0.6)
        ax.axhline(4.5, color='white', ls='--', lw=0.7, alpha=0.6)

        if i < len(channel_data) - 1:
            ax.tick_params(labelbottom=False, labelsize=11)
        else:
            ax.set_xlabel('Time (minutes)', fontsize=13)
            ax.tick_params(labelsize=11)

        ax.grid(axis='x', alpha=0.15, color='white')

    plt.tight_layout(h_pad=0.5)
    suffix = f'_{t_start_min:.0f}_{t_end_min:.0f}min'
    out_path = OUT_DIR / f'diagnostic_{m["label"]}{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_path}')
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, default=None)
    parser.add_argument('--start', type=float, default=None)
    parser.add_argument('--end', type=float, default=None)
    args = parser.parse_args()
    if args.session is not None:
        if args.start is not None or args.end is not None:
            plot_session_zoomed(args.session, args.start or 0, args.end)
        else:
            plot_session(args.session)
    else:
        for idx in range(12):
            plot_session(idx)
