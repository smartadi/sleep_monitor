"""
EEG delta power time series vs all-channel smooth spectrograms.

For each session: top row = EEG delta (1-4.5 Hz) band power,
rows below = smooth spectrograms for EEG, CLE, CRE, CH, CLE-CRE.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor.loader import load_session
from sleep_monitor.sessions import SESSION_META
from swa_pipeline import bandpass_fir, EPOCH_SEC

OUT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'swa_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPEC_NPERSEG = 1024
SPEC_NOVERLAP = 896
FMAX = 8.0
DELTA_BAND = (1.0, 4.5)
CHANNELS = ['EEG', 'CLE', 'CRE', 'CH', 'CLE-CRE']


def smooth_spectrogram(sig, fs, nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP):
    from scipy.signal import spectrogram as sp_spectrogram
    f, t, Sxx = sp_spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                window='hamming', mode='psd')
    return f, t, Sxx


def compute_delta_power_timeseries(sig, fs, epoch_sec=EPOCH_SEC):
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

    sigs = {
        'EEG': s.psg['EEG'].astype(np.float64),
        'CLE': s.cap['CLE'].astype(np.float64),
        'CRE': s.cap['CRE'].astype(np.float64),
        'CH':  s.cap['CH'].astype(np.float64),
        'CLE-CRE': (s.cap['CLE'] - s.cap['CRE']).astype(np.float64),
    }

    t_delta, delta_power = compute_delta_power_timeseries(sigs['EEG'], fs)

    fig, axes = plt.subplots(6, 1, figsize=(20, 18),
                             gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5, 1.5, 1.5]})

    # Row 0: EEG delta power
    ax = axes[0]
    ax.plot(t_delta, delta_power, color='#1f77b4', lw=0.6)
    ax.set_ylabel('EEG Delta\n(1-4.5 Hz)\nPower')
    ax.set_xlim(0, t_delta[-1])
    ax.set_title(f'{m["label"]}  ({m["subject"]} {m["date"]})', fontsize=14, fontweight='bold')
    ax.tick_params(labelbottom=False)

    # Rows 1-5: spectrograms
    for i, ch_name in enumerate(CHANNELS):
        ax = axes[i + 1]
        sig = sigs[ch_name]
        f, t, Sxx = smooth_spectrogram(sig, fs)

        fmask = f <= FMAX
        t_min = t / 60.0

        log_Sxx = 10 * np.log10(np.maximum(Sxx[fmask], 1e-20))
        vmin = np.percentile(log_Sxx, 2)
        vmax = np.percentile(log_Sxx, 98)

        ax.pcolormesh(t_min, f[fmask], log_Sxx, shading='gouraud',
                      cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_ylabel(f'{ch_name}\nFreq (Hz)')
        ax.set_ylim(0, FMAX)
        ax.set_xlim(0, t_delta[-1])

        if i < len(CHANNELS) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time (minutes)')

    plt.tight_layout(h_pad=0.3)
    out_path = OUT_DIR / f'eeg_delta_vs_spectrograms_{m["label"]}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_path.name}')


def main():
    print(f'Output: {OUT_DIR}/')
    for idx in range(12):
        plot_session(idx)
    print('Done.')


if __name__ == '__main__':
    main()
