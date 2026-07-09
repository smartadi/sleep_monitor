"""
Example spindle windows: raw + sigma-filtered EEG and CAP channels around
individual PSG-scored spindles.

The aggregate triggered-average figures show CAP is flat *on average*. This
script zooms into single spindle events so we can look for anything the average
would wash out (phase-varying bursts, transient morphology). For each of the
strongest N2 spindles in a session it plots, in a shared time window:
    row 1 : EEG raw           (spindle span shaded)
    row 2 : EEG 11-16 Hz      (the spindle, unmistakable)
    row 3 : CAP CLE-CRE raw
    row 4 : CAP CLE-CRE 11-16 Hz
    row 5 : CAP CLE-CRE sigma spectrogram (is there ANY transient sigma power?)

Usage:  python -m analysis.spindles.plot_spindle_windows [session_idx] [n_examples]
Outputs -> analysis/spindles/outputs/fig_spindle_windows_<label>.png
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, spectrogram

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles

FS = 100.0
SIGMA_LO, SIGMA_HI = 11.0, 16.0
N2_CODE = 2
WIN_HALF = 3.0          # +/- s plotted around each spindle center
OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)


def bp(sig, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return filtfilt(b, a, sig.astype(np.float64))


def stage_at(t_hr, prof):
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(t_hr), -1, np.int8)
    for i, t in enumerate(t_hr):
        j = np.argmin(np.abs(tep - t))
        if abs(tep[j] - t) < 30.0 / 3600.0:
            out[i] = codes[j]
    return out


def main(idx=2, n_examples=6):
    meta = SESSION_META[idx]
    s = load_session(idx)
    s.sleep_profile = load_sleep_profile(s)
    sp = load_spindles(s)
    if sp is None or s.sleep_profile is None:
        print(f'{meta["label"]}: no spindles / no profile')
        return

    stg = stage_at(sp['center_hr'], s.sleep_profile)
    n2 = stg == N2_CODE
    cen_hr = sp['center_hr'][n2]
    dur_s = sp['duration_s'][n2]
    freq = sp['freq_hz'][n2]

    eeg = s.psg['EEG'].astype(np.float64)
    cle_cre = s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    eeg_sig = bp(eeg, SIGMA_LO, SIGMA_HI)
    cap_sig = bp(cle_cre, SIGMA_LO, SIGMA_HI)
    eeg_env = np.abs(hilbert(eeg_sig))

    # rank spindles by EEG sigma amplitude at center -> pick the clearest ones
    cen_samp = np.round(cen_hr * 3600.0 * FS).astype(int)
    half = int(WIN_HALF * FS)
    ok = (cen_samp - half > 0) & (cen_samp + half < len(eeg))
    cen_samp, cen_hr, dur_s, freq = cen_samp[ok], cen_hr[ok], dur_s[ok], freq[ok]
    amp = eeg_env[cen_samp]
    pick = np.argsort(amp)[::-1][:n_examples]

    ncol = n_examples
    fig, axes = plt.subplots(5, ncol, figsize=(3.1 * ncol, 11), squeeze=False)
    t = np.arange(-half, half + 1) / FS

    for col, p in enumerate(pick):
        c = cen_samp[p]
        a, b = c - half, c + half + 1
        sl = slice(a, b)
        span = 0.5 * dur_s[p]

        # normalise each CAP window to its own std so morphology is visible
        cap_raw = cle_cre[sl] - np.mean(cle_cre[sl])

        rows = [
            (eeg[sl] - np.mean(eeg[sl]), 'EEG raw', '#2C3E50'),
            (eeg_sig[sl], 'EEG 11-16 Hz', '#2C3E50'),
            (cap_raw, 'CAP CLE-CRE raw', '#E67E22'),
            (cap_sig[sl], 'CAP CLE-CRE 11-16 Hz', '#E67E22'),
        ]
        for r, (y, lab, color) in enumerate(rows):
            ax = axes[r][col]
            ax.plot(t, y, color=color, lw=0.7)
            ax.axvspan(-span, span, color='gold', alpha=0.25, zorder=0)
            ax.axvline(0, color='k', ls=':', lw=0.6)
            ax.set_xlim(-WIN_HALF, WIN_HALF)
            if col == 0:
                ax.set_ylabel(lab, fontsize=8)
            ax.tick_params(labelsize=7)
            if r < 4:
                ax.set_xticklabels([])

        # sigma spectrogram of the CAP window (0-20 Hz)
        axg = axes[4][col]
        f, tt, Sxx = spectrogram(cle_cre[sl], fs=FS, nperseg=128, noverlap=112)
        band = f <= 20
        axg.pcolormesh(tt - WIN_HALF, f[band], np.log10(Sxx[band] + 1e-12),
                       shading='auto', cmap='magma')
        axg.axhspan(SIGMA_LO, SIGMA_HI, color='cyan', alpha=0.0)
        axg.axhline(SIGMA_LO, color='cyan', ls='--', lw=0.6)
        axg.axhline(SIGMA_HI, color='cyan', ls='--', lw=0.6)
        axg.set_xlim(-WIN_HALF, WIN_HALF)
        if col == 0:
            axg.set_ylabel('CAP spec (Hz)', fontsize=8)
        axg.tick_params(labelsize=7)
        axg.set_xlabel('t from center (s)', fontsize=8)

        axes[0][col].set_title(f'{cen_hr[p]:.2f} h\n{freq[p]:.1f} Hz, {dur_s[p]*1000:.0f} ms',
                               fontsize=8)

    fig.suptitle(f'{meta["label"]}: individual N2 spindle windows — EEG shows the '
                 f'spindle, CAP (raw / sigma / spectrogram) shows nothing at 11-16 Hz',
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(OUT, f'fig_spindle_windows_{meta["label"]}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}  ({len(cen_hr)} N2 spindles, showing top {n_examples} by EEG sigma)')


if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(idx, n)
