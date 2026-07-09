"""
Per-spindle window figures for EVERY spindle in one session.

Same panel as plot_spindle_windows.py but one figure PER spindle (single column):
    EEG raw / EEG 11-16 Hz / CAP 11-16 Hz for CLE-CRE, CLE, CRE, CH / CAP spectrogram
over a +/-3 s window with the scored spindle span shaded. Written to a per-session
subfolder so you can flip through the whole night.

Usage:  python -m analysis.spindles.plot_spindle_windows_all [session_idx] [--n2-only]
Outputs -> analysis/spindles/outputs/spindle_windows_<label>/spindle_<NNNN>_<t>.png
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles

FS = 100.0
SIGMA_LO, SIGMA_HI = 11.0, 16.0
N2_CODE = 2
WIN_HALF = 3.0
CAP_CHANNELS = ['CLE-CRE', 'CLE', 'CRE', 'CH']
CHAN_COLORS = {'CLE-CRE': '#E67E22', 'CLE': '#27AE60', 'CRE': '#8E44AD', 'CH': '#2980B9'}
STAGE_NAME = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM', -1: '?'}
OUT = os.path.join(os.path.dirname(__file__), 'outputs')


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


def main(idx=2, n2_only=False):
    meta = SESSION_META[idx]
    s = load_session(idx)
    s.sleep_profile = load_sleep_profile(s)
    sp = load_spindles(s)
    if sp is None or s.sleep_profile is None:
        print(f'{meta["label"]}: no spindles / no profile')
        return

    cen_hr = sp['center_hr']
    dur_s = sp['duration_s']
    freq = sp['freq_hz']
    stg = stage_at(cen_hr, s.sleep_profile)
    if n2_only:
        keep = stg == N2_CODE
        cen_hr, dur_s, freq, stg = cen_hr[keep], dur_s[keep], freq[keep], stg[keep]

    eeg = s.psg['EEG'].astype(np.float64)
    cap = {ch: (s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)) if ch == 'CLE-CRE'
           else s.cap[ch].astype(np.float64) for ch in CAP_CHANNELS}
    eeg_sig = bp(eeg, SIGMA_LO, SIGMA_HI)
    cap_sig = {ch: bp(cap[ch], SIGMA_LO, SIGMA_HI) for ch in CAP_CHANNELS}

    cen_samp = np.round(cen_hr * 3600.0 * FS).astype(int)
    half = int(WIN_HALF * FS)
    ok = (cen_samp - half > 0) & (cen_samp + half < len(eeg))

    subdir = os.path.join(OUT, f'spindle_windows_{meta["label"]}' + ('_N2' if n2_only else ''))
    os.makedirs(subdir, exist_ok=True)
    t = np.arange(-half, half + 1) / FS
    nrow = 3 + len(CAP_CHANNELS)      # EEG raw, EEG sigma, 4 CAP sigma, spectrogram

    n_written = 0
    for j in np.where(ok)[0]:
        c = cen_samp[j]
        sl = slice(c - half, c + half + 1)
        span = 0.5 * dur_s[j]

        fig, axes = plt.subplots(nrow, 1, figsize=(3.4, 1.55 * nrow), squeeze=False)
        axes = axes[:, 0]
        rows = [(eeg[sl] - np.mean(eeg[sl]), 'EEG raw', '#2C3E50'),
                (eeg_sig[sl], 'EEG 11-16', '#2C3E50')]
        for ch in CAP_CHANNELS:
            rows.append((cap_sig[ch][sl], f'{ch}\n11-16', CHAN_COLORS[ch]))
        for r, (y, lab, color) in enumerate(rows):
            ax = axes[r]
            ax.plot(t, y, color=color, lw=0.7)
            ax.axvspan(-span, span, color='gold', alpha=0.25, zorder=0)
            ax.axvline(0, color='k', ls=':', lw=0.6)
            ax.set_xlim(-WIN_HALF, WIN_HALF)
            ax.set_ylabel(lab, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xticklabels([])

        axg = axes[nrow - 1]
        f, tt, Sxx = spectrogram(cap['CLE-CRE'][sl], fs=FS, nperseg=128, noverlap=112)
        fb = f <= 20
        axg.pcolormesh(tt - WIN_HALF, f[fb], np.log10(Sxx[fb] + 1e-12), shading='auto', cmap='magma')
        axg.axhline(SIGMA_LO, color='cyan', ls='--', lw=0.6)
        axg.axhline(SIGMA_HI, color='cyan', ls='--', lw=0.6)
        axg.set_xlim(-WIN_HALF, WIN_HALF)
        axg.set_ylabel('CAP spec (Hz)', fontsize=8)
        axg.set_xlabel('t from center (s)', fontsize=8)
        axg.tick_params(labelsize=7)

        fr = f'{freq[j]:.1f}Hz' if np.isfinite(freq[j]) else 'NA'
        axes[0].set_title(f'{meta["label"]} spindle #{j}  [{STAGE_NAME.get(int(stg[j]),"?")}]\n'
                          f'{cen_hr[j]:.3f} h · {fr} · {dur_s[j]*1000:.0f} ms', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f'spindle_{j:04d}_{cen_hr[j]:.3f}h.png'),
                    dpi=110, bbox_inches='tight')
        plt.close(fig)
        n_written += 1
        if n_written % 100 == 0:
            print(f'  {n_written} written...')

    print(f'{meta["label"]}: wrote {n_written} per-spindle figures to {subdir}')


if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 2
    n2_only = '--n2-only' in sys.argv
    main(idx, n2_only)
