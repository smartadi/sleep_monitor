"""
Spindle-triggered ERSP — does ANY part of each channel's own spectrum change
during a spindle, relative to that same channel's surrounding baseline?

The direct/indirect tests pre-selected bands (sigma, cardiac). This is the
hypothesis-free version: for every channel (EEG + all CAP), average the
time-frequency power around N2 spindles over 0-45 Hz and baseline-correct each
frequency against the window edges (|t|>5 s). The result is "activity minus
itself": any transient band change — sigma, but also respiration, cardiac,
motion, broadband — is revealed as a colored patch. EEG is the positive control
(must light up at 11-16 Hz, t=0).

Outputs -> analysis/spindles/outputs/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles

FS = 100.0
N2_CODE = 2
WIN_HALF = 8.0          # +/- s extracted per spindle
FMAX = 45.0
NPERSEG = 128
NOVERLAP = 96
BASE_EDGE = 5.0         # |t|>BASE_EDGE is baseline
CORE = 1.0              # |t|<CORE is the "during spindle" core
MAX_EVENTS = 400        # cap per session for tractability
CHANNELS = ['EEG', 'CLE-CRE', 'CLE', 'CRE', 'CH']
OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)


def get_channel(s, ch):
    if ch == 'EEG':
        return s.psg['EEG'].astype(np.float64)
    if ch == 'CLE-CRE':
        return s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    return s.cap[ch].astype(np.float64)


def stage_at(t_hr, prof):
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(t_hr), -1, np.int8)
    for i, t in enumerate(t_hr):
        j = np.argmin(np.abs(tep - t))
        if abs(tep[j] - t) < 30.0 / 3600.0:
            out[i] = codes[j]
    return out


def session_ersp(sig, centers_samp, half):
    """Mean baseline-corrected dB time-frequency map over events for one channel."""
    acc = None
    faxis = tcen = None
    k = 0
    for c in centers_samp:
        a, b = c - half, c + half + 1
        if a < 0 or b > len(sig):
            continue
        f, t, Sxx = spectrogram(sig[a:b], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        fb = f <= FMAX
        dB = 10.0 * np.log10(Sxx[fb] + 1e-12)
        if acc is None:
            acc = np.zeros_like(dB)
            tcen = t - (t[-1] / 2.0)   # center time axis on 0
            faxis = f[fb]
        acc += dB
        k += 1
    if k == 0:
        return None
    mean_dB = acc / k
    base = mean_dB[:, np.abs(tcen) > BASE_EDGE].mean(axis=1, keepdims=True)
    ersp = mean_dB - base            # dB change vs own baseline
    return {'f': faxis, 't': tcen, 'ersp': ersp, 'k': k}


def main():
    rng = np.random.default_rng(11)
    stacks = {ch: [] for ch in CHANNELS}
    core_spec = {ch: [] for ch in CHANNELS}   # per-session core-vs-baseline dB by freq
    f_axis = t_axis = None
    rows = []

    for idx in range(len(SESSION_META)):
        meta = SESSION_META[idx]
        try:
            s = load_session(idx)
            s.sleep_profile = load_sleep_profile(s)
            sp = load_spindles(s)
            if sp is None or s.sleep_profile is None:
                continue
            stg = stage_at(sp['center_hr'], s.sleep_profile)
            cen_hr = sp['center_hr'][stg == N2_CODE]
            if len(cen_hr) < 20:
                continue
            if len(cen_hr) > MAX_EVENTS:
                cen_hr = rng.choice(cen_hr, size=MAX_EVENTS, replace=False)
            cen_samp = np.round(cen_hr * 3600.0 * FS).astype(int)
            half = int(WIN_HALF * FS)

            for ch in CHANNELS:
                r = session_ersp(get_channel(s, ch), cen_samp, half)
                if r is None:
                    continue
                f_axis, t_axis = r['f'], r['t']
                stacks[ch].append(r['ersp'])
                core = r['ersp'][:, np.abs(t_axis) < CORE].mean(axis=1)
                core_spec[ch].append(core)
                # sigma-band core change, for the table
                sig = (f_axis >= 11) & (f_axis <= 16)
                rows.append({'session': meta['label'], 'channel': ch,
                             'sigma_core_dB': float(core[sig].mean()),
                             'peak_abs_dB': float(np.max(np.abs(core))),
                             'peak_freq_hz': float(f_axis[np.argmax(np.abs(core))])})
            print(f"{meta['label']}: ERSP done ({len(cen_samp)} events)")
        except Exception as e:
            print(f'[{idx}] ERSP failed: {e}')
            continue

    save = {'f': f_axis, 't': t_axis}
    for ch in CHANNELS:
        save[f'ersp_{ch}'] = np.array(stacks[ch])       # (sessions, nf, nt)
        save[f'core_{ch}'] = np.array(core_spec[ch])     # (sessions, nf)
    np.savez(os.path.join(OUT, 'spindle_ersp.npz'), **save)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'spindle_ersp.csv'), index=False)
    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()
