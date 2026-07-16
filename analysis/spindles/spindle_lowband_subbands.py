"""
Break the CAP low-band (0-3 Hz) spindle-onset response into three sub-bands to
see WHERE in the low band the bump lives:

    slow    0.0-0.5 Hz   (sub-respiratory / baseline-wander / very slow mechanical)
    mid     0.5-1.5 Hz   (respiratory upper + low cardiac)
    high    1.5-3.0 Hz   (cardiac fundamental + harmonics)

Onset-triggered average of the CH power in each sub-band, per N2 spindle, for all
12 sessions. Uses the same +/-8 s window and own-baseline dB contrast as
`spindle_lowband_detection.py`, but with a finer STFT (nperseg=256 -> 0.39 Hz
bins) so the three sub-bands are actually resolved (the 0-3 Hz analysis used
nperseg=128 = 0.78 Hz bins, too coarse to split 0-0.5 from 0.5-1.5).

Recomputes from raw sessions (the sub-band traces are not cached), then caches
the onset-triggered curves to spindle_lowband_subbands.npz so the figure can be
re-tuned without reloading. Pass --recompute to force a rebuild.

Outputs:
  writeup/figures/spindles/fig_spindle_lowband_subbands.png
  analysis/spindles/outputs/spindle_lowband_subbands.npz

Usage:
  .venv/Scripts/python.exe -m analysis.spindles.spindle_lowband_subbands
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles

FS = 100.0
N2_CODE = 2
WIN_HALF = 8.0
CORE_HALF = 1.0
BASE_EDGE = 5.0
NPERSEG = 256          # 0.39 Hz resolution (finer than the 0-3 Hz analysis)
NOVERLAP = 224

SUBBANDS = {
    'slow 0–0.5 Hz':  (0.0, 0.5),
    'mid 0.5–1.5 Hz': (0.5, 1.5),
    'high 1.5–3 Hz':  (1.5, 3.0),
}
SUB_COLORS = {'slow 0–0.5 Hz': '#C0392B', 'mid 0.5–1.5 Hz': '#2980B9',
              'high 1.5–3 Hz': '#27AE60'}
CHAN = 'CH'

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, 'outputs')
NPZ = os.path.join(OUT, 'spindle_lowband_subbands.npz')
FIG = os.path.join(HERE, '..', '..', 'writeup', 'figures', 'spindles',
                   'fig_spindle_lowband_subbands.png')
os.makedirs(OUT, exist_ok=True)


def stage_at(t_hr, prof):
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(t_hr), -1, np.int8)
    for i, t in enumerate(t_hr):
        j = np.argmin(np.abs(tep - t))
        if abs(tep[j] - t) < 30.0 / 3600.0:
            out[i] = codes[j]
    return out


def subband_triggered(sig, centers_samp):
    """Onset-triggered average baseline-corrected dB(t) per sub-band, plus counts."""
    n = len(sig)
    win = int(WIN_HALF * FS)
    tcen = core_t = base_t = None
    fmask = {}
    acc = {b: None for b in SUBBANDS}
    k = 0
    for c in centers_samp:
        a, b = c - win, c + win + 1
        if a < 0 or b > n:
            continue
        f, t, Sxx = spectrogram(sig[a:b], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        dB = 10.0 * np.log10(Sxx + 1e-12)
        if tcen is None:
            tcen = t - t[-1] / 2.0
            core_t = np.abs(tcen) < CORE_HALF
            base_t = np.abs(tcen) > BASE_EDGE
            for name, (lo, hi) in SUBBANDS.items():
                fmask[name] = (f >= lo) & (f < hi) if hi < 3.0 else (f >= lo) & (f <= hi)
        for name in SUBBANDS:
            band_dB = dB[fmask[name]].mean(axis=0)
            curve = band_dB - band_dB[base_t].mean()
            acc[name] = curve if acc[name] is None else acc[name] + curve
        k += 1
    if k == 0:
        return None, None, 0, None
    curves = {name: acc[name] / k for name in SUBBANDS}
    core_db = {name: curves[name][core_t].mean() for name in SUBBANDS}
    return curves, tcen, k, core_db


def compute():
    labels, ns = [], []
    per = {name: [] for name in SUBBANDS}
    coredb = {name: [] for name in SUBBANDS}
    t_axis = None
    for m in SESSION_META:
        s = load_session(m['idx'])
        s.sleep_profile = load_sleep_profile(s)
        sp = load_spindles(s)
        if s.sleep_profile is None or sp is None:
            print(f"  skip {m['label']}: missing profile/spindles")
            continue
        stg = stage_at(sp['center_hr'], s.sleep_profile)
        cen = np.round(sp['center_hr'][stg == N2_CODE] * 3600.0 * FS).astype(int)
        if len(cen) < 20:
            print(f"  skip {m['label']}: <20 N2 spindles")
            continue
        sig = s.cap['CH'].astype(np.float64)
        curves, tcen, k, cdb = subband_triggered(sig, cen)
        if curves is None:
            continue
        t_axis = tcen
        labels.append(m['label']); ns.append(k)
        for name in SUBBANDS:
            per[name].append(curves[name]); coredb[name].append(cdb[name])
        print(f"  {m['label']}: {k} spindles  "
              + "  ".join(f"{name.split()[0]} {cdb[name]:+.2f}" for name in SUBBANDS))
        del s, sig
    save = {'t_axis': t_axis, 'labels': np.array(labels), 'n': np.array(ns)}
    for name in SUBBANDS:
        key = name.split()[0]
        save[f'trig_{key}'] = np.vstack(per[name])
        save[f'coredb_{key}'] = np.array(coredb[name])
    np.savez(NPZ, **save)
    print('cached', NPZ)
    return save


def load_cache():
    d = np.load(NPZ, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    if os.path.exists(NPZ) and '--recompute' not in sys.argv:
        print('loading cache (pass --recompute to rebuild)')
        data = load_cache()
    else:
        data = compute()

    t = data['t_axis']
    labels = [str(x) for x in data['labels']]
    ns = data['n']
    core = np.abs(t) < CORE_HALF
    keys = [name.split()[0] for name in SUBBANDS]
    names = list(SUBBANDS)

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.35, 1, 1, 1, 1],
                          hspace=0.5, wspace=0.22)

    # ---- grand-mean panel (top, spanning) ----
    axg = fig.add_subplot(gs[0, :])
    axg.axhline(0, color='gray', lw=0.7, ls=':')
    axg.axvspan(-CORE_HALF, CORE_HALF, color='gray', alpha=0.08)
    axg.axvline(0, color='k', lw=0.7, alpha=0.5)
    for name in names:
        key = name.split()[0]
        gm = data[f'trig_{key}'].mean(axis=0)
        axg.plot(t, gm, color=SUB_COLORS[name], lw=2.2,
                 label=f'{name}  (+{gm[core].mean():.2f} dB)')
    axg.set_title('Grand mean over 12 sessions — the CH 0–3 Hz spindle bump, split by sub-band',
                  fontsize=11, fontweight='bold')
    axg.set_xlabel('Time from spindle center (s)', fontsize=9)
    axg.set_ylabel('CH power (dB vs baseline)', fontsize=9)
    axg.legend(fontsize=9, loc='upper left')
    axg.set_xlim(t.min(), t.max())

    # ---- per-session grid ----
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.axvspan(-CORE_HALF, CORE_HALF, color='gray', alpha=0.08)
        ax.axvline(0, color='k', lw=0.6, alpha=0.5)
        for name in names:
            key = name.split()[0]
            ax.plot(t, data[f'trig_{key}'][i], color=SUB_COLORS[name], lw=1.5)
        # which sub-band peaks at onset
        peaks = {name: data[f'trig_{name.split()[0]}'][i][core].mean() for name in names}
        top = max(peaks, key=peaks.get)
        ax.set_title(f'{lab} (n={int(ns[i]):,})  peak: {top.split()[0]}',
                     fontsize=9, fontweight='bold', color=SUB_COLORS[top])
        ax.set_xlim(t.min(), t.max())
        ax.tick_params(labelsize=7)
        if i % 3 == 0:
            ax.set_ylabel('CH (dB)', fontsize=8)
        if i >= 9:
            ax.set_xlabel('Time from center (s)', fontsize=8)

    fig.suptitle('Where the 0–3 Hz spindle bump lives — CH power split into '
                 '0–0.5 / 0.5–1.5 / 1.5–3 Hz (onset-triggered average, all 12 sessions)',
                 fontsize=13, fontweight='bold', y=1.0)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', FIG)

    # console summary
    print('\nGrand-mean onset core dB by sub-band:')
    for name in names:
        key = name.split()[0]
        arr = data[f'coredb_{key}']
        print(f'  {name:16s}: mean +{arr.mean():.3f} dB  (positive in {int((arr>0).sum())}/12)')


if __name__ == '__main__':
    main()
