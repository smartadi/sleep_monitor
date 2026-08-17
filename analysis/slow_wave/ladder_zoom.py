"""
High-contrast zoom diagnostic to SEE harmonic ladders the detector may be
missing.  Plots the combined spectrogram + each channel over a time span, with
per-column background removal so narrow rungs pop regardless of local noise
level.

Run:  python ladder_zoom.py S4N2 4 5
      python ladder_zoom.py S6N1 2 4
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
import harmonic_ladder_overlay as H

OUT = Path(r"C:/Users/adity/AppData/Local/Temp/claude/C--Users-adity-Documents-sleep-monitor-code/331097c7-ff4f-45af-9a10-5cec26a81f37/scratchpad")


def _enh(sig, fs, t0, t1):
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=int(30 * fs), noverlap=int(15 * fs))
    m = f <= 3.0
    f, Sxx = f[m], Sxx[m]
    thr = (t / 3600 >= t0) & (t / 3600 <= t1)
    t, Sxx = t[thr], Sxx[:, thr]
    db = 10 * np.log10(Sxx + 1e-20)
    dfq = f[1] - f[0]
    k = max(3, int(0.4 / dfq) | 1)
    return f, t / 3600, db - median_filter(db, size=(k, 1), mode='nearest')


def main():
    lab, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    for i in range(12):
        s = load_session(i)
        if s.label == lab:
            s.sleep_profile = load_sleep_profile(s)
            break
    rows = ['CH', 'CLE', 'CRE']
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    combs = []
    for ax, ch in zip(axes[:3], rows):
        f, t, enh = _enh(H._sig(s, ch), s.fs, t0, t1)
        combs.append(enh)
        ax.pcolormesh(t, f, enh, shading='gouraud', cmap='magma',
                      vmin=0, vmax=np.percentile(enh, 98), rasterized=True)
        ax.set_ylabel(f'{ch}\nFreq (Hz)', fontsize=9); ax.set_ylim(0, 3)
    n = min(c.shape[1] for c in combs)
    comb = np.mean([c[:, :n] for c in combs], axis=0)
    axes[3].pcolormesh(t[:n], f, comb, shading='gouraud', cmap='magma',
                       vmin=0, vmax=np.percentile(comb, 98), rasterized=True)
    axes[3].set_ylabel('COMBINED\nFreq (Hz)', fontsize=9); axes[3].set_ylim(0, 3)
    axes[3].set_xlabel('Time (hr)', fontsize=10)
    fig.suptitle(f'{lab} — {t0}-{t1} hr, per-column enhanced (rungs should show)', fontsize=13)
    fig.tight_layout()
    out = OUT / f'zoom_{lab}_{t0:.0f}_{t1:.0f}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
