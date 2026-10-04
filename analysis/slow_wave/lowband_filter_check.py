"""
Why the low-band panel changed: one night, three versions of the same channel.

The ridge_tune figures drew the 0-0.3 Hz band from the rate pipeline's
motion-cancelled signal (remove_acc_artifact, 0.05-4 Hz). ridge_lowband_smooth.py
uses the channel band-passed at 0.005-0.5 Hz with no accelerometer regression.
This puts the evidence for that choice on one page, for S1N1 CRE:

  A  whole-night spectrum of the channel, of the channel after the regression,
     and of the accelerometer magnitude -- the regression lifts the band and
     copies in the accelerometer's 0.1447 Hz line
  B  the smoothed map built from the rate pipeline's signal: a constant hump
     just above its 0.05 Hz corner
  C  the same map built from the channel itself

Writes  writeup/figures/harmonics/ridges_lowband/lowband_filter_check_S1N1_CRE.png

Usage
    .venv/Scripts/python.exe analysis/slow_wave/lowband_filter_check.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
from scipy.signal import welch     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import ridge_lowband_smooth as R                       # noqa: E402
from ridge_overlay_tune import _sig                    # noqa: E402
from sleep_monitor import load_session                 # noqa: E402

LABEL, CH = 'S1N1', 'CRE'


def spectrum(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(600 * fs), nfft=2 ** 15)
    m = (f >= 0.01) & (f <= 0.3)
    return f[m], 10 * np.log10(p[m])


def main():
    s = load_session(0)
    y, fs2 = R.low_band_signal(s.cap[CH], s.fs)
    a, _ = R.low_band_signal(s.cap['acc_mag'], s.fs)
    resid = y - np.dot(a, y) / np.dot(a, a) * a
    old, _ = R._to_low_rate(_sig(s, CH), s.fs)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    for x, lbl, c in ((y, f'{CH}, band-passed 0.005–0.5 Hz', '#1F618D'),
                      (resid, f'{CH} after accelerometer regression', '#C0392B')):
        f, p = spectrum(x, fs2)
        ax.plot(f, p, lw=1.8, color=c, label=lbl)
    ax2 = ax.twinx()
    f, p = spectrum(a, fs2)
    ax2.plot(f, p, lw=1.2, color='#7F8C8D', label='accelerometer magnitude (right axis)')
    ax2.set_ylabel('acc (dB)', color='#7F8C8D')
    ax.axvline(0.1447, color='#7F8C8D', ls=':', lw=1)
    ax.set_xlim(0.01, 0.3)
    ax.set_ylabel('dB')
    ax.set_xlabel('Frequency (Hz)')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=11, loc='upper right')
    ax.set_title(f'A  {LABEL} {CH}, whole night: the regression lifts the band and '
                 'copies in the accelerometer line at 0.1447 Hz', loc='left', fontsize=14)

    for k, (x, ttl) in enumerate(((old, 'B  map from the rate pipeline signal '
                                   '(0.05–4 Hz band-pass + regression): a hump at its corner'),
                                  (y, 'C  map from the channel itself (0.005–0.5 Hz, '
                                   'no regression)'))):
        f, t, db = R.low_band_map(x, fs2)
        axm = fig.add_subplot(gs[k + 1])
        lo, hi = np.percentile(db, [5, 99.5])
        axm.pcolormesh(t, f, db, shading='gouraud', cmap='viridis', vmin=lo, vmax=hi,
                       rasterized=True)
        axm.set_ylim(0, 0.3)
        axm.set_ylabel('Hz')
        axm.set_title(ttl, loc='left', fontsize=14)
    axm.set_xlabel('Time (hr)')
    out = R.FIG / f'lowband_filter_check_{LABEL}_{CH}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print('wrote', out)


if __name__ == '__main__':
    main()
