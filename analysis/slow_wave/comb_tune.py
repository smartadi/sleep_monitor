"""
Harmonic-comb detection, tuned by eye: one figure per night, all three channels.

Detector (adaptive, no fixed dB rule)
-------------------------------------
1. Spectrogram of the channel, 30-s windows every 15 s, up to FMAX = 5 Hz, each
   column referenced to its own smooth background (the paper's enhancement), so
   a value is "dB above the local floor".
2. Comb score per window. For every fundamental f0 in F0_LO..F0_HI, take the
   peak height near each multiple k*f0 up to FMAX (at most KMAX multiples),
   clip at 0, and average: H(f0) = mean_k max(0, peak_k). The window's score is
   the best H over f0. A comb is many harmonics standing above the floor at
   once; no single harmonic has to pass a fixed level.
3. The score is smoothed over SMOOTH_MIN minutes and converted to a z-score
   against THIS night and channel (median and MAD), so the threshold adapts to
   how strong combs are on that recording.
4. Episodes by hysteresis: start where z >= Z_ON, extend both ways while
   z >= Z_OFF, bridge gaps up to GAP_MIN, keep episodes >= MIN_MIN.
5. Bands inside each episode: the paper's band tracker (paper/stages/
   s06_harmonic_comb.py: track_bands), run up to FMAX.

Figure: stage ladder (hypnogram), then CH / CLE / CRE spectrograms to 5 Hz with
each episode boxed and numbered (CLE-2 ...), bands in cyan (thick = lasting
>= half the episode). Grid every 15 min.

Writes  writeup/figures/harmonics/comb_tune/comb_tune_{S}.png
        reports/slow_wave/comb_tune_episodes.csv

Usage
    .venv/Scripts/python.exe analysis/slow_wave/comb_tune.py              all nights
    .venv/Scripts/python.exe analysis/slow_wave/comb_tune.py S6N1 S4N2    some nights
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
import matplotlib.ticker as mticker                   # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d  # noqa: E402
from scipy.signal import spectrogram                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))

import seclib                                         # noqa: E402
from stages import s06_harmonic_comb as C             # noqa: E402

OUT = ROOT / 'writeup' / 'figures' / 'harmonics' / 'comb_tune'
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / 'reports' / 'slow_wave'
CHANNELS = ['CH', 'CLE', 'CRE']
PERSIST = 0.5

P = dict(
    FMAX=5.0, WIN_SEC=30.0, STEP_SEC=15.0, BG_HZ=0.4,
    F0_LO=0.15, F0_HI=0.55, F0_STEP=0.005, KMAX=4, TOL_HZ=0.05,
    SMOOTH_MIN=1.5,          # smoothing of the comb score
    Z_ON=2.5,                # start an episode (robust z within the night)
    Z_OFF=1.0,               # keep extending while above this
    GAP_MIN=7.0,             # bridge gaps up to this long
    MIN_MIN=3.0,             # shortest episode kept
)


def enhanced(sig, fs):
    f, t, S = spectrogram(sig, fs=fs, nperseg=int(P['WIN_SEC'] * fs),
                          noverlap=int((P['WIN_SEC'] - P['STEP_SEC']) * fs))
    m = f <= P['FMAX']
    f, S = f[m], S[m]
    db = 10 * np.log10(S + 1e-20)
    k = max(3, int(P['BG_HZ'] / (f[1] - f[0])) | 1)
    return f, t / 3600.0, db - median_filter(db, size=(k, 1), mode='nearest')


def comb_score(f, enh):
    """Best mean harmonic height over f0, per window, and the f0 that gives it."""
    df = f[1] - f[0]
    near = maximum_filter1d(enh, size=2 * int(round(P['TOL_HZ'] / df)) + 1, axis=0)
    f0s = np.arange(P['F0_LO'], P['F0_HI'] + 1e-9, P['F0_STEP'])
    best = np.full(enh.shape[1], -np.inf)
    arg = np.zeros(enh.shape[1])
    for f0 in f0s:
        ks = np.arange(1, min(P['KMAX'], int(P['FMAX'] / f0)) + 1)
        idx = np.clip(np.round(ks * f0 / df).astype(int), 0, len(f) - 1)
        h = np.clip(near[idx], 0, None).mean(axis=0)
        better = h > best
        best[better], arg[better] = h[better], f0
    return best, arg


def episodes_from(z):
    n = len(z)
    on = z >= P['Z_ON']
    keep = np.zeros(n, bool)
    i = 0
    while i < n:                       # grow each seed while above Z_OFF
        if on[i] and not keep[i]:
            a = i
            while a > 0 and z[a - 1] >= P['Z_OFF']:
                a -= 1
            b = i
            while b + 1 < n and z[b + 1] >= P['Z_OFF']:
                b += 1
            keep[a:b + 1] = True
            i = b + 1
        else:
            i += 1
    step_min = P['STEP_SEC'] / 60
    idx = np.flatnonzero(keep)
    for a, b in zip(idx[:-1], idx[1:]):     # bridge short gaps
        if 1 < b - a <= P['GAP_MIN'] / step_min:
            keep[a:b] = True
    out, i = [], 0
    while i < n:
        if keep[i]:
            j = i
            while j + 1 < n and keep[j + 1]:
                j += 1
            if (j - i + 1) * step_min >= P['MIN_MIN']:
                out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def detect(session, ch):
    f, t, enh = enhanced(C._sig(session, ch), session.fs)
    sc, f0 = comb_score(f, enh)
    sm = uniform_filter1d(sc, max(1, int(round(P['SMOOTH_MIN'] * 60 / P['STEP_SEC']))))
    med = np.median(sm)
    mad = 1.4826 * np.median(np.abs(sm - med)) + 1e-9
    z = (sm - med) / mad
    eps = []
    for lo, hi in episodes_from(z):
        eps.append(dict(lo=lo, hi=hi, f0=float(np.median(f0[lo:hi])),
                        zmax=float(z[lo:hi].max()),
                        bands=C.track_bands(enh, f, lo, hi)))
    return f, t, enh, z, eps


def draw_night(label):
    s = seclib.get_session(label)
    C.FMAX = P['FMAX']
    fig, axes = plt.subplots(4, 1, figsize=(20, 15), sharex=True,
                             gridspec_kw={'height_ratios': [0.35, 1, 1, 1]})
    C.draw_stage_ladder(axes[0], s.sleep_profile)
    rows = []
    for ax, ch in zip(axes[1:], CHANNELS):
        f, t, enh, z, eps = detect(s, ch)
        ax.pcolormesh(t, f, enh, shading='gouraud', cmap='magma', vmin=0,
                      vmax=np.percentile(enh, 99.5), rasterized=True)
        for k, ep in enumerate(eps, 1):
            a, b = t[ep['lo']], t[min(ep['hi'], len(t) - 1)]
            ax.add_patch(plt.Rectangle((a, 0.03), b - a, P['FMAX'] - 0.12, fill=False,
                                       ec='white', lw=1.6))
            ax.text(a, P['FMAX'] - 0.08, f'{ch}-{k}', color='white', fontsize=11,
                    va='top', fontweight='bold')
            span = ep['hi'] - ep['lo']
            n_sus = 0
            for fr, s0, s1 in ep['bands']:
                sus = (s1 - s0 + 1) >= PERSIST * span
                n_sus += sus
                ax.plot([t[s0], t[s1]], [fr, fr], color='#00E5FF',
                        lw=2.2 if sus else 0.8, alpha=1 if sus else 0.5)
            rows.append(dict(session=label, channel=ch, episode=f'{ch}-{k}',
                             t0_hr=round(a, 3), t1_hr=round(b, 3),
                             dur_min=round((b - a) * 60, 1), f0_hz=round(ep['f0'], 3),
                             zmax=round(ep['zmax'], 1), n_bands=len(ep['bands']),
                             n_sustained=n_sus))
        ax.set_ylim(0, P['FMAX'])
        ax.set_ylabel(f'{ch}\nfrequency (Hz)', fontsize=13)
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.25))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.grid(which='minor', axis='x', color='w', alpha=0.15, lw=0.6)
        ax.grid(which='major', axis='x', color='w', alpha=0.4, lw=0.9)
    axes[0].xaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    axes[0].grid(which='both', axis='x', alpha=0.25)
    axes[-1].set_xlabel('time (h)   [grid every 15 min]', fontsize=13)
    n = {ch: sum(r['channel'] == ch for r in rows) for ch in CHANNELS}
    axes[0].set_title(f'{label}: comb episodes  CH {n["CH"]}, CLE {n["CLE"]}, CRE {n["CRE"]}'
                      f'   (z on {P["Z_ON"]}, z off {P["Z_OFF"]}, gap {P["GAP_MIN"]} min, '
                      f'min {P["MIN_MIN"]} min)', loc='left', fontsize=14)
    fig.tight_layout(h_pad=0.4)
    out = OUT / f'comb_tune_{label}.png'
    fig.savefig(out, dpi=80, bbox_inches='tight')
    plt.close(fig)
    print(f'  {label}: CH {n["CH"]}  CLE {n["CLE"]}  CRE {n["CRE"]}')
    return rows


def main(labels):
    rows = []
    for lab in labels:
        rows += draw_night(lab)
    df = pd.DataFrame(rows)
    p = TAB / 'comb_tune_episodes.csv'
    if p.exists() and len(labels) < 12:
        old = pd.read_csv(p)
        if 'zmax' in old:
            df = pd.concat([old[~old.session.isin(labels)], df])
    df.sort_values(['session', 'channel', 't0_hr']).to_csv(p, index=False)
    return df


if __name__ == '__main__':
    main(sys.argv[1:] or seclib.LABELS)
