"""
Figure 3 — in-band SNR split into respiratory and cardiac bands, with the
unworn mask as a measured negative control.

Why this replaces the pooled version (`inband_snr.py`)
------------------------------------------------------
The pooled figure reports one number per session per channel over 0.1-3.0 Hz,
which is respiration and cardiac activity added together. Respiration dominates
that sum, so a channel can look healthy in the pooled bar while carrying almost
no cardiac signal. Splitting the band is the only way to see that, and it is the
same evidence that supports stating the cardiac result as the weaker one.

Why the unworn recording is plotted
-----------------------------------
The SNR denominator is each recording's own 10-50 Hz density, extrapolated down
across the signal band. That extrapolation is licensed by `baseline noise/
SM2_33.txt` -- the mask recorded with nobody wearing it -- which shows the floor
is spectrally white from 0.1 to 50 Hz. The manuscript already says so in Methods.

Running the identical SNR on that recording turns the assumption into a
measurement. With no subject there is nothing in the physiological band but the
same white floor, so the control should land at ~0 dB. Whatever it actually
reads is the zero every other bar in the figure should be read against: if it
sits above 0, every session SNR here is inflated by that much.

Definition, unchanged from the pooled figure so the numbers stay comparable:

    signal  =  mean PSD/Hz over the band   (resp 0.1-0.5, cardiac 0.5-3.0)
    noise   =  mean PSD/Hz over 10 Hz - Nyquist
    SNR(dB) =  10 * log10(signal / noise)

Densities, not band-integrated powers: integrating a 0.4 Hz band against a 40 Hz
noise band would impose a bandwidth penalty that differs between the respiratory
and cardiac panels and make them incomparable.

The unworn recording is 111.11 Hz native (9 ms tick) against the sessions' 100 Hz,
so it is resampled 9/10 -- exact, no interpolation error -- before the PSD, which
also puts its Nyquist at 50 Hz so the noise band matches the sessions'.

Reads   signal_characterization_cache.pkl  (per-session PSDs)
        baseline noise/SM2_33.txt          (unworn mask, 15.8 min)
Writes  writeup/figures/signal_validation/fig3_inband_snr_split.png
        writeup/figures/signal_validation/inband_snr_split_summary.csv

Usage:
    .venv/Scripts/python.exe writeup/figures/signal_validation/inband_snr_split.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
from scipy.signal import resample_poly, welch   # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parents[2]
CACHE = OUT_DIR / 'signal_characterization_cache.pkl'
UNWORN = ROOT / 'baseline noise' / 'SM2_33.txt'

CHANNELS = ['CH', 'CLE', 'CRE']
CHAN_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}

BANDS = {'respiratory': (0.1, 0.5), 'cardiac': (0.5, 3.0)}
NOISE_LO = 10.0
FS = 100.0                 # session sampling rate (sleep_monitor.config.FS)
UNWORN_FS = 1000.0 / 9.0   # mask tick is exactly 9 ms
SEG_SEC = 30.0             # Welch sub-segment, as used for the session PSDs


def band_density(freqs, psd, lo, hi):
    """Mean PSD per Hz over [lo, hi) -- bandwidth-independent."""
    m = (freqs > lo) & (freqs < hi)
    return float(np.mean(psd[m]))


def snr_db(freqs, psd, band):
    p_sig = band_density(freqs, psd, *band)
    p_noise = band_density(freqs, psd, NOISE_LO, freqs.max())
    return 10.0 * np.log10(p_sig / (p_noise + 1e-30) + 1e-30)


def unworn_psds():
    """PSD per channel for the unworn mask, on the sessions' sampling grid."""
    d = pd.read_csv(UNWORN)
    d.columns = [c.strip() for c in d.columns]
    dt = np.diff(d['timeMS'].to_numpy())
    assert np.median(dt) == 9, 'unworn tick is not 9 ms; check the export'
    out = {}
    for ch in CHANNELS:
        x = d[ch].to_numpy(float)
        x = resample_poly(x, 9, 10)          # 1000/9 Hz -> 100 Hz, exact
        f, p = welch(x, fs=FS, nperseg=int(SEG_SEC * FS),
                     noverlap=int(SEG_SEC * FS) // 2, scaling='density')
        out[ch] = (f, p)
    return out, len(d) / UNWORN_FS / 60.0


def main():
    if not CACHE.exists():
        sys.exit(f'Cache not found: {CACHE.name}\n'
                 'Run signal_characterization.py --recompute first.')
    if not UNWORN.exists():
        sys.exit(f'Unworn recording not found: {UNWORN}')

    with open(CACHE, 'rb') as f:
        cached = pickle.load(f)
    labels, psd_by_session = cached['labels'], cached['psd_by_session']

    unworn, unworn_min = unworn_psds()

    rows = []
    for lab in labels:
        for band_name, band in BANDS.items():
            r = {'recording': lab, 'band': band_name}
            for ch in CHANNELS:
                freqs, psd = psd_by_session[lab][ch]
                r[ch] = round(snr_db(freqs, psd, band), 2)
            rows.append(r)
    for band_name, band in BANDS.items():
        r = {'recording': 'unworn (control)', 'band': band_name}
        for ch in CHANNELS:
            freqs, psd = unworn[ch]
            r[ch] = round(snr_db(freqs, psd, band), 2)
        rows.append(r)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / 'inband_snr_split_summary.csv', index=False)

    pd.set_option('display.width', 200)
    print(f'unworn recording: {unworn_min:.1f} min, resampled '
          f'{UNWORN_FS:.2f} -> {FS:.0f} Hz\n')
    print(summary.to_string(index=False))

    ctrl = summary[summary.recording == 'unworn (control)'].set_index('band')
    print('\nnegative control (should sit near 0 dB if the noise floor is white):')
    for band_name in BANDS:
        vals = {ch: ctrl.loc[band_name, ch] for ch in CHANNELS}
        print(f'  {band_name:12s} ' + '  '.join(f'{c} {v:+.2f} dB'
                                                for c, v in vals.items()))

    # ── figure: one panel per band, sessions then the control, set apart ──────
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 10.0), sharex=True)
    x = np.arange(len(labels))
    gap = 0.9
    x_ctrl = len(labels) - 1 + gap + 1.0
    w = 0.26

    for ax, (band_name, band) in zip(axes, BANDS.items()):
        sub = summary[summary.band == band_name].set_index('recording')
        floor = max(sub.loc['unworn (control)', ch] for ch in CHANNELS)
        ax.axhline(floor, color='#b3282d', lw=1.1, ls='--', zorder=2)

        for i, ch in enumerate(CHANNELS):
            vals = [sub.loc[lab, ch] for lab in labels]
            off = (i - 1) * w
            ax.bar(x + off, vals, w, color=CHAN_COLORS[ch], edgecolor='white',
                   linewidth=0.5, zorder=3,
                   label=f'{ch}  (median {np.median(vals):.1f} dB)')
            cv = sub.loc['unworn (control)', ch]
            ax.bar(x_ctrl + off, cv, w, color=CHAN_COLORS[ch],
                   edgecolor='#b3282d', linewidth=1.2, hatch='//', zorder=3)
            # the control bars are ~0 dB by design, so print the value: the
            # reader needs to see it is zero, not merely that it is short
            ax.annotate(f'{cv:+.1f}', (x_ctrl + off, cv), ha='center',
                        va='top' if cv < 0 else 'bottom',
                        xytext=(0, -4 if cv < 0 else 4),
                        textcoords='offset points', fontsize=10,
                        color='#b3282d', zorder=5)
        ax.annotate('unworn control ≈ 0 dB', (x_ctrl, floor),
                    xytext=(0, 26), textcoords='offset points', ha='center',
                    va='bottom', fontsize=11, color='#b3282d')

        ax.axhline(0, color='gray', lw=1.0, zorder=2)
        ax.set_ylabel(f'{band_name.capitalize()}-band SNR\n(dB, per-Hz density)',
                      fontsize=14)
        ax.set_title(f'{band_name.capitalize()} band '
                     f'({band[0]}–{band[1]} Hz)',
                     fontsize=15, fontweight='bold', loc='left')
        ax.grid(True, axis='y', alpha=0.25, zorder=0)
        ax.legend(loc='upper right', fontsize=12, frameon=True,
                  framealpha=0.95, ncol=3)
        ax.margins(y=0.16)

    axes[-1].set_xticks(list(x) + [x_ctrl])
    axes[-1].set_xticklabels(list(labels) + ['unworn\n(control)'],
                             rotation=45, ha='right', fontsize=12)
    axes[-1].set_xlabel('Recording', fontsize=14)
    fig.suptitle('Physiological-band SNR per recording, per raw channel, '
                 'split by band\n'
                 'Hatched bars: the same measurement on the mask with nobody '
                 'wearing it — the level at which the figure says nothing',
                 fontsize=15, fontweight='bold', y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT_DIR / 'fig3_inband_snr_split.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved {p.name}')


if __name__ == '__main__':
    main()
