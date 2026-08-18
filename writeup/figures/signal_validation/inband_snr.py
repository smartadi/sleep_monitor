"""
Figure 2 — Physiological-band signal-to-noise ratio per session, per channel.

We report the three raw capacitive channels (CH, CLE, CRE) directly. We do NOT
report the CLE-CRE differential: the two temple electrodes see respiration
largely in phase, so the difference cancels the common-mode physiological signal
while keeping the uncorrelated noise — the differential loses SNR relative to the
better single electrode in 10/12 sessions (e.g. S5N2: CH +16.8 dB but
CLE-CRE -4.9 dB). Reporting the raw channels avoids that self-inflicted loss.

Density-based, baseline-free SNR (per channel, on the raw signal):

    signal  =  mean PSD/Hz over 0.1 - 3 Hz  (respiration 0.1-0.5 + cardiac 0.5-3)
    noise   =  mean PSD/Hz over 10 - 50 Hz  (10 Hz to Nyquist; physiology-free floor)

    SNR(dB) = 10 * log10( <P_signal> / <P_noise> )

We compare *spectral densities* (mean power per Hz), not band-integrated powers.
Integrating signal over 2.9 Hz against noise over 40 Hz imposes a fixed ~11 dB
bandwidth penalty that made otherwise-usable channels read as near-zero or
negative SNR (e.g. S5N1 CRE was -5.1 dB under the integrated form, yet its raw
0.1-3 Hz waveform shows clear respiratory/cardiac oscillation). The density form
removes that mismatch, is gain-invariant (a per-channel scaling cancels in the
ratio, so the S4N1 gain anomaly no longer distorts the bar), and tracks what the
raw traces actually show.

The numerator is the physiological band only; sub-0.1 Hz baseline drift/movement
is excluded so the number reflects respiratory+cardiac pickup, not baseline
wander. The >=10 Hz floor is a stable per-channel instrument constant, and the
no-subject baseline recording (baseline noise/SM2_33.txt) confirms this floor is
spectrally white from 0.1-50 Hz, which is what licenses extrapolating it across
the signal band. Welch removes the DC offset per segment.

Reads signal_characterization_cache.pkl.

Output (writeup/figures/signal_validation/):
    fig2_inband_snr.png
    inband_snr_summary.csv

Usage:
    .venv/Scripts/python.exe writeup/figures/signal_validation/inband_snr.py
"""

from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent
CACHE = OUT_DIR / 'signal_characterization_cache.pkl'

CHANNELS = ['CH', 'CLE', 'CRE']
CHAN_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}
SIG_LO, SIG_HI = 0.1, 3.0        # physiological signal band (resp + cardiac)
NOISE_LO = 10.0                  # noise = NOISE_LO .. Nyquist

_trapz = getattr(np, 'trapezoid', np.trapz)


def band_density(freqs, psd, lo, hi):
    """Mean PSD per Hz over [lo, hi) — a bandwidth-independent spectral density."""
    m = (freqs > lo) & (freqs < hi)
    return float(np.mean(psd[m]))


def snr_db(freqs, psd):
    p_sig = band_density(freqs, psd, SIG_LO, SIG_HI)
    p_noise = band_density(freqs, psd, NOISE_LO, freqs.max())
    return 10.0 * np.log10(p_sig / (p_noise + 1e-30) + 1e-30)


# ── Load cached PSDs ──────────────────────────────────────────────────────────
if not CACHE.exists():
    sys.exit(f"Cache not found: {CACHE.name}\nRun signal_characterization.py --recompute first.")

with open(CACHE, 'rb') as f:
    cached = pickle.load(f)
labels = cached['labels']
psd_by_session = cached['psd_by_session']

# ── Per-session, per-channel SNR from the full-night PSD ─────────────────────
rows = []
for lab in labels:
    row = {'session': lab}
    for ch in CHANNELS:
        freqs, psd = psd_by_session[lab][ch]
        row[ch] = round(snr_db(freqs, psd), 2)
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / 'inband_snr_summary.csv', index=False)

means = {ch: summary[ch].mean() for ch in CHANNELS}
print(summary.to_string(index=False))
print('\nchannel means (dB):', {ch: round(v, 1) for ch, v in means.items()})

# ── Figure: grouped bars, one group per session, 3 channels ──────────────────
fig, ax = plt.subplots(figsize=(12.5, 5.8))
x = np.arange(len(labels))
w = 0.26
for i, ch in enumerate(CHANNELS):
    vals = summary[ch].to_numpy()
    off = (i - 1) * w
    ax.bar(x + off, vals, w, color=CHAN_COLORS[ch], edgecolor='white', linewidth=0.5,
           zorder=3, label=f'{ch}  (mean {means[ch]:.1f} dB)')

ax.axhline(0, color='gray', lw=1.0, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Physiological-band SNR (dB, per-Hz density)', fontsize=11)
ax.set_xlabel('Session', fontsize=11)
ax.set_title('Respiratory + cardiac band SNR per session, per raw channel',
             fontsize=13, fontweight='bold')
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.legend(loc='upper right', fontsize=9.5, frameon=True, framealpha=0.95, ncol=3)
ax.margins(y=0.14)

# The SNR definition lives in the figure caption, not on the axes: repeating it
# inside the panel crowds the plot and duplicates the text a reader has already
# been given.

fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_inband_snr.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('\nSaved fig2_inband_snr.png')
