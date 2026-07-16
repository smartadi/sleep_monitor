"""
Figure 2 — Broadband signal-to-noise ratio, one value per session.

Simplified, baseline-free SNR
-----------------------------
The capacitive temple sensor concentrates all of its physiological content
(respiration, cardiac, movement, slow drift) below ~10 Hz.  Above 10 Hz there
is no physiology — only the sensor / electronic noise floor.  So we define,
directly on the raw CLE-CRE signal, with no baseline model required:

    signal power  =  P( f < 10 Hz )
    noise  power  =  P( f >= 10 Hz )            (up to Nyquist, 50 Hz)

    SNR(dB) = 10 * log10( P_signal / P_noise )

Power is the integral of the full-night Welch PSD over each frequency range
(Welch removes the DC offset per segment, so the drifting baseline does not
inflate the signal term).  One number per session; higher = the physiological
band sits further above the sensor noise floor.

Reads the summaries cached by signal_characterization.py.  Run that once
(optionally with --recompute) if the cache is missing.

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

CHANNEL   = 'CLE-CRE'
SIG_HI    = 10.0          # Hz — signal = everything below this
CHANNEL_COLOR = '#2C7FB8'

_trapz = getattr(np, 'trapezoid', np.trapz)


def band_power(freqs, psd, lo, hi):
    m = (freqs > lo) & (freqs < hi)
    return float(_trapz(psd[m], freqs[m]))


def snr_db(freqs, psd, split=SIG_HI):
    nyq = freqs.max()
    p_sig = band_power(freqs, psd, 0.0, split)
    p_noise = band_power(freqs, psd, split, nyq)
    return 10.0 * np.log10(p_sig / (p_noise + 1e-30) + 1e-30)


# ── Load cached PSDs ──────────────────────────────────────────────────────────
if not CACHE.exists():
    sys.exit(f"Cache not found: {CACHE.name}\n"
             f"Run signal_characterization.py --recompute first.")

with open(CACHE, 'rb') as f:
    cached = pickle.load(f)

labels = cached['labels']
psd_by_session = cached['psd_by_session']
records = {r['session']: r for r in cached['records']}

# ── Per-session SNR from the full-night PSD ──────────────────────────────────
rows = []
for lab in labels:
    freqs, psd = psd_by_session[lab][CHANNEL]
    rows.append(dict(
        session=lab,
        subject=records[lab]['subject'],
        snr_db=round(snr_db(freqs, psd), 2),
    ))
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / 'inband_snr_summary.csv', index=False)

snr = summary['snr_db'].to_numpy()
mean_snr, med_snr = float(np.mean(snr)), float(np.median(snr))
print(summary.to_string(index=False))
print(f"\nSNR (dB): mean {mean_snr:.1f}, median {med_snr:.1f}, "
      f"range {snr.min():.1f}–{snr.max():.1f}")

# ── Mean PSD across sessions (for the inset that defines signal vs noise) ─────
freq0 = psd_by_session[labels[0]][CHANNEL][0]
psd_stack = np.vstack([np.interp(freq0, *psd_by_session[l][CHANNEL]) for l in labels])
psd_mean = psd_stack.mean(axis=0)

# ── Figure: single panel, one bar per session ────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(len(labels))
ax.bar(x, snr, color=CHANNEL_COLOR, alpha=0.9, edgecolor='white', linewidth=0.6, zorder=3)
for xi, v in zip(x, snr):
    ax.text(xi, v + (0.4 if v >= 0 else -0.4), f'{v:.0f}',
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=8, color='#333')

ax.axhline(0, color='gray', ls='-', lw=1.0, zorder=2)
ax.axhline(mean_snr, color='#E74C3C', ls='--', lw=1.3, zorder=2,
           label=f'Mean {mean_snr:.1f} dB')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Broadband SNR (dB)', fontsize=11)
ax.set_xlabel('Session', fontsize=11)
ax.set_title(f'In-band signal-to-noise ratio per session ({CHANNEL})',
             fontsize=13, fontweight='bold')
ax.grid(True, axis='y', alpha=0.25, zorder=0)
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.margins(y=0.12)
ax.set_ylim(top=max(snr.max() + 5.0, ax.get_ylim()[1]))

# definition text
ax.text(0.985, 0.04,
        r'signal = power at $f<10$ Hz' '\n'
        r'noise  = power at $f\geq10$ Hz' '\n'
        r'SNR $= 10\log_{10}(P_\mathrm{sig}/P_\mathrm{noise})$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='#F5F7FA', ec='#B0BEC5', lw=0.8))

# inset: mean PSD showing the signal / noise split (opaque PiP over the bars)
axin = ax.inset_axes([0.63, 0.62, 0.34, 0.34], zorder=6)
axin.set_facecolor('white')
for sp in axin.spines.values():
    sp.set_edgecolor('#B0BEC5')
    sp.set_linewidth(1.0)
    sp.set_zorder(7)
fmask = (freq0 > 0.03) & (freq0 <= 50.0)
axin.semilogy(freq0[fmask], psd_mean[fmask], color='#2C3E50', lw=1.1)
axin.axvspan(0.03, SIG_HI, color=CHANNEL_COLOR, alpha=0.18)
axin.axvspan(SIG_HI, 50.0, color='#7F8C8D', alpha=0.22)
axin.axvline(SIG_HI, color='#333', lw=1.0, ls='--')
axin.text(2.5, axin.get_ylim()[1], 'signal', ha='center', va='top',
          fontsize=7.5, color=CHANNEL_COLOR, fontweight='bold')
axin.text(25, axin.get_ylim()[1], 'noise', ha='center', va='top',
          fontsize=7.5, color='#5D6D7E', fontweight='bold')
axin.set_xlim(0, 50)
axin.set_xlabel('Frequency (Hz)', fontsize=7.5)
axin.set_ylabel('PSD', fontsize=7.5)
axin.tick_params(labelsize=6.5)
axin.set_title('Mean PSD (12 sessions)', fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_inband_snr.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nSaved fig2_inband_snr.png")
