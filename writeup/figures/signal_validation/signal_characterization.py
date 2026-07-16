"""
Signal validation / characterization for the paper — three deliverables over
all 12 sessions.

1. Simplified in-band SNR
   ----------------------
   A single, easy-to-state ratio computed per 60 s window on the raw broadband
   CLE-CRE signal:

       SNR_band(dB) = 10 * log10( mean_PSD_in_band / mean_PSD_in_noise_band )

   where mean_PSD_in_band = (band power) / (band width) is the average power
   spectral density in the target band, and the noise reference is a fixed
   physiology-free high band (5-10 Hz).  Because both terms are per-Hz
   densities, the ratio is a fair "signal PSD vs noise-floor PSD" comparison
   that does not depend on how wide the bands are.  Reported per session as the
   median across the night, for the respiratory (0.1-0.5 Hz) and cardiac
   (0.5-3.0 Hz) bands.

2. Mean capacitance value changes across the session
   -------------------------------------------------
   The raw CAP channels carry a slowly drifting DC baseline (coupling offset)
   that is independent of the AC physiological signal.  For each session the
   baseline is extracted as the <0.01 Hz trend of the raw CLE, CRE and CH
   channels and plotted across the night.  Drift magnitude (5-95th percentile
   span) is quantified, both in raw units and relative to the AC signal
   amplitude, to show the sensor baseline is non-stationary.

3. Characteristics of signal frequency
   -----------------------------------
   Full-night Welch PSD of every channel, averaged across sessions; the
   dominant frequency in each physiological band per session; and the fraction
   of total power that falls in the respiratory band, cardiac band and
   broadband noise band.

Outputs (writeup/figures/signal_validation/):
  fig8_simplified_snr.png
  fig9_mean_capacitance_drift.png
  fig10_frequency_characteristics.png
  signal_characterization_summary.csv   (per-session, per-channel)

Usage:
  .venv/Scripts/python.exe writeup/figures/signal_validation/signal_characterization.py
"""

from __future__ import annotations
import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI, CAP_COLORS,
)
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.loader import load_session

OUT_DIR = Path(__file__).resolve().parent

# ── Band definitions ──────────────────────────────────────────────────────────
RESP_BAND  = (RESP_LO, RESP_HI)   # 0.1-0.5 Hz
CARD_BAND  = (CARD_LO, CARD_HI)   # 0.5-3.0 Hz
NOISE_BAND = (5.0, 10.0)          # physiology-free reference floor
TOTAL_BAND = (0.05, 10.0)         # denominator for power fractions

CHANNELS = ['CH', 'CLE', 'CRE', 'CLE-CRE']  # order for frequency panels
PRIMARY  = 'CLE-CRE'

# ── Window parameters ─────────────────────────────────────────────────────────
SNR_WIN_SEC   = 60.0
SNR_STEP_SEC  = 30.0
SNR_SEG_SEC   = 20.0     # Welch sub-segment inside each window
FULL_SEG_SEC  = 30.0     # Welch sub-segment for full-night PSD
BASELINE_FC   = 0.01     # Hz — DC-baseline cutoff (period ~100 s)
BASE_DS_SEC   = 10.0     # baseline plotting resolution


_trapz = getattr(np, 'trapezoid', np.trapz)  # np>=2.0 renamed trapz -> trapezoid


def _band_power(freqs, psd, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    return float(_trapz(psd[m], freqs[m]))


def _mean_density(freqs, psd, band):
    """Average PSD (power per Hz) inside a band."""
    bw = band[1] - band[0]
    return _band_power(freqs, psd, band) / bw if bw > 0 else np.nan


def simplified_snr(sig, band, noise=NOISE_BAND, fs=FS,
                   win_sec=SNR_WIN_SEC, step_sec=SNR_STEP_SEC,
                   seg_sec=SNR_SEG_SEC):
    """Per-window simplified SNR (dB): band PSD density / noise PSD density."""
    win_n  = int(win_sec * fs)
    step_n = int(step_sec * fs)
    nperseg = int(seg_sec * fs)
    starts = np.arange(0, len(sig) - win_n + 1, step_n)
    t_hr = (starts + win_n / 2) / fs / 3600.0
    snr = np.empty(len(starts))
    for i, s0 in enumerate(starts):
        chunk = sig[s0:s0 + win_n].astype(np.float64)
        freqs, psd = welch(chunk, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2, scaling='density')
        d_sig = _mean_density(freqs, psd, band)
        d_noise = _mean_density(freqs, psd, noise) + 1e-15
        snr[i] = 10.0 * np.log10(d_sig / d_noise + 1e-15)
    return t_hr, snr


def dc_baseline(sig, fs=FS, fc=BASELINE_FC):
    """Slow DC baseline via decimate-then-lowpass (memory-cheap)."""
    # Decimate by block-mean to ~1 Hz first, then rolling median for the trend.
    q = int(fs)  # 1 s blocks
    n = (len(sig) // q) * q
    coarse = sig[:n].astype(np.float64).reshape(-1, q).mean(axis=1)  # 1 Hz
    # Rolling median over 1/fc seconds to isolate the <fc baseline.
    win = max(3, int(round(1.0 / fc)))
    if win % 2 == 0:
        win += 1
    s = pd.Series(coarse)
    base = s.rolling(win, center=True, min_periods=1).median().to_numpy()
    t_hr = (np.arange(len(base)) + 0.5) / 3600.0
    return t_hr, base


# ══════════════════════════════════════════════════════════════════════════════
# Pass over all sessions — collect lightweight summaries
# ══════════════════════════════════════════════════════════════════════════════
# Loading all 12 sessions is slow; cache the computed summaries so figures can be
# re-tuned without reloading raw data.  Pass --recompute to force a fresh pass.

CACHE = OUT_DIR / 'signal_characterization_cache.pkl'
RECOMPUTE = '--recompute' in sys.argv

records = []          # per-session summary rows (one per session, primary chan)
snr_by_session = {}   # label -> {'resp': arr, 'card': arr}
psd_by_session = {}   # label -> {chan: (freqs, psd)}
drift_by_session = {} # label -> {chan: (t_hr, base)}
per_chan_snr = {}     # label -> {chan: {'resp': med, 'card': med}}
labels = []

if CACHE.exists() and not RECOMPUTE:
    print(f"Loading cached summaries from {CACHE.name} (pass --recompute to rebuild)")
    with open(CACHE, 'rb') as f:
        cached = pickle.load(f)
    records          = cached['records']
    snr_by_session   = cached['snr_by_session']
    psd_by_session   = cached['psd_by_session']
    drift_by_session = cached['drift_by_session']
    per_chan_snr     = cached['per_chan_snr']
    labels           = cached['labels']

for idx in ([] if labels else range(len(SESSION_META))):
    s = load_session(idx)
    label = s.label
    labels.append(label)

    cle = s.cap['CLE'].astype(np.float64)
    cre = s.cap['CRE'].astype(np.float64)
    ch  = s.cap['CH'].astype(np.float64)
    raw = {'CH': ch, 'CLE': cle, 'CRE': cre, 'CLE-CRE': cle - cre}

    # ── 1. Simplified SNR (per window) on the primary channel ────────────────
    t_snr, snr_resp = simplified_snr(raw[PRIMARY], RESP_BAND)
    _,     snr_card = simplified_snr(raw[PRIMARY], CARD_BAND)
    snr_by_session[label] = {'resp': snr_resp, 'card': snr_card}

    # per-channel median SNR (for the channel-comparison panel)
    per_chan_snr[label] = {}
    for ch_name, sig in raw.items():
        _, sr = simplified_snr(sig, RESP_BAND)
        _, sc = simplified_snr(sig, CARD_BAND)
        per_chan_snr[label][ch_name] = {
            'resp': float(np.nanmedian(sr)),
            'card': float(np.nanmedian(sc)),
        }

    # ── 3. Full-night PSD per channel ────────────────────────────────────────
    psd_by_session[label] = {}
    for ch_name, sig in raw.items():
        freqs, psd = welch(sig, fs=FS, nperseg=int(FULL_SEG_SEC * FS),
                           noverlap=int(FULL_SEG_SEC * FS) // 2, scaling='density')
        psd_by_session[label][ch_name] = (freqs, psd)

    # ── 2. DC baseline drift (raw channels) ──────────────────────────────────
    drift_by_session[label] = {}
    drift_span = {}
    for ch_name in ['CLE', 'CRE', 'CH']:
        t_b, base = dc_baseline(raw[ch_name])
        drift_by_session[label][ch_name] = (t_b, base)
        drift_span[ch_name] = float(np.nanpercentile(base, 95)
                                    - np.nanpercentile(base, 5))

    # ── Summary numbers on the primary channel ───────────────────────────────
    freqs, psd = psd_by_session[label][PRIMARY]
    p_resp  = _band_power(freqs, psd, RESP_BAND)
    p_card  = _band_power(freqs, psd, CARD_BAND)
    p_noise = _band_power(freqs, psd, NOISE_BAND)
    p_total = _band_power(freqs, psd, TOTAL_BAND)

    # dominant frequency in each band (full-night PSD peak)
    def _dom_freq(band):
        m = (freqs >= band[0]) & (freqs <= band[1])
        return float(freqs[m][np.argmax(psd[m])])
    dom_resp = _dom_freq(RESP_BAND)
    dom_card = _dom_freq(CARD_BAND)

    # AC amplitude of the primary channel (std of broadband-ish 0.1-3 Hz)
    ac_amp = float(np.sqrt(_band_power(freqs, psd, (0.1, 3.0))))

    records.append(dict(
        session=label, subject=s.subject, night=s.meta['night'],
        duration_hr=round(s.duration_hr, 2),
        snr_resp_db=round(float(np.nanmedian(snr_resp)), 2),
        snr_card_db=round(float(np.nanmedian(snr_card)), 2),
        dom_resp_hz=round(dom_resp, 4),
        dom_card_hz=round(dom_card, 4),
        dom_resp_brmin=round(dom_resp * 60, 1),
        dom_card_bpm=round(dom_card * 60, 1),
        resp_pow_frac=round(p_resp / p_total, 4),
        card_pow_frac=round(p_card / p_total, 4),
        noise_pow_frac=round(p_noise / p_total, 4),
        dc_drift_CLE=round(drift_span['CLE'], 2),
        dc_drift_CRE=round(drift_span['CRE'], 2),
        dc_drift_CH=round(drift_span['CH'], 2),
        ac_amp_primary=round(ac_amp, 3),
        dc_drift_over_ac=round(drift_span['CLE'] / (ac_amp + 1e-9), 1),
    ))

    del s, cle, cre, ch, raw
    gc.collect()

if not CACHE.exists() or RECOMPUTE:
    with open(CACHE, 'wb') as f:
        pickle.dump(dict(records=records, snr_by_session=snr_by_session,
                         psd_by_session=psd_by_session,
                         drift_by_session=drift_by_session,
                         per_chan_snr=per_chan_snr, labels=labels), f)
    print(f"Cached summaries to {CACHE.name}")

summary = pd.DataFrame(records)
csv_path = OUT_DIR / 'signal_characterization_summary.csv'
summary.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")
print(summary.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Simplified in-band SNR
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating fig8 — simplified SNR...")
fig8, axes = plt.subplots(1, 3, figsize=(17, 5.4),
                          gridspec_kw={'wspace': 0.30,
                                       'width_ratios': [2.2, 1.1, 1.3]})

# (A) per-session SNR box (primary channel), resp vs cardiac
ax = axes[0]
x = np.arange(len(labels))
w = 0.36
resp_data = [snr_by_session[l]['resp'][np.isfinite(snr_by_session[l]['resp'])]
             for l in labels]
card_data = [snr_by_session[l]['card'][np.isfinite(snr_by_session[l]['card'])]
             for l in labels]
bp_r = ax.boxplot(resp_data, positions=x - w/2, widths=w*0.8,
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color='black', lw=1.4))
bp_c = ax.boxplot(card_data, positions=x + w/2, widths=w*0.8,
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color='black', lw=1.4))
for p in bp_r['boxes']:
    p.set_facecolor('#27AE60'); p.set_alpha(0.65)
for p in bp_c['boxes']:
    p.set_facecolor('#E74C3C'); p.set_alpha(0.65)
ax.axhline(0, color='gray', ls=':', lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Simplified SNR (dB)', fontsize=9)
ax.set_title(f'(A) Per-session in-band SNR — {PRIMARY}',
             fontsize=10, fontweight='bold')
ax.legend([bp_r['boxes'][0], bp_c['boxes'][0]],
          ['Respiratory (0.1–0.5 Hz)', 'Cardiac (0.5–3.0 Hz)'],
          fontsize=8, loc='upper right')
ax.grid(True, axis='y', alpha=0.25)

# (B) channel comparison — median SNR averaged over sessions
ax = axes[1]
chan_order = ['CH', 'CLE', 'CRE', 'CLE-CRE']
resp_means = [np.nanmean([per_chan_snr[l][c]['resp'] for l in labels])
              for c in chan_order]
card_means = [np.nanmean([per_chan_snr[l][c]['card'] for l in labels])
              for c in chan_order]
xc = np.arange(len(chan_order))
ax.bar(xc - 0.2, resp_means, 0.4, color='#27AE60', alpha=0.8, label='Resp')
ax.bar(xc + 0.2, card_means, 0.4, color='#E74C3C', alpha=0.8, label='Cardiac')
ax.axhline(0, color='gray', ls=':', lw=0.9)
ax.set_xticks(xc); ax.set_xticklabels(chan_order, fontsize=8)
ax.set_ylabel('Median SNR (dB)', fontsize=9)
ax.set_title('(B) SNR by channel\n(mean over 12 sessions)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.25)

# (C) definition schematic — mean PSD with the three bands shaded
ax = axes[2]
# average the primary-channel PSD across sessions on a common grid
freqs0 = psd_by_session[labels[0]][PRIMARY][0]
psd_stack = np.vstack([np.interp(freqs0, psd_by_session[l][PRIMARY][0],
                                 psd_by_session[l][PRIMARY][1]) for l in labels])
psd_mean = psd_stack.mean(axis=0)
fm = (freqs0 >= 0.05) & (freqs0 <= 12.0)
ax.semilogy(freqs0[fm], psd_mean[fm], color='#2C3E50', lw=1.4)
ax.axvspan(*RESP_BAND, color='#27AE60', alpha=0.22, label='Resp (signal)')
ax.axvspan(*CARD_BAND, color='#E74C3C', alpha=0.15, label='Cardiac (signal)')
ax.axvspan(*NOISE_BAND, color='#7F8C8D', alpha=0.25, label='Noise floor (5–10 Hz)')
ax.set_xlabel('Frequency (Hz)', fontsize=9)
ax.set_ylabel('PSD (a.u.²/Hz)', fontsize=9)
ax.set_title('(C) SNR definition\nsignal-band vs noise-floor density',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.grid(True, which='both', alpha=0.2)
ax.set_xlim(0.05, 12.0)

fig8.suptitle('Fig. 8 — Simplified In-Band Signal-to-Noise Ratio (all 12 sessions)',
              fontsize=13, fontweight='bold', y=1.02)
fig8.savefig(OUT_DIR / 'fig8_simplified_snr.png', dpi=200,
             bbox_inches='tight', facecolor='white')
plt.close(fig8)
print("  Saved fig8_simplified_snr.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Mean capacitance drift across the session
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig9 — mean capacitance drift...")
n = len(labels)
ncol = 3
nrow = int(np.ceil(n / ncol))
fig9, axes = plt.subplots(nrow, ncol, figsize=(16, 3.0 * nrow),
                          gridspec_kw={'hspace': 0.55, 'wspace': 0.28})
axes = np.atleast_1d(axes).ravel()
chan_colors = {'CLE': CAP_COLORS['CLE'], 'CRE': CAP_COLORS['CRE'],
               'CH': CAP_COLORS['CH']}
# Each channel is plotted as deviation from its own session-median so the three
# baselines (which sit at very different absolute offsets) overlay near zero and
# the *within-night change* is directly readable. y-limits use robust
# percentiles so a single dropout spike does not flatten the gradual drift.
for i, label in enumerate(labels):
    ax = axes[i]
    iqrs = []
    for ch_name in ['CLE', 'CRE', 'CH']:
        t_b, base = drift_by_session[label][ch_name]
        centered = base - np.nanmedian(base)
        ax.plot(t_b, centered, lw=1.1, color=chan_colors[ch_name], label=ch_name)
        q25, q75 = np.nanpercentile(centered, [25, 75])
        iqrs.append(q75 - q25)
    ax.axhline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
    # IQR-based limits: robust even to sustained sensor-dropout segments, which
    # then clip off-screen while the gradual physiological drift stays visible.
    spread = max(max(iqrs), 2.0)
    ax.set_ylim(-6 * spread, 6 * spread)
    row = summary[summary.session == label].iloc[0]
    ax.set_title(f"{label} — {row.subject}  |  DC span "
                 f"CLE {row.dc_drift_CLE:g}, CRE {row.dc_drift_CRE:g}, "
                 f"CH {row.dc_drift_CH:g}",
                 fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Time (hr)', fontsize=8)
    ax.set_ylabel('Δ from median (a.u.)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7, loc='upper right', ncol=3)
for j in range(n, len(axes)):
    axes[j].axis('off')
fig9.suptitle('Fig. 9 — Mean Capacitance (DC Baseline) Drift Across the Night '
              '(<0.01 Hz trend, each channel centered on its session median)',
              fontsize=13, fontweight='bold', y=1.005)
fig9.savefig(OUT_DIR / 'fig9_mean_capacitance_drift.png', dpi=200,
             bbox_inches='tight', facecolor='white')
plt.close(fig9)
print("  Saved fig9_mean_capacitance_drift.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Frequency characteristics
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig10 — frequency characteristics...")
fig10, axes = plt.subplots(1, 3, figsize=(17, 5.4),
                           gridspec_kw={'wspace': 0.30})

# (A) mean PSD per channel (averaged over sessions)
ax = axes[0]
for ch_name in CHANNELS:
    stack = np.vstack([np.interp(freqs0, psd_by_session[l][ch_name][0],
                                 psd_by_session[l][ch_name][1]) for l in labels])
    m = stack.mean(axis=0)
    ax.semilogy(freqs0[fm], m[fm], lw=1.3,
                color=CAP_COLORS[ch_name], label=ch_name)
ax.axvspan(*RESP_BAND, color='#27AE60', alpha=0.15)
ax.axvspan(*CARD_BAND, color='#E74C3C', alpha=0.10)
ax.set_xlabel('Frequency (Hz)', fontsize=9)
ax.set_ylabel('PSD (a.u.²/Hz)', fontsize=9)
ax.set_title('(A) Mean PSD by channel\n(12-session average)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, which='both', alpha=0.2)
ax.set_xlim(0.05, 12.0)

# (B) spectral centroid (energy-weighted mean frequency) per session per band
ax = axes[1]

def _centroid(freqs, psd, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    p = psd[m]
    return float(np.sum(freqs[m] * p) / (np.sum(p) + 1e-30))

cen_resp = np.array([_centroid(*psd_by_session[l][PRIMARY], RESP_BAND)
                     for l in labels])
cen_card = np.array([_centroid(*psd_by_session[l][PRIMARY], CARD_BAND)
                     for l in labels])
rng = np.random.default_rng(0)
ax.scatter(rng.uniform(-0.06, 0.06, n), cen_resp,
           color='#27AE60', s=45, alpha=0.85, zorder=3, edgecolor='white', lw=0.5)
ax.scatter(1 + rng.uniform(-0.06, 0.06, n), cen_card,
           color='#E74C3C', s=45, alpha=0.85, zorder=3, edgecolor='white', lw=0.5)
for xc, data in [(0, cen_resp), (1, cen_card)]:
    ax.hlines(np.median(data), xc - 0.18, xc + 0.18, color='black', lw=1.8, zorder=4)
# expected physiological band extents
ax.axhspan(RESP_LO, RESP_HI, xmin=0.02, xmax=0.48, color='#27AE60', alpha=0.10)
ax.axhspan(CARD_LO, CARD_HI, xmin=0.52, xmax=0.98, color='#E74C3C', alpha=0.08)
ax.set_xlim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Respiratory\nband', 'Cardiac\nband'], fontsize=9)
ax.set_ylabel('Spectral centroid (Hz)', fontsize=9)
ax.set_title('(B) Per-session spectral centroid\n(energy-weighted mean freq., CLE−CRE)',
             fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.25)
axr = ax.twinx()
axr.set_ylim(np.array(ax.get_ylim()) * 60)
axr.set_ylabel('Rate (cycles/min)', fontsize=9)

# (C) band power fractions per session (stacked)
ax = axes[2]
rf = summary['resp_pow_frac'].to_numpy()
cf = summary['card_pow_frac'].to_numpy()
nf = summary['noise_pow_frac'].to_numpy()
other = np.clip(1 - rf - cf - nf, 0, None)
xs = np.arange(n)
ax.bar(xs, rf, color='#27AE60', label='Resp 0.1–0.5 Hz')
ax.bar(xs, cf, bottom=rf, color='#E74C3C', label='Cardiac 0.5–3 Hz')
ax.bar(xs, nf, bottom=rf + cf, color='#7F8C8D', label='Noise 5–10 Hz')
ax.bar(xs, other, bottom=rf + cf + nf, color='#BDC3C7', alpha=0.6, label='Other')
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
ax.set_ylabel('Fraction of 0.05–10 Hz power', fontsize=9)
ax.set_title('(C) Band power fractions\nper session', fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.set_ylim(0, 1.0)

fig10.suptitle('Fig. 10 — CAP Signal Frequency Characteristics (all 12 sessions)',
               fontsize=13, fontweight='bold', y=1.02)
fig10.savefig(OUT_DIR / 'fig10_frequency_characteristics.png', dpi=200,
              bbox_inches='tight', facecolor='white')
plt.close(fig10)
print("  Saved fig10_frequency_characteristics.png")

print("\nDone.")
