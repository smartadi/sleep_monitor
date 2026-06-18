"""
Generate figures demonstrating CAP mask signal energy in respiratory
(0.1–0.5 Hz) and cardiac (0.5–3.0 Hz) bands.

Outputs (writeup/figures/signal_validation/):
  fig5_cap_spectrogram_bands.png   — annotated CAP spectrograms for 3 sessions
  fig6_bandpower_vs_psg_rate.png   — band-power time course vs PSG GT rate
  fig7_inband_snr.png              — in-band vs out-of-band SNR summary
"""

from __future__ import annotations
import sys, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from scipy.signal import welch, spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI, CAP_COLORS,
)
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.preprocessing import preprocess_full
from sleep_monitor.ground_truth import gt_sliding_rates
from sleep_monitor.viz import plot_hypnogram

OUT_DIR = Path(__file__).resolve().parent
REPRESENTATIVE = [0, 4, 10]  # S1N1, S3N1, S6N1 — varied subjects
CHANNEL = 'CLE-CRE'

RESP_BAND = (RESP_LO, RESP_HI)
CARD_BAND = (CARD_LO, CARD_HI)
NOISE_LO  = (3.0, 5.0)
NOISE_HI  = (5.0, 10.0)

BAND_POWER_WIN = 60.0   # seconds per sliding window
BAND_POWER_STEP = 10.0  # seconds step


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sliding_band_power(sig, fs, f_lo, f_hi, win_sec, step_sec,
                        welch_seg_sec=4.0):
    """Absolute band power on a sliding window grid."""
    win_n = int(win_sec * fs)
    step_n = int(step_sec * fs)
    nperseg = int(welch_seg_sec * fs)
    starts = np.arange(0, len(sig) - win_n + 1, step_n)
    t_hr = (starts + win_n / 2) / fs / 3600.0
    power = np.empty(len(starts))
    for i, s0 in enumerate(starts):
        chunk = sig[s0:s0 + win_n].astype(np.float64)
        freqs, psd = welch(chunk, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2, scaling='density')
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        power[i] = np.trapz(psd[mask], freqs[mask])
    return t_hr, power


def _compute_snr_windows(sig, fs, in_band, noise_bands,
                         win_sec=60.0, step_sec=30.0,
                         welch_seg_sec=4.0):
    """Per-window in-band / out-of-band power ratio (dB)."""
    win_n = int(win_sec * fs)
    step_n = int(step_sec * fs)
    nperseg = int(welch_seg_sec * fs)
    starts = np.arange(0, len(sig) - win_n + 1, step_n)
    t_hr = (starts + win_n / 2) / fs / 3600.0
    snr = np.empty(len(starts))
    for i, s0 in enumerate(starts):
        chunk = sig[s0:s0 + win_n].astype(np.float64)
        freqs, psd = welch(chunk, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2, scaling='density')
        in_mask = (freqs >= in_band[0]) & (freqs <= in_band[1])
        in_pwr = np.trapz(psd[in_mask], freqs[in_mask])
        noise_pwr = 0.0
        noise_bw = 0.0
        for nlo, nhi in noise_bands:
            nm = (freqs >= nlo) & (freqs <= nhi)
            noise_pwr += np.trapz(psd[nm], freqs[nm])
            noise_bw += (nhi - nlo)
        in_bw = in_band[1] - in_band[0]
        in_density = in_pwr / in_bw if in_bw > 0 else 1e-12
        noise_density = noise_pwr / noise_bw if noise_bw > 0 else 1e-12
        snr[i] = 10 * np.log10(in_density / (noise_density + 1e-15))
    return t_hr, snr


# ── Load sessions ────────────────────────────────────────────────────────────

print("Loading sessions...")
sessions = []
for idx in REPRESENTATIVE:
    s = load_session(idx)
    s.sleep_profile = load_sleep_profile(s)
    sessions.append(s)

print("Preprocessing...")
prepped = []
for s in sessions:
    full_sigs, gt_sigs = preprocess_full(s, acc_removal=True)
    cap_sig = full_sigs[CHANNEL]['resp']  # full signal (resp-band filtered)
    raw_diff = s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    prepped.append({
        'session': s,
        'raw': raw_diff,
        'full_sigs': full_sigs,
        'gt_sigs': gt_sigs,
    })

print("Computing GT rates...")
gt_rates = []
for s in sessions:
    gt = gt_sliding_rates(s, win_sec=30.0, step_sec=10.0)
    gt_rates.append(gt)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Annotated CAP spectrograms with band overlays
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating fig5 — annotated spectrograms...")
n_sess = len(sessions)
fig5, axes5 = plt.subplots(n_sess, 1, figsize=(14, 3.8 * n_sess),
                           gridspec_kw={'hspace': 0.35})
if n_sess == 1:
    axes5 = [axes5]

for i, (s, p) in enumerate(zip(sessions, prepped)):
    ax = axes5[i]
    raw = p['raw']
    f_max = 5.0

    nperseg_spec = int(FS * 10)
    noverlap_spec = int(FS * 9)
    f, t, Sxx = spectrogram(raw.astype(np.float64), fs=FS,
                             nperseg=nperseg_spec,
                             noverlap=noverlap_spec,
                             scaling='density')
    mask = f <= f_max
    psd_db = 10 * np.log10(Sxx[mask] + 1e-15)

    vmin, vmax = np.nanpercentile(psd_db, [5, 98])
    im = ax.pcolormesh(t / 3600.0, f[mask], psd_db,
                       shading='gouraud', cmap='inferno',
                       vmin=vmin, vmax=vmax, rasterized=True)

    # Resp band overlay
    ax.axhspan(RESP_LO, RESP_HI, color='#27AE60', alpha=0.18, zorder=2)
    ax.axhline(RESP_LO, color='#27AE60', lw=1.2, ls='--', alpha=0.8, zorder=3)
    ax.axhline(RESP_HI, color='#27AE60', lw=1.2, ls='--', alpha=0.8, zorder=3)

    # Cardiac band overlay
    ax.axhspan(CARD_LO, CARD_HI, color='#E74C3C', alpha=0.12, zorder=2)
    ax.axhline(CARD_LO, color='#E74C3C', lw=1.2, ls='--', alpha=0.8, zorder=3)
    ax.axhline(CARD_HI, color='#E74C3C', lw=1.2, ls='--', alpha=0.8, zorder=3)

    # Labels on right side
    ax.text(t[-1] / 3600.0 + 0.02 * s.duration_hr, (RESP_LO + RESP_HI) / 2,
            f'Resp\n{RESP_LO}–{RESP_HI} Hz',
            va='center', ha='left', fontsize=7.5, color='#27AE60',
            fontweight='bold', clip_on=False)
    ax.text(t[-1] / 3600.0 + 0.02 * s.duration_hr, (CARD_LO + CARD_HI) / 2,
            f'Cardiac\n{CARD_LO}–{CARD_HI} Hz',
            va='center', ha='left', fontsize=7.5, color='#E74C3C',
            fontweight='bold', clip_on=False)

    ax.set_ylabel('Frequency (Hz)', fontsize=9)
    ax.set_xlabel('Time (hr)', fontsize=9)
    ax.set_title(f'{s.label} — {s.subject} ({s.duration_hr:.1f} hr)  |  '
                 f'CLE−CRE spectrogram (0–{f_max} Hz)',
                 fontsize=10, fontweight='bold')
    ax.set_ylim(0, f_max)
    cb = fig5.colorbar(im, ax=ax, label='PSD (dB)', shrink=0.8, pad=0.08)
    cb.ax.tick_params(labelsize=7)

fig5.suptitle('Fig. 5 — CAP Mask Spectrogram: Energy in Respiratory & Cardiac Bands',
              fontsize=13, fontweight='bold', y=1.01)
fig5.savefig(OUT_DIR / 'fig5_cap_spectrogram_bands.png',
             dpi=200, bbox_inches='tight', facecolor='white')
print(f"  Saved fig5_cap_spectrogram_bands.png")
plt.close(fig5)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Band-power time course vs PSG rate
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating fig6 — band power vs PSG rate...")
fig6, axes6 = plt.subplots(n_sess * 2, 1, figsize=(14, 2.8 * n_sess * 2),
                            gridspec_kw={'hspace': 0.45})

for i, (s, p, gt) in enumerate(zip(sessions, prepped, gt_rates)):
    raw = p['raw']

    # Respiratory band power
    t_resp, bp_resp = _sliding_band_power(raw, FS, RESP_LO, RESP_HI,
                                          BAND_POWER_WIN, BAND_POWER_STEP)
    # Cardiac band power
    t_card, bp_card = _sliding_band_power(raw, FS, CARD_LO, CARD_HI,
                                          BAND_POWER_WIN, BAND_POWER_STEP)

    # --- Resp panel ---
    ax_r = axes6[i * 2]
    color_bp = '#27AE60'
    color_gt = '#2C3E50'

    ax_r.fill_between(t_resp, 0, bp_resp / np.nanmax(bp_resp),
                      color=color_bp, alpha=0.25, label='CAP resp band power')
    ax_r.plot(t_resp, bp_resp / np.nanmax(bp_resp),
              color=color_bp, lw=1.0, alpha=0.8)
    ax_r.set_ylabel('Norm. band power', fontsize=8, color=color_bp)
    ax_r.set_ylim(0, 1.15)
    ax_r.tick_params(axis='y', labelcolor=color_bp, labelsize=7)

    ax_r2 = ax_r.twinx()
    ok = np.isfinite(gt['resp_hz'])
    ax_r2.plot(gt['t_hr'][ok], gt['resp_hz'][ok] * 60,
               color=color_gt, lw=1.0, alpha=0.7, label='PSG resp rate')
    ax_r2.set_ylabel('PSG resp (br/min)', fontsize=8, color=color_gt)
    ax_r2.tick_params(axis='y', labelcolor=color_gt, labelsize=7)

    lines1, labels1 = ax_r.get_legend_handles_labels()
    lines2, labels2 = ax_r2.get_legend_handles_labels()
    ax_r.legend(lines1 + lines2, labels1 + labels2,
                fontsize=7, loc='upper right', ncol=2)
    ax_r.set_title(f'{s.label} — Respiratory band (0.1–0.5 Hz): '
                   f'CAP power tracks breathing',
                   fontsize=9, fontweight='bold')
    ax_r.grid(True, alpha=0.2)

    # --- Cardiac panel ---
    ax_c = axes6[i * 2 + 1]
    color_bp_c = '#E74C3C'

    ax_c.fill_between(t_card, 0, bp_card / np.nanmax(bp_card),
                      color=color_bp_c, alpha=0.25, label='CAP cardiac band power')
    ax_c.plot(t_card, bp_card / np.nanmax(bp_card),
              color=color_bp_c, lw=1.0, alpha=0.8)
    ax_c.set_ylabel('Norm. band power', fontsize=8, color=color_bp_c)
    ax_c.set_ylim(0, 1.15)
    ax_c.tick_params(axis='y', labelcolor=color_bp_c, labelsize=7)

    ax_c2 = ax_c.twinx()
    ok_c = np.isfinite(gt['card_hz'])
    ax_c2.plot(gt['t_hr'][ok_c], gt['card_hz'][ok_c] * 60,
               color=color_gt, lw=1.0, alpha=0.7, label='PSG cardiac rate')
    ax_c2.set_ylabel('PSG cardiac (BPM)', fontsize=8, color=color_gt)
    ax_c2.tick_params(axis='y', labelcolor=color_gt, labelsize=7)

    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2,
                fontsize=7, loc='upper right', ncol=2)
    ax_c.set_title(f'{s.label} — Cardiac band (0.5–3.0 Hz): '
                   f'CAP power tracks heart rate',
                   fontsize=9, fontweight='bold')
    ax_c.set_xlabel('Time (hr)', fontsize=8)
    ax_c.grid(True, alpha=0.2)

fig6.suptitle('Fig. 6 — CAP Band Power Time Course vs PSG Ground-Truth Rate',
              fontsize=13, fontweight='bold', y=1.005)
fig6.savefig(OUT_DIR / 'fig6_bandpower_vs_psg_rate.png',
             dpi=200, bbox_inches='tight', facecolor='white')
print(f"  Saved fig6_bandpower_vs_psg_rate.png")
plt.close(fig6)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: In-band vs out-of-band SNR
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating fig7 — in/out-of-band SNR...")

# Noise reference bands for each target
resp_noise_bands = [(3.5, 5.0)]       # well above cardiac
card_noise_bands = [(3.5, 5.0)]       # above cardiac upper bound

# Collect per-session SNR distributions
all_resp_snr = {}
all_card_snr = {}

for i, (s, p) in enumerate(zip(sessions, prepped)):
    raw = p['raw']
    t_r, snr_r = _compute_snr_windows(raw, FS, RESP_BAND, resp_noise_bands)
    t_c, snr_c = _compute_snr_windows(raw, FS, CARD_BAND, card_noise_bands)
    all_resp_snr[s.label] = snr_r
    all_card_snr[s.label] = snr_c

fig7, axes7 = plt.subplots(1, 3, figsize=(16, 5.5),
                            gridspec_kw={'wspace': 0.35})

# --- Panel A: SNR boxplot by session ---
ax_a = axes7[0]
labels_list = list(all_resp_snr.keys())
resp_data = [all_resp_snr[k][np.isfinite(all_resp_snr[k])] for k in labels_list]
card_data = [all_card_snr[k][np.isfinite(all_card_snr[k])] for k in labels_list]

x = np.arange(len(labels_list))
w = 0.35

bp_r = ax_a.boxplot([d for d in resp_data],
                     positions=x - w/2, widths=w * 0.8,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', lw=1.5))
bp_c = ax_a.boxplot([d for d in card_data],
                     positions=x + w/2, widths=w * 0.8,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', lw=1.5))
for patch in bp_r['boxes']:
    patch.set_facecolor('#27AE60')
    patch.set_alpha(0.6)
for patch in bp_c['boxes']:
    patch.set_facecolor('#E74C3C')
    patch.set_alpha(0.6)

ax_a.axhline(0, color='gray', lw=0.8, ls=':', alpha=0.5)
ax_a.set_xticks(x)
ax_a.set_xticklabels(labels_list, fontsize=8)
ax_a.set_ylabel('SNR vs noise floor (dB)', fontsize=9)
ax_a.set_title('(A) In-band SNR by session', fontsize=10, fontweight='bold')
ax_a.legend([bp_r['boxes'][0], bp_c['boxes'][0]],
            ['Resp (0.1–0.5 Hz)', 'Cardiac (0.5–3.0 Hz)'],
            fontsize=8, loc='upper right')
ax_a.grid(True, axis='y', alpha=0.25)

# --- Panel B: Time course of SNR (first session) ---
ax_b = axes7[1]
raw0 = prepped[0]['raw']
s0 = sessions[0]
t_sr, snr_resp_0 = _compute_snr_windows(raw0, FS, RESP_BAND, resp_noise_bands,
                                          win_sec=60, step_sec=15)
t_sc, snr_card_0 = _compute_snr_windows(raw0, FS, CARD_BAND, card_noise_bands,
                                          win_sec=60, step_sec=15)

ax_b.plot(t_sr, snr_resp_0, color='#27AE60', lw=0.9, alpha=0.8,
          label='Resp band SNR')
ax_b.plot(t_sc, snr_card_0, color='#E74C3C', lw=0.9, alpha=0.8,
          label='Cardiac band SNR')
ax_b.axhline(0, color='gray', lw=0.8, ls=':', alpha=0.5)
ax_b.fill_between(t_sr, 0, snr_resp_0, where=snr_resp_0 > 0,
                  color='#27AE60', alpha=0.1)
ax_b.fill_between(t_sc, 0, snr_card_0, where=snr_card_0 > 0,
                  color='#E74C3C', alpha=0.1)
ax_b.set_xlabel('Time (hr)', fontsize=9)
ax_b.set_ylabel('SNR (dB)', fontsize=9)
ax_b.set_title(f'(B) SNR time course — {s0.label}', fontsize=10, fontweight='bold')
ax_b.legend(fontsize=8, loc='upper right')
ax_b.grid(True, alpha=0.25)

# --- Panel C: Aggregate PSD with band annotations ---
ax_c = axes7[2]
# Compute full-night PSD for each session and average
psd_all = []
for p in prepped:
    raw = p['raw']
    nperseg_psd = int(FS * 30)
    freqs, psd = welch(raw.astype(np.float64), fs=FS,
                       nperseg=nperseg_psd,
                       noverlap=nperseg_psd // 2,
                       scaling='density')
    psd_all.append(psd)

psd_mean = np.mean(psd_all, axis=0)
psd_std = np.std(psd_all, axis=0)

f_plot_mask = (freqs >= 0.05) & (freqs <= 8.0)
ax_c.semilogy(freqs[f_plot_mask], psd_mean[f_plot_mask],
              color='#2C3E50', lw=1.5, label='Mean PSD (3 sessions)')
ax_c.fill_between(freqs[f_plot_mask],
                  (psd_mean - psd_std)[f_plot_mask],
                  (psd_mean + psd_std)[f_plot_mask],
                  color='#2C3E50', alpha=0.15, label='±1 SD')

# Shade bands
ax_c.axvspan(RESP_LO, RESP_HI, color='#27AE60', alpha=0.2,
             label=f'Resp ({RESP_LO}–{RESP_HI} Hz)')
ax_c.axvspan(CARD_LO, CARD_HI, color='#E74C3C', alpha=0.15,
             label=f'Cardiac ({CARD_LO}–{CARD_HI} Hz)')

# Compute and annotate median band powers
for band_name, (flo, fhi), color in [
    ('Resp', RESP_BAND, '#27AE60'),
    ('Cardiac', CARD_BAND, '#E74C3C'),
]:
    bm = (freqs >= flo) & (freqs <= fhi)
    bp = np.trapz(psd_mean[bm], freqs[bm])
    ax_c.annotate(f'{band_name}\n{bp:.2e}',
                  xy=((flo + fhi) / 2, psd_mean[bm].max()),
                  xytext=(0, 15), textcoords='offset points',
                  fontsize=7.5, ha='center', fontweight='bold', color=color,
                  arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

ax_c.set_xlabel('Frequency (Hz)', fontsize=9)
ax_c.set_ylabel('PSD (a.u.²/Hz)', fontsize=9)
ax_c.set_title('(C) Mean PSD — band energy', fontsize=10, fontweight='bold')
ax_c.legend(fontsize=7, loc='upper right')
ax_c.grid(True, alpha=0.25)
ax_c.set_xlim(0.05, 8.0)

fig7.suptitle('Fig. 7 — In-Band Signal-to-Noise Ratio: CAP Mask Energy Concentration',
              fontsize=13, fontweight='bold', y=1.02)
fig7.savefig(OUT_DIR / 'fig7_inband_snr.png',
             dpi=200, bbox_inches='tight', facecolor='white')
print(f"  Saved fig7_inband_snr.png")
plt.close(fig7)

# ── Print summary statistics ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BAND ENERGY SUMMARY")
print("=" * 70)
for s, p in zip(sessions, prepped):
    raw = p['raw']
    freqs, psd = welch(raw.astype(np.float64), fs=FS,
                       nperseg=int(FS * 30),
                       noverlap=int(FS * 15),
                       scaling='density')
    resp_mask = (freqs >= RESP_LO) & (freqs <= RESP_HI)
    card_mask = (freqs >= CARD_LO) & (freqs <= CARD_HI)
    noise_mask = (freqs >= 3.5) & (freqs <= 5.0)
    total_mask = (freqs >= 0.05) & (freqs <= 10.0)

    resp_pwr = np.trapz(psd[resp_mask], freqs[resp_mask])
    card_pwr = np.trapz(psd[card_mask], freqs[card_mask])
    noise_pwr = np.trapz(psd[noise_mask], freqs[noise_mask])
    total_pwr = np.trapz(psd[total_mask], freqs[total_mask])

    resp_frac = resp_pwr / total_pwr * 100
    card_frac = card_pwr / total_pwr * 100

    resp_bw = RESP_HI - RESP_LO
    card_bw = CARD_HI - CARD_LO
    noise_bw = 1.5

    resp_snr = 10 * np.log10((resp_pwr / resp_bw) / (noise_pwr / noise_bw + 1e-15))
    card_snr = 10 * np.log10((card_pwr / card_bw) / (noise_pwr / noise_bw + 1e-15))

    print(f"\n{s.label} ({s.subject}, {s.duration_hr:.1f} hr):")
    print(f"  Resp band power:    {resp_pwr:.4e}  ({resp_frac:.1f}% of total)")
    print(f"  Cardiac band power: {card_pwr:.4e}  ({card_frac:.1f}% of total)")
    print(f"  Resp SNR vs noise:  {resp_snr:+.1f} dB")
    print(f"  Cardiac SNR vs noise: {card_snr:+.1f} dB")

print("\nDone.")
