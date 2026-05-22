"""Validation breath rate — peak-to-rate scaling ratio k per experiment mode.

Uses z(Cvl)−z(Cvr) (z-scored per mode) as the cap signal, with a loose peak
detector vs Thorax ground truth.  Runs on the first validation subject (S0001).

Outputs (saved to notebooks/plots/):
  - validation_k_and_rates.png   : k by mode + GT vs raw vs scaled bar chart
  - validation_peak_overlay.png  : per-mode signal + peak overlay
  - validation_k_comparison.png  : per-mode k vs session-wide k error
  - validation_results.csv       : full results table
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from load_validation import (
    load_subject, extract_by_mode,
    MODE_ORDER, FS, SUBJECT_FILES,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.rates import rate_acf

# ── Config ───────────────────────────────────────────────────────────────────
RESP_LO, RESP_HI = 0.1, 0.5

PROM_CAP   = 0.05
DIST_CAP_S = 0.4
PROM_GT    = 0.5
DIST_GT_S  = 1.5

OUT_DIR = Path(__file__).resolve().parent.parent / 'notebooks' / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'figure.dpi': 150, 'font.size': 9})


# ── Helpers ──────────────────────────────────────────────────────────────────

def count_peaks(sig, prom_factor, min_dist_s, fs):
    md = max(1, int(round(min_dist_s * fs)))
    sw = max(3, md // 4)
    sm = np.convolve(sig.astype(np.float64), np.ones(sw) / sw, mode='same')
    pks, _ = find_peaks(sm, distance=md, prominence=prom_factor * np.std(sm))
    return len(pks), pks


def zscore_per_mode(series):
    x = series.values.astype(np.float64)
    sd = np.std(x)
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / sd


# ── Load first subject ──────────────────────────────────────────────────────

print("Loading first validation subject...")
df = load_subject(SUBJECT_FILES[0])
subject = df['subject'].iloc[0]
df['acc_mag'] = np.sqrt(df['aX']**2 + df['aY']**2 + df['aZ']**2)
print(f"  {subject}: {len(df):,} rows, {df['t_sec'].iloc[-1]:.1f}s, "
      f"modes={df['experimentMode'].nunique()}")


# ── Compute k per mode ──────────────────────────────────────────────────────

print("\nComputing per-mode scaling ratio k (z(Cvl)-z(Cvr) vs Thorax)...")
rows = []
mode_signals = {}  # stash for plotting

for mode in MODE_ORDER:
    mdf = extract_by_mode(df, mode)
    if len(mdf) < int(5.0 * FS):
        print(f"  {mode}: SKIP ({len(mdf)} samples)")
        continue

    z_cvl = zscore_per_mode(mdf['Cvl'])
    z_cvr = zscore_per_mode(mdf['Cvr'])
    raw_diff = z_cvl - z_cvr

    acc = mdf['acc_mag'].values.astype(np.float64)
    sig = remove_acc_artifact(raw_diff, acc, RESP_LO, RESP_HI, FS)
    n_cap, pks_cap = count_peaks(sig, PROM_CAP, DIST_CAP_S, FS)

    thx = mdf['Thorax'].values.astype(np.float64)
    thx_bp = bandpass(thx, RESP_LO, RESP_HI, FS)
    n_gt, pks_gt = count_peaks(thx_bp, PROM_GT, DIST_GT_S, FS)
    gt_rate_bpm = rate_acf(thx_bp, RESP_LO, RESP_HI, FS) * 60.0

    k = n_cap / n_gt if n_gt > 0 else np.nan
    cap_rate_raw = n_cap / (len(mdf) / FS) * 60.0
    cap_rate_scaled = (cap_rate_raw / k) if np.isfinite(k) and k > 0 else np.nan

    rows.append({
        'subject': subject, 'mode': mode,
        'duration_s': len(mdf) / FS,
        'n_cap_peaks': n_cap, 'n_gt_peaks': n_gt, 'k': k,
        'gt_rate_bpm': gt_rate_bpm,
        'cap_rate_raw_bpm': cap_rate_raw,
        'cap_rate_scaled_bpm': cap_rate_scaled,
        'error_raw_bpm': cap_rate_raw - gt_rate_bpm,
        'error_scaled_bpm': (cap_rate_scaled - gt_rate_bpm
                             if np.isfinite(cap_rate_scaled) else np.nan),
    })
    mode_signals[mode] = dict(sig=sig, thx_bp=thx_bp,
                              pks_cap=pks_cap, pks_gt=pks_gt,
                              n_samples=len(mdf))

    print(f"  {mode}: {len(mdf)/FS:.1f}s  gt={gt_rate_bpm:.1f} br/min  "
          f"cap_raw={cap_rate_raw:.1f}  k={k:.2f}  scaled={cap_rate_scaled:.1f}")

results = pd.DataFrame(rows)
results['mode'] = pd.Categorical(results['mode'], categories=MODE_ORDER, ordered=True)
results = results.sort_values('mode').reset_index(drop=True)

csv_out = OUT_DIR / 'validation_results.csv'
results.to_csv(csv_out, index=False)
print(f"\nResults table -> {csv_out}")
print(results.to_string(index=False))


# ── Plot 1: k by mode + rate comparison ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
ax.plot(results['mode'].astype(str), results['k'], 'o-',
        color='steelblue', markersize=6)
ax.axhline(results['k'].median(), color='red', ls='--', alpha=0.6,
           label=f'median k = {results["k"].median():.2f}')
ax.axhline(1.0, color='gray', ls='--', alpha=0.4)
ax.set_ylabel('k  (n_cap / n_thorax)')
ax.set_title(f'{subject} — Scaling ratio k by experiment mode (Cvl−Cvr)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
x = np.arange(len(results))
w = 0.25
ax.bar(x - w, results['gt_rate_bpm'], w,
       color='black', alpha=0.7, label='GT (Thorax)')
ax.bar(x, results['cap_rate_raw_bpm'], w,
       color='gray', alpha=0.6, label='Cap raw')
ax.bar(x + w, results['cap_rate_scaled_bpm'], w,
       color='steelblue', alpha=0.8, label='Cap scaled (per-mode k)')
ax.set_xticks(x)
ax.set_xticklabels(results['mode'].astype(str), rotation=45, ha='right')
ax.set_ylabel('br/min')
ax.set_title(f'{subject} — Breath rate: GT vs raw vs scaled')
ax.grid(True, axis='y', alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
out1 = OUT_DIR / 'validation_k_and_rates.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot 1 -> {out1}")

print(f"\nk summary: median={results['k'].median():.3f}  "
      f"mean={results['k'].mean():.3f}  std={results['k'].std():.3f}  "
      f"range=[{results['k'].min():.3f}, {results['k'].max():.3f}]")


# ── Plot 2: per-mode signal + peak overlay ──────────────────────────────────

modes_present = [m for m in MODE_ORDER if m in mode_signals]
n_modes = len(modes_present)
fig, axes = plt.subplots(n_modes, 1, figsize=(16, 2.5 * n_modes), sharex=False)
if n_modes == 1:
    axes = [axes]

for ax, mode in zip(axes, modes_present):
    ms = mode_signals[mode]
    t = np.arange(ms['n_samples']) / FS
    sig, thx_bp = ms['sig'], ms['thx_bp']
    pks_cap, pks_gt = ms['pks_cap'], ms['pks_gt']

    sig_n = (sig - sig.mean()) / (sig.std() + 1e-12)
    thx_n = (thx_bp - thx_bp.mean()) / (thx_bp.std() + 1e-12)

    ax.plot(t, thx_n, color='black', lw=0.6, alpha=0.7, label='Thorax')
    ax.plot(t, sig_n, color='steelblue', lw=0.5, alpha=0.7,
            label='z(Cvl)−z(Cvr)')
    ax.plot(t[pks_gt], thx_n[pks_gt], 'kv', markersize=4, alpha=0.6)
    ax.plot(t[pks_cap], sig_n[pks_cap], 'r^', markersize=3, alpha=0.6)

    n_cap, n_gt = len(pks_cap), len(pks_gt)
    k = n_cap / n_gt if n_gt > 0 else np.nan
    ax.set_title(f'{mode}  |  k={k:.2f}  cap={n_cap}  gt={n_gt}  '
                 f'({ms["n_samples"]/FS:.0f}s)', fontsize=9, loc='left')
    ax.set_ylabel('z', fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)

axes[0].legend(loc='upper right', fontsize=7)
axes[-1].set_xlabel('Time within phase (s)')
fig.suptitle(f'{subject} — z(Cvl)−z(Cvr) vs Thorax by mode (resp bandpass)',
             fontsize=12, y=1.005)
fig.tight_layout()
out2 = OUT_DIR / 'validation_peak_overlay.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot 2 -> {out2}")


# ── Plot 3: per-mode k vs session-wide k ────────────────────────────────────

k_session = results['k'].median()
results['cap_rate_session_k_bpm'] = results['cap_rate_raw_bpm'] / k_session
results['error_session_k_bpm'] = (results['cap_rate_session_k_bpm']
                                  - results['gt_rate_bpm'])

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(results))
w = 0.3
ax.bar(x - w/2, results['error_scaled_bpm'].abs(), w,
       color='steelblue', alpha=0.8, label='Per-mode k (|error|)')
ax.bar(x + w/2, results['error_session_k_bpm'].abs(), w,
       color='coral', alpha=0.8,
       label=f'Session-wide k={k_session:.2f} (|error|)')
ax.set_xticks(x)
ax.set_xticklabels(results['mode'].astype(str), rotation=45, ha='right')
ax.set_ylabel('|error| (br/min)')
ax.set_title(f'{subject} — Per-mode k vs session-wide k: absolute rate error')
ax.grid(True, axis='y', alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
out3 = OUT_DIR / 'validation_k_comparison.png'
fig.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot 3 -> {out3}")

mae_permode = results['error_scaled_bpm'].abs().mean()
mae_session = results['error_session_k_bpm'].abs().mean()
print(f"\nMAE per-mode k:      {mae_permode:.2f} br/min  "
      f"(by construction ~= 0)")
print(f"MAE session-wide k={k_session:.2f}: {mae_session:.2f} br/min")
print("\nDone.")
