"""Phase-by-phase peak/rate analysis across all validation subjects.

Compares lenient CAP peak detection (z(Cvl)-z(Cvr)) against Thorax GT peaks
for each subject and validation phase (experimentMode).

Uses updated GT detection aligned with sleep_monitor.ground_truth fallback:
    1) Thorax bandpass (resp band)
    2) Prominence-based peak detection (prom=0.05*std, distance~0.6/f_hi)
    3) Physiological interval quality filtering

Outputs (saved to notebooks/plots/validation/):
    - val_peaks_results.csv             : full per-subject/per-phase table
    - val_peaks_k_heatmap.png           : k heatmap (subjects x phases)
    - val_peaks_k_by_mode.png           : k per phase, all subjects overlaid
    - val_peaks_counts_grid.png         : CAP vs GT peak counts per subject
    - val_peaks_per_subject.png         : per-subject signal+peak overlays
    - val_phase_rate_error_summary.png  : phase-wise absolute/signed rate error
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_validation import (
    load_subject, load_all, extract_by_mode,
    MODE_ORDER, FS, SUBJECT_FILES,
)
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact

# ── Config ───────────────────────────────────────────────────────────────────
RESP_LO, RESP_HI = 0.1, 0.5

# Lenient cap detector
PROM_CAP   = 0.05
DIST_CAP_S = 0.4

# Updated Thorax GT detector (aligned with ground_truth fallback)
PROM_GT_FACTOR = 0.05
GT_DIST_SCALE  = 0.6  # distance ~= GT_DIST_SCALE * fs / RESP_HI

MIN_PHASE_S = 5.0

OUT_DIR = Path(__file__).resolve().parent.parent / 'notebooks' / 'plots' / 'validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'figure.dpi': 150, 'font.size': 9})


# ── Helpers ──────────────────────────────────────────────────────────────────

def count_peaks(sig, prom_factor, min_dist_s, fs):
    md = max(1, int(round(min_dist_s * fs)))
    sw = max(3, md // 4)
    sm = np.convolve(sig.astype(np.float64), np.ones(sw) / sw, mode='same')
    pks, props = find_peaks(sm, distance=md, prominence=prom_factor * np.std(sm))
    mean_prom = float(np.mean(props['prominences'])) if len(pks) > 0 else np.nan
    return len(pks), pks, mean_prom


def zscore_per_mode(vals):
    x = vals.astype(np.float64)
    sd = np.std(x)
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / sd


def _quality_filter_peaks(peak_indices, fs, rate_lo_hz, rate_hi_hz):
    """Remove peaks that imply non-physiological inter-peak intervals."""
    if len(peak_indices) < 2:
        return peak_indices
    intervals = np.diff(peak_indices) / fs
    min_interval = 1.0 / rate_hi_hz
    max_interval = 1.0 / rate_lo_hz
    good = (intervals >= min_interval) & (intervals <= max_interval)
    keep = np.ones(len(peak_indices), dtype=bool)
    for i in range(len(good)):
        if not good[i]:
            keep[i + 1] = False
    return peak_indices[keep]


def detect_gt_peaks_updated(thorax_sig, f_lo, f_hi, fs):
    """Updated GT peak detection on Thorax using fallback-style logic."""
    thx_bp = bandpass(thorax_sig.astype(np.float64), f_lo, f_hi, fs)
    min_dist = max(1, int(fs / f_hi * GT_DIST_SCALE))
    prom = PROM_GT_FACTOR * np.std(thx_bp)
    pks, props = find_peaks(thx_bp, distance=min_dist, prominence=prom)
    pks = _quality_filter_peaks(pks, fs, f_lo, f_hi)
    mean_prom = float(np.mean(props['prominences'])) if len(props.get('prominences', [])) else np.nan
    return thx_bp, pks, mean_prom


# ── Load all subjects ────────────────────────────────────────────────────────

print("Loading all validation subjects...")
all_df = load_all()
all_df['acc_mag'] = np.sqrt(all_df['aX']**2 + all_df['aY']**2 + all_df['aZ']**2)
subjects = sorted(all_df['subject'].unique())
print(f"  {len(subjects)} subjects: {subjects}")


# ── Compute peaks per subject per mode ───────────────────────────────────────

print("\nAnalysing peaks/rates (phase-by-phase, lenient CAP vs updated GT) ...")
rows = []

for subj in subjects:
    sdf = all_df[all_df['subject'] == subj]
    for mode in MODE_ORDER:
        mdf = extract_by_mode(sdf, mode)
        if len(mdf) < int(MIN_PHASE_S * FS):
            continue

        duration_s = len(mdf) / FS

        # Cap signal: per-mode z-score difference
        z_cvl = zscore_per_mode(mdf['Cvl'].values)
        z_cvr = zscore_per_mode(mdf['Cvr'].values)
        raw_diff = z_cvl - z_cvr
        acc = mdf['acc_mag'].values.astype(np.float64)
        sig = remove_acc_artifact(raw_diff, acc, RESP_LO, RESP_HI, FS)
        n_cap, pks_cap, prom_cap = count_peaks(sig, PROM_CAP, DIST_CAP_S, FS)

        # GT: updated Thorax detector (fallback-style + quality filter)
        thx_bp, pks_gt, prom_gt = detect_gt_peaks_updated(
            mdf['Thorax'].values.astype(np.float64), RESP_LO, RESP_HI, FS
        )
        n_gt = len(pks_gt)

        k = n_cap / n_gt if n_gt > 0 else np.nan
        gt_rate_bpm = (n_gt / duration_s) * 60.0
        cap_rate_bpm = (n_cap / duration_s) * 60.0
        cap_rate_scaled_bpm = (cap_rate_bpm / k) if np.isfinite(k) and k > 0 else np.nan

        # Peak interval stats (IPI = inter-peak interval)
        ipi_cap = np.diff(pks_cap) / FS if len(pks_cap) > 1 else np.array([])
        ipi_gt  = np.diff(pks_gt)  / FS if len(pks_gt)  > 1 else np.array([])

        rows.append({
            'subject': subj, 'mode': mode,
            'duration_s': duration_s,
            'n_cap': n_cap, 'n_gt': n_gt, 'k': k,
            'peaks_per_sec_cap': n_cap / duration_s,
            'peaks_per_sec_gt': n_gt / duration_s,
            'gt_rate_bpm': gt_rate_bpm,
            'cap_rate_raw_bpm': cap_rate_bpm,
            'cap_rate_scaled_bpm': cap_rate_scaled_bpm,
            'error_raw_bpm': cap_rate_bpm - gt_rate_bpm,
            'error_scaled_bpm': (cap_rate_scaled_bpm - gt_rate_bpm
                                 if np.isfinite(cap_rate_scaled_bpm) else np.nan),
            'mean_prom_cap': prom_cap, 'mean_prom_gt': prom_gt,
            'ipi_mean_cap': float(np.mean(ipi_cap)) if len(ipi_cap) else np.nan,
            'ipi_std_cap':  float(np.std(ipi_cap))  if len(ipi_cap) else np.nan,
            'ipi_mean_gt':  float(np.mean(ipi_gt))  if len(ipi_gt)  else np.nan,
            'ipi_std_gt':   float(np.std(ipi_gt))   if len(ipi_gt)  else np.nan,
        })

    print(f"  {subj}: done")

results = pd.DataFrame(rows)
results['mode'] = pd.Categorical(results['mode'], categories=MODE_ORDER, ordered=True)
results = results.sort_values(['subject', 'mode']).reset_index(drop=True)

csv_out = OUT_DIR / 'val_peaks_results.csv'
results.to_csv(csv_out, index=False)
print(f"\nResults -> {csv_out}  ({len(results)} rows)")
print(results[['subject', 'mode', 'duration_s', 'n_cap', 'n_gt', 'k']].to_string(index=False))


# ── Plot 1: k heatmap (subjects x modes) ────────────────────────────────────

pivot_k = results.pivot(index='subject', columns='mode', values='k')
pivot_k = pivot_k.reindex(columns=MODE_ORDER)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(pivot_k.values, aspect='auto', cmap='RdYlBu_r', vmin=0.5, vmax=4.0)
ax.set_xticks(range(len(MODE_ORDER)))
ax.set_xticklabels(MODE_ORDER, rotation=45, ha='right')
ax.set_yticks(range(len(subjects)))
ax.set_yticklabels(pivot_k.index)
ax.set_xlabel('Experiment Mode')
ax.set_ylabel('Subject')
ax.set_title('Peak scaling ratio k = n_cap / n_gt  (all subjects x modes)')
for i in range(pivot_k.shape[0]):
    for j in range(pivot_k.shape[1]):
        v = pivot_k.iloc[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if v > 2.5 else 'black')
fig.colorbar(im, ax=ax, label='k', shrink=0.8)
fig.tight_layout()
out1 = OUT_DIR / 'val_peaks_k_heatmap.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot 1 -> {out1}")


# ── Plot 2: k per mode, all subjects overlaid ───────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: k traces
ax = axes[0]
for subj in subjects:
    sub = results[results['subject'] == subj]
    ax.plot(sub['mode'].astype(str), sub['k'], 'o-', markersize=4,
            alpha=0.7, label=subj)
# Cross-subject median per mode
med_k = results.groupby('mode', observed=True)['k'].median()
ax.plot(med_k.index.astype(str), med_k.values, 's--', color='black',
        markersize=6, lw=2, label='median', zorder=10)
ax.axhline(1.0, color='gray', ls='--', alpha=0.4)
ax.axhline(2.0, color='gray', ls='--', alpha=0.4)
ax.set_ylabel('k  (n_cap / n_gt)')
ax.set_title('Scaling ratio k by mode (all subjects)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=2)

# Right: k boxplot per mode
ax = axes[1]
mode_data = [results.loc[results['mode'] == m, 'k'].dropna().values
             for m in MODE_ORDER]
bp = ax.boxplot(mode_data, tick_labels=MODE_ORDER, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.5)
ax.axhline(1.0, color='gray', ls='--', alpha=0.4)
ax.set_ylabel('k')
ax.set_title('k distribution per mode (across subjects)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, axis='y', alpha=0.3)

fig.tight_layout()
out2 = OUT_DIR / 'val_peaks_k_by_mode.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot 2 -> {out2}")


# ── Plot 3: cap vs GT peak counts per subject ───────────────────────────────

n_subj = len(subjects)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes_flat = axes.ravel()

for i, subj in enumerate(subjects):
    ax = axes_flat[i]
    sub = results[results['subject'] == subj].copy()
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w/2, sub['n_gt'].values, w, color='black', alpha=0.7, label='GT (Thorax)')
    ax.bar(x + w/2, sub['n_cap'].values, w, color='steelblue', alpha=0.7, label='Cap (lenient)')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['mode'].astype(str), rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('peak count')
    ax.set_title(f'{subj}', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')

fig.suptitle('Peak counts per phase: updated GT vs CAP (lenient detector)', fontsize=13)
fig.tight_layout()
out3 = OUT_DIR / 'val_peaks_counts_grid.png'
fig.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot 3 -> {out3}")


# ── Plot 4: per-subject signal + peak overlay for two contrasting modes ─────
# Pick the longest resting mode and one postural mode to show per subject.

SHOW_MODES = ['layDownRest', 'degree30', 'turnLeft', 'valsavaMild']

fig, axes = plt.subplots(len(subjects), len(SHOW_MODES),
                         figsize=(4.5 * len(SHOW_MODES), 2.8 * len(subjects)),
                         sharex=False)

for row, subj in enumerate(subjects):
    sdf = all_df[all_df['subject'] == subj]
    for col, mode in enumerate(SHOW_MODES):
        ax = axes[row, col]
        mdf = extract_by_mode(sdf, mode)
        if len(mdf) < int(MIN_PHASE_S * FS):
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='gray')
            ax.set_title(f'{subj} / {mode}', fontsize=8)
            ax.set_yticks([])
            continue

        t = np.arange(len(mdf)) / FS
        z_cvl = zscore_per_mode(mdf['Cvl'].values)
        z_cvr = zscore_per_mode(mdf['Cvr'].values)
        raw_diff = z_cvl - z_cvr
        acc = mdf['acc_mag'].values.astype(np.float64)
        sig = remove_acc_artifact(raw_diff, acc, RESP_LO, RESP_HI, FS)
        n_cap, pks_cap, _ = count_peaks(sig, PROM_CAP, DIST_CAP_S, FS)

        thx_bp, pks_gt, _ = detect_gt_peaks_updated(
            mdf['Thorax'].values.astype(np.float64), RESP_LO, RESP_HI, FS
        )
        n_gt = len(pks_gt)

        sig_n = (sig - sig.mean()) / (sig.std() + 1e-12)
        thx_n = (thx_bp - thx_bp.mean()) / (thx_bp.std() + 1e-12)

        ax.plot(t, thx_n, color='black', lw=0.5, alpha=0.7)
        ax.plot(t, sig_n, color='steelblue', lw=0.4, alpha=0.6)
        ax.plot(t[pks_gt], thx_n[pks_gt], 'kv', markersize=3, alpha=0.6)
        ax.plot(t[pks_cap], sig_n[pks_cap], 'r^', markersize=2, alpha=0.6)

        k = n_cap / n_gt if n_gt > 0 else np.nan
        ax.set_title(f'{subj}/{mode}  k={k:.2f}  cap={n_cap} gt={n_gt}',
                     fontsize=7, loc='left')
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    axes[row, 0].set_ylabel(subj, fontsize=9)

for col, mode in enumerate(SHOW_MODES):
    axes[-1, col].set_xlabel('Time (s)', fontsize=7)

fig.suptitle('Signal + peaks: updated GT Thorax (black/v) vs z(Cvl)-z(Cvr) (blue/^)',
             fontsize=11, y=1.005)
fig.tight_layout()
out4 = OUT_DIR / 'val_peaks_per_subject.png'
fig.savefig(out4, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot 4 -> {out4}")


# ── Plot 5: phase-by-phase rate error summary ───────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
abs_err_data = [
    results.loc[results['mode'] == m, 'error_scaled_bpm'].abs().dropna().values
    for m in MODE_ORDER
]
bp = ax.boxplot(abs_err_data, tick_labels=MODE_ORDER, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.55)
ax.set_title('Phase-by-phase |scaled rate error| (br/min)')
ax.set_ylabel('|error_scaled_bpm|')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, axis='y', alpha=0.3)

ax = axes[1]
bias_by_mode = results.groupby('mode', observed=True)['error_scaled_bpm'].mean()
ax.bar(bias_by_mode.index.astype(str), bias_by_mode.values,
       color='coral', alpha=0.75)
ax.axhline(0.0, color='black', lw=1.0, alpha=0.6)
ax.set_title('Phase-by-phase scaled rate bias (br/min)')
ax.set_ylabel('mean error_scaled_bpm')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, axis='y', alpha=0.3)

fig.tight_layout()
out5 = OUT_DIR / 'val_phase_rate_error_summary.png'
fig.savefig(out5, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot 5 -> {out5}")


# ── Summary stats ────────────────────────────────────────────────────────────

print("\n=== Summary ===")
valid = results.dropna(subset=['k'])
# Exclude valsavaHigh as outlier (very short, GT often fails)
stable = valid[~valid['mode'].isin(['valsavaHigh'])]
print(f"All modes:        k median={valid['k'].median():.3f}  "
      f"mean={valid['k'].mean():.3f}  std={valid['k'].std():.3f}  "
      f"range=[{valid['k'].min():.2f}, {valid['k'].max():.2f}]")
print(f"Excl valsavaHigh: k median={stable['k'].median():.3f}  "
      f"mean={stable['k'].mean():.3f}  std={stable['k'].std():.3f}  "
      f"range=[{stable['k'].min():.2f}, {stable['k'].max():.2f}]")

print("\nPer-mode median k (across subjects):")
for mode in MODE_ORDER:
    mk = valid.loc[valid['mode'] == mode, 'k']
    if len(mk):
        print(f"  {mode:>15s}: median={mk.median():.3f}  "
              f"IQR=[{mk.quantile(0.25):.2f}, {mk.quantile(0.75):.2f}]  n={len(mk)}")

print("\nPer-subject median k (across modes, excl valsavaHigh):")
for subj in subjects:
    sk = stable.loc[stable['subject'] == subj, 'k']
    if len(sk):
        print(f"  {subj}: median={sk.median():.3f}  "
              f"std={sk.std():.3f}  range=[{sk.min():.2f}, {sk.max():.2f}]")

print("\nDone.")
