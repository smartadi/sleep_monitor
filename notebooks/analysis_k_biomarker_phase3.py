"""
Phase 3: Correlate k(t) with PSG biomarkers.

Builds on Phase 1+2 (analysis_k_biomarker.py).
For each session, compute k(t) alongside:
  - Sleep stage labels (from PSG sleep profile)
  - EEG delta/theta/alpha band power (from PSG EEG)
  - HRV metrics (SDNN, RMSSD from ECG R-R intervals)
  - Respiratory regularity (breath-to-breath interval std from Flow)
  - Movement (accelerometer RMS)

Then test for correlations across all sessions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import SESSION_META, load_session
from sleep_monitor.loader import load_sleep_profile
from sleep_monitor.config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI, STAGE_LABELS
from sleep_monitor.preprocessing import preprocess_full
from sleep_monitor.rates import rate_hilbert
from sleep_monitor.ground_truth import gt_sliding_rates, gt_heart_rate, gt_resp_rate
from scipy.signal import find_peaks

PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'k_biomarker'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

WIN_SEC = 60.0
STEP_SEC = 10.0
K_LO, K_HI = 0.5, 4.0
RESP_PROM_FACTOR = 0.05
RESP_MIN_DIST_S = 0.4

STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = {
    'Wake': '#e74c3c', 'N1': '#f39c12', 'N2': '#2ecc71',
    'N3': '#9b59b6', 'REM': '#3498db',
}
CODE_TO_LABEL = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}


def rate_peaks_loose(x, fs=FS):
    x = x.astype(np.float64)
    if len(x) < 16:
        return np.nan
    min_dist = max(1, int(round(RESP_MIN_DIST_S * fs)))
    smooth_win = max(3, min_dist // 4)
    x_sm = np.convolve(x, np.ones(smooth_win) / smooth_win, mode='same')
    pks, _ = find_peaks(x_sm, distance=min_dist,
                        prominence=RESP_PROM_FACTOR * np.std(x_sm))
    if len(pks) < 2:
        return np.nan
    return len(pks) / (len(x) / fs)


def sliding_k(cap_sig, gt_times_s, gt_rates_hz, rate_fn, fs=FS):
    win_n = int(round(WIN_SEC * fs))
    step_n = max(1, int(round(STEP_SEC * fs)))
    t_list, k_list = [], []
    for start in range(0, len(cap_sig) - win_n + 1, step_n):
        seg = cap_sig[start:start + win_n]
        t_centre_s = (start + win_n / 2.0) / fs
        raw_rate = rate_fn(seg)
        if not np.isfinite(raw_rate) or raw_rate <= 0:
            t_list.append(t_centre_s); k_list.append(np.nan); continue
        gt_rate = np.interp(t_centre_s, gt_times_s, gt_rates_hz)
        if not np.isfinite(gt_rate) or gt_rate <= 0:
            t_list.append(t_centre_s); k_list.append(np.nan); continue
        k_val = raw_rate / gt_rate
        t_list.append(t_centre_s)
        k_list.append(k_val if K_LO <= k_val <= K_HI else np.nan)
    return np.array(t_list), np.array(k_list)


def eeg_band_power(eeg_raw, fs, win_sec=WIN_SEC, step_sec=STEP_SEC):
    """Sliding-window EEG band power: delta (0.5-4), theta (4-8), alpha (8-13)."""
    win_n = int(round(win_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13)}
    t_list = []
    power = {b: [] for b in bands}

    for start in range(0, len(eeg_raw) - win_n + 1, step_n):
        seg = eeg_raw[start:start + win_n].astype(np.float64)
        t_list.append((start + win_n / 2.0) / fs)
        nperseg = min(len(seg), int(fs * 4))
        freqs, psd = welch(seg, fs=fs, nperseg=nperseg)
        for bname, (flo, fhi) in bands.items():
            mask = (freqs >= flo) & (freqs <= fhi)
            power[bname].append(np.trapz(psd[mask], freqs[mask]) if mask.any() else np.nan)

    return np.array(t_list), {b: np.array(v) for b, v in power.items()}


def sliding_hrv(gt_card, win_sec=WIN_SEC, step_sec=STEP_SEC):
    """Sliding-window HRV (SDNN, RMSSD) from GT R-peak times."""
    peak_times = gt_card.peak_times_s
    intervals = gt_card.intervals_s
    t_list, sdnn_list, rmssd_list = [], [], []
    t_max = peak_times[-1] if len(peak_times) > 0 else 0

    for t_centre in np.arange(win_sec / 2, t_max - win_sec / 2, step_sec):
        t_lo, t_hi = t_centre - win_sec / 2, t_centre + win_sec / 2
        mask = (peak_times[:-1] >= t_lo) & (peak_times[:-1] < t_hi)
        rr = intervals[mask]
        t_list.append(t_centre)
        if len(rr) < 5:
            sdnn_list.append(np.nan); rmssd_list.append(np.nan)
            continue
        rr_ms = rr * 1000
        sdnn_list.append(np.std(rr_ms, ddof=1))
        rmssd_list.append(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))

    return np.array(t_list), np.array(sdnn_list), np.array(rmssd_list)


def sliding_resp_regularity(gt_resp, win_sec=WIN_SEC, step_sec=STEP_SEC):
    """Sliding-window breath-to-breath interval variability from GT Flow."""
    peak_times = gt_resp.peak_times_s
    intervals = gt_resp.intervals_s
    t_list, cv_list = [], []
    t_max = peak_times[-1] if len(peak_times) > 0 else 0

    for t_centre in np.arange(win_sec / 2, t_max - win_sec / 2, step_sec):
        t_lo, t_hi = t_centre - win_sec / 2, t_centre + win_sec / 2
        mask = (peak_times[:-1] >= t_lo) & (peak_times[:-1] < t_hi)
        ibi = intervals[mask]
        t_list.append(t_centre)
        if len(ibi) < 3 or np.mean(ibi) == 0:
            cv_list.append(np.nan); continue
        cv_list.append(np.std(ibi) / np.mean(ibi))

    return np.array(t_list), np.array(cv_list)


def sliding_acc_rms(acc_mag, fs, win_sec=WIN_SEC, step_sec=STEP_SEC):
    """Sliding-window accelerometer RMS."""
    win_n = int(round(win_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))
    t_list, rms_list = [], []
    for start in range(0, len(acc_mag) - win_n + 1, step_n):
        seg = acc_mag[start:start + win_n].astype(np.float64)
        t_list.append((start + win_n / 2.0) / fs)
        rms_list.append(np.sqrt(np.mean(seg ** 2)))
    return np.array(t_list), np.array(rms_list)


def assign_stage(t_s, sleep_profile):
    """Map each time point to a sleep stage label."""
    if sleep_profile is None:
        return np.full(len(t_s), '', dtype=object)
    t_ep_s = sleep_profile['t_ep_hr'] * 3600
    codes = sleep_profile['codes']
    labels = np.full(len(t_s), '', dtype=object)
    for i, t in enumerate(t_s):
        idx = np.searchsorted(t_ep_s, t, side='right') - 1
        if 0 <= idx < len(codes):
            code = codes[idx]
            labels[i] = CODE_TO_LABEL.get(code, '')
    return labels


# ── Main: compute all biomarkers per session ─────────────────────────────────

print('=' * 60)
print('Phase 3: Correlating k(t) with PSG biomarkers')
print('=' * 60)

all_k_resp, all_k_card = [], []
all_stages_resp, all_stages_card = [], []
all_delta_resp, all_delta_card = [], []
all_sdnn, all_rmssd = [], []
all_resp_cv = []
all_acc_resp, all_acc_card = [], []
all_labels = []

for meta in SESSION_META:
    label = meta['label']
    print(f'\n--- {label} ({meta["subject"]}) ---')

    if not meta['csv'].exists():
        print('  SKIP'); continue
    try:
        session = load_session(meta['idx'])
    except Exception as e:
        print(f'  SKIP: {e}'); continue

    fs = session.fs
    full_sigs, gt_sigs = preprocess_full(session, acc_removal=True)
    sig_resp = full_sigs['CLE-CRE']['resp']
    sig_card = full_sigs['CLE-CRE']['card']

    gt = gt_sliding_rates(session, win_sec=WIN_SEC, step_sec=STEP_SEC)
    gt_t_s = gt['t_hr'] * 3600.0

    # k(t)
    t_resp, k_resp = sliding_k(
        sig_resp, gt_t_s, gt['resp_hz'],
        lambda seg: rate_peaks_loose(seg, fs), fs)
    t_card, k_card = sliding_k(
        sig_card, gt_t_s, gt['card_hz'],
        lambda seg: rate_hilbert(seg, CARD_LO, CARD_HI, fs), fs)

    # Sleep stages
    sp = load_sleep_profile(session)
    stages_resp = assign_stage(t_resp, sp)
    stages_card = assign_stage(t_card, sp)

    # EEG band power
    eeg_raw = session.psg.get('EEG')
    if eeg_raw is not None:
        t_eeg, eeg_power = eeg_band_power(eeg_raw, fs)
        delta_resp = np.interp(t_resp, t_eeg, eeg_power['delta'])
        delta_card = np.interp(t_card, t_eeg, eeg_power['delta'])
    else:
        delta_resp = np.full_like(t_resp, np.nan)
        delta_card = np.full_like(t_card, np.nan)

    # HRV
    gt_card_result = gt['card_gt']
    t_hrv, sdnn, rmssd = sliding_hrv(gt_card_result)
    sdnn_interp = np.interp(t_card, t_hrv, sdnn) if len(t_hrv) > 0 else np.full_like(t_card, np.nan)
    rmssd_interp = np.interp(t_card, t_hrv, rmssd) if len(t_hrv) > 0 else np.full_like(t_card, np.nan)

    # Resp regularity
    gt_resp_result = gt['resp_gt']
    t_rcv, resp_cv = sliding_resp_regularity(gt_resp_result)
    resp_cv_interp = np.interp(t_resp, t_rcv, resp_cv) if len(t_rcv) > 0 else np.full_like(t_resp, np.nan)

    # Movement
    acc_mag = session.cap['acc_mag']
    t_acc, acc_rms = sliding_acc_rms(acc_mag, fs)
    acc_resp = np.interp(t_resp, t_acc, acc_rms)
    acc_card = np.interp(t_card, t_acc, acc_rms)

    # Accumulate
    all_k_resp.append(k_resp)
    all_k_card.append(k_card)
    all_stages_resp.append(stages_resp)
    all_stages_card.append(stages_card)
    all_delta_resp.append(delta_resp)
    all_delta_card.append(delta_card)
    all_sdnn.append(sdnn_interp)
    all_rmssd.append(rmssd_interp)
    all_resp_cv.append(resp_cv_interp)
    all_acc_resp.append(acc_resp)
    all_acc_card.append(acc_card)
    all_labels.append(label)

    n_r = np.isfinite(k_resp).sum()
    n_c = np.isfinite(k_card).sum()
    print(f'  k_resp valid={n_r}, k_card valid={n_c}, stages loaded={sp is not None}')

# Concatenate all sessions
k_resp_all = np.concatenate(all_k_resp)
k_card_all = np.concatenate(all_k_card)
stages_resp_all = np.concatenate(all_stages_resp)
stages_card_all = np.concatenate(all_stages_card)
delta_resp_all = np.concatenate(all_delta_resp)
delta_card_all = np.concatenate(all_delta_card)
sdnn_all = np.concatenate(all_sdnn)
rmssd_all = np.concatenate(all_rmssd)
resp_cv_all = np.concatenate(all_resp_cv)
acc_resp_all = np.concatenate(all_acc_resp)
acc_card_all = np.concatenate(all_acc_card)


# ── Plot 1: k by sleep stage (box plots + Kruskal-Wallis) ───────────────────

print('\n--- Sleep stage analysis ---')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for j, (k_all, stages_all, title) in enumerate([
    (k_resp_all, stages_resp_all, 'k_resp by sleep stage'),
    (k_card_all, stages_card_all, 'k_cardiac by sleep stage'),
]):
    ax = axes[j]
    data_by_stage = {}
    for stage in STAGE_ORDER:
        mask = (stages_all == stage) & np.isfinite(k_all)
        vals = k_all[mask]
        if len(vals) > 0:
            data_by_stage[stage] = vals

    if len(data_by_stage) >= 2:
        stage_names = [s for s in STAGE_ORDER if s in data_by_stage]
        stage_data = [data_by_stage[s] for s in stage_names]

        bp = ax.boxplot(stage_data, tick_labels=stage_names, showfliers=False,
                        patch_artist=True, medianprops=dict(color='red', lw=2))
        for patch, sname in zip(bp['boxes'], stage_names):
            patch.set_facecolor(STAGE_COLORS.get(sname, '#bdc3c7'))
            patch.set_alpha(0.5)

        medians = [np.median(d) for d in stage_data]
        for xi, m in enumerate(medians):
            ax.text(xi + 1, m + 0.01, f'{m:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        stat, pval = stats.kruskal(*stage_data)
        ax.set_title(f'{title}\nKruskal-Wallis H={stat:.1f}, p={pval:.2e}')
    else:
        ax.set_title(f'{title}\n(insufficient stage data)')

    ax.set_ylabel('k')
    ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(PLOT_DIR / 'phase3_k_by_stage.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved -> phase3_k_by_stage.png')

for j, (k_all, stages_all, band) in enumerate([
    (k_resp_all, stages_resp_all, 'resp'),
    (k_card_all, stages_card_all, 'cardiac'),
]):
    print(f'\n  {band.upper()} k by stage:')
    for stage in STAGE_ORDER:
        mask = (stages_all == stage) & np.isfinite(k_all)
        vals = k_all[mask]
        if len(vals) > 0:
            print(f'    {stage:5s}: median={np.median(vals):.3f}  '
                  f'mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  n={len(vals)}')


# ── Plot 2: Per-session k by stage (to check consistency) ───────────────────

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

for j, (k_list, stages_list, title) in enumerate([
    (all_k_resp, all_stages_resp, 'k_resp by stage -- per session'),
    (all_k_card, all_stages_card, 'k_cardiac by stage -- per session'),
]):
    ax = axes[j]
    x_positions = []
    x_labels = []
    box_data = []
    box_colors = []
    pos = 0

    for si, label in enumerate(all_labels):
        for stage in STAGE_ORDER:
            mask = (stages_list[si] == stage) & np.isfinite(k_list[si])
            vals = k_list[si][mask]
            if len(vals) < 10:
                continue
            box_data.append(vals)
            box_colors.append(STAGE_COLORS[stage])
            x_positions.append(pos)
            x_labels.append(f'{label}\n{stage}')
            pos += 1
        pos += 0.5

    if box_data:
        bp = ax.boxplot(box_data, positions=x_positions, widths=0.7,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', lw=1))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=5, rotation=90)
    ax.set_ylabel('k')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=STAGE_COLORS[s], alpha=0.5)
               for s in STAGE_ORDER]
    ax.legend(handles, STAGE_ORDER, fontsize=8, loc='upper right', ncol=5)

fig.tight_layout()
fig.savefig(PLOT_DIR / 'phase3_k_by_stage_per_session.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
print('Saved -> phase3_k_by_stage_per_session.png')


# ── Plot 3: k vs EEG delta power (scatter + correlation) ────────────────────

print('\n--- EEG delta power correlation ---')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for j, (k_all, delta_all, title) in enumerate([
    (k_resp_all, delta_resp_all, 'k_resp vs EEG delta power'),
    (k_card_all, delta_card_all, 'k_cardiac vs EEG delta power'),
]):
    ax = axes[j]
    valid = np.isfinite(k_all) & np.isfinite(delta_all) & (delta_all > 0)
    if valid.sum() > 10:
        log_delta = np.log10(delta_all[valid])
        k_v = k_all[valid]

        ax.scatter(log_delta, k_v, s=1, alpha=0.05, color='steelblue')

        r, p = stats.spearmanr(log_delta, k_v)
        ax.set_title(f'{title}\nSpearman r={r:.3f}, p={p:.2e}')

        # Binned means for trend
        bins = np.linspace(np.percentile(log_delta, 1),
                           np.percentile(log_delta, 99), 20)
        bin_idx = np.digitize(log_delta, bins)
        bin_means_x, bin_means_y = [], []
        for bi in range(1, len(bins)):
            m = bin_idx == bi
            if m.sum() > 20:
                bin_means_x.append(bins[bi - 1])
                bin_means_y.append(np.median(k_v[m]))
        ax.plot(bin_means_x, bin_means_y, 'ro-', lw=2, markersize=5,
                label='binned median')
        ax.legend(fontsize=8)
    else:
        ax.set_title(f'{title}\n(insufficient data)')

    ax.set_xlabel('log10(EEG delta power)')
    ax.set_ylabel('k')
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(PLOT_DIR / 'phase3_k_vs_eeg_delta.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved -> phase3_k_vs_eeg_delta.png')


# ── Plot 4: k_cardiac vs HRV (SDNN, RMSSD) ─────────────────────────────────

print('\n--- HRV correlation ---')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for j, (hrv_all, hrv_name) in enumerate([
    (sdnn_all, 'SDNN (ms)'),
    (rmssd_all, 'RMSSD (ms)'),
]):
    ax = axes[j]
    valid = np.isfinite(k_card_all) & np.isfinite(hrv_all) & (hrv_all > 0)
    if valid.sum() > 10:
        h_v = hrv_all[valid]
        k_v = k_card_all[valid]
        ax.scatter(h_v, k_v, s=1, alpha=0.05, color='steelblue')

        r, p = stats.spearmanr(h_v, k_v)
        ax.set_title(f'k_cardiac vs {hrv_name}\nSpearman r={r:.3f}, p={p:.2e}')

        bins = np.linspace(np.percentile(h_v, 1), np.percentile(h_v, 99), 20)
        bin_idx = np.digitize(h_v, bins)
        bx, by = [], []
        for bi in range(1, len(bins)):
            m = bin_idx == bi
            if m.sum() > 20:
                bx.append((bins[bi - 1] + bins[bi]) / 2)
                by.append(np.median(k_v[m]))
        ax.plot(bx, by, 'ro-', lw=2, markersize=5, label='binned median')
        ax.legend(fontsize=8)
    else:
        ax.set_title(f'k_cardiac vs {hrv_name}\n(insufficient data)')

    ax.set_xlabel(hrv_name)
    ax.set_ylabel('k_cardiac')
    ax.grid(alpha=0.3)

fig.suptitle('Cardiac k vs Heart Rate Variability', fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'phase3_k_vs_hrv.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved -> phase3_k_vs_hrv.png')


# ── Plot 5: k_resp vs respiratory regularity ─────────────────────────────────

print('\n--- Respiratory regularity correlation ---')

fig, ax = plt.subplots(figsize=(7, 6))
valid = np.isfinite(k_resp_all) & np.isfinite(resp_cv_all) & (resp_cv_all > 0)
if valid.sum() > 10:
    cv_v = resp_cv_all[valid]
    k_v = k_resp_all[valid]
    ax.scatter(cv_v, k_v, s=1, alpha=0.05, color='steelblue')
    r, p = stats.spearmanr(cv_v, k_v)
    ax.set_title(f'k_resp vs breath interval CV\nSpearman r={r:.3f}, p={p:.2e}')

    bins = np.linspace(np.percentile(cv_v, 1), np.percentile(cv_v, 99), 20)
    bin_idx = np.digitize(cv_v, bins)
    bx, by = [], []
    for bi in range(1, len(bins)):
        m = bin_idx == bi
        if m.sum() > 20:
            bx.append((bins[bi - 1] + bins[bi]) / 2)
            by.append(np.median(k_v[m]))
    ax.plot(bx, by, 'ro-', lw=2, markersize=5, label='binned median')
    ax.legend(fontsize=8)
else:
    ax.set_title('k_resp vs breath interval CV\n(insufficient data)')

ax.set_xlabel('Breath interval CV (std/mean)')
ax.set_ylabel('k_resp')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'phase3_k_vs_resp_regularity.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
print('Saved -> phase3_k_vs_resp_regularity.png')


# ── Plot 6: k vs accelerometer RMS (movement) ───────────────────────────────

print('\n--- Movement correlation ---')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for j, (k_all, acc_all, title) in enumerate([
    (k_resp_all, acc_resp_all, 'k_resp vs acc RMS'),
    (k_card_all, acc_card_all, 'k_cardiac vs acc RMS'),
]):
    ax = axes[j]
    valid = np.isfinite(k_all) & np.isfinite(acc_all) & (acc_all > 0)
    if valid.sum() > 10:
        a_v = acc_all[valid]
        k_v = k_all[valid]
        ax.scatter(a_v, k_v, s=1, alpha=0.05, color='steelblue')
        r, p = stats.spearmanr(a_v, k_v)
        ax.set_title(f'{title}\nSpearman r={r:.3f}, p={p:.2e}')

        bins = np.linspace(np.percentile(a_v, 1), np.percentile(a_v, 99), 20)
        bin_idx = np.digitize(a_v, bins)
        bx, by = [], []
        for bi in range(1, len(bins)):
            m = bin_idx == bi
            if m.sum() > 20:
                bx.append((bins[bi - 1] + bins[bi]) / 2)
                by.append(np.median(k_v[m]))
        ax.plot(bx, by, 'ro-', lw=2, markersize=5, label='binned median')
        ax.legend(fontsize=8)
    else:
        ax.set_title(f'{title}\n(insufficient data)')
    ax.set_xlabel('Accelerometer RMS')
    ax.set_ylabel('k')
    ax.grid(alpha=0.3)

fig.suptitle('k vs Movement (accelerometer)', fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'phase3_k_vs_movement.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved -> phase3_k_vs_movement.png')


# ── Plot 7: Correlation summary heatmap ──────────────────────────────────────

print('\n--- Correlation summary ---')

corr_results = []

pairs = [
    ('k_resp', 'delta_power', k_resp_all, np.log10(np.clip(delta_resp_all, 1e-20, None))),
    ('k_resp', 'resp_cv', k_resp_all, resp_cv_all),
    ('k_resp', 'acc_rms', k_resp_all, acc_resp_all),
    ('k_card', 'delta_power', k_card_all, np.log10(np.clip(delta_card_all, 1e-20, None))),
    ('k_card', 'sdnn', k_card_all, sdnn_all),
    ('k_card', 'rmssd', k_card_all, rmssd_all),
    ('k_card', 'acc_rms', k_card_all, acc_card_all),
]

for k_name, bio_name, k_arr, bio_arr in pairs:
    valid = np.isfinite(k_arr) & np.isfinite(bio_arr)
    if valid.sum() > 50:
        r, p = stats.spearmanr(k_arr[valid], bio_arr[valid])
    else:
        r, p = np.nan, np.nan
    corr_results.append(dict(k=k_name, biomarker=bio_name, spearman_r=r, p_value=p,
                             n=valid.sum()))
    print(f'  {k_name} vs {bio_name}: r={r:.3f}  p={p:.2e}  n={valid.sum()}')

corr_df = pd.DataFrame(corr_results)

fig, ax = plt.subplots(figsize=(8, 5))
pivot = corr_df.pivot(index='k', columns='biomarker', values='spearman_r')
pivot = pivot.reindex(columns=['delta_power', 'sdnn', 'rmssd', 'resp_cv', 'acc_rms'])
im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for jj in range(len(pivot.columns)):
        val = pivot.values[i, jj]
        if np.isfinite(val):
            ax.text(jj, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if abs(val) > 0.3 else 'black')
plt.colorbar(im, ax=ax, label='Spearman r')
ax.set_title('k(t) vs PSG biomarker correlations (all sessions pooled)')
fig.tight_layout()
fig.savefig(PLOT_DIR / 'phase3_correlation_heatmap.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
print('Saved -> phase3_correlation_heatmap.png')

corr_df.to_csv(ROOT / 'artifacts' / 'k_biomarker_correlations.csv',
               index=False, float_format='%.4f')
print(f'\nSaved correlations -> artifacts/k_biomarker_correlations.csv')
print('\nPhase 3 complete.')
