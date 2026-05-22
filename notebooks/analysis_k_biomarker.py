"""
Phase 1+2: Compute k(t) time series and characterize its temporal structure.

k(t) = raw_CAP_rate(t) / GT_rate(t) at each sliding window.
  - Resp:    k_resp(t)    = rate_peaks_loose(t) / GT_flow_rate(t)
  - Cardiac: k_cardiac(t) = rate_hilbert(t)     / GT_ecg_rate(t)

Phase 1 outputs per-session k(t) arrays + summary CSV.
Phase 2 plots: per-session k(t) overlaid on hypnogram, distributions,
autocorrelation, and cross-session summary.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import SESSION_META, load_session
from sleep_monitor.loader import load_sleep_profile
from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI, STAGE_LABELS,
)
from sleep_monitor.preprocessing import preprocess_full
from sleep_monitor.rates import rate_hilbert, rate_peaks
from sleep_monitor.ground_truth import gt_sliding_rates

PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'k_biomarker'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

WIN_SEC = 60.0
STEP_SEC = 10.0
K_LO, K_HI = 0.5, 4.0

RESP_PROM_FACTOR = 0.05
RESP_MIN_DIST_S = 0.4


def rate_peaks_loose(x, fs=FS):
    """Loose peak detector matching rate_peaks_scaled_resp internals, returns Hz."""
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
    duration = len(x) / fs
    return len(pks) / duration


def sliding_k(cap_sig, gt_times_s, gt_rates_hz, rate_fn, fs=FS):
    """Compute k(t) = rate_fn(window) / GT_rate(window_centre)."""
    win_n = int(round(WIN_SEC * fs))
    step_n = max(1, int(round(STEP_SEC * fs)))

    t_list, k_list, raw_list, gt_list = [], [], [], []

    for start in range(0, len(cap_sig) - win_n + 1, step_n):
        seg = cap_sig[start:start + win_n]
        t_centre_s = (start + win_n / 2.0) / fs

        raw_rate = rate_fn(seg)
        if not np.isfinite(raw_rate) or raw_rate <= 0:
            t_list.append(t_centre_s)
            k_list.append(np.nan)
            raw_list.append(np.nan)
            gt_list.append(np.nan)
            continue

        gt_rate = np.interp(t_centre_s, gt_times_s, gt_rates_hz)
        if not np.isfinite(gt_rate) or gt_rate <= 0:
            t_list.append(t_centre_s)
            k_list.append(np.nan)
            raw_list.append(raw_rate)
            gt_list.append(np.nan)
            continue

        k_val = raw_rate / gt_rate
        if not (K_LO <= k_val <= K_HI):
            k_val = np.nan

        t_list.append(t_centre_s)
        k_list.append(k_val)
        raw_list.append(raw_rate)
        gt_list.append(gt_rate)

    return (np.array(t_list), np.array(k_list),
            np.array(raw_list), np.array(gt_list))


def autocorr(x, max_lag):
    """Normalized autocorrelation of x (NaN-aware) up to max_lag samples."""
    x = x.copy()
    valid = np.isfinite(x)
    x[~valid] = 0.0
    n = len(x)
    mean = np.nanmean(x[valid]) if valid.any() else 0
    x_centered = x - mean
    x_centered[~valid] = 0.0
    var = np.nanmean(x_centered[valid] ** 2) if valid.any() else 1.0
    if var == 0:
        return np.zeros(max_lag)
    lags = np.arange(1, max_lag + 1)
    ac = np.array([
        np.sum(x_centered[:n - lag] * x_centered[lag:]) / (valid[:n - lag] & valid[lag:]).sum()
        for lag in lags
    ])
    return ac / var


# ── Phase 1: Compute k(t) for all sessions ──────────────────────────────────

print('=' * 60)
print('Phase 1: Computing k(t) time series for all 12 sessions')
print('=' * 60)

all_rows = []
session_data = {}

for meta in SESSION_META:
    label = meta['label']
    print(f'\n--- {label} ({meta["subject"]}) ---')

    if not meta['csv'].exists():
        print('  SKIP: csv not found')
        continue

    try:
        session = load_session(meta['idx'])
    except Exception as e:
        print(f'  SKIP load failed: {e}')
        continue

    fs = session.fs
    full_sigs, gt_sigs = preprocess_full(session, acc_removal=True)
    sig_resp = full_sigs['CLE-CRE']['resp']
    sig_card = full_sigs['CLE-CRE']['card']

    gt = gt_sliding_rates(session, win_sec=WIN_SEC, step_sec=STEP_SEC)
    gt_t_s = gt['t_hr'] * 3600.0
    gt_resp_hz = gt['resp_hz']
    gt_card_hz = gt['card_hz']

    t_resp, k_resp, raw_resp, gtv_resp = sliding_k(
        sig_resp, gt_t_s, gt_resp_hz,
        lambda seg: rate_peaks_loose(seg, fs), fs,
    )
    t_card, k_card, raw_card, gtv_card = sliding_k(
        sig_card, gt_t_s, gt_card_hz,
        lambda seg: rate_hilbert(seg, CARD_LO, CARD_HI, fs), fs,
    )

    sp = load_sleep_profile(session)

    valid_resp = np.isfinite(k_resp)
    valid_card = np.isfinite(k_card)

    row = dict(
        label=label, subject=meta['subject'],
        dur_hr=session.duration_hr,
        k_resp_median=np.nanmedian(k_resp) if valid_resp.any() else np.nan,
        k_resp_mean=np.nanmean(k_resp) if valid_resp.any() else np.nan,
        k_resp_std=np.nanstd(k_resp) if valid_resp.any() else np.nan,
        k_resp_iqr=np.subtract(*np.nanpercentile(k_resp, [75, 25])) if valid_resp.any() else np.nan,
        k_resp_cov=valid_resp.mean(),
        k_card_median=np.nanmedian(k_card) if valid_card.any() else np.nan,
        k_card_mean=np.nanmean(k_card) if valid_card.any() else np.nan,
        k_card_std=np.nanstd(k_card) if valid_card.any() else np.nan,
        k_card_iqr=np.subtract(*np.nanpercentile(k_card, [75, 25])) if valid_card.any() else np.nan,
        k_card_cov=valid_card.mean(),
    )
    all_rows.append(row)

    session_data[label] = dict(
        t_resp_hr=t_resp / 3600, k_resp=k_resp,
        t_card_hr=t_card / 3600, k_card=k_card,
        raw_resp=raw_resp, gtv_resp=gtv_resp,
        raw_card=raw_card, gtv_card=gtv_card,
        sleep_profile=sp, meta=meta,
    )

    print(f'  resp k: median={row["k_resp_median"]:.3f}  std={row["k_resp_std"]:.3f}  '
          f'IQR={row["k_resp_iqr"]:.3f}  cov={row["k_resp_cov"]:.1%}')
    print(f'  card k: median={row["k_card_median"]:.3f}  std={row["k_card_std"]:.3f}  '
          f'IQR={row["k_card_iqr"]:.3f}  cov={row["k_card_cov"]:.1%}')

df = pd.DataFrame(all_rows)
csv_path = ROOT / 'artifacts' / 'k_biomarker_summary.csv'
df.to_csv(csv_path, index=False, float_format='%.4f')
print(f'\nSaved summary -> {csv_path}')
print(df.to_string(index=False))


# ── Phase 2: Characterize k(t) ──────────────────────────────────────────────

print('\n' + '=' * 60)
print('Phase 2: Characterizing k(t) temporal structure')
print('=' * 60)

STAGE_COLORS = {
    'Wake': '#e74c3c', 'REM': '#3498db', 'N1': '#f39c12',
    'N2': '#2ecc71', 'N3': '#9b59b6', '?': '#bdc3c7',
}
STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

# ── Plot 1: Per-session k(t) overlaid on hypnogram (grid) ───────────────────

n_sessions = len(session_data)
fig, axes = plt.subplots(n_sessions, 2, figsize=(18, 3 * n_sessions),
                         sharex=False, squeeze=False)
fig.suptitle('k(t) time series — all sessions\n'
             'Left: respiratory (peaks_loose / GT_flow)   '
             'Right: cardiac (hilbert / GT_ecg)', fontsize=13, y=1.01)

for i, (label, sd) in enumerate(session_data.items()):
    sp = sd['sleep_profile']

    for j, (band, t_key, k_key, ylbl, ylim) in enumerate([
        ('resp', 't_resp_hr', 'k_resp', 'k_resp', (0.5, 3.0)),
        ('card', 't_card_hr', 'k_card', 'k_card', (0.8, 3.5)),
    ]):
        ax = axes[i, j]
        t_hr = sd[t_key]
        k_vals = sd[k_key]

        if sp is not None:
            t_ep = sp['t_ep_hr']
            labels_sp = sp['labels']
            epoch_dur_hr = 30.0 / 3600
            for ei in range(len(t_ep)):
                stage = labels_sp[ei]
                color = STAGE_COLORS.get(stage, '#bdc3c7')
                ax.axvspan(t_ep[ei], t_ep[ei] + epoch_dur_hr,
                           alpha=0.15, color=color, lw=0)

        ax.plot(t_hr, k_vals, lw=0.5, alpha=0.6, color='gray')
        win_median = 30
        if np.isfinite(k_vals).sum() > win_median:
            k_smooth = pd.Series(k_vals).rolling(win_median, center=True,
                                                  min_periods=5).median()
            ax.plot(t_hr, k_smooth, lw=1.5, color='black', label='30-win median')

        med = np.nanmedian(k_vals)
        ax.axhline(med, ls='--', color='red', lw=0.8, alpha=0.7,
                    label=f'median={med:.2f}')
        ax.set_ylim(ylim)
        ax.set_ylabel(ylbl)
        if i == 0:
            ax.set_title('Respiratory' if j == 0 else 'Cardiac')
        if i == n_sessions - 1:
            ax.set_xlabel('Time (hr)')
        ax.text(0.01, 0.95, label, transform=ax.transAxes,
                fontsize=9, va='top', fontweight='bold')
        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc='upper right')

fig.tight_layout()
fig.savefig(PLOT_DIR / 'all_sessions_k_timeseries.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
print(f'Saved -> all_sessions_k_timeseries.png')


# ── Plot 2: k distributions per session (violin / box) ──────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for j, (band, k_key, title) in enumerate([
    ('resp', 'k_resp', 'k_resp (peaks_loose / GT_flow)'),
    ('card', 'k_card', 'k_cardiac (hilbert / GT_ecg)'),
]):
    ax = axes[j]
    labels_list = []
    data_list = []
    for label, sd in session_data.items():
        k_vals = sd[k_key]
        valid = k_vals[np.isfinite(k_vals)]
        if len(valid) > 0:
            labels_list.append(label)
            data_list.append(valid)

    bp = ax.boxplot(data_list, labels=labels_list, showfliers=False,
                    patch_artist=True, medianprops=dict(color='red', lw=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db' if j == 1 else '#2ecc71')
        patch.set_alpha(0.4)
    ax.set_title(title)
    ax.set_ylabel('k')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('k distribution per session', fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'all_sessions_k_distributions.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
print(f'Saved -> all_sessions_k_distributions.png')


# ── Plot 3: Autocorrelation of k(t) ─────────────────────────────────────────

max_lag_min = 30
max_lag_samples = int(max_lag_min * 60 / STEP_SEC)
lag_minutes = np.arange(1, max_lag_samples + 1) * STEP_SEC / 60

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for j, (band, k_key, title) in enumerate([
    ('resp', 'k_resp', 'Autocorrelation of k_resp(t)'),
    ('card', 'k_card', 'Autocorrelation of k_cardiac(t)'),
]):
    ax = axes[j]
    ac_all = []
    for label, sd in session_data.items():
        k_vals = sd[k_key]
        if np.isfinite(k_vals).sum() < max_lag_samples * 2:
            continue
        ac = autocorr(k_vals, max_lag_samples)
        ax.plot(lag_minutes, ac, lw=0.7, alpha=0.4, label=label)
        ac_all.append(ac)

    if ac_all:
        ac_mean = np.nanmean(ac_all, axis=0)
        ax.plot(lag_minutes, ac_mean, lw=2.5, color='black', label='mean')

    ax.axhline(0, ls='-', color='gray', lw=0.5)
    ax.set_title(title)
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('Autocorrelation')
    ax.set_xlim(0, max_lag_min)
    ax.legend(fontsize=6, ncol=2, loc='upper right')
    ax.grid(alpha=0.3)

fig.suptitle('k(t) autocorrelation — slow drift = physiological signal', fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'all_sessions_k_autocorrelation.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
print(f'Saved -> all_sessions_k_autocorrelation.png')


# ── Plot 4: Per-session detail panels (hypnogram + k + raw vs GT rates) ─────

for label, sd in session_data.items():
    sp = sd['sleep_profile']
    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex='col')
    fig.suptitle(f'{label} — k(t) biomarker detail', fontsize=13)

    for j, (band, t_key, k_key, raw_key, gt_key, ylim_k, rate_unit) in enumerate([
        ('resp', 't_resp_hr', 'k_resp', 'raw_resp', 'gtv_resp',
         (0.5, 3.0), 'Hz'),
        ('card', 't_card_hr', 'k_card', 'raw_card', 'gtv_card',
         (0.8, 3.5), 'Hz'),
    ]):
        t_hr = sd[t_key]
        k_vals = sd[k_key]
        raw = sd[raw_key]
        gtv = sd[gt_key]

        # Row 0: hypnogram
        ax0 = axes[0, j]
        if sp is not None:
            t_ep = sp['t_ep_hr']
            codes = sp['codes']
            ax0.step(t_ep, codes, where='post', lw=1, color='black')
            ax0.set_yticks([0, 1, 2, 3, 4])
            ax0.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'])
            ax0.invert_yaxis()
        ax0.set_title(f'{"Respiratory" if j == 0 else "Cardiac"}')
        ax0.set_ylabel('Stage')

        # Row 1: k(t)
        ax1 = axes[1, j]
        ax1.plot(t_hr, k_vals, lw=0.4, alpha=0.5, color='gray')
        if np.isfinite(k_vals).sum() > 30:
            k_smooth = pd.Series(k_vals).rolling(30, center=True,
                                                  min_periods=5).median()
            ax1.plot(t_hr, k_smooth, lw=1.5, color='black')
        ax1.axhline(np.nanmedian(k_vals), ls='--', color='red', lw=0.8)
        ax1.set_ylabel('k(t)')
        ax1.set_ylim(ylim_k)
        ax1.grid(alpha=0.3)

        # Row 2: raw CAP rate vs GT rate
        ax2 = axes[2, j]
        ax2.plot(t_hr, np.array(raw) * 60, lw=0.5, alpha=0.5,
                 color='steelblue', label='CAP raw')
        ax2.plot(t_hr, np.array(gtv) * 60, lw=0.5, alpha=0.5,
                 color='green', label='GT')
        ax2.set_ylabel('Rate (per min)')
        ax2.set_xlabel('Time (hr)')
        ax2.legend(fontsize=7, loc='upper right')
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'{label}_k_detail.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

print(f'Saved -> per-session detail panels')


# ── Summary statistics table ─────────────────────────────────────────────────

print('\n' + '=' * 60)
print('Phase 2 Summary: k(t) temporal characteristics')
print('=' * 60)

for band, k_key in [('resp', 'k_resp'), ('card', 'k_card')]:
    print(f'\n--- {band.upper()} ---')
    all_ac_halflife = []
    for label, sd in session_data.items():
        k_vals = sd[k_key]
        n_valid = np.isfinite(k_vals).sum()
        n_total = len(k_vals)
        if n_valid < 100:
            continue
        ac = autocorr(k_vals, min(max_lag_samples, n_valid // 2))
        halflife_idx = np.where(ac < 0.5)[0]
        halflife_min = (halflife_idx[0] * STEP_SEC / 60) if len(halflife_idx) > 0 else float('inf')
        all_ac_halflife.append(halflife_min)
        print(f'  {label}: median={np.nanmedian(k_vals):.3f}  '
              f'std={np.nanstd(k_vals):.3f}  '
              f'AC halflife={halflife_min:.1f} min  '
              f'valid={n_valid}/{n_total} ({n_valid/n_total:.0%})')

    if all_ac_halflife:
        print(f'  Across sessions: AC halflife median={np.median(all_ac_halflife):.1f} min  '
              f'range=[{min(all_ac_halflife):.1f}, {max(all_ac_halflife):.1f}] min')

print('\nDone. All plots saved to notebooks/plots/k_biomarker/')
