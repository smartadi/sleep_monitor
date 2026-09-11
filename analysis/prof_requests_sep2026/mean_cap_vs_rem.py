"""
Request A (Prof, 2026-09-08): "Correlate the mean Cap value changes (positive and
negative) with REM. According to my observation, it is coupled with sleep cycles."

Question
--------
Does the SLOW mean (DC baseline) of the capacitive channels swing up and down with
the NREM-REM ultradian cycle, and does it move in a consistent direction around REM?

Approach (descriptive, per-subject — n=6, no headline p-values)
--------
Per 30 s PSG epoch, for each channel (CH, CLE, CRE, CLE-CRE) take the mean sample
value (the DC level), z-scored per session. Two slow traces are drawn: a light
per-epoch trace and a ~15 min rolling-median "sleep-cycle" trend.

Three views:
  1. Per-session overlay: slow mean vs hypnogram, REM periods shaded  (12-panel).
  2. REM-onset-triggered average of the slow mean, +/-40 min, per channel
     (pooled + per subject) — does the baseline systematically rise/fall into REM?
  3. REM-minus-NREM slow mean per session/subject (the +/- direction the prof asked
     about) and the per-session correlation of the slow mean with a smoothed REM-
     occupancy cycle (the ultradian coupling, one number per night).

Outputs
-------
  reports/prof_requests_sep2026/mean_cap_vs_rem_session.csv
  notebooks/plots/prof_requests_sep2026/A_meancap_rem_overlay_grid.png
  notebooks/plots/prof_requests_sep2026/A_meancap_rem_triggered.png
  notebooks/plots/prof_requests_sep2026/A_meancap_rem_direction.png

Usage:
    .venv/Scripts/python.exe analysis/prof_requests_sep2026/mean_cap_vs_rem.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import spectrogram as sp_spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_session, load_sleep_profile, FS
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, CAP_COLORS, CAP_SCALE_TO_FF,
)
from sleep_monitor.sessions import SESSION_META

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'prof_requests_sep2026'
REPORT_DIR = ROOT / 'reports' / 'prof_requests_sep2026'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
CHANNELS = ['CH', 'CLE', 'CRE', 'CLE-CRE']
CYCLE_WIN_EPOCHS = 31       # ~15.5 min rolling median -> "sleep-cycle" slow trend
TRIG_HALF_MIN = 40.0        # +/- window for REM-onset-triggered average (minutes)
TRIG_HALF_EP = int(TRIG_HALF_MIN * 60 / EPOCH_SEC)


# ── Per-session epoch features ────────────────────────────────────────────────

def epoch_means(idx):
    """Per-epoch DC mean of each channel (z-scored per session) + stage codes."""
    meta = SESSION_META[idx]
    s = load_session(idx)
    sp = load_sleep_profile(s)
    if sp is None:
        print(f'  {meta["label"]}: no sleep profile, skipping')
        return None
    s.sleep_profile = sp

    t_hr = s.time_hr.astype(np.float64)
    raw = {
        'CH':  s.cap['CH'].astype(np.float64) * CAP_SCALE_TO_FF,
        'CLE': s.cap['CLE'].astype(np.float64) * CAP_SCALE_TO_FF,
        'CRE': s.cap['CRE'].astype(np.float64) * CAP_SCALE_TO_FF,
    }
    raw['CLE-CRE'] = raw['CLE'] - raw['CRE']

    ep_t = sp['t_ep_hr']
    codes = sp['codes']
    dt_hr = EPOCH_SEC / 3600.0

    rows = []
    for j in range(len(ep_t)):
        t0 = ep_t[j]
        t1 = t0 + dt_hr
        if t0 < 0 or t1 > t_hr[-1]:
            continue
        i0 = np.searchsorted(t_hr, t0)
        i1 = np.searchsorted(t_hr, t1)
        if i1 - i0 < int(0.5 * FS * EPOCH_SEC):
            continue
        row = {'session': meta['label'], 'subject': meta['subject'],
               'epoch': j, 't_hr': t0 + dt_hr / 2.0, 'stage_code': int(codes[j])}
        for ch in CHANNELS:
            row[f'mean_{ch}'] = float(raw[ch][i0:i1].mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # per-session z-score + cycle-scale slow trend
    for ch in CHANNELS:
        x = df[f'mean_{ch}'].to_numpy()
        df[f'z_{ch}'] = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
        df[f'slow_{ch}'] = (pd.Series(df[f'z_{ch}'])
                            .rolling(CYCLE_WIN_EPOCHS, center=True, min_periods=5)
                            .median().to_numpy())
    return df, s


# ── REM-period bookkeeping ────────────────────────────────────────────────────

def rem_onset_epochs(codes):
    """Indices (into the epoch DataFrame order) where a REM period begins."""
    is_rem = (codes == 0).astype(int)
    onsets = np.flatnonzero((np.diff(is_rem) == 1)) + 1
    if is_rem.size and is_rem[0] == 1:
        onsets = np.r_[0, onsets]
    return onsets


def rem_occupancy(codes, win=CYCLE_WIN_EPOCHS):
    """Smoothed REM fraction on the epoch grid — the ultradian REM cycle."""
    r = (codes == 0).astype(float)
    return pd.Series(r).rolling(win, center=True, min_periods=5).mean().to_numpy()


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_overlay_grid(dfs, out):
    """12-panel: cycle-scale slow mean (CLE-CRE, CH) vs hypnogram, REM shaded."""
    fig, axes = plt.subplots(6, 2, figsize=(20, 16), sharex=False)
    axes = axes.ravel()
    for ax, df in zip(axes, dfs):
        codes = df['stage_code'].to_numpy()
        t = df['t_hr'].to_numpy()
        # shade REM
        for j in np.flatnonzero(codes == 0):
            ax.axvspan(t[j] - EPOCH_SEC / 7200.0, t[j] + EPOCH_SEC / 7200.0,
                       color=STAGE_COLORS[0], alpha=0.22, lw=0)
        for ch, lw in [('CLE-CRE', 2.0), ('CH', 1.6)]:
            ax.plot(t, df[f'slow_{ch}'], color=CAP_COLORS[ch], lw=lw, label=ch)
        ax.axhline(0, color='gray', ls=':', lw=0.7)
        ax.set_title(df['session'].iloc[0], fontsize=13, fontweight='bold')
        ax.set_ylabel('slow mean (z)', fontsize=10)
        ax.tick_params(labelsize=9)
    axes[0].legend(fontsize=10, loc='upper right', ncol=2,
                   handles=[plt.Line2D([], [], color=CAP_COLORS['CLE-CRE'], lw=2, label='CLE-CRE'),
                            plt.Line2D([], [], color=CAP_COLORS['CH'], lw=2, label='CH'),
                            mpatches.Patch(color=STAGE_COLORS[0], alpha=0.3, label='REM')])
    for ax in axes[-2:]:
        ax.set_xlabel('Time (hours)', fontsize=11)
    fig.suptitle('A. Slow (sleep-cycle) mean capacitance vs REM — REM periods shaded',
                 fontsize=17, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def fig_triggered(dfs, out):
    """REM-onset-triggered average of the slow mean, per channel: pooled + per subject."""
    lags_min = (np.arange(-TRIG_HALF_EP, TRIG_HALF_EP + 1) * EPOCH_SEC) / 60.0
    stacks = {ch: [] for ch in CHANNELS}      # pooled (per-onset rows)
    per_sub = {}                              # subject -> {ch: [rows]}
    for df in dfs:
        sub = df['subject'].iloc[0]
        per_sub.setdefault(sub, {ch: [] for ch in CHANNELS})
        codes = df['stage_code'].to_numpy()
        n = len(df)
        for on in rem_onset_epochs(codes):
            if on - TRIG_HALF_EP < 0 or on + TRIG_HALF_EP >= n:
                continue
            for ch in CHANNELS:
                # use z (not the pre-smoothed slow) so the average itself is the smoother;
                # re-baseline each window to its pre-onset (-40..-20 min) level
                seg = df[f'z_{ch}'].to_numpy()[on - TRIG_HALF_EP: on + TRIG_HALF_EP + 1]
                base = np.nanmean(seg[:TRIG_HALF_EP // 2])
                seg = seg - base
                stacks[ch].append(seg)
                per_sub[sub][ch].append(seg)

    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(5.2 * len(CHANNELS), 4.6),
                             sharex=True, sharey=True)
    for ax, ch in zip(axes, CHANNELS):
        arr = np.array(stacks[ch])
        # per-subject light lines
        for sub, d in per_sub.items():
            if d[ch]:
                ax.plot(lags_min, np.nanmean(np.array(d[ch]), axis=0),
                        color='0.7', lw=0.9, alpha=0.8)
        if arr.size:
            m = np.nanmean(arr, axis=0)
            se = np.nanstd(arr, axis=0) / np.sqrt(max(1, arr.shape[0]))
            ax.plot(lags_min, m, color=CAP_COLORS[ch], lw=2.6)
            ax.fill_between(lags_min, m - se, m + se, color=CAP_COLORS[ch], alpha=0.25)
            ax.text(0.03, 0.95, f'n={arr.shape[0]} REM onsets', transform=ax.transAxes,
                    fontsize=10, va='top')
        ax.axvline(0, color=STAGE_COLORS[0], lw=1.6, ls='--')
        ax.axhline(0, color='gray', ls=':', lw=0.7)
        ax.set_title(ch, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time from REM onset (min)', fontsize=11)
    axes[0].set_ylabel('slow mean, re-baselined (z)', fontsize=11)
    fig.suptitle('A. REM-onset-triggered slow mean capacitance '
                 '(bold = pooled, grey = per subject)',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def fig_direction(sess_df, out):
    """Per-session REM-minus-NREM slow mean (the +/- swing) for each channel."""
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(5.0 * len(CHANNELS), 4.6),
                             sharey=True)
    order = sess_df.sort_values('session')['session'].tolist()
    for ax, ch in zip(axes, CHANNELS):
        vals = sess_df.set_index('session').loc[order, f'rem_minus_nrem_{ch}']
        colors = ['#9B59B6' if v > 0 else '#E67E22' for v in vals]
        ax.barh(range(len(order)), vals.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title(ch, fontsize=13, fontweight='bold')
        ax.set_xlabel('REM − NREM slow mean (z)', fontsize=10)
        ax.grid(True, axis='x', alpha=0.15)
    axes[0].invert_yaxis()
    fig.suptitle('A. REM vs NREM slow-mean shift per night '
                 '(purple = higher in REM, orange = lower)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 66)
    print('A. Slow mean capacitance vs REM / sleep cycle')
    print('=' * 66)

    dfs = []
    for idx in range(len(SESSION_META)):
        res = epoch_means(idx)
        if res is not None:
            dfs.append(res[0])
    dfs.sort(key=lambda d: d['session'].iloc[0])

    # per-session summary: REM-NREM shift + correlation with the REM ultradian cycle
    rows = []
    for df in dfs:
        codes = df['stage_code'].to_numpy()
        rem = codes == 0
        nrem = np.isin(codes, [1, 2, 3])
        occ = rem_occupancy(codes)
        row = {'session': df['session'].iloc[0], 'subject': df['subject'].iloc[0],
               'n_rem_epochs': int(rem.sum()), 'n_rem_onsets': int(len(rem_onset_epochs(codes)))}
        for ch in CHANNELS:
            z = df[f'z_{ch}'].to_numpy()
            row[f'rem_minus_nrem_{ch}'] = (float(np.nanmean(z[rem])) -
                                           float(np.nanmean(z[nrem]))) if rem.any() and nrem.any() else np.nan
            # descriptive correlation of the slow mean with the REM-occupancy cycle
            sl = df[f'slow_{ch}'].to_numpy()
            ok = np.isfinite(sl) & np.isfinite(occ)
            row[f'cycle_corr_{ch}'] = (float(np.corrcoef(sl[ok], occ[ok])[0, 1])
                                       if ok.sum() > 20 else np.nan)
        rows.append(row)
    sess_df = pd.DataFrame(rows)
    sess_df.to_csv(REPORT_DIR / 'mean_cap_vs_rem_session.csv', index=False)

    print('\nPer-session REM-minus-NREM slow mean (z) and REM-cycle correlation:')
    for ch in CHANNELS:
        rm = sess_df[f'rem_minus_nrem_{ch}']
        cc = sess_df[f'cycle_corr_{ch}']
        pos = int((rm > 0).sum()); neg = int((rm < 0).sum())
        print(f'  {ch:8s}  REM>NREM in {pos}/{len(rm)} nights, REM<NREM in {neg}; '
              f'median REM-NREM={rm.median():+.2f} z; '
              f'median |cycle r|={cc.abs().median():.2f}')

    print('\nWriting figures...')
    fig_overlay_grid(dfs, PLOT_DIR / 'A_meancap_rem_overlay_grid.png')
    fig_triggered(dfs, PLOT_DIR / 'A_meancap_rem_triggered.png')
    fig_direction(sess_df, PLOT_DIR / 'A_meancap_rem_direction.png')
    print('\nDone. -> reports/prof_requests_sep2026/  &  notebooks/plots/prof_requests_sep2026/')


if __name__ == '__main__':
    main()
