"""
Absolute (native a.u.) CAP mean value vs sleep stage — the "not just z-score" analysis.

Motivation
----------
Every other mean-value figure in this project z-scores each channel per session
(subtract median, divide by MAD/SD). Z-scoring discards BOTH the between-session
DC offset AND the native amplitude scale, so it can only speak to the *shape* of
the slow mean, never to "how many a.u. does the baseline actually move by stage".
The professor's Results directive ("Start with mean value (raw signal changes) in
comparison to sleep stages") asks the opposite question, in absolute units.

This script quantifies, in NATIVE a.u.:
  1. Between-session absolute baseline offset  (per-session median DC level).
  2. Within-night drift                        (last-30min minus first-30min).
  3. Within-session stage excursion            (per-session-centered mean by stage),
     preserving a.u. scale (subtract per-session Wake reference; do NOT divide by SD).
  4. Sign consistency of the Wake/N3 absolute shift across the 6 subjects.
  5. Kruskal-Wallis across stages on the per-session-centered a.u. value.

The headline is a scale comparison: between-session offset >> within-night drift
>~ within-session stage excursion, and the stage excursion has no universal sign.
That is the honest-characterization reason the rest of the pipeline normalizes.

Inputs
------
reports/mean_value/mean_value_epochs.csv   (raw a.u. columns mean_CLE, mean_CRE,
    mean_CLE-CRE, mean_CLE+CRE already present; produced by mean_value_vs_stage.py)

Outputs
-------
reports/mean_value/abs_mean_scale.csv          scale comparison per channel
reports/mean_value/abs_mean_stage_au.csv        per-stage centered a.u. + KW
reports/mean_value/abs_mean_subject_direction.csv  per-subject Wake/N3 a.u. signs
notebooks/plots/mean_value/abs_baseline_by_session.png    (offset dominance)
notebooks/plots/mean_value/abs_stage_boxplot_au.png       (a.u. stage excursion)
notebooks/plots/mean_value/abs_scale_comparison.png       (offset vs drift vs stage)
notebooks/plots/mean_value/abs_trace_<SESSION>.png        (raw a.u. + spectrogram)

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/abs_mean_vs_stage.py
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
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile, FS
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, CAP_COLORS,
)
from sleep_monitor.sessions import SESSION_META

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'mean_value'
REPORT_DIR = ROOT / 'reports' / 'mean_value'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CLE', 'CRE', 'CLE-CRE', 'CLE+CRE']
CH_COLOR = {'CLE': '#27AE60', 'CRE': '#8E44AD',
            'CLE-CRE': '#E67E22', 'CLE+CRE': '#7F8C8D'}
WAKE, N3 = 4, 1
REPRESENTATIVE = ['S1N1', 'S3N1', 'S5N1']
DRIFT_WIN_HR = 0.5   # 30-min window for start/end drift estimate


# ── Stats in native a.u. ──────────────────────────────────────────────────────

def centered_by_session(df, col, ref_stage=WAKE):
    """Subtract per-session median of ref_stage; keep a.u. scale (no /SD)."""
    d = df[df['stage_code'] >= 0].copy()
    ref = d[d['stage_code'] == ref_stage].groupby('session')[col].median()
    d = d[d['session'].isin(ref.index)].copy()
    d['centered_au'] = d[col] - d['session'].map(ref)
    return d


def scale_table(df):
    rows = []
    for ch in CHANNELS:
        col = f'mean_{ch}'
        offset = df.groupby('session')[col].median()
        # within-night drift per session (a.u.)
        drift = []
        for _, g in df.groupby('session'):
            g = g.sort_values('t_hr')
            t = g['t_hr'].values
            early = g[g['t_hr'] <= t[0] + DRIFT_WIN_HR][col].median()
            late = g[g['t_hr'] >= t[-1] - DRIFT_WIN_HR][col].median()
            drift.append(late - early)
        drift = np.array(drift, float)
        d = centered_by_session(df, col)
        iqr = d.groupby('session')['centered_au'].apply(
            lambda x: x.quantile(.75) - x.quantile(.25))
        rows.append({
            'channel': ch,
            'between_session_offset_SD_au': offset.std(),
            'between_session_offset_range_au': offset.max() - offset.min(),
            'within_night_drift_absmedian_au': np.median(np.abs(drift)),
            'within_session_centered_IQR_median_au': iqr.median(),
        })
    return pd.DataFrame(rows)


def stage_au_table(df):
    rows = []
    for ch in CHANNELS:
        col = f'mean_{ch}'
        d = centered_by_session(df, col)
        groups = [d[d['stage_code'] == sc]['centered_au'].dropna().values
                  for sc in STAGE_ORDER]
        valid = [g for g in groups if len(g) > 10]
        H, p = kruskal(*valid) if len(valid) >= 2 else (np.nan, np.nan)
        row = {'channel': ch, 'KW_H': H, 'KW_p': p}
        for sc in STAGE_ORDER:
            v = d[d['stage_code'] == sc]['centered_au'].dropna()
            row[f'med_{STAGE_LABELS[sc]}_au'] = float(np.median(v)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def subject_direction_table(df):
    rows = []
    for ch in CHANNELS:
        col = f'mean_{ch}'
        d = centered_by_session(df, col)
        for contrast, mask in (('N3_vs_Wake', d['stage_code'] == N3),
                               ('Sleep_vs_Wake', d['stage_code'].isin([0, 1, 2, 3]))):
            sub = d[mask].groupby('subject')['centered_au'].median()
            pos, neg = int((sub > 0).sum()), int((sub < 0).sum())
            rows.append({
                'channel': ch, 'contrast': contrast,
                'per_subject_au': ', '.join(f'{v:.0f}' for v in sub.values),
                'n_pos': pos, 'n_neg': neg,
                'verdict': 'CONSISTENT' if 0 in (pos, neg) else 'SUBJECT-DEPENDENT',
            })
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_baseline_by_session(df, out):
    """Per-session absolute DC level (raw a.u.) — visualises offset dominance."""
    sessions = sorted(df['session'].unique())
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(17, 5), sharex=False)
    for ax, ch in zip(axes, CHANNELS):
        col = f'mean_{ch}'
        data = [df[df['session'] == s][col].dropna().values for s in sessions]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, vert=True,
                        widths=0.6, medianprops=dict(color='k', lw=1.4))
        for patch in bp['boxes']:
            patch.set_facecolor(CH_COLOR[ch]); patch.set_alpha(0.55)
        ax.set_title(f'{ch}', fontsize=11, fontweight='bold', color=CH_COLOR[ch])
        ax.set_xticks(range(1, len(sessions) + 1))
        ax.set_xticklabels(sessions, rotation=90, fontsize=6)
        ax.set_ylabel('absolute mean value (a.u.)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.15)
    fig.suptitle('Absolute CAP DC baseline per session (raw a.u.) — dominated by '
                 'between-session coupling offset', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches='tight'); plt.close(fig)


def fig_stage_boxplot_au(df, out):
    """Per-session-centered mean value by stage, in NATIVE a.u. (scale preserved)."""
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(17, 4.6), sharey=False)
    for ax, ch in zip(axes, CHANNELS):
        col = f'mean_{ch}'
        d = centered_by_session(df, col)
        data, labels, colors = [], [], []
        for sc in STAGE_ORDER:
            v = d[d['stage_code'] == sc]['centered_au'].dropna().values
            if len(v) > 10:
                data.append(v); labels.append(STAGE_LABELS[sc])
                colors.append(STAGE_COLORS[sc])
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.62,
                        medianprops=dict(color='k', lw=1.8))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for i, v in enumerate(data):
            ax.text(i + 1, np.median(v), f'{np.median(v):+.1f}', ha='center',
                    va='bottom', fontsize=8, fontweight='bold')
        ax.axhline(0, color='gray', ls=':', lw=0.9)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(ch, fontsize=11, fontweight='bold', color=CH_COLOR[ch])
        ax.set_ylabel('mean value re. session Wake (a.u.)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.15)
    fig.suptitle('Within-session absolute mean-value excursion by stage (native a.u., '
                 'per-session Wake reference) — a few a.u., no universal direction',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches='tight'); plt.close(fig)


def fig_scale_comparison(scale_df, out):
    """Bar chart: between-session offset vs within-night drift vs stage IQR (a.u.)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CHANNELS))
    w = 0.26
    off = scale_df.set_index('channel').loc[CHANNELS, 'between_session_offset_SD_au']
    dr = scale_df.set_index('channel').loc[CHANNELS, 'within_night_drift_absmedian_au']
    iq = scale_df.set_index('channel').loc[CHANNELS, 'within_session_centered_IQR_median_au']
    ax.bar(x - w, off, w, label='between-session offset (SD)', color='#34495E')
    ax.bar(x, dr, w, label='within-night drift (|median|)', color='#E67E22')
    ax.bar(x + w, iq, w, label='within-session stage IQR (median)', color='#2ECC71')
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS, fontsize=10)
    ax.set_ylabel('a.u.', fontsize=11)
    ax.set_title('Where the absolute mean-value variance lives: coupling offset '
                 '≫ night drift ≳ stage effect', fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches='tight'); plt.close(fig)


def fig_abs_trace(label, out):
    """Raw absolute a.u. mean-value trace (NOT z-scored) + low-freq spectrogram."""
    idx = next(i for i, m in enumerate(SESSION_META) if m['label'] == label)
    s = load_session(idx)
    sp = load_sleep_profile(s)
    if sp is None:
        return None
    s.sleep_profile = sp

    cle = s.cap['CLE'].astype(np.float64)
    cre = s.cap['CRE'].astype(np.float64)
    raw = {'CLE': cle, 'CRE': cre, 'CLE-CRE': cle - cre, 'CLE+CRE': 0.5 * (cle + cre)}
    t_hr = s.time_hr.astype(np.float64)

    # 1 Hz block-mean per channel for a light absolute baseline trace
    def block_mean(x, fs, sec=5.0):
        n = int(round(fs * sec)); m = len(x) // n
        y = x[:m * n].reshape(m, n).mean(axis=1)
        tt = (np.arange(m) + 0.5) * sec / 3600.0
        return tt, y

    f, tsp, Sxx = sp_spectrogram(cle, fs=FS, nperseg=4096, noverlap=3072,
                                 nfft=8192, scaling='density')
    fmask = f <= 1.0
    Sdb = 10 * np.log10(Sxx[fmask] + 1e-30)
    th = tsp / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [0.22, 1.0, 0.9]},
                             sharex=True)
    # hypnogram band
    ax = axes[0]
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                   color=STAGE_COLORS.get(c, '#AAA'), alpha=0.75)
    ax.set_yticks([]); ax.set_ylabel('Stage', fontsize=9)
    ax.legend(handles=[mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
                       for c in STAGE_ORDER], loc='upper right', ncol=5, fontsize=8)
    ax.set_title(f'{label} — ABSOLUTE mean value (raw a.u., not z-scored) + '
                 'low-freq spectrogram', fontsize=12, fontweight='bold')
    # spectrogram
    ax = axes[1]
    vmin, vmax = np.nanpercentile(Sdb, [5, 97])
    ax.pcolormesh(th, f[fmask], Sdb, shading='gouraud', cmap='inferno',
                  vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_ylabel('Frequency (Hz)', fontsize=10); ax.set_ylim(0, 1.0)
    # absolute traces in a.u.
    ax = axes[2]
    for ch in CHANNELS:
        tt, y = block_mean(raw[ch], s.fs, 5.0)
        ax.plot(tt, y, lw=1.0, color=CH_COLOR[ch], label=f'{ch} (a.u.)')
    ax.set_ylabel('absolute mean value (a.u.)', fontsize=10)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc='upper right')
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    csv = REPORT_DIR / 'mean_value_epochs.csv'
    if not csv.exists():
        sys.exit(f'Missing {csv}. Run mean_value_vs_stage.py first.')
    df = pd.read_csv(csv)
    print('=' * 68)
    print('ABSOLUTE (native a.u.) CAP mean value vs sleep stage')
    print('=' * 68)
    print(f'{df["session"].nunique()} sessions, {df["subject"].nunique()} subjects, '
          f'{len(df)} epochs\n')

    scale_df = scale_table(df)
    scale_df.to_csv(REPORT_DIR / 'abs_mean_scale.csv', index=False)
    print('Scale comparison (a.u.):')
    for _, r in scale_df.iterrows():
        print(f'  {r["channel"]:9s} offset_SD={r["between_session_offset_SD_au"]:7.1f}  '
              f'drift|med|={r["within_night_drift_absmedian_au"]:6.1f}  '
              f'stage_IQR={r["within_session_centered_IQR_median_au"]:6.1f}')

    stage_df = stage_au_table(df)
    stage_df.to_csv(REPORT_DIR / 'abs_mean_stage_au.csv', index=False)
    print('\nPer-stage centered median (a.u.) + Kruskal-Wallis:')
    for _, r in stage_df.iterrows():
        print(f'  {r["channel"]:9s} KW_H={r["KW_H"]:7.1f} p={r["KW_p"]:.2e}  '
              f'Wake={r["med_Wake_au"]:+.1f} N1={r["med_N1_au"]:+.1f} '
              f'N2={r["med_N2_au"]:+.1f} N3={r["med_N3_au"]:+.1f} REM={r["med_REM_au"]:+.1f}')

    dir_df = subject_direction_table(df)
    dir_df.to_csv(REPORT_DIR / 'abs_mean_subject_direction.csv', index=False)
    print('\nPer-subject direction consistency (a.u.):')
    for _, r in dir_df.iterrows():
        print(f'  {r["channel"]:9s} {r["contrast"]:14s} pos={r["n_pos"]} '
              f'neg={r["n_neg"]} -> {r["verdict"]}')

    print('\nWriting figures...')
    fig_baseline_by_session(df, PLOT_DIR / 'abs_baseline_by_session.png')
    fig_stage_boxplot_au(df, PLOT_DIR / 'abs_stage_boxplot_au.png')
    fig_scale_comparison(scale_df, PLOT_DIR / 'abs_scale_comparison.png')
    print('  abs_baseline_by_session.png, abs_stage_boxplot_au.png, abs_scale_comparison.png')
    for lbl in REPRESENTATIVE:
        o = fig_abs_trace(lbl, PLOT_DIR / f'abs_trace_{lbl}.png')
        if o:
            print(f'  abs_trace_{lbl}.png')

    print(f'\nDone. Reports -> {REPORT_DIR}   Figures -> {PLOT_DIR}')


if __name__ == '__main__':
    main()
