"""
Thorax-CAP relationship analysis across all 12 sessions.

Part 1: Per-epoch (30s) feature extraction — CAP (5 channels), thorax, accelerometer
Part 2: Correlation analysis — pooled, per-session, per-sleep-stage
Part 3: Full-night session plots (12 stacked-panel PNGs)
Part 4: Summary correlation plots (heatmap, scatter, bars, stage-stratified)

Output:
  artifacts/thorax_cap_epochs.parquet
  notebooks/plots/thorax_analysis/fullnight_{label}.png   (×12)
  notebooks/plots/thorax_analysis/corr_heatmap.png
  notebooks/plots/thorax_analysis/scatter_best_corr.png
  notebooks/plots/thorax_analysis/per_session_corr_bars.png
  notebooks/plots/thorax_analysis/corr_by_stage.png
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch, find_peaks
from scipy.stats import pearsonr, spearmanr

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    CAP_COLORS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, PSG_EPOCH_SEC,
)
from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.filters import bandpass
from sleep_monitor.motion import epoch_motion

ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT / "notebooks" / "plots" / "thorax_analysis"
ART_DIR = ROOT / "artifacts"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
EPOCH_N = int(EPOCH_SEC * FS)

CAP_LABELS = ['CLE', 'CRE', 'CH', 'CLE-CRE', 'avg']
EXTENDED_CAP_COLORS = {
    **CAP_COLORS,
    'avg': '#E74C3C',
}
THORAX_COLOR = '#2C3E50'


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Per-epoch feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _stage_at_time(sp, t_hr):
    if sp is None:
        return -1, '?'
    idx = int(np.searchsorted(sp['t_ep_hr'], t_hr, side='right')) - 1
    idx = np.clip(idx, 0, len(sp['codes']) - 1)
    code = int(sp['codes'][idx])
    return code, STAGE_LABELS.get(code, '?')


def compute_session_epochs(session) -> list:
    fs = session.fs
    n_samples = session.n_samples

    raw_cle = session.cap['CLE'].astype(np.float64)
    raw_cre = session.cap['CRE'].astype(np.float64)
    raw_ch = session.cap['CH'].astype(np.float64)
    raw_diff = raw_cle - raw_cre
    raw_avg = (raw_cle + raw_cre) / 2.0
    raw_caps = {'CLE': raw_cle, 'CRE': raw_cre, 'CH': raw_ch,
                'CLE-CRE': raw_diff, 'avg': raw_avg}
    acc_mag = session.cap['acc_mag'].astype(np.float64)

    resp_bp = {}
    card_bp = {}
    for ch in CAP_LABELS:
        resp_bp[ch] = remove_acc_artifact(raw_caps[ch], acc_mag, RESP_LO, RESP_HI, fs)
        card_bp[ch] = remove_acc_artifact(raw_caps[ch], acc_mag, CARD_LO, CARD_HI, fs)

    thorax_raw = session.psg['Thorax'].astype(np.float64)
    thorax_bp = bandpass(thorax_raw, RESP_LO, RESP_HI, fs)

    motion = epoch_motion(session, epoch_sec=EPOCH_SEC)
    sp = session.sleep_profile

    n_epochs = n_samples // EPOCH_N
    n_motion = len(motion['t_hr'])
    n_ep = min(n_epochs, n_motion)

    rows = []
    for i in range(n_ep):
        s, e = i * EPOCH_N, (i + 1) * EPOCH_N
        t_center_hr = float(np.mean(session.time_hr[s:e]))
        stage_code, stage_label = _stage_at_time(sp, t_center_hr)

        row = {
            'session': session.label,
            'subject': session.subject,
            't_hr': t_center_hr,
            'stage_code': stage_code,
            'stage_label': stage_label,
        }

        for ch in CAP_LABELS:
            seg_raw = raw_caps[ch][s:e]
            row[f'{ch}_raw_mean'] = float(np.mean(seg_raw))
            row[f'{ch}_raw_std'] = float(np.std(seg_raw))
            row[f'{ch}_resp_rms'] = float(np.sqrt(np.mean(resp_bp[ch][s:e] ** 2)))
            row[f'{ch}_card_rms'] = float(np.sqrt(np.mean(card_bp[ch][s:e] ** 2)))

        seg_thorax_raw = thorax_raw[s:e]
        seg_thorax_bp = thorax_bp[s:e]
        row['thorax_raw_mean'] = float(np.mean(seg_thorax_raw))
        row['thorax_raw_std'] = float(np.std(seg_thorax_raw))
        row['thorax_resp_rms'] = float(np.sqrt(np.mean(seg_thorax_bp ** 2)))
        row['thorax_resp_p2p'] = float(np.max(seg_thorax_bp) - np.min(seg_thorax_bp))

        nperseg = min(len(seg_thorax_bp), int(fs * 4))
        freqs, psd = welch(seg_thorax_bp, fs=fs, nperseg=nperseg)
        mask_f = (freqs >= RESP_LO) & (freqs <= RESP_HI)
        if np.any(mask_f) and np.any(psd[mask_f] > 0):
            row['thorax_dom_freq_hz'] = float(freqs[mask_f][np.argmax(psd[mask_f])])
        else:
            row['thorax_dom_freq_hz'] = np.nan

        peaks_idx, _ = find_peaks(
            seg_thorax_bp,
            distance=int(fs / RESP_HI * 0.6),
            prominence=0.05 * np.std(seg_thorax_bp),
        )
        if len(peaks_idx) >= 2:
            intervals = np.diff(peaks_idx) / fs
            row['thorax_regularity_cov'] = float(np.std(intervals) / (np.mean(intervals) + 1e-12))
        else:
            row['thorax_regularity_cov'] = np.nan

        row['movement_rms'] = float(motion['movement_rms'][i])
        row['movement_peak'] = float(motion['movement_peak'][i])
        row['roll_deg'] = float(motion['roll_deg'][i])
        row['pitch_deg'] = float(motion['pitch_deg'][i])

        rows.append(row)

    return rows


def build_epoch_table():
    all_rows = []
    sleep_profiles = {}
    for i in range(len(SESSION_META)):
        sess = load_session(i)
        sess.sleep_profile = load_sleep_profile(sess)
        print(f'  {sess.label}: extracting epochs...', flush=True)
        rows = compute_session_epochs(sess)
        all_rows.extend(rows)
        sleep_profiles[sess.label] = sess.sleep_profile
        print(f'    {len(rows)} epochs')
        del sess
    df = pd.DataFrame(all_rows)
    return df, sleep_profiles


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Correlation analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_corr(a, b, method='pearson'):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return np.nan, np.nan
    if method == 'pearson':
        return pearsonr(a[mask], b[mask])
    return spearmanr(a[mask], b[mask])


CAP_FEAT_SUFFIXES = ['_raw_mean', '_raw_std', '_resp_rms', '_card_rms']
THORAX_FEAT_COLS = [
    'thorax_raw_mean', 'thorax_raw_std', 'thorax_resp_rms',
    'thorax_resp_p2p', 'thorax_dom_freq_hz', 'thorax_regularity_cov',
]


def _cap_feat_cols():
    cols = []
    for ch in CAP_LABELS:
        for suf in CAP_FEAT_SUFFIXES:
            cols.append(f'{ch}{suf}')
    return cols


def correlation_analysis(df):
    cap_cols = _cap_feat_cols()

    # ── Pooled correlations ───────────────────────────────────────────────────
    pooled_rows = []
    for cc in cap_cols:
        for tc in THORAX_FEAT_COLS:
            r_p, p_p = _safe_corr(df[cc].values, df[tc].values, 'pearson')
            r_s, p_s = _safe_corr(df[cc].values, df[tc].values, 'spearman')
            pooled_rows.append({
                'cap_feat': cc, 'thorax_feat': tc,
                'r_pearson': r_p, 'p_pearson': p_p,
                'r_spearman': r_s, 'p_spearman': p_s,
            })
    pooled_df = pd.DataFrame(pooled_rows)

    print('\n  Top 10 pooled correlations (|Pearson r|):')
    top = pooled_df.dropna(subset=['r_pearson']).copy()
    top['abs_r'] = top['r_pearson'].abs()
    top = top.nlargest(10, 'abs_r')
    for _, row in top.iterrows():
        print(f'    {row.cap_feat:25s} vs {row.thorax_feat:25s}  '
              f'r={row.r_pearson:+.3f}  (Spearman {row.r_spearman:+.3f})')

    # ── Per-session correlations ──────────────────────────────────────────────
    per_sess_rows = []
    for label, gdf in df.groupby('session'):
        for cc in cap_cols:
            for tc in THORAX_FEAT_COLS:
                r, _ = _safe_corr(gdf[cc].values, gdf[tc].values, 'pearson')
                per_sess_rows.append({
                    'session': label, 'cap_feat': cc, 'thorax_feat': tc,
                    'r_pearson': r,
                })
    per_sess_df = pd.DataFrame(per_sess_rows)

    # ── Per-stage correlations ────────────────────────────────────────────────
    per_stage_rows = []
    for code in STAGE_ORDER:
        sdf = df[df['stage_code'] == code]
        if len(sdf) < 20:
            continue
        for cc in cap_cols:
            for tc in THORAX_FEAT_COLS:
                r, _ = _safe_corr(sdf[cc].values, sdf[tc].values, 'pearson')
                per_stage_rows.append({
                    'stage_code': code, 'stage_label': STAGE_LABELS[code],
                    'cap_feat': cc, 'thorax_feat': tc,
                    'r_pearson': r,
                })
    per_stage_df = pd.DataFrame(per_stage_rows)

    return {'pooled': pooled_df, 'per_session': per_sess_df, 'per_stage': per_stage_df}


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Full-night session plots
# ═══════════════════════════════════════════════════════════════════════════════

def _zsc(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / (s if s > 0 else 1.0)


def _hypnogram_strip(ax, sp, dur_hr):
    if sp is None:
        ax.set_visible(False)
        return
    import matplotlib.patches as mpatches
    t = sp['t_ep_hr']
    codes = sp['codes']
    for i in range(len(t) - 1):
        c = int(codes[i])
        ax.axvspan(t[i], t[i + 1], color=STAGE_COLORS.get(c, '#AAA'), alpha=0.85)
    patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
               for c in STAGE_ORDER]
    ax.legend(handles=patches, fontsize=6, loc='upper right', ncol=5, framealpha=0.8)
    ax.set_yticks([])
    ax.set_ylabel('Stage', fontsize=7)
    ax.tick_params(axis='x', labelbottom=False)


def _row_cap_means(ax, df_s):
    t = df_s['t_hr'].values
    for ch in CAP_LABELS:
        ax.plot(t, df_s[f'{ch}_raw_mean'].values,
                color=EXTENDED_CAP_COLORS[ch], lw=0.7, alpha=0.85, label=ch)
    ax.set_ylabel('CAP raw mean', fontsize=7)
    ax.legend(fontsize=6, loc='upper right', ncol=5)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelbottom=False, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)


def _row_cap_stds(ax, df_s):
    t = df_s['t_hr'].values
    for ch in CAP_LABELS:
        ax.plot(t, df_s[f'{ch}_raw_std'].values,
                color=EXTENDED_CAP_COLORS[ch], lw=0.7, alpha=0.85, label=ch)
    ax.set_ylabel('CAP raw std', fontsize=7)
    ax.legend(fontsize=6, loc='upper right', ncol=5)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelbottom=False, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)


def _row_cap_vs_thorax_resp(ax, df_s):
    t = df_s['t_hr'].values
    for ch in CAP_LABELS:
        vals = _zsc(df_s[f'{ch}_resp_rms'].values)
        ax.plot(t, vals, color=EXTENDED_CAP_COLORS[ch], lw=0.6, alpha=0.7, label=ch)
    thorax_z = _zsc(df_s['thorax_resp_rms'].values)
    ax.plot(t, thorax_z, color=THORAX_COLOR, lw=1.5, alpha=0.9, label='Thorax', zorder=5)
    ax.set_ylabel('Resp RMS (z)', fontsize=7)
    ax.legend(fontsize=6, loc='upper right', ncol=6)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelbottom=False, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)


def _row_cap_card_rms(ax, df_s):
    t = df_s['t_hr'].values
    for ch in CAP_LABELS:
        ax.plot(t, df_s[f'{ch}_card_rms'].values,
                color=EXTENDED_CAP_COLORS[ch], lw=0.7, alpha=0.85, label=ch)
    ax.set_ylabel('Cardiac RMS', fontsize=7)
    ax.legend(fontsize=6, loc='upper right', ncol=5)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelbottom=False, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)


def _row_thorax_features(ax, df_s):
    t = df_s['t_hr'].values
    ax.plot(t, _zsc(df_s['thorax_resp_rms'].values),
            color='#27AE60', lw=0.8, label='RMS')
    ax.plot(t, _zsc(df_s['thorax_dom_freq_hz'].values),
            color='#E67E22', lw=0.8, ls='--', label='Dom freq')
    ax.plot(t, _zsc(df_s['thorax_regularity_cov'].values),
            color='#8E44AD', lw=0.8, ls=':', label='Regularity')
    ax.set_ylabel('Thorax (z)', fontsize=7)
    ax.legend(fontsize=6, loc='upper right', ncol=3)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelbottom=False, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)


def _row_thorax_raw(ax, df_s):
    t = df_s['t_hr'].values
    mean = df_s['thorax_raw_mean'].values
    std = df_s['thorax_raw_std'].values
    ax.plot(t, mean, color=THORAX_COLOR, lw=0.7, alpha=0.9, label='Thorax mean')
    ax.fill_between(t, mean - std, mean + std, color=THORAX_COLOR, alpha=0.15,
                    label='Thorax +/-std')
    ax.set_ylabel('Thorax raw\nmean+/-std', fontsize=7)
    ax.legend(fontsize=6, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='x', labelbottom=False, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)


def _row_accel(ax, df_s):
    t = df_s['t_hr'].values
    ax.plot(t, df_s['roll_deg'].values, color='#2980B9', lw=0.6, alpha=0.8, label='Roll')
    ax.plot(t, df_s['pitch_deg'].values, color='#27AE60', lw=0.6, alpha=0.8, label='Pitch')
    ax.set_ylabel('Angle (°)', fontsize=7, color='#2C3E50')
    ax.tick_params(axis='y', labelsize=6)

    ax2 = ax.twinx()
    ax2.fill_between(t, 0, df_s['movement_rms'].values,
                     color='#E74C3C', alpha=0.25, label='Move RMS')
    ax2.set_ylabel('Move RMS', fontsize=7, color='#E74C3C')
    ax2.tick_params(axis='y', labelsize=6, colors='#E74C3C')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper right', ncol=3)
    ax.grid(True, alpha=0.2)


def plot_fullnight_session(df_s, meta, sp=None):
    dur_hr = df_s['t_hr'].max()
    has_sp = sp is not None

    n_rows = 8 + int(has_sp)
    height_ratios = ([0.6] if has_sp else []) + [1, 1, 1, 1.2, 1, 1.2, 1, 1]

    fig = plt.figure(figsize=(18, 2.2 * n_rows))
    gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.35, height_ratios=height_ratios)

    row = 0
    first_ax = None

    if has_sp:
        ax = fig.add_subplot(gs[row])
        _hypnogram_strip(ax, sp, dur_hr)
        first_ax = ax
        row += 1

    ax = fig.add_subplot(gs[row], sharex=first_ax) if first_ax else fig.add_subplot(gs[row])
    if first_ax is None:
        first_ax = ax
    _row_cap_means(ax, df_s)
    row += 1

    ax = fig.add_subplot(gs[row], sharex=first_ax)
    _row_cap_stds(ax, df_s)
    row += 1

    ax = fig.add_subplot(gs[row], sharex=first_ax)
    _row_thorax_raw(ax, df_s)
    row += 1

    ax = fig.add_subplot(gs[row], sharex=first_ax)
    _row_cap_vs_thorax_resp(ax, df_s)
    row += 1

    ax = fig.add_subplot(gs[row], sharex=first_ax)
    _row_cap_card_rms(ax, df_s)
    row += 1

    ax = fig.add_subplot(gs[row], sharex=first_ax)
    _row_thorax_features(ax, df_s)
    row += 1

    ax = fig.add_subplot(gs[row], sharex=first_ax)
    _row_accel(ax, df_s)
    ax.set_xlabel('Time (hr)', fontsize=9)
    ax.set_xlim(0, dur_hr)

    label = meta['label']
    fig.suptitle(
        f"{label}  {meta['subject']}-{meta['initials']}  {meta['date']}  ({dur_hr:.1f} hr)",
        fontsize=13, fontweight='bold', y=0.995,
    )

    out = PLOT_DIR / f'fullnight_{label}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {out.name}')


def plot_all_fullnights(df, sleep_profiles):
    meta_by_label = {m['label']: m for m in SESSION_META}
    for label in sorted(df['session'].unique()):
        df_s = df[df['session'] == label].sort_values('t_hr').reset_index(drop=True)
        meta = meta_by_label[label]
        sp = sleep_profiles.get(label)
        plot_fullnight_session(df_s, meta, sp=sp)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — Summary correlation plots
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_corr_heatmap(pooled_df):
    pivot = pooled_df.pivot(index='cap_feat', columns='thorax_feat', values='r_pearson')
    cap_order = _cap_feat_cols()
    thorax_order = THORAX_FEAT_COLS
    pivot = pivot.reindex(index=cap_order, columns=thorax_order)

    fig, ax = plt.subplots(figsize=(10, 14))
    im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5,
                        color='white' if abs(v) > 0.5 else 'black')
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([c.replace('thorax_', '') for c in pivot.columns],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=6)
    fig.colorbar(im, ax=ax, label='Pearson r', shrink=0.5)
    ax.set_title('CAP features vs Thorax features — Pearson r (pooled)', fontsize=11)
    fig.savefig(PLOT_DIR / 'corr_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('    Saved corr_heatmap.png')


def _plot_best_scatter(df, pooled_df):
    sub = pooled_df[pooled_df['thorax_feat'] == 'thorax_resp_rms'].dropna(subset=['r_pearson'])
    if sub.empty:
        return
    best_row = sub.loc[sub['r_pearson'].abs().idxmax()]
    cap_col = best_row['cap_feat']

    fig, ax = plt.subplots(figsize=(8, 6))
    ok = df[cap_col].notna() & df['thorax_resp_rms'].notna() & (df['stage_code'] >= 0)
    d = df[ok]
    for code in STAGE_ORDER:
        m = d['stage_code'] == code
        if m.any():
            ax.scatter(d.loc[m, 'thorax_resp_rms'], d.loc[m, cap_col],
                       s=6, alpha=0.3, color=STAGE_COLORS[code],
                       label=STAGE_LABELS[code], rasterized=True)
    r = best_row['r_pearson']
    ax.set_xlabel('Thorax resp RMS', fontsize=9)
    ax.set_ylabel(cap_col, fontsize=9)
    ax.set_title(f'Best correlation: {cap_col} vs Thorax resp RMS  (r={r:.3f})', fontsize=10)
    ax.legend(fontsize=7, markerscale=3)
    ax.grid(True, alpha=0.2)
    fig.savefig(PLOT_DIR / 'scatter_best_corr.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('    Saved scatter_best_corr.png')


def _plot_session_corr_bars(per_sess_df, pooled_df):
    sub = pooled_df[pooled_df['thorax_feat'] == 'thorax_resp_rms'].dropna(subset=['r_pearson'])
    if sub.empty:
        return
    best_row = sub.loc[sub['r_pearson'].abs().idxmax()]
    best_cap = best_row['cap_feat']
    best_thorax = 'thorax_resp_rms'

    pair = per_sess_df[(per_sess_df['cap_feat'] == best_cap) &
                       (per_sess_df['thorax_feat'] == best_thorax)]
    pair = pair.sort_values('session')

    fig, ax = plt.subplots(figsize=(12, 5))
    sessions = pair['session'].values
    r_vals = pair['r_pearson'].values
    x = np.arange(len(sessions))
    colors = ['#3498DB' if r >= 0 else '#E74C3C' for r in r_vals]
    ax.bar(x, r_vals, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontsize=8, rotation=45)
    mean_r = np.nanmean(r_vals)
    ax.axhline(mean_r, color='#E74C3C', ls='--', lw=1.2, label=f'Mean r={mean_r:.3f}')
    ax.set_ylabel('Pearson r', fontsize=9)
    ax.set_title(f'{best_cap} vs {best_thorax} per session', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.2)
    fig.savefig(PLOT_DIR / 'per_session_corr_bars.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('    Saved per_session_corr_bars.png')


def _plot_corr_by_stage(per_stage_df):
    if per_stage_df.empty:
        return
    stages = [c for c in STAGE_ORDER if c in per_stage_df['stage_code'].values]
    n = len(stages)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 10), sharey=True)
    if n == 1:
        axes = [axes]

    cap_order = _cap_feat_cols()
    for ax, code in zip(axes, stages):
        sdf = per_stage_df[per_stage_df['stage_code'] == code]
        pivot = sdf.pivot(index='cap_feat', columns='thorax_feat', values='r_pearson')
        pivot = pivot.reindex(index=cap_order, columns=THORAX_FEAT_COLS)
        im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(STAGE_LABELS[code], fontsize=10, color=STAGE_COLORS[code], fontweight='bold')
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([c.replace('thorax_', '') for c in pivot.columns],
                           rotation=45, ha='right', fontsize=6)
        if ax == axes[0]:
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index, fontsize=5)
        else:
            ax.set_yticks([])

    fig.suptitle('CAP vs Thorax correlations by sleep stage', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=axes, label='Pearson r', shrink=0.4)
    fig.savefig(PLOT_DIR / 'corr_by_stage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('    Saved corr_by_stage.png')


def plot_summary_correlations(df, corr_results):
    _plot_corr_heatmap(corr_results['pooled'])
    _plot_best_scatter(df, corr_results['pooled'])
    _plot_session_corr_bars(corr_results['per_session'], corr_results['pooled'])
    _plot_corr_by_stage(corr_results['per_stage'])


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('Thorax-CAP Relationship Analysis')
    print('=' * 60)

    # -- Part 1 --
    print('\n-- Part 1: Epoch feature extraction --')
    df, sleep_profiles = build_epoch_table()
    pq_path = ART_DIR / 'thorax_cap_epochs.parquet'
    df.to_parquet(pq_path, index=False)
    print(f'Saved {len(df)} rows x {len(df.columns)} cols to {pq_path}')

    # -- Part 2 --
    print('\n-- Part 2: Correlation analysis --')
    corr_results = correlation_analysis(df)

    # -- Part 3 --
    print('\n-- Part 3: Full-night session plots --')
    plot_all_fullnights(df, sleep_profiles)

    # -- Part 4 --
    print('\n-- Part 4: Summary correlation plots --')
    plot_summary_correlations(df, corr_results)

    print(f'\nAll plots saved to {PLOT_DIR}')
    print(f'Epoch table: {pq_path}')


if __name__ == '__main__':
    main()
