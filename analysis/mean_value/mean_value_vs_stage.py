"""
Raw CAP mean value (DC baseline) vs PSG sleep stage.

Professor directive (Results opener): "Start with mean value (raw signal changes)
in comparison to sleep stages."

Question
--------
Does the raw capacitive temple-sensor DC level (the slow baseline / mean value of
CLE, CRE and the CLE-CRE differential) vary systematically with PSG sleep stage,
consistently across subjects?

Approach
--------
Per 30 s PSG epoch, for each channel (CLE, CRE, CLE-CRE) compute:
  * mean_raw  : mean of the raw samples in the epoch      (the DC level)
  * base_vlf  : very-low-frequency baseline (<0.05 Hz), sampled at epoch centre
Per-session z-score removes cross-subject offset/scale drift.
A per-session slow-trend-removed version (mean_raw_detr) controls for the
time-of-night baseline drift confound (DC drifts monotonically over the night,
and stages have temporal structure -> potential confound).

Quantification
--------------
  * Kruskal-Wallis across the 5 stages (pooled, z-scored).
  * Per-stage medians -> effect direction.
  * Per-subject consistency: sign of Wake-vs-sleep and N3-vs-rest contrasts across
    the 6 subjects (make-or-break: universal vs subject-dependent, cf. HER result).
  * LOSO AUC for the most promising contrast.

Outputs
-------
  reports/mean_value/mean_value_epochs.csv
  reports/mean_value/mean_value_stage_stats.csv
  reports/mean_value/mean_value_subject_direction.csv
  notebooks/plots/mean_value/spectrogram_hypno_<label>.png   (2-3 sessions)
  notebooks/plots/mean_value/meanvalue_hypno_<label>.png     (2-3 sessions)
  notebooks/plots/mean_value/boxplot_by_stage.png
  notebooks/plots/mean_value/subject_direction.png

Usage:
    py analysis/mean_value/mean_value_vs_stage.py
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
from scipy.stats import kruskal, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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

EPOCH_SEC = 30.0
# All CAP channels + the accelerometer magnitude, treated as mean-value channels.
CHANNELS = ['CLE', 'CRE', 'CLE-CRE', 'CLE+CRE', 'acc']
CAP_CH = ['CLE', 'CRE', 'CLE-CRE']       # channels drawn in the spectrogram overlay
VLF_HZ = 0.05           # slow baseline cutoff (below respiratory band 0.1-0.5 Hz)
DETREND_WIN_EPOCHS = 61  # ~30 min rolling-median trend to remove (odd -> centred)
REPRESENTATIVE = ['S1N1', 'S3N1', 'S5N1']  # sessions for time-series/spectrogram figs


# ── Per-epoch feature extraction ─────────────────────────────────────────────

def block_average(x, fs, block_sec=1.0):
    """Downsample by non-overlapping block mean. Returns (t_s, y) at 1/block_sec Hz."""
    n = int(round(fs * block_sec))
    m = len(x) // n
    y = x[: m * n].reshape(m, n).mean(axis=1)
    t = (np.arange(m) + 0.5) * block_sec
    return t, y


def vlf_baseline(sig, fs):
    """Very-low-frequency (<VLF_HZ) baseline via 1 Hz block-average + lowpass."""
    from scipy.signal import butter, filtfilt
    t1, y1 = block_average(sig, fs, 1.0)          # -> 1 Hz series
    nyq = 0.5                                       # fs_new = 1 Hz
    b, a = butter(3, VLF_HZ / nyq, btype='low')
    yb = filtfilt(b, a, y1)
    return t1, yb                                   # seconds, baseline


def extract_session(idx):
    """Return per-epoch DataFrame of mean-value features for one session."""
    meta = SESSION_META[idx]
    s = load_session(idx)
    sp = load_sleep_profile(s)
    if sp is None:
        print(f'  {meta["label"]}: no sleep profile, skipping')
        return None
    s.sleep_profile = sp

    fs = s.fs
    t_hr = s.time_hr.astype(np.float64)
    raw = {
        'CLE': s.cap['CLE'].astype(np.float64),
        'CRE': s.cap['CRE'].astype(np.float64),
    }
    raw['CLE-CRE'] = raw['CLE'] - raw['CRE']
    raw['CLE+CRE'] = 0.5 * (raw['CLE'] + raw['CRE'])
    acc = s.cap['acc_mag'].astype(np.float64)
    raw['acc'] = acc

    # VLF baselines (1 Hz series) per channel, interpolated onto epoch centres.
    vlf = {}
    for ch in CHANNELS:
        t1, yb = vlf_baseline(raw[ch], fs)
        vlf[ch] = (t1 / 3600.0, yb)   # hours

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
        if i1 - i0 < int(0.5 * fs * EPOCH_SEC):   # need >=half epoch of samples
            continue
        tc = t0 + dt_hr / 2.0
        row = {
            'session': meta['label'], 'subject': meta['subject'],
            'night': meta['night'], 'epoch': j, 't_hr': tc,
            'stage_code': int(codes[j]),
            'acc_std': float(acc[i0:i1].std()),
        }
        for ch in CHANNELS:
            seg = raw[ch][i0:i1]
            row[f'mean_{ch}'] = float(seg.mean())
            tb, yb = vlf[ch]
            row[f'vlf_{ch}'] = float(np.interp(tc, tb, yb))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # Per-session normalisation + slow-trend removal.
    for ch in CHANNELS:
        for pref in ('mean', 'vlf'):
            col = f'{pref}_{ch}'
            x = df[col].to_numpy()
            z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
            df[f'{col}_z'] = z
            # rolling-median slow trend (time-of-night confound) then re-z
            trend = pd.Series(x).rolling(
                DETREND_WIN_EPOCHS, center=True, min_periods=5).median().to_numpy()
            resid = x - trend
            zr = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-12)
            df[f'{col}_detr_z'] = zr
    # per-session motion flag (top-decile acc std)
    thr = np.nanpercentile(df['acc_std'], 90)
    df['motion'] = df['acc_std'] > thr
    return df, s


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_spectrogram_hypno(s, df, out):
    """Low-freq spectrogram of raw CLE + hypnogram + mean-value trace (context)."""
    sig = s.cap['CLE'].astype(np.float64)
    f, t, Sxx = sp_spectrogram(sig, fs=FS, nperseg=4096, noverlap=3072,
                               nfft=8192, scaling='density')
    mask = f <= 1.0
    Sdb = 10 * np.log10(Sxx[mask] + 1e-30)
    th = t / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 7),
                             gridspec_kw={'height_ratios': [0.25, 1.0, 0.6]},
                             sharex=True)
    ax = axes[0]
    sp = s.sleep_profile
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                   color=STAGE_COLORS.get(c, '#AAA'), alpha=0.7)
    ax.set_yticks([]); ax.set_ylabel('Stage', fontsize=10)
    ax.legend(handles=[mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
                       for c in STAGE_ORDER],
              loc='upper right', fontsize=8, ncol=5, framealpha=0.9)
    ax.set_title(f'{s.label} — raw CLE low-freq spectrogram + DC baseline vs stage',
                 fontsize=12, fontweight='bold')

    ax = axes[1]
    vmin, vmax = np.nanpercentile(Sdb, [5, 97])
    ax.pcolormesh(th, f[mask], Sdb, shading='gouraud', cmap='inferno',
                  vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_ylabel('Frequency (Hz)', fontsize=10); ax.set_ylim(0, 1.0)

    ax = axes[2]
    for ch in CAP_CH:
        ax.plot(df['t_hr'], df[f'mean_{ch}_z'], lw=1.1,
                color=CAP_COLORS.get(ch, None), label=f'{ch} mean (z)')
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel('Mean value (z)', fontsize=10)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_meanvalue_hypno(s, df, out):
    """Mean-value trace overlaid with colour-banded hypnogram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    sp = s.sleep_profile
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                   color=STAGE_COLORS.get(c, '#AAA'), alpha=0.18)
    ax.plot(df['t_hr'], df['mean_CLE-CRE_z'], lw=1.4, color=CAP_COLORS['CLE-CRE'],
            label='CLE-CRE mean (z)')
    ax.plot(df['t_hr'], df['mean_CLE_z'], lw=1.0, color=CAP_COLORS['CLE'],
            alpha=0.8, label='CLE mean (z)')
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('Mean value (z)', fontsize=10)
    ax.set_title(f'{s.label} — raw mean value vs hypnogram', fontsize=12,
                 fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
                       for c in STAGE_ORDER]
              + [plt.Line2D([], [], color=CAP_COLORS['CLE-CRE'], label='CLE-CRE mean'),
                 plt.Line2D([], [], color=CAP_COLORS['CLE'], label='CLE mean')],
              loc='upper right', fontsize=7, ncol=4, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_boxplot(df, feat, out):
    """Per-stage boxplot of a z-scored mean-value feature (pooled, all subjects)."""
    d = df[df['stage_code'] >= 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []
    for sc in STAGE_ORDER:
        v = d[d['stage_code'] == sc][feat].dropna().values
        if len(v) > 0:
            data.append(v); labels.append(STAGE_LABELS[sc])
            colors.append(STAGE_COLORS[sc])
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=2), widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for i, v in enumerate(data):
        ax.text(i + 1, np.median(v) + 0.05, f'{np.median(v):.2f}',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_xticklabels(labels)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel(f'{feat}', fontsize=10)
    ax.set_title(f'Raw mean value by stage — {feat} (12 sessions pooled)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_subject_direction(df, feat, out):
    """Per-subject per-stage median heatmap -> direction consistency check."""
    d = df[df['stage_code'] >= 0]
    subs = sorted(d['subject'].unique())
    H = np.full((len(subs), len(STAGE_ORDER)), np.nan)
    for si, sub in enumerate(subs):
        ds = d[d['subject'] == sub]
        for sj, sc in enumerate(STAGE_ORDER):
            v = ds[ds['stage_code'] == sc][feat].dropna().values
            if len(v) >= 10:
                H[si, sj] = np.median(v)
    fig, ax = plt.subplots(figsize=(7, 5))
    vlim = np.nanmax(np.abs(H))
    im = ax.imshow(H, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_LABELS[c] for c in STAGE_ORDER])
    ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs)
    for si in range(len(subs)):
        for sj in range(len(STAGE_ORDER)):
            if np.isfinite(H[si, sj]):
                ax.text(sj, si, f'{H[si, sj]:.2f}', ha='center', va='center',
                        fontsize=8, color='black')
    plt.colorbar(im, ax=ax, shrink=0.8, label=f'median {feat}')
    ax.set_title(f'Per-subject median {feat} by stage\n(direction consistency)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── Stats ────────────────────────────────────────────────────────────────────

def stage_stats(df, feat):
    d = df[df['stage_code'] >= 0]
    groups = [d[d['stage_code'] == sc][feat].dropna().values for sc in STAGE_ORDER]
    groups_lbl = [STAGE_LABELS[sc] for sc in STAGE_ORDER]
    valid = [(g, l) for g, l in zip(groups, groups_lbl) if len(g) > 10]
    gs = [g for g, _ in valid]
    H, p = kruskal(*gs)
    meds = {STAGE_LABELS[sc]: float(np.median(d[d['stage_code'] == sc][feat].dropna()))
            for sc in STAGE_ORDER if len(d[d['stage_code'] == sc]) > 10}
    return H, p, meds


def contrast_auc(d, feat, pos_mask):
    """AUC of |feat| separating pos vs rest; return best-direction AUC + sign."""
    y = pos_mask.astype(int).values
    x = d[feat].values
    ok = np.isfinite(x)
    if ok.sum() < 20 or y[ok].sum() < 5 or (1 - y[ok]).sum() < 5:
        return np.nan, ''
    a_pos = roc_auc_score(y[ok], x[ok])
    a_neg = roc_auc_score(y[ok], -x[ok])
    if a_neg > a_pos:
        return a_neg, 'low'   # positive class has LOWER mean value
    return a_pos, 'high'


def loso_auc(df, feat, pos_def):
    """LOSO logistic AUC for a binary stage contrast using a single mean-value feat."""
    d = df[df['stage_code'] >= 0].copy()
    d = d[np.isfinite(d[feat])]
    y = pos_def(d).astype(int).values
    X = d[[feat]].values
    subs = d['subject'].values
    uniq = sorted(set(subs))
    yt_all, yp_all = [], []
    for ts in uniq:
        tr = subs != ts; te = subs == ts
        if y[tr].sum() < 10 or y[te].sum() < 3:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        yt_all.extend(y[te]); yp_all.extend(p)
    if len(set(yt_all)) < 2:
        return np.nan
    return roc_auc_score(yt_all, yp_all)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 64)
    print('Raw CAP mean value (DC baseline) vs sleep stage')
    print('=' * 64)

    all_dfs = []
    sess_cache = {}
    for idx in range(len(SESSION_META)):
        res = extract_session(idx)
        if res is None:
            continue
        df, s = res
        all_dfs.append(df)
        if s.label in REPRESENTATIVE:
            sess_cache[s.label] = (s, df)
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(REPORT_DIR / 'mean_value_epochs.csv', index=False)
    print(f'\nPooled {len(df)} epochs across {df["session"].nunique()} sessions, '
          f'{df["subject"].nunique()} subjects')

    # ── Stage-wise stats for each feature ──
    feats = []
    for ch in CHANNELS:
        feats += [f'mean_{ch}_z', f'mean_{ch}_detr_z', f'vlf_{ch}_z']
    stat_rows = []
    print('\nKruskal-Wallis across 5 stages (pooled, z-scored):')
    for feat in feats:
        H, p, meds = stage_stats(df, feat)
        order = sorted(meds, key=meds.get)
        stat_rows.append({'feature': feat, 'KW_H': H, 'KW_p': p,
                          **{f'med_{k}': v for k, v in meds.items()},
                          'low_to_high': ' < '.join(order)})
        print(f'  {feat:22s} H={H:8.1f}  p={p:.2e}  order: {" < ".join(order)}')
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(REPORT_DIR / 'mean_value_stage_stats.csv', index=False)

    # motion-clean sensitivity on primary feature
    prim = 'mean_CLE-CRE_z'
    Hc, pc, medc = stage_stats(df[~df['motion']], prim)
    print(f'\nMotion-clean (drop top-decile acc): {prim} H={Hc:.1f} p={pc:.2e}')
    Hd, pd_, medd = stage_stats(df, 'mean_CLE-CRE_detr_z')
    print(f'Trend-removed confound check: mean_CLE-CRE_detr_z H={Hd:.1f} p={pd_:.2e}')

    # ── Per-subject direction consistency (every channel) ──
    print('\nPer-subject direction consistency (mean_<ch>_z):')
    d = df[df['stage_code'] >= 0]
    subs = sorted(d['subject'].unique())
    dir_rows = []
    contrasts = {
        'Wake_vs_sleep': lambda x: x['stage_code'] == 4,
        'N3_vs_rest':    lambda x: x['stage_code'] == 1,
        'REM_vs_rest':   lambda x: x['stage_code'] == 0,
    }
    for ch in CHANNELS:
        feat = f'mean_{ch}_z'
        for cname, cfun in contrasts.items():
            signs = []
            for sub in subs:
                ds = d[d['subject'] == sub]
                auc, sign = contrast_auc(ds, feat, cfun(ds))
                signs.append(sign)
                dir_rows.append({'channel': ch, 'contrast': cname, 'subject': sub,
                                 'feature': feat, 'auc': auc, 'direction': sign})
            highs = signs.count('high'); lows = signs.count('low')
            verdict = 'CONSISTENT' if 0 in (highs, lows) else 'SUBJECT-DEPENDENT'
            print(f'  {ch:8s} {cname:16s}: high={highs} low={lows} -> {verdict}')
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(REPORT_DIR / 'mean_value_subject_direction.csv', index=False)

    # ── LOSO AUC for each contrast, every channel ──
    print('\nLOSO AUC (single mean-value feature, logistic):')
    loso_rows = []
    for cname, cfun in contrasts.items():
        for ch in CHANNELS:
            for feat in [f'mean_{ch}_z', f'mean_{ch}_detr_z']:
                auc = loso_auc(df, feat, cfun)
                loso_rows.append({'contrast': cname, 'channel': ch,
                                  'feature': feat, 'loso_auc': auc})
                print(f'  {cname:16s} {feat:22s} AUC={auc:.3f}')
    pd.DataFrame(loso_rows).to_csv(REPORT_DIR / 'mean_value_loso_auc.csv', index=False)

    # ── Figures ──
    print('\nWriting figures...')
    for lbl, (s, sdf) in sess_cache.items():
        fig_spectrogram_hypno(s, sdf, PLOT_DIR / f'spectrogram_hypno_{lbl}.png')
        fig_meanvalue_hypno(s, sdf, PLOT_DIR / f'meanvalue_hypno_{lbl}.png')
        print(f'  {lbl}: spectrogram + mean-value overlays')
    for ch in CHANNELS:
        safe = ch.replace('+', 'plus').replace('-', '_')
        fig_boxplot(df, f'mean_{ch}_z', PLOT_DIR / f'boxplot_{safe}.png')
        fig_boxplot(df, f'mean_{ch}_detr_z', PLOT_DIR / f'boxplot_{safe}_detr.png')
        fig_subject_direction(df, f'mean_{ch}_z', PLOT_DIR / f'subject_direction_{safe}.png')
        print(f'  {ch}: boxplot + detr boxplot + subject-direction heatmap')

    print('\nDone. Reports -> reports/mean_value/  Figures -> notebooks/plots/mean_value/')


if __name__ == '__main__':
    main()
