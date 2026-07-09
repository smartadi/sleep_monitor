"""
Raw per-window signal statistics vs PSG sleep stage.

Extends `mean_value_vs_stage.py` (which covered only the DC mean) to a spanning
set of *time-domain* raw statistics, computed per 30 s PSG epoch for every
channel (CLE, CRE, CLE-CRE, CLE+CRE, acc) and compared against sleep stage.

Statistic families
------------------
  level       mean        DC baseline / coupling offset          (also in mean_value)
  dispersion  std         per-window amplitude (resp+cardiac+motion energy)
  dispersion  iqr         robust amplitude (outlier-insensitive)
  trend       slope_win   within-epoch smoothed slope (local DC drift rate)
  trend       slope_vlf   slope of the <0.05 Hz baseline (the "smoothed slope")
  roughness   linelen     mean |Δ| — how busy/active the trace is
  complexity  mobility    Hjorth mobility (dominant-freq proxy)
  complexity  complexity  Hjorth complexity (bandwidth proxy)
  shape       skew        waveform asymmetry
  shape       kurt        peakedness / spikiness

Per-session z-score removes cross-subject offset/scale; a rolling-median
slow-trend-removed variant (`_detr_z`) controls the time-of-night drift confound.

Question: does any raw per-window statistic separate sleep stages *consistently
across subjects* (unlike the DC mean, which was significant-but-subject-dependent)?

Outputs
-------
  reports/mean_value/raw_stats_epochs.csv
  reports/mean_value/raw_stats_kw.csv                (KW H,p + ordering per feature)
  reports/mean_value/raw_stats_subject_direction.csv (per-subject contrast direction)
  reports/mean_value/raw_stats_loso_auc.csv          (LOSO AUC grid)
  notebooks/plots/mean_value/loso_grid_<contrast>.png       (stat x channel overview)
  notebooks/plots/mean_value/boxplot_<stat>_CLE_CRE.png     (std, slope_vlf)
  notebooks/plots/mean_value/subject_direction_<stat>_CLE_CRE.png

Usage:
    .venv/Scripts/python.exe analysis/mean_value/raw_stats_vs_stage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mean_value_vs_stage as mv          # reuse loaders, helpers, fig functions
from mean_value_vs_stage import (
    block_average, vlf_baseline, contrast_auc, loso_auc, stage_stats,
    fig_boxplot, fig_subject_direction,
    CHANNELS, EPOCH_SEC, DETREND_WIN_EPOCHS, PLOT_DIR, REPORT_DIR,
)
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import STAGE_LABELS, STAGE_ORDER
from sleep_monitor.sessions import SESSION_META

STAT_KEYS = ['mean', 'std', 'iqr', 'slope_win', 'slope_vlf',
             'linelen', 'mobility', 'complexity', 'skew', 'kurt']
CONTRASTS = {
    'N3_vs_rest':    lambda x: x['stage_code'] == 1,
    'Wake_vs_sleep': lambda x: x['stage_code'] == 4,
    'REM_vs_rest':   lambda x: x['stage_code'] == 0,
}


# ── Per-window statistic helpers ─────────────────────────────────────────────

def hjorth(x):
    """Hjorth mobility and complexity (time-domain complexity descriptors)."""
    dx = np.diff(x)
    ddx = np.diff(dx)
    v0 = x.var()
    v1 = dx.var()
    v2 = ddx.var()
    mob = np.sqrt(v1 / v0) if v0 > 1e-20 else np.nan
    mob1 = np.sqrt(v2 / v1) if v1 > 1e-20 else np.nan
    comp = (mob1 / mob) if (mob and mob > 1e-20) else np.nan
    return mob, comp


def window_slope(seg, fs):
    """OLS slope (per second) of the 1 Hz block-averaged samples within the epoch."""
    t, y = block_average(seg, fs, 1.0)
    if len(y) < 3:
        return np.nan
    # slope of least-squares line y ~ a + b t
    tm = t - t.mean()
    denom = (tm * tm).sum()
    if denom < 1e-20:
        return np.nan
    return float((tm * (y - y.mean())).sum() / denom)


def epoch_stats(seg, fs, slope_vlf_val):
    """All raw statistics for one epoch segment of one channel."""
    q75, q25 = np.percentile(seg, [75, 25])
    mob, comp = hjorth(seg)
    return {
        'mean': float(seg.mean()),
        'std': float(seg.std()),
        'iqr': float(q75 - q25),
        'slope_win': window_slope(seg, fs),
        'slope_vlf': float(slope_vlf_val),
        'linelen': float(np.abs(np.diff(seg)).mean()),
        'mobility': mob,
        'complexity': comp,
        'skew': float(sp_skew(seg)) if seg.std() > 1e-12 else 0.0,
        'kurt': float(sp_kurtosis(seg)) if seg.std() > 1e-12 else 0.0,
    }


def extract_session(idx):
    """Per-epoch DataFrame of the full raw-statistic battery for one session."""
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
    raw['acc'] = s.cap['acc_mag'].astype(np.float64)

    # VLF baseline + its time-gradient (smoothed slope) per channel.
    vlf_grad = {}
    for ch in CHANNELS:
        t1, yb = vlf_baseline(raw[ch], fs)          # t1 seconds, yb baseline
        g = np.gradient(yb, t1)                       # d(baseline)/dt, per second
        vlf_grad[ch] = (t1 / 3600.0, g)               # hours, slope

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
        if i1 - i0 < int(0.5 * fs * EPOCH_SEC):
            continue
        tc = t0 + dt_hr / 2.0
        row = {
            'session': meta['label'], 'subject': meta['subject'],
            'night': meta['night'], 'epoch': j, 't_hr': tc,
            'stage_code': int(codes[j]),
            'acc_std': float(raw['acc'][i0:i1].std()),
        }
        for ch in CHANNELS:
            seg = raw[ch][i0:i1]
            tb, gb = vlf_grad[ch]
            sv = float(np.interp(tc, tb, gb))
            st = epoch_stats(seg, fs, sv)
            for k, v in st.items():
                row[f'{k}_{ch}'] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # per-session z-score + slow-trend-removed z for every stat/channel
    for ch in CHANNELS:
        for stat in STAT_KEYS:
            col = f'{stat}_{ch}'
            x = df[col].to_numpy(dtype=np.float64)
            df[f'{col}_z'] = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
            trend = pd.Series(x).rolling(
                DETREND_WIN_EPOCHS, center=True, min_periods=5).median().to_numpy()
            resid = x - trend
            df[f'{col}_detr_z'] = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-12)
    thr = np.nanpercentile(df['acc_std'], 90)
    df['motion'] = df['acc_std'] > thr
    return df


# ── LOSO-AUC overview grid (stat x channel) ──────────────────────────────────

def fig_loso_grid(auc_mat, contrast, out):
    """Heatmap of LOSO AUC over stat (rows) x channel (cols) for one contrast."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(auc_mat, aspect='auto', cmap='RdBu_r', vmin=0.35, vmax=0.65)
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS, rotation=20, ha='right')
    ax.set_yticks(range(len(STAT_KEYS)))
    ax.set_yticklabels(STAT_KEYS)
    for i in range(len(STAT_KEYS)):
        for k in range(len(CHANNELS)):
            v = auc_mat[i, k]
            if np.isfinite(v):
                ax.text(k, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(v - 0.5) > 0.11 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8, label='LOSO AUC')
    ax.set_title(f'LOSO AUC by raw statistic x channel — {contrast}\n'
                 f'(0.50 = chance; single-feature logistic, leave-one-subject-out)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('Raw per-window statistics vs sleep stage (all channels)')
    print('=' * 70)

    dfs = []
    for idx in range(len(SESSION_META)):
        r = extract_session(idx)
        if r is not None:
            dfs.append(r)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(REPORT_DIR / 'raw_stats_epochs.csv', index=False)
    print(f'\nPooled {len(df)} epochs, {df["subject"].nunique()} subjects, '
          f'{len(STAT_KEYS)} stats x {len(CHANNELS)} channels')

    # ── Kruskal-Wallis per (stat, channel) ──
    kw_rows = []
    for ch in CHANNELS:
        for stat in STAT_KEYS:
            feat = f'{stat}_{ch}_z'
            H, p, meds = stage_stats(df, feat)
            order = ' < '.join(sorted(meds, key=meds.get))
            kw_rows.append({'stat': stat, 'channel': ch, 'feature': feat,
                            'KW_H': H, 'KW_p': p, 'order_low_to_high': order})
    kw = pd.DataFrame(kw_rows)
    kw.to_csv(REPORT_DIR / 'raw_stats_kw.csv', index=False)

    # ── Per-subject direction consistency + LOSO AUC ──
    d = df[df['stage_code'] >= 0]
    subs = sorted(d['subject'].unique())
    dir_rows, loso_rows = [], []
    auc_grids = {c: np.full((len(STAT_KEYS), len(CHANNELS)), np.nan) for c in CONTRASTS}
    for ci, ch in enumerate(CHANNELS):
        for si, stat in enumerate(STAT_KEYS):
            feat = f'{stat}_{ch}_z'
            for cname, cfun in CONTRASTS.items():
                signs = []
                for sub in subs:
                    ds = d[d['subject'] == sub]
                    auc, sign = contrast_auc(ds, feat, cfun(ds))
                    signs.append(sign)
                highs, lows = signs.count('high'), signs.count('low')
                consistent = 0 in (highs, lows) and (highs + lows) >= 5
                lauc = loso_auc(df, feat, cfun)
                auc_grids[cname][si, ci] = lauc
                dir_rows.append({'stat': stat, 'channel': ch, 'contrast': cname,
                                 'high': highs, 'low': lows,
                                 'consistent': consistent, 'loso_auc': lauc})
                loso_rows.append({'stat': stat, 'channel': ch,
                                  'contrast': cname, 'loso_auc': lauc})
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(REPORT_DIR / 'raw_stats_subject_direction.csv', index=False)
    pd.DataFrame(loso_rows).to_csv(REPORT_DIR / 'raw_stats_loso_auc.csv', index=False)

    # ── Report: strongest features ──
    print('\nTop LOSO AUC (|AUC-0.5|), any contrast:')
    dd = dir_df.copy()
    dd['dist'] = (dd['loso_auc'] - 0.5).abs()
    for _, r in dd.sort_values('dist', ascending=False).head(12).iterrows():
        flag = '  <-- direction-consistent' if r['consistent'] else ''
        print(f'  {r["stat"]:11s} {r["channel"]:8s} {r["contrast"]:14s} '
              f'AUC={r["loso_auc"]:.3f}{flag}')

    print('\nDirection-consistent AND AUC>=0.60 (candidate real markers):')
    cand = dir_df[(dir_df['consistent']) & (dir_df['loso_auc'] >= 0.60)]
    if cand.empty:
        print('  (none)')
    else:
        for _, r in cand.iterrows():
            print(f'  {r["stat"]:11s} {r["channel"]:8s} {r["contrast"]:14s} '
                  f'AUC={r["loso_auc"]:.3f}  ({r["high"]}/{r["low"]})')

    # ── Figures ──
    print('\nWriting figures...')
    for cname in CONTRASTS:
        fig_loso_grid(auc_grids[cname], cname, PLOT_DIR / f'loso_grid_{cname}.png')
        print(f'  loso_grid_{cname}.png')
    # detailed figures for the two user-requested stats on the primary channel
    for stat in ['std', 'slope_vlf']:
        feat = f'{stat}_CLE-CRE_z'
        fig_boxplot(df, feat, PLOT_DIR / f'boxplot_{stat}_CLE_CRE.png')
        fig_subject_direction(df, feat, PLOT_DIR / f'subject_direction_{stat}_CLE_CRE.png')
        print(f'  boxplot_{stat}_CLE_CRE.png + subject_direction_{stat}_CLE_CRE.png')

    print('\nDone. Reports -> reports/mean_value/  Figures -> notebooks/plots/mean_value/')


if __name__ == '__main__':
    main()
