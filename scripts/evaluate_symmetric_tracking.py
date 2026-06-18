#!/usr/bin/env python
"""
Symmetric resp + cardiac rate evaluation from cached CAP estimates.

Phase A cache only (artifacts/mask_phase_a.parquet). No raw reprocessing.

Deliverables:
  1. Multichannel × multimethod MAE tables (per-session spread, not just mean)
  2. Detector B: responsive tracker (peaks+hilbert mean-fusion, minimal smooth)
  3. Tracking battery (within-session r, Delta-tracking, transient vs steady,
     temporal-shuffle null) — IDENTICAL for both bands
  4. Two operating points: robust-mean (spectral) vs responsive (Detector B)
  5. Achievable ceiling (resp Flow-vs-RIPSum r≈0.48)

Figures => writeup/figures/mask_rate_detection/fig18-fig22
Data   => reports/rates/mask/symmetric_tracking_*.csv
"""
from __future__ import annotations
import sys, functools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ART = ROOT / 'artifacts'
RPT = ROOT / 'reports' / 'rates' / 'mask'
FIG = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'
RPT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CLE', 'CRE', 'CH', 'avg', 'diff']
METHODS = ['r_spectral', 'r_hilbert', 'r_peaks_loose', 'r_peaks_strict']
BANDS = ['resp', 'card']
N_SHUFFLE = 200

plt.rcParams.update({
    'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.titlesize': 10,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def per_session_k(raw, gt):
    v = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
    if v.sum() < 20:
        return np.nan
    r = raw[v] / gt[v]
    r = r[(r > 0.3) & (r < 5.0)]
    return float(np.median(r)) if len(r) >= 10 else np.nan


def roll_med(x, k=5):
    out = np.full_like(x, np.nan, dtype=float)
    h = k // 2
    for i in range(len(x)):
        seg = x[max(0, i - h):i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def wcorr(a, b, min_n=20):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < min_n:
        return np.nan
    a_, b_ = a[v], b[v]
    if np.std(a_) < 1e-9 or np.std(b_) < 1e-9:
        return np.nan
    return float(np.corrcoef(a_, b_)[0, 1])


def median_ae(est, gt):
    v = np.isfinite(est) & np.isfinite(gt)
    if v.sum() < 10:
        return np.nan
    return float(np.median(np.abs(est[v] - gt[v])))


def delta_corr(est, gt, min_n=20):
    v = np.isfinite(est) & np.isfinite(gt)
    if v.sum() < min_n + 1:
        return np.nan
    e_, g_ = est[v], gt[v]
    de, dg = np.diff(e_), np.diff(g_)
    if np.std(de) < 1e-9 or np.std(dg) < 1e-9:
        return np.nan
    return float(np.corrcoef(de, dg)[0, 1])


def transient_steady_r(est, gt, threshold_frac=0.5, smooth_k=5):
    v = np.isfinite(est) & np.isfinite(gt)
    if v.sum() < 40:
        return np.nan, np.nan
    e_, g_ = est[v], gt[v]
    gs = roll_med(g_, smooth_k)
    dg = np.abs(np.diff(gs))
    dg = np.concatenate([[0], dg])
    thr = np.nanmedian(dg) + threshold_frac * np.nanstd(dg)
    trans = dg > thr
    steady = ~trans
    r_trans = wcorr(e_[trans], g_[trans], min_n=15) if trans.sum() > 15 else np.nan
    r_steady = wcorr(e_[steady], g_[steady], min_n=15) if steady.sum() > 15 else np.nan
    return r_trans, r_steady


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Multichannel × Multimethod MAE (with per-session spread)
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_mae_tables(df):
    print('\n' + '=' * 70)
    print('PHASE 1: Multichannel × Multimethod MAE (with variation)')
    print('=' * 70)

    rows = []
    for band in BANDS:
        bdf = df[df.band == band].copy()
        scale = 60.0  # Hz => per-min
        for method in METHODS:
            for ch in CHANNELS:
                sub = bdf[bdf.channel == ch].copy()
                sess_maes = []
                for sess, g in sub.groupby('session'):
                    raw = g[method].values
                    gt = g['gt_hz'].values
                    k = per_session_k(raw, gt)
                    if np.isnan(k):
                        continue
                    cal = raw / k
                    mae = median_ae(cal * scale, gt * scale)
                    if not np.isnan(mae):
                        sess_maes.append(mae)
                if not sess_maes:
                    continue
                arr = np.array(sess_maes)
                rows.append({
                    'band': band, 'method': method.replace('r_', ''),
                    'channel': ch,
                    'mae_median': float(np.median(arr)),
                    'mae_mean': float(np.mean(arr)),
                    'mae_q25': float(np.percentile(arr, 25)),
                    'mae_q75': float(np.percentile(arr, 75)),
                    'mae_min': float(np.min(arr)),
                    'mae_max': float(np.max(arr)),
                    'n_sessions': len(arr),
                })

    mae_df = pd.DataFrame(rows)
    mae_df.to_csv(RPT / 'symmetric_tracking_mae_table.csv', index=False)
    print(f'  Saved {len(mae_df)} rows to symmetric_tracking_mae_table.csv')

    # Print summary
    for band in BANDS:
        print(f'\n  {band.upper()} — best per method (median MAE across sessions):')
        b = mae_df[mae_df.band == band]
        for method in b['method'].unique():
            m = b[b.method == method]
            best = m.loc[m.mae_median.idxmin()]
            print(f'    {method:15s} / {best.channel:4s}: '
                  f'{best.mae_median:.2f} [{best.mae_q25:.2f}–{best.mae_q75:.2f}]')

    return mae_df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Detector B — responsive tracker
# ═══════════════════════════════════════════════════════════════════════════════

def build_detector_b(df, band):
    """Build Detector B: mean-fusion of peaks_loose + hilbert across channels.

    Returns per-(session, epoch): detB_hz, spectral_hz, gt_hz (all in Hz).
    """
    bdf = df[df.band == band].copy()
    records = []

    for sess, sg in bdf.groupby('session'):
        # Pivot to get per-channel rates
        piv = sg.pivot_table(
            index='epoch', columns='channel',
            values=['r_peaks_loose', 'r_hilbert', 'r_spectral', 'gt_hz'],
            aggfunc='first',
        )
        epochs = piv.index.values
        gt_all = sg.groupby('epoch')['gt_hz'].first().reindex(epochs).values
        t_hr = sg.groupby('epoch')['t_hr'].first().reindex(epochs).values

        # k-calibrate each method × channel, then fuse
        detb_vals = []
        for method in ['r_peaks_loose', 'r_hilbert']:
            for ch in CHANNELS:
                if (method, ch) not in piv.columns:
                    continue
                raw = piv[(method, ch)].values
                k = per_session_k(raw, gt_all)
                if np.isnan(k):
                    continue
                detb_vals.append(raw / k)

        if not detb_vals:
            continue

        # Mean fusion across all method × channel combos
        mat = np.vstack(detb_vals)
        detb_fused = np.nanmean(mat, axis=0)
        # Minimal smoothing: rolling median k=3
        detb_smooth = roll_med(detb_fused, k=3)

        # Spectral baseline (diff channel, k-calibrated)
        spec_raw = piv[('r_spectral', 'diff')].values if ('r_spectral', 'diff') in piv.columns else np.full(len(epochs), np.nan)
        k_spec = per_session_k(spec_raw, gt_all)
        spec_cal = spec_raw / k_spec if not np.isnan(k_spec) else spec_raw

        for i, ep in enumerate(epochs):
            records.append({
                'session': sess, 'epoch': ep, 't_hr': t_hr[i],
                'detB_hz': detb_smooth[i],
                'detB_raw_hz': detb_fused[i],
                'spectral_hz': spec_cal[i],
                'gt_hz': gt_all[i],
            })

    return pd.DataFrame(records)


def phase2_detector_b(df):
    print('\n' + '=' * 70)
    print('PHASE 2: Detector B — responsive tracker (peaks+hilbert mean-fusion)')
    print('=' * 70)

    results = {}
    for band in BANDS:
        detb = build_detector_b(df, band)
        detb.to_parquet(ART / f'detB_{band}.parquet', index=False)
        results[band] = detb
        n_epochs = detb['epoch'].nunique()
        print(f'  {band}: {len(detb)} rows, {n_epochs} epochs, '
              f'{detb["session"].nunique()} sessions')
        # Quick MAE
        scale = 60.0
        v = np.isfinite(detb.detB_hz) & np.isfinite(detb.gt_hz)
        mae_b = np.median(np.abs(detb.loc[v, 'detB_hz'] - detb.loc[v, 'gt_hz'])) * scale
        v2 = np.isfinite(detb.spectral_hz) & np.isfinite(detb.gt_hz)
        mae_s = np.median(np.abs(detb.loc[v2, 'spectral_hz'] - detb.loc[v2, 'gt_hz'])) * scale
        print(f'    Pooled MAE — FWD: {mae_b:.2f}, Spectral: {mae_s:.2f}')

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Tracking battery — identical for both bands
# ═══════════════════════════════════════════════════════════════════════════════

def tracking_battery(detb_df, band):
    """Run the full tracking battery on Detector B output for one band.

    Returns: per-session tracking metrics + shuffle null distribution.
    """
    scale = 60.0
    rng = np.random.default_rng(42)
    session_rows = []
    null_dists = {}  # session -> array of shuffled r values

    for sess, g in detb_df.groupby('session'):
        gt = g['gt_hz'].values * scale
        est_b = g['detB_hz'].values * scale
        est_spec = g['spectral_hz'].values * scale

        gt_s = roll_med(gt, 5)
        est_b_s = roll_med(est_b, 5)
        est_spec_s = roll_med(est_spec, 5)

        # Within-session r
        r_b = wcorr(est_b_s, gt_s)
        r_spec = wcorr(est_spec_s, gt_s)

        # Delta-tracking
        dr_b = delta_corr(est_b_s, gt_s)
        dr_spec = delta_corr(est_spec_s, gt_s)

        # Transient vs steady
        rt_b, rs_b = transient_steady_r(est_b_s, gt_s)
        rt_spec, rs_spec = transient_steady_r(est_spec_s, gt_s)

        # MAE
        mae_b = median_ae(est_b, gt)
        mae_spec = median_ae(est_spec, gt)

        # GT variation stats
        v = np.isfinite(gt_s)
        gt_std = float(np.std(gt_s[v])) if v.sum() > 10 else np.nan
        gt_range = float(np.ptp(gt_s[v])) if v.sum() > 10 else np.nan

        session_rows.append({
            'band': band, 'session': sess,
            'r_detB': r_b, 'r_spectral': r_spec,
            'dr_detB': dr_b, 'dr_spectral': dr_spec,
            'r_transient_detB': rt_b, 'r_steady_detB': rs_b,
            'r_transient_spec': rt_spec, 'r_steady_spec': rs_spec,
            'mae_detB': mae_b, 'mae_spectral': mae_spec,
            'gt_std': gt_std, 'gt_range': gt_range,
            'n_epochs': len(g),
        })

        # Temporal-shuffle null (for Detector B only)
        null_rs = []
        valid_gt = np.isfinite(gt)
        for _ in range(N_SHUFFLE):
            gt_shuf = gt.copy()
            gt_shuf[valid_gt] = rng.permutation(gt_shuf[valid_gt])
            gt_shuf_s = roll_med(gt_shuf, 5)
            null_rs.append(wcorr(est_b_s, gt_shuf_s))
        null_dists[sess] = np.array(null_rs)

    sess_df = pd.DataFrame(session_rows)
    return sess_df, null_dists


def phase3_tracking(detb_results):
    print('\n' + '=' * 70)
    print('PHASE 3: Tracking battery (within-session r, delta, transient, shuffle null)')
    print('=' * 70)

    all_sess = []
    all_nulls = {}
    for band in BANDS:
        sess_df, null_dists = tracking_battery(detb_results[band], band)
        all_sess.append(sess_df)
        all_nulls[band] = null_dists

        # Print summary
        r_med = sess_df['r_detB'].median()
        r_mean = sess_df['r_detB'].mean()
        dr_med = sess_df['dr_detB'].median()

        # Significance: one-sample Wilcoxon that r > 0
        valid_r = sess_df['r_detB'].dropna().values
        if len(valid_r) >= 5:
            try:
                _, p_wilc = stats.wilcoxon(valid_r, alternative='greater')
            except Exception:
                p_wilc = np.nan
        else:
            p_wilc = np.nan

        # Shuffle null: fraction of sessions where real r > 95th percentile of null
        n_pass = 0
        for sess in sess_df['session']:
            real_r = sess_df.loc[sess_df.session == sess, 'r_detB'].values[0]
            null_arr = null_dists.get(sess, np.array([]))
            if len(null_arr) > 0 and np.isfinite(real_r):
                if real_r > np.percentile(null_arr[np.isfinite(null_arr)], 95):
                    n_pass += 1

        null_means = [np.nanmean(null_dists[s]) for s in null_dists]
        null_95s = [np.nanpercentile(null_dists[s][np.isfinite(null_dists[s])], 95)
                    for s in null_dists if np.isfinite(null_dists[s]).sum() > 0]

        print(f'\n  {band.upper()}:')
        print(f'    Within-session r (FWD):  median={r_med:+.3f}, '
              f'mean={r_mean:+.3f}')
        print(f'    Delta-tracking r:           median={dr_med:+.3f}')
        print(f'    Transient r (FWD):       median='
              f'{sess_df["r_transient_detB"].median():+.3f}')
        print(f'    Steady r (FWD):          median='
              f'{sess_df["r_steady_detB"].median():+.3f}')
        print(f'    Wilcoxon p (r > 0):       {p_wilc:.4f}')
        print(f'    Shuffle null:             mean null r={np.nanmean(null_means):+.3f}, '
              f'95th={np.nanmean(null_95s):+.3f}')
        print(f'    Sessions passing null:    {n_pass}/{len(sess_df)}')
        print(f'    Spectral r (baseline):    median='
              f'{sess_df["r_spectral"].median():+.3f}')

        # PASS/FAIL verdict
        if p_wilc < 0.05 and n_pass >= len(sess_df) // 2:
            print(f'    => VERDICT: PASS (weak tracking signal present)')
        else:
            print(f'    => VERDICT: FAIL (no significant tracking)')

    tracking_df = pd.concat(all_sess, ignore_index=True)
    tracking_df.to_csv(RPT / 'symmetric_tracking_battery.csv', index=False)
    print(f'\n  Saved tracking battery to symmetric_tracking_battery.csv')
    return tracking_df, all_nulls


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Achievable ceiling
# ═══════════════════════════════════════════════════════════════════════════════

def phase4_ceiling():
    print('\n' + '=' * 70)
    print('PHASE 4: Achievable ceiling — resp GT inter-signal agreement')
    print('=' * 70)

    gt_path = ART / 'consolidated_resp_gt.parquet'
    if not gt_path.exists():
        print('  Consensus GT not found — skipping ceiling analysis')
        return None

    gt = pd.read_parquet(gt_path)
    scale = 60.0
    rows = []
    for sess, g in gt.groupby('session'):
        flow = g['rate_flow'].values * scale
        ripsum = g['rate_ripsum'].values * scale
        consensus = g['rate_consensus'].values * scale

        flow_s = roll_med(flow, 5)
        rip_s = roll_med(ripsum, 5)
        cons_s = roll_med(consensus, 5)

        r_fr = wcorr(flow_s, rip_s)
        dr_fr = delta_corr(flow_s, rip_s)

        r_fc = wcorr(flow_s, cons_s)
        r_rc = wcorr(rip_s, cons_s)

        rows.append({
            'session': sess,
            'r_flow_ripsum': r_fr,
            'dr_flow_ripsum': dr_fr,
            'r_flow_consensus': r_fc,
            'r_ripsum_consensus': r_rc,
            'flow_std': float(np.nanstd(flow_s)),
            'ripsum_std': float(np.nanstd(rip_s)),
        })

    ceil_df = pd.DataFrame(rows)
    ceil_df.to_csv(RPT / 'symmetric_tracking_ceiling.csv', index=False)

    r_med = ceil_df['r_flow_ripsum'].median()
    dr_med = ceil_df['dr_flow_ripsum'].median()
    print(f'  Flow vs RIPSum (independent PSG sensors):')
    print(f'    Within-session r:   median={r_med:+.3f}, '
          f'range=[{ceil_df["r_flow_ripsum"].min():+.3f}, '
          f'{ceil_df["r_flow_ripsum"].max():+.3f}]')
    print(f'    Delta-tracking r:   median={dr_med:+.3f}')
    print(f'  => This is the CEILING for any resp sensor on 30s windows')

    return ceil_df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Figures
# ═══════════════════════════════════════════════════════════════════════════════

def fig18_mae_heatmap(mae_df):
    """Multichannel × multimethod MAE heatmap, resp + cardiac side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, band in enumerate(BANDS):
        ax = axes[idx]
        b = mae_df[mae_df.band == band].copy()
        methods = sorted(b['method'].unique())
        channels = CHANNELS

        mat = np.full((len(methods), len(channels)), np.nan)
        iqr_text = np.full((len(methods), len(channels)), '', dtype=object)
        for i, m in enumerate(methods):
            for j, ch in enumerate(channels):
                row = b[(b.method == m) & (b.channel == ch)]
                if len(row):
                    mat[i, j] = row.iloc[0]['mae_median']
                    iqr_text[i, j] = (f'{row.iloc[0]["mae_median"]:.1f}\n'
                                      f'[{row.iloc[0]["mae_q25"]:.1f}–'
                                      f'{row.iloc[0]["mae_q75"]:.1f}]')

        vmax = np.nanpercentile(mat, 90) if not np.all(np.isnan(mat)) else 10
        im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                       vmin=0, vmax=vmax)
        for i in range(len(methods)):
            for j in range(len(channels)):
                if iqr_text[i, j]:
                    ax.text(j, i, iqr_text[i, j], ha='center', va='center',
                            fontsize=7, color='black')

        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(channels)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        unit = 'br/min' if band == 'resp' else 'BPM'
        ax.set_title(f'{band.upper()} — Median MAE ({unit})\n[IQR across 12 sessions]',
                     fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label=f'MAE ({unit})')

    plt.tight_layout()
    fig.savefig(FIG / 'fig18_mae_heatmap.png')
    plt.close(fig)
    print('  fig18 saved: MAE heatmap')


def fig19_tracking_r_bars(tracking_df, null_dists):
    """Per-session tracking r bar chart with shuffle-null band, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, band in enumerate(BANDS):
        ax = axes[idx]
        b = tracking_df[tracking_df.band == band].sort_values('session')
        sessions = b['session'].values
        x = np.arange(len(sessions))

        # Real r
        r_vals = b['r_detB'].values
        ax.bar(x - 0.15, r_vals, 0.3, label='Fused Window Detection',
               color='#2ECC71', alpha=0.85, zorder=3)
        ax.bar(x + 0.15, b['r_spectral'].values, 0.3, label='Spectral',
               color='#E74C3C', alpha=0.85, zorder=3)

        # Shuffle null band (per-session 5th–95th percentile)
        for i, sess in enumerate(sessions):
            nd = null_dists[band].get(sess, np.array([]))
            nd = nd[np.isfinite(nd)]
            if len(nd) > 0:
                lo, hi = np.percentile(nd, [5, 95])
                ax.fill_between([i - 0.4, i + 0.4], lo, hi,
                                color='gray', alpha=0.2, zorder=1)
                ax.plot([i - 0.4, i + 0.4], [np.median(nd)] * 2,
                        color='gray', lw=0.7, ls='--', zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Within-session Pearson r (smoothed)')
        ax.set_title(f'{band.upper()} — Tracking r per session\n'
                     f'(gray band = shuffle null 5th–95th %ile)',
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.5, 0.8)

    plt.tight_layout()
    fig.savefig(FIG / 'fig19_tracking_r_bars.png')
    plt.close(fig)
    print('  fig19 saved: tracking r bars')


def fig20_delta_transient(tracking_df):
    """Delta-tracking and transient vs steady analysis, side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for col, band in enumerate(BANDS):
        b = tracking_df[tracking_df.band == band].sort_values('session')

        # Row 0: Delta-tracking
        ax = axes[0, col]
        sessions = b['session'].values
        x = np.arange(len(sessions))
        ax.bar(x - 0.15, b['dr_detB'].values, 0.3, label='FWD dr',
               color='#3498DB', alpha=0.85)
        ax.bar(x + 0.15, b['dr_spectral'].values, 0.3, label='Spectral dr',
               color='#E74C3C', alpha=0.85)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Delta-tracking r')
        ax.set_title(f'{band.upper()} — Delta-tracking (corr of epoch-to-epoch changes)',
                     fontweight='bold')
        ax.legend(fontsize=7)

        # Row 1: Transient vs steady
        ax = axes[1, col]
        w = 0.2
        ax.bar(x - w, b['r_transient_detB'].values, w * 2, label='FWD transient',
               color='#E67E22', alpha=0.85)
        ax.bar(x + w, b['r_steady_detB'].values, w * 2, label='FWD steady',
               color='#9B59B6', alpha=0.85)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Within-session r')
        ax.set_title(f'{band.upper()} — Transient (|dGT|>thr) vs Steady segments',
                     fontweight='bold')
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG / 'fig20_delta_transient.png')
    plt.close(fig)
    print('  fig20 saved: Delta-tracking + transient/steady')


def fig21_operating_points(tracking_df):
    """Two operating points: MAE vs tracking r, per band."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, band in enumerate(BANDS):
        ax = axes[idx]
        b = tracking_df[tracking_df.band == band].copy()
        # Resp spectral r is NaN (constant predictor) — fill with 0.0
        b['r_spectral'] = b['r_spectral'].fillna(0.0)

        # Each session is a point, colored by detector type
        for _, row in b.iterrows():
            ax.scatter(row['mae_detB'], row['r_detB'],
                       c='#2ECC71', s=60, zorder=3, alpha=0.8,
                       edgecolors='black', linewidth=0.5)
            if pd.notna(row['mae_spectral']):
                ax.scatter(row['mae_spectral'], row['r_spectral'],
                           c='#E74C3C', s=60, zorder=4, alpha=0.8,
                           edgecolors='black', linewidth=0.5)
            ax.annotate(row['session'],
                        (row['mae_detB'], row['r_detB']),
                        fontsize=5, ha='left', va='bottom', color='#27AE60')

        # Summary markers (medians)
        ax.scatter(b['mae_detB'].median(), b['r_detB'].median(),
                   c='#2ECC71', s=200, marker='*', zorder=5,
                   edgecolors='black', linewidth=1.5, label='FWD (median)')
        spec_mae = b['mae_spectral'].dropna()
        spec_r = b.loc[spec_mae.index, 'r_spectral']
        ax.scatter(spec_mae.median(), spec_r.median(),
                   c='#E74C3C', s=200, marker='*', zorder=5,
                   edgecolors='black', linewidth=1.5, label='Spectral (median)')

        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlabel('Median MAE (per-min)')
        ax.set_ylabel('Within-session r')
        unit = 'br/min' if band == 'resp' else 'BPM'
        ax.set_title(f'{band.upper()} — Two operating points\n'
                     f'(MAE in {unit} vs tracking r)', fontweight='bold')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG / 'fig21_operating_points.png')
    plt.close(fig)
    print('  fig21 saved: two operating points')


def fig22_fullnight_traces(detb_results):
    """Full-night traces: GT vs DetB vs spectral for 4 representative sessions."""
    picks = {
        'resp': ['S1N1', 'S3N2', 'S5N1', 'S4N2'],
        'card': ['S1N1', 'S2N1', 'S4N1', 'S6N1'],
    }
    scale = 60.0

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=False)

    for col, band in enumerate(BANDS):
        detb = detb_results[band]
        for row_idx, sess in enumerate(picks[band]):
            ax = axes[row_idx, col]
            g = detb[detb.session == sess].sort_values('t_hr')
            if len(g) == 0:
                ax.text(0.5, 0.5, f'{sess} not found', transform=ax.transAxes)
                continue

            t = g['t_hr'].values
            gt = g['gt_hz'].values * scale
            db = g['detB_hz'].values * scale
            sp = g['spectral_hz'].values * scale

            ax.plot(t, gt, 'k-', lw=0.8, alpha=0.6, label='GT')
            ax.plot(t, db, '-', color='#2ECC71', lw=0.7, alpha=0.8, label='FWD')
            ax.plot(t, sp, '-', color='#E74C3C', lw=0.5, alpha=0.6, label='Spectral')

            unit = 'br/min' if band == 'resp' else 'BPM'
            ax.set_ylabel(f'{unit}')
            ax.set_title(f'{band.upper()} — {sess}', fontsize=9, fontweight='bold')
            if row_idx == 0:
                ax.legend(fontsize=7, loc='upper right')
            if row_idx == 3:
                ax.set_xlabel('Time (hours)')

    plt.tight_layout()
    fig.savefig(FIG / 'fig22_fullnight_traces.png')
    plt.close(fig)
    print('  fig22 saved: full-night traces')


def fig23_ceiling(ceil_df, tracking_df):
    """Achievable ceiling: mask tracking r vs gold-standard inter-sensor r."""
    if ceil_df is None:
        print('  fig23 skipped (no ceiling data)')
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    sessions = sorted(ceil_df['session'].unique())
    x = np.arange(len(sessions))

    # Gold standard ceiling (Flow vs RIPSum)
    ceil_vals = []
    for s in sessions:
        r = ceil_df.loc[ceil_df.session == s, 'r_flow_ripsum']
        ceil_vals.append(r.values[0] if len(r) else np.nan)
    ax.bar(x - 0.25, ceil_vals, 0.25, label='Ceiling (Flow vsRIPSum)',
           color='#F39C12', alpha=0.85)

    # Mask resp DetB
    resp_t = tracking_df[tracking_df.band == 'resp']
    resp_vals = []
    for s in sessions:
        r = resp_t.loc[resp_t.session == s, 'r_detB']
        resp_vals.append(r.values[0] if len(r) else np.nan)
    ax.bar(x, resp_vals, 0.25, label='Mask resp (FWD)',
           color='#2ECC71', alpha=0.85)

    # Mask cardiac DetB
    card_t = tracking_df[tracking_df.band == 'card']
    card_vals = []
    for s in sessions:
        r = card_t.loc[card_t.session == s, 'r_detB']
        card_vals.append(r.values[0] if len(r) else np.nan)
    ax.bar(x + 0.25, card_vals, 0.25, label='Mask cardiac (FWD)',
           color='#3498DB', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha='right')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel('Within-session Pearson r')
    ax.set_title('Tracking capacity: mask vs achievable ceiling\n'
                 '(Flow vsRIPSum = best-case resp agreement between independent PSG sensors)',
                 fontweight='bold')
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG / 'fig23_ceiling_comparison.png')
    plt.close(fig)
    print('  fig23 saved: ceiling comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('Loading mask_phase_a.parquet...')
    df = pd.read_parquet(ART / 'mask_phase_a.parquet')
    print(f'  {len(df)} rows, {df["session"].nunique()} sessions')

    mae_df = phase1_mae_tables(df)
    detb_results = phase2_detector_b(df)
    tracking_df, null_dists = phase3_tracking(detb_results)
    ceil_df = phase4_ceiling()

    print('\n' + '=' * 70)
    print('PHASE 5: Figures')
    print('=' * 70)
    fig18_mae_heatmap(mae_df)
    fig19_tracking_r_bars(tracking_df, null_dists)
    fig20_delta_transient(tracking_df)
    fig21_operating_points(tracking_df)
    fig22_fullnight_traces(detb_results)
    fig23_ceiling(ceil_df, tracking_df)

    # Final summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    for band in BANDS:
        b = tracking_df[tracking_df.band == band]
        print(f'\n  {band.upper()}:')
        print(f'    FWD MAE (median across sessions): '
              f'{b["mae_detB"].median():.2f}')
        print(f'    Spectral MAE:                      '
              f'{b["mae_spectral"].median():.2f}')
        print(f'    FWD within-session r:              '
              f'{b["r_detB"].median():+.3f} '
              f'[{b["r_detB"].min():+.3f}, {b["r_detB"].max():+.3f}]')
        print(f'    FWD Delta-tracking r:                  '
              f'{b["dr_detB"].median():+.3f}')
        n_pass = 0
        for _, row in b.iterrows():
            nd = null_dists[band].get(row['session'], np.array([]))
            nd = nd[np.isfinite(nd)]
            if len(nd) > 0 and np.isfinite(row['r_detB']):
                if row['r_detB'] > np.percentile(nd, 95):
                    n_pass += 1
        print(f'    Sessions > shuffle null 95th:       '
              f'{n_pass}/{len(b)}')

    if ceil_df is not None:
        print(f'\n  RESP CEILING (Flow vsRIPSum): '
              f'median r = {ceil_df["r_flow_ripsum"].median():+.3f}')

    print('\nDone. Figures in writeup/figures/mask_rate_detection/fig18-23.')
    print('Data in reports/rates/mask/symmetric_tracking_*.csv')


if __name__ == '__main__':
    main()
