#!/usr/bin/env python
"""
Adaptive-k + Oracle channel analysis — operates ENTIRELY on cached Phase A
(artifacts/mask_phase_a.parquet). No raw-signal reprocessing.

Answers two open questions raised after the first mask-rate pipeline:

  Q1. Is spectral winning for resp only because it is effectively k-free?
      -> Test SELF-SUPERVISED adaptive k(t): anchor peaks to spectral
         within session (no GT). If adaptive-k peaks matches/beats spectral,
         the static-k comparison was unfair.

  Q2. Why does the single 'diff' channel win when earlier validation showed
      channels carry DIFFERENT information and an oracle-over-channels wins?
      -> Recompute per-epoch oracle (channel diversity headroom) and diagnose
         why SQI-weighted fusion fails to capture it.

Outputs (checkpoints for the next session):
  reports/rates/mask/adaptive_k_results.csv
  reports/rates/mask/oracle_headroom.csv
  reports/rates/mask/channel_win_distribution.csv
  writeup/figures/mask_rate_detection/fig7_oracle_headroom.png
  writeup/figures/mask_rate_detection/fig8_adaptive_k.png
  writeup/figures/mask_rate_detection/fig9_channel_diversity.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ART = ROOT / 'artifacts'
RPT = ROOT / 'reports' / 'rates' / 'mask'
FIG = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'
RPT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CLE', 'CRE', 'CH', 'avg', 'diff']
METHODS = {'spectral': 'r_spectral', 'hilbert': 'r_hilbert',
           'peaks_loose': 'r_peaks_loose', 'peaks_strict': 'r_peaks_strict'}
STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})


def per_session_k(raw, gt):
    """Static per-session k = median(raw/gt) over valid epochs (GT-based)."""
    v = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
    if v.sum() < 20:
        return 1.0
    r = raw[v] / gt[v]
    r = r[(r > 0.3) & (r < 5.0)]
    return float(np.median(r)) if len(r) >= 10 else 1.0


def causal_rolling_median(x, win):
    """Causal rolling median ignoring NaNs (trailing window of length win)."""
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        seg = x[max(0, i - win + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) > 0:
            out[i] = np.median(seg)
    return out


def mae_bpm(pred, gt):
    v = np.isfinite(pred) & np.isfinite(gt)
    if v.sum() < 10:
        return np.nan, 0
    return float(np.median(np.abs(pred[v] - gt[v])) * 60), int(v.sum())


def main():
    df = pd.read_parquet(ART / 'mask_phase_a.parquet')
    print(f'Loaded cache: {df.shape}')

    # ════════════════════════════════════════════════════════════════════
    # Q2a — Per-(channel,method) static-k MAE table  (which single combo best)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('Q2a — Single (channel x method) leaderboard, per-session k applied')
    print('=' * 70)

    rows = []
    for band in ['resp', 'card']:
        unit = 'br/min' if band == 'resp' else 'BPM'
        bsub = df[df.band == band]
        for mname, mcol in METHODS.items():
            for ch in CHANNELS:
                preds, gts = [], []
                for sess, g in bsub[bsub.channel == ch].groupby('session'):
                    g = g.sort_values('epoch')
                    raw = g[mcol].values
                    gt = g['gt_hz'].values
                    k = per_session_k(raw, gt)
                    preds.append(raw / k)
                    gts.append(gt)
                preds = np.concatenate(preds)
                gts = np.concatenate(gts)
                mae, n = mae_bpm(preds, gts)
                rows.append({'band': band, 'method': mname, 'channel': ch,
                             'MAE': mae, 'n': n})
    leaderboard = pd.DataFrame(rows)
    for band in ['resp', 'card']:
        unit = 'br/min' if band == 'resp' else 'BPM'
        top = leaderboard[leaderboard.band == band].nsmallest(8, 'MAE')
        print(f'\n  {band.upper()} top 8 ({unit}):')
        print(top.to_string(index=False))
    leaderboard.to_csv(RPT / 'single_combo_leaderboard.csv', index=False)

    # ════════════════════════════════════════════════════════════════════
    # Q2b — Oracle headroom (channel diversity, method diversity, full)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('Q2b — Oracle headroom analysis')
    print('=' * 70)

    oracle_rows = []
    for band in ['resp', 'card']:
        unit = 'br/min' if band == 'resp' else 'BPM'
        bsub = df[df.band == band]
        # Pick primary method per band (from main pipeline conclusion)
        prim = 'spectral' if band == 'resp' else 'peaks_loose'
        pcol = METHODS[prim]

        # Build a wide table: per (session, epoch) the k-scaled rate for every
        # channel (primary method) and every method (diff channel).
        # --- channel oracle (primary method, best channel per epoch) ---
        ch_scaled = {}   # channel -> dict (session,epoch)->scaled rate
        gt_map = {}
        for ch in CHANNELS:
            g_all = bsub[bsub.channel == ch]
            for sess, g in g_all.groupby('session'):
                g = g.sort_values('epoch')
                raw = g[pcol].values
                gt = g['gt_hz'].values
                k = per_session_k(raw, gt)
                scaled = raw / k
                for ep, sc, gv in zip(g['epoch'].values, scaled, gt):
                    ch_scaled.setdefault(ch, {})[(sess, ep)] = sc
                    gt_map[(sess, ep)] = gv

        keys = sorted(gt_map.keys())
        gt_arr = np.array([gt_map[k] for k in keys])
        chan_mat = np.full((len(keys), len(CHANNELS)), np.nan)
        for j, ch in enumerate(CHANNELS):
            d = ch_scaled.get(ch, {})
            chan_mat[:, j] = [d.get(k, np.nan) for k in keys]

        # diff-only baseline
        diff_idx = CHANNELS.index('diff')
        mae_diff, _ = mae_bpm(chan_mat[:, diff_idx], gt_arr)
        # mean-fusion across channels
        mae_meanfuse, _ = mae_bpm(np.nanmean(chan_mat, axis=1), gt_arr)
        # oracle channel (closest to GT per epoch)
        err = np.abs(chan_mat - gt_arr[:, None])
        oracle_ch = np.nanmin(err, axis=1) / 60.0  # back to Hz space? no—keep
        # build oracle prediction
        best_j = np.nanargmin(np.where(np.isfinite(err), err, np.inf), axis=1)
        oracle_pred = chan_mat[np.arange(len(keys)), best_j]
        mae_oracle_ch, _ = mae_bpm(oracle_pred, gt_arr)

        # --- method oracle (diff channel, best method per epoch) ---
        gd = bsub[bsub.channel == 'diff']
        meth_scaled = {}
        for mname, mcol in METHODS.items():
            for sess, g in gd.groupby('session'):
                g = g.sort_values('epoch')
                raw = g[mcol].values
                gt = g['gt_hz'].values
                k = per_session_k(raw, gt)
                scaled = raw / k
                for ep, sc in zip(g['epoch'].values, scaled):
                    meth_scaled.setdefault(mname, {})[(sess, ep)] = sc
        meth_mat = np.full((len(keys), len(METHODS)), np.nan)
        for j, mname in enumerate(METHODS):
            d = meth_scaled.get(mname, {})
            meth_mat[:, j] = [d.get(k, np.nan) for k in keys]
        err_m = np.abs(meth_mat - gt_arr[:, None])
        best_jm = np.nanargmin(np.where(np.isfinite(err_m), err_m, np.inf), axis=1)
        oracle_meth_pred = meth_mat[np.arange(len(keys)), best_jm]
        mae_oracle_meth, _ = mae_bpm(oracle_meth_pred, gt_arr)

        # --- full oracle (all channel x method) ---
        full_cols = []
        for ch in CHANNELS:
            for mname, mcol in METHODS.items():
                col = {}
                for sess, g in bsub[bsub.channel == ch].groupby('session'):
                    g = g.sort_values('epoch')
                    raw = g[mcol].values
                    gt = g['gt_hz'].values
                    k = per_session_k(raw, gt)
                    scaled = raw / k
                    for ep, sc in zip(g['epoch'].values, scaled):
                        col[(sess, ep)] = sc
                full_cols.append([col.get(k, np.nan) for k in keys])
        full_mat = np.array(full_cols).T  # (epochs, 20)
        err_f = np.abs(full_mat - gt_arr[:, None])
        best_jf = np.nanargmin(np.where(np.isfinite(err_f), err_f, np.inf), axis=1)
        oracle_full_pred = full_mat[np.arange(len(keys)), best_jf]
        mae_oracle_full, _ = mae_bpm(oracle_full_pred, gt_arr)

        print(f'\n  {band.upper()} ({unit}), primary method = {prim}:')
        print(f'    diff channel only        : {mae_diff:.2f}')
        print(f'    mean-fusion (5 channels) : {mae_meanfuse:.2f}')
        print(f'    ORACLE channel           : {mae_oracle_ch:.2f}  (headroom {mae_diff - mae_oracle_ch:.2f})')
        print(f'    ORACLE method (diff)     : {mae_oracle_meth:.2f}')
        print(f'    ORACLE channel x method  : {mae_oracle_full:.2f}  (headroom {mae_diff - mae_oracle_full:.2f})')

        oracle_rows.append({'band': band, 'diff_only': mae_diff,
                            'mean_fusion': mae_meanfuse,
                            'oracle_channel': mae_oracle_ch,
                            'oracle_method_diff': mae_oracle_meth,
                            'oracle_full': mae_oracle_full})

        # Channel-win distribution (which channel is closest, primary method)
        win_counts = pd.Series(best_j).value_counts().sort_index()
        win_dist = {CHANNELS[i]: int(win_counts.get(i, 0)) for i in range(len(CHANNELS))}
        total = sum(win_dist.values())
        print(f'    Channel-win distribution (primary method, oracle picks):')
        for ch, c in win_dist.items():
            print(f'      {ch:>5s}: {c:5d} ({100*c/total:.1f}%)')

        pd.DataFrame([{'band': band, 'channel': ch, 'wins': c,
                       'pct': 100*c/total} for ch, c in win_dist.items()]
                     ).to_csv(RPT / f'channel_win_{band}.csv', index=False)

    pd.DataFrame(oracle_rows).to_csv(RPT / 'oracle_headroom.csv', index=False)

    # ════════════════════════════════════════════════════════════════════
    # Q1 — Self-supervised adaptive k(t)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('Q1 — Self-supervised adaptive k(t): peaks anchored to spectral')
    print('=' * 70)

    ADAPT_WIN = 12  # epochs (~6 min trailing window)
    adapt_rows = []
    for band in ['resp', 'card']:
        unit = 'br/min' if band == 'resp' else 'BPM'
        bsub = df[(df.band == band) & (df.channel == 'diff')]

        all_spec, all_peaks_static, all_peaks_adapt, all_peaks_first10, all_gt = \
            [], [], [], [], []
        for sess, g in bsub.groupby('session'):
            g = g.sort_values('epoch')
            spec = g['r_spectral'].values
            peaks = g['r_peaks_loose'].values
            gt = g['gt_hz'].values

            # static GT-based k
            k_static = per_session_k(peaks, gt)

            # first-10-min k (realistic, no full-session GT)
            v = np.isfinite(peaks) & np.isfinite(gt) & (gt > 0)
            first = v.copy(); first[20:] = False
            if first.sum() >= 5:
                r = peaks[first] / gt[first]
                r = r[(r > 0.3) & (r < 5.0)]
                k_first10 = float(np.median(r)) if len(r) >= 3 else k_static
            else:
                k_first10 = k_static

            # SELF-SUPERVISED adaptive k(t): ratio peaks/spectral, causal median
            ratio = np.where((np.isfinite(spec)) & (spec > 0) & np.isfinite(peaks),
                             peaks / spec, np.nan)
            ratio = np.where((ratio > 0.3) & (ratio < 5.0), ratio, np.nan)
            k_t = causal_rolling_median(ratio, ADAPT_WIN)
            # backfill leading NaNs with first finite value
            if np.any(np.isfinite(k_t)):
                first_finite = k_t[np.isfinite(k_t)][0]
                k_t = np.where(np.isfinite(k_t), k_t, first_finite)
            else:
                k_t = np.full_like(peaks, k_static)

            all_spec.append(spec)
            all_peaks_static.append(peaks / k_static)
            all_peaks_first10.append(peaks / k_first10)
            all_peaks_adapt.append(peaks / k_t)
            all_gt.append(gt)

        spec = np.concatenate(all_spec)
        ps_static = np.concatenate(all_peaks_static)
        ps_first10 = np.concatenate(all_peaks_first10)
        ps_adapt = np.concatenate(all_peaks_adapt)
        gt = np.concatenate(all_gt)

        m_spec, _ = mae_bpm(spec, gt)
        m_static, _ = mae_bpm(ps_static, gt)
        m_first10, _ = mae_bpm(ps_first10, gt)
        m_adapt, _ = mae_bpm(ps_adapt, gt)

        print(f'\n  {band.upper()} ({unit}), diff channel:')
        print(f'    spectral (k-free)              : {m_spec:.2f}')
        print(f'    peaks / k_static (GT, full)    : {m_static:.2f}')
        print(f'    peaks / k_first10min (GT, 10m) : {m_first10:.2f}')
        print(f'    peaks / k(t) self-sup (no GT)  : {m_adapt:.2f}')

        adapt_rows.append({'band': band, 'spectral': m_spec,
                           'peaks_k_static': m_static,
                           'peaks_k_first10': m_first10,
                           'peaks_k_adaptive_selfsup': m_adapt})

    pd.DataFrame(adapt_rows).to_csv(RPT / 'adaptive_k_results.csv', index=False)

    # ════════════════════════════════════════════════════════════════════
    # Figures
    # ════════════════════════════════════════════════════════════════════
    # Fig 7: oracle headroom
    odf = pd.DataFrame(oracle_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, band in zip(axes, ['resp', 'card']):
        unit = 'br/min' if band == 'resp' else 'BPM'
        r = odf[odf.band == band].iloc[0]
        labels = ['diff\nonly', 'mean\nfusion', 'oracle\nchannel',
                  'oracle\nmethod', 'oracle\nch×meth']
        vals = [r['diff_only'], r['mean_fusion'], r['oracle_channel'],
                r['oracle_method_diff'], r['oracle_full']]
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#27AE60', '#16A085']
        ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02 * max(vals), f'{v:.2f}', ha='center',
                    fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band == "resp" else "Cardiac"}')
    fig.suptitle('Oracle headroom: how much can better fusion gain?',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG / 'fig7_oracle_headroom.png')
    plt.close(fig)
    print('\n  Fig 7 (oracle headroom) saved')

    # Fig 8: adaptive k
    adf = pd.DataFrame(adapt_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, band in zip(axes, ['resp', 'card']):
        unit = 'br/min' if band == 'resp' else 'BPM'
        r = adf[adf.band == band].iloc[0]
        labels = ['spectral\n(k-free)', 'peaks\n/k static', 'peaks\n/k 10min',
                  'peaks\n/k(t) self-sup']
        vals = [r['spectral'], r['peaks_k_static'], r['peaks_k_first10'],
                r['peaks_k_adaptive_selfsup']]
        colors = ['#9B59B6', '#F39C12', '#E67E22', '#2ECC71']
        ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02 * max(vals), f'{v:.2f}', ha='center',
                    fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f'Median MAE ({unit})')
        ax.set_title(f'{"Respiratory" if band == "resp" else "Cardiac"}')
    fig.suptitle('Self-supervised adaptive k(t): does k-free peaks rival spectral?',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG / 'fig8_adaptive_k.png')
    plt.close(fig)
    print('  Fig 8 (adaptive k) saved')

    # Fig 9: channel diversity (win distribution)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, band in zip(axes, ['resp', 'card']):
        wdf = pd.read_csv(RPT / f'channel_win_{band}.csv')
        ax.bar(wdf['channel'], wdf['pct'], color='#3498DB', alpha=0.85)
        for i, v in enumerate(wdf['pct']):
            ax.text(i, v + 0.5, f'{v:.0f}%', ha='center', fontsize=9)
        ax.set_ylabel('% epochs this channel is best (oracle)')
        ax.set_title(f'{"Respiratory" if band == "resp" else "Cardiac"}')
    fig.suptitle('Channel diversity: no single channel dominates oracle picks',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG / 'fig9_channel_diversity.png')
    plt.close(fig)
    print('  Fig 9 (channel diversity) saved')

    print('\nDONE. Checkpoints in reports/rates/mask/, figures in '
          'writeup/figures/mask_rate_detection/')


if __name__ == '__main__':
    main()
