#!/usr/bin/env python
"""
Cardiac tracking diagnostic — does within-session HR variation actually track,
once GT R-peak noise is controlled? Gates the 'push for real tracking' work.

Cache-only (artifacts/mask_phase_a.parquet). For cardiac, per channel + simple
fusions, computes WITHIN-session Pearson r vs GT under three noise treatments:
  (1) raw peaks vs raw GT          (what we reported: ~0)
  (2) raw peaks vs SMOOTHED GT     (removes GT R-peak detection spikes)
  (3) smoothed peaks vs smoothed GT (removes estimator jitter too)

If (2)/(3) >> (1): tracking EXISTS, the ~0 was a metric/noise artifact -> build
the fusion tracker. If still ~0: the signal genuinely doesn't carry instantaneous
HR -> pivot. Also reports the per-epoch oracle's within-session r as an (optimistic)
upper bound, and how much trackable variance survives smoothing.

Output: reports/rates/mask/cardiac_tracking_diagnostic.csv
        writeup/figures/mask_rate_detection/fig12_cardiac_tracking.png
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
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})


def per_session_k(raw, gt):
    v = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
    if v.sum() < 20:
        return 1.0
    r = raw[v] / gt[v]
    r = r[(r > 0.3) & (r < 5.0)]
    return float(np.median(r)) if len(r) >= 10 else 1.0


def roll_med(x, k=5):
    """Centered rolling median ignoring NaN."""
    out = np.full_like(x, np.nan, dtype=float)
    h = k // 2
    for i in range(len(x)):
        seg = x[max(0, i - h):i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def wcorr(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 20 or np.std(a[v]) < 1e-9 or np.std(b[v]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[v], b[v])[0, 1])


def main():
    df = pd.read_parquet(ART / 'mask_phase_a.parquet')
    c = df[df.band == 'card'].copy()

    # Build per (session, epoch) x channel matrix of k-scaled peaks rate + GT
    rows = []
    # also collect within-session r per channel/treatment
    agg = {ch: {'r_raw': [], 'r_gtsmooth': [], 'r_bothsmooth': []} for ch in CHANNELS}
    agg['mean_fuse'] = {'r_raw': [], 'r_gtsmooth': [], 'r_bothsmooth': []}
    agg['oracle'] = {'r_raw': [], 'r_gtsmooth': [], 'r_bothsmooth': []}
    var_retained = []  # fraction of GT variance surviving smoothing

    for sess, g in c.groupby('session'):
        # pivot channels
        piv = g.pivot_table(index='epoch', columns='channel',
                            values='r_peaks_loose')
        gt = g.groupby('epoch')['gt_hz'].first().reindex(piv.index).values
        if np.isfinite(gt).sum() < 30:
            continue
        gt_s = roll_med(gt, 5)
        # variance retained after smoothing (how much real HR variation vs noise)
        vv = np.isfinite(gt) & np.isfinite(gt_s)
        if vv.sum() > 20 and np.var(gt[vv]) > 0:
            var_retained.append(np.var(gt_s[vv]) / np.var(gt[vv]))

        scaled = {}
        for ch in CHANNELS:
            if ch not in piv:
                continue
            raw = piv[ch].values
            k = per_session_k(raw, gt)
            sc = raw / k
            scaled[ch] = sc
            sc_s = roll_med(sc, 5)
            agg[ch]['r_raw'].append(wcorr(sc, gt))
            agg[ch]['r_gtsmooth'].append(wcorr(sc, gt_s))
            agg[ch]['r_bothsmooth'].append(wcorr(sc_s, gt_s))

        # mean fusion across available channels
        mat = np.vstack([scaled[ch] for ch in scaled])
        mfuse = np.nanmean(mat, axis=0)
        mfuse_s = roll_med(mfuse, 5)
        agg['mean_fuse']['r_raw'].append(wcorr(mfuse, gt))
        agg['mean_fuse']['r_gtsmooth'].append(wcorr(mfuse, gt_s))
        agg['mean_fuse']['r_bothsmooth'].append(wcorr(mfuse_s, gt_s))

        # oracle: per-epoch channel closest to GT (optimistic upper bound)
        err = np.abs(mat - gt[None, :])
        with np.errstate(invalid='ignore'):
            bj = np.nanargmin(np.where(np.isfinite(err), err, np.inf), axis=0)
        oracle = mat[bj, np.arange(mat.shape[1])]
        oracle_s = roll_med(oracle, 5)
        agg['oracle']['r_raw'].append(wcorr(oracle, gt))
        agg['oracle']['r_gtsmooth'].append(wcorr(oracle, gt_s))
        agg['oracle']['r_bothsmooth'].append(wcorr(oracle_s, gt_s))

    # ── Extra gates: per-session best FIXED channel + oracle inflation check ──
    best_fixed = []      # per session: max over channels of within-session r (smooth)
    best_fixed_ch = []
    oracle_shuf = []     # oracle r against temporally-shuffled GT (inflation baseline)
    rng = np.random.default_rng(0)
    for sess, g in c.groupby('session'):
        piv = g.pivot_table(index='epoch', columns='channel', values='r_peaks_loose')
        gt = g.groupby('epoch')['gt_hz'].first().reindex(piv.index).values
        if np.isfinite(gt).sum() < 30:
            continue
        gt_s = roll_med(gt, 5)
        scaled = {}
        for ch in CHANNELS:
            if ch in piv:
                scaled[ch] = piv[ch].values / per_session_k(piv[ch].values, gt)
        rs = {ch: wcorr(roll_med(scaled[ch], 5), gt_s) for ch in scaled}
        rs = {k: v for k, v in rs.items() if np.isfinite(v)}
        if rs:
            bch = max(rs, key=rs.get)
            best_fixed.append(rs[bch]); best_fixed_ch.append(bch)
        # oracle against shuffled GT (selection-bias floor)
        mat = np.vstack([scaled[ch] for ch in scaled])
        gt_sh = gt.copy(); m = np.isfinite(gt_sh)
        gt_sh[m] = rng.permutation(gt_sh[m])
        err = np.abs(mat - gt_sh[None, :])
        with np.errstate(invalid='ignore'):
            bj = np.nanargmin(np.where(np.isfinite(err), err, np.inf), axis=0)
        orc = mat[bj, np.arange(mat.shape[1])]
        oracle_shuf.append(wcorr(roll_med(orc, 5), roll_med(gt_sh, 5)))

    print(f'GT variance retained after 5-epoch median smoothing: '
          f'{np.nanmean(var_retained)*100:.0f}% '
          f'(=> {100-np.nanmean(var_retained)*100:.0f}% of raw GT variance is '
          f'high-freq noise/jitter)')
    print(f'Best FIXED channel per session (within-session r, both smoothed): '
          f'mean={np.nanmean(best_fixed):+.3f}, '
          f'range=[{np.nanmin(best_fixed):+.3f},{np.nanmax(best_fixed):+.3f}], '
          f'channels picked={dict(pd.Series(best_fixed_ch).value_counts())}')
    print(f'Oracle r vs SHUFFLED GT (selection-bias floor): '
          f'mean={np.nanmean(oracle_shuf):+.3f}  '
          f'(real oracle 0.63 minus this floor = genuine trackable signal)')
    print()
    print(f'{"source":>10s} | within-session r:  raw-GT   smooth-GT  both-smooth')
    print('-' * 60)
    for src in CHANNELS + ['mean_fuse', 'oracle']:
        rr = np.nanmean(agg[src]['r_raw'])
        rg = np.nanmean(agg[src]['r_gtsmooth'])
        rb = np.nanmean(agg[src]['r_bothsmooth'])
        rows.append({'source': src, 'r_raw': rr, 'r_gtsmooth': rg,
                     'r_bothsmooth': rb})
        print(f'{src:>10s} |               {rr:+.3f}    {rg:+.3f}     {rb:+.3f}')

    res = pd.DataFrame(rows)
    res.to_csv(RPT / 'cardiac_tracking_diagnostic.csv', index=False)

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(res))
    w = 0.26
    ax.bar(x - w, res.r_raw, w, label='vs raw GT', color='#E74C3C', alpha=0.85)
    ax.bar(x, res.r_gtsmooth, w, label='vs smoothed GT', color='#3498DB', alpha=0.85)
    ax.bar(x + w, res.r_bothsmooth, w, label='both smoothed', color='#2ECC71', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(res.source)
    ax.set_ylabel('Mean within-session Pearson r with GT')
    ax.axhline(0, color='gray', lw=0.5)
    ax.legend()
    ax.set_title('Cardiac tracking: does within-session HR variation survive '
                 'GT-noise control?\n(if blue/green >> red, tracking exists and '
                 'the ~0 was a noise artifact)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG / 'fig12_cardiac_tracking.png')
    plt.close(fig)
    print('\nFig 12 saved.')


if __name__ == '__main__':
    main()
