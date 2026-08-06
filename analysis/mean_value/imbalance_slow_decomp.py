"""
Separate SLOW imbalance (drift) from LARGE motion jumps in the differential mean.

Refinement of the hypothesis: the large discontinuous steps in CLE-CRE are the
head re-seating on the pillow (electrode re-coupling) — those are motion, not
imbalance.  The capacitance imbalance is the slow drift that happens *between* those resets, while the
mask is stationary.  So we decompose the sleep-period differential mean into:

  segments   = contiguous non-motion runs (>= MIN_SEG epochs).  The mask is
               stationary within a segment; a motion epoch / big jump ends it.
  WITHIN-segment variance   = slow drift  -> the candidate imbalance
  BETWEEN-segment variance  = the jump/reset ladder -> MOTION

Metrics (per session, on the sleep period)
------------------------------------------
  slow_frac        within/(within+between)  fraction of differential variance
                   that is slow drift (imbalance) rather than motion resets. HIGH = imbalance.
  slow_std         sqrt(within-var)  a.u.  -> physical amplitude of the slow imbalance
  slow_diff_common slow_std(CLE-CRE) / slow_std(CLE+CRE)  offset-free amplitude,
                   comparable across sessions, restricted to the slow band.
  drift_rate       median |within-segment slope|  (a.u./min) -> how fast it drifts
  slow_monotonic   fraction of within-segment drift that is one-signed (does the
                   slow imbalance keep going one way, or wander?)
  n_segments, med_seg_min  segmentation context

De-jumped reconstruction (for plotting): concatenate segments, shifting each so
it continues from the previous segment's end — removes the inter-segment jumps
and leaves a continuous SLOW-imbalance trace.

Outputs
    reports/mean_value/imbalance_slow_decomp.csv
    notebooks/plots/mean_value/imbalance_slow_decomp_grid.png     (raw z vs de-jumped slow)
    notebooks/plots/mean_value/imbalance_slow_summary.png

Usage:
    .venv/Scripts/python.exe analysis/mean_value/imbalance_slow_decomp.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'mean_value'
REPORT_DIR = ROOT / 'reports' / 'mean_value'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_MIN = 0.5
SLEEP_CODES = {0, 1, 2, 3}
MIN_SEG = 6           # >= 3 min of stationary mask to count as a slow segment
JUMP_MAD = 5.0        # within a segment, an increment > this*MAD also splits it
ZCLIP = 6.0


def robust_z_scale(x):
    x = x[np.isfinite(x)]
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return med, 1.4826 * mad


def sleep_window(codes):
    is_sleep = np.isin(codes, list(SLEEP_CODES))
    if is_sleep.sum() < 20:
        return None
    on = int(np.argmax(is_sleep))
    off = len(is_sleep) - 1 - int(np.argmax(is_sleep[::-1]))
    w = np.zeros(len(codes), bool)
    w[on:off + 1] = True
    return w


def segments_of(imbalance, motion, win):
    """Contiguous non-motion index runs within the sleep window, further split at
    large within-run increments (residual jumps motion flag missed)."""
    valid = win & (~motion) & np.isfinite(imbalance)
    # split at large increments too
    segs, cur = [], []
    di = np.diff(imbalance)
    good_di = np.isfinite(di)
    mad = np.median(np.abs(di[good_di] - np.median(di[good_di]))) + 1e-9
    thr = JUMP_MAD * 1.4826 * mad
    for i in range(len(imbalance)):
        if not valid[i]:
            if len(cur) >= MIN_SEG:
                segs.append(np.array(cur))
            cur = []
            continue
        if cur and abs(imbalance[i] - imbalance[cur[-1]]) > thr:
            if len(cur) >= MIN_SEG:
                segs.append(np.array(cur))
            cur = [i]
            continue
        cur.append(i)
    if len(cur) >= MIN_SEG:
        segs.append(np.array(cur))
    return segs


def decomp(g):
    g = g.sort_values('epoch').reset_index(drop=True)
    codes = g['stage_code'].to_numpy()
    motion = g['motion'].to_numpy().astype(bool) if 'motion' in g else np.zeros(len(g), bool)
    imbalance = g['vlf_CLE-CRE'].to_numpy()
    common = g['vlf_CLE+CRE'].to_numpy()
    win = sleep_window(codes)
    if win is None:
        return None, None
    segs = segments_of(imbalance, motion, win)
    if len(segs) < 2:
        return None, None

    def within_between(x):
        seg_means, within_ss, tot_n = [], 0.0, 0
        allvals, weights = [], []
        for s in segs:
            v = x[s]
            v = v[np.isfinite(v)]
            if len(v) < 2:
                continue
            seg_means.append((np.mean(v), len(v)))
            within_ss += np.sum((v - np.mean(v)) ** 2)
            tot_n += len(v)
        if tot_n == 0 or len(seg_means) < 2:
            return np.nan, np.nan
        within_var = within_ss / tot_n
        m = np.array([a for a, _ in seg_means]); w = np.array([n for _, n in seg_means])
        gm = np.average(m, weights=w)
        between_var = np.average((m - gm) ** 2, weights=w)
        return within_var, between_var

    wv_d, bv_d = within_between(imbalance)
    wv_c, bv_c = within_between(common)
    slow_frac = wv_d / (wv_d + bv_d + 1e-12)

    # drift rate + monotonicity within segments
    slopes, mono_pos = [], []
    for s in segs:
        v = imbalance[s]
        tt = np.arange(len(v)) * EPOCH_MIN
        if len(v) >= 3 and np.isfinite(v).all():
            b = np.polyfit(tt, v, 1)[0]
            slopes.append(b)
            # fraction of increments with the segment's dominant sign
            di = np.diff(v)
            if len(di):
                mono_pos.append(np.mean(np.sign(di) == np.sign(np.sum(di))))
    drift_rate = float(np.median(np.abs(slopes))) if slopes else np.nan
    slow_monotonic = float(np.mean(mono_pos)) if mono_pos else np.nan

    metrics = {
        'slow_frac': float(slow_frac),
        'slow_std': float(np.sqrt(wv_d)),
        'jump_std': float(np.sqrt(bv_d)),
        'slow_diff_common': float(np.sqrt(wv_d) / (np.sqrt(wv_c) + 1e-9)),
        'drift_rate': drift_rate,
        'slow_monotonic': slow_monotonic,
        'n_segments': len(segs),
        'med_seg_min': float(np.median([len(s) for s in segs]) * EPOCH_MIN),
    }

    # de-jumped reconstruction
    med, scale = robust_z_scale(imbalance)
    recon = np.full(len(imbalance), np.nan)
    prev_end = None
    for s in segs:
        v = imbalance[s].astype(float)
        if prev_end is not None:
            v = v + (prev_end - v[0])
        recon[s] = v
        prev_end = v[-1]
    trace = {'t': g['t_hr'].to_numpy(),
             'zimb': (imbalance - med) / scale,
             'zrecon': (recon - med) / scale,
             'codes': codes, 'motion': motion}
    return metrics, trace


def fig_grid(traces, sessions):
    fig, axes = plt.subplots(6, 2, figsize=(20, 20), sharey=True)
    for ax, lbl in zip(axes.ravel(), sessions):
        tr = traces[lbl]
        t = tr['t']; codes = tr['codes']; motion = tr['motion']
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                       alpha=0.10, lw=0)
        for j in np.where(motion)[0]:
            if j < len(t) - 1:
                ax.axvspan(t[j], t[j + 1], color='0.5', alpha=0.20, lw=0, zorder=2)
        ax.plot(t, np.clip(tr['zimb'], -ZCLIP, ZCLIP), lw=0.7, color='#f0b27a',
                zorder=3, label='raw imbalance (z)')
        ax.plot(t, np.clip(tr['zrecon'], -ZCLIP, ZCLIP), lw=2.0, color='#B9380B',
                zorder=4, label='de-jumped slow imbalance')
        ax.axhline(0, color='#2C3E50', ls='--', lw=0.8, zorder=2)
        ax.set_ylim(-ZCLIP - 0.5, ZCLIP + 0.5)
        ax.set_title(lbl, fontsize=10, fontweight='bold')
        ax.set_ylabel('imbalance (z)', fontsize=8)
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    handles = [mpatches.Patch(color='0.5', alpha=0.35, label='motion epoch'),
               plt.Line2D([], [], color='#f0b27a', lw=1.5, label='raw imbalance (z)'),
               plt.Line2D([], [], color='#B9380B', lw=2.5, label='de-jumped SLOW imbalance')]
    handles += [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
                for k in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=9, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle('Slow imbalance (drift between motion resets) vs raw jumpy imbalance — '
                 'CLE-CRE, z per session', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = PLOT_DIR / 'imbalance_slow_decomp_grid.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    return out


def fig_summary(sess):
    s = sess.sort_values('slow_frac', ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    a1.bar(s['session'], s['slow_frac'], color='#B9380B', edgecolor='k', lw=0.5)
    a1.axhline(0.5, color='#555', ls='--', lw=1)
    a1.set_ylabel('slow_frac  =  within/(within+between) variance')
    a1.set_title('Fraction of differential variance that is SLOW imbalance\n'
                 '(vs motion jumps);  >0.5 = imbalance-dominated', fontsize=11, fontweight='bold')
    a1.set_ylim(0, 1)
    plt.setp(a1.get_xticklabels(), rotation=45, ha='right')
    a1.grid(True, alpha=0.15, axis='y')

    a2.scatter(sess['slow_diff_common'], sess['slow_frac'], s=90,
               c=sess['drift_rate'], cmap='viridis', edgecolor='k', lw=0.5)
    for _, r in sess.iterrows():
        a2.annotate(r['session'], (r['slow_diff_common'], r['slow_frac']),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    cb = plt.colorbar(a2.collections[0], ax=a2); cb.set_label('drift_rate (a.u./min)')
    a2.axhline(0.5, color='#555', ls='--', lw=1)
    a2.set_xlabel('slow_diff_common  (slow differential / slow common amplitude)')
    a2.set_ylabel('slow_frac')
    a2.set_title('Slow-imbalance amplitude vs slow-imbalance share', fontsize=11, fontweight='bold')
    a2.grid(True, alpha=0.15)
    fig.tight_layout()
    out = PLOT_DIR / 'imbalance_slow_summary.png'
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
    return out


def main():
    csv = REPORT_DIR / 'mean_value_epochs.csv'
    if not csv.exists():
        sys.exit(f'Missing {csv}. Run mean_value_vs_stage.py first.')
    df = pd.read_csv(csv)
    sessions = sorted(df['session'].unique())

    rows, traces = [], {}
    for (sess, sub, night), g in df.groupby(['session', 'subject', 'night']):
        m, tr = decomp(g)
        if m is None:
            print(f'  {sess}: skipped'); continue
        rows.append({'session': sess, 'subject': sub, 'night': night, **m})
        traces[sess] = tr
    sess = pd.DataFrame(rows).sort_values('session').reset_index(drop=True)
    sess.to_csv(REPORT_DIR / 'imbalance_slow_decomp.csv', index=False)

    grid = fig_grid(traces, [s for s in sessions if s in traces])
    summ = fig_summary(sess)

    show = sess[['session', 'subject', 'slow_frac', 'slow_std', 'jump_std',
                 'slow_diff_common', 'drift_rate', 'slow_monotonic',
                 'n_segments', 'med_seg_min']].copy()
    for c in ['slow_frac', 'slow_std', 'jump_std', 'slow_diff_common',
              'drift_rate', 'slow_monotonic', 'med_seg_min']:
        show[c] = show[c].round(3)
    print('=' * 104)
    print('Slow-imbalance decomposition (drift between motion resets) — sleep period')
    print('=' * 104)
    print(show.to_string(index=False))
    print(f'\nslow_frac: median {sess["slow_frac"].median():.2f}  '
          f'(range {sess["slow_frac"].min():.2f}-{sess["slow_frac"].max():.2f})')
    print('  slow_frac > 0.5  => the differential mean is driven more by slow drift '
          '(imbalance) than by motion resets.')
    imbalance_dom = sess[sess['slow_frac'] > 0.5]['session'].tolist()
    print(f'  imbalance-dominated sessions: {imbalance_dom or "none"}')
    print(f'\nGrid -> {grid}\nSummary -> {summ}')
    print(f'Table -> {REPORT_DIR / "imbalance_slow_decomp.csv"}')


if __name__ == '__main__':
    main()
