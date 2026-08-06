"""
Motion-demarcated imbalance, and a principled cross-session imbalance quantifier.

Two things:

1. DEMARCATE motion-related changes in the mean.
   Work on per-session robust-z imbalance (offset removed).  A "change in the mean"
   is the epoch-to-epoch step  dz = z[i] - z[i-1].  A step is MOTION-LINKED if a
   high-motion epoch sits at i-1, i, or i+1 (top-decile acc flag, +/-1 epoch
   window because the accel spike and the baseline shift are not perfectly
   simultaneous).  We shade motion epochs and mark motion-linked vs clean steps,
   and report what fraction of the total mean-change energy is motion-linked.

2. QUANTIFY imbalance across sessions on the z-scored CLE-CRE.

   Why absolute one-sidedness is NOT the metric on z-data
   -----------------------------------------------------
   Per-session z-scoring removes the DC offset — which is exactly the
   mask-mount instrumental term we do NOT trust.  But it also removes any
   physiological "zero", so "fraction of time imbalance>0" / signed bias are
   undefined on z-data (median z ~ 0 by construction).  Directional imbalance
   is fundamentally unidentifiable from CLE-CRE without an absolute calibration.

   What IS well-defined and comparable across sessions
   ---------------------------------------------------
   (A) DYNAMICS (scale-free, survive z):   computed on motion-gated sleep epochs
        imbalance_reversal_clean  mean-crossings / hour   (low = stuck one-sided)
        imbalance_dwell_max_clean longest one-sided run (min)
        imbalance_ac1             lag-1 autocorr of z-imbalance (stickiness)
   (B) AMPLITUDE (offset-free, physically meaningful, on RAW not z):
        diff_common_ratio    std(vlf_CLE-CRE) / std(vlf_CLE+CRE)
          -> differential swing measured in units of the common-mode swing.
             Dimensionless, offset-free, and — unlike per-session z — actually
             comparable in magnitude across sessions.  This is the recommended
             single imbalance-strength number.
   (C) CLEANLINESS:
        motion_var_frac      fraction of mean-change energy that is motion-linked
          -> a session with high motion_var_frac has "imbalance" that is mostly the
             head moving on the pillow, not physiology.

   RECOMMENDATION (printed): rank sessions by diff_common_ratio computed on
   motion-free sleep epochs, and read it together with motion_var_frac.  Report
   dynamics (reversal/dwell) as the "stuck one-sided vs alternating" axis.  Do
   NOT report absolute directional bias from this sensor.

Outputs
    reports/mean_value/imbalance_motion_quantify.csv
    notebooks/plots/mean_value/imbalance_motion_grid.png
    notebooks/plots/mean_value/imbalance_quantifier_summary.png

Usage:
    .venv/Scripts/python.exe analysis/mean_value/imbalance_motion_quantify.py
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
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_MIN = 0.5
SLEEP_CODES = {0, 1, 2, 3}
ZCLIP = 6.0
STEP_Z = 1.0          # |dz| > 1 robust-sigma per 30 s counts as a "step"
MOTION_PAD = 1        # motion window +/- this many epochs around a step


def robust_z(x):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    med = np.median(x[good])
    mad = np.median(np.abs(x[good] - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def _run_lengths(sign_arr):
    runs, cur, n = [], 0, 0
    for s in sign_arr:
        if s == cur:
            n += 1
            continue
        if cur != 0:
            runs.append(n)
        cur, n = s, 1
    if cur != 0:
        runs.append(n)
    return runs


def sleep_window(codes):
    is_sleep = np.isin(codes, list(SLEEP_CODES))
    if is_sleep.sum() < 20:
        return None
    on = int(np.argmax(is_sleep))
    off = len(is_sleep) - 1 - int(np.argmax(is_sleep[::-1]))
    w = np.zeros(len(codes), bool)
    w[on:off + 1] = True
    return w


def analyse(g):
    g = g.sort_values('epoch').reset_index(drop=True)
    codes = g['stage_code'].to_numpy()
    motion = g['motion'].to_numpy().astype(bool) if 'motion' in g else np.zeros(len(g), bool)
    z = robust_z(g['vlf_CLE-CRE'].to_numpy())
    win = sleep_window(codes)
    if win is None:
        return None, None

    # ── mean-change (step) energy, motion-linked vs clean ──
    dz = np.diff(z)
    # motion window aligned to the step between i-1 and i
    mot_pad = motion.copy()
    for k in range(1, MOTION_PAD + 1):
        mot_pad[k:] |= motion[:-k]
        mot_pad[:-k] |= motion[k:]
    step_motion = mot_pad[1:] | mot_pad[:-1]      # motion near either endpoint
    e_total = float(np.nansum(dz ** 2))
    e_motion = float(np.nansum(dz[step_motion] ** 2))
    motion_var_frac = e_motion / (e_total + 1e-12)

    big = np.abs(dz) > STEP_Z
    steps_motion = int(np.sum(big & step_motion))
    steps_clean = int(np.sum(big & ~step_motion))

    # ── dynamics on motion-free sleep epochs ──
    clean = win & (~motion) & np.isfinite(z)
    zc = z[clean]
    metrics = {'motion_var_frac': motion_var_frac,
               'steps_motion': steps_motion, 'steps_clean': steps_clean,
               'n_clean_epochs': int(clean.sum())}
    if len(zc) >= 20:
        n_hr = len(zc) * EPOCH_MIN / 60.0
        med = np.median(zc)
        s = np.sign(zc - med); s = s[s != 0]
        rev = int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0
        runs = _run_lengths(np.sign(zc - med))
        ac1 = float(np.corrcoef(zc[:-1], zc[1:])[0, 1]) if len(zc) > 3 else np.nan
        metrics.update({
            'imbalance_reversal_clean': rev / n_hr,
            'imbalance_dwell_max_clean': (max(runs) if runs else 0) * EPOCH_MIN,
            'imbalance_ac1': ac1,
        })
    else:
        metrics.update({'imbalance_reversal_clean': np.nan,
                        'imbalance_dwell_max_clean': np.nan, 'imbalance_ac1': np.nan})

    # ── amplitude: differential-to-common std ratio on motion-free sleep (RAW) ──
    d = g['vlf_CLE-CRE'].to_numpy()
    c = g['vlf_CLE+CRE'].to_numpy()
    m = win & (~motion) & np.isfinite(d) & np.isfinite(c)
    if m.sum() >= 20:
        metrics['diff_common_ratio'] = float(np.std(d[m]) / (np.std(c[m]) + 1e-9))
    else:
        metrics['diff_common_ratio'] = np.nan

    trace = {'t': g['t_hr'].to_numpy(), 'z': z, 'codes': codes,
             'motion': motion, 'big': big, 'step_motion': step_motion}
    return metrics, trace


def fig_grid(traces, sessions):
    fig, axes = plt.subplots(6, 2, figsize=(20, 20), sharey=True)
    for ax, lbl in zip(axes.ravel(), sessions):
        tr = traces[lbl]
        t, z = tr['t'], np.clip(tr['z'], -ZCLIP, ZCLIP)
        codes, motion = tr['codes'], tr['motion']
        # faint hypnogram
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                       alpha=0.10, lw=0)
        # motion epochs shaded (grey hatch band)
        for j in np.where(motion)[0]:
            if j < len(t) - 1:
                ax.axvspan(t[j], t[j + 1], color='0.4', alpha=0.28, lw=0, zorder=2)
        ax.plot(t, z, lw=1.0, color='#E67E22', zorder=3)
        ax.axhline(0, color='#2C3E50', ls='--', lw=0.8, zorder=2)
        # step markers at the later endpoint of each big step
        big, sm = tr['big'], tr['step_motion']
        idx = np.where(big)[0] + 1
        is_mot = sm[big]
        if idx.size:
            ax.scatter(t[idx[is_mot]], z[idx[is_mot]], s=22, color='#7f0000',
                       marker='o', zorder=5, label='motion-linked step')
            ax.scatter(t[idx[~is_mot]], z[idx[~is_mot]], s=22, color='#1560bd',
                       marker='D', zorder=5, label='clean step')
        ax.set_ylim(-ZCLIP - 0.5, ZCLIP + 0.5)
        ax.set_title(lbl, fontsize=10, fontweight='bold')
        ax.set_ylabel('imbalance (z)', fontsize=8)
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    handles = [mpatches.Patch(color='0.4', alpha=0.4, label='motion epoch'),
               plt.Line2D([], [], color='#E67E22', label='imbalance (CLE-CRE), z'),
               plt.Line2D([], [], color='#7f0000', marker='o', ls='', label='motion-linked step'),
               plt.Line2D([], [], color='#1560bd', marker='D', ls='', label='clean step')]
    handles += [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
                for k in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=9, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle('Motion-demarcated imbalance (CLE-CRE, z per session) — steps classified '
                 'motion-linked vs clean', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = PLOT_DIR / 'imbalance_motion_grid.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    return out


def fig_summary(sess):
    s = sess.sort_values('diff_common_ratio', ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.get_cmap('RdYlGn_r')(plt.Normalize(0, 1)(s['motion_var_frac']))
    a1.bar(s['session'], s['diff_common_ratio'], color=colors, edgecolor='k', lw=0.5)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, 1)); sm.set_array([])
    plt.colorbar(sm, ax=a1, label='motion_var_frac (red = motion-driven)')
    a1.set_ylabel('diff_common_ratio  =  std(CLE-CRE)/std(CLE+CRE)')
    a1.set_title('Recommended imbalance-strength metric\n(offset-free, motion-gated sleep)',
                 fontsize=11, fontweight='bold')
    plt.setp(a1.get_xticklabels(), rotation=45, ha='right')
    a1.grid(True, alpha=0.15, axis='y')

    a2.scatter(sess['imbalance_reversal_clean'], sess['diff_common_ratio'],
               s=90, c=sess['motion_var_frac'], cmap='RdYlGn_r', vmin=0, vmax=1,
               edgecolor='k', lw=0.5)
    for _, r in sess.iterrows():
        a2.annotate(r['session'], (r['imbalance_reversal_clean'], r['diff_common_ratio']),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    a2.set_xlabel('imbalance_reversal_clean  (crossings/hr; low = stuck one-sided)')
    a2.set_ylabel('diff_common_ratio (amplitude)')
    a2.set_title('Dynamics vs amplitude\n(colour = motion fraction)',
                 fontsize=11, fontweight='bold')
    a2.grid(True, alpha=0.15)
    fig.tight_layout()
    out = PLOT_DIR / 'imbalance_quantifier_summary.png'
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
        m, tr = analyse(g)
        if m is None:
            print(f'  {sess}: skipped'); continue
        rows.append({'session': sess, 'subject': sub, 'night': night, **m})
        traces[sess] = tr
    sess = pd.DataFrame(rows).sort_values('session').reset_index(drop=True)
    sess.to_csv(REPORT_DIR / 'imbalance_motion_quantify.csv', index=False)

    grid = fig_grid(traces, [s for s in sessions if s in traces])
    summ = fig_summary(sess)

    # ── console report ──
    pd.set_option('display.width', 160)
    show = sess[['session', 'subject', 'diff_common_ratio', 'imbalance_reversal_clean',
                 'imbalance_dwell_max_clean', 'imbalance_ac1', 'motion_var_frac',
                 'steps_motion', 'steps_clean']].copy()
    for c in ['diff_common_ratio', 'imbalance_reversal_clean', 'imbalance_dwell_max_clean',
              'imbalance_ac1', 'motion_var_frac']:
        show[c] = show[c].round(3)
    print('=' * 96)
    print('Motion demarcation + imbalance quantifiers (motion-gated, sleep period)')
    print('=' * 96)
    print(show.to_string(index=False))

    print('\nMotion-linked share of all mean-change energy:')
    print(f'  median {sess["motion_var_frac"].median():.1%}  '
          f'(range {sess["motion_var_frac"].min():.1%}-{sess["motion_var_frac"].max():.1%})')
    print('\nRECOMMENDED single imbalance-strength number: diff_common_ratio')
    print('  = std(vlf_CLE-CRE)/std(vlf_CLE+CRE) on motion-free sleep epochs.')
    print('  Offset-free (survives the mask-mount DC term) AND magnitude-comparable')
    print('  across sessions (unlike per-session z, which normalises amplitude away).')
    print('  Read alongside motion_var_frac; discount sessions where it is high.')
    top = sess.sort_values('diff_common_ratio', ascending=False)
    print('\n  Strongest differential imbalance (low motion_var_frac = trustworthy):')
    for _, r in top.head(4).iterrows():
        print(f'    {r["session"]}: ratio={r["diff_common_ratio"]:.2f}  '
              f'motion_frac={r["motion_var_frac"]:.0%}  '
              f'reversal={r["imbalance_reversal_clean"]:.1f}/hr')
    print(f'\nGrid -> {grid}\nSummary -> {summ}')
    print(f'Table -> {REPORT_DIR / "imbalance_motion_quantify.csv"}')


if __name__ == '__main__':
    main()
