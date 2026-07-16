"""
Directional "flow" from the L-R mean value, and its link to sleep quality.

Hypothesis (user)
-----------------
The slowly-moving mean (DC baseline) of the capacitive temple sensors tracks a
slow "flow". The LEFT-minus-RIGHT baseline,  flow(t) = base_CLE - base_CRE,
is a *signed direction of flow*. If, across the night, flow stays prominently in
ONE direction (imbalanced / one-sided) rather than alternating, sleep quality is
worse.

Signal
------
flow(t) := vlf_CLE-CRE  (the <0.05 Hz baseline of CLE-CRE, already computed by
mean_value_vs_stage.py and stored per 30 s epoch in mean_value_epochs.csv).
This is the "slowly moving mean value" the hypothesis is about.

IMPORTANT confound — static sensor offset
------------------------------------------
CLE-CRE has an unknown per-session DC offset (electrode gain / coupling
mismatch). So metrics defined relative to *absolute zero* ("how far from 0",
"fraction of time positive") are partly instrumental, not physiological. We
therefore report TWO families of imbalance metrics:

  offset-SENSITIVE (literal reading of the hypothesis; interpret with care)
     flow_bias        median(flow)                    signed, raw units
     flow_onesided    |mean(flow)| / rms(flow) in[0,1]  1 = never crosses abs 0

  offset-INVARIANT (robust test of "stuck one-sided vs alternating")
     flow_skew        skewness of flow                one-sided magnitude asym.
     flow_reversal    sign-changes of (flow-median)/hr  low = stuck, high = alternating
     flow_gini_dwell  dwell asymmetry about session median (Gini of run lengths)

The offset-invariant metrics are the ones we trust for the hypothesis.

Sleep quality (PSG hypnogram, from the same epoch table)
--------------------------------------------------------
  sleep_eff   sleep epochs / scored epochs
  waso_min    wake minutes after sleep onset (before final awakening)
  sol_min     sleep-onset latency
  n_awake     number of wake bouts after onset
  pct_N3      % of sleep in N3            (deep sleep -> good)
  pct_REM     % of sleep in REM
  frag_idx    stage transitions per hour (fragmentation -> bad)

Analysis
--------
  * Per-session flow + quality metrics table.
  * Spearman correlation grid: each flow metric vs each quality metric (n=12).
  * Within-subject paired check: does the worse-quality night have the more
    imbalanced flow? (6 subject-pairs, sign test.)
  * "How flow changes across sleep": per-stage flow activity (pooled, z-scored).
  * Figures: per-session flow-vs-hypnogram traces, imbalance-vs-quality scatters,
    per-session reversal-rate bars, per-stage flow-activity boxplot.

n = 12 sessions / 6 subjects -> EXPLORATORY. Report effect sizes + caveats,
not p-value verdicts.

Usage:
    .venv/Scripts/python.exe analysis/mean_value/flow_imbalance_vs_quality.py

Requires reports/mean_value/mean_value_epochs.csv (produced by
mean_value_vs_stage.py). Regenerate that first if missing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import skew as sp_skew, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'mean_value'
REPORT_DIR = ROOT / 'reports' / 'mean_value'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_MIN = 0.5                       # 30 s epochs
FLOW_COL = 'vlf_CLE-CRE'              # the slowly-moving L-R mean = "flow"
SLEEP_CODES = {0, 1, 2, 3}           # REM, N3, N2, N1
WAKE_CODE = 4
REPRESENTATIVE = ['S1N1', 'S3N1', 'S5N1']

# Quality metrics where a LARGER value means WORSE sleep (for sign bookkeeping).
WORSE_IF_HIGH = {'waso_min', 'sol_min', 'n_awake', 'frag_idx'}
# ... and where LARGER means BETTER sleep.
BETTER_IF_HIGH = {'sleep_eff', 'pct_N3', 'pct_REM'}


# ── Per-session flow (imbalance) metrics ──────────────────────────────────────

def flow_metrics(g):
    """
    Flow imbalance metrics for one session.

    g : per-epoch rows for a session, sorted by epoch, with FLOW_COL, stage_code,
        motion (top-decile acc flag). Flow is analysed over the SLEEP period only
        (onset..final sleep epoch) with high-motion epochs dropped, because motion
        throws large transient swings into the baseline.
    """
    g = g.sort_values('epoch').reset_index(drop=True)
    code = g['stage_code'].to_numpy()

    # sleep period = first..last sleep epoch
    is_sleep = np.isin(code, list(SLEEP_CODES))
    if is_sleep.sum() < 20:
        return None
    on, off = np.argmax(is_sleep), len(is_sleep) - 1 - np.argmax(is_sleep[::-1])
    win = np.zeros(len(g), dtype=bool)
    win[on:off + 1] = True
    clean = win & (~g['motion'].to_numpy()) & np.isfinite(g[FLOW_COL].to_numpy())

    f = g.loc[clean, FLOW_COL].to_numpy()
    if len(f) < 20:
        return None
    n_hr = len(f) * EPOCH_MIN / 60.0

    med = np.median(f)
    sd = np.std(f)
    rms = np.sqrt(np.mean(f ** 2))
    # reversals: sign changes of (f - median). Robust to static offset.
    s = np.sign(f - med)
    s = s[s != 0]
    reversals = int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0

    # longest one-sided run about the session median (min) — "stuck" duration.
    runs_pos, runs_neg = _run_lengths(np.sign(f - med))
    all_runs = runs_pos + runs_neg
    max_run_min = (max(all_runs) if all_runs else 0) * EPOCH_MIN

    return {
        # offset-SENSITIVE (interpret with care — dominated by mask-placement offset)
        'flow_bias':      float(med),                       # signed operating point (a.u.)
        'flow_onesided':  float(abs(np.mean(f)) / (rms + 1e-9)),  # saturates ~1 (offset)
        'flow_offset_dom': float(abs(med) / (sd + 1e-9)),   # offset / modulation ratio
        # offset-INVARIANT (the real test of "stuck one-sided vs alternating")
        'flow_skew':      float(sp_skew(f)),
        'flow_absskew':   float(abs(sp_skew(f))),
        'flow_reversal':  float(reversals / n_hr),          # crossings per hour
        'flow_maxrun':    float(max_run_min),               # longest one-sided run (min)
        # context
        'flow_std':       float(sd),
        'flow_range':     float(np.ptp(f)),
        'n_clean_epochs': int(len(f)),
    }


def _run_lengths(sign_arr):
    """Return (pos_runs, neg_runs) lengths of contiguous same-sign stretches.
    Zeros (samples exactly at the median) are ignored."""
    pos, neg = [], []
    cur, n = 0, 0
    for s in sign_arr:
        if s == cur:
            n += 1
            continue
        if cur > 0:
            pos.append(n)
        elif cur < 0:
            neg.append(n)
        cur, n = s, 1
    if cur > 0:
        pos.append(n)
    elif cur < 0:
        neg.append(n)
    return pos, neg


# ── Per-session sleep-quality metrics (from hypnogram) ────────────────────────

def quality_metrics(g):
    g = g.sort_values('epoch').reset_index(drop=True)
    code = g['stage_code'].to_numpy()
    scored = code >= 0
    c = code[scored]
    if len(c) < 20:
        return None
    is_sleep = np.isin(c, list(SLEEP_CODES))
    if is_sleep.sum() < 10:
        return None

    n_scored = len(c)
    on = int(np.argmax(is_sleep))
    off = len(is_sleep) - 1 - int(np.argmax(is_sleep[::-1]))
    within = c[on:off + 1]                         # sleep period incl. mid-night wakes
    n_sleep = int(is_sleep.sum())

    waso = int(np.sum(within == WAKE_CODE))
    # wake bouts after onset
    wake_flag = (within == WAKE_CODE).astype(int)
    n_awake = int(np.sum(np.diff(np.r_[0, wake_flag]) == 1))
    # fragmentation: stage-code changes across the whole scored night
    frag = int(np.sum(np.diff(c) != 0))
    hours = n_scored * EPOCH_MIN / 60.0

    def pct(stage):
        return 100.0 * np.sum(c == stage) / n_sleep

    return {
        'sleep_eff': 100.0 * n_sleep / n_scored,      # %
        'waso_min':  waso * EPOCH_MIN,
        'sol_min':   on * EPOCH_MIN,
        'n_awake':   n_awake,
        'pct_N3':    pct(1),
        'pct_REM':   pct(0),
        'frag_idx':  frag / hours,
        'tst_min':   n_sleep * EPOCH_MIN,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_flow_hypno(g, label, out):
    """Signed flow trace across the night, over a colour-banded hypnogram."""
    g = g.sort_values('epoch')
    t = g['t_hr'].to_numpy()
    f = g[FLOW_COL].to_numpy()
    med = np.nanmedian(f[np.isin(g['stage_code'], list(SLEEP_CODES))])

    fig, ax = plt.subplots(figsize=(14, 4))
    codes = g['stage_code'].to_numpy()
    for j in range(len(t) - 1):
        ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                   alpha=0.16, lw=0)
    ax.plot(t, f, lw=1.2, color='#E67E22', label='flow = CLE-CRE baseline')
    ax.axhline(med, color='#2C3E50', ls='--', lw=1.0, label='session median')
    ax.fill_between(t, med, f, where=(f >= med), color='#C0392B', alpha=0.25,
                    interpolate=True, label='flow → left')
    ax.fill_between(t, med, f, where=(f < med), color='#2980B9', alpha=0.25,
                    interpolate=True, label='flow → right')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('flow  (CLE-CRE, a.u.)')
    ax.set_title(f'{label} — directional flow (L-R slow mean) vs hypnogram',
                 fontsize=12, fontweight='bold')
    handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
               for k in STAGE_ORDER]
    handles += [plt.Line2D([], [], color='#E67E22', label='flow'),
                plt.Line2D([], [], color='#2C3E50', ls='--', label='median')]
    ax.legend(handles=handles, loc='upper right', ncol=4, fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)


def fig_scatter(sess, xcol, ycol, out):
    """Per-session scatter of a flow metric vs a quality metric, Spearman annotated."""
    x = sess[xcol].to_numpy(); y = sess[ycol].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    rho, p = spearmanr(x[ok], y[ok])
    fig, ax = plt.subplots(figsize=(6, 5))
    subs = sorted(sess['subject'].unique())
    cmap = plt.get_cmap('tab10')
    for i, sub in enumerate(subs):
        m = sess['subject'] == sub
        ax.scatter(sess.loc[m, xcol], sess.loc[m, ycol], s=70, color=cmap(i),
                   label=sub, edgecolor='k', lw=0.5, zorder=3)
    for _, r in sess.iterrows():
        ax.annotate(r['session'], (r[xcol], r[ycol]), fontsize=6,
                    xytext=(3, 3), textcoords='offset points', alpha=0.7)
    if ok.sum() >= 3 and np.unique(x[ok]).size >= 2:
        b, a = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, a + b * xs, color='#555', ls='--', lw=1.2, zorder=2)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.set_title(f'{ycol} vs {xcol}\nSpearman ρ={rho:.2f}  p={p:.3f}  (n={ok.sum()})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2, title='subject')
    ax.grid(True, alpha=0.15)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)


def fig_reversal_bars(sess, out):
    """Per-session flow reversal rate, coloured by sleep efficiency."""
    s = sess.sort_values('flow_reversal')
    fig, ax = plt.subplots(figsize=(10, 5))
    norm = plt.Normalize(sess['sleep_eff'].min(), sess['sleep_eff'].max())
    colors = plt.get_cmap('RdYlGn')(norm(s['sleep_eff']))
    ax.bar(s['session'], s['flow_reversal'], color=colors, edgecolor='k', lw=0.5)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label='sleep efficiency (%)')
    ax.set_ylabel('flow reversals / hour'); ax.set_xlabel('session')
    ax.set_title('Flow alternation rate per session (low = stuck one-sided)\n'
                 'coloured by sleep efficiency', fontsize=11, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.15, axis='y')
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)


def fig_flow_by_stage(df, out):
    """How flow behaves across stages: per-session-z |flow-median| by stage (pooled)."""
    d = df[df['stage_code'] >= 0].copy()
    # per-session centre + scale, so sessions are comparable
    d['flow_c'] = np.nan
    for sess, g in d.groupby('session'):
        f = g[FLOW_COL].to_numpy()
        sl = np.isin(g['stage_code'], list(SLEEP_CODES))
        med = np.median(f[sl]) if sl.any() else np.median(f)
        sd = np.std(f) + 1e-9
        d.loc[g.index, 'flow_c'] = np.abs(f - med) / sd
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []
    for sc in STAGE_ORDER:
        v = d.loc[d['stage_code'] == sc, 'flow_c'].dropna().to_numpy()
        if len(v):
            data.append(v); labels.append(STAGE_LABELS[sc]); colors.append(STAGE_COLORS[sc])
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=2), widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticklabels(labels)
    ax.set_ylabel('|flow − session median|  (per-session z)')
    ax.set_title('Flow excursion magnitude by sleep stage (12 sessions pooled)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    epochs_csv = REPORT_DIR / 'mean_value_epochs.csv'
    if not epochs_csv.exists():
        sys.exit(f'Missing {epochs_csv}. Run mean_value_vs_stage.py first.')

    print('=' * 68)
    print('Directional flow (L-R slow mean) imbalance  vs  sleep quality')
    print('=' * 68)
    df = pd.read_csv(epochs_csv)
    print(f'Loaded {len(df)} epochs, {df["session"].nunique()} sessions, '
          f'{df["subject"].nunique()} subjects')

    # ── per-session flow + quality metrics ──
    rows = []
    for (sess, sub, night), g in df.groupby(['session', 'subject', 'night']):
        fm = flow_metrics(g)
        qm = quality_metrics(g)
        if fm is None or qm is None:
            print(f'  {sess}: skipped (insufficient sleep/clean epochs)')
            continue
        rows.append({'session': sess, 'subject': sub, 'night': night, **fm, **qm})
    sess = pd.DataFrame(rows).sort_values('session').reset_index(drop=True)
    sess.to_csv(REPORT_DIR / 'flow_imbalance_session.csv', index=False)
    print(f'\nPer-session table -> flow_imbalance_session.csv  ({len(sess)} sessions)')

    flow_metric_cols = ['flow_bias', 'flow_onesided', 'flow_offset_dom',
                        'flow_skew', 'flow_absskew', 'flow_reversal', 'flow_maxrun']
    qual_cols = ['sleep_eff', 'waso_min', 'sol_min', 'n_awake',
                 'pct_N3', 'pct_REM', 'frag_idx']

    # Confound diagnostic: how much bigger is the static offset than the modulation?
    od = sess['flow_offset_dom']
    print(f'\nOffset-dominance |median|/std : median={od.median():.1f}x '
          f'(range {od.min():.1f}-{od.max():.1f}). '
          f'Offset >> modulation -> absolute one-sidedness is instrumental.')

    # ── Spearman correlation grid ──
    print('\nSpearman rho  (flow metric x quality metric, n = %d sessions):' % len(sess))
    print('  offset-invariant metrics starred (trusted); watch sign vs hypothesis')
    invariant = {'flow_skew', 'flow_absskew', 'flow_reversal', 'flow_maxrun'}
    corr_rows = []
    header = f'{"flow metric":16s} ' + ' '.join(f'{q[:8]:>9s}' for q in qual_cols)
    print('  ' + header)
    for fmname in flow_metric_cols:
        cells = []
        for q in qual_cols:
            x = sess[fmname].to_numpy(); y = sess[q].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            rho, p = spearmanr(x[ok], y[ok]) if ok.sum() >= 4 else (np.nan, np.nan)
            corr_rows.append({'flow_metric': fmname, 'quality': q,
                              'spearman_rho': rho, 'p': p, 'n': int(ok.sum()),
                              'offset_invariant': fmname in invariant})
            star = '*' if (np.isfinite(p) and p < 0.05) else ' '
            cells.append(f'{rho:>+8.2f}{star}')
        tag = '*' if fmname in invariant else ' '
        print(f'  {fmname+tag:16s} ' + ' '.join(cells))
    pd.DataFrame(corr_rows).to_csv(REPORT_DIR / 'flow_quality_corr.csv', index=False)

    # ── within-subject de-meaned correlation (removes fixed per-subject offset) ──
    print('\nWithin-subject de-meaned Spearman (per-subject mean removed, isolates '
          'night-to-night STATE):')
    within_subject_corr(sess, flow_metric_cols, qual_cols)

    # ── within-subject paired night check ──
    print('\nWithin-subject paired check (worse-quality night vs more-imbalanced night):')
    paired_report(sess)

    # ── figures ──
    print('\nWriting figures...')
    for lbl in REPRESENTATIVE:
        g = df[df['session'] == lbl]
        if len(g):
            fig_flow_hypno(g, lbl, PLOT_DIR / f'flow_hypno_{lbl}.png')
            print(f'  {lbl}: flow-vs-hypnogram trace')
    # key scatters: the offset-invariant imbalance metrics vs core quality
    for fmname, q in [('flow_reversal', 'sleep_eff'), ('flow_reversal', 'frag_idx'),
                      ('flow_absskew', 'sleep_eff'), ('flow_absskew', 'waso_min'),
                      ('flow_maxrun', 'sleep_eff'), ('flow_bias', 'sleep_eff')]:
        fig_scatter(sess, fmname, q, PLOT_DIR / f'scatter_{fmname}_vs_{q}.png')
    print('  scatter grid: reversal/skew/maxrun/dwell vs efficiency/WASO/frag')
    fig_reversal_bars(sess, PLOT_DIR / 'flow_reversal_bars.png')
    fig_flow_by_stage(df, PLOT_DIR / 'flow_by_stage.png')
    print('  reversal-rate bars + flow-by-stage boxplot')

    print('\nDone.  Reports -> reports/mean_value/  Figures -> notebooks/plots/mean_value/')
    _print_takeaways(sess, corr_rows)


def within_subject_corr(sess, flow_cols, qual_cols):
    """Correlate flow vs quality AFTER removing each subject's own mean.

    Removing the per-subject mean strips the fixed mask-placement offset (and any
    stable anatomy), so a surviving correlation reflects night-to-night STATE that
    co-varies with sleep quality — the confound-free version of the hypothesis.
    Only subjects with both nights contribute (paired). n_pairs = 6.
    """
    d = sess.copy()
    both = d.groupby('subject')['night'].transform('nunique') >= 2
    d = d[both]
    if d.empty:
        print('  (no subjects with two nights)')
        return
    cols = flow_cols + qual_cols
    for c in cols:
        d[c + '_w'] = d[c] - d.groupby('subject')[c].transform('mean')
    rows = []
    focus_q = ['sleep_eff', 'waso_min', 'frag_idx']
    print(f'  {"flow metric":16s} ' + ' '.join(f'{q[:8]:>9s}' for q in focus_q))
    for fm in ['flow_bias', 'flow_reversal', 'flow_absskew', 'flow_maxrun']:
        cells = []
        for q in focus_q:
            x = d[fm + '_w'].to_numpy(); y = d[q + '_w'].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            r, p = spearmanr(x[ok], y[ok]) if ok.sum() >= 4 else (np.nan, np.nan)
            rows.append({'flow_metric': fm, 'quality': q, 'within_rho': r,
                         'p': p, 'n_pairs': int(ok.sum())})
            star = '*' if (np.isfinite(p) and p < 0.05) else ' '
            cells.append(f'{r:>+8.2f}{star}')
        # for bias, |bias| deviation is the imbalance reading
        print(f'  {fm:16s} ' + ' '.join(cells))
    pd.DataFrame(rows).to_csv(REPORT_DIR / 'flow_quality_within_subject.csv', index=False)


def paired_report(sess):
    """For each subject with both nights, does the worse night have worse flow balance?"""
    agree = {}
    for fmname in ['flow_reversal', 'flow_absskew', 'flow_maxrun', 'flow_bias']:
        hits = tot = 0
        for sub, g in sess.groupby('subject'):
            if g['night'].nunique() < 2:
                continue
            g = g.sort_values('night')
            n1, n2 = g.iloc[0], g.iloc[1]
            # worse sleep = lower efficiency
            worse = n1 if n1['sleep_eff'] < n2['sleep_eff'] else n2
            better = n2 if worse is n1 else n1
            # more imbalanced: lower reversal, higher skew/maxrun, larger |bias|
            if fmname == 'flow_reversal':
                more_imb = worse['flow_reversal'] < better['flow_reversal']
            elif fmname == 'flow_bias':
                more_imb = abs(worse['flow_bias']) > abs(better['flow_bias'])
            else:
                more_imb = worse[fmname] > better[fmname]
            hits += int(more_imb); tot += 1
        agree[fmname] = (hits, tot)
        print(f'  {fmname:16s}: worse-sleep night more imbalanced in {hits}/{tot} subjects')
    return agree


def _print_takeaways(sess, corr_rows):
    cdf = pd.DataFrame(corr_rows)
    inv = cdf[cdf['offset_invariant']].dropna(subset=['spearman_rho'])
    if inv.empty:
        return
    strongest = inv.reindex(inv['spearman_rho'].abs().sort_values(ascending=False).index).head(3)
    print('\nStrongest offset-invariant associations:')
    for _, r in strongest.iterrows():
        print(f'  {r["flow_metric"]} x {r["quality"]}: '
              f'rho={r["spearman_rho"]:+.2f} (p={r["p"]:.3f}, n={int(r["n"])})')


if __name__ == '__main__':
    main()
