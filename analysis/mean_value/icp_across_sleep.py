"""
How does intracranial fluid pressure change across a night of sleep?

Frames the overnight capacitive mean as a regional-ICP proxy and quantifies the
change, in femtofarads and (indicatively) in mmHg.

The physical chain
------------------
The mask reads capacitance in fF. The source papers relate capacitance change to
intracranial fluid mechanically (vessel diameter, CSF layer thickness, tissue
displacement) and report ONE sensitivity figure: 4.79 fF/mmHg (CLE, respiratory
band, PORCINE), with linearity established only over <1 mm displacement and
0-8.2 mmHg (r = 0.92). So:

    dP (mmHg)  ~=  dC (fF) / 4.79

Three limits on that arrow, applied honestly throughout this script:
  * RELATIVE ONLY. There is no absolute-ICP calibration for humans anywhere in
    the source work, so every number here is a CHANGE from a within-session
    reference, never an absolute pressure.
  * The direction is site- and mechanism-specific: higher ICP correlated
    POSITIVELY with capacitance at the eye and right-back sensors but NEGATIVELY
    at left-back, and a thicker CSF layer LOWERS capacitance while a wider vessel
    RAISES it. The mask's sensors are the periorbital/forehead ones, so the
    positive convention is used — flagged, not assumed silently.
  * The sensitivity is porcine and band-specific. mmHg axes are therefore drawn
    as a secondary, explicitly "indicative" scale; the primary unit is fF.

What is quantified
------------------
1. Reference        per-session Wake median (the within-night zero).
2. Stage effect     per-session median dC by stage -> pooled, Kruskal-Wallis,
                    and per-SUBJECT sign consistency (6 subjects, 2 nights each)
                    — the sign test matters more than the p-value here, because
                    epochs within a night are heavily autocorrelated.
3. Trajectory       dC against time since sleep onset, so a monotone overnight
                    pressure drift is separated from a stage-locked one.
4. Motion control   everything recomputed on motion-quiet epochs only, so a
                    stage effect is not just posture/coupling change.
5. Excursion        total overnight swing per session, in fF and mmHg.

Input
-----
reports/mean_value/mean_value_epochs.csv  (regenerate via mean_value_vs_stage.py
    after the CH + femtofarad change, so mean_CH is present)

Outputs
-------
    reports/mean_value/icp_stage_change.csv
    reports/mean_value/icp_subject_direction.csv
    reports/mean_value/icp_session_excursion.csv
    writeup/figures/channel_evolution/icp_across_sleep.png

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/icp_across_sleep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kruskal

# Channel labels carry a real minus sign (U+2212); the default Windows console
# codepage is cp1252 and would raise on it mid-report.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, CAP_UNIT, ICP_SENS_FF_PER_MMHG,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'mean_value'
PLOT_DIR = ROOT / 'writeup' / 'figures' / 'channel_evolution'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS_CSV = REPORT_DIR / 'mean_value_epochs.csv'
CHANNELS = ['CH', 'CLE', 'CRE', 'CLE-CRE', 'CLE+CRE']
CH_LABEL = {'CH': 'CH (hardware L−R)', 'CLE': 'CLE (left)', 'CRE': 'CRE (right)',
            'CLE-CRE': 'CLE−CRE (arithmetic L−R)', 'CLE+CRE': 'CLE+CRE (bilateral mean)'}
LADDER_ORDER = [4, 3, 2, 1, 0]          # Wake, N1, N2, N3, REM — sleep-depth order
WAKE, REM, N1, N2, N3 = 4, 0, 3, 2, 1
TRAJ_BIN_MIN = 20.0                     # bin width for the overnight trajectory
MOTION_QUIET_PCT = 50                   # epochs below this motion pct are "quiet"
DETREND_WIN_EPOCHS = 61                 # ~30 min centred rolling median (project convention)


def to_mmhg(dc):
    return np.asarray(dc, float) / ICP_SENS_FF_PER_MMHG


def load_epochs():
    if not EPOCHS_CSV.exists():
        sys.exit(f'missing {EPOCHS_CSV}\n'
                 'run: .venv/Scripts/python.exe analysis/mean_value/mean_value_vs_stage.py')
    df = pd.read_csv(EPOCHS_CSV)
    missing = [c for c in CHANNELS if f'mean_{c}' not in df.columns]
    if missing:
        sys.exit(f'epoch table lacks mean_{{{",".join(missing)}}} — re-run '
                 'mean_value_vs_stage.py (it now emits CH, in femtofarads)')
    return df


def add_referenced(df):
    """Two referenced series per channel, both in fF:

    dC_<ch>   mean - the session's Wake median. Keeps the slow overnight drift,
              so a stage contrast computed on it is CONFOUNDED by that drift —
              the overnight swing is ~10x the stage effect, so one night with a
              strong monotone ramp can invert a subject's contrast.
    dCd_<ch>  the same series minus a ~30 min centred rolling-median trend, i.e.
              the STAGE-LOCKED component with the drift removed. This is the
              series a "does pressure change with sleep stage" question wants.
    """
    out = []
    for sess, g in df.groupby('session', sort=False):
        g = g.copy().sort_values('t_hr')
        wake = g['stage_code'] == WAKE
        for ch in CHANNELS:
            col = f'mean_{ch}'
            ref = g.loc[wake, col].median() if wake.sum() >= 5 else g[col].median()
            g[f'dC_{ch}'] = g[col] - ref
            trend = (g[col].rolling(DETREND_WIN_EPOCHS, center=True, min_periods=10)
                     .median())
            g[f'dCd_{ch}'] = g[col] - trend
        g['quiet'] = g['acc_std'] <= g['acc_std'].quantile(MOTION_QUIET_PCT / 100)
        # time since the first non-Wake epoch, for the trajectory
        slept = g.loc[g['stage_code'] != WAKE, 't_hr']
        onset = slept.iloc[0] if len(slept) else g['t_hr'].iloc[0]
        g['t_since_onset_hr'] = g['t_hr'] - onset
        out.append(g)
    return pd.concat(out, ignore_index=True)


def stage_change(df, quiet_only=False, pre='dC'):
    """Per-session median dC by stage, then pooled across the 12 sessions."""
    d = df[df['quiet']] if quiet_only else df
    rows = []
    per_sess = (d.groupby(['session', 'subject', 'stage_code'])[
                    [f'{pre}_{c}' for c in CHANNELS]].median().reset_index())
    for ch in CHANNELS:
        col = f'{pre}_{ch}'
        groups = [per_sess.loc[per_sess['stage_code'] == c, col].dropna().values
                  for c in LADDER_ORDER]
        groups = [g for g in groups if len(g) >= 3]
        H, p = kruskal(*groups) if len(groups) >= 3 else (np.nan, np.nan)
        for c in LADDER_ORDER:
            v = per_sess.loc[per_sess['stage_code'] == c, col].dropna().values
            if not len(v):
                continue
            rows.append(dict(
                channel=ch, stage=STAGE_LABELS[c], stage_code=c, series=pre,
                n_sessions=len(v), motion=('quiet' if quiet_only else 'all'),
                dC_median_fF=float(np.median(v)),
                dC_q1_fF=float(np.percentile(v, 25)),
                dC_q3_fF=float(np.percentile(v, 75)),
                dP_median_mmHg=float(to_mmhg(np.median(v))),
                KW_H=H, KW_p=p))
    return pd.DataFrame(rows), per_sess


def subject_direction(per_sess, pre='dC'):
    """Per-SUBJECT contrast sign — the honest consistency check (n=6, not n=epochs).
    Nights are averaged within subject first so a subject cannot vote twice. The
    per-NIGHT tally is reported alongside, because averaging two nights lets one
    drift-dominated night invert a subject whose other night agrees."""
    rows = []
    contrasts = [('N3_vs_Wake', N3, WAKE), ('N2_vs_Wake', N2, WAKE),
                 ('REM_vs_Wake', REM, WAKE), ('N3_vs_REM', N3, REM)]
    for ch in CHANNELS:
        col = f'{pre}_{ch}'
        wide = per_sess.pivot_table(index=['subject', 'session'],
                                    columns='stage_code', values=col)
        for name, a, b in contrasts:
            if a not in wide.columns or b not in wide.columns:
                continue
            per_night = (wide[a] - wide[b]).dropna()
            per_subj = per_night.groupby(level='subject').mean()
            if len(per_subj) < 3:
                continue
            npos = int((per_subj > 0).sum()); nneg = int((per_subj < 0).sum())
            rows.append(dict(
                channel=ch, contrast=name, series=pre, n_subjects=len(per_subj),
                median_dC_fF=float(per_subj.median()),
                median_dP_mmHg=float(to_mmhg(per_subj.median())),
                n_pos=npos, n_neg=nneg,
                n_nights=len(per_night),
                n_nights_pos=int((per_night > 0).sum()),
                n_nights_neg=int((per_night < 0).sum()),
                per_subject_fF=', '.join(f'{v:+.1f}' for v in per_subj.values),
                verdict=('CONSISTENT' if max(npos, nneg) >= len(per_subj) - 1
                         else 'SUBJECT-DEPENDENT')))
    return pd.DataFrame(rows)


def session_excursion(df):
    rows = []
    for (sess, subj), g in df.groupby(['session', 'subject'], sort=False):
        g = g.sort_values('t_hr')
        row = dict(session=sess, subject=subj)
        for ch in CHANNELS:
            y = g[f'dC_{ch}'].to_numpy(float)
            # 15-epoch (~7.5 min) rolling median, so the swing is drift not spikes
            ys = pd.Series(y).rolling(15, center=True, min_periods=5).median().to_numpy()
            ys = ys[np.isfinite(ys)]
            if len(ys) < 20:
                continue
            row[f'swing_fF_{ch}'] = float(np.nanmax(ys) - np.nanmin(ys))
            row[f'swing_mmHg_{ch}'] = float(to_mmhg(np.nanmax(ys) - np.nanmin(ys)))
            row[f'net_fF_{ch}'] = float(np.nanmedian(ys[-40:]) - np.nanmedian(ys[:40]))
        rows.append(row)
    return pd.DataFrame(rows)


def trajectory(df, ch):
    """Median dC vs time since sleep onset, across sessions."""
    d = df[df['t_since_onset_hr'] >= 0].copy()
    d['bin'] = (d['t_since_onset_hr'] * 60 // TRAJ_BIN_MIN) * TRAJ_BIN_MIN / 60
    # median within session-bin first, so long sessions don't dominate
    per = d.groupby(['session', 'bin'])[f'dC_{ch}'].median().reset_index()
    g = per.groupby('bin')[f'dC_{ch}']
    out = pd.DataFrame({'bin': g.median().index, 'med': g.median().values,
                        'q1': g.quantile(0.25).values, 'q3': g.quantile(0.75).values,
                        'n': g.size().values})
    return out[out['n'] >= 6]


def mmhg_axis(ax, label='Δ pressure (mmHg, indicative)'):
    sec = ax.secondary_yaxis('right', functions=(to_mmhg,
                                                 lambda p: np.asarray(p) * ICP_SENS_FF_PER_MMHG))
    sec.set_ylabel(label, fontsize=9.5, color='#7B241C')
    sec.tick_params(labelsize=8.5, colors='#7B241C')
    return sec


def make_figure(stage_all, stage_quiet, subj, subj_d, exc, df, out):
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.60)

    # A — per-stage dC, all channels
    ax = fig.add_subplot(gs[0, 0])
    order = [STAGE_LABELS[c] for c in LADDER_ORDER]
    w = 0.15
    for k, ch in enumerate(CHANNELS):
        sub = stage_all[stage_all['channel'] == ch].set_index('stage')
        vals = [sub.loc[st, 'dC_median_fF'] if st in sub.index else np.nan for st in order]
        ax.bar(np.arange(len(order)) + (k - 2) * w, vals, w, label=CH_LABEL[ch], alpha=0.9)
    ax.axhline(0, color='k', lw=1.0)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel(f'Δ capacitance vs Wake ({CAP_UNIT})')
    ax.legend(fontsize=7.5, loc='lower left'); ax.grid(True, axis='y', alpha=0.2)
    kw = stage_all.groupby('channel')['KW_p'].first()
    nsig = int((kw < 0.05).sum())
    verdict = (f'{nsig}/{len(kw)} channels p<0.05' if nsig
               else f'all n.s., best p={kw.min():.2f}')
    ax.set_title('A  Stage effect (per-session medians)', fontsize=11,
                 fontweight='bold', loc='left')
    ax.text(0.99, 0.985, f'KW across stages: {verdict} (n=12)',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            style='italic', color='#7B241C')
    mmhg_axis(ax)

    # B — motion control on the strongest channel
    ax = fig.add_subplot(gs[0, 1])
    for lbl, tab, colr, hatch in [('all epochs', stage_all, '#2980B9', None),
                                  ('motion-quiet half', stage_quiet, '#F39C12', '//')]:
        sub = tab[tab['channel'] == 'CLE'].set_index('stage')
        vals = [sub.loc[st, 'dC_median_fF'] if st in sub.index else np.nan for st in order]
        off = -0.19 if lbl == 'all epochs' else 0.19
        ax.bar(np.arange(len(order)) + off, vals, 0.38, label=lbl, color=colr,
               alpha=0.9, hatch=hatch)
    ax.axhline(0, color='k', lw=1.0)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel(f'Δ capacitance vs Wake ({CAP_UNIT})')
    ax.legend(fontsize=8.5); ax.grid(True, axis='y', alpha=0.2)
    ax.set_title('B  CLE — is the stage effect just motion/posture?', fontsize=11,
                 fontweight='bold', loc='left')

    # C — per-subject consistency of N3 vs Wake, drift kept vs drift removed
    ax = fig.add_subplot(gs[0, 2])
    a = subj[subj['contrast'] == 'N3_vs_Wake'].reset_index(drop=True)
    b = subj_d[subj_d['contrast'] == 'N3_vs_Wake'].set_index('channel')
    ys = np.arange(len(a))
    for y, (_, r) in zip(ys, a.iterrows()):
        vals = [float(v) for v in r['per_subject_fF'].split(',')]
        ax.scatter(vals, [y - 0.16] * len(vals), s=38, color='#5D6D7E',
                   alpha=0.7, zorder=3)
        ax.scatter([r['median_dC_fF']], [y - 0.16], s=130, marker='|',
                   color='#C0392B', linewidth=2.5, zorder=4)
        if r['channel'] in b.index:
            rd = b.loc[r['channel']]
            vd = [float(v) for v in rd['per_subject_fF'].split(',')]
            ax.scatter(vd, [y + 0.16] * len(vd), s=38, color='#16A085',
                       alpha=0.8, zorder=3)
            ax.scatter([rd['median_dC_fF']], [y + 0.16], s=130, marker='|',
                       color='#0E6251', linewidth=2.5, zorder=4)
    ax.axvline(0, color='k', lw=1.2)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{r['channel']}\ndrift {r['n_pos']}+/{r['n_neg']}−   "
         f"detr {b.loc[r['channel'], 'n_pos'] if r['channel'] in b.index else '?'}+/"
         f"{b.loc[r['channel'], 'n_neg'] if r['channel'] in b.index else '?'}−"
         for _, r in a.iterrows()], fontsize=7.5)
    ax.plot([], [], 'o', color='#5D6D7E', label='drift kept')
    ax.plot([], [], 'o', color='#16A085', label='drift removed')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlabel(f'N3 − Wake, per subject ({CAP_UNIT})')
    ax.set_title('C  Per-subject direction (n=6, nights averaged)', fontsize=11,
                 fontweight='bold', loc='left')
    ax.grid(True, axis='x', alpha=0.2); ax.invert_yaxis()

    # D — overnight trajectory
    ax = fig.add_subplot(gs[1, 0])
    for ch, colr in [('CLE', '#27AE60'), ('CRE', '#8E44AD'), ('CLE+CRE', '#2C3E50')]:
        tr = trajectory(df, ch)
        ax.fill_between(tr['bin'], tr['q1'], tr['q3'], color=colr, alpha=0.15)
        ax.plot(tr['bin'], tr['med'], lw=1.8, color=colr, label=CH_LABEL[ch])
    ax.axhline(0, color='k', lw=1.0, ls=':')
    ax.set_xlabel('Time since sleep onset (hours)')
    ax.set_ylabel(f'Δ capacitance vs Wake ({CAP_UNIT})')
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.2)
    ax.set_title(f'D  Overnight trajectory (median ± IQR, {TRAJ_BIN_MIN:g}-min bins)',
                 fontsize=11, fontweight='bold', loc='left')
    mmhg_axis(ax)

    # E — per-session total swing
    ax = fig.add_subplot(gs[1, 1])
    e = exc.sort_values('session')
    ax.barh(e['session'], e['swing_fF_CLE'], color='#27AE60', alpha=0.85)
    ax.set_xlabel(f'Overnight swing in CLE ({CAP_UNIT})')
    ax.set_title('E  Total overnight excursion per session', fontsize=11,
                 fontweight='bold', loc='left')
    ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)
    sec = ax.secondary_xaxis('top', functions=(to_mmhg,
                                               lambda p: np.asarray(p) * ICP_SENS_FF_PER_MMHG))
    sec.set_xlabel('mmHg (indicative)', fontsize=9.5, color='#7B241C')
    sec.tick_params(labelsize=8.5, colors='#7B241C')

    # F — the caveat panel, on the figure so no one reads mmHg as calibrated ICP
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    ax.text(0, 1.0, 'How to read the mmHg axis', fontsize=11.5, fontweight='bold',
            va='top', transform=ax.transAxes)
    ax.text(0, 0.90,
            f'Primary unit is capacitance ({CAP_UNIT}). The right-hand mmHg scale\n'
            f'divides by {ICP_SENS_FF_PER_MMHG} fF/mmHg, the only sensitivity the source\n'
            'work reports — and it is:\n\n'
            '  • PORCINE, not human. No human absolute-ICP\n'
            '    calibration exists in the source papers.\n'
            '  • measured on CLE in the respiratory band, then\n'
            '    applied here to every channel and to the slow mean.\n'
            '  • linear only over <1 mm displacement / 0–8.2 mmHg\n'
            '    (r = 0.92); sensitivity is stated to depend on\n'
            '    vascular and tissue geometry.\n\n'
            'So these are RELATIVE pressure changes against a within-\n'
            'night Wake reference, never absolute ICP.\n\n'
            'Sign convention: capacitance rises with regional ICP at\n'
            'the periorbital/forehead sensors (positive correlation at\n'
            'LE/RE/RB in the source work). A thicker CSF layer LOWERS\n'
            'capacitance, so a rise here is read as more blood volume /\n'
            'higher regional pressure, not more CSF.',
            fontsize=8.6, va='top', transform=ax.transAxes, linespacing=1.45)

    fig.suptitle('Intracranial fluid pressure across sleep, from the capacitive mean',
                 fontsize=14, fontweight='bold')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    df = add_referenced(load_epochs())
    # Two passes: the raw Wake-referenced series (drift-confounded) and the
    # drift-removed series that isolates the stage-locked component.
    stage_all, per_sess = stage_change(df, quiet_only=False, pre='dC')
    stage_quiet, _ = stage_change(df, quiet_only=True, pre='dC')
    stage_detr, per_sess_d = stage_change(df, quiet_only=False, pre='dCd')
    subj = subject_direction(per_sess, pre='dC')
    subj_d = subject_direction(per_sess_d, pre='dCd')
    exc = session_excursion(df)

    pd.concat([stage_all, stage_quiet, stage_detr], ignore_index=True).to_csv(
        REPORT_DIR / 'icp_stage_change.csv', index=False)
    pd.concat([subj, subj_d], ignore_index=True).to_csv(
        REPORT_DIR / 'icp_subject_direction.csv', index=False)
    exc.to_csv(REPORT_DIR / 'icp_session_excursion.csv', index=False)
    make_figure(stage_all, stage_quiet, subj, subj_d, exc, df,
                PLOT_DIR / 'icp_across_sleep.png')

    for lbl, st, sj in [('RAW (drift kept — confounded)', stage_all, subj),
                        ('DETRENDED (stage-locked)', stage_detr, subj_d)]:
        print(f'\n===== {lbl} =====')
        print('  Stage effect vs Wake (per-session medians):')
        for ch in CHANNELS:
            s = st[st['channel'] == ch]
            bits = ' '.join(f"{r['stage']}:{r['dC_median_fF']:+6.1f}" for _, r in s.iterrows())
            p = s['KW_p'].iloc[0] if len(s) else np.nan
            print(f'    {CH_LABEL[ch]:28s} {bits}   KW p={p:.2g}')
        print('  Direction (per subject, and per night):')
        for _, r in sj[sj['contrast'].isin(['N3_vs_Wake', 'REM_vs_Wake'])].iterrows():
            print(f"    {CH_LABEL[r['channel']]:28s} {r['contrast']:12s} "
                  f"{r['median_dC_fF']:+6.1f} fF ({r['median_dP_mmHg']:+5.2f} mmHg)  "
                  f"subj {r['n_pos']}+/{r['n_neg']}−  "
                  f"nights {r['n_nights_pos']}+/{r['n_nights_neg']}−  {r['verdict']}")
    print(f'\n== Overnight swing (CLE) ==')
    print(f"  median {exc['swing_fF_CLE'].median():.1f} fF "
          f"({to_mmhg(exc['swing_fF_CLE'].median()):.2f} mmHg indicative), "
          f"range [{exc['swing_fF_CLE'].min():.1f}, {exc['swing_fF_CLE'].max():.1f}] fF")
    print(f'\nReports -> {REPORT_DIR}\nFigure  -> {PLOT_DIR / "icp_across_sleep.png"}')


if __name__ == '__main__':
    main()
