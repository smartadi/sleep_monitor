"""
Mean-value traces expressed as a deviation from the session mean, in femtofarads.

Replaces the z-scored presentation used in earlier mean-value figures. A z-score
hides the two numbers a reader actually needs: what the operating point was, and
how many femtofarads the signal actually moved. Here every trace is simply

    y(t) = x(t) - mu,        mu = mean of x over the session's sleep period

so zero is the session mean and the y-axis is in real fF. mu itself is printed
on every panel and tabulated per session, so nothing is lost by centring.

What is reported
----------------
  * per-session mu for every channel (CH, CLE, CRE, CLE-CRE, CLE+CRE), in fF
  * per-session excursion of the centred trace: sd, p5, p95, full range
  * per-stage median deviation from the session mean, in fF, per channel,
    pooled and per subject -- the exact values, annotated on the figure

Channels: CH is the sensor's own hardware difference channel and is NOT equal to
the arithmetic CLE-CRE (see ch_vs_clecre_sessions.py); both are carried.

Outputs
-------
    reports/mean_value/mean_centred_session_means.csv
    reports/mean_value/mean_centred_stage_deviation.csv
    writeup/figures/mean_value/mean_centred_<SESSION>.png        (12)
    writeup/figures/mean_value/mean_centred_grid_CLE_CRE.png
    writeup/figures/mean_value/mean_centred_stage_deviation.png
    writeup/figures/mean_value/mean_centred_operating_points.png

Reads reports/mean_value/mean_value_epochs.csv (mean_value_vs_stage.py).

Usage:
    .venv/Scripts/python.exe analysis/mean_value/mean_centred_traces.py
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
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, CAP_UNIT,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'mean_value'
FIG_DIR = ROOT / 'writeup' / 'figures' / 'mean_value'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_CODES = [0, 1, 2, 3]
STAGE_LADDER = [4, 3, 2, 1, 0]          # Wake, N1, N2, N3, REM

# channel -> (column stem, colour, linewidth, display label)
CHANNELS = [
    ('CLE',     '#27AE60', 1.0, 'CLE (left temple)'),
    ('CRE',     '#8E44AD', 1.0, 'CRE (right temple)'),
    ('CH',      '#2980B9', 1.0, 'CH (hardware difference)'),
    ('CLE-CRE', '#E67E22', 1.7, 'CLE-CRE (arithmetic difference)'),
]
# Grid panels clip to this window so one motion spike cannot flatten a night.
GRID_CLIP = 300.0       # fF about the session mean


def sleep_mask(g):
    """Epoch mask for the sleep period (first to last scored sleep epoch)."""
    codes = g['stage_code'].to_numpy()
    is_sleep = np.isin(codes, SLEEP_CODES)
    if is_sleep.sum() < 20:
        return np.ones(len(g), bool)
    on = int(np.argmax(is_sleep))
    off = len(is_sleep) - 1 - int(np.argmax(is_sleep[::-1]))
    m = np.zeros(len(g), bool)
    m[on:off + 1] = True
    return m


def _hypno_bands(ax, t, codes, alpha=0.16):
    for j in range(len(t) - 1):
        ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                   alpha=alpha, lw=0)


# ── per-session figure ───────────────────────────────────────────────────────

def _pct_lim(y, pad_frac=0.12, lo_p=0.5, hi_p=99.5):
    """Y-limits set by percentiles so one re-seat spike cannot flatten the trace."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(y, [lo_p, hi_p])
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def fig_session(g, label, mus, out, ha=None):
    """
    Per-session mean-value figure, three stacked panels.

    The four channels are NOT drawn on one axis. CH moves ~2.8x as far as the
    temple channels (up to 11x -- see ch_vs_clecre_sessions.py), so a shared
    axis is set by CH and flattens CLE, CRE and CLE-CRE onto the zero line.
    Absolute channels, difference channels and posture each get their own row.
    """
    g = g.sort_values('epoch')
    t = g['t_hr'].to_numpy()
    codes = g['stage_code'].to_numpy()
    motion = g['motion'].to_numpy().astype(bool) if 'motion' in g else np.zeros(len(g), bool)
    has_ha = ha is not None and len(ha) > 0

    nrow = 3 if has_ha else 2
    heights = [1.0, 1.0, 0.62][:nrow]
    fig, axes = plt.subplots(nrow, 1, figsize=(14, 3.1 * nrow + 1.0), sharex=True,
                             gridspec_kw={'height_ratios': heights})

    # ── row 1: the two absolute temple channels ──
    ax = axes[0]
    _hypno_bands(ax, t, codes)
    ys = []
    for ch, colr, _, lbl in [c for c in CHANNELS if c[0] in ('CLE', 'CRE')]:
        y = g[f'mean_{ch}'].to_numpy() - mus[ch]
        ys.append(y)
        ax.plot(t, y, lw=1.2, color=colr, alpha=0.9,
                label=f'{lbl}   mean = {mus[ch]:,.0f} {CAP_UNIT}', zorder=3)
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.0, zorder=2)
    lo, hi = _pct_lim(np.concatenate(ys))
    ax.set_ylim(lo, hi)
    if motion.any():
        ax.plot(t[motion], np.full(motion.sum(), hi - 0.04 * (hi - lo)), '|',
                color='k', ms=5, alpha=0.55, zorder=5)
    ax.set_ylabel(f'Δ from session mean\n({CAP_UNIT})')
    ax.set_title(f'{label} — mean value as a deviation from the session mean  '
                 f'(zero = session mean, axis in {CAP_UNIT})',
                 fontsize=12.5, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.13)

    # ── row 2: the two difference channels, independent axes ──
    ax = axes[1]
    _hypno_bands(ax, t, codes)
    yd = g['mean_CLE-CRE'].to_numpy() - mus['CLE-CRE']
    ax.plot(t, yd, lw=1.6, color='#E67E22', zorder=4)
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.0, zorder=2)
    ax.set_ylim(*_pct_lim(yd))
    ax.set_ylabel(f'CLE−CRE − mean\n({CAP_UNIT})', color='#E67E22')
    ax.tick_params(axis='y', labelcolor='#E67E22')
    ax.grid(True, alpha=0.13)

    # CH is deliberately not drawn on this row. It was on a twin axis of its own
    # scale, which put two channels over one another and made each one's step
    # structure harder to read; the row is the CLE-CRE mean evolution.
    ax.legend(handles=[
        plt.Line2D([], [], color='#E67E22', lw=2.2,
                   label=f'CLE-CRE (arithmetic)   mean = {mus["CLE-CRE"]:,.0f} {CAP_UNIT}')],
        loc='upper left', fontsize=8, framealpha=0.92)

    # ── row 3: head posture, which is what most of the steps above are ──
    if has_ha:
        ax = axes[2]
        _hypno_bands(ax, t, codes)
        ax.plot(ha['t_hr'], ha['turn_deg'], lw=1.3, color='#16A085', zorder=3)
        for lv, lb in [(90, 'left'), (0, 'supine'), (-90, 'right')]:
            ax.axhline(lv, color='#555', ls=':', lw=0.8, zorder=2)
            ax.annotate(lb, (t[0], lv), fontsize=7.5, color='#555', va='bottom')
        ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_ylabel('head turn\n(deg)', color='#16A085')
        ax.tick_params(axis='y', labelcolor='#16A085')
        ax.grid(True, alpha=0.13)

    axes[-1].set_xlabel('Time (hours)')
    handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
               for k in STAGE_ORDER]
    handles.append(plt.Line2D([], [], color='k', marker='|', ls='',
                              label='high-motion epoch'))
    fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out, dpi=185, bbox_inches='tight'); plt.close(fig)


def fig_grid(df, sessions, mu_tbl, ch, out):
    fig, axes = plt.subplots(6, 2, figsize=(20, 20), sharey=True)
    for ax, lbl in zip(axes.ravel(), sessions):
        g = df[df['session'] == lbl].sort_values('epoch')
        t = g['t_hr'].to_numpy()
        codes = g['stage_code'].to_numpy()
        mu = float(mu_tbl.loc[mu_tbl['session'] == lbl, f'mean_{ch}'].iloc[0])
        y = np.clip(g[f'mean_{ch}'].to_numpy() - mu, -GRID_CLIP, GRID_CLIP)
        _hypno_bands(ax, t, codes)
        ax.plot(t, y, lw=1.0, color='#E67E22', zorder=3)
        ax.axhline(0, color='#2C3E50', ls='--', lw=0.9, zorder=2)
        ax.fill_between(t, 0, y, where=(y >= 0), color='#C0392B', alpha=0.22,
                        interpolate=True, zorder=1)
        ax.fill_between(t, 0, y, where=(y < 0), color='#2980B9', alpha=0.22,
                        interpolate=True, zorder=1)
        ax.set_ylim(-GRID_CLIP, GRID_CLIP)
        ax.set_title(f'{lbl}   session mean = {mu:,.0f} {CAP_UNIT}',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Δ from mean ({CAP_UNIT})', fontsize=8)
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
               for k in STAGE_ORDER]
    handles += [plt.Line2D([], [], color='#E67E22', label=f'{ch} − session mean'),
                mpatches.Patch(color='#C0392B', alpha=0.4, label='above session mean'),
                mpatches.Patch(color='#2980B9', alpha=0.4, label='below session mean')]
    fig.legend(handles=handles, loc='upper center', ncol=8, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f'{ch} mean value referenced to each session mean — all 12 sessions '
                 f'(axis in {CAP_UNIT}, clipped at ±{GRID_CLIP:.0f} {CAP_UNIT})',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)


def fig_stage_deviation(dev, out):
    """Per-stage median deviation from the session mean, in fF, with the numbers."""
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4.4 * len(CHANNELS), 5.4),
                             sharex=True)
    order = [STAGE_LABELS[c] for c in STAGE_LADDER]
    for ax, (ch, colr, _, lbl) in zip(np.atleast_1d(axes), CHANNELS):
        d = dev[dev['channel'] == ch]
        med, lo, hi, vals = [], [], [], []
        for st in order:
            v = d.loc[d['stage'] == st, 'dev_fF'].to_numpy()
            vals.append(v)
            med.append(np.median(v) if len(v) else np.nan)
            q = np.percentile(v, [25, 75]) if len(v) else (np.nan, np.nan)
            lo.append(q[0]); hi.append(q[1])
        xs = np.arange(len(order))
        ax.bar(xs, med, color=[STAGE_COLORS[c] for c in STAGE_LADDER],
               alpha=0.75, edgecolor='k', lw=0.6, zorder=2)
        ax.errorbar(xs, med, yerr=[np.array(med) - np.array(lo),
                                   np.array(hi) - np.array(med)],
                    fmt='none', ecolor='#333', elinewidth=1.1, capsize=4, zorder=3)
        for x, v in zip(xs, vals):                       # per-session dots
            ax.plot(np.full(len(v), x) + np.linspace(-0.16, 0.16, len(v)), v, 'o',
                    ms=3, color='k', alpha=0.45, zorder=4)
        for x, m in zip(xs, med):
            if np.isfinite(m):
                ax.annotate(f'{m:+.1f}', (x, m), fontsize=9, fontweight='bold',
                            ha='center', va='bottom' if m >= 0 else 'top',
                            xytext=(0, 6 if m >= 0 else -12),
                            textcoords='offset points')
        ax.axhline(0, color='k', lw=1.1)
        ax.set_xticks(xs); ax.set_xticklabels(order)
        ax.set_title(lbl, fontsize=10.5, fontweight='bold', color=colr)
        ax.grid(True, axis='y', alpha=0.18)
    np.atleast_1d(axes)[0].set_ylabel(
        f'Median deviation from session mean ({CAP_UNIT})')
    fig.suptitle('How far each sleep stage sits from the session mean, in femtofarads\n'
                 'bar = median across the 12 sessions, whisker = IQR, dot = one session '
                 '(zero = that session\'s own mean)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)


def fig_operating_points(mu_tbl, out):
    """The mu values themselves — what was subtracted, per session and channel."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    sessions = mu_tbl['session'].tolist()
    xs = np.arange(len(sessions))

    ax = axes[0]
    for ch, colr, _, lbl in CHANNELS:
        if ch == 'CLE-CRE':
            continue
        ax.plot(xs, mu_tbl[f'mean_{ch}'], 'o-', color=colr, ms=7, lw=1.4, label=lbl)
        for x, v in zip(xs, mu_tbl[f'mean_{ch}']):
            ax.annotate(f'{v:,.0f}', (x, v), fontsize=6.5, ha='center',
                        xytext=(0, 7), textcoords='offset points')
    ax.set_xticks(xs); ax.set_xticklabels(sessions, rotation=45, ha='right')
    ax.set_ylabel(f'Session mean ({CAP_UNIT})')
    ax.set_title('A  Operating point subtracted from each trace',
                 fontsize=11.5, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.18)

    ax = axes[1]
    w = 0.38
    ax.bar(xs - w / 2, mu_tbl['mean_CLE-CRE'], w, color='#E67E22', alpha=0.85,
           edgecolor='k', lw=0.5, label='CLE−CRE session mean')
    ax.bar(xs + w / 2, mu_tbl['sd_CLE-CRE'], w, color='#7F8C8D', alpha=0.85,
           edgecolor='k', lw=0.5, label='CLE−CRE sd about that mean')
    for x, v in zip(xs, mu_tbl['mean_CLE-CRE']):
        ax.annotate(f'{v:,.0f}', (x - w / 2, v), fontsize=6.5, ha='center',
                    va='bottom' if v >= 0 else 'top',
                    xytext=(0, 5 if v >= 0 else -10), textcoords='offset points')
    ax.axhline(0, color='k', lw=1.1)
    ax.set_xticks(xs); ax.set_xticklabels(sessions, rotation=45, ha='right')
    ax.set_ylabel(CAP_UNIT)
    ax.set_title('B  Differential operating point vs how much it moves\n'
                 '(S5N1 is the one session with a positive L−R offset)',
                 fontsize=11.5, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, axis='y', alpha=0.18)

    fig.suptitle('The numbers removed by mean-centring — reported, not hidden',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    csv = REPORT_DIR / 'mean_value_epochs.csv'
    if not csv.exists():
        sys.exit(f'Missing {csv}. Run mean_value_vs_stage.py first.')
    df = pd.read_csv(csv)
    sessions = sorted(df['session'].unique())

    # ── per-session mu (over the sleep period) + excursion ──
    mu_rows, dev_rows = [], []
    mus_by_session = {}
    for lbl in sessions:
        g = df[df['session'] == lbl].sort_values('epoch')
        m = sleep_mask(g)
        row = {'session': lbl, 'subject': g['subject'].iloc[0],
               'night': int(g['night'].iloc[0]), 'n_epochs_sleep': int(m.sum())}
        mus = {}
        for ch, *_ in CHANNELS:
            x = g[f'mean_{ch}'].to_numpy()
            mu = float(np.nanmean(x[m]))
            mus[ch] = mu
            c = x - mu
            row[f'mean_{ch}'] = mu
            row[f'sd_{ch}'] = float(np.nanstd(c[m]))
            row[f'p5_{ch}'] = float(np.nanpercentile(c[m], 5))
            row[f'p95_{ch}'] = float(np.nanpercentile(c[m], 95))
            row[f'range_{ch}'] = float(np.nanmax(c[m]) - np.nanmin(c[m]))
            # per-stage deviation from THIS session's mean, in fF
            for sc in STAGE_LADDER:
                v = c[(g['stage_code'].to_numpy() == sc)]
                if len(v) >= 10:
                    dev_rows.append({'session': lbl, 'subject': row['subject'],
                                     'channel': ch, 'stage': STAGE_LABELS[sc],
                                     'stage_code': sc, 'n_epochs': len(v),
                                     'dev_fF': float(np.median(v)),
                                     'session_mean_fF': mu})
        mus_by_session[lbl] = mus
        mu_rows.append(row)

    mu_tbl = pd.DataFrame(mu_rows)
    dev = pd.DataFrame(dev_rows)
    mu_tbl.to_csv(REPORT_DIR / 'mean_centred_session_means.csv', index=False)
    dev.to_csv(REPORT_DIR / 'mean_centred_stage_deviation.csv', index=False)

    # ── figures ──
    # Head angle is optional context: the large steps in the differential are
    # mostly the head re-seating, so posture is drawn underneath when available.
    ha_csv = REPORT_DIR / 'head_angle_epochs.csv'
    ha_all = pd.read_csv(ha_csv) if ha_csv.exists() else None
    if ha_all is None:
        print('  (no head_angle_epochs.csv - per-session figures will omit the '
              'posture row; run head_angle_validate.py for it)')
    for lbl in sessions:
        ha = ha_all[ha_all['session'] == lbl].sort_values('t_hr') \
            if ha_all is not None else None
        fig_session(df[df['session'] == lbl], lbl, mus_by_session[lbl],
                    FIG_DIR / f'mean_centred_{lbl}.png', ha=ha)
    fig_grid(df, sessions, mu_tbl, 'CLE-CRE',
             FIG_DIR / 'mean_centred_grid_CLE_CRE.png')
    fig_stage_deviation(dev, FIG_DIR / 'mean_centred_stage_deviation.png')
    fig_operating_points(mu_tbl, FIG_DIR / 'mean_centred_operating_points.png')

    # ── console report: the exact numbers ──
    print('=' * 96)
    print(f'Session means subtracted (sleep period), and excursion about them [{CAP_UNIT}]')
    print('=' * 96)
    show = mu_tbl[['session', 'mean_CLE', 'mean_CRE', 'mean_CH', 'mean_CLE-CRE',
                   'sd_CLE-CRE', 'p5_CLE-CRE', 'p95_CLE-CRE', 'range_CLE-CRE']].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(1)
    print(show.to_string(index=False))

    print(f'\nCohort operating point [{CAP_UNIT}]:')
    for ch, *_ in CHANNELS:
        v = mu_tbl[f'mean_{ch}']
        print(f'  {ch:8s} mean of session means = {v.mean():9,.1f}  '
              f'[{v.min():,.1f}, {v.max():,.1f}]   '
              f'typical sd about own mean = {mu_tbl[f"sd_{ch}"].median():7.1f}')

    print(f'\nMedian deviation from the session mean, by stage [{CAP_UNIT}] '
          f'(median across sessions, IQR):')
    hdr = f'  {"channel":10s}' + ''.join(f'{STAGE_LABELS[c]:>18s}' for c in STAGE_LADDER)
    print(hdr)
    for ch, *_ in CHANNELS:
        d = dev[dev['channel'] == ch]
        line = f'  {ch:10s}'
        for sc in STAGE_LADDER:
            v = d.loc[d['stage_code'] == sc, 'dev_fF']
            if len(v):
                q1, q3 = np.percentile(v, [25, 75])
                line += f'{v.median():+8.1f} [{q1:+.0f},{q3:+.0f}]'.rjust(18)
            else:
                line += f'{"--":>18s}'
        print(line)

    print(f'\nTables  -> {REPORT_DIR}')
    print(f'Figures -> {FIG_DIR}')


if __name__ == '__main__':
    main()
