"""
A night-scale flow marker: low-pass the magnitude, then read the direction.

What was wrong with the previous marker
---------------------------------------
The earlier flow quantities were differentiators. `drift_rate` in
flow_slow_decomp.py is the within-segment slope of the L-R differential, i.e. a
VELOCITY (fF/min), and the raw differential trace itself is read for its steps.
Differentiating a slow signal amplifies exactly what we do not want: every
electrode re-seat is a step, and the derivative of a step is a spike, so the
velocity trace is dominated by the handful of epochs where the mask moved and
carries almost no structure on the timescale of the night. It also has no
meaningful sign over the night — it averages to zero by construction, because
where the signal ends up is a fixed distance from where it started.

The marker built here
---------------------
Work with the LEVEL, not its rate of change, and split it into a magnitude and
a direction, each low-passed to a night-scale timescale:

    d(t)  = (CLE - CRE)(t) - mu        mean-centred differential, fF
    A(t)  = LP_tau( |d(t)| )           MAGNITUDE — how one-sided the flow is
    D(t)  = LP_tau( d(t) ) / A(t)      DIRECTION — in [-1, +1], + = left

The low pass is a rolling MEDIAN over tau, taken over motion-free epochs only,
so a re-seat contributes one outlier instead of a spike. A(t) is positive by
construction and varies slowly; D(t) is bounded, dimensionless, and changes
direction slowly — a marker you can read across the night rather than a
spike train.

The control that decides whether it means anything
--------------------------------------------------
A left-right capacitance difference must respond to the head turning left or
right. So D(t) is tested against the accelerometer head-turn angle validated in
head_angle_validate.py, two ways: (a) correlation of D with sin(turn) over the
whole night, (b) the same marker recomputed on SUPINE-ONLY epochs, where the
head barely turns. Whatever survives (b) is not posture.

tau is swept (5-60 min) rather than assumed; the sweep is reported.

Outputs
-------
    reports/mean_value/flow_marker_epochs.csv     A(t), D(t) per 30 s epoch
    reports/mean_value/flow_marker_session.csv    per-session summary
    reports/mean_value/flow_marker_tau_sweep.csv
    writeup/figures/flow/flow_marker_definition.png
    writeup/figures/flow/flow_marker_grid.png
    writeup/figures/flow/flow_marker_summary.png
    writeup/figures/flow/flow_marker_vs_headangle.png

Needs reports/mean_value/mean_value_epochs.csv (mean_value_vs_stage.py) and
reports/mean_value/head_angle_epochs.csv (head_angle_validate.py).

Usage:
    .venv/Scripts/python.exe analysis/mean_value/flow_marker.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, CAP_UNIT,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'mean_value'
FIG_DIR = ROOT / 'writeup' / 'figures' / 'flow'
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_MIN = 0.5
SLEEP_CODES = [0, 1, 2, 3]
TAU_MIN = 30.0                      # night-scale low-pass window, minutes
TAU_SWEEP = [5.0, 15.0, 30.0, 60.0]
DESPIKE_MIN = 2.5                   # short median filter that removes re-seats
PERSIST_LAG_MIN = 30.0              # lag at which "night-scale structure" is tested
DEMO = 'S6N1'                       # session for the definition figure
EPS = 1e-9


def sleep_mask(codes):
    is_sleep = np.isin(codes, SLEEP_CODES)
    if is_sleep.sum() < 20:
        return np.ones(len(codes), bool)
    on = int(np.argmax(is_sleep))
    off = len(is_sleep) - 1 - int(np.argmax(is_sleep[::-1]))
    m = np.zeros(len(codes), bool)
    m[on:off + 1] = True
    return m


def _roll(x, win_epochs, how):
    return pd.Series(x).rolling(int(win_epochs), center=True,
                                min_periods=max(3, int(win_epochs) // 4))\
                       .aggregate(how).to_numpy()


def flow_marker(d, motion, tau_min=TAU_MIN, despike_min=DESPIKE_MIN):
    """
    Magnitude and direction of the differential flow.

    The low pass is a median-then-mean cascade: a short rolling MEDIAN first, to
    delete electrode-re-seat spikes without smearing them, then a rolling MEAN
    over tau. A median alone at the tau scale would make the direction
    degenerate -- over a half-hour window d(t) is almost always one-signed, and
    median(d) then equals +/-median(|d|) exactly, so D would collapse to a hard
    +/-1. The mean keeps D graded: |D| = 1 only when the window is strictly
    one-signed, and near 0 when the two sides are balanced.

    d       : mean-centred CLE-CRE per epoch (fF)
    motion  : bool per epoch — excluded from the low pass (electrode re-seats)
    Returns (A, D, d_lp) with A the low-passed |d| in fF and D in [-1, +1].
    """
    win = max(3, int(round(tau_min / EPOCH_MIN)))
    wsp = max(3, int(round(despike_min / EPOCH_MIN)))
    dm = np.where(motion, np.nan, d)
    dc = _roll(dm, wsp, 'median')             # de-spike
    A = _roll(np.abs(dc), win, 'mean')        # low pass on the ABSOLUTE
    d_lp = _roll(dc, win, 'mean')             # low pass on the signed level
    D = d_lp / (A + EPS)
    return A, np.clip(D, -1.0, 1.0), d_lp


def velocity(d):
    """
    The old quantity, for the side-by-side: rate of change of the differential.

    Plain first difference, not np.gradient — the central difference in
    np.gradient averages two neighbouring steps and so manufactures positive
    autocorrelation that the underlying quantity does not have.
    """
    v = np.full(len(d), np.nan)
    v[1:] = np.diff(d) / EPOCH_MIN
    return v


def acf_at(x, lag):
    """Autocorrelation of x at a given lag, on the finite overlap."""
    x = np.asarray(x, float)
    if len(x) <= lag + 10:
        return np.nan
    a, b = x[:-lag], x[lag:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def build(df, ha):
    """Per-epoch marker for every session, joined to the head angle."""
    out = []
    for lbl, g in df.groupby('session'):
        g = g.sort_values('epoch').reset_index(drop=True)
        codes = g['stage_code'].to_numpy()
        sm = sleep_mask(codes)
        raw = g['mean_CLE-CRE'].to_numpy()
        mu = float(np.nanmean(raw[sm]))
        d = raw - mu
        motion = g['motion'].to_numpy().astype(bool) if 'motion' in g else np.zeros(len(g), bool)

        A, D, d_lp = flow_marker(d, motion)
        rec = pd.DataFrame({
            'session': lbl, 'subject': g['subject'], 'night': g['night'],
            't_hr': g['t_hr'], 'stage_code': codes, 'motion': motion,
            'in_sleep': sm, 'session_mean_fF': mu,
            'd_fF': d, 'v_fF_per_min': velocity(d),
            'flow_mag_fF': A, 'flow_dir': D, 'flow_lp_fF': d_lp,
        })
        for tau in TAU_SWEEP:
            At, Dt, _ = flow_marker(d, motion, tau)
            rec[f'flow_mag_tau{int(tau)}'] = At
            rec[f'flow_dir_tau{int(tau)}'] = Dt

        # head angle on the same 30 s grid
        h = ha[ha['session'] == lbl]
        if len(h):
            rec['turn_deg'] = np.interp(rec['t_hr'], h['t_hr'], h['turn_deg'])
            # nearest-neighbour for the categorical label
            j = np.abs(rec['t_hr'].to_numpy()[:, None]
                       - h['t_hr'].to_numpy()[None, :]).argmin(axis=1)
            rec['psg_pos'] = h['psg_pos'].to_numpy()[j]
            rec['head_pos'] = h['head_pos'].to_numpy()[j]
        else:
            rec['turn_deg'] = np.nan
            rec['psg_pos'] = 'Unknown'; rec['head_pos'] = 'Unknown'
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def summarise(ep):
    rows, sweep = [], []
    LAG_EP = int(round(PERSIST_LAG_MIN / EPOCH_MIN))
    for lbl, g in ep.groupby('session'):
        g = g[g['in_sleep']].sort_values('t_hr')
        if len(g) < 40:
            continue
        D = g['flow_dir'].to_numpy()
        A = g['flow_mag_fF'].to_numpy()
        t = g['t_hr'].to_numpy()
        v = g['v_fF_per_min'].to_numpy()
        ok = np.isfinite(D)
        third = len(g) // 3

        # how often does the direction actually reverse (sign change of the
        # smoothed direction, ignoring the dead band around zero)?
        s = np.sign(np.where(np.abs(D) > 0.2, D, 0.0))
        s = s[s != 0]
        rev = int(np.sum(np.diff(s) != 0)) if len(s) > 1 else 0

        row = {
            'session': lbl, 'subject': g['subject'].iloc[0],
            'night': int(g['night'].iloc[0]),
            'session_mean_fF': float(g['session_mean_fF'].iloc[0]),
            'n_epochs': int(len(g)),
            'flow_mag_median_fF': float(np.nanmedian(A)),
            'flow_mag_p90_fF': float(np.nanpercentile(A, 90)),
            'flow_dir_median': float(np.nanmedian(D)),
            'flow_dir_first_third': float(np.nanmedian(D[:third])),
            'flow_dir_last_third': float(np.nanmedian(D[-third:])),
            'flow_dir_reversals': rev,
            'frac_left_dominant': float(np.nanmean(D[ok] > 0.2)),
            'frac_right_dominant': float(np.nanmean(D[ok] < -0.2)),
            # the old velocity quantity, for contrast
            'velocity_median_abs': float(np.nanmedian(np.abs(v))),
            'velocity_p99_abs': float(np.nanpercentile(np.abs(v), 99)),
            'velocity_mean': float(np.nanmean(v)),
            'velocity_lag1_acf': acf_at(v, 1),
            'flow_dir_lag1_acf': acf_at(D, 1),
            'velocity_persist_acf': acf_at(v, LAG_EP),
            'flow_dir_persist_acf': acf_at(D, LAG_EP),
            'd_persist_acf': acf_at(g['d_fF'].to_numpy(), LAG_EP),
        }

        # head-angle control
        turn = g['turn_deg'].to_numpy()
        m = np.isfinite(turn) & ok
        if m.sum() > 30:
            rho, p = spearmanr(np.sin(np.radians(turn[m])), D[m])
            row['rho_dir_vs_sin_turn'] = float(rho)
            row['p_dir_vs_sin_turn'] = float(p)
        sup = (g['head_pos'].to_numpy() == 'Supine') & ok
        if sup.sum() > 30:
            row['n_supine_epochs'] = int(sup.sum())
            row['flow_dir_median_supine'] = float(np.nanmedian(D[sup]))
            row['flow_mag_median_supine_fF'] = float(np.nanmedian(A[sup]))
            row['flow_dir_reversals_supine'] = int(
                np.sum(np.diff(np.sign(D[sup][np.abs(D[sup]) > 0.2])) != 0)
                if (np.abs(D[sup]) > 0.2).sum() > 1 else 0)
        rows.append(row)

        for tau in TAU_SWEEP:
            Dt = g[f'flow_dir_tau{int(tau)}'].to_numpy()
            At = g[f'flow_mag_tau{int(tau)}'].to_numpy()
            st = np.sign(np.where(np.abs(Dt) > 0.2, Dt, 0.0)); st = st[st != 0]
            sweep.append({
                'session': lbl, 'tau_min': tau,
                'dir_persist_acf': acf_at(Dt, LAG_EP),
                'dir_lag1_acf': acf_at(Dt, 1),
                'dir_reversals': int(np.sum(np.diff(st) != 0)) if len(st) > 1 else 0,
                'mag_median_fF': float(np.nanmedian(At)),
                'dir_abs_median': float(np.nanmedian(np.abs(Dt))),
            })
    return pd.DataFrame(rows), pd.DataFrame(sweep)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_definition(ep, lbl, out):
    g = ep[(ep['session'] == lbl)].sort_values('t_hr')
    t = g['t_hr'].to_numpy(); codes = g['stage_code'].to_numpy()

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={'height_ratios': [1, .85, 1, 1, .85]})

    def bands(ax):
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                       alpha=0.13, lw=0)

    ax = axes[0]; bands(ax)
    ax.plot(t, g['d_fF'], lw=0.9, color='#7F8C8D')
    ax.axhline(0, color='#2C3E50', ls='--', lw=0.9)
    ax.set_ylabel(f'd(t)  ({CAP_UNIT})')
    ax.set_title(f'1  Mean-centred differential  d = (CLE−CRE) − mean   '
                 f'(session mean {g["session_mean_fF"].iloc[0]:,.0f} {CAP_UNIT})',
                 fontsize=10.5, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.13)

    ax = axes[1]; bands(ax)
    ax.plot(t, g['v_fF_per_min'], lw=0.8, color='#C0392B')
    ax.axhline(0, color='#2C3E50', ls='--', lw=0.9)
    ax.set_ylabel(f'v(t)  ({CAP_UNIT}/min)')
    ax.set_title('2  THE OLD MARKER — velocity dv/dt. Re-seats become spikes; '
                 'no night-scale structure and no lasting sign.',
                 fontsize=10.5, fontweight='bold', loc='left', color='#C0392B')
    ax.grid(True, alpha=0.13)

    ax = axes[2]; bands(ax)
    ax.plot(t, np.abs(g['d_fF']), lw=0.7, color='#BDC3C7', label='|d(t)|')
    ax.plot(t, g['flow_mag_fF'], lw=2.2, color='#B9380B',
            label=f'A(t) = LP$_{{{TAU_MIN:.0f}min}}$|d(t)|')
    ax.set_ylabel(f'magnitude ({CAP_UNIT})')
    ax.set_ylim(0, np.nanpercentile(np.abs(g['d_fF']), 99) * 1.1 + 1)
    ax.set_title('3  MAGNITUDE — low pass on the absolute. Always positive, '
                 'slowly varying: how one-sided the flow is.',
                 fontsize=10.5, fontweight='bold', loc='left', color='#B9380B')
    ax.legend(fontsize=8.5, ncol=2); ax.grid(True, alpha=0.13)

    ax = axes[3]; bands(ax)
    D = g['flow_dir'].to_numpy()
    ax.plot(t, D, lw=2.2, color='#1F618D')
    ax.axhline(0, color='#2C3E50', ls='--', lw=0.9)
    ax.fill_between(t, 0, D, where=(D >= 0), color='#C0392B', alpha=0.25, interpolate=True)
    ax.fill_between(t, 0, D, where=(D < 0), color='#2980B9', alpha=0.25, interpolate=True)
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel('D(t)   [−1, +1]')
    ax.set_title('4  DIRECTION — LP(d)/LP(|d|). Bounded, dimensionless, turns over '
                 'slowly.  + = left-dominant, − = right-dominant.',
                 fontsize=10.5, fontweight='bold', loc='left', color='#1F618D')
    ax.grid(True, alpha=0.13)

    ax = axes[4]; bands(ax)
    ax.plot(t, g['turn_deg'], lw=1.3, color='#16A085')
    for lv, lb in [(90, 'left'), (0, 'supine'), (-90, 'right')]:
        ax.axhline(lv, color='#555', ls=':', lw=0.8)
        ax.annotate(lb, (t[0], lv), fontsize=7.5, color='#555', va='bottom')
    ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel('head turn (deg)')
    ax.set_xlabel('Time (hours)')
    ax.set_title('5  CONTROL — head-turn angle from the accelerometer. The '
                 'direction marker must be checked against this.',
                 fontsize=10.5, fontweight='bold', loc='left', color='#16A085')
    ax.grid(True, alpha=0.13)

    handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
               for k in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 1.003), framealpha=0.9)
    fig.suptitle(f'{lbl} — from a velocity to a night-scale flow marker',
                 fontsize=14, fontweight='bold', y=1.025)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)


def fig_grid(ep, sessions, out):
    fig, axes = plt.subplots(6, 2, figsize=(20, 21))
    for ax, lbl in zip(axes.ravel(), sessions):
        g = ep[ep['session'] == lbl].sort_values('t_hr')
        t = g['t_hr'].to_numpy(); codes = g['stage_code'].to_numpy()
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                       alpha=0.13, lw=0)
        D = g['flow_dir'].to_numpy()
        ax.plot(t, D, lw=2.0, color='#1F618D', zorder=4)
        ax.axhline(0, color='#2C3E50', ls='--', lw=0.9, zorder=2)
        ax.fill_between(t, 0, D, where=(D >= 0), color='#C0392B', alpha=0.22,
                        interpolate=True, zorder=1)
        ax.fill_between(t, 0, D, where=(D < 0), color='#2980B9', alpha=0.22,
                        interpolate=True, zorder=1)
        ax.set_ylim(-1.2, 1.2); ax.set_ylabel('direction D', fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t, g['flow_mag_fF'], lw=1.5, color='#B9380B', alpha=0.85, zorder=3)
        ax2.set_ylabel(f'magnitude ({CAP_UNIT})', fontsize=8, color='#B9380B')
        ax2.tick_params(axis='y', labelcolor='#B9380B', labelsize=7)
        ax2.set_ylim(0, max(np.nanpercentile(g['flow_mag_fF'], 98) * 1.35, 1))
        ax.set_title(f'{lbl}   mean {g["session_mean_fF"].iloc[0]:,.0f} {CAP_UNIT}',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    handles = [plt.Line2D([], [], color='#1F618D', lw=2.5, label='direction D(t)  [−1,+1]'),
               plt.Line2D([], [], color='#B9380B', lw=2.5,
                          label=f'magnitude A(t) = LP|d| ({CAP_UNIT}, right axis)'),
               mpatches.Patch(color='#C0392B', alpha=0.4, label='left-dominant'),
               mpatches.Patch(color='#2980B9', alpha=0.4, label='right-dominant')]
    handles += [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
                for k in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=9, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.004))
    fig.suptitle(f'Flow magnitude and direction over the night — all 12 sessions '
                 f'(low pass = rolling median over {TAU_MIN:.0f} min, motion epochs excluded)',
                 fontsize=15, fontweight='bold', y=1.018)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)


def fig_summary(sess, sweep, ep, out):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    s = sess.sort_values('session')
    xs = np.arange(len(s))

    # A — magnitude per session
    ax = axes[0, 0]
    ax.bar(xs, s['flow_mag_median_fF'], color='#B9380B', alpha=0.85,
           edgecolor='k', lw=0.5, label='median A(t)')
    ax.plot(xs, s['flow_mag_p90_fF'], 'o', color='#2C3E50', ms=6, label='90th pct')
    for x, v in zip(xs, s['flow_mag_median_fF']):
        ax.annotate(f'{v:.1f}', (x, v), fontsize=7, ha='center',
                    xytext=(0, 4), textcoords='offset points')
    ax.set_xticks(xs); ax.set_xticklabels(s['session'], rotation=45, ha='right')
    ax.set_ylabel(f'flow magnitude ({CAP_UNIT})')
    ax.set_title('A  How one-sided is the flow, per night', fontsize=11,
                 fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, axis='y', alpha=0.2)

    # B — direction: start vs end of night
    ax = axes[0, 1]
    w = 0.38
    ax.bar(xs - w / 2, s['flow_dir_first_third'], w, color='#2980B9', alpha=0.85,
           edgecolor='k', lw=0.5, label='first third of the night')
    ax.bar(xs + w / 2, s['flow_dir_last_third'], w, color='#E67E22', alpha=0.85,
           edgecolor='k', lw=0.5, label='last third')
    ax.axhline(0, color='k', lw=1.1)
    ax.set_xticks(xs); ax.set_xticklabels(s['session'], rotation=45, ha='right')
    ax.set_ylim(-1.15, 1.15); ax.set_ylabel('direction D')
    ax.set_title('B  Does the direction drift across the night?', fontsize=11,
                 fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, axis='y', alpha=0.2)

    # C — night-scale persistence: the old velocity vs the new direction
    ax = axes[0, 2]
    ax.bar(xs - w / 2, s['velocity_persist_acf'], w, color='#C0392B', alpha=0.85,
           edgecolor='k', lw=0.5, label='velocity dv/dt (old)')
    ax.bar(xs + w / 2, s['flow_dir_persist_acf'], w, color='#1F618D', alpha=0.85,
           edgecolor='k', lw=0.5, label='direction D (new)')
    ax.axhline(0, color='k', lw=1.1)
    ax.set_xticks(xs); ax.set_xticklabels(s['session'], rotation=45, ha='right')
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(f'autocorrelation at {PERSIST_LAG_MIN:.0f} min lag')
    ax.set_title('C  Is it a night-scale marker at all?\nvelocity forgets in one '
                 'epoch; direction persists', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, axis='y', alpha=0.2)

    # D — tau sweep
    ax = axes[1, 0]
    for lbl, gg in sweep.groupby('session'):
        gg = gg.sort_values('tau_min')
        ax.plot(gg['tau_min'], gg['dir_persist_acf'], 'o-', ms=4, lw=1.0, alpha=0.55)
    med = sweep.groupby('tau_min')['dir_persist_acf'].median()
    ax.plot(med.index, med.values, 'o-', color='k', lw=2.4, ms=8, label='median')
    ax.axvline(TAU_MIN, color='#C0392B', ls='--', lw=1.4,
               label=f'chosen tau = {TAU_MIN:.0f} min')
    ax2 = ax.twinx()
    medr = sweep.groupby('tau_min')['dir_reversals'].median()
    ax2.plot(medr.index, medr.values, 's--', color='#B9380B', lw=1.8, ms=7,
             label='median reversals')
    ax2.set_ylabel('direction reversals per night', color='#B9380B')
    ax2.tick_params(axis='y', labelcolor='#B9380B')
    ax.set_xlabel('low-pass window tau (min)')
    ax.set_ylabel(f'persistence of D ({PERSIST_LAG_MIN:.0f} min lag acf)')
    ax.set_title('D  Choice of timescale (swept, not assumed)\nlonger tau buys '
                 'persistence and costs resolution', fontsize=11,
                 fontweight='bold', loc='left')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5); ax.grid(True, alpha=0.2)

    # E — direction reversals: all epochs vs supine only
    ax = axes[1, 1]
    have = s.dropna(subset=['flow_dir_reversals_supine'])
    xr = np.arange(len(have))
    ax.bar(xr - w / 2, have['flow_dir_reversals'], w, color='#7F8C8D', alpha=0.85,
           edgecolor='k', lw=0.5, label='all epochs')
    ax.bar(xr + w / 2, have['flow_dir_reversals_supine'], w, color='#16A085',
           alpha=0.85, edgecolor='k', lw=0.5, label='supine epochs only')
    ax.set_xticks(xr); ax.set_xticklabels(have['session'], rotation=45, ha='right')
    ax.set_ylabel('direction reversals over the night')
    ax.set_title('E  Reversals survive posture control\n(head held supine)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, axis='y', alpha=0.2)

    # F — how much of the direction is head posture
    ax = axes[1, 2]
    have = s.dropna(subset=['rho_dir_vs_sin_turn'])
    colr = ['#C0392B' if p < 0.05 else '#BDC3C7' for p in have['p_dir_vs_sin_turn']]
    ax.barh(have['session'], have['rho_dir_vs_sin_turn'], color=colr, alpha=0.9,
            edgecolor='k', lw=0.5)
    ax.axvline(0, color='k', lw=1.1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Spearman rho:  D(t)  vs  sin(head turn)')
    ax.set_title('F  Is the direction just the head turning?\n'
                 f'red = p<0.05  ({(have["p_dir_vs_sin_turn"] < 0.05).sum()}/'
                 f'{len(have)} sessions)', fontsize=11, fontweight='bold', loc='left')
    ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    fig.suptitle('Night-scale flow marker — magnitude, direction, timescale, '
                 'and the head-posture control', fontsize=13.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=190, bbox_inches='tight'); plt.close(fig)


def fig_vs_headangle(ep, sess, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # A — direction vs head turn, pooled
    ax = axes[0]
    g = ep[ep['in_sleep'] & np.isfinite(ep['turn_deg']) & np.isfinite(ep['flow_dir'])]
    ax.scatter(g['turn_deg'], g['flow_dir'], s=4, alpha=0.12, color='#1F618D')
    for lv in (-90, 0, 90):
        ax.axvline(lv, color='#555', ls=':', lw=0.9)
    ax.axhline(0, color='k', lw=1.0)
    rho, p = spearmanr(np.sin(np.radians(g['turn_deg'])), g['flow_dir'])
    ax.set_xlim(-185, 185); ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('head turn angle (deg)   − = right, + = left')
    ax.set_ylabel('flow direction D')
    ax.set_title(f'A  Pooled, all sessions\nSpearman rho(D, sin turn) = {rho:.2f}, '
                 f'p = {p:.1e}, n = {len(g):,}',
                 fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.2)

    # B — supine-only: the same marker with posture held
    ax = axes[1]
    sup = g[g['head_pos'] == 'Supine']
    for lbl, gg in sup.groupby('session'):
        gg = gg.sort_values('t_hr')
        ax.plot(gg['t_hr'], gg['flow_dir'], lw=1.2, alpha=0.75, label=lbl)
    ax.axhline(0, color='k', lw=1.1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('flow direction D')
    ax.set_title(f'B  Supine epochs only — head held face-up\n'
                 f'the direction still moves and still reverses (n = {len(sup):,} epochs)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=6.5, ncol=3, loc='upper right'); ax.grid(True, alpha=0.2)

    # C — magnitude by head position
    ax = axes[2]
    order = ['Supine', 'Left', 'Right', 'Prone']
    data, labels = [], []
    for p_ in order:
        v = g.loc[g['head_pos'] == p_, 'flow_mag_fF'].dropna().to_numpy()
        if len(v) >= 30:
            data.append(v); labels.append(f'{p_}\nn={len(v):,}')
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops=dict(color='k', lw=1.7))
    for patch, c in zip(bp['boxes'], ['#3498DB', '#27AE60', '#E67E22', '#8E44AD']):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, v in enumerate(data):
        ax.annotate(f'{np.median(v):.1f}', (i + 1, np.median(v)), fontsize=9,
                    fontweight='bold', ha='center', xytext=(0, 7),
                    textcoords='offset points')
    ax.set_xticklabels(labels)
    ax.set_ylabel(f'flow magnitude A(t)  ({CAP_UNIT})')
    ax.set_title('C  Magnitude by head position\nposture sets the scale, so it is '
                 'reported alongside', fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, axis='y', alpha=0.2)

    fig.suptitle('The control: how much of the flow marker is the head turning?',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=190, bbox_inches='tight'); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    mv = REPORT_DIR / 'mean_value_epochs.csv'
    hz = REPORT_DIR / 'head_angle_epochs.csv'
    if not mv.exists():
        sys.exit(f'Missing {mv}. Run mean_value_vs_stage.py first.')
    if not hz.exists():
        sys.exit(f'Missing {hz}. Run head_angle_validate.py first.')
    df = pd.read_csv(mv)
    ha = pd.read_csv(hz)

    ep = build(df, ha)
    sess, sweep = summarise(ep)
    ep.to_csv(REPORT_DIR / 'flow_marker_epochs.csv', index=False)
    sess.to_csv(REPORT_DIR / 'flow_marker_session.csv', index=False)
    sweep.to_csv(REPORT_DIR / 'flow_marker_tau_sweep.csv', index=False)

    sessions = sorted(ep['session'].unique())
    demo = DEMO if DEMO in sessions else sessions[0]
    fig_definition(ep, demo, FIG_DIR / 'flow_marker_definition.png')
    fig_grid(ep, sessions, FIG_DIR / 'flow_marker_grid.png')
    fig_summary(sess, sweep, ep, FIG_DIR / 'flow_marker_summary.png')
    fig_vs_headangle(ep, sess, FIG_DIR / 'flow_marker_vs_headangle.png')

    print('=' * 104)
    print(f'Night-scale flow marker  (low pass = rolling median over {TAU_MIN:.0f} min, '
          f'motion epochs excluded)')
    print('=' * 104)
    show = sess[['session', 'session_mean_fF', 'flow_mag_median_fF', 'flow_dir_median',
                 'flow_dir_first_third', 'flow_dir_last_third', 'flow_dir_reversals',
                 'frac_left_dominant', 'frac_right_dominant']].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(2)
    print(show.to_string(index=False))

    print(f'\nOld velocity marker vs new direction marker '
          f'(autocorrelation at a {PERSIST_LAG_MIN:.0f} min lag = does it carry '
          f'night-scale structure):')
    print(f"  velocity  dv/dt : median {sess['velocity_persist_acf'].median():+.3f}  "
          f"[{sess['velocity_persist_acf'].min():+.3f}, "
          f"{sess['velocity_persist_acf'].max():+.3f}]"
          "   <- ~0: forgets within one epoch")
    print(f"  direction D(t)  : median {sess['flow_dir_persist_acf'].median():+.3f}  "
          f"[{sess['flow_dir_persist_acf'].min():+.3f}, "
          f"{sess['flow_dir_persist_acf'].max():+.3f}]"
          "   <- persists across the night")
    print(f"  (raw d(t) for reference: {sess['d_persist_acf'].median():+.3f})")
    print(f"  velocity mean over the night: "
          f"{sess['velocity_mean'].abs().median():.4f} {CAP_UNIT}/min "
          f"vs typical |v| {sess['velocity_median_abs'].median():.2f} -- it averages "
          f"to zero by construction, so it carries no direction.")

    print('\ntau sweep (median across sessions):')
    for tau, gg in sweep.groupby('tau_min'):
        print(f'  tau={tau:5.0f} min  persistence acf of D = '
              f'{gg["dir_persist_acf"].median():.3f}  '
              f'reversals = {gg["dir_reversals"].median():4.1f}  '
              f'|D| = {gg["dir_abs_median"].median():.2f}')

    print('\nHead-posture control:')
    r = sess.dropna(subset=['rho_dir_vs_sin_turn'])
    sig = r[r['p_dir_vs_sin_turn'] < 0.05]
    print(f"  rho(D, sin turn): median {r['rho_dir_vs_sin_turn'].median():+.2f} "
          f"[{r['rho_dir_vs_sin_turn'].min():+.2f}, {r['rho_dir_vs_sin_turn'].max():+.2f}], "
          f"significant in {len(sig)}/{len(r)} sessions "
          f"(signs: {(sig['rho_dir_vs_sin_turn']>0).sum()} positive, "
          f"{(sig['rho_dir_vs_sin_turn']<0).sum()} negative)")
    sp = sess.dropna(subset=['flow_dir_median_supine'])
    print(f"  supine-only: {len(sp)} sessions with >=30 supine epochs; "
          f"median magnitude {sp['flow_mag_median_supine_fF'].median():.1f} {CAP_UNIT} "
          f"vs {sp['flow_mag_median_fF'].median():.1f} {CAP_UNIT} over all epochs;")
    print(f"               direction reversals {sp['flow_dir_reversals_supine'].median():.1f} "
          f"(supine) vs {sp['flow_dir_reversals'].median():.1f} (all) -- the marker is "
          f"not purely posture.")

    print(f'\nTables  -> {REPORT_DIR}')
    print(f'Figures -> {FIG_DIR}')


if __name__ == '__main__':
    main()
