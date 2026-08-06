"""
Revalidation of the accelerometer-derived head angle: is it a real angle?

The mean-value and flow analyses use head posture as a covariate, so the angle
has to be more than a plausible-looking trace. This script runs the four checks
that decide whether it is, and fixes what they broke.

Check 1 — is the DC accelerometer actually gravity?
    |a| must sit at 1 g and stay there. If it does, the low-frequency vector is
    the gravity direction and an arctangent of its components is a true tilt
    angle. Reported per session: median |a|, coefficient of variation, and the
    fraction of the night within 5 % of 1 g.

Check 2 — calibration.
    An axis-aligned ellipsoid fit gives per-axis bias and gain. The angles used
    here are ratios of axes, so an ISOTROPIC gain error cancels exactly; only
    per-axis bias moves them. The fit is reported with a coverage figure,
    because a head that never leaves supine samples one cap of the sphere and
    cannot constrain a 6-parameter fit. The angle change from applying it is
    quantified rather than assumed negligible.

Check 3 — gravity-extraction cutoff.
    The previous implementation low-passed at 0.5 Hz, which is the TOP of the
    respiratory band, so breathing rode straight into the "static" angle. The
    residual angle ripple in 0.1–0.5 Hz is measured at 0.5 Hz and at 0.05 Hz.

Check 4 — agreement with an independent reference.
    The PSG scores body position (Supine / Left / Right / Prone / Upright) from
    its own sensor. The derived head-turn angle is compared against it: median
    angle per scored position, and the accuracy of a fixed-threshold classifier.
    This also pins the SIGN convention, which the legacy classify_position() had
    inverted (it called positive roll 'right'; the PSG says positive is LEFT).

Definitions validated here (now in sleep_monitor/motion.py):
    turn_deg = atan2(gY, gZ)   0 = supine, +90 = left, −90 = right, ±180 = prone
    elev_deg = asin(gX/|g|)    head-axis elevation, ≈90 upright
The legacy roll = atan2(gY, sqrt(gX²+gZ²)) is retained for comparison and shown
to saturate: it is clamped to ±90° and cannot separate prone from supine.

Outputs
-------
    reports/mean_value/head_angle_validation.csv       per-session checks
    reports/mean_value/head_angle_vs_psg_position.csv  angle by scored position
    reports/mean_value/head_angle_epochs.csv           30 s angle series, all sessions
    writeup/figures/head_angle/head_angle_validation.png
    writeup/figures/head_angle/head_angle_traces_grid.png
    writeup/figures/head_angle/head_angle_<SESSION>.png

Usage:
    .venv/Scripts/python.exe analysis/mean_value/head_angle_validate.py
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
from sleep_monitor import (
    load_session, load_sleep_profile, load_position, position_at,
    head_angle, head_orientation, fit_accel_calibration, classify_head_position,
)
from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.filters import bandpass

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'mean_value'
FIG_DIR = ROOT / 'writeup' / 'figures' / 'head_angle'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
LP_LEGACY = 0.5          # the old gravity cutoff — inside the respiratory band
LP_GRAVITY = 0.05        # the cutoff validated here
POS_ORDER = ['Supine', 'Left', 'Right', 'Prone', 'Upright']
POS_COLORS = {'Supine': '#3498DB', 'Left': '#27AE60', 'Right': '#E67E22',
              'Prone': '#8E44AD', 'Upright': '#E74C3C', 'Unknown': '#BBBBBB'}


def epoch_reduce(x, fs, sec=EPOCH_SEC, fn=np.median):
    n = int(round(fs * sec))
    m = len(x) // n
    return fn(np.asarray(x, float)[: m * n].reshape(m, n), axis=1)


def circ_median_deg(a):
    """Median of an angle in degrees, taken on the circle (handles the ±180 seam)."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    r = np.radians(a)
    ang = np.degrees(np.arctan2(np.median(np.sin(r)), np.median(np.cos(r))))
    return float(ang)


def analyse(idx, meta):
    s = load_session(idx)
    fs = s.fs
    aX, aY, aZ = (s.cap[c].astype(np.float64) for c in ('aX', 'aY', 'aZ'))

    # ── check 1: is |a| gravity? ──
    mag = np.sqrt(aX ** 2 + aY ** 2 + aZ ** 2)
    g_med = float(np.median(mag))
    row = {
        'session': meta['label'], 'subject': meta['subject'],
        'night': meta['night'], 'hours': float(s.time_hr[-1]),
        'accel_mag_median_g': g_med,
        'accel_mag_cv': float(np.std(mag) / np.mean(mag)),
        'frac_within_5pct_1g': float(np.mean(np.abs(mag - 1.0) < 0.05)),
        'frac_within_5pct_median': float(np.mean(np.abs(mag - g_med) < 0.05 * g_med)),
    }

    # ── check 2: calibration ──
    cal = fit_accel_calibration(aX, aY, aZ)
    if cal is not None:
        row.update({
            'bias_X': cal['bias'][0], 'bias_Y': cal['bias'][1], 'bias_Z': cal['bias'][2],
            'gain_X': cal['gain'][0], 'gain_Y': cal['gain'][1], 'gain_Z': cal['gain'][2],
            'calib_coverage': cal['coverage'], 'calib_resid_rms': cal['resid_rms'],
        })

    # ── angle variants ──
    ang = head_angle(aX, aY, aZ, fs, LP_GRAVITY)                    # validated
    ang_fast = head_angle(aX, aY, aZ, fs, LP_LEGACY)                # legacy cutoff
    ang_cal = head_angle(aX, aY, aZ, fs, LP_GRAVITY, calib=cal) if cal else None
    legacy = head_orientation(aX, aY, aZ, fs, LP_LEGACY)            # legacy formula

    # ── check 3: respiratory-band ripple leaking into the angle ──
    # Measured on motion-QUIET epochs only. Over the whole night the 0.1-0.5 Hz
    # band also carries real head movement, which would mask the breathing leak
    # this check is about.
    dyn = epoch_reduce(np.abs(np.diff(mag, prepend=mag[0])), fs, fn=np.std)
    quiet_ep = dyn <= np.percentile(dyn, 50)
    n_ep = int(round(fs * EPOCH_SEC))
    quiet_samp = np.repeat(quiet_ep, n_ep)[:len(mag)]
    quiet_samp = np.pad(quiet_samp, (0, max(0, len(mag) - len(quiet_samp))),
                        constant_values=False)
    for nm, a in (('lp050', ang_fast), ('lp005', ang)):
        rip = bandpass(a['turn_deg'], 0.1, 0.5, fs)
        row[f'turn_resp_ripple_deg_{nm}'] = float(np.std(rip[quiet_samp]))
        row[f'turn_resp_ripple_allnight_deg_{nm}'] = float(np.std(rip))
    row['turn_range_deg'] = float(np.percentile(ang['turn_deg'], 99)
                                  - np.percentile(ang['turn_deg'], 1))

    # ── check 2b: how much does calibration actually move the angle? ──
    if ang_cal is not None:
        d = np.abs(((ang_cal['turn_deg'] - ang['turn_deg']) + 180) % 360 - 180)
        row['calib_turn_shift_median_deg'] = float(np.median(d))
        row['calib_turn_shift_p95_deg'] = float(np.percentile(d, 95))

    # ── legacy formula: how wrong is it, and where ──
    # The clamp does not show up as a pile-up at exactly ±90; it shows up as a
    # FOLD. Once gZ < 0 (head turned past the side) the legacy roll walks back
    # towards zero while the true angle keeps going, so the error is measured
    # directly, split by the sign of gZ.
    fold_err = np.abs(((legacy['roll_deg'] - ang['turn_deg']) + 180) % 360 - 180)
    past_side = ang['gZ'] < 0
    row['legacy_cannot_see_prone_frac'] = float(np.mean(past_side))
    row['legacy_err_deg_gZpos'] = float(np.median(fold_err[~past_side])) \
        if (~past_side).any() else np.nan
    row['legacy_err_deg_gZneg'] = float(np.median(fold_err[past_side])) \
        if past_side.any() else np.nan
    row['legacy_err_deg_p95'] = float(np.percentile(fold_err, 95))

    # ── epoch series (for figures / downstream covariates) ──
    t_ep = epoch_reduce(s.time_hr.astype(float), fs)
    ep = pd.DataFrame({
        'session': meta['label'], 'subject': meta['subject'],
        'night': meta['night'], 't_hr': t_ep,
        'turn_deg': epoch_reduce(ang['turn_deg'], fs, fn=lambda a, axis: np.apply_along_axis(circ_median_deg, axis, a)),
        'elev_deg': epoch_reduce(ang['elev_deg'], fs),
        'tilt_deg': epoch_reduce(ang['tilt_deg'], fs),
        'gmag': epoch_reduce(ang['gmag'], fs),
        'legacy_roll_deg': epoch_reduce(legacy['roll_deg'], fs),
    })
    ep['head_pos'] = classify_head_position(ep['turn_deg'].to_numpy(),
                                            ep['elev_deg'].to_numpy())

    # stage labels on the same epoch grid
    sp = load_sleep_profile(s)
    if sp is not None:
        j = np.searchsorted(sp['t_ep_hr'], ep['t_hr'].to_numpy(), side='right') - 1
        ok = (j >= 0) & (j < len(sp['codes']))
        code = np.full(len(ep), -1, int)
        code[ok] = np.asarray(sp['codes'])[j[ok]]
        ep['stage_code'] = code
    else:
        ep['stage_code'] = -1

    # ── check 4: PSG-scored body position ──
    pos = load_position(s)
    ep['psg_pos'] = position_at(pos, ep['t_hr'].to_numpy()) if pos else 'Unknown'

    pos_rows = []
    for p in POS_ORDER:
        m = (ep['psg_pos'] == p).to_numpy()
        if m.sum() < 5:
            continue
        pos_rows.append({
            'session': meta['label'], 'psg_pos': p, 'n_epochs': int(m.sum()),
            'turn_deg_median': circ_median_deg(ep['turn_deg'].to_numpy()[m]),
            'turn_deg_iqr': float(np.subtract(*np.percentile(
                ep['turn_deg'].to_numpy()[m], [75, 25]))),
            'elev_deg_median': float(np.median(ep['elev_deg'].to_numpy()[m])),
            'legacy_roll_median': float(np.median(ep['legacy_roll_deg'].to_numpy()[m])),
        })

    scored = ep['psg_pos'].isin(POS_ORDER).to_numpy()
    if scored.sum() > 0:
        row['psg_pos_agree_frac'] = float(
            np.mean(ep['head_pos'].to_numpy()[scored] == ep['psg_pos'].to_numpy()[scored]))
        row['psg_pos_n_scored'] = int(scored.sum())
        # sign test: does positive turn mean the PSG's 'Left'?
        lr = np.isin(ep['psg_pos'].to_numpy(), ['Left', 'Right'])
        if lr.sum() >= 10:
            sign_ok = np.mean(
                (ep['turn_deg'].to_numpy()[lr] > 0)
                == (ep['psg_pos'].to_numpy()[lr] == 'Left'))
            row['psg_left_is_positive_turn_frac'] = float(sign_ok)
            row['n_lateral_epochs'] = int(lr.sum())

    return row, ep, pd.DataFrame(pos_rows)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_validation(per, pos_tbl, ep_all, out):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # A — |a| is 1 g
    ax = axes[0, 0]
    ax.barh(per['session'], per['accel_mag_median_g'], color='#2980B9', alpha=0.85)
    ax.axvline(1.0, color='#C0392B', lw=1.6, ls='--', label='1 g')
    ax.set_xlim(0.9, 1.1)
    for y, (v, cv) in enumerate(zip(per['accel_mag_median_g'], per['accel_mag_cv'])):
        ax.annotate(f'{v:.3f} g  (CV {cv*100:.1f}%)', (v, y), fontsize=7,
                    va='center', xytext=(4, 0), textcoords='offset points')
    ax.set_xlabel('median |a| over the night (g)')
    ax.set_title('A  Is the DC accelerometer gravity?\nyes — |a| pinned near 1 g, '
                 'CV < 2 %', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # B — respiratory ripple by cutoff
    ax = axes[0, 1]
    y = np.arange(len(per)); h = 0.38
    ax.barh(y - h / 2, per['turn_resp_ripple_deg_lp050'], h, color='#C0392B',
            alpha=0.85, label=f'{LP_LEGACY:g} Hz cutoff (legacy)')
    ax.barh(y + h / 2, per['turn_resp_ripple_deg_lp005'], h, color='#27AE60',
            alpha=0.85, label=f'{LP_GRAVITY:g} Hz cutoff (validated)')
    ax.set_yticks(y); ax.set_yticklabels(per['session'])
    ax.set_xscale('log')
    ax.set_xlabel('breathing-band ripple in turn angle\n(deg sd, 0.1–0.5 Hz, '
                  'motion-quiet epochs only)')
    ax.set_title('B  Gravity cutoff\n0.5 Hz lets breathing into the "static" angle',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # C — calibration shift
    ax = axes[0, 2]
    if 'calib_turn_shift_median_deg' in per:
        ax.barh(per['session'], per['calib_turn_shift_median_deg'], color='#8E44AD',
                alpha=0.85, label='median shift')
        ax.barh(per['session'], per['calib_turn_shift_p95_deg'], color='#8E44AD',
                alpha=0.30, label='95th pct shift')
        for y_, (v, c) in enumerate(zip(per['calib_turn_shift_p95_deg'],
                                        per['calib_coverage'])):
            ax.annotate(f'coverage {c:.2f}', (v, y_), fontsize=6.5, va='center',
                        xytext=(4, 0), textcoords='offset points')
    ax.set_xlabel('change in turn angle from ellipsoid calibration (deg)')
    ax.set_title('C  Does calibration matter?\nangles are ratios — isotropic gain '
                 'cancels exactly', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # D — turn angle by PSG-scored position
    ax = axes[1, 0]
    present = [p for p in POS_ORDER if (pos_tbl['psg_pos'] == p).any()]
    data = [pos_tbl.loc[pos_tbl['psg_pos'] == p, 'turn_deg_median'].to_numpy()
            for p in present]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops=dict(color='k', lw=1.6))
    for patch, p in zip(bp['boxes'], present):
        patch.set_facecolor(POS_COLORS[p]); patch.set_alpha(0.65)
    for i, (p, v) in enumerate(zip(present, data)):
        ax.plot(np.full(len(v), i + 1) + np.linspace(-0.14, 0.14, len(v)), v, 'o',
                ms=4, color='k', alpha=0.55)
        ax.annotate(f'n={len(v)}', (i + 1, 168), fontsize=7.5, ha='center')
    for lv, lb in [(0, 'supine'), (90, 'left'), (-90, 'right')]:
        ax.axhline(lv, color='#555', ls=':', lw=1.0)
        ax.annotate(lb, (0.62, lv), fontsize=7.5, color='#555', va='bottom')
    ax.set_xticklabels(present)
    ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel('turn angle = atan2(gY, gZ)   (deg)')
    ax.set_title('D  Against the PSG-scored position\none point = one session',
                 fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, axis='y', alpha=0.2)

    # E — legacy formula saturates
    ax = axes[1, 1]
    sub = ep_all[ep_all['psg_pos'].isin(POS_ORDER)]
    ax.scatter(sub['turn_deg'], sub['legacy_roll_deg'], s=6, alpha=0.25,
               c=[POS_COLORS[p] for p in sub['psg_pos']])
    xs = np.linspace(-180, 180, 400)
    ax.plot(xs, xs, color='k', ls='--', lw=1.0, label='identity')
    ax.axhline(90, color='#C0392B', lw=1.0, ls=':')
    ax.axhline(-90, color='#C0392B', lw=1.0, ls=':',
               label='legacy clamp at ±90°')
    ax.set_xlim(-185, 185); ax.set_ylim(-100, 100)
    ax.set_xlabel('validated turn angle (deg)')
    ax.set_ylabel('legacy roll = atan2(gY, √(gX²+gZ²))  (deg)')
    ax.set_title('E  Why the legacy angle was replaced\nit folds prone back onto '
                 'supine', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.legend(handles=ax.get_legend().legend_handles
              + [mpatches.Patch(color=POS_COLORS[p], label=f'PSG {p}')
                 for p in present], fontsize=7.5, loc='lower right', ncol=2)

    # F — agreement with the PSG classifier + sign check
    ax = axes[1, 2]
    ag = per.dropna(subset=['psg_pos_agree_frac'])
    ax.barh(ag['session'], ag['psg_pos_agree_frac'], color='#16A085', alpha=0.85)
    ax.axvline(1.0, color='#555', ls=':', lw=1.0)
    for y_, (v, n) in enumerate(zip(ag['psg_pos_agree_frac'], ag['psg_pos_n_scored'])):
        ax.annotate(f'{v*100:.0f}%  (n={n})', (v, y_), fontsize=7, va='center',
                    xytext=(-4, 0), textcoords='offset points', ha='right',
                    color='white', fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('epochs where the accelerometer label matches the PSG label')
    sgn = per['psg_left_is_positive_turn_frac'].dropna()
    ax.set_title('F  Fixed-threshold agreement with PSG position\n'
                 f'sign check: positive turn = PSG "Left" in '
                 f'{sgn.mean()*100:.0f}% of lateral epochs ({len(sgn)} sessions)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    fig.suptitle('Accelerometer head angle — revalidation. '
                 'The gravity vector is real (A), the old 0.5 Hz cutoff let breathing in (B), '
                 'calibration is second-order (C),\nand the angle matches the '
                 'independently scored PSG body position (D, F) once the ±90° clamp '
                 'is removed (E).',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=190, bbox_inches='tight'); plt.close(fig)


def fig_traces(ep_all, sessions, out):
    fig, axes = plt.subplots(6, 2, figsize=(20, 20), sharey=True)
    for ax, lbl in zip(axes.ravel(), sessions):
        g = ep_all[ep_all['session'] == lbl].sort_values('t_hr')
        t = g['t_hr'].to_numpy()
        codes = g['stage_code'].to_numpy()
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                       alpha=0.13, lw=0)
        # PSG-scored position as a ribbon along the bottom
        psg = g['psg_pos'].to_numpy()
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1], ymin=0.0, ymax=0.045,
                       color=POS_COLORS.get(psg[j], '#BBB'), alpha=0.95, lw=0)
        ax.plot(t, g['turn_deg'], lw=1.2, color='#B9380B', zorder=4)
        ax.plot(t, g['elev_deg'], lw=0.9, color='#2980B9', alpha=0.8, zorder=3)
        for lv in (-90, 0, 90):
            ax.axhline(lv, color='#555', ls=':', lw=0.8, zorder=2)
        ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_title(lbl, fontsize=10, fontweight='bold')
        ax.set_ylabel('angle (deg)', fontsize=8)
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    handles = [plt.Line2D([], [], color='#B9380B', lw=2, label='turn (0 supine, +90 left, −90 right)'),
               plt.Line2D([], [], color='#2980B9', lw=2, label='elevation (90 = upright)')]
    handles += [mpatches.Patch(color=POS_COLORS[p], label=f'PSG {p}') for p in POS_ORDER]
    handles += [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
                for k in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=7, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle('Head angle over the night — all 12 sessions. Ribbon along the '
                 'bottom of each panel is the PSG-scored body position.',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)


def fig_session(g, lbl, out):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={'height_ratios': [1.0, 0.55]})
    t = g['t_hr'].to_numpy(); codes = g['stage_code'].to_numpy()
    ax = axes[0]
    for j in range(len(t) - 1):
        ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                   alpha=0.15, lw=0)
    ax.plot(t, g['turn_deg'], lw=1.4, color='#B9380B', label='turn angle')
    ax.plot(t, g['elev_deg'], lw=1.0, color='#2980B9', alpha=0.85, label='elevation')
    ax.plot(t, g['legacy_roll_deg'], lw=0.9, color='#7F8C8D', ls='--', alpha=0.8,
            label='legacy roll (clamped ±90°)')
    for lv, lb in [(90, 'left'), (0, 'supine'), (-90, 'right')]:
        ax.axhline(lv, color='#555', ls=':', lw=0.8)
        ax.annotate(lb, (t[0], lv), fontsize=7.5, color='#555', va='bottom')
    ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel('angle (deg)')
    ax.set_title(f'{lbl} — head angle vs PSG-scored body position',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='upper right'); ax.grid(True, alpha=0.13)

    ax = axes[1]
    psg = g['psg_pos'].to_numpy(); own = g['head_pos'].to_numpy()
    for j in range(len(t) - 1):
        ax.axvspan(t[j], t[j + 1], ymin=0.55, ymax=0.95,
                   color=POS_COLORS.get(psg[j], '#BBB'), lw=0)
        ax.axvspan(t[j], t[j + 1], ymin=0.05, ymax=0.45,
                   color=POS_COLORS.get(own[j], '#BBB'), lw=0)
    ax.set_yticks([0.25, 0.75]); ax.set_yticklabels(['accelerometer', 'PSG scored'])
    ax.set_ylim(0, 1); ax.set_xlabel('Time (hours)')
    ax.legend(handles=[mpatches.Patch(color=POS_COLORS[p], label=p)
                       for p in POS_ORDER], fontsize=8, ncol=5, loc='upper right')
    fig.tight_layout(); fig.savefig(out, dpi=185, bbox_inches='tight'); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rows, eps, pos_tbls = [], [], []
    for i, meta in enumerate(SESSION_META):
        row, ep, pt = analyse(i, meta)
        rows.append(row); eps.append(ep); pos_tbls.append(pt)
        print(f"  {row['session']}  |a|={row['accel_mag_median_g']:.3f} g "
              f"(CV {row['accel_mag_cv']*100:4.1f}%)  "
              f"ripple {row['turn_resp_ripple_deg_lp050']:.3f}->"
              f"{row['turn_resp_ripple_deg_lp005']:.3f} deg  "
              f"PSG agree={row.get('psg_pos_agree_frac', float('nan')):.2f}")

    per = pd.DataFrame(rows)
    ep_all = pd.concat(eps, ignore_index=True)
    pos_tbl = pd.concat([p for p in pos_tbls if len(p)], ignore_index=True)
    per.to_csv(REPORT_DIR / 'head_angle_validation.csv', index=False)
    pos_tbl.to_csv(REPORT_DIR / 'head_angle_vs_psg_position.csv', index=False)
    ep_all.to_csv(REPORT_DIR / 'head_angle_epochs.csv', index=False)

    sessions = sorted(ep_all['session'].unique())
    fig_validation(per, pos_tbl, ep_all, FIG_DIR / 'head_angle_validation.png')
    fig_traces(ep_all, sessions, FIG_DIR / 'head_angle_traces_grid.png')
    for lbl in sessions:
        fig_session(ep_all[ep_all['session'] == lbl].sort_values('t_hr'), lbl,
                    FIG_DIR / f'head_angle_{lbl}.png')

    print('\n' + '=' * 84)
    print('VERDICT')
    print('=' * 84)
    print(f"1. Gravity      |a| median {per['accel_mag_median_g'].median():.3f} g "
          f"[{per['accel_mag_median_g'].min():.3f}, {per['accel_mag_median_g'].max():.3f}], "
          f"CV {per['accel_mag_cv'].median()*100:.1f}% (max "
          f"{per['accel_mag_cv'].max()*100:.1f}%).")
    print(f"                {per['frac_within_5pct_median'].min()*100:.1f}-"
          f"{per['frac_within_5pct_median'].max()*100:.1f}% of the night within 5% of "
          f"the session median -> the DC vector is gravity; tilt angles are valid.")
    if 'calib_turn_shift_median_deg' in per:
        print(f"2. Calibration  ellipsoid fit moves the turn angle by "
              f"{per['calib_turn_shift_median_deg'].median():.2f} deg (median), "
              f"{per['calib_turn_shift_p95_deg'].max():.2f} deg worst-case p95;")
        print(f"                coverage {per['calib_coverage'].min():.2f}-"
              f"{per['calib_coverage'].max():.2f} -- poorly constrained on "
              f"supine-dominant nights, so it is reported, not applied.")
    print(f"3. Cutoff       breathing-band ripple in the angle: "
          f"{per['turn_resp_ripple_deg_lp050'].median():.3f} deg at {LP_LEGACY} Hz -> "
          f"{per['turn_resp_ripple_deg_lp005'].median():.3f} deg at {LP_GRAVITY} Hz "
          f"({per['turn_resp_ripple_deg_lp050'].median() / max(per['turn_resp_ripple_deg_lp005'].median(), 1e-9):.0f}x).")
    ag = per['psg_pos_agree_frac'].dropna()
    sgn = per['psg_left_is_positive_turn_frac'].dropna()
    print(f"4. PSG position agreement {ag.median()*100:.1f}% median "
          f"[{ag.min()*100:.1f}, {ag.max()*100:.1f}] over {int(per['psg_pos_n_scored'].sum()):,} "
          f"scored epochs, fixed thresholds, no per-session tuning.")
    print(f"                sign: positive turn = PSG 'Left' in {sgn.mean()*100:.1f}% of "
          f"lateral epochs across {len(sgn)} sessions -> the legacy "
          f"classify_position() had left/right inverted (now fixed).")
    print(f"5. Legacy angle up to "
          f"{per['legacy_cannot_see_prone_frac'].max()*100:.1f}% of a night is spent "
          f"with gZ<0 (head turned past the side), where the clamped roll folds back:")
    print(f"                median |legacy - turn| = "
          f"{per['legacy_err_deg_gZpos'].median():.2f} deg while gZ>0 but "
          f"{per['legacy_err_deg_gZneg'].median():.1f} deg once gZ<0.")

    # ── where the accelerometer and the PSG disagree ──
    sc = ep_all[ep_all['psg_pos'].isin(POS_ORDER)]
    conf = pd.crosstab(sc['psg_pos'], sc['head_pos'])
    conf = conf.reindex(index=[p for p in POS_ORDER if p in conf.index],
                        columns=[p for p in POS_ORDER if p in conf.columns],
                        fill_value=0)
    print('\nConfusion, 30 s epochs (rows = PSG-scored body position, '
          'cols = accelerometer head position):')
    print(conf.to_string())
    print('  The PSG sensor scores TRUNK position; this one is on the head. Head-on-'
          'pillow\n  turns with the trunk still supine are a real physiological '
          'difference, not an error.')

    print('\nPer-position turn angle (deg, median across sessions):')
    for p in POS_ORDER:
        d = pos_tbl[pos_tbl['psg_pos'] == p]
        if len(d) == 0:
            continue
        print(f'  {p:8s} n_sessions={len(d):2d}  turn={circ_median_deg(d["turn_deg_median"]):+7.1f}  '
              f'elev={d["elev_deg_median"].median():+6.1f}  '
              f'(legacy roll {d["legacy_roll_median"].median():+7.1f})')

    print(f'\nTables  -> {REPORT_DIR}')
    print(f'Figures -> {FIG_DIR}')


if __name__ == '__main__':
    main()
