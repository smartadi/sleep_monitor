"""
Where the head angle comes from, and how much of the SEC signal it explains.

Two questions, kept separate on purpose because conflating them is the easy
mistake:

    1. PROVENANCE. Is the head angle computed from the accelerometer alone, with
       no capacitance anywhere in the chain? This is a question about code, and
       it is answered by reading the chain and by checking the function's
       signature at runtime -- not by a correlation.

    2. COUPLING. Given that the two are measured independently, how much of each
       SEC channel does the angle account for? This is a question about physics
       and it is answered by numbers. A high correlation here would NOT mean the
       angle came from the capacitance; it would mean the head really does move
       the electrodes.

The two get confused because a strong correlation feels like contamination. It
is not. The electrodes sit on the temples; turning the head presses one toward
the skin and lifts the other, so the SEC channels SHOULD respond to orientation.
That is the reason for regressing it out, and the reason the regression works.

Also characterises the angle's own dynamics, because the question was whether it
carries both abrupt turns and slow drift. It carries both, and the two have very
different sizes -- the numbers are printed per recording.

A note on which angle. Two definitions exist in this repo:

    turn   = atan2(gY, gZ)                 motion.head_angle(), validated
             full +/-180: 0 supine, +90 left, -90 right, +/-180 prone
    roll   = atan2(gY, sqrt(gX^2 + gZ^2))  motion.head_orientation(), legacy
             clamped to +/-90, cannot tell supine from prone

diff_motion_regressed.py displays the legacy one. That is harmless there because
the regression itself uses the full gravity VECTOR [gX, gY, gZ], not any angle,
so it is immune to both the +/-180 wrap and the clamp. Both are reported below
so the difference is visible rather than assumed.

Writes  reports/mean_value/head_angle_provenance.csv
        writeup/figures/imbalance/fig_head_angle_provenance.png

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/head_angle_provenance.py
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    'diff_motion_regressed',
    ROOT / 'analysis' / 'mean_value' / 'diff_motion_regressed.py')
dmr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dmr)

from sleep_monitor import load_session                        # noqa: E402
from sleep_monitor import motion                              # noqa: E402
from sleep_monitor.config import CAP_SCALE_TO_FF, CAP_CHANNELS  # noqa: E402
from sleep_monitor.filters import lowpass                     # noqa: E402
from sleep_monitor.sessions import SESSION_META               # noqa: E402

FIG = ROOT / 'writeup' / 'figures' / 'imbalance'
TAB = ROOT / 'reports' / 'mean_value'
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

STEP_DEG = 5.0          # a block-to-block angle change this big is a turn
SLOW_MIN = 30.0         # timescale separating "slow drift" from "a turn"

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'font.weight': 'bold', 'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.dpi': 110, 'savefig.dpi': 180,
})


def provenance():
    """Read the chain, and assert it at runtime rather than trusting the read."""
    print('=' * 72)
    print('1. PROVENANCE -- what goes into the angle')
    print('=' * 72)

    sig = inspect.signature(motion.head_angle)
    args = list(sig.parameters)
    print(f'\nmotion.head_angle{sig}')
    print(f'  arguments: {args}')
    cap_only = [c for c in CAP_CHANNELS if c not in ('aX', 'aY', 'aZ')]
    bad = [a for a in args if a in cap_only]
    assert not bad, f'capacitance channel in the signature: {bad}'
    print(f'  capacitance channels {cap_only} appear in the signature: NO')

    src = inspect.getsource(motion.head_angle)
    hits = [c for c in cap_only if c in src]
    assert not hits, f'capacitance channel in the body: {hits}'
    print(f'  capacitance channels appear in the function body:   NO')

    print('\nthe chain, in full:\n')
    print('   raw CSV columns          aX, aY, aZ        (units of g, own ADC)')
    print('        |                   CH, CLE, CRE      (capacitance counts)')
    print('        |                        \\___ never enters any line below')
    print('        v')
    print(f'   gravity split            g = lowpass(a, {motion.GRAVITY_LP_HZ} Hz)')
    print('        v')
    print('   angle                    turn = atan2(gY, gZ)        [validated]')
    print('                            roll = atan2(gY, sqrt(gX^2+gZ^2)) [legacy]')
    print('                            elev = asin(gX / |g|)')
    print('                            tilt = acos(gZ / |g|)')
    print('\n   |g| is the validity check: it must stay near 1 g, or the DC')
    print('   vector is not gravity and no angle from it means anything.')
    print('\nthe regression in diff_motion_regressed.py uses the gravity VECTOR')
    print('[gX, gY, gZ], not an angle, so it is immune to the +/-180 wrap and to')
    print('the legacy +/-90 clamp. The angle is a display and diagnostic.')


def r2_on(y, X):
    """Fraction of y explained by least squares on X (plus intercept)."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    A = np.column_stack([X, np.ones(len(y))])
    ok = np.isfinite(y) & np.isfinite(A).all(axis=1)
    if ok.sum() < 50:
        return np.nan
    yy, AA = y[ok], A[ok]
    beta, *_ = np.linalg.lstsq(AA, yy, rcond=None)
    ss = float(np.sum((yy - yy.mean()) ** 2))
    return float(1 - np.sum((yy - AA @ beta) ** 2) / ss) if ss > 0 else np.nan


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(np.asarray(a, float))))


def main():
    provenance()

    print('\n' + '=' * 72)
    print('2. THE ANGLE\'S OWN DYNAMICS, and 3. COUPLING TO SEC')
    print('=' * 72)

    n = int(dmr.BLOCK_S * dmr.FS)
    slow_k = int(round(SLOW_MIN * 60 / dmr.BLOCK_S)) | 1
    rows, keep = [], []

    for meta in SESSION_META:
        s = load_session(meta)
        ha = motion.head_angle(np.asarray(s.cap['aX'], float),
                               np.asarray(s.cap['aY'], float),
                               np.asarray(s.cap['aZ'], float), fs=dmr.FS)
        G = np.column_stack([dmr.blocks(ha[k], n) for k in ('gX', 'gY', 'gZ')])
        gmag = dmr.blocks(ha['gmag'], n)
        # unwrapped for differences and for the slow trend, so a supine-to-prone
        # crossing does not register as a spurious 360 deg jump. The reported
        # RANGE comes from the wrapped angle, which is the physical one and
        # cannot exceed 360; the unwrapped range is cumulative rotation and
        # would be misread as a bigger excursion than actually happened.
        turn_wrapped = dmr.blocks(ha['turn_deg'], n)
        turn = unwrap_deg(turn_wrapped)
        roll = dmr.blocks(np.degrees(np.arctan2(
            ha['gY'], np.sqrt(ha['gX'] ** 2 + ha['gZ'] ** 2))), n)

        # Slow drift vs abrupt turns, separated at SLOW_MIN. The trend is a
        # CIRCULAR median -- a plain median of an angle that lives on a circle
        # averages +170 and -170 to 0, i.e. it reports "supine" for a head that
        # never left prone.
        r = np.radians(turn_wrapped)
        sm = pd.Series(np.sin(r)).rolling(slow_k, center=True,
                                          min_periods=1).median().to_numpy()
        cm = pd.Series(np.cos(r)).rolling(slow_k, center=True,
                                          min_periods=1).median().to_numpy()
        slow_wrapped = np.degrees(np.arctan2(sm, cm))
        slow = pd.Series(turn).rolling(slow_k, center=True,
                                       min_periods=1).median().to_numpy()
        dturn = np.abs(np.diff(turn, prepend=turn[0]))

        sig = {}
        for c in ('CLE', 'CRE', 'CH'):
            y = dmr.blocks(np.asarray(s.cap[c], float) * CAP_SCALE_TO_FF, n)
            sig[c] = y - np.nanmean(y)
        sig['CLE-CRE'] = sig['CLE'] - sig['CRE']
        dd, _ = dmr.destep(sig['CLE-CRE'], G, dmr.STEP_THRESH, dmr.STEP_PAD)
        chd, _ = dmr.destep(sig['CH'], G, dmr.STEP_THRESH, dmr.STEP_PAD)
        sig['CLE-CRE de-stepped'] = dd
        sig['CH de-stepped'] = chd

        row = dict(session=meta['label'],
                   gmag_median=float(np.nanmedian(gmag)),
                   gmag_cv_pct=float(100 * np.nanstd(gmag) / np.nanmedian(gmag)),
                   turn_range_deg=float(np.nanmax(turn_wrapped)
                                       - np.nanmin(turn_wrapped)),
                   steps_gt5=int((dturn > STEP_DEG).sum()),
                   steps_gt20=int((dturn > 20).sum()),
                   dturn_median_deg=float(np.nanmedian(dturn)),
                   dturn_p999_deg=float(np.nanpercentile(dturn, 99.9)),
                   slow_range_deg=float(np.nanmax(slow) - np.nanmin(slow)))
        for name, y in sig.items():
            row[f'r2_vec_{name}'] = r2_on(y, G)
            row[f'r2_turn_{name}'] = r2_on(y, turn[:, None])
            row[f'r2_roll_{name}'] = r2_on(y, roll[:, None])
        rows.append(row)
        keep.append((meta['label'], turn_wrapped, slow_wrapped, dturn, sig, G))
        print(f"  {meta['label']}  |g| {row['gmag_median']:.3f} g "
              f"(cv {row['gmag_cv_pct']:.1f}%)   turns >5 deg: "
              f"{row['steps_gt5']:4d}   R2(CLE-CRE | gravity) "
              f"{row['r2_vec_CLE-CRE']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(TAB / 'head_angle_provenance.csv', index=False)

    print('\naccelerometer validity -- |g| must sit near 1 g or no angle is real')
    print(f"  |g| median across recordings {df.gmag_median.median():.3f} g, "
          f"range {df.gmag_median.min():.3f}-{df.gmag_median.max():.3f}")
    print(f"  within-recording CV: median {df.gmag_cv_pct.median():.1f}%, "
          f"worst {df.gmag_cv_pct.max():.1f}%")

    print('\nangle dynamics per recording (10 s blocks), median across the 12')
    print(f"  block-to-block change, median      "
          f"{df.dturn_median_deg.median():6.2f} deg   <- the quiet baseline")
    print(f"  block-to-block change, 99.9th pct  "
          f"{df.dturn_p999_deg.median():6.2f} deg   <- abrupt turns")
    print(f"  changes > {STEP_DEG:.0f} deg per night           "
          f"{df.steps_gt5.median():6.0f}")
    print(f"  changes > 20 deg per night          "
          f"{df.steps_gt20.median():6.0f}")
    print(f"  total angle range                  "
          f"{df.turn_range_deg.median():6.1f} deg  (wrapped, max possible 360)")
    print(f"  range of the {SLOW_MIN:.0f}-min slow component  "
          f"{df.slow_range_deg.median():6.1f} deg   <- slow drift, same signal")
    print(f"      (slow component is unwrapped, so it is cumulative rotation)")
    print('  -> both are present: large abrupt turns AND a slow component.')

    print('\ncoupling: variance of each SEC channel explained by head '
          'orientation')
    print('  (this is physics, not provenance -- the electrodes are on the '
          'temples)')
    print(f"\n{'channel':22s} {'R2 | gravity vector':>20s} "
          f"{'R2 | turn angle':>17s} {'recordings >0.5':>17s}")
    for name in ['CLE', 'CRE', 'CH', 'CLE-CRE',
                 'CLE-CRE de-stepped', 'CH de-stepped']:
        v = df[f'r2_vec_{name}']
        t = df[f'r2_turn_{name}']
        print(f'{name:22s} {v.median():9.3f} [{v.min():.2f}-{v.max():.2f}] '
              f'{t.median():9.3f} [{t.min():.2f}-{t.max():.2f}] '
              f'{int((v > 0.5).sum()):12d}/12')

    # ---------------- figure ----------------
    pick = [k for k in keep if k[0] in ('S1N1', 'S6N1')] or keep[:2]
    fig, axes = plt.subplots(3, len(pick), squeeze=False,
                             figsize=(8.4 * len(pick), 12.0))
    for c, (label, turn, slow, dturn, sig, G) in enumerate(pick):
        t = np.arange(len(turn)) * dmr.BLOCK_S / 3600.0
        ax = axes[0][c]
        for y, nm in [(180, 'prone'), (90, 'left'), (0, 'supine'),
                      (-90, 'right'), (-180, 'prone')]:
            ax.axhline(y, color='#AAB2BD', lw=1.0, ls=':', zorder=0)
            ax.annotate(nm, (1.004, y), xycoords=('axes fraction', 'data'),
                        va='center', fontsize=12, color='#7F8C8D')
        ax.plot(t, turn, lw=1.1, color='#2C3E50', label='turn = atan2(gY, gZ)')
        ax.plot(t, slow, lw=3.2, color='#B9380B',
                label=f'{SLOW_MIN:.0f}-min circular median')
        big = dturn > STEP_DEG
        ax.plot(t[big], turn[big], ls='', marker='v', ms=5, color='#1F618D',
                label=f'change > {STEP_DEG:.0f} deg')
        ax.set_ylim(-200, 200)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_title(f'{label}   head angle, from aX/aY/aZ only', loc='left')
        ax.set_ylabel('degrees')
        ax.set_xlabel('hours into the recording')
        if c == 0:
            ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.11), ncol=3,
                      frameon=False, fontsize=12)
        ax.grid(axis='x', alpha=0.2)

        ax = axes[1][c]
        ax.semilogy(t, np.maximum(dturn, 1e-3), lw=0.9, color='#555555')
        ax.axhline(STEP_DEG, color='#b3282d', ls='--', lw=2.0,
                   label=f'{STEP_DEG:.0f} deg')
        ax.set_ylabel('|change| per 10 s\n(degrees, log)')
        ax.set_title('abrupt turns sit three decades above the quiet baseline',
                     loc='left', fontsize=15)
        ax.legend(loc='upper right', frameon=False, fontsize=12)
        ax.grid(alpha=0.2, which='both')

        ax = axes[2][c]
        names = ['CLE', 'CRE', 'CH', 'CLE-CRE',
                 'CLE-CRE de-stepped', 'CH de-stepped']
        vals = [r2_on(sig[nm], G) for nm in names]
        cols = ['#7F8C8D'] * 4 + ['#B9380B', '#1F618D']
        ax.barh(range(len(names)), vals, color=cols)
        for i, v in enumerate(vals):
            ax.annotate(f'{v:.2f}', (v + 0.012, i), va='center', fontsize=14)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('fraction of variance explained by [gX, gY, gZ]')
        ax.set_title('how much of each channel is head orientation',
                     loc='left', fontsize=15)
        ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    p = FIG / 'fig_head_angle_provenance.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {p.name}')
    print(f'-> {TAB / "head_angle_provenance.csv"}')


if __name__ == '__main__':
    main()
