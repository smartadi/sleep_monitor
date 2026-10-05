"""
Does the slow-trend velocity of CLE-CRE and CH change at REM onset?

The literature gives one specific prediction for a slow intracranial-volume
signal during sleep: blood volume and ICP rise at the transition from NREM into
REM, quickly (within minutes; Näsi 2011 NIRS, Riedel 2023 invasive ICP), while
changes into N3 are small and gradual. This tests that prediction on the
motion-removed (de-stepped) traces.

Events
    REM-episode onset: the first REM epoch after at least REM_GAP_MIN minutes
    without REM, with REM making up at least half of the following
    REM_HOLD_MIN minutes. The hypnograms are fragmented (REM bouts of a few
    epochs), so an episode is allowed brief interruptions.

Quantity
    The same 30-min least-squares slope as the display marker, but CENTRED on
    each moment instead of trailing it. The display marker is trailing so it
    cannot anticipate; for an event-locked average that would push every
    response ~15 min late. Blocks inside a head movement are left out of every
    fit, as in the marker. Units fF/h.

Per event
    the velocity curve from -WIN to +WIN min around onset, and the onset
    response = mean velocity over [0, +RESP_MIN] min minus mean over
    [-BASE_MIN, 0] min (how much faster the trace rises after onset than
    before it).

Null
    The same response at random times in NREM sleep in the same recording, at
    least NULL_GAP_MIN from any REM onset, NULL_DRAWS times. It absorbs each
    night's own drift and autocorrelation.

Summary
    events are averaged within a participant first (both nights pooled per
    participant), and the result is the direction count across participants
    and how each participant's response sits in its own null. Descriptive:
    six participants, no pooled p-value.

What it finds (2026-10-04)
    15 REM-episode onsets in 5 participants (S6's hypnograms have none that
    qualify). CLE-CRE velocity rises after onset in 5/5 participants (4/4
    using only onsets with no head movement within 2 min), median +8 fF/h, but
    only 1/5 clears its own null 95%. CH rises in 3/5. The CLE-CRE direction
    holds for a 10- or 15-min REM-free gap and is lost (3/6) once brief REM
    re-entries count as onsets. A weak, definition-dependent effect in the
    predicted direction -- not a finding on its own.

Writes  reports/mean_value/rem_onset_velocity_events.csv
        reports/mean_value/rem_onset_velocity_sensitivity.csv
        reports/mean_value/rem_onset_velocity_subjects.csv
        writeup/figures/imbalance/fig_rem_onset_velocity.png

Usage
    .venv/Scripts/python.exe analysis/mean_value/rem_onset_velocity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import diff_motion_regressed as dmr                 # noqa: E402
from sleep_monitor.sessions import SESSION_META     # noqa: E402

TREND_MIN = 30.0        # slope window, as in the display marker (centred here)
REM_GAP_MIN = 10.0      # REM-free time before an episode onset
REM_HOLD_MIN = 5.0      # REM must fill >= half of this after onset
WIN_MIN = 30.0          # event window, either side
BASE_MIN = 15.0         # pre-onset baseline span
RESP_MIN = 15.0         # post-onset response span
NULL_GAP_MIN = 20.0     # null times this far from any REM onset
NULL_DRAWS = 500
REM, WAKE = 0, 4
SIG = {'CLE-CRE': ('d_destep', '#B9380B'), 'CH': ('ch_destep', '#1F618D')}

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'font.weight': 'normal', 'axes.labelweight': 'normal',
                     'axes.titleweight': 'normal', 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def centred_slope(y, moving):
    """30-min least-squares slope centred on each block, movement left out, fF/h."""
    n = int(round(TREND_MIN * 60 / dmr.BLOCK_S))
    t_h = pd.Series(np.arange(len(y)) * dmr.BLOCK_S / 3600.0)
    yy = pd.Series(np.where(moving, np.nan, y))
    tt = t_h.where(yy.notna())
    cov = tt.rolling(n, min_periods=n // 2, center=True).cov(yy)
    var = tt.rolling(n, min_periods=n // 2, center=True).var()
    return (cov / var).to_numpy()


def rem_onsets(codes):
    """Block indices of REM-episode onsets (see module docstring)."""
    gap = int(round(REM_GAP_MIN * 60 / dmr.BLOCK_S))
    hold = int(round(REM_HOLD_MIN * 60 / dmr.BLOCK_S))
    is_rem = codes == REM
    out = []
    for i in np.flatnonzero(is_rem):
        if i < gap or i + hold > len(codes):
            continue
        if is_rem[i - gap:i].any():
            continue
        if is_rem[i:i + hold].mean() >= 0.5:
            out.append(int(i))
    return out


def windows(v, idx, w):
    """Event-locked segments of v, NaN-padded at the recording edges."""
    seg = np.full((len(idx), 2 * w + 1), np.nan)
    for k, i in enumerate(idx):
        a, b = i - w, i + w + 1
        lo, hi = max(a, 0), min(b, len(v))
        seg[k, lo - a:hi - a] = v[lo:hi]
    return seg


def response(seg, w):
    b = int(round(BASE_MIN * 60 / dmr.BLOCK_S))
    r = int(round(RESP_MIN * 60 / dmr.BLOCK_S))
    return np.nanmean(seg[:, w:w + r], axis=1) - np.nanmean(seg[:, w - b:w], axis=1)


def null_times(codes, onsets, n, rng):
    """n random NREM-sleep blocks at least NULL_GAP_MIN from any REM onset."""
    gap = int(round(NULL_GAP_MIN * 60 / dmr.BLOCK_S))
    w = int(round(WIN_MIN * 60 / dmr.BLOCK_S))
    ok = (codes != REM) & (codes != WAKE)
    ok[:w] = ok[-w:] = False
    for o in onsets:
        ok[max(0, o - gap):o + gap] = False
    pool = np.flatnonzero(ok)
    return rng.choice(pool, size=n, replace=True) if len(pool) else np.array([], int)


def sensitivity(sessions, w):
    """The participant direction count under looser and stricter REM-onset rules.

    A brief REM re-entry inside an ongoing episode counts as an onset once the
    REM-free gap is short; this shows how much the result depends on that.
    """
    global REM_GAP_MIN, REM_HOLD_MIN
    keep = REM_GAP_MIN, REM_HOLD_MIN
    vel = {k: [centred_slope(r[f], r['moving']) for r in sessions]
           for k, (f, _) in SIG.items()}
    rows = []
    for gap, hold in [(15, 5), (10, 5), (5, 3), (5, 2)]:
        REM_GAP_MIN, REM_HOLD_MIN = gap, hold
        for k in SIG:
            per = {}
            for r, v in zip(sessions, vel[k]):
                on = rem_onsets(np.asarray(r['codes']))
                if on:
                    per.setdefault(r['label'][:2], []).extend(response(windows(v, on, w), w))
            vals = np.array([np.nanmean(x) for x in per.values()])
            rows.append(dict(rem_free_gap_min=gap, rem_hold_min=hold, signal=k,
                             n_events=sum(len(x) for x in per.values()),
                             n_participants=len(vals), n_rise=int((vals > 0).sum()),
                             median_response_fF_h=float(np.median(vals))))
    REM_GAP_MIN, REM_HOLD_MIN = keep
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(0)
    w = int(round(WIN_MIN * 60 / dmr.BLOCK_S))
    lag_min = (np.arange(-w, w + 1) * dmr.BLOCK_S) / 60.0
    ev_rows, curves, null_curves = [], {k: {} for k in SIG}, {k: {} for k in SIG}
    null_resp = {k: {} for k in SIG}

    sessions = [dmr.one_session(meta) for meta in SESSION_META]
    for r in sessions:
        codes = np.asarray(r['codes'])
        on = rem_onsets(codes)
        subj = r['label'][:2]
        print(f"  {r['label']}: {len(on)} REM-episode onsets")
        if not on:
            continue
        nt = null_times(codes, on, len(on) * NULL_DRAWS, rng)
        for key, (field, _) in SIG.items():
            v = centred_slope(r[field], r['moving'])
            seg = windows(v, on, w)
            resp = response(seg, w)
            curves[key].setdefault(subj, []).append(seg)
            nseg = windows(v, nt, w)
            null_curves[key].setdefault(subj, []).append(nseg)
            nr = response(nseg, w).reshape(NULL_DRAWS, len(on))
            null_resp[key].setdefault(subj, []).append(nr)
            for i, o in enumerate(on):
                mv = r['moving'][max(0, o - 12):o + 13].any()   # +/-2 min
                ev_rows.append(dict(session=r['label'], subject=subj, signal=key,
                                    onset_hr=float(r['t'][o]), response_fF_h=resp[i],
                                    movement_within_2min=bool(mv)))

    ev = pd.DataFrame(ev_rows)
    ev.to_csv(dmr.TAB / 'rem_onset_velocity_events.csv', index=False)

    # per participant: pool both nights' events, compare with the matched null
    rows = []
    for key in SIG:
        for subj in sorted(curves[key]):
            segs = np.vstack(curves[key][subj])
            obs = float(np.nanmean(response(segs, w)))
            # null: same number of events per night, NULL_DRAWS times
            nr = np.hstack(null_resp[key][subj])               # draws x events
            nd = np.nanmean(nr, axis=1)
            sub_ev = ev[(ev.subject == subj) & (ev.signal == key)]
            still = sub_ev[~sub_ev.movement_within_2min].response_fF_h
            rows.append(dict(signal=key, subject=subj, n_events=len(segs),
                             response_fF_h=obs,
                             response_still_only=float(still.mean()) if len(still) else np.nan,
                             n_still=len(still),
                             null_median=float(np.nanmedian(nd)),
                             null_p05=float(np.nanpercentile(nd, 5)),
                             null_p95=float(np.nanpercentile(nd, 95)),
                             percentile_in_null=float((nd < obs).mean() * 100)))
    sj = pd.DataFrame(rows)
    sj.to_csv(dmr.TAB / 'rem_onset_velocity_subjects.csv', index=False)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6),
                             gridspec_kw={'width_ratios': [1.6, 1]})
    for row, (key, (_, color)) in enumerate(SIG.items()):
        ax = axes[row, 0]
        subj_means = np.array([np.nanmean(np.vstack(curves[key][s]), axis=0)
                               for s in sorted(curves[key])])
        null_means = np.array([np.nanmean(np.vstack(null_curves[key][s]), axis=0)
                               for s in sorted(null_curves[key])])
        for sm in subj_means:
            ax.plot(lag_min, sm, color=color, lw=0.8, alpha=0.35)
        m = np.nanmean(subj_means, axis=0)
        se = np.nanstd(subj_means, axis=0) / np.sqrt(len(subj_means))
        nm = np.nanmean(null_means, axis=0)
        ax.fill_between(lag_min, m - se, m + se, color=color, alpha=0.18, lw=0)
        ax.plot(lag_min, m, color=color, lw=2.6, label='REM onset, mean of participants')
        ax.plot(lag_min, nm, color='#7F8C8D', lw=1.8, ls='--',
                label='random NREM times (same nights)')
        ax.axvline(0, color='k', lw=1)
        ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_xlim(-WIN_MIN, WIN_MIN)
        ax.set_ylabel(f'{key} velocity (fF/h)')
        ax.set_title(f'{"ab"[row]}   {key}: velocity around REM-episode onset '
                     f'({len(subj_means)} participants)', loc='left')
        ax.grid(alpha=0.2)
        if row == 1:
            ax.set_xlabel('minutes from REM onset')
        ax.legend(fontsize=9, loc='upper left', frameon=False)

        ax2 = axes[row, 1]
        s = sj[sj.signal == key].reset_index(drop=True)
        y = np.arange(len(s))
        ax2.hlines(y, s.null_p05, s.null_p95, color='#BDC3C7', lw=6,
                   label='null 5–95%')
        ax2.plot(s.null_median, y, '|', color='#7F8C8D', ms=14, mew=2)
        ax2.plot(s.response_fF_h, y, 'o', color=color, ms=9, label='REM onset')
        ax2.axvline(0, color='k', lw=0.6, ls=':')
        ax2.set_yticks(y)
        ax2.set_yticklabels([f'{a}  (n={b})' for a, b in zip(s.subject, s.n_events)])
        ax2.invert_yaxis()
        up = int((s.response_fF_h > 0).sum())
        above = int((s.percentile_in_null > 95).sum())
        ax2.set_title(f'{"cd"[row]}   onset response: {up}/{len(s)} rise, '
                      f'{above}/{len(s)} above null 95%', loc='left')
        ax2.set_xlabel(f'onset response (fF/h)\n'
                       f'mean {RESP_MIN:.0f} min after − mean {BASE_MIN:.0f} min before')
        ax2.grid(alpha=0.2, axis='x')
        if row == 0:
            ax2.legend(fontsize=9, loc='lower right', frameon=False)
    fig.tight_layout()
    out = dmr.FIG / 'fig_rem_onset_velocity.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)

    sens = sensitivity(sessions, w)
    sens.to_csv(dmr.TAB / 'rem_onset_velocity_sensitivity.csv', index=False)
    print('\nevent-definition sensitivity:')
    print(sens.to_string(index=False))

    print('\nper participant (both nights pooled):')
    print(sj.round(2).to_string(index=False))
    print('wrote', out.name)


if __name__ == '__main__':
    main()
