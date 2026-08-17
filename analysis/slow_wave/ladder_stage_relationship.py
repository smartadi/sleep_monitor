"""
Stage relationship of the harmonic-comb LADDER events — with a focus on REM:
do the events cluster in a stage, and does REM tend to PRECEDE or FOLLOW them?

Method:
  * Detect ladder episodes on all 12 sessions x 3 channels (harmonic_ladder_overlay
    .detect_channel).  Per session, merge episodes overlapping in time across
    channels into single EVENTS (so the same physiological event is counted once).
  * Stage enrichment: fraction of event-time in each stage vs that stage's base
    rate over the night (enrichment = P(stage | event) / P(stage)).
  * REM timing: for each event, minutes to the nearest REM epoch BEFORE onset and
    AFTER offset; fraction of pre-onset [-30,0] min and post-offset [0,+30] min
    that is REM.  Compared against a matched-random-NREM null.
  * Event-triggered stage occupancy aligned at onset (-30..+30 min).

Outputs -> reports/slow_wave/revamp/
  ladder_events.csv           one row per merged event (times, dur, stages, REM gaps)
  ladder_stage_summary.csv    stage enrichment + REM pre/post (obs vs null), per subject
Figure -> writeup/figures/harmonics/ladder_stage_relationship.png

Run:  python ladder_stage_relationship.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import STAGE_LABELS, STAGE_ORDER, STAGE_COLORS
import harmonic_ladder_overlay as H

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'revamp'
FIG_DIR = Path(__file__).resolve().parents[2] / 'writeup' / 'figures' / 'harmonics'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REM = 0                       # stage code for REM
WIN_MIN = 30.0               # peri-event window for REM search / occupancy
TAUS = np.arange(-WIN_MIN, WIN_MIN + 0.01, 0.5)   # minutes, for occupancy curve
N_NULL = 200                 # random NREM control points per session


def merge_spans(spans, tol_hr=1.0 / 60):
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= out[-1][1] + tol_hr:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(m) for m in out]


def rem_intervals(sp):
    """List of (start_hr, end_hr) for REM epochs."""
    t = np.asarray(sp['t_ep_hr'], float)
    codes = np.asarray(sp['codes'])
    n = min(len(t), len(codes))
    ivs = []
    for i in range(n - 1):
        if int(codes[i]) == REM:
            ivs.append((t[i], t[i + 1]))
    return ivs


def rem_gaps(rem_ivs, t0, t1):
    """Minutes from onset t0 back to nearest REM end before it, and from offset
    t1 forward to nearest REM start after it (nan if none)."""
    before = [t0 - e for (s, e) in rem_ivs if e <= t0]
    after = [s - t1 for (s, e) in rem_ivs if s >= t1]
    db = min(before) * 60 if before else np.nan
    da = min(after) * 60 if after else np.nan
    return db, da


def rem_frac(sp, a_hr, b_hr):
    """Fraction of time in [a,b] hr scored REM (sampled at 0.5 min)."""
    ts = np.arange(a_hr, b_hr, 0.5 / 60)
    if len(ts) == 0:
        return np.nan
    st = np.array([H._stage_at(sp, t) for t in ts])
    valid = st >= 0
    return float(np.mean(st[valid] == REM)) if valid.any() else np.nan


def main():
    events = []
    occ = {s: np.zeros(len(TAUS)) for s in STAGE_ORDER}
    n_ev = 0
    rows = []
    rng = np.random.default_rng(7)

    for idx in range(12):
        s = load_session(idx)
        s.sleep_profile = load_sleep_profile(s)
        sp = s.sleep_profile
        t = np.asarray(sp['t_ep_hr'], float)
        codes = np.asarray(sp['codes'])
        nrem_mask = np.isin(codes, [1, 2, 3])        # N3,N2,N1
        rem_ivs = rem_intervals(sp)

        spans = []
        for ch in H.CHANNELS:
            f, t_hr, enh, active, episodes = H.detect_channel(s, ch)
            for ep in episodes:
                spans.append((t_hr[ep['lo']], t_hr[ep['hi'] - 1]))
        evs = merge_spans(spans)

        # base rates for enrichment
        base = {st: float(np.mean(codes == st)) for st in STAGE_ORDER}

        for (t0, t1) in evs:
            n_ev += 1
            dur = (t1 - t0) * 60
            # stage composition during event
            ts = np.arange(t0, t1, 0.5 / 60)
            est = np.array([H._stage_at(sp, tt) for tt in ts])
            dom = STAGE_LABELS.get(int(np.bincount(est[est >= 0]).argmax()), '?') if (est >= 0).any() else '?'
            db, da = rem_gaps(rem_ivs, t0, t1)
            pre = rem_frac(sp, t0 - WIN_MIN / 60, t0)
            post = rem_frac(sp, t1, t1 + WIN_MIN / 60)
            events.append(dict(session=s.label, subject=s.subject, t0_hr=round(t0, 3),
                               t1_hr=round(t1, 3), dur_min=round(dur, 1), dom_stage=dom,
                               rem_min_before=round(db, 1) if np.isfinite(db) else np.nan,
                               rem_min_after=round(da, 1) if np.isfinite(da) else np.nan,
                               rem_frac_pre=round(pre, 3) if np.isfinite(pre) else np.nan,
                               rem_frac_post=round(post, 3) if np.isfinite(post) else np.nan))
            # onset-aligned occupancy
            for ti, tau in enumerate(TAUS):
                st = H._stage_at(sp, t0 + tau / 60)
                if st in occ:
                    occ[st][ti] += 1

        # matched-random-NREM null for REM pre/post fractions
        nrem_t = t[:len(codes)][nrem_mask[:len(t)]]
        null_pre, null_post = [], []
        if len(nrem_t) and evs:
            durs = [(b - a) for a, b in evs]
            for k in range(N_NULL):
                d = durs[k % len(durs)]
                rt = nrem_t[rng.integers(len(nrem_t))]
                null_pre.append(rem_frac(sp, rt - WIN_MIN / 60, rt))
                null_post.append(rem_frac(sp, rt + d, rt + d + WIN_MIN / 60))

        # per-session/subject summary row
        se = [e for e in events if e['session'] == s.label]
        if se:
            rows.append(dict(
                session=s.label, subject=s.subject, n_events=len(se),
                dom_stage_mode=pd.Series([e['dom_stage'] for e in se]).mode().iat[0],
                rem_frac_pre=np.nanmean([e['rem_frac_pre'] for e in se]),
                rem_frac_post=np.nanmean([e['rem_frac_post'] for e in se]),
                null_rem_frac_pre=np.nanmean(null_pre) if null_pre else np.nan,
                null_rem_frac_post=np.nanmean(null_post) if null_post else np.nan,
                med_rem_min_before=np.nanmedian([e['rem_min_before'] for e in se]),
                med_rem_min_after=np.nanmedian([e['rem_min_after'] for e in se]),
            ))

    ev_df = pd.DataFrame(events)
    sm_df = pd.DataFrame(rows)
    ev_df.to_csv(REPORT_DIR / 'ladder_events.csv', index=False)
    sm_df.to_csv(REPORT_DIR / 'ladder_stage_summary.csv', index=False)

    # ── figure: onset-aligned stage occupancy ──
    P = {s: occ[s] / max(n_ev, 1) for s in STAGE_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    for st in STAGE_ORDER:
        ax.plot(TAUS, P[st], color=STAGE_COLORS[st], lw=2, label=STAGE_LABELS[st])
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('minutes relative to event onset')
    ax.set_ylabel('P(stage)')
    ax.set_title(f'Stage occupancy around ladder-event onset (n={n_ev} events)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    ax = axes[1]
    remc = STAGE_COLORS[REM]
    ax.plot(TAUS, P[REM], color=remc, lw=2.5, label='P(REM) around onset')
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('minutes relative to event onset')
    ax.set_ylabel('P(REM)')
    ax.set_title('REM occupancy vs event onset (left=before, right=during/after)')
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ladder_stage_relationship.png', dpi=150, facecolor='white')
    plt.close(fig)

    # ── console summary ──
    print(f"\n{'='*64}\nLADDER EVENTS — stage relationship\n{'='*64}")
    print(f"Total merged events: {n_ev} across {ev_df['session'].nunique()} sessions, "
          f"{ev_df['subject'].nunique()} subjects")
    print("\nEvent dominant-stage distribution:")
    print(ev_df['dom_stage'].value_counts().to_string())
    print(f"\nREM within {WIN_MIN:.0f} min BEFORE onset  : mean frac "
          f"{np.nanmean(ev_df['rem_frac_pre']):.3f}")
    print(f"REM within {WIN_MIN:.0f} min AFTER offset  : mean frac "
          f"{np.nanmean(ev_df['rem_frac_post']):.3f}")
    print(f"Null (random NREM) pre/post REM frac       : "
          f"{np.nanmean(sm_df['null_rem_frac_pre']):.3f} / "
          f"{np.nanmean(sm_df['null_rem_frac_post']):.3f}")
    print(f"\nMedian minutes to nearest REM  before={np.nanmedian(ev_df['rem_min_before']):.1f}  "
          f"after={np.nanmedian(ev_df['rem_min_after']):.1f}")
    # per-subject direction: is REM more BEFORE or AFTER?
    print("\nPer-subject: REM closer BEFORE onset or AFTER offset?")
    for subj, g in ev_df.groupby('subject'):
        b = np.nanmedian(g['rem_min_before'])
        a = np.nanmedian(g['rem_min_after'])
        who = 'BEFORE' if (np.isfinite(b) and (not np.isfinite(a) or b < a)) else \
              ('AFTER' if np.isfinite(a) else '—')
        print(f"  {subj}: median before={b:.0f}m after={a:.0f}m -> REM nearer {who}")
    print(f"\nWrote {REPORT_DIR/'ladder_events.csv'} and {REPORT_DIR/'ladder_stage_summary.csv'}")
    print(f"Figure {FIG_DIR/'ladder_stage_relationship.png'}")


if __name__ == '__main__':
    main()
