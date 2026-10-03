"""
Event-locked averages of the de-stepped SEC markers at sleep-stage transitions.

Explaining a whole night has not worked. Head orientation accounts for a third
of the slow signal, the overnight drift turned out to be posture, and no stage
contrast survived per recording. Event-locking asks a narrower question that
this kind of record can actually answer: does the marker MOVE in the minutes
around a change of brain state, and does it move the same way on more than one
night?

Three decisions make or break it, and each comes from something that went wrong
earlier in this work:

1. The marker is DE-STEPPED (diff_motion_regressed.destep). Without that, any
   transition near a movement returns the movement.
2. Transitions within EXCLUDE of a head movement are DROPPED. Stage changes and
   position changes co-occur often enough that keeping them would measure
   posture and report it as physiology. This is the test, not a nicety -- the
   printed table shows what each criterion costs in events.
3. The trace is NOT causally smoothed. The 5-minute causal median used for
   display lags by half its window, which would shift every response later by
   the same amount. Averaging across events supplies the smoothing instead.

The window length is set by the hypnograms, not by preference. These scorings
are heavily fragmented: N3 bouts run a median of one to two 30-second epochs and
REM bouts reach only five to seven, so the twelve nights contain almost no
transition with fifteen clear minutes either side. A long window is therefore
not available, and the analysis runs at two scales that the data does support:

    transitions   +/- 5 min, each stage held >= 1 min
    sleep onset   +/- 20 min, sleep sustained >= 10 min

The short scale is honest about what it is. With N3 bouts of 30-60 s, the window
around an N3 onset also contains the N3 offset; the average is a response to a
brief excursion into N3, not to a sustained state. Sleep onset is the one event
in these records that is long on both sides by construction, which is why it
gets its own panel.

Each event is baselined on its own pre-window and averaged WITHIN a recording
before anything is averaged across recordings, so one restless night with many
transitions cannot carry the result. The null is the same procedure at matched
random times in the same recording, away from movement -- it absorbs the drift
and the autocorrelation of the trace, which a flat zero line would not.

Reported descriptively: the response per recording, the sign count across the
twelve, how many exceed the recording's own null spread, and the per-subject
mean. No pooled p-values; the recording is the unit.

What it finds
-------------
CH responds to N3 and the differential does not.

    into N3     CH  10/12 recordings fall,  median -1.6 fF
    out of N3   CH   9/11 recordings rise,  median +0.8 fF
    into REM    CH   5/6  recordings fall,  median -0.9 fF
    into N3     CLE-CRE  6/12   -- chance
    out of N3   CLE-CRE  7/11   -- chance

The control the result rests on is the mirror: 10 of the 11 recordings with
both event types move one way into N3 and the other way out of it. Shifting
every event of a recording by a common random offset, which preserves the trace
and the spacing between entries and exits but destroys the alignment to the
scorer, gives a mean of 4.5 opposite-sign recordings and never once reaches 10
in 400 shifts (p = 0.003).

Two honest limits. Per recording the response does not clear its own null
spread (|z| > 2 in 0-1 of 12), so what carries this is consistency of direction
across nights, not size within a night. And one night, S6N2, moves about twenty
times further than the rest; it is excluded from no statistic here, but the
median and the sign count are what should be quoted, not the mean.

This agrees in direction with the independent variance result -- band amplitude
falls from Wake to N3 and returns in REM, also carried by CH -- which is the
same channel and the same sign from a different quantity on the same nights.

Writes  reports/mean_value/stage_event_locked.csv
        reports/mean_value/stage_event_counts.csv
        reports/mean_value/hypnogram_fragmentation.csv
        writeup/figures/imbalance/fig_stage_event_locked.png
        writeup/figures/imbalance/fig_sleep_onset_locked.png

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/stage_event_locked.py
"""

from __future__ import annotations

import importlib.util
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

from sleep_monitor.sessions import SESSION_META      # noqa: E402

FIG = ROOT / 'writeup' / 'figures' / 'imbalance'
TAB = ROOT / 'reports' / 'mean_value'
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

NULL_PER_EVENT = 8               # matched random times per real event
MIN_REC = 6                      # fewest recordings an event may be plotted from

# stage codes, from sleep_monitor: 4 Wake, 3 N1, 2 N2, 1 N3, 0 REM
TRANSITIONS = {
    'into N3':    lambda a, b: a == 2 and b == 1,
    'out of N3':  lambda a, b: a == 1 and b == 2,
    'into REM':   lambda a, b: a in (1, 2, 3) and b == 0,
    'out of REM': lambda a, b: a == 0 and b in (1, 2, 3),
    'into Wake':  lambda a, b: a in (0, 1, 2) and b == 4,
}
ONSET = {'return to sleep': lambda a, b: a == 4 and b in (0, 1, 2, 3)}

# win, baseline, response, stability either side, movement exclusion -- minutes.
#
# `collapse` is the series the stability test runs on, which is not always the
# stage series. The second scale has to require that SLEEP persists, not that
# one stage does: no single stage holds for five minutes often enough in these
# scorings, so measuring stability on the raw codes rejects nearly every return
# to sleep in the dataset. The collapse to wake-versus-sleep asks the question
# that was meant.
#
# `stable_pre` is 0 for that scale, and deliberately. Requiring even one minute
# of continuous prior wakefulness leaves ONE usable event in the whole dataset,
# because wake bouts here run a median of a single 30-second epoch. So this is
# not sleep onset in the clinical sense -- nothing in these recordings is, once
# the settling movement is excluded. It is a return to sustained sleep after a
# brief awakening, which is the version of the event the data contains.
SCALES = {
    'transitions': dict(win=5.0, base=(-5.0, -3.0), resp=(0.0, 2.0),
                        stable_pre=1.0, stable_post=1.0, excl=1.5,
                        events=TRANSITIONS, collapse=None,
                        fname='fig_stage_event_locked.png'),
    'return to sleep': dict(win=10.0, base=(-10.0, -6.0), resp=(0.0, 5.0),
                            stable_pre=0.0, stable_post=5.0, excl=2.0,
                            events=ONSET,
                            collapse=lambda c: (c == 4).astype(int),
                            fname='fig_sleep_onset_locked.png'),
}
CHANS = [('CLE-CRE', 'd_destep', '#B9380B'), ('CH', 'ch_destep', '#1F618D')]

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


def run_lengths(codes):
    """Per sample: blocks the current stage run has lasted, blocks remaining."""
    n = len(codes)
    bounds = np.r_[np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]), n]
    before = np.empty(n, int)
    after = np.empty(n, int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = np.arange(a, b)
        before[idx] = idx - a
        after[idx] = b - idx
    return before, after


def sign_p(agree, n):
    """Two-sided exact binomial on the recording-level sign count.

    Reported as a column, not as a headline. With n = 12 recordings there is no
    pooling of epochs here, so the number means what it says -- but the quantity
    a reader should look at first is the sign count itself.
    """
    if n == 0:
        return np.nan
    from math import comb
    tail = sum(comb(n, k) for k in range(agree, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def extract(y, idx, half, b0, b1):
    """Windows around idx, each minus the median of its own baseline."""
    keep = []
    for i in np.atleast_1d(np.asarray(idx, int)).ravel():
        a, b = i - half, i + half
        if a < 0 or b > len(y):
            continue
        seg = np.asarray(y[a:b], float)
        base = np.nanmedian(seg[b0:b1])
        if np.isfinite(base):
            keep.append(seg - base)
    return np.array(keep) if keep else np.empty((0, 2 * half))


def fragmentation(sessions):
    """Bout lengths per stage, in 30-second epochs, as the scale constraint."""
    rows = []
    for label, codes_ep in sessions:
        c = np.asarray(codes_ep)
        e = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
        rl, st = np.diff(np.r_[e, len(c)]), c[e]
        for code, name in [(1, 'N3'), (0, 'REM'), (2, 'N2'), (3, 'N1'),
                           (4, 'Wake')]:
            r = rl[st == code]
            rows.append(dict(session=label, stage=name, n_bouts=len(r),
                             median_epochs=float(np.median(r)) if len(r) else np.nan,
                             max_epochs=int(r.max()) if len(r) else 0,
                             bouts_ge_10min=int((r >= 20).sum())))
    return pd.DataFrame(rows)


def run_scale(name, cfg, cache):
    bs = dmr.BLOCK_S
    half = int(round(cfg['win'] * 60 / bs))
    b0 = int(round((cfg['base'][0] + cfg['win']) * 60 / bs))
    b1 = int(round((cfg['base'][1] + cfg['win']) * 60 / bs))
    r0 = int(round((cfg['resp'][0] + cfg['win']) * 60 / bs))
    r1 = int(round((cfg['resp'][1] + cfg['win']) * 60 / bs))
    excl = int(round(cfg['excl'] * 60 / bs))
    s_pre = int(round(cfg['stable_pre'] * 60 / bs))
    s_post = int(round(cfg['stable_post'] * 60 / bs))
    tax = np.arange(-half, half) * bs / 60.0
    events = cfg['events']

    subj = {m['label']: m['subject'] for m in SESSION_META}
    real = {e: {c: [] for c, _, _ in CHANS} for e in events}
    null = {e: {c: [] for c, _, _ in CHANS} for e in events}
    rows, counts = [], []
    rng = np.random.default_rng(0)

    for r in cache:
        codes, moving = r['codes'], r['moving']
        near = np.convolve(moving, np.ones(2 * excl + 1, bool), 'same') > 0
        col = cfg['collapse']
        before, after = run_lengths(codes if col is None else col(codes))
        pool = np.flatnonzero(~near)
        pool = pool[(pool >= half) & (pool <= len(codes) - half)]

        for ename, test in events.items():
            hit = np.array([test(codes[i - 1], codes[i])
                            for i in range(1, len(codes))], bool)
            idx = np.flatnonzero(hit) + 1
            rec = dict(session=r['label'], event=ename, all=len(idx))
            idx = idx[(before[idx - 1] >= s_pre) & (after[idx] >= s_post)]
            rec['stage_held'] = len(idx)
            idx = idx[~near[idx]]
            rec['movement_free'] = len(idx)
            counts.append(rec)
            if len(idx) == 0:
                continue

            nid = (rng.choice(pool, size=min(len(idx) * NULL_PER_EVENT,
                                             pool.size), replace=False)
                   if pool.size else np.array([], int))
            for cname, key, _ in CHANS:
                A = extract(r[key], idx, half, b0, b1)
                if not len(A):
                    continue
                mu = np.nanmean(A, axis=0)
                real[ename][cname].append(mu)
                resp = float(np.nanmean(mu[r0:r1]))

                N = extract(r[key], nid, half, b0, b1)
                if len(N) > 1:
                    null[ename][cname].append(np.nanmean(N, axis=0))
                    per = np.array([np.nanmean(w[r0:r1]) for w in N])
                    nmean, nsd = float(np.nanmean(per)), float(np.nanstd(per, ddof=1))
                    # the null for an AVERAGE of len(A) events narrows by sqrt(n)
                    nse = nsd / np.sqrt(len(A))
                else:
                    nmean = nse = np.nan
                rows.append(dict(
                    session=r['label'], subject=subj[r['label']], scale=name,
                    event=ename, channel=cname, n_events=len(A),
                    response_fF=resp, null_mean_fF=nmean, null_se_fF=nse,
                    z_vs_null=(resp - nmean) / nse
                    if nse and np.isfinite(nse) and nse > 0 else np.nan))

    cdf = pd.DataFrame(counts)
    df = pd.DataFrame(rows)

    print(f'\n=== {name}:  +/-{cfg["win"]:.0f} min window, stage held '
          f'>={cfg["stable_pre"]:.0f}/{cfg["stable_post"]:.0f} min, '
          f'>={cfg["excl"]:.1f} min clear of movement ===')
    print(f"{'event':12s} {'all':>5s} {'stage held':>11s} "
          f"{'movement-free':>14s}  recordings")
    for e in events:
        s = cdf[cdf.event == e]
        print(f'{e:12s} {s["all"].sum():5d} {s["stage_held"].sum():11d} '
              f'{s["movement_free"].sum():14d}  '
              f'{(s["movement_free"] > 0).sum()}/12')

    if df.empty:
        return cdf, df, None
    print(f'\nresponse = mean of {cfg["resp"][0]:.0f} to {cfg["resp"][1]:.0f} '
          f'min, baselined on {cfg["base"][0]:.0f} to {cfg["base"][1]:.0f} min '
          f'(fF)')
    print(f"{'event':15s} {'chan':8s} {'rec':>4s} {'ev':>4s} {'median':>8s} "
          f"{'same sign':>11s} {'sign p':>8s} {'|z|>2':>8s}  per-subject means")
    for e in events:
        for cname, _, _ in CHANS:
            s = df[(df.event == e) & (df.channel == cname)].dropna(
                subset=['response_fF'])
            if s.empty:
                continue
            pos = int((s.response_fF > 0).sum())
            agree = max(pos, len(s) - pos)
            bysub = s.groupby('subject').response_fF.mean()
            print(f'{e:15s} {cname:8s} {len(s):4d} {int(s.n_events.sum()):4d} '
                  f'{s.response_fF.median():+8.2f} '
                  f'{agree:>7d}/{len(s):<3d} {sign_p(agree, len(s)):8.3f} '
                  f'{int((s.z_vs_null.abs() > 2).sum()):5d}/{len(s):<3d}  '
                  + ' '.join(f'{v:+.1f}' for v in bysub))

    # ---- figure: a column per event, a row per channel ------------------
    # Only events that reached MIN_REC recordings are drawn. Three of the five
    # transition types survive the filters on fewer than half the nights, and a
    # panel built from three recordings invites a reading it cannot support.
    # Their counts stay in the table above and in the CSV.
    shown = [e for e in events
             if max(len(real[e][c]) for c, _, _ in CHANS) >= MIN_REC]
    dropped = [e for e in events if e not in shown]
    if dropped:
        print(f'  not plotted (<{MIN_REC} recordings): ' + ', '.join(dropped))
    if not shown:
        return cdf, df, None

    ne = len(shown)
    fig, axes = plt.subplots(len(CHANS), ne, squeeze=False,
                             figsize=(max(5.2, 4.6 * ne), 4.6 * len(CHANS)),
                             sharex=True)
    for rr, (cname, _, col) in enumerate(CHANS):
        for cc, ename in enumerate(shown):
            ax = axes[rr][cc]
            stack = real[ename][cname]
            lo, hi = None, None
            if stack:
                A = np.vstack(stack)
                for w in A:
                    ax.plot(tax, w, lw=1.0, color=col, alpha=0.32)
                mu = np.nanmean(A, axis=0)
                if len(A) > 1:
                    se = np.nanstd(A, axis=0, ddof=1) / np.sqrt(len(A))
                    ax.fill_between(tax, mu - se, mu + se, color=col,
                                    alpha=0.22, lw=0)
                ax.plot(tax, mu, lw=3.4, color=col,
                        label=f'mean of {len(A)} recordings')
                # Robust limits. One night reaches -110 fF entering N3 and on a
                # shared scale it flattens every other recording in the panel
                # to a horizontal line. Take the 10th-90th percentile ACROSS
                # recordings at each time point, which excludes roughly the one
                # extreme night per side and keeps the rest readable, then widen
                # to whatever the mean needs. Nights that leave the axis are
                # counted on the panel rather than silently cropped.
                lo = float(np.nanmin(np.nanpercentile(A, 10, axis=0)))
                hi = float(np.nanmax(np.nanpercentile(A, 90, axis=0)))
                lo, hi = min(lo, np.nanmin(mu)), max(hi, np.nanmax(mu))
                off = max(hi - lo, 1.0) * 0.18
                lo, hi = lo - off, hi + off
                ax.set_ylim(lo, hi)
                n_out = int(((A < lo) | (A > hi)).any(axis=1).sum())
                if n_out:
                    ax.annotate(f'{n_out} recording'
                                f'{"s" if n_out > 1 else ""} off scale',
                                (0.97, 0.04), xycoords='axes fraction',
                                ha='right', fontsize=12, color='#777777')
            ns = null[ename][cname]
            if ns:
                ax.plot(tax, np.nanmean(np.vstack(ns), axis=0), lw=2.2,
                        color='#555555', ls=':', label='matched random times')
            ax.axvspan(*cfg['resp'], color='#f0c000', alpha=0.13, lw=0)
            ax.axvline(0, color='#b3282d', ls='--', lw=1.8)
            ax.axhline(0, color='#2C3E50', lw=1.0)
            ax.grid(alpha=0.20)
            s = df[(df.event == ename) & (df.channel == cname)].dropna(
                subset=['response_fF'])
            if len(s):
                pos = int((s.response_fF > 0).sum())
                ax.set_title(f'{ename}\n{max(pos, len(s) - pos)}/{len(s)} '
                             f'recordings agree in sign', loc='left',
                             fontsize=16)
            if rr == len(CHANS) - 1:
                ax.set_xlabel('minutes from transition')
            if cc == 0:
                ax.set_ylabel(f'{cname}\nfF, baselined')
            if rr == 0 and cc == 0:
                ax.legend(loc='upper left', frameon=False, fontsize=12)
    fig.tight_layout()
    p = FIG / cfg['fname']
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {p.name}')
    return cdf, df, p


def mirror(df):
    """Does each recording move one way into N3 and the other way out of it?

    This is the control the design turns on. A marker that genuinely tracks
    sleep depth has to reverse when the transition reverses. A marker that is
    drifting, or that is still carrying residual motion, has no reason to: it
    will give whatever sign the local trend happens to have, independently at
    each event type. Nothing about the two event sets shares a baseline or a
    window, so the pairing is not built in.
    """
    print('\nmirror check: entering and leaving N3 should move oppositely')
    print(f"{'chan':8s} {'recordings':>11s} {'opposite signs':>15s} "
          f"{'p':>7s}  per recording")
    for cname, _, _ in CHANS:
        a = df[(df.event == 'into N3') & (df.channel == cname)].set_index(
            'session').response_fF
        b = df[(df.event == 'out of N3') & (df.channel == cname)].set_index(
            'session').response_fF
        both = pd.concat([a.rename('in'), b.rename('out')], axis=1).dropna()
        if both.empty:
            continue
        opp = int(((both['in'] < 0) & (both['out'] > 0)).sum()
                  + ((both['in'] > 0) & (both['out'] < 0)).sum())
        print(f'{cname:8s} {len(both):11d} {opp:15d} '
              f'{sign_p(max(opp, len(both) - opp), len(both)):7.3f}  '
              + ' '.join(f'{s}:{"opp" if (x < 0) != (y < 0) else "same"}'
                         for s, x, y in zip(both.index, both["in"],
                                            both["out"])))


def mirror_null(cache, cfg, n_perm=400, seed=1):
    """Is the mirror real, or is it the shape of the windows?

    N3 bouts last a median of 30-60 s, which looks like it should make an entry
    window and the matching exit window overlap almost completely and share
    their baseline. Measured on the events actually used, it does not: the
    stability and movement filters keep so few events that an entry's nearest
    surviving exit sits a median of 10 minutes later, well outside the 2-minute
    response window. The function prints that gap, because the assumption was
    wrong and the number is what settles it.

    The control shifts every event of a recording by one common random offset,
    wrapped around the night. That keeps the trace, the number of events of each
    type, and crucially the relative timing between entries and exits, so any
    mirror the window geometry can produce by itself survives the shift. What
    the shift destroys is the alignment to the scorer's stages. The real count
    is then read against the distribution of shifted counts.

    Also prints how far apart the paired events actually are, so the overlap is
    a measured quantity in the record rather than an assumption.
    """
    bs = dmr.BLOCK_S
    half = int(round(cfg['win'] * 60 / bs))
    b0 = int(round((cfg['base'][0] + cfg['win']) * 60 / bs))
    b1 = int(round((cfg['base'][1] + cfg['win']) * 60 / bs))
    r0 = int(round((cfg['resp'][0] + cfg['win']) * 60 / bs))
    r1 = int(round((cfg['resp'][1] + cfg['win']) * 60 / bs))
    excl = int(round(cfg['excl'] * 60 / bs))
    stable = int(round(cfg['stable_pre'] * 60 / bs))
    rng = np.random.default_rng(seed)

    sel, gaps = [], []
    for r in cache:
        codes, n = r['codes'], len(r['codes'])
        near = np.convolve(r['moving'], np.ones(2 * excl + 1, bool), 'same') > 0
        before, after = run_lengths(codes)
        picked = {}
        for ename in ('into N3', 'out of N3'):
            test = TRANSITIONS[ename]
            idx = np.flatnonzero(np.array(
                [test(codes[i - 1], codes[i]) for i in range(1, n)], bool)) + 1
            idx = idx[(before[idx - 1] >= stable) & (after[idx] >= stable)]
            picked[ename] = idx[~near[idx]]
        if len(picked['into N3']) and len(picked['out of N3']):
            sel.append((r, picked))
            for i in picked['into N3']:
                nxt = picked['out of N3'][picked['out of N3'] > i]
                if nxt.size:
                    gaps.append((nxt[0] - i) * bs / 60.0)

    def count_opposite(shift_each):
        opp = 0
        for r, picked in sel:
            n = len(r['codes'])
            sh = rng.integers(half, n - half) if shift_each else 0
            vals = {}
            for ename, idx in picked.items():
                j = (idx + sh) % n
                A = extract(r['ch_destep'], j, half, b0, b1)
                vals[ename] = (float(np.nanmean(np.nanmean(A, axis=0)[r0:r1]))
                               if len(A) else np.nan)
            a, b = vals['into N3'], vals['out of N3']
            if np.isfinite(a) and np.isfinite(b) and (a < 0) != (b < 0):
                opp += 1
        return opp

    real = count_opposite(False)
    perm = np.array([count_opposite(True) for _ in range(n_perm)])
    p = float((perm >= real).mean())
    print(f'\nmirror control on CH, {len(sel)} recordings with both event types')
    if gaps:
        g = np.array(gaps)
        share = float((g < cfg['resp'][1]).mean())
        print(f'  entry to matching exit: median {np.median(g):.1f} min '
              f'[{np.percentile(g, 25):.1f}-{np.percentile(g, 75):.1f}]; '
              f'{share:.0%} of pairs are closer than the '
              f'{cfg["resp"][1]:.0f}-min response window')
    print(f'  opposite signs, as scored:     {real}/{len(sel)}')
    print(f'  opposite signs, events shifted: mean {perm.mean():.1f}, '
          f'95th pct {np.percentile(perm, 95):.0f}, max {perm.max()}  '
          f'({n_perm} shifts)')
    print(f'  shifted runs reaching the real count: p = {p:.3f}')
    return real, perm, p


def main():
    cache, hyp = [], []
    for meta in SESSION_META:
        r = dmr.one_session(meta)
        cache.append(r)
        s = dmr.load_session(meta)
        hyp.append((meta['label'], dmr.load_sleep_profile(s)['codes']))

    frag = fragmentation(hyp)
    frag.to_csv(TAB / 'hypnogram_fragmentation.csv', index=False)
    print('\nhypnogram bout lengths, 30-second epochs, across the 12 nights')
    print(f"{'stage':6s} {'bouts/night':>12s} {'median bout':>12s} "
          f"{'longest bout':>13s} {'bouts >=10 min':>15s}")
    for st in ['Wake', 'N1', 'N2', 'N3', 'REM']:
        s = frag[frag.stage == st]
        print(f'{st:6s} {s.n_bouts.median():12.0f} '
              f'{s.median_epochs.median():10.1f} ep '
              f'{s.max_epochs.max():11d} ep {s.bouts_ge_10min.sum():15d}')

    cdfs, dfs = [], []
    for name, cfg in SCALES.items():
        c, d, _ = run_scale(name, cfg, cache)
        cdfs.append(c.assign(scale=name))
        if not d.empty:
            dfs.append(d)

    mirror(pd.concat(dfs, ignore_index=True))
    mirror_null(cache, SCALES['transitions'])

    pd.concat(cdfs, ignore_index=True).to_csv(
        TAB / 'stage_event_counts.csv', index=False)
    pd.concat(dfs, ignore_index=True).to_csv(
        TAB / 'stage_event_locked.csv', index=False)
    print(f'\n-> {TAB / "stage_event_locked.csv"}')
    print(f'-> {TAB / "stage_event_counts.csv"}')
    print(f'-> {TAB / "hypnogram_fragmentation.csv"}')


if __name__ == '__main__':
    main()
