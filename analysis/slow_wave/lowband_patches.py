"""
Bright low-frequency patches (0.01-0.03 Hz) in SEC spectrograms: what are they?

Zooming the spectrogram below respiration shows short bright patches below
0.03 Hz (5-15 min; e.g. S4N2 CLE ~3.7 h, CH ~2.7 and ~3.6 h), and many of them
appear to sit on stage changes or head movements. This detects them in every
night and channel and asks how often they coincide with each, against chance.

Two versions of each channel
    raw        the channel decimated to 2 Hz, band-passed 0.005-0.9 Hz
    corrected  the channel with head-movement steps removed (10-s blocks,
               diff_motion_regressed.destep), i.e. the trace of Figures S10-S12

A head movement moves the SEC level in a step, and a step has broadband power
that is largest at the lowest frequencies -- so a patch that disappears after
correction was the step. 10-s blocks (Nyquist 0.05 Hz) still cover 0.01-0.03 Hz.

Spectrogram   4-min windows, 30-s step (0.0042 Hz resolution), the 1/f trend
              removed per column, light smoothing -- the same treatment as the
              Fig. 5 low band, on a window long enough for 0.01 Hz.
Patch         the 0.01-0.03 Hz band power at least PATCH_DB above that night's
              median, for at least MIN_MIN minutes.
Coincidence   a patch "has movement" if any movement block falls within it or
              PAD_MIN either side; "has a stage change" likewise for a change of
              scored stage.
Null          the same patches moved to random times in the same night (circular
              shift, durations kept), N_NULL times; the observed fraction is
              reported against that null's median and 95th percentile.

Writes  reports/slow_wave/low_band/lowband_patches.csv            one row per patch
        reports/slow_wave/low_band/lowband_patches_summary.csv    per night x channel x version
        writeup/figures/harmonics/lowband_patches/patches_{S}.png
        writeup/figures/harmonics/lowband_patches/patches_summary.png

Usage
    .venv/Scripts/python.exe analysis/slow_wave/lowband_patches.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from scipy.ndimage import gaussian_filter            # noqa: E402
from scipy.signal import butter, decimate, sosfiltfilt, spectrogram  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / 'mean_value'))

import diff_motion_regressed as dmr                  # noqa: E402
from sleep_monitor import load_session, load_sleep_profile           # noqa: E402
from sleep_monitor.config import STAGE_COLORS, STAGE_LABELS, STAGE_ORDER  # noqa: E402
from sleep_monitor.filters import lowpass            # noqa: E402
from sleep_monitor.sessions import SESSION_META      # noqa: E402

FIG = ROOT / 'writeup' / 'figures' / 'harmonics' / 'lowband_patches'
TAB = ROOT / 'reports' / 'slow_wave' / 'low_band'
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
WIN_S, STEP_S = 240.0, 30.0
BAND = (0.01, 0.03)
SHOW = (0.008, 0.12)        # frequency range drawn
PATCH_DB = 6.0
MIN_MIN = 5.0
PAD_MIN = 2.0
N_NULL = 500
BLUR = (2.0, 2.0)           # (freq bins, time columns)


def lowband_map(x, fs):
    """Spectrogram up to SHOW[1] Hz, each frequency referenced to its own night median.

    Per-frequency referencing, not a per-column 1/f fit: a patch is a stretch of
    time in which the low band is stronger than it usually is, and a per-column
    fit removes exactly that (a column-wide low-frequency boost).
    """
    f, t, P = spectrogram(x, fs=fs, nperseg=int(WIN_S * fs),
                          noverlap=int((WIN_S - STEP_S) * fs),
                          nfft=max(int(WIN_S * fs), 2048), detrend='linear')
    m = (f >= SHOW[0]) & (f <= SHOW[1])
    f, P = f[m], P[m]
    db = 10 * np.log10(P + 1e-20)
    db = db - np.median(db, axis=1, keepdims=True)
    return f, t / 3600.0, gaussian_filter(db, BLUR)


def band_excess(f, db):
    """0.01-0.03 Hz band power per column, dB above the night's median."""
    b = (f >= BAND[0]) & (f <= BAND[1])
    p = db[b].mean(axis=0)
    return p - np.median(p), f[b][np.argmax(db[b], axis=0)]


def runs(mask, need):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= need:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def signals(meta):
    """Raw (2 Hz) and movement-corrected (10-s blocks) versions of each channel."""
    with contextlib.redirect_stdout(io.StringIO()):
        s = load_session(meta)
        prof = load_sleep_profile(s)
    fs = s.fs
    n = int(dmr.BLOCK_S * fs)
    g = np.column_stack([dmr.blocks(lowpass(np.asarray(s.cap[a], float), dmr.GRAV_HZ, fs), n)
                         for a in ('aX', 'aY', 'aZ')])
    raw, cor = {}, {}
    sos = butter(2, [0.005, 0.9], btype='band', fs=2.0, output='sos')
    for ch in CHANNELS:
        x = np.asarray(s.cap[ch], float)
        y = decimate(decimate(x, 10, ftype='fir', zero_phase=True), int(fs / 20),
                     ftype='fir', zero_phase=True)
        raw[ch] = sosfiltfilt(sos, y)
        yb = dmr.blocks(x, n)
        yb = yb - np.nanmean(yb)
        d, moving = dmr.destep(yb, g, dmr.STEP_THRESH, dmr.STEP_PAD)
        cor[ch] = d
    t_blk = (np.arange(len(moving)) * dmr.BLOCK_S + dmr.BLOCK_S / 2) / 3600.0
    tep, cc = np.asarray(prof['t_ep_hr']), np.asarray(prof['codes'])
    return dict(label=meta['label'], fs_raw=2.0, fs_cor=1.0 / dmr.BLOCK_S, raw=raw,
                cor=cor, moving=moving, t_blk=t_blk, t_ep=tep, codes=cc)


def events_on_grid(S, t_col):
    """Per spectrogram column: movement in the column window, and stage change."""
    half = WIN_S / 2 / 3600.0
    mv = np.array([S['moving'][(S['t_blk'] >= t - half) & (S['t_blk'] <= t + half)].any()
                   for t in t_col])
    chg_t = S['t_ep'][1:][np.diff(S['codes']) != 0]
    return mv, chg_t


def coincidence(patches, t_col, mv_cols, chg_t):
    """Fraction of patches with movement / a stage change within PAD of them."""
    pad = PAD_MIN / 60.0
    step = STEP_S / 3600.0
    has_mv, has_ch = [], []
    for a, b in patches:
        t0, t1 = t_col[a] - pad, t_col[b] + pad
        cols = (t_col >= t0) & (t_col <= t1)
        has_mv.append(bool(mv_cols[cols].any()))
        has_ch.append(bool(((chg_t >= t0) & (chg_t <= t1)).any()))
    return np.array(has_mv, bool), np.array(has_ch, bool)


def null_fractions(patches, t_col, mv_cols, chg_t, rng):
    n = len(t_col)
    out = np.zeros((N_NULL, 2))
    for k in range(N_NULL):
        sh = int(rng.integers(n))
        moved = [((a + sh) % n, (a + sh) % n + (b - a)) for a, b in patches]
        moved = [(a, min(b, n - 1)) for a, b in moved]
        m, c = coincidence(moved, t_col, mv_cols, chg_t)
        out[k] = m.mean(), c.mean()
    return out


def stage_at(S, t):
    i = np.clip(np.searchsorted(S['t_ep'], t) - 1, 0, len(S['codes']) - 1)
    return STAGE_LABELS.get(int(S['codes'][i]), '?')


def figure(S, maps, patches_all):
    fig, axes = plt.subplots(7, 1, figsize=(15, 15), sharex=True,
                             gridspec_kw={'height_ratios': [0.25] + [1, 1] * 3})
    ax0 = axes[0]
    pos = {c: k for k, c in enumerate(STAGE_ORDER)}
    ax0.step(S['t_ep'], [pos.get(int(c), np.nan) for c in S['codes']], where='post', color='k', lw=1)
    ax0.set_yticks(range(5))
    ax0.set_yticklabels([STAGE_LABELS[c] for c in STAGE_ORDER], fontsize=8)
    for a, b in dmr._spans(S['moving']):
        ax0.axvspan(S['t_blk'][a], S['t_blk'][min(b, len(S['t_blk']) - 1)],
                    color='#E74C3C', alpha=0.35, lw=0)
    ax0.set_title(f"{S['label']}: 0.008–0.12 Hz, 4-min windows. Red bands = head "
                  f"movement. Boxes = patches ({BAND[0]}–{BAND[1]} Hz ≥ {PATCH_DB:g} dB "
                  f"above night median, ≥ {MIN_MIN:g} min)", loc='left', fontsize=11)
    k = 1
    for ch in CHANNELS:
        for ver in ('raw', 'cor'):
            ax = axes[k]
            f, t, db = maps[(ch, ver)]
            lo, hi = -6, 10
            ax.pcolormesh(t, f, db, shading='auto', cmap='viridis', vmin=lo, vmax=hi,
                          rasterized=True)
            ax.set_yscale('log')
            ax.set_ylim(*SHOW)
            ax.set_yticks([0.01, 0.02, 0.03, 0.05, 0.1])
            ax.set_yticklabels(['.01', '.02', '.03', '.05', '.1'], fontsize=8)
            for a, b in patches_all[(ch, ver)]:
                ax.add_patch(plt.Rectangle((t[a], BAND[0]), t[b] - t[a], BAND[1] - BAND[0],
                                           fill=False, ec='#FF2D55', lw=1.6))
            for a, b in dmr._spans(S['moving']):
                ax.axvspan(S['t_blk'][a], S['t_blk'][min(b, len(S['t_blk']) - 1)],
                           ymin=0.97, ymax=1.0, color='#E74C3C', lw=0)
            ax.set_ylabel(f"{ch}\n{'raw' if ver == 'raw' else 'steps removed'}",
                          fontsize=9)
            k += 1
    axes[-1].set_xlabel('time (h)')
    fig.tight_layout(h_pad=0.3)
    out = FIG / f"patches_{S['label']}.png"
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    prow, srow = [], []
    for meta in SESSION_META:
        S = signals(meta)
        maps, patches_all = {}, {}
        for ch in CHANNELS:
            for ver, fs in (('raw', S['fs_raw']), ('cor', S['fs_cor'])):
                x = S[ver][ch]
                x = np.where(np.isfinite(x), x, np.nanmedian(x))
                f, t, db = lowband_map(x, fs)
                ex, fpk = band_excess(f, db)
                P = runs(ex >= PATCH_DB, int(round(MIN_MIN * 60 / STEP_S)))
                maps[(ch, ver)], patches_all[(ch, ver)] = (f, t, db), P
                mv_cols, chg_t = events_on_grid(S, t)
                if P:
                    hm, hc = coincidence(P, t, mv_cols, chg_t)
                    nl = null_fractions(P, t, mv_cols, chg_t, rng)
                else:
                    hm = hc = np.array([], bool)
                    nl = np.full((1, 2), np.nan)
                for (a, b), m, c in zip(P, hm, hc):
                    prow.append(dict(session=S['label'], channel=ch, version=ver,
                                     t0_hr=t[a], t1_hr=t[b],
                                     dur_min=(b - a + 1) * STEP_S / 60,
                                     peak_db=float(ex[a:b + 1].max()),
                                     f_peak_hz=float(np.median(fpk[a:b + 1])),
                                     stage=stage_at(S, (t[a] + t[b]) / 2),
                                     movement=bool(m), stage_change=bool(c)))
                srow.append(dict(session=S['label'], channel=ch, version=ver,
                                 n_patches=len(P),
                                 frac_movement=float(hm.mean()) if len(P) else np.nan,
                                 null_movement_med=float(np.nanmedian(nl[:, 0])),
                                 null_movement_p95=float(np.nanpercentile(nl[:, 0], 95)),
                                 frac_stage_change=float(hc.mean()) if len(P) else np.nan,
                                 null_change_med=float(np.nanmedian(nl[:, 1])),
                                 null_change_p95=float(np.nanpercentile(nl[:, 1], 95))))
        figure(S, maps, patches_all)
        s_ = [r for r in srow if r['session'] == S['label']]
        print(f"  {S['label']}: " + '  '.join(
            f"{r['channel']}/{r['version']} {r['n_patches']}" for r in s_))
    pt = pd.DataFrame(prow)
    sm = pd.DataFrame(srow)
    pt.to_csv(TAB / 'lowband_patches.csv', index=False)
    sm.to_csv(TAB / 'lowband_patches_summary.csv', index=False)
    summary_figure(pt, sm)
    report(pt, sm)


def summary_figure(pt, sm):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, (ver, lbl) in enumerate((('raw', 'raw channel'), ('cor', 'steps removed'))):
        g = sm[sm.version == ver]
        tot = g.groupby('channel').n_patches.sum()
        axes[0].bar(np.arange(3) + (i - 0.5) * 0.38, [tot.get(c, 0) for c in CHANNELS],
                    0.38, label=lbl, color=['#7F8C8D', '#2E86C1'][i])
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(CHANNELS)
    axes[0].set_ylabel('patches, all 12 nights')
    axes[0].legend(frameon=False)
    for j, (col, ncol, title) in enumerate((('frac_movement', 'null_movement_med',
                                             'with head movement'),
                                            ('frac_stage_change', 'null_change_med',
                                             'with a stage change'))):
        ax = axes[j + 1]
        for i, ver in enumerate(('raw', 'cor')):
            g = sm[(sm.version == ver) & (sm.n_patches > 0)]
            x = np.arange(len(g)) * 0 + i
            ax.scatter(x + np.random.default_rng(j).uniform(-0.15, 0.15, len(g)), g[col],
                       s=22, color=['#7F8C8D', '#2E86C1'][i], zorder=3)
            ax.hlines(g[ncol].median(), i - 0.3, i + 0.3, color='k', ls='--', lw=1.2)
            ax.hlines(g[col].median(), i - 0.3, i + 0.3, color='k', lw=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['raw', 'steps removed'])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('fraction of patches')
        ax.set_title(f'{title}\n(solid = median night×channel, dashed = chance)',
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / 'patches_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def report(pt, sm):
    print('\npatches per version (12 nights x 3 channels):')
    for ver in ('raw', 'cor'):
        p = pt[pt.version == ver]
        g = sm[(sm.version == ver) & (sm.n_patches > 0)]
        print(f"  {ver}: {len(p)} patches, median {p.dur_min.median():.1f} min, "
              f"f {p.f_peak_hz.median():.3f} Hz; with movement {p.movement.mean():.0%} "
              f"(chance, median over night×channel: {g.null_movement_med.median():.0%}); "
              f"with stage change {p.stage_change.mean():.0%} "
              f"(chance {g.null_change_med.median():.0%}); "
              f"night×channel above null p95: movement "
              f"{int((g.frac_movement > g.null_movement_p95).sum())}/{len(g)}, change "
              f"{int((g.frac_stage_change > g.null_change_p95).sum())}/{len(g)}")
        print('   stage at patch centre:', p.stage.value_counts().to_dict())
        q = p[~p.movement]
        print(f"   without movement: {len(q)}; of those with stage change "
              f"{q.stage_change.mean():.0%}; stages {q.stage.value_counts().to_dict()}")


# ── sharper tests ────────────────────────────────────────────────────────────
# The coincidence fractions above turn out to be uninformative: movements and
# stage changes are so frequent in these fragmented hypnograms that ~85% of
# randomly placed 7-min windows contain both. Two measures that are not swamped:
#   1. stage enrichment of patch time (observed / expected), per night
#   2. event-locked band power around movement onsets, and around stage changes
#      with no movement within ISOL_MIN, each against random times

ISOL_MIN = 5.0
LOCK_MIN = 15.0


def _onsets(mask):
    return np.flatnonzero(np.diff(mask.astype(int), prepend=0) == 1)


def locked(ex, t, ev_t, rng, n_null=500):
    """Mean band excess from -LOCK to +LOCK min around event times, and a null."""
    w = int(round(LOCK_MIN * 60 / STEP_S))
    idx = [int(np.argmin(np.abs(t - e))) for e in ev_t]
    idx = [i for i in idx if w <= i < len(t) - w]
    if not idx:
        return None, None, 0
    seg = np.array([ex[i - w:i + w + 1] for i in idx])
    # null: the mean over the SAME number of random times, n_null times
    win = np.lib.stride_tricks.sliding_window_view(ex, 2 * w + 1)
    nul = np.array([win[rng.integers(0, len(win), len(idx))].mean(0)
                    for _ in range(n_null)])
    return seg.mean(0), nul, len(idx)


def sharper():
    rng = np.random.default_rng(1)
    w = int(round(LOCK_MIN * 60 / STEP_S))
    lag = np.arange(-w, w + 1) * STEP_S / 60
    enr, lk = [], []
    curves = {(v, k): [] for v in ('raw', 'cor') for k in ('movement', 'stage change')}
    for meta in SESSION_META:
        S = signals(meta)
        chg_all = S['t_ep'][1:][np.diff(S['codes']) != 0]
        mv_on = S['t_blk'][_onsets(S['moving'])]
        # stage changes with no movement within ISOL_MIN
        iso = np.array([c for c in chg_all
                        if not S['moving'][np.abs(S['t_blk'] - c) <= ISOL_MIN / 60].any()])
        for ch in CHANNELS:
            for ver, fs in (('raw', S['fs_raw']), ('cor', S['fs_cor'])):
                x = np.nan_to_num(S[ver][ch], nan=np.nanmedian(S[ver][ch]))
                f, t, db = lowband_map(x, fs)
                ex, _ = band_excess(f, db)
                on = ex >= PATCH_DB
                st = np.array([stage_at(S, ti) for ti in t])
                for s_ in ('Wake', 'N1', 'N2', 'N3', 'REM'):
                    e_ = (st == s_).mean()
                    if e_ > 0.01 and on.any():
                        enr.append(dict(session=S['label'], channel=ch, version=ver,
                                        stage=s_, enrichment=(st[on] == s_).mean() / e_))
                for name, ev in (('movement', mv_on), ('stage change', iso)):
                    m, nul, n = locked(ex, t, ev, rng)
                    if m is None:
                        continue
                    curves[(ver, name)].append(m)
                    peak = m[w - 2:w + 3].mean()                 # +/- 1 min around event
                    nd = nul[:, w - 2:w + 3].mean(1)
                    lk.append(dict(session=S['label'], channel=ch, version=ver,
                                   event=name, n_events=n, peak_db=peak,
                                   null_p95=float(np.percentile(nd, 95)),
                                   above=bool(peak > np.percentile(nd, 95))))
        print('  sharper:', S['label'], f'{len(mv_on)} movement onsets, '
              f'{len(iso)} isolated stage changes')
    enr, lk = pd.DataFrame(enr), pd.DataFrame(lk)
    enr.to_csv(TAB / 'lowband_patches_stage_enrichment.csv', index=False)
    lk.to_csv(TAB / 'lowband_patches_event_locked.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, ver in enumerate(('raw', 'cor')):
        ax = axes[i]
        for name, col in (('movement', '#C0392B'), ('stage change', '#2E86C1')):
            c = np.array(curves[(ver, name)])
            if not len(c):
                continue
            ax.plot(lag, c.T, color=col, lw=0.4, alpha=0.25)
            ax.plot(lag, c.mean(0), color=col, lw=2.4,
                    label=f'{name} ({len(c)} night×channel)')
        ax.axvline(0, color='k', lw=0.8)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_title(f"{'raw channel' if ver == 'raw' else 'steps removed'}: "
                     f'0.01–0.03 Hz power around events', fontsize=11)
        ax.set_xlabel('minutes from event')
        ax.set_ylabel('dB above night median')
        ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    for i, ver in enumerate(('raw', 'cor')):
        g = enr[enr.version == ver].groupby('stage').enrichment.median()
        ax.bar(np.arange(5) + (i - 0.5) * 0.38,
               [g.get(s_, np.nan) for s_ in ('Wake', 'N1', 'N2', 'N3', 'REM')], 0.38,
               color=['#7F8C8D', '#2E86C1'][i],
               label='raw' if ver == 'raw' else 'steps removed')
    ax.axhline(1, color='k', ls='--', lw=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    ax.set_ylabel('patch time: observed / expected')
    ax.set_title('stage of patch time (median night×channel)', fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'patches_events_and_stages.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('\nevent-locked peak (+/-1 min), night×channel above own null p95:')
    for (ver, ev), g in lk.groupby(['version', 'event']):
        print(f'  {ver:3s} {ev:12s} {int(g.above.sum())}/{len(g)}  median peak '
              f'{g.peak_db.median():+.1f} dB (null p95 median {g.null_p95.median():+.1f})')
    print('stage enrichment of patch time (median over night×channel):')
    for ver, g in enr.groupby('version'):
        m = g.groupby('stage').enrichment.median()
        print(f'  {ver}: ' + '  '.join(f'{k} {m[k]:.2f}' for k in
                                       ('Wake', 'N1', 'N2', 'N3', 'REM') if k in m))


if __name__ == '__main__':
    main()
    sharper()
