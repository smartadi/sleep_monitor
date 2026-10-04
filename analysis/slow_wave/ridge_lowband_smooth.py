"""
Low band (0-0.3 Hz) redone: a smooth, finer spectrogram, and a ridge tracked on it.

The bottom panel of the ridge_tune figures (ridge_overlay_tune.py) is a single
5-minute periodogram per column, blurred afterwards. A single periodogram has
~100% variance per bin, so the blur is fighting speckle it cannot remove, and
the 0.0033 Hz bin spacing renders as visible blocks. No ridge is drawn on it.

What this does instead
----------------------
0. Signal. The channel itself, decimated to 2 Hz and band-passed 0.005-0.5 Hz.
   NOT the rate pipeline's motion-cancelled signal: its 0.05 Hz corner makes
   the 0.05-0.08 Hz band the old panel showed, and its accelerometer
   regression adds ~6 dB of accelerometer content plus a 0.1447 Hz
   instrumental tone. Measured in low_band_signal's docstring.
1. Each column is a WELCH estimate over a 10-minute window (4-minute Hann
   segments, 75% overlap -> ~7 averaged segments), stepped every 30 s. The
   averaging is what removes the speckle; smoothing is no longer asked to.
   nfft is padded to a 0.001 Hz grid so the display is not blocky (padding
   interpolates, it does not add resolution; the true resolution is the
   segment's ~0.004 Hz).
2. The 1/f trend is removed per column by a straight-line fit of dB against
   log frequency, then a light Gaussian blur (1 min x 0.003 Hz).
3. Ridge: a Viterbi path through the smoothed map, restricted to 0.02-0.20 Hz
   (the top of the low band is respiration), with a frequency-change penalty
   set for this band -- much higher than the rate bands, because a 0.02 Hz
   move is a 25% change here.
4. Gate. The path counts as a ridge where it stands at least `gate` dB above
   that column's in-band median for at least MIN_RUN_MIN minutes. The gate is
   set per night from the night's own map, block-shuffled in time
   (shuffle_gate): the lowest level at which the shuffled maps show no more
   than 5% of the night as ridge. Elsewhere the path is drawn faint.

The tuning differs from the respiratory and cardiac trackers on purpose:
those rhythms are always present, so their path is never gated. A slow
oscillation is episodic, so presence is decided per column, against a null.

Outputs
    writeup/figures/harmonics/ridges_lowband/ridge_lowband_{S}_{CH}.png
    writeup/figures/harmonics/ridges_lowband/ridge_lowband_allsessions_{CH}.png
    reports/slow_wave/low_band/ridge_lowband_smooth.csv  (one row per night)
    reports/slow_wave/low_band/ridge_lowband_smooth_episodes.csv

Usage
    .venv/Scripts/python.exe analysis/slow_wave/ridge_lowband_smooth.py            # all 12, CRE
    .venv/Scripts/python.exe analysis/slow_wave/ridge_lowband_smooth.py --session S1N1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402
from scipy.ndimage import gaussian_filter, median_filter  # noqa: E402
from scipy.signal import butter, decimate, sosfiltfilt, welch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sleep_monitor import load_session, load_sleep_profile   # noqa: E402
from sleep_monitor.sessions import SESSION_META               # noqa: E402
from ridge_overlay_tune import (_sig, _enhance_spec, draw_stage_ladder,  # noqa: E402
                                track_single_ridge, TRACK, BAND_COLOR)

FIG = ROOT / 'writeup' / 'figures' / 'harmonics' / 'ridges_lowband'
TAB = ROOT / 'reports' / 'slow_wave' / 'low_band'
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

FS_LOW = 2.0             # decimated rate
WIN_S = 600.0            # one spectrogram column: 10 min
STEP_S = 30.0            # column step, on the epoch grid
SEG_S = 240.0            # Welch segment inside the column
NFFT_DF = 0.001          # display grid
FMIN, FMAX = 0.01, 0.30
HP_HZ = 0.005           # band-pass corner for the low band (see low_band_signal)
BLUR = (2.0, 3.0)        # (time cols = 1 min, freq bins = 0.003 Hz)

RIDGE_BAND = (0.02, 0.20)
PENALTY = 150.0          # dB per Hz: a 0.02 Hz jump costs 3 dB
PROM_DB = 3.0            # fallback only; the gate is set per night by shuffle_gate
N_SURR = 20              # block-shuffled nulls per night
SURR_FALSE = 0.05        # gate: surrogates may show at most 5% of the night as ridge
MIN_RUN_MIN = 15.0       # and for at least this long (> one 10-min shuffle block)
SMOOTH_COLS = 5          # median on the path (2.5 min)
RIDGE_COLOR = '#FF2D55'


def _to_low_rate(x, fs):
    q1, q2 = 10, int(round(fs / 10 / FS_LOW))
    y = decimate(decimate(np.asarray(x, float), q1, ftype='fir', zero_phase=True),
                 q2, ftype='fir', zero_phase=True)
    return y, fs / q1 / q2


def low_band_signal(cap, fs, hp=None):
    """The low-band channel, built at 2 Hz: band-passed, NOT motion-regressed.

    Two things went wrong with the rate pipeline's canceller here, and both are
    measured, on S1N1 CRE (whole-night Welch, 10-min segments):

    1. Its 0.05 Hz band-pass corner. In a 0-0.3 Hz spectrogram that is
       1/f-detrended, the corner becomes a hump just above 0.05 Hz in every
       column of every night -- the "0.05-0.08 Hz" band the old panel showed.
    2. The accelerometer regression itself. Below 0.5 Hz acc_mag is mostly
       head orientation, plus an instrumental tone at 0.1447 Hz (and 0.289 Hz)
       present on all twelve nights. Subtracting beta * acc_mag RAISES the
       channel's power by ~6 dB across the band (5.9 -> 11.6 dB at 0.13 Hz)
       and writes the tone into it (+19 dB at 0.1447 Hz); notching the tone
       first only leaves a dark line where it was.

    So the low band uses the channel itself, band-passed HP_HZ-0.5 Hz with
    second-order sections at 2 Hz. Movement shows up as broadband vertical
    stripes, which the ridge tracker does not follow.
    """
    hp = HP_HZ if hp is None else hp
    y, fs2 = _to_low_rate(cap, fs)
    sos = butter(2, [hp, 0.5], btype='band', fs=fs2, output='sos')
    return sosfiltfilt(sos, y), fs2


def low_band_map(y, fs2):
    """Smoothed, 1/f-detrended Welch spectrogram of the 0-0.3 Hz band."""
    win, step, seg = int(WIN_S * fs2), int(STEP_S * fs2), int(SEG_S * fs2)
    nfft = int(2 ** np.ceil(np.log2(fs2 / NFFT_DF)))
    starts = np.arange(0, len(y) - win + 1, step)
    cols = []
    for a in starts:
        f, p = welch(y[a:a + win], fs=fs2, nperseg=seg, noverlap=int(0.75 * seg),
                     nfft=nfft, window='hann', detrend='linear')
        cols.append(p)
    P = np.array(cols).T
    m = (f >= FMIN) & (f <= FMAX)
    f, P = f[m], P[m]
    db = 10 * np.log10(P + 1e-20)
    A = np.vstack([np.ones(m.sum()), np.log10(f)]).T
    coef, *_ = np.linalg.lstsq(A, db, rcond=None)
    db = gaussian_filter(db - A @ coef, sigma=BLUR)
    t = (starts + win / 2) / fs2 / 3600.0
    return f, t, db


def _runs_keep(on, need):
    keep = np.zeros(len(on), bool)
    i, n = 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            if j - i + 1 >= need:
                keep[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return keep


def shuffle_gate(f, db, n=N_SURR, seed=0):
    """Presence gate calibrated on the night's own map, block-shuffled in time.

    The map is cut into blocks one analysis window long (10 min, the span over
    which neighbouring columns share data) and the blocks are permuted. Every
    column keeps its real spectrum -- including motion bursts, which a
    stationary noise surrogate smears across the whole night and turns into
    permanent features -- but continuity across blocks is destroyed. A ridge
    is a claim of continuity, so this is the null it has to beat.

    The gate is the lowest prominence (dB) at which the shuffled maps show no
    more than SURR_FALSE of the night as ridge under the same run-length rule.

    Two earlier nulls were rejected: phase randomisation (every chance spike of
    the 8-hour periodogram becomes an all-night tone) and coloured noise from
    the smoothed spectrum (stationary, so a few motion bursts dominate it).
    """
    rng = np.random.default_rng(seed)
    blk = int(round(WIN_S / STEP_S))
    nt = db.shape[1]
    idx = [np.arange(i, min(i + blk, nt)) for i in range(0, nt, blk)]
    need = int(round(MIN_RUN_MIN * 60 / STEP_S))
    proms = []
    for _ in range(n):
        order = rng.permutation(len(idx))
        dbs = db[:, np.concatenate([idx[k] for k in order])]
        proms.append(track_lowband(f, dbs, thr=np.inf)[1])
    for thr in np.arange(1.0, 20.0, 0.25):
        frac = np.mean([_runs_keep(p >= thr, need).mean() for p in proms])
        if frac <= SURR_FALSE:
            return float(thr)
    return float('inf')


def track_lowband(f, db, thr=PROM_DB):
    """Viterbi path through the smoothed map, plus a per-column presence gate."""
    bm = (f >= RIDGE_BAND[0]) & (f <= RIDGE_BAND[1])
    fb, E = f[bm], db[bm]
    nf, nt = E.shape
    D = np.abs(fb[:, None] - fb[None, :])
    score = E[:, 0].copy()
    back = np.zeros((nf, nt), int)
    for i in range(1, nt):
        M = score[None, :] - PENALTY * D
        back[:, i] = np.argmax(M, axis=1)
        score = E[:, i] + M[np.arange(nf), back[:, i]]
    path = np.zeros(nt, int)
    path[-1] = int(np.argmax(score))
    for i in range(nt - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    path = median_filter(path, size=SMOOTH_COLS, mode='nearest')
    fr = fb[path]
    prom = E[path, np.arange(nt)] - np.median(E, axis=0)
    on = prom >= thr
    keep = _runs_keep(on, int(round(MIN_RUN_MIN * 60 / STEP_S)))
    return fr, prom, keep


def episodes(t, fr, prom, keep):
    out, i, n = [], 0, len(keep)
    while i < n:
        if keep[i]:
            j = i
            while j + 1 < n and keep[j + 1]:
                j += 1
            out.append(dict(t0_hr=t[i], t1_hr=t[j], dur_min=(j - i + 1) * STEP_S / 60,
                            f_med_hz=float(np.median(fr[i:j + 1])),
                            prom_med_db=float(np.median(prom[i:j + 1]))))
            i = j + 1
        else:
            i += 1
    return out


def figure(label, ch, prof, sig, fs, f, t, db, fr, keep):
    t_resp, resp_tr, resp_p = track_single_ridge(sig, fs, **TRACK['resp'])
    t_card, card_tr, card_p = track_single_ridge(sig, fs, **TRACK['card'])

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.32, 1.0, 0.72], hspace=0.14)
    ax0 = fig.add_subplot(gs[0])
    draw_stage_ladder(ax0, prof)
    ax0.tick_params(labelsize=12)
    ax0.set_title(f'{label} ({ch}) — respiratory / cardiac Viterbi traces '
                  f'(conf {resp_p:.0%}/{card_p:.0%}); low-band ridge '
                  f'{100 * keep.mean():.0f}% of the night', fontsize=15)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    f1, t1, db1 = _enhance_spec(sig, fs, 30, 15, 3.0, bg_hz=0.4)
    ax1.pcolormesh(t1 / 3600, f1, db1, shading='gouraud', cmap='magma',
                   vmin=0, vmax=np.percentile(db1, 99.5), rasterized=True)
    ax1.plot(t_resp, resp_tr, color=BAND_COLOR['resp'], lw=2.0, label='respiratory')
    ax1.plot(t_card, card_tr, color=BAND_COLOR['card'], lw=2.0, label='cardiac')
    ax1.set_ylim(0, 3.0)
    ax1.set_ylabel('Frequency (Hz)', fontsize=13)
    ax1.tick_params(labelsize=12)
    ax1.legend(loc='upper right', fontsize=11)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    lo, hi = np.percentile(db, [5, 99.5])
    ax2.pcolormesh(t, f, db, shading='gouraud', cmap='viridis', vmin=lo, vmax=hi,
                   rasterized=True)
    faint = np.where(keep, np.nan, fr)
    solid = np.where(keep, fr, np.nan)
    ax2.plot(t, faint, color='white', lw=1.0, alpha=0.45, ls=':')
    ax2.plot(t, solid, color=RIDGE_COLOR, lw=2.6, label='low-band ridge')
    ax2.set_ylim(0.0, FMAX)
    ax2.set_ylabel('Low band (Hz)', fontsize=13)
    ax2.tick_params(labelsize=12)
    ax2.set_xlabel('Time (hr)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=11)
    out = FIG / f'ridge_lowband_{label}_{ch}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


def sheet(results, ch):
    """All twelve low-band panels, three columns by four rows (fits a 16:9 slide)."""
    ncol, nrow = 3, 4
    fig = plt.figure(figsize=(27, 15))
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[0.22, 1.0] * nrow,
                          hspace=0.34, wspace=0.10)
    for k, r in enumerate(results):
        c, rw = k % ncol, k // ncol
        a0 = fig.add_subplot(gs[rw * 2, c])
        a1 = fig.add_subplot(gs[rw * 2 + 1, c], sharex=a0)
        draw_stage_ladder(a0, r['prof'])
        a0.set_ylabel('')
        a0.tick_params(labelbottom=False, labelsize=7)
        a0.set_title(f"{r['label']}  —  ridge {100 * r['keep'].mean():.0f}% of night, "
                     f"median {r['f_med']:.3f} Hz", loc='left', fontsize=14)
        lo, hi = np.percentile(r['db'], [5, 99.5])
        a1.pcolormesh(r['t'], r['f'], r['db'], shading='gouraud', cmap='viridis',
                      vmin=lo, vmax=hi, rasterized=True)
        a1.plot(r['t'], np.where(r['keep'], r['fr'], np.nan), color=RIDGE_COLOR, lw=2.4)
        a1.set_ylim(0, FMAX)
        a1.set_xlim(0, 9)
        a1.set_ylabel('Hz', fontsize=12)
        a1.tick_params(labelsize=11, labelbottom=(rw == nrow - 1))
        if rw == nrow - 1:
            a1.set_xlabel('Time (hr)', fontsize=13)
    out = FIG / f'ridge_lowband_allsessions_{ch}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session')
    ap.add_argument('--channel', default='CRE')
    ap.add_argument('--hp', type=float, default=None,
                    help='band-pass corner; 0.05 reproduces the old filter edge')
    args = ap.parse_args()
    rows, eps, results = [], [], []
    for meta in SESSION_META:
        if args.session and meta['label'] != args.session:
            continue
        s = load_session(meta)
        prof = load_sleep_profile(s)
        sig = _sig(s, args.channel)
        y, fs2 = low_band_signal(s.cap[args.channel], s.fs, hp=args.hp)
        f, t, db = low_band_map(y, fs2)
        thr = shuffle_gate(f, db)
        fr, prom, keep = track_lowband(f, db, thr=thr)
        out = figure(s.label, args.channel, prof, sig, s.fs, f, t, db, fr, keep)
        ep = episodes(t, fr, prom, keep)
        for e in ep:
            eps.append(dict(session=s.label, channel=args.channel, **e))
        f_med = float(np.median(fr[keep])) if keep.any() else np.nan
        rows.append(dict(session=s.label, channel=args.channel, gate_db=thr,
                         pct_night=100 * keep.mean(), n_episodes=len(ep),
                         f_median_hz=f_med,
                         longest_min=max([e['dur_min'] for e in ep], default=0.0)))
        results.append(dict(label=s.label, prof=prof, t=t, f=f, db=db, fr=fr,
                            keep=keep, f_med=f_med))
        print(f'  {s.label}: ridge {100 * keep.mean():4.0f}% of night, '
              f'{len(ep)} episodes, median {f_med:.3f} Hz, gate {thr:.1f} dB -> {out.name}')
        del s, sig
    if not args.session:
        pd.DataFrame(rows).to_csv(TAB / 'ridge_lowband_smooth.csv', index=False)
        pd.DataFrame(eps).to_csv(TAB / 'ridge_lowband_smooth_episodes.csv', index=False)
        print('  sheet ->', sheet(results, args.channel).name)


if __name__ == '__main__':
    main()
