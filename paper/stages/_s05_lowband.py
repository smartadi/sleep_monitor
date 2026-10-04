"""Low band (0-0.3 Hz) for s05_ridges: smoothed spectrogram, Viterbi ridge, shuffle-gated presence.

Ported verbatim from analysis/slow_wave/ridge_lowband_smooth.py (low_band_signal,
low_band_map, _runs_keep, shuffle_gate, track_lowband, episodes). This replaces the
low band of analysis/slow_wave/ridge_overlay_tune.py, whose input was the rate
pipeline's motion-cancelled signal: that canceller's 0.05 Hz band-pass corner,
after per-column 1/f detrending, is a constant hump just above 0.05 Hz in every
column of every night (the "0.05-0.08 Hz" of V11), and its accelerometer
regression writes an instrumental 0.1447 Hz accelerometer tone into the channel.

Method
    0. Signal: the channel itself, decimated to 2 Hz, band-passed 0.005-0.5 Hz
       (2nd-order Butterworth, zero phase). No accelerometer regression.
    1. Each column: Welch over a 10-min window (4-min Hann segments, 75% overlap),
       stepped every 30 s; nfft padded to a 0.001 Hz display grid.
    2. 1/f trend removed per column (straight line of dB on log10 f), then a
       Gaussian blur of 1 min x 0.003 Hz.
    3. Ridge: Viterbi path restricted to 0.02-0.20 Hz, penalty 150 dB/Hz,
       median-smoothed over 5 columns.
    4. Gate: the path is a ridge where it stands >= gate dB above the column's
       in-band median for >= 15 min. The gate is set per night from the night's
       own map, block-shuffled in 10-min blocks (20 shuffles): the lowest level
       at which the shuffled maps show <= 5% of the night as ridge.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import butter, decimate, sosfiltfilt, welch

FS_LOW = 2.0             # decimated rate
WIN_S = 600.0            # one spectrogram column: 10 min
STEP_S = 30.0            # column step, on the epoch grid
SEG_S = 240.0            # Welch segment inside the column
NFFT_DF = 0.001          # display grid
FMIN, FMAX = 0.01, 0.30
HP_HZ = 0.005            # band-pass corner for the low band (see low_band_signal)
BLUR = (2.0, 3.0)        # (time cols = 1 min, freq bins = 0.003 Hz)

RIDGE_BAND = (0.02, 0.20)
PENALTY = 150.0          # dB per Hz: a 0.02 Hz jump costs 3 dB
N_SURR = 20              # block-shuffled nulls per night
SURR_FALSE = 0.05        # gate: surrogates may show at most 5% of the night as ridge
MIN_RUN_MIN = 15.0       # and for at least this long (> one 10-min shuffle block)
SMOOTH_COLS = 5          # median on the path (2.5 min)


def _to_low_rate(x, fs):
    q1, q2 = 10, int(round(fs / 10 / FS_LOW))
    y = decimate(decimate(np.asarray(x, float), q1, ftype='fir', zero_phase=True),
                 q2, ftype='fir', zero_phase=True)
    return y, fs / q1 / q2


def low_band_signal(cap, fs):
    """The low-band channel at 2 Hz: band-passed HP_HZ-0.5 Hz, NOT motion-regressed."""
    y, fs2 = _to_low_rate(cap, fs)
    sos = butter(2, [HP_HZ, 0.5], btype='band', fs=fs2, output='sos')
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
    """True on every run of `on` that is at least `need` columns long."""
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


def track_lowband(f, db, thr):
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


def shuffle_gate(f, db, n=N_SURR, seed=0):
    """Presence gate calibrated on the night's own map, block-shuffled in time.

    Blocks are one analysis window long (10 min, the span over which neighbouring
    columns share data). Every column keeps its real spectrum, motion bursts
    included, but continuity across blocks is destroyed; a ridge is a claim of
    continuity, so this is the null it has to beat. The gate is the lowest
    prominence (dB) at which the shuffled maps show no more than SURR_FALSE of
    the night as ridge under the same run-length rule.
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


def episodes(t, fr, prom, keep):
    """One dict per contiguous gated ridge run."""
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


def night(cap, fs):
    """Everything the low band gives for one night of one channel."""
    y, fs2 = low_band_signal(cap, fs)
    f, t, db = low_band_map(y, fs2)
    thr = shuffle_gate(f, db)
    fr, prom, keep = track_lowband(f, db, thr=thr)
    return dict(f=f, t=t, db=db, gate_db=thr, fr=fr, prom=prom, keep=keep,
                episodes=episodes(t, fr, prom, keep))
