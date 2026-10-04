"""Signal machinery for s08_delta_kcomplex.

The delta-burst onset detector, the three per-recording feature sets the analyses
run on, and the event-locked maths (peri-event stacks, random-NREM nulls,
SEC->EEG cross-correlation, forecasting AUC). Ported from analysis/delta_onset/;
see the stage docstring for the function-by-function provenance.

Feature sets (all cached under _cache/s08_delta_kcomplex/, float32 like the legacy
caches, so the stage reproduces the legacy numbers):

    detector    100 Hz: EEG delta envelope, per-sample stage code, motion flag
    zerophase   20 Hz, decimated FIRST, then zero-phase band-pass + Hilbert +
                centred 1-s smoother (delta_cap_precursor.load_features)
    fullrate    100 Hz band-pass, then subsampled to 20 Hz, for both estimators
                (lowband_precursor_check.process): zero-phase (sosfiltfilt +
                Hilbert + centred 1-s smoother) and strictly causal (sosfilt +
                trailing 1-s RMS); plus a causal EEG delta envelope
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, hilbert, sosfilt, sosfiltfilt

from seclib import CACHE_DIR, FS, STAGE_LABELS, get_session

CACHE = CACHE_DIR / 's08_delta_kcomplex'

# ── Detector (delta_onset_detection.py) ──────────────────────────────────────
DELTA = (0.5, 4.0)               # EEG delta band (Hz)
DET_SMOOTH_S = 2.0               # detector envelope smoothing (s)
K_HIGH = 2.0                     # burst threshold  = med + K_HIGH*MAD
K_LOW = 0.5                      # onset threshold  = med + K_LOW*MAD
MIN_BURST_S = 4.0                # burst must stay above `high` this long
MIN_IEI_S = 25.0                 # refractory gap between accepted onsets
PRE_S = 30.0                     # pre-onset window: motion check + quiet gate
POST_S = 15.0                    # onset must leave this much record after it
MAX_MOTION_FRAC = 0.10           # max motion fraction in the pre-window
QUIET_PRE_S = 30.0               # EEG-quiescence gate window ("q30")
NREM_CODES = (1, 2)              # 1 = N3, 2 = N2
MOTION_WIN_S, MOTION_PCTL = 2.0, 90.0

# ── Event-locked analysis (delta_cap_precursor.py, lowband_precursor_check.py) ─
AFS = 20.0                       # analysis grid (Hz)
Q = int(round(FS / AFS))         # 100 -> 20 Hz
BANDS = {'0-0.5': (0.03, 0.5), '0.5-1': (0.5, 1.0), '1-3': (1.0, 3.0)}
CHANNELS = ['CLE', 'CRE', 'CH']
KEYS = [(ch, b) for ch in CHANNELS for b in BANDS]
ENV_SMOOTH_S = 1.0
ONSET_GUARD_S = 60.0             # null centres stay this far from any event
LEAD_WIN = (-12.0, -2.0)         # forecasting window before onset (s)
LAG_MAX_S = 30.0                 # cross-correlation lag range (s)
MIN_ONSETS = 5                   # recordings with fewer onsets are not analysed


# ═════════════════════════════════════════════════════════════════════════════
# Filters and envelopes
# ═════════════════════════════════════════════════════════════════════════════
def _bandpass(sig, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, np.asarray(sig, dtype=np.float64))


def _smooth(sig, fs, win_s):
    """Centred moving average (non-causal)."""
    n = max(1, int(fs * win_s))
    return np.convolve(sig, np.ones(n) / n, mode='same')


def _trail_ma(x, n):
    """Causal trailing moving average: y[i] = mean(x[i-n+1 .. i])."""
    return np.convolve(x, np.ones(n) / n, mode='full')[:len(x)]


def _rolling_std(sig, fs, win_s):
    """Rolling std via convolution (E[x^2]-E[x]^2), same length."""
    n = max(1, int(fs * win_s))
    k = np.ones(n) / n
    x = np.asarray(sig, dtype=np.float64)
    m = np.convolve(x, k, mode='same')
    m2 = np.convolve(x * x, k, mode='same')
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def env_zerophase(sig, fs, lo, hi):
    """Zero-phase band-pass -> Hilbert amplitude -> centred 1-s smoother."""
    return _smooth(np.abs(hilbert(_bandpass(sig, fs, lo, hi))), fs, ENV_SMOOTH_S)


def env_causal(sig, fs, lo, hi):
    """Strictly causal: forward-only band-pass -> trailing 1-s RMS."""
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    y = sosfilt(sos, np.asarray(sig, dtype=np.float64))
    return np.sqrt(_trail_ma(y * y, int(fs * ENV_SMOOTH_S)))


def stage_code_per_sample(profile, n, fs):
    """Broadcast 30-s epoch stage codes onto every sample (-1 where unscored)."""
    codes = np.full(n, -1, dtype=np.int8)
    if profile is None:
        return codes
    esamp = int(30.0 * fs)
    for t_hr, c in zip(profile['t_ep_hr'], profile['codes']):
        s = int(round(t_hr * 3600.0 * fs))
        e = min(s + esamp, n)
        if s < n and e > 0:
            codes[max(s, 0):e] = c
    return codes


def motion_mask(acc_mag, fs):
    """Per-sample motion flag: rolling std of accel magnitude above its 90th pct."""
    rs = _rolling_std(acc_mag, fs, MOTION_WIN_S)
    return rs > np.percentile(rs, MOTION_PCTL)


def zscore_on(x, mask):
    """z-score x with the mean/SD of x[mask] (NREM), applied to all of x."""
    ref = x[mask]
    mu, sd = ref.mean(), ref.std()
    return (x - mu) / sd if sd > 0 else x - mu


# ═════════════════════════════════════════════════════════════════════════════
# Feature sets (cached)
# ═════════════════════════════════════════════════════════════════════════════
def _cached(name, build):
    path = CACHE / f'{name}.npz'
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    d = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    return d


def detector_features(label):
    """100 Hz EEG delta envelope (float32), stage codes, motion flag."""
    def build():
        s = get_session(label)
        eeg = s.psg['EEG'].astype(np.float64)
        env = _smooth(np.abs(hilbert(_bandpass(eeg, FS, *DELTA))), FS, DET_SMOOTH_S)
        return {'env': env.astype(np.float32),
                'codes': stage_code_per_sample(s.sleep_profile, len(eeg), FS),
                'motion': motion_mask(s.cap['acc_mag'].astype(np.float64), FS)}
    return _cached(f'detector_{label}', build)


def zerophase_features(label):
    """20 Hz, decimate-first zero-phase envelopes (delta_cap_precursor.load_features).

    Keys: eeg_delta, nrem, motion, and cap_<CH>_<band>.
    """
    def build():
        s = get_session(label)

        def dec(x):
            return decimate(np.asarray(x, np.float64), Q, ftype='fir', zero_phase=True)

        eeg_d = dec(s.psg['EEG'])
        cap_d = {ch: dec(s.cap[ch]) for ch in CHANNELS}
        acc_d = dec(s.cap['acc_mag'])
        m = min(len(eeg_d), len(acc_d), *[len(v) for v in cap_d.values()])
        codes = stage_code_per_sample(s.sleep_profile, m, AFS)
        mot = _rolling_std(acc_d[:m], AFS, 2.0)
        out = {'eeg_delta': env_zerophase(eeg_d[:m], AFS, *DELTA).astype(np.float32),
               'nrem': np.isin(codes, NREM_CODES),
               'motion': mot > np.percentile(mot, 90.0)}
        for ch in CHANNELS:
            for b, (lo, hi) in BANDS.items():
                out[f'cap_{ch}_{b}'] = env_zerophase(cap_d[ch][:m], AFS, lo, hi).astype(np.float32)
        return out
    return _cached(f'zerophase_{label}', build)


def fullrate_features(label):
    """100 Hz envelopes subsampled to 20 Hz, zero-phase ('zp_') and causal ('ca_').

    Keys: nrem, motion (the detector's 100 Hz masks, subsampled), ca_eeg_delta,
    and {zp,ca}_<CH>_<band>.
    """
    def build():
        s = get_session(label)
        det = detector_features(label)
        out = {'nrem': np.isin(det['codes'], NREM_CODES)[::Q], 'motion': det['motion'][::Q]}
        out['ca_eeg_delta'] = env_causal(s.psg['EEG'], FS, *DELTA)[::Q].astype(np.float32)
        for ch in CHANNELS:
            sig = s.cap[ch].astype(np.float64)
            for b, (lo, hi) in BANDS.items():
                out[f'zp_{ch}_{b}'] = env_zerophase(sig, FS, lo, hi)[::Q].astype(np.float32)
                out[f'ca_{ch}_{b}'] = env_causal(sig, FS, lo, hi)[::Q].astype(np.float32)
        return out
    return _cached(f'fullrate_{label}', build)


# ═════════════════════════════════════════════════════════════════════════════
# Delta-burst onset detector (delta_onset_detection.detect_onsets)
# ═════════════════════════════════════════════════════════════════════════════
def detect_onsets(env, codes, motion, fs=FS, quiet_pre_s=QUIET_PRE_S):
    """Accepted delta-burst onsets (DataFrame) and the (low, high) thresholds.

    Burst = envelope above `high` for >= MIN_BURST_S; onset = the last upward
    crossing of `low` before it. Kept iff in NREM, >= MIN_IEI_S after the last
    kept onset, with a motion-clean PRE_S pre-window and a quiet (mean < low)
    quiet_pre_s pre-window.
    """
    is_nrem = np.isin(codes, NREM_CODES)
    nrem_env = env[is_nrem]
    if nrem_env.size < int(60 * fs):
        return pd.DataFrame(), (np.nan, np.nan)

    med = np.median(nrem_env)
    mad = np.median(np.abs(nrem_env - med)) * 1.4826 + 1e-12
    high = med + K_HIGH * mad
    low = med + K_LOW * mad

    above_high = env > high
    above_low = env > low
    high_starts = np.flatnonzero(above_high & ~np.roll(above_high, 1))
    high_starts = high_starts[high_starts > 0]

    min_burst = int(MIN_BURST_S * fs)
    pre = int(PRE_S * fs)
    quiet = int(quiet_pre_s * fs)
    n = len(env)

    rows = []
    last_onset = -np.inf
    for hs in high_starts:
        e = min(hs + min_burst, n)
        if (e - hs) < min_burst or not np.all(above_high[hs:e]):
            continue
        j = hs
        while j > 0 and above_low[j - 1]:
            j -= 1
        onset = j
        if (onset - last_onset) / fs < MIN_IEI_S:
            continue
        if not is_nrem[onset]:
            continue
        if onset - pre < 0 or onset + int(POST_S * fs) >= n:
            continue
        mfrac = motion[onset - pre:onset].mean()
        if mfrac > MAX_MOTION_FRAC:
            continue
        pre_env_mean = float(env[onset - quiet:onset].mean()) if quiet > 0 else 0.0
        if quiet > 0 and pre_env_mean >= low:
            continue

        last_onset = onset
        rows.append({'onset_samp': int(onset), 'onset_hr': onset / fs / 3600.0,
                     'stage_code': int(codes[onset]),
                     'stage': STAGE_LABELS.get(int(codes[onset]), '?'),
                     'peak_env': float(env[hs:e].max()),
                     'pre_motion_frac': float(mfrac), 'pre_env_mean': pre_env_mean})
    return pd.DataFrame(rows), (low, high)


def longest_run_s(mask, fs=FS):
    """Longest run of True in mask, in seconds (kcomplex_morphology.longest_run_s)."""
    best, start = 0.0, None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            best = max(best, (i - start) / fs)
            start = None
    if start is not None:
        best = max(best, (len(mask) - start) / fs)
    return best


# ═════════════════════════════════════════════════════════════════════════════
# Event-locked maths
# ═════════════════════════════════════════════════════════════════════════════
def peri_stack(x, centers, pre, post):
    """Stack x[c-pre : c+post] for centres that fit; (k, pre+post) or None."""
    n = len(x)
    rows = [x[c - pre:c + post] for c in centers if c - pre >= 0 and c + post < n]
    return np.stack(rows) if rows else None


def random_nrem_centers(nrem, motion, events, pre, post, k, rng):
    """k random NREM indices with a motion-clean pre-window, >= ONSET_GUARD_S
    from every event."""
    n = len(nrem)
    guard = int(ONSET_GUARD_S * AFS)
    excl = np.zeros(n, bool)
    for o in events:
        excl[max(0, o - guard):min(n, o + guard)] = True
    valid = np.flatnonzero(nrem & ~excl)
    valid = valid[(valid >= pre) & (valid + post < n)]
    if len(valid) == 0:
        return np.array([], int)
    out = []
    for c in rng.permutation(valid):
        if motion[c - pre:c].mean() <= 0.10:
            out.append(c)
        if len(out) >= k:
            break
    return np.array(out, int)


def _lagwin(cc, nf, lag_max):
    return np.concatenate([cc[nf - lag_max:], cc[:lag_max + 1]])


def nrem_concat(sig_z, nrem, lag_max):
    """Concatenate the z-scored contiguous NREM runs >= 2*lag_max long."""
    idx = np.flatnonzero(nrem)
    if len(idx) < 4 * lag_max:
        return None
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    stops = np.concatenate([idx[breaks], [idx[-1]]])
    parts = []
    for s, e in zip(starts, stops):
        if e - s < 2 * lag_max:
            continue
        seg = sig_z[s:e + 1]
        parts.append((seg - seg.mean()) / (seg.std() + 1e-12))
    return np.concatenate(parts) if parts else None


def nrem_xcorr(cap_z, eeg_z, nrem):
    """SEC (drive) vs EEG delta (target) over contiguous NREM; +lag = SEC leads.

    Returns (lags_s, xcorr) or None. (Legacy also built a 150-draw circular-shift
    null here; only its discarded xcorr figure used it, so it is not computed.)
    """
    lag_max = int(LAG_MAX_S * AFS)
    d = nrem_concat(cap_z, nrem, lag_max)
    t = nrem_concat(eeg_z, nrem, lag_max)
    if d is None or t is None or len(d) != len(t):
        return None
    n = len(d)
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    P = np.conj(np.fft.rfft(d, nf)) * np.fft.rfft(t, nf)
    xc = _lagwin(np.fft.irfft(P, nf) / n, nf, lag_max)
    return np.arange(-lag_max, lag_max + 1) / AFS, xc


def auc(pos, neg):
    """Mann-Whitney AUC of pos vs neg (0.5 = chance)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def per_subject(sessions, field):
    """{key: (n_subj, T)} -- each subject's mean over its recordings."""
    subj = sorted({s['subject'] for s in sessions})
    out = {}
    for key in sessions[0][field]:
        out[key] = np.array([np.nanmean([s[field][key] for s in sessions
                                         if s['subject'] == sb], axis=0) for sb in subj])
    return subj, out
