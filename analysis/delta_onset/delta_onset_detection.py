"""
Delta-onset event detection — the "trial split" for the professor's hypothesis.

Hypothesis under test (built elsewhere): at the ONSET of a delta burst in the
contact EEG, a preceding event appears in specific CAP frequency bands
(0-0.5, 0.5-1, 1-3 Hz) of CLE/CRE/CH. This module ONLY builds and validates the
trigger — the delta-onset event times — so we can eyeball them before committing
to the peri-event / lead-lag analysis (delta_cap_precursor.py, to follow).

Everything downstream depends on what counts as an "onset". A loose definition
lets slow drifts smear across t=0 and fake a precursor, so we detect bursts with
a Schmitt trigger (two thresholds), require the burst to be sustained, walk back
to the true rising edge for the onset time, enforce a refractory gap so trials
are ~independent, and keep only onsets that sit in NREM with a motion-clean
pre-window.

Detector (per session)
-----------------------
1. EEG -> bandpass 0.5-4 Hz (delta) -> Hilbert envelope -> smooth (~2 s).
2. Per-session robust baseline over NREM samples: median + MAD of the envelope.
   high = med + K_HIGH*MAD, low = med + K_LOW*MAD.
3. Burst = envelope crosses `high` and stays above it for >= MIN_BURST_S.
4. Onset = last upward crossing of `low` before that high crossing (the rising
   edge of the burst, not its peak).
5. Keep onset iff: it lands in NREM (N2/N3), the pre-window [-PRE_S, 0] is inside
   the recording and has motion fraction < MAX_MOTION_FRAC, and it is >= MIN_IEI_S
   after the previous accepted onset (refractory).

Outputs -> analysis/delta_onset/outputs/
    delta_onsets_<label>.npz      per-session onset times + metadata
    delta_onsets_summary.csv      one row per session (counts, stage mix, IEI)
    fig_onsets_overview_<label>.png   whole-night hypnogram + envelope + onsets
    fig_onset_gallery_<label>.png     zoomed single-event panels (eyeball check)

Usage
-----
    py analysis/delta_onset/delta_onset_detection.py                 # all sessions
    py analysis/delta_onset/delta_onset_detection.py --session 3     # just S2N2
    py analysis/delta_onset/delta_onset_detection.py -s 3 --show 12  # 12 gallery panels
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import FS, EEG_BANDS, STAGE_LABELS, STAGE_COLORS

# ── Detector parameters ─────────────────────────────────────────────────────────
DELTA_LO, DELTA_HI = EEG_BANDS['delta']     # 0.5-4.0 Hz
ENV_SMOOTH_S = 2.0                          # envelope smoothing (s)
K_HIGH = 2.0                                # burst threshold  = med + K_HIGH*MAD
K_LOW = 0.5                                 # onset threshold  = med + K_LOW*MAD
MIN_BURST_S = 4.0                           # burst must stay above `high` this long
MIN_IEI_S = 25.0                            # refractory: min gap between onsets
PRE_S = 30.0                                # pre-onset window checked for motion
POST_S = 15.0                               # post-onset window (for gallery only)
MAX_MOTION_FRAC = 0.10                      # max fraction of motion samples in pre-window
REQUIRE_QUIET_PRE = True                    # EEG-quiescence gate: pre-window delta must be quiet
                                            # (mean pre-window env < `low` -> true quiet->delta onset)
NREM_CODES = (1, 2)                         # 1=N3, 2=N2

# Motion: rolling std of acc_mag over MOTION_WIN_S, flagged above MOTION_PCTL.
MOTION_WIN_S = 2.0
MOTION_PCTL = 90.0

OUT = Path(__file__).resolve().parent / 'outputs'

# Fast-iteration cache (delta envelope + motion + profile are expensive to recompute)
CACHE_DIR = Path(os.environ.get('TEMP', '/tmp')) / 'delta_onset_cache'


# ── Signal helpers ──────────────────────────────────────────────────────────────
def _bandpass(sig, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, np.asarray(sig, dtype=np.float64))


def _smooth(sig, fs, win_s):
    n = max(1, int(fs * win_s))
    k = np.ones(n) / n
    return np.convolve(sig, k, mode='same')


def _rolling_std(sig, fs, win_s):
    """Rolling std via convolution (E[x^2]-E[x]^2), same length."""
    n = max(1, int(fs * win_s))
    k = np.ones(n) / n
    x = np.asarray(sig, dtype=np.float64)
    m = np.convolve(x, k, mode='same')
    m2 = np.convolve(x * x, k, mode='same')
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


# ── Per-sample stage / NREM / motion masks ──────────────────────────────────────
def stage_code_per_sample(profile, n_samples, fs):
    """Broadcast 30-s epoch stage codes onto every sample (-1 where unscored)."""
    codes = np.full(n_samples, -1, dtype=np.int8)
    if profile is None:
        return codes
    epoch_samp = int(30.0 * fs)
    for t_hr, c in zip(profile['t_ep_hr'], profile['codes']):
        s = int(round(t_hr * 3600.0 * fs))
        e = min(s + epoch_samp, n_samples)
        if s < n_samples and e > 0:
            codes[max(s, 0):e] = c
    return codes


def motion_mask(acc_mag, fs):
    """Boolean per-sample motion flag from rolling std of accel magnitude."""
    rs = _rolling_std(acc_mag, fs, MOTION_WIN_S)
    thr = np.percentile(rs, MOTION_PCTL)
    return rs > thr


# ── Cache ────────────────────────────────────────────────────────────────────────
def load_features(idx):
    """EEG delta envelope, per-sample stage codes, motion mask, EEG raw (all @ FS)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f'feat_{idx}.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (str(z['label']), z['env'], z['codes'], z['motion'], z['eeg'])

    sess = load_session(idx)
    profile = load_sleep_profile(sess)
    eeg = sess.psg['EEG'].astype(np.float64)
    env = _smooth(np.abs(hilbert(_bandpass(eeg, FS, DELTA_LO, DELTA_HI))), FS, ENV_SMOOTH_S)
    codes = stage_code_per_sample(profile, len(eeg), FS)
    motion = motion_mask(sess.cap['acc_mag'].astype(np.float64), FS)
    label = sess.label

    np.savez_compressed(
        cache, label=label,
        env=env.astype(np.float32), codes=codes,
        motion=motion, eeg=eeg.astype(np.float32),
    )
    return label, env, codes, motion, eeg


# ── Onset detection ──────────────────────────────────────────────────────────────
def detect_onsets(env, codes, motion, fs):
    """
    Returns a DataFrame of accepted delta-burst onsets with metadata, plus the
    (low, high) thresholds used (for plotting).
    """
    is_nrem = np.isin(codes, NREM_CODES)
    nrem_env = env[is_nrem]
    if nrem_env.size < int(60 * fs):          # need a minute of NREM to set a baseline
        return pd.DataFrame(), (np.nan, np.nan)

    med = np.median(nrem_env)
    mad = np.median(np.abs(nrem_env - med)) * 1.4826 + 1e-12
    high = med + K_HIGH * mad
    low = med + K_LOW * mad

    above_high = env > high
    above_low = env > low
    # rising edges into the high state
    high_starts = np.flatnonzero(above_high & ~np.roll(above_high, 1))
    high_starts = high_starts[high_starts > 0]

    min_burst = int(MIN_BURST_S * fs)
    pre = int(PRE_S * fs)
    n = len(env)

    rows = []
    last_onset = -np.inf
    for hs in high_starts:
        # sustained burst?
        e = min(hs + min_burst, n)
        if (e - hs) < min_burst or not np.all(above_high[hs:e]):
            continue
        # walk back to the rising edge of `low` -> onset
        j = hs
        while j > 0 and above_low[j - 1]:
            j -= 1
        onset = j
        # refractory
        if (onset - last_onset) / fs < MIN_IEI_S:
            continue
        # onset must be in NREM
        if not is_nrem[onset]:
            continue
        # pre-window inside recording and motion-clean
        if onset - pre < 0 or onset + int(POST_S * fs) >= n:
            continue
        mfrac = motion[onset - pre:onset].mean()
        if mfrac > MAX_MOTION_FRAC:
            continue
        # EEG-quiescence gate: pre-window delta must be quiet (near baseline)
        pre_env_mean = float(env[onset - pre:onset].mean())
        if REQUIRE_QUIET_PRE and pre_env_mean >= low:
            continue

        last_onset = onset
        rows.append({
            'onset_samp': int(onset),
            'onset_hr': onset / fs / 3600.0,
            'stage_code': int(codes[onset]),
            'stage': STAGE_LABELS.get(int(codes[onset]), '?'),
            'peak_env': float(env[hs:e].max()),
            'pre_motion_frac': float(mfrac),
            'pre_env_mean': pre_env_mean,
        })

    return pd.DataFrame(rows), (low, high)


# ── Figures ──────────────────────────────────────────────────────────────────────
def fig_overview(label, env, codes, motion, onsets, thr, fs):
    low, high = thr
    n = len(env)
    t_hr = np.arange(n) / fs / 3600.0
    ds = max(1, int(fs))                       # decimate to ~1 Hz for plotting
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'{label}: delta-onset detection overview  '
                 f'(n={len(onsets)} onsets)', fontsize=13)

    # hypnogram
    ax[0].step(t_hr[::ds], codes[::ds], where='post', color='#2C3E50', lw=0.6)
    ax[0].set_yticks(list(STAGE_LABELS.keys()))
    ax[0].set_yticklabels([STAGE_LABELS[k] for k in STAGE_LABELS])
    ax[0].set_ylabel('stage')
    ax[0].set_title('hypnogram', fontsize=9)

    # delta envelope + thresholds + onsets
    ax[1].plot(t_hr[::ds], env[::ds], color='#C0392B', lw=0.5)
    ax[1].axhline(high, ls='--', color='k', lw=0.7, label=f'high ({K_HIGH} MAD)')
    ax[1].axhline(low, ls=':', color='gray', lw=0.7, label=f'low ({K_LOW} MAD)')
    for _, r in onsets.iterrows():
        ax[1].axvline(r['onset_hr'], color='#27AE60', lw=0.4, alpha=0.6)
    ax[1].set_ylabel('delta env')
    ax[1].set_ylim(0, high * 4 if np.isfinite(high) else None)
    ax[1].legend(fontsize=8, loc='upper right')
    ax[1].set_title('delta envelope (green = accepted onset)', fontsize=9)

    # motion
    ax[2].plot(t_hr[::ds], motion[::ds].astype(float), color='#E67E22', lw=0.4)
    ax[2].set_ylabel('motion')
    ax[2].set_xlabel('time (hr)')
    ax[2].set_title('motion flag', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / f'fig_onsets_overview_{label}.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_gallery(label, env, eeg, onsets, fs, n_show, rng):
    if len(onsets) == 0:
        return None
    pick = onsets.sort_values('peak_env', ascending=False)
    idx = pick['onset_samp'].to_numpy()[:n_show]      # strongest bursts
    pre, post = int(PRE_S * fs), int(POST_S * fs)
    ncol = 3
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.6 * nrow), squeeze=False)
    fig.suptitle(f'{label}: strongest delta onsets — raw EEG (grey) + delta env (red)\n'
                 f'onset at t=0 (green); shaded = pre-window checked in precursor test',
                 fontsize=12)
    tax = np.arange(-pre, post) / fs
    for a, c in enumerate(idx):
        ax = axes[a // ncol][a % ncol]
        seg = slice(c - pre, c + post)
        ax.plot(tax, _zscore(eeg[seg]), color='#95A5A6', lw=0.4)
        ax.plot(tax, _zscore(env[seg]), color='#C0392B', lw=1.2)
        ax.axvline(0, color='#27AE60', lw=1.2)
        ax.axvspan(-PRE_S, 0, color='#27AE60', alpha=0.06)
        ax.set_title(f't={c/fs/3600:.2f} hr', fontsize=8)
        ax.set_xlabel('s from onset', fontsize=8)
    for a in range(len(idx), nrow * ncol):
        axes[a // ncol][a % ncol].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / f'fig_onset_gallery_{label}.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ── Runner ───────────────────────────────────────────────────────────────────────
def run_session(idx, n_show, rng):
    label, env, codes, motion, eeg = load_features(idx)
    onsets, thr = detect_onsets(env, codes, motion, FS)

    dur_hr = len(env) / FS / 3600.0
    nrem_hr = np.isin(codes, NREM_CODES).sum() / FS / 3600.0
    stage_mix = onsets['stage'].value_counts().to_dict() if len(onsets) else {}
    iei = np.diff(np.sort(onsets['onset_hr'].to_numpy())) * 3600.0 if len(onsets) > 1 else np.array([])

    np.savez_compressed(
        OUT / f'delta_onsets_{label}.npz',
        onset_samp=onsets['onset_samp'].to_numpy() if len(onsets) else np.array([]),
        onset_hr=onsets['onset_hr'].to_numpy() if len(onsets) else np.array([]),
        stage_code=onsets['stage_code'].to_numpy() if len(onsets) else np.array([]),
        low=thr[0], high=thr[1], fs=FS,
    )
    ov = fig_overview(label, env, codes, motion, onsets, thr, FS)
    ga = fig_gallery(label, env, eeg, onsets, FS, n_show, rng)

    n = len(onsets)
    n_nrem = int(onsets['stage_code'].isin(NREM_CODES).sum()) if n else 0
    print(f'{label}: {n:4d} onsets  ({n / max(nrem_hr, 1e-6):.1f}/hr NREM)  '
          f'stages={stage_mix}  '
          f'median IEI={np.median(iei):.0f}s' if len(iei) else
          f'{label}: {n:4d} onsets')
    print(f'   overview: {ov.name}' + (f'   gallery: {ga.name}' if ga else ''))

    return {
        'session': label, 'n_onsets': n,
        'dur_hr': round(dur_hr, 2), 'nrem_hr': round(nrem_hr, 2),
        'onsets_per_nrem_hr': round(n / max(nrem_hr, 1e-6), 2),
        'pct_in_nrem': round(100.0 * n_nrem / max(n, 1), 1),
        'n_N3': int(stage_mix.get('N3', 0)), 'n_N2': int(stage_mix.get('N2', 0)),
        'median_iei_s': round(float(np.median(iei)), 1) if len(iei) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--session', type=int, default=None,
                    help='session index 0-11 (default: all)')
    ap.add_argument('--show', type=int, default=9, help='gallery panels per session')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    indices = [args.session] if args.session is not None else list(range(len(SESSION_META)))

    summary = []
    for i in indices:
        try:
            summary.append(run_session(i, args.show, rng))
        except Exception as e:
            print(f'[{i}] FAILED: {e}')

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(OUT / 'delta_onsets_summary.csv', index=False)
        print('\n' + df.to_string(index=False))
        print(f'\nSaved summary -> {OUT / "delta_onsets_summary.csv"}')


if __name__ == '__main__':
    main()
