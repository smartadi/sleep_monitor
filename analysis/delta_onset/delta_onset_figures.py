"""
Delta-onset CAP figures — the paper set.

One question, four views: when an EEG delta burst starts, what do ALL THREE CAP
bands do, in every channel, across the cohort / in one night / per subject / vs age?

Everything here uses a STRICTLY CAUSAL envelope estimator (forward-only Butterworth
+ trailing RMS). A zero-phase envelope has a symmetric impulse response, so a large
post-onset rise leaks BACKWARD and manufactures a pre-onset ramp; the causal
estimator cannot do that, so a flat pre-onset baseline in these figures is a real
absence of a precursor rather than a filtering choice.

Two accuracy fixes relative to `delta_cap_precursor.py` / `lowband_precursor_check.py`:

  1. SELECTION-MATCHED NULL. Real onsets are required to sit in NREM, to have a
     motion-clean pre-window, AND to have a QUIESCENT EEG-delta pre-window. The old
     random-NREM null imposed only the first two. Since CAP band power tracks tonic
     delta (the CAP-SWA state marker), the unmatched null baseline is drawn from
     epochs that are on average more delta-rich than the onsets' pre-windows, which
     pushes the null ABOVE the real curve before t=0 and can be misread as a
     pre-onset dip. The null here passes the identical three gates (only the burst
     itself is absent), so real and null are comparable before t=0 by construction.
     Both nulls are computed; the difference is reported in the summary CSV.

  2. TIMING BIAS IS QUANTIFIED, NOT ASSUMED AWAY. A causal envelope reports an event
     LATE (filter group delay + trailing window), and the detector's own envelope is
     zero-phase with a 2 s centered smoother, so the trigger fires EARLY relative to
     the true burst start. Both biases are measured on synthetic bursts
     (`calibrate_timing`) and the corrected latency is reported alongside the raw one.
     Both corrections shorten the measured latency, so raw causal latency is an UPPER
     bound on the true CAP lag -- and the no-precursor conclusion is unaffected, since
     both biases push the CAP event later, never earlier.

Figures -> writeup/figures/delta_onset/
    fig_delta_onset_cohort.png    trigger + motion control + all 3 bands x 3 channels (n=6)
    fig_delta_onset_session.png   the same view within one night (S2N2, 80 onsets)
    fig_delta_onset_subjects.png  per subject x channel, all bands
    fig_delta_onset_age.png       response amplitude / latency vs subject age (n=6)
CSVs -> writeup/figures/delta_onset/
    delta_onset_response_summary.csv   per channel x band: peak z, latency (raw+corrected),
                                       pre-onset lead vs both nulls, subject direction counts
    delta_onset_timing_calibration.csv estimator delay + trigger bias per band

Usage
-----
    .venv/Scripts/python.exe analysis/delta_onset/delta_onset_figures.py
    .venv/Scripts/python.exe analysis/delta_onset/delta_onset_figures.py --tag q15
    .venv/Scripts/python.exe analysis/delta_onset/delta_onset_figures.py --recompute
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, hilbert
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import FS, EEG_BANDS

# ── Parameters (mirror delta_cap_precursor.py / lowband_precursor_check.py) ─────
ANALYSIS_FS = 20.0
BANDS = {'0-0.5': (0.03, 0.5), '0.5-1': (0.5, 1.0), '1-3': (1.0, 3.0)}
CHANNELS = ['CLE', 'CRE', 'CH']
DELTA_LO, DELTA_HI = EEG_BANDS['delta']          # 0.5-4 Hz
ENV_SMOOTH_S = 1.0                               # CAP envelope smoothing
TRIG_SMOOTH_S = 2.0                              # detector's EEG envelope smoothing
PRE_S, POST_S = 30.0, 15.0
NREM_CODES = (1, 2)
ONSET_GUARD_S = 60.0                             # null centers this far from any onset
MAX_MOTION_FRAC = 0.10                           # same gate the detector applies
MIN_ONSETS = 5                                   # a session needs this many to contribute
WARMUP_S = 120.0                                 # ignore this much of the recording start
RESP_WIN = (0.0, 10.0)                           # window the response amplitude is read over
LEAD_WIN = (-12.0, -2.0)                         # pre-onset window the precursor is read over
EXEMPLAR = 'S2N2'

# Subject demographics — source: analysis/rates/age_features.py (manuscript Table 1)
DEMO = {'S1': dict(age=61, sex='F'), 'S2': dict(age=66, sex='M'),
        'S3': dict(age=37, sex='M'), 'S4': dict(age=54, sex='M'),
        'S5': dict(age=55, sex='F'), 'S6': dict(age=25, sex='M')}

OUT_FIG = Path(__file__).resolve().parents[2] / 'writeup' / 'figures' / 'delta_onset'
OUT_DATA = Path(__file__).resolve().parent / 'outputs'
CACHE = Path(os.environ.get('TEMP', '/tmp')) / 'delta_onset_figs'

BAND_COLORS = {'0-0.5': '#1ABC9C', '0.5-1': '#8E44AD', '1-3': '#E67E22'}
NULL_COLOR = '#999999'
ONSET_COLOR = '#27AE60'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False})


# ── Envelope estimators ────────────────────────────────────────────────────────
def _trail_ma(x, n):
    """Causal trailing moving average: y[i] = mean(x[i-n+1 .. i])."""
    return np.convolve(x, np.ones(n) / n, mode='full')[:len(x)]


def _center_ma(x, n):
    return np.convolve(x, np.ones(n) / n, mode='same')


def env_causal(sig, lo, hi, fs=FS):
    """Strictly causal: forward-only bandpass -> trailing RMS. Cannot leak backward.

    The filter state is initialised to the steady state for the signal's starting DC
    level. Without this, `sosfilt` starts from zero state while the CAP channels sit
    at ~2000 fF, and the resulting step transient rings for minutes at these corner
    frequencies (settling ~1/0.03 Hz for the slow band) -- which both corrupts any
    window early in the recording and inflates the NREM standard deviation the whole
    trace is z-scored by.
    """
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    y, _ = sosfilt(sos, sig, zi=sosfilt_zi(sos) * float(sig[0]))
    return np.sqrt(_trail_ma(y * y, int(fs * ENV_SMOOTH_S)))


def env_zerophase(sig, lo, hi, fs=FS):
    """Zero-phase reference (unbiased latency, but leaks backward in time)."""
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    y = sosfiltfilt(sos, sig)
    return _center_ma(np.abs(hilbert(y)), int(fs * ENV_SMOOTH_S))


def env_trigger(eeg, fs=FS):
    """The detector's own EEG delta envelope (zero-phase, 2 s centered smoother)."""
    sos = butter(4, [DELTA_LO, DELTA_HI], btype='band', fs=fs, output='sos')
    return _center_ma(np.abs(hilbert(sosfiltfilt(sos, eeg))), int(fs * TRIG_SMOOTH_S))


# ── Timing calibration ─────────────────────────────────────────────────────────
def _rise_time(env, t, base_win, plateau_win, frac=0.5):
    """Time at which env first reaches `frac` of the way from baseline to plateau."""
    b = env[base_win].mean()
    p = env[plateau_win].mean()
    thr = b + frac * (p - b)
    i = np.flatnonzero(env >= thr)
    return float(t[i[0]]) if i.size else np.nan


def calibrate_timing(fs=FS):
    """Measure how late/early each estimator reports a burst that truly starts at t=0.

    Synthetic burst: a sinusoid at the band's geometric-center frequency switched on
    at t=0 (amplitude 1, zero before). The true event time is 0 by construction, so
    the 50%-rise time of each envelope IS its bias.
    """
    t = np.arange(-60.0 * fs, 60.0 * fs) / fs
    base = (t >= -40) & (t <= -10)
    plateau = (t >= 20) & (t <= 40)
    rows = []
    for bn, (lo, hi) in BANDS.items():
        fc = float(np.sqrt(lo * hi))
        x = np.sin(2 * np.pi * fc * t) * (t >= 0)
        rows.append({
            'band_hz': bn, 'center_freq_hz': round(fc, 3),
            'causal_delay_s': round(_rise_time(env_causal(x, lo, hi, fs), t, base, plateau), 2),
            'zerophase_delay_s': round(_rise_time(env_zerophase(x, lo, hi, fs), t, base, plateau), 2),
        })
    # trigger bias: the detector walks back to a LOW crossing, so use a 10% rise
    fc = float(np.sqrt(DELTA_LO * DELTA_HI))
    x = np.sin(2 * np.pi * fc * t) * (t >= 0)
    trig = round(_rise_time(env_trigger(x, fs), t, base, plateau, frac=0.10), 2)
    for r in rows:
        r['trigger_bias_s'] = trig                       # negative = trigger fires early
        r['latency_correction_s'] = round(r['causal_delay_s'] - trig, 2)
    return pd.DataFrame(rows)


# ── Per-session peri-onset extraction ──────────────────────────────────────────
def _sub(x, q):
    return x[::q]


def _rolling_std(sig, fs, win_s):
    n = max(1, int(fs * win_s)); k = np.ones(n) / n
    m = np.convolve(sig, k, 'same'); m2 = np.convolve(sig * sig, k, 'same')
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def stage_per_sample(profile, n, fs):
    codes = np.full(n, -1, np.int8)
    if profile is None:
        return codes
    es = int(30.0 * fs)
    for t_hr, c in zip(profile['t_ep_hr'], profile['codes']):
        s = int(round(t_hr * 3600.0 * fs)); e = min(s + es, n)
        if s < n and e > 0:
            codes[max(s, 0):e] = c
    return codes


def zscore_on(x, mask):
    ref = x[mask]; mu, sd = ref.mean(), ref.std()
    return (x - mu) / sd if sd > 0 else x - mu


def robust_z_on(x, mask):
    """Median/MAD version of zscore_on, for comparing AMPLITUDES across subjects.

    These envelopes are very heavy-tailed (a night's peak runs ~3000x its median),
    so the standard deviation is set by a handful of movement artifacts and varies
    several-fold between nights. Dividing by it makes 'z' mean something different on
    a quiet night than on a restless one -- which is why the quietest night (S5, 6
    onsets) reads 5-13 z where every other subject reads 1-3. MAD is insensitive to
    those tails, so cross-subject amplitude claims are made on this scale.
    """
    ref = x[mask]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    return (x - med) / mad if mad > 0 else x - med


def peri(x, centers, pre, post):
    n = len(x)
    rows = [x[c - pre:c + post] for c in centers if c - pre >= 0 and c + post < n]
    return np.stack(rows) if rows else None


def null_centers(nrem, motion, trig_env, low, quiet_n, onsets, pre, post, k, rng,
                 match_quiescence, clean_post=False):
    """Random NREM centers passing the same gates the detector applied to onsets.

    match_quiescence=True adds the EEG-delta quiescence gate (mean pre-window envelope
    below the detector's low threshold) that the real onsets had to pass.
    clean_post=True additionally requires the POST window to be motion-clean, matching
    the motion-clean onset subset.
    """
    n = len(nrem)
    guard = int(ONSET_GUARD_S * ANALYSIS_FS)
    excl = np.zeros(n, bool)
    for o in onsets:
        excl[max(0, o - guard):min(n, o + guard)] = True
    valid = np.flatnonzero(nrem & ~excl)
    warm = int(WARMUP_S * ANALYSIS_FS)
    valid = valid[(valid >= max(pre, quiet_n, warm)) & (valid + post < n)]
    out = []
    for c in rng.permutation(valid):
        if motion[c - pre:c].mean() > MAX_MOTION_FRAC:
            continue
        if clean_post and motion[c:c + post].mean() > MAX_MOTION_FRAC:
            continue
        if match_quiescence and trig_env[c - quiet_n:c].mean() >= low:
            continue
        out.append(c)
        if len(out) >= k:
            break
    return np.array(out, int)


def process_session(idx, tag, rng, recompute=False):
    """Peri-onset matrices for one session, cached to disk (figures then rerun fast)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f'onsetfig_{idx}_{tag}.npz'
    if cf.exists() and not recompute:
        z = np.load(cf, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d['label'] = str(d['label']); d['subject'] = str(d['subject'])
        return d if int(d['n_onsets']) >= MIN_ONSETS else None

    onset_f = OUT_DATA / f'delta_onsets_{SESSION_META[idx]["label"]}_{tag}.npz'
    if not onset_f.exists():
        return None
    oz = np.load(onset_f)
    if oz['onset_samp'].size < MIN_ONSETS:
        return None
    q = int(round(FS / ANALYSIS_FS))
    onsets = np.round(np.asarray(oz['onset_samp']) / q).astype(int)
    low = float(oz['low'])
    quiet_n = int(float(oz['quiet_pre_s']) * ANALYSIS_FS)

    sess = load_session(idx)
    profile = load_sleep_profile(sess)
    codes = stage_per_sample(profile, sess.n_samples, FS)
    nrem = _sub(np.isin(codes, NREM_CODES), q)
    mo = _rolling_std(sess.cap['acc_mag'].astype(np.float64), FS, 2.0)
    motion = _sub(mo > np.percentile(mo, 90.0), q)
    trig = _sub(env_trigger(sess.psg['EEG'].astype(np.float64)), q)
    m = min(len(nrem), len(motion), len(trig))
    nrem, motion, trig = nrem[:m], motion[:m], trig[:m]
    ref = nrem.copy()                    # z-score reference: NREM after the warm-up
    ref[:int(WARMUP_S * ANALYSIS_FS)] = False

    pre, post = int(PRE_S * ANALYSIS_FS), int(POST_S * ANALYSIS_FS)
    k = len(onsets)
    null_m = null_centers(nrem, motion, trig, low, quiet_n, onsets, pre, post, k, rng, True)
    null_u = null_centers(nrem, motion, trig, low, quiet_n, onsets, pre, post, k, rng, False)

    # Motion-clean subset: the detector only screened the PRE window, so the post-onset
    # window -- where the CAP response lives -- is unconstrained. Head movement moves the
    # electrodes and raises every band at once, which is exactly the observed response
    # shape, so the response has to be re-measured on onsets that are motion-clean AFTER
    # t=0 as well. The null for this subset carries the same post-window gate.
    keep = np.array([motion[o:o + post].mean() <= MAX_MOTION_FRAC for o in onsets], bool)
    onsets_c = onsets[keep]
    null_c = null_centers(nrem, motion, trig, low, quiet_n, onsets, pre, post,
                          max(len(onsets_c), 1), rng, True, clean_post=True)

    out = {'label': sess.label, 'subject': sess.label.split('N')[0],
           'n_onsets': k, 'n_null_matched': len(null_m),
           'n_onsets_clean': len(onsets_c), 'n_null_clean': len(null_c),
           'post_motion_frac': float(np.mean([motion[o:o + post].mean() for o in onsets])),
           'tax': np.arange(-pre, post) / ANALYSIS_FS}

    sets = (('real', onsets), ('nullm', null_m), ('nullu', null_u),
            ('realc', onsets_c), ('nullc', null_c))
    trig_z = zscore_on(trig, ref)
    for nm, cen in sets:
        s = peri(trig_z, cen, pre, post)
        out[f'eeg_{nm}'] = s.mean(0) if s is not None else np.full(pre + post, np.nan)
        s = peri(motion.astype(float), cen, pre, post)
        out[f'motion_{nm}'] = s.mean(0) if s is not None else np.full(pre + post, np.nan)

    for ch in CHANNELS:
        sig = sess.cap[ch].astype(np.float64)
        for bn, (lo, hi) in BANDS.items():
            e_ca = _sub(env_causal(sig, lo, hi), q)[:m]
            ca = zscore_on(e_ca, ref)
            rz = robust_z_on(e_ca, ref)
            zp = zscore_on(_sub(env_zerophase(sig, lo, hi), q)[:m], ref)
            for est, env in (('ca', ca), ('rz', rz), ('zp', zp)):
                for nm, cen in sets:
                    if est in ('zp', 'rz') and nm in ('realc', 'nullc'):
                        continue                       # zero-phase / robust: reference only
                    s = peri(env, cen, pre, post)
                    out[f'{est}_{nm}_{ch}_{bn}'] = (s.mean(0) if s is not None
                                                    else np.full(pre + post, np.nan))
    np.savez_compressed(cf, **out)
    return out


def per_subject(sessions, key):
    """Stack per-subject means (sessions averaged within subject) -> (n_subj, T)."""
    subj = sorted({s['subject'] for s in sessions})
    return subj, np.array([np.nanmean([s[key] for s in sessions if s['subject'] == sb], axis=0)
                           for sb in subj])


# ── Small plotting helpers ─────────────────────────────────────────────────────
def motion_backdrop(ax, tax, motion, mmax, label=False):
    """Shade the head-motion profile behind a channel's band curves.

    Motion is drawn inside every channel panel rather than in a panel of its own, so the
    band-power rise cannot be read without also seeing the movement that occupies the same
    window. Placed on a twinned axis underneath the curves.
    """
    ax2 = ax.twinx()
    ax2.fill_between(tax, 0, motion * 100, color='#C0392B', alpha=0.11, lw=0, zorder=0)
    ax2.plot(tax, motion * 100, color='#C0392B', lw=0.7, alpha=0.40, zorder=0)
    ax2.set_ylim(0, mmax * 100 * 1.75)       # keep the backdrop subordinate to the curves
    ax2.spines['top'].set_visible(False)
    if label:
        ax2.set_ylabel('% samples with motion', color='#C0392B', fontsize=8.5)
        ax2.tick_params(axis='y', colors='#C0392B', labelsize=7.5)
    else:
        ax2.set_yticks([])
    ax.set_zorder(ax2.get_zorder() + 1)      # band curves draw on top of the shading
    ax.patch.set_visible(False)
    return ax2


def band_panel(ax, tax, curves, nulls=None, sem=None, mark_peak=True, lw=1.5):
    """One channel panel: the three band curves (+ optional SEM band and null)."""
    for bn in BANDS:
        y = curves[bn]
        ax.plot(tax, y, color=BAND_COLORS[bn], lw=lw, label=f'{bn} Hz', zorder=3)
        if sem is not None:
            ax.fill_between(tax, y - sem[bn], y + sem[bn], color=BAND_COLORS[bn],
                            alpha=0.18, lw=0, zorder=2)
        if nulls is not None:
            ax.plot(tax, nulls[bn], color=NULL_COLOR, ls='--', lw=0.8, zorder=1)
        if mark_peak:
            post = tax >= 0
            j = np.nanargmax(y[post])
            ax.plot(tax[post][j], y[post][j], 'o', ms=3.5, color=BAND_COLORS[bn], zorder=4)
    ax.axvline(0, color=ONSET_COLOR, lw=1.1, zorder=1)
    ax.axhline(0, color='k', lw=0.5, zorder=1)
    ax.set_xlim(tax[0], tax[-1])


def band_legend(fig, extra_null=True, **kw):
    h = [Line2D([], [], color=BAND_COLORS[b], lw=2, label=f'{b} Hz') for b in BANDS]
    if extra_null:
        h.append(Line2D([], [], color=NULL_COLOR, ls='--', lw=1, label='matched null'))
    fig.legend(handles=h, frameon=False, **kw)


# ── Figure 1: cohort ───────────────────────────────────────────────────────────
def fig_cohort(sessions, tag, cal):
    subj, _ = per_subject(sessions, 'eeg_real')
    tax = sessions[0]['tax']
    ns = len(subj)

    fig = plt.figure(figsize=(11.0, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1.25], hspace=0.42, wspace=0.30,
                          left=0.07, right=0.925, top=0.90, bottom=0.09)

    # A — the trigger itself
    ax = fig.add_subplot(gs[0, 0])
    _, R = per_subject(sessions, 'eeg_real'); _, N = per_subject(sessions, 'eeg_nullm')
    m, sem = np.nanmean(R, 0), np.nanstd(R, 0) / np.sqrt(ns)
    ax.plot(tax, m, color='#2C3E50', lw=1.5)
    ax.fill_between(tax, m - sem, m + sem, color='#2C3E50', alpha=0.18, lw=0)
    ax.plot(tax, np.nanmean(N, 0), color=NULL_COLOR, ls='--', lw=0.8)
    ax.axvline(0, color=ONSET_COLOR, lw=1.1); ax.axhline(0, color='k', lw=0.5)
    ax.set_xlim(tax[0], tax[-1]); ax.set_title('EEG delta (trigger)', fontsize=9.5)
    ax.set_ylabel('z'); ax.set_xlabel('s from delta onset')

    # B — the same measurement with movement excluded (CH, the largest channel).
    # The motion profile itself is drawn behind every channel panel below, so this slot
    # carries the consequence rather than repeating the motion trace.
    ax = fig.add_subplot(gs[0, 1])
    clean_s = [s for s in sessions if int(s['n_onsets_clean']) >= MIN_ONSETS]
    for bn in BANDS:
        _, R = per_subject(sessions, f'ca_real_CH_{bn}')
        _, C = per_subject(clean_s, f'ca_realc_CH_{bn}')
        ax.plot(tax, np.nanmean(R, 0), color=BAND_COLORS[bn], lw=0.9, alpha=0.35)
        ax.plot(tax, np.nanmean(C, 0), color=BAND_COLORS[bn], lw=1.5)
    ax.axvline(0, color=ONSET_COLOR, lw=1.1); ax.axhline(0, color='k', lw=0.5)
    ax.set_xlim(tax[0], tax[-1])
    ax.set_title(f'CH — motion-free onsets only ({sum(int(s["n_onsets_clean"]) for s in sessions)}'
                 f'/{sum(int(s["n_onsets"]) for s in sessions)})', fontsize=9.5)
    ax.set_ylabel('CAP band power (z)'); ax.set_xlabel('s from delta onset')
    ax.legend(handles=[Line2D([], [], color='#555', lw=0.9, alpha=0.5, label='all onsets'),
                       Line2D([], [], color='#555', lw=1.5, label='motion-free')],
              frameon=False, fontsize=7, loc='upper left')

    # C — response amplitude. A window MEAN, not a peak: the max of a noisy per-subject
    # curve is biased upward, and the bias is largest for the subjects with fewest
    # onsets, which inflates the null as much as the response.
    ax = fig.add_subplot(gs[0, 2])
    win = (tax >= RESP_WIN[0]) & (tax <= RESP_WIN[1])
    clean = [s for s in sessions if int(s['n_onsets_clean']) >= MIN_ONSETS]
    width = 0.26
    for bi, bn in enumerate(BANDS):
        amp, nul, cln = [], [], []
        for ch in CHANNELS:
            _, R = per_subject(sessions, f'ca_real_{ch}_{bn}')
            _, N = per_subject(sessions, f'ca_nullm_{ch}_{bn}')
            _, C = per_subject(clean, f'ca_realc_{ch}_{bn}')
            amp.append(np.nanmean(R[:, win], axis=1)); nul.append(np.nanmean(N[:, win], axis=1))
            cln.append(np.nanmean(C[:, win], axis=1))
        x = np.arange(len(CHANNELS)) + (bi - 1) * width
        ax.bar(x, [a.mean() for a in amp], width * 0.9, color=BAND_COLORS[bn], alpha=0.85)
        for xi, a in zip(x, amp):
            ax.plot(np.full(len(a), xi), a, 'o', ms=2.6, color='k', alpha=0.55, zorder=3)
        ax.plot(x, [n.mean() for n in nul], '_', ms=9, color=NULL_COLOR, mew=1.6, zorder=4)
        # motion-free onsets: same measurement, movement excluded
        ax.plot(x, [c.mean() for c in cln], 'o', ms=5, mfc='none', mec='#C0392B', mew=1.3,
                zorder=5)
    ax.set_xticks(np.arange(len(CHANNELS))); ax.set_xticklabels(CHANNELS)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'Response, {RESP_WIN[0]:.0f}–{RESP_WIN[1]:.0f} s', fontsize=9.5)
    ax.set_ylabel('mean (z)')
    ax.legend(handles=[Line2D([], [], ls='none', marker='o', mfc='none', mec='#C0392B',
                              mew=1.3, ms=5, label='motion-free onsets'),
                       Line2D([], [], ls='none', marker='_', color=NULL_COLOR, mew=1.6,
                              ms=9, label='null')],
              frameon=False, fontsize=7, loc='upper left')

    # D-F — the response, all bands, per channel (shared y so channel rank is readable),
    # with the motion profile shaded behind each panel so the two are read together
    axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
    _, MO = per_subject(sessions, 'motion_real')
    mo = np.nanmean(MO, 0)
    lim = 0
    for j, ch in enumerate(CHANNELS):
        cur, nul, sem = {}, {}, {}
        for bn in BANDS:
            _, R = per_subject(sessions, f'ca_real_{ch}_{bn}')
            _, N = per_subject(sessions, f'ca_nullm_{ch}_{bn}')
            cur[bn] = np.nanmean(R, 0); sem[bn] = np.nanstd(R, 0) / np.sqrt(ns)
            nul[bn] = np.nanmean(N, 0)
            lim = max(lim, np.nanmax(cur[bn] + sem[bn]), np.nanmax(nul[bn]))
        motion_backdrop(axes[j], tax, mo, np.nanmax(mo), label=(j == len(CHANNELS) - 1))
        band_panel(axes[j], tax, cur, nul, sem)
        axes[j].set_title(ch, fontsize=9.5)
        axes[j].set_xlabel('s from delta onset')
    axes[0].set_ylabel('CAP band power (z)')
    for a in axes:
        a.set_ylim(-0.45, lim * 1.12)
    for a in axes[1:]:
        a.set_yticklabels([])

    fig.suptitle(f'CAP band power at EEG delta-burst onset  (n={ns} subjects, '
                 f'{sum(s["n_onsets"] for s in sessions)} onsets, causal envelope)',
                 fontsize=11, y=0.975)
    band_legend(fig, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.004), fontsize=8.5)
    p = OUT_FIG / 'fig_delta_onset_cohort.png'
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f'  saved {p.name}')


# ── Figure 1b: motion control ──────────────────────────────────────────────────
def fig_motion(sessions, tag):
    """Does the response survive on onsets that are motion-clean AFTER t=0 as well?"""
    tax = sessions[0]['tax']
    subj = sorted({s['subject'] for s in sessions})
    keep = [s for s in sessions if int(s['n_onsets_clean']) >= MIN_ONSETS]
    n_all = sum(int(s['n_onsets']) for s in sessions)
    n_cln = sum(int(s['n_onsets_clean']) for s in sessions)

    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.2), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.79, bottom=0.30, wspace=0.24)

    ax = axes[0]
    for nm, col, lab in (('real', '#C0392B', 'all onsets'), ('realc', '#2C3E50', 'motion-clean')):
        src = sessions if nm == 'real' else keep
        _, M = per_subject(src, f'motion_{nm}')
        ax.plot(tax, np.nanmean(M, 0) * 100, color=col, lw=1.5, label=lab)
    _, M = per_subject(keep, 'motion_nullc')
    ax.plot(tax, np.nanmean(M, 0) * 100, color=NULL_COLOR, ls='--', lw=0.8)
    ax.axvline(0, color=ONSET_COLOR, lw=1.1); ax.set_ylim(bottom=0)
    ax.set_xlim(tax[0], tax[-1]); ax.set_title('Head motion', fontsize=9.5)
    ax.set_ylabel('% samples flagged'); ax.set_xlabel('s from delta onset')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')

    lim = 0
    for j, ch in enumerate(CHANNELS):
        ax = axes[j + 1]
        for bn in BANDS:
            _, R = per_subject(sessions, f'ca_real_{ch}_{bn}')
            _, C = per_subject(keep, f'ca_realc_{ch}_{bn}')
            _, N = per_subject(keep, f'ca_nullc_{ch}_{bn}')
            ax.plot(tax, np.nanmean(R, 0), color=BAND_COLORS[bn], lw=0.9, alpha=0.35)
            ax.plot(tax, np.nanmean(C, 0), color=BAND_COLORS[bn], lw=1.6)
            ax.plot(tax, np.nanmean(N, 0), color=NULL_COLOR, ls='--', lw=0.8)
            lim = max(lim, np.nanmax(np.nanmean(R, 0)), np.nanmax(np.nanmean(C, 0)))
        ax.axvline(0, color=ONSET_COLOR, lw=1.1); ax.axhline(0, color='k', lw=0.5)
        ax.set_xlim(tax[0], tax[-1])
        ax.set_title(ch, fontsize=9.5); ax.set_xlabel('s from delta onset')
    axes[1].set_ylabel('CAP band power (z)')
    for a in axes[1:]:
        a.set_ylim(-0.45, lim * 1.12)
    for a in axes[2:]:
        a.set_yticklabels([])

    fig.suptitle(f'Motion control — faint = all {n_all} onsets, bold = the {n_cln} with a '
                 f'motion-clean post-onset window  '
                 f'(n={len({s["subject"] for s in keep})}/{len(subj)} subjects)',
                 fontsize=10.5, y=0.965)
    band_legend(fig, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=8.5)
    p = OUT_FIG / 'fig_delta_onset_motion_control.png'
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f'  saved {p.name}')
    return keep


# ── Figure 2: one session ──────────────────────────────────────────────────────
def fig_session(sessions, tag):
    s = next((x for x in sessions if x['label'] == EXEMPLAR), sessions[0])
    tax = s['tax']
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.1), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.925, top=0.80, bottom=0.30, wspace=0.30)

    ax = axes[0]
    ax.plot(tax, s['eeg_real'], color='#2C3E50', lw=1.5)
    ax.plot(tax, s['eeg_nullm'], color=NULL_COLOR, ls='--', lw=0.8)
    ax.axvline(0, color=ONSET_COLOR, lw=1.1); ax.axhline(0, color='k', lw=0.5)
    ax.set_xlim(tax[0], tax[-1]); ax.set_title('EEG delta (trigger)', fontsize=9.5)
    ax.set_ylabel('z'); ax.set_xlabel('s from delta onset')

    lim = 0
    mo = s['motion_real']
    for j, ch in enumerate(CHANNELS):
        cur = {bn: s[f'ca_real_{ch}_{bn}'] for bn in BANDS}
        nul = {bn: s[f'ca_nullm_{ch}_{bn}'] for bn in BANDS}
        motion_backdrop(axes[j + 1], tax, mo, np.nanmax(mo), label=(j == len(CHANNELS) - 1))
        band_panel(axes[j + 1], tax, cur, nul)
        lim = max(lim, max(np.nanmax(v) for v in cur.values()),
                  max(np.nanmax(v) for v in nul.values()))
        axes[j + 1].set_title(ch, fontsize=9.5); axes[j + 1].set_xlabel('s from delta onset')
    axes[1].set_ylabel('CAP band power (z)')
    for a in axes[1:]:
        a.set_ylim(-0.45, lim * 1.12)
    for a in axes[2:]:
        a.set_yticklabels([])

    fig.suptitle(f'Single night — {s["label"]}  ({s["n_onsets"]} delta onsets)',
                 fontsize=11, y=0.965)
    band_legend(fig, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=8.5)
    p = OUT_FIG / 'fig_delta_onset_session.png'
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f'  saved {p.name}')


# ── Figure 3: per subject ──────────────────────────────────────────────────────
def fig_subjects(sessions, tag):
    subj = sorted({s['subject'] for s in sessions})
    tax = sessions[0]['tax']
    fig, axes = plt.subplots(len(CHANNELS), len(subj), figsize=(2.05 * len(subj), 5.6),
                             sharex=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.15, wspace=0.34, hspace=0.24)
    # Each panel carries its own y-scale, with the peak printed. A shared scale is
    # unreadable here: S5 contributes 6 onsets on a very quiet night and peaks 5-13 z,
    # roughly 5x every other subject, so one panel would set the axis for all 18.
    for i, ch in enumerate(CHANNELS):
        for j, sb in enumerate(subj):
            ss = [x for x in sessions if x['subject'] == sb]
            cur = {bn: np.nanmean([x[f'ca_real_{ch}_{bn}'] for x in ss], 0) for bn in BANDS}
            ax = axes[i, j]
            # no null here: with 6-179 onsets per subject the per-subject null is noisy
            # enough to dominate the panel it is drawn in. The null lives in the cohort figure.
            band_panel(ax, tax, cur, lw=1.2)
            pk = max(np.nanmax(v) for v in cur.values())
            ax.set_ylim(-0.35 * pk, pk * 1.25)
            ax.text(0.03, 0.95, f'{pk:.1f} z', transform=ax.transAxes, fontsize=7.5,
                    va='top', color='#555')
            if i == 0:
                n = sum(int(x['n_onsets']) for x in ss)
                ax.set_title(f'{sb} · {DEMO[sb]["age"]}y · n={n}', fontsize=8.5)
            if i == len(CHANNELS) - 1:
                ax.set_xlabel('s from onset', fontsize=8)
            ax.set_ylabel(f'{ch}\nband power (z)' if j == 0 else '', fontsize=8.5)
    fig.suptitle('Per subject — CAP band power at EEG delta onset (causal envelope; '
                 'per-panel y-scale, peak printed)', fontsize=11, y=0.965)
    band_legend(fig, extra_null=False, loc='lower center', ncol=3,
                bbox_to_anchor=(0.5, 0.005), fontsize=8.5)
    p = OUT_FIG / 'fig_delta_onset_subjects.png'
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f'  saved {p.name}')


# ── Figure 4: age ──────────────────────────────────────────────────────────────
def fig_age(sessions, tag, cal):
    subj = sorted({s['subject'] for s in sessions})
    ages = np.array([DEMO[s]['age'] for s in subj], float)
    tax = sessions[0]['tax']
    post = tax >= 0
    corr = {r['band_hz']: r['latency_correction_s'] for _, r in cal.iterrows()}

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.6), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.87, bottom=0.16, wspace=0.24, hspace=0.30)
    rows = []
    for j, ch in enumerate(CHANNELS):
        for bn in BANDS:
            _, R = per_subject(sessions, f'ca_real_{ch}_{bn}')
            _, RZ = per_subject(sessions, f'rz_real_{ch}_{bn}')
            amp = np.nanmean(RZ[:, (tax >= RESP_WIN[0]) & (tax <= RESP_WIN[1])], axis=1)
            lat = np.array([half_rise(tax, r) for r in R]) - corr[bn]
            for metric, val, ax in (('resp_robust', amp, axes[0, j]), ('latency_s', lat, axes[1, j])):
                rho, p = spearmanr(ages, val)
                ax.plot(ages, val, 'o', ms=5, color=BAND_COLORS[bn], mec='k', mew=0.4)
                if np.isfinite(rho):
                    z = np.polyfit(ages, val, 1)
                    xs = np.linspace(ages.min() - 3, ages.max() + 3, 20)
                    ax.plot(xs, np.polyval(z, xs), color=BAND_COLORS[bn], lw=0.9, alpha=0.5)
                rows.append({'tag': tag, 'channel': ch, 'band_hz': bn, 'metric': metric,
                             'spearman_rho': round(float(rho), 3), 'p_value': round(float(p), 3),
                             'n_subj': len(subj)})
        axes[0, j].set_title(ch, fontsize=9.5)
        axes[1, j].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel(f'response {RESP_WIN[0]:.0f}–{RESP_WIN[1]:.0f} s\n(robust z, MAD)')
    axes[1, 0].set_ylabel('half-rise latency (s, corrected)')

    # annotate the strongest |rho| per panel, so the figure carries no unearned claims
    df = pd.DataFrame(rows)
    for j, ch in enumerate(CHANNELS):
        for r, metric in enumerate(('resp_robust', 'latency_s')):
            d = df[(df.channel == ch) & (df.metric == metric)]
            best = d.loc[d.spearman_rho.abs().idxmax()]
            axes[r, j].text(0.03, 0.93,
                            f"max |rho| {best.band_hz}: {best.spearman_rho:+.2f} (p={best.p_value:.2f})",
                            transform=axes[r, j].transAxes, fontsize=7.5, va='top', color='#444')
    fig.suptitle(f'Response vs subject age (n={len(subj)}, exploratory — no correction '
                 f'for {len(df)} tests)', fontsize=11, y=0.965)
    band_legend(fig, extra_null=False, loc='lower center', ncol=3,
                bbox_to_anchor=(0.5, 0.005), fontsize=8.5)
    p = OUT_FIG / 'fig_delta_onset_age.png'
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f'  saved {p.name}')
    return df


# ── Summary CSV ────────────────────────────────────────────────────────────────
def half_rise(tax, y):
    """First post-onset time the curve reaches half its post-onset peak.

    Less noisy than argmax as a latency estimate, and it measures the ONSET of the
    response rather than its crest, which is what the direction claim rests on.
    """
    post = tax >= 0
    yp, tp = y[post], tax[post]
    pk = np.nanmax(yp)
    if not np.isfinite(pk) or pk <= 0:
        return np.nan
    i = np.flatnonzero(yp >= 0.5 * pk)
    return float(tp[i[0]]) if i.size else np.nan


def summary_table(sessions, clean, tag, cal):
    tax = sessions[0]['tax']
    post = tax >= 0
    win = (tax >= RESP_WIN[0]) & (tax <= RESP_WIN[1])
    lead = (tax >= LEAD_WIN[0]) & (tax <= LEAD_WIN[1])
    corr = {r['band_hz']: r['latency_correction_s'] for _, r in cal.iterrows()}
    rows = []
    for ch in CHANNELS:
        for bn in BANDS:
            subj, R = per_subject(sessions, f'ca_real_{ch}_{bn}')
            _, Nm = per_subject(sessions, f'ca_nullm_{ch}_{bn}')
            _, Nu = per_subject(sessions, f'ca_nullu_{ch}_{bn}')
            _, Z = per_subject(sessions, f'zp_real_{ch}_{bn}')
            _, Zn = per_subject(sessions, f'zp_nullm_{ch}_{bn}')
            _, C = per_subject(clean, f'ca_realc_{ch}_{bn}')
            _, Cn = per_subject(clean, f'ca_nullc_{ch}_{bn}')
            _, RZ = per_subject(sessions, f'rz_real_{ch}_{bn}')
            _, RZn = per_subject(sessions, f'rz_nullm_{ch}_{bn}')
            amp = np.nanmean(R[:, win], axis=1) - np.nanmean(Nm[:, win], axis=1)
            ampr = np.nanmean(RZ[:, win], axis=1) - np.nanmean(RZn[:, win], axis=1)
            ampc = np.nanmean(C[:, win], axis=1) - np.nanmean(Cn[:, win], axis=1)
            gm = np.nanmean(R, 0)
            lat_pk = float(tax[post][np.nanargmax(gm[post])])
            lat_hr = half_rise(tax, gm)
            rows.append({
                'tag': tag, 'channel': ch, 'band_hz': bn,
                'resp_z': round(float(np.nanmean(np.nanmean(R[:, win], axis=1))), 3),
                'resp_minus_null_z': round(float(np.nanmean(amp)), 3),
                'resp_minus_null_robust': round(float(np.nanmean(ampr)), 3),
                'n_subj_response': int(np.sum(amp > 0)), 'n_subj': len(subj),
                'peak_latency_raw_s': round(lat_pk, 1),
                'peak_latency_corrected_s': round(lat_pk - corr[bn], 1),
                'halfrise_latency_corrected_s': round(lat_hr - corr[bn], 1),
                'clean_resp_minus_null_z': round(float(np.nanmean(ampc)), 3),
                'n_subj_response_clean': int(np.sum(ampc > 0)),
                'n_subj_clean': len({s['subject'] for s in clean}),
                'lead_matched_null_z': round(float(np.nanmean((R - Nm)[:, lead])), 3),
                'lead_unmatched_null_z': round(float(np.nanmean((R - Nu)[:, lead])), 3),
                'lead_zerophase_z': round(float(np.nanmean((Z - Zn)[:, lead])), 3),
                'n_subj_lead_pos': int(np.sum(np.nanmean((R - Nm)[:, lead], axis=1) > 0)),
            })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='q30')
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    cal = calibrate_timing()
    cal.insert(0, 'tag', args.tag)
    cal.to_csv(OUT_FIG / 'delta_onset_timing_calibration.csv', index=False)
    print('\nTiming calibration (bias of each estimator on a burst that truly starts at t=0):')
    print(cal.to_string(index=False))

    rng = np.random.default_rng(0)
    sessions = []
    for i in range(len(SESSION_META)):
        try:
            r = process_session(i, args.tag, rng, args.recompute)
        except Exception as e:
            print(f'  [{i}] failed: {e}'); continue
        if r is None:
            print(f'  [{SESSION_META[i]["label"]}] skipped (<{MIN_ONSETS} onsets)'); continue
        sessions.append(r)
        print(f'  {r["label"]}: {int(r["n_onsets"])} onsets, '
              f'{int(r["n_null_matched"])} matched-null centers')
    if not sessions:
        print('no sessions'); return

    print()
    fig_cohort(sessions, args.tag, cal)
    clean = fig_motion(sessions, args.tag)
    fig_session(sessions, args.tag)
    fig_subjects(sessions, args.tag)
    age_df = fig_age(sessions, args.tag, cal)

    pm = np.array([float(s['post_motion_frac']) for s in sessions])
    print(f'\nPost-onset motion: {pm.mean() * 100:.1f}% of samples flagged on average '
          f'(per-session {pm.min() * 100:.1f}-{pm.max() * 100:.1f}%); '
          f'{sum(int(s["n_onsets_clean"]) for s in sessions)}/'
          f'{sum(int(s["n_onsets"]) for s in sessions)} onsets are motion-clean after t=0.')

    summ = summary_table(sessions, clean, args.tag, cal)
    summ.to_csv(OUT_FIG / 'delta_onset_response_summary.csv', index=False)
    age_df.to_csv(OUT_FIG / 'delta_onset_age_correlations.csv', index=False)
    print('\nResponse summary:')
    print(summ.to_string(index=False))
    print('\nAge correlations (|rho| >= 0.7):')
    print(age_df[age_df.spearman_rho.abs() >= 0.7].to_string(index=False))


if __name__ == '__main__':
    main()
