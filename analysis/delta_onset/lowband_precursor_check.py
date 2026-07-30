"""
Is the 0-0.5 Hz pre-onset RISE a genuine CAP precursor, or a filter artifact?

In `delta_cap_precursor.py` the peri-onset 0-0.5 Hz curves appear to start rising
BEFORE t=0. But that pipeline uses ZERO-PHASE filtering (sosfiltfilt) + a CENTERED
moving-average smoother -- both non-causal. For a slow 0-0.5 Hz band these have a
multi-second symmetric impulse response, so a real post-onset power increase LEAKS
BACKWARD in time and can manufacture an apparent pre-onset rise.

This script re-measures the pre-onset behaviour with a STRICTLY CAUSAL estimator
(forward-only Butterworth `sosfilt` + trailing RMS) alongside the original
zero-phase one, on the same delta onsets, and reports -- WITHOUT hypothesizing --
whether any pre-onset rise above the random-NREM null survives causal filtering,
and with what per-subject consistency (n=6).

Report read-outs (per channel, 0-0.5 Hz, both estimators):
  * pre-window amplitude (real minus null) at [-3,0] s and [-8,-3] s, per subject,
    and the count of subjects with real>null there;
  * first onset-relative time the across-subject mean(real-null) crosses +0.05 z
    and stays up (a "rise-onset" time) -- negative = before delta onset.

Outputs -> analysis/delta_onset/outputs/
    fig_lowband_causal_check_q30.png
    lowband_precursor_check_q30.csv
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, sosfilt, hilbert

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import FS

ANALYSIS_FS = 20.0
BANDS = {'0-0.5': (0.03, 0.5), '0.5-1': (0.5, 1.0), '1-3': (1.0, 3.0)}
CHANNELS = ['CLE', 'CRE', 'CH']
NREM_CODES = (1, 2)
PRE_S, POST_S = 30.0, 15.0
ENV_SMOOTH_S = 1.0
ONSET_GUARD_S = 60.0
EPS = 0.05                                   # z threshold for "rise-onset" crossing
OUT = Path(__file__).resolve().parent / 'outputs'
CACHE = Path(os.environ.get('TEMP', '/tmp')) / 'delta_precursor_cache'   # reuse feature dir
BAND_COLORS = {'0-0.5': '#1ABC9C', '0.5-1': '#8E44AD', '1-3': '#E67E22'}


def _sub(x, q):
    return x[::q]


def _trail_ma(x, n):
    """Causal trailing moving average: y[i] = mean(x[i-n+1 .. i])."""
    k = np.ones(n) / n
    return np.convolve(x, k, mode='full')[:len(x)]


def _center_ma(x, n):
    return np.convolve(x, np.ones(n) / n, mode='same')


def env_zerophase(sig, lo, hi):
    """Original estimator: zero-phase bandpass -> Hilbert -> centered smooth."""
    sos = butter(4, [lo, hi], btype='band', fs=FS, output='sos')
    y = sosfiltfilt(sos, sig)
    e = _center_ma(np.abs(hilbert(y)), int(FS * ENV_SMOOTH_S))
    return e


def env_causal(sig, lo, hi):
    """Strictly causal estimator: forward-only bandpass -> trailing RMS power."""
    sos = butter(4, [lo, hi], btype='band', fs=FS, output='sos')
    y = sosfilt(sos, sig)
    p = _trail_ma(y * y, int(FS * ENV_SMOOTH_S))
    return np.sqrt(p)


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


def _roll_std(sig, fs, w):
    n = max(1, int(fs * w)); k = np.ones(n) / n
    m = np.convolve(sig, k, 'same'); m2 = np.convolve(sig * sig, k, 'same')
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def zscore_on(x, mask):
    ref = x[mask]; mu, sd = ref.mean(), ref.std()
    return (x - mu) / sd if sd > 0 else x - mu


def load_onsets(label, tag):
    f = OUT / f'delta_onsets_{label}_{tag}.npz'
    if not f.exists():
        return np.array([], int)
    z = np.load(f); s = z['onset_samp']
    q = int(round(FS / ANALYSIS_FS))
    return np.round(np.asarray(s) / q).astype(int) if s.size else np.array([], int)


def peri(env_z, centers, pre, post):
    n = len(env_z)
    rows = [env_z[c - pre:c + post] for c in centers if c - pre >= 0 and c + post < n]
    return np.stack(rows) if rows else None


def rand_centers(nrem, motion, onsets, pre, post, k, rng):
    n = len(nrem); guard = int(ONSET_GUARD_S * ANALYSIS_FS)
    excl = np.zeros(n, bool)
    for o in onsets:
        excl[max(0, o - guard):min(n, o + guard)] = True
    valid = np.flatnonzero(nrem & ~excl)
    valid = valid[(valid >= pre) & (valid + post < n)]
    out = []
    for c in rng.permutation(valid):
        if motion[c - pre:c].mean() <= 0.10:
            out.append(c)
        if len(out) >= k:
            break
    return np.array(out, int)


def process(idx, tag, rng):
    q = int(round(FS / ANALYSIS_FS))
    sess = load_session(idx)
    profile = load_sleep_profile(sess)
    n = sess.n_samples
    codes = stage_per_sample(profile, n, FS)
    nrem = _sub(np.isin(codes, NREM_CODES), q)
    mo = _roll_std(sess.cap['acc_mag'].astype(np.float64), FS, 2.0)
    motion = _sub(mo > np.percentile(mo, 90.0), q)
    onsets = load_onsets(sess.label, tag)
    if len(onsets) < 5:
        return None
    pre, post = int(PRE_S * ANALYSIS_FS), int(POST_S * ANALYSIS_FS)
    rand = rand_centers(nrem, motion, onsets, pre, post, len(onsets), rng)

    out = {'label': sess.label, 'subject': sess.label.split('N')[0], 'zp': {}, 'ca': {},
           'zp_null': {}, 'ca_null': {}, 'tax': (np.arange(-pre, post) / ANALYSIS_FS)}
    m = min(len(nrem), len(motion))
    for ch in CHANNELS:
        sig = sess.cap[ch].astype(np.float64)
        for bname, (lo, hi) in BANDS.items():
            zp = zscore_on(_sub(env_zerophase(sig, lo, hi), q)[:m], nrem[:m])
            ca = zscore_on(_sub(env_causal(sig, lo, hi), q)[:m], nrem[:m])
            key = (ch, bname)
            for tagn, env in (('zp', zp), ('ca', ca)):
                s_on = peri(env, onsets, pre, post)
                s_rn = peri(env, rand, pre, post) if len(rand) else None
                out[tagn][key] = s_on.mean(0) if s_on is not None else np.full(pre + post, np.nan)
                out[f'{tagn}_null'][key] = (s_rn.mean(0) if s_rn is not None
                                            else np.full(pre + post, np.nan))
    return out


def per_subject(sessions, estimator, null=False):
    subj = sorted({s['subject'] for s in sessions})
    keys = [(ch, b) for ch in CHANNELS for b in BANDS]
    src = f'{estimator}_null' if null else estimator
    D = {}
    for key in keys:
        rows = []
        for sb in subj:
            ss = [s for s in sessions if s['subject'] == sb]
            rows.append(np.nanmean([s[src][key] for s in ss], axis=0))
        D[key] = np.array(rows)          # (n_subj, T)
    return subj, D


def rise_onset_time(tax, mean_diff):
    """First time mean_diff crosses +EPS and stays >=0 for >=1 s afterward."""
    hold = int(1.0 * ANALYSIS_FS)
    for i in range(len(tax)):
        if mean_diff[i] >= EPS and np.all(mean_diff[i:i + hold] >= 0):
            return float(tax[i])
    return np.nan


def run(indices, tag):
    rng = np.random.default_rng(0)
    sessions = []
    for i in indices:
        try:
            r = process(i, tag, rng)
            if r is not None:
                sessions.append(r); print(f'  {r["label"]} ok')
        except Exception as e:
            print(f'  [{i}] failed: {e}')
    if not sessions:
        print('no sessions'); return
    tax = sessions[0]['tax']
    subj_zp, ZP = per_subject(sessions, 'zp')
    _, ZPn = per_subject(sessions, 'zp', null=True)
    _, CA = per_subject(sessions, 'ca')
    _, CAn = per_subject(sessions, 'ca', null=True)

    near = (tax >= -3) & (tax <= 0)
    mid = (tax >= -8) & (tax < -3)

    rows = []
    for ch in CHANNELS:
        for bname in BANDS:
            key = (ch, bname)
            for est, R, Rn in (('zerophase', ZP, ZPn), ('causal', CA, CAn)):
                diff = R[key] - Rn[key]                       # (n_subj, T)
                near_ps = diff[:, near].mean(1)
                mid_ps = diff[:, mid].mean(1)
                mdiff = np.nanmean(diff, 0)
                rows.append({
                    'tag': tag, 'channel': ch, 'band_hz': bname, 'estimator': est,
                    'near[-3,0]_realMinusNull_z': round(float(np.nanmean(near_ps)), 3),
                    'near_n_subj_pos': int(np.sum(near_ps > 0)),
                    'mid[-8,-3]_realMinusNull_z': round(float(np.nanmean(mid_ps)), 3),
                    'mid_n_subj_pos': int(np.sum(mid_ps > 0)),
                    'rise_onset_t_s': round(rise_onset_time(tax, mdiff), 1),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'lowband_precursor_check_{tag}.csv', index=False)

    # figure: focus 0-0.5 Hz, zp vs causal, per channel, zoomed
    fig, axes = plt.subplots(2, len(CHANNELS), figsize=(4.6 * len(CHANNELS), 7), sharex=True)
    fig.suptitle(f'0-0.5 Hz pre-onset rise: zero-phase (leaks backward) vs strictly causal ({tag})\n'
                 f'solid=onset-locked mean, dashed=random-NREM null, band=SEM across subjects (n={len(subj_zp)}); '
                 f'green=delta onset', fontsize=12)
    zoom = (tax >= -20) & (tax <= 8)
    for row, (est, R, Rn) in enumerate((('zero-phase', ZP, ZPn), ('causal', CA, CAn))):
        for j, ch in enumerate(CHANNELS):
            ax = axes[row, j]; key = (ch, '0-0.5')
            m = np.nanmean(R[key], 0); sem = np.nanstd(R[key], 0) / np.sqrt(R[key].shape[0])
            nm = np.nanmean(Rn[key], 0)
            ax.plot(tax[zoom], m[zoom], color=BAND_COLORS['0-0.5'], lw=1.8)
            ax.fill_between(tax[zoom], (m - sem)[zoom], (m + sem)[zoom],
                            color=BAND_COLORS['0-0.5'], alpha=0.25)
            ax.plot(tax[zoom], nm[zoom], color='#888', ls='--', lw=1.0)
            ax.axvline(0, color='#27AE60', lw=1.2); ax.axhline(0, color='k', lw=0.5)
            ro = rise_onset_time(tax, np.nanmean(R[key] - Rn[key], 0))
            if np.isfinite(ro):
                ax.axvline(ro, color='#C0392B', lw=1.0, ls=':')
                ax.text(ro, ax.get_ylim()[1] * 0.9, f'{ro:+.1f}s', fontsize=8, color='#C0392B')
            if row == 0:
                ax.set_title(ch, fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{est}\n0-0.5 Hz power (z)', fontsize=9)
            ax.set_xlabel('s from delta onset', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fp = OUT / f'fig_lowband_causal_check_{tag}.png'
    fig.savefig(fp, dpi=120); plt.close(fig)

    print(f'\n[{tag}] 0-0.5 Hz precursor check (real minus null):')
    show = df[df.band_hz == '0-0.5'][['channel', 'estimator', 'near[-3,0]_realMinusNull_z',
                                      'near_n_subj_pos', 'mid[-8,-3]_realMinusNull_z',
                                      'mid_n_subj_pos', 'rise_onset_t_s']]
    print(show.to_string(index=False))
    print(f'  saved {fp.name}, lowband_precursor_check_{tag}.csv')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--session', type=int, default=None)
    ap.add_argument('--tag', default='q30')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    idxs = [args.session] if args.session is not None else list(range(len(SESSION_META)))
    run(idxs, args.tag)
