"""
Delta-onset CAP precursor test — does a CAP band-power event PRECEDE EEG delta onset?

Consumes the delta-onset triggers built by `delta_onset_detection.py` (npz files,
one set per quiescence window q15/q30) and asks the professor's question: in the
seconds BEFORE an EEG delta-burst onset, is there a preceding event in specific
CAP bands (0-0.5, 0.5-1, 1-3 Hz) of CLE/CRE/CH?

Three complementary tests, each per band x channel, each with a null:
  1. Peri-onset average  — mean z-scored CAP band power locked to onset (t=0),
     aggregated per-subject (never pooling raw epochs). A precursor shows as a
     rise ABOVE baseline BEFORE t=0. Null: same average locked to random NREM
     times (matched count, motion-clean, away from real onsets) -> should be flat.
  2. CAP->EEG cross-correlation — continuous NREM, CAP band env (drive) vs EEG
     delta env (target). Positive lag = CAP LEADS EEG delta (the hypothesis).
     Circular-shift null gives a 95% band.
  3. Forecasting AUC — can CAP band power in a pre-onset window [-12,-2] s separate
     "about to have an onset" from random NREM? AUC>0.5 = predictive precursor.
     Reported pooled and per-subject.

Controls baked in: onsets already require a motion-clean pre-window (from the
detector); the 0-0.5 band is high-passed at 0.03 Hz to reject DC/coupling drift
(the known cap-mean-drift confound); CAP is never gated on, so it is the free
independent variable. CAP does not pick up cortical EEG electrically (established
negative result), so any CAP change is mechanical/hemodynamic.

Outputs -> analysis/delta_onset/outputs/
    fig_precursor_grid_<tag>.png    3x3 peri-onset averages + null
    fig_precursor_xcorr_<tag>.png   3x3 CAP->EEG cross-correlations + null band
    fig_precursor_auc_<tag>.png     forecasting AUC bars (pooled + per-subject)
    precursor_summary_<tag>.csv     per band x channel stats

Usage
-----
    py analysis/delta_onset/delta_cap_precursor.py                 # both q15 & q30
    py analysis/delta_onset/delta_cap_precursor.py --tag q30
    py analysis/delta_onset/delta_cap_precursor.py -s 3 --tag q30  # single session
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, decimate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import FS

# ── Parameters ────────────────────────────────────────────────────────────────
ANALYSIS_FS = 20.0                          # common rate for all envelopes (Hz)
BANDS = {                                   # label -> (lo, hi) Hz; 0.03 floor kills DC drift
    '0-0.5': (0.03, 0.5),
    '0.5-1': (0.5, 1.0),
    '1-3':   (1.0, 3.0),
}
CHANNELS = ['CLE', 'CRE', 'CH']
EEG_DELTA = (0.5, 4.0)
ENV_SMOOTH_S = 1.0
PRE_S, POST_S = 30.0, 15.0                  # peri-onset window
LEAD_WIN = (-12.0, -2.0)                    # forecasting pre-onset window (s)
LAG_MAX_S = 30.0                            # xcorr +/- lag
N_SHUFFLE = 150
NREM_CODES = (1, 2)
ONSET_GUARD_S = 60.0                        # random-null samples must be >= this from any onset

OUT = Path(__file__).resolve().parent / 'outputs'
CACHE_DIR = Path(os.environ.get('TEMP', '/tmp')) / 'delta_precursor_cache'

BAND_COLORS = {'0-0.5': '#1ABC9C', '0.5-1': '#8E44AD', '1-3': '#E67E22'}


# ── Signal helpers ──────────────────────────────────────────────────────────────
def _bandpass(sig, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, np.asarray(sig, dtype=np.float64))


def _smooth(sig, fs, win_s):
    n = max(1, int(fs * win_s))
    return np.convolve(sig, np.ones(n) / n, mode='same')


def _env(sig, fs, lo, hi):
    """Amplitude envelope of the band-passed signal."""
    return _smooth(np.abs(hilbert(_bandpass(sig, fs, lo, hi))), fs, ENV_SMOOTH_S)


def _zscore_on(x, mask):
    """z-score x using stats from x[mask] only (NREM), applied to all of x."""
    ref = x[mask]
    mu, sd = ref.mean(), ref.std()
    return (x - mu) / sd if sd > 0 else x - mu


def _subsample(x, q):
    return x[::q]


def stage_code_per_sample(profile, n, fs):
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


def _rolling_std(sig, fs, win_s):
    n = max(1, int(fs * win_s))
    k = np.ones(n) / n
    x = np.asarray(sig, float)
    m = np.convolve(x, k, mode='same')
    m2 = np.convolve(x * x, k, mode='same')
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


# ── Feature cache (envelopes at ANALYSIS_FS) ────────────────────────────────────
def load_features(idx):
    """Per-session at ANALYSIS_FS: EEG delta env, CAP band envs (raw), nrem & motion masks.

    Raw channels are decimated 100->ANALYSIS_FS Hz FIRST, so the Hilbert envelopes
    run on ~5x fewer samples (the bands top out at 3 Hz, well under the 10 Hz Nyquist).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f'prec_{idx}.npz'
    q = int(round(FS / ANALYSIS_FS))         # 100 -> 20 Hz
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (str(z['label']),
                {k: z[k] for k in z.files if k.startswith('cap_')},
                z['eeg_delta'], z['nrem'], z['motion'])

    sess = load_session(idx)
    profile = load_sleep_profile(sess)

    def dec(x):
        return decimate(np.asarray(x, np.float64), q, ftype='fir', zero_phase=True)

    eeg_d = dec(sess.psg['EEG'])
    cap_d = {ch: dec(sess.cap[ch]) for ch in CHANNELS}
    acc_d = dec(sess.cap['acc_mag'])
    m = min(len(eeg_d), len(acc_d), *[len(v) for v in cap_d.values()])

    codes = stage_code_per_sample(profile, m, ANALYSIS_FS)
    nrem = np.isin(codes, NREM_CODES)
    mot = _rolling_std(acc_d[:m], ANALYSIS_FS, 2.0)
    motion = mot > np.percentile(mot, 90.0)

    eeg_delta = _env(eeg_d[:m], ANALYSIS_FS, *EEG_DELTA)
    cap_envs = {}
    for ch in CHANNELS:
        for bname, (lo, hi) in BANDS.items():
            cap_envs[f'cap_{ch}_{bname}'] = _env(cap_d[ch][:m], ANALYSIS_FS, lo, hi).astype(np.float32)

    np.savez_compressed(cache, label=sess.label, eeg_delta=eeg_delta.astype(np.float32),
                        nrem=nrem, motion=motion, **cap_envs)
    return sess.label, cap_envs, eeg_delta, nrem, motion


def load_onsets(label, tag):
    f = OUT / f'delta_onsets_{label}_{tag}.npz'
    if not f.exists():
        return np.array([], dtype=int)
    z = np.load(f)
    samp = z['onset_samp']
    if samp.size == 0:
        return np.array([], dtype=int)
    q = int(round(FS / ANALYSIS_FS))
    return np.round(np.asarray(samp) / q).astype(int)


# ── Core computations ───────────────────────────────────────────────────────────
def peri_stack(env_z, centers, pre, post):
    """Stack windows [-pre,+post] around centers; returns (k, win) array."""
    n = len(env_z)
    rows = [env_z[c - pre:c + post] for c in centers if c - pre >= 0 and c + post < n]
    return np.stack(rows) if rows else None


def random_nrem_centers(nrem, motion, onsets, pre, post, k, rng):
    """k random NREM indices, motion-clean pre-window, >= ONSET_GUARD from any onset."""
    n = len(nrem)
    guard = int(ONSET_GUARD_S * ANALYSIS_FS)
    excl = np.zeros(n, bool)
    for o in onsets:
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
    """Concatenate z-scored contiguous NREM runs >= 2*lag_max."""
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


def nrem_xcorr(cap_z, eeg_z, nrem, lag_max_s, rng):
    """CAP(drive) vs EEG-delta(target) over contiguous NREM. +lag = CAP leads EEG.

    Circular-shift null via phase ramp: rfft(roll(t,sh)) = T*exp(-2πi k sh/nf), so
    each shuffle is one irfft of a precomputed product — no per-shuffle forward FFT.
    """
    lag_max = int(lag_max_s * ANALYSIS_FS)
    d = nrem_concat(cap_z, nrem, lag_max)
    t = nrem_concat(eeg_z, nrem, lag_max)
    if d is None or t is None or len(d) != len(t):
        return None
    n = len(d)
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    D = np.fft.rfft(d, nf); T = np.fft.rfft(t, nf)
    P = np.conj(D) * T                                   # base product
    xc = _lagwin(np.fft.irfft(P, nf) / n, nf, lag_max)
    lags = np.arange(-lag_max, lag_max + 1) / ANALYSIS_FS
    k = np.arange(nf // 2 + 1)
    shifts = rng.integers(lag_max, n - lag_max, size=N_SHUFFLE)
    peak_null = np.empty(N_SHUFFLE)
    for j, sh in enumerate(shifts):
        ramp = np.exp(-2j * np.pi * k * sh / nf)
        cc = np.fft.irfft(P * ramp, nf) / n
        peak_null[j] = np.max(np.abs(_lagwin(cc, nf, lag_max)))
    return lags, xc, np.percentile(peak_null, 95)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ── Per-session processing ──────────────────────────────────────────────────────
def build_base(idx, rng):
    """Tag-independent per-session work: z-scored envelopes + CAP->EEG xcorr (once)."""
    label, cap_envs, eeg_delta, nrem, motion = load_features(idx)
    eeg_z = _zscore_on(eeg_delta.astype(np.float64), nrem)
    envs_z, xcorr = {}, {}
    for ch in CHANNELS:
        for bname in BANDS:
            key = (ch, bname)
            env_z = _zscore_on(cap_envs[f'cap_{ch}_{bname}'].astype(np.float64), nrem)
            envs_z[key] = env_z
            xcorr[key] = nrem_xcorr(env_z, eeg_z, nrem, LAG_MAX_S, rng)
    return {'label': label, 'subject': label.split('N')[0],
            'nrem': nrem, 'motion': motion, 'envs_z': envs_z, 'xcorr': xcorr}


def process_tag(base, tag, rng):
    """Tag-specific: peri-onset curves, random-NREM null, forecasting AUC."""
    onsets = load_onsets(base['label'], tag)
    if len(onsets) < 5:
        return None
    nrem, motion = base['nrem'], base['motion']
    pre, post = int(PRE_S * ANALYSIS_FS), int(POST_S * ANALYSIS_FS)
    rand = random_nrem_centers(nrem, motion, onsets, pre, post, len(onsets), rng)
    lw0 = int((LEAD_WIN[0] + PRE_S) * ANALYSIS_FS)
    lw1 = int((LEAD_WIN[1] + PRE_S) * ANALYSIS_FS)

    res = {'label': base['label'], 'subject': base['subject'], 'n_onsets': len(onsets),
           'curves': {}, 'null_curves': {}, 'xcorr': base['xcorr'], 'auc': {},
           'tax': np.arange(-pre, post) / ANALYSIS_FS}
    for key, env_z in base['envs_z'].items():
        stk = peri_stack(env_z, onsets, pre, post)
        res['curves'][key] = stk.mean(0) if stk is not None else np.full(pre + post, np.nan)
        nstk = peri_stack(env_z, rand, pre, post) if len(rand) else None
        res['null_curves'][key] = nstk.mean(0) if nstk is not None else np.full(pre + post, np.nan)
        pos = stk[:, lw0:lw1].mean(1) if stk is not None else np.array([])
        neg = nstk[:, lw0:lw1].mean(1) if nstk is not None else np.array([])
        res['auc'][key] = auc(pos, neg)
    return res


# ── Aggregation across sessions (per-subject, then across subjects) ─────────────
def aggregate(sessions):
    keys = [(ch, b) for ch in CHANNELS for b in BANDS]
    tax = sessions[0]['tax']
    subj = sorted({s['subject'] for s in sessions})
    agg = {'tax': tax, 'subjects': subj, 'curve': {}, 'null': {}, 'auc_subj': {}, 'auc_pool': {}}
    for key in keys:
        # per-subject mean curve = mean of that subject's session curves
        subj_curves, subj_nulls, subj_aucs = [], [], []
        for sb in subj:
            ss = [s for s in sessions if s['subject'] == sb]
            c = np.nanmean([s['curves'][key] for s in ss], axis=0)
            nl = np.nanmean([s['null_curves'][key] for s in ss], axis=0)
            a = np.nanmean([s['auc'][key] for s in ss])
            subj_curves.append(c); subj_nulls.append(nl); subj_aucs.append(a)
        agg['curve'][key] = np.array(subj_curves)     # (n_subj, T)
        agg['null'][key] = np.array(subj_nulls)
        agg['auc_subj'][key] = np.array(subj_aucs)
        agg['auc_pool'][key] = float(np.nanmean(subj_aucs))
    return agg


# ── Figures ──────────────────────────────────────────────────────────────────────
def fig_grid(agg, tag):
    tax = agg['tax']
    fig, axes = plt.subplots(len(BANDS), len(CHANNELS), figsize=(4.4 * len(CHANNELS), 3.2 * len(BANDS)),
                             sharex=True)
    fig.suptitle(f'Peri-onset CAP band power locked to EEG delta onset ({tag})\n'
                 f'mean +/- SEM across subjects (n={len(agg["subjects"])}); '
                 f'grey = random-NREM null; green line = onset; shaded = pre-onset lead window',
                 fontsize=12)
    for i, bname in enumerate(BANDS):
        for j, ch in enumerate(CHANNELS):
            ax = axes[i, j]
            key = (ch, bname)
            C = agg['curve'][key]; N = agg['null'][key]
            m = np.nanmean(C, 0); sem = np.nanstd(C, 0) / np.sqrt(max(1, C.shape[0]))
            nm = np.nanmean(N, 0)
            ax.plot(tax, m, color=BAND_COLORS[bname], lw=1.6)
            ax.fill_between(tax, m - sem, m + sem, color=BAND_COLORS[bname], alpha=0.25)
            ax.plot(tax, nm, color='#888888', lw=0.9, ls='--')
            ax.axvline(0, color='#27AE60', lw=1.2)
            ax.axvspan(LEAD_WIN[0], LEAD_WIN[1], color='#27AE60', alpha=0.07)
            ax.axhline(0, color='k', lw=0.5)
            if i == 0:
                ax.set_title(ch, fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{bname} Hz\nband power (z)', fontsize=9)
            if i == len(BANDS) - 1:
                ax.set_xlabel('s from delta onset', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / f'fig_precursor_grid_{tag}.png'
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig_xcorr(sessions, tag):
    """Pool per-session xcorr (mean) with mean null threshold."""
    fig, axes = plt.subplots(len(BANDS), len(CHANNELS), figsize=(4.4 * len(CHANNELS), 3.2 * len(BANDS)),
                             sharex=True)
    fig.suptitle(f'CAP -> EEG delta cross-correlation over NREM ({tag})\n'
                 f'+lag = CAP LEADS EEG delta (the hypothesis); dashed = mean 95% shuffle null',
                 fontsize=12)
    for i, bname in enumerate(BANDS):
        for j, ch in enumerate(CHANNELS):
            ax = axes[i, j]; key = (ch, bname)
            xcs, lags, thrs = [], None, []
            for s in sessions:
                xc = s['xcorr'].get(key)
                if xc is None:
                    continue
                lags, curve, thr = xc
                xcs.append(curve); thrs.append(thr)
            if xcs:
                M = np.nanmean(xcs, 0)
                ax.plot(lags, M, color=BAND_COLORS[bname], lw=1.4)
                thr = np.nanmean(thrs)
                ax.axhline(thr, ls='--', color='gray', lw=0.7)
                ax.axhline(-thr, ls='--', color='gray', lw=0.7)
                pk = lags[np.argmax(np.abs(M))]
                ax.axvline(pk, color='#C0392B', lw=0.9, label=f'peak {pk:+.1f}s')
                ax.legend(fontsize=7, loc='upper right')
            ax.axvline(0, color='k', lw=0.5)
            if i == 0:
                ax.set_title(ch, fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{bname} Hz\ncorr', fontsize=9)
            if i == len(BANDS) - 1:
                ax.set_xlabel('lag (s)  [+ = CAP leads]', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / f'fig_precursor_xcorr_{tag}.png'
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig_auc(agg, tag):
    keys = [(ch, b) for ch in CHANNELS for b in BANDS]
    labels = [f'{ch}\n{b}' for ch in CHANNELS for b in BANDS]
    pool = [agg['auc_pool'][k] for k in keys]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(keys))
    colors = [BAND_COLORS[b] for ch in CHANNELS for b in BANDS]
    ax.bar(x, pool, color=colors, alpha=0.8)
    for xi, k in enumerate(keys):
        pts = agg['auc_subj'][k]
        ax.scatter(np.full(len(pts), xi), pts, color='k', s=14, zorder=3, alpha=0.7)
    ax.axhline(0.5, color='k', lw=0.8, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('forecasting AUC\n(pre-onset CAP vs random NREM)')
    ax.set_ylim(0.3, 0.9)
    ax.set_title(f'Can pre-onset CAP band power predict an imminent delta onset? ({tag})\n'
                 f'bar = across-subject mean; dots = per-subject; 0.5 = chance', fontsize=11)
    fig.tight_layout()
    out = OUT / f'fig_precursor_auc_{tag}.png'
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


# ── Runner ───────────────────────────────────────────────────────────────────────
def run_tag(tag, bases, rng):
    sessions = []
    for b in bases:
        try:
            r = process_tag(b, tag, rng)
            if r is not None:
                sessions.append(r)
                print(f'  {r["label"]}: n={r["n_onsets"]}')
        except Exception as e:
            print(f'  [{b["label"]}] failed: {e}')
    if not sessions:
        print(f'[{tag}] no usable sessions.')
        return
    agg = aggregate(sessions)

    g = fig_grid(agg, tag)
    x = fig_xcorr(sessions, tag)
    a = fig_auc(agg, tag)

    # summary table
    rows = []
    tax = agg['tax']
    pre_mask = (tax >= LEAD_WIN[0]) & (tax <= LEAD_WIN[1])
    for ch in CHANNELS:
        for bname in BANDS:
            key = (ch, bname)
            m = np.nanmean(agg['curve'][key], 0)
            lead_amp = float(np.nanmean(m[pre_mask]))          # mean z in lead window
            peak_pre_t = float(tax[:len(tax)//2][np.nanargmax(m[:len(tax)//2])])
            xc_peaks = [s['xcorr'][key][1] for s in sessions if s['xcorr'].get(key) is not None]
            xc_lags = sessions[0]['xcorr'][key][0] if sessions[0]['xcorr'].get(key) is not None else None
            if xc_peaks and xc_lags is not None:
                M = np.nanmean(xc_peaks, 0)
                xc_peak_lag = float(xc_lags[np.argmax(np.abs(M))])
                xc_peak_r = float(M[np.argmax(np.abs(M))])
            else:
                xc_peak_lag = xc_peak_r = np.nan
            rows.append({
                'tag': tag, 'channel': ch, 'band_hz': bname,
                'lead_amp_z': round(lead_amp, 3),
                'peak_pre_t_s': round(peak_pre_t, 1),
                'xcorr_peak_lag_s': round(xc_peak_lag, 1),
                'xcorr_peak_r': round(xc_peak_r, 3),
                'auc_pooled': round(agg['auc_pool'][key], 3),
                'auc_subj_min': round(float(np.nanmin(agg['auc_subj'][key])), 3),
                'auc_subj_max': round(float(np.nanmax(agg['auc_subj'][key])), 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'precursor_summary_{tag}.csv', index=False)
    print(f'\n[{tag}] {len(sessions)} sessions, subjects={agg["subjects"]}')
    print(df.to_string(index=False))
    print(f'  figs: {g.name}, {x.name}, {a.name}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--session', type=int, default=None)
    ap.add_argument('--tag', default=None, help='q15 or q30 (default: both)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    indices = [args.session] if args.session is not None else list(range(len(SESSION_META)))
    tags = [args.tag] if args.tag else ['q15', 'q30']

    print('Building per-session features + CAP->EEG xcorr (once)...')
    bases = []
    for i in indices:
        try:
            bases.append(build_base(i, rng))
            print(f'  base {bases[-1]["label"]} ready')
        except Exception as e:
            print(f'  [{i}] base failed: {e}')

    for tag in tags:
        print(f'\n=== precursor test, window {tag} ===')
        run_tag(tag, bases, rng)


if __name__ == '__main__':
    main()
