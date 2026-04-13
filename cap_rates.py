"""
cap_rates.py — Heart rate & respiratory rate from capacitor sensor data.

Usage examples
--------------
# 1-minute peak inspection at hour 2.0 (default method: peaks)
python cap_rates.py --subject OS001 --night 2 --start 2.0 --mode inspect

# Choose a different peak detection method (same method applied to cap AND GT)
python cap_rates.py --subject OS001 --night 2 --start 2.0 --mode inspect --method acf
python cap_rates.py --subject OS001 --night 2 --start 2.0 --mode inspect --method zerocross
python cap_rates.py --subject OS001 --night 2 --start 2.0 --mode inspect --method hilbert

# Shorter or longer inspection window
python cap_rates.py --subject OS001 --night 2 --start 2.0 --mode inspect --win-min 0.5
python cap_rates.py --subject OS001 --night 2 --start 2.0 --mode inspect --win-min 2

# Whole-night sliding-window rates
python cap_rates.py --subject OS001 --night 2 --mode rates

# Accuracy metrics vs PSG ground truth (whole night)
python cap_rates.py --subject OS001 --night 2 --mode metrics

# All three
python cap_rates.py --subject OS001 --night 2 --start 2.0 --mode all

# Disable accelerometer motion removal
python cap_rates.py --subject OS001 --night 2 --mode inspect --no-acc-removal

# Adjust sliding window parameters
python cap_rates.py --subject OS001 --night 2 --mode rates --win-sec 30 --step-sec 2

Available methods: peaks, zerocross, acf, spectral, hilbert
  peaks     — find_peaks with prominence threshold (default)
  zerocross — upward zero-crossing positions
  acf       — ACF-estimated period used as min-distance for find_peaks
  spectral  — Welch-estimated period used as min-distance for find_peaks
  hilbert   — Hilbert inst-freq median used as min-distance for find_peaks
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks, hilbert
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# ── File layout ────────────────────────────────────────────────────────────────
BASE_DIR = Path(
    r'C:\Users\adity\Documents\sleep monitor'
    r'\overnight_6subject_pelthupdate_030526'
    r'\overnight_6subject_pelthupdate_030526'
)

CAP_CHANNELS = ['CH', 'CLE', 'CRE', 'aX', 'aY', 'aZ']
PSG_CHANNELS = ['EEG', 'EOGl', 'EOGr', 'ECG', 'Flow', 'Pleth', 'Thorax', 'Abdomen']
ALL_SIG_COLS = CAP_CHANNELS + PSG_CHANNELS
FS = 100.0

def _csv(sid, ini, d, var):
    tag = '_1point_sync' if var == '1point' else ''
    return (BASE_DIR / f'{sid} - {ini}' / d / f'Sync_{d}'
            / f'SleepMask_PSG_100Hz{tag}_combined_{d}.csv.gz')

def _edf(sid, ini, d, var):
    tag = '_1point_sync' if var == '1point' else '_sync'
    return BASE_DIR / f'{sid} - {ini}' / d / f'Sync_{d}' / f'SleepMask{tag}_{d}.edf'

_S = [
    ('OS001','KJK','09-17-2024',''), ('OS001','KJK','09-18-2024',''),
    ('OS002','LDI','09-19-2024',''), ('OS002','LDI','09-20-2024',''),
    ('OS003','LCW','12-18-2025',''), ('OS003','LCW','12-19-2025',''),
    ('OS004','CJH','12-25-2025','1point'), ('OS004','CJH','12-26-2025','1point'),
    ('OS005','CJY','01-03-2026','1point'), ('OS005','CJY','12-27-2025','1point'),
    ('OS006','SK', '01-14-2026',''),  ('OS006','SK', '01-15-2026',''),
]
SESSION_META = [
    {'idx': i, 'subject': sid, 'initials': ini,
     'night': (i % 2) + 1, 'label': f'S{(i//2)+1}N{(i%2)+1}',
     'date': d, 'csv': _csv(sid, ini, d, var), 'edf': _edf(sid, ini, d, var)}
    for i, (sid, ini, d, var) in enumerate(_S)
]

# Band definitions
RESP_LO, RESP_HI = 0.1, 0.5
CARD_LO, CARD_HI = 0.5, 3.0

CAP_CHANS  = ['CH', 'CLE', 'CRE', 'CLE-CRE']
CAP_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD', 'CLE-CRE': '#E67E22'}
GT_COLOR   = '#2C3E50'
METHOD_NAMES = ['spectral', 'acf', 'hilbert', 'zerocross', 'peaks']
METHOD_LABELS = {
    'spectral':  'Spectral peak',
    'acf':       'ACF',
    'hilbert':   'Hilbert inst. freq.',
    'zerocross': 'Zero-crossing',
    'peaks':     'Peak counting',
}
METHOD_COLORS = {
    'spectral': '#3498DB', 'acf': '#E74C3C', 'hilbert': '#27AE60',
    'zerocross': '#9B59B6', 'peaks': '#E67E22',
}

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_session(meta, dtype=np.float32):
    """Load a single session; only the requested session is ever read."""
    print(f"Loading {meta['label']} ({meta['subject']}-{meta['initials']} "
          f"{meta['date']})...", end=' ', flush=True)
    df = pd.read_csv(
        meta['csv'], compression='gzip',
        dtype={c: np.float32 for c in ALL_SIG_COLS + ['timeMS']},
        usecols=['timeMS'] + ALL_SIG_COLS,
    )
    t_ms = df['timeMS'].to_numpy(dtype=dtype)
    t_ms -= t_ms[0]
    t_hr = t_ms / 3_600_000.0
    cap = {ch: df[ch].to_numpy(dtype=dtype) for ch in CAP_CHANNELS}
    cap['acc_mag'] = np.sqrt(cap['aX']**2 + cap['aY']**2 + cap['aZ']**2).astype(dtype)
    psg = {ch: df[ch].to_numpy(dtype=dtype) for ch in PSG_CHANNELS}
    print(f'{t_hr[-1]:.2f} hr  ({len(t_hr):,} samples)')
    return {'meta': meta, 'time_hr': t_hr, 'cap': cap, 'psg': psg, 'fs': FS}


def find_meta(subject, night):
    matches = [m for m in SESSION_META
               if m['subject'] == subject and m['night'] == night]
    if not matches:
        available = sorted({m['subject'] for m in SESSION_META})
        raise ValueError(
            f"No session found for subject={subject!r} night={night}. "
            f"Available subjects: {available}"
        )
    return matches[0]

# ══════════════════════════════════════════════════════════════════════════════
# Signal processing
# ══════════════════════════════════════════════════════════════════════════════

def bandpass(x, f_lo, f_hi, fs=FS, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    return filtfilt(b, a, x.astype(np.float64))


def remove_acc_artifact(cap_sig, acc_mag, f_lo, f_hi):
    """OLS-regress bandpassed acc_mag out of the cap channel in-band."""
    cap_bp = bandpass(cap_sig, f_lo, f_hi)
    acc_bp = bandpass(acc_mag, f_lo, f_hi)
    beta = np.dot(acc_bp, cap_bp) / (np.dot(acc_bp, acc_bp) + 1e-12)
    return cap_bp - beta * acc_bp


def preprocess_window(session, start_hr, win_hr, acc_removal=True):
    """
    Extract a time window and bandpass-filter all cap channels.
    Returns a dict with t_s, idx, sigs, gt_resp, gt_card.
    """
    t = session['time_hr']
    end_hr = start_hr + win_hr
    mask = (t >= start_hr) & (t <= end_hr)
    if not mask.any():
        raise ValueError(f'No samples in [{start_hr:.3f}, {end_hr:.3f}] hr')
    idx = np.where(mask)[0]
    t_s = (t[idx] - t[idx[0]]) * 3600.0

    cap = session['cap']
    psg = session['psg']
    acc = cap['acc_mag'][idx].astype(np.float64)

    raw = {
        'CH':  cap['CH'][idx].astype(np.float64),
        'CLE': cap['CLE'][idx].astype(np.float64),
        'CRE': cap['CRE'][idx].astype(np.float64),
    }
    raw['CLE-CRE'] = raw['CLE'] - raw['CRE']

    sigs = {}
    for ch, sig in raw.items():
        if acc_removal:
            sigs[ch] = {
                'resp': remove_acc_artifact(sig, acc, RESP_LO, RESP_HI),
                'card': remove_acc_artifact(sig, acc, CARD_LO, CARD_HI),
            }
        else:
            sigs[ch] = {
                'resp': bandpass(sig, RESP_LO, RESP_HI),
                'card': bandpass(sig, CARD_LO, CARD_HI),
            }

    return dict(
        t_s      = t_s,
        idx      = idx,
        raw      = raw,
        sigs     = sigs,
        gt_resp  = bandpass(psg['Thorax'][idx], RESP_LO, RESP_HI),
        gt_card  = bandpass(psg['Pleth'][idx],  CARD_LO, CARD_HI),
        gt_thorax_raw = psg['Thorax'][idx].astype(np.float64),
        gt_pleth_raw  = psg['Pleth'][idx].astype(np.float64),
    )


def preprocess_full(session, acc_removal=True):
    """Bandpass-filter the entire session for whole-night analysis."""
    cap = session['cap']
    psg = session['psg']
    acc = cap['acc_mag'].astype(np.float64)
    print('Preprocessing...', end=' ', flush=True)

    full = {}
    raw_ch = {
        'CH':  cap['CH'].astype(np.float64),
        'CLE': cap['CLE'].astype(np.float64),
        'CRE': cap['CRE'].astype(np.float64),
    }
    raw_ch['CLE-CRE'] = raw_ch['CLE'] - raw_ch['CRE']
    for ch, sig in raw_ch.items():
        if acc_removal:
            full[ch] = {
                'resp': remove_acc_artifact(sig, acc, RESP_LO, RESP_HI),
                'card': remove_acc_artifact(sig, acc, CARD_LO, CARD_HI),
            }
        else:
            full[ch] = {
                'resp': bandpass(sig, RESP_LO, RESP_HI),
                'card': bandpass(sig, CARD_LO, CARD_HI),
            }
    gt = {
        'thorax_bp': bandpass(psg['Thorax'].astype(np.float64), RESP_LO, RESP_HI),
        'pleth_bp':  bandpass(psg['Pleth'].astype(np.float64),  CARD_LO, CARD_HI),
    }
    print('done')
    return full, gt

# ══════════════════════════════════════════════════════════════════════════════
# Rate estimators  (each returns rate in Hz, or np.nan on failure)
# ══════════════════════════════════════════════════════════════════════════════

def rate_spectral(x, f_lo, f_hi, fs=FS):
    nperseg = min(len(x), max(64, int(fs * 4)))
    if len(x) < nperseg:
        return np.nan
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(freqs[mask][np.argmax(psd[mask])]) if mask.any() else np.nan


def rate_acf(x, f_lo, f_hi, fs=FS, prominence=0.10):
    x = x.astype(np.float64) - x.mean()
    n = len(x)
    lag_min = max(1, int(np.floor(fs / f_hi)))
    lag_max = min(n - 1, int(np.ceil(fs / f_lo)))
    if lag_min >= lag_max or n < lag_min + 2:
        return np.nan
    nfft = 1
    while nfft < 2 * n - 1:
        nfft <<= 1
    X   = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(X * np.conj(X), n=nfft)[:n].real
    acf /= (acf[0] + 1e-12)
    peaks, props = find_peaks(acf[lag_min:lag_max + 1], prominence=prominence)
    if not len(peaks):
        return np.nan
    k = peaks[np.argmax(props['prominences'])] + lag_min
    if 0 < k < n - 1:
        a0, a1, a2 = acf[k-1], acf[k], acf[k+1]
        d = a0 - 2*a1 + a2
        delta = np.clip(0.5*(a0-a2)/(d+1e-12) if abs(d) > 1e-12 else 0.0, -0.5, 0.5)
        period_s = (k + delta) / fs
    else:
        period_s = k / fs
    return 1.0 / period_s if period_s > 0 else np.nan


def rate_hilbert(x, f_lo, f_hi, fs=FS):
    analytic  = hilbert(x.astype(np.float64))
    phase     = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    amplitude = np.abs(analytic)[:-1]
    amp_thresh = np.percentile(amplitude, 25)
    valid = (inst_freq >= f_lo) & (inst_freq <= f_hi) & (amplitude >= amp_thresh)
    return float(np.median(inst_freq[valid])) if valid.sum() >= 10 else np.nan


def rate_zerocross(x, fs=FS):
    x = x.astype(np.float64)
    signs = np.sign(x)
    signs[signs == 0] = 1
    cross_idx = np.where(np.diff(signs) > 0)[0]
    if len(cross_idx) < 2:
        return np.nan
    t_cross = []
    for i in cross_idx:
        if i + 1 < len(x):
            frac = -x[i] / (x[i+1] - x[i] + 1e-12)
            t_cross.append((i + frac) / fs)
    t_cross = np.array(t_cross)
    if len(t_cross) < 2:
        return np.nan
    return (len(t_cross) - 1) / (t_cross[-1] - t_cross[0])


def rate_peaks(x, f_lo, f_hi, fs=FS, prom_factor=0.4):
    min_dist = max(1, int(0.9 * fs / f_hi))
    # Mild smoothing to bridge small intra-peak valleys (dual peaks)
    smooth_win = max(3, min_dist // 4)
    k = np.ones(smooth_win) / smooth_win
    x_sm = np.convolve(x.astype(np.float64), k, mode='same')
    pks, _   = find_peaks(x_sm, distance=min_dist, prominence=prom_factor * np.std(x_sm))
    if len(pks) < 2:
        return np.nan
    duration = (pks[-1] - pks[0]) / fs
    return (len(pks) - 1) / duration if duration > 0 else np.nan


def estimate_rate(x, f_lo, f_hi, fs=FS):
    return {
        'spectral':  rate_spectral(x,  f_lo, f_hi, fs),
        'acf':       rate_acf(x,       f_lo, f_hi, fs),
        'hilbert':   rate_hilbert(x,   f_lo, f_hi, fs),
        'zerocross': rate_zerocross(x, fs),
        'peaks':     rate_peaks(x,     f_lo, f_hi, fs),
    }


def detect_peaks(sig, f_lo, f_hi, fs=FS, prom_factor=0.4):
    dist = max(1, int(0.9 * fs / f_hi))
    # Mild smoothing to bridge small intra-peak valleys (dual peaks)
    smooth_win = max(3, dist // 4)
    k = np.ones(smooth_win) / smooth_win
    sig_sm = np.convolve(sig.astype(np.float64), k, mode='same')
    pks, _ = find_peaks(sig_sm, distance=dist, prominence=prom_factor * np.std(sig_sm))
    return pks


def zerocross_indices(x):
    """Return sample indices of upward zero crossings (sub-sample interpolated → nearest int)."""
    x = x.astype(np.float64)
    signs = np.sign(x)
    signs[signs == 0] = 1
    cross_raw = np.where(np.diff(signs) > 0)[0]
    result = []
    for i in cross_raw:
        if i + 1 < len(x):
            frac = -x[i] / (x[i+1] - x[i] + 1e-12)
            result.append(int(round(i + frac)))
    return np.array(result, dtype=int)


def peaks_by_method(sig, f_lo, f_hi, method, fs=FS, prom_factor=0.4):
    """
    Return peak (or crossing) indices using the chosen method.

    peaks / zerocross  → directly give sample positions.
    acf / spectral / hilbert → estimate the period first, then run find_peaks
                                with that period as the minimum distance so the
                                positions are real signal maxima consistent with
                                the estimated rate.
    """
    if method == 'peaks':
        return detect_peaks(sig, f_lo, f_hi, fs, prom_factor)

    if method == 'zerocross':
        return zerocross_indices(sig)

    # Rate-based methods: estimate period → constrain find_peaks distance
    rate_fn = {'acf': rate_acf, 'spectral': rate_spectral, 'hilbert': rate_hilbert}[method]
    rate_hz = rate_fn(sig, f_lo, f_hi, fs)
    if np.isnan(rate_hz) or rate_hz <= 0:
        return detect_peaks(sig, f_lo, f_hi, fs, prom_factor)   # fallback
    min_dist = max(1, int(0.85 / rate_hz * fs))
    pks, _ = find_peaks(sig, distance=min_dist, prominence=prom_factor * np.std(sig))
    return pks


def rate_by_method(sig, f_lo, f_hi, method, fs=FS):
    """Return scalar rate in Hz for annotation, using the chosen method."""
    fn = {
        'peaks':     lambda: rate_peaks(sig,    f_lo, f_hi, fs),
        'zerocross': lambda: rate_zerocross(sig, fs),
        'acf':       lambda: rate_acf(sig,      f_lo, f_hi, fs),
        'spectral':  lambda: rate_spectral(sig, f_lo, f_hi, fs),
        'hilbert':   lambda: rate_hilbert(sig,  f_lo, f_hi, fs),
    }[method]
    return fn()

# ══════════════════════════════════════════════════════════════════════════════
# Sliding window
# ══════════════════════════════════════════════════════════════════════════════

def sliding_rates(signal, f_lo, f_hi, fs=FS, win_sec=20.0, step_sec=1.0):
    """Return (centre_times_s, {method: rates_hz_array})."""
    win_n  = int(round(win_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))
    t_list = []
    r_list = {m: [] for m in METHOD_NAMES}
    for start in range(0, len(signal) - win_n + 1, step_n):
        seg = signal[start:start + win_n]
        t_list.append((start + win_n / 2.0) / fs)
        res = estimate_rate(seg, f_lo, f_hi, fs)
        for m in METHOD_NAMES:
            r_list[m].append(res[m])
    return np.array(t_list), {m: np.array(r_list[m]) for m in METHOD_NAMES}

# ══════════════════════════════════════════════════════════════════════════════
# Accuracy helpers
# ══════════════════════════════════════════════════════════════════════════════

def accuracy_metrics(gt_t, gt_hz, cap_t, cap_hz):
    valid_gt  = ~np.isnan(gt_hz)
    valid_cap = ~np.isnan(cap_hz)
    if valid_gt.sum() < 2 or valid_cap.sum() < 2:
        return dict(n=0, mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan)
    f_gt = interp1d(gt_t[valid_gt], gt_hz[valid_gt],
                    kind='linear', bounds_error=False, fill_value=np.nan)
    t_lo = max(gt_t[valid_gt][0],   cap_t[valid_cap][0])
    t_hi = min(gt_t[valid_gt][-1],  cap_t[valid_cap][-1])
    mask = valid_cap & (cap_t >= t_lo) & (cap_t <= t_hi)
    if mask.sum() < 5:
        return dict(n=0, mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan)
    ref  = f_gt(cap_t[mask])
    pred = cap_hz[mask]
    ok   = ~np.isnan(ref) & ~np.isnan(pred)
    ref, pred = ref[ok], pred[ok]
    if len(ref) < 5:
        return dict(n=0, mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan)
    err = pred - ref
    return dict(n=len(ref), mae=np.mean(np.abs(err)), rmse=np.sqrt(np.mean(err**2)),
                r=pearsonr(ref, pred)[0], bias=np.mean(err))

# ══════════════════════════════════════════════════════════════════════════════
# Mode: inspect  — 1-minute peak inspection
# ══════════════════════════════════════════════════════════════════════════════

def mode_inspect(session, start_hr, win_hr=1/60, acc_removal=True, method='peaks'):
    m   = session['meta']
    win = preprocess_window(session, start_hr, win_hr, acc_removal)
    t_s = win['t_s']

    def zsc(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    raw_z = {ch: zsc(win['raw'][ch]) for ch in CAP_CHANS}

    # GT peaks use the same method as cap — consistent comparison
    gt_resp_pks = peaks_by_method(win['gt_resp'], RESP_LO, RESP_HI, method)
    gt_card_pks = peaks_by_method(win['gt_card'], CARD_LO, CARD_HI, method)

    ROWS = CAP_CHANS + ['GT']
    fig, axes = plt.subplots(
        len(ROWS), 2,
        figsize=(15, 2.8 * len(ROWS)),
        sharex=True,
        gridspec_kw={'hspace': 0.45, 'wspace': 0.28}
    )
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})

    BAND_SPECS = [
        ('resp', RESP_LO, RESP_HI, win['gt_resp'], win['gt_thorax_raw'],
         gt_resp_pks, 'GT Thorax', 'br/min'),
        ('card', CARD_LO, CARD_HI, win['gt_card'], win['gt_pleth_raw'],
         gt_card_pks, 'GT Pleth',  'BPM'),
    ]

    for ri, row_id in enumerate(ROWS):
        is_gt = (row_id == 'GT')
        color = GT_COLOR if is_gt else CAP_COLORS[row_id]

        for ci, (band, f_lo, f_hi, gt_bp, gt_raw, gt_pks, gt_lbl, unit) in enumerate(BAND_SPECS):
            ax = axes[ri, ci]
            gt_filt_z = zsc(gt_bp)
            gt_raw_z  = zsc(gt_raw)

            if is_gt:
                ax.plot(t_s, gt_raw_z,  color='#AAAAAA', lw=0.7,  alpha=0.70, label='raw (z)')
                ax.plot(t_s, gt_filt_z, color=GT_COLOR,  lw=1.2,  alpha=0.95, label=gt_lbl)
                ax.plot(t_s[gt_pks], gt_filt_z[gt_pks], '^',
                        color=GT_COLOR, ms=8, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.6,
                        label=f'GT peaks  n={len(gt_pks)}')
                r_gt  = rate_acf(gt_bp, f_lo, f_hi)
                r_str = f'{r_gt*60:.1f} {unit}' if not np.isnan(r_gt) else 'n/a'
                ax.text(0.01, 0.97, f'ACF: {r_str}   peaks: {len(gt_pks)}',
                        transform=ax.transAxes, va='top', fontsize=7.5,
                        bbox=dict(facecolor='white', alpha=0.78, pad=2, edgecolor='none'))
                ax.set_ylabel(f'{gt_lbl}\nNorm. amp.', fontsize=8)

            else:
                cap_filt   = win['sigs'][row_id][band]
                cap_filt_z = zsc(cap_filt)
                cap_pks    = detect_peaks(cap_filt, f_lo, f_hi)

                ax.plot(t_s, raw_z[row_id],  color='#CCCCCC', lw=0.6, alpha=0.85, label='raw (z)')
                ax.plot(t_s, cap_filt_z,     color=color,     lw=1.1, alpha=0.92,
                        label=f'{row_id} filtered')
                ax.plot(t_s[cap_pks], cap_filt_z[cap_pks], 'v',
                        color=color, ms=8, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.6,
                        label=f'cap peaks  n={len(cap_pks)}')
                for pk in gt_pks:
                    ax.axvline(t_s[pk], color=GT_COLOR, lw=0.7, alpha=0.28, zorder=1)

                r_cap = rate_acf(cap_filt, f_lo, f_hi)
                r_gt  = rate_acf(gt_bp,    f_lo, f_hi)
                c_s = f'{r_cap*60:.1f}' if not np.isnan(r_cap) else 'n/a'
                g_s = f'{r_gt*60:.1f}'  if not np.isnan(r_gt)  else 'n/a'
                ax.text(0.01, 0.97,
                        f'cap ACF: {c_s} {unit}   GT ACF: {g_s} {unit}   cap peaks: {len(cap_pks)}',
                        transform=ax.transAxes, va='top', fontsize=7.5,
                        bbox=dict(facecolor='white', alpha=0.78, pad=2, edgecolor='none'))
                ax.set_ylabel(f'{row_id}\nNorm. amp.', fontsize=8)

            ax.axhline(0, color='gray', lw=0.4, alpha=0.35)
            ax.legend(fontsize=6.5, loc='lower right', ncol=3)
            if ri == 0:
                band_lbl = 'Respiratory' if ci == 0 else 'Cardiac'
                ax.set_title(f'{band_lbl} band  ({f_lo}–{f_hi} Hz)',
                             fontsize=9, fontweight='bold')

    for ci in range(2):
        axes[-1, ci].set_xlabel('Time in window (s)', fontsize=8)

    t0m = start_hr * 60
    t1m = (start_hr + win_hr) * 60
    fig.suptitle(
        f"1-min peak inspection  |  {m['label']} {m['subject']}-{m['initials']} {m['date']}  "
        f"| {t0m:.1f}–{t1m:.1f} min  "
        "| grey=raw(z)  colour=filtered  "
        "\u25bd=cap peaks  \u25b3=GT peaks  dashed=GT peak times",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Mode: rates  — whole-night sliding window
# ══════════════════════════════════════════════════════════════════════════════

def mode_rates(session, acc_removal=True, win_sec=20.0, step_sec=1.0):
    m    = session['meta']
    t_hr = session['time_hr']
    full, gt = preprocess_full(session, acc_removal)

    print(f'Running sliding window (win={win_sec}s, step={step_sec}s)...')
    slide, slide_t = {}, {}
    for ch in CAP_CHANS:
        print(f'  {ch}', end=' ', flush=True)
        slide[ch], slide_t[ch] = {}, {}
        for band, f_lo, f_hi in [('resp', RESP_LO, RESP_HI), ('card', CARD_LO, CARD_HI)]:
            t_sl, rates_sl = sliding_rates(full[ch][band], f_lo, f_hi,
                                            win_sec=win_sec, step_sec=step_sec)
            slide[ch][band]   = rates_sl
            slide_t[ch][band] = t_sl
        print('done')

    print('  GT Thorax', end=' ', flush=True)
    gt_t_resp, gt_rates_resp = sliding_rates(gt['thorax_bp'], RESP_LO, RESP_HI,
                                              win_sec=win_sec, step_sec=step_sec)
    print('done')
    print('  GT Pleth', end=' ', flush=True)
    gt_t_card, gt_rates_card = sliding_rates(gt['pleth_bp'], CARD_LO, CARD_HI,
                                              win_sec=win_sec, step_sec=step_sec)
    print('done')

    t0_min = float(t_hr[0]) * 60
    def to_min(t_s): return t0_min + t_s / 60.0

    plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True,
                         'grid.alpha': 0.3, 'font.size': 9})

    for ch in CAP_CHANS:
        fig, axes = plt.subplots(
            len(METHOD_NAMES), 2,
            figsize=(16, 2.5 * len(METHOD_NAMES)),
            sharex='col', sharey='col',
            gridspec_kw={'hspace': 0.4, 'wspace': 0.25}
        )
        t_min = to_min(slide_t[ch]['resp'])

        for row, meth in enumerate(METHOD_NAMES):
            color = METHOD_COLORS[meth]
            for col, (band, gt_t, gt_rts, unit, ylim) in enumerate([
                ('resp', gt_t_resp, gt_rates_resp, 'br/min', (0, 40)),
                ('card', gt_t_card, gt_rates_card, 'BPM',    (20, 180)),
            ]):
                ax = axes[row, col]
                hz = slide[ch][band][meth]
                valid = ~np.isnan(hz)
                ax.plot(to_min(slide_t[ch][band])[valid], hz[valid] * 60,
                        lw=0.9, color=color, alpha=0.85, label=METHOD_LABELS[meth])
                gt_v = ~np.isnan(gt_rts['acf'])
                ax.plot(to_min(gt_t)[gt_v], gt_rts['acf'][gt_v] * 60,
                        '-.', color=GT_COLOR, lw=1.2, alpha=0.7,
                        label='GT ACF')
                ax.set_ylim(*ylim)
                ax.set_ylabel(f'{METHOD_LABELS[meth]}\n{unit}', fontsize=7)
                ax.legend(fontsize=6, loc='upper right')
                if row == 0:
                    lbl = 'Respiratory Rate' if col == 0 else 'Heart Rate'
                    ax.set_title(f'{ch}  —  {lbl}', fontsize=9, fontweight='bold')

        axes[-1, 0].set_xlabel('Time (min)', fontsize=8)
        axes[-1, 1].set_xlabel('Time (min)', fontsize=8)
        fig.suptitle(
            f"Sliding rates — {ch}  |  {m['label']} {m['subject']}-{m['initials']} {m['date']}  "
            f"| win={win_sec:.0f}s step={step_sec:.0f}s",
            fontsize=10)
        plt.tight_layout()
        plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Mode: metrics  — accuracy vs GT
# ══════════════════════════════════════════════════════════════════════════════

def mode_metrics(session, acc_removal=True, win_sec=20.0, step_sec=1.0):
    m    = session['meta']
    full, gt = preprocess_full(session, acc_removal)

    print(f'Running sliding window for metrics (win={win_sec}s, step={step_sec}s)...')
    slide, slide_t = {}, {}
    for ch in CAP_CHANS:
        print(f'  {ch}', end=' ', flush=True)
        slide[ch], slide_t[ch] = {}, {}
        for band, f_lo, f_hi in [('resp', RESP_LO, RESP_HI), ('card', CARD_LO, CARD_HI)]:
            t_sl, rates_sl = sliding_rates(full[ch][band], f_lo, f_hi,
                                            win_sec=win_sec, step_sec=step_sec)
            slide[ch][band]   = rates_sl
            slide_t[ch][band] = t_sl
        print('done')

    print('  GT...', end=' ', flush=True)
    gt_t_resp, gt_rates_resp = sliding_rates(gt['thorax_bp'], RESP_LO, RESP_HI,
                                              win_sec=win_sec, step_sec=step_sec)
    gt_t_card, gt_rates_card = sliding_rates(gt['pleth_bp'],  CARD_LO, CARD_HI,
                                              win_sec=win_sec, step_sec=step_sec)
    print('done')

    rows = []
    for ch in CAP_CHANS:
        for band, gt_t, gt_rts, unit, scale in [
            ('resp', gt_t_resp, gt_rates_resp, 'br/min', 60),
            ('card', gt_t_card, gt_rates_card, 'BPM',    60),
        ]:
            gt_hz = gt_rts['acf']
            for meth in METHOD_NAMES:
                met = accuracy_metrics(gt_t, gt_hz, slide_t[ch][band], slide[ch][band][meth])
                rows.append({
                    'Channel': ch, 'Band': band, 'Method': meth,
                    'n': met['n'],
                    f'MAE ({unit})':  round(met['mae']  * scale, 2) if not np.isnan(met['mae'])  else np.nan,
                    f'RMSE ({unit})': round(met['rmse'] * scale, 2) if not np.isnan(met['rmse']) else np.nan,
                    'Pearson r':      round(met['r'],    3)         if not np.isnan(met['r'])    else np.nan,
                    f'Bias ({unit})': round(met['bias'] * scale, 2) if not np.isnan(met['bias']) else np.nan,
                })

    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"Accuracy vs GT  |  {m['label']} {m['subject']}-{m['initials']} {m['date']}")
    print(f"{'='*60}")
    for band, unit in [('resp', 'br/min'), ('card', 'BPM')]:
        label = 'Respiratory Rate' if band == 'resp' else 'Heart Rate'
        print(f'\n--- {label} ---')
        sub = (df[df['Band'] == band]
               .drop(columns='Band')
               .sort_values(['Channel', f'MAE ({unit})'])
               .reset_index(drop=True))
        print(sub.to_string(index=False))

    # Heatmap
    plt.rcParams.update({'figure.dpi': 110, 'font.size': 9})
    for band, unit in [('resp', 'br/min'), ('card', 'BPM')]:
        pivot = (df[df['Band'] == band]
                 .pivot(index='Channel', columns='Method', values=f'MAE ({unit})')
                 [METHOD_NAMES])
        fig, ax = plt.subplots(figsize=(9, 3))
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r')
        ax.set_xticks(range(len(METHOD_NAMES)))
        ax.set_xticklabels([METHOD_LABELS[k] for k in METHOD_NAMES],
                            rotation=25, ha='right', fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j, meth in enumerate(METHOD_NAMES):
                v = pivot.values[i, j]
                ax.text(j, i, f'{v:.1f}' if not np.isnan(v) else 'NaN',
                        ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, label=f'MAE ({unit})')
        lbl = 'Respiratory Rate' if band == 'resp' else 'Heart Rate'
        ax.set_title(f'MAE  —  {lbl}  |  {m["label"]} {m["subject"]}-{m["initials"]}',
                     fontsize=10)
        plt.tight_layout()
        plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Heart rate & respiratory rate from cap sensor data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--subject',  default='OS001',
                   help='Subject ID, e.g. OS001  (default: OS001)')
    p.add_argument('--night',    default=2, type=int,
                   help='Night number 1 or 2  (default: 2)')
    p.add_argument('--start',    default=2.0, type=float,
                   help='Window start time in hours  (default: 2.0)')
    p.add_argument('--win-min',  default=1.0, type=float,
                   help='Inspection window length in minutes  (default: 1.0)')
    p.add_argument('--mode',     default='inspect',
                   choices=['inspect', 'rates', 'metrics', 'all'],
                   help='What to run  (default: inspect)')
    p.add_argument('--no-acc-removal', action='store_true',
                   help='Disable accelerometer motion removal')
    p.add_argument('--win-sec',  default=20.0, type=float,
                   help='Sliding window length in seconds  (default: 20)')
    p.add_argument('--step-sec', default=1.0,  type=float,
                   help='Sliding window step in seconds  (default: 1)')
    return p.parse_args()


def main():
    args    = parse_args()
    acc_rem = not args.no_acc_removal

    meta    = find_meta(args.subject, args.night)
    session = load_session(meta)

    if args.mode in ('inspect', 'all'):
        mode_inspect(session, args.start, win_hr=args.win_min / 60,
                     acc_removal=acc_rem)

    if args.mode in ('rates', 'all'):
        mode_rates(session, acc_removal=acc_rem,
                   win_sec=args.win_sec, step_sec=args.step_sec)

    if args.mode in ('metrics', 'all'):
        mode_metrics(session, acc_removal=acc_rem,
                     win_sec=args.win_sec, step_sec=args.step_sec)


if __name__ == '__main__':
    main()
