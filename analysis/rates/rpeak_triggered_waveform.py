"""
Does the capacitive cardiac pulse actually have TWO peaks per heartbeat?

The paper's k ~ 2 story (mean-rate overcount, cardiac tracking failure, harmonic ladder,
age-invariant cardiac k) all rest on the claim that the CAP pulse is biphasic (systolic
peak + dicrotic notch). This shows it directly: an ECG R-peak-triggered ensemble average
of the CAP cardiac-band signal. If it shows two maxima per cardiac cycle, the k ~ 2
mechanism is demonstrated, not asserted.

ECG R-peaks are the fiducial (already used as cardiac ground truth). Per beat we take the
CAP window R-0.15 s .. R+ (median RR) s, de-mean it, and average over all asleep beats.

Outputs -> analysis/rates/outputs/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.ground_truth import _ecg_rpeaks

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)
FS = 100.0
PRE = 0.15          # s before R-peak
POST_PAD = 0.15     # s after one median-RR cycle
CARD_BAND = (0.5, 8.0)   # wide enough to preserve pulse morphology (notch)


def bp(sig, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return filtfilt(b, a, sig.astype(np.float64))


def stage_codes_at(samp_idx, prof):
    """stage code per sample index using nearest 30-s epoch (or -1)."""
    t_hr = samp_idx / FS / 3600.0
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(samp_idx), -1, np.int8)
    j = np.searchsorted(tep, t_hr) - 1
    j = np.clip(j, 0, len(tep) - 1)
    out[:] = codes[j]
    return out


def run_session(idx, channel='CLE-CRE'):
    s = load_session(idx)
    ecg = s.psg.get('ECG')
    if ecg is None:
        return None
    try:
        rpeaks = _ecg_rpeaks(ecg.astype(np.float64), FS)
    except Exception:
        return None
    if len(rpeaks) < 200:
        return None
    rr = np.median(np.diff(rpeaks)) / FS
    if not (0.4 < rr < 1.5):
        return None

    if channel == 'CLE-CRE':
        sig = s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    else:
        sig = s.cap[channel].astype(np.float64)
    sig = bp(sig, *CARD_BAND)

    # keep only asleep beats (exclude Wake=4 and unscored=-1) to reduce motion
    prof = load_sleep_profile(s)
    if prof is not None:
        st = stage_codes_at(rpeaks, prof)
        rpeaks = rpeaks[(st != 4) & (st != -1)]
    if len(rpeaks) < 200:
        return None

    pre = int(PRE * FS)
    post = int((rr + POST_PAD) * FS)
    win = pre + post
    beats = []
    for r in rpeaks:
        a, b = r - pre, r + post
        if a < 0 or b > len(sig):
            continue
        seg = sig[a:b]
        beats.append(seg - seg.mean())
    beats = np.array(beats)
    ens = beats.mean(0)
    ens_norm = ens / np.max(np.abs(ens))
    t = (np.arange(win) - pre) / FS

    # count maxima within one cardiac cycle (0 .. rr), prominence-gated
    cyc = (t >= 0.0) & (t <= rr)
    seg = ens[cyc]
    prom = 0.15 * (seg.max() - seg.min())
    pk, _ = find_peaks(seg, prominence=prom)
    n_peaks = len(pk)

    return dict(label=SESSION_META[idx]['label'], subj=SESSION_META[idx]['label'].split('N')[0],
                t=t, ens=ens_norm, rr=rr, n_beats=len(beats), n_peaks_per_cycle=n_peaks)


def main():
    results = []
    for idx in range(len(SESSION_META)):
        try:
            r = run_session(idx)
        except Exception as e:
            print(f'[{idx}] FAIL {e}'); continue
        if r is None:
            print(f'[{SESSION_META[idx]["label"]}] skipped (ECG unusable)'); continue
        results.append(r)
        print(f'{r["label"]}: RR={r["rr"]:.2f}s beats={r["n_beats"]:5d}  '
              f'peaks/cycle={r["n_peaks_per_cycle"]}')

    npk = [r['n_peaks_per_cycle'] for r in results]
    print(f'\nSessions with 2 peaks per cardiac cycle: {sum(n==2 for n in npk)}/{len(results)}'
          f'  (>=2: {sum(n>=2 for n in npk)}/{len(results)})')
    pd.DataFrame([{k: r[k] for k in ('label','subj','rr','n_beats','n_peaks_per_cycle')}
                  for r in results]).to_csv(os.path.join(OUT,'rpeak_waveform_peaks.csv'), index=False)

    # ── figure: per-session ensemble averages ─────────────────────────────────
    n = len(results)
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1*ncol, 2.5*nrow), squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        ax.plot(r['t'], r['ens'], color='#8E44AD', lw=1.6)
        ax.axvline(0, color='r', ls=':', lw=1, label='ECG R')
        ax.axvline(r['rr'], color='gray', ls='--', lw=0.8)
        cyc = (r['t'] >= 0) & (r['t'] <= r['rr'])
        seg = r['ens'][cyc]; tt = r['t'][cyc]
        prom = 0.15*(seg.max()-seg.min())
        pk,_ = find_peaks(seg, prominence=prom)
        ax.plot(tt[pk], seg[pk], 'k^', ms=6, zorder=5)
        ax.set_title(f"{r['label']}  ({r['n_peaks_per_cycle']} peaks/beat)", fontsize=9)
        ax.set_xlabel('t from R-peak (s)', fontsize=8)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle('ECG R-peak-triggered CAP cardiac waveform (CLE-CRE, asleep beats): '
                 'biphasic pulse underlies k~2', y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_rpeak_triggered_waveform.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()
