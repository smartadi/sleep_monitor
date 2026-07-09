"""
Two follow-up tests of whether the capacitive mask carries ANY spindle signature,
after the direct sigma-band test came back negative (AUC 0.50).

  #3 HEART-RATE ROUTE (the only physiologically plausible positive):
     Sleep spindles carry a stereotyped biphasic autonomic beat: a transient
     heart-rate change time-locked to the spindle. The mask measures cardiac
     pulsation well, so if that HR modulation exists it could leak into CAP even
     though 11-16 Hz does not. We build instantaneous HR from (a) ECG R-peaks
     (does the autonomic signature exist in THIS cohort? -> positive control) and
     (b) CAP cardiac-band pulse peaks (does the mask capture it?), then take the
     spindle-triggered average of each.

  #2 COHERENCE ROUTE (converts the negative into a mechanistic statement):
     Magnitude-squared coherence between EEG and CAP in the sigma band tests for
     ANY electrical leakage/volume-conduction, not just detectable spindles.
     Anchor: CLE vs CRE coherence (two CAP channels sharing one physical source)
     must be high at resp/cardiac freqs -> proves the estimator works, so a
     near-zero EEG-CAP sigma coherence is a true absence of electrical pickup.

Outputs -> analysis/spindles/outputs/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, coherence

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.ground_truth import gt_heart_rate
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles

FS = 100.0
CARD_LO, CARD_HI = 0.5, 3.0
SIGMA_LO, SIGMA_HI = 11.0, 16.0
N2_CODE = 2
HR_HALF = 15.0          # +/- s for HR triggered average
HR_GRID = 4.0           # Hz, resampled HR grid
OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)


def bp(sig, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return filtfilt(b, a, sig.astype(np.float64))


def stage_at(t_hr, prof):
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(t_hr), -1, np.int8)
    for i, t in enumerate(t_hr):
        j = np.argmin(np.abs(tep - t))
        if abs(tep[j] - t) < 30.0 / 3600.0:
            out[i] = codes[j]
    return out


def instantaneous_hr(peak_t_s, t_grid_s):
    """Interpolate 1/IBI (bpm) onto a uniform grid. NaN outside peak coverage."""
    if len(peak_t_s) < 3:
        return np.full_like(t_grid_s, np.nan)
    ibi = np.diff(peak_t_s)
    bpm = 60.0 / ibi
    t_mid = 0.5 * (peak_t_s[:-1] + peak_t_s[1:])
    # physiological gate
    ok = (bpm > 35) & (bpm < 110)
    t_mid, bpm = t_mid[ok], bpm[ok]
    hr = np.interp(t_grid_s, t_mid, bpm, left=np.nan, right=np.nan)
    return hr


def cap_pulse_peaks(cle_cre):
    card = bp(cle_cre, CARD_LO, CARD_HI)
    min_dist = int(FS / CARD_HI * 0.6)
    prom = 0.3 * np.std(card)
    pk, _ = find_peaks(card, distance=min_dist, prominence=prom)
    return pk / FS


def triggered_hr(hr, t_grid_s, centers_s, half_s):
    half_n = int(half_s * HR_GRID)
    win = 2 * half_n + 1
    dt = 1.0 / HR_GRID
    acc = np.zeros(win)
    cnt = np.zeros(win)
    for c in centers_s:
        i0 = int(round((c - t_grid_s[0]) * HR_GRID)) - half_n
        seg = hr[i0:i0 + win] if i0 >= 0 and i0 + win <= len(hr) else None
        if seg is None or np.all(np.isnan(seg)):
            continue
        m = np.isfinite(seg)
        acc[m] += seg[m]
        cnt[m] += 1
    mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    tax = (np.arange(win) - half_n) * dt
    return tax, mean, int(np.nanmax(cnt))


def run_hr_route():
    rng = np.random.default_rng(7)
    ecg_curves, cap_curves, ecg_null, cap_null = [], [], [], []
    tax = None
    rows = []
    for idx in range(len(SESSION_META)):
        meta = SESSION_META[idx]
        try:
            s = load_session(idx)
            s.sleep_profile = load_sleep_profile(s)
            sp = load_spindles(s)
            if sp is None or s.sleep_profile is None:
                continue
            stg = stage_at(sp['center_hr'], s.sleep_profile)
            cen_s = sp['center_hr'][stg == N2_CODE] * 3600.0
            if len(cen_s) < 20:
                continue

            n = s.n_samples
            t_grid = np.arange(0, n / FS, 1.0 / HR_GRID)

            ecg_gt = gt_heart_rate(s)
            hr_ecg = instantaneous_hr(ecg_gt.peak_times_s, t_grid)

            cle_cre = s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
            hr_cap = instantaneous_hr(cap_pulse_peaks(cle_cre), t_grid)

            # random N2 control centers (same count)
            n2_starts = s.sleep_profile['t_ep_hr'][s.sleep_profile['codes'] == N2_CODE]
            null_s = (rng.choice(n2_starts, size=min(len(cen_s), len(n2_starts)),
                                 replace=len(n2_starts) < len(cen_s))
                      + 0.5 * 30.0 / 3600.0) * 3600.0

            tax, m_ecg, _ = triggered_hr(hr_ecg, t_grid, cen_s, HR_HALF)
            _,   m_cap, _ = triggered_hr(hr_cap, t_grid, cen_s, HR_HALF)
            _,   n_ecg, _ = triggered_hr(hr_ecg, t_grid, null_s, HR_HALF)
            _,   n_cap, _ = triggered_hr(hr_cap, t_grid, null_s, HR_HALF)

            # express as delta-bpm from window mean
            def dz(m):
                return m - np.nanmean(m)
            ecg_curves.append(dz(m_ecg)); cap_curves.append(dz(m_cap))
            ecg_null.append(dz(n_ecg));   cap_null.append(dz(n_cap))

            # peak-to-trough of the ECG-HR response in [-2,10]s
            band = (tax >= -2) & (tax <= 10)
            ecg_pp = np.nanmax(dz(m_ecg)[band]) - np.nanmin(dz(m_ecg)[band])
            cap_pp = np.nanmax(dz(m_cap)[band]) - np.nanmin(dz(m_cap)[band])
            rows.append({'session': meta['label'], 'n_spindles_N2': len(cen_s),
                         'ecg_hr_ptp_bpm': ecg_pp, 'cap_hr_ptp_bpm': cap_pp})
            print(f"{meta['label']}: HR route  ECG ptp={ecg_pp:.2f} bpm  CAP ptp={cap_pp:.2f} bpm")
        except Exception as e:
            print(f'[{idx}] HR route failed: {e}')
            continue

    np.savez(os.path.join(OUT, 'spindle_hr_triggered.npz'),
             tax=tax,
             ecg=np.array(ecg_curves), cap=np.array(cap_curves),
             ecg_null=np.array(ecg_null), cap_null=np.array(cap_null))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'spindle_hr_route.csv'), index=False)
    return rows


def run_coherence_route():
    freqs = None
    coh_eeg_cap, coh_cle_cre = [], []
    rows = []
    for idx in range(len(SESSION_META)):
        meta = SESSION_META[idx]
        try:
            s = load_session(idx)
            eeg = s.psg['EEG'].astype(np.float64)
            cle = s.cap['CLE'].astype(np.float64)
            cre = s.cap['CRE'].astype(np.float64)
            cle_cre = cle - cre

            nper = 2048
            f, c1 = coherence(eeg, cle_cre, fs=FS, nperseg=nper)
            _, c2 = coherence(cle, cre, fs=FS, nperseg=nper)
            freqs = f
            coh_eeg_cap.append(c1); coh_cle_cre.append(c2)

            sig = (f >= SIGMA_LO) & (f <= SIGMA_HI)
            rows.append({
                'session': meta['label'],
                'eeg_cap_coh_sigma': float(np.mean(c1[sig])),
                'cle_cre_coh_sigma': float(np.mean(c2[sig])),
                'cle_cre_coh_resp': float(np.mean(c2[(f >= 0.1) & (f <= 0.5)])),
                'cle_cre_coh_card': float(np.mean(c2[(f >= 0.8) & (f <= 2.0)])),
            })
            print(f"{meta['label']}: coh  EEG-CAP sigma={rows[-1]['eeg_cap_coh_sigma']:.3f}  "
                  f"CLE-CRE sigma={rows[-1]['cle_cre_coh_sigma']:.3f}  "
                  f"card={rows[-1]['cle_cre_coh_card']:.3f}")
        except Exception as e:
            print(f'[{idx}] coherence failed: {e}')
            continue

    np.savez(os.path.join(OUT, 'spindle_coherence.npz'),
             freqs=freqs,
             eeg_cap=np.array(coh_eeg_cap), cle_cre=np.array(coh_cle_cre))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'spindle_coherence.csv'), index=False)
    return rows


if __name__ == '__main__':
    print('=== #3 heart-rate route ===')
    run_hr_route()
    print('\n=== #2 coherence route ===')
    run_coherence_route()
    print(f'\nSaved outputs to {OUT}')
