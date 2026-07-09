"""
Spindle validation — does the capacitive mask sense sleep spindles?

Design mirrors the SWA validation (Lucey-style): run an identical detector on
contact EEG (positive control) and on the CAP channels, using PSG-scored spindle
times as ground truth.

Two questions:
  (1) DIRECT: is there 11-16 Hz (sigma) power time-locked to spindles in CAP?
      -> spindle-triggered sigma-envelope average + event-vs-control AUC.
      EEG must light up (validates timing); CAP is the test.
  (2) INDIRECT: is there an autonomic/hemodynamic correlate — a transient in the
      CAP cardiac-band (0.5-3 Hz) pulsation envelope time-locked to spindles?

Outputs -> analysis/spindles/outputs/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles

FS = 100.0
SIGMA_LO, SIGMA_HI = 11.0, 16.0
CARD_LO, CARD_HI = 0.5, 3.0
OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)

N2_CODE = 2
TRIG_HALF = 4.0        # +/- s for sigma triggered average
TRIG_HALF_CARD = 10.0  # +/- s for cardiac triggered average
CORE_HALF = 0.25       # +/- s window used for event-vs-control power


def bp_env(sig, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    f = filtfilt(b, a, sig.astype(np.float64))
    return np.abs(hilbert(f))


def zscore(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s if s > 0 else 1.0)


def stage_at(t_hr, prof):
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(t_hr), -1, np.int8)
    for i, t in enumerate(t_hr):
        j = np.argmin(np.abs(tep - t))
        if abs(tep[j] - t) < 30.0 / 3600.0:
            out[i] = codes[j]
    return out


def triggered_average(env_z, centers_samp, half_samp):
    """Mean peri-event curve; events too close to edges are skipped."""
    n = len(env_z)
    win = 2 * half_samp + 1
    acc = np.zeros(win)
    k = 0
    for c in centers_samp:
        a, b = c - half_samp, c + half_samp + 1
        if a < 0 or b > n:
            continue
        acc += env_z[a:b]
        k += 1
    return (acc / k if k else acc), k


def event_vs_control_auc(env, centers_samp, control_samp, core_samp):
    """AUC that core-window mean power separates spindle from control windows."""
    def core_power(idx):
        vals = []
        for c in idx:
            a, b = c - core_samp, c + core_samp + 1
            if a < 0 or b > len(env):
                continue
            vals.append(np.mean(env[a:b]))
        return np.array(vals)

    pe = core_power(centers_samp)
    pc = core_power(control_samp)
    if len(pe) < 5 or len(pc) < 5:
        return np.nan, np.nan
    # rank-based AUC
    allv = np.concatenate([pe, pc])
    ranks = pd.Series(allv).rank().to_numpy()
    r_e = ranks[:len(pe)].sum()
    auc = (r_e - len(pe) * (len(pe) + 1) / 2) / (len(pe) * len(pc))
    log_ratio = np.log2(np.median(pe) / np.median(pc)) if np.median(pc) > 0 else np.nan
    return auc, log_ratio


def run_session(idx, rng):
    meta = SESSION_META[idx]
    s = load_session(idx)
    s.sleep_profile = load_sleep_profile(s)
    if s.sleep_profile is None:
        return None
    sp = load_spindles(s)
    if sp is None:
        return None

    # restrict to N2 spindles for a clean, physiology-matched test
    stg = stage_at(sp['center_hr'], s.sleep_profile)
    n2 = stg == N2_CODE
    cen_hr = sp['center_hr'][n2]
    if len(cen_hr) < 20:
        return None
    cen_samp = np.round(cen_hr * 3600.0 * FS).astype(int)

    # control samples: N2 timepoints >=3 s from any spindle
    prof = s.sleep_profile
    n2_mask_ep = prof['codes'] == N2_CODE
    n2_starts = prof['t_ep_hr'][n2_mask_ep]
    cand = []
    for t0 in n2_starts:
        for frac in (0.25, 0.5, 0.75):
            cand.append((t0 + frac * 30.0 / 3600.0))
    cand = np.array(cand)
    # keep candidates >=3 s from any spindle center
    if len(cand):
        d = np.min(np.abs(cand[:, None] - sp['center_hr'][None, :]), axis=1) * 3600.0
        cand = cand[d >= 3.0]
    if len(cand) > len(cen_hr):
        cand = rng.choice(cand, size=len(cen_hr), replace=False)
    ctrl_samp = np.round(cand * 3600.0 * FS).astype(int)

    channels = {
        'EEG':     s.psg['EEG'].astype(np.float64),
        'CLE-CRE': (s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)),
        'CLE':     s.cap['CLE'].astype(np.float64),
        'CRE':     s.cap['CRE'].astype(np.float64),
        'CH':      s.cap['CH'].astype(np.float64),
    }

    half = int(TRIG_HALF * FS)
    half_c = int(TRIG_HALF_CARD * FS)
    core = int(CORE_HALF * FS)

    rows = []
    trig_sigma = {}
    trig_card = {}
    for name, sig in channels.items():
        env_s = bp_env(sig, SIGMA_LO, SIGMA_HI)
        env_sz = zscore(env_s)
        ta, k = triggered_average(env_sz, cen_samp, half)
        trig_sigma[name] = ta
        auc, lr = event_vs_control_auc(env_s, cen_samp, ctrl_samp, core)

        # cardiac-band triggered average (autonomic coupling)
        env_c = bp_env(sig, CARD_LO, CARD_HI)
        env_cz = zscore(env_c)
        tac, _ = triggered_average(env_cz, cen_samp, half_c)
        trig_card[name] = tac

        rows.append({
            'session': meta['label'], 'subject': meta['subject'], 'channel': name,
            'n_spindles_N2': len(cen_hr), 'n_events_used': k,
            'sigma_auc': auc, 'sigma_log2ratio': lr,
        })

    return {
        'rows': rows,
        'trig_sigma': trig_sigma, 'trig_card': trig_card,
        'label': meta['label'], 'subject': meta['subject'],
        'n_spindles_N2': len(cen_hr),
        't_axis_sigma': np.arange(-half, half + 1) / FS,
        't_axis_card': np.arange(-half_c, half_c + 1) / FS,
    }


def main():
    rng = np.random.default_rng(42)
    all_rows = []
    trig_sigma_stack = {}   # channel -> list of curves
    trig_card_stack = {}
    t_sig = t_card = None
    per_session = []

    for idx in range(len(SESSION_META)):
        try:
            res = run_session(idx, rng)
        except Exception as e:
            print(f'[{idx}] FAILED: {e}')
            continue
        if res is None:
            print(f'[{idx}] skipped (no spindles / no N2)')
            continue
        all_rows.extend(res['rows'])
        t_sig = res['t_axis_sigma']; t_card = res['t_axis_card']
        for ch, curve in res['trig_sigma'].items():
            trig_sigma_stack.setdefault(ch, []).append(curve)
        for ch, curve in res['trig_card'].items():
            trig_card_stack.setdefault(ch, []).append(curve)
        per_session.append(res)
        eeg_auc = [r['sigma_auc'] for r in res['rows'] if r['channel'] == 'EEG'][0]
        cle_auc = [r['sigma_auc'] for r in res['rows'] if r['channel'] == 'CLE-CRE'][0]
        print(f"{res['label']}: n_N2={res['n_spindles_N2']:4d}  EEG_AUC={eeg_auc:.3f}  CLE-CRE_AUC={cle_auc:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, 'spindle_per_session.csv'), index=False)

    # cross-session summary
    summ = (df.groupby('channel')
              .agg(sigma_auc_median=('sigma_auc', 'median'),
                   sigma_auc_mean=('sigma_auc', 'mean'),
                   sigma_auc_std=('sigma_auc', 'std'),
                   log2ratio_median=('sigma_log2ratio', 'median'),
                   n_sessions=('sigma_auc', 'count'))
              .reset_index())
    summ.to_csv(os.path.join(OUT, 'spindle_summary.csv'), index=False)
    print('\n=== Cross-session sigma-band AUC (spindle vs control) ===')
    print(summ.to_string(index=False))

    # save triggered-average arrays for plotting
    np.savez(os.path.join(OUT, 'triggered_averages.npz'),
             t_sig=t_sig, t_card=t_card,
             **{f'sig_{ch}': np.array(v) for ch, v in trig_sigma_stack.items()},
             **{f'card_{ch}': np.array(v) for ch, v in trig_card_stack.items()})
    print(f'\nSaved outputs to {OUT}')
    return df, summ


if __name__ == '__main__':
    main()
