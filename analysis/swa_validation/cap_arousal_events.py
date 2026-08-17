#!/usr/bin/env python
"""
CAP arousal events — an inventory.

This is deliberately a REPORTING script, not a hypothesis test. It detects
transient events in the CAP signal, throws out the ones that are head motion or
electrode re-seat, and then reports what each surviving event looks like in the
other channels. No p-values, no AUCs, no classifier. The output is a catalogue:
how many events of each kind, where in the night they fall, and what a typical
one looks like.

Detection (CLE-CRE, 1 s grid)
    envelope   : detrended std of the CAP signal in a 2 s window, stepped 1 s
    baseline   : centred 4 min rolling median of log10 envelope
    event      : robust z >= Z_EVENT for >= MIN_DUR_S, gaps < MERGE_GAP_S merged

Rejection (an event must survive all three to be reported)
    motion     : dynamic-acceleration z >= Z_MOTION anywhere in onset +/- 5 s
    head turn  : head angle moves > TURN_DEG over onset +/- 10 s
    level step : CAP mean shifts more than 5 robust sd across the event
                 (electrode re-seat / coupling change, not an event)
    impulse    : a single sample-to-sample jump above IMPULSE_SD robust sd
                 (an electrical glitch raises the 2 s envelope just as well as
                 a real transient does, and must not be counted as one)

Classes (evaluated in order; every event lands in exactly one)
    1 cortical_scored    a PSG-scored arousal overlaps onset +/- 3 s
    2 cortical_unscored  no scored arousal, but EEG 8-30 Hz rises (z >= Z_EEG)
    3 autonomic_only     no scored arousal, no EEG rise, but HR surge and/or
                         pulse-wave amplitude drop
    4 unexplained        none of the above

Class 4 is reported, not hidden — it is the error bar on classes 2 and 3.

The thresholds are descriptive detection criteria, not tested hypotheses. They
are printed in the summary so any number in the paper can be traced to them.

Outputs
    reports/swa_validation/cap_events/cap_arousal_events.csv     per event
    reports/swa_validation/cap_events/cap_arousal_summary.csv    per session
    reports/swa_validation/cap_events/cap_arousal_by_stage.csv   class x stage
    writeup/figures/cap_events/event_inventory.png
    writeup/figures/cap_events/event_profiles.png
    writeup/figures/cap_events/event_examples_<SESSION>.png
    writeup/figures/cap_events/event_night_<SESSION>.png

Usage
    .venv/Scripts/python.exe analysis/swa_validation/cap_arousal_events.py --sessions S1N1
    .venv/Scripts/python.exe analysis/swa_validation/cap_arousal_events.py --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import (
    load_session, load_sleep_profile, load_arousals, load_autonomic_arousals,
    load_apnea_events, gt_heart_rate, bandpass,
)
from sleep_monitor.motion import head_angle, dynamic_acceleration
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, CAP_SCALE_TO_FF,
)
from sleep_monitor.sessions import SESSION_META

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'swa_validation' / 'cap_events'
FIG_DIR = ROOT / 'writeup' / 'figures' / 'cap_events'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── detection / rejection / classification criteria ──────────────────────────
WIN_S, HOP_S = 2.0, 1.0      # envelope window and step -> the 1 s analysis grid
BASE_WIN_S = 240.0           # rolling baseline for the robust z
Z_EVENT = 3.0                # CAP envelope z to open an event
MIN_DUR_S = 2.0              # an event must stay above threshold this long
MERGE_GAP_S = 5.0            # events closer than this are one event

Z_MOTION = 3.0               # dynamic-acceleration z that rejects an event
MOTION_PAD_S = 5.0
TURN_DEG = 5.0               # head-angle change that rejects an event
TURN_PAD_S = 10.0
STEP_SD = 5.0                # CAP level step, in robust sd, that rejects
IMPULSE_SD = 20.0            # single-sample jump, in robust sd, that rejects

Z_EEG = 2.0                  # EEG 8-30 Hz z that counts as a cortical shift
HR_SURGE_BPM = 3.0           # HR rise over baseline that counts as autonomic
PWA_DROP_FRAC = 0.70         # pulse amplitude falling below this fraction

BASE_LO_S, BASE_HI_S = -30.0, -5.0    # pre-event reference window
EEG_LO_S, EEG_HI_S = -2.0, 8.0
HR_LO_S, HR_HI_S = 0.0, 15.0
PWA_LO_S, PWA_HI_S = 0.0, 20.0
AROUSAL_PAD_S = 3.0                   # scored-arousal overlap tolerance
APNEA_PAD_S = 15.0                    # "follows a respiratory event" window

PROFILE_PAD_S = 60                    # peri-event figure half-width

CLASSES = ['cortical_scored', 'cortical_unscored', 'autonomic_only', 'unexplained']
CLASS_LABELS = {
    'cortical_scored':   'Class 1\nscored arousal',
    'cortical_unscored': 'Class 2\nEEG shift, unscored',
    'autonomic_only':    'Class 3\nautonomic only',
    'unexplained':       'Class 4\nunexplained',
}
CLASS_COLORS = {
    'cortical_scored':   '#2C7BB6',
    'cortical_unscored': '#7FB3D5',
    'autonomic_only':    '#D7191C',
    'unexplained':       '#BDBDBD',
}


# ── small helpers ────────────────────────────────────────────────────────────
def grid_windows(x, fs, win_s=WIN_S, hop_s=HOP_S):
    """Sliding windows on the shared 1 s analysis grid. Returns (W, t_sec)."""
    n, h = int(round(fs * win_s)), int(round(fs * hop_s))
    W = sliding_window_view(np.asarray(x, np.float64), n)[::h]
    t = (np.arange(W.shape[0]) * h + n / 2.0) / fs
    return W, t


def win_std_detrend(W):
    """Per-window std after removing the window's own linear trend.

    The detrend matters: a 2 s slice of the overnight drift is not an event,
    and plain np.var would count it as one.
    """
    n = W.shape[1]
    tc = np.arange(n, dtype=np.float64)
    tc -= tc.mean()
    denom = (tc ** 2).sum()
    m = W.mean(axis=1, keepdims=True)
    slope = (W * tc).sum(axis=1) / denom
    resid = W - m - slope[:, None] * tc
    return resid.std(axis=1)


def robust_z(v, win_n):
    """Deviation from a centred rolling median, scaled by a rolling MAD."""
    s = pd.Series(v)
    base = s.rolling(win_n, center=True, min_periods=win_n // 4).median()
    dev = s - base
    scale = dev.abs().rolling(win_n, center=True,
                              min_periods=win_n // 4).median() * 1.4826
    scale = scale.replace(0.0, np.nan)
    return (dev / scale).to_numpy(), dev.to_numpy(), base.to_numpy()


def runs_of_true(mask):
    """[(start, end_exclusive), ...] for each contiguous True run."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def merge_runs(runs, gap):
    out = []
    for a, b in runs:
        if out and a - out[-1][1] < gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def win_slice(i0, lo_s, hi_s, n):
    """Grid indices for [onset+lo_s, onset+hi_s]; the grid is 1 s per step."""
    a = int(max(0, i0 + lo_s / HOP_S))
    b = int(min(n, i0 + hi_s / HOP_S + 1))
    return a, b


def safe(fn, arr, default=np.nan):
    a = arr[np.isfinite(arr)]
    return float(fn(a)) if a.size else default


def intervals_hit(ev, t0_hr, pad_hr):
    """Does any scored interval overlap [t0 - pad, t0 + pad]?"""
    if ev is None or len(ev['start_hr']) == 0:
        return False
    return bool(np.any((ev['start_hr'] <= t0_hr + pad_hr) &
                       (ev['end_hr'] >= t0_hr - pad_hr)))


# ── per-session extraction ───────────────────────────────────────────────────
def process(idx, meta, verbose=True):
    s = load_session(idx)
    sp = load_sleep_profile(s)
    if sp is None:
        return None
    fs = s.fs
    lbl = meta['label']

    cle = s.cap['CLE'].astype(np.float64) * CAP_SCALE_TO_FF
    cre = s.cap['CRE'].astype(np.float64) * CAP_SCALE_TO_FF
    d = cle - cre

    # ---- everything onto the shared 1 s grid -------------------------------
    W, t_g = grid_windows(d, fs)
    cap_std = win_std_detrend(W)
    cap_level = W.mean(axis=1)
    del W
    ng = len(t_g)
    t_hr = t_g / 3600.0

    eeg_fast = bandpass(s.psg['EEG'].astype(np.float64), 8.0, 30.0, fs)
    Wf, _ = grid_windows(eeg_fast, fs)
    eeg_rms = Wf.std(axis=1)[:ng]
    del Wf

    dyn = dynamic_acceleration(s.cap['aX'], s.cap['aY'], s.cap['aZ'], fs)
    Wm, _ = grid_windows(dyn, fs)
    mot_rms = Wm.std(axis=1)[:ng]
    del Wm

    ang = head_angle(s.cap['aX'], s.cap['aY'], s.cap['aZ'], fs)
    turn = ang['turn_deg'][::int(round(fs * HOP_S))][:ng]

    pleth_bp = bandpass(s.psg['Pleth'].astype(np.float64), 0.5, 3.0, fs)
    Wp, _ = grid_windows(pleth_bp, fs)
    pwa = (Wp.max(axis=1) - Wp.min(axis=1))[:ng]
    del Wp

    try:
        gt = gt_heart_rate(s)
        hr = np.interp(t_g, gt.peak_times_s[1:], gt.instant_rate_bpm,
                       left=np.nan, right=np.nan)
        hr = pd.Series(hr).rolling(3, center=True, min_periods=1).median().to_numpy()
        hr_src = gt.signal_used
    except Exception:
        hr = np.full(ng, np.nan)
        hr_src = 'none'

    # ---- events ------------------------------------------------------------
    base_n = int(round(BASE_WIN_S / HOP_S))
    cap_log = np.log10(cap_std + 1e-9)
    z_cap, dev_cap, _ = robust_z(cap_log, base_n)
    z_eeg, dev_eeg, _ = robust_z(np.log10(eeg_rms + 1e-12), base_n)
    z_mot, dev_mot, _ = robust_z(np.log10(mot_rms + 1e-12), base_n)

    above = np.nan_to_num(z_cap, nan=0.0) >= Z_EVENT
    runs = merge_runs(runs_of_true(above), int(MERGE_GAP_S / HOP_S))
    runs = [(a, b) for a, b in runs if (b - a) * HOP_S >= MIN_DUR_S]

    # level-step scale: how much does the CAP mean normally move in 10 s?
    d10 = cap_level[10:] - cap_level[:-10]
    step_scale = 1.4826 * np.median(np.abs(d10 - np.median(d10)))
    step_thresh = STEP_SD * max(step_scale, 1e-9)

    # impulse scale: the normal sample-to-sample jump of the raw CAP signal
    d1 = np.diff(d)
    imp_scale = max(1.4826 * np.median(np.abs(d1 - np.median(d1))), 1e-12)
    hop_n = int(round(fs * HOP_S))

    ar = load_arousals(s)
    au = load_autonomic_arousals(s)
    ap = load_apnea_events(s)

    rows = []
    for a, b in runs:
        i0 = int(a)
        t0_s, t0_hr = t_g[i0], t_hr[i0]

        ma, mb = win_slice(i0, -MOTION_PAD_S, MOTION_PAD_S, ng)
        ta, tb = win_slice(i0, -TURN_PAD_S, TURN_PAD_S, ng)
        pa, pb = win_slice(i0, 2.0, 12.0, ng)
        qa, qb = win_slice(i0, -12.0, -2.0, ng)

        mot_z = safe(np.nanmax, z_mot[ma:mb])
        turn_swing = (float(np.nanmax(turn[ta:tb]) - np.nanmin(turn[ta:tb]))
                      if tb > ta else np.nan)
        step = (safe(np.nanmedian, cap_level[pa:pb])
                - safe(np.nanmedian, cap_level[qa:qb]))

        rej_motion = np.isfinite(mot_z) and mot_z >= Z_MOTION
        rej_turn = np.isfinite(turn_swing) and turn_swing > TURN_DEG
        rej_step = np.isfinite(step) and abs(step) > step_thresh

        ra = max(0, int((i0 - 1) * hop_n))
        rb = min(len(d1), int((b + 1) * hop_n))
        impulse = (float(np.max(np.abs(d1[ra:rb]))) / imp_scale
                   if rb > ra else np.nan)
        rej_impulse = np.isfinite(impulse) and impulse > IMPULSE_SD

        ba, bb = win_slice(i0, BASE_LO_S, BASE_HI_S, ng)
        ea, eb = win_slice(i0, EEG_LO_S, EEG_HI_S, ng)
        ha, hb = win_slice(i0, HR_LO_S, HR_HI_S, ng)
        wa, wb = win_slice(i0, PWA_LO_S, PWA_HI_S, ng)

        eeg_z = safe(np.nanmax, z_eeg[ea:eb])
        hr_base = safe(np.nanmedian, hr[ba:bb])
        hr_peak = safe(np.nanmax, hr[ha:hb])
        hr_rise = hr_peak - hr_base
        pwa_base = safe(np.nanmedian, pwa[ba:bb])
        pwa_min = safe(np.nanmin, pwa[wa:wb])
        pwa_ratio = pwa_min / pwa_base if pwa_base and np.isfinite(pwa_base) else np.nan

        scored = intervals_hit(ar, t0_hr, AROUSAL_PAD_S / 3600.0)
        eeg_shift = np.isfinite(eeg_z) and eeg_z >= Z_EEG
        hr_surge = np.isfinite(hr_rise) and hr_rise >= HR_SURGE_BPM
        pwa_drop = np.isfinite(pwa_ratio) and pwa_ratio <= PWA_DROP_FRAC

        if scored:
            cls = 'cortical_scored'
        elif eeg_shift:
            cls = 'cortical_unscored'
        elif hr_surge or pwa_drop:
            cls = 'autonomic_only'
        else:
            cls = 'unexplained'

        j = int(np.searchsorted(sp['t_ep_hr'], t0_hr) - 1)
        stage = int(sp['codes'][j]) if 0 <= j < len(sp['codes']) else -1

        post_resp = False
        if ap is not None and len(ap['end_hr']):
            dt = (t0_hr - ap['end_hr']) * 3600.0
            post_resp = bool(np.any((dt >= -2.0) & (dt <= APNEA_PAD_S)))

        rows.append({
            'session': lbl, 'subject': meta['subject'], 'night': meta['night'],
            't_hr': t0_hr, 'grid_i': i0,
            'duration_s': (b - a) * HOP_S,
            'peak_z': safe(np.nanmax, z_cap[a:b]),
            'cap_dev_log10': safe(np.nanmax, dev_cap[a:b]),
            'stage_code': stage, 'stage': STAGE_LABELS.get(stage, '?'),
            'motion_z': mot_z, 'turn_swing_deg': turn_swing,
            'level_step_ff': step, 'step_thresh_ff': step_thresh,
            'impulse_sd': impulse,
            'rej_motion': rej_motion, 'rej_turn': rej_turn, 'rej_step': rej_step,
            'rej_impulse': rej_impulse,
            'kept': not (rej_motion or rej_turn or rej_step or rej_impulse),
            'eeg_z': eeg_z, 'hr_base_bpm': hr_base, 'hr_rise_bpm': hr_rise,
            'pwa_ratio': pwa_ratio,
            'scored_arousal': scored,
            'autonomic_scored': intervals_hit(au, t0_hr, AROUSAL_PAD_S / 3600.0),
            'eeg_shift': eeg_shift, 'hr_surge': hr_surge, 'pwa_drop': pwa_drop,
            'post_respiratory': post_resp,
            'klass': cls,
            'hr_source': hr_src,
        })

    ev = pd.DataFrame(rows)
    if verbose:
        n_keep = int(ev['kept'].sum()) if len(ev) else 0
        print(f'  {lbl}: {len(ev)} detected, {n_keep} kept '
              f'({100 * n_keep / max(len(ev), 1):.0f}%), HR from {hr_src}')

    series = {
        't_g': t_g, 'z_cap': z_cap, 'dev_cap': dev_cap, 'dev_eeg': dev_eeg,
        'dev_mot': dev_mot, 'hr': hr, 'pwa': pwa, 'cap_std': cap_std,
        'cap_level': cap_level, 'turn': turn, 'sp': sp,
        'dur_hr': float(t_hr[-1]),
    }
    return ev, series, s


# ── peri-event profiles (median across events, descriptive only) ─────────────
def profiles(ev, series_by_session):
    pad = int(PROFILE_PAD_S / HOP_S)
    keys = ['dev_cap', 'dev_eeg', 'hr', 'pwa', 'dev_mot']
    acc = {c: {k: [] for k in keys} for c in CLASSES}
    for lbl, sr in series_by_session.items():
        sub = ev[(ev['session'] == lbl) & ev['kept']]
        ng = len(sr['t_g'])
        for _, r in sub.iterrows():
            i0 = int(r['grid_i'])
            if i0 - pad < 0 or i0 + pad + 1 > ng:
                continue
            sl = slice(i0 - pad, i0 + pad + 1)
            base = slice(i0 + int(BASE_LO_S), i0 + int(BASE_HI_S))
            for k in keys:
                seg = np.asarray(sr[k][sl], float).copy()
                b = safe(np.nanmedian, np.asarray(sr[k][base], float))
                if k == 'pwa':
                    seg = seg / b if b else seg * np.nan
                else:
                    seg = seg - (b if np.isfinite(b) else 0.0)
                acc[r['klass']][k].append(seg)
    return acc, pad


# ── figures ──────────────────────────────────────────────────────────────────
def fig_inventory(ev, out):
    sessions = sorted(ev['session'].unique())
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.2))

    ax = axes[0]
    x = np.arange(len(sessions))
    tot = ev.groupby('session').size().reindex(sessions).fillna(0).to_numpy()
    kept_n = (ev[ev['kept']].groupby('session').size()
              .reindex(sessions).fillna(0).to_numpy())
    # An event can trip more than one rejection gate, so these are grouped, not
    # stacked — stacking them would sum past the detection count.
    ax.bar(x, tot, width=0.92, facecolor='none', edgecolor='#2C3E50', lw=1.0,
           label='detected')
    groups = [('rej_motion', '#7F8C8D', 'head motion'),
              ('rej_turn', '#B2BABB', 'head turn'),
              ('rej_step', '#D5D8DC', 'level step'),
              ('rej_impulse', '#5D6D7E', 'impulse'),
              (None, '#E67E22', 'kept')]
    w, off = 0.16, -0.32
    for k, c, lab in groups:
        v = (kept_n if k is None else
             ev[ev[k]].groupby('session').size().reindex(sessions).fillna(0).to_numpy())
        ax.bar(x + off, v, w, color=c, label=lab, edgecolor='white', lw=0.3)
        off += 0.16
    ax.set_xticks(x); ax.set_xticklabels(sessions, rotation=60, fontsize=8)
    ax.set_ylabel('CAP transients')
    ax.set_title('A  What the detector found, and what survived\n'
                 'outline = detected; an event can trip several gates, so the '
                 'bars are grouped, not stacked', fontsize=10, loc='left')
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[1]
    keep = ev[ev['kept']]
    bot = np.zeros(len(sessions))
    for c in CLASSES:
        v = keep[keep['klass'] == c].groupby('session').size()
        v = v.reindex(sessions).fillna(0).to_numpy()
        ax.bar(x, v, bottom=bot, color=CLASS_COLORS[c],
               label=CLASS_LABELS[c].replace('\n', ' — '),
               edgecolor='white', lw=0.4)
        bot += v
    ax.set_xticks(x); ax.set_xticklabels(sessions, rotation=60, fontsize=8)
    ax.set_ylabel('kept events')
    ax.set_title('B  Class composition per night', fontsize=10, loc='left')
    ax.legend(fontsize=7.5)

    ax = axes[2]
    comp = (keep.groupby(['session', 'klass']).size()
            .unstack(fill_value=0).reindex(sessions).fillna(0))
    comp = comp.reindex(columns=CLASSES, fill_value=0)
    frac = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0) * 100
    bot = np.zeros(len(sessions))
    for c in CLASSES:
        v = frac[c].to_numpy()
        ax.bar(x, v, bottom=bot, color=CLASS_COLORS[c], edgecolor='white', lw=0.4)
        bot += np.nan_to_num(v)
    ax.set_xticks(x); ax.set_xticklabels(sessions, rotation=60, fontsize=8)
    ax.set_ylabel('% of kept events'); ax.set_ylim(0, 100)
    ax.set_title('C  Same, as a share of the night', fontsize=10, loc='left')

    ax = axes[3]
    order = [STAGE_LABELS[c] for c in STAGE_ORDER]
    xs = np.arange(len(order))
    bot = np.zeros(len(order))
    for c in CLASSES:
        v = (keep[keep['klass'] == c].groupby('stage').size()
             .reindex(order).fillna(0).to_numpy())
        ax.bar(xs, v, bottom=bot, color=CLASS_COLORS[c], edgecolor='white', lw=0.4)
        bot += v
    ax.set_xticks(xs); ax.set_xticklabels(order)
    ax.set_ylabel('kept events, all sessions pooled')
    ax.set_title('D  Where in sleep the events fall', fontsize=10, loc='left')

    # E — the motion gate and the scored arousals are not independent
    ax = axes[4]
    det = (ev.groupby('session')['scored_arousal'].mean()
           .reindex(sessions).fillna(0).to_numpy() * 100)
    kpt = (keep.groupby('session')['scored_arousal'].mean()
           .reindex(sessions).fillna(0).to_numpy() * 100)
    w = 0.38
    ax.bar(x - w / 2, det, w, color='#34495E', label='all detected transients')
    ax.bar(x + w / 2, kpt, w, color='#E67E22', label='after motion/step rejection')
    ax.set_xticks(x); ax.set_xticklabels(sessions, rotation=60, fontsize=8)
    ax.set_ylabel('% overlapping a scored arousal')
    ax.set_title('E  Why class 1 is small\n'
                 'scored arousals are movement-coupled, so the motion gate '
                 'removes them', fontsize=10, loc='left')
    ax.legend(fontsize=7.5)

    fig.suptitle('CAP transient inventory — every event that is not head motion '
                 'and not an electrode step, sorted by what the other channels show',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=190, bbox_inches='tight'); plt.close(fig)


def fig_profiles(acc, pad, out):
    rows = [('dev_cap', 'CAP envelope\nlog10 dev', None),
            ('dev_eeg', 'EEG 8-30 Hz\nlog10 dev', None),
            ('hr', 'heart rate\nbpm from baseline', 0.0),
            ('pwa', 'pulse amplitude\nratio to baseline', 1.0),
            ('dev_mot', 'head motion\nlog10 dev', None)]
    tt = np.arange(-pad, pad + 1) * HOP_S
    fig, axes = plt.subplots(len(rows), len(CLASSES),
                             figsize=(4.3 * len(CLASSES), 2.15 * len(rows)),
                             sharex=True)
    for j, c in enumerate(CLASSES):
        n_ev = len(acc[c]['dev_cap'])
        for i, (k, ylab, ref) in enumerate(rows):
            ax = axes[i, j]
            arr = np.asarray(acc[c][k], float) if acc[c][k] else np.empty((0, len(tt)))
            if arr.shape[0]:
                med = np.nanmedian(arr, axis=0)
                lo = np.nanpercentile(arr, 25, axis=0)
                hi = np.nanpercentile(arr, 75, axis=0)
                ax.fill_between(tt, lo, hi, color=CLASS_COLORS[c], alpha=0.22, lw=0)
                ax.plot(tt, med, color=CLASS_COLORS[c], lw=1.6)
            if ref is not None:
                ax.axhline(ref, color='0.5', lw=0.7, ls=':')
            ax.axvline(0, color='k', lw=0.8, ls='--')
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if i == 0:
                ax.set_title(f'{CLASS_LABELS[c]}   n = {n_ev}',
                             fontsize=10, color=CLASS_COLORS[c])
            if i == len(rows) - 1:
                ax.set_xlabel('time from event onset (s)')
    fig.suptitle('What a typical event of each class looks like — median and IQR '
                 'across events, all sessions pooled.\n'
                 'Descriptive only: no test is performed on these curves.',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=190, bbox_inches='tight'); plt.close(fig)


def fig_examples(ev, sr, s, lbl, out):
    fs = s.fs
    cle = s.cap['CLE'].astype(np.float64) * CAP_SCALE_TO_FF
    cre = s.cap['CRE'].astype(np.float64) * CAP_SCALE_TO_FF
    d = cle - cre
    eeg = s.psg['EEG'].astype(np.float64)
    eeg_f = bandpass(eeg, 8.0, 30.0, fs)
    pleth = s.psg['Pleth'].astype(np.float64)
    dyn = dynamic_acceleration(s.cap['aX'], s.cap['aY'], s.cap['aZ'], fs)

    keep = ev[(ev['session'] == lbl) & ev['kept']]
    picks = []
    for c in CLASSES:
        # the MEDIAN-strength event, not the strongest: the strongest events are
        # the most artifact-like, so they are the wrong thing to show as typical
        sub = keep[keep['klass'] == c].sort_values('peak_z')
        if len(sub):
            picks.append((c, sub.iloc[len(sub) // 2]))
    if not picks:
        return

    pad_s = 30.0
    rows = ['CLE-CRE (fF)', 'EEG raw', 'EEG 8-30 Hz', 'Pleth', 'head motion']
    fig, axes = plt.subplots(len(rows), len(picks),
                             figsize=(4.6 * len(picks), 1.9 * len(rows)),
                             sharex=True, squeeze=False)
    for j, (c, r) in enumerate(picks):
        t0 = r['t_hr'] * 3600.0
        i0 = int((t0 - pad_s) * fs)
        i1 = int((t0 + pad_s) * fs)
        i0, i1 = max(0, i0), min(len(d), i1)
        tt = np.arange(i0, i1) / fs - t0
        for i, (sig, ylab, col) in enumerate([
                (d, rows[0], '#E67E22'), (eeg, rows[1], '#34495E'),
                (eeg_f, rows[2], '#2C7BB6'), (pleth, rows[3], '#C0392B'),
                (dyn, rows[4], '#7F8C8D')]):
            ax = axes[i, j]
            ax.plot(tt, sig[i0:i1], lw=0.6, color=col)
            ax.axvline(0, color='k', lw=0.9, ls='--')
            ax.margins(x=0)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.5)
            if i == 0:
                ax.set_title(
                    f'{CLASS_LABELS[c]}\n{lbl}  {r["t_hr"]:.2f} h  {r["stage"]}  '
                    f'z={r["peak_z"]:.1f}  EEGz={r["eeg_z"]:.1f}  '
                    f'dHR={r["hr_rise_bpm"]:.1f}  PWA={r["pwa_ratio"]:.2f}',
                    fontsize=8.5, color=CLASS_COLORS[c])
            if i == len(rows) - 1:
                ax.set_xlabel('time from onset (s)')
    fig.suptitle(f'{lbl} — the median-strength kept event of each class, '
                 f'raw traces', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)


def fig_night(ev, sr, lbl, out):
    sp = sr['sp']
    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True,
                             gridspec_kw={'height_ratios': [1.1, 1.6, 1.2]})

    ax = axes[0]
    ax.step(sp['t_ep_hr'], sp['codes'], where='post', lw=0.9, color='#2C3E50')
    ax.set_yticks(STAGE_ORDER)
    ax.set_yticklabels([STAGE_LABELS[c] for c in STAGE_ORDER], fontsize=8)
    ax.set_ylabel('stage', fontsize=9)
    ax.set_title(f'{lbl} — CAP transients across the night', fontsize=12,
                 fontweight='bold', loc='left')

    ax = axes[1]
    ax.plot(sr['t_g'] / 3600.0, sr['z_cap'], lw=0.4, color='#E67E22')
    ax.axhline(Z_EVENT, color='k', lw=0.8, ls='--')
    ax.set_ylabel('CAP envelope z', fontsize=9)
    ax.set_ylim(-3, 12)

    ax = axes[2]
    sub = ev[ev['session'] == lbl]
    lanes = {'rejected': 0}
    lanes.update({c: i + 1 for i, c in enumerate(CLASSES)})
    for _, r in sub.iterrows():
        lane = lanes[r['klass']] if r['kept'] else 0
        col = CLASS_COLORS[r['klass']] if r['kept'] else '#BDC3C7'
        ax.plot([r['t_hr'], r['t_hr']], [lane - 0.4, lane + 0.4], lw=0.8, color=col)
    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels(['rejected'] + [CLASS_LABELS[c].replace('\n', ' ')
                                       for c in CLASSES], fontsize=8)
    ax.set_ylim(-0.7, len(CLASSES) + 0.7)
    ax.set_xlabel('time (hours)')
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sessions', nargs='*', default=None,
                   help='session labels, e.g. S1N1 S1N2')
    p.add_argument('--all', action='store_true')
    p.add_argument('--demo', default=None, help='session for the example figures')
    a = p.parse_args()

    todo = [(i, m) for i, m in enumerate(SESSION_META)
            if a.all or (a.sessions and m['label'] in a.sessions)]
    if not todo:
        todo = [(i, m) for i, m in enumerate(SESSION_META)
                if m['label'] == 'S1N1']

    print('Criteria: '
          f'event z>={Z_EVENT} for >={MIN_DUR_S}s | reject motion z>={Z_MOTION}, '
          f'turn>{TURN_DEG}deg, step>{STEP_SD}sd | '
          f'EEG z>={Z_EEG}, dHR>={HR_SURGE_BPM}bpm, PWA<={PWA_DROP_FRAC}')
    print(f'Processing {len(todo)} session(s)')

    all_ev, series_by_session, sess_obj = [], {}, {}
    for i, m in todo:
        try:
            out = process(i, m)
        except Exception as e:
            print(f'  {m["label"]}: FAILED — {type(e).__name__}: {e}')
            continue
        if out is None:
            print(f'  {m["label"]}: no sleep profile, skipped')
            continue
        ev, sr, s = out
        all_ev.append(ev)
        series_by_session[m['label']] = sr
        sess_obj[m['label']] = s

    if not all_ev:
        print('nothing processed')
        return

    ev = pd.concat(all_ev, ignore_index=True)
    ev.to_csv(REPORT_DIR / 'cap_arousal_events.csv', index=False)

    keep = ev[ev['kept']]
    rows = []
    for lbl, g in ev.groupby('session'):
        k = g[g['kept']]
        dur = series_by_session[lbl]['dur_hr']
        r = {'session': lbl, 'hours': round(dur, 2), 'detected': len(g),
             'det_scored': int(g['scored_arousal'].sum()),
             'rej_motion': int(g['rej_motion'].sum()),
             'rej_turn': int(g['rej_turn'].sum()),
             'rej_step': int(g['rej_step'].sum()),
             'rej_impulse': int(g['rej_impulse'].sum()),
             'kept': len(k), 'kept_per_hour': round(len(k) / dur, 2),
             'kept_scored': int(k['scored_arousal'].sum())}
        for c in CLASSES:
            n = int((k['klass'] == c).sum())
            r[c] = n
            r[f'{c}_per_hour'] = round(n / dur, 2)
        r['post_respiratory'] = int(k['post_respiratory'].sum())
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(REPORT_DIR / 'cap_arousal_summary.csv', index=False)

    by_stage = (keep.groupby(['klass', 'stage']).size().unstack(fill_value=0)
                .reindex(CLASSES).fillna(0).astype(int))
    by_stage.to_csv(REPORT_DIR / 'cap_arousal_by_stage.csv')

    print('\n--- per session ---')
    print(summ.to_string(index=False))
    print('\n--- class x stage (kept events, pooled) ---')
    print(by_stage.to_string())
    print('\n--- pooled ---')
    tot = len(ev)
    print(f'detected {tot}, kept {len(keep)} ({100 * len(keep) / max(tot, 1):.0f}%)')
    ds, ks = int(ev['scored_arousal'].sum()), int(keep['scored_arousal'].sum())
    print(f'overlapping a scored arousal: {ds}/{tot} of detected '
          f'({100 * ds / max(tot, 1):.0f}%) -> {ks}/{len(keep)} of kept '
          f'({100 * ks / max(len(keep), 1):.0f}%)')
    rej_scored = ev[ev['scored_arousal'] & ~ev['kept']]
    if len(rej_scored):
        print(f'  of the {len(rej_scored)} scored-arousal detections rejected: '
              f'{int(rej_scored["rej_motion"].sum())} motion, '
              f'{int(rej_scored["rej_turn"].sum())} turn, '
              f'{int(rej_scored["rej_step"].sum())} level step, '
              f'{int(rej_scored["rej_impulse"].sum())} impulse '
              f'(categories overlap)')
    for c in CLASSES:
        n = int((keep['klass'] == c).sum())
        print(f'  {c:<20} {n:>6}  ({100 * n / max(len(keep), 1):>5.1f}% of kept)')

    fig_inventory(ev, FIG_DIR / 'event_inventory.png')
    acc, pad = profiles(ev, series_by_session)
    fig_profiles(acc, pad, FIG_DIR / 'event_profiles.png')

    demo = a.demo or ('S1N1' if 'S1N1' in series_by_session
                      else sorted(series_by_session)[0])
    if demo in series_by_session:
        fig_examples(ev, series_by_session[demo], sess_obj[demo], demo,
                     FIG_DIR / f'event_examples_{demo}.png')
        fig_night(ev, series_by_session[demo], demo,
                  FIG_DIR / f'event_night_{demo}.png')

    print(f'\nTables  -> {REPORT_DIR}')
    print(f'Figures -> {FIG_DIR}')


if __name__ == '__main__':
    main()
