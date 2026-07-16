"""
Per-spindle, onset-level DETECTION-RATE analysis of the low-band (0-3 Hz) CAP
response to sleep spindles.

Known story (established elsewhere):
  * CAP carries NO electrical sigma -> per-spindle sigma AUC ~ 0.50.
  * CAP DOES carry a small low-band (0-3 Hz) mechanical/hemodynamic bump
    time-locked to N2 spindle onsets (ERSP work: ~+0.45 dB, strongest on CH).

This script quantifies HOW OFTEN that bump is detectable per spindle.

Method (reuses spindle_loader + the ERSP baseline-correction idea, but scores
EACH spindle individually instead of session-averaging):
  * For every N2 spindle, take a +/-8 s window, run a short-time spectrogram
    (nperseg=128, noverlap=96 -- identical to spindle_ersp), average power over
    the low band, and form a per-spindle dB = 10*log10(core / baseline), where
    core = |t|<1 s and baseline = |t|>5 s of that SAME spindle's window. This is
    exactly the ERSP contrast that validated the +0.45 dB low-band CH bump, done
    per event rather than averaged.
  * DETECTION RULE: a spindle is "detected" if its core low-band power exceeds
    its own local baseline (per-spindle dB > 0). Under the null this sits at 50%,
    so chance = 0.50 and detection rate > 0.50 = a real low-band bump.
  * Matched controls (N2 timepoints >=3 s from any spindle) are scored the same
    way; their detection fraction is the empirical chance reference (~0.50).
  * Per-spindle effect size = the per-spindle dB itself.

Bands: sigma 11-16 Hz (built-in negative control, must be ~chance for CAP),
low 0-3 Hz (primary -- the validated bump), and a 0.5-3 Hz variant (the env_c band).
Channels: CLE, CRE, CLE-CRE, CH (+ EEG kept for reference). N2 spindles, 12 sessions.

Outputs:
  writeup/figures/spindles/fig_spindle_lowband_detection.png
  analysis/spindles/outputs/spindle_lowband_detection.csv   (per session + pooled)
  analysis/spindles/outputs/spindle_lowband_detection.npz   (per-spindle dB arrays)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles

FS = 100.0
N2_CODE = 2
WIN_HALF = 8.0            # +/- s extracted per event (matches spindle_ersp)
CORE_HALF = 1.0           # |t| < CORE_HALF is the per-event "during spindle" core
BASE_EDGE = 5.0           # |t| > BASE_EDGE is the per-event baseline
NPERSEG = 128
NOVERLAP = 96

# bands
SIGMA = (11.0, 16.0)
LOW_03 = (0.0, 3.0)       # primary low band (the validated 0-3 Hz bump)
LOW_C = (0.5, 3.0)        # 0.5-3 Hz variant (the env_c cardiac band)

CAP_CHANNELS = ['CLE', 'CRE', 'CLE-CRE', 'CH']
ALL_CHANNELS = CAP_CHANNELS + ['EEG']

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
FIGDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'writeup', 'figures', 'spindles')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


def get_channel(s, ch):
    if ch == 'EEG':
        return s.psg['EEG'].astype(np.float64)
    if ch == 'CLE-CRE':
        return s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    return s.cap[ch].astype(np.float64)


def stage_at(t_hr, prof):
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(t_hr), -1, np.int8)
    for i, t in enumerate(t_hr):
        j = np.argmin(np.abs(tep - t))
        if abs(tep[j] - t) < 30.0 / 3600.0:
            out[i] = codes[j]
    return out


def channel_event_metrics(sig, centers_samp, win_samp, bands, want_trace=None):
    """Per-event spectrogram band-power dB (core vs that event's own baseline).

    For every event: short-time spectrogram of the +/-win window, average power
    over each band, then dB = 10*log10(mean core power / mean baseline power)
    with core = |t|<CORE_HALF and baseline = |t|>BASE_EDGE. This is the ERSP
    contrast applied per event. Returns {band: np.array of per-event dB}. If
    want_trace is a band name, also returns the mean baseline-corrected dB(t)
    curve for that band (for the onset-triggered figure panel).
    """
    n = len(sig)
    per_band = {b: [] for b in bands}
    trace_acc = None
    trace_k = 0
    tcen = None
    fmask = {}
    core_t = base_t = None

    for c in centers_samp:
        a, b = c - win_samp, c + win_samp + 1
        if a < 0 or b > n:
            continue
        f, t, Sxx = spectrogram(sig[a:b], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        dB = 10.0 * np.log10(Sxx + 1e-12)             # dB per freq bin (as in spindle_ersp)
        if tcen is None:
            tcen = t - t[-1] / 2.0
            core_t = np.abs(tcen) < CORE_HALF
            base_t = np.abs(tcen) > BASE_EDGE
            for bn, (lo, hi) in bands.items():
                fmask[bn] = (f >= lo) & (f <= hi)
        for bn in bands:
            band_dB = dB[fmask[bn]].mean(axis=0)       # mean over band freq bins, in dB, vs time
            base = band_dB[base_t].mean()
            per_band[bn].append(band_dB[core_t].mean() - base)   # core minus own baseline (dB)
            if bn == want_trace:
                curve = band_dB - base
                trace_acc = curve if trace_acc is None else trace_acc + curve
                trace_k += 1
    out = {bn: np.array(v) for bn, v in per_band.items()}
    trace = (trace_acc / trace_k) if trace_k else None
    return out, trace, tcen


def run_session(idx, rng):
    meta = SESSION_META[idx]
    s = load_session(idx)
    s.sleep_profile = load_sleep_profile(s)
    if s.sleep_profile is None:
        return None
    sp = load_spindles(s)
    if sp is None:
        return None

    stg = stage_at(sp['center_hr'], s.sleep_profile)
    n2 = stg == N2_CODE
    cen_hr = sp['center_hr'][n2]
    if len(cen_hr) < 20:
        return None
    cen_samp = np.round(cen_hr * 3600.0 * FS).astype(int)

    # matched control samples: N2 epoch timepoints >=3 s from any spindle center
    prof = s.sleep_profile
    n2_starts = prof['t_ep_hr'][prof['codes'] == N2_CODE]
    cand = []
    for t0 in n2_starts:
        for frac in (0.25, 0.5, 0.75):
            cand.append(t0 + frac * 30.0 / 3600.0)
    cand = np.array(cand)
    if len(cand):
        d = np.min(np.abs(cand[:, None] - sp['center_hr'][None, :]), axis=1) * 3600.0
        cand = cand[d >= 3.0]
    ctrl_samp = np.round(cand * 3600.0 * FS).astype(int)
    if len(ctrl_samp) < 10:
        return None

    win = int(WIN_HALF * FS)
    bands = {'sigma': SIGMA, 'low_c': LOW_C, 'low_03': LOW_03}

    per_channel = {}   # ch -> dict of band -> metrics
    trig_low = {}      # ch -> onset-triggered low_03 band-power dB curve
    t_axis = None
    for ch in ALL_CHANNELS:
        sig = get_channel(s, ch)
        per_channel[ch] = {}
        db_e, trace, tcen = channel_event_metrics(sig, cen_samp, win, bands,
                                                   want_trace='low_03')
        db_c, _, _ = channel_event_metrics(sig, ctrl_samp, win, bands)
        trig_low[ch] = trace
        if t_axis is None:
            t_axis = tcen
        for bname in bands:
            de = db_e[bname][np.isfinite(db_e[bname])]
            dc = db_c[bname][np.isfinite(db_c[bname])]
            if len(de) < 5 or len(dc) < 5:
                per_channel[ch][bname] = None
                continue
            # DETECTION RULE: spindle detected if its core band power exceeds its
            # own local baseline (per-spindle dB > 0).
            per_channel[ch][bname] = {
                'det_rate': float(np.mean(de > 0)),
                'mean_db': float(np.mean(de)),
                'median_db': float(np.median(de)),
                'null_rate': float(np.mean(dc > 0)),   # control detection, ~0.5 sanity
                'n_spindles': int(len(de)),
                'db_per_spindle': de,      # kept for pooled distribution
            }

    return {
        'label': meta['label'], 'subject': meta['subject'],
        'n_spindles_N2': int(len(cen_hr)), 'n_controls': int(len(ctrl_samp)),
        'per_channel': per_channel, 'trig_low': trig_low,
        't_axis': t_axis,
    }


def main():
    rng = np.random.default_rng(42)
    sessions = []
    for idx in range(len(SESSION_META)):
        try:
            res = run_session(idx, rng)
        except Exception as e:
            print(f'[{idx}] FAILED: {e}')
            continue
        if res is None:
            print(f'[{idx}] skipped')
            continue
        sessions.append(res)
        cinfo = res['per_channel']['CH']['low_03']
        print(f"{res['label']}: n_N2={res['n_spindles_N2']:4d}  "
              f"CH low-band det={cinfo['det_rate']:.3f}  mean={cinfo['mean_db']:+.3f} dB")

    # tag map: 'low' = primary 0-3 Hz (low_03); 'lowc' = 0.5-3 Hz (low_c); 'sigma'
    TAGS = [('low_03', 'low'), ('low_c', 'lowc'), ('sigma', 'sigma')]

    # ---- per-session table + pooled ----
    rows = []
    pooled_db = {ch: {'low_03': [], 'sigma': []} for ch in ALL_CHANNELS}
    for res in sessions:
        row = {'session': res['label'], 'subject': res['subject'],
               'n_spindles_N2': res['n_spindles_N2']}
        for ch in ALL_CHANNELS:
            pc = res['per_channel'][ch]
            for bname, tag in TAGS:
                m = pc.get(bname)
                if m is None:
                    row[f'{ch}_{tag}_detrate'] = np.nan
                    row[f'{ch}_{tag}_meandB'] = np.nan
                    continue
                row[f'{ch}_{tag}_detrate'] = m['det_rate']
                row[f'{ch}_{tag}_meandB'] = m['mean_db']
                if bname in ('low_03', 'sigma'):
                    pooled_db[ch][bname].append(m['db_per_spindle'])
            # empirical null (primary low band) for reference
            m = pc.get('low_03')
            row[f'{ch}_low_nullrate'] = m['null_rate'] if m else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)

    # pooled row (spindle-weighted mean of detection rate; mean dB over all spindles)
    pooled = {'session': 'POOLED', 'subject': 'ALL',
              'n_spindles_N2': int(df['n_spindles_N2'].sum())}
    for ch in ALL_CHANNELS:
        for bname, tag in TAGS:
            num = den = 0.0
            for res in sessions:
                m = res['per_channel'][ch].get(bname)
                if m is None:
                    continue
                num += m['det_rate'] * m['n_spindles']
                den += m['n_spindles']
            pooled[f'{ch}_{tag}_detrate'] = num / den if den else np.nan
        for bname, tag in [('low_03', 'low'), ('sigma', 'sigma')]:
            alldb = np.concatenate(pooled_db[ch][bname]) if pooled_db[ch][bname] else np.array([])
            pooled[f'{ch}_{tag}_meandB'] = float(np.mean(alldb)) if len(alldb) else np.nan
        pooled[f'{ch}_low_nullrate'] = float(np.nanmean(df[f'{ch}_low_nullrate']))
    df = pd.concat([df, pd.DataFrame([pooled])], ignore_index=True)

    csv_path = os.path.join(OUT, 'spindle_lowband_detection.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}')

    # ---- save per-spindle dB arrays + triggered averages ----
    save = {}
    for ch in ALL_CHANNELS:
        for bname in ('low_03', 'sigma'):
            arr = np.concatenate(pooled_db[ch][bname]) if pooled_db[ch][bname] else np.array([])
            save[f'db_{ch}_{bname}'] = arr
    t_axis = sessions[0]['t_axis']
    save['t_axis'] = t_axis
    for ch in ALL_CHANNELS:
        save[f'trig_low_{ch}'] = np.array([res['trig_low'][ch] for res in sessions])
    npz_path = os.path.join(OUT, 'spindle_lowband_detection.npz')
    np.savez(npz_path, **save)
    print(f'Wrote {npz_path}')

    make_figure(df, sessions, save)
    return df


def make_figure(df, sessions, save):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sess_df = df[df['session'] != 'POOLED'].copy()
    pooled = df[df['session'] == 'POOLED'].iloc[0]
    colors = {'CLE': '#4C72B0', 'CRE': '#55A868', 'CLE-CRE': '#C44E52', 'CH': '#8172B3'}

    fig = plt.figure(figsize=(15, 5.2), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.32)

    # --- Panel A: per-session low-band detection rate bars, per channel ---
    axA = fig.add_subplot(gs[0, 0])
    labels = sess_df['session'].tolist()
    x = np.arange(len(labels))
    w = 0.2
    for i, ch in enumerate(CAP_CHANNELS):
        vals = sess_df[f'{ch}_low_detrate'].values * 100
        axA.bar(x + (i - 1.5) * w, vals, w, label=ch, color=colors[ch],
                edgecolor='none', alpha=0.9)
    axA.axhline(50, color='k', ls='--', lw=1.2, label='chance (50%)')
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    axA.set_ylabel('Low-band (0-3 Hz) detection rate (%)')
    axA.set_title('A  Per-session low-band spindle detection rate', fontsize=10, loc='left')
    axA.set_ylim(40, max(62, sess_df[[f'{c}_low_detrate' for c in CAP_CHANNELS]].max().max() * 100 + 3))
    axA.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9)
    axA.grid(axis='y', alpha=0.25)

    # --- Panel B: per-spindle effect-size distributions, sigma vs low-band (CH) ---
    axB = fig.add_subplot(gs[0, 1])
    db_low = save['db_CH_low_03']
    db_sig = save['db_CH_sigma']
    bins = np.linspace(-6, 6, 61)
    axB.hist(db_sig, bins=bins, density=True, color='#999999', alpha=0.75,
             label=f'sigma 11-16 Hz\n(mean {np.mean(db_sig):+.2f} dB, det {np.mean(db_sig>0)*100:.0f}%)')
    axB.hist(db_low, bins=bins, density=True, color=colors['CH'], alpha=0.6,
             label=f'low 0-3 Hz\n(mean {np.mean(db_low):+.2f} dB, det {np.mean(db_low>0)*100:.0f}%)')
    axB.axvline(0, color='k', ls='--', lw=1.2)
    axB.axvline(np.mean(db_low), color=colors['CH'], lw=2)
    axB.axvline(np.mean(db_sig), color='#555555', lw=2)
    axB.set_xlabel('Per-spindle effect size (dB, core vs own baseline)')
    axB.set_ylabel('density')
    axB.set_title('B  Per-spindle effect: CH low-band vs sigma', fontsize=10, loc='left')
    axB.legend(fontsize=7.5, loc='upper left')
    axB.set_xlim(-6, 6)

    # --- Panel C: onset-triggered low-band average, all CAP channels (CH bold) ---
    axC = fig.add_subplot(gs[0, 2])
    t = save['t_axis']
    for ch in CAP_CHANNELS:
        curves = save[f'trig_low_{ch}']              # (sessions, T)
        m = curves.mean(axis=0)
        se = curves.std(axis=0) / np.sqrt(curves.shape[0])
        lw = 2.4 if ch == 'CH' else 1.2
        alpha = 1.0 if ch == 'CH' else 0.6
        axC.plot(t, m, color=colors[ch], lw=lw, alpha=alpha, label=ch)
        if ch == 'CH':
            axC.fill_between(t, m - se, m + se, color=colors[ch], alpha=0.2)
    axC.axvline(0, color='k', ls='--', lw=1.0)
    axC.set_xlim(-6, 6)
    axC.set_xlabel('time from spindle center (s)')
    axC.set_ylabel('0-3 Hz power (dB re baseline)')
    axC.set_title('C  Onset-triggered low-band average', fontsize=10, loc='left')
    axC.legend(fontsize=7.5, loc='upper right')
    axC.grid(alpha=0.25)

    ch_low = pooled['CH_low_detrate'] * 100
    ch_sig = pooled['CH_sigma_detrate'] * 100
    fig.suptitle(
        f'Per-spindle low-band (0-3 Hz) CAP detection of N2 sleep spindles  |  '
        f'CH pooled: low-band {ch_low:.1f}% vs sigma {ch_sig:.1f}% (chance 50%)',
        fontsize=11, y=1.02)
    fig_path = os.path.join(FIGDIR, 'fig_spindle_lowband_detection.png')
    fig.savefig(fig_path, bbox_inches='tight', dpi=200)
    print(f'Wrote {fig_path}')


if __name__ == '__main__':
    main()
