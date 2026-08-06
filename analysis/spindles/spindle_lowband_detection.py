"""
Per-spindle DETECTION-RATE analysis of the low-band (0.1-3 Hz) CAP
response to sleep spindles.

Known story (established elsewhere):
  * CAP carries NO electrical sigma -> per-spindle sigma AUC ~ 0.50.
  * CAP DOES carry a small low-band (0.1-3 Hz) mechanical/hemodynamic bump
    time-locked to N2 spindle centers (ERSP work: ~+0.5 dB, strongest on CH),
    but it is heavy-tailed: the mean is several times the median.

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
low 0.1-3 Hz (primary -- the validated bump), and a 0.5-3 Hz variant (the env_c band).
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
from scipy.stats import trim_mean

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
# Primary low band. Lower edge is 0.1 Hz, not 0, so the f=0 STFT bin is excluded:
# after per-segment mean removal that bin holds only the residual within-window
# drift, which is a windowing/detrending quantity rather than a low-band oscillation.
# At nperseg=128 (0.78 Hz bins) this selects the 0.78/1.56/2.34 Hz bins.
LOW_03 = (0.1, 3.0)
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
    curve for that band (for the center-triggered figure panel).
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
            # Event sits WIN_HALF seconds into the epoch; t[-1]/2 is one STFT
            # hop short of that and shifts the whole axis 0.32 s early.
            tcen = t - WIN_HALF
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
    trig_low = {}      # ch -> center-triggered low_03 band-power dB curve
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
            # The per-event dB distribution is heavy-tailed: a small minority of
            # events carries most of the mean. Report the mean alongside robust
            # statistics so the two can be compared directly.
            per_channel[ch][bname] = {
                'det_rate': float(np.mean(de > 0)),
                'mean_db': float(np.mean(de)),
                'median_db': float(np.median(de)),
                'trim_db': float(trim_mean(de, 0.1)),   # 10% trimmed each tail
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
                    row[f'{ch}_{tag}_mediandB'] = np.nan
                    row[f'{ch}_{tag}_trimdB'] = np.nan
                    continue
                row[f'{ch}_{tag}_detrate'] = m['det_rate']
                row[f'{ch}_{tag}_meandB'] = m['mean_db']
                row[f'{ch}_{tag}_mediandB'] = m['median_db']
                row[f'{ch}_{tag}_trimdB'] = m['trim_db']
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
            if not len(alldb):
                for k in ('meandB', 'mediandB', 'trimdB', 'semdB', 'ci_lo', 'ci_hi',
                          'top5pct_share'):
                    pooled[f'{ch}_{tag}_{k}'] = np.nan
                continue
            sem = float(alldb.std(ddof=1) / np.sqrt(len(alldb)))
            pooled[f'{ch}_{tag}_meandB'] = float(alldb.mean())
            pooled[f'{ch}_{tag}_mediandB'] = float(np.median(alldb))
            pooled[f'{ch}_{tag}_trimdB'] = float(trim_mean(alldb, 0.1))
            pooled[f'{ch}_{tag}_semdB'] = sem
            # 95% CI on the trial-averaged effect. For the sigma band this is the
            # quantity the manuscript should quote: an upper bound on any
            # spindle-locked change, not a claim that the change is exactly zero.
            pooled[f'{ch}_{tag}_ci_lo'] = float(alldb.mean() - 1.96 * sem)
            pooled[f'{ch}_{tag}_ci_hi'] = float(alldb.mean() + 1.96 * sem)
            # Fraction of the summed effect carried by the strongest 5% of events.
            # >1 means the remaining 95% sum negative, i.e. the mean is tail-driven.
            top = np.sort(alldb)[::-1][:max(1, len(alldb) // 20)]
            pooled[f'{ch}_{tag}_top5pct_share'] = (float(top.sum() / alldb.sum())
                                                   if alldb.sum() != 0 else np.nan)
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

    # ---- manuscript-ready summary ----
    p = df[df['session'] == 'POOLED'].iloc[0]
    ss = df[df['session'] != 'POOLED']
    print(f"\n=== pooled over {int(p['n_spindles_N2']):,} N2 spindles ===")
    print(f"{'ch':8} {'low mean':>9} {'median':>8} {'trim10':>8} {'det':>7} {'null':>7} "
          f"{'top5%share':>11} {'<0.5 sessions':>14}")
    for ch in ALL_CHANNELS:
        below = int((ss[f'{ch}_low_detrate'] < 0.5).sum())
        print(f"{ch:8} {p[f'{ch}_low_meandB']:+9.3f} {p[f'{ch}_low_mediandB']:+8.3f} "
              f"{p[f'{ch}_low_trimdB']:+8.3f} {p[f'{ch}_low_detrate']:7.3f} "
              f"{p[f'{ch}_low_nullrate']:7.3f} {p[f'{ch}_low_top5pct_share']:11.2f} "
              f"{below:>10d}/12")
    print('\nSigma band — quote as a bound, not a null:')
    for ch in ALL_CHANNELS:
        m, se = p[f'{ch}_sigma_meandB'], p[f'{ch}_sigma_semdB']
        bound = abs(m) + 1.96 * se
        print(f"  {ch:8} mean {m:+.4f} dB  95% CI [{p[f'{ch}_sigma_ci_lo']:+.4f}, "
              f"{p[f'{ch}_sigma_ci_hi']:+.4f}]  ->  |effect| < {bound:.3f} dB "
              f"({100*(10**(bound/10)-1):.1f}% power)")

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
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.3], wspace=0.32)

    # --- Panel A: onset-triggered low-band average, all CAP channels (CH bold) ---
    axA = fig.add_subplot(gs[0, 0])
    t = save['t_axis']
    for ch in CAP_CHANNELS:
        curves = save[f'trig_low_{ch}']              # (sessions, T)
        m = curves.mean(axis=0)
        se = curves.std(axis=0) / np.sqrt(curves.shape[0])
        lw = 2.4 if ch == 'CH' else 1.2
        alpha = 1.0 if ch == 'CH' else 0.6
        axA.plot(t, m, color=colors[ch], lw=lw, alpha=alpha, label=ch)
        if ch == 'CH':
            axA.fill_between(t, m - se, m + se, color=colors[ch], alpha=0.2)
    axA.axvline(0, color='k', ls='--', lw=1.0)
    axA.set_xlim(-6, 6)
    axA.set_xlabel('time from spindle center (s)')
    axA.set_ylabel('0.1-3 Hz power (dB re baseline)')
    axA.set_title('A  Center-triggered low-band average (all channels)', fontsize=10, loc='left')
    axA.legend(fontsize=7.5, loc='upper right')
    axA.grid(alpha=0.25)

    # --- Panel B: per-spindle effect-size distributions, low-band vs sigma (CH) ---
    axB = fig.add_subplot(gs[0, 1])
    db_low = save['db_CH_low_03']
    db_sig = save['db_CH_sigma']
    bins = np.linspace(-6, 6, 61)
    axB.hist(db_sig, bins=bins, density=True, color='#999999', alpha=0.75,
             label=f'sigma 11-16 Hz\n(mean {np.mean(db_sig):+.2f}, '
                   f'median {np.median(db_sig):+.2f} dB)')
    axB.hist(db_low, bins=bins, density=True, color=colors['CH'], alpha=0.6,
             label=f'low 0.1-3 Hz\n(mean {np.mean(db_low):+.2f}, '
                   f'median {np.median(db_low):+.2f} dB)')
    axB.axvline(0, color='k', ls='--', lw=1.2)
    # Solid = mean, dashed = median. The gap between them is the point: the
    # distribution is heavy-tailed, so the mean overstates the typical spindle.
    axB.axvline(np.mean(db_low), color=colors['CH'], lw=2)
    axB.axvline(np.median(db_low), color=colors['CH'], lw=2, ls=':')
    axB.axvline(np.mean(db_sig), color='#555555', lw=2)
    axB.axvline(np.median(db_sig), color='#555555', lw=2, ls=':')
    det = 100 * np.mean(db_low > 0)
    axB.set_xlabel('Per-spindle power change (dB, core vs own baseline)\n'
                   'vertical lines: solid = mean, dotted = median')
    axB.set_ylabel('density')
    axB.set_title('B  Low-band vs sigma at spindles (CH)', fontsize=10, loc='left')
    axB.legend(fontsize=7.5, loc='upper left')
    axB.set_xlim(-6, 6)
    axB.text(0.98, 0.55, f'{det:.1f}% of spindles exceed\ntheir own baseline\n(chance 50%)',
             transform=axB.transAxes, ha='right', va='top', fontsize=7.5, zorder=6,
             bbox=dict(boxstyle='round', fc='white', ec='#bbbbbb', alpha=0.95))

    # --- Panel C: per-session low-band power change per channel ---
    axC = fig.add_subplot(gs[0, 2])
    labels = sess_df['session'].tolist()
    x = np.arange(len(labels))
    w = 0.2
    for i, ch in enumerate(CAP_CHANNELS):
        vals = sess_df[f'{ch}_low_meandB'].values
        axC.bar(x + (i - 1.5) * w, vals, w, label=f'{ch} (mean)', color=colors[ch],
                edgecolor='none', alpha=0.9)
        # Per-session median overlaid as a tick, so the tail-driven gap between
        # mean and median is visible night by night, not just in the pooled panel.
        med = sess_df[f'{ch}_low_mediandB'].values
        axC.scatter(x + (i - 1.5) * w, med, marker='_', s=45, lw=1.4,
                    color='k', zorder=4)
    axC.scatter([], [], marker='_', s=45, lw=1.4, color='k', label='median')
    axC.axhline(0, color='k', ls='-', lw=0.8)
    axC.set_xticks(x)
    axC.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    axC.set_ylabel('Low-band (0.1-3 Hz) power change (dB)')
    axC.set_title('C  Per-session low-band increase at spindles', fontsize=10, loc='left')
    axC.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9)
    axC.grid(axis='y', alpha=0.25)

    fig.suptitle(
        f'Low-band (0.1-3 Hz) capacitive response to N2 sleep spindles  |  CH pooled '
        f'mean {pooled["CH_low_meandB"]:+.2f} dB, median {pooled["CH_low_mediandB"]:+.2f} dB, '
        f'{100*pooled["CH_low_detrate"]:.1f}% of spindles above own baseline  |  '
        f'sigma band bounded at |{abs(pooled["CH_sigma_meandB"]) + 1.96*pooled["CH_sigma_semdB"]:.2f}| dB',
        fontsize=10.5, y=1.02)
    fig_path = os.path.join(FIGDIR, 'fig_spindle_lowband_detection.png')
    fig.savefig(fig_path, bbox_inches='tight', dpi=200)
    print(f'Wrote {fig_path}')


if __name__ == '__main__':
    import sys
    if '--figure-only' in sys.argv:
        # Rebuild the figure from saved CSV + NPZ without re-running the
        # per-session analysis (used to re-render after a figure reframe).
        import pandas as pd
        df = pd.read_csv(os.path.join(OUT, 'spindle_lowband_detection.csv'))
        save = np.load(os.path.join(OUT, 'spindle_lowband_detection.npz'))
        make_figure(df, None, save)
    else:
        main()
