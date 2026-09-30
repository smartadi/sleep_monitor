"""
Ridge and harmonic-ladder detection on the optical control session.

Why this exists
---------------
Every ridge / harmonic-ladder number we have so far comes from the capacitive
mask alone, scored against PSG.  The control session of 2026-09-11 is the first
recording where a *second*, independent sensor sat on the same head on the same
clock: a Muse S, whose `Optics1-4` channels carry a textbook pulsatile
photoplethysmogram.  The shipped sync report already notes a sharp optical
cardiac line at 1.083 Hz with visible 1st and 2nd harmonics, and *no* resolvable
cardiac line in CH / CLE / CRE over the same window.

That makes this session a positive control for the detector itself, not for the
mask: we know a real harmonic ladder is present in one modality, so

  1. the detector must find it in the optics (if it does not, the detector is
     broken, and every negative CAP result is uninterpretable);
  2. whatever the detector then reports for the CAP channels over the *same*
     windows is measured against a ladder we know exists in the room.

So this runs the unmodified pipeline -- `detect_persistent_ridges`,
`label_harmonic_ladder_windows`, `detect_harmonics` -- with the same parameters
`analysis/slow_wave/band_ridge_analysis.py` uses for the 12 overnight sessions,
across three modalities on one 20 Hz wall-clock grid:

  CAP   CH, CLE, CRE          mask, 111.11 Hz  (from the shipped clock-corrected export)
  OPT   Optics1-4             Muse, ~21.4 Hz true update rate
  EEG   RAW_AF7, RAW_AF8      Muse, 256 Hz     (context; TP9/TP10 are noisier in this rig)

AUX_1-4 are excluded: the sync report established they are floating analog
inputs (flat DC plus a comb of fixed tones), not physiology.

There is no PSG here, so there are no stage statistics.  The reference is the
Muse's own reported `Heart_Rate`, which is independent of our spectral estimate.

Bands are the pipeline's, with one addition: `card_ext` raises the cardiac
ceiling from 3.0 to 5.0 Hz so a 1.08 Hz fundamental can show three harmonics
inside one detector run -- a ladder cannot be found across two separate
band-restricted runs.

Outputs -> reports/slow_wave/optical_control/
  ridges.csv            one row per detected persistent ridge
  windows.csv           per 30 s window x channel x band: ridge + ladder features
  harmonics.csv         per 30 s window x channel: HPS / cepstral / explicit traces
  summary.csv           per channel x band headline numbers
  common_20hz.parquet   cached preprocessed grid (delete to force a reload)
-> writeup/figures/optical_control/
  fig_ridges.png        spectrogram + ridge overlay, all channels
  fig_ladders.png       ladder f0 / harmonic count traces vs Muse Heart_Rate
  fig_psd.png           motion-free median PSD per channel with harmonic markers

Run:
  python optical_control_ridges.py            # full, uses the cache if present
  python optical_control_ridges.py --reload   # rebuild the 20 Hz grid from raw
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import (
    detect_persistent_ridges,
    label_harmonic_ladder_windows,
    detect_harmonics,
)

# ── Paths ────────────────────────────────────────────────────────────────────
SESSION_DIR = ROOT / 'writeup' / 'optical test data' / 'optical_sensor_control_test'
SYNC_DIR = SESSION_DIR / 'optical_control_sync'
MASK_CSV = SYNC_DIR / 'mask_realtime_111Hz.csv.gz'          # clock-corrected export
MUSE_CSV = SESSION_DIR / 'mindMonitor_2026-09-11--02-07-30.csv'

OUT = ROOT / 'reports' / 'slow_wave' / 'optical_control'
FIG = ROOT / 'writeup' / 'figures' / 'optical_control'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ── Grid ─────────────────────────────────────────────────────────────────────
# 20 Hz is the optics' own working rate in the sync report (true update 21.44 Hz)
# and leaves 10 Hz of headroom above the 5 Hz band ceiling.  Both other streams
# decimate onto it with exact integer ratios: 111.11 x 9/50 and 256 x 5/64.
FS = 20.0
FS_MASK = 1000.0 / 9.0
FS_MUSE = 256.0
SETTLE_SEC = 120.0        # drop the tap burst and filter settling, as fig 10 does

CAP_CH = ['CH', 'CLE', 'CRE']
OPT_CH = ['Optics1', 'Optics2', 'Optics3', 'Optics4']
EEG_CH = ['RAW_AF7', 'RAW_AF8']
CHANNELS = CAP_CH + OPT_CH + EEG_CH
MODALITY = ({c: 'CAP' for c in CAP_CH} | {c: 'OPT' for c in OPT_CH}
            | {c: 'EEG' for c in EEG_CH})
MOD_COLOR = {'CAP': '#8E44AD', 'OPT': '#E67E22', 'EEG': '#2980B9'}

# ── Detector parameters (identical to band_ridge_analysis.py) ────────────────
BANDS = {
    'resp': dict(min_freq=0.1, max_freq=0.5, welch_seg_sec=30.0,
                 max_freq_jump=0.05, peak_prominence_frac=0.3),
    'card': dict(min_freq=0.5, max_freq=3.0, welch_seg_sec=8.0,
                 max_freq_jump=0.10, peak_prominence_frac=0.5),
    # cardiac parametrisation with the ceiling raised so f0 + 3 harmonics fit
    # inside one run; a ladder cannot form across two band-restricted runs.
    'card_ext': dict(min_freq=0.5, max_freq=5.0, welch_seg_sec=8.0,
                     max_freq_jump=0.10, peak_prominence_frac=0.5),
}
BAND_LABEL = {'resp': 'Respiratory 0.1-0.5 Hz',
              'card': 'Cardiac 0.5-3.0 Hz',
              'card_ext': 'Cardiac extended 0.5-5.0 Hz'}

WIN_SEC = 30.0
STEP_SEC = 30.0
SMOOTH_WINDOWS = 7
MIN_PERSIST_SEC = 300.0
MAX_GAP_WINDOWS = 5

PREP_BAND = (0.05, 4.0)      # band_ridge_analysis pre-filters here before detection
PREP_BAND_EXT = (0.05, 8.0)  # card_ext needs headroom above its 5 Hz ceiling


# ── Loading ──────────────────────────────────────────────────────────────────

def _uniform_from_updates(t_ns, values, grid_ns):
    """Interpolate a forward-filled stream onto `grid_ns` using only the instants
    where it actually changed.  Mind Monitor repeats the previous value on every
    EEG row, so the raw column looks like 256 Hz while the sensor updates at
    ~21 Hz; taking every row would fabricate a step-hold spectrum."""
    v = np.asarray(values, dtype=float)
    ok = np.all(np.isfinite(v), axis=1) if v.ndim == 2 else np.isfinite(v)
    v, t = v[ok], t_ns[ok]
    if v.ndim == 1:
        v = v[:, None]
    chg = np.r_[True, np.any(v[1:] != v[:-1], axis=1)]
    t_u, v_u = t[chg].astype(float), v[chg]
    out = np.column_stack([np.interp(grid_ns.astype(float), t_u, v_u[:, i])
                           for i in range(v_u.shape[1])])
    rate = chg.sum() / ((t_u[-1] - t_u[0]) / 1e9)
    return out, float(rate)


def build_common_grid():
    """Both devices onto one 20 Hz wall-clock grid over their overlap."""
    print('Loading mask (111.11 Hz, clock-corrected export)...', flush=True)
    m = pd.read_csv(MASK_CSV, usecols=['time', 'CH', 'CLE', 'CRE', 'aX', 'aY', 'aZ'])
    m['time'] = pd.to_datetime(m['time'])
    print(f'  {len(m):,} samples, {m["time"].iloc[0]} -> {m["time"].iloc[-1]}')

    print('Loading Muse...', flush=True)
    use = (['TimeStamp'] + EEG_CH + OPT_CH
           + [f'Accelerometer_{c}' for c in 'XYZ'] + ['Heart_Rate'])
    mu = pd.read_csv(MUSE_CSV, usecols=use, low_memory=False)
    mu['time'] = pd.to_datetime(mu['TimeStamp'], format='mixed')
    mu = mu.sort_values('time', kind='stable').reset_index(drop=True)
    print(f'  {len(mu):,} rows, {mu["time"].iloc[0]} -> {mu["time"].iloc[-1]}')

    t0 = max(m['time'].iloc[0], mu['time'].iloc[0]) + pd.Timedelta(seconds=SETTLE_SEC)
    t1 = min(m['time'].iloc[-1], mu['time'].iloc[-1])
    print(f'Overlap (after {SETTLE_SEC:.0f}s settle): {t0} -> {t1} '
          f'= {(t1 - t0).total_seconds() / 60:.1f} min')

    grid = pd.date_range(t0, t1, freq=pd.Timedelta(seconds=1.0 / FS))
    grid_ns = grid.view('int64')
    out = pd.DataFrame(index=grid)

    # -- mask: uniform 9 ms tick, so anti-alias + decimate exactly 111.11 -> 20
    ms = m[(m['time'] >= t0 - pd.Timedelta(seconds=5))
           & (m['time'] <= t1 + pd.Timedelta(seconds=5))].reset_index(drop=True)
    mt0 = ms['time'].iloc[0].value
    for c in CAP_CH + ['aX', 'aY', 'aZ']:
        d = resample_poly(ms[c].to_numpy(float), up=9, down=50)
        dt = mt0 + (np.arange(len(d)) / FS * 1e9)
        out[c] = np.interp(grid_ns.astype(float), dt, d)
    out['cap_acc_mag'] = np.sqrt(out['aX'] ** 2 + out['aY'] ** 2 + out['aZ'] ** 2)

    # -- optics + Muse accelerometer + HR: forward-filled, use change instants
    mut = mu['time'].to_numpy().astype('datetime64[ns]').astype('int64')
    o, rate = _uniform_from_updates(mut, mu[OPT_CH].to_numpy(float), grid_ns)
    print(f'  optics true update rate: {rate:.2f} Hz')
    for i, c in enumerate(OPT_CH):
        out[c] = o[:, i]
    a, arate = _uniform_from_updates(
        mut, mu[[f'Accelerometer_{c}' for c in 'XYZ']].to_numpy(float), grid_ns)
    print(f'  Muse accelerometer update rate: {arate:.2f} Hz')
    out['muse_acc_mag'] = np.sqrt((a ** 2).sum(axis=1))
    hr, _ = _uniform_from_updates(mut, mu['Heart_Rate'].to_numpy(float), grid_ns)
    out['muse_hr'] = hr[:, 0]

    # -- EEG: rows are packet-stamped but arrive at a steady 256 Hz row rate;
    #    treat as uniform (the sync report's own convention) and decimate 256->20.
    es = mu[(mu['time'] >= t0 - pd.Timedelta(seconds=5))
            & (mu['time'] <= t1 + pd.Timedelta(seconds=5))].reset_index(drop=True)
    et0 = es['time'].iloc[0].value
    for c in EEG_CH:
        x = es[c].to_numpy(float)
        x = np.nan_to_num(x, nan=np.nanmedian(x))
        d = resample_poly(x, up=5, down=64)
        dt = et0 + (np.arange(len(d)) / FS * 1e9)
        out[c] = np.interp(grid_ns.astype(float), dt, d)

    out.index.name = 'time'
    return out


def load_grid(reload: bool):
    cache = OUT / 'common_20hz.parquet'
    if cache.exists() and not reload:
        print(f'Using cached grid {cache}')
        return pd.read_parquet(cache)
    g = build_common_grid()
    g.to_parquet(cache)
    print(f'Cached -> {cache}  ({len(g):,} rows @ {FS:g} Hz)')
    return g


# ── Preprocessing ────────────────────────────────────────────────────────────

def prepare(grid, ch, band_hi):
    """Pipeline preprocessing for one channel, plus the gate accelerometer.

    CAP gets the OLS accelerometer regression the overnight pipeline applies;
    optics and EEG get the same bandpass but no regression, because the mask
    accelerometer is not the motion reference for a device sitting on a
    different strap.  Each modality is gated by its own accelerometer.
    """
    acc = grid['cap_acc_mag'].to_numpy() if MODALITY[ch] == 'CAP' \
        else grid['muse_acc_mag'].to_numpy()
    x = grid[ch].to_numpy(float)
    if MODALITY[ch] == 'CAP':
        sig = remove_acc_artifact(x, acc, PREP_BAND[0], band_hi, fs=FS)
    else:
        sig = bandpass(x, PREP_BAND[0], band_hi, FS)
    return sig, acc


def spectrogram(sig, fmax=5.0, seg_sec=8.0):
    """Display spectrogram on the same window geometry as the detector."""
    win_n, step_n = int(WIN_SEC * FS), int(STEP_SEC * FS)
    nperseg = min(int(seg_sec * FS), win_n)
    starts = np.arange(0, len(sig) - win_n + 1, step_n)
    f, _ = welch(sig[:win_n], fs=FS, nperseg=nperseg, noverlap=nperseg // 2,
                 scaling='density')
    keep = f <= fmax
    P = np.array([
        welch(sig[s:s + win_n], fs=FS, nperseg=nperseg, noverlap=nperseg // 2,
              scaling='density')[1][keep]
        for s in starts
    ])
    t_hr = (starts + win_n / 2) / FS / 3600.0
    return t_hr, f[keep], P


# ── Detection ────────────────────────────────────────────────────────────────

def run_detection(grid):
    ridge_rows, window_rows, harm_rows = [], [], []
    runs = {}

    n_win = None
    for ch in CHANNELS:
        print(f'\n--- {ch} ({MODALITY[ch]}) ---', flush=True)
        for band, bp in BANDS.items():
            hi = PREP_BAND_EXT[1] if band == 'card_ext' else PREP_BAND[1]
            sig, acc = prepare(grid, ch, hi)
            rr = detect_persistent_ridges(
                sig, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC,
                min_freq=bp['min_freq'], max_freq=bp['max_freq'],
                smooth_windows=SMOOTH_WINDOWS,
                min_persistence_sec=MIN_PERSIST_SEC,
                max_freq_jump=bp['max_freq_jump'],
                peak_prominence_frac=bp['peak_prominence_frac'],
                welch_seg_sec=bp['welch_seg_sec'],
                max_gap_windows=MAX_GAP_WINDOWS,
                acc_mag=acc,
            )
            lad = label_harmonic_ladder_windows(rr, min_harmonics=2, min_f0=0.1)
            runs[(ch, band)] = (rr, lad)
            n_win = len(rr['t_hr'])

            for r in rr['ridges']:
                ridge_rows.append(dict(
                    channel=ch, modality=MODALITY[ch], band=band,
                    median_freq=float(r['median_freq']),
                    duration_min=float(r['duration_sec']) / 60.0,
                    start_hr=float(rr['t_hr'][r['start_idx']]),
                    end_hr=float(rr['t_hr'][r['end_idx']]),
                    n_present=int(r['n_present']),
                ))

            t_hr = rr['t_hr']
            n_active = np.zeros(n_win, dtype=int)
            min_f = np.full(n_win, np.nan)
            tot_p = np.full(n_win, np.nan)
            for i in range(n_win):
                fs_i = [r['freq_trace'][i] for r in rr['ridges']
                        if np.isfinite(r['freq_trace'][i])]
                as_i = [r['amp_trace'][i] for r in rr['ridges']
                        if np.isfinite(r['freq_trace'][i])]
                n_active[i] = len(fs_i)
                if fs_i:
                    min_f[i] = min(fs_i)
                    tot_p[i] = float(np.nansum(as_i))
            window_rows.append(pd.DataFrame(dict(
                channel=ch, modality=MODALITY[ch], band=band,
                t_hr=t_hr, motion_masked=rr['motion_mask'],
                n_ridges=n_active, min_ridge_freq=min_f, total_ridge_power=tot_p,
                ridge_present=(n_active > 0).astype(int),
                is_ladder=lad['is_ladder'], ladder_f0=lad['ladder_f0'],
                ladder_n=lad['ladder_n'], ladder_power=lad['ladder_power'],
            )))
            print(f'  {band:<9} {len(rr["ridges"]):3d} ridges, '
                  f'{len(rr["harmonic_groups"])} harmonic groups, '
                  f'ladder in {int(lad["is_ladder"].sum())}/{n_win} windows',
                  flush=True)

        # per-window 3-method harmonic detector, cardiac f0 range
        sig, acc = prepare(grid, ch, PREP_BAND_EXT[1])
        h = detect_harmonics(sig, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC,
                             f0_range=(0.5, 2.0), max_harmonics=4,
                             f_tolerance=0.08, welch_seg_sec=8.0,
                             acc_mag=acc)
        h['channel'] = ch
        h['modality'] = MODALITY[ch]
        harm_rows.append(h)
        ok = h[~h['motion_masked']]
        print(f'  harmonics: median f0={np.nanmedian(ok["f0_hz"]):.3f} Hz, '
              f'median n_harm={np.nanmedian(ok["n_harmonics"]):.1f}, '
              f'median HER={np.nanmedian(ok["harmonic_energy_ratio"]):.3f}',
              flush=True)

    return (pd.DataFrame(ridge_rows),
            pd.concat(window_rows, ignore_index=True),
            pd.concat(harm_rows, ignore_index=True),
            runs)


def window_hr(grid, n_win):
    """Muse Heart_Rate averaged onto the detector's window grid, in Hz."""
    win_n, step_n = int(WIN_SEC * FS), int(STEP_SEC * FS)
    hr = grid['muse_hr'].to_numpy(float)
    starts = np.arange(0, len(hr) - win_n + 1, step_n)[:n_win]
    return np.array([np.nanmedian(hr[s:s + win_n]) for s in starts]) / 60.0


def summarise(windows, harm, hr_hz):
    rows = []
    for (ch, band), g in windows.groupby(['channel', 'band'], sort=False):
        ok = g[~g['motion_masked']]
        lad = ok[ok['is_ladder']]
        # Frequency agreement with the Muse's own HR, on ladder windows.  Only
        # meaningful for the cardiac bands: a respiratory-band ladder has a
        # respiratory fundamental and comparing it to a heart rate is nonsense.
        if len(lad) and band != 'resp':
            idx = lad.index - g.index[0]
            ref = hr_hz[np.clip(idx.to_numpy(), 0, len(hr_hz) - 1)]
            err = lad['ladder_f0'].to_numpy() - ref
            med_err, iqr_err = float(np.nanmedian(err)), float(
                np.nanpercentile(err, 75) - np.nanpercentile(err, 25))
        else:
            med_err = iqr_err = np.nan
        rows.append(dict(
            channel=ch, modality=MODALITY[ch], band=band,
            n_windows=len(g), n_motion_free=len(ok),
            ridge_frac=float(ok['ridge_present'].mean()) if len(ok) else np.nan,
            median_n_ridges=float(ok['n_ridges'].median()) if len(ok) else np.nan,
            ladder_frac=float(ok['is_ladder'].mean()) if len(ok) else np.nan,
            median_ladder_f0=float(lad['ladder_f0'].median()) if len(lad) else np.nan,
            median_ladder_n=float(lad['ladder_n'].median()) if len(lad) else np.nan,
            f0_minus_museHR_median=med_err, f0_minus_museHR_iqr=iqr_err,
        ))
    s = pd.DataFrame(rows)

    hrows = []
    for ch, g in harm.groupby('channel', sort=False):
        ok = g[~g['motion_masked']]
        hrows.append(dict(
            channel=ch, modality=MODALITY[ch],
            median_f0=float(np.nanmedian(ok['f0_hz'])),
            median_n_harmonics=float(np.nanmedian(ok['n_harmonics'])),
            median_harmonic_energy_ratio=float(
                np.nanmedian(ok['harmonic_energy_ratio'])),
            median_cepstral_prominence=float(np.nanmedian(ok['cep_prominence'])),
            median_hps_score=float(np.nanmedian(ok['hps_score'])),
            frac_ge2_harmonics=float(np.nanmean(ok['n_harmonics'] >= 2)),
        ))
    return s, pd.DataFrame(hrows)


def cross_modal(windows, harm):
    """The control's actual question: at the windows where the optics show a
    cardiac ladder, what does the detector report for CAP and EEG?

    This is the sensitivity statement.  A CAP channel that finds nothing in
    windows where a ladder is demonstrably present on the same head is evidence
    about the sensor, not about the detector.
    """
    w = windows[windows['band'] == 'card_ext']
    opt = w[w['channel'].isin(OPT_CH)]
    # a window counts as "ladder present in the room" when >= 3 of the 4 optical
    # channels agree -- they are four views of one pulsatile source
    votes = opt.groupby('t_hr')['is_ladder'].sum()
    clean = opt.groupby('t_hr')['motion_masked'].max()
    ref_t = votes[(votes >= 3) & (~clean.astype(bool))].index.to_numpy()

    rows = []
    for ch in CHANNELS:
        g = w[w['channel'] == ch].set_index('t_hr')
        h = harm[harm['channel'] == ch].set_index('t_hr')
        sel = g.index.isin(ref_t)
        hsel = h.index.isin(ref_t)
        rows.append(dict(
            channel=ch, modality=MODALITY[ch],
            n_reference_windows=int(sel.sum()),
            ladder_frac=float(g.loc[sel, 'is_ladder'].mean()),
            any_cardiac_ridge_frac=float(g.loc[sel, 'ridge_present'].mean()),
            median_n_harmonics=float(np.nanmedian(h.loc[hsel, 'n_harmonics'])),
            median_harmonic_energy_ratio=float(
                np.nanmedian(h.loc[hsel, 'harmonic_energy_ratio'])),
            median_cepstral_prominence=float(
                np.nanmedian(h.loc[hsel, 'cep_prominence'])),
        ))
    return pd.DataFrame(rows)


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_ridges(grid, runs, path):
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(13, 2.0 * len(CHANNELS)),
                             sharex=True)
    for ax, ch in zip(np.atleast_1d(axes), CHANNELS):
        sig, _ = prepare(grid, ch, PREP_BAND_EXT[1])
        t_hr, f, P = spectrogram(sig, fmax=5.0)
        D = 10 * np.log10(np.maximum(P, np.finfo(float).tiny))
        vmin, vmax = np.nanpercentile(D, [10, 99])
        ax.pcolormesh(t_hr, f, D.T, cmap='magma', vmin=vmin, vmax=vmax,
                      shading='auto', rasterized=True)
        for band, style in (('resp', dict(color='#00E5FF', lw=1.3)),
                            ('card_ext', dict(color='#7CFF00', lw=1.3))):
            rr, _ = runs[(ch, band)]
            for r in rr['ridges']:
                ax.plot(rr['t_hr'], r['freq_trace'], alpha=0.9, **style)
        ax.set_ylim(0, 5)
        ax.set_ylabel(f'{ch}\n({MODALITY[ch]})  Hz', fontsize=8,
                      color=MOD_COLOR[MODALITY[ch]])
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel('Time from session start (hr)', fontsize=9)
    axes[0].set_title('Optical control session 2026-09-11 — persistent ridges over '
                      'the spectrogram\ncyan = respiratory band run, '
                      'green = cardiac-extended run', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f'  wrote {path}')


def fig_ladders(windows, hr_hz, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    w = windows[windows['band'] == 'card_ext']
    t_ref = w[w['channel'] == CHANNELS[0]]['t_hr'].to_numpy()

    ax = axes[0]
    ax.plot(t_ref, hr_hz[:len(t_ref)], color='k', lw=2.0, alpha=0.7,
            label='Muse Heart_Rate / 60 (reference)')
    for ch in CHANNELS:
        g = w[w['channel'] == ch]
        lad = g[g['is_ladder'] & ~g['motion_masked']]
        if len(lad) == 0:
            continue
        ax.plot(lad['t_hr'], lad['ladder_f0'], '.', ms=3.5,
                color=MOD_COLOR[MODALITY[ch]], alpha=0.55,
                label=f'{ch} ({MODALITY[ch]})')
    ax.set_ylabel('Ladder fundamental f0 (Hz)', fontsize=9)
    ax.set_ylim(0.4, 2.2)
    ax.legend(fontsize=7, ncol=4, loc='upper right')
    ax.grid(alpha=0.15)
    ax.set_title('Harmonic ladders, cardiac-extended band (0.5-5.0 Hz)', fontsize=11)

    ax = axes[1]
    for i, ch in enumerate(CHANNELS):
        g = w[w['channel'] == ch].sort_values('t_hr')
        y = np.where(g['motion_masked'], np.nan, g['is_ladder'].astype(float))
        ax.fill_between(g['t_hr'], i, i + 0.85 * np.nan_to_num(y),
                        color=MOD_COLOR[MODALITY[ch]], alpha=0.75, lw=0)
        ax.axhline(i, color='0.85', lw=0.5)
    ax.set_yticks(np.arange(len(CHANNELS)) + 0.4)
    ax.set_yticklabels([f'{c} ({MODALITY[c]})' for c in CHANNELS], fontsize=8)
    ax.set_ylim(0, len(CHANNELS))
    ax.set_xlabel('Time from session start (hr)', fontsize=9)
    ax.set_ylabel('Ladder present', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f'  wrote {path}')


def fig_psd(grid, runs, path):
    """Motion-free median PSD per channel with k*f0 markers from the optics."""
    opt_rr, opt_lad = runs[(OPT_CH[0], 'card_ext')]
    f0 = float(np.nanmedian(opt_lad['ladder_f0'])) if opt_lad['is_ladder'].any() \
        else np.nan

    ncol = 3
    nrow = int(np.ceil(len(CHANNELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.7 * nrow))
    for ax, ch in zip(axes.ravel(), CHANNELS):
        sig, _ = prepare(grid, ch, PREP_BAND_EXT[1])
        rr, _ = runs[(ch, 'card_ext')]
        _, f, P = spectrogram(sig, fmax=5.0)
        keep = ~rr['motion_mask'][:len(P)]
        med = np.nanmedian(P[keep], axis=0)
        ax.semilogy(f, med, color=MOD_COLOR[MODALITY[ch]], lw=1.2)
        if np.isfinite(f0):
            for k in range(1, 5):
                if k * f0 <= 5.0:
                    ax.axvline(k * f0, color='0.5', ls=':', lw=0.9)
        ax.set_title(f'{ch} ({MODALITY[ch]})', fontsize=9)
        ax.set_xlim(0, 5)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(CHANNELS):]:
        ax.axis('off')
    fig.suptitle(f'Motion-free median PSD, dotted lines at k x {f0:.3f} Hz '
                 f'(optical ladder fundamental)', fontsize=11)
    fig.supxlabel('Frequency (Hz)', fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f'  wrote {path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reload', action='store_true',
                    help='rebuild the 20 Hz grid from the raw files')
    ap.add_argument('--no-figs', action='store_true')
    args = ap.parse_args()

    grid = load_grid(args.reload)
    print(f'\nGrid: {len(grid):,} samples @ {FS:g} Hz '
          f'= {len(grid) / FS / 60:.1f} min\n')

    ridges, windows, harm, runs = run_detection(grid)
    n_win = int(windows['t_hr'].nunique())
    hr_hz = window_hr(grid, n_win)

    summary, hsummary = summarise(windows, harm, hr_hz)
    xmod = cross_modal(windows, harm)

    ridges.to_csv(OUT / 'ridges.csv', index=False)
    windows.to_csv(OUT / 'windows.csv', index=False)
    harm.to_csv(OUT / 'harmonics.csv', index=False)
    summary.to_csv(OUT / 'summary.csv', index=False)
    hsummary.to_csv(OUT / 'harmonic_summary.csv', index=False)
    xmod.to_csv(OUT / 'cross_modal.csv', index=False)

    print('\n' + '=' * 78)
    print(f'Muse reported HR over the window: median {np.nanmedian(hr_hz) * 60:.1f} bpm '
          f'= {np.nanmedian(hr_hz):.3f} Hz')
    print('\nPersistent ridges + ladders (motion-free windows):')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nPer-window harmonic detector, f0 searched in 0.5-2.0 Hz:')
    print(hsummary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nCross-modal: windows where >=3 of 4 optical channels agree a '
          f'cardiac ladder is present (n={int(xmod["n_reference_windows"].max())}):')
    print(xmod.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    if not args.no_figs:
        print('\nFigures:')
        fig_ridges(grid, runs, FIG / 'fig_ridges.png')
        fig_ladders(windows, hr_hz, FIG / 'fig_ladders.png')
        fig_psd(grid, runs, FIG / 'fig_psd.png')

    print(f'\nwrote -> {OUT}')


if __name__ == '__main__':
    main()
