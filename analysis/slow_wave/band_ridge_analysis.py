"""
Band-restricted ridge and harmonic detection.

The Stage-3 ridge pipeline detected persistent spectral ridges across the full
0-5 Hz band. This script repeats the analysis restricted to the two
physiological bands the sensor is known to carry:

  - Respiratory band : 0.1-0.5 Hz  (fine-resolution Welch, one periodogram/window)
  - Cardiac band     : 0.5-3.0 Hz  (8 s Welch segments, harmonic ladders possible)

For each of 12 sessions x 3 CAP channels x 2 bands we run
detect_persistent_ridges with a band-limited (min_freq, max_freq), summarise
per-epoch ridge structure, align to PSG stage, and test stage association
(Kruskal-Wallis over stages; N3-vs-rest Mann-Whitney with per-subject direction).

Outputs:
  reports/slow_wave/band_ridge_epochs.parquet         (per-epoch, per-band, per-channel)
  reports/slow_wave/band_ridge_stage_summary.csv      (KW + N3-vs-rest per band/feature)
  writeup/figures/harmonics/band_ridge_by_stage.png   (paper figure)
  writeup/figures/harmonics/band_ridge_overlay_<label>.png (representative session)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kruskal, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_all_sessions, FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import detect_persistent_ridges

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
FIG_DIR = Path(__file__).resolve().parents[2] / 'writeup' / 'figures' / 'harmonics'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}

# Per-band detector parameters.  Respiratory needs fine frequency resolution
# (single 30 s periodogram, df~0.033 Hz) since the whole band is only 0.4 Hz
# wide; cardiac uses 8 s Welch segments and permits integer-ratio harmonics.
BANDS = {
    'resp': dict(min_freq=0.1, max_freq=0.5, welch_seg_sec=30.0,
                 max_freq_jump=0.05, peak_prominence_frac=0.3),
    'card': dict(min_freq=0.5, max_freq=3.0, welch_seg_sec=8.0,
                 max_freq_jump=0.10, peak_prominence_frac=0.5),
}
BAND_LABEL = {'resp': 'Respiratory (0.1-0.5 Hz)', 'card': 'Cardiac (0.5-3.0 Hz)'}

WIN_SEC = 30.0
STEP_SEC = 30.0
SMOOTH_WINDOWS = 7
MIN_PERSIST_SEC = 300.0
MAX_GAP_WINDOWS = 5

# Channel used for pooled stage statistics per band (dominant ridge channel).
POOL_CH = {'resp': 'CRE', 'card': 'CRE'}


def ridge_epoch_features(rr: dict, ch: str, band: str) -> pd.DataFrame:
    """Per-epoch summary of band-limited ridge structure."""
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    groups = rr['harmonic_groups']
    n_win = len(t_hr)

    n_ridges = np.zeros(n_win, dtype=int)
    n_groups_active = np.zeros(n_win, dtype=int)
    min_ridge_freq = np.full(n_win, np.nan)
    mean_ridge_freq = np.full(n_win, np.nan)
    freq_spread = np.full(n_win, np.nan)
    total_ridge_power = np.full(n_win, np.nan)
    max_prominence = np.full(n_win, np.nan)

    # prominence traces keyed by ridge index
    for i in range(n_win):
        freqs_i, amps_i, proms_i = [], [], []
        for ridge in ridges:
            f = ridge['freq_trace'][i]
            if np.isfinite(f):
                freqs_i.append(f)
                amps_i.append(ridge['amp_trace'][i])
                pr = ridge.get('prominence_trace')
                if pr is not None and np.isfinite(pr[i]):
                    proms_i.append(pr[i])
        n_ridges[i] = len(freqs_i)
        if freqs_i:
            fa = np.array(freqs_i)
            min_ridge_freq[i] = fa.min()
            mean_ridge_freq[i] = fa.mean()
            freq_spread[i] = fa.std() if len(fa) > 1 else 0.0
            total_ridge_power[i] = float(np.sum(amps_i))
        if proms_i:
            max_prominence[i] = float(np.max(proms_i))

        grp_count = 0
        for grp in groups:
            active = sum(
                1 for mi in grp['harmonic_idxs']
                if mi < len(ridges) and np.isfinite(ridges[mi]['freq_trace'][i])
            )
            if active >= 2:
                grp_count += 1
        n_groups_active[i] = grp_count

    return pd.DataFrame({
        't_hr': t_hr,
        'band': band,
        'channel': ch,
        'motion_masked': rr['motion_mask'],
        'n_ridges': n_ridges,
        'n_groups_active': n_groups_active,
        'min_ridge_freq': min_ridge_freq,
        'mean_ridge_freq': mean_ridge_freq,
        'freq_spread': freq_spread,
        'total_ridge_power': total_ridge_power,
        'max_prominence': max_prominence,
        'ridge_present': (n_ridges > 0).astype(int),
    })


def align_sleep_stages(df: pd.DataFrame, sp: dict) -> pd.DataFrame:
    if sp is None:
        df['stage_code'] = -1
        df['stage_label'] = '?'
        return df
    t_ep = sp['t_ep_hr']
    codes = sp['codes']
    scodes = []
    for t in df['t_hr']:
        idx = np.searchsorted(t_ep, t, side='right') - 1
        idx = np.clip(idx, 0, len(codes) - 1)
        scodes.append(int(codes[idx]))
    df['stage_code'] = scodes
    df['stage_label'] = [STAGE_LABELS.get(c, '?') for c in scodes]
    return df


FEATURES = [
    ('ridge_present', 'Ridge present (frac)'),
    ('n_ridges', 'Active ridges'),
    ('min_ridge_freq', 'Lowest ridge freq (Hz)'),
    ('total_ridge_power', 'Total ridge power'),
    ('freq_spread', 'Ridge freq spread (Hz)'),
    ('n_groups_active', 'Harmonic groups'),
]


def stage_summary(all_epochs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        pool = all_epochs[
            (all_epochs['band'] == band)
            & (all_epochs['channel'] == POOL_CH[band])
            & (~all_epochs['motion_masked'])
            & (all_epochs['stage_code'] >= 0)
        ].copy()
        pool_ex_wake = pool[pool['stage_code'] != 4].copy()
        pool_ex_wake['is_N3'] = pool_ex_wake['stage_code'] == 1
        subjects = sorted(pool['subject'].unique())
        for feat, label in FEATURES:
            groups = []
            for sc in STAGE_ORDER:
                vals = pool.loc[pool['stage_code'] == sc, feat].dropna()
                if len(vals) > 0:
                    groups.append(vals.values)
            kw_p = kruskal(*groups)[1] if len(groups) >= 2 else np.nan

            n3 = pool_ex_wake.loc[pool_ex_wake['is_N3'], feat].dropna()
            oth = pool_ex_wake.loc[~pool_ex_wake['is_N3'], feat].dropna()
            if len(n3) > 5 and len(oth) > 5:
                mwu_p = mannwhitneyu(n3, oth, alternative='two-sided')[1]
            else:
                mwu_p = np.nan
            dirs = []
            for subj in subjects:
                sv = pool_ex_wake[pool_ex_wake['subject'] == subj]
                a = sv.loc[sv['is_N3'], feat].dropna()
                b = sv.loc[~sv['is_N3'], feat].dropna()
                if len(a) > 3 and len(b) > 3:
                    dirs.append('N3>' if a.median() > b.median() else 'N3<')
                else:
                    dirs.append('?')
            n_up = dirs.count('N3>')
            n_dn = dirs.count('N3<')
            rows.append(dict(
                band=band, channel=POOL_CH[band], feature=feat,
                n3_median=float(n3.median()) if len(n3) else np.nan,
                other_median=float(oth.median()) if len(oth) else np.nan,
                kw_p=kw_p, mwu_p=mwu_p,
                n_subj_N3_up=n_up, n_subj_N3_dn=n_dn,
                directions=','.join(dirs),
            ))
    return pd.DataFrame(rows)


def plot_by_stage(all_epochs: pd.DataFrame, out_path: Path):
    """One row per band, columns = key features, boxplots by stage."""
    show_feats = [
        ('min_ridge_freq', 'Lowest ridge freq (Hz)'),
        ('total_ridge_power', 'Total ridge power'),
        ('n_groups_active', 'Harmonic groups'),
    ]
    fig, axes = plt.subplots(len(BANDS), len(show_feats),
                             figsize=(15, 8), squeeze=False)
    for r, band in enumerate(BANDS):
        pool = all_epochs[
            (all_epochs['band'] == band)
            & (all_epochs['channel'] == POOL_CH[band])
            & (~all_epochs['motion_masked'])
            & (all_epochs['stage_code'] >= 0)
        ]
        for c, (feat, label) in enumerate(show_feats):
            ax = axes[r, c]
            data, labs, cols = [], [], []
            for sc in STAGE_ORDER:
                vals = pool.loc[pool['stage_code'] == sc, feat].dropna()
                if len(vals) > 0:
                    data.append(vals.values)
                    labs.append(STAGE_LABELS[sc])
                    cols.append(STAGE_COLORS[sc])
            if not data:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center')
                continue
            bp = ax.boxplot(data, labels=labs, patch_artist=True, widths=0.6,
                            showfliers=False, medianprops=dict(color='black', lw=1.5))
            for j, col in enumerate(cols):
                bp['boxes'][j].set_facecolor(col)
                bp['boxes'][j].set_alpha(0.6)
            kw_p = kruskal(*data)[1] if len(data) >= 2 else np.nan
            sig = '***' if kw_p < 1e-3 else '**' if kw_p < 1e-2 else '*' if kw_p < 0.05 else 'ns'
            ax.set_title(f'{label}\nKW p={kw_p:.1e} {sig}', fontsize=9)
            ax.grid(True, alpha=0.15, axis='y')
            if c == 0:
                ax.set_ylabel(f'{BAND_LABEL[band]}\n({POOL_CH[band]})', fontsize=9)
    fig.suptitle('Band-restricted ridge features by sleep stage', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_overlay(label, band_rr: dict, sp: dict, out_path: Path):
    """Representative session: hypnogram + resp-band and cardiac-band ridge freq traces."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={'height_ratios': [0.5, 1, 1]})
    ax = axes[0]
    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.6)
    ax.set_yticks([])
    ax.set_ylabel('Stage', fontsize=8)
    patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c]) for c in STAGE_ORDER]
    ax.legend(handles=patches, loc='upper right', fontsize=6, ncol=5)
    ax.set_title(f'{label} — band-restricted ridges ({POOL_CH["resp"]})', fontsize=11)

    for row, band in enumerate(['resp', 'card']):
        ax = axes[row + 1]
        rr = band_rr[band]
        t = rr['t_hr']
        for ridge in rr['ridges']:
            ft = ridge['freq_trace']
            ax.plot(t, ft, lw=1.2, alpha=0.85)
        ax.set_ylabel(f'{BAND_LABEL[band]}\nridge freq (Hz)', fontsize=8)
        ax.set_ylim(BANDS[band]['min_freq'], BANDS[band]['max_freq'])
        ax.grid(True, alpha=0.15)
    axes[-1].set_xlabel('Time (hr)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'  wrote {out_path}')


if __name__ == '__main__':
    print('Loading sessions with sleep profiles...')
    sessions = load_all_sessions(with_sleep_profiles=True)

    all_dfs = []
    overlay_target = 'S1N1'
    overlay_rr = {}

    for s in sessions:
        print(f'\n=== {s.label} ({s.subject}, {s.duration_hr:.1f} hr) ===')
        acc_mag = s.cap['acc_mag']
        for ch in CHANNELS:
            sig = remove_acc_artifact(s.cap[ch], acc_mag, 0.05, 4.0)
            for band, bp in BANDS.items():
                rr = detect_persistent_ridges(
                    sig, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC,
                    min_freq=bp['min_freq'], max_freq=bp['max_freq'],
                    smooth_windows=SMOOTH_WINDOWS,
                    min_persistence_sec=MIN_PERSIST_SEC,
                    max_freq_jump=bp['max_freq_jump'],
                    peak_prominence_frac=bp['peak_prominence_frac'],
                    welch_seg_sec=bp['welch_seg_sec'],
                    max_gap_windows=MAX_GAP_WINDOWS,
                    acc_mag=acc_mag,
                )
                df = ridge_epoch_features(rr, ch, band)
                df = align_sleep_stages(df, s.sleep_profile)
                df['session'] = s.label
                df['subject'] = s.subject
                all_dfs.append(df)
                print(f'  {ch} {band:>4}: {len(rr["ridges"])} ridges, '
                      f'{len(rr["harmonic_groups"])} groups')
                if s.label == overlay_target and ch == POOL_CH[band]:
                    overlay_rr[band] = rr

    all_epochs = pd.concat(all_dfs, ignore_index=True)
    pq = REPORT_DIR / 'band_ridge_epochs.parquet'
    all_epochs.to_parquet(pq, index=False)
    print(f'\nSaved {len(all_epochs)} rows -> {pq}')

    summary = stage_summary(all_epochs)
    csv = REPORT_DIR / 'band_ridge_stage_summary.csv'
    summary.to_csv(csv, index=False)
    print(f'Summary -> {csv}')
    print(summary.to_string(index=False))

    plot_by_stage(all_epochs, FIG_DIR / 'band_ridge_by_stage.png')
    if len(overlay_rr) == 2:
        sp = next(s.sleep_profile for s in sessions if s.label == overlay_target)
        plot_overlay(overlay_target, overlay_rr, sp,
                     FIG_DIR / f'band_ridge_overlay_{overlay_target}.png')

    print('\nDone.')
