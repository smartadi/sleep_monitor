"""
Stage 3: Persistent ridge features vs physiology.

For all 12 sessions × 3 CAP channels:
  1. Run detect_persistent_ridges (temporal smoothing + continuity tracking)
  2. Per-epoch (30 s): count active ridges, harmonic groups, amplitudes, frequencies
  3. Align with PSG sleep stage labels
  4. Statistical tests: ridge features by stage (Kruskal-Wallis + post-hoc Dunn)
  5. Per-subject and pooled analysis
  6. Plots saved to reports/slow_wave/

Outputs:
  - reports/slow_wave/stage3_ridge_epochs.parquet   (per-epoch feature table)
  - reports/slow_wave/stage3_ridge_features_by_stage.png
  - reports/slow_wave/stage3_ridge_timeseries_<label>.png  (per-session)
  - reports/slow_wave/stage3_summary.csv
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
from sleep_monitor.config import PSG_EPOCH_SEC
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import detect_persistent_ridges

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}

# Ridge detector parameters (tuned from Stage 2 verification)
WIN_SEC = 30.0
STEP_SEC = 30.0
MAX_FREQ = 5.0
SMOOTH_WINDOWS = 7
MIN_PERSIST_SEC = 300.0
MAX_FREQ_JUMP = 0.08
PEAK_PROM_FRAC = 0.5
MAX_GAP_WINDOWS = 5
WELCH_SEG_SEC = 8.0


# ── Per-epoch feature extraction from ridge results ─────────────────────────

def ridge_epoch_features(rr: dict, ch: str) -> pd.DataFrame:
    """
    From detect_persistent_ridges output, compute per-epoch summary features.

    Returns DataFrame with one row per epoch window.
    """
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    groups = rr['harmonic_groups']
    n_win = len(t_hr)

    n_ridges = np.zeros(n_win, dtype=int)
    n_groups_active = np.zeros(n_win, dtype=int)
    mean_ridge_freq = np.full(n_win, np.nan)
    min_ridge_freq = np.full(n_win, np.nan)
    max_ridge_freq = np.full(n_win, np.nan)
    freq_spread = np.full(n_win, np.nan)
    mean_ridge_amp = np.full(n_win, np.nan)
    max_ridge_amp = np.full(n_win, np.nan)
    total_ridge_power = np.full(n_win, np.nan)
    strongest_f0 = np.full(n_win, np.nan)
    max_group_size = np.zeros(n_win, dtype=int)

    for i in range(n_win):
        freqs_i = []
        amps_i = []
        for ridge in ridges:
            f = ridge['freq_trace'][i]
            a = ridge['amp_trace'][i]
            if np.isfinite(f):
                freqs_i.append(f)
                amps_i.append(a)

        n_ridges[i] = len(freqs_i)

        if len(freqs_i) > 0:
            freqs_arr = np.array(freqs_i)
            amps_arr = np.array(amps_i)
            mean_ridge_freq[i] = np.mean(freqs_arr)
            min_ridge_freq[i] = np.min(freqs_arr)
            max_ridge_freq[i] = np.max(freqs_arr)
            freq_spread[i] = np.std(freqs_arr) if len(freqs_arr) > 1 else 0.0
            mean_ridge_amp[i] = np.mean(amps_arr)
            max_ridge_amp[i] = np.max(amps_arr)
            total_ridge_power[i] = np.sum(amps_arr)

        # Count harmonic groups active at this window
        grp_count = 0
        best_grp_size = 0
        for grp in groups:
            members_active = 0
            for mi in grp['harmonic_idxs']:
                if mi < len(ridges) and np.isfinite(ridges[mi]['freq_trace'][i]):
                    members_active += 1
            if members_active >= 2:
                grp_count += 1
                best_grp_size = max(best_grp_size, members_active)
                if members_active > best_grp_size or np.isnan(strongest_f0[i]):
                    fi = grp['fundamental_idx']
                    f0_val = ridges[fi]['freq_trace'][i]
                    if np.isfinite(f0_val):
                        strongest_f0[i] = f0_val

        n_groups_active[i] = grp_count
        max_group_size[i] = best_grp_size

    df = pd.DataFrame({
        't_hr': t_hr,
        'channel': ch,
        'motion_masked': rr['motion_mask'],
        'n_ridges': n_ridges,
        'n_groups_active': n_groups_active,
        'max_group_size': max_group_size,
        'mean_ridge_freq': mean_ridge_freq,
        'min_ridge_freq': min_ridge_freq,
        'max_ridge_freq': max_ridge_freq,
        'freq_spread': freq_spread,
        'mean_ridge_amp': mean_ridge_amp,
        'max_ridge_amp': max_ridge_amp,
        'total_ridge_power': total_ridge_power,
        'strongest_f0': strongest_f0,
    })
    return df


def align_sleep_stages(df: pd.DataFrame, sp: dict) -> pd.DataFrame:
    """Add sleep stage code and label to each epoch row."""
    if sp is None:
        df['stage_code'] = -1
        df['stage_label'] = '?'
        return df

    t_ep = sp['t_ep_hr']
    codes = sp['codes']

    stage_codes = []
    for t in df['t_hr']:
        idx = np.searchsorted(t_ep, t, side='right') - 1
        idx = np.clip(idx, 0, len(codes) - 1)
        stage_codes.append(int(codes[idx]))

    df['stage_code'] = stage_codes
    df['stage_label'] = [STAGE_LABELS.get(c, '?') for c in stage_codes]
    return df


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_timeseries(session_label: str, session_dfs: dict, sp: dict, out_path: Path):
    """
    Per-session time series: hypnogram + ridge features over time for each channel.
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={'height_ratios': [0.5, 1, 1, 1, 1]})

    # Row 0: Hypnogram
    ax = axes[0]
    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.6)
    ax.set_yticks([])
    ax.set_ylabel('Stage', fontsize=8)
    patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
               for c in STAGE_ORDER]
    ax.legend(handles=patches, loc='upper right', fontsize=6, ncol=5)
    ax.set_title(f'{session_label} — Persistent Ridge Features vs Sleep Stage', fontsize=11)

    # Row 1: n_ridges per channel
    ax = axes[1]
    for ch in CHANNELS:
        df = session_dfs[ch]
        valid = ~df['motion_masked']
        ax.plot(df.loc[valid, 't_hr'], df.loc[valid, 'n_ridges'],
                color=CH_COLORS[ch], lw=0.8, alpha=0.8, label=ch)
    ax.set_ylabel('Active ridges', fontsize=8)
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.15)

    # Row 2: n_groups_active
    ax = axes[2]
    for ch in CHANNELS:
        df = session_dfs[ch]
        valid = ~df['motion_masked']
        ax.plot(df.loc[valid, 't_hr'], df.loc[valid, 'n_groups_active'],
                color=CH_COLORS[ch], lw=0.8, alpha=0.8)
    ax.set_ylabel('Harmonic groups', fontsize=8)
    ax.grid(True, alpha=0.15)

    # Row 3: mean_ridge_freq (lowest ridge -> likely f0)
    ax = axes[3]
    for ch in CHANNELS:
        df = session_dfs[ch]
        valid = ~df['motion_masked'] & df['min_ridge_freq'].notna()
        ax.plot(df.loc[valid, 't_hr'], df.loc[valid, 'min_ridge_freq'],
                color=CH_COLORS[ch], lw=0.8, alpha=0.8)
    ax.set_ylabel('Lowest ridge\nfreq (Hz)', fontsize=8)
    ax.set_ylim(0, 2.0)
    ax.grid(True, alpha=0.15)

    # Row 4: total_ridge_power
    ax = axes[4]
    for ch in CHANNELS:
        df = session_dfs[ch]
        valid = ~df['motion_masked'] & df['total_ridge_power'].notna()
        ax.plot(df.loc[valid, 't_hr'], df.loc[valid, 'total_ridge_power'],
                color=CH_COLORS[ch], lw=0.8, alpha=0.8)
    ax.set_ylabel('Total ridge\npower', fontsize=8)
    ax.set_xlabel('Time (hr)', fontsize=9)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_features_by_stage(all_epochs: pd.DataFrame, out_path: Path):
    """
    Pooled box plots of ridge features by sleep stage, one panel per feature.
    """
    features = [
        ('n_ridges', 'Active ridges'),
        ('n_groups_active', 'Harmonic groups'),
        ('max_group_size', 'Max group size'),
        ('min_ridge_freq', 'Lowest ridge freq (Hz)'),
        ('total_ridge_power', 'Total ridge power'),
        ('freq_spread', 'Ridge freq spread (Hz)'),
    ]

    stage_order_labels = [STAGE_LABELS[c] for c in STAGE_ORDER]
    valid = all_epochs[~all_epochs['motion_masked'] & (all_epochs['stage_code'] >= 0)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (feat, label) in enumerate(features):
        ax = axes[i]
        data_by_stage = []
        labels_used = []
        for sc in STAGE_ORDER:
            sl = STAGE_LABELS[sc]
            vals = valid.loc[valid['stage_code'] == sc, feat].dropna()
            if len(vals) > 0:
                data_by_stage.append(vals.values)
                labels_used.append(sl)

        if not data_by_stage:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        bp = ax.boxplot(data_by_stage, labels=labels_used, patch_artist=True,
                        widths=0.6, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        for j, sc in enumerate(STAGE_ORDER[:len(bp['boxes'])]):
            bp['boxes'][j].set_facecolor(STAGE_COLORS[sc])
            bp['boxes'][j].set_alpha(0.6)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.15, axis='y')

        # Kruskal-Wallis test
        if len(data_by_stage) >= 2:
            try:
                stat, pval = kruskal(*data_by_stage)
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                ax.set_title(f'{label}\nKW p={pval:.2e} {sig}', fontsize=8)
            except ValueError:
                ax.set_title(label, fontsize=8)
        else:
            ax.set_title(label, fontsize=8)

    fig.suptitle('Ridge Features by Sleep Stage (all sessions, CH channel)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_features_by_stage_per_subject(all_epochs: pd.DataFrame, out_path: Path):
    """
    Per-subject ridge feature comparison across stages — check universality.
    """
    valid = all_epochs[
        ~all_epochs['motion_masked'] & (all_epochs['stage_code'] >= 0)
        & (all_epochs['channel'] == 'CH')
    ].copy()

    subjects = sorted(valid['subject'].unique())
    features = [
        ('n_ridges', 'Active ridges'),
        ('n_groups_active', 'Harmonic groups'),
        ('total_ridge_power', 'Total ridge power'),
    ]

    fig, axes = plt.subplots(len(features), len(subjects),
                             figsize=(3.2 * len(subjects), 3.5 * len(features)),
                             squeeze=False)

    for row, (feat, label) in enumerate(features):
        for col, subj in enumerate(subjects):
            ax = axes[row, col]
            sv = valid[valid['subject'] == subj]

            data_by_stage = []
            labels_used = []
            for sc in STAGE_ORDER:
                sl = STAGE_LABELS[sc]
                vals = sv.loc[sv['stage_code'] == sc, feat].dropna()
                if len(vals) > 0:
                    data_by_stage.append(vals.values)
                    labels_used.append(sl)

            if not data_by_stage:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            bp = ax.boxplot(data_by_stage, labels=labels_used, patch_artist=True,
                            widths=0.6, showfliers=False,
                            medianprops=dict(color='black', lw=1.5))
            for j, sc in enumerate(STAGE_ORDER[:len(bp['boxes'])]):
                bp['boxes'][j].set_facecolor(STAGE_COLORS[sc])
                bp['boxes'][j].set_alpha(0.6)

            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.15, axis='y')

            if row == 0:
                ax.set_title(subj, fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(label, fontsize=8)

            # KW test
            if len(data_by_stage) >= 2:
                try:
                    _, pval = kruskal(*data_by_stage)
                    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                    ax.text(0.98, 0.95, f'p={pval:.1e} {sig}', transform=ax.transAxes,
                            fontsize=5, ha='right', va='top')
                except ValueError:
                    pass

    fig.suptitle('Ridge Features by Stage — Per Subject (CH)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_n3_vs_rest(all_epochs: pd.DataFrame, out_path: Path):
    """
    Focused comparison: N3 vs non-N3 (pooled N1+N2+REM) for key features.
    Tests if ridge structure marks deep sleep specifically.
    """
    valid = all_epochs[
        ~all_epochs['motion_masked'] & (all_epochs['stage_code'] >= 0)
        & (all_epochs['stage_code'] != 4)  # exclude Wake
        & (all_epochs['channel'] == 'CH')
    ].copy()

    valid['is_N3'] = valid['stage_code'] == 1

    features = [
        ('n_ridges', 'Active ridges'),
        ('n_groups_active', 'Harmonic groups'),
        ('max_group_size', 'Max group size'),
        ('min_ridge_freq', 'Lowest ridge freq (Hz)'),
        ('total_ridge_power', 'Total ridge power'),
        ('freq_spread', 'Ridge freq spread (Hz)'),
    ]

    subjects = sorted(valid['subject'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    summary_rows = []

    for i, (feat, label) in enumerate(features):
        ax = axes[i]

        # Pooled
        n3_vals = valid.loc[valid['is_N3'], feat].dropna()
        other_vals = valid.loc[~valid['is_N3'], feat].dropna()

        if len(n3_vals) > 5 and len(other_vals) > 5:
            stat, pval = mannwhitneyu(n3_vals, other_vals, alternative='two-sided')
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        else:
            pval = np.nan
            sig = '?'

        bp = ax.boxplot([other_vals.values, n3_vals.values],
                        labels=['non-N3\n(N1+N2+REM)', 'N3'],
                        patch_artist=True, widths=0.5, showfliers=False,
                        medianprops=dict(color='black', lw=2))
        bp['boxes'][0].set_facecolor('#3498DB')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('#2ECC71')
        bp['boxes'][1].set_alpha(0.5)

        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f'{label}\nMW-U p={pval:.2e} {sig}', fontsize=8)
        ax.grid(True, alpha=0.15, axis='y')

        # Per-subject direction
        directions = []
        for subj in subjects:
            sv = valid[valid['subject'] == subj]
            n3_s = sv.loc[sv['is_N3'], feat].dropna()
            oth_s = sv.loc[~sv['is_N3'], feat].dropna()
            if len(n3_s) > 3 and len(oth_s) > 3:
                diff = n3_s.median() - oth_s.median()
                directions.append('N3>other' if diff > 0 else 'N3<other')
            else:
                directions.append('?')

        dir_text = ', '.join(f'{s}:{d}' for s, d in zip(subjects, directions))
        ax.text(0.5, -0.18, dir_text, transform=ax.transAxes,
                fontsize=5, ha='center', style='italic')

        summary_rows.append({
            'feature': feat,
            'n3_median': float(n3_vals.median()) if len(n3_vals) > 0 else np.nan,
            'other_median': float(other_vals.median()) if len(other_vals) > 0 else np.nan,
            'mwu_pval': float(pval),
            'significance': sig,
            'per_subject_directions': dir_text,
        })

    fig.suptitle('N3 vs non-N3: Ridge Feature Comparison (CH, all sessions)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return pd.DataFrame(summary_rows)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading all sessions with sleep profiles...')
    sessions = load_all_sessions(with_sleep_profiles=True)

    all_epoch_dfs = []

    for s in sessions:
        print(f'\n{"="*60}')
        print(f'{s.label} ({s.subject}, {s.duration_hr:.1f} hr)')
        print(f'{"="*60}')

        acc_mag = s.cap['acc_mag']
        session_dfs = {}

        for ch in CHANNELS:
            print(f'  {ch}: artifact removal + ridge detection...')
            sig = remove_acc_artifact(s.cap[ch], acc_mag, 0.05, 4.0)

            rr = detect_persistent_ridges(
                sig, fs=FS,
                win_sec=WIN_SEC, step_sec=STEP_SEC,
                max_freq=MAX_FREQ,
                smooth_windows=SMOOTH_WINDOWS,
                min_persistence_sec=MIN_PERSIST_SEC,
                max_freq_jump=MAX_FREQ_JUMP,
                peak_prominence_frac=PEAK_PROM_FRAC,
                max_gap_windows=MAX_GAP_WINDOWS,
                welch_seg_sec=WELCH_SEG_SEC,
                acc_mag=acc_mag,
            )

            n_r = len(rr['ridges'])
            n_g = len(rr['harmonic_groups'])
            print(f'    -> {n_r} ridges, {n_g} harmonic groups')

            df = ridge_epoch_features(rr, ch)
            df = align_sleep_stages(df, s.sleep_profile)
            df['session'] = s.label
            df['subject'] = s.subject

            session_dfs[ch] = df
            all_epoch_dfs.append(df)

            # Quick per-stage summary
            valid = df[~df['motion_masked'] & (df['stage_code'] >= 0)]
            for sc in STAGE_ORDER:
                sv = valid[valid['stage_code'] == sc]
                if len(sv) > 0:
                    med_nr = sv['n_ridges'].median()
                    med_ng = sv['n_groups_active'].median()
                    print(f'      {STAGE_LABELS[sc]:>4}: {len(sv):3d} ep, '
                          f'ridges={med_nr:.1f}, groups={med_ng:.1f}')

        # Per-session time series plot
        out = REPORT_DIR / f'stage3_ridge_timeseries_{s.label}.png'
        print(f'  Plotting time series -> {out.name}')
        plot_timeseries(s.label, session_dfs, s.sleep_profile, out)

    # ── Combine all epochs ──
    all_epochs = pd.concat(all_epoch_dfs, ignore_index=True)
    pq_path = REPORT_DIR / 'stage3_ridge_epochs.parquet'
    all_epochs.to_parquet(pq_path, index=False)
    print(f'\nSaved {len(all_epochs)} epoch rows -> {pq_path.name}')

    # ── Pooled plots (CH channel only for clarity) ──
    ch_epochs = all_epochs[all_epochs['channel'] == 'CH'].copy()

    print('\nPlotting features by stage (pooled)...')
    plot_features_by_stage(ch_epochs, REPORT_DIR / 'stage3_ridge_features_by_stage.png')

    print('Plotting features by stage per subject...')
    plot_features_by_stage_per_subject(all_epochs, REPORT_DIR / 'stage3_ridge_features_per_subject.png')

    print('Plotting N3 vs non-N3 comparison...')
    summary_df = plot_n3_vs_rest(all_epochs, REPORT_DIR / 'stage3_n3_vs_rest.png')

    # Save summary table
    csv_path = REPORT_DIR / 'stage3_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f'Summary -> {csv_path.name}')

    # ── Print key findings ──
    print('\n' + '='*60)
    print('STAGE 3 KEY FINDINGS')
    print('='*60)
    for _, row in summary_df.iterrows():
        print(f"\n  {row['feature']}:")
        print(f"    N3 median:    {row['n3_median']:.3f}")
        print(f"    Other median: {row['other_median']:.3f}")
        print(f"    MW-U p={row['mwu_pval']:.2e} {row['significance']}")
        print(f"    Per-subject:  {row['per_subject_directions']}")

    print('\nDone. Check reports/slow_wave/ for all outputs.')
