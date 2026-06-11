"""
Trial-based SWS exploration: find sleep segments matching physiological criteria.

Phase 1 criteria:
  C1 — Mean capacitance slowly changes (low abs DC slope over rolling window)
  C2 — Settling after head movement (acc_rms low AND was higher in preceding window)
  C3 — Thorax amplitude smoother (variance of thorax RMS in window < preceding window)

A "trial" = run of >= MIN_TRIAL_EPOCHS consecutive epochs where all 3 criteria hold.

Output: reports/slow_wave/trials/<session>/  — per-trial multi-panel plots + summary CSV
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, STAGE_LABELS, STAGE_COLORS, EEG_BANDS,
)
from sleep_monitor.filters import bandpass, lowpass
from sleep_monitor.sessions import SESSION_META

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'trials'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = PSG_EPOCH_SEC  # 30 s
MIN_TRIAL_EPOCHS = 10      # trial must last >= 5 minutes


# ── Per-epoch feature computation ────────────────────────────────────────────

def compute_epoch_traces(session):
    """
    Compute per-epoch traces for criteria 1, 2, 3 + validation signals.

    Returns DataFrame with one row per epoch.
    """
    sp = session.sleep_profile
    if sp is None:
        raise ValueError(f'{session.label}: no sleep profile')

    fs = session.fs
    n_epochs = len(sp['codes'])
    epoch_n = int(EPOCH_SEC * fs)
    t_hr = session.time_hr

    cle = session.cap['CLE'].astype(np.float64)
    cre = session.cap['CRE'].astype(np.float64)
    diff = cle - cre
    acc = session.cap['acc_mag'].astype(np.float64)
    thorax = session.psg['Thorax'].astype(np.float64)
    eeg = session.psg['EEG'].astype(np.float64)

    rows = []
    for ei in range(n_epochs):
        t_start = sp['t_ep_hr'][ei]
        t_end = t_start + EPOCH_SEC / 3600.0
        mask = (t_hr >= t_start) & (t_hr < t_end)
        if mask.sum() < epoch_n * 0.5:
            continue

        idx = np.where(mask)[0]
        seg_diff = diff[idx]
        seg_acc = acc[idx]
        seg_thorax = thorax[idx]
        seg_eeg = eeg[idx]

        # C1: epoch mean of CLE-CRE (will compute slope over rolling window later)
        dc_mean = float(np.mean(seg_diff))

        # C2: accelerometer RMS (AC component)
        acc_ac = seg_acc - np.mean(seg_acc)
        acc_rms = float(np.sqrt(np.mean(acc_ac ** 2)))

        # C3: thorax RMS (will compute variance ratio over rolling window later)
        thorax_bp = bandpass(seg_thorax, RESP_LO, RESP_HI, fs)
        thorax_rms = float(np.sqrt(np.mean(thorax_bp ** 2)))

        # Validation: EEG delta power (0.5-4 Hz)
        nperseg = min(len(seg_eeg), int(4.0 * fs))
        if len(seg_eeg) >= nperseg:
            freqs, psd = welch(seg_eeg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            df_ = freqs[1] - freqs[0]
            delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
            total_mask = (freqs >= 0.5) & (freqs <= 30.0)
            delta_power = float(np.trapezoid(psd[delta_mask], dx=df_))
            total_power = float(np.trapezoid(psd[total_mask], dx=df_))
            delta_ratio = delta_power / total_power if total_power > 0 else 0.0
        else:
            delta_ratio = np.nan

        rows.append({
            'epoch_idx': ei,
            't_hr': float(t_start),
            'stage_code': int(sp['codes'][ei]),
            'stage_label': sp['labels'][ei],
            'dc_mean': dc_mean,
            'acc_rms': acc_rms,
            'thorax_rms': thorax_rms,
            'eeg_delta_ratio': delta_ratio,
        })

    df = pd.DataFrame(rows)

    # ── Rolling window features ──────────────────────────────────────────────
    W = 5  # epochs = 2.5 min window

    # C1: absolute slope of dc_mean over rolling window
    dc_slope = []
    for i in range(len(df)):
        lo = max(0, i - W // 2)
        hi = min(len(df), i + W // 2 + 1)
        vals = df['dc_mean'].iloc[lo:hi].values
        if len(vals) < 3:
            dc_slope.append(np.nan)
            continue
        t = np.arange(len(vals), dtype=np.float64)
        coeffs = np.polyfit(t, vals, 1)
        dc_slope.append(abs(float(coeffs[0])))
    df['dc_abs_slope'] = dc_slope

    # C2: acc_rms relative to preceding window (ratio = current / preceding)
    acc_ratio = []
    for i in range(len(df)):
        current = df['acc_rms'].iloc[max(0, i - W // 2):i + 1].mean()
        preceding_lo = max(0, i - W)
        preceding_hi = max(0, i - W // 2)
        if preceding_hi <= preceding_lo:
            acc_ratio.append(np.nan)
            continue
        preceding = df['acc_rms'].iloc[preceding_lo:preceding_hi].mean()
        acc_ratio.append(current / preceding if preceding > 1e-9 else np.nan)
    df['acc_ratio'] = acc_ratio

    # C3: variance of thorax_rms in current window vs preceding window
    thorax_var_ratio = []
    for i in range(len(df)):
        cur_lo = max(0, i - W // 2)
        cur_hi = min(len(df), i + W // 2 + 1)
        pre_lo = max(0, cur_lo - W)
        pre_hi = cur_lo
        if pre_hi <= pre_lo or cur_hi <= cur_lo:
            thorax_var_ratio.append(np.nan)
            continue
        var_cur = df['thorax_rms'].iloc[cur_lo:cur_hi].var()
        var_pre = df['thorax_rms'].iloc[pre_lo:pre_hi].var()
        thorax_var_ratio.append(var_cur / var_pre if var_pre > 1e-12 else np.nan)
    df['thorax_var_ratio'] = thorax_var_ratio

    return df


# ── Criterion thresholding ───────────────────────────────────────────────────

def apply_criteria(df, c1_pct=50, c2_ratio=1.3, c3_ratio=1.3):
    """
    Flag epochs meeting each criterion.

    C1: dc_abs_slope below c1_pct percentile (slow change)
    C2: acc_ratio < c2_ratio (calmer than or comparable to preceding window)
    C3: thorax_var_ratio < c3_ratio (smoother than or comparable to preceding)
    """
    slope_thresh = np.nanpercentile(df['dc_abs_slope'].values, c1_pct)

    df['c1_met'] = df['dc_abs_slope'] < slope_thresh
    df['c2_met'] = df['acc_ratio'] < c2_ratio
    df['c3_met'] = df['thorax_var_ratio'] < c3_ratio
    df['all_met'] = df['c1_met'] & df['c2_met'] & df['c3_met']

    return df, slope_thresh


# ── Trial extraction ─────────────────────────────────────────────────────────

def extract_trials(df, min_epochs=MIN_TRIAL_EPOCHS, context_epochs=10):
    """
    Find contiguous runs of all_met=True with at least min_epochs length.

    Returns list of dicts with trial info + context window.
    """
    met = df['all_met'].values.astype(bool)
    trials = []
    i = 0
    while i < len(met):
        if met[i]:
            start = i
            while i < len(met) and met[i]:
                i += 1
            length = i - start
            if length >= min_epochs:
                ctx_lo = max(0, start - context_epochs)
                ctx_hi = min(len(df), i + context_epochs)
                n3_in_trial = (df['stage_code'].iloc[start:i] == 1).sum()
                n3_frac = n3_in_trial / length
                dominant_stage_code = df['stage_code'].iloc[start:i].mode().iloc[0]
                trials.append({
                    'start_epoch': start,
                    'end_epoch': i,
                    'length_epochs': length,
                    'duration_min': length * EPOCH_SEC / 60,
                    't_start_hr': float(df['t_hr'].iloc[start]),
                    't_end_hr': float(df['t_hr'].iloc[i - 1]) + EPOCH_SEC / 3600,
                    'n3_epochs': int(n3_in_trial),
                    'n3_fraction': float(n3_frac),
                    'dominant_stage': STAGE_LABELS.get(int(dominant_stage_code), '?'),
                    'ctx_lo': ctx_lo,
                    'ctx_hi': ctx_hi,
                    'mean_dc_slope': float(df['dc_abs_slope'].iloc[start:i].mean()),
                    'mean_acc_rms': float(df['acc_rms'].iloc[start:i].mean()),
                    'mean_thorax_var_ratio': float(df['thorax_var_ratio'].iloc[start:i].mean()),
                    'mean_eeg_delta': float(df['eeg_delta_ratio'].iloc[start:i].mean()),
                })
        else:
            i += 1
    return trials


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_trial(df, trial, trial_num, session_label, session):
    """Multi-panel plot for one trial with context."""
    ctx = df.iloc[trial['ctx_lo']:trial['ctx_hi']]
    t_min = (ctx['t_hr'].values - ctx['t_hr'].values[0]) * 60  # minutes from context start
    trial_start_min = (trial['t_start_hr'] - ctx['t_hr'].values[0]) * 60
    trial_end_min = (trial['t_end_hr'] - ctx['t_hr'].values[0]) * 60

    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(8, 1, figure=fig, hspace=0.3,
                  height_ratios=[0.4, 1, 1, 1, 1, 1, 1, 0.6])

    fig.suptitle(
        f'{session_label} — Trial {trial_num} '
        f'({trial["duration_min"]:.1f} min, '
        f't={trial["t_start_hr"]:.2f}-{trial["t_end_hr"]:.2f} hr, '
        f'N3={trial["n3_fraction"]:.0%}, dominant={trial["dominant_stage"]})',
        fontsize=13, fontweight='bold',
    )

    # Panel 0: Sleep stage ribbon
    ax = fig.add_subplot(gs[0])
    for _, row in ctx.iterrows():
        ep_start = (row['t_hr'] - ctx['t_hr'].values[0]) * 60
        ep_end = ep_start + EPOCH_SEC / 60
        color = STAGE_COLORS.get(row['stage_code'], '#AAAAAA')
        alpha = 0.8 if row['stage_code'] == 1 else 0.4
        ax.axvspan(ep_start, ep_end, color=color, alpha=alpha)
        if ep_end - ep_start > 0.3:
            ax.text((ep_start + ep_end) / 2, 0.5, row['stage_label'],
                    ha='center', va='center', fontsize=6)
    ax.axvspan(trial_start_min, trial_end_min, facecolor='none',
               edgecolor='red', linewidth=2, linestyle='--')
    ax.set_ylabel('Stage')
    ax.set_yticks([])
    ax.set_xlim(t_min[0], t_min[-1])

    # Panel 1: DC mean (raw CLE-CRE mean per epoch)
    ax = fig.add_subplot(gs[1])
    ax.plot(t_min, ctx['dc_mean'].values, 'o-', color='#E67E22', markersize=3, linewidth=1)
    ax.axvspan(trial_start_min, trial_end_min, color='#2ECC71', alpha=0.15)
    ax.set_ylabel('DC mean\n(CLE-CRE)')
    ax.set_title('C1: Mean capacitance — slow change in shaded region', fontsize=10, loc='left')

    # Panel 2: DC absolute slope
    ax = fig.add_subplot(gs[2])
    ax.plot(t_min, ctx['dc_abs_slope'].values, 'o-', color='#C0392B', markersize=3, linewidth=1)
    ax.axvspan(trial_start_min, trial_end_min, color='#2ECC71', alpha=0.15)
    c1_vals = ctx['c1_met'].values
    ax.fill_between(t_min, 0, ctx['dc_abs_slope'].values,
                    where=c1_vals, color='#2ECC71', alpha=0.3, label='C1 met')
    ax.set_ylabel('|DC slope|')
    ax.legend(fontsize=8)

    # Panel 3: Accelerometer RMS
    ax = fig.add_subplot(gs[3])
    ax.plot(t_min, ctx['acc_rms'].values, 'o-', color='#34495E', markersize=3, linewidth=1)
    ax.axvspan(trial_start_min, trial_end_min, color='#2ECC71', alpha=0.15)
    ax.set_ylabel('Acc RMS')
    ax.set_title('C2: Stillness (low acc, calmer than preceding)', fontsize=10, loc='left')

    # Panel 4: Acc ratio (current / preceding)
    ax = fig.add_subplot(gs[4])
    ax.plot(t_min, ctx['acc_ratio'].values, 'o-', color='#2980B9', markersize=3, linewidth=1)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax.axvspan(trial_start_min, trial_end_min, color='#2ECC71', alpha=0.15)
    c2_vals = ctx['c2_met'].values
    ax.fill_between(t_min, 0, ctx['acc_ratio'].values,
                    where=c2_vals, color='#2ECC71', alpha=0.3, label='C2 met')
    ax.set_ylabel('Acc ratio\n(cur/prev)')
    ax.legend(fontsize=8)

    # Panel 5: Thorax RMS
    ax = fig.add_subplot(gs[5])
    ax.plot(t_min, ctx['thorax_rms'].values, 'o-', color='#8E44AD', markersize=3, linewidth=1)
    ax.axvspan(trial_start_min, trial_end_min, color='#2ECC71', alpha=0.15)
    ax.set_ylabel('Thorax RMS')
    ax.set_title('C3: Smooth thorax (low variance vs preceding window)', fontsize=10, loc='left')

    # Panel 6: Thorax variance ratio
    ax = fig.add_subplot(gs[6])
    ax.plot(t_min, ctx['thorax_var_ratio'].values, 'o-', color='#8E44AD', markersize=3, linewidth=1)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax.axvspan(trial_start_min, trial_end_min, color='#2ECC71', alpha=0.15)
    c3_vals = ctx['c3_met'].values
    ax.fill_between(t_min, 0, np.clip(ctx['thorax_var_ratio'].values, 0, 5),
                    where=c3_vals, color='#2ECC71', alpha=0.3, label='C3 met')
    ax.set_ylabel('Thorax var\nratio (cur/prev)')
    ax.set_ylim(0, 5)
    ax.legend(fontsize=8)

    # Panel 7: EEG delta ratio (validation)
    ax = fig.add_subplot(gs[7])
    ax.plot(t_min, ctx['eeg_delta_ratio'].values, 'o-', color='#C0392B', markersize=3, linewidth=1)
    ax.axvspan(trial_start_min, trial_end_min, color='#2ECC71', alpha=0.15)
    ax.set_ylabel('EEG delta\nratio')
    ax.set_xlabel('Time from context start (min)')
    ax.set_title('Validation: EEG delta power (high = conventional SWS)', fontsize=10, loc='left')

    for a in fig.axes:
        a.set_xlim(t_min[0], t_min[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_night_overview(df, trials, session_label):
    """Full-night overview: all traces + trial regions highlighted."""
    t_hr = df['t_hr'].values

    fig, axes = plt.subplots(7, 1, figsize=(22, 16), sharex=True)
    fig.suptitle(f'{session_label} — Night Overview ({len(trials)} trials detected)',
                 fontsize=14, fontweight='bold')

    for trial in trials:
        for ax in axes:
            ax.axvspan(trial['t_start_hr'], trial['t_end_hr'],
                       color='#2ECC71', alpha=0.15)

    # 0: Stage ribbon
    ax = axes[0]
    for _, row in df.iterrows():
        color = STAGE_COLORS.get(row['stage_code'], '#AAAAAA')
        alpha = 0.8 if row['stage_code'] == 1 else 0.4
        ax.axvspan(row['t_hr'], row['t_hr'] + EPOCH_SEC / 3600, color=color, alpha=alpha)
    ax.set_ylabel('Stage')
    ax.set_yticks([])

    # 1: DC mean
    ax = axes[1]
    ax.plot(t_hr, df['dc_mean'], color='#E67E22', linewidth=0.8)
    ax.set_ylabel('DC mean')

    # 2: DC slope
    ax = axes[2]
    ax.plot(t_hr, df['dc_abs_slope'], color='#C0392B', linewidth=0.8)
    ax.set_ylabel('|DC slope|')

    # 3: Acc RMS
    ax = axes[3]
    ax.plot(t_hr, df['acc_rms'], color='#34495E', linewidth=0.8)
    ax.set_ylabel('Acc RMS')

    # 4: Thorax RMS
    ax = axes[4]
    ax.plot(t_hr, df['thorax_rms'], color='#8E44AD', linewidth=0.8)
    ax.set_ylabel('Thorax RMS')

    # 5: Thorax var ratio
    ax = axes[5]
    ax.plot(t_hr, np.clip(df['thorax_var_ratio'], 0, 5), color='#8E44AD', linewidth=0.8)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('Thorax var\nratio')
    ax.set_ylim(0, 5)

    # 6: EEG delta
    ax = axes[6]
    ax.plot(t_hr, df['eeg_delta_ratio'], color='#C0392B', linewidth=0.8)
    ax.set_ylabel('EEG delta')
    ax.set_xlabel('Time (hr)')

    # Mark criteria met
    for ax_idx, col, color in [(2, 'c1_met', '#2ECC71'),
                                (3, 'c2_met', '#2ECC71'),
                                (5, 'c3_met', '#2ECC71')]:
        met = df[col].values
        for i in range(len(met)):
            if met[i]:
                axes[ax_idx].axvspan(t_hr[i], t_hr[i] + EPOCH_SEC / 3600,
                                     color=color, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def process_session(session_idx, save=True):
    """Run full trial detection pipeline on one session."""
    session = load_session(session_idx)
    session.sleep_profile = load_sleep_profile(session)
    label = session.label

    print(f"\n{'='*60}")
    print(f"Trial detection: {label}")
    print(f"{'='*60}")

    print("  Computing per-epoch traces...")
    df = compute_epoch_traces(session)
    print(f"  {len(df)} epochs computed")

    print("  Applying criteria...")
    df, slope_thresh = apply_criteria(df)
    c1_count = df['c1_met'].sum()
    c2_count = df['c2_met'].sum()
    c3_count = df['c3_met'].sum()
    all_count = df['all_met'].sum()
    print(f"  C1 (slow DC): {c1_count}/{len(df)} epochs (thresh={slope_thresh:.6f})")
    print(f"  C2 (settling): {c2_count}/{len(df)} epochs")
    print(f"  C3 (smooth thorax): {c3_count}/{len(df)} epochs")
    print(f"  All 3 met: {all_count}/{len(df)} epochs")

    print("  Extracting trials...")
    trials = extract_trials(df)
    print(f"  Found {len(trials)} trials (>= {MIN_TRIAL_EPOCHS} consecutive epochs)")

    for i, trial in enumerate(trials):
        n3_tag = f"N3={trial['n3_fraction']:.0%}" if trial['n3_fraction'] > 0 else "no N3"
        print(f"    Trial {i+1}: {trial['duration_min']:.1f} min @ "
              f"t={trial['t_start_hr']:.2f}-{trial['t_end_hr']:.2f} hr, "
              f"{n3_tag}, stage={trial['dominant_stage']}, "
              f"EEG delta={trial['mean_eeg_delta']:.3f}")

    if not save:
        return df, trials

    # Save outputs
    sess_dir = REPORT_DIR / label
    sess_dir.mkdir(parents=True, exist_ok=True)

    # Epoch traces CSV
    csv_path = sess_dir / 'epoch_traces.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved epoch traces: {csv_path}")

    # Night overview plot
    print("  Plotting night overview...")
    fig = plot_night_overview(df, trials, label)
    overview_path = sess_dir / 'night_overview.png'
    fig.savefig(overview_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {overview_path}")

    # Per-trial plots
    for i, trial in enumerate(trials):
        fig = plot_trial(df, trial, i + 1, label, session)
        trial_path = sess_dir / f'trial_{i+1:02d}.png'
        fig.savefig(trial_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    if trials:
        print(f"  Saved {len(trials)} trial plots to {sess_dir}")

    # Trial summary
    if trials:
        summary = pd.DataFrame(trials)
        summary['session'] = label
        summary_path = sess_dir / 'trial_summary.csv'
        summary.to_csv(summary_path, index=False)

    return df, trials


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Trial-based SWS exploration')
    parser.add_argument('--session', type=int, default=0,
                        help='Session index (0-11), default 0 for single run')
    parser.add_argument('--all', action='store_true', help='Run on all 12 sessions')
    args = parser.parse_args()

    print("Trial-Based SWS Exploration — Phase 1 (C1: slow DC, C2: settling, C3: smooth thorax)")
    print("=" * 70)

    all_trials = []

    if args.all:
        for idx in range(12):
            try:
                df, trials = process_session(idx)
                for t in trials:
                    t['session'] = SESSION_META[idx]['label']
                all_trials.extend(trials)
            except Exception as e:
                print(f"  ERROR on session {idx}: {e}")
                import traceback
                traceback.print_exc()
    else:
        df, trials = process_session(args.session)
        for t in trials:
            t['session'] = SESSION_META[args.session]['label']
        all_trials.extend(trials)

    # Global summary
    if all_trials:
        summary = pd.DataFrame(all_trials)
        summary_path = REPORT_DIR / 'all_trials_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f"\n{'='*70}")
        print(f"SUMMARY: {len(all_trials)} trials across {summary['session'].nunique()} sessions")
        print(f"  Mean duration: {summary['duration_min'].mean():.1f} min")
        print(f"  Mean N3 fraction: {summary['n3_fraction'].mean():.2f}")
        print(f"  Trials with any N3: {(summary['n3_fraction'] > 0).sum()}/{len(summary)}")
        stage_dist = summary['dominant_stage'].value_counts()
        print(f"  Dominant stages: {dict(stage_dist)}")
        print(f"  Saved: {summary_path}")


if __name__ == '__main__':
    main()
