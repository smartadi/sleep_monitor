"""
Harmonic ladder detection via concurrent persistent ridges.

For each session x channel:
  1. Run detect_persistent_ridges (temporally smoothed, continuity-tracked)
  2. At each window, check if active ridges form integer-ratio ladders
  3. Label each window as "ladder" or "not"
  4. Plot: spectrogram with ladder windows highlighted, ridge lines overlaid
  5. Summary stats: ladder prevalence by sleep stage

Channels: CH, CLE, CRE, acc_mag (raw accelerometer for reference).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_all_sessions, FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.config import PSG_EPOCH_SEC
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import detect_persistent_ridges, label_harmonic_ladder_windows

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE', 'acc_mag']
CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD', 'acc_mag': '#E67E22'}

# Ridge detector parameters
WIN_SEC = 30.0
STEP_SEC = 30.0
MAX_FREQ = 5.0
SMOOTH_WINDOWS = 7
MIN_PERSIST_SEC = 300.0
MAX_FREQ_JUMP = 0.08
PEAK_PROM_FRAC = 0.5
MAX_GAP_WINDOWS = 5
WELCH_SEG_SEC = 8.0

# Ladder labeling
RATIO_TOL = 0.12
MIN_HARMONICS = 2
MIN_F0 = 0.1


def prepare_signals(session):
    acc_mag = session.cap['acc_mag']
    signals = {}
    for ch in CHANNELS:
        if ch == 'acc_mag':
            signals[ch] = acc_mag.astype(np.float64)
        else:
            signals[ch] = remove_acc_artifact(session.cap[ch], acc_mag, 0.05, 4.0)
    return signals, acc_mag


def run_ridges_and_ladders(signals, acc_mag):
    """Run ridge tracker + ladder labeling on all channels."""
    results = {}
    for ch in CHANNELS:
        sig = signals[ch]
        am = None if ch == 'acc_mag' else acc_mag

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
            acc_mag=am,
        )

        ladder = label_harmonic_ladder_windows(
            rr,
            ratio_tol=RATIO_TOL,
            min_harmonics=MIN_HARMONICS,
            min_f0=MIN_F0,
        )

        results[ch] = {'rr': rr, 'ladder': ladder}
    return results


def align_stages(t_hr, sp):
    """Map each window time to a sleep stage code."""
    if sp is None:
        return np.full(len(t_hr), -1, dtype=int)
    codes = sp['codes']
    t_ep = sp['t_ep_hr']
    out = np.empty(len(t_hr), dtype=int)
    for i, t in enumerate(t_hr):
        idx = np.clip(np.searchsorted(t_ep, t, side='right') - 1, 0, len(codes) - 1)
        out[i] = int(codes[idx])
    return out


def plot_session(session, results, out_path):
    """
    Per-session plot: hypnogram + spectrogram with ridges and ladder highlighting
    for each channel.
    """
    sp = session.sleep_profile
    n_ch = len(CHANNELS)
    fig, axes = plt.subplots(n_ch + 1, 1, figsize=(18, 3.2 * (n_ch + 1)),
                             gridspec_kw={'height_ratios': [0.5] + [1.5] * n_ch},
                             sharex=True)

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
    ax.set_title(f'{session.label} -- Harmonic Ladder Windows (persistent ridges)',
                 fontsize=11, fontweight='bold')

    ridge_cmap = plt.cm.tab20

    for idx, ch in enumerate(CHANNELS):
        ax = axes[idx + 1]
        sig = results[ch]['rr']
        rr = results[ch]['rr']
        ladder = results[ch]['ladder']
        t_hr = rr['t_hr']

        # Spectrogram from raw signal
        raw_sig = np.zeros(1)  # placeholder
        # Use the smoothed PSDs directly for a cleaner image
        psds_plot = rr['psds_smooth']
        freqs = rr['freqs']

        # If psds_smooth is available, plot as image
        valid_rows = ~np.all(np.isnan(psds_plot), axis=1)
        if valid_rows.sum() > 0:
            Sxx_db = 10 * np.log10(np.where(np.isnan(psds_plot), 1e-30, psds_plot) + 1e-30)
            vmin, vmax = np.nanpercentile(Sxx_db[valid_rows], [5, 95])
            ax.pcolormesh(t_hr, freqs, Sxx_db.T,
                          shading='nearest', cmap='inferno',
                          vmin=vmin, vmax=vmax, rasterized=True)

        # Overlay persistent ridges as colored lines
        ridges = rr['ridges']
        n_ridges = max(len(ridges), 1)
        for ri, ridge in enumerate(ridges):
            color = ridge_cmap(ri % 20)
            valid = ~np.isnan(ridge['freq_trace'])
            ax.plot(t_hr[valid], ridge['freq_trace'][valid], '-',
                    color=color, lw=1.5, alpha=0.85, zorder=3)

        # Highlight ladder windows: green vertical bars at bottom
        ladder_mask = ladder['is_ladder']
        for i in range(len(t_hr)):
            if ladder_mask[i]:
                ax.axvspan(t_hr[i] - STEP_SEC / 7200.0,
                           t_hr[i] + STEP_SEC / 7200.0,
                           ymin=0, ymax=0.08, color='#2ECC71', alpha=0.9, zorder=5)
                # Draw the ladder member frequencies as bright dots
                for f in ladder['ladder_freqs'][i]:
                    ax.plot(t_hr[i], f, 'o', color='lime', markersize=3,
                            markeredgewidth=0, alpha=0.9, zorder=6)

        # Motion mask: red ticks at top
        motion = rr['motion_mask']
        for i in range(len(t_hr)):
            if motion[i]:
                ax.axvspan(t_hr[i] - STEP_SEC / 7200.0,
                           t_hr[i] + STEP_SEC / 7200.0,
                           ymin=0.95, ymax=1.0, color='red', alpha=0.6, zorder=5)

        n_ladder = int(ladder_mask.sum())
        n_total = int((~rr['motion_mask']).sum())
        pct = 100.0 * n_ladder / max(n_total, 1)
        ax.set_ylim(0, MAX_FREQ)
        ax.set_ylabel(f'{ch}\nFreq (Hz)', fontsize=8)
        ax.text(0.005, 0.92,
                f'{ch}: {len(ridges)} ridges, {n_ladder}/{n_total} ladder windows ({pct:.1f}%)',
                transform=ax.transAxes, fontsize=7, fontweight='bold',
                color='white', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    axes[-1].set_xlabel('Time (hr)', fontsize=9)

    # Legend for markers
    legend_elements = [
        mpatches.Patch(color='#2ECC71', label='Ladder window'),
        plt.Line2D([0], [0], marker='o', color='lime', lw=0,
                   markersize=5, label='Ladder freq'),
        mpatches.Patch(color='red', alpha=0.6, label='Motion'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=6, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_epoch_table(session, results):
    """Build per-epoch DataFrame with ladder labels and sleep stages."""
    rows = []
    sp = session.sleep_profile

    for ch in CHANNELS:
        rr = results[ch]['rr']
        ladder = results[ch]['ladder']
        t_hr = rr['t_hr']
        stages = align_stages(t_hr, sp)

        for i in range(len(t_hr)):
            rows.append({
                'session': session.label,
                'subject': session.subject,
                'channel': ch,
                't_hr': t_hr[i],
                'motion_masked': bool(rr['motion_mask'][i]),
                'is_ladder': bool(ladder['is_ladder'][i]),
                'ladder_f0': ladder['ladder_f0'][i],
                'ladder_n': int(ladder['ladder_n'][i]),
                'ladder_power': ladder['ladder_power'][i],
                'stage_code': int(stages[i]),
                'stage_label': STAGE_LABELS.get(int(stages[i]), '?'),
            })

    return pd.DataFrame(rows)


def plot_ladder_prevalence(all_df, out_path):
    """Bar chart: % of windows with harmonic ladders by sleep stage, per channel."""
    valid = all_df[~all_df['motion_masked'] & (all_df['stage_code'] >= 0)].copy()

    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4.5 * len(CHANNELS), 5),
                             sharey=True)
    if len(CHANNELS) == 1:
        axes = [axes]

    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        cv = valid[valid['channel'] == ch]

        prevalences = []
        labels_used = []
        colors = []
        for sc in STAGE_ORDER:
            sv = cv[cv['stage_code'] == sc]
            if len(sv) == 0:
                continue
            pct = 100.0 * sv['is_ladder'].sum() / len(sv)
            prevalences.append(pct)
            labels_used.append(STAGE_LABELS[sc])
            colors.append(STAGE_COLORS[sc])

        bars = ax.bar(labels_used, prevalences, color=colors, alpha=0.7, edgecolor='black', lw=0.5)
        for bar, pct in zip(bars, prevalences):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{pct:.1f}%', ha='center', fontsize=7, fontweight='bold')

        ax.set_title(ch, fontsize=10, fontweight='bold', color=CH_COLORS.get(ch, 'black'))
        ax.set_ylabel('Ladder prevalence (%)' if ci == 0 else '', fontsize=9)
        ax.grid(True, alpha=0.15, axis='y')

    fig.suptitle('Harmonic Ladder Prevalence by Sleep Stage (all sessions)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ladder_prevalence_per_subject(all_df, out_path):
    """Per-subject ladder prevalence heatmap."""
    valid = all_df[
        ~all_df['motion_masked'] & (all_df['stage_code'] >= 0)
        & (all_df['channel'] == 'CH')
    ].copy()

    subjects = sorted(valid['subject'].unique())
    stage_labels = [STAGE_LABELS[sc] for sc in STAGE_ORDER]

    matrix = np.full((len(subjects), len(STAGE_ORDER)), np.nan)
    for si, subj in enumerate(subjects):
        sv = valid[valid['subject'] == subj]
        for sj, sc in enumerate(STAGE_ORDER):
            ss = sv[sv['stage_code'] == sc]
            if len(ss) > 0:
                matrix[si, sj] = 100.0 * ss['is_ladder'].sum() / len(ss)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlGn', vmin=0)
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, fontsize=9)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=9)

    for si in range(len(subjects)):
        for sj in range(len(STAGE_ORDER)):
            val = matrix[si, sj]
            if np.isfinite(val):
                ax.text(sj, si, f'{val:.1f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if val > 15 else 'black')

    plt.colorbar(im, ax=ax, label='Ladder prevalence (%)')
    ax.set_title('Harmonic Ladder Prevalence -- CH channel, per subject x stage', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading all sessions...')
    sessions = load_all_sessions(with_sleep_profiles=True)

    if len(sys.argv) > 1:
        labels = sys.argv[1:]
        sessions = [s for s in sessions if s.label in labels]

    all_epoch_dfs = []

    for s in sessions:
        print(f'\n{"="*60}')
        print(f'{s.label} ({s.subject}, {s.duration_hr:.1f} hr)')
        print(f'{"="*60}')

        signals, acc_mag = prepare_signals(s)
        results = run_ridges_and_ladders(signals, acc_mag)

        for ch in CHANNELS:
            rr = results[ch]['rr']
            ladder = results[ch]['ladder']
            n_ladder = int(ladder['is_ladder'].sum())
            n_valid = int((~rr['motion_mask']).sum())
            pct = 100.0 * n_ladder / max(n_valid, 1)
            print(f'  {ch:>7}: {len(rr["ridges"]):2d} ridges, '
                  f'{n_ladder:3d}/{n_valid:3d} ladder windows ({pct:5.1f}%)')

        # Per-session spectrogram + ladder plot
        out = REPORT_DIR / f'harmonic_ladders_{s.label}.png'
        print(f'  Plotting -> {out.name}')
        plot_session(s, results, out)

        # Epoch table
        df = build_epoch_table(s, results)
        all_epoch_dfs.append(df)

        # Quick per-stage summary
        valid = df[~df['motion_masked'] & (df['stage_code'] >= 0) & (df['channel'] == 'CH')]
        for sc in STAGE_ORDER:
            sv = valid[valid['stage_code'] == sc]
            if len(sv) > 0:
                pct = 100.0 * sv['is_ladder'].sum() / len(sv)
                print(f'    CH {STAGE_LABELS[sc]:>4}: {pct:5.1f}% ladder ({int(sv["is_ladder"].sum())}/{len(sv)})')

    # ── Combine all sessions ──
    all_df = pd.concat(all_epoch_dfs, ignore_index=True)
    pq_path = REPORT_DIR / 'harmonic_ladders_epochs.parquet'
    all_df.to_parquet(pq_path, index=False)
    print(f'\nSaved {len(all_df)} rows -> {pq_path.name}')

    # ── Pooled plots ──
    print('\nPlotting ladder prevalence by stage...')
    plot_ladder_prevalence(all_df, REPORT_DIR / 'harmonic_ladders_by_stage.png')

    print('Plotting per-subject heatmap...')
    plot_ladder_prevalence_per_subject(all_df, REPORT_DIR / 'harmonic_ladders_per_subject.png')

    # ── Print summary ──
    print('\n' + '=' * 60)
    print('HARMONIC LADDER SUMMARY (CH channel)')
    print('=' * 60)
    valid = all_df[~all_df['motion_masked'] & (all_df['stage_code'] >= 0) & (all_df['channel'] == 'CH')]
    for sc in STAGE_ORDER:
        sv = valid[valid['stage_code'] == sc]
        if len(sv) > 0:
            n_lad = int(sv['is_ladder'].sum())
            pct = 100.0 * n_lad / len(sv)
            med_f0 = sv.loc[sv['is_ladder'], 'ladder_f0'].median()
            med_n = sv.loc[sv['is_ladder'], 'ladder_n'].median()
            print(f'  {STAGE_LABELS[sc]:>4}: {pct:5.1f}% ({n_lad:4d}/{len(sv):4d})  '
                  f'median f0={med_f0:.2f} Hz  median members={med_n:.0f}')

    print(f'\nTotal: {int(valid["is_ladder"].sum())}/{len(valid)} '
          f'({100*valid["is_ladder"].sum()/len(valid):.1f}%)')
    print('\nDone. Check reports/slow_wave/ for all outputs.')
