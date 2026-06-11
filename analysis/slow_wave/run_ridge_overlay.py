"""
Full-session harmonic ridge overlay — the consolidated workflow.

For each session:
  1. Artifact-remove CAP channels (CH, CLE, CRE)
  2. Run persistent ridge detection with tuned parameters + fragment merging
  3. Compute continuous per-window harmonic strength score
  4. Produce a rich overlay plot:
       Row 0  Hypnogram (sleep stages)
       Row 1  Spectrogram + ridge traces (colored by strength)
       Row 2  Continuous harmonic score + labeled event windows
       Row 3  Ridge stats (n_ridges, n_ladder members, ladder f0)
  5. Save per-epoch table with scores to parquet

Tuning vs run_harmonic_ladders.py:
  - Lower peak prominence (0.25 vs 0.5) to catch weaker but real ridges
  - Ridge fragment merging (new) — stitches broken ridges back together
  - Continuous harmonic score (new) — replaces binary ladder labeling
  - Overlapping step (15s) for smoother temporal resolution
  - Longer gap tolerance (6 windows) — ridges survive brief dropouts

Usage:
    python run_ridge_overlay.py              # all 12 sessions
    python run_ridge_overlay.py S1N1 S2N1    # specific sessions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_all_sessions, FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.config import PSG_EPOCH_SEC
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import (
    detect_persistent_ridges,
    compute_harmonic_score,
    label_harmonic_ladder_windows,
)

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tuned parameters ──────────────────────────────────────────────────────────

CHANNELS = ['CH', 'CLE', 'CRE']
CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}

WIN_SEC = 30.0
STEP_SEC = 15.0           # overlapping for smoother traces
MAX_FREQ = 5.0
SMOOTH_WINDOWS = 9        # wider smoothing for more stable PSDs
MIN_PERSIST_SEC = 180.0   # 3 minutes minimum ridge
MAX_FREQ_JUMP = 0.10      # slightly more tolerant continuity
PEAK_PROM_FRAC = 0.25     # lower threshold — catch weaker ridges
MAX_GAP_WINDOWS = 6       # survive ~90s of dropout
WELCH_SEG_SEC = 10.0      # longer segment for better freq resolution

RATIO_TOL = 0.12
MIN_F0 = 0.1

# Score thresholds for event labeling
SCORE_THRESH_STRONG = 0.3
SCORE_THRESH_MODERATE = 0.1


def prepare_signals(session):
    """Load and artifact-remove CAP channels."""
    acc_mag = session.cap['acc_mag']
    signals = {}
    for ch in CHANNELS:
        signals[ch] = remove_acc_artifact(session.cap[ch], acc_mag, 0.05, 4.0)
    return signals, acc_mag


def run_detection(signals, acc_mag):
    """Run ridge detection + harmonic scoring on all channels."""
    results = {}
    for ch in CHANNELS:
        rr = detect_persistent_ridges(
            signals[ch], fs=FS,
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
        hs = compute_harmonic_score(rr, ratio_tol=RATIO_TOL, min_f0=MIN_F0)
        results[ch] = {'rr': rr, 'hs': hs}
    return results


def align_stages(t_hr, sp):
    """Map window times to sleep stage codes."""
    if sp is None:
        return np.full(len(t_hr), -1, dtype=int)
    codes = sp['codes']
    t_ep = sp['t_ep_hr']
    out = np.empty(len(t_hr), dtype=int)
    for i, t in enumerate(t_hr):
        idx = np.clip(np.searchsorted(t_ep, t, side='right') - 1, 0, len(codes) - 1)
        out[i] = int(codes[idx])
    return out


def _find_events(score, t_hr, thresh, min_windows=3):
    """Find contiguous windows above threshold, return list of (start_hr, end_hr, peak_score)."""
    above = score >= thresh
    events = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i - 1
            if (end - start + 1) >= min_windows:
                peak = float(np.max(score[start:end + 1]))
                events.append((t_hr[start], t_hr[end], peak))
        else:
            i += 1
    return events


def plot_session(session, results, best_ch, out_path):
    """
    Per-session overlay: hypnogram + spectrogram with ridges + harmonic score + stats.
    """
    sp = session.sleep_profile
    rr = results[best_ch]['rr']
    hs = results[best_ch]['hs']
    t_hr = rr['t_hr']
    freqs = rr['freqs']
    ridges = rr['ridges']
    score = hs['harmonic_score']
    n_lad = hs['n_ladder']
    lad_f0 = hs['ladder_f0']

    fig, axes = plt.subplots(4, 1, figsize=(20, 14),
                             gridspec_kw={'height_ratios': [0.4, 2.0, 0.8, 0.8]},
                             sharex=True)

    # ── Row 0: Hypnogram ──
    ax = axes[0]
    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.6)
    ax.set_yticks([])
    ax.set_ylabel('Stage', fontsize=9)
    patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
               for c in STAGE_ORDER]
    ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=5)
    ax.set_title(f'{session.label}  —  Harmonic Ridge Overlay  ({best_ch} channel)',
                 fontsize=13, fontweight='bold')

    # ── Row 1: Spectrogram + ridge traces ──
    ax = axes[1]
    psds_plot = rr['psds_smooth']
    valid_rows = ~np.all(np.isnan(psds_plot), axis=1)
    if valid_rows.sum() > 0:
        Sxx_db = 10 * np.log10(np.where(np.isnan(psds_plot), 1e-30, psds_plot) + 1e-30)
        vmin, vmax = np.nanpercentile(Sxx_db[valid_rows], [5, 95])
        ax.pcolormesh(t_hr, freqs, Sxx_db.T,
                      shading='nearest', cmap='inferno',
                      vmin=vmin, vmax=vmax, rasterized=True)

    # Ridge traces — color by amplitude strength, label only strongest
    if ridges:
        all_amps = np.concatenate([r['amp_trace'][~np.isnan(r['amp_trace'])]
                                   for r in ridges if np.any(~np.isnan(r['amp_trace']))])
        if len(all_amps) > 0:
            amp_norm = Normalize(vmin=np.percentile(all_amps, 10),
                                 vmax=np.percentile(all_amps, 95))
        else:
            amp_norm = Normalize(0, 1)

        # Rank ridges by total amplitude for labeling
        ridge_power = [float(np.nansum(r['amp_trace'])) for r in ridges]
        top_n = min(20, len(ridges))
        top_idxs = set(np.argsort(ridge_power)[-top_n:])

        cmap_ridge = plt.cm.cool
        for ri, ridge in enumerate(ridges):
            valid = ~np.isnan(ridge['freq_trace'])
            if valid.sum() < 2:
                continue
            t_r = t_hr[valid]
            f_r = ridge['freq_trace'][valid]
            a_r = ridge['amp_trace'][valid]

            lw = 2.0 if ri in top_idxs else 1.0
            alpha = 0.9 if ri in top_idxs else 0.5
            for seg_start in range(len(t_r) - 1):
                color = cmap_ridge(amp_norm(a_r[seg_start]))
                ax.plot(t_r[seg_start:seg_start + 2], f_r[seg_start:seg_start + 2],
                        '-', color=color, lw=lw, alpha=alpha, zorder=3)

            if ri in top_idxs:
                ax.text(t_r[0], f_r[0] + 0.08, ridge['label'],
                        fontsize=5.5, color='white', fontweight='bold',
                        bbox=dict(fc='black', alpha=0.5, pad=0.5, lw=0), zorder=4)

    # Highlight strong harmonic event windows
    strong_events = _find_events(score, t_hr, SCORE_THRESH_STRONG, min_windows=4)
    for (t0, t1, peak) in strong_events:
        ax.axvspan(t0, t1, ymin=0, ymax=1, color='lime', alpha=0.12, zorder=1)
        ax.axvspan(t0, t1, ymin=0, ymax=0.04, color='lime', alpha=0.9, zorder=5)

    # Harmonic ladder member dots
    for i in range(len(t_hr)):
        if score[i] >= SCORE_THRESH_MODERATE:
            for f in hs['ladder_freqs'][i]:
                ax.plot(t_hr[i], f, '.', color='cyan',
                        markersize=1.5, alpha=0.6, zorder=5)

    # Motion mask
    for i in range(len(t_hr)):
        if rr['motion_mask'][i]:
            dt = STEP_SEC / 7200.0
            ax.axvspan(t_hr[i] - dt, t_hr[i] + dt,
                       ymin=0.96, ymax=1.0, color='red', alpha=0.6, zorder=5)

    ax.set_ylim(0, MAX_FREQ)
    ax.set_ylabel('Frequency (Hz)', fontsize=9)

    n_ridges = len(ridges)
    n_valid = int((~rr['motion_mask']).sum())
    n_strong = sum(1 for i in range(len(t_hr)) if score[i] >= SCORE_THRESH_STRONG)
    ax.text(0.005, 0.97,
            f'{n_ridges} ridges  |  {n_strong}/{n_valid} strong harmonic windows '
            f'({100*n_strong/max(n_valid,1):.1f}%)',
            transform=ax.transAxes, fontsize=8, fontweight='bold',
            color='white', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.7))

    legend_elements = [
        mpatches.Patch(color='lime', alpha=0.4, label='Strong harmonic event'),
        plt.Line2D([0], [0], color='cyan', marker='.', lw=0, markersize=5,
                   label='Ladder frequency'),
        mpatches.Patch(color='red', alpha=0.6, label='Motion'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=3)

    # ── Row 2: Continuous harmonic score ──
    ax = axes[2]
    valid_mask = ~rr['motion_mask']
    t_plot = t_hr.copy()
    s_plot = score.copy()
    s_plot[~valid_mask] = np.nan

    ax.fill_between(t_plot, 0, s_plot, color='#3498DB', alpha=0.4, step='mid')
    ax.plot(t_plot, s_plot, color='#2980B9', lw=0.8, alpha=0.8)

    ax.axhline(SCORE_THRESH_STRONG, color='lime', lw=1, ls='--', alpha=0.7,
               label=f'Strong ({SCORE_THRESH_STRONG})')
    ax.axhline(SCORE_THRESH_MODERATE, color='yellow', lw=0.8, ls=':', alpha=0.5,
               label=f'Moderate ({SCORE_THRESH_MODERATE})')

    # Stage-colored background
    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.08, zorder=0)

    ax.set_ylim(0, min(1.0, np.nanmax(s_plot) * 1.3 + 0.05) if np.any(np.isfinite(s_plot)) else 1.0)
    ax.set_ylabel('Harmonic\nScore', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── Row 3: Ridge stats ──
    ax = axes[3]

    # Count active ridges per window
    n_active = np.zeros(len(t_hr), dtype=int)
    for ridge in ridges:
        active = ~np.isnan(ridge['freq_trace'])
        n_active += active.astype(int)

    ax.fill_between(t_hr, 0, n_active, color='#9B59B6', alpha=0.3, step='mid',
                    label='Active ridges')
    ax.plot(t_hr, n_active, color='#8E44AD', lw=0.8)

    ax2 = ax.twinx()
    f0_plot = lad_f0.copy()
    f0_plot[n_lad < 2] = np.nan
    ax2.plot(t_hr, f0_plot, '.', color='#E67E22', markersize=2, alpha=0.6,
             label='Ladder f0')
    ax2.set_ylabel('Ladder f0 (Hz)', fontsize=8, color='#E67E22')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='y', colors='#E67E22')

    ax.set_ylabel('Active\nRidges', fontsize=9)
    ax.set_xlabel('Time (hr)', fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return strong_events


def pick_best_channel(results):
    """Pick channel with highest mean harmonic score (non-zero windows only)."""
    best_ch = None
    best_val = -1
    for ch in CHANNELS:
        rr = results[ch]['rr']
        hs = results[ch]['hs']
        valid = ~rr['motion_mask']
        if valid.sum() == 0:
            continue
        scores = hs['harmonic_score'][valid]
        nz = scores[scores > 0]
        val = float(np.mean(nz)) * len(nz) / max(valid.sum(), 1) if len(nz) > 0 else 0
        if val > best_val:
            best_val = val
            best_ch = ch
    return best_ch or 'CH'


def build_epoch_table(session, results):
    """Per-epoch DataFrame with harmonic scores and sleep stages."""
    rows = []
    sp = session.sleep_profile

    for ch in CHANNELS:
        rr = results[ch]['rr']
        hs = results[ch]['hs']
        t_hr = rr['t_hr']
        stages = align_stages(t_hr, sp)

        n_active = np.zeros(len(t_hr), dtype=int)
        for ridge in rr['ridges']:
            n_active += (~np.isnan(ridge['freq_trace'])).astype(int)

        for i in range(len(t_hr)):
            rows.append({
                'session': session.label,
                'subject': session.subject,
                'channel': ch,
                't_hr': t_hr[i],
                'motion_masked': bool(rr['motion_mask'][i]),
                'harmonic_score': float(hs['harmonic_score'][i]),
                'ratio_quality': float(hs['ratio_quality'][i]),
                'n_ladder': int(hs['n_ladder'][i]),
                'ladder_f0': float(hs['ladder_f0'][i]),
                'ladder_power': float(hs['ladder_power'][i]),
                'n_active_ridges': int(n_active[i]),
                'stage_code': int(stages[i]),
                'stage_label': STAGE_LABELS.get(int(stages[i]), '?'),
            })

    return pd.DataFrame(rows)


def plot_score_by_stage(all_df, out_path):
    """Boxplots of harmonic score by sleep stage, per channel."""
    valid = all_df[~all_df['motion_masked'] & (all_df['stage_code'] >= 0)].copy()

    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(5 * len(CHANNELS), 5),
                             sharey=True)
    if len(CHANNELS) == 1:
        axes = [axes]

    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        cv = valid[valid['channel'] == ch]

        data = []
        labels_used = []
        colors = []
        for sc in STAGE_ORDER:
            sv = cv[cv['stage_code'] == sc]
            if len(sv) > 0:
                data.append(sv['harmonic_score'].values)
                labels_used.append(STAGE_LABELS[sc])
                colors.append(STAGE_COLORS[sc])

        bp = ax.boxplot(data, tick_labels=labels_used, patch_artist=True,
                        widths=0.6, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for si, (d, lbl) in enumerate(zip(data, labels_used)):
            med = np.median(d)
            ax.text(si + 1, med + 0.005, f'{med:.3f}', ha='center',
                    fontsize=7, fontweight='bold')

        ax.set_title(ch, fontsize=10, fontweight='bold', color=CH_COLORS.get(ch, 'black'))
        ax.set_ylabel('Harmonic Score' if ci == 0 else '', fontsize=9)
        ax.grid(True, alpha=0.15, axis='y')

    fig.suptitle('Harmonic Score by Sleep Stage (all sessions)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_multichannel_comparison(session, results, out_path):
    """Compare harmonic score across channels for one session."""
    fig, axes = plt.subplots(len(CHANNELS) + 1, 1, figsize=(18, 3 * (len(CHANNELS) + 1)),
                             gridspec_kw={'height_ratios': [0.4] + [1.0] * len(CHANNELS)},
                             sharex=True)

    sp = session.sleep_profile
    ax = axes[0]
    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.6)
    ax.set_yticks([])
    ax.set_ylabel('Stage', fontsize=8)
    ax.set_title(f'{session.label}  —  Multi-channel harmonic score comparison',
                 fontsize=11, fontweight='bold')

    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci + 1]
        rr = results[ch]['rr']
        hs = results[ch]['hs']
        t_hr = rr['t_hr']
        score = hs['harmonic_score'].copy()
        score[rr['motion_mask']] = np.nan

        ax.fill_between(t_hr, 0, score, color=CH_COLORS[ch], alpha=0.4, step='mid')
        ax.plot(t_hr, score, color=CH_COLORS[ch], lw=0.8)
        ax.axhline(SCORE_THRESH_STRONG, color='lime', lw=0.8, ls='--', alpha=0.5)

        n_ridges = len(rr['ridges'])
        med_score = float(np.nanmedian(score[~rr['motion_mask']])) if (~rr['motion_mask']).sum() > 0 else 0
        ax.text(0.005, 0.92,
                f'{ch}: {n_ridges} ridges, median score={med_score:.3f}',
                transform=ax.transAxes, fontsize=8, fontweight='bold',
                color='white', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

        ax.set_ylim(0, 1)
        ax.set_ylabel(f'{ch}\nScore', fontsize=8)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel('Time (hr)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

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
        results = run_detection(signals, acc_mag)

        best_ch = pick_best_channel(results)

        for ch in CHANNELS:
            rr = results[ch]['rr']
            hs = results[ch]['hs']
            n_strong = int((hs['harmonic_score'] >= SCORE_THRESH_STRONG).sum())
            n_valid = int((~rr['motion_mask']).sum())
            med_score = float(np.median(hs['harmonic_score'][~rr['motion_mask']])) if n_valid > 0 else 0
            marker = ' <-- best' if ch == best_ch else ''
            print(f'  {ch:>3}: {len(rr["ridges"]):2d} ridges, '
                  f'med_score={med_score:.3f}, '
                  f'{n_strong:3d}/{n_valid:3d} strong ({100*n_strong/max(n_valid,1):.1f}%)'
                  f'{marker}')

        # Per-session overlay plot (best channel)
        out = REPORT_DIR / f'ridge_overlay_{s.label}.png'
        print(f'  Overlay plot ({best_ch}) -> {out.name}')
        strong_events = plot_session(s, results, best_ch, out)
        if strong_events:
            print(f'  {len(strong_events)} strong harmonic events:')
            for t0, t1, pk in strong_events[:5]:
                print(f'    {t0:.2f}–{t1:.2f} hr  (peak score={pk:.3f})')
            if len(strong_events) > 5:
                print(f'    ... and {len(strong_events) - 5} more')

        # Multi-channel comparison
        mc_out = REPORT_DIR / f'ridge_multichannel_{s.label}.png'
        print(f'  Multi-channel comparison -> {mc_out.name}')
        plot_multichannel_comparison(s, results, mc_out)

        # Epoch table
        df = build_epoch_table(s, results)
        all_epoch_dfs.append(df)

    # ── Combine all sessions ──
    all_df = pd.concat(all_epoch_dfs, ignore_index=True)
    pq_path = REPORT_DIR / 'ridge_overlay_epochs.parquet'
    all_df.to_parquet(pq_path, index=False)
    print(f'\nSaved {len(all_df)} rows -> {pq_path.name}')

    # ── Pooled score-by-stage plot ──
    print('\nPlotting harmonic score by stage...')
    plot_score_by_stage(all_df, REPORT_DIR / 'ridge_overlay_score_by_stage.png')

    # ── Summary ──
    print('\n' + '=' * 60)
    print('HARMONIC SCORE SUMMARY (best channel per session)')
    print('=' * 60)
    for ch in CHANNELS:
        cv = all_df[~all_df['motion_masked'] & (all_df['stage_code'] >= 0)
                    & (all_df['channel'] == ch)]
        print(f'\n  {ch}:')
        for sc in STAGE_ORDER:
            sv = cv[cv['stage_code'] == sc]
            if len(sv) > 0:
                med = sv['harmonic_score'].median()
                strong_pct = 100 * (sv['harmonic_score'] >= SCORE_THRESH_STRONG).sum() / len(sv)
                print(f'    {STAGE_LABELS[sc]:>4}: med={med:.4f}  strong={strong_pct:.1f}%  (n={len(sv)})')

    print('\nDone. Check reports/slow_wave/ for ridge_overlay_*.png files.')
