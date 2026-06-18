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
from scipy.signal import spectrogram as sp_spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_all_sessions, FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.config import PSG_EPOCH_SEC
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import (
    detect_persistent_ridges,
    compute_prominence_score,
)

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'overlay'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tuned parameters ──────────────────────────────────────────────────────────

CHANNELS = ['CH', 'CLE', 'CRE']
CH_COLORS = {'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD'}

WIN_SEC = 30.0
STEP_SEC = 15.0           # overlapping for smoother traces
MAX_FREQ = 5.0
SMOOTH_WINDOWS = 9        # wider smoothing for more stable PSDs
MIN_PERSIST_SEC = 300.0   # 5 minutes minimum ridge
MAX_FREQ_JUMP = 0.10      # slightly more tolerant continuity
PEAK_PROM_FRAC = 0.25     # lower threshold — catch weaker ridges
MAX_GAP_WINDOWS = 6       # survive ~90s of dropout
WELCH_SEG_SEC = 10.0      # longer segment for better freq resolution

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
    """Run ridge detection + prominence scoring on all channels."""
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
        ps = compute_prominence_score(rr)
        results[ch] = {'rr': rr, 'ps': ps}
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


def compute_fine_spectrogram(sig, fs=100.0, max_freq=5.0):
    """High-res spectrogram for visual background (decoupled from ridge detection)."""
    f, t, Sxx = sp_spectrogram(sig, fs=fs, nperseg=2048, noverlap=1920,
                                nfft=4096, scaling='density')
    mask = f <= max_freq
    return t / 3600.0, f[mask], 10 * np.log10(Sxx[mask] + 1e-30)


def _plot_spectrogram_row(ax, sig, fs, rr, ps, ch, ch_color, motion_mask_regions):
    """Draw one spectrogram row: fine background + ridges colored by prominence + events."""
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    score = ps['prominence_score']

    # Fine spectrogram background
    t_spec, f_spec, Sxx_db = compute_fine_spectrogram(sig, fs=fs, max_freq=MAX_FREQ)
    vmin, vmax = np.nanpercentile(Sxx_db, [5, 95])
    ax.pcolormesh(t_spec, f_spec, Sxx_db,
                  shading='gouraud', cmap='inferno',
                  vmin=vmin, vmax=vmax, rasterized=True)

    # Semi-transparent red overlay on motion-masked regions
    for (t0, t1) in motion_mask_regions:
        ax.axvspan(t0, t1, color='red', alpha=0.15, zorder=2)

    # Ridge traces — color by prominence, label top 20
    if ridges:
        all_prom = np.concatenate([
            r['prominence_trace'][np.isfinite(r['prominence_trace'])]
            for r in ridges if np.any(np.isfinite(r.get('prominence_trace', [])))
        ])
        prom_norm = Normalize(
            vmin=1.0,
            vmax=np.percentile(all_prom, 95) if len(all_prom) > 0 else 10.0,
        )

        ridge_prom = [r.get('median_prominence', 0.0) for r in ridges]
        top_n = min(20, len(ridges))
        top_idxs = set(np.argsort(ridge_prom)[-top_n:])

        cmap_ridge = plt.cm.cool
        for ri, ridge in enumerate(ridges):
            valid = ~np.isnan(ridge['freq_trace'])
            pt = ridge.get('prominence_trace', np.full_like(ridge['freq_trace'], np.nan))
            if valid.sum() < 2:
                continue
            t_r = t_hr[valid]
            f_r = ridge['freq_trace'][valid]
            p_r = pt[valid]
            lw = 2.0 if ri in top_idxs else 1.0
            alpha = 0.9 if ri in top_idxs else 0.5
            for si in range(len(t_r) - 1):
                pval = p_r[si] if np.isfinite(p_r[si]) else 1.0
                color = cmap_ridge(prom_norm(pval))
                ax.plot(t_r[si:si + 2], f_r[si:si + 2],
                        '-', color=color, lw=lw, alpha=alpha, zorder=3)
            if ri in top_idxs:
                med_p = ridge.get('median_prominence', 0)
                lbl = f'{ridge["label"]} ({med_p:.1f}x)'
                ax.text(t_r[0], f_r[0] + 0.08, lbl,
                        fontsize=5.5, color='white', fontweight='bold',
                        bbox=dict(fc='black', alpha=0.5, pad=0.5, lw=0), zorder=4)

    # Green event bars for strong prominence windows
    strong_events = _find_events(score, t_hr, SCORE_THRESH_STRONG, min_windows=4)
    for (t0, t1, _) in strong_events:
        ax.axvspan(t0, t1, ymin=0, ymax=1, color='lime', alpha=0.12, zorder=1)
        ax.axvspan(t0, t1, ymin=0, ymax=0.04, color='lime', alpha=0.9, zorder=5)

    ax.set_ylim(0, MAX_FREQ)
    ax.set_ylabel(f'{ch}\nFreq (Hz)', fontsize=9, color=ch_color)

    # Per-channel annotation
    n_ridges = len(ridges)
    n_valid = int((~rr['motion_mask']).sum())
    n_strong = int((score >= SCORE_THRESH_STRONG).sum())
    med_prom = float(np.median(ps['max_prominence'][~rr['motion_mask']])) if n_valid > 0 else 0
    ax.text(0.005, 0.97,
            f'{ch}: {n_ridges} ridges | {n_strong} strong ({100*n_strong/max(n_valid,1):.1f}%) '
            f'| median prominence {med_prom:.1f}x',
            transform=ax.transAxes, fontsize=8, fontweight='bold',
            color='white', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.7))

    return strong_events


def _get_motion_regions(rr):
    """Convert per-window motion mask to (t_start_hr, t_end_hr) spans."""
    t_hr = rr['t_hr']
    mask = rr['motion_mask']
    dt = STEP_SEC / 3600.0
    regions = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = t_hr[i] - dt / 2
            while i < len(mask) and mask[i]:
                i += 1
            end = t_hr[i - 1] + dt / 2
            regions.append((start, end))
        else:
            i += 1
    return regions


def plot_session(session, signals, results, out_path):
    """
    6-row stacked overlay: hypnogram, 3x channel spectrograms, score, stats.
    """
    sp = session.sleep_profile
    ref_rr = results[CHANNELS[0]]['rr']
    t_hr = ref_rr['t_hr']

    fig, axes = plt.subplots(6, 1, figsize=(22, 20),
                             gridspec_kw={'height_ratios': [0.4, 1.8, 1.8, 1.8, 0.8, 0.8]},
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
    ax.set_title(f'{session.label}  —  Harmonic Ridge Overlay v2  (all channels)',
                 fontsize=13, fontweight='bold')

    # ── Rows 1-3: Per-channel spectrograms ──
    all_strong_events = {}
    for ci, ch in enumerate(CHANNELS):
        rr = results[ch]['rr']
        ps = results[ch]['ps']
        motion_regions = _get_motion_regions(rr)
        events = _plot_spectrogram_row(
            axes[ci + 1], signals[ch], FS, rr, ps, ch,
            CH_COLORS[ch], motion_regions,
        )
        all_strong_events[ch] = events

    # Legend on first spectrogram row only
    legend_elements = [
        mpatches.Patch(color='lime', alpha=0.4, label='Strong prominence event'),
        mpatches.Patch(color='red', alpha=0.15, label='Motion artifact'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=2)

    # ── Row 4: Prominence score (all 3 channels overlaid) ──
    ax = axes[4]
    for ch in CHANNELS:
        rr = results[ch]['rr']
        ps = results[ch]['ps']
        s_plot = ps['prominence_score'].copy()
        s_plot[rr['motion_mask']] = np.nan
        ax.fill_between(rr['t_hr'], 0, s_plot, color=CH_COLORS[ch], alpha=0.2, step='mid')
        ax.plot(rr['t_hr'], s_plot, color=CH_COLORS[ch], lw=0.8, alpha=0.8, label=ch)

    ax.axhline(SCORE_THRESH_STRONG, color='lime', lw=1, ls='--', alpha=0.7)
    ax.axhline(SCORE_THRESH_MODERATE, color='yellow', lw=0.8, ls=':', alpha=0.5)

    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.08, zorder=0)

    all_scores = np.concatenate([results[ch]['ps']['prominence_score'] for ch in CHANNELS])
    smax = np.nanmax(all_scores) if np.any(np.isfinite(all_scores)) else 1.0
    ax.set_ylim(0, min(1.0, smax * 1.3 + 0.05))
    ax.set_ylabel('Prominence\nScore', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.grid(True, alpha=0.15)

    # ── Row 5: Ridge stats (active count + strong count per channel) ──
    ax = axes[5]
    for ch in CHANNELS:
        rr = results[ch]['rr']
        n_active = np.zeros(len(rr['t_hr']), dtype=int)
        for ridge in rr['ridges']:
            n_active += (~np.isnan(ridge['freq_trace'])).astype(int)
        ax.fill_between(rr['t_hr'], 0, n_active, color=CH_COLORS[ch], alpha=0.15, step='mid')
        ax.plot(rr['t_hr'], n_active, color=CH_COLORS[ch], lw=0.8, label=f'{ch} ridges')

    ax2 = ax.twinx()
    for ch in CHANNELS:
        ps = results[ch]['ps']
        rr = results[ch]['rr']
        ns = ps['n_strong_ridges'].astype(float)
        ns[rr['motion_mask']] = np.nan
        ax2.plot(rr['t_hr'], ns, '.', color=CH_COLORS[ch],
                 markersize=2, alpha=0.4)
    ax2.set_ylabel('Strong ridges\n(>5x floor)', fontsize=8, color='#E67E22')
    ax2.set_ylim(0, None)

    ax.set_ylabel('Active\nRidges', fontsize=9)
    ax.set_xlabel('Time (hr)', fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.legend(loc='upper left', fontsize=7, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return all_strong_events


def build_epoch_table(session, results):
    """Per-epoch DataFrame with prominence scores and sleep stages."""
    rows = []
    sp = session.sleep_profile

    for ch in CHANNELS:
        rr = results[ch]['rr']
        ps = results[ch]['ps']
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
                'prominence_score': float(ps['prominence_score'][i]),
                'max_prominence': float(ps['max_prominence'][i]),
                'n_strong_ridges': int(ps['n_strong_ridges'][i]),
                'n_active_ridges': int(n_active[i]),
                'stage_code': int(stages[i]),
                'stage_label': STAGE_LABELS.get(int(stages[i]), '?'),
            })

    return pd.DataFrame(rows)


def plot_score_by_stage(all_df, out_path):
    """Boxplots of prominence score by sleep stage, per channel."""
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
                data.append(sv['prominence_score'].values)
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
        ax.set_ylabel('Prominence Score' if ci == 0 else '', fontsize=9)
        ax.grid(True, alpha=0.15, axis='y')

    fig.suptitle('Ridge Prominence by Sleep Stage (all sessions)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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

        for ch in CHANNELS:
            rr = results[ch]['rr']
            ps = results[ch]['ps']
            n_strong = int((ps['prominence_score'] >= SCORE_THRESH_STRONG).sum())
            n_valid = int((~rr['motion_mask']).sum())
            med_prom = float(np.median(ps['max_prominence'][~rr['motion_mask']])) if n_valid > 0 else 0
            print(f'  {ch:>3}: {len(rr["ridges"]):2d} ridges, '
                  f'med_prom={med_prom:.1f}x, '
                  f'{n_strong:3d}/{n_valid:3d} strong ({100*n_strong/max(n_valid,1):.1f}%)')

        # Per-session 6-row stacked overlay plot
        out = REPORT_DIR / f'ridge_overlay_{s.label}.png'
        print(f'  Overlay plot (3-channel stacked) -> {out.name}')
        all_strong_events = plot_session(s, signals, results, out)
        total_events = sum(len(v) for v in all_strong_events.values())
        if total_events:
            print(f'  {total_events} strong harmonic events across channels:')
            for ch, events in all_strong_events.items():
                for t0, t1, pk in events[:3]:
                    print(f'    {ch} {t0:.2f}–{t1:.2f} hr  (peak={pk:.3f})')
                if len(events) > 3:
                    print(f'    ... and {len(events) - 3} more on {ch}')

        # Epoch table
        df = build_epoch_table(s, results)
        all_epoch_dfs.append(df)

    # ── Combine all sessions ──
    all_df = pd.concat(all_epoch_dfs, ignore_index=True)
    pq_path = REPORT_DIR / 'ridge_overlay_epochs.parquet'
    all_df.to_parquet(pq_path, index=False)
    print(f'\nSaved {len(all_df)} rows -> {pq_path.name}')

    # ── Pooled score-by-stage plot ──
    print('\nPlotting prominence score by stage...')
    plot_score_by_stage(all_df, REPORT_DIR / 'ridge_overlay_score_by_stage.png')

    # ── Summary ──
    print('\n' + '=' * 60)
    print('RIDGE PROMINENCE SUMMARY (all channels)')
    print('=' * 60)
    for ch in CHANNELS:
        cv = all_df[~all_df['motion_masked'] & (all_df['stage_code'] >= 0)
                    & (all_df['channel'] == ch)]
        print(f'\n  {ch}:')
        for sc in STAGE_ORDER:
            sv = cv[cv['stage_code'] == sc]
            if len(sv) > 0:
                med = sv['prominence_score'].median()
                med_raw = sv['max_prominence'].median()
                strong_pct = 100 * (sv['prominence_score'] >= SCORE_THRESH_STRONG).sum() / len(sv)
                print(f'    {STAGE_LABELS[sc]:>4}: score={med:.3f}  prom={med_raw:.1f}x  '
                      f'strong={strong_pct:.1f}%  (n={len(sv)})')

    print('\nDone. Check reports/slow_wave/overlay/ for ridge_overlay_*.png files.')
