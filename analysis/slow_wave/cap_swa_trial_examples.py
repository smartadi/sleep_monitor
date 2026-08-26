"""
CAP-SWA trial EXAMPLES with pre/post context — now with high-res raw CAP + PSG.

For each session, pick a few representative trials (ranked by N3 content then
duration) and plot each with a fixed pad of epochs BEFORE and AFTER the trial.
Two artifacts per session:

  examples/<session>.png                 compact per-session overview (epoch level,
                                          3 example trials as columns)
  examples/<session>_trial<NNN>.png       ONE full-width HIGH-RES figure per example
                                          trial showing the raw signals:
    1. sleep stage strip (+ head-movement ticks)
    2. hold criteria C1 (single-ch slow-drift) / C3 (slow thorax), epoch level
    3. raw CAP  — CLE, CRE, CH   (100 Hz, demeaned per window)
    4. raw PSG  — EEG (0.3–40 Hz), ECG, Thorax, Flow, Pleth   (100 Hz)
    trial span shaded green; initiating head movement(s) marked orange.

Reads:  reports/slow_wave/cap_swa/trials/{trial_epochs.parquet, trials.csv}
        raw sessions via sleep_monitor.load_session (for the 100 Hz waveforms)
Writes: reports/slow_wave/cap_swa/trials/examples/

Run: .venv/Scripts/python.exe analysis/slow_wave/cap_swa_trial_examples.py
     .venv/Scripts/python.exe analysis/slow_wave/cap_swa_trial_examples.py --n 4
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, SESSION_META, FS
from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS
from sleep_monitor.filters import bandpass

ROOT = Path(__file__).resolve().parents[2]
TRIALS_DIR = ROOT / 'reports' / 'slow_wave' / 'cap_swa' / 'trials'
OUT = TRIALS_DIR / 'examples'
OUT.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
N3_CODE = 1
PAD_EPOCHS = 10        # 5 min of context before and after the trial
N_EXAMPLES = 3         # example trials per session
Q_HOLD = 0.50          # must match the value used to build the trials
HIRES_DPI = 150

LABEL_TO_IDX = {m['label']: m['idx'] for m in SESSION_META}

RAW_CAP = ['CLE', 'CRE', 'CH']
RAW_PSG = ['EEG', 'ECG', 'Thorax', 'Flow', 'Pleth']
RAW_COLORS = {
    'CLE': '#2980B9', 'CRE': '#C0392B', 'CH': '#16A085',
    'EEG': '#2C3E50', 'ECG': '#C0392B', 'Thorax': '#D35400',
    'Flow': '#2980B9', 'Pleth': '#8E44AD',
}


def pick_examples(trials_sess, n=N_EXAMPLES):
    """Rank a session's trials by N3 content then duration; take the top n."""
    t = trials_sess.sort_values(['frac_N3', 'duration_min'], ascending=False)
    return t.head(n)


def _trial_window(g, trial):
    """Return (w epochs, onset_hr, x_epoch_min, x_off_min, move_x_min) for a trial."""
    tid = trial['trial_id']
    idx = np.where(g['trial_id'].values == tid)[0]
    a, b = idx[0], idx[-1] + 1
    lo, hi = max(0, a - PAD_EPOCHS), min(len(g), b + PAD_EPOCHS)
    w = g.iloc[lo:hi]
    onset_hr = g['t_hr'].iloc[a]
    x = (w['t_hr'].values - onset_hr) * 60.0
    x_off = (g['t_hr'].iloc[b - 1] - onset_hr) * 60.0 + EPOCH_SEC / 60.0
    move_x = ((w.loc[w['head_move'], 't_hr'].values - onset_hr) * 60.0
              if 'head_move' in w else np.array([]))
    return w, onset_hr, x, x_off, move_x


# ── compact per-session overview (epoch level) ───────────────────────────────

def plot_trial(axes, g_sess, trial, col):
    g = g_sess.reset_index(drop=True)
    w, onset_hr, x, x_off, move_x = _trial_window(g, trial)

    def shade(ax):
        ax.axvspan(0.0, x_off, color='#2ECC71', alpha=0.16)
        ax.axvline(0.0, color='#27AE60', lw=1.0)
        ax.axvline(x_off, color='#27AE60', lw=1.0, ls='--')
        for mx in move_x:
            ax.axvline(mx, color='#E67E22', lw=0.7, alpha=0.5)

    ax = axes[0]
    for _, r in w.iterrows():
        xx = (r['t_hr'] - onset_hr) * 60.0
        ax.axvspan(xx, xx + EPOCH_SEC / 60.0,
                   color=STAGE_COLORS.get(int(r['stage_code']), '#AAA'),
                   alpha=0.9 if r['stage_code'] == N3_CODE else 0.45)
    ax.axvline(0.0, color='k', lw=0.8); ax.axvline(x_off, color='k', lw=0.8, ls='--')
    for mx in move_x:
        ax.axvline(mx, color='#E67E22', lw=1.2, alpha=0.8)
    ax.set_yticks([])
    ax.set_title(f"trial {int(trial['trial_id'])} · {trial['duration_min']:.1f} min · "
                 f"dom {trial['dom_stage']} · N3 {trial['frac_N3']*100:.0f}% "
                 f"(orange = head move)", fontsize=9)

    ax = axes[1]; shade(ax)
    ax.plot(x, w['c1_slow_drift'], lw=1.0, label='C1 slow single-ch drift', color='#2980B9')
    ax.plot(x, w['c3_slow_thorax'], lw=1.0, label='C3 slow thorax', color='#E67E22')
    ax.axhline(Q_HOLD, color='k', ls=':', alpha=0.6); ax.set_ylim(0, 1.02)
    if col == 0:
        ax.set_ylabel('hold criteria\nsub-scores'); ax.legend(fontsize=6, loc='lower left')

    ax = axes[2]; shade(ax)
    for ch, c in [('cle_mean', '#2980B9'), ('cre_mean', '#C0392B'), ('ch_mean', '#16A085')]:
        v = w[ch].values.astype(float)
        ax.plot(x, v - np.nanmean(v), lw=1.0, color=c, label=ch.split('_')[0].upper())
    if col == 0:
        ax.set_ylabel('single-ch mean\n(demeaned, C1)'); ax.legend(fontsize=6, ncol=3, loc='lower left')

    ax = axes[3]; shade(ax)
    ax.plot(x, w['ecg_hr_hz'] * 60.0, lw=1.0, color='#C0392B')
    if col == 0:
        ax.set_ylabel('HR (BPM)')
    ax = axes[4]; shade(ax)
    ax.plot(x, w['eeg_delta_ratio'], lw=1.0, color='#8E44AD')
    if col == 0:
        ax.set_ylabel('EEG delta')
    ax = axes[5]; shade(ax)
    ax.plot(x, w['thorax_rms'], lw=1.0, color='#27AE60')
    ax2 = ax.twinx(); ax2.plot(x, w['acc_rms'], lw=1.0, color='#7F8C8D', alpha=0.8)
    ax2.set_yticks([])
    if col == 0:
        ax.set_ylabel('thorax /\naccel')
    ax.set_xlabel('minutes from trial onset')


def plot_session_overview(g_sess, trials_sess, sess):
    ex = pick_examples(trials_sess)
    n = len(ex)
    if n == 0:
        return False
    fig, axes = plt.subplots(6, n, figsize=(6.2 * n, 12), squeeze=False, sharex='col')
    fig.suptitle(f'{sess} — example CAP-SWA trials with pre/post context '
                 f'({PAD_EPOCHS*EPOCH_SEC/60:.0f} min pad, green = trial)',
                 fontsize=13, fontweight='bold')
    for col, (_, tr) in enumerate(ex.iterrows()):
        plot_trial([axes[r][col] for r in range(6)], g_sess, tr, col)
    handles = [Patch(facecolor=STAGE_COLORS.get(c, '#AAA'), label=STAGE_LABELS[c])
               for c in [4, 3, 2, 1, 0]]
    fig.legend(handles=handles, loc='upper right', fontsize=8, ncol=5,
               bbox_to_anchor=(0.995, 0.985))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f'{sess}.png', dpi=115, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


# ── high-res per-trial figure with raw CAP + PSG ─────────────────────────────

def plot_trial_hires(session, g_sess, trial, sess):
    """One full-width high-res figure: epoch context + raw CAP + raw PSG waveforms."""
    g = g_sess.reset_index(drop=True)
    w, onset_hr, x_ep, x_off, move_x = _trial_window(g, trial)

    # raw sample window [onset-pad, offset+pad] via the shared 100 Hz time grid
    t_hr = session.time_hr
    t0 = onset_hr - PAD_EPOCHS * EPOCH_SEC / 3600.0
    t1 = (g['t_hr'].iloc[np.where(g['trial_id'].values == trial['trial_id'])[0][-1]]
          + (PAD_EPOCHS + 1) * EPOCH_SEC / 3600.0)
    i0 = int(np.searchsorted(t_hr, t0)); i1 = int(np.searchsorted(t_hr, t1))
    xr = (t_hr[i0:i1] - onset_hr) * 60.0

    n_rows = 2 + len(RAW_CAP) + len(RAW_PSG)
    heights = [0.6, 1.0] + [1.0] * len(RAW_CAP) + [1.2, 1.0, 1.0, 1.0, 1.0]
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 1.15 * sum(heights) + 2),
                             sharex=True, gridspec_kw={'height_ratios': heights})
    fig.suptitle(f'{sess} · trial {int(trial["trial_id"])} — {trial["duration_min"]:.1f} min · '
                 f'dom {trial["dom_stage"]} · N3 {trial["frac_N3"]*100:.0f}%  '
                 f'(raw CAP + PSG, 100 Hz; green = trial, orange = head movement)',
                 fontsize=13, fontweight='bold')

    def shade(ax, marks=True):
        ax.axvspan(0.0, x_off, color='#2ECC71', alpha=0.14)
        ax.axvline(0.0, color='#27AE60', lw=1.0)
        ax.axvline(x_off, color='#27AE60', lw=1.0, ls='--')
        if marks:
            for mx in move_x:
                ax.axvline(mx, color='#E67E22', lw=1.0, ls=':', alpha=0.8)

    # 1 stage strip
    ax = axes[0]
    for _, r in w.iterrows():
        xx = (r['t_hr'] - onset_hr) * 60.0
        ax.axvspan(xx, xx + EPOCH_SEC / 60.0,
                   color=STAGE_COLORS.get(int(r['stage_code']), '#AAA'),
                   alpha=0.9 if r['stage_code'] == N3_CODE else 0.45)
    ax.axvline(0.0, color='k', lw=0.8); ax.axvline(x_off, color='k', lw=0.8, ls='--')
    for mx in move_x:
        ax.axvline(mx, color='#E67E22', lw=1.4)
    ax.set_yticks([]); ax.set_ylabel('stage', fontsize=9)
    handles = [Patch(facecolor=STAGE_COLORS.get(c, '#AAA'), label=STAGE_LABELS[c])
               for c in [4, 3, 2, 1, 0]]
    ax.legend(handles=handles, fontsize=7, ncol=5, loc='upper right')

    # 2 criteria (epoch)
    ax = axes[1]; shade(ax)
    ax.plot(x_ep, w['c1_slow_drift'], lw=1.2, color='#2980B9', label='C1 slow-drift')
    ax.plot(x_ep, w['c3_slow_thorax'], lw=1.2, color='#E67E22', label='C3 slow-thorax')
    ax.axhline(Q_HOLD, color='k', ls=':', alpha=0.6); ax.set_ylim(0, 1.02)
    ax.set_ylabel('criteria', fontsize=9); ax.legend(fontsize=7, ncol=2, loc='lower left')

    # 3 raw CAP (demeaned per window)
    row = 2
    for ch in RAW_CAP:
        ax = axes[row]; shade(ax, marks=False)
        sig = session.cap[ch][i0:i1].astype(np.float64)
        ax.plot(xr, sig - np.nanmean(sig), lw=0.4, color=RAW_COLORS[ch], rasterized=True)
        ax.set_ylabel(f'{ch}\n(raw)', fontsize=9)
        row += 1

    # 4 raw PSG
    for ch in RAW_PSG:
        ax = axes[row]; shade(ax, marks=False)
        sig = session.psg[ch][i0:i1].astype(np.float64)
        if ch == 'EEG':
            sig = bandpass(sig, 0.3, 40.0, FS)   # readability
            lbl = 'EEG\n(0.3–40)'
        else:
            sig = sig - np.nanmean(sig)
            lbl = f'{ch}\n(raw)'
        ax.plot(xr, sig, lw=0.4, color=RAW_COLORS[ch], rasterized=True)
        ax.set_ylabel(lbl, fontsize=9)
        row += 1

    axes[-1].set_xlabel('minutes from trial onset', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = OUT / f'{sess}_trial{int(trial["trial_id"]):03d}.png'
    fig.savefig(out, dpi=HIRES_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=N_EXAMPLES,
                    help='example trials per session')
    ap.add_argument('--overview-only', action='store_true',
                    help='skip the raw-signal high-res figures (fast)')
    args = ap.parse_args()

    epochs = pd.read_parquet(TRIALS_DIR / 'trial_epochs.parquet')
    trials = pd.read_csv(TRIALS_DIR / 'trials.csv')

    made_ov, made_hi = [], 0
    for sess in sorted(epochs['session'].unique()):
        g = epochs[epochs['session'] == sess].sort_values('t_hr')
        ts = trials[trials['session'] == sess]
        if plot_session_overview(g, ts, sess):
            made_ov.append(sess)
        if args.overview_only or ts.empty:
            continue
        ex = pick_examples(ts, args.n)
        idx = LABEL_TO_IDX.get(sess)
        if idx is None:
            print(f'  {sess}: no session idx, skipping raw figures'); continue
        session = load_session(idx)          # 100 Hz raw, loaded once per session
        for _, tr in ex.iterrows():
            out = plot_trial_hires(session, g, tr, sess)
            made_hi += 1
            print(f'  {out.name}')

    print(f'\noverview figures: {len(made_ov)}  |  high-res raw figures: {made_hi}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
