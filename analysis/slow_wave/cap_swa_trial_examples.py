"""
CAP-SWA trial EXAMPLES with pre/post context.

For each session, pick a few representative trials (ranked by N3 content then
duration) and plot each one with a fixed pad of epochs BEFORE and AFTER the trial,
so you can see what leads into and out of the state. The trial span is shaded; the
onset (t=0) and offset are marked.

Per example trial, stacked panels (shared x = minutes from trial onset):
  1. sleep stage strip (PSG)
  2. the three definition criteria as sub-scores (D1 slow-DC / D3 slow-thorax /
     Dq quiescent) with the q "holds" line
  3. mean capacitance CLE-CRE (dc_mean) — the actual slow-DC quantity (D1)
  4. heart rate (BPM, from ECG)
  5. EEG delta ratio
  6. thorax amplitude (RMS) + accelerometer RMS (twin axis)

Reads:  reports/slow_wave/cap_swa/trials/trial_epochs.parquet  (from cap_swa_trials.py)
        reports/slow_wave/cap_swa/trials/trials.csv
Writes: reports/slow_wave/cap_swa/trials/examples/<session>.png

Run: .venv/Scripts/python.exe analysis/slow_wave/cap_swa_trial_examples.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS

ROOT = Path(__file__).resolve().parents[2]
TRIALS_DIR = ROOT / 'reports' / 'slow_wave' / 'cap_swa' / 'trials'
OUT = TRIALS_DIR / 'examples'
OUT.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
N3_CODE = 1
PAD_EPOCHS = 10        # 5 min of context before and after the trial
N_EXAMPLES = 3         # example trials per session
Q_HOLD = 0.50          # must match the value used to build the trials


def pick_examples(trials_sess, n=N_EXAMPLES):
    """Rank a session's trials by N3 content then duration; take the top n."""
    t = trials_sess.sort_values(['frac_N3', 'duration_min'], ascending=False)
    return t.head(n)


def plot_trial(axes, g_sess, trial, col):
    """Draw one example trial (a column of stacked panels) with pre/post pad."""
    g = g_sess.reset_index(drop=True)
    tid = trial['trial_id']
    idx = np.where(g['trial_id'].values == tid)[0]
    a, b = idx[0], idx[-1] + 1                      # trial epoch span [a, b)
    lo = max(0, a - PAD_EPOCHS)
    hi = min(len(g), b + PAD_EPOCHS)
    w = g.iloc[lo:hi]
    onset_hr = g['t_hr'].iloc[a]
    # x in minutes from trial onset
    x = (w['t_hr'].values - onset_hr) * 60.0
    x_on = 0.0
    x_off = (g['t_hr'].iloc[b - 1] - onset_hr) * 60.0 + EPOCH_SEC / 60.0

    def shade(ax):
        ax.axvspan(x_on, x_off, color='#2ECC71', alpha=0.16)
        ax.axvline(x_on, color='#27AE60', lw=1.0)
        ax.axvline(x_off, color='#27AE60', lw=1.0, ls='--')

    # ── 1. stage strip ──
    ax = axes[0]
    for _, r in w.iterrows():
        xx = (r['t_hr'] - onset_hr) * 60.0
        ax.axvspan(xx, xx + EPOCH_SEC / 60.0,
                   color=STAGE_COLORS.get(int(r['stage_code']), '#AAA'),
                   alpha=0.9 if r['stage_code'] == N3_CODE else 0.45)
    ax.axvline(x_on, color='k', lw=0.8); ax.axvline(x_off, color='k', lw=0.8, ls='--')
    ax.set_yticks([])
    stages_present = [STAGE_LABELS.get(int(s), '?') for s in w['stage_code'].unique()]
    ax.set_title(f"trial {int(tid)} · {trial['duration_min']:.1f} min · "
                 f"dom {trial['dom_stage']} · N3 {trial['frac_N3']*100:.0f}%",
                 fontsize=9)

    # ── 2. criteria sub-scores ──
    ax = axes[1]; shade(ax)
    ax.plot(x, w['swa_s_dc'], lw=1.0, label='D1 slow-DC', color='#2980B9')
    ax.plot(x, w['swa_s_thorax'], lw=1.0, label='D3 slow-thorax', color='#E67E22')
    ax.plot(x, w['swa_s_still'], lw=1.0, label='Dq quiescent', color='#16A085')
    ax.axhline(Q_HOLD, color='k', ls=':', alpha=0.6)
    ax.set_ylim(0, 1.02)
    if col == 0:
        ax.set_ylabel('criteria\nsub-scores')
        ax.legend(fontsize=6, loc='lower left', ncol=1)

    # ── 3. mean capacitance CLE-CRE ──
    ax = axes[2]; shade(ax)
    ax.plot(x, w['dc_mean'], lw=1.0, color='#34495E')
    if col == 0:
        ax.set_ylabel('CLE-CRE\nmean (D1)')

    # ── 4. heart rate ──
    ax = axes[3]; shade(ax)
    ax.plot(x, w['ecg_hr_hz'] * 60.0, lw=1.0, color='#C0392B')
    if col == 0:
        ax.set_ylabel('HR (BPM)')

    # ── 5. EEG delta ──
    ax = axes[4]; shade(ax)
    ax.plot(x, w['eeg_delta_ratio'], lw=1.0, color='#8E44AD')
    if col == 0:
        ax.set_ylabel('EEG delta')

    # ── 6. thorax + accel ──
    ax = axes[5]; shade(ax)
    ax.plot(x, w['thorax_rms'], lw=1.0, color='#27AE60', label='thorax RMS')
    ax2 = ax.twinx()
    ax2.plot(x, w['acc_rms'], lw=1.0, color='#7F8C8D', alpha=0.8, label='accel RMS')
    ax2.set_yticks([])
    if col == 0:
        ax.set_ylabel('thorax /\naccel')
    ax.set_xlabel('minutes from trial onset')


def plot_session(g_sess, trials_sess, sess):
    ex = pick_examples(trials_sess)
    n = len(ex)
    if n == 0:
        return False
    fig, axes = plt.subplots(6, n, figsize=(6.2 * n, 12), squeeze=False,
                             sharex='col')
    fig.suptitle(f'{sess} — example CAP-SWA trials with pre/post context '
                 f'({PAD_EPOCHS*EPOCH_SEC/60:.0f} min pad, green = trial)',
                 fontsize=13, fontweight='bold')
    for col, (_, tr) in enumerate(ex.iterrows()):
        plot_trial([axes[r][col] for r in range(6)], g_sess, tr, col)
    # shared stage legend
    handles = [Patch(facecolor=STAGE_COLORS.get(c, '#AAA'),
                     label=STAGE_LABELS[c]) for c in [4, 3, 2, 1, 0]]
    fig.legend(handles=handles, loc='upper right', fontsize=8, ncol=5,
               bbox_to_anchor=(0.995, 0.985))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f'{sess}.png', dpi=115, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


def main():
    epochs = pd.read_parquet(TRIALS_DIR / 'trial_epochs.parquet')
    trials = pd.read_csv(TRIALS_DIR / 'trials.csv')
    made = []
    for sess in sorted(epochs['session'].unique()):
        g = epochs[epochs['session'] == sess].sort_values('t_hr')
        ts = trials[trials['session'] == sess]
        if plot_session(g, ts, sess):
            made.append(sess)
    print(f'wrote {len(made)} session example figures -> {OUT}')
    print('sessions:', ', '.join(made))


if __name__ == '__main__':
    main()
