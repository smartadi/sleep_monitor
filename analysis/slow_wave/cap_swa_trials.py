"""
CAP-SWA TRIAL analysis (Workstream C — observational, not a classifier).

A "trial" is a sustained window where the THREE mechanical definition points all
hold at once (a conjunction, not the geometric-mean score):
    D1  slow mean-capacitance change   — |DC slope| of CLE-CRE is LOW
    D3  slow thorax amplitude change   — thorax envelope drift/variance is LOW
    Dq  quiescent                      — accelerometer RMS is LOW

Each criterion is judged per session against its own distribution: the existing
per-session percentile sub-scores (swa_s_dc / swa_s_thorax / swa_s_still, high =
favorable) must each clear a threshold q. Epochs satisfying all three, in runs of
>= MIN_EPOCHS, form trials. We then OBSERVE what the trials look like — we do not
fit anything.

Characteristics reported:
  - Trial inventory: count, duration distribution, per session/subject
  - Sleep-stage composition of trial epochs + N3 enrichment (lift) and coverage
  - Trial vs non-trial physiology contrasts (HR, RR, EEG delta, k, cardiac freq),
    per-subject direction counts (n=6 — direction is the evidence)
  - Onset-triggered averages (+/-5 min around trial onset): does a trial mark
    entry into a deep, bradycardic, delta-rich state?
  - Time-of-night placement of trials
  - Sensitivity of trial yield to the conjunction threshold q

Reads:  reports/slow_wave/cap_swa/all_epoch_features.parquet   (no re-extraction)
Writes: reports/slow_wave/cap_swa/trials/
  trials.csv                 one row per trial (timing, duration, stage mix, physiology)
  trial_epochs.parquet       per-epoch table with trial id + in-trial flag
  stage_composition.csv      stage mix + N3 enrichment/coverage per q
  physiology_contrasts.csv   trial vs non-trial per-feature, per-subject direction
  onset_triggered.csv        mean +/- sem trace of each feature around trial onset
  cap_swa_trials.png         summary figure
  per_session/<sess>.png     hypnogram + 3 criteria + trial shading

Run: .venv/Scripts/python.exe analysis/slow_wave/cap_swa_trials.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS

ROOT = Path(__file__).resolve().parents[2]
CAP_SWA = ROOT / 'reports' / 'slow_wave' / 'cap_swa'
OUT = CAP_SWA / 'trials'
(OUT / 'per_session').mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
N3_CODE = 1
Q_HOLD = 0.50          # each per-session sub-score must clear this to "hold"
MIN_EPOCHS = 4         # sustained >= 2 min to be a trial
Q_SWEEP = [0.40, 0.50, 0.60, 0.70]
ONSET_HALF = 10        # +/- epochs (5 min) for onset-triggered averages

SUBS = ['swa_s_dc', 'swa_s_thorax', 'swa_s_still']
PHYS = {
    'ecg_hr_hz':          ('Heart rate', 60.0, 'BPM'),
    'flow_rr_hz':         ('Resp rate', 60.0, 'br/min'),
    'eeg_delta_ratio':    ('EEG delta', 1.0, 'ratio'),
    'k_resp_dev':         ('CAP-thorax rate dev', 1.0, 'Hz'),
    'cap_card_hz':        ('CAP cardiac freq', 1.0, 'Hz'),
    'card_freq_divergence': ('PPG-CAP card div', 1.0, 'Hz'),
    'thorax_rms':         ('Thorax amplitude', 1.0, 'a.u.'),
    'acc_rms':            ('Accel RMS', 1.0, 'a.u.'),
}


# ── Trial detection ──────────────────────────────────────────────────────────

def holds_mask(g, q):
    """Boolean per-epoch: all three sub-scores clear q (all criteria hold)."""
    ok = np.ones(len(g), dtype=bool)
    for c in SUBS:
        ok &= (g[c].values >= q)
    ok &= np.isfinite(g[SUBS].values).all(axis=1)
    return ok


def runs(mask, min_len):
    """Yield (start, stop) index runs of True with length >= min_len."""
    i, n = 0, len(mask)
    out = []
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def build_trials(df, q=Q_HOLD, min_epochs=MIN_EPOCHS):
    """Per-session conjunction trials → per-trial table + per-epoch trial ids."""
    trial_rows, epoch_frames = [], []
    tid = 0
    for (sess, subj), g in df.groupby(['session', 'subject'], sort=False):
        g = g.sort_values('t_hr').reset_index(drop=True)
        mask = holds_mask(g, q)
        g = g.assign(in_trial=mask, trial_id=-1)
        for (a, b) in runs(mask, min_epochs):
            seg = g.iloc[a:b]
            g.loc[a:b - 1, 'trial_id'] = tid
            stage_frac = seg['stage_code'].value_counts(normalize=True).to_dict()
            dom = seg['stage_code'].mode()
            row = dict(
                trial_id=tid, session=sess, subject=subj,
                t_start_hr=float(seg['t_hr'].iloc[0]),
                n_epochs=int(b - a), duration_min=(b - a) * EPOCH_SEC / 60.0,
                dom_stage=STAGE_LABELS.get(int(dom.iloc[0]), '?') if len(dom) else '?',
                frac_N3=stage_frac.get(N3_CODE, 0.0),
                frac_N2=stage_frac.get(2, 0.0),
                frac_REM=stage_frac.get(0, 0.0),
                frac_Wake=stage_frac.get(4, 0.0),
                frac_N1=stage_frac.get(3, 0.0),
            )
            for f in PHYS:
                row[f'{f}_mean'] = float(seg[f].mean()) if f in seg else np.nan
            trial_rows.append(row)
            tid += 1
        epoch_frames.append(g)
    trials = pd.DataFrame(trial_rows)
    epochs = pd.concat(epoch_frames, ignore_index=True)
    return trials, epochs


# ── Characterization ─────────────────────────────────────────────────────────

def stage_composition(epochs, q):
    """Stage mix of trial epochs + N3 enrichment (lift) and coverage."""
    scored = epochs[epochs['stage_code'] >= 0]
    intr = scored[scored['in_trial']]
    rows = []
    base_rates = scored['stage_code'].value_counts(normalize=True)
    for code, lab in STAGE_LABELS.items():
        if code < 0:
            continue
        in_frac = (intr['stage_code'] == code).mean() if len(intr) else np.nan
        base = base_rates.get(code, 0.0)
        # coverage: fraction of this stage's epochs that fall in a trial
        stg = scored[scored['stage_code'] == code]
        cover = stg['in_trial'].mean() if len(stg) else np.nan
        rows.append(dict(q=q, stage=lab, trial_epoch_frac=in_frac,
                         base_rate=base, enrichment=(in_frac / base if base else np.nan),
                         stage_coverage_by_trials=cover))
    return pd.DataFrame(rows)


def physiology_contrasts(epochs):
    """Per-subject trial vs non-trial median contrast + direction counts."""
    scored = epochs[epochs['stage_code'] >= 0]
    rows = []
    for f, (name, scale, unit) in PHYS.items():
        deltas = []
        for subj, g in scored.groupby('subject'):
            a = g.loc[g['in_trial'], f].dropna()
            b = g.loc[~g['in_trial'], f].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            deltas.append((a.median() - b.median()) * scale)
        deltas = np.array(deltas)
        if len(deltas) == 0:
            continue
        n_up = int((deltas > 0).sum())
        rows.append(dict(feature=f, name=name, unit=unit, n_subjects=len(deltas),
                         median_delta=float(np.median(deltas)),
                         n_increase=n_up,
                         consistency=f'{max(n_up, len(deltas)-n_up)}/{len(deltas)}'))
    return pd.DataFrame(rows)


def onset_triggered(epochs, half=ONSET_HALF):
    """Mean±sem of each feature in [-half,+half] epochs around trial onsets.
    Per-session z-scored so subjects combine on equal footing."""
    feats = list(PHYS.keys())
    stacks = {f: [] for f in feats}
    for sess, g in epochs.groupby('session', sort=False):
        g = g.sort_values('t_hr').reset_index(drop=True)
        tid = g['trial_id'].values
        # onset = first epoch of each trial run
        onsets = [i for i in range(len(g))
                  if tid[i] >= 0 and (i == 0 or tid[i - 1] != tid[i])]
        zg = {}
        for f in feats:
            v = g[f].values.astype(float)
            mu, sd = np.nanmean(v), np.nanstd(v)
            zg[f] = (v - mu) / sd if sd > 1e-9 else v * np.nan
        for o in onsets:
            lo, hi = o - half, o + half + 1
            if lo < 0 or hi > len(g):
                continue
            for f in feats:
                stacks[f].append(zg[f][lo:hi])
    rows = []
    lags = np.arange(-half, half + 1)
    for f in feats:
        arr = np.array(stacks[f]) if stacks[f] else np.empty((0, 2 * half + 1))
        if len(arr) == 0:
            continue
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(max(1, arr.shape[0]))
        for k, lag in enumerate(lags):
            rows.append(dict(feature=f, name=PHYS[f][0], lag_epoch=int(lag),
                             lag_min=lag * EPOCH_SEC / 60.0, n=arr.shape[0],
                             z_mean=float(mean[k]), z_sem=float(sem[k])))
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_summary(trials, comp, contrasts, onset):
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f'CAP-SWA trials — windows where all three mechanical criteria hold '
                 f'(q={Q_HOLD}, ≥{MIN_EPOCHS} epochs): {len(trials)} trials',
                 fontsize=14, fontweight='bold')

    # A: duration distribution
    ax = axes[0, 0]
    ax.hist(trials['duration_min'], bins=30, color='#16A085', alpha=0.85)
    ax.set_xlabel('trial duration (min)'); ax.set_ylabel('# trials')
    ax.set_title(f'A. Trial durations (median {trials["duration_min"].median():.1f} min)')

    # B: stage composition of trial epochs + enrichment
    ax = axes[0, 1]
    c = comp[comp['q'] == Q_HOLD].set_index('stage')
    order = [s for s in ['Wake', 'N1', 'N2', 'N3', 'REM'] if s in c.index]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, c.loc[order, 'trial_epoch_frac'], w, color='#16A085',
           label='in trials')
    ax.bar(x + w/2, c.loc[order, 'base_rate'], w, color='#95A5A6', label='overall')
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel('fraction of epochs'); ax.set_title('B. Stage mix: trials vs night')
    ax.legend(fontsize=8)
    for i, s in enumerate(order):
        e = c.loc[s, 'enrichment']
        ax.text(i, max(c.loc[s, 'trial_epoch_frac'], c.loc[s, 'base_rate']) + 0.01,
                f'{e:.1f}×', ha='center', fontsize=8)

    # C: stage coverage by trials
    ax = axes[0, 2]
    ax.bar(x, c.loc[order, 'stage_coverage_by_trials'],
           color=[STAGE_COLORS.get({'Wake':4,'N1':3,'N2':2,'N3':1,'REM':0}[s], '#888')
                  for s in order])
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel('fraction of stage epochs inside a trial')
    ax.set_title('C. How much of each stage the trials capture')

    # D: physiology contrasts (trial - non-trial), per-subject direction
    ax = axes[1, 0]
    cc = contrasts.copy()
    ax.barh(np.arange(len(cc)), cc['median_delta'],
            color=['#C0392B' if d < 0 else '#2980B9' for d in cc['median_delta']])
    ax.set_yticks(np.arange(len(cc)))
    ax.set_yticklabels([f'{r["name"]} ({r["consistency"]})' for _, r in cc.iterrows()],
                       fontsize=8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('median (trial − non-trial), native units')
    ax.set_title('D. Physiology during trials (per-subject direction)')
    ax.invert_yaxis()

    # E: onset-triggered HR + EEG delta + acc
    ax = axes[1, 1]
    for f, col in [('ecg_hr_hz', '#C0392B'), ('eeg_delta_ratio', '#8E44AD'),
                   ('acc_rms', '#7F8C8D'), ('thorax_rms', '#27AE60')]:
        d = onset[onset['feature'] == f].sort_values('lag_min')
        if len(d):
            ax.plot(d['lag_min'], d['z_mean'], label=PHYS[f][0])
            ax.fill_between(d['lag_min'], d['z_mean'] - d['z_sem'],
                            d['z_mean'] + d['z_sem'], alpha=0.15)
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('minutes from trial onset'); ax.set_ylabel('z (per session)')
    ax.set_title('E. Onset-triggered averages'); ax.legend(fontsize=7)

    # F: time-of-night placement
    ax = axes[1, 2]
    # normalize each session's trial onset to fraction of that session's span
    fr = []
    for sess, g in trials.groupby('session'):
        fr.extend(g['t_start_hr'].values - g['t_start_hr'].min())
    ax.hist(trials['t_start_hr'], bins=24, color='#E67E22', alpha=0.85)
    ax.set_xlabel('trial onset (clock hour into recording)')
    ax.set_ylabel('# trials'); ax.set_title('F. When trials occur')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / 'cap_swa_trials.png', dpi=120, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def plot_session(epochs, sess):
    g = epochs[epochs['session'] == sess].sort_values('t_hr')
    if g.empty:
        return
    t = g['t_hr'].values
    fig, axes = plt.subplots(4, 1, figsize=(18, 9), sharex=True)
    fig.suptitle(f'{sess} — CAP-SWA trials (shaded) over the 3 criteria + hypnogram',
                 fontsize=12, fontweight='bold')
    trial_t = t[g['in_trial'].values]
    for ax in axes:
        for tt in trial_t:
            ax.axvspan(tt, tt + EPOCH_SEC / 3600, color='#2ECC71', alpha=0.14)
    ax = axes[0]
    for _, r in g.iterrows():
        ax.axvspan(r['t_hr'], r['t_hr'] + EPOCH_SEC / 3600,
                   color=STAGE_COLORS.get(int(r['stage_code']), '#AAA'),
                   alpha=0.85 if r['stage_code'] == N3_CODE else 0.4)
    ax.set_ylabel('stage'); ax.set_yticks([])
    axes[1].plot(t, g['swa_s_dc'], lw=0.8, label='D1 slow-DC')
    axes[1].plot(t, g['swa_s_thorax'], lw=0.8, label='D3 slow-thorax')
    axes[1].plot(t, g['swa_s_still'], lw=0.8, label='Dq quiescent')
    axes[1].axhline(Q_HOLD, color='k', ls='--', alpha=0.4)
    axes[1].set_ylabel('sub-scores'); axes[1].legend(fontsize=7, ncol=3)
    axes[2].plot(t, g['ecg_hr_hz'] * 60, color='#C0392B', lw=0.8)
    axes[2].set_ylabel('HR (BPM)'); axes[2].set_ylim(35, 100)
    axes[3].plot(t, g['eeg_delta_ratio'], color='#8E44AD', lw=0.8)
    axes[3].set_ylabel('EEG delta'); axes[3].set_xlabel('time (hr)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / 'per_session' / f'{sess}.png', dpi=110,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_parquet(CAP_SWA / 'all_epoch_features.parquet')
    print(f'loaded {len(df)} epochs, {df["session"].nunique()} sessions')

    trials, epochs = build_trials(df, q=Q_HOLD)
    trials.to_csv(OUT / 'trials.csv', index=False)
    epochs.to_parquet(OUT / 'trial_epochs.parquet')
    print(f'\n{len(trials)} trials at q={Q_HOLD}, min {MIN_EPOCHS} epochs')
    print(f'  total trial epochs: {int(epochs["in_trial"].sum())} '
          f'({100*epochs["in_trial"].mean():.1f}% of all epochs)')
    print(f'  median duration: {trials["duration_min"].median():.1f} min '
          f'(max {trials["duration_min"].max():.1f})')
    print(f'  dominant stage of trials: '
          f'{trials["dom_stage"].value_counts().to_dict()}')

    # stage composition across the q sweep
    comp_all = []
    for q in Q_SWEEP:
        _, ep_q = build_trials(df, q=q)
        comp_all.append(stage_composition(ep_q, q))
        n_tr = ep_q.loc[ep_q['trial_id'] >= 0, 'trial_id'].nunique()
        print(f'  q={q}: {int(ep_q["in_trial"].sum())} trial epochs, {n_tr} trials')
    comp = pd.concat(comp_all, ignore_index=True)
    comp.to_csv(OUT / 'stage_composition.csv', index=False)

    contrasts = physiology_contrasts(epochs)
    contrasts.to_csv(OUT / 'physiology_contrasts.csv', index=False)
    print('\nphysiology (trial minus non-trial), per-subject direction:')
    for _, r in contrasts.iterrows():
        print(f'  {r["name"]:22s} d={r["median_delta"]:+.4f} {r["unit"]:7s} '
              f'{r["consistency"]}')

    onset = onset_triggered(epochs)
    onset.to_csv(OUT / 'onset_triggered.csv', index=False)

    plot_summary(trials, comp, contrasts, onset)
    for sess in sorted(df['session'].unique()):
        plot_session(epochs, sess)
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
