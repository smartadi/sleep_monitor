"""
CAP-SWA TRIAL analysis (Workstream C — observational).

Trials use the user's EXACT three-point definition:
    C1  slow mean drift of a SINGLE capacitive channel (CLE, CRE, or CH) —
        NOT the CLE-CRE difference.  "or" = the criterion is met when the
        slowest-drifting of the three channels is flat (low |DC slope|).
    C2  the trial is INITIATED BY A HEAD MOVEMENT — a distinct accelerometer
        movement event occurs in the epochs just before the window starts.
    C3  low-variance, slowly-changing thorax amplitude.

Note vs the earlier version: (a) single-channel DC drift replaces the CLE-CRE
differential; (b) quiescence-throughout (low accel RMS) is REPLACED by the
movement-initiation trigger — the accelerometer now marks the onset, it is not a
hold condition. C1 and C3 must hold (as per-session percentile sub-scores >= q) in
a sustained run of >= MIN_EPOCHS, AND a head movement must precede the run onset
within MOVE_LOOKBACK epochs.

Because single-channel DC and movement events are not in the cap_swa parquet, this
recomputes per-epoch quantities from the raw sessions, then merges physiology
(HR, RR, EEG delta, ...) from all_epoch_features.parquet on (session, epoch_idx)
for characterization.

Writes: reports/slow_wave/cap_swa/trials/
  trials.csv, trial_epochs.parquet, stage_composition.csv,
  physiology_contrasts.csv, onset_triggered.csv, cap_swa_trials.png
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
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import (PSG_EPOCH_SEC, RESP_LO, RESP_HI,
                                  STAGE_LABELS, STAGE_COLORS)
from sleep_monitor.filters import bandpass
from sleep_monitor.sessions import SESSION_META
from analysis.slow_wave.cap_swa_definition import detect_movements  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CAP_SWA = ROOT / 'reports' / 'slow_wave' / 'cap_swa'
OUT = CAP_SWA / 'trials'
(OUT / 'per_session').mkdir(parents=True, exist_ok=True)

EPOCH_SEC = PSG_EPOCH_SEC       # 30 s
N3_CODE = 1
ROLL_W = 5                      # epochs in the rolling drift/variance window (2.5 min)
Q_HOLD = 0.50                   # per-session sub-score cut for C1 and C3
MIN_EPOCHS = 4                  # sustained >= 2 min
MOVE_LOOKBACK = 3               # a head movement within this many epochs before onset (C2)
Q_SWEEP = [0.40, 0.50, 0.60, 0.70]
ONSET_HALF = 10                 # +/- epochs for onset-triggered averages

PHYS = {
    'ecg_hr_hz':            ('Heart rate', 60.0, 'BPM'),
    'flow_rr_hz':           ('Resp rate', 60.0, 'br/min'),
    'eeg_delta_ratio':      ('EEG delta', 1.0, 'ratio'),
    'k_resp_dev':           ('CAP-thorax rate dev', 1.0, 'Hz'),
    'cap_card_hz':          ('CAP cardiac freq', 1.0, 'Hz'),
    'card_freq_divergence': ('PPG-CAP card div', 1.0, 'Hz'),
    'thorax_rms':           ('Thorax amplitude', 1.0, 'a.u.'),
    'acc_rms':              ('Accel RMS', 1.0, 'a.u.'),
}


# ── Per-epoch quantities from raw signal ─────────────────────────────────────

def _inv_pct(x):
    """Low raw value -> high score in [0,1] (per-session percentile, inverted)."""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    v = np.isfinite(x)
    if v.sum() < 3:
        return out
    out[v] = 1.0 - pd.Series(x[v]).rank(pct=True).values
    return out


def epoch_quantities(idx):
    """One row per epoch: single-channel DC means, thorax RMS, accel RMS, movements."""
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    sp = session.sleep_profile
    fs = session.fs
    t_hr = session.time_hr
    epoch_n = int(EPOCH_SEC * fs)
    label = session.label

    cle = session.cap['CLE'].astype(np.float64)
    cre = session.cap['CRE'].astype(np.float64)
    ch = session.cap['CH'].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)
    thorax = session.psg['Thorax'].astype(np.float64)

    rows = []
    for ei in range(len(sp['codes'])):
        t0 = sp['t_ep_hr'][ei]
        t1 = t0 + EPOCH_SEC / 3600.0
        m = (t_hr >= t0) & (t_hr < t1)
        if m.sum() < epoch_n * 0.5:
            continue
        j = np.where(m)[0]
        thb = bandpass(thorax[j], RESP_LO, RESP_HI, fs)
        rows.append(dict(
            session=label, subject=SESSION_META[idx]['subject'], epoch_idx=ei,
            t_hr=float(t0), stage_code=int(sp['codes'][ei]),
            cle_mean=float(np.mean(cle[j])), cre_mean=float(np.mean(cre[j])),
            ch_mean=float(np.mean(ch[j])),
            thorax_rms=float(np.sqrt(np.mean(thb ** 2))),
            acc_rms=float(np.sqrt(np.mean((acc[j] - np.mean(acc[j])) ** 2))),
        ))
    df = pd.DataFrame(rows).reset_index(drop=True)

    # rolling |slope| per single channel; C1 = slowest channel is flat
    slopes = {c: [] for c in ['cle_mean', 'cre_mean', 'ch_mean']}
    thx_std = []
    for i in range(len(df)):
        lo = max(0, i - ROLL_W // 2)
        hi = min(len(df), i + ROLL_W // 2 + 1)
        x = np.arange(hi - lo, dtype=float)
        for c in slopes:
            v = df[c].iloc[lo:hi].values
            slopes[c].append(abs(float(np.polyfit(x, v, 1)[0])) if hi - lo >= 3 else np.nan)
        tv = df['thorax_rms'].iloc[lo:hi].values
        thx_std.append(float(np.std(tv)) if hi - lo >= 3 else np.nan)
    for c in slopes:
        df[f'{c}_slope'] = slopes[c]
    # C1 raw = minimum single-channel |slope| ("or": any channel flat qualifies)
    df['drift_min_slope'] = df[['cle_mean_slope', 'cre_mean_slope', 'ch_mean_slope']].min(axis=1)
    df['thorax_std'] = thx_std              # C3 raw: low std = low-var & slow

    df['c1_slow_drift'] = _inv_pct(df['drift_min_slope'].values)
    df['c3_slow_thorax'] = _inv_pct(df['thorax_std'].values)

    # head-movement events -> epoch flags
    mv = detect_movements(acc, fs)
    mv_t_hr = mv / fs / 3600.0
    move_epoch = np.zeros(len(df), dtype=bool)
    for mt in mv_t_hr:
        k = np.argmin(np.abs(df['t_hr'].values - mt))
        if abs(df['t_hr'].values[k] - mt) <= EPOCH_SEC / 3600.0:
            move_epoch[k] = True
    df['head_move'] = move_epoch
    return df


# ── Trial building (C1 & C3 hold, C2 movement-initiated) ─────────────────────

def runs(mask, min_len):
    out, i, n = [], 0, len(mask)
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


def build_trials(feat, q=Q_HOLD, min_epochs=MIN_EPOCHS, lookback=MOVE_LOOKBACK):
    trial_rows, epoch_frames = [], []
    tid = 0
    for (sess, subj), g in feat.groupby(['session', 'subject'], sort=False):
        g = g.sort_values('t_hr').reset_index(drop=True)
        holds = ((g['c1_slow_drift'].values >= q) &
                 (g['c3_slow_thorax'].values >= q) &
                 np.isfinite(g['c1_slow_drift'].values) &
                 np.isfinite(g['c3_slow_thorax'].values))
        move = g['head_move'].values
        g = g.assign(in_trial=False, trial_id=-1, holds=holds)
        for (a, b) in runs(holds, min_epochs):
            # C2: a head movement in the lookback epochs just before onset
            lo = max(0, a - lookback)
            initiated = bool(move[lo:a + 1].any())
            if not initiated:
                continue
            g.loc[a:b - 1, 'in_trial'] = True
            g.loc[a:b - 1, 'trial_id'] = tid
            seg = g.iloc[a:b]
            frac = seg['stage_code'].value_counts(normalize=True).to_dict()
            dom = seg['stage_code'].mode()
            trial_rows.append(dict(
                trial_id=tid, session=sess, subject=subj,
                t_start_hr=float(seg['t_hr'].iloc[0]),
                n_epochs=int(b - a), duration_min=(b - a) * EPOCH_SEC / 60.0,
                move_epoch_before=int(a - lo - np.argmax(move[lo:a + 1][::-1]))
                if move[lo:a + 1].any() else -1,
                dom_stage=STAGE_LABELS.get(int(dom.iloc[0]), '?') if len(dom) else '?',
                frac_N3=frac.get(N3_CODE, 0.0), frac_N2=frac.get(2, 0.0),
                frac_N1=frac.get(3, 0.0), frac_REM=frac.get(0, 0.0),
                frac_Wake=frac.get(4, 0.0)))
            tid += 1
        epoch_frames.append(g)
    trials = pd.DataFrame(trial_rows)
    epochs = pd.concat(epoch_frames, ignore_index=True)
    return trials, epochs


# ── Characterization ─────────────────────────────────────────────────────────

def stage_composition(epochs, q):
    scored = epochs[epochs['stage_code'] >= 0]
    intr = scored[scored['in_trial']]
    base = scored['stage_code'].value_counts(normalize=True)
    rows = []
    for code, lab in STAGE_LABELS.items():
        if code < 0:
            continue
        inf = (intr['stage_code'] == code).mean() if len(intr) else np.nan
        b = base.get(code, 0.0)
        stg = scored[scored['stage_code'] == code]
        rows.append(dict(q=q, stage=lab, trial_epoch_frac=inf, base_rate=b,
                         enrichment=(inf / b if b else np.nan),
                         stage_coverage_by_trials=(stg['in_trial'].mean() if len(stg) else np.nan)))
    return pd.DataFrame(rows)


def physiology_contrasts(epochs):
    scored = epochs[epochs['stage_code'] >= 0]
    rows = []
    for f, (name, scale, unit) in PHYS.items():
        if f not in scored:
            continue
        deltas = []
        for subj, g in scored.groupby('subject'):
            a = g.loc[g['in_trial'], f].dropna()
            b = g.loc[~g['in_trial'], f].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            deltas.append((a.median() - b.median()) * scale)
        deltas = np.array(deltas)
        if not len(deltas):
            continue
        n_up = int((deltas > 0).sum())
        rows.append(dict(feature=f, name=name, unit=unit, n_subjects=len(deltas),
                         median_delta=float(np.median(deltas)), n_increase=n_up,
                         consistency=f'{max(n_up, len(deltas)-n_up)}/{len(deltas)}'))
    return pd.DataFrame(rows)


def onset_triggered(epochs, half=ONSET_HALF):
    feats = [f for f in PHYS if f in epochs.columns] + ['acc_rms']
    feats = list(dict.fromkeys(feats))
    stacks = {f: [] for f in feats}
    for sess, g in epochs.groupby('session', sort=False):
        g = g.sort_values('t_hr').reset_index(drop=True)
        tid = g['trial_id'].values
        onsets = [i for i in range(len(g))
                  if tid[i] >= 0 and (i == 0 or tid[i - 1] != tid[i])]
        z = {}
        for f in feats:
            v = g[f].values.astype(float)
            mu, sd = np.nanmean(v), np.nanstd(v)
            z[f] = (v - mu) / sd if sd > 1e-9 else v * np.nan
        for o in onsets:
            if o - half < 0 or o + half + 1 > len(g):
                continue
            for f in feats:
                stacks[f].append(z[f][o - half:o + half + 1])
    rows, lags = [], np.arange(-half, half + 1)
    for f in feats:
        arr = np.array(stacks[f]) if stacks[f] else np.empty((0, 2 * half + 1))
        if not len(arr):
            continue
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(max(1, arr.shape[0]))
        for k, lag in enumerate(lags):
            rows.append(dict(feature=f, name=PHYS.get(f, (f,))[0], lag_min=lag * EPOCH_SEC / 60.0,
                             n=arr.shape[0], z_mean=float(mean[k]), z_sem=float(sem[k])))
    return pd.DataFrame(rows)


# ── Summary plot ─────────────────────────────────────────────────────────────

def plot_summary(trials, comp, contrasts, onset):
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f'CAP-SWA trials — movement-initiated, single-channel slow-drift + '
                 f'slow-thorax (q={Q_HOLD}, ≥{MIN_EPOCHS} ep): {len(trials)} trials',
                 fontsize=14, fontweight='bold')
    ax = axes[0, 0]
    if len(trials):
        ax.hist(trials['duration_min'], bins=25, color='#16A085', alpha=0.85)
        ax.set_title(f'A. Durations (median {trials["duration_min"].median():.1f} min)')
    ax.set_xlabel('trial duration (min)'); ax.set_ylabel('# trials')

    ax = axes[0, 1]
    c = comp[comp['q'] == Q_HOLD].set_index('stage')
    order = [s for s in ['Wake', 'N1', 'N2', 'N3', 'REM'] if s in c.index]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, c.loc[order, 'trial_epoch_frac'], w, color='#16A085', label='in trials')
    ax.bar(x + w/2, c.loc[order, 'base_rate'], w, color='#95A5A6', label='overall')
    for i, s in enumerate(order):
        e = c.loc[s, 'enrichment']
        ax.text(i, max(c.loc[s, 'trial_epoch_frac'], c.loc[s, 'base_rate']) + 0.01,
                f'{e:.1f}×', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel('fraction of epochs')
    ax.set_title('B. Stage mix: trials vs night'); ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.bar(x, c.loc[order, 'stage_coverage_by_trials'],
           color=[STAGE_COLORS.get({'Wake':4,'N1':3,'N2':2,'N3':1,'REM':0}[s], '#888') for s in order])
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel('fraction of stage inside a trial')
    ax.set_title('C. How much of each stage the trials capture')

    ax = axes[1, 0]
    if len(contrasts):
        ax.barh(np.arange(len(contrasts)), contrasts['median_delta'],
                color=['#C0392B' if d < 0 else '#2980B9' for d in contrasts['median_delta']])
        ax.set_yticks(np.arange(len(contrasts)))
        ax.set_yticklabels([f'{r["name"]} ({r["consistency"]})' for _, r in contrasts.iterrows()],
                           fontsize=8)
        ax.axvline(0, color='k', lw=0.6); ax.invert_yaxis()
    ax.set_xlabel('median (trial − non-trial)')
    ax.set_title('D. Physiology during trials (per-subject direction)')

    ax = axes[1, 1]
    for f, col in [('ecg_hr_hz', '#C0392B'), ('eeg_delta_ratio', '#8E44AD'),
                   ('acc_rms', '#7F8C8D'), ('thorax_rms', '#27AE60')]:
        d = onset[onset['feature'] == f].sort_values('lag_min')
        if len(d):
            ax.plot(d['lag_min'], d['z_mean'], label=PHYS.get(f, (f,))[0])
            ax.fill_between(d['lag_min'], d['z_mean'] - d['z_sem'], d['z_mean'] + d['z_sem'], alpha=0.15)
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('minutes from trial onset'); ax.set_ylabel('z (per session)')
    ax.set_title('E. Onset-triggered averages'); ax.legend(fontsize=7)

    ax = axes[1, 2]
    if len(trials):
        ax.hist(trials['t_start_hr'], bins=24, color='#E67E22', alpha=0.85)
    ax.set_xlabel('trial onset (hour into recording)'); ax.set_ylabel('# trials')
    ax.set_title('F. When trials occur')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / 'cap_swa_trials.png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('recomputing single-channel DC + movements per session...')
    feats = []
    for idx in range(12):
        f = epoch_quantities(idx)
        feats.append(f)
        print(f"  {f['session'].iloc[0]}: {len(f)} epochs, "
              f"{int(f['head_move'].sum())} head-move epochs")
    feat = pd.concat(feats, ignore_index=True)

    # merge physiology from the aligned cap_swa parquet (same epoch loop -> same idx)
    phys = pd.read_parquet(CAP_SWA / 'all_epoch_features.parquet')
    keep = ['session', 'epoch_idx'] + [c for c in PHYS if c in phys.columns and c != 'acc_rms']
    m = feat.merge(phys[keep], on=['session', 'epoch_idx'], how='left', suffixes=('', '_p'))
    # sanity: stage codes align
    chk = phys[['session', 'epoch_idx', 'stage_code']].rename(columns={'stage_code': 'sc_p'})
    m2 = m.merge(chk, on=['session', 'epoch_idx'], how='left')
    mism = (m2['stage_code'] != m2['sc_p']).mean()
    print(f"  stage-label alignment with parquet: {(1-mism)*100:.1f}% agree")
    feat = m

    trials, epochs = build_trials(feat, q=Q_HOLD)
    trials.to_csv(OUT / 'trials.csv', index=False)
    epochs.to_parquet(OUT / 'trial_epochs.parquet')
    print(f"\n{len(trials)} movement-initiated trials at q={Q_HOLD}")
    if len(trials):
        print(f"  trial epochs: {int(epochs['in_trial'].sum())} "
              f"({100*epochs['in_trial'].mean():.1f}% of all)")
        print(f"  median duration: {trials['duration_min'].median():.1f} min")
        print(f"  dominant stage: {trials['dom_stage'].value_counts().to_dict()}")

    comp_all = []
    for q in Q_SWEEP:
        _, ep_q = build_trials(feat, q=q)
        comp_all.append(stage_composition(ep_q, q))
        nt = ep_q.loc[ep_q['trial_id'] >= 0, 'trial_id'].nunique()
        print(f"  q={q}: {int(ep_q['in_trial'].sum())} trial epochs, {nt} trials")
    comp = pd.concat(comp_all, ignore_index=True)
    comp.to_csv(OUT / 'stage_composition.csv', index=False)

    contrasts = physiology_contrasts(epochs)
    contrasts.to_csv(OUT / 'physiology_contrasts.csv', index=False)
    print('\nphysiology (trial minus non-trial), per-subject direction:')
    for _, r in contrasts.iterrows():
        print(f'  {r["name"]:22s} d={r["median_delta"]:+.4f} {r["unit"]:7s} {r["consistency"]}')

    onset = onset_triggered(epochs)
    onset.to_csv(OUT / 'onset_triggered.csv', index=False)
    plot_summary(trials, comp, contrasts, onset)
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
