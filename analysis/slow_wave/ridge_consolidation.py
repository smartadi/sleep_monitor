"""
Ridge consolidation for the paper (Workstream A).

Three goals:
  1. FLATNESS — detect and quantify flat (near-constant-frequency) ridges, using
     the new per-ridge flatness metrics in sleep_monitor.harmonics.
  2. CONSISTENCY — compare the baseline ridge-detection config against a
     "flat-favoring" config (tighter continuity, longer persistence, more
     smoothing) and report flat-ridge yield + cross-session reproducibility.
  3. SECONDARY SLEEP STATE — the N3 story is weak (AUC~0.53). Test every ridge
     feature against ALL FIVE stages (one-vs-rest AUC + KW + per-subject
     direction) and report whichever stage the ridge structure best marks.

Channel: CRE (dominant ridge channel in 9/12 sessions per manuscript).

Outputs -> reports/slow_wave/ridge_consolidation/
  per_ridge.csv            one row per detected ridge (flatness, duration, dominant stage)
  per_epoch_features.parquet   per-window ridge features aligned to PSG stage
  stage_association.csv    KW + per-stage one-vs-rest AUC for every feature
  retune_comparison.csv    baseline vs flat-favoring config
  ridge_consolidation.png  flatness-by-stage + AUC heatmap summary

Run:
  python ridge_consolidation.py --session 0
  python ridge_consolidation.py --all
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import FS, STAGE_LABELS, STAGE_ORDER, STAGE_COLORS
from sleep_monitor.harmonics import detect_persistent_ridges
from sleep_monitor.sessions import SESSION_META

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'ridge_consolidation'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL = 'CRE'
WIN_SEC = 30.0
STEP_SEC = 30.0
FLAT_THRESHOLD = 0.95   # flatness >= this (freq CV < ~0.05) counts as a "flat" ridge

# Detection configs compared for consistency. welch_seg_sec=20 gives ~0.05 Hz
# bins (vs the 0.125 Hz default) so flat-vs-wandering ridges are distinguishable;
# the flat-favoring config additionally tightens continuity, lengthens the
# minimum persistence, and smooths more.
BASELINE_CFG = dict(max_freq_jump=0.12, min_persistence_sec=300.0,
                    smooth_windows=5, welch_seg_sec=20.0)
FLAT_CFG     = dict(max_freq_jump=0.08, min_persistence_sec=420.0,
                    smooth_windows=7, welch_seg_sec=20.0)

RIDGE_FEATURES = ['n_active_ridges', 'min_ridge_freq', 'total_ridge_power',
                  'freq_spread', 'mean_flatness', 'n_flat_ridges']


def _get_signal(session, channel):
    if channel == 'CLE-CRE':
        return (session.cap['CLE'].astype(np.float64)
                - session.cap['CRE'].astype(np.float64))
    return session.cap[channel].astype(np.float64)


def _stage_at(sp, t_hr):
    """Stage code at time t_hr (hours)."""
    idx = np.searchsorted(sp['t_ep_hr'], t_hr, side='right') - 1
    if 0 <= idx < len(sp['codes']):
        return int(sp['codes'][idx])
    return -1


def per_ridge_table(rr, sp, session_label, subject):
    """One row per ridge: flatness metrics + dominant stage over its lifetime."""
    t_hr = rr['t_hr']
    rows = []
    for r in rr['ridges']:
        si, ei = r['start_idx'], r['end_idx']
        present = np.where(np.isfinite(r['freq_trace']))[0]
        stages = [_stage_at(sp, t_hr[i]) for i in present]
        stages = [s for s in stages if s >= 0]
        if stages:
            vals, cnts = np.unique(stages, return_counts=True)
            dom = int(vals[np.argmax(cnts)])
            n3_frac = float(np.mean(np.array(stages) == 1))
        else:
            dom, n3_frac = -1, 0.0
        rows.append(dict(
            session=session_label, subject=subject,
            median_freq=r['median_freq'], duration_sec=r['duration_sec'],
            flatness=r.get('flatness', np.nan), freq_cv=r.get('freq_cv', np.nan),
            freq_std=r.get('freq_std', np.nan), drift_slope=r.get('drift_slope', np.nan),
            coverage=r.get('coverage', np.nan),
            peak_prominence=r.get('peak_prominence', np.nan),
            median_prominence=r.get('median_prominence', np.nan),
            dominant_stage=STAGE_LABELS.get(dom, '?'), dominant_stage_code=dom,
            n3_fraction=n3_frac,
        ))
    return pd.DataFrame(rows)


def per_epoch_features(rr, sp, session_label, subject):
    """Per-window ridge features aligned to the PSG stage."""
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    n_win = len(t_hr)
    rows = []
    for i in range(n_win):
        if rr['motion_mask'][i]:
            continue
        active = [(r['freq_trace'][i], r['amp_trace'][i], r.get('flatness', np.nan))
                  for r in ridges if np.isfinite(r['freq_trace'][i])]
        stage = _stage_at(sp, t_hr[i])
        if stage < 0:
            continue
        if active:
            freqs = np.array([a[0] for a in active])
            amps = np.array([a[1] for a in active])
            flats = np.array([a[2] for a in active])
            n_flat = int(np.sum(np.isfinite(flats) & (flats >= FLAT_THRESHOLD)))
            row = dict(
                n_active_ridges=len(active),
                min_ridge_freq=float(np.min(freqs)),
                total_ridge_power=float(np.sum(amps)),
                freq_spread=float(np.max(freqs) - np.min(freqs)),
                mean_flatness=float(np.nanmean(flats)) if np.isfinite(flats).any() else np.nan,
                n_flat_ridges=n_flat,
            )
        else:
            row = dict(n_active_ridges=0, min_ridge_freq=np.nan,
                       total_ridge_power=0.0, freq_spread=0.0,
                       mean_flatness=np.nan, n_flat_ridges=0)
        row.update(session=session_label, subject=subject, t_hr=float(t_hr[i]),
                   stage_code=stage, stage_label=STAGE_LABELS.get(stage, '?'))
        rows.append(row)
    return pd.DataFrame(rows)


def detect(session, cfg):
    sig = _get_signal(session, CHANNEL)
    acc = session.cap['acc_mag'].astype(np.float64)
    return detect_persistent_ridges(
        sig, fs=session.fs, win_sec=WIN_SEC, step_sec=STEP_SEC,
        acc_mag=acc, **cfg)


def process_session(idx):
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    sp = session.sleep_profile
    label, subject = session.label, session.subject
    print(f"\n{'='*60}\nRidge consolidation: {label}\n{'='*60}")

    # Baseline + flat-favoring detection for the consistency comparison
    rr_base = detect(session, BASELINE_CFG)
    rr_flat = detect(session, FLAT_CFG)

    def _summ(rr, tag):
        rd = per_ridge_table(rr, sp, label, subject)
        n = len(rd)
        n_flat = int((rd['flatness'] >= FLAT_THRESHOLD).sum()) if n else 0
        med_flat = float(rd['flatness'].median()) if n else np.nan
        print(f"  {tag:14s}: {n:4d} ridges | {n_flat:3d} flat (>={FLAT_THRESHOLD}) "
              f"| median flatness {med_flat:.3f}")
        return dict(config=tag, session=label, subject=subject, n_ridges=n,
                    n_flat=n_flat, flat_frac=(n_flat / n if n else np.nan),
                    median_flatness=med_flat,
                    median_drift=float(rd['drift_slope'].median()) if n else np.nan)

    retune = [_summ(rr_base, 'baseline'), _summ(rr_flat, 'flat_favoring')]

    # Use flat-favoring config as the consolidated detector going forward
    ridge_tbl = per_ridge_table(rr_flat, sp, label, subject)
    epoch_tbl = per_epoch_features(rr_flat, sp, label, subject)
    return ridge_tbl, epoch_tbl, pd.DataFrame(retune)


# ── Cross-session stage-association stats ────────────────────────────────────

def stage_association(epoch_df):
    """KW across stages + one-vs-rest AUC per stage for each ridge feature."""
    rows = []
    for feat in RIDGE_FEATURES:
        sub = epoch_df[[feat, 'stage_code', 'subject']].dropna()
        groups = [sub.loc[sub['stage_code'] == s, feat].values
                  for s in STAGE_ORDER if (sub['stage_code'] == s).sum() >= 10]
        if len(groups) >= 2:
            try:
                _, kw_p = kruskal(*groups)
            except ValueError:
                kw_p = np.nan
        else:
            kw_p = np.nan
        rec = dict(feature=feat, kruskal_p=kw_p)
        # one-vs-rest AUC per stage (pooled) + per-subject direction consistency
        for s in STAGE_ORDER:
            y = (sub['stage_code'] == s).astype(int)
            if y.nunique() == 2:
                auc = roc_auc_score(y, sub[feat])
            else:
                auc = np.nan
            # per-subject: does feature go up or down for this stage?
            ups = 0; tot = 0
            for subj, g in sub.groupby('subject'):
                a = g.loc[g['stage_code'] == s, feat]
                b = g.loc[g['stage_code'] != s, feat]
                if len(a) >= 10 and len(b) >= 10:
                    tot += 1
                    if a.median() > b.median():
                        ups += 1
            rec[f'AUC_{STAGE_LABELS[s]}'] = auc
            rec[f'dir_{STAGE_LABELS[s]}'] = f'{ups}/{tot}' if tot else '-'
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_summary(ridge_df, assoc_df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Flatness distribution by dominant stage
    ax = axes[0]
    data, labels, colors = [], [], []
    for s in STAGE_ORDER:
        sl = STAGE_LABELS[s]
        vals = ridge_df.loc[ridge_df['dominant_stage'] == sl, 'flatness'].dropna()
        if len(vals) >= 3:
            data.append(vals.values); labels.append(f'{sl}\n(n={len(vals)})')
            colors.append(STAGE_COLORS[s])
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel('Ridge flatness'); ax.set_title('Ridge flatness by dominant sleep stage')

    # AUC heatmap: feature x stage
    ax = axes[1]
    stages = [STAGE_LABELS[s] for s in STAGE_ORDER]
    mat = np.array([[assoc_df.loc[assoc_df['feature'] == f, f'AUC_{s}'].values[0]
                     for s in stages] for f in RIDGE_FEATURES], dtype=float)
    im = ax.imshow(mat, cmap='RdBu_r', vmin=0.35, vmax=0.65, aspect='auto')
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages)
    ax.set_yticks(range(len(RIDGE_FEATURES))); ax.set_yticklabels(RIDGE_FEATURES)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title('One-vs-rest AUC (feature × stage)')
    fig.colorbar(im, ax=ax, label='AUC')
    plt.tight_layout()
    return fig


def run_all():
    all_ridge, all_epoch, all_retune = [], [], []
    for idx in range(12):
        try:
            rt, et, rc = process_session(idx)
            all_ridge.append(rt); all_epoch.append(et); all_retune.append(rc)
        except Exception as e:
            print(f"  ERROR session {idx}: {e}")
            import traceback; traceback.print_exc()

    ridge_df = pd.concat(all_ridge, ignore_index=True)
    epoch_df = pd.concat(all_epoch, ignore_index=True)
    retune_df = pd.concat(all_retune, ignore_index=True)
    assoc_df = stage_association(epoch_df)

    ridge_df.to_csv(REPORT_DIR / 'per_ridge.csv', index=False)
    epoch_df.to_parquet(REPORT_DIR / 'per_epoch_features.parquet')
    retune_df.to_csv(REPORT_DIR / 'retune_comparison.csv', index=False)
    assoc_df.to_csv(REPORT_DIR / 'stage_association.csv', index=False)
    fig = plot_summary(ridge_df, assoc_df)
    fig.savefig(REPORT_DIR / 'ridge_consolidation.png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Consistency summary
    print(f"\n{'='*60}\nCONSISTENCY (baseline vs flat-favoring)\n{'='*60}")
    for cfg, g in retune_df.groupby('config'):
        print(f"  {cfg:14s}: mean {g['n_ridges'].mean():.0f} ridges/session, "
              f"flat-frac {g['flat_frac'].mean():.2f}, "
              f"median flatness {g['median_flatness'].median():.3f}, "
              f"cross-session ridge-count CV "
              f"{g['n_ridges'].std()/g['n_ridges'].mean():.2f}")

    print(f"\n{'='*60}\nSTAGE ASSOCIATION (best secondary stage per feature)\n{'='*60}")
    stages = [STAGE_LABELS[s] for s in STAGE_ORDER]
    for _, r in assoc_df.iterrows():
        aucs = {s: r[f'AUC_{s}'] for s in stages if np.isfinite(r[f'AUC_{s}'])}
        # best = AUC furthest from 0.5 (either direction)
        best = max(aucs, key=lambda s: abs(aucs[s] - 0.5)) if aucs else '-'
        print(f"  {r['feature']:18s} KW p={r['kruskal_p']:.1e} | "
              f"best stage: {best} (AUC={aucs.get(best, np.nan):.3f}, "
              f"dir {r.get('dir_'+best, '-')})")

    # Flat-ridge stage concentration
    flat = ridge_df[ridge_df['flatness'] >= FLAT_THRESHOLD]
    print(f"\n  Flat ridges (flatness>={FLAT_THRESHOLD}): {len(flat)}/{len(ridge_df)}")
    print("  Dominant-stage distribution of FLAT ridges:")
    print("   ", dict(flat['dominant_stage'].value_counts()))
    print("  Dominant-stage distribution of ALL ridges:")
    print("   ", dict(ridge_df['dominant_stage'].value_counts()))

    return ridge_df, epoch_df, assoc_df, retune_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=int, default=0)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all:
        run_all()
    else:
        rt, et, rc = process_session(args.session)
        print(rc.to_string(index=False))


if __name__ == '__main__':
    main()
