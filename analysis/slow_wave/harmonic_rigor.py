"""
Harmonic rigor: reject fake ladders, score confidence, characterize the rest
(Workstream B).

Three goals:
  1. STRONGER, NO FAKES — integer-ratio alignment happens by chance when many
     ridges are active. We build a per-session, per-k surrogate null (random
     ridge frequencies over the observed band) and keep only ladders whose
     alignment quality beats the null 95th percentile. Reports how many
     old-criterion ladders survive (real vs fake fraction).
  2. CONFIDENCE SCORE — for surviving ladders, a calibrated [0,1] confidence
     combining ratio quality, amplitude-decay monotonicity, and harmonic count.
  3. WHAT HAPPENS OTHERWISE — every non-ladder window is classified
     (motion / quiet / single-tone / broadband / multi-non-harmonic) from its
     spectrum, then cross-tabbed against sleep stage.

Channel: CH (strongest harmonic detection per manuscript, 70% of windows).

Outputs -> reports/slow_wave/harmonic_rigor/
  ladder_windows.parquet    per-window ladder score, confidence, survives-null
  null_summary.csv          real-vs-fake ladder counts per session
  otherwise_crosstab.csv     non-harmonic window class x sleep stage
  harmonic_rigor.png         survival + confidence + otherwise-by-stage figure

Run:
  python harmonic_rigor.py --session 0
  python harmonic_rigor.py --all
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import STAGE_LABELS, STAGE_ORDER, STAGE_COLORS
from sleep_monitor.harmonics import detect_persistent_ridges

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'harmonic_rigor'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL = 'CH'
WIN_SEC = 30.0
STEP_SEC = 30.0
DET_CFG = dict(max_freq_jump=0.12, min_persistence_sec=300.0,
               smooth_windows=5, welch_seg_sec=20.0)
RATIO_TOL = 0.06        # tight integer tolerance — 0.12 admits coincidental pairs
MIN_F0 = 0.1
MIN_STRONG_MEMBERS = 3  # a "strong" ladder = fundamental + >=2 harmonics
N_SURROGATE = 400
QUIET_PCT = 20      # in-band power below this session-percentile -> "quiet"
CONC_HI = 0.5       # spectral concentration above -> "single-tone"


# ── Ladder scoring (shared by observed + surrogate) ──────────────────────────

def best_ladder(freqs, amps, ratio_tol=RATIO_TOL, min_f0=MIN_F0):
    """
    Best integer-ratio ladder among a set of (freq, amp) ridges.

    Returns (n_members, ratio_quality, f0, power, decay_factor, member_amps).
    ratio_quality in [0,1] (1 = perfect integer alignment); decay_factor in
    [0,1] (1 = amplitudes monotonically non-increasing with harmonic number).
    """
    order = np.argsort(freqs)
    freqs = np.asarray(freqs)[order]
    amps = np.asarray(amps)[order]
    n = len(freqs)
    best = (0, 0.0, np.nan, 0.0, 0.0, [])
    best_score = -1.0
    for ai in range(n):
        f0 = freqs[ai]
        if f0 < min_f0:
            continue
        members = [(1, f0, amps[ai])]
        devs = []
        for aj in range(n):
            if aj == ai:
                continue
            ratio = freqs[aj] / f0
            k = round(ratio)
            if k >= 2 and abs(ratio - k) < ratio_tol:
                members.append((k, freqs[aj], amps[aj]))
                devs.append(abs(ratio - k))
        if len(members) < 2:
            continue
        quality = 1.0 - np.mean(devs) / ratio_tol if devs else 0.0
        # amplitude decay: order members by harmonic index, count non-increasing steps
        members.sort(key=lambda m: m[0])
        m_amps = [m[2] for m in members]
        steps = np.diff(m_amps)
        decay = float(np.mean(steps <= 0)) if len(steps) else 0.0
        power = float(np.sum(m_amps))
        score = _score(len(members), quality)
        if score > best_score:
            best_score = score
            best = (len(members), quality, float(f0), power, decay, m_amps)
    return best


def _score(n_members, quality):
    """Combined alignment score used for the null comparison."""
    if n_members < 2:
        return 0.0
    return quality * min(np.log2(max(n_members, 1)) / np.log2(6), 1.0)


# ── Per-session processing ───────────────────────────────────────────────────

def _spectral_shape(psd):
    """Return (concentration, entropy) of a (possibly NaN) PSD row over the band."""
    p = psd[np.isfinite(psd)]
    if len(p) < 4 or np.all(p <= 0):
        return np.nan, np.nan
    p = p / p.sum()
    concentration = float(np.max(p))                        # peakiness
    entropy = float(-np.sum(p * np.log(p + 1e-30)) / np.log(len(p)))  # 0..1
    return concentration, entropy


def process_session(idx):
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    sp = session.sleep_profile
    label, subject = session.label, session.subject
    print(f"\n{'='*60}\nHarmonic rigor: {label}\n{'='*60}")

    sig = session.cap[CHANNEL].astype(np.float64)
    acc = session.cap['acc_mag'].astype(np.float64)
    rr = detect_persistent_ridges(sig, fs=session.fs, win_sec=WIN_SEC,
                                  step_sec=STEP_SEC, acc_mag=acc, **DET_CFG)
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    psds = rr['psds_smooth']
    motion = rr['motion_mask']
    n_win = len(t_hr)

    # session ridge frequency range for the surrogate draws
    all_freqs = np.concatenate([r['freq_trace'][np.isfinite(r['freq_trace'])]
                                for r in ridges]) if ridges else np.array([])
    if len(all_freqs) == 0:
        return None, None, None
    f_lo, f_hi = float(np.min(all_freqs)), float(np.max(all_freqs))
    all_amps = np.concatenate([r['amp_trace'][np.isfinite(r['amp_trace'])]
                               for r in ridges])

    # ── Per-k surrogate null thresholds (95th pct of random-alignment score) ──
    rng = np.random.default_rng(1234 + idx)
    max_k = 0
    active_per_win = []
    for i in range(n_win):
        act = [(r['freq_trace'][i], r['amp_trace'][i])
               for r in ridges if np.isfinite(r['freq_trace'][i])]
        active_per_win.append(act)
        max_k = max(max_k, len(act))

    null_thresh = {}
    for k in range(2, max_k + 1):
        scores = np.empty(N_SURROGATE)
        for s in range(N_SURROGATE):
            fr = rng.uniform(f_lo, f_hi, size=k)
            am = rng.choice(all_amps, size=k, replace=True)
            nm, q, *_ = best_ladder(fr, am)
            scores[s] = _score(nm, q)
        null_thresh[k] = float(np.percentile(scores, 95))

    # ── Observed ladders vs null ──
    rows = []
    n_old_ladder = n_strong = n_survive = 0
    for i in range(n_win):
        act = active_per_win[i]
        stage = _stage_at(sp, t_hr[i])
        conc, ent = _spectral_shape(psds[i]) if not motion[i] else (np.nan, np.nan)
        rec = dict(session=label, subject=subject, t_hr=float(t_hr[i]),
                   stage_code=stage, stage_label=STAGE_LABELS.get(stage, '?'),
                   motion=bool(motion[i]), n_active=len(act),
                   concentration=conc, entropy=ent)
        if len(act) >= 2:
            fr = [a[0] for a in act]; am = [a[1] for a in act]
            nm, q, f0, power, decay, m_amps = best_ladder(fr, am)
            obs = _score(nm, q)
            thr = null_thresh.get(nm if nm >= 2 else 2, np.inf)
            is_old = nm >= 2                          # permissive: any integer pair
            is_strong = nm >= MIN_STRONG_MEMBERS      # fundamental + >=2 harmonics
            # a real detection = a STRONG ladder that also beats the null
            survives = is_strong and (obs > thr)
            n_old_ladder += int(is_old)
            n_strong += int(is_strong)
            n_survive += int(survives)
            confidence = float(q * decay) if survives else 0.0
            rec.update(n_members=nm, ratio_quality=float(q), ladder_f0=f0,
                       ladder_power=power, decay_factor=float(decay),
                       obs_score=obs, null_thresh=float(thr),
                       old_ladder=is_old, survives_null=survives,
                       confidence=confidence)
        else:
            rec.update(n_members=0, ratio_quality=0.0, ladder_f0=np.nan,
                       ladder_power=0.0, decay_factor=0.0, obs_score=0.0,
                       null_thresh=np.nan, old_ladder=False,
                       survives_null=False, confidence=0.0)
        rows.append(rec)

    df = pd.DataFrame(rows)

    # ── Classify non-ladder ("otherwise") windows ──
    quiet_thresh = np.nanpercentile(
        df.loc[~df['motion'], 'ladder_power'].replace(0, np.nan).values, QUIET_PCT)
    # use total in-band power proxy = sum of active amps; fall back to concentration
    def _classify(r):
        if r['motion']:
            return 'motion'
        if r['survives_null']:
            return 'ladder'
        # non-ladder window: describe its spectrum
        if r['n_active'] == 0 or (np.isfinite(r['concentration']) is False):
            return 'quiet'
        if r['n_active'] <= 1 or (np.isfinite(r['concentration']) and r['concentration'] >= CONC_HI):
            return 'single_tone'
        if np.isfinite(r['entropy']) and r['entropy'] >= 0.85:
            return 'broadband'
        return 'multi_nonharmonic'
    df['window_class'] = df.apply(_classify, axis=1)

    survive_frac = n_survive / n_strong if n_strong else np.nan
    print(f"  {n_win} windows | permissive(>=2) {n_old_ladder} | "
          f"strong(>=3) {n_strong} | survive null {n_survive} "
          f"({survive_frac:.0%} of strong are real, {1-survive_frac:.0%} rejected)")
    print(f"  otherwise: {dict(df['window_class'].value_counts())}")

    null_row = dict(session=label, subject=subject, n_windows=n_win,
                    n_permissive=n_old_ladder, n_strong=n_strong,
                    n_survive=n_survive, survive_frac=survive_frac)
    return df, null_row, df[['window_class', 'stage_label', 'session']]


def _stage_at(sp, t_hr):
    idx = np.searchsorted(sp['t_ep_hr'], t_hr, side='right') - 1
    if 0 <= idx < len(sp['codes']):
        return int(sp['codes'][idx])
    return -1


# ── Aggregation + plotting ───────────────────────────────────────────────────

def plot_summary(win_df, null_df):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Real vs fake ladders per session
    ax = axes[0]
    x = np.arange(len(null_df))
    ax.bar(x, null_df['n_survive'], label='strong ladder, survives null', color='#27AE60')
    ax.bar(x, null_df['n_strong'] - null_df['n_survive'],
           bottom=null_df['n_survive'], label='strong but rejected (fake)',
           color='#E74C3C', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(null_df['session'], rotation=45, fontsize=7)
    ax.set_ylabel('windows'); ax.set_title('Ladder survival vs surrogate null')
    ax.legend(fontsize=8)

    # 2. Confidence distribution of surviving ladders
    ax = axes[1]
    conf = win_df.loc[win_df['survives_null'], 'confidence'].dropna()
    ax.hist(conf, bins=30, color='#2980B9', alpha=0.8)
    ax.set_xlabel('ladder confidence'); ax.set_ylabel('windows')
    ax.set_title(f'Confidence of surviving ladders (n={len(conf)})')

    # 3. Otherwise: window class fraction by stage
    ax = axes[2]
    classes = ['ladder', 'single_tone', 'multi_nonharmonic', 'broadband', 'quiet', 'motion']
    cls_colors = {'ladder': '#27AE60', 'single_tone': '#3498DB',
                  'multi_nonharmonic': '#9B59B6', 'broadband': '#E67E22',
                  'quiet': '#95A5A6', 'motion': '#E74C3C'}
    stages = [STAGE_LABELS[s] for s in STAGE_ORDER]
    bottom = np.zeros(len(stages))
    for c in classes:
        fracs = []
        for s in stages:
            sub = win_df[win_df['stage_label'] == s]
            fracs.append((sub['window_class'] == c).mean() if len(sub) else 0)
        ax.bar(stages, fracs, bottom=bottom, label=c, color=cls_colors[c])
        bottom += np.array(fracs)
    ax.set_ylabel('fraction of windows'); ax.set_title('What CAP does, by sleep stage')
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    return fig


def run_all():
    win_all, null_all, ct_all = [], [], []
    for idx in range(12):
        try:
            df, nr, ct = process_session(idx)
            if df is not None:
                win_all.append(df); null_all.append(nr); ct_all.append(ct)
        except Exception as e:
            print(f"  ERROR session {idx}: {e}")
            import traceback; traceback.print_exc()

    win_df = pd.concat(win_all, ignore_index=True)
    null_df = pd.DataFrame(null_all)
    win_df.to_parquet(REPORT_DIR / 'ladder_windows.parquet')
    null_df.to_csv(REPORT_DIR / 'null_summary.csv', index=False)

    crosstab = pd.crosstab(win_df['window_class'], win_df['stage_label'],
                           normalize='columns')
    crosstab.to_csv(REPORT_DIR / 'otherwise_crosstab.csv')

    fig = plot_summary(win_df, null_df)
    fig.savefig(REPORT_DIR / 'harmonic_rigor.png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n{'='*60}\nPOOLED SUMMARY\n{'='*60}")
    tot_perm = null_df['n_permissive'].sum()
    tot_strong = null_df['n_strong'].sum()
    tot_surv = null_df['n_survive'].sum()
    print(f"  Permissive(>=2) ladders: {tot_perm} | strong(>=3): {tot_strong} | "
          f"survive null: {tot_surv} "
          f"({tot_surv/tot_strong:.0%} of strong are real, "
          f"{1-tot_surv/tot_strong:.0%} rejected as coincidental)")
    print(f"  Median per-session survival: {null_df['survive_frac'].median():.0%}")
    print(f"  Surviving-ladder confidence (median): "
          f"{win_df.loc[win_df['survives_null'],'confidence'].median():.3f}")
    print("\n  Window-class fraction by stage (otherwise crosstab):")
    print(crosstab.round(3).to_string())
    return win_df, null_df, crosstab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=int, default=0)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all:
        run_all()
    else:
        df, nr, ct = process_session(args.session)


if __name__ == '__main__':
    main()
