"""
Multi-channel ladder quantification (0-5 Hz band, all channels).

For every session and every window, run the comb-fit ladder detector
(analysis/slow_wave/ladder_spectrogram.comb_fit) on ALL CAP channels
(CH, CLE, CRE). A window is counted as carrying a ladder if ANY channel
detects one (>= MIN_RUNGS equally-spaced prominent ridges). Per-channel results
are all retained.

The comb spacing Δf is data-driven (a ladder need not be integer harmonics);
`harmonic` flags the special case where the comb passes through ~0 (Δf ≈ f0).
Fundamental < 0.6 Hz is tagged respiratory, >= 0.6 Hz cardiac (for band split).

Outputs -> reports/slow_wave/ladder_quantify/
  per_window_channels.parquet   one row per (session, window, channel)
  per_window_combined.parquet   one row per (session, window): any-channel ladder
  summary.csv                   prevalence / harmonic-frac / band split by stage
  channel_ladder_counts.csv     which channel detects ladders most

Run:
  python ladder_quantify.py --session 0
  python ladder_quantify.py --all
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sleep_monitor import load_session, load_sleep_profile, FS
from sleep_monitor.config import STAGE_LABELS, STAGE_ORDER
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.harmonics import detect_persistent_ridges
from run_ridge_overlay import (
    prepare_signals, WIN_SEC, STEP_SEC, MAX_FREQ, SMOOTH_WINDOWS,
    MIN_PERSIST_SEC, MAX_FREQ_JUMP, PEAK_PROM_FRAC, MAX_GAP_WINDOWS, WELCH_SEG_SEC,
)
from ladder_spectrogram import comb_fit, PROM_MIN, MIN_RUNGS

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'ladder_quantify'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
RESP_MAX_F0 = 0.6   # fundamental below this -> respiratory ladder, else cardiac


def _stage_at(sp, t_hr):
    idx = np.searchsorted(sp['t_ep_hr'], t_hr, side='right') - 1
    if 0 <= idx < len(sp['codes']):
        return int(sp['codes'][idx])
    return -1


def _prominent_freqs(rr, i):
    out = []
    for r in rr['ridges']:
        f = r['freq_trace'][i]
        if not np.isfinite(f):
            continue
        pt = r.get('prominence_trace')
        prom = pt[i] if (pt is not None and np.isfinite(pt[i])) else 0.0
        if prom >= PROM_MIN:
            out.append(f)
    return out


def process_session(idx):
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    sp = session.sleep_profile
    label, subject = session.label, session.subject
    print(f"\n{'='*60}\nLadder quantify: {label}\n{'='*60}")

    signals, acc_mag = prepare_signals(session)
    det = {}
    for ch in CHANNELS:
        det[ch] = detect_persistent_ridges(
            signals[ch], fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC, max_freq=MAX_FREQ,
            smooth_windows=SMOOTH_WINDOWS, min_persistence_sec=MIN_PERSIST_SEC,
            max_freq_jump=MAX_FREQ_JUMP, peak_prominence_frac=PEAK_PROM_FRAC,
            max_gap_windows=MAX_GAP_WINDOWS, welch_seg_sec=WELCH_SEG_SEC, acc_mag=acc_mag)

    t_hr = det[CHANNELS[0]]['t_hr']
    n_win = len(t_hr)
    ch_rows, comb_rows = [], []

    for i in range(n_win):
        stage = _stage_at(sp, t_hr[i])
        best = None
        for ch in CHANNELS:
            rr = det[ch]
            motion = bool(rr['motion_mask'][i])
            if motion:
                c = dict(df=np.nan, n_rungs=0, coverage=0.0, harmonic=False,
                         fundamental=np.nan, regularity=0.0)
            else:
                freqs = _prominent_freqs(rr, i)
                c = comb_fit(freqs)
            is_ladder = (c['n_rungs'] >= MIN_RUNGS) and (c.get('coverage', 0) > 0)
            ch_rows.append(dict(
                session=label, subject=subject, channel=ch, t_hr=float(t_hr[i]),
                stage_code=stage, stage_label=STAGE_LABELS.get(stage, '?'),
                motion=motion, is_ladder=is_ladder,
                n_rungs=int(c['n_rungs']) if is_ladder else 0,
                df_hz=float(c['df']) if is_ladder else np.nan,
                fundamental=float(c.get('fundamental', np.nan)) if is_ladder else np.nan,
                coverage=float(c.get('coverage', 0.0)) if is_ladder else 0.0,
                harmonic=bool(c.get('harmonic', False)) if is_ladder else False,
            ))
            if is_ladder:
                score = c['n_rungs'] * c['coverage']
                if best is None or score > best['score']:
                    best = dict(score=score, channel=ch, n_rungs=c['n_rungs'],
                                df=c['df'], fundamental=c.get('fundamental', np.nan),
                                coverage=c['coverage'], harmonic=c.get('harmonic', False))
        any_motion = all(det[ch]['motion_mask'][i] for ch in CHANNELS)
        comb_rows.append(dict(
            session=label, subject=subject, t_hr=float(t_hr[i]),
            stage_code=stage, stage_label=STAGE_LABELS.get(stage, '?'),
            motion=any_motion, any_ladder=best is not None,
            best_channel=best['channel'] if best else '',
            n_rungs=int(best['n_rungs']) if best else 0,
            df_hz=float(best['df']) if best else np.nan,
            fundamental=float(best['fundamental']) if best else np.nan,
            coverage=float(best['coverage']) if best else 0.0,
            harmonic=bool(best['harmonic']) if best else False,
            band=('respiratory' if (best and best['fundamental'] < RESP_MAX_F0)
                  else ('cardiac' if best else '')),
        ))

    ch_df = pd.DataFrame(ch_rows)
    comb_df = pd.DataFrame(comb_rows)
    nonmotion = comb_df[~comb_df['motion']]
    prev = nonmotion['any_ladder'].mean() if len(nonmotion) else np.nan
    harm = nonmotion.loc[nonmotion['any_ladder'], 'harmonic'].mean() if nonmotion['any_ladder'].any() else np.nan
    print(f"  {n_win} windows | any-channel ladder prevalence (non-motion) {prev:.0%} | "
          f"{harm:.0%} harmonic")
    for ch in CHANNELS:
        n = int(ch_df[(ch_df['channel'] == ch) & ch_df['is_ladder']].shape[0])
        print(f"    {ch}: {n} ladder windows")
    return ch_df, comb_df


def summarize(ch_all, comb_all):
    nm = comb_all[~comb_all['motion']]
    rows = []
    for s in STAGE_ORDER:
        sub = nm[nm['stage_code'] == s]
        if len(sub) < 10:
            continue
        lad = sub[sub['any_ladder']]
        rows.append(dict(
            stage=STAGE_LABELS[s], n_windows=len(sub),
            ladder_prevalence=round(sub['any_ladder'].mean(), 3),
            harmonic_frac=round(lad['harmonic'].mean(), 3) if len(lad) else np.nan,
            resp_frac=round((lad['band'] == 'respiratory').mean(), 3) if len(lad) else np.nan,
            card_frac=round((lad['band'] == 'cardiac').mean(), 3) if len(lad) else np.nan,
            median_rungs=int(lad['n_rungs'].median()) if len(lad) else 0,
            median_df=round(lad['df_hz'].median(), 3) if len(lad) else np.nan,
        ))
    summ = pd.DataFrame(rows)

    # channel comparison
    ch_counts = []
    for ch in CHANNELS:
        cc = ch_all[(ch_all['channel'] == ch) & ~ch_all['motion']]
        lad = cc[cc['is_ladder']]
        ch_counts.append(dict(
            channel=ch, ladder_windows=len(lad),
            ladder_prevalence=round(cc['is_ladder'].mean(), 3),
            harmonic_frac=round(lad['harmonic'].mean(), 3) if len(lad) else np.nan,
            median_fundamental=round(lad['fundamental'].median(), 3) if len(lad) else np.nan,
        ))
    return summ, pd.DataFrame(ch_counts)


def run_all():
    ch_all, comb_all = [], []
    for idx in range(12):
        try:
            cd, kd = process_session(idx)
            ch_all.append(cd); comb_all.append(kd)
        except Exception as e:
            print(f"  ERROR session {idx}: {e}")
            import traceback; traceback.print_exc()
    ch_all = pd.concat(ch_all, ignore_index=True)
    comb_all = pd.concat(comb_all, ignore_index=True)
    ch_all.to_parquet(REPORT_DIR / 'per_window_channels.parquet')
    comb_all.to_parquet(REPORT_DIR / 'per_window_combined.parquet')
    summ, ch_counts = summarize(ch_all, comb_all)
    summ.to_csv(REPORT_DIR / 'summary.csv', index=False)
    ch_counts.to_csv(REPORT_DIR / 'channel_ladder_counts.csv', index=False)

    print(f"\n{'='*60}\nLADDER PREVALENCE / TYPE BY STAGE (any-channel, non-motion)\n{'='*60}")
    print(summ.to_string(index=False))
    print(f"\n{'='*60}\nPER-CHANNEL LADDER DETECTION\n{'='*60}")
    print(ch_counts.to_string(index=False))
    nm = comb_all[~comb_all['motion'] & comb_all['any_ladder']]
    print(f"\n  Overall any-channel ladder prevalence (non-motion): "
          f"{comb_all[~comb_all['motion']]['any_ladder'].mean():.0%}")
    print(f"  Band split of detected ladders: "
          f"respiratory {np.mean(nm['band']=='respiratory'):.0%}, "
          f"cardiac {np.mean(nm['band']=='cardiac'):.0%}")
    print(f"  Harmonic (Δf≈f0) vs inharmonic: "
          f"{nm['harmonic'].mean():.0%} / {1-nm['harmonic'].mean():.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=int, default=0)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all:
        run_all()
    else:
        process_session(args.session)


if __name__ == '__main__':
    main()
