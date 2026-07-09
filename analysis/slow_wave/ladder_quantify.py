"""
Multi-channel ladder quantification, band-isolated (0-5 Hz rungs).

For every session/window/channel we run the comb-fit ladder detector
(analysis/slow_wave/ladder_spectrogram.comb_fit) on the prominent persistent
ridges in THREE modes:

  combined : Δf free over 0.15-1.6 Hz — the single dominant ladder (kept setup)
  resp     : Δf restricted to 0.12-0.50 Hz — breathing-harmonic ladder
  cardiac  : Δf restricted to 0.50-1.60 Hz — heartbeat-harmonic ladder

Respiratory and cardiac ladders are detected INDEPENDENTLY, so one window can
carry both at once (they no longer compete for the single best comb). Rungs may
extend across the full 0-5 Hz range in every mode; only the spacing Δf is banded.

A window carries a ladder (of a given mode) if ANY channel detects >= MIN_RUNGS
equally-spaced prominent rungs. Per-channel results are all retained.

Outputs -> reports/slow_wave/ladder_quantify/
  per_window_channels.parquet   one row per (session, window, channel), all modes
  per_window_combined.parquet   one row per (session, window), any-channel per mode
  summary_by_band.csv           prevalence / harmonic-frac / Δf by stage x mode
  channel_ladder_counts.csv     per-channel detection counts by mode

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
# Δf (rung-spacing) bands that isolate each harmonic family
BANDS = {
    'combined': (0.15, 1.60),
    'resp':     (0.12, 0.50),
    'cardiac':  (0.50, 1.60),
}


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


def _ladder_fields(c):
    """Flatten a comb_fit result to a compact ladder record."""
    is_ladder = (c['n_rungs'] >= MIN_RUNGS) and (c.get('coverage', 0) > 0)
    return dict(
        is_ladder=is_ladder,
        n_rungs=int(c['n_rungs']) if is_ladder else 0,
        df_hz=float(c['df']) if is_ladder else np.nan,
        fundamental=float(c.get('fundamental', np.nan)) if is_ladder else np.nan,
        coverage=float(c.get('coverage', 0.0)) if is_ladder else 0.0,
        harmonic=bool(c.get('harmonic', False)) if is_ladder else False,
    )


def process_session(idx):
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    sp = session.sleep_profile
    label, subject = session.label, session.subject
    print(f"\n{'='*60}\nLadder quantify (band-isolated): {label}\n{'='*60}")

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
        # best ladder per band across channels (for the combined per-window row)
        best = {m: None for m in BANDS}
        for ch in CHANNELS:
            rr = det[ch]
            motion = bool(rr['motion_mask'][i])
            freqs = [] if motion else _prominent_freqs(rr, i)
            row = dict(session=label, subject=subject, channel=ch,
                       t_hr=float(t_hr[i]), stage_code=stage,
                       stage_label=STAGE_LABELS.get(stage, '?'), motion=motion)
            for mode, (lo, hi) in BANDS.items():
                # max_min_k=2 requires a real low rung so a banded spacing can't
                # be spuriously fit to only high harmonics of the other rhythm
                c = comb_fit(freqs, df_lo=lo, df_hi=hi, max_min_k=2) if freqs else \
                    dict(n_rungs=0, df=np.nan, coverage=0.0, harmonic=False, fundamental=np.nan)
                lf = _ladder_fields(c)
                for k, v in lf.items():
                    row[f'{mode}_{k}'] = v
                if lf['is_ladder']:
                    score = lf['n_rungs'] * lf['coverage']
                    if best[mode] is None or score > best[mode]['score']:
                        best[mode] = dict(score=score, channel=ch, **lf)
            ch_rows.append(row)

        any_motion = all(det[ch]['motion_mask'][i] for ch in CHANNELS)
        crow = dict(session=label, subject=subject, t_hr=float(t_hr[i]),
                    stage_code=stage, stage_label=STAGE_LABELS.get(stage, '?'),
                    motion=any_motion)
        for mode in BANDS:
            b = best[mode]
            crow[f'{mode}_ladder'] = b is not None
            crow[f'{mode}_channel'] = b['channel'] if b else ''
            crow[f'{mode}_n_rungs'] = int(b['n_rungs']) if b else 0
            crow[f'{mode}_df'] = float(b['df_hz']) if b else np.nan
            crow[f'{mode}_fundamental'] = float(b['fundamental']) if b else np.nan
            crow[f'{mode}_harmonic'] = bool(b['harmonic']) if b else False
        comb_rows.append(crow)

    ch_df = pd.DataFrame(ch_rows)
    comb_df = pd.DataFrame(comb_rows)
    nm = comb_df[~comb_df['motion']]
    print(f"  {n_win} windows | non-motion {len(nm)} | any-channel prevalence: "
          + ", ".join(f"{m} {nm[f'{m}_ladder'].mean():.0%}" for m in BANDS))
    for ch in CHANNELS:
        cc = ch_df[ch_df['channel'] == ch]
        print(f"    {ch}: " + ", ".join(
            f"{m} {int(cc[f'{m}_is_ladder'].sum())}" for m in BANDS))
    return ch_df, comb_df


def summarize(ch_all, comb_all):
    nm = comb_all[~comb_all['motion']]
    rows = []
    for mode in BANDS:
        for s in STAGE_ORDER:
            sub = nm[nm['stage_code'] == s]
            if len(sub) < 10:
                continue
            lad = sub[sub[f'{mode}_ladder']]
            rows.append(dict(
                mode=mode, stage=STAGE_LABELS[s], n_windows=len(sub),
                prevalence=round(sub[f'{mode}_ladder'].mean(), 3),
                harmonic_frac=round(lad[f'{mode}_harmonic'].mean(), 3) if len(lad) else np.nan,
                median_df=round(lad[f'{mode}_df'].median(), 3) if len(lad) else np.nan,
                median_f0=round(lad[f'{mode}_fundamental'].median(), 3) if len(lad) else np.nan,
                median_rungs=int(lad[f'{mode}_n_rungs'].median()) if len(lad) else 0,
            ))
    summ = pd.DataFrame(rows)

    ch_counts = []
    for ch in CHANNELS:
        cc = ch_all[(ch_all['channel'] == ch) & ~ch_all['motion']]
        rec = dict(channel=ch, n_windows=len(cc))
        for mode in BANDS:
            lad = cc[cc[f'{mode}_is_ladder']]
            rec[f'{mode}_windows'] = len(lad)
            rec[f'{mode}_prev'] = round(cc[f'{mode}_is_ladder'].mean(), 3)
            rec[f'{mode}_med_f0'] = round(lad[f'{mode}_fundamental'].median(), 3) if len(lad) else np.nan
        ch_counts.append(rec)
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
    summ.to_csv(REPORT_DIR / 'summary_by_band.csv', index=False)
    ch_counts.to_csv(REPORT_DIR / 'channel_ladder_counts.csv', index=False)

    print(f"\n{'='*70}\nLADDER PREVALENCE / TYPE BY STAGE x BAND (any-channel, non-motion)\n{'='*70}")
    print(summ.to_string(index=False))
    print(f"\n{'='*70}\nPER-CHANNEL LADDER DETECTION BY BAND\n{'='*70}")
    print(ch_counts.to_string(index=False))
    nm = comb_all[~comb_all['motion']]
    print(f"\n  Overall any-channel prevalence (non-motion):")
    for mode in BANDS:
        lad = nm[nm[f'{mode}_ladder']]
        co = (nm['resp_ladder'] & nm['cardiac_ladder']).mean()
        print(f"    {mode:8s}: {nm[f'{mode}_ladder'].mean():.1%} | "
              f"{lad[f'{mode}_harmonic'].mean():.0%} harmonic | "
              f"median f0={lad[f'{mode}_fundamental'].median():.2f} Hz")
    print(f"    resp & cardiac co-occur in {(nm['resp_ladder'] & nm['cardiac_ladder']).mean():.1%} of non-motion windows")


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
