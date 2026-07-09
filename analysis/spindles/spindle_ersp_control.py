"""
Is the CAP 0-3 Hz spindle bump a real spindle signature, or an artifact / an
arousal-K-complex confound?

Two decisive controls, reusing the ERSP machinery from spindle_ersp.py:

  (A) ARTIFACT NULL — run the identical ERSP at RANDOM N2 timepoints. Event-
      triggered spectrograms with a within-window baseline can manufacture a
      center bump; if random N2 also bumps, the effect is not spindle-locked.

  (B) AROUSAL CONFOUND — most N2 spindles ride a K-complex / micro-arousal whose
      autonomic-motor transient IS a 0-3 Hz mechanical signal. We (i) trigger the
      ERSP on scored arousals (Classification Arousal, in N2) and (ii) split
      spindles by whether an arousal onset falls within +/-5 s. If only
      arousal-coupled spindles bump, the mask is sensing the arousal, not the
      spindle.

Channels: CH (strongest bump) + CLE-CRE (canonical). Outputs -> outputs/.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles, _SPINDLE_RE, _tod_sec
from analysis.spindles.spindle_ersp import (
    FS, N2_CODE, WIN_HALF, FMAX, session_ersp, get_channel, stage_at)

CHANNELS = ['CH', 'CLE-CRE']
MAX_EVENTS = 400
AROUSAL_NEAR = 5.0     # s: spindle "has an arousal" if one is within this
CORE = 1.0
OUT = os.path.join(os.path.dirname(__file__), 'outputs')


def load_arousals(session):
    """Center times (hr from CAP start) of scored cortical arousals."""
    psg_dir = session.meta.get('psg_dir')
    if psg_dir is None:
        return None
    pat = str(psg_dir / 'PSG_analysis_*' / 'Classification Arousal*.txt')
    fs = [m for m in glob.glob(pat) if 'MACOSX' not in m and os.path.basename(m)[:2] != '._']
    if not fs:
        return None
    starts, ends = [], []
    with open(sorted(fs)[0], 'r', encoding='latin-1') as fh:
        for line in fh:
            mt = _SPINDLE_RE.match(line.strip())
            if not mt:
                continue
            starts.append(_tod_sec(*mt.group(1, 2, 3, 4)))
            ends.append(_tod_sec(*mt.group(5, 6, 7, 8)))
    if not starts:
        return None
    starts, ends = np.array(starts), np.array(ends)
    ts = session.time_start
    if ts is None:
        return None
    if hasattr(ts, 'tz') and ts.tz is not None:
        ts = ts.tz_localize(None) if hasattr(ts, 'tz_localize') else ts.replace(tzinfo=None)
    csv_start = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6

    def to_hr(tod):
        off = tod - csv_start
        off = np.where(off < -43200, off + 86400, off)
        off = np.where(off > 43200, off - 86400, off)
        return off / 3600.0

    center = 0.5 * (to_hr(starts) + to_hr(ends))
    dur_hr = float(session.time_hr[-1])
    return center[(center >= 0) & (center <= dur_hr)]


def core_spectrum(sig, centers_hr, half, rng):
    if len(centers_hr) > MAX_EVENTS:
        centers_hr = rng.choice(centers_hr, size=MAX_EVENTS, replace=False)
    cen = np.round(centers_hr * 3600.0 * FS).astype(int)
    r = session_ersp(sig, cen, half)
    if r is None:
        return None, None, 0
    core = r['ersp'][:, np.abs(r['t']) < CORE].mean(axis=1)
    return r['f'], core, r['k']


def main():
    rng = np.random.default_rng(3)
    conds = ['spindle', 'randN2', 'arousal', 'spindle_arous', 'spindle_noarous']
    stacks = {ch: {c: [] for c in conds} for ch in CHANNELS}
    f_axis = None
    counts = {c: [] for c in conds}

    for idx in range(len(SESSION_META)):
        meta = SESSION_META[idx]
        try:
            s = load_session(idx)
            s.sleep_profile = load_sleep_profile(s)
            sp = load_spindles(s)
            if sp is None or s.sleep_profile is None:
                continue
            stg = stage_at(sp['center_hr'], s.sleep_profile)
            spin_hr = sp['center_hr'][stg == N2_CODE]
            if len(spin_hr) < 20:
                continue

            # random N2 timepoints
            n2_starts = s.sleep_profile['t_ep_hr'][s.sleep_profile['codes'] == N2_CODE]
            rand_hr = rng.choice(n2_starts, size=min(len(spin_hr), len(n2_starts)),
                                 replace=len(n2_starts) < len(spin_hr)) + 0.5 * 30.0 / 3600.0

            # arousals (restrict to N2), and spindle split by arousal proximity
            ar = load_arousals(s)
            if ar is not None and len(ar):
                ar_stg = stage_at(ar, s.sleep_profile)
                ar_n2 = ar[ar_stg == N2_CODE]
                d = np.min(np.abs(spin_hr[:, None] - ar[None, :]), axis=1) * 3600.0
                spin_ar = spin_hr[d <= AROUSAL_NEAR]
                spin_no = spin_hr[d > AROUSAL_NEAR]
            else:
                ar_n2 = np.array([]); spin_ar = np.array([]); spin_no = spin_hr

            half = int(WIN_HALF * FS)
            cond_centers = {'spindle': spin_hr, 'randN2': rand_hr, 'arousal': ar_n2,
                            'spindle_arous': spin_ar, 'spindle_noarous': spin_no}
            for ch in CHANNELS:
                sig = get_channel(s, ch)
                for c in conds:
                    cc = cond_centers[c]
                    if len(cc) < 15:
                        continue
                    f, core, k = core_spectrum(sig, cc, half, rng)
                    if core is None:
                        continue
                    f_axis = f
                    stacks[ch][c].append(core)
                    if ch == CHANNELS[0]:
                        counts[c].append(k)
            print(f"{meta['label']}: spin={len(spin_hr)} rand={len(rand_hr)} "
                  f"arousalN2={len(ar_n2)} spin+ar={len(spin_ar)} spin-ar={len(spin_no)}")
        except Exception as e:
            print(f'[{idx}] control failed: {e}')
            continue

    save = {'f': f_axis}
    for ch in CHANNELS:
        for c in conds:
            arr = np.array(stacks[ch][c]) if stacks[ch][c] else np.zeros((0,))
            save[f'{ch}__{c}'] = arr
    np.savez(os.path.join(OUT, 'spindle_ersp_control.npz'), **save)
    print('\nevent counts (median/session):',
          {c: int(np.median(counts[c])) if counts[c] else 0 for c in conds})
    # 0-3 Hz core change per condition/channel
    print('\n=== mean 0-3 Hz core-vs-baseline (dB) ===')
    lo = (f_axis >= 0.5) & (f_axis <= 3.0)
    rows = []
    for ch in CHANNELS:
        for c in conds:
            a = save[f'{ch}__{c}']
            v = float(np.nanmean(a[:, lo])) if a.size else np.nan
            rows.append({'channel': ch, 'cond': c, 'lowfreq_dB': round(v, 3),
                         'n_sessions': a.shape[0] if a.ndim == 2 else 0})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'spindle_ersp_control.csv'), index=False)
    print(df.to_string(index=False))
    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()
