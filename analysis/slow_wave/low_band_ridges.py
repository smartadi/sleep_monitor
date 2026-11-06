"""
Low-frequency (infra-slow) persistent ridges: reproduce, pin the spec, and null-test.

Why this exists
---------------
The manuscript reports recurrent low-frequency ridges (§3.3: "67 total; median
6 per night", energy "near 0.05-0.08 Hz") and leans on them in the discussion,
but the band is NOT in our pipeline: reports/slow_wave/band_ridge_epochs.parquet
carries band in {resp, card} only.  The numbers come from a co-author script we
do not hold, so we cannot regenerate or audit them.

The paper also carries three mutually inconsistent specs for the same analysis:

  §2.7 methods        0.01-0.3 Hz,  5-min windows / 50% overlap,  ridge >= 5 min
  co-author figure    0.01-0.10 Hz, ">= 10 min continuous ridge"
  §3.3 prose          "most below 0.12 Hz ... some extended to approximately 0.30 Hz"

So this script does three things:

  1. REPRODUCE  - implements the low band in our own detector.
  2. PIN        - runs a small spec grid (band top x persistence threshold) so the
                  sensitivity of the headline count to the spec is visible, and
                  the paper can commit to one spec on evidence.
  3. NULL-TEST  - a ridge at 0.02-0.08 Hz with 5-min windows sits a handful of
                  bins off DC, inside the 1/f knee, where a ridge tracker will
                  happily follow colored noise.  For each recording we build
                  phase-randomized surrogates of that recording's own low-band
                  signal.  The surrogate has the SAME power spectrum (same 1/f)
                  and no time structure, so any ridge it yields is the detector
                  talking to itself.  Observed counts are scored against that.

Outputs -> reports/slow_wave/low_band/
  low_band_ridges.csv        one row per detected ridge (observed)
  low_band_per_night.csv     per session x channel x spec summary + null p
  low_band_null.csv          per-surrogate metrics (long form)

Run:
  python low_band_ridges.py --pilot                # S1N1, CRE, 20 surrogates
  python low_band_ridges.py --n-surrogates 200     # full
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor import load_session, SESSION_META, FS
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import detect_persistent_ridges

OUT = ROOT / 'reports' / 'slow_wave' / 'low_band'
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']

# Window geometry is fixed by the methods text: 5-min windows, 50% overlap.
# One periodogram per window -> df = 1/300 = 0.0033 Hz.
WIN_SEC = 300.0
STEP_SEC = 150.0
WELCH_SEG_SEC = 300.0
BAND_LO = 0.01

# A ridge may wander by ~3 bins between windows.
MAX_FREQ_JUMP = 0.010
PEAK_PROM_FRAC = 0.25
SMOOTH_WINDOWS = 3
MAX_GAP_WINDOWS = 2

# spec grid: (band top Hz, minimum persistence s)
SPECS = [
    (0.10, 300.0),
    (0.10, 600.0),
    (0.30, 300.0),
    (0.30, 600.0),
]


def spec_name(top, persist):
    return f'{BAND_LO:g}-{top:g}Hz_>={int(persist / 60)}min'


def detect(sig, acc_mag, top, persist):
    return detect_persistent_ridges(
        sig, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC,
        min_freq=BAND_LO, max_freq=top,
        smooth_windows=SMOOTH_WINDOWS,
        min_persistence_sec=persist,
        max_freq_jump=MAX_FREQ_JUMP,
        peak_prominence_frac=PEAK_PROM_FRAC,
        welch_seg_sec=WELCH_SEG_SEC,
        max_gap_windows=MAX_GAP_WINDOWS,
        acc_mag=acc_mag,
    )


def ridge_rows(rr, **keys):
    rows = []
    for r in rr['ridges']:
        rows.append(dict(
            median_freq=float(r['median_freq']),
            duration_min=float(r['duration_sec']) / 60.0,
            start_hr=float(rr['t_hr'][r['start_idx']]),
            end_hr=float(rr['t_hr'][r['end_idx']]),
            n_present=int(r['n_present']),
            **keys,
        ))
    return rows


def summarise(rr):
    """Headline metrics a null has to beat."""
    durs = [r['duration_sec'] / 60.0 for r in rr['ridges']]
    freqs = [r['median_freq'] for r in rr['ridges']]
    return dict(
        n_ridges=len(durs),
        total_ridge_min=float(np.sum(durs)) if durs else 0.0,
        max_duration_min=float(np.max(durs)) if durs else 0.0,
        median_ridge_freq=float(np.median(freqs)) if freqs else np.nan,
    )


def phase_randomize(x, rng):
    """Surrogate with an identical power spectrum and destroyed time structure."""
    n = len(x)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phase = rng.uniform(0, 2 * np.pi, len(X))
    phase[0] = 0.0                      # keep the DC term real
    if n % 2 == 0:
        phase[-1] = 0.0                 # and the Nyquist term
    return np.fft.irfft(mag * np.exp(1j * phase), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true',
                    help='S1N1 / CRE / 20 surrogates, for timing')
    ap.add_argument('--sessions', type=int, nargs='*', default=None)
    ap.add_argument('--channels', nargs='*', default=None)
    ap.add_argument('--n-surrogates', type=int, default=200)
    ap.add_argument('--null-channel', default='CRE',
                    help='surrogates are expensive; run them on this channel')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    if args.pilot:
        idxs, channels, n_sur = [0], ['CRE'], 20
    else:
        idxs = args.sessions if args.sessions is not None else list(range(len(SESSION_META)))
        channels = args.channels or CHANNELS
        n_sur = args.n_surrogates

    rng = np.random.default_rng(20260929)
    ridges_all, summary_all, null_all = [], [], []

    for idx in idxs:
        t0 = time.time()
        s = load_session(idx, with_profile=True)
        acc_mag = s.cap['acc_mag']
        print(f'\n=== {s.label} ({s.subject}, {s.duration_hr:.1f} hr) ===', flush=True)

        for ch in channels:
            # band-limit once at the widest top; the detector re-restricts per spec
            sig = remove_acc_artifact(s.cap[ch], acc_mag, BAND_LO, 0.30)

            for top, persist in SPECS:
                name = spec_name(top, persist)
                rr = detect(sig, acc_mag, top, persist)
                summ = summarise(rr)
                ridges_all += ridge_rows(rr, session=s.label, subject=s.subject,
                                         channel=ch, spec=name)

                row = dict(session=s.label, subject=s.subject, channel=ch,
                           spec=name, band_top=top, min_persist_min=persist / 60,
                           duration_hr=float(s.duration_hr), **summ)

                # ---- surrogate null (one channel, one spec grid point at a time)
                if ch == args.null_channel and n_sur > 0:
                    null_counts, null_maxdur = [], []
                    for k in range(n_sur):
                        sur = phase_randomize(sig, rng)
                        rs = detect(sur, acc_mag, top, persist)
                        ns = summarise(rs)
                        null_counts.append(ns['n_ridges'])
                        null_maxdur.append(ns['max_duration_min'])
                        null_all.append(dict(session=s.label, channel=ch, spec=name,
                                             surrogate=k, **ns))
                    nc = np.array(null_counts)
                    nd = np.array(null_maxdur)
                    row.update(
                        null_n_mean=float(nc.mean()),
                        null_n_p95=float(np.percentile(nc, 95)),
                        null_maxdur_p95=float(np.percentile(nd, 95)),
                        # +1 correction: an observed value is never p = 0
                        p_count=float((np.sum(nc >= summ['n_ridges']) + 1) / (n_sur + 1)),
                        p_maxdur=float((np.sum(nd >= summ['max_duration_min']) + 1) / (n_sur + 1)),
                        n_surrogates=n_sur,
                    )

                summary_all.append(row)
                msg = (f'  {ch} {name:<20} {summ["n_ridges"]:3d} ridges  '
                       f'max {summ["max_duration_min"]:5.1f} min  '
                       f'median f {summ["median_ridge_freq"]:.4f} Hz')
                if 'p_count' in row:
                    msg += (f'   | null {row["null_n_mean"]:.1f} '
                            f'(p95 {row["null_n_p95"]:.0f})  p={row["p_count"]:.3f}')
                print(msg, flush=True)

        print(f'  [{time.time() - t0:.0f}s]', flush=True)

    tag = f'_{args.tag}' if args.tag else ''
    pd.DataFrame(ridges_all).to_csv(OUT / f'low_band_ridges{tag}.csv', index=False)
    pd.DataFrame(summary_all).to_csv(OUT / f'low_band_per_night{tag}.csv', index=False)
    if null_all:
        pd.DataFrame(null_all).to_csv(OUT / f'low_band_null{tag}.csv', index=False)
    print(f'\nwrote -> {OUT}')


if __name__ == '__main__':
    main()
