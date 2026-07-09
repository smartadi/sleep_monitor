"""
Does the accelerometer motion canceller remove ONLY motion, or also physiology?

Tests the highlighted manuscript claim "...so that only motion energy within that band
was removed." For each session and band, on CLE-CRE:
  (1) fraction of in-band variance removed by the OLS canceller;
  (2) whether removed energy tracks motion — per-window corr(removed energy, acc energy);
  (3) whether physiology is preserved — mean in-band magnitude-squared coherence with the
      PSG reference (Flow for respiration, Pleth for cardiac) BEFORE vs AFTER cancellation.
If coherence with the reference is preserved/improved while variance (esp. in high-motion
windows) is removed, the canceller targets motion, not physiology.

Outputs -> analysis/rates/outputs/motion_cancel_validation.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import coherence

from sleep_monitor.loader import load_session
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.config import RESP_LO, RESP_HI, CARD_LO, CARD_HI

FS = 100.0
OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)
BANDS = {'resp': (RESP_LO, RESP_HI, 'Flow'), 'card': (CARD_LO, CARD_HI, 'Pleth')}
WIN = int(30 * FS)


def mean_band_coh(x, y, lo, hi):
    f, Cxy = coherence(x, y, fs=FS, nperseg=int(60 * FS))
    m = (f >= lo) & (f <= hi)
    return float(np.mean(Cxy[m]))


def win_energy(sig):
    n = len(sig) // WIN
    return np.var(sig[:n * WIN].reshape(n, WIN), axis=1)


def run_session(idx):
    s = load_session(idx)
    cap = s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    acc = s.cap['acc_mag'].astype(np.float64)
    rows = []
    for band, (lo, hi, refname) in BANDS.items():
        ref = s.psg.get(refname)
        before = bandpass(cap, lo, hi, FS)
        after = remove_acc_artifact(cap, acc, lo, hi, FS)
        removed = before - after
        acc_bp = bandpass(acc, lo, hi, FS)

        frac_removed = 1.0 - np.var(after) / (np.var(before) + 1e-20)
        # removed energy vs motion energy, per 30-s window
        n = min(len(win_energy(removed)), len(win_energy(acc_bp)))
        r_mot = np.corrcoef(win_energy(removed)[:n], win_energy(acc_bp)[:n])[0, 1]

        coh_before = coh_after = np.nan
        if ref is not None:
            ref_bp = bandpass(ref.astype(np.float64), lo, hi, FS)
            coh_before = mean_band_coh(before, ref_bp, lo, hi)
            coh_after = mean_band_coh(after, ref_bp, lo, hi)

        rows.append(dict(session=s.label, band=band, ref=refname,
                         frac_var_removed=frac_removed,
                         corr_removed_vs_motion=r_mot,
                         coh_ref_before=coh_before, coh_ref_after=coh_after,
                         coh_delta=coh_after - coh_before))
    return rows


def main():
    all_rows = []
    for idx in range(len(SESSION_META)):
        try:
            rows = run_session(idx)
        except Exception as e:
            print(f'[{idx}] FAIL {e}'); continue
        all_rows.extend(rows)
        for r in rows:
            print(f"{r['session']} {r['band']:4s}: removed={r['frac_var_removed']*100:4.1f}%  "
                  f"corr(removed,motion)={r['corr_removed_vs_motion']:+.2f}  "
                  f"coh_ref {r['coh_ref_before']:.3f}->{r['coh_ref_after']:.3f} "
                  f"(Δ{r['coh_delta']:+.3f})")
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, 'motion_cancel_validation.csv'), index=False)
    print('\n=== medians by band ===')
    for band in ('resp', 'card'):
        b = df[df.band == band]
        print(f'{band}: frac_removed={b.frac_var_removed.median()*100:.1f}%  '
              f'corr(removed,motion)={b.corr_removed_vs_motion.median():+.2f}  '
              f'coh_before={b.coh_ref_before.median():.3f}  coh_after={b.coh_ref_after.median():.3f}  '
              f'coh_delta_median={b.coh_delta.median():+.3f}  '
              f'(sessions where coh preserved/improved: {int((b.coh_delta>=-0.01).sum())}/{len(b)})')
    print(f'\nSaved {OUT}/motion_cancel_validation.csv')


if __name__ == '__main__':
    main()
