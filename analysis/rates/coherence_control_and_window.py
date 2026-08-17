"""What does 4.1's coherence of 0.31 mean? Two questions, one script.

1. THE CONTROL. 4.1 reports coherence between the capacitive channel and the PSG
   reference at the ground-truth rate frequency, and compares it to a
   phase-randomized surrogate null and to a two-PSG-sensor upper bound. It has no
   negative *channel*. This runs the identical statistic with the contact EEG in
   place of the reference: a real, simultaneously recorded signal that shares the
   recording environment but carries no respiratory or cardiac mechanics.

2. THE WINDOW. Every analysis epoch in the paper is 30 s, and the coherence in
   4.1 is computed inside that epoch with 10 s segments for respiration -- five
   segments. Magnitude-squared coherence estimated from N segments has an
   expected value of about 1/N under independence, so a five-segment estimate
   carries a bias floor near 0.2 whatever the signals are. This measures that
   floor empirically, by pairing the capacitive channel with a time-reversed and
   with a circularly shifted reference, and repeats the whole comparison at a
   five-minute window (19 segments) so the two window lengths can be judged
   side by side.

Nothing here changes the epoch convention. It establishes what a coherence value
means at 30 s, so 4.1 can report the number with its floor rather than against
zero.

Outputs -> reports/rates/coupling/coherence_control.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sleep_monitor.config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI  # noqa: E402
from sleep_monitor.filters import bandpass                                # noqa: E402
from sleep_monitor.loader import load_session                             # noqa: E402
from sleep_monitor.sessions import SESSION_META                           # noqa: E402
from signal_validation_enhanced import coherence_at_frequency             # noqa: E402

OUT = ROOT / "reports" / "rates" / "coupling"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = {"resp": (RESP_LO, RESP_HI, "Flow"), "card": (CARD_LO, CARD_HI, "Pleth")}
WINDOWS = {"30 s (the analysis epoch)": 30.0, "5 min": 300.0}


def gt_freq(sig, fs, lo, hi):
    """Dominant frequency of the reference inside the band, for this window."""
    from scipy.signal import welch
    nper = min(len(sig), int(fs * 10.0))
    f, p = welch(sig, fs=fs, nperseg=nper)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(p[m])]) if m.any() else np.nan


def main():
    rows = []
    for meta in SESSION_META:
        s = load_session(meta)
        cap = (s.cap["CLE"].astype(float) + s.cap["CRE"].astype(float)) / 2.0
        for band, (lo, hi, ref_name) in BANDS.items():
            if ref_name not in s.psg or "EEG" not in s.psg:
                continue
            ref = s.psg[ref_name].astype(float)
            eeg = s.psg["EEG"].astype(float)
            n = min(len(cap), len(ref), len(eeg))
            c = bandpass(cap[:n], lo, hi, FS)
            r = bandpass(ref[:n], lo, hi, FS)
            e = bandpass(eeg[:n], lo, hi, FS)

            for wname, wsec in WINDOWS.items():
                win = int(wsec * FS)
                vals = {"reference": [], "eeg": [], "shifted": [], "reversed": []}
                nseg = max(1, int(win / (FS * (10.0 if band == "resp" else 4.0))))
                for i in range(n // win):
                    sl = slice(i * win, (i + 1) * win)
                    cw, rw, ew = c[sl], r[sl], e[sl]
                    if np.std(cw) < 1e-12 or np.std(rw) < 1e-12:
                        continue
                    f0 = gt_freq(rw, FS, lo, hi)
                    if not np.isfinite(f0):
                        continue
                    vals["reference"].append(
                        coherence_at_frequency(cw, rw, FS, f0, (lo, hi)))
                    vals["eeg"].append(
                        coherence_at_frequency(cw, ew, FS, f0, (lo, hi)))
                    # estimator floor: same signals, destroyed alignment
                    vals["shifted"].append(
                        coherence_at_frequency(np.roll(cw, len(cw) // 2), rw, FS,
                                               f0, (lo, hi)))
                    vals["reversed"].append(
                        coherence_at_frequency(cw[::-1], rw, FS, f0, (lo, hi)))
                if not vals["reference"]:
                    continue
                rows.append({
                    "session": s.meta["label"], "band": band, "window": wname,
                    "n_windows": len(vals["reference"]),
                    "segments": nseg,
                    **{k: float(np.median(v)) for k, v in vals.items()},
                })
        print("  %s done" % s.meta["label"])

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "coherence_control.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nMedian coherence at the reference frequency, across the twelve recordings\n")
    g = (d.groupby(["band", "window"])
         [["reference", "eeg", "shifted", "reversed", "segments"]]
         .median().round(3))
    print(g.to_string())
    print("\nMargin over each control (reference minus control):")
    for (band, win), r in g.iterrows():
        print("  %-5s %-24s  over EEG %+.3f   over shifted %+.3f   "
              "(1/N floor = %.2f with %d segments)"
              % (band, win, r["reference"] - r["eeg"],
                 r["reference"] - r["shifted"], 1.0 / r["segments"], r["segments"]))

    print("\nPer-session sign test, reference vs EEG:")
    from scipy import stats
    for (band, win), sub in d.groupby(["band", "window"]):
        diff = (sub["reference"] - sub["eeg"]).values
        p = stats.wilcoxon(diff).pvalue if len(diff) > 5 else np.nan
        print("  %-5s %-24s  %d/%d recordings positive, median %+.3f, p = %.3f"
              % (band, win, (diff > 0).sum(), len(diff), np.median(diff), p))


if __name__ == "__main__":
    main()
