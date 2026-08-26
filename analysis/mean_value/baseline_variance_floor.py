"""Is the variance seen during sleep the sensor's noise, or the subject?

The variance of the capacitive signal is read in this project as a proxy for
arousal, and it varies with sleep stage. That reading only holds if the variance
is dominated by the subject rather than by the instrument. The mask was recorded
unworn for 15.8 minutes, on the same device, which gives the instrument's own
variance directly.

Both quantities are computed identically for the unworn mask and for every
recording:

  block variance   the quantity plotted in Figure 3 row D -- 10 s block variance
                   of the <10 Hz signal, in fF^2
  band amplitude   the quantity behind the stage profile -- log10 RMS in the
                   respiratory and cardiac bands on the 30 s grid

Outputs -> reports/mean_value/baseline_variance_floor.csv
           writeup/figures/mean_value/baseline_variance_floor.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import RESP_LO, RESP_HI, CARD_LO, CARD_HI   # noqa: E402
from sleep_monitor.filters import bandpass, lowpass                    # noqa: E402

OUT = ROOT / "reports" / "mean_value"
FIG = ROOT / "writeup" / "figures" / "mean_value"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

BLOCK_SEC = 10.0
EPOCH_SEC = 30.0
LP_HZ = 10.0
CHANNELS = ["CH", "CLE", "CRE", "CLE-CRE"]
STAGES = ["Wake", "REM", "N1", "N2", "N3"]


def baseline_signals():
    b = pd.read_csv(ROOT / "baseline noise" / "SM2_33.txt").dropna()
    fs = 1000.0 / np.median(np.diff(b.timeMS))
    sig = {c: b[c].to_numpy(float) * 1000.0 for c in ("CH", "CLE", "CRE")}
    sig["CLE-CRE"] = sig["CLE"] - sig["CRE"]
    return sig, fs


def block_var(x, fs):
    n = int(BLOCK_SEC * fs)
    m = len(x) // n
    return np.var(x[: m * n].reshape(-1, n), axis=1)


def band_rms(x, fs, lo, hi):
    y = bandpass(x, lo, hi, fs)
    n = int(EPOCH_SEC * fs)
    m = len(y) // n
    return np.sqrt((y[: m * n].reshape(-1, n) ** 2).mean(axis=1))


def main():
    sig, fs = baseline_signals()
    rows = []
    for ch in CHANNELS:
        x = sig[ch]
        v = block_var(lowpass(x, LP_HZ, fs), fs)
        rows.append({"channel": ch, "measure": "block variance (fF^2)",
                     "baseline_median": float(np.median(v))})
        for band, lo, hi in (("resp", RESP_LO, RESP_HI), ("card", CARD_LO, CARD_HI)):
            r = band_rms(x, fs, lo, hi)
            rows.append({"channel": ch, "measure": "%s log10 RMS" % band,
                         "baseline_median": float(np.log10(np.median(r)))})
    base = pd.DataFrame(rows)

    # sleep-time values, already computed per stage
    stage = pd.read_csv(OUT / "variance_by_stage.csv")
    comp = []
    for _, r in stage.iterrows():
        key = "%s log10 RMS" % r.band
        b = base[(base.channel == r.channel) & (base.measure == key)].baseline_median
        if not len(b):
            continue
        floor = float(b.iloc[0])
        for st in STAGES:
            if st in r and np.isfinite(r[st]):
                comp.append({"channel": r.channel, "band": r.band, "stage": st,
                             "sleep_log10_rms": float(r[st]), "baseline_log10_rms": floor,
                             "ratio": 10 ** (float(r[st]) - floor)})
    c = pd.DataFrame(comp)
    c.to_csv(OUT / "baseline_variance_floor.csv", index=False)

    pd.set_option("display.width", 200)
    print(base.round(4).to_string(index=False))
    print("\nSleep amplitude as a multiple of the unworn mask, by stage\n")
    piv = c.pivot_table(index=["band", "channel"], columns="stage",
                        values="ratio")[STAGES]
    print(piv.round(1).to_string())
    print("\nsmallest multiple anywhere: %.1fx  (%s)"
          % (c.ratio.min(), c.loc[c.ratio.idxmin(), ["band", "channel", "stage"]].to_dict()))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, band in enumerate(("resp", "card")):
        sub = piv.loc[band]
        for ch in sub.index:
            ax[i].plot(range(len(STAGES)), sub.loc[ch].values, "o-", lw=1.5, ms=5, label=ch)
        ax[i].axhline(1, color="k", ls="--", lw=1.0)
        ax[i].text(0.05, 1.15, "unworn mask", fontsize=8, color="k")
        ax[i].set_yscale("log")
        ax[i].set_xticks(range(len(STAGES)))
        ax[i].set_xticklabels(STAGES)
        ax[i].set_ylabel("amplitude ÷ unworn mask")
        ax[i].set_title("%s band" % band, loc="left", fontsize=10)
        ax[i].grid(alpha=0.3, lw=0.5, which="both")
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Sleep-time signal against the instrument's own floor "
                 "(same mask, unworn)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_variance_floor.png", dpi=200)
    print("\nwrote %s" % (FIG / "baseline_variance_floor.png"))


main()
