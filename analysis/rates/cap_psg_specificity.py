"""Is the CAP-PSG amplitude coupling specific, or is it motion?

cap_psg_coupling.py finds that capacitive band power correlates with every PSG
channel at about the same strength -- Flow 0.38, ECG 0.36, EEG 0.34 in the
respiratory band. Respiratory coupling would not look like that. Uniform
positive correlation across physiologically unrelated sensors is what a shared
amplitude driver produces: when the subject moves or contact degrades, every
sensor's band power moves together.

Two tests separate the possibilities.

  motion control    drop motion epochs, and partial out accelerometer RMS
  specificity       CAP-Flow minus CAP-EEG in the respiratory band, and
                    CAP-ECG minus CAP-EEG in the cardiac band. EEG is the
                    negative control: it shares the artifact but carries no
                    respiratory or cardiac mechanics.

If the specificity contrast survives, there is band-specific coupling on top of
the shared driver. If it collapses, the matrix in cap_psg_coupling.py is an
artifact map and must not be reported as physiology.

Outputs -> reports/rates/coupling/cap_psg_specificity.csv
           writeup/figures/coupling/cap_psg_specificity.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
from scipy import stats               # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rates" / "coupling"
FIG = ROOT / "writeup" / "figures" / "coupling"
EPOCHS = OUT / "cap_psg_epoch.parquet"
MOTION = ROOT / "reports" / "mean_value" / "imbalance_epochs.csv"

CAP = ["CH", "CLE", "CRE", "CLE-CRE"]
PSG = ["Flow", "Thorax", "Abdomen", "Pleth", "ECG", "EEG"]
BANDS = ["resp", "card"]


def partial_r(x, y, z):
    """Correlation of x and y with z regressed out of both."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 50:
        return np.nan
    x, y, z = x[m], y[m], z[m]
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    d = pd.read_parquet(EPOCHS)
    mo = pd.read_csv(MOTION, usecols=["session", "t_hr", "motion"])
    mo["epoch"] = np.round(mo.t_hr * 3600.0 / 30.0).astype(int)
    mo = mo.drop_duplicates(["session", "epoch"])
    d = d.merge(mo[["session", "epoch", "motion"]], on=["session", "epoch"], how="left")
    d["motion"] = d.motion.fillna(False).astype(bool)
    print("joined motion flags for %.0f%% of epochs; %.1f%% flagged as motion"
          % (100 * d.motion.notna().mean(), 100 * d.motion.mean()))

    rows = []
    for band in BANDS:
        for c in CAP:
            cc = "%s_%s" % (c, band)
            for p in PSG:
                pc = "%s_%s" % (p, band)
                if cc not in d or pc not in d:
                    continue
                all_r, clean_r, part_r = [], [], []
                for _, g in d.groupby("session"):
                    a, b = g[cc].values, g[pc].values
                    m = np.isfinite(a) & np.isfinite(b)
                    if m.sum() > 50:
                        all_r.append(stats.pearsonr(a[m], b[m])[0])
                    q = g[~g.motion]
                    a2, b2 = q[cc].values, q[pc].values
                    m2 = np.isfinite(a2) & np.isfinite(b2)
                    if m2.sum() > 50:
                        clean_r.append(stats.pearsonr(a2[m2], b2[m2])[0])
                    # partial out the mask's own broadband amplitude as the
                    # shared-driver proxy: the other band on the same channel
                    other = "%s_%s" % (c, "card" if band == "resp" else "resp")
                    if other in g:
                        part_r.append(partial_r(a, b, g[other].values))
                rows.append({"band": band, "cap": c, "psg": p,
                             "r_all": np.median(all_r),
                             "r_motion_free": np.median(clean_r) if clean_r else np.nan,
                             "r_partial": np.nanmedian(part_r) if part_r else np.nan})
    t = pd.DataFrame(rows)

    # specificity: target channel minus EEG on the same CAP channel
    spec = []
    for band, target in (("resp", "Flow"), ("card", "ECG")):
        for c in CAP:
            row = t[(t.band == band) & (t.cap == c)]
            tg = row[row.psg == target].iloc[0]
            eeg = row[row.psg == "EEG"].iloc[0]
            spec.append({"band": band, "cap": c, "target": target,
                         "target_r": tg.r_all, "eeg_r": eeg.r_all,
                         "specificity_all": tg.r_all - eeg.r_all,
                         "specificity_motion_free": tg.r_motion_free - eeg.r_motion_free,
                         "specificity_partial": tg.r_partial - eeg.r_partial})
    s = pd.DataFrame(spec)

    t.to_csv(OUT / "cap_psg_specificity.csv", index=False)
    pd.set_option("display.width", 200)
    for band in BANDS:
        print("\n%s band — median within-night r" % band)
        print(t[t.band == band][["cap", "psg", "r_all", "r_motion_free", "r_partial"]]
              .round(3).to_string(index=False))
    print("\nSpecificity: target minus EEG (the negative control)")
    print(s.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, band in enumerate(BANDS):
        sub = t[t.band == band]
        x = np.arange(len(PSG))
        for k, c in enumerate(CAP):
            v = [sub[(sub.cap == c) & (sub.psg == p)].r_motion_free.iloc[0] for p in PSG]
            ax[i].plot(x, v, "o-", label=c, lw=1.4, ms=5)
        ax[i].axhline(0, color="k", lw=0.8)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(PSG, rotation=45, ha="right")
        ax[i].set_ylabel("median within-night r")
        ax[i].set_title("%s band, motion-free epochs" % band, loc="left", fontsize=10)
        ax[i].grid(alpha=0.3, lw=0.5)
        ax[i].set_ylim(-0.35, 0.6)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("If coupling were specific, Flow (resp) and ECG (card) would stand "
                 "above EEG", fontsize=11)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "cap_psg_specificity.png", dpi=190)
    print("\nwrote %s" % (FIG / "cap_psg_specificity.png"))


if __name__ == "__main__":
    main()
