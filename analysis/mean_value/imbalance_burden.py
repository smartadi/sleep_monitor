"""Per-session integrated capacitance imbalance.

The night-scale imbalance marker is referenced to each session's own mean,
because the absolute operating point of CLE-CRE is set by where the mask sits
rather than by the head under it. That reference is what makes the marker
comparable across sessions -- and it is also why a *signed* integral is not a
usable session metric: integrating a baseline-referenced signal returns its
baseline. This script reports the integral that survives the reference:

    mean |imbalance|      time-average of the signed marker's magnitude, fF
    integral |imbalance|  the same accumulated over the night, fF*h
    left-dominant frac    share of clean epochs with the marker positive
    asymmetry index       (int+ - int-) / (int+ + int-), reported as the null
                          it is -- near zero by construction, and shown so that
                          the magnitude result is not read as lateralization

Motion epochs are excluded, matching the marker definition in section 3.4.

Outputs
    reports/mean_value/imbalance_burden.csv
    writeup/figures/mean_value/imbalance_burden.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = ROOT / "reports" / "mean_value" / "imbalance_epochs.csv"
OUT_CSV = ROOT / "reports" / "mean_value" / "imbalance_burden.csv"
OUT_FIG = ROOT / "writeup" / "figures" / "mean_value" / "imbalance_burden.png"

EPOCH_HOURS = 30.0 / 3600.0


def build() -> pd.DataFrame:
    e = pd.read_csv(EPOCHS)
    e = e[~e["motion"].astype(bool)]
    rows = []
    for sess, g in e.groupby("session", sort=True):
        x = g["imb_lp_fF"].to_numpy(float)
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        pos = x[x > 0].sum() * EPOCH_HOURS
        neg = -x[x < 0].sum() * EPOCH_HOURS
        rows.append({
            "session": sess,
            "subject": g["subject"].iloc[0],
            "n_epochs": len(x),
            "hours": len(x) * EPOCH_HOURS,
            "mean_abs_fF": float(np.abs(x).mean()),
            "integral_abs_fFh": float(np.abs(x).sum() * EPOCH_HOURS),
            "integral_pos_fFh": float(pos),
            "integral_neg_fFh": float(neg),
            "asymmetry": float((pos - neg) / (pos + neg)) if (pos + neg) else np.nan,
            "frac_left_dominant": float((x > 0).mean()),
        })
    return pd.DataFrame(rows)


def figure(d: pd.DataFrame) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
    x = np.arange(len(d))
    subj = d["subject"].astype("category").cat.codes
    colors = plt.cm.tab10(subj % 10)

    ax[0].bar(x, d["integral_abs_fFh"], color=colors)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("∫|imbalance| dt  (fF·h)")
    ax[0].set_title("A  Imbalance burden per night", loc="left", fontsize=10)

    # B: the same burden, paired within subject. A quantity that were a subject
    # trait would give flat lines; these span 2.0x to 5.8x.
    for k, (subj, g) in enumerate(d.groupby("subject", sort=True)):
        g = g.sort_values("session")
        ax[1].plot([0, 1], g["integral_abs_fFh"].values, "o-",
                   color=plt.cm.tab10(k % 10), label=subj, lw=1.6, ms=5)
    ax[1].set_yscale("log")
    ax[1].set_xlim(-0.3, 1.3)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["night 1", "night 2"])
    ax[1].set_ylabel("∫|imbalance| dt  (fF·h)")
    ax[1].set_title("B  Same subject, two nights", loc="left", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False, ncol=2)

    ax[2].bar(x, d["asymmetry"], color=colors)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylim(-1, 1)
    ax[2].set_ylabel("asymmetry index")
    ax[2].set_title("C  Net lateralization (null)", loc="left", fontsize=10)

    for a in (ax[0], ax[2]):
        a.set_xticks(x)
        a.set_xticklabels(d["session"], rotation=90, fontsize=7)
    for a in ax:
        a.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    print("wrote %s" % OUT_FIG)


def main():
    d = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_CSV, index=False)
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))
    print("\nwrote %s" % OUT_CSV)
    print("\nburden      %.1f to %.1f fF*h  (%.0fx)"
          % (d.integral_abs_fFh.min(), d.integral_abs_fFh.max(),
             d.integral_abs_fFh.max() / d.integral_abs_fFh.min()))
    print("mean |imb|  %.2f to %.2f fF" % (d.mean_abs_fF.min(), d.mean_abs_fF.max()))
    print("asymmetry   %+.2f to %+.2f  (near zero: the marker is baseline-referenced)"
          % (d.asymmetry.min(), d.asymmetry.max()))
    print("left frac   %.2f to %.2f" % (d.frac_left_dominant.min(), d.frac_left_dominant.max()))
    within = d.groupby("subject").integral_abs_fFh.agg(lambda s: s.max() / s.min())
    print("\nnight-to-night ratio within a subject: %s"
          % ", ".join("%s %.1fx" % (k, v) for k, v in within.items()))
    figure(d)


if __name__ == "__main__":
    main()
