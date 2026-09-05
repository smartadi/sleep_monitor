"""Why the mask finds almost no arousals on some subjects.

Two subjects -- S3 and S4, all four of their nights -- register roughly one
capacitive event for every six arousals the technologist scored, while S1 and S2
sit near parity. The question is whether the mask disagrees with the EEG on those
subjects or simply has nothing to disagree with.

It is the second. Detection yield tracks how far the capacitive signal moves at
all: the 90th percentile of per-epoch variance is about 1 fF^2 on S3 and S4
against 5-35 fF^2 on S1, S2 and S6. A transient detector needs transients, and on
those subjects the trace sits near flat all night. The effect is a property of the
subject rather than the night -- both nights of each subject land in the same
place -- which points at coupling: how the mask sits on that person's temples.

Run from the repo root:
    .venv/Scripts/python.exe analysis/swa_validation/arousal_yield_vs_signal.py

Outputs -> reports/psg/arousal_yield_vs_signal.csv
           writeup/figures/prof_metrics/arousal_yield_vs_signal.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "reports" / "swa_validation" / "cap_events" / "cap_arousal_summary.csv"
VAR = ROOT / "reports" / "mean_value" / "high_variance_epochs.parquet"
COUNTS = ROOT / "reports" / "psg" / "arousal_counts.csv"
OUT = ROOT / "reports" / "psg"
FIG = ROOT / "writeup" / "figures" / "prof_metrics"
FIG.mkdir(parents=True, exist_ok=True)

C_INK = "#1B2A41"
C_MUTED = "#5A6472"
C_FAINT = "#D6DBE1"
SUBJ_COLOR = {"S1": "#2980B9", "S2": "#16A085", "S3": "#C0392B",
              "S4": "#E67E22", "S5": "#8E44AD", "S6": "#7F8C8D"}

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.edgecolor": C_MUTED,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def build():
    sm = pd.read_csv(SUMMARY)
    ac = pd.read_csv(COUNTS)
    v = pd.read_parquet(VAR)
    vs = (v.groupby("session")["var_CLE-CRE"]
          .agg(var_med="median", var_p90=lambda s: s.quantile(0.90))
          .reset_index())
    t = (sm[["session", "hours", "detected", "kept"]]
         .merge(vs, on="session")
         .merge(ac[["session", "subject", "arousal_index"]], on="session"))
    t["det_per_hour"] = t.detected / t.hours
    # events the mask finds for each arousal the technologist scored
    t["cap_per_scored"] = t.det_per_hour / t.arousal_index
    return t.sort_values("session")


def figure(t):
    fig, ax = plt.subplots(1, 2, figsize=(250 * MM, 105 * MM))

    a = ax[0]
    for sub, g in t.groupby("subject"):
        a.plot(g.var_p90, g.cap_per_scored, "o", ms=9, color=SUBJ_COLOR[sub],
               label=sub, alpha=0.9)
        if len(g) == 2:                     # join a subject's two nights
            a.plot(g.var_p90, g.cap_per_scored, "-", color=SUBJ_COLOR[sub],
                   lw=1.0, alpha=0.45)
    a.set_xscale("log")
    a.set_xlabel("capacitive excursion size\n(90th pct of epoch variance, fF$^2$)")
    a.set_ylabel("mask events per scored arousal")
    a.axhline(1.0, color=C_MUTED, ls="--", lw=0.8)
    a.text(0.02, 1.02, "parity", transform=a.get_yaxis_transform(),
           fontsize=8.5, color=C_MUTED)
    a.set_title("A  ·  yield follows how much the signal moves",
                loc="left", fontsize=10)
    a.legend(ncol=3, fontsize=8.5)

    b = ax[1]
    order = sorted(t.subject.unique())
    x = np.arange(len(order))
    for i, sub in enumerate(order):
        g = t[t.subject == sub].sort_values("session")
        for j, (_, r) in enumerate(g.iterrows()):
            b.bar(i + (j - 0.5) * 0.36, r.cap_per_scored, width=0.33,
                  color=SUBJ_COLOR[sub], alpha=0.95 if j == 0 else 0.55)
    b.set_xticks(x)
    b.set_xticklabels(order)
    b.axhline(1.0, color=C_MUTED, ls="--", lw=0.8)
    b.set_ylabel("mask events per scored arousal")
    b.set_xlabel("subject  (two bars = two nights)")
    b.set_title("B  ·  it is a property of the subject, not the night",
                loc="left", fontsize=10)

    for a_ in ax:
        a_.grid(color=C_FAINT, lw=0.5)
        a_.set_axisbelow(True)
    fig.suptitle("Why the mask finds few arousals on some subjects",
                 fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = FIG / "arousal_yield_vs_signal.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote %s" % p)


def main():
    t = build()
    t.to_csv(OUT / "arousal_yield_vs_signal.csv", index=False)
    pd.set_option("display.width", 220)
    print(t[["session", "subject", "arousal_index", "det_per_hour",
             "cap_per_scored", "var_p90"]].round(2).to_string(index=False))
    figure(t)


if __name__ == "__main__":
    main()
