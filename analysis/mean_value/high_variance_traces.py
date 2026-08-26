"""High-variance zones shown as traces, with a hypnogram ladder above each.

The tick-mark version showed where the zones were but not what they looked like.
This draws the quantity itself: per-recording epoch variance of the <10 Hz
capacitive signal, on a log axis because motion spikes span orders of magnitude,
with the top-decile threshold marked and the epochs above it picked out. Epochs
that coincide with movement are drawn in a separate colour, because roughly half
of them do.

Reads reports/mean_value/high_variance_epochs.parquet, written by
high_variance_zones.py.

Output -> writeup/figures/mean_value/high_variance_traces.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec   # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "mean_value" / "high_variance_epochs.parquet"
FIG = ROOT / "writeup" / "figures" / "mean_value"
FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from sleep_monitor.config import STAGE_LADDER, STAGE_COLORS, STAGE_ORDER  # noqa: E402

CHANNEL = "CH"
LADDER = list(STAGE_LADDER)                       # Wake, N1, N2, N3, REM
YPOS = {s: -i for i, s in enumerate(LADDER)}
STAGE_COLOR = {n: STAGE_COLORS[c] for n, c in zip(STAGE_LADDER, STAGE_ORDER)}


def main():
    d = pd.read_parquet(SRC)
    sessions = sorted(d.session.unique())

    fig = plt.figure(figsize=(13.5, 1.75 * len(sessions)))
    gs = gridspec.GridSpec(2 * len(sessions), 1, figure=fig,
                           height_ratios=[0.42, 1.0] * len(sessions),
                           hspace=0.0)
    tmax = d.t_hr.max()

    for i, sname in enumerate(sessions):
        g = d[d.session == sname].sort_values("t_hr")
        thr = g["var_" + CHANNEL].quantile(0.90)
        hi = g[g["var_" + CHANNEL] > thr]

        # --- hypnogram ladder
        axh = fig.add_subplot(gs[2 * i, 0])
        y = g.stage.map(YPOS).to_numpy(float)
        axh.step(g.t_hr, y, where="mid", lw=1.0, color="#2C3E50")
        for st in LADDER:
            m = (g.stage == st).to_numpy()
            axh.plot(g.t_hr[m], np.full(m.sum(), YPOS[st]), "|",
                     color=STAGE_COLOR[st], ms=4, mew=2.0, alpha=0.9)
        axh.set_yticks([YPOS[s] for s in LADDER])
        axh.set_yticklabels(LADDER, fontsize=6.5)
        axh.set_ylim(-4.6, 0.6)
        axh.set_xlim(0, tmax)
        axh.set_xticks([])
        axh.tick_params(axis="y", length=0)
        axh.grid(axis="y", alpha=0.18, lw=0.4)
        for side in ("top", "right", "bottom"):
            axh.spines[side].set_visible(False)
        axh.set_ylabel(sname, rotation=0, ha="right", va="center",
                       fontsize=9, labelpad=26, fontweight="bold")

        # --- the variance trace itself
        axv = fig.add_subplot(gs[2 * i + 1, 0])
        axv.plot(g.t_hr, g["var_" + CHANNEL], lw=0.7, color="#34495E",
                 alpha=0.9, zorder=2)
        axv.axhline(thr, color="#C0392B", ls="--", lw=0.9, zorder=3)
        still = hi[~hi.motion]
        moving = hi[hi.motion]
        axv.scatter(moving.t_hr, moving["var_" + CHANNEL], s=13, zorder=4,
                    color="#BDC3C7", edgecolor="none")
        axv.scatter(still.t_hr, still["var_" + CHANNEL], s=15, zorder=5,
                    color="#C0392B", edgecolor="none")
        axv.set_yscale("log")
        axv.set_xlim(0, tmax)
        axv.set_ylabel("var (fF$^2$)", fontsize=7)
        axv.tick_params(labelsize=6.5)
        axv.grid(alpha=0.2, lw=0.4, which="both")
        for side in ("top", "right"):
            axv.spines[side].set_visible(False)
        if i < len(sessions) - 1:
            axv.set_xticklabels([])
        else:
            axv.set_xlabel("time (hours)", fontsize=9)


    handles = [plt.Line2D([], [], marker="o", ls="none", ms=5, color="#C0392B",
                          label="top decile, still"),
               plt.Line2D([], [], marker="o", ls="none", ms=5, color="#BDC3C7",
                          label="top decile, moving"),
               plt.Line2D([], [], ls="--", color="#C0392B", lw=0.9,
                          label="90th percentile of the recording")]
    fig.legend(handles=handles, fontsize=8.5, ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Capacitive variance (%s) with the scored hypnogram above each recording"
                 % CHANNEL, fontsize=12, y=1.006)
    fig.savefig(FIG / "high_variance_traces.png", dpi=170, bbox_inches="tight")
    print("wrote %s" % (FIG / "high_variance_traces.png"))


main()
