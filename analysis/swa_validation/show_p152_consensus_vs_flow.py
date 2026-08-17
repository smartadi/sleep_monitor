#!/usr/bin/env python
"""
Substantiate the manuscript p152 claim: agreement between the consensus
respiratory GT and Flow-only.

Claim under test:
  "median absolute difference between consensus and Flow-only was 0.06 br/min,
   though 29% of epochs differed by more than 1 br/min."

Loads artifacts/consolidated_resp_gt.parquet, computes the per-epoch
|consensus - Flow| distribution (br/min), prints the numbers, and writes a
two-panel figure (ECDF + Bland-Altman) we can show / drop into the supplement.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "artifacts" / "consolidated_resp_gt.parquet"
OUT = ROOT / "writeup" / "figures" / "mask_rate_detection"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET)
# rates stored in Hz -> br/min
flow = df["rate_flow"].to_numpy(float) * 60.0
cons = df["rate_consensus"].to_numpy(float) * 60.0

both = np.isfinite(flow) & np.isfinite(cons)
flow, cons = flow[both], cons[both]
diff = cons - flow
adiff = np.abs(diff)

n = adiff.size
median_abs = float(np.median(adiff))
pct_gt1 = float(np.mean(adiff > 1.0) * 100)
pct_gt2 = float(np.mean(adiff > 2.0) * 100)
mean_bias = float(np.mean(diff))
loa = (mean_bias - 1.96 * np.std(diff), mean_bias + 1.96 * np.std(diff))

print(f"epochs with both signals : {n}")
print(f"median |consensus-Flow|  : {median_abs:.3f} br/min")
print(f"% epochs |diff| > 1 br/min: {pct_gt1:.1f}%")
print(f"% epochs |diff| > 2 br/min: {pct_gt2:.1f}%")
print(f"mean bias (cons-Flow)    : {mean_bias:+.3f} br/min")
print(f"95% limits of agreement  : [{loa[0]:+.2f}, {loa[1]:+.2f}] br/min")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: ECDF of |difference|
xs = np.sort(adiff)
ys = np.arange(1, n + 1) / n
axA.plot(xs, ys, color="#2C3E50", lw=1.6)
axA.axvline(median_abs, color="#27AE60", ls="--", lw=1.2,
            label=f"median = {median_abs:.2f} br/min")
axA.axvline(1.0, color="#C0392B", ls=":", lw=1.2,
            label=f"1 br/min ({pct_gt1:.0f}% exceed)")
axA.set_xlim(0, 3)
axA.set_xlabel("|consensus − Flow-only|  (br/min)")
axA.set_ylabel("cumulative fraction of epochs")
axA.set_title("A. Per-epoch agreement (ECDF)")
axA.legend(fontsize=8, loc="lower right")
axA.grid(alpha=0.3)

# Panel B: Bland-Altman consensus vs Flow
mean_rate = (cons + flow) / 2.0
axB.scatter(mean_rate, diff, s=3, alpha=0.15, color="#2980B9", edgecolors="none")
axB.axhline(mean_bias, color="#27AE60", lw=1.3, label=f"bias {mean_bias:+.2f}")
axB.axhline(loa[1], color="#C0392B", ls="--", lw=1.0,
            label=f"95% LoA [{loa[0]:+.1f}, {loa[1]:+.1f}]")
axB.axhline(loa[0], color="#C0392B", ls="--", lw=1.0)
axB.set_ylim(-4, 4)
axB.set_xlabel("mean of consensus & Flow  (br/min)")
axB.set_ylabel("consensus − Flow  (br/min)")
axB.set_title("B. Bland–Altman")
axB.legend(fontsize=8, loc="upper right")
axB.grid(alpha=0.3)

fig.suptitle(
    f"Consensus vs Flow-only respiratory GT  (N = {n:,} epochs)  |  "
    f"median |Δ| = {median_abs:.2f} br/min, {pct_gt1:.0f}% exceed 1 br/min",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_png = OUT / "figS_consensus_vs_flow_agreement.png"
fig.savefig(out_png, dpi=160)
print(f"\nwrote {out_png}")
