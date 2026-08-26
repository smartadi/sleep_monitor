"""Figure: how the forehead channel relates to the temple differential.

Figure 3 used to carry CH beside CLE-CRE on the level and imbalance rows, and
the comparison was read off those overlays. With CH removed from both rows for
legibility, the relationship itself has nowhere to be seen. This gives it a
figure of its own.

The point the three panels make together: the two mounts agree about the body
and disagree about the mount. Band amplitude tracks closely in every recording,
their coherence falls away with frequency, and the forehead channel swings
several times further across a night. The per-session slow-level correlation is
reported in section 4.1 rather than plotted -- it added a panel without adding
an idea.

Inputs  reports/mean_value/ch_vs_diff_per_session.csv   (scale)
        reports/mean_value/ch_vs_diff_coherence.csv     (coherence spectrum)
        reports/mean_value/cle_cre_vs_ch.csv            (band amplitude)
Output  writeup/figures/mean_value/ch_vs_diff_relationship.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "writeup" / "figures" / "mean_value"
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(ROOT / "reports/mean_value/ch_vs_diff_per_session.csv").sort_values("session")
b = pd.read_csv(ROOT / "reports/mean_value/cle_cre_vs_ch.csv")
coh = pd.read_csv(ROOT / "reports/mean_value/ch_vs_diff_coherence.csv")
x = np.arange(len(d))

fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))

# A -- band amplitude: the agreement
w = 0.38
for k, (band, colr, lbl) in enumerate([("resp", "#2980B9", "respiratory"),
                                       ("card", "#E67E22", "cardiac")]):
    v = b[b.band == band].set_index("session").reindex(d.session).r.to_numpy()
    ax[0].bar(x + (k - 0.5) * w, v, width=w, color=colr, label=lbl)
ax[0].set_ylim(0, 1)
ax[0].set_ylabel("r,  band amplitude")
ax[0].set_title("A  Band amplitude tracks\n0 of 12 negative in either band",
                loc="left", fontsize=9.5)
ax[0].legend(fontsize=8, frameon=False, loc="lower right")

# B -- coherence against frequency, every recording
coh = coh[(coh.freq >= 0.002) & (coh.freq <= 3.0)]


def smooth(v, n=25):
    return np.convolve(v, np.ones(n) / n, mode="same")


grid, stack = None, []
for sess, g in coh.groupby("session"):
    g = g.sort_values("freq")
    f, y = g.freq.to_numpy(), smooth(g.coherence.to_numpy())
    if grid is None:
        grid = f
    stack.append(np.interp(grid, f, y))
    ax[1].plot(f, y, lw=0.7, color="#95A5A6", alpha=0.55, zorder=2)
stack = np.vstack(stack)
med = np.median(stack, axis=0)
for lo, hi, colr, lbl in [(0.1, 0.5, "#2980B9", "respiratory"),
                          (0.5, 3.0, "#E67E22", "cardiac")]:
    ax[1].axvspan(lo, hi, color=colr, alpha=0.07, zorder=1)
    ax[1].text(np.sqrt(lo * hi), 0.045, lbl, ha="center", fontsize=8,
               color=colr, style="italic")
ax[1].fill_between(grid, np.percentile(stack, 25, axis=0),
                   np.percentile(stack, 75, axis=0), color="#2C3E50",
                   alpha=0.15, zorder=3, label="interquartile range")
ax[1].plot(grid, med, lw=2.4, color="#2C3E50", zorder=4,
           label="median of 12 recordings")
ax[1].set_xscale("log")
ax[1].set_xlim(0.002, 3.0)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel("frequency (Hz)")
ax[1].set_ylabel("magnitude-squared coherence")
ax[1].set_title("B  Agreement falls with frequency", loc="left", fontsize=9.5)
ax[1].legend(fontsize=8, frameon=False, loc="upper left")
ax[1].grid(alpha=0.3, lw=0.5, which="both")

# C -- scale
ratio = (d.sd_CH / d.sd_DIFF).to_numpy()
ax[2].bar(x, ratio, color="#16A085")
ax[2].axhline(1, color="k", ls="--", lw=0.9)
ax[2].set_yscale("log")
ax[2].set_ylabel("SD(CH) / SD(CLE\u2212CRE)")
ax[2].set_title("C  CH swings further\nmedian %.1f\u00d7, up to %.1f\u00d7"
                % (np.median(ratio), ratio.max()), loc="left", fontsize=9.5)

for a in (ax[0], ax[2]):
    a.set_xticks(x)
    a.set_xticklabels(d.session, rotation=90, fontsize=7)
    a.grid(axis="y", alpha=0.3, lw=0.5)

fig.suptitle("Forehead channel (CH) against the temple differential (CLE\u2212CRE)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "ch_vs_diff_relationship.png", dpi=210)
print("wrote %s" % (OUT / "ch_vs_diff_relationship.png"))
print("band amp resp %.2f card %.2f | scale median %.1fx max %.1fx"
      % (b[b.band == "resp"].r.median(), b[b.band == "card"].r.median(),
         np.median(ratio), ratio.max()))
print("median coherence at 0.01 / 0.2 / 1.0 Hz: %.2f / %.2f / %.2f"
      % tuple(float(np.interp(v, grid, med)) for v in (0.01, 0.2, 1.0)))
