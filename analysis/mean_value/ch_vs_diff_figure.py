"""Figure: how the forehead channel relates to the temple differential.

Figure 3 used to carry CH beside CLE-CRE on the level and imbalance rows, and
the comparison was read off those overlays. With CH removed from both rows for
legibility, the relationship itself has nowhere to be seen -- it survives only
as numbers in the text. This gives it a figure of its own.

The point the four panels make together: the two mounts agree about the body and
disagree about the mount. Band amplitude tracks closely in every recording;
the slow level does not, and in three recordings runs the other way.

Inputs  reports/mean_value/ch_vs_diff_per_session.csv   (levels, coherence, scale)
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
x = np.arange(len(d))

fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.3))

# A -- slow level: the disagreement
cols = ["#C0392B" if v < 0 else "#7F8C8D" for v in d.r_level]
ax[0].bar(x, d.r_level, color=cols)
ax[0].axhline(0, color="k", lw=0.9)
ax[0].set_ylim(-1, 1)
ax[0].set_ylabel("r,  slow level")
ax[0].set_title("A  Slow level: mount-specific\n%d of %d recordings negative"
                % ((d.r_level < 0).sum(), len(d)), loc="left", fontsize=9.5)

# B -- band amplitude: the agreement
w = 0.38
for k, (band, colr, lbl) in enumerate([("resp", "#2980B9", "respiratory"),
                                       ("card", "#E67E22", "cardiac")]):
    v = (b[b.band == band].set_index("session").reindex(d.session).r.to_numpy())
    ax[1].bar(x + (k - 0.5) * w, v, width=w, color=colr, label=lbl)
ax[1].axhline(0, color="k", lw=0.9)
ax[1].set_ylim(-1, 1)
ax[1].set_ylabel("r,  band amplitude")
ax[1].set_title("B  Band amplitude: shared\n0 of 12 negative in either band",
                loc="left", fontsize=9.5)
ax[1].legend(fontsize=8, frameon=False, loc="lower right")

# C -- coherence by band
for k, (c, colr, lbl) in enumerate([("coh_slow", "#8E44AD", "slow (<0.01 Hz)"),
                                    ("coh_resp", "#2980B9", "respiratory"),
                                    ("coh_card", "#E67E22", "cardiac")]):
    ax[2].plot(x, d[c], "o-", color=colr, lw=1.3, ms=4,
               label="%s  median %.2f" % (lbl, d[c].median()))
ax[2].set_ylim(0, 1)
ax[2].set_ylabel("magnitude-squared coherence")
ax[2].set_title("C  Agreement falls with frequency", loc="left", fontsize=9.5)
ax[2].legend(fontsize=8, frameon=False)

# D -- scale
ratio = (d.sd_CH / d.sd_DIFF).to_numpy()
ax[3].bar(x, ratio, color="#16A085")
ax[3].axhline(1, color="k", ls="--", lw=0.9)
ax[3].set_yscale("log")
ax[3].set_ylabel("SD(CH) / SD(CLE−CRE)")
ax[3].set_title("D  CH swings further\nmedian %.1f×, up to %.1f×"
                % (np.median(ratio), ratio.max()), loc="left", fontsize=9.5)

for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(d.session, rotation=90, fontsize=7)
    a.grid(axis="y", alpha=0.3, lw=0.5)

fig.suptitle("Forehead channel (CH) against the temple differential (CLE−CRE)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "ch_vs_diff_relationship.png", dpi=210)
print("wrote %s" % (OUT / "ch_vs_diff_relationship.png"))
print("level r %.2f..%.2f (%d neg) | band amp resp %.2f card %.2f | scale median %.1fx max %.1fx"
      % (d.r_level.min(), d.r_level.max(), (d.r_level < 0).sum(),
         b[b.band == "resp"].r.median(), b[b.band == "card"].r.median(),
         np.median(ratio), ratio.max()))
