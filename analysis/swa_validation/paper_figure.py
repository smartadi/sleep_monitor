"""Paper figure for the slow-wave activity validation.

The existing roc_curves.png carries a 24-entry legend and a title that calls the
EEG arm a sanity check, which is not how the comparison should be presented: the
EEG arm is the positive control that makes the capacitive null interpretable.

Two panels, both from the per-session results table:

  A  N3 discrimination, capacitive against contact EEG, paired per recording
  B  capacitive SWA against EEG SWA, per-recording correlation, with zero marked

Output -> writeup/figures/swa/swa_validation_paper.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "analysis" / "swa_validation" / "outputs" / "swa_validation_results.csv"
OUT = ROOT / "writeup" / "figures" / "swa"
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(RES).sort_values("session").reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

# A -- paired AUC
x = np.arange(len(d))
for i, r in d.iterrows():
    ax[0].plot([0, 1], [r.sws_auc, r.eeg_n3_auc], "-", color="#BBBBBB", lw=1.0, zorder=1)
ax[0].scatter(np.zeros(len(d)), d.sws_auc, s=44, color="#C0392B", zorder=3,
              label="capacitive  mean %.3f" % d.sws_auc.mean())
ax[0].scatter(np.ones(len(d)), d.eeg_n3_auc, s=44, color="#2980B9", zorder=3,
              label="contact EEG  mean %.3f" % d.eeg_n3_auc.mean())
ax[0].axhline(0.5, color="k", ls="--", lw=1.0)
ax[0].text(1.04, 0.5, "chance", fontsize=8, va="center")
ax[0].set_xlim(-0.35, 1.35)
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(["capacitive", "contact EEG"])
ax[0].set_ylabel("N3 discrimination (AUC)")
ax[0].set_ylim(0.35, 0.9)
ax[0].set_title("A  Same pipeline, two sensors", loc="left", fontsize=10)
ax[0].legend(fontsize=8, frameon=False, loc="upper left")
ax[0].grid(axis="y", alpha=0.3, lw=0.5)

# B -- per-session correlation with EEG SWA
ax[1].bar(x, d.swa_total_r_pearson, color="#7F8C8D")
ax[1].axhline(0, color="k", lw=0.9)
ax[1].set_xticks(x)
ax[1].set_xticklabels(d.session, rotation=90, fontsize=7)
ax[1].set_ylabel("r,  capacitive SWA vs EEG SWA")
ax[1].set_ylim(-0.25, 0.25)
ax[1].set_title("B  Slow-wave activity, epoch by epoch  (mean r = %+.3f)"
                % d.swa_total_r_pearson.mean(), loc="left", fontsize=10)
ax[1].grid(axis="y", alpha=0.3, lw=0.5)

fig.tight_layout()
fig.savefig(OUT / "swa_validation_paper.png", dpi=220)
print("wrote %s" % (OUT / "swa_validation_paper.png"))
print("cap AUC %.3f +- %.3f | eeg AUC %.3f +- %.3f | r %+.3f +- %.3f"
      % (d.sws_auc.mean(), d.sws_auc.std(), d.eeg_n3_auc.mean(), d.eeg_n3_auc.std(),
         d.swa_total_r_pearson.mean(), d.swa_total_r_pearson.std()))
