"""Two questions from the co-author's notes.

1. VARIANCE AND REM. Signal variance is read in this project as a proxy for
   cortical arousal. Does it track sleep stage, and REM in particular? Section
   4.6 already reports that harmonic-comb episodes follow REM, so a second,
   simpler REM relationship in the same recordings is worth testing rather than
   assuming.

2. CLE-CRE AGAINST CH. Section 4.1 states that the two differ in scale and that
   their level correlation ranges widely, but the relationship is asserted
   rather than reported. This quantifies it per session, in amplitude and by
   band.

Uses the cached per-epoch band amplitudes from cap_psg_coupling.py (log RMS on
the 30 s grid) joined to the PSG staging from the rate rerun, so no raw pass.

Outputs -> reports/mean_value/variance_by_stage.csv
           reports/mean_value/cle_cre_vs_ch.csv
           writeup/figures/mean_value/variance_by_stage.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from scipy import stats                 # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COUPLING = ROOT / "reports" / "rates" / "coupling" / "cap_psg_epoch.parquet"
PHASE_A = ROOT / "artifacts" / "rate_rerun_phase_a.parquet"
OUT = ROOT / "reports" / "mean_value"
FIG = ROOT / "writeup" / "figures" / "mean_value"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

CAP = ["CH", "CLE", "CRE", "CLE-CRE"]
STAGES = ["Wake", "N1", "N2", "N3", "REM"]


def load():
    d = pd.read_parquet(COUPLING)
    a = pd.read_parquet(PHASE_A)
    a = (a[(a.band == "resp") & (a.channel == "CRE")][["session", "epoch", "stage"]]
         .drop_duplicates())
    d = d.merge(a, on=["session", "epoch"], how="inner")
    d["stage"] = d["stage"].astype(str)
    return d[d.stage.isin(STAGES)]


def variance_by_stage(d):
    """Per-subject stage means, so a long recording cannot dominate."""
    rows = []
    for band in ("resp", "card"):
        for ch in CAP:
            col = "%s_%s" % (ch, band)
            if col not in d:
                continue
            per = (d.groupby(["session", "stage"])[col].mean().reset_index())
            per["subject"] = per.session.str[:2]
            per = per.groupby(["subject", "stage"])[col].mean().reset_index()
            piv = per.pivot(index="subject", columns="stage", values=col)
            if not {"REM"}.issubset(piv.columns):
                continue
            other = piv[[c for c in ["N1", "N2", "N3"] if c in piv]].mean(axis=1)
            diff = (piv["REM"] - other).dropna()
            if len(diff) < 4:
                continue
            rows.append({
                "band": band, "channel": ch,
                "REM_minus_NREM": float(diff.mean()),
                "n_subj_REM_higher": int((diff > 0).sum()), "n_subj": len(diff),
                "wilcoxon_p": float(stats.wilcoxon(diff).pvalue),
                **{s: float(piv[s].mean()) for s in STAGES if s in piv},
            })
    return pd.DataFrame(rows)


def cle_cre_vs_ch(d):
    rows = []
    for band in ("resp", "card"):
        a, b = "CLE-CRE_%s" % band, "CH_%s" % band
        if a not in d or b not in d:
            continue
        for sess, g in d.groupby("session"):
            x, y = g[a].to_numpy(float), g[b].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 100:
                continue
            r = float(np.corrcoef(x[m], y[m])[0, 1])
            # amplitude ratio in linear units: these are log10 RMS
            ratio = float(10 ** (np.median(y[m]) - np.median(x[m])))
            rows.append({"band": band, "session": sess, "r": r,
                         "CH_over_diff_amplitude": ratio})
    t = pd.DataFrame(rows)
    summ = (t.groupby("band")
            .agg(median_r=("r", "median"), min_r=("r", "min"), max_r=("r", "max"),
                 n_negative=("r", lambda s: int((s < 0).sum())),
                 median_ratio=("CH_over_diff_amplitude", "median"))
            .reset_index())
    return t, summ


def figure(d, v):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, band in enumerate(("resp", "card")):
        col = "CLE-CRE_%s" % band
        per = (d.groupby(["session", "stage"])[col].mean().reset_index())
        per["subject"] = per.session.str[:2]
        per = per.groupby(["subject", "stage"])[col].mean().reset_index()
        piv = per.pivot(index="subject", columns="stage", values=col)
        piv = piv[[s for s in STAGES if s in piv.columns]]
        # centre each subject so between-subject offset does not hide the shape
        centred = piv.sub(piv.mean(axis=1), axis=0)
        for subj, row in centred.iterrows():
            ax[i].plot(range(len(row)), row.values, "o-", lw=1.2, ms=4, alpha=0.75,
                       label=subj)
        ax[i].axhline(0, color="k", lw=0.8)
        ax[i].set_xticks(range(len(centred.columns)))
        ax[i].set_xticklabels(centred.columns)
        ax[i].set_ylabel("log$_{10}$ RMS, centered per subject")
        ax[i].set_title("%s band, CLE−CRE" % band, loc="left", fontsize=10)
        ax[i].grid(alpha=0.3, lw=0.5)
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Capacitive band amplitude by sleep stage", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "variance_by_stage.png", dpi=200)
    print("wrote %s" % (FIG / "variance_by_stage.png"))


def main():
    d = load()
    print("epochs with stage labels: %d across %d sessions"
          % (len(d), d.session.nunique()))

    v = variance_by_stage(d)
    v.to_csv(OUT / "variance_by_stage.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nREM minus mean(N1,N2,N3), per subject, log10 RMS\n")
    print(v[["band", "channel", "REM_minus_NREM", "n_subj_REM_higher", "n_subj",
             "wilcoxon_p"]].round(4).to_string(index=False))

    print("\nStage means (log10 RMS), CLE−CRE\n")
    print(v[v.channel == "CLE-CRE"][["band"] + STAGES].round(3).to_string(index=False))

    t, summ = cle_cre_vs_ch(d)
    t.to_csv(OUT / "cle_cre_vs_ch.csv", index=False)
    print("\nCLE−CRE against CH, per-epoch amplitude, within night\n")
    print(summ.round(3).to_string(index=False))

    figure(d, v)


if __name__ == "__main__":
    main()
