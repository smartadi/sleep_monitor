"""The k-vs-age analysis, recomputed on the operational (fused) pipeline.

`k_age_prior.py` reads `outputs/k_vs_age_per_subject.csv`, whose k values come
from the older per-band single-channel estimators.  The manuscript now reports
rates from the agreement-gated five-channel fusion of loose peak counting, so
Table 3 / Table S1 and the age figure were quoting two different k series
(respiratory median 0.99 against a figure showing 1.04 -> 0.91).

This rebuilds the per-subject table from the same file that populates Table S1
and repeats the analysis, with two method changes the manuscript already states:

  * exact permutation p over all 720 orderings, instead of the asymptotic
    approximation, which is unreliable at n = 6;
  * the leave-one-subject-out calibration test reported as the primary result,
    since the correlation alone is not significant at this sample size.

Inputs : reports/rates/mask/table_s1_review_single_pipeline.csv   (per-session k)
         analysis/rates/outputs/k_vs_age_per_subject.csv          (demographics)
Outputs: analysis/rates/outputs/k_age_fused.csv
         analysis/rates/outputs/k_vs_age_per_subject_fused.csv
         writeup/figures/k_biomarker/fig_k_vs_age_3panel.png      (paper figure)

Run from the repo root:  python analysis/rates/k_age_fused.py
"""

from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "rates" / "outputs"
FIG = ROOT / "writeup" / "figures" / "k_biomarker" / "fig_k_vs_age_3panel.png"
SRC = ROOT / "reports" / "rates" / "mask" / "table_s1_review_single_pipeline.csv"
DEMO = OUT / "k_vs_age_per_subject.csv"

BANDS = {"resp": "Respiratory", "card": "Cardiac"}
SEX_COLOR = {"M": "#2a78d6", "F": "#eb6834"}


def exact_spearman_p(x, y) -> float:
    """Two-sided permutation p over every ordering. n = 6 -> 720 permutations."""
    rho = spearmanr(x, y).statistic
    y = np.asarray(y)
    null = [spearmanr(x, y[list(perm)]).statistic for perm in permutations(range(len(y)))]
    return float(np.mean(np.abs(null) >= abs(rho) - 1e-12))


def build_per_subject() -> pd.DataFrame:
    """Per-subject k from the fused pipeline, carrying demographics across."""
    ses = pd.read_csv(SRC)
    ses["subj"] = ses.Session.str[:2]
    demo = (pd.read_csv(DEMO)[["band", "subj", "age", "sex", "psqi"]]
            .drop_duplicates(subset=["band", "subj"]))

    rows = []
    for band, col in [("resp", "Resp k"), ("card", "Card k")]:
        g = ses.groupby("subj")[col]
        t = pd.DataFrame({
            "band": band,
            "subj": g.mean().index,
            "k_mean": g.mean().values,
            "k_n1": g.first().values,
            "k_n2": g.last().values,
        })
        t["dk"] = (t.k_n1 - t.k_n2).abs()
        rows.append(t)
    per = pd.concat(rows, ignore_index=True).merge(demo, on=["band", "subj"])
    return per.sort_values(["band", "age"]).reset_index(drop=True)


def loso(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in sub.iterrows():
        train = sub.drop(i)
        slope, intercept = np.polyfit(train.age, train.k_mean, 1)
        rows.append(dict(subj=row.subj, age=row.age, k_true=row.k_mean,
                         k_fixed=1.0, k_population=train.k_mean.mean(),
                         k_agefit=slope * row.age + intercept))
    pred = pd.DataFrame(rows)
    for s in ("fixed", "population", "agefit"):
        pred[f"err_{s}"] = (pred[f"k_{s}"] - pred.k_true).abs()
    return pred


def loo_rho_range(sub: pd.DataFrame) -> tuple[float, float]:
    rhos = [spearmanr(sub.drop(i).age, sub.drop(i).k_mean).statistic for i in sub.index]
    return min(rhos), max(rhos)


def main() -> None:
    per = build_per_subject()
    per.to_csv(OUT / "k_vs_age_per_subject_fused.csv", index=False)

    results, preds = [], {}
    for band in BANDS:
        sub = per[per.band == band].reset_index(drop=True)
        pred = loso(sub)
        preds[band] = pred
        rho = spearmanr(sub.age, sub.k_mean).statistic
        lo, hi = loo_rho_range(sub)
        results.append(dict(
            band=band, n=len(sub),
            k_median=sub.k_mean.median(), k_min=sub.k_mean.min(), k_max=sub.k_mean.max(),
            spearman_rho=rho,
            p_exact=exact_spearman_p(sub.age.values, sub.k_mean.values),
            p_asymptotic=spearmanr(sub.age, sub.k_mean).pvalue,
            loo_rho_min=lo, loo_rho_max=hi, sign_stable=bool(lo * hi > 0),
            dk_median=sub.dk.median(), dk_max=sub.dk.max(),
            n_dk_le_003=int((sub.dk <= 0.03).sum()),
            mae_k_fixed=pred.err_fixed.mean(),
            mae_k_population=pred.err_population.mean(),
            mae_k_agefit=pred.err_agefit.mean(),
        ))

    stats = pd.DataFrame(results)
    stats.to_csv(OUT / "k_age_fused.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(stats.round(4).to_string(index=False))

    plot_three_panel(per, preds["resp"])
    print("figure -> %s" % FIG)


def plot_three_panel(per: pd.DataFrame, resp_pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (band, label) in zip(axes, BANDS.items()):
        sub = per[per.band == band]
        for _, r in sub.iterrows():
            ax.errorbar(r.age, r.k_mean,
                        yerr=[[r.k_mean - min(r.k_n1, r.k_n2)],
                              [max(r.k_n1, r.k_n2) - r.k_mean]],
                        fmt="o" if r.sex == "M" else "s", color=SEX_COLOR[r.sex],
                        markersize=8, capsize=3, ecolor="#888780")
            dy = -12 if (band == "card" and r.subj == "S4") else 2
            ax.annotate(r.subj, (r.age, r.k_mean), textcoords="offset points",
                        xytext=(8, dy), fontsize=9)
        slope, intercept = np.polyfit(sub.age, sub.k_mean, 1)
        xs = np.array([sub.age.min() - 3, sub.age.max() + 3])
        ax.plot(xs, slope * xs + intercept, "--", color="#898781", lw=1.2)

        rho = spearmanr(sub.age, sub.k_mean).statistic
        p = exact_spearman_p(sub.age.values, sub.k_mean.values)
        lo, hi = loo_rho_range(sub.reset_index(drop=True))
        ax.set_title(f"{label} k vs age\n"
                     rf"$\rho$={rho:+.2f}, exact p={p:.3f}"
                     rf"  (leave-one-out $\rho$: {lo:+.2f} to {hi:+.2f})", fontsize=10)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel(f"{label} k")
        ax.grid(alpha=0.3)

    ax = axes[2]
    w, xs = 0.35, np.arange(len(resp_pred))
    ax.bar(xs - w / 2, resp_pred.err_agefit, w, label="age prior", color="#2a78d6")
    ax.bar(xs + w / 2, resp_pred.err_fixed, w, label="no calibration (k = 1)",
           color="#c8c6bf")
    ax.set_xticks(xs)
    ax.set_xticklabels(resp_pred.subj)
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("|k predicted − k true|")
    ax.set_title("Respiratory k: leave-one-subject-out\n"
                 f"mean |error| {resp_pred.err_agefit.mean():.3f} (age) vs "
                 f"{resp_pred.err_fixed.mean():.3f} (none)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    fig.savefig(OUT / "fig_k_vs_age_3panel_fused.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
