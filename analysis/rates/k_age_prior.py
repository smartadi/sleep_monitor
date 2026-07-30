"""Leave-one-subject-out test: can subject age substitute for per-session k calibration?

The per-session k factor requires a reference rate to estimate. If k varies
systematically with age, an age-based prior could replace calibration entirely.
This tests that out-of-sample: fit k ~ age on five subjects, predict the sixth.

Baselines compared, per band:
  fixed        -- k = 1.0 (no calibration at all; raw estimator output)
  population   -- k = mean of the five training subjects (best constant prior)
  age-prior    -- k = a + b * age, fit on the five training subjects

Inputs : analysis/rates/outputs/k_vs_age_per_subject.csv  (from k_vs_age.py)
Outputs: analysis/rates/outputs/k_age_prior.csv
         analysis/rates/outputs/fig_k_age_prior.png
         analysis/rates/outputs/fig_k_vs_age_3panel.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent / "outputs"
BANDS = {"resp": "Respiratory", "card": "Cardiac"}
SEX_COLOR = {"M": "#2a78d6", "F": "#eb6834"}


def loso_predictions(sub: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-subject-out k predictions under each strategy."""
    rows = []
    for i, row in sub.iterrows():
        train = sub.drop(i)
        slope, intercept = np.polyfit(train.age, train.k_mean, 1)
        rows.append(
            dict(
                subj=row.subj,
                age=row.age,
                k_true=row.k_mean,
                k_fixed=1.0,
                k_population=train.k_mean.mean(),
                k_agefit=slope * row.age + intercept,
            )
        )
    pred = pd.DataFrame(rows)
    for strat in ("fixed", "population", "agefit"):
        pred[f"err_{strat}"] = (pred[f"k_{strat}"] - pred.k_true).abs()
    return pred


def loo_spearman(sub: pd.DataFrame) -> tuple[float, float]:
    """Range of Spearman rho when each subject is dropped in turn."""
    rhos = [spearmanr(sub.drop(i).age, sub.drop(i).k_mean).statistic for i in sub.index]
    return min(rhos), max(rhos)


def main() -> None:
    per_subj = pd.read_csv(OUT / "k_vs_age_per_subject.csv")

    results, preds = [], {}
    for band in BANDS:
        sub = per_subj[per_subj.band == band].reset_index(drop=True)
        pred = loso_predictions(sub)
        preds[band] = pred

        rho, p = spearmanr(sub.age, sub.k_mean)
        lo, hi = loo_spearman(sub)
        results.append(
            dict(
                band=band,
                n=len(sub),
                spearman_rho=rho,
                spearman_p=p,
                loo_rho_min=lo,
                loo_rho_max=hi,
                sign_stable=bool(lo * hi > 0),
                mae_k_fixed=pred.err_fixed.mean(),
                mae_k_population=pred.err_population.mean(),
                mae_k_agefit=pred.err_agefit.mean(),
            )
        )

    stats = pd.DataFrame(results)
    stats.to_csv(OUT / "k_age_prior.csv", index=False)
    print(stats.to_string(index=False))

    _plot_three_panel(per_subj, preds["resp"])
    _plot_error_bars(preds)


def _plot_three_panel(per_subj: pd.DataFrame, resp_pred: pd.DataFrame) -> None:
    """Main-text figure: k vs age for both bands, plus the LOSO calibration test."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (band, label) in zip(axes, BANDS.items()):
        sub = per_subj[per_subj.band == band]
        for _, r in sub.iterrows():
            ax.errorbar(
                r.age,
                r.k_mean,
                yerr=[[r.k_mean - min(r.k_n1, r.k_n2)], [max(r.k_n1, r.k_n2) - r.k_mean]],
                fmt="o" if r.sex == "M" else "s",
                color=SEX_COLOR[r.sex],
                markersize=8,
                capsize=3,
                ecolor="#888780",
            )
            # S4 and S5 are one year apart; nudge one label clear of the other
            dy = -12 if (band == "card" and r.subj == "S4") else 2
            ax.annotate(r.subj, (r.age, r.k_mean), textcoords="offset points",
                        xytext=(8, dy), fontsize=9)
        slope, intercept = np.polyfit(sub.age, sub.k_mean, 1)
        xs = np.array([sub.age.min() - 3, sub.age.max() + 3])
        ax.plot(xs, slope * xs + intercept, "--", color="#898781", lw=1.2)

        rho, p = spearmanr(sub.age, sub.k_mean)
        lo, hi = loo_spearman(sub)
        ax.set_title(
            f"{label} k vs age\n"
            rf"$\rho$={rho:+.2f}, p={p:.2f}  (leave-one-out $\rho$: {lo:+.2f} to {hi:+.2f})",
            fontsize=10,
        )
        ax.set_xlabel("Age (years)")
        ax.set_ylabel(f"{label} k")
        ax.grid(alpha=0.3)

    ax = axes[2]
    x = np.arange(len(resp_pred))
    ax.bar(x - 0.2, resp_pred.err_fixed, width=0.4, label="no calibration (k = 1.0)",
           color="#b4b2a9")
    ax.bar(x + 0.2, resp_pred.err_agefit, width=0.4, label="age prior (leave-one-out)",
           color="#2a78d6")
    ax.set_xticks(x)
    ax.set_xticklabels(resp_pred.subj)
    ax.set_ylabel("|predicted k − true k|")
    ax.set_title(
        "Respiratory k: age prior vs no calibration\n"
        f"mean error {resp_pred.err_agefit.mean():.3f} vs {resp_pred.err_fixed.mean():.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Calibration factor k and subject age (n = 6 subjects, 2 nights each)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_k_vs_age_3panel.png", dpi=200)
    print("wrote", OUT / "fig_k_vs_age_3panel.png")


def _plot_error_bars(preds: dict[str, pd.DataFrame]) -> None:
    """Supplementary figure: all three strategies, both bands."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (band, label) in zip(axes, BANDS.items()):
        pred = preds[band]
        x = np.arange(len(pred))
        for off, strat, color, name in [
            (-0.27, "fixed", "#b4b2a9", "k = 1.0"),
            (0.0, "population", "#eb6834", "population mean k"),
            (0.27, "agefit", "#2a78d6", "age prior"),
        ]:
            ax.bar(x + off, pred[f"err_{strat}"], width=0.26, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(pred.subj)
        ax.set_ylabel("|predicted k − true k|")
        ax.set_title(f"{label}  (leave-one-subject-out)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_k_age_prior.png", dpi=200)
    print("wrote", OUT / "fig_k_age_prior.png")


if __name__ == "__main__":
    main()
