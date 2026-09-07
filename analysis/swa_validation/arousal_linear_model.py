"""Can a linear model predict the EEG arousal rate from the capacitive signal?

Fitted on non-overlapping 20-minute windows rather than a smoothed grid, so
neighbouring observations are close to independent and the sample size is honest
(about 230 windows over twelve nights).

Everything is scored leave-one-SUBJECT-out. In-sample fit on twelve nights would
look good and mean nothing; the question is whether a model fitted on five
subjects says anything useful about the sixth.

Two baselines, because "beats zero" is not the bar:

  NIGHT MEAN -- predict every window of a night as that night's own average
  arousal rate. This is the score to beat. Anything that cannot is telling you
  about differences between nights, not about what is happening within one.

  STAGE COMPOSITION -- the same model given the fraction of each window spent in
  each sleep stage, from the PSG. Not available to a wearable, so not a competitor;
  it is the ceiling, showing how much of the within-night variation is reachable
  at all at this resolution.

Reported as correlation between predicted and observed on held-out subjects, and
as R^2 against each baseline.

Run from the repo root:
    .venv/Scripts/python.exe analysis/swa_validation/arousal_linear_model.py

Outputs -> reports/psg/arousal_linear_model.csv
           writeup/figures/prof_metrics/arousal_linear_model.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from sklearn.linear_model import LinearRegression   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VAR = ROOT / "reports" / "mean_value" / "high_variance_epochs.parquet"
EVENTS = ROOT / "reports" / "psg" / "arousal_events.csv"
OUT = ROOT / "reports" / "psg" / "arousal_linear_model.csv"
FIG = ROOT / "writeup" / "figures" / "prof_metrics"
FIG.mkdir(parents=True, exist_ok=True)

CH = "var_CLE-CRE"
WIN_MIN = 20.0
THRESH = 10.0
STAGES = ["Wake", "N1", "N2", "N3", "REM"]

C_INK = "#1B2A41"
C_MUTED = "#5A6472"
C_FAINT = "#D6DBE1"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.edgecolor": C_MUTED,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 250,
    "savefig.bbox": "tight",
})


def windows():
    """One row per non-overlapping 20-minute window."""
    var = pd.read_parquet(VAR)
    ev = pd.read_csv(EVENTS)
    w_hr = WIN_MIN / 60.0
    rows = []
    for sess, g in var.groupby("session"):
        g = g.sort_values("t_hr")
        e = ev[ev.session == sess].t_hr.to_numpy(float)
        edges = np.arange(0, g.t_hr.max(), w_hr)
        for lo in edges:
            hi = lo + w_hr
            m = (g.t_hr >= lo) & (g.t_hr < hi)
            if m.sum() < 30:                    # need a near-full window
                continue
            sub = g[m]
            v = np.log10(np.clip(sub[CH].to_numpy(float), 1e-3, None))
            rec = {"session": sess, "subject": sess[:2], "t_hr": lo,
                   "arousal_rate": float(((e >= lo) & (e < hi)).sum() / w_hr),
                   "log_var_mean": float(v.mean()),
                   "log_var_sd": float(v.std()),
                   "frac_above": float((sub[CH] > THRESH).mean()),
                   "acc_sd": float(sub.acc_sd.mean())}
            for s in STAGES:
                rec["frac_" + s] = float((sub.stage == s).mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def loso(d, feats):
    """Predict held-out subjects; returns predictions aligned to d."""
    pred = np.full(len(d), np.nan)
    for subj in d.subject.unique():
        te = (d.subject == subj).to_numpy()
        tr = ~te
        m = LinearRegression().fit(d.loc[tr, feats], d.loc[tr, "arousal_rate"])
        pred[te] = m.predict(d.loc[te, feats])
    return pred


def score(y, p):
    ok = np.isfinite(p) & np.isfinite(y)
    r = np.corrcoef(y[ok], p[ok])[0, 1]
    ss = np.sum((y[ok] - p[ok]) ** 2)
    return r, ss


def main():
    d = windows()
    print("%d windows over %d nights, %d subjects"
          % (len(d), d.session.nunique(), d.subject.nunique()))
    y = d.arousal_rate.to_numpy(float)

    models = {
        "CAP variance only": ["log_var_mean"],
        "CAP variance + spread + threshold": ["log_var_mean", "log_var_sd", "frac_above"],
        "CAP + motion": ["log_var_mean", "log_var_sd", "frac_above", "acc_sd"],
        "PSG stage composition (ceiling)": ["frac_" + s for s in STAGES],
    }

    night_mean = d.groupby("session").arousal_rate.transform("mean").to_numpy()
    cohort_mean = np.full(len(d), y.mean())
    ss_night = np.sum((y - night_mean) ** 2)
    ss_cohort = np.sum((y - cohort_mean) ** 2)

    rows, preds = [], {}
    for name, feats in models.items():
        p = loso(d, feats)
        r, ss = score(y, p)
        rows.append({"model": name, "n_features": len(feats), "loso_r": r,
                     "r2_vs_cohort_mean": 1 - ss / ss_cohort,
                     "r2_vs_night_mean": 1 - ss / ss_night})
        preds[name] = p
    rows.append({"model": "night's own mean (baseline)", "n_features": 0,
                 "loso_r": score(y, night_mean)[0],
                 "r2_vs_cohort_mean": 1 - ss_night / ss_cohort,
                 "r2_vs_night_mean": 0.0})

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print("\nLeave-one-subject-out\n")
    print(res.round(3).to_string(index=False))
    print("\nr2_vs_night_mean above 0 means the model beats simply knowing the "
          "night's average arousal rate.")

    fig, ax = plt.subplots(1, 2, figsize=(250 * MM, 100 * MM))
    best = "CAP + motion"
    ax[0].scatter(preds[best], y, s=16, alpha=0.6, color="#16A085")
    lim = [0, max(y.max(), np.nanmax(preds[best])) * 1.05]
    ax[0].plot(lim, lim, ls="--", lw=0.9, color=C_MUTED)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("predicted arousals/h (held-out subject)")
    ax[0].set_ylabel("observed arousals/h")
    ax[0].set_title("A  ·  %s, r = %.2f" % (best, res.set_index("model").loc[best, "loso_r"]),
                    loc="left", fontsize=10)

    m = res.set_index("model")
    names = [n for n in models] + ["night's own mean (baseline)"]
    vals = [m.loc[n, "r2_vs_night_mean"] for n in names]
    cols = ["#16A085" if v > 0 else "#C0392B" for v in vals]
    ax[1].barh(range(len(names)), vals, color=cols, height=0.6)
    ax[1].axvline(0, color=C_INK, lw=1.0)
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("R² against the night's own mean")
    ax[1].set_title("B  ·  anything left of zero is worse than the baseline",
                    loc="left", fontsize=10)

    for a in ax:
        a.grid(color=C_FAINT, lw=0.5)
        a.set_axisbelow(True)
    fig.suptitle("Predicting the EEG arousal rate from the mask, %g-minute windows, LOSO"
                 % WIN_MIN, fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p_out = FIG / "arousal_linear_model.png"
    fig.savefig(p_out); plt.close(fig)
    print("\nwrote %s" % p_out)


if __name__ == "__main__":
    main()
