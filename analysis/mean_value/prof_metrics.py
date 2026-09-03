"""The three per-night metrics from the professor's 2026-08-26 email.

Measured on the two panels of the channel-evolution figure, per overnight
recording:

  1. Absolute area under the CLE-CRE curve. The curve is the DC mean of CLE-CRE
     referenced to its own session mean -- the quantity the top panel plots --
     so the area is the integral of its absolute deviation from that mean.
  2. Duration for which the CLE-CRE variance sits above a threshold.
  3. Frequency of variance impulses above the same threshold, an impulse being
     one contiguous run of epochs above it.

Two choices the email left open, both of which change the answer:

  THE THRESHOLD IS ABSOLUTE, SHARED BY EVERY NIGHT. A per-session percentile
  would put the same fraction of epochs above threshold in every recording by
  construction, so metric 2 would be a constant and could not correlate with
  anything. Reported over a sweep (2, 5, 10, 20, 50 fF^2) rather than at one
  value, since the distribution is heavy-tailed and no single cut is obviously
  right.

  EVERYTHING IS REPORTED BOTH AS A TOTAL AND PER HOUR. Recordings run 4.1 to
  8.7 hours. Totals scale with that, so a raw area or a raw impulse count would
  carry night length into any comparison against PSQI or age before physiology
  got a look in.

Motion is not excluded from the headline numbers: restlessness is part of what
these metrics are meant to capture. A motion-free column is reported alongside
so the choice can be revisited.

Reads the cached epoch tables, so no raw pass:
    reports/mean_value/imbalance_epochs.csv          d_fF   -> metric 1
    reports/mean_value/high_variance_epochs.parquet  var_*  -> metrics 2 and 3

Run from the repo root:
    .venv/Scripts/python.exe analysis/mean_value/prof_metrics.py

Outputs -> reports/mean_value/prof_metrics_per_night.csv
           reports/mean_value/prof_metrics_threshold_sweep.csv
           reports/mean_value/prof_metrics_threshold_policy.csv
           writeup/figures/prof_metrics/{method,metric1_area,metric2_duration,
                                          metric3_impulses,results_table,
                                          threshold_policy}.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMB = ROOT / "reports" / "mean_value" / "imbalance_epochs.csv"
VAR = ROOT / "reports" / "mean_value" / "high_variance_epochs.parquet"
AGE = ROOT / "analysis" / "rates" / "outputs" / "k_vs_age_per_subject.csv"
OUT = ROOT / "reports" / "mean_value"
FIG = ROOT / "writeup" / "figures" / "prof_metrics"
FIG.mkdir(parents=True, exist_ok=True)

CH = "CLE-CRE"
EPOCH_MIN = 0.5                       # 30 s epochs
EPOCH_HR = EPOCH_MIN / 60.0
THRESHOLDS = [2.0, 5.0, 10.0, 20.0, 50.0]     # fF^2
HEADLINE = 10.0

C_INK = "#1B2A41"
C_MUTED = "#5A6472"
C_FAINT = "#D6DBE1"
C_A = "#E67E22"      # metric 1, the CLE-CRE trace colour in the source figure
C_B = "#2980B9"      # metrics 2 and 3, the variance colour
C_RED = "#C0392B"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 11,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.edgecolor": C_MUTED,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def runs_above(mask):
    """Start and end indices of each contiguous True run — one run = one impulse."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], m.view(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def metric1(imb):
    """Absolute area under the mean-referenced CLE-CRE curve, per night."""
    rows = []
    for sess, g in imb.groupby("session"):
        d = g["d_fF"].to_numpy(float)
        d = d[np.isfinite(d)]
        hours = len(d) * EPOCH_HR
        rows.append({
            "session": sess,
            "hours": hours,
            "area_fF_h": float(np.abs(d).sum() * EPOCH_HR),
            "area_per_hour_fF": float(np.abs(d).mean()),
        })
    return pd.DataFrame(rows)


def metric23(var, thr, motion_free=False):
    """Duration above threshold and impulse count, per night."""
    rows = []
    col = "var_" + CH
    for sess, g in var.groupby("session"):
        g = g.sort_values("epoch")
        if motion_free:
            g = g[~g.motion]
        v = g[col].to_numpy(float)
        hours = len(v) * EPOCH_HR
        above = v > thr
        runs = runs_above(above)
        lens = np.array([b - a for a, b in runs], float) * EPOCH_MIN
        rows.append({
            "session": sess,
            "hours": hours,
            "dur_above_min": float(above.sum() * EPOCH_MIN),
            "dur_above_pct": float(100.0 * above.mean()) if len(v) else np.nan,
            "n_impulses": len(runs),
            "impulses_per_hour": float(len(runs) / hours) if hours else np.nan,
            "median_impulse_min": float(np.median(lens)) if len(lens) else 0.0,
        })
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_method(imb, var, sess="S1N1"):
    """One night, showing exactly what each of the three numbers measures."""
    gi = imb[imb.session == sess].sort_values("t_hr")
    gv = var[var.session == sess].sort_values("epoch")
    v = gv["var_" + CH].to_numpy(float)

    fig, ax = plt.subplots(2, 1, figsize=(250 * MM, 125 * MM), sharex=True,
                           gridspec_kw={"hspace": 0.22})

    a = ax[0]
    a.fill_between(gi.t_hr, 0, gi.d_fF, color=C_A, alpha=0.35, lw=0,
                   label="shaded area = metric 1")
    a.plot(gi.t_hr, gi.d_fF, color=C_A, lw=0.8)
    a.axhline(0, color=C_INK, ls="--", lw=0.8)
    a.set_ylabel("CLE−CRE − session mean\n(fF)")
    a.set_title("%s — what the three metrics measure" % sess, loc="left", pad=8)
    a.legend(loc="lower right", borderpad=0.3)

    b = ax[1]
    b.plot(gv.t_hr, v, color=C_B, lw=0.7)
    b.axhline(HEADLINE, color=C_RED, ls="--", lw=1.0)
    for i, (s, e) in enumerate(runs_above(v > HEADLINE)):
        b.axvspan(gv.t_hr.iloc[s], gv.t_hr.iloc[min(e, len(gv) - 1)],
                  color=C_RED, alpha=0.30, lw=0,
                  label="impulse above threshold" if i == 0 else None)
    b.set_yscale("log")
    b.set_ylabel("variance, CLE−CRE\n(fF$^2$)")
    b.set_xlabel("time (h)")
    b.text(1.0, 1.015, "threshold %g fF$^2$   ·   shaded total = metric 2   ·   "
           "shaded count = metric 3" % HEADLINE,
           transform=b.transAxes, ha="right", va="bottom", fontsize=8.5, color=C_MUTED)
    b.legend(loc="upper left")

    for x in ax:
        x.grid(axis="y", color=C_FAINT, lw=0.5)
        x.set_axisbelow(True)
    p = FIG / "method.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


def _bars(ax, labels, values, color, ylabel):
    ax.bar(range(len(values)), values, color=color, width=0.68)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=C_FAINT, lw=0.5)
    ax.set_axisbelow(True)


def fig_metric1(m1):
    m1 = m1.sort_values("session")
    fig, ax = plt.subplots(1, 2, figsize=(250 * MM, 105 * MM))
    _bars(ax[0], m1.session, m1.area_fF_h, C_A, "absolute area  (fF·h)")
    ax[0].set_title("Total area over the night", loc="left", fontsize=10)
    _bars(ax[1], m1.session, m1.area_per_hour_fF, C_A, "mean |deviation|  (fF)")
    ax[1].set_title("Area per hour — night length removed", loc="left", fontsize=10)
    fig.suptitle("Metric 1 — absolute area under the CLE−CRE curve", fontsize=11.5,
                 x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG / "metric1_area.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


def fig_metric2(sweep):
    head = sweep[sweep.threshold == HEADLINE].sort_values("session")
    fig, ax = plt.subplots(1, 2, figsize=(250 * MM, 105 * MM))
    _bars(ax[0], head.session, head.dur_above_pct, C_B, "% of the night")
    ax[0].set_title("At %g fF$^2$" % HEADLINE, loc="left", fontsize=10)
    for thr in THRESHOLDS:
        s = sweep[sweep.threshold == thr].sort_values("session")
        ax[1].plot(range(len(s)), s.dur_above_pct, "o-", lw=1.2, ms=4,
                   label="%g fF$^2$" % thr)
    ax[1].set_xticks(range(len(head)))
    ax[1].set_xticklabels(head.session, fontsize=8.5)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("% of the night")
    ax[1].set_title("Threshold sweep", loc="left", fontsize=10)
    ax[1].grid(axis="y", color=C_FAINT, lw=0.5, which="both")
    ax[1].set_axisbelow(True)
    ax[1].legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Metric 2 — time with CLE−CRE variance above threshold",
                 fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG / "metric2_duration.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


def fig_metric3(sweep):
    head = sweep[sweep.threshold == HEADLINE].sort_values("session")
    fig, ax = plt.subplots(1, 2, figsize=(250 * MM, 105 * MM))
    _bars(ax[0], head.session, head.impulses_per_hour, C_B, "impulses per hour")
    ax[0].set_title("At %g fF$^2$" % HEADLINE, loc="left", fontsize=10)
    for thr in THRESHOLDS:
        s = sweep[sweep.threshold == thr].sort_values("session")
        ax[1].plot(range(len(s)), s.impulses_per_hour, "o-", lw=1.2, ms=4,
                   label="%g fF$^2$" % thr)
    ax[1].set_xticks(range(len(head)))
    ax[1].set_xticklabels(head.session, fontsize=8.5)
    ax[1].set_ylabel("impulses per hour")
    ax[1].set_title("Threshold sweep", loc="left", fontsize=10)
    ax[1].grid(axis="y", color=C_FAINT, lw=0.5)
    ax[1].set_axisbelow(True)
    ax[1].legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Metric 3 — frequency of CLE−CRE variance impulses",
                 fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG / "metric3_impulses.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


def fig_table(t):
    """All twelve nights as a rendered table, for the deck."""
    cols = ["session", "age", "psqi", "hours", "area_fF_h", "area_per_hour_fF",
            "dur_above_min", "dur_above_pct", "n_impulses", "impulses_per_hour"]
    head = ["night", "age", "PSQI", "hours", "area\n(fF·h)", "area/hour\n(fF)",
            "time > thr\n(min)", "time > thr\n(%)", "impulses\n(n)", "impulses\n(per h)"]
    fmt = ["%s", "%.0f", "%.0f", "%.1f", "%.0f", "%.1f", "%.1f", "%.2f", "%.0f", "%.2f"]

    d = t.sort_values("session")
    cells = [[f % v if not (isinstance(v, float) and np.isnan(v)) else "—"
              for f, v in zip(fmt, row)] for row in d[cols].to_numpy()]

    fig, ax = plt.subplots(figsize=(250 * MM, 118 * MM))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="right")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.75)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(C_FAINT)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(color=C_INK, fontweight="bold")
            cell.set_facecolor("#EFF1F5")
        elif r % 2 == 0:
            cell.set_facecolor("#FAFBFC")
        if c == 0:
            cell.set_text_props(ha="left")
    ax.set_title("All twelve nights — threshold %g fF$^2$" % HEADLINE,
                 loc="left", fontsize=11.5, pad=16)
    p = FIG / "results_table.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


# ── threshold policy ─────────────────────────────────────────────────────────

POLICIES = [
    ("absolute %g fF$^2$" % HEADLINE, lambda v: HEADLINE, C_B),
    ("per-night 90th pct", lambda v: float(np.percentile(v, 90)), "#8E44AD"),
    ("10x night median", lambda v: 10.0 * float(np.median(v)), "#27AE60"),
]


def policy_table(var):
    """Every night under each threshold policy, for the comparison figure."""
    rows = []
    col = "var_" + CH
    for name, fn, _ in POLICIES:
        for sess, g in var.groupby("session"):
            g = g.sort_values("epoch")
            v = g[col].to_numpy(float)
            thr = fn(v)
            above = v > thr
            hours = len(v) * EPOCH_HR
            rows.append({
                "policy": name, "session": sess, "subject": sess[:2], "night": sess[-1],
                "threshold_fF2": thr, "dur_above_pct": 100.0 * above.mean(),
                "impulses_per_hour": len(runs_above(above)) / hours,
                "night_median_fF2": float(np.median(v)),
            })
    return pd.DataFrame(rows)


def fig_policy(pt):
    """Why the threshold is held the same on every night."""
    from scipy import stats as _st
    fig, ax = plt.subplots(1, 3, figsize=(250 * MM, 100 * MM))
    sess = sorted(pt.session.unique())

    for name, _, c in POLICIES:
        d = pt[pt.policy == name].set_index("session").loc[sess]
        ax[0].plot(range(len(sess)), d.dur_above_pct, "o-", color=c, lw=1.3, ms=4,
                   label=name)
    ax[0].set_xticks(range(len(sess)))
    ax[0].set_xticklabels(sess, rotation=90, fontsize=7.5)
    ax[0].set_ylabel("time above threshold  (% of night)")
    ax[0].set_title("A  ·  the percentile policy is flat by construction",
                    loc="left", fontsize=9.5)
    ax[0].legend(fontsize=8)

    for name, _, c in POLICIES:
        d = pt[pt.policy == name]
        w = d.pivot(index="subject", columns="night", values="dur_above_pct")
        r = _st.spearmanr(w["1"], w["2"]).statistic
        ax[1].scatter(w["1"], w["2"], s=34, color=c, alpha=0.85,
                      label="%s  ρ=%+.2f" % (name, r))
    lim = [0, max(pt.dur_above_pct) * 1.1]
    ax[1].plot(lim, lim, ls="--", lw=0.8, color=C_MUTED)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("night 1  (% of night)"); ax[1].set_ylabel("night 2")
    ax[1].set_title("B  ·  does it reproduce across a subject's two nights?",
                    loc="left", fontsize=9.5)
    ax[1].legend(fontsize=8, loc="upper left")

    for name, _, c in POLICIES:
        d = pt[pt.policy == name]
        r = _st.spearmanr(d.dur_above_pct, d.night_median_fF2).statistic
        ax[2].scatter(d.night_median_fF2, d.dur_above_pct, s=34, color=c, alpha=0.85,
                      label="%s  ρ=%+.2f" % (name, r))
    ax[2].set_xscale("log")
    ax[2].set_xlabel("night's median variance  (fF$^2$)")
    ax[2].set_ylabel("time above threshold  (%)")
    ax[2].set_title("C  ·  all policies track the night's overall level",
                    loc="left", fontsize=9.5)
    ax[2].legend(fontsize=8, loc="upper left")

    for a in ax:
        a.grid(color=C_FAINT, lw=0.5)
        a.set_axisbelow(True)
    fig.suptitle("Choosing the variance threshold", fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = FIG / "threshold_policy.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


def main():
    imb = pd.read_csv(IMB)
    var = pd.read_parquet(VAR)

    m1 = metric1(imb)

    sweep = []
    for thr in THRESHOLDS:
        s = metric23(var, thr)
        s["threshold"] = thr
        f = metric23(var, thr, motion_free=True)
        s["dur_above_pct_motionfree"] = f["dur_above_pct"].to_numpy()
        s["impulses_per_hour_motionfree"] = f["impulses_per_hour"].to_numpy()
        sweep.append(s)
    sweep = pd.concat(sweep, ignore_index=True)

    # subject-level age and PSQI, carried through for convenience only
    a = pd.read_csv(AGE)[["subj", "age", "psqi"]].drop_duplicates("subj")
    head = sweep[sweep.threshold == HEADLINE].drop(columns=["threshold", "hours"])
    t = m1.merge(head, on="session")
    t["subj"] = t.session.str[:2]
    t = t.merge(a, left_on="subj", right_on="subj", how="left").drop(columns="subj")

    t.to_csv(OUT / "prof_metrics_per_night.csv", index=False)
    sweep.to_csv(OUT / "prof_metrics_threshold_sweep.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nPer night, threshold %g fF^2\n" % HEADLINE)
    print(t[["session", "age", "psqi", "hours", "area_fF_h", "area_per_hour_fF",
             "dur_above_min", "dur_above_pct", "n_impulses",
             "impulses_per_hour"]].round(2).to_string(index=False))

    print("\nFigures")
    fig_method(imb, var)
    fig_metric1(m1)
    fig_metric2(sweep)
    fig_metric3(sweep)
    fig_table(t)

    pt = policy_table(var)
    pt.to_csv(OUT / "prof_metrics_threshold_policy.csv", index=False)
    fig_policy(pt)
    print("\ntables -> %s" % OUT)


if __name__ == "__main__":
    main()
