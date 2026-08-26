"""Where are the high-variance zones, and what stage are they in?

Variance is read in this project as an arousal proxy. This locates the epochs
that actually carry it and asks what the polysomnograph was scoring at the time,
specifically whether they concentrate in REM.

Motion decides whether the question is interesting at all: if the high-variance
epochs are simply movement, any stage association is a statement about how much
people move in each stage rather than about the sensor. Every result is reported
twice, once over all epochs and once with motion epochs removed.

Per 30 s epoch, per channel: variance of the <10 Hz signal (the quantity in
Figure 3 row D), accelerometer SD, and the scored stage. An epoch is a
high-variance zone when its variance is in the top decile of that recording.

Outputs -> reports/mean_value/high_variance_epochs.parquet
           reports/mean_value/high_variance_enrichment.csv
           writeup/figures/mean_value/high_variance_zones.png
           writeup/figures/mean_value/high_variance_enrichment.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.filters import lowpass                          # noqa: E402
from sleep_monitor.loader import load_session, load_sleep_profile   # noqa: E402
from sleep_monitor.sessions import SESSION_META                     # noqa: E402

OUT = ROOT / "reports" / "mean_value"
FIG = ROOT / "writeup" / "figures" / "mean_value"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

EPOCH = 30.0
LP_HZ = 10.0
CHANNELS = ["CH", "CLE", "CRE", "CLE-CRE"]
STAGES = ["Wake", "REM", "N1", "N2", "N3"]
CODE2STAGE = {4: "Wake", 3: "N1", 2: "N2", 1: "N3", 0: "REM"}
TOP_DECILE = 0.90


def epoch_reduce(x, n, fn):
    m = len(x) // n
    return fn(x[: m * n].reshape(-1, n), axis=1)


def build():
    rows = []
    for idx, meta in enumerate(SESSION_META):
        s = load_session(idx)
        sp = load_sleep_profile(s)
        if sp is None:
            continue
        fs = s.fs
        n = int(EPOCH * fs)
        cap = {c: s.cap[c].astype(np.float64) for c in ("CH", "CLE", "CRE")}
        cap["CLE-CRE"] = cap["CLE"] - cap["CRE"]
        nep = min(len(v) for v in cap.values()) // n

        codes = np.full(nep, -1)
        t_ep = (np.arange(nep) * EPOCH + EPOCH / 2) / 3600.0
        prof_t = np.asarray(sp["t_ep_hr"], float)
        prof_c = np.asarray(sp["codes"], int)
        j = np.searchsorted(prof_t, t_ep) - 1
        ok = (j >= 0) & (j < len(prof_c))
        codes[ok] = prof_c[j[ok]]

        acc = epoch_reduce(s.cap["acc_mag"].astype(np.float64), n, np.std)[:nep]
        rec = {"session": s.meta["label"], "subject": s.meta["label"][:2],
               "epoch": np.arange(nep), "t_hr": t_ep,
               "stage": [CODE2STAGE.get(c, "?") for c in codes], "acc_sd": acc}
        for ch in CHANNELS:
            rec["var_" + ch] = epoch_reduce(lowpass(cap[ch], LP_HZ, fs), n, np.var)[:nep]
        rows.append(pd.DataFrame(rec))
        print("  %s  %d epochs" % (s.meta["label"], nep))

    d = pd.concat(rows, ignore_index=True)
    d = d[d.stage.isin(STAGES)].copy()
    d["motion"] = d.groupby("session").acc_sd.transform(
        lambda v: v > v.quantile(TOP_DECILE))
    for ch in CHANNELS:
        d["hi_" + ch] = d.groupby("session")["var_" + ch].transform(
            lambda v: v > v.quantile(TOP_DECILE))
    d.to_parquet(OUT / "high_variance_epochs.parquet", index=False)
    return d


def enrichment(d):
    """Observed over expected occupancy of each stage among high-variance epochs."""
    rows = []
    for subset, sub in (("all epochs", d), ("motion-free", d[~d.motion])):
        for ch in CHANNELS:
            for sess, g in sub.groupby("session"):
                hi = g[g["hi_" + ch]]
                if len(hi) < 10:
                    continue
                for st in STAGES:
                    exp = (g.stage == st).mean()
                    obs = (hi.stage == st).mean()
                    if exp > 0.01:
                        rows.append({"subset": subset, "channel": ch,
                                     "session": sess, "stage": st,
                                     "observed": obs, "expected": exp,
                                     "enrichment": obs / exp})
    return pd.DataFrame(rows)


def fig_zones(d):
    sess = sorted(d.session.unique())
    fig, axes = plt.subplots(len(sess), 1, figsize=(13, 1.05 * len(sess)), sharex=True)
    colors = {"Wake": "#E74C3C", "N1": "#F39C12", "N2": "#3498DB",
              "N3": "#27AE60", "REM": "#8E44AD"}
    for ax, sname in zip(np.atleast_1d(axes), sess):
        g = d[d.session == sname].sort_values("t_hr")
        for st, c in colors.items():
            m = (g.stage == st).to_numpy()
            ax.fill_between(g.t_hr, 0, 1, where=m, color=c, alpha=0.30,
                            step="mid", lw=0)
        hi = g[g["hi_CH"] & ~g.motion]
        him = g[g["hi_CH"] & g.motion]
        ax.plot(him.t_hr, np.full(len(him), 0.5), "|", color="#7F8C8D", ms=8, mew=1.2)
        ax.plot(hi.t_hr, np.full(len(hi), 0.5), "|", color="#111111", ms=10, mew=1.6)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(sname, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_xlim(0, d.t_hr.max())
    axes[-1].set_xlabel("time (hours)")
    handles = [plt.Line2D([], [], color=c, lw=6, alpha=0.5, label=s)
               for s, c in colors.items()]
    handles += [plt.Line2D([], [], color="#111111", lw=1.6, label="high variance (CH)"),
                plt.Line2D([], [], color="#7F8C8D", lw=1.2, label="high variance, moving")]
    axes[0].legend(handles=handles, fontsize=7.5, ncol=7, frameon=False,
                   loc="lower left", bbox_to_anchor=(0, 1.05))
    fig.suptitle("High-variance epochs against the hypnogram, all twelve recordings",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "high_variance_zones.png", dpi=190)
    print("wrote %s" % (FIG / "high_variance_zones.png"))


def fig_enrichment(e):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, subset in enumerate(("all epochs", "motion-free")):
        sub = e[(e.subset == subset) & (e.channel == "CH")]
        for k, st in enumerate(STAGES):
            v = sub[sub.stage == st].enrichment.to_numpy()
            if not len(v):
                continue
            ax[i].scatter(np.full(len(v), k) + rng.uniform(-.12, .12, len(v)), v,
                          s=22, color="#34495E", alpha=0.65, zorder=3)
            ax[i].plot([k - .28, k + .28], [np.median(v)] * 2, lw=2.4,
                       color="#C0392B", zorder=4)
        ax[i].axhline(1, color="k", ls="--", lw=1.0)
        ax[i].set_xticks(range(len(STAGES)))
        ax[i].set_xticklabels(STAGES)
        ax[i].set_yscale("log")
        ax[i].set_title("%s  (CH)" % subset, loc="left", fontsize=10)
        ax[i].grid(axis="y", alpha=0.3, lw=0.5, which="both")
    ax[0].set_ylabel("enrichment  (observed / expected)")
    fig.suptitle("Which stages the high-variance epochs fall in", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "high_variance_enrichment.png", dpi=200)
    print("wrote %s" % (FIG / "high_variance_enrichment.png"))


def main():
    pq = OUT / "high_variance_epochs.parquet"
    d = pd.read_parquet(pq) if pq.exists() else build()
    e = enrichment(d)
    e.to_csv(OUT / "high_variance_enrichment.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nEnrichment of high-variance epochs by stage (median across recordings)\n")
    piv = e.pivot_table(index=["subset", "channel"], columns="stage",
                        values="enrichment", aggfunc="median")[STAGES]
    print(piv.round(2).to_string())
    print("\nhigh-variance epochs that are also motion epochs: %.1f%%"
          % (100 * d[d.hi_CH].motion.mean()))
    print("stage occupancy overall: %s"
          % d.stage.value_counts(normalize=True).round(3).to_dict())
    fig_zones(d)
    fig_enrichment(e)


main()
