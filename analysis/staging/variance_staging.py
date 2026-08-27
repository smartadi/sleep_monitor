"""
Variance and staging — given a window's capacitive variance, what sleep stage?

Signal variance is read in this project as a proxy for cortical arousal: it is
large in wake and light sleep and small in deep sleep. This asks the reporting
question directly — over a window of some length, does the variance of the
capacitive signal tell you the stage?

Two directions, both descriptive, subject as the unit of analysis:

  FORWARD  — per-subject median variance by stage: is there a depth ordering
             (Wake high → N3 low, REM returning up), and how many of the six
             subjects show it?
  INVERSE  — bin each subject's window variance into terciles and report the
             stage composition of each bin: what stage is a low/mid/high-variance
             window most likely to be?

Separability is summarized by a single descriptive AUC per subject (N3-vs-rest
by low variance, Wake-vs-rest by high variance) — not a trained classifier. A
window-length sweep (30 s to 5 min) shows whether longer windows sharpen the
relationship.

No inferential p-value hunting: per-subject direction counts and medians are the
evidence, with one grouped (subject-level) Friedman reported for scale only.

Uses the cached <10 Hz per-epoch variance in
reports/mean_value/high_variance_epochs.parquet (var_CH etc., with stage, motion
flag), so no raw pass.

Outputs -> reports/staging/variance_stage_depth.csv
           reports/staging/variance_stage_inverse.csv
           reports/staging/variance_stage_separability.csv
           writeup/figures/staging/variance_staging.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.group_stats import pooled_vs_grouped_stage_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "reports" / "mean_value" / "high_variance_epochs.parquet"
OUT = ROOT / "reports" / "staging"
FIG = ROOT / "writeup" / "figures" / "staging"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
VARCOL = "var_CH"          # forehead carries the depth signal (project_variance_sleep_depth)
WINDOWS = [1, 5, 10]       # epochs: 30 s, 2.5 min, 5 min


def load(drop_motion=True):
    d = pd.read_parquet(CACHE)
    d = d[d.stage.isin(STAGES)].copy()
    d["subject"] = d.session.str[:2]
    d["logvar"] = np.log10(np.clip(d[VARCOL], 1e-6, None))
    if drop_motion:
        d = d[~d.motion]
    return d.sort_values(["session", "epoch"]).reset_index(drop=True)


def smooth_window(d, win):
    """Rolling-median log-variance over `win` consecutive epochs, within a session."""
    if win == 1:
        d = d.copy()
        d["wvar"] = d["logvar"]
        return d
    out = []
    for _, g in d.groupby("session"):
        g = g.copy()
        g["wvar"] = g["logvar"].rolling(win, center=True, min_periods=win).median()
        out.append(g)
    return pd.concat(out, ignore_index=True).dropna(subset=["wvar"])


# ── FORWARD: depth ordering ───────────────────────────────────────────────────

def depth_ordering(d):
    per = d.groupby(["subject", "stage"])["logvar"].median().reset_index()
    piv = per.pivot(index="subject", columns="stage", values="logvar")
    piv = piv[[s for s in STAGES if s in piv.columns]]
    counts = dict(
        n_subj=len(piv),
        n3_lowest=int((piv.idxmin(axis=1) == "N3").sum()),
        wake_highest=int((piv.idxmax(axis=1) == "Wake").sum()),
        wake_to_n3_monotone=int(((piv["Wake"] > piv["N1"]) & (piv["N1"] > piv["N2"])
                                 & (piv["N2"] > piv["N3"])).sum()),
        rem_above_n3=int((piv["REM"] > piv["N3"]).sum()),
    )
    return piv, counts


# ── INVERSE: variance level -> stage composition ──────────────────────────────

def inverse_mapping(dw):
    dw = dw.copy()
    dw["level"] = dw.groupby("subject")["wvar"].transform(
        lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop"))
    tab = pd.crosstab(dw["level"], dw["stage"], normalize="index")
    tab = tab.reindex(columns=[s for s in STAGES if s in tab.columns]).fillna(0.0)
    return tab


# ── separability (descriptive AUC) ────────────────────────────────────────────

def separability(dw):
    rows = []
    for target, sign in (("N3", -1.0), ("Wake", +1.0)):
        for subj, g in dw.groupby("subject"):
            y = (g.stage == target).astype(int)
            if y.sum() < 10 or (1 - y).sum() < 10:
                continue
            auc = roc_auc_score(y, sign * g["wvar"])
            rows.append({"target": target, "subject": subj, "auc": float(auc),
                         "n_target": int(y.sum())})
    return pd.DataFrame(rows)


def peri_rem_onset(d, half_epochs=20, min_rem=3):
    """Average log-variance in a +/- window around REM onset, per subject.

    A REM onset is an epoch entering REM from non-REM, with at least `min_rem`
    REM epochs following and a non-REM baseline before. Motion epochs are kept
    (motion may itself be a REM-state signal). Each subject's onsets are averaged,
    then the cohort mean is taken, so a session with many onsets cannot dominate.
    Baseline is the -10..-5 min pre-onset window.
    """
    def onsets(g):
        s = g.stage.values
        out = []
        for i in range(1, len(s)):
            if (s[i] == "REM" and s[i - 1] != "REM"
                    and (s[i:i + min_rem] == "REM").all()
                    and (s[max(0, i - 3):i] != "REM").all()):
                out.append(i)
        return out

    subj = {}
    n_onsets = {}
    for sj, gs in d.groupby("subject"):
        curves = []
        for _, g in gs.groupby("session"):
            g = g.sort_values("epoch").reset_index(drop=True)
            lv = g["logvar"].values
            for i in onsets(g):
                if i - half_epochs >= 0 and i + half_epochs < len(lv):
                    curves.append(lv[i - half_epochs:i + half_epochs + 1])
        if curves:
            subj[sj] = np.nanmean(curves, axis=0)
            n_onsets[sj] = len(curves)
    M = np.vstack(list(subj.values()))
    base = M[:, :half_epochs - 10].mean(axis=1, keepdims=True)
    Mc = M - base
    t = np.arange(-half_epochs, half_epochs + 1) * 0.5
    pre_last5 = Mc[:, half_epochs - 10:half_epochs].mean(1)
    pre_prior5 = Mc[:, half_epochs - 20:half_epochs - 10].mean(1)
    post5 = Mc[:, half_epochs:half_epochs + 10].mean(1)
    summary = dict(
        n_subjects=len(subj), total_onsets=int(sum(n_onsets.values())),
        onsets_per_subject=n_onsets,
        rise_in_last5min_over_prior5=int((pre_last5 - pre_prior5 > 0).sum()),
        step_up_post_onset=int((post5 > 0).sum()),
        cohort_pre_last5=float(np.nanmean(pre_last5)),
        cohort_post5=float(np.nanmean(post5)),
    )
    return t, Mc, subj, summary


def window_sweep(d):
    rows = []
    for win in WINDOWS:
        dw = smooth_window(d, win)
        sep = separability(dw)
        for target in ("N3", "Wake"):
            a = sep[sep.target == target]["auc"]
            rows.append({"win_epochs": win, "win_min": win * 0.5, "target": target,
                         "median_auc": float(a.median()),
                         "n_subj_auc_above_0.6": int((a > 0.6).sum()),
                         "n_subj": int(len(a))})
    return pd.DataFrame(rows)


# ── figure ────────────────────────────────────────────────────────────────────

def figure(piv, inv, sweep):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    # A: per-subject depth profile, centered
    c = piv.sub(piv.mean(axis=1), axis=0)
    for subj, row in c.iterrows():
        ax[0].plot(range(len(row)), row.values, "o-", lw=1.3, ms=4, alpha=0.8, label=subj)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(range(len(c.columns)))
    ax[0].set_xticklabels(c.columns)
    ax[0].set_ylabel("log$_{10}$ variance, centered per subject")
    ax[0].set_title("A. Variance falls with sleep depth\n(N3 lowest in "
                    f"{int((piv.idxmin(axis=1)=='N3').sum())}/{len(piv)} subjects)",
                    loc="left", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    ax[0].grid(alpha=0.3, lw=0.5)

    # B: inverse stacked bars
    bottom = np.zeros(len(inv))
    colors = {"Wake": "#E74C3C", "N1": "#F39C12", "N2": "#3498DB",
              "N3": "#2ECC71", "REM": "#9B59B6"}
    for s in inv.columns:
        ax[1].bar(range(len(inv)), inv[s].values * 100, bottom=bottom,
                  label=s, color=colors.get(s, "gray"))
        bottom += inv[s].values * 100
    ax[1].set_xticks(range(len(inv)))
    ax[1].set_xticklabels([str(x) for x in inv.index])
    ax[1].set_xlabel("per-subject variance tercile")
    ax[1].set_ylabel("stage composition (%)")
    ax[1].set_title("B. Low variance is deep sleep,\nhigh variance is wake", loc="left", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    # C: window-length sweep of AUC
    for target, mk in (("N3", "o-"), ("Wake", "s--")):
        s = sweep[sweep.target == target]
        ax[2].plot(s["win_min"], s["median_auc"], mk, lw=1.5, ms=6, label=f"{target} vs rest")
    ax[2].axhline(0.5, color="gray", ls=":", lw=1)
    ax[2].set_xlabel("window length (min)")
    ax[2].set_ylabel("median AUC across subjects")
    ax[2].set_ylim(0.45, 0.85)
    ax[2].set_title("C. Separability vs window length", loc="left", fontsize=10)
    ax[2].legend(fontsize=8, frameon=False)
    ax[2].grid(alpha=0.3, lw=0.5)

    fig.suptitle("Capacitive variance as a sleep-depth indicator (forehead CH, motion-free epochs)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "variance_staging.png", dpi=200, bbox_inches="tight")
    print("wrote", FIG / "variance_staging.png")


def peri_rem_figure(t, Mc, subj_curves):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    subjects = list(subj_curves.keys())
    base = np.array([subj_curves[s] for s in subjects])
    base = base - base[:, :len(t) // 2 - 10].mean(axis=1, keepdims=True)
    for i, s in enumerate(subjects):
        ax.plot(t, base[i], lw=1.0, alpha=0.45, label=s)
    m = np.nanmean(Mc, axis=0)
    se = np.nanstd(Mc, axis=0) / np.sqrt(Mc.shape[0])
    ax.plot(t, m, "k-", lw=2.4, label="cohort mean")
    ax.fill_between(t, m - se, m + se, color="k", alpha=0.15)
    ax.axvline(0, color="#9B59B6", lw=1.5, ls="--")
    ax.axhline(0, color="gray", lw=0.8)
    ax.text(0.3, ax.get_ylim()[1] * 0.9, "REM onset", color="#9B59B6", fontsize=9)
    ax.set_xlabel("time relative to REM onset (min)")
    ax.set_ylabel("log$_{10}$ variance change from pre-onset baseline")
    ax.set_title("Capacitive variance steps up at REM onset, without a lead-up rise\n"
                 "(forehead CH, motion included)", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "variance_peri_rem.png", dpi=200, bbox_inches="tight")
    print("wrote", FIG / "variance_peri_rem.png")


def main():
    # Primary analysis is motion-free; motion-included is reported alongside,
    # because motion may itself track the brain state (e.g. wake, phasic REM).
    d = load(drop_motion=True)
    d_all = load(drop_motion=False)
    print(f"{len(d)} motion-free ({len(d_all)} motion-included) epochs "
          f"across {d.session.nunique()} sessions")

    print("\n=== depth ordering: motion-free vs motion-included (direction counts /6) ===")
    for label, dd in (("motion-free", d), ("motion-incl", d_all)):
        _, c = depth_ordering(dd)
        print(f"  {label:12s}: N3 lowest {c['n3_lowest']}/6, "
              f"Wake->N3 monotone {c['wake_to_n3_monotone']}/6, "
              f"REM>N3 {c['rem_above_n3']}/6")
    print("  motion fraction by stage:",
          d_all.groupby("stage").motion.mean().round(3).to_dict())

    piv, counts = depth_ordering(d)
    print("\n=== FORWARD: per-subject median log10 variance by stage ===")
    print(piv.round(2).to_string())
    print("cohort median:", piv.median().round(2).to_dict())
    print("direction counts:", counts)
    piv.assign(**counts).to_csv(OUT / "variance_stage_depth.csv")

    # one grouped test, reported for scale only (subject as unit)
    gt = pooled_vs_grouped_stage_test(d["logvar"].values, d["stage"].values, d["subject"].values)
    print(f"\ngrouped Friedman across stages (n={gt['n_groups']} subjects): "
          f"p={gt['p_grouped_friedman']:.4f}  | pooled would give {gt['p_pooled_kruskal']:.1e} "
          f"(non-independent, not used)")

    dw1 = smooth_window(d, 1)
    inv = inverse_mapping(dw1)
    print("\n=== INVERSE: P(stage | per-subject variance tercile), 30 s windows, % ===")
    print((inv * 100).round(1).to_string())
    (inv * 100).round(2).to_csv(OUT / "variance_stage_inverse.csv")

    sep = separability(dw1)
    sep.to_csv(OUT / "variance_stage_separability.csv", index=False)
    print("\n=== descriptive separability (AUC, 30 s windows) ===")
    for target in ("N3", "Wake"):
        a = sep[sep.target == target]["auc"]
        print(f"  {target} vs rest: median AUC {a.median():.2f} "
              f"(per-subject {sorted(round(x,2) for x in a)})")

    sweep = window_sweep(d)
    print("\n=== window-length sweep ===")
    print(sweep.round(3).to_string(index=False))

    figure(piv, inv, sweep)

    # peri-REM lead-up (motion included, per the brain-state-of-motion point)
    t, Mc, subj_curves, prem = peri_rem_onset(d_all)
    print("\n=== peri-REM-onset variance (motion included) ===")
    print(f"  {prem['total_onsets']} onsets across {prem['n_subjects']} subjects "
          f"({prem['onsets_per_subject']})")
    print(f"  lead-up: variance rises in the last 5 min before onset in only "
          f"{prem['rise_in_last5min_over_prior5']}/6 subjects "
          f"(cohort change {prem['cohort_pre_last5']:+.2f}) — no anticipatory rise")
    print(f"  onset: variance steps up in the 5 min after onset in "
          f"{prem['step_up_post_onset']}/6 subjects (cohort {prem['cohort_post5']:+.2f})")
    pd.DataFrame({"t_min": t, "cohort_change": np.nanmean(Mc, axis=0)}).to_csv(
        OUT / "variance_peri_rem.csv", index=False)
    peri_rem_figure(t, Mc, subj_curves)

    print("\nReporting summary:")
    print("  Capacitive variance is a sleep-DEPTH axis: it falls Wake->N3 "
          f"({counts['wake_to_n3_monotone']}/{counts['n_subj']} monotone, "
          f"N3 lowest {counts['n3_lowest']}/{counts['n_subj']}) and returns in REM "
          f"({counts['rem_above_n3']}/{counts['n_subj']}).")
    print("  As a stage indicator it separates deep sleep and wake from the middle "
          "at a modest, subject-dependent level; it does not resolve all five stages "
          "(N2 dominates every variance level by base rate).")


if __name__ == "__main__":
    main()
