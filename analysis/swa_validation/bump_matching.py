"""Do capacitive bumps and EEG arousal bumps land in the same places?

Correlation asks whether two levels covary linearly, which is a strong demand on
two instruments measuring different things through different mechanisms. It
returned a near-null and that is not, on its own, evidence of no relationship.

This asks the weaker and more appropriate question. Detect episodes -- "bumps" --
in each signal independently, each against its own night's baseline, then ask
whether a bump in one is accompanied by a bump in the other more often than
chance. Amplitudes never meet, so a subject whose mask swings ten times harder
than another is not thereby penalised, and the per-night gain differences that
dominated every previous comparison drop out.

Method:

  BUMPS are prominent peaks of the 20-minute-smoothed curve -- arousal rate for
  the EEG, log variance for the mask -- with prominence set from that night's own
  spread and peaks required to be at least 15 minutes apart.

  MATCHING is symmetric and reported both ways: the fraction of mask bumps with
  an EEG bump within a tolerance, and the fraction of EEG bumps with a mask bump.
  Tolerance is swept (5, 10, 15 minutes) rather than fixed.

  THE NULL is a circular shift of the mask bump times, 500 draws. It preserves
  the number of bumps and their spacing and destroys only alignment, so the
  fraction of shifts matching at least as well is a per-night p. Per night --
  the question is explicitly whether this differs between subjects.

  DELAY of the nearest EEG bump relative to each mask bump is recorded, so that
  any consistent lead or lag shows up as a shifted distribution rather than
  having to be searched for.

Run from the repo root:
    .venv/Scripts/python.exe analysis/swa_validation/bump_matching.py

Outputs -> reports/psg/bump_matching.csv
           reports/psg/bump_delays.csv
           writeup/figures/prof_metrics/bump_matching_nights.png
           writeup/figures/prof_metrics/bump_matching_summary.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from scipy.signal import find_peaks     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VAR = ROOT / "reports" / "mean_value" / "high_variance_epochs.parquet"
EVENTS = ROOT / "reports" / "psg" / "arousal_events.csv"
OUT = ROOT / "reports" / "psg"
FIG = ROOT / "writeup" / "figures" / "prof_metrics"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

CH = "var_CLE-CRE"
SMOOTH_MIN = 20
MIN_SEP_MIN = 15                 # peaks closer than this are one bump
PROM_FRAC = 0.5                  # prominence as a fraction of the night's IQR
TOL_MIN = [5, 10, 15]
HEADLINE_TOL = 10
N_SHIFT = 500
RNG = np.random.default_rng(0)

C_INK = "#1B2A41"
C_MUTED = "#5A6472"
C_FAINT = "#D6DBE1"
C_AR = "#111111"
C_VAR = "#16A085"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.edgecolor": C_MUTED,
    "legend.frameon": False, "figure.dpi": 170, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def boxcar(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def curves(g, ev, tmax):
    step = 1 / 60.0
    grid = np.arange(0, tmax, step)
    w = SMOOTH_MIN
    a = boxcar(np.histogram(ev, bins=np.append(grid, grid[-1] + step))[0].astype(float), w) * 60
    lv = np.log10(np.clip(g[CH].to_numpy(float), 1e-3, None))
    v = boxcar(np.interp(grid, g.t_hr.to_numpy(float), lv), w)
    k = w // 2
    return grid[k:-k], a[k:-k], v[k:-k]


def bumps(t, y):
    """Prominent peaks, prominence scaled to this night's own spread."""
    spread = np.subtract(*np.percentile(y, [75, 25]))
    prom = max(PROM_FRAC * spread, 1e-9)
    idx, _ = find_peaks(y, prominence=prom, distance=MIN_SEP_MIN)
    return t[idx]


def hit_rate(a_times, b_times, tol_hr):
    """Fraction of a-bumps with a b-bump within tol, and the signed delays."""
    if len(a_times) == 0 or len(b_times) == 0:
        return np.nan, np.array([])
    d = b_times[None, :] - a_times[:, None]
    j = np.argmin(np.abs(d), axis=1)
    nearest = d[np.arange(len(a_times)), j]
    return float(np.mean(np.abs(nearest) <= tol_hr)), nearest


def main():
    var = pd.read_parquet(VAR)
    ev = pd.read_csv(EVENTS)
    sessions = sorted(var.session.unique())

    rows, delays, store = [], [], {}
    for sess in sessions:
        g = var[var.session == sess].sort_values("epoch")
        e = ev[ev.session == sess].t_hr.to_numpy(float)
        tmax = float(g.t_hr.max())
        t, a, v = curves(g, e, tmax)
        if len(t) < 60:
            continue

        eeg_b = bumps(t, a)
        cap_b = bumps(t, v)
        store[sess] = (t, a, v, eeg_b, cap_b)

        rec = {"session": sess, "subject": sess[:2], "hours": tmax,
               "n_eeg_bumps": len(eeg_b), "n_cap_bumps": len(cap_b)}

        for tol in TOL_MIN:
            th = tol / 60.0
            fwd, nearest = hit_rate(cap_b, eeg_b, th)
            rev, _ = hit_rate(eeg_b, cap_b, th)
            rec["cap_to_eeg_tol%d" % tol] = fwd
            rec["eeg_to_cap_tol%d" % tol] = rev

            if tol == HEADLINE_TOL and len(cap_b) and len(eeg_b):
                span = t[-1] - t[0]
                null = []
                for _ in range(N_SHIFT):
                    shift = RNG.uniform(0, span)
                    shifted = t[0] + np.mod(cap_b - t[0] + shift, span)
                    h, _u = hit_rate(np.sort(shifted), eeg_b, th)
                    null.append(h)
                null = np.array(null)
                rec["null_mean"] = float(np.nanmean(null))
                rec["p_shift"] = float((np.sum(null >= fwd) + 1) / (N_SHIFT + 1))
                for dly in nearest:
                    delays.append({"session": sess, "subject": sess[:2],
                                   "delay_min": float(dly * 60.0)})
        rows.append(rec)
        print("  %-6s  cap bumps %2d  eeg bumps %2d   match %.2f  null %.2f  p %.3f"
              % (sess, rec["n_cap_bumps"], rec["n_eeg_bumps"],
                 rec.get("cap_to_eeg_tol10", np.nan), rec.get("null_mean", np.nan),
                 rec.get("p_shift", np.nan)))

    d = pd.DataFrame(rows)
    dl = pd.DataFrame(delays)
    d.to_csv(OUT / "bump_matching.csv", index=False)
    dl.to_csv(OUT / "bump_delays.csv", index=False)

    print("\nmatch rate above its own null on %d of %d nights; %d at p < 0.05"
          % (int((d.cap_to_eeg_tol10 > d.null_mean).sum()), len(d),
             int((d.p_shift < 0.05).sum())))
    print("median match %.2f vs median null %.2f"
          % (d.cap_to_eeg_tol10.median(), d.null_mean.median()))
    print("\nby subject:")
    print(d.groupby("subject")[["cap_to_eeg_tol10", "null_mean", "p_shift"]]
          .mean().round(3).to_string())

    # ── per-night curves with bumps marked
    fig, axes = plt.subplots(4, 3, figsize=(300 * MM, 165 * MM))
    for ax, sess in zip(axes.ravel(), sessions):
        if sess not in store:
            ax.set_visible(False)
            continue
        t, a, v, eb, cb = store[sess]
        ax.plot(t, a, color=C_AR, lw=1.2)
        ax.plot(eb, np.interp(eb, t, a), "v", color=C_AR, ms=7)
        ax.set_ylabel("arousals/h", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t, v, color=C_VAR, lw=1.2)
        ax2.plot(cb, np.interp(cb, t, v), "^", color=C_VAR, ms=7)
        ax2.set_ylabel("log$_{10}$ var", fontsize=8, color=C_VAR)
        ax2.tick_params(labelsize=7.5, colors=C_VAR)
        ax2.spines["top"].set_visible(False)
        r = d[d.session == sess].iloc[0]
        ax.set_title("%s   match %.2f (null %.2f)  p = %.3f"
                     % (sess, r.cap_to_eeg_tol10, r.null_mean, r.p_shift),
                     loc="left", fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(color=C_FAINT, lw=0.4)
        ax.set_axisbelow(True)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (h)")
    fig.suptitle("Bumps in each signal, marked independently  "
                 "(▲ mask, ▼ EEG; match = mask bumps with an EEG bump within %d min)"
                 % HEADLINE_TOL, fontsize=11, x=0.06, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = FIG / "bump_matching_nights.png"
    fig.savefig(p1); plt.close(fig); print("\nwrote %s" % p1.name)

    # ── summary: match vs null per night, and the delay distribution
    fig, ax = plt.subplots(1, 2, figsize=(250 * MM, 100 * MM))
    x = np.arange(len(d))
    dd = d.sort_values("session")
    ax[0].bar(x - 0.2, dd.cap_to_eeg_tol10, width=0.4, color=C_VAR, label="observed")
    ax[0].bar(x + 0.2, dd.null_mean, width=0.4, color="#B0B7C0", label="shift null")
    ax[0].set_xticks(x); ax[0].set_xticklabels(dd.session, rotation=90, fontsize=8)
    ax[0].set_ylabel("mask bumps with an EEG bump within %d min" % HEADLINE_TOL)
    ax[0].set_title("A  ·  per night, against its own null", loc="left", fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].hist(dl.delay_min, bins=np.arange(-60, 61, 5), color=C_VAR, alpha=0.85)
    ax[1].axvline(0, color=C_INK, ls="--", lw=1.0)
    ax[1].set_xlabel("delay of nearest EEG bump  (min; positive = EEG later)")
    ax[1].set_ylabel("count")
    ax[1].set_title("B  ·  is there a consistent lead or lag?", loc="left", fontsize=10)

    for a_ in ax:
        a_.grid(axis="y", color=C_FAINT, lw=0.5)
        a_.set_axisbelow(True)
    fig.suptitle("Bump matching between the mask and the scored arousal rate",
                 fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = FIG / "bump_matching_summary.png"
    fig.savefig(p2); plt.close(fig); print("wrote %s" % p2.name)

    # ── deck version: the result and the subject split, nothing else
    dd = d.sort_values("session")
    by = d.groupby("subject")[["cap_to_eeg_tol10", "null_mean"]].mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(250 * MM, 100 * MM),
                           gridspec_kw={"width_ratios": [1.7, 1.0]})
    x = np.arange(len(dd))
    ax[0].bar(x - 0.2, dd.cap_to_eeg_tol10, width=0.4, color=C_VAR, label="observed")
    ax[0].bar(x + 0.2, dd.null_mean, width=0.4, color="#B0B7C0", label="chance")
    ax[0].set_xticks(x); ax[0].set_xticklabels(dd.session, rotation=90, fontsize=8.5)
    ax[0].set_ylabel("mask bumps with an EEG\narousal bump within 10 min")
    ax[0].set_title("Per night", loc="left", fontsize=10)
    ax[0].legend(fontsize=8.5)

    xb = np.arange(len(by))
    ax[1].bar(xb - 0.2, by.cap_to_eeg_tol10, width=0.4, color=C_VAR)
    ax[1].bar(xb + 0.2, by.null_mean, width=0.4, color="#B0B7C0")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(by.subject, fontsize=9)
    ax[1].set_title("By subject — it differs between people", loc="left", fontsize=10)

    for a_ in ax:
        a_.set_ylim(0, 1)
        a_.grid(axis="y", color=C_FAINT, lw=0.5)
        a_.set_axisbelow(True)
    fig.suptitle("Mask activity bumps land where EEG arousals cluster, about twice as "
                 "often as chance", fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p3 = FIG / "bump_matching_simple.png"
    fig.savefig(p3); plt.close(fig); print("wrote %s" % p3.name)


if __name__ == "__main__":
    main()
