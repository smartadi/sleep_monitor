"""Does smoothed capacitive variance track the scored arousal rate?

The observation this tests came from looking at the per-night timeseries: where
the capacitive variance stays elevated for a while, the scored arousal rate looks
elevated too. That is a different claim from the one already tested and rejected.
Counting threshold crossings asks whether discrete mask events line up with
discrete arousals, and throws away magnitude; this asks whether the *level* of
variance, integrated over tens of minutes, tracks the *rate* of arousals over the
same window.

Both signals are put on a common one-minute grid and smoothed with the same
20-minute boxcar, then compared per night.

Two things are reported rather than one, because they are different questions:

  CORRELATION of the two smoothed curves, per night. Smoothing makes neighbouring
  points dependent, so this number is inflated as a significance test and is used
  only to rank nights.

  A CIRCULAR-SHIFT NULL for each night: the same correlation after rotating the
  variance curve by a random offset, 500 times. Rotation preserves the smoothing
  and the shape of both curves and destroys only their alignment, so the fraction
  of shifts beating the true value is an honest per-night p. This is the repo's
  standard idiom for autocorrelated series and is the whole statistical apparatus
  here -- no pooling, no cohort test.

Run from the repo root:
    .venv/Scripts/python.exe analysis/swa_validation/variance_vs_arousal_density.py

Outputs -> reports/psg/variance_vs_arousal_density.csv
           writeup/figures/prof_metrics/variance_vs_arousal_density.png
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

VAR = ROOT / "reports" / "mean_value" / "high_variance_epochs.parquet"
EVENTS = ROOT / "reports" / "psg" / "arousal_events.csv"
OUT = ROOT / "reports" / "psg" / "variance_vs_arousal_density.csv"
FIG = ROOT / "writeup" / "figures" / "prof_metrics"
FIG.mkdir(parents=True, exist_ok=True)

CH = "var_CLE-CRE"
SMOOTH_MIN = 20          # same window for both signals
GRID_MIN = 1.0
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
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def curves(g, ev, tmax):
    """Arousal rate and capacitive variance level on a shared smoothed grid."""
    step = GRID_MIN / 60.0
    grid = np.arange(0, tmax, step)
    w = int(SMOOTH_MIN / GRID_MIN)

    counts = np.histogram(ev, bins=np.append(grid, grid[-1] + step))[0]
    arous = boxcar(counts.astype(float), w) * (60.0 / GRID_MIN)

    # variance interpolated onto the same grid, in log units so that a single
    # motion spike does not set the level for the whole window
    lv = np.log10(np.clip(g[CH].to_numpy(float), 1e-3, None))
    vgrid = np.interp(grid, g.t_hr.to_numpy(float), lv)
    varc = boxcar(vgrid, w)

    edge = w // 2                    # boxcar ends are partial; drop them
    return grid[edge:-edge], arous[edge:-edge], varc[edge:-edge]


def shift_null(a, b, n=N_SHIFT):
    """p from rotating one curve: same shape and smoothing, alignment destroyed."""
    r = np.corrcoef(a, b)[0, 1]
    n_pts = len(a)
    hits = 0
    for _ in range(n):
        k = RNG.integers(1, n_pts)
        if np.corrcoef(a, np.roll(b, k))[0, 1] >= r:
            hits += 1
    return r, (hits + 1) / (n + 1)


def main():
    var = pd.read_parquet(VAR)
    ev = pd.read_csv(EVENTS)
    sessions = sorted(var.session.unique())

    rows, store = [], {}
    for sess in sessions:
        g = var[var.session == sess].sort_values("epoch")
        e = ev[ev.session == sess].t_hr.to_numpy(float)
        tmax = float(g.t_hr.max())
        t, a, v = curves(g, e, tmax)
        if len(t) < 30:
            continue
        r, p = shift_null(a, v)
        rows.append({"session": sess, "n_points": len(t), "r": r, "p_shift": p,
                     "arousal_mean": a.mean(), "var_level_mean": v.mean()})
        store[sess] = (t, a, v, r, p)
        print("  %-6s r = %+.2f   p = %.3f" % (sess, r, p))

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    pos = (d.r > 0).sum()
    sig = (d.p_shift < 0.05).sum()
    print("\n%d of %d nights positive; %d of %d beat the shift null at p < 0.05"
          % (pos, len(d), sig, len(d)))

    fig, axes = plt.subplots(4, 3, figsize=(300 * MM, 165 * MM))
    for ax, sess in zip(axes.ravel(), sessions):
        if sess not in store:
            ax.set_visible(False)
            continue
        t, a, v, r, p = store[sess]
        ax.plot(t, a, color=C_AR, lw=1.4, label="arousals/h")
        ax.set_ylabel("arousals/h", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t, v, color=C_VAR, lw=1.4, label="variance level")
        ax2.set_ylabel("log$_{10}$ var", fontsize=8, color=C_VAR)
        ax2.tick_params(labelsize=7.5, colors=C_VAR)
        ax2.spines["top"].set_visible(False)
        ax.set_title("%s   r = %+.2f   p = %.3f" % (sess, r, p), loc="left",
                     fontsize=9.5, color=C_INK)
        ax.tick_params(labelsize=7.5)
        ax.grid(color=C_FAINT, lw=0.4)
        ax.set_axisbelow(True)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (h)")

    fig.suptitle("Smoothed capacitive variance against scored arousal rate "
                 "(%d min window, circular-shift null)" % SMOOTH_MIN,
                 fontsize=11, x=0.06, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p_out = FIG / "variance_vs_arousal_density.png"
    fig.savefig(p_out)
    plt.close(fig)
    print("wrote %s" % p_out)


if __name__ == "__main__":
    main()
