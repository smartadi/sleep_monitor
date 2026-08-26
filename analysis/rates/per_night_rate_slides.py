"""One night, one band, one figure — the per-slide version of the rate traces.

The deck only had the summary forms: a single representative night carrying both
bands, and the twelve-panel grids. Neither is readable projected. This writes the
same plot the representative-night figure draws, once per recording per band, at
a 16:9-friendly aspect so it fills a slide.

Nothing about the estimate changes: same channel, same loose peak counting, same
per-session k fitted as the median raw/reference ratio over the whole night, same
clipping. Only the framing moves.

Run from the repo root:
    .venv/Scripts/python.exe analysis/rates/per_night_rate_slides.py

Output -> writeup/figures/rate_rerun/per_night/<session>_<band>.png  (24 files)
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "rates"))

SRC = ROOT / "artifacts" / "rate_rerun_phase_a.parquet"
OUT = ROOT / "writeup" / "figures" / "rate_rerun" / "per_night"
OUT.mkdir(parents=True, exist_ok=True)

CHANNEL = "CRE"
K_LO, K_HI = 0.3, 5.0
BAND_NAME = {"resp": "Respiratory", "card": "Cardiac"}
UNIT = {"resp": "br/min", "card": "BPM"}
# No fixed y-limits: a shared scale clipped the reference on the fastest nights,
# which reads as a flat top rather than as data running off the axis. Limits are
# taken per figure from the reference in full -- every scored beat stays on the
# plot -- widened to hold the bulk of the estimate. The estimate's dropout
# spikes are allowed off the bottom; they are absent epochs, not rates.
EST_PCT = (1.0, 99.0)
PAD = 0.06

C_INK = "#1B2A41"
C_MUTED = "#5A6472"
C_FAINT = "#D6DBE1"
C_PK = "#16A085"
C_REF = "#111111"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.edgecolor": C_MUTED,
    "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def fit_k(raw, gt):
    """Per-session scale: median of the raw/reference ratio, ratios clipped."""
    r = raw / gt
    r = r[(r > K_LO) & (r < K_HI) & np.isfinite(r)]
    return float(np.median(r)) if len(r) >= 10 else np.nan


def one(df, session, band):
    g = (df[(df.band == band) & (df.channel == CHANNEL) & (df.session == session)]
         .dropna(subset=["gt_hz"]).sort_values("epoch"))
    if g.empty:
        print("  %s %s -- no data" % (session, band))
        return None

    t = g.t_hr.values
    gt = g.gt_hz.values * 60.0
    raw = g.peaks_loose.values * 60.0
    m = np.isfinite(raw) & np.isfinite(gt) & (gt > 0)
    k = fit_k(raw[m], gt[m])
    if not np.isfinite(k):
        print("  %s %s -- k not estimable" % (session, band))
        return None
    est = raw[m] / k

    lo = min(np.nanmin(gt), np.nanpercentile(est, EST_PCT[0]))
    hi = max(np.nanmax(gt), np.nanpercentile(est, EST_PCT[1]))
    pad = PAD * (hi - lo)

    fig, ax = plt.subplots(figsize=(250 * MM, 116 * MM))
    ax.plot(t[m], est, "-", color=C_PK, lw=0.7, alpha=0.9, zorder=2,
            label="capacitive mask (peak counting, per-session k)")
    ax.plot(t, gt, "-", color=C_REF, lw=1.2, zorder=3, label="PSG reference")
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylabel("%s rate  (%s)" % (BAND_NAME[band].lower(), UNIT[band]))
    ax.set_xlabel("time (h)")
    ax.grid(axis="y", color=C_FAINT, lw=0.5)
    ax.set_axisbelow(True)

    err = float(np.median(np.abs(est - gt[m])))
    ax.text(1.0, 1.012, "k = %.2f    median |error| %.2f %s" % (k, err, UNIT[band]),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=C_MUTED)
    ax.legend(loc="lower left", ncol=2, handlelength=1.8, borderpad=0.2)
    ax.set_title("%s — %s rate across the night" % (session, BAND_NAME[band].lower()),
                 loc="left", color=C_INK, pad=8)

    out = OUT / ("%s_%s.png" % (session, band))
    fig.savefig(out)
    plt.close(fig)
    return out, k, err


def main():
    df = pd.read_parquet(SRC)
    sessions = sorted(df.session.unique())
    print("sessions: %d" % len(sessions))
    n = 0
    for sess in sessions:
        for band in ("resp", "card"):
            r = one(df, sess, band)
            if r:
                out, k, err = r
                print("  %-6s %-5s k=%.2f  err=%.2f %s" % (sess, band, k, err, UNIT[band]))
                n += 1
    print("\nwrote %d figures -> %s" % (n, OUT))


if __name__ == "__main__":
    main()
