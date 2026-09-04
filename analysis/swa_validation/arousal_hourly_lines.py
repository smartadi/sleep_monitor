"""Hourly arousal rate as dots and lines — every night, both instruments.

The heatmap held all 288 numbers but made the shape of a single night hard to
read. This is the same data drawn conventionally: hour along x, rate up y, one
dot-and-line per night. Individual nights are thin and translucent so the twelve
can overlap without turning into mud; the cohort mean for each instrument is
drawn thick on top. The nights that sit furthest from their instrument's mean are
labelled directly, since a twenty-four entry legend would be unreadable.

The per-hour table is cached on first build (loading twelve recordings takes a
few minutes) so that re-plotting is instant. Delete the CSV to force a rebuild.

Run from the repo root:
    .venv/Scripts/python.exe analysis/swa_validation/arousal_hourly_lines.py

Outputs -> reports/psg/arousal_hourly.csv
           writeup/figures/prof_metrics/arousal_hourly_lines.png
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

CAP_EVENTS = ROOT / "reports" / "swa_validation" / "cap_events" / "cap_arousal_events.csv"
CACHE = ROOT / "reports" / "psg" / "arousal_hourly.csv"
FIG = ROOT / "writeup" / "figures" / "prof_metrics"
FIG.mkdir(parents=True, exist_ok=True)
CACHE.parent.mkdir(parents=True, exist_ok=True)

C_INK = "#1B2A41"
C_MUTED = "#5A6472"
C_FAINT = "#D6DBE1"
C_EEG = "#2980B9"
C_CAP = "#E67E22"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.edgecolor": C_MUTED,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def build_table():
    """Arousals per hour of recording, per night, from both instruments."""
    from sleep_monitor.loader import load_session, load_arousals
    from sleep_monitor.sessions import SESSION_META

    cap = pd.read_csv(CAP_EVENTS) if CAP_EVENTS.exists() else None
    rows = []
    for i, _ in enumerate(SESSION_META):
        s = load_session(i)
        lab = s.meta["label"]
        ar = load_arousals(s)
        if ar is None:
            continue
        dur = float(s.time_hr[-1])
        edges = np.arange(0, np.ceil(dur) + 1)
        # divide by the recording time actually in each bin, so a partial final
        # hour reads as a rate instead of an artificially short bar
        cover = np.clip(np.minimum(edges[1:], dur) - edges[:-1], 1e-6, None)
        eeg = np.histogram(ar["start_hr"], bins=edges)[0] / cover
        cp = (np.histogram(cap[cap.session == lab].t_hr, bins=edges)[0] / cover
              if cap is not None else np.full(len(eeg), np.nan))
        for h, (e, c, cv) in enumerate(zip(eeg, cp, cover)):
            rows.append({"session": lab, "subject": lab[:2], "hour": h,
                         "coverage_hr": cv, "eeg_per_hour": e, "cap_per_hour": c})
        print("  %s  %d hours" % (lab, len(eeg)))
    return pd.DataFrame(rows)


def figure(d):
    fig, ax = plt.subplots(figsize=(250 * MM, 120 * MM))

    for col, colr, name in ((("eeg_per_hour"), C_EEG, "EEG scored"),
                            (("cap_per_hour"), C_CAP, "CAP detected")):
        for sess, g in d.groupby("session"):
            g = g.sort_values("hour")
            ax.plot(g.hour, g[col], "o-", color=colr, lw=0.9, ms=3.2,
                    alpha=0.32, zorder=2)
        m = d.groupby("hour")[col].mean()
        ax.plot(m.index, m.values, "o-", color=colr, lw=2.8, ms=7,
                zorder=4, label="%s — cohort mean" % name,
                markeredgecolor="white", markeredgewidth=1.0)

    # label the nights furthest from their instrument's mean, rather than a
    # twenty-four entry legend nobody can read
    mean_eeg = d.groupby("session").eeg_per_hour.mean()
    mean_cap = d.groupby("session").cap_per_hour.mean()
    for series, col, colr in ((mean_eeg, "eeg_per_hour", C_EEG),
                              (mean_cap, "cap_per_hour", C_CAP)):
        for sess in (series.idxmax(), series.idxmin()):
            g = d[d.session == sess].sort_values("hour")
            last = g.dropna(subset=[col]).iloc[-1]
            ax.annotate(sess, (last.hour, last[col]), fontsize=8.5, color=colr,
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontweight="bold")

    ax.set_xlabel("hour of recording")
    ax.set_ylabel("arousals per hour")
    ax.set_xticks(range(int(d.hour.max()) + 1))
    ax.set_xlim(-0.3, d.hour.max() + 0.9)
    ax.set_ylim(bottom=0)
    ax.grid(color=C_FAINT, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.set_title("Arousals per hour through the night — twelve nights, both instruments",
                 loc="left", pad=10)
    p = FIG / "arousal_hourly_lines.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote %s" % p)


def figure_by_subject(d):
    """Twelve panels, one per night, laid out a subject to a row.

    Putting a subject's two nights side by side is the point of the layout: it
    shows at a glance whether a night's arousal profile is a property of the
    person or of that particular night. Axes are shared, so panels compare
    directly without reading the ticks.
    """
    subs = sorted(d.subject.unique())
    fig, axes = plt.subplots(len(subs), 2, figsize=(250 * MM, 175 * MM),
                             sharex=True, sharey=True)
    hmax = int(d.hour.max())

    for r, sub in enumerate(subs):
        for c, night in enumerate(("1", "2")):
            ax = axes[r, c]
            sess = "%sN%s" % (sub, night)
            g = d[d.session == sess].sort_values("hour")
            if g.empty:
                ax.set_visible(False)
                continue
            ax.plot(g.hour, g.eeg_per_hour, "o-", color=C_EEG, lw=1.8, ms=5,
                    label="EEG scored", markeredgecolor="white", markeredgewidth=0.8)
            ax.plot(g.hour, g.cap_per_hour, "s-", color=C_CAP, lw=1.8, ms=4.5,
                    label="CAP detected", markeredgecolor="white", markeredgewidth=0.8)
            ax.set_title(sess, loc="left", fontsize=10, color=C_INK, pad=4)
            ax.grid(color=C_FAINT, lw=0.5)
            ax.set_axisbelow(True)
            ax.set_xticks(range(hmax + 1))
            ax.set_xlim(-0.3, hmax + 0.3)
            ax.set_ylim(0, d[["eeg_per_hour", "cap_per_hour"]].max().max() * 1.06)

    for r in range(len(subs)):
        axes[r, 0].set_ylabel("arousals\nper hour", fontsize=9)
    for c in range(2):
        axes[-1, c].set_xlabel("hour of recording")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=10, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.972))
    fig.suptitle("Arousals per hour — each subject's two nights side by side",
                 fontsize=12, x=0.055, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = FIG / "arousal_hourly_by_subject.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote %s" % p)


def main():
    if CACHE.exists():
        d = pd.read_csv(CACHE)
        print("loaded cached table (%d rows) — delete %s to rebuild"
              % (len(d), CACHE.name))
    else:
        print("building hourly table from the recordings")
        d = build_table()
        d.to_csv(CACHE, index=False)
        print("cached -> %s" % CACHE)
    figure(d)
    figure_by_subject(d)


if __name__ == "__main__":
    main()
