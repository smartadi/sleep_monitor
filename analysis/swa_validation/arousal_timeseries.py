"""Full-night timeseries per session: capacitive signal against EEG arousal tags.

Built for eyeballing rather than for a number -- the point is to look at twelve
nights with the scored arousals laid over the capacitive traces and decide what
is worth measuring, before measuring anything.

Each night gets five stacked panels on a shared clock:

  1. Hypnogram, Wake at the top rung through to REM at the bottom.
  2. Event lanes. Every scored arousal as a tick coloured by subtype, every
     mask-detected transient as a tick below it, and a smoothed density for each
     so co-variation is visible without counting ticks.
  3. CLE-CRE epoch variance on a log axis -- the quantity the mask's own event
     detector keys on -- with the detector's threshold drawn.
  4. CLE-CRE slow mean, referenced to the session mean.
  5. Accelerometer SD, so anything that is just movement is obvious.

Arousal event times are cached on first run, since reading them costs a load of
all twelve recordings. Everything else comes from tables already on disk.

Run from the repo root:
    .venv/Scripts/python.exe analysis/swa_validation/arousal_timeseries.py
    .venv/Scripts/python.exe analysis/swa_validation/arousal_timeseries.py S1N1

Outputs -> reports/psg/arousal_events.csv
           writeup/figures/prof_metrics/arousal_timeseries/<session>.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VAR = ROOT / "reports" / "mean_value" / "high_variance_epochs.parquet"
IMB = ROOT / "reports" / "mean_value" / "imbalance_epochs.csv"
CAP_EVENTS = ROOT / "reports" / "swa_validation" / "cap_events" / "cap_arousal_events.csv"
CACHE = ROOT / "reports" / "psg" / "arousal_events.csv"
FIG = ROOT / "writeup" / "figures" / "prof_metrics" / "arousal_timeseries"
FIG.mkdir(parents=True, exist_ok=True)
CACHE.parent.mkdir(parents=True, exist_ok=True)

LADDER = ["Wake", "N1", "N2", "N3", "REM"]      # top rung to bottom
YPOS = {s: -i for i, s in enumerate(LADDER)}
STAGE_COLOR = {"Wake": "#E74C3C", "N1": "#F39C12", "N2": "#3498DB",
               "N3": "#27AE60", "REM": "#8E44AD"}
GROUP_COLOR = {"spontaneous": "#2980B9", "respiratory": "#C0392B",
               "limb": "#E67E22", "cardiac": "#8E44AD", "other": "#7F8C8D"}
LABEL_GROUP = {"Arousal": "spontaneous", "Respiratory Arousal": "respiratory",
               "SpO2 Arousal": "respiratory", "LM Arousal": "limb",
               "PLM Arousal": "limb", "Cardiac Arousal": "cardiac"}

C_INK = "#1B2A41"
C_MUTED = "#5A6472"
C_FAINT = "#D6DBE1"
C_VAR = "#2980B9"
C_DC = "#E67E22"
C_RED = "#C0392B"
THRESH = 10.0                # the mask detector's variance threshold, fF^2
SMOOTH_MIN = 20.0            # density smoothing window

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 7.5, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.edgecolor": C_MUTED,
    "legend.frameon": False, "figure.dpi": 170, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def build_events():
    """Read every scored arousal once and cache start time, duration and type."""
    from sleep_monitor.loader import load_session, load_arousals, load_sleep_profile
    from sleep_monitor.sessions import SESSION_META
    rows = []
    for i, _ in enumerate(SESSION_META):
        s = load_session(i)
        ar = load_arousals(s)
        prof = load_sleep_profile(s)
        if ar is None:
            continue
        lab = s.meta["label"]
        st = np.full(len(ar["start_hr"]), -1)
        if prof is not None:
            pt = np.asarray(prof["t_ep_hr"], float)
            pc = np.asarray(prof["codes"], int)
            j = np.searchsorted(pt, ar["start_hr"]) - 1
            ok = (j >= 0) & (j < len(pc))
            st[ok] = pc[j[ok]]
        for k in range(len(ar["start_hr"])):
            lab_t = ar["types"][k]
            rows.append({"session": lab, "t_hr": float(ar["start_hr"][k]),
                         "duration_s": float(ar["duration_s"][k]),
                         "type": lab_t,
                         "group": LABEL_GROUP.get(lab_t, "other"),
                         "stage_code": int(st[k])})
        print("  %s  %d arousals" % (lab, len(ar["start_hr"])))
    return pd.DataFrame(rows)


def density(times, tmax, win_min=SMOOTH_MIN):
    """Events per hour on a fine grid, boxcar-smoothed."""
    step = 1.0 / 60.0
    grid = np.arange(0, tmax + step, step)
    counts = np.histogram(times, bins=np.append(grid, grid[-1] + step))[0]
    w = max(1, int(win_min))
    kern = np.ones(w) / w
    return grid, np.convolve(counts, kern, mode="same") * 60.0


def one(sess, ev, var, imb, cap):
    g = var[var.session == sess].sort_values("epoch")
    gi = imb[imb.session == sess].sort_values("t_hr")
    ge = ev[ev.session == sess]
    gc = cap[(cap.session == sess)] if cap is not None else pd.DataFrame()
    if g.empty:
        return
    tmax = float(g.t_hr.max())

    fig = plt.figure(figsize=(340 * MM, 165 * MM))
    gs = gridspec.GridSpec(5, 1, figure=fig,
                           height_ratios=[0.55, 0.75, 1.25, 0.9, 0.5], hspace=0.14)

    # 1 ── hypnogram
    ax0 = fig.add_subplot(gs[0])
    y = g.stage.map(YPOS).to_numpy(float)
    ax0.step(g.t_hr, y, where="mid", lw=0.9, color="#2C3E50")
    for st in LADDER:
        m = (g.stage == st).to_numpy()
        ax0.plot(g.t_hr[m], np.full(m.sum(), YPOS[st]), "|",
                 color=STAGE_COLOR[st], ms=4, mew=1.8, alpha=0.9)
    ax0.set_yticks([YPOS[s] for s in LADDER]); ax0.set_yticklabels(LADDER)
    ax0.set_ylim(-4.6, 0.6)

    # 2 ── event lanes + density
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    for grp, gg in ge.groupby("group"):
        ax1.plot(gg.t_hr, np.full(len(gg), 1.0), "|", ms=9, mew=1.1,
                 color=GROUP_COLOR.get(grp, "#7F8C8D"), label="EEG %s" % grp)
    if len(gc):
        ax1.plot(gc.t_hr, np.full(len(gc), 0.55), "|", ms=7, mew=1.0,
                 color="#95A5A6", label="mask, all detections")
        kept = gc[gc.kept]
        ax1.plot(kept.t_hr, np.full(len(kept), 0.55), "|", ms=9, mew=1.4,
                 color="#E67E22", label="mask, kept")
    ax1.set_ylim(0.25, 1.35)
    ax1.set_yticks([1.0, 0.55]); ax1.set_yticklabels(["EEG", "mask"])
    axd = ax1.twinx()
    gt, de = density(ge.t_hr.to_numpy(), tmax)
    axd.plot(gt, de, color="#2980B9", lw=1.3, alpha=0.9)
    if len(gc):
        gt2, dc = density(gc.t_hr.to_numpy(), tmax)
        axd.plot(gt2, dc, color="#E67E22", lw=1.3, alpha=0.9)
    axd.set_ylabel("events/h\n(%g min smooth)" % SMOOTH_MIN, fontsize=7.5)
    axd.tick_params(labelsize=7.5)
    axd.spines["top"].set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()

    # 3 ── capacitive variance
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.plot(g.t_hr, g["var_CLE-CRE"], lw=0.6, color=C_VAR)
    ax2.axhline(THRESH, color=C_RED, ls="--", lw=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("variance CLE−CRE\n(fF$^2$)")

    # 4 ── slow mean
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3.plot(gi.t_hr, gi.d_fF, lw=0.8, color=C_DC)
    ax3.axhline(0, color=C_INK, ls="--", lw=0.7)
    ax3.set_ylabel("CLE−CRE − mean\n(fF)")

    # 5 ── motion
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    ax4.fill_between(g.t_hr, 0, g.acc_sd, color="#7F8C8D", lw=0, alpha=0.85)
    ax4.set_ylabel("accel SD")
    ax4.set_xlabel("time (h)")

    for a in (ax0, ax1, ax2, ax3, ax4):
        a.set_xlim(0, tmax)
        a.grid(axis="x", color=C_FAINT, lw=0.4)
        a.set_axisbelow(True)
    for a in (ax0, ax1, ax2, ax3):
        a.tick_params(labelbottom=False)

    n_eeg, n_cap = len(ge), int(gc.kept.sum()) if len(gc) else 0
    # legend above everything -- inside the axes it covered the hypnogram
    fig.legend(handles, labels, ncol=6, fontsize=8, frameon=False,
               loc="upper left", bbox_to_anchor=(0.075, 0.955))
    fig.suptitle("%s — capacitive signal against scored arousals   "
                 "(%d EEG arousals, %d mask events kept)"
                 % (sess, n_eeg, n_cap), fontsize=11, x=0.075, ha="left", y=0.985)
    p = FIG / ("%s.png" % sess)
    fig.savefig(p)
    plt.close(fig)
    print("  %s" % p.name)


def main():
    if CACHE.exists():
        ev = pd.read_csv(CACHE)
        print("cached arousal events: %d rows" % len(ev))
    else:
        print("reading arousal scoring from the recordings")
        ev = build_events()
        ev.to_csv(CACHE, index=False)
        print("cached -> %s" % CACHE)

    var = pd.read_parquet(VAR)
    imb = pd.read_csv(IMB)
    cap = pd.read_csv(CAP_EVENTS) if CAP_EVENTS.exists() else None

    want = sys.argv[1:] or sorted(var.session.unique())
    print("\nfigures")
    for sess in want:
        one(sess, ev, var, imb, cap)
    print("\n-> %s" % FIG)


if __name__ == "__main__":
    main()
