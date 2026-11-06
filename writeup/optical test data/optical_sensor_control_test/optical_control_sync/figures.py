"""Figures for the mask <-> Muse synchronisation."""

import os

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CH_COL = {"CH": "#1b6ca8", "CLE": "#e07b39", "CRE": "#3f8f5b"}
MUSE_COL = "#7b4ea3"
GREY = "#5a5a5a"


def _apply_style(sizes=(9, 8, 7)):
    base, small, tick = sizes
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": base,
        "axes.titlesize": base, "axes.labelsize": base,
        "axes.titleweight": "regular", "axes.titlelocation": "left",
        "legend.fontsize": small, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.0,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def _clock_axis(ax, fmt="%H:%M:%S"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def fig1_tap_anchor(sm, taps, anchor, outdir, fname="fig1_tap_anchor.png"):
    """The six tap pulses that put the mask on wall time, and the agreement
    between the two bursts."""
    _apply_style()
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, :])

    t0 = taps["time"].iloc[0] - pd.Timedelta(seconds=8)
    t1 = taps["time"].iloc[-1] + pd.Timedelta(seconds=8)
    w = sm[(sm["time"] >= t0) & (sm["time"] <= t1)]
    ax0.plot(w["time"], w["CH"], color=CH_COL["CH"], lw=0.7)
    ax0.plot(taps["time"], taps["CH"], "v", ms=4, color="#b3282d", zorder=5)
    for clk in anchor["per_burst"]["reported"]:
        ax0.axvline(pd.Timestamp(clk), color=GREY, ls=":", lw=0.9)
    ax0.set_ylabel("Forehead capacitance CH")
    ax0.set_title("Six finger presses on the mask sensors put its uptime counter on wall time")
    ax0.text(0.42, 0.42, "dotted = time reported by tester\ntriangles = detected tap\n"
                         f"offsets from the two bursts agree to "
                         f"{anchor['offset_spread_s']*1000:.0f} ms",
             transform=ax0.transAxes, ha="left", va="center", fontsize=7, color=GREY)
    _clock_axis(ax0)
    ax0.margins(y=0.12)

    reported = list(anchor["per_burst"]["reported"])
    for k, b in enumerate(sorted(taps["burst"].unique())):
        ax = fig.add_subplot(gs[1, k])
        sub = taps[taps["burst"] == b]
        a = sub["time"].iloc[0] - pd.Timedelta(seconds=1.2)
        z = sub["time"].iloc[-1] + pd.Timedelta(seconds=1.2)
        ww = sm[(sm["time"] >= a) & (sm["time"] <= z)]
        ax.plot(ww["time"], ww["CH"], color=CH_COL["CH"], lw=0.9, label="CH forehead")
        ax2 = ax.twinx()
        ax2.plot(ww["time"], ww["CRE"], color=CH_COL["CRE"], lw=0.9, label="CRE right eye")
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(labelsize=mpl.rcParams["ytick.labelsize"])
        for _, r in sub.iterrows():
            ax.plot(r["time"], r["CH"], "v", ms=4, color="#b3282d", zorder=5)
        iv = np.diff(sub["t_dev"].values)
        ax.set_title(f"burst {k+1}: reported {pd.Timestamp(reported[k]).strftime('%H:%M:%S')}"
                     f"  ·  gaps {iv[0]:.2f}, {iv[1]:.2f} s")
        # label each channel once, on the outer edge of the pair, and colour the
        # tick marks so the unlabelled side is still identifiable
        ax.set_ylabel("CH forehead" if k == 0 else "", color=CH_COL["CH"])
        ax2.set_ylabel("CRE right eye" if k == 1 else "", color=CH_COL["CRE"])
        ax.tick_params(axis="y", colors=CH_COL["CH"])
        ax2.tick_params(axis="y", colors=CH_COL["CRE"])
        _clock_axis(ax, "%H:%M:%S")
        ax.tick_params(axis="x", rotation=0)
    p = os.path.join(outdir, fname)
    fig.savefig(p)
    return fig, p


def fig2_synced_overview(both, cc, taps, outdir, fname="fig2_synced_overview.png"):
    """Both devices on one wall-clock axis, plus the movement cross-correlation
    that verifies the alignment."""
    _apply_style()
    fig = plt.figure(figsize=(7.6, 8.6))
    gs = fig.add_gridspec(7, 1, height_ratios=[1, 1, 1, 1, 1, 0.30, 1.15], hspace=0.42)
    t = both["time"]

    ax = fig.add_subplot(gs[0])
    for c in ("CH", "CLE", "CRE"):
        ax.plot(t, both[f"sm_{c}_sd"], lw=0.6, color=CH_COL[c], label=c)
    ax.set_yscale("log"); ax.set_ylabel("mask amplitude\n(0.1 s SD)")
    ax.set_title("Sleep mask and Muse S on one wall clock, anchored by the mask tap markers "
                 f"({pd.Timestamp(cc['overlap_start']).strftime('%Y-%m-%d')})")
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    ax_share = ax

    ax = fig.add_subplot(gs[1], sharex=ax_share)
    if "sm_headUDTilt" in both:
        ax.plot(t, both["sm_headUDTilt"], lw=0.6, color="#1b6ca8", label="up-down (pitch)")
        ax.plot(t, both["sm_headLRTilt"], lw=0.6, color="#e07b39", label="left-right (roll)")
        ax.axhline(0, color=GREY, lw=0.5, ls=":")
        ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.set_ylabel("mask head tilt\n(deg)")

    ax = fig.add_subplot(gs[2], sharex=ax_share)
    m = np.sqrt(both["sm_aX"]**2 + both["sm_aY"]**2 + both["sm_aZ"]**2)
    ax.plot(t, m.rolling(10, center=True).std(), lw=0.6, color=GREY)
    ax.set_yscale("log"); ax.set_ylabel("mask motion\n(|acc| SD, g)")

    ax = fig.add_subplot(gs[3], sharex=ax_share)
    ax.plot(t, both["muse_RAW_AF7_sd"], lw=0.6, color=MUSE_COL)
    ax.set_yscale("log"); ax.set_ylabel("Muse EEG AF7\namplitude (µV)")

    ax = fig.add_subplot(gs[4], sharex=ax_share)
    mm = np.sqrt(both["muse_Accelerometer_X"]**2 + both["muse_Accelerometer_Y"]**2
                 + both["muse_Accelerometer_Z"]**2)
    ax.plot(t, mm.rolling(10, center=True).std(), lw=0.6, color=MUSE_COL)
    ax.set_yscale("log"); ax.set_ylabel("Muse motion\n(|acc| SD, g)")
    ax.set_xlabel("clock time")
    _clock_axis(ax, "%H:%M")
    for a in fig.axes:
        for tp in taps["time"]:
            a.axvline(tp, color="#b3282d", lw=0.5, alpha=0.6)
    ax_share.set_xlim(t.iloc[0], t.iloc[-1])

    ax = fig.add_subplot(gs[6])
    ax.plot(cc["lag_s"], cc["r"], color=GREY, lw=0.9)
    ax.axhline(cc["r_background_p95"], color="#b3282d", ls="--", lw=0.8)
    ax.axvline(0, color="k", lw=0.6)
    ax.plot(cc["lag_peak_s"], cc["r_peak"], "o", ms=5, color=MUSE_COL, zorder=5)
    ax.annotate(f"peak r = {cc['r_peak']:.2f} at {cc['lag_peak_s']:+.0f} s",
                xy=(cc["lag_peak_s"], cc["r_peak"]),
                xytext=(28, cc["r_peak"] * 0.92), fontsize=7, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))
    ax.text(-118, cc["r_background_p95"] + 0.012,
            f"95th pct of |r| beyond ±30 s = {cc['r_background_p95']:.2f}",
            fontsize=7, color="#b3282d")
    ax.set_xlabel("lag added to the mask clock (s)")
    ax.set_ylabel("correlation of\nhead-movement level")
    ax.set_title("Movement shared by both units confirms the shipped alignment")
    ax.margins(x=0.02)
    p = os.path.join(outdir, fname)
    fig.savefig(p)
    return fig, p


def fig3_motion_refinement(ref, sm, upd, outdir, fname="fig3_motion_refinement.png"):
    """Per-event lags from large head movements, and one event before/after."""
    _apply_style()
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.62, wspace=0.28)
    ev, used = ref["events"], ref["used"]
    med, se = ref["lag_median_s"], ref["lag_se_s"]
    rej = ev[~ev.index.isin(used.index)]

    ax = fig.add_subplot(gs[0, :])
    ax.axhspan(med - se, med + se, color=MUSE_COL, alpha=0.18, lw=0)
    ax.axhline(med, color=MUSE_COL, lw=1.0)
    ax.scatter(rej["t_rel_s"] / 60, rej["lag_s"], s=14, facecolors="none",
               edgecolors=GREY, lw=0.6, label=f"rejected (weak match, n={len(rej)})")
    sc = ax.scatter(used["t_rel_s"] / 60, used["lag_s"], s=16 + 60 * (used["r"] - 0.45),
                    c=used["r"], cmap="viridis", vmin=0.45, vmax=0.9, zorder=4,
                    label=f"used (n={len(used)})")
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.035)
    cb.set_label("peak r", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("minutes into the overlap")
    ax.set_ylabel("lag to add to\nmask clock (s)")
    ax.set_title(f"Each large head movement gives its own estimate of the residual offset: "
                 f"median {med:+.2f} s")
    ax.legend(loc="lower left", ncol=2, fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(used["lag_s"], bins=np.arange(-3, 3.01, 0.2), color="#8fb8d8",
            edgecolor="white", lw=0.4)
    ax.axvline(med, color=MUSE_COL, lw=1.2)
    ax.set_xlabel("lag (s)"); ax.set_ylabel("events")
    ax.set_title(f"median {med:+.2f} s, SE {se:.2f} s\nrobust SD {ref['lag_robust_sd_s']:.2f} s")

    # one clean event, before and after the shift
    ax = fig.add_subplot(gs[1, 1])
    best = used.sort_values("r", ascending=False).iloc[0]
    grid, Em, Eu = ref["grid"], ref["Em"], ref["Eu"]
    c = best["t_rel_s"]
    sel = (grid > c - 2.5) & (grid < c + 2.5)
    g = grid[sel] - c
    nrm = lambda v: v / (np.nanmax(v) or 1.0)
    ax.plot(g, nrm(Eu[sel]), color=MUSE_COL, lw=1.1, label="Muse jerk")
    ax.plot(g, nrm(np.interp(grid[sel] - med, grid, Em)), color="#1b6ca8", lw=1.1,
            label="mask jerk, shifted")
    ax.plot(g, nrm(Em[sel]), color=GREY, lw=0.8, ls="--", label="mask jerk, tap anchor only")
    ax.set_xlabel("seconds around the event")
    ax.set_ylabel("normalised jerk")
    ax.set_title(f"largest-r event, r = {best['r']:.2f}")
    ax.set_ylim(-0.05, 1.65)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    p = os.path.join(outdir, fname)
    fig.savefig(p)
    return fig, p
