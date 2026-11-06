"""First-10-minute overview of every populated Mind Monitor (Muse S) channel.

Raw export, no filtering: the point is to see what each channel actually
contains, including rail hits and dropouts. Panel y limits are robust
(median +/- 10 * 1.4826 * MAD) so a few railed samples cannot squash the trace;
those samples run off-panel by design.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

EEG_CH = ["TP9", "AF7", "AF8", "TP10"]
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
C_EEG = {"TP9": "#7f3b08", "AF7": "#e08214", "AF8": "#b2532a", "TP10": "#d9a05b"}
C_OPT = ["#1b5e20", "#2e7d32", "#4caf50", "#81c784"]
C_AUX = ["#5d4037", "#8d6e63", "#a1887f", "#bcaaa4"]
C_XYZ = ["#1f4e79", "#2e75b6", "#7fb3d9"]
GREY = "#555555"
RED = "#c0392b"


def envelope(t, x, npix=1500):
    """Per-pixel min / max / median so no transient is lost to decimation."""
    x = np.asarray(x, float)
    n = len(x)
    npix = int(min(npix, n))
    edges = np.unique(np.linspace(0, n, npix + 1).astype(int))
    starts = edges[:-1]
    lo = np.minimum.reduceat(x, starts)
    hi = np.maximum.reduceat(x, starts)
    mid = np.array([np.median(x[a:b]) for a, b in zip(starts, edges[1:])])
    tt = pd.to_datetime(pd.Series(t)).to_numpy()[starts]
    return tt, lo, hi, mid


def rlim(*arrays, k=10.0, pad=0.05):
    """Robust limits: median +/- k * 1.4826 * MAD over the pooled arrays."""
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    med = np.median(v)
    sig = 1.4826 * np.median(np.abs(v - med))
    if sig <= 0:
        sig = np.std(v) or 1.0
    lo, hi = med - k * sig, med + k * sig
    lo, hi = max(lo, v.min()), min(hi, v.max())
    if hi <= lo:
        lo, hi = v.min() - 1, v.max() + 1
    m = pad * (hi - lo)
    return lo - m, hi + m


def load(path, minutes=10.0):
    m = pd.read_csv(path, low_memory=False)
    m["time"] = pd.to_datetime(m["TimeStamp"])
    w = m[m.time < m.time.iloc[0] + pd.Timedelta(minutes=minutes)].copy()
    ev = w.loc[w["Elements"].notna(), ["time", "Elements"]].copy()
    w = w[w["RAW_TP9"].notna()].reset_index(drop=True)
    fs = len(w) / (w.time.iloc[-1] - w.time.iloc[0]).total_seconds()
    dead = [c for c in w.columns
            if c.startswith("Optics") and w[c].nunique(dropna=True) <= 1]
    return w, ev, fs, dead


def figure(w, ev, fs, dead, outdir, fname="fig11_muse_first10min.png"):
    rows = (["RAW_" + c for c in EEG_CH] + BANDS +
            ["AUX", "Optics", "Accel", "Gyro", "HR", "HSI", "Battery"])
    nr = len(rows)
    fig = plt.figure(figsize=(7.4, 1.02 * nr + 1.0))
    gs = fig.add_gridspec(nr, 1, hspace=0.22, left=0.13, right=0.985,
                          top=1 - 0.62 / (1.02 * nr + 1.0), bottom=0.052)
    t = w["time"].to_numpy()
    axes, ax0 = [], None

    def newax(i):
        nonlocal ax0
        ax = fig.add_subplot(gs[i], sharex=ax0)
        if ax0 is None:
            ax0 = ax
        axes.append(ax)
        if i < nr - 1:
            ax.tick_params(labelbottom=False)
        ax.tick_params(labelsize=6)
        return ax

    # --- raw EEG, one row per electrode ---------------------------------
    for i, c in enumerate(EEG_CH):
        ax = newax(i)
        col = "RAW_" + c
        tt, lo, hi, mid = envelope(t, w[col])
        ax.fill_between(tt, lo, hi, color=C_EEG[c], lw=0, alpha=0.85)
        ax.plot(tt, mid, color="#ffffff", lw=0.35, alpha=0.7)
        ax.set_ylim(*rlim(w[col]))
        ax.set_ylabel(f"RAW {c}\n\u00b5V", fontsize=7)
        rail = int(((w[col] <= 0.001) | (w[col] >= 1449.999)).sum())
        if rail:
            ax.annotate(f"{rail:,} railed samples", (0.995, 0.06),
                        xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=6, color=RED)

    # --- relative band power, four electrodes per row -------------------
    for j, b in enumerate(BANDS):
        ax = newax(4 + j)
        for c in EEG_CH:
            tt, lo, hi, mid = envelope(t, w[f"{b}_{c}"], npix=900)
            ax.plot(tt, mid, color=C_EEG[c], lw=0.5,
                    label=c if j == 0 else None)
        ax.set_ylabel(f"{b}\nlog(rel.)", fontsize=7)
        ax.set_ylim(*rlim(*[w[f"{b}_{c}"] for c in EEG_CH], k=12))
        if j == 0:
            ax.legend(ncol=4, fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1.1, columnspacing=1.0, borderaxespad=0.0)

    # --- AUX -------------------------------------------------------------
    ax = newax(9)
    for k, c in enumerate(["AUX_1", "AUX_2", "AUX_3", "AUX_4"]):
        tt, lo, hi, mid = envelope(t, w[c], npix=900)
        ax.plot(tt, mid, color=C_AUX[k], lw=0.5, label=c.replace("_", " "))
    ax.set_ylabel("AUX 1\u20134\n\u00b5V", fontsize=7)
    ax.set_ylim(*rlim(*[w[c] for c in ["AUX_1", "AUX_2", "AUX_3", "AUX_4"]],
                      k=22))
    ax.legend(ncol=4, fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1.1, columnspacing=1.0, borderaxespad=0.0)

    # --- optics ----------------------------------------------------------
    ax = newax(10)
    for k in range(4):
        c = f"Optics{k+1}"
        tt, lo, hi, mid = envelope(t, w[c], npix=900)
        ax.plot(tt, mid, color=C_OPT[k], lw=0.6, label=str(k + 1))
    ax.set_ylabel("Optics 1\u20134\nsensor units", fontsize=7)
    ax.set_ylim(*rlim(*[w[f"Optics{k+1}"] for k in range(4)], k=12))
    ax.legend(ncol=4, fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1.1, columnspacing=1.0, borderaxespad=0.0)
    if dead:
        ax.annotate("Optics 5\u201316 identically zero in this export",
                    (0.005, 0.46), xycoords="axes fraction", ha="left",
                    va="center", fontsize=6, color=GREY)

    # --- accelerometer / gyroscope ---------------------------------------
    for i, (kind, cols, unit) in enumerate(
            [("Accel", ["Accelerometer_X", "Accelerometer_Y",
                        "Accelerometer_Z"], "g"),
             ("Gyro", ["Gyro_X", "Gyro_Y", "Gyro_Z"], "\u00b0/s")]):
        ax = newax(11 + i)
        for k, c in enumerate(cols):
            tt, lo, hi, mid = envelope(t, w[c], npix=1200)
            ax.plot(tt, mid, color=C_XYZ[k], lw=0.5, label=c[-1])
        ax.set_ylabel(f"{kind}\n{unit}", fontsize=7)
        ax.set_ylim(*rlim(*[w[c] for c in cols], k=14))
        ax.legend(ncol=3, fontsize=6, frameon=False, loc="upper right",
                  handlelength=1.1, columnspacing=1.0, borderaxespad=0.15)

    # --- heart rate -------------------------------------------------------
    ax = newax(13)
    tt, lo, hi, mid = envelope(t, w["Heart_Rate"], npix=900)
    ax.plot(tt, mid, color="#b2182b", lw=0.6)
    ax.set_ylabel("Heart rate\nbpm", fontsize=7)
    ax.set_ylim(*rlim(w["Heart_Rate"], k=8))

    # --- fit indicator ----------------------------------------------------
    ax = newax(14)
    for c in EEG_CH:
        y = w[f"HSI_{c}"].to_numpy(float)
        tt, lo, hi, mid = envelope(t, y, npix=1200)
        ax.step(tt, hi, where="post", color=C_EEG[c], lw=0.7, label=c)
    ax.set_ylabel("Fit (HSI)\n1 good \u2192 4 bad", fontsize=7)
    ax.set_yticks([1, 2, 4])
    ax.set_ylim(0.7, 4.4)
    ax.legend(ncol=4, fontsize=6, frameon=False, loc="upper right",
              handlelength=1.1, columnspacing=1.0, borderaxespad=0.15)

    # --- battery + headband-on + element events ---------------------------
    ax = newax(15)
    ax.plot(t, w["Battery"].to_numpy(float), color=GREY, lw=0.8)
    ax.set_ylabel("Battery\n%", fontsize=7)
    ax.set_ylim(*rlim(w["Battery"], k=6))
    on = w["HeadBandOn"].to_numpy(float)
    ax.annotate("headband-on = 1 throughout", (0.005, 0.12),
                xycoords="axes fraction", fontsize=6, color=GREY,
                ha="left", va="bottom") if np.all(on == 1) else None

    # element events as a rug across the frontal EEG rows
    kinds = {"blink": ("#2e75b6", axes[1]), "jaw_clench": (RED, axes[2])}
    for key, (col, target) in kinds.items():
        te = ev.loc[ev["Elements"].str.contains(key), "time"].to_numpy()
        if len(te) == 0:
            continue
        target.vlines(te, 0.93, 1.0, transform=target.get_xaxis_transform(),
                      color=col, lw=0.5, alpha=0.9)
        target.annotate(f"{key.replace('_', ' ')} ({len(te)})", (0.005, 0.90),
                        xycoords="axes fraction", fontsize=6, color=col,
                        ha="left", va="top")

    ax0.set_xlim(t[0], t[-1])
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("wall clock (local)", fontsize=8)
    for ax in axes:
        ax.margins(x=0)

    t0 = pd.to_datetime(t[0])
    fig.suptitle(
        "Mind Monitor (Muse S) raw export, first 10 minutes \u2014 every "
        "populated channel\n"
        f"{t0:%Y-%m-%d %H:%M:%S} onward \u00b7 {fs:.1f} Hz \u00b7 "
        f"{len(w):,} samples \u00b7 no filtering; panel limits are robust "
        "(median \u00b1 10 MAD\u2011\u03c3), so rail hits run off-panel",
        fontsize=8.5, ha="left", x=0.012, y=0.995, va="top")
    p = os.path.join(outdir, fname)
    fig.savefig(p, dpi=200)
    return fig, p


def build(path, outdir):
    w, ev, fs, dead = load(path)
    fig, p = figure(w, ev, fs, dead, outdir)
    return dict(fig=fig, path=p, fs=fs, n=len(w), dead=dead,
                events=ev["Elements"].value_counts().to_dict(),
                t0=str(w.time.iloc[0]), t1=str(w.time.iloc[-1]))
