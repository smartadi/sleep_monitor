"""
Time series + spectrogram figures for every channel of the two devices.

One figure per acquisition stream; inside a figure each channel gets a pair of
stacked panels sharing the x (wall-clock) axis:

    - a time trace, drawn as a per-pixel min/max envelope so no transient is
      lost to decimation,
    - a spectrogram in dB, Hann window, 50% overlap.

All frequency axes are labelled with the true sample rate of that stream.
Muse's Optics channels are forward-filled in the Mind Monitor CSV, so they are
rebuilt on a uniform grid from their change instants before transforming
(see muse_optics_uniform).
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import spectrogram

try:
    from figure_style import apply_figure_style
except Exception:  # helper is injected into the kernel by the skill
    apply_figure_style = globals().get("apply_figure_style", lambda *a, **k: None)

GREY = "#6b6b6b"
FS_MASK = 1000.0 / 9.0          # mask tick is exactly 9 ms
FS_MUSE = 256.0                 # Mind Monitor EEG/AUX row rate
FS_OPTICS = 20.0                # uniform grid for the ~21.5 Hz optical stream

MASK_LABELS = {"CH": "CH\n(head)", "CLE": "CLE\n(left eye)", "CRE": "CRE\n(right eye)"}


# ----------------------------------------------------------------- utilities
def envelope(t, x, n_px=1400):
    """Per-pixel min/max envelope: returns (t_px, lo, hi, mid)."""
    x = np.asarray(x, float)
    n = len(x)
    edges = np.linspace(0, n, min(n_px, n) + 1).astype(int)
    lo = np.empty(len(edges) - 1)
    hi = np.empty(len(edges) - 1)
    mid = np.empty(len(edges) - 1)
    tc = np.empty(len(edges) - 1, dtype="datetime64[ns]")
    tv = np.asarray(t, dtype="datetime64[ns]").astype("int64")
    for i in range(len(edges) - 1):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        seg = x[a:b]
        good = seg[np.isfinite(seg)]
        if good.size == 0:
            lo[i] = hi[i] = mid[i] = np.nan
        else:
            lo[i], hi[i], mid[i] = good.min(), good.max(), np.median(good)
        tc[i] = np.int64(tv[a:b].mean()).astype("datetime64[ns]")
    return pd.to_datetime(tc), lo, hi, mid


def spec(x, fs, t0, win_s=30.0, fmin=None, fmax=None, n_avg=1, n_logbin=0):
    """Spectrogram in dB on a wall-clock x axis. Returns (t, f, S_dB).

    A raw periodogram carries chi2(2) speckle of about +/-5 dB, which buries a
    narrow-dynamic-range channel in visual noise. Two variance reductions are
    available and both are plain Welch averaging:

    n_avg    average this many adjacent time windows (time resolution becomes
             n_avg * win_s / 2); variance falls ~n_avg-fold.
    n_logbin average the linear power into this many geometric frequency bins
             before taking dB, matching a log frequency axis and averaging
             progressively more bins per pixel as frequency rises.
    """
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    x = x - np.mean(x)
    nper = int(round(win_s * fs))
    f, tsec, S = spectrogram(x, fs=fs, window="hann", nperseg=nper,
                             noverlap=nper // 2, detrend="linear",
                             scaling="density", mode="psd")
    if n_avg > 1:
        k = (S.shape[1] // n_avg) * n_avg
        S = S[:, :k].reshape(len(f), -1, n_avg).mean(axis=2)
        tsec = tsec[:k].reshape(-1, n_avg).mean(axis=1)
    keep = np.ones_like(f, bool)
    if fmin is not None:
        keep &= f >= fmin
    if fmax is not None:
        keep &= f <= fmax
    f, S = f[keep], S[keep]
    if n_logbin:
        edges = np.geomspace(f[0], f[-1], n_logbin + 1)
        idx = np.clip(np.digitize(f, edges) - 1, 0, n_logbin - 1)
        Sb = np.full((n_logbin, S.shape[1]), np.nan)
        fb = np.sqrt(edges[:-1] * edges[1:])
        for i in range(n_logbin):
            m = idx == i
            if m.any():
                Sb[i] = S[m].mean(axis=0)
        ok = np.isfinite(Sb).all(axis=1)
        f, S = fb[ok], Sb[ok]
    S_db = 10.0 * np.log10(np.maximum(S, np.finfo(float).tiny))
    t_ax = pd.to_datetime(t0) + pd.to_timedelta(tsec, unit="s")
    return t_ax, f, S_db


def muse_optics_uniform(muse, cols, fs=FS_OPTICS):
    """Rebuild the forward-filled Optics channels on a uniform grid."""
    v = muse[cols].to_numpy(float)
    ok = np.all(np.isfinite(v), axis=1)
    v, t = v[ok], muse["time"].to_numpy()[ok]
    chg = np.r_[True, np.any(v[1:] != v[:-1], axis=1)]
    t_u, v_u = t[chg], v[chg]
    grid = pd.date_range(t_u[0].astype("datetime64[ns]"),
                         t_u[-1].astype("datetime64[ns]"),
                         freq=pd.Timedelta(seconds=1.0 / fs))
    gx = grid.view("int64").astype(float)
    tx = t_u.astype("datetime64[ns]").astype("int64").astype(float)
    out = pd.DataFrame({c: np.interp(gx, tx, v_u[:, i]) for i, c in enumerate(cols)},
                       index=grid)
    return out, float(chg.sum() / ((tx[-1] - tx[0]) / 1e9))


# ------------------------------------------------------------------- plotting
def _stream_figure(chans, get_series, fs, t0, outdir, fname, title,
                   win_s, fmin, fmax, logf, unit, spec_cmap="magma",
                   time_pctl=(0.2, 99.8), db_pctl=(5, 99.5), figw=7.2,
                   row_h=1.55, notes=None, n_avg=1, n_logbin=0):
    apply_figure_style()
    n = len(chans)
    fig = plt.figure(figsize=(figw, 0.55 + row_h * n))
    gs = fig.add_gridspec(2 * n, 1, height_ratios=[0.85, 1.0] * n,
                          hspace=0.14, left=0.125, right=0.885,
                          top=1 - 0.52 / (0.55 + row_h * n), bottom=0.055)
    share = None
    for i, c in enumerate(chans):
        t, x = get_series(c)

        ax = fig.add_subplot(gs[2 * i], sharex=share)
        share = share or ax
        te, lo, hi, mid = envelope(t, x)
        ax.fill_between(te, lo, hi, color="#2b6cb0", lw=0, alpha=0.55)
        ax.plot(te, mid, color="#1a365d", lw=0.5)
        good = np.asarray(x, float)
        good = good[np.isfinite(good)]
        ax.set_ylim(np.percentile(good, time_pctl[0]), np.percentile(good, time_pctl[1]))
        ax.set_ylabel(f"{chans[c] if isinstance(chans, dict) else c}\n{unit}")
        ax.tick_params(labelbottom=False)
        ax.margins(x=0)

        axs = fig.add_subplot(gs[2 * i + 1], sharex=share)
        ts, f, S = spec(x, fs, t[0], win_s=win_s, fmin=fmin, fmax=fmax,
                        n_avg=n_avg, n_logbin=n_logbin)
        vmin, vmax = np.percentile(S, db_pctl)
        m = axs.pcolormesh(ts, f, S, cmap=spec_cmap, vmin=vmin, vmax=vmax,
                           shading="nearest", rasterized=True)
        axs.set_ylim(f.min(), f.max())
        if logf:
            axs.set_yscale("log")
            axs.set_yticks([0.1, 1, 10])
            axs.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda v, _: f"{v:g}"))
        axs.set_ylabel("Hz")
        axs.margins(x=0)
        cax = axs.inset_axes([1.012, 0.0, 0.018, 1.0])
        cb = fig.colorbar(m, cax=cax)
        cb.set_label("dB", labelpad=13)
        cb.ax.tick_params(length=1.8, pad=1.2)
        if i < n - 1:
            axs.tick_params(labelbottom=False)
        else:
            axs.xaxis.set_major_locator(mdates.HourLocator())
            axs.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            axs.set_xlabel("wall clock (local)")

    sub = f"{fs:g} Hz · {win_s:g} s Hann window, 50% overlap"
    if n_avg > 1:
        sub += f", {n_avg}-window Welch average ({n_avg * win_s / 2:.0f} s/column)"
    if n_logbin:
        sub2 = f"{n_logbin} geometric frequency bins"
    else:
        sub2 = "linear frequency axis"
    if notes:
        sub2 += f" · {notes}"
    fig.suptitle(f"{title}\n{sub}\n{sub2}", fontsize=7.6, x=0.125, ha="left",
                 va="top", y=0.998)
    p = os.path.join(outdir, fname)
    fig.savefig(p)
    return fig, p


def fig4_mask_spectra(sm, outdir, fname="fig4_mask_time_spectrogram.png"):
    def get(c):
        return sm["time"].to_numpy(), sm[c].to_numpy(float)
    return _stream_figure(MASK_LABELS, get, FS_MASK, sm["time"].iloc[0], outdir,
                          fname, "Sleep-mask capacitive channels",
                          win_s=30.0, fmin=0.04, fmax=50.0, logf=True,
                          unit="(a.u.)", spec_cmap="magma", row_h=1.6,
                          n_avg=8, n_logbin=56)


def fig5_muse_eeg_spectra(muse, outdir, fname="fig5_muse_eeg_time_spectrogram.png"):
    chans = {"RAW_TP9": "TP9\n(µV)", "RAW_AF7": "AF7\n(µV)",
             "RAW_AF8": "AF8\n(µV)", "RAW_TP10": "TP10\n(µV)"}
    m = muse.dropna(subset=["RAW_AF7"]).reset_index(drop=True)

    def get(c):
        return m["time"].to_numpy(), m[c].to_numpy(float)
    return _stream_figure(chans, get, FS_MUSE, m["time"].iloc[0], outdir, fname,
                          "Muse S EEG", win_s=30.0, fmin=0.5, fmax=35.0,
                          logf=False, unit="", spec_cmap="viridis", row_h=1.5,
                          n_avg=4)


def fig6_optics_spectra(muse, outdir, fname="fig6_optics_time_spectrogram.png"):
    cols = ["Optics1", "Optics2", "Optics3", "Optics4"]
    uni, rate = muse_optics_uniform(muse, cols)
    chans = {c: c.replace("Optics", "Optics ") + "\n(a.u.)" for c in cols}

    def get(c):
        return uni.index.to_numpy(), uni[c].to_numpy(float)
    return _stream_figure(chans, get, FS_OPTICS, uni.index[0], outdir, fname,
                          "Muse S optical (fNIRS/PPG) channels", win_s=60.0,
                          fmin=0.04, fmax=9.5, logf=True, unit="",
                          spec_cmap="magma", row_h=1.5, n_avg=4, n_logbin=48,
                          notes=f"resampled from {rate:.1f} Hz native updates")


def fig7_aux_spectra(muse, outdir, fname="fig7_aux_time_spectrogram.png"):
    cols = ["AUX_1", "AUX_2", "AUX_3", "AUX_4"]
    m = muse.dropna(subset=cols).reset_index(drop=True)
    chans = {c: c.replace("_", " ") + "\n(a.u.)" for c in cols}

    def get(c):
        return m["time"].to_numpy(), m[c].to_numpy(float)
    return _stream_figure(chans, get, FS_MUSE, m["time"].iloc[0], outdir, fname,
                          "Muse S auxiliary analog channels", win_s=30.0,
                          fmin=0.5, fmax=100.0, logf=True, unit="",
                          spec_cmap="viridis", row_h=1.5, n_avg=4, n_logbin=56,
                          time_pctl=(2, 98))


def band_table(sm, muse, outdir, fname="channel_band_power.csv"):
    """Median band power per channel over the two-device overlap window."""
    rows = []
    lo, hi = muse["time"].min(), muse["time"].max()
    smo = sm[(sm.time >= lo) & (sm.time <= hi)]
    bands = [("0.04-0.5", 0.04, 0.5), ("0.5-2", 0.5, 2), ("2-8", 2, 8),
             ("8-16", 8, 16), ("16-50", 16, 50)]
    streams = [("sleepmask", FS_MASK, smo, ["CH", "CLE", "CRE"], 30.0),
               ("muse_eeg", FS_MUSE, muse.dropna(subset=["RAW_AF7"]),
                ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"], 30.0),
               ("muse_aux", FS_MUSE, muse.dropna(subset=["AUX_1"]),
                ["AUX_1", "AUX_2", "AUX_3", "AUX_4"], 30.0)]
    uni, rate = muse_optics_uniform(muse, ["Optics1", "Optics2", "Optics3", "Optics4"])
    streams.append(("muse_optics", FS_OPTICS, uni.reset_index(names="time"),
                    ["Optics1", "Optics2", "Optics3", "Optics4"], 60.0))
    for stream, fs, df, cols, win in streams:
        for c in cols:
            _, f, S = spec(df[c].to_numpy(float), fs, df["time"].iloc[0], win_s=win)
            r = {"stream": stream, "channel": c, "fs_hz": round(fs, 2)}
            for name, a, b in bands:
                m = (f >= a) & (f < min(b, fs / 2))
                r[f"P_{name}Hz_dB"] = round(float(np.median(S[m])), 2) if m.any() else np.nan
                if m.any():
                    r[f"peak_{name}Hz"] = round(float(f[m][np.median(S[m], axis=1).argmax()]), 3)
            rows.append(r)
    out = pd.DataFrame(rows)
    p = os.path.join(outdir, fname)
    out.to_csv(p, index=False)
    return out, p
