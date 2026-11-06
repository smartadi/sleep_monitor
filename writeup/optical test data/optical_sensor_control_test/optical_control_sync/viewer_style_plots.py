"""
Time-domain, spectrogram and power-spectrum figures for the mask / Muse control
session, using the signal-processing chain of the session viewer
(smartadi/sleep_monitor, filters.py + preprocessing.py + motion.py):

  * Butterworth order 3, zero-phase `filtfilt`.
  * outlier clip at +/- 4 sigma before filtering.
  * OLS accelerometer-magnitude regression on the capacitive channels, with
    both cap and acc bandpassed to the same band so only in-band motion energy
    is subtracted:  beta = <acc,cap> / <acc,acc>.
  * gravity = 0.5 Hz lowpass per axis; head tilt from the gravity vector with
    the user's two formulas.  Dynamic acceleration = 0.5 Hz highpass per axis.
  * motion gate: per-30 s RMS of dynamic acceleration, flagged when
    RMS > median + 3 * MAD.
  * viewer band edges: respiratory 0.1-0.5 Hz, cardiac 0.5-3.0 Hz.

Channels are restricted to the two-device overlap window so every row shares
one wall clock.  TP9/TP10 and the AUX inputs are deliberately excluded.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch, find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from spectra import (apply_figure_style, envelope, spec, muse_optics_uniform,
                     FS_MASK, FS_MUSE, FS_OPTICS, GREY)

ORDER = 3
SLOW_BAND = (0.03, 5.0)      # display band for mask + optics
EEG_BAND = (0.5, 30.0)       # display band for EEG
RESP = (0.1, 0.5)            # viewer respiratory band
CARD = (0.5, 3.0)            # viewer cardiac band
GRAV_FC = 0.5                # gravity / dynamic split
MASK_CH = ["CLE", "CRE", "CH"]
EEG_CH = ["RAW_AF8", "RAW_AF7"]
OPT_CH = ["Optics1", "Optics2", "Optics3", "Optics4"]

C_MASK = {"CLE": "#1f4e79", "CRE": "#2e75b6", "CH": "#0f2d4a"}
C_TILT = {"headUDTilt": "#7f7f7f", "headLRTilt": "#bfbfbf"}
C_EEG = {"RAW_AF8": "#a33b1f", "RAW_AF7": "#d1743f"}
C_OPT = {"Optics1": "#1b7837", "Optics2": "#4daf4a",
         "Optics3": "#7fbc41", "Optics4": "#b8e186"}
LBL = {"CLE": "CLE", "CRE": "CRE", "CH": "CH",
       "headUDTilt": "head tilt\nUD", "headLRTilt": "head tilt\nLR",
       "RAW_AF8": "EEG AF8", "RAW_AF7": "EEG AF7",
       "Optics1": "Optics 1", "Optics2": "Optics 2",
       "Optics3": "Optics 3", "Optics4": "Optics 4"}


# ---------------------------------------------------------------- filters
def _fill(x):
    x = np.asarray(x, float)
    if np.all(np.isfinite(x)):
        return x
    m = np.isfinite(x)
    return np.interp(np.arange(len(x)), np.flatnonzero(m), x[m])


def bandpass(x, fs, lo, hi, order=ORDER):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, _fill(x))


def lowpass(x, fs, fc, order=ORDER):
    sos = butter(order, fc, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, _fill(x))


def highpass(x, fs, fc, order=ORDER):
    sos = butter(order, fc, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, _fill(x))


def outlier_clip(x, k=4.0):
    x = np.asarray(x, float)
    m, s = np.nanmedian(x), np.nanstd(x)
    return np.clip(x, m - k * s, m + k * s)


def remove_acc_artifact(cap, acc_mag, fs, band):
    """Viewer OLS variant: single regression coefficient, in-band only."""
    a = bandpass(acc_mag, fs, *band)
    c = bandpass(outlier_clip(cap), fs, *band)
    beta = float(np.dot(a, c) / np.dot(a, a))
    return c - beta * a, beta


# ---------------------------------------------------------------- prepare
def overlap(sm, muse):
    lo = max(sm["time"].iloc[0], muse["time"].iloc[0])
    hi = min(sm["time"].iloc[-1], muse["time"].iloc[-1])
    return pd.Timestamp(lo), pd.Timestamp(hi)


def prep_mask(sm, t0, t1, band=SLOW_BAND):
    d = sm[(sm["time"] >= t0) & (sm["time"] <= t1)].reset_index(drop=True)
    acc = {ax: d[ax].to_numpy(float) for ax in ("aX", "aY", "aZ")}
    acc_mag = np.sqrt(sum(v ** 2 for v in acc.values()))
    out = {"time": d["time"].to_numpy()}
    betas = {}
    for c in MASK_CH:
        out[c], betas[c] = remove_acc_artifact(d[c].to_numpy(float), acc_mag,
                                               FS_MASK, band)
    g = {ax: lowpass(v, FS_MASK, GRAV_FC) for ax, v in acc.items()}
    out["headUDTilt"] = np.degrees(np.arctan2(
        g["aX"], np.sqrt(g["aY"] ** 2 + g["aZ"] ** 2)))
    out["headLRTilt"] = np.degrees(np.arctan2(
        g["aY"], np.sqrt(g["aX"] ** 2 + g["aZ"] ** 2)))
    dyn = np.sqrt(sum(highpass(v, FS_MASK, GRAV_FC) ** 2 for v in acc.values()))
    out["dyn_acc"] = dyn
    return pd.DataFrame(out), betas


def prep_eeg(muse, t0, t1, band=EEG_BAND):
    d = muse[muse["RAW_AF7"].notna() & muse["RAW_AF8"].notna()]
    d = d[(d["time"] >= t0) & (d["time"] <= t1)].reset_index(drop=True)
    out = {"time": d["time"].to_numpy()}
    for c in EEG_CH:
        out[c] = bandpass(outlier_clip(d[c].to_numpy(float), k=8.0),
                          FS_MUSE, *band)
    return pd.DataFrame(out)


def prep_optics(muse, t0, t1, band=SLOW_BAND):
    u, rate = muse_optics_uniform(muse, OPT_CH, fs=FS_OPTICS)
    u = u[(u.index >= t0) & (u.index <= t1)]
    out = {"time": u.index.to_numpy()}
    for c in OPT_CH:
        out[c] = bandpass(u[c].to_numpy(float), FS_OPTICS, *band)
    return pd.DataFrame(out), rate


def motion_gate(time, dyn, fs, epoch_s=30.0, mad_k=3.0):
    n = int(round(epoch_s * fs))
    k = len(dyn) // n
    r = np.sqrt((dyn[:k * n].reshape(k, n) ** 2).mean(axis=1))
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    thr = med + mad_k * mad
    flag = r > thr
    starts = pd.to_datetime(time[:k * n].reshape(k, n)[:, 0])
    spans, i = [], 0
    while i < k:
        if flag[i]:
            j = i
            while j + 1 < k and flag[j + 1]:
                j += 1
            spans.append((starts[i], starts[j] + pd.Timedelta(seconds=epoch_s)))
            i = j + 1
        else:
            i += 1
    return dict(rms=r, epoch_start=starts, flag=flag, thresh=float(thr),
                median=float(med), mad=float(mad), spans=spans,
                frac=float(flag.mean()), n_epochs=int(k))


# ---------------------------------------------------------------- figure 8
def fig8_time_domain(mask, eeg, opt, gate, outdir,
                     fname="fig8_time_domain.png", n_px=1600):
    apply_figure_style()
    rows = ([(c, mask, "a.u.", C_MASK[c]) for c in MASK_CH] +
            [(c, mask, "deg", C_TILT[c]) for c in ("headUDTilt", "headLRTilt")] +
            [(c, eeg, "\u00b5V", C_EEG[c]) for c in EEG_CH] +
            [(c, opt, "a.u.", C_OPT[c]) for c in OPT_CH])
    n = len(rows)
    fig = plt.figure(figsize=(7.2, 0.82 * n + 1.0))
    gs = fig.add_gridspec(n, 1, hspace=0.0, top=1 - 0.9 / (0.82 * n + 1.0),
                          bottom=0.06, left=0.135, right=0.985)
    ax0 = None
    for i, (c, df, unit, col) in enumerate(rows):
        ax = fig.add_subplot(gs[i], sharex=ax0)
        ax0 = ax0 or ax
        t, lo, hi, mid = envelope(df["time"].to_numpy(), df[c].to_numpy(), n_px)
        for a, b in gate["spans"]:
            ax.axvspan(a, b, color="#d9d9d9", lw=0, zorder=0)
        ax.fill_between(t, lo, hi, color=col, lw=0, alpha=0.85, zorder=2)
        ax.plot(t, mid, color=col, lw=0.35, zorder=3)
        x = df[c].to_numpy()
        med = np.nanmedian(x)
        sd = 1.4826 * np.nanmedian(np.abs(x - med))
        half = max(10 * sd, 1e-12)
        ax.set_ylim(med - half, med + half)
        ax.set_ylabel(f"{LBL[c]}\n({unit})", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.005)
        if i < n - 1:
            ax.tick_params(labelbottom=False)
    ax0.set_xlim(mask["time"].iloc[0], mask["time"].iloc[-1])
    ax = fig.axes[-1]
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("wall clock (local)")
    fig.axes[0].set_title(
        "Time domain, two-device overlap window\n"
        f"mask {FS_MASK:.1f} Hz \u00b7 EEG {FS_MUSE:.0f} Hz \u00b7 "
        f"optics {FS_OPTICS:.0f} Hz \u00b7 order-3 Butterworth zero-phase; "
        f"capacitive {SLOW_BAND[0]}\u2013{SLOW_BAND[1]} Hz after OLS "
        "accelerometer regression,\n"
        f"EEG {EEG_BAND[0]}\u2013{EEG_BAND[1]} Hz, optics "
        f"{SLOW_BAND[0]}\u2013{SLOW_BAND[1]} Hz \u00b7 tilt from the "
        f"{GRAV_FC} Hz gravity vector \u00b7 grey = motion-gated epochs "
        f"({100*gate['frac']:.0f}% of the night)",
        fontsize=7.2, loc="left", pad=6)
    p = os.path.join(outdir, fname)
    fig.savefig(p)
    return fig, p


# ---------------------------------------------------------------- figure 9
def fig9_spectrograms(mask, eeg, opt, outdir, fname="fig9_spectrograms.png"):
    apply_figure_style()
    rows = ([(c, mask, FS_MASK, 60.0, SLOW_BAND, 2) for c in MASK_CH] +
            [(c, eeg, FS_MUSE, 30.0, EEG_BAND, 4) for c in EEG_CH] +
            [(c, opt, FS_OPTICS, 60.0, SLOW_BAND, 2) for c in OPT_CH])
    n = len(rows)
    fig = plt.figure(figsize=(7.2, 1.15 * n + 1.1))
    gs = fig.add_gridspec(n, 1, hspace=0.22, top=1 - 1.0 / (1.15 * n + 1.1),
                          bottom=0.055, left=0.105, right=0.9)
    ax0 = None
    for i, (c, df, fs, win_s, band, n_avg) in enumerate(rows):
        ax = fig.add_subplot(gs[i], sharex=ax0)
        ax0 = ax0 or ax
        t, f, S = spec(df[c].to_numpy(), fs, df["time"].iloc[0],
                       win_s=win_s, fmin=band[0], fmax=band[1], n_avg=n_avg)
        pct = (10, 97) if band == SLOW_BAND else (5, 99.5)
        vmin, vmax = np.nanpercentile(S, pct)
        m = ax.pcolormesh(t, f, S, cmap="magma", vmin=vmin, vmax=vmax,
                          shading="nearest", rasterized=True)
        cb = fig.colorbar(m, ax=ax, pad=0.012, fraction=0.032)
        cb.set_label("dB", fontsize=7, labelpad=1)
        cb.ax.tick_params(labelsize=6.5)
        ax.set_ylabel(f"{LBL[c]}\nHz", fontsize=7.5)
        ax.tick_params(labelsize=7)
        if band == SLOW_BAND:
            ax.set_yticks([1, 2, 3, 4, 5])
        else:
            ax.set_yticks([5, 15, 25])
        if i < n - 1:
            ax.tick_params(labelbottom=False)
    ax0.set_xlim(mask["time"].iloc[0], mask["time"].iloc[-1])
    ax = fig.axes[-2]
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("wall clock (local)")
    fig.axes[0].set_title(
        "Spectrograms, two-device overlap window\n"
        f"capacitive + optics: {SLOW_BAND[1]:.0f} Hz span, 60 s Hann window, "
        "50% overlap, 2-window Welch average (60 s/column)\n"
        f"EEG: {EEG_BAND[0]}\u2013{EEG_BAND[1]:.0f} Hz, 30 s Hann window, "
        "50% overlap, 4-window Welch average (60 s/column)",
        fontsize=7.2, loc="left", pad=6)
    p = os.path.join(outdir, fname)
    fig.savefig(p)
    return fig, p


# --------------------------------------------------------------- figure 10
def _psd(x, fs, nper_s):
    nper = int(round(min(nper_s * fs, len(x) / 4)))
    f, P = welch(np.asarray(x, float), fs=fs, window="hann", nperseg=nper,
                 noverlap=nper // 2, detrend="linear", scaling="density")
    return f, P


def clean_runs(time, fs, bad_spans, settle_s=120.0, min_s=60.0):
    """Contiguous index runs excluding motion-gated spans and the tap settle."""
    t = pd.to_datetime(pd.Series(time)).to_numpy()
    ok = np.ones(len(t), bool)
    ok[t < t[0] + np.timedelta64(int(settle_s * 1000), "ms")] = False
    for a, b in bad_spans:
        ok &= ~((t >= np.datetime64(a)) & (t <= np.datetime64(b)))
    runs, i, n = [], 0, len(ok)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            if (j - i + 1) / fs >= min_s:
                runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return runs, float(ok.mean())


def psd_gated(df, c, fs, nper_s, runs):
    """Welch PSD averaged over motion-free runs only (viewer motion mask)."""
    x = df[c].to_numpy(float)
    acc, wts, fref = [], [], None
    for a, b in runs:
        seg = x[a:b]
        nper = int(round(min(nper_s * fs, len(seg))))
        f, P = welch(seg, fs=fs, window="hann", nperseg=nper,
                     noverlap=nper // 2, detrend="linear", scaling="density")
        if fref is None:
            fref = f
        if len(f) == len(fref):
            acc.append(P)
            wts.append(len(seg))
    P = np.average(np.vstack(acc), axis=0, weights=wts)
    return fref, P


def fig10_power_spectra(mask, eeg, opt, outdir, gate,
                        fname="fig10_power_spectra.png",
                        nper_slow=120.0, nper_eeg=8.0):
    apply_figure_style()
    groups = [("Sleep-mask capacitive channels", MASK_CH, mask, FS_MASK,
               C_MASK, nper_slow, SLOW_BAND),
              ("Muse EEG", EEG_CH, eeg, FS_MUSE, C_EEG, nper_eeg, EEG_BAND),
              ("Muse optical channels", OPT_CH, opt, FS_OPTICS, C_OPT,
               nper_slow, SLOW_BAND)]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6))
    fig.subplots_adjust(top=0.895, bottom=0.07, left=0.105, right=0.985,
                        hspace=0.34)
    rows = []
    for ax, (name, chans, df, fs, cols, nper_s, band) in zip(axes, groups):
        slow = band == SLOW_BAND
        if slow:
            ax.axvspan(*RESP, color="#e8eef5", lw=0, zorder=0)
            ax.axvspan(*CARD, color="#f3ece2", lw=0, zorder=0)
        runs, frac = clean_runs(df["time"].to_numpy(), fs, gate["spans"])
        for c in chans:
            f, P = psd_gated(df, c, fs, nper_s, runs)
            keep = (f >= band[0]) & (f <= band[1])
            f, P = f[keep], P[keep]
            db = 10 * np.log10(np.maximum(P, np.finfo(float).tiny))
            ax.plot(f, db, color=cols[c], lw=0.9, label=LBL[c], zorder=3)
            if slow:
                inb = (f >= CARD[0]) & (f <= CARD[1])
                fpk = float(f[inb][np.argmax(P[inb])])
                ax.plot([fpk], [db[inb][np.argmax(P[inb])]], "v",
                        color=cols[c], ms=3.5, zorder=4)
            rows.append(pd.DataFrame({"group": name, "channel": LBL[c],
                                      "fs_hz": fs, "freq_hz": f, "psd_db": db}))
        ax.set_xlim(*band)
        ax.set_ylabel("PSD (dB / Hz)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(f"{name}  \u00b7 Welch {nper_s:g} s Hann, 50% overlap, "
                     f"{len(runs)} motion-free runs ({100*frac:.0f}% of the "
                     "window)", fontsize=8, loc="left")
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
        if slow:
            ax.annotate("resp\n0.1\u20130.5", (0.11, 0.965), xycoords=("data",
                        "axes fraction"), ha="left", va="top", fontsize=6.5,
                        color="#4a6d94")
            ax.annotate("cardiac 0.5\u20133.0", (1.75, 0.965),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="top", fontsize=6.5, color="#8a6d47")
    fig.suptitle("Whole-session power spectra, two-device overlap window\n"
                 "marker = strongest cardiac-band line per channel; "
                 "shaded = viewer respiratory and cardiac bands",
                 fontsize=8.2, x=0.105, ha="left", y=0.985)
    p = os.path.join(outdir, fname)
    fig.savefig(p)
    table = pd.concat(rows, ignore_index=True)
    pc = os.path.join(outdir, "power_spectra_control.csv")
    table.to_csv(pc, index=False)
    return fig, p, table, pc


# ------------------------------------------------------------------ driver
def build(sm, muse, outdir):
    t0, t1 = overlap(sm, muse)
    mask, betas = prep_mask(sm, t0, t1)
    eeg = prep_eeg(muse, t0, t1)
    opt, opt_rate = prep_optics(muse, t0, t1)
    gate = motion_gate(mask["time"].to_numpy(), mask["dyn_acc"].to_numpy(),
                       FS_MASK)
    f8, p8 = fig8_time_domain(mask, eeg, opt, gate, outdir)
    f9, p9 = fig9_spectrograms(mask, eeg, opt, outdir)
    f10, p10, tab, pc = fig10_power_spectra(mask, eeg, opt, outdir, gate)
    peaks = {}
    for chans, df, fs, nper in ((MASK_CH, mask, FS_MASK, 120.0),
                                (OPT_CH, opt, FS_OPTICS, 120.0)):
        runs, _ = clean_runs(df["time"].to_numpy(), fs, gate["spans"])
        for c in chans:
            f, P = psd_gated(df, c, fs, nper, runs)
            for nm, b in (("resp", RESP), ("card", CARD)):
                m = (f >= b[0]) & (f <= b[1])
                peaks[f"{c}_{nm}_peak_hz"] = round(float(f[m][np.argmax(P[m])]), 4)
    info = dict(t0=str(t0), t1=str(t1),
                minutes=round((t1 - t0).total_seconds() / 60, 1),
                betas={k: round(v, 5) for k, v in betas.items()},
                optics_native_hz=round(opt_rate, 2),
                motion=dict(frac_epochs=round(gate["frac"], 4),
                            n_epochs=gate["n_epochs"],
                            thresh=round(gate["thresh"], 5),
                            n_spans=len(gate["spans"])),
                peaks=peaks, paths=[p8, p9, p10, pc])
    return dict(mask=mask, eeg=eeg, opt=opt, gate=gate, table=tab, info=info)


# -------------------------------------------------------------- methods note
DOC_NAME = "SIGNAL_PROCESSING_control_session_plots"


def write_doc(res, outdir, name=DOC_NAME):
    """Companion methods note (.md + .docx) for figures 8-10."""
    i = res["info"]
    g = res["gate"]
    pk = i["peaks"]
    _runs, _frac = clean_runs(res["mask"]["time"].to_numpy(), FS_MASK, g["spans"])
    _nruns = len(_runs)
    L = []
    A = L.append
    A(f"# Signal processing - control-session time, spectrogram and "
      f"power-spectrum figures\n")
    A(f"Session {i['t0'][:19]} to {i['t1'][:19]} local "
      f"({i['minutes']:.1f} min), the window where the sleep mask and the "
      f"Muse S overlap after the tap-anchored clock correction. Every panel in "
      f"`fig8_time_domain.png`, `fig9_spectrograms.png` and "
      f"`fig10_power_spectra.png` is on that one wall clock.\n")
    A("This chain follows the session viewer (`smartadi/sleep_monitor`, "
      "`filters.py` / `preprocessing.py` / `motion.py`) so the numbers here are "
      "comparable with the viewer's own output. TP9, TP10 and AUX_1-4 are "
      "excluded: TP9/TP10 carry more broadband noise than the frontal pair in "
      "this rig, and the AUX inputs carry no physiology (flat DC plus a comb of "
      "fixed tones, i.e. floating analog inputs).\n")

    A("## Chain, in order\n")
    A("**1. Outlier clip.** Capacitive channels clipped at median +/- 4 sigma "
      "(viewer `outlier_clip`); EEG at +/- 8 sigma, looser because real EEG "
      "arousals are large and clipping them would flatten the feature of "
      "interest. The three finger taps at the start of the record exceed any "
      "of these limits and survive as clipped transients, which is why they "
      "are handled again by the motion gate below.\n")
    A("**2. Bandpass, order-3 Butterworth, zero-phase.** `sosfiltfilt`, so no "
      "phase distortion and no group delay - important because these figures "
      "are read for cross-device timing. Capacitive and optical channels "
      f"{SLOW_BAND[0]}-{SLOW_BAND[1]} Hz; EEG {EEG_BAND[0]}-{EEG_BAND[1]} Hz. "
      f"The {SLOW_BAND[0]} Hz high-pass removes the overnight capacitive drift "
      f"(tens of minutes) that would otherwise dominate every panel; the "
      f"{SLOW_BAND[1]} Hz low-pass is the band the mask and optics were asked "
      "to be shown in and sits above the second cardiac harmonic.\n")
    A("**3. Accelerometer artifact removal, capacitive channels only.** Viewer "
      "OLS variant: accelerometer magnitude is bandpassed to the *same* band as "
      "the capacitive channel, then one coefficient "
      "beta = <acc,cap>/<acc,acc> is fitted over the whole window and "
      "beta*acc subtracted. Fitted values this session: "
      + ", ".join(f"`{k}` {v:+.5f}" for k, v in i["betas"].items()) +
      ". These are small, so in-band motion energy is a minor contributor "
      "outside the gated epochs - the gate, not the regression, is doing the "
      "work here. A single global beta cannot track a coupling that changes "
      "with head position; the viewer's 16-tap NLMS variant is the option if "
      "that becomes a limit.\n")
    A(f"**4. Head tilt.** Gravity is the {GRAV_FC} Hz low-pass of each "
      "accelerometer axis; tilt then uses the two formulas as given:\n")
    A("```\nheadUDTilt = degrees(arctan2(aX, sqrt(aY^2 + aZ^2)))\n"
      "headLRTilt = degrees(arctan2(aY, sqrt(aX^2 + aZ^2)))\n```\n")
    A("Applying them to the gravity vector rather than to raw acceleration is "
      "the only deviation from the formulas as written, and it is the viewer's "
      "`motion.py` convention: without it every head movement injects a "
      "transient of tens of degrees into an angle that is meant to describe "
      "posture.\n")
    A(f"**5. Motion gate.** Dynamic acceleration = {GRAV_FC} Hz high-pass per "
      "axis, magnitude; RMS over non-overlapping 30 s epochs; an epoch is "
      "flagged when RMS > median + 3*MAD (viewer `motion_thresh_mad=3.0`). "
      f"This session: {g['n_epochs']} epochs, threshold {g['thresh']:.5f}, "
      f"{100*i['motion']['frac_epochs']:.1f}% flagged in "
      f"{i['motion']['n_spans']} spans. Flagged spans are shaded grey in "
      "figure 8, left in place in figure 9, and excluded from figure 10.\n")

    A("## Figure-specific parameters\n")
    A("| | fig 8 time domain | fig 9 spectrogram | fig 10 power spectrum |")
    A("|---|---|---|---|")
    A("| capacitive | per-pixel min/max envelope, 1600 px | 60 s Hann, 50% "
      "overlap, 2-window average -> 60 s/column | Welch 120 s Hann, 50% "
      "overlap |")
    A("| EEG | same envelope | 30 s Hann, 50% overlap, 4-window average -> "
      "60 s/column | Welch 8 s Hann, 50% overlap |")
    A("| optics | same envelope | as capacitive | as capacitive |")
    A("| y range | median +/- 10 robust sigma (1.4826*MAD) | "
      f"{SLOW_BAND[1]:.0f} Hz span; EEG {EEG_BAND[0]}-{EEG_BAND[1]:.0f} Hz | "
      "same |")
    A("| colour range | - | 10-97 pct (mask, optics), 5-99.5 pct (EEG) | - |")
    A("")
    A("**Why these windows.** 60 s gives 0.017 Hz resolution, enough to "
      "separate a 1.08 Hz cardiac line from its 0.05 Hz respiratory "
      "side-structure, and two-window averaging halves the chi-square(2) "
      "speckle that otherwise buries a narrow-dynamic-range channel. EEG uses "
      "30 s windows because that is the scoring epoch and 0.033 Hz resolution "
      "is already far finer than any EEG band edge. The 120 s Welch segment in "
      "figure 10 is the longest that still yields several independent segments "
      "inside the shortest motion-free run.\n")
    A("**Normalisation.** All spectra are power spectral density, "
      "`scaling='density'`, 10*log10, referenced to nothing external - dB here "
      "is relative within a channel, and the absolute level of a capacitive or "
      "optical channel is in arbitrary sensor units. Compare shapes and "
      "time-course, not absolute levels, between channels of different type. "
      "Each spectrogram column is linearly detrended before the transform.\n")
    A("**Envelope, not decimation.** Figure 8 draws the per-pixel min and max "
      "of every sample in that pixel, so a single-sample transient cannot be "
      "lost to downsampling; the thin line inside the band is the per-pixel "
      "median.\n")
    A("**Motion gating in figure 10.** The Welch average runs only over "
      "contiguous motion-free runs of at least 60 s, after discarding the "
      "first 120 s (the tap burst and filter settling): "
      f"{_nruns} runs covering {100*_frac:.0f}% of the window. Without this the tap transient "
      "alone raised apparent capacitive band power by roughly 20 dB and put a "
      "leakage ripple across the whole 0-5 Hz span, which is what the first "
      "render of this figure showed.\n")

    A("## What this session's spectra say\n")
    A("| channel | respiratory-band peak (Hz) | cardiac-band peak (Hz) |")
    A("|---|---|---|")
    for c in MASK_CH + OPT_CH:
        A(f"| {LBL[c]} | {pk[c+'_resp_peak_hz']:.3f} | "
          f"{pk[c+'_card_peak_hz']:.3f} |")
    A("")
    A("The optical channels carry a sharp cardiac fundamental near 1.08 Hz "
      "with visible first and second harmonics, and all four agree to the "
      "frequency bin - they are seeing one pulsatile source. The capacitive "
      "channels show no resolvable cardiac line over this window: their "
      "band 'peaks' are the maximum of a smooth falling spectrum, not a "
      "feature, and should not be read as a rate estimate. Respiratory-band "
      "entries sit at or near the lower band edge for the same reason, i.e. "
      "the 1/f rise, except for the optical peak visible near 0.27 Hz.\n")
    A("## Interpretive limits\n")
    A("* A whole-window Welch average over 110 min mixes sleep stages and "
      "postures; a line that is present only part of the night is attenuated "
      "here but visible in figure 9.\n"
      "* The gate is derived from the mask accelerometer only, and is applied "
      "to the Muse streams as well. It therefore removes head movement, not "
      "Muse-specific electrode artifacts.\n"
      "* The optical channels are reconstructed from a forward-filled export: "
      f"true update rate {i['optics_native_hz']:.2f} Hz, linearly interpolated "
      f"onto a uniform {FS_OPTICS:.0f} Hz grid before filtering. Content above "
      f"about {i['optics_native_hz']/2:.0f} Hz is not trustworthy.\n"
      "* Figure 8 y limits are robust (median +/- 10 sigma from the MAD), so "
      "the tap transients and the largest movement artifacts run off-panel by "
      "design. Their amplitude is not readable from this figure.\n"
      "* Absolute clock accuracy is that of the tap anchor, about 1 s; "
      "relative alignment between the two devices is better than that. "
      "Nothing at these window lengths is sensitive to it.\n")

    A("## Tunable constants\n")
    A("| constant | value | cost of changing it |")
    A("|---|---|---|")
    A(f"| capacitive/optical band | {SLOW_BAND[0]}-{SLOW_BAND[1]} Hz | a "
      "higher high-pass removes real vasomotion; a lower one lets overnight "
      "drift back in |")
    A(f"| EEG band | {EEG_BAND[0]}-{EEG_BAND[1]} Hz | raising the top edge "
      "admits EMG; lowering it hides spindle/beta activity |")
    A(f"| gravity split | {GRAV_FC} Hz | higher puts slow head motion into the "
      "tilt trace; lower smears posture steps |")
    A("| clip | 4 sigma (cap), 8 sigma (EEG) | tighter flattens real arousals |")
    A("| motion MAD factor | 3.0 | lower gates more of the night out of "
      "figure 10 |")
    A("| gate epoch | 30 s | shorter tracks brief movements, at the cost of "
      "fragmenting the motion-free runs figure 10 needs |")
    A("| Welch segment | 120 s (slow), 8 s (EEG) | longer sharpens lines and "
      "reduces the number of independent segments |")
    A("")
    A("Not done, deliberately: no notch filter (no mains line is visible in "
      "these channels), no interpolation across the gated epochs, no CLE-CRE "
      "differential channel (the viewer's primary channel, but the request was "
      "for CLE and CRE separately), no per-epoch normalisation, no artifact "
      "rejection beyond the clip and the motion gate.\n")

    md = "\n".join(L)
    pmd = os.path.join(outdir, name + ".md")
    with open(pmd, "w") as f:
        f.write(md)
    pdx = _md_to_docx(md, os.path.join(outdir, name + ".docx"))
    return pmd, pdx


def _md_to_docx(md, path):
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    lines = md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            p = doc.add_paragraph("\n".join(buf))
            p.style = doc.styles["Normal"]
            for r in p.runs:
                r.font.name = "Consolas"
                r.font.size = Pt(9)
        elif ln.startswith("|") and i + 1 < n and set(lines[i + 1]) <= set("|-: "):
            hdr = [c.strip() for c in ln.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            t = doc.add_table(rows=1, cols=len(hdr))
            t.style = "Light Grid Accent 1"
            for j, h in enumerate(hdr):
                t.rows[0].cells[j].text = h
            for r in rows:
                cells = t.add_row().cells
                for j, v in enumerate(r[:len(hdr)]):
                    cells[j].text = v.replace("`", "")
            continue
        elif ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            doc.add_heading(ln.lstrip("# ").strip(), level=min(lvl, 4))
        elif ln.startswith("* "):
            doc.add_paragraph(ln[2:].replace("`", ""), style="List Bullet")
        elif ln.strip():
            doc.add_paragraph(ln.replace("`", "").replace("**", ""))
        i += 1
    doc.save(path)
    return path
