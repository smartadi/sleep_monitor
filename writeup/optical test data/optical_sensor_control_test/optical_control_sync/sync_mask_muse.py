"""
Sleep-mask <-> Muse S synchronisation for the optical_sensor_control_test session
(SM2_4_20260910_Giang.txt + mindMonitor_2026-09-11--02-07-30.csv).

The sleep mask writes only a device uptime counter (``timeMS``, ms since power-on)
and has no wall clock.  The Muse / Mind Monitor CSV carries phone wall-clock
timestamps.  The bridge between the two clocks is the physical tap marker: the
tester pressed the mask sensors three times at 02:10:03 and three more times at
02:10:40, which appear in the mask stream as saturating pulses on CH / CRE.

Usage
-----
    from sync_mask_muse import *
    sm    = load_sleepmask(SM_PATH)            # + real 'time' column once anchored
    taps  = detect_taps(sm)                    # 6 tap pulses, mask clock
    anch  = anchor_from_taps(taps, TAP_MARKERS)
    sm    = apply_clock(sm, anch)
    muse  = load_muse(MUSE_PATH)               # wall clock already present
    both  = resample_common(sm, muse, fs=10)   # aligned on real time

Nothing here writes to the data folder; callers choose the output directory.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_DATA_FOLDER = "optical_sensor_control_test"


def _find_data_dir() -> str:
    """Locate the raw-data folder without hard-coding a user path.

    Order: $SLEEP_OPTICAL_DIR, this file's own folder (if it holds the data),
    then the folder itself / its parent / cwd searched for
    ``optical_sensor_control_test``.  Mirrors the _find_root() convention used
    by first_pass/run_all_sessions.py so a folder move needs no edit here.
    """
    env = os.environ.get("SLEEP_OPTICAL_DIR")
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    up = os.path.dirname(here)
    cands = [here, os.path.join(here, _DATA_FOLDER),
             up, os.path.join(up, _DATA_FOLDER),
             os.path.join(os.getcwd(), _DATA_FOLDER), os.getcwd()]
    for c in cands:
        if os.path.isdir(c) and any(f.startswith("mindMonitor_") for f in os.listdir(c)):
            return c
    raise SystemExit(
        f"Could not find the raw-data folder '{_DATA_FOLDER}' (a directory "
        f"containing mindMonitor_*.csv). Set SLEEP_OPTICAL_DIR to it.")


DATA_DIR = _find_data_dir()
SM_PATH = os.path.join(DATA_DIR, "SM2_4_20260910_Giang.txt")
MUSE_PATH = os.path.join(DATA_DIR, "mindMonitor_2026-09-11--02-07-30.csv")

# Session date: the mask file is named 20260910 but the recording crosses midnight;
# the Muse file's own timestamps put the taps on 2026-09-11.
SESSION_DATE = "2026-09-11"

# Wall-clock times of the two triple-tap marker sequences, as reported by the
# tester.  Each entry is (clock string, number of taps in that burst).  These are
# the ONLY link between mask uptime and real time, so they are stated explicitly
# here rather than buried in a script.
TAP_MARKERS = [(f"{SESSION_DATE} 02:10:03", 3),
               (f"{SESSION_DATE} 02:10:40", 3)]

SM_CHANNELS = ["CH", "CLE", "CRE"]          # forehead, left eye, right eye capacitive
SM_ACC = ["aX", "aY", "aZ"]
SM_TILT = ["headUDTilt", "headLRTilt"]      # pitch (nod) and roll (side tilt), deg
SM_FS_NOMINAL = 1000.0 / 9.0                # 9 ms tick -> 111.11 Hz

MUSE_KEEP = (["TimeStamp"]
             + [f"RAW_{c}" for c in ("TP9", "AF7", "AF8", "TP10")]
             + [f"AUX_{i}" for i in range(1, 5)]
             + [f"Optics{i}" for i in range(1, 5)]
             + [f"Accelerometer_{c}" for c in "XYZ"]
             + [f"Gyro_{c}" for c in "XYZ"]
             + ["Heart_Rate", "HeadBandOn",
                "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10", "Elements"])


# ----------------------------------------------------------------- sleep mask

def load_sleepmask(path: str = SM_PATH, tilt: bool = True) -> pd.DataFrame:
    """Read the mask text export.  Returns columns timeMS, CH, CLE, CRE, aX..aZ
    plus ``t_dev`` = seconds of device uptime (timeMS/1000, not zeroed) and, with
    ``tilt=True``, the two head-tilt angles in degrees."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["t_dev"] = df["timeMS"].astype(np.int64) / 1000.0
    if tilt:
        df = add_head_tilt(df)
    return df


def add_head_tilt(df: pd.DataFrame) -> pd.DataFrame:
    """Head tilt angles (deg) from the mask accelerometer.

    headUDTilt = nodding / pitch (up-down), headLRTilt = side-to-side / roll.
    Each axis is referenced to the plane of the other two, so gravity gives an
    absolute angle whenever the head is still.
    """
    df = df.copy()
    df["headUDTilt"] = np.degrees(np.arctan2(df["aX"], np.sqrt(df["aY"]**2 + df["aZ"]**2)))
    df["headLRTilt"] = np.degrees(np.arctan2(df["aY"], np.sqrt(df["aX"]**2 + df["aZ"]**2)))
    return df


def sleepmask_timing(df: pd.DataFrame) -> dict:
    """Sampling-integrity summary: tick spacing, effective rate, dropouts."""
    dt = np.diff(df["timeMS"].values)
    return dict(n_samples=len(df),
                t_first_ms=int(df["timeMS"].iloc[0]),
                t_last_ms=int(df["timeMS"].iloc[-1]),
                span_s=float((df["timeMS"].iloc[-1] - df["timeMS"].iloc[0]) / 1000),
                dt_median_ms=float(np.median(dt)),
                dt_min_ms=int(dt.min()), dt_max_ms=int(dt.max()),
                fs_effective=float(1000.0 / np.median(dt)),
                n_gaps_over_50ms=int((dt > 50).sum()))


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.abs(x - med) / (1.4826 * mad + 1e-12)


def tap_score(df: pd.DataFrame) -> np.ndarray:
    """Per-sample tap evidence: max robust |z| across the capacitive channels.
    A finger press saturates CH negative and CRE positive, giving z of order 1e3
    against a night-long baseline, so no band-pass is needed."""
    return np.maximum.reduce([_robust_z(df[c].values) for c in SM_CHANNELS])


def detect_taps(df: pd.DataFrame, z_thresh: float = 50.0,
                min_sep_s: float = 0.5, burst_gap_s: float = 10.0) -> pd.DataFrame:
    """Locate individual tap pulses and group them into bursts.

    Returns one row per tap with t_dev (device seconds), burst index, and the
    channel excursions.  ``burst_gap_s`` separates the triple-tap sequences.
    """
    s = tap_score(df)
    t = df["t_dev"].values
    pk, _ = find_peaks(s, height=z_thresh, distance=int(min_sep_s * SM_FS_NOMINAL))
    if len(pk) == 0:
        return pd.DataFrame(columns=["burst", "tap", "t_dev", "z", "CH", "CLE", "CRE"])
    burst = np.zeros(len(pk), dtype=int)
    burst[1:] = np.cumsum(np.diff(t[pk]) > burst_gap_s)
    out = pd.DataFrame(dict(burst=burst, t_dev=t[pk], z=s[pk],
                            CH=df["CH"].values[pk], CLE=df["CLE"].values[pk],
                            CRE=df["CRE"].values[pk]))
    out["tap"] = out.groupby("burst").cumcount()
    return out[["burst", "tap", "t_dev", "z", "CH", "CLE", "CRE"]].reset_index(drop=True)


def anchor_from_taps(taps: pd.DataFrame, markers=TAP_MARKERS) -> dict:
    """Map mask device time onto wall clock using the tap bursts.

    The reported clock time is taken as the moment of the FIRST tap of each
    burst (that is when the tester started the sequence).  With two bursts only
    ~37 s apart, the reported times are quoted to the second, so a drift fit
    would be meaningless: we use a constant offset (mean of the per-burst
    offsets) and report the disagreement between them as the sync uncertainty.
    """
    bursts = sorted(taps["burst"].unique())
    if len(bursts) != len(markers):
        raise ValueError(f"found {len(bursts)} tap bursts but {len(markers)} "
                         f"marker times were supplied")
    offsets, rows = [], []
    for b, (clock, n_expected) in zip(bursts, markers):
        sub = taps[taps["burst"] == b]
        t0 = float(sub["t_dev"].iloc[0])
        wall = pd.Timestamp(clock)
        off = wall.value / 1e9 - t0                       # epoch seconds - device seconds
        offsets.append(off)
        rows.append(dict(burst=int(b), n_taps=len(sub), reported=clock,
                         first_tap_t_dev=t0,
                         last_tap_t_dev=float(sub["t_dev"].iloc[-1]),
                         intervals_s=list(np.round(np.diff(sub["t_dev"].values), 3)),
                         offset_s=off, n_taps_expected=n_expected))
    offsets = np.array(offsets)
    return dict(offset_s=float(offsets.mean()),
                offset_spread_s=float(offsets.max() - offsets.min()),
                per_burst=pd.DataFrame(rows),
                epoch0=pd.Timestamp(0, tz=None))


def apply_clock(df: pd.DataFrame, anchor: dict) -> pd.DataFrame:
    """Add a wall-clock ``time`` column to the mask frame (device clock assumed
    linear at its own rate; only a constant offset is applied)."""
    df = df.copy()
    df["time"] = pd.to_datetime((df["t_dev"].values + anchor["offset_s"]) * 1e9)
    return df


# ----------------------------------------------------------------------- Muse

def load_muse(path: str = MUSE_PATH, usecols=MUSE_KEEP) -> pd.DataFrame:
    """Read the Mind Monitor export.  Keeps the wall-clock ``time`` column and
    drops the two connect/disconnect event rows that carry no samples."""
    num = [c for c in usecols if c not in ("TimeStamp", "Elements")]
    df = pd.read_csv(path, usecols=usecols,
                     dtype={c: "float32" for c in num}, low_memory=False)
    df["time"] = pd.to_datetime(df["TimeStamp"], format="mixed")
    df = df.drop(columns=["TimeStamp"])
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    return df


def muse_events(df: pd.DataFrame) -> pd.DataFrame:
    e = df.loc[df["Elements"].notna(), ["time", "Elements"]].copy()
    return e.reset_index(drop=True)


def muse_timing(df: pd.DataFrame) -> dict:
    data = df[df["RAW_AF7"].notna()]
    span = (data["time"].iloc[-1] - data["time"].iloc[0]).total_seconds()
    d = data["time"].diff().dt.total_seconds().values[1:]
    return dict(n_rows=len(df), n_sample_rows=len(data),
                t_start=str(data["time"].iloc[0]), t_end=str(data["time"].iloc[-1]),
                span_s=span, fs_mean=len(data) / span,
                n_timestamp_backsteps=int((d < 0).sum()),
                frac_duplicate_timestamps=float((d == 0).mean()))


# --------------------------------------------------------------- common clock

def resample_common(sm: pd.DataFrame, muse: pd.DataFrame, fs: float = 10.0,
                    t_start=None, t_end=None) -> pd.DataFrame:
    """Put both devices on one regular wall-clock grid.

    Mask channels are averaged within each bin (they are 111 Hz and smooth);
    Muse EEG/optics are averaged too, and per-bin SD is added for the channels
    where amplitude, not level, is the signal.  Requires ``sm`` to carry the
    real ``time`` column from :func:`apply_clock`.
    """
    if "time" not in sm:
        raise ValueError("run apply_clock(sm, anchor) first")
    lo = pd.Timestamp(t_start) if t_start is not None else max(sm["time"].iloc[0], muse["time"].iloc[0])
    hi = pd.Timestamp(t_end) if t_end is not None else min(sm["time"].iloc[-1], muse["time"].iloc[-1])
    period = pd.Timedelta(seconds=1.0 / fs)

    def binned(df, cols, prefix, with_sd=()):
        d = df[(df["time"] >= lo) & (df["time"] <= hi)]
        gb = d.set_index("time")[list(cols)].resample(period)
        out = gb.mean()
        out.columns = [prefix + c for c in out.columns]
        if with_sd:
            sd = d.set_index("time")[list(with_sd)].resample(period).std()
            sd.columns = [prefix + c + "_sd" for c in sd.columns]
            out = out.join(sd)
        return out

    tilt = [c for c in SM_TILT if c in sm.columns]
    a = binned(sm, SM_CHANNELS + SM_ACC + tilt, "sm_", with_sd=SM_CHANNELS)
    mcols = [c for c in muse.columns if c.startswith(("RAW_", "AUX_", "Optics",
                                                      "Accelerometer_", "Gyro_"))]
    b = binned(muse, mcols + ["Heart_Rate"], "muse_",
               with_sd=[c for c in mcols if c.startswith("RAW_")])
    both = a.join(b, how="outer")
    both.index.name = "time"
    return both.reset_index()


# ------------------------------------------------------- independent sync check

def refine_anchor(anchor: dict, refinement: dict) -> dict:
    """Add the motion-event lag to the tap-derived offset.

    The tap anchor fixes the clock to about a second (the reported tap times are
    quoted to the second); the motion-event lag measures the residual.  Keeps
    the original value as ``offset_s_tap`` so the correction stays auditable.
    """
    if not refinement.get("converged"):
        return dict(anchor)
    out = dict(anchor)
    out["offset_s_tap"] = anchor["offset_s"]
    out["offset_s"] = anchor["offset_s"] + refinement["lag_median_s"]
    out["refinement_lag_s"] = refinement["lag_median_s"]
    out["refinement_se_s"] = refinement["lag_se_s"]
    out["refinement_n_events"] = refinement["n_used"]
    return out


def muse_accel_updates(path: str = MUSE_PATH) -> pd.DataFrame:
    """True accelerometer sample instants from the Mind Monitor export.

    Mind Monitor repeats the last accelerometer value on every EEG row, so the
    file looks like 256 Hz accelerometer data but 93% of consecutive values are
    identical.  The instants where the (X, Y, Z) triple actually CHANGES are the
    real sensor updates (~17 Hz, median spacing 55 ms) and carry real
    timestamps; using them instead of the repeated rows is what makes
    sub-second alignment possible.
    """
    cols = ["TimeStamp"] + [f"Accelerometer_{c}" for c in "XYZ"]
    a = pd.read_csv(path, usecols=cols).dropna()
    a["time"] = pd.to_datetime(a["TimeStamp"])
    V = a[[f"Accelerometer_{c}" for c in "XYZ"]].values.astype(float)
    keep = np.r_[True, np.any(np.diff(V, axis=0) != 0, axis=1)]
    out = a.loc[keep, ["time"] + [f"Accelerometer_{c}" for c in "XYZ"]]
    return out.sort_values("time").reset_index(drop=True)


def _jerk_envelope(t_s: np.ndarray, xyz: np.ndarray, grid: np.ndarray,
                   smooth_s: float = 0.06) -> np.ndarray:
    """|d(acceleration vector)/dt|, smoothed, sampled on ``grid`` (seconds).

    The derivative is what makes a movement onset sharp: a posture change is a
    slow ramp in tilt but a fast spike in jerk, and the spike is what localises
    the event in time.
    """
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    dt = np.diff(t_s)
    good = dt > 0
    j = np.zeros_like(d)
    j[good] = d[good] / dt[good]
    tm = 0.5 * (t_s[1:] + t_s[:-1])
    e = np.interp(grid, tm[good], j[good], left=0.0, right=0.0)
    n = max(1, int(round(smooth_s / (grid[1] - grid[0]))))
    k = np.ones(n) / n
    return np.convolve(e, k, mode="same")


def refine_sync_by_motion_events(sm: pd.DataFrame, upd: pd.DataFrame,
                                 n_events: int = 40, half_window_s: float = 4.0,
                                 max_lag_s: float = 3.0, grid_dt: float = 0.01,
                                 min_r: float = 0.5, min_sep_s: float = 20.0,
                                 smooth_s: float = 0.06,
                                 height_mult: float = 20.0) -> dict:
    """Refine the clock offset on individual large head movements.

    The tap anchor is quantised to the second and the whole-night motion
    correlation cannot beat ~1 s, but a single large movement has a fast jerk
    edge that both devices see.  For each of the largest well-separated mask
    movement events, the mask and Muse jerk envelopes are cross-correlated
    inside a short window; the per-event lags then give an offset estimate whose
    standard error falls as 1/sqrt(n_events).

    Sign convention: ``lag_s`` is the number of seconds to ADD to the mask clock
    so that mask motion lines up with Muse motion (refined_offset = offset +
    median lag).  Requires ``sm`` to already carry a real-time column.
    """
    lo = max(sm["time"].iloc[0], upd["time"].iloc[0])
    hi = min(sm["time"].iloc[-1], upd["time"].iloc[-1])
    t0 = lo

    sm_t = (sm["time"].values - np.datetime64(t0)).astype("timedelta64[us]").astype(float) / 1e6
    mu_t = (upd["time"].values - np.datetime64(t0)).astype("timedelta64[us]").astype(float) / 1e6
    T = (hi - lo).total_seconds()
    grid = np.arange(0.0, T, grid_dt)

    sm_xyz = sm[SM_ACC].values.astype(float)
    mu_xyz = upd[[f"Accelerometer_{c}" for c in "XYZ"]].values.astype(float)
    Em = _jerk_envelope(sm_t, sm_xyz, grid, smooth_s)
    Eu = _jerk_envelope(mu_t, mu_xyz, grid, smooth_s)

    # candidate events: largest well-separated mask jerk peaks, away from the edges
    from scipy.signal import find_peaks
    pk, props = find_peaks(Em, height=np.median(Em) * height_mult,
                           distance=int(min_sep_s / grid_dt))
    edge = (grid[pk] > half_window_s + max_lag_s) & (grid[pk] < T - half_window_s - max_lag_s)
    pk, h = pk[edge], props["peak_heights"][edge]
    order = np.argsort(h)[::-1][:n_events]
    pk, h = pk[order], h[order]

    nlag = int(max_lag_s / grid_dt)
    hw = int(half_window_s / grid_dt)
    rows = []
    for p, hh in zip(pk, h):
        lo_i, hi_i = p - hw, p + hw
        y = Eu[lo_i:hi_i]
        if len(y) < 2 * hw or not np.isfinite(y).all() or y.std() == 0:
            continue
        rs = np.empty(2 * nlag + 1)
        for k, s in enumerate(range(-nlag, nlag + 1)):
            x = Em[lo_i - s:hi_i - s]        # mask shifted LATER by +s*grid_dt
            rs[k] = np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan
        if not np.isfinite(rs).any():
            continue
        j = int(np.nanargmax(rs))
        lag = (j - nlag) * grid_dt
        # parabolic refinement on the correlation peak
        if 0 < j < len(rs) - 1 and np.isfinite(rs[j - 1: j + 2]).all():
            den = rs[j - 1] - 2 * rs[j] + rs[j + 1]
            if den != 0:
                lag += 0.5 * (rs[j - 1] - rs[j + 1]) / den * grid_dt
        far = np.abs(np.arange(-nlag, nlag + 1) * grid_dt - (j - nlag) * grid_dt) > 1.0
        rows.append(dict(time=str(t0 + pd.Timedelta(seconds=float(grid[p]))),
                         t_rel_s=float(grid[p]), mask_jerk_peak=float(hh),
                         lag_s=float(lag), r=float(rs[j]),
                         r_elsewhere_p95=float(np.nanpercentile(rs[far], 95))))

    ev = pd.DataFrame(rows)
    used = ev[(ev["r"] >= min_r) & (ev["r"] - ev["r_elsewhere_p95"] > 0.1)] if len(ev) else ev
    if len(used) < 3:
        return dict(events=ev, used=used, n_used=int(len(used)), converged=False,
                    grid=grid, Em=Em, Eu=Eu, t0=str(t0))
    lags = used["lag_s"].values
    med = float(np.median(lags))
    mad = float(np.median(np.abs(lags - med)))
    return dict(events=ev, used=used, n_used=int(len(used)), converged=True,
                lag_median_s=med, lag_mad_s=mad,
                lag_robust_sd_s=1.4826 * mad,
                lag_se_s=1.4826 * mad / np.sqrt(len(lags)),
                lag_iqr_s=float(np.percentile(lags, 75) - np.percentile(lags, 25)),
                r_median=float(np.median(used["r"])),
                grid=grid, Em=Em, Eu=Eu, t0=str(t0))


def motion_cross_correlation(sm: pd.DataFrame, muse: pd.DataFrame,
                             bin_s: float = 1.0, max_lag_s: float = 120.0) -> dict:
    """Verify the wall-clock alignment on a signal both devices really share.

    Both units sit on the head, so body movement appears in both accelerometers.
    Per-bin SD of the accelerometer vector magnitude is log-transformed and
    z-scored, then cross-correlated.  A sync error of X s shows up as the peak
    sitting at lag X; the background level at distant lags gives the noise floor
    against which the peak is judged.  This is a CHECK on the tap anchor, not a
    replacement for it -- Muse accelerometer rows are interpolated to the EEG
    row rate, so it cannot resolve sub-second offsets.
    """
    lo = max(sm["time"].iloc[0], muse["time"].iloc[0])
    hi = min(sm["time"].iloc[-1], muse["time"].iloc[-1])

    def env(time, cols_xyz):
        mag = np.sqrt(sum(c.astype(float) ** 2 for c in cols_xyz))
        s = pd.Series(mag, index=pd.DatetimeIndex(time))
        s = s[(s.index >= lo) & (s.index <= hi)]
        return s.resample(pd.Timedelta(seconds=bin_s)).std()

    a = env(sm["time"].values, [sm[c].values for c in SM_ACC]).rename("mask")
    md = muse[muse["RAW_AF7"].notna()]
    b = env(md["time"].values,
            [md[f"Accelerometer_{c}"].values for c in "XYZ"]).rename("muse")
    E = pd.concat([a, b], axis=1).dropna()
    if len(E) < 10:
        raise ValueError("no overlapping motion bins")

    def zs(v, floor):
        v = np.log10(v + floor)
        return (v - v.mean()) / v.std()

    x, y = zs(E["mask"].values, 1e-4), zs(E["muse"].values, 1e-6)
    nlag = int(max_lag_s / bin_s)
    lags = np.arange(-nlag, nlag + 1)
    cc = np.array([np.corrcoef(x[max(0, l):len(x) + min(0, l)],
                               y[max(0, -l):len(y) + min(0, -l)])[0, 1] for l in lags])
    far = np.abs(lags) * bin_s > 30
    return dict(lag_s=lags * bin_s, r=cc,
                overlap_start=str(lo), overlap_end=str(hi),
                n_bins=int(len(E)), bin_s=bin_s,
                r_at_zero=float(cc[lags == 0][0]),
                r_peak=float(cc.max()), lag_peak_s=float(lags[np.argmax(cc)] * bin_s),
                r_background_p95=float(np.percentile(np.abs(cc[far]), 95)))
