"""LayDown-only validation analysis for respiratory and cardiac rates.

Runs on all validation subjects, restricted to layDown phases, and computes:
  - CAP peak-count rates (raw and scaled)
  - GT peak-count rates (updated detector)
  - scaling factor k = n_cap / n_gt per subject/phase/band

Outputs (saved to notebooks/plots/validation/laydown/):
  - val_laydown_phase_results.csv
    - val_laydown_window_results.csv
  - val_laydown_subject_summary.csv
  - val_laydown_scaling_by_subject.csv
  - val_laydown_k_by_subject.png
  - val_laydown_error_raw_vs_scaled.png
  - val_laydown_gt_vs_cap_scatter.png
    - timeseries/val_laydown_timeseries_<subject>.png
    - signals/val_laydown_signals_<subject>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_validation import FS, MODE_ORDER, load_all
from sleep_monitor.config import CARD_HI, CARD_LO, RESP_HI, RESP_LO
from sleep_monitor.filters import bandpass
from sleep_monitor.preprocessing import remove_acc_artifact


# CAP channel-pairs for validation rig
RESP_CAP_PAIR = ("Cvl", "Cvr")
CARD_CAP_PAIR = ("Cbl", "Cbr")

# Updated GT detector style (aligned with fallback in ground_truth)
GT_PROM_FACTOR = 0.05
GT_DIST_SCALE = 0.6

# Lenient CAP detectors (band-specific)
CAP_RESP_PROM = 0.05
CAP_RESP_MIN_DIST_S = 0.4
CAP_CARD_PROM = 0.08
CAP_CARD_MIN_DIST_S = 0.25

MIN_PHASE_S = 5.0
LAYDOWN_TAG = "laydown"
WIN_SEC = 30.0
STEP_SEC = 5.0

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "plots"
    / "validation"
    / "laydown"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9})


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    sd = np.std(x)
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / sd


def _quality_filter_peaks(
    peak_indices: np.ndarray, fs: float, rate_lo_hz: float, rate_hi_hz: float
) -> np.ndarray:
    """Drop peaks that imply non-physiological inter-peak intervals."""
    if len(peak_indices) < 2:
        return peak_indices
    intervals = np.diff(peak_indices) / fs
    min_interval = 1.0 / rate_hi_hz
    max_interval = 1.0 / rate_lo_hz
    good = (intervals >= min_interval) & (intervals <= max_interval)
    keep = np.ones(len(peak_indices), dtype=bool)
    for i in range(len(good)):
        if not good[i]:
            keep[i + 1] = False
    return peak_indices[keep]


def _detect_gt_peaks_updated(
    x: np.ndarray, f_lo: float, f_hi: float, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass + fallback-style peak detection + physiological filtering."""
    bp = bandpass(x.astype(np.float64), f_lo, f_hi, fs)
    min_dist = max(1, int(fs / f_hi * GT_DIST_SCALE))
    prom = GT_PROM_FACTOR * np.std(bp)
    peaks, _ = find_peaks(bp, distance=min_dist, prominence=prom)
    peaks = _quality_filter_peaks(peaks, fs, f_lo, f_hi)
    return bp, peaks


def _detect_cap_peaks_lenient(
    x: np.ndarray, fs: float, prom_factor: float, min_dist_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed lenient CAP peak detector used for scaling analysis."""
    min_dist = max(1, int(round(min_dist_s * fs)))
    smooth_win = max(3, min_dist // 4)
    sm = np.convolve(x.astype(np.float64), np.ones(smooth_win) / smooth_win, mode="same")
    peaks, _ = find_peaks(sm, distance=min_dist, prominence=prom_factor * np.std(sm))
    return sm, peaks


def _select_gt_cardiac_channel(df_mode: pd.DataFrame) -> tuple[str | None, np.ndarray | None]:
    """Use Puls first, fallback to Pleth, based on finite sample coverage."""
    for ch in ("Puls", "Pleth"):
        if ch in df_mode.columns:
            vals = df_mode[ch].to_numpy(dtype=float)
            if np.isfinite(vals).mean() > 0.8:
                return ch, vals
    return None, None


def _safe_rate_bpm(n_peaks: int, duration_s: float) -> float:
    if duration_s <= 0:
        return np.nan
    return (n_peaks / duration_s) * 60.0


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def _windowed_peak_rates(
    cap_sig: np.ndarray,
    gt_sig: np.ndarray,
    f_lo: float,
    f_hi: float,
    fs: float,
    cap_prom: float,
    cap_min_dist_s: float,
    win_sec: float,
    step_sec: float,
) -> pd.DataFrame:
    """Compute window-wise CAP/GT peak-count rates and scaling factor k."""
    win_n = int(round(win_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))
    if len(cap_sig) < win_n or len(gt_sig) < win_n:
        return pd.DataFrame()

    rows = []
    for start in range(0, min(len(cap_sig), len(gt_sig)) - win_n + 1, step_n):
        seg_cap = cap_sig[start:start + win_n]
        seg_gt = gt_sig[start:start + win_n]

        _, pks_cap = _detect_cap_peaks_lenient(seg_cap, fs, cap_prom, cap_min_dist_s)
        _, pks_gt = _detect_gt_peaks_updated(seg_gt, f_lo, f_hi, fs)

        n_cap = len(pks_cap)
        n_gt = len(pks_gt)
        duration_s = win_n / fs
        cap_rate_raw_bpm = _safe_rate_bpm(n_cap, duration_s)
        gt_rate_bpm = _safe_rate_bpm(n_gt, duration_s)
        k = n_cap / n_gt if n_gt > 0 else np.nan
        cap_rate_scaled_bpm = (
            cap_rate_raw_bpm / k if np.isfinite(k) and k > 0 else np.nan
        )

        rows.append(
            {
                "t_center_s": (start + win_n / 2.0) / fs,
                "win_start_s": start / fs,
                "win_end_s": (start + win_n) / fs,
                "n_cap": n_cap,
                "n_gt": n_gt,
                "k": k,
                "gt_rate_bpm": gt_rate_bpm,
                "cap_rate_raw_bpm": cap_rate_raw_bpm,
                "cap_rate_scaled_bpm": cap_rate_scaled_bpm,
                "error_raw_bpm": cap_rate_raw_bpm - gt_rate_bpm,
                "error_scaled_bpm": (
                    cap_rate_scaled_bpm - gt_rate_bpm
                    if np.isfinite(cap_rate_scaled_bpm)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("Loading all validation subjects...")
    all_df = load_all()
    all_df["acc_mag"] = np.sqrt(all_df["aX"] ** 2 + all_df["aY"] ** 2 + all_df["aZ"] ** 2)

    lay_modes = [m for m in MODE_ORDER if LAYDOWN_TAG in str(m).lower()]
    if not lay_modes:
        raise RuntimeError("No layDown modes found in MODE_ORDER")
    print(f"Using layDown modes: {lay_modes}")

    subjects = sorted(all_df["subject"].unique())
    rows: list[dict] = []
    window_rows: list[dict] = []
    subject_signal_segments: dict[str, dict[str, list[dict]]] = {
        s: {"resp": [], "cardiac": []} for s in subjects
    }

    for subj in subjects:
        sdf = all_df[all_df["subject"] == subj].copy()
        phase_offset_s = 0.0
        for mode in lay_modes:
            mdf = sdf[sdf["experimentMode"] == mode].copy()
            if len(mdf) < int(MIN_PHASE_S * FS):
                continue

            duration_s = len(mdf) / FS

            # Respiratory: CAP z(Cvl)-z(Cvr) vs GT Thorax
            z_l = _zscore(mdf[RESP_CAP_PAIR[0]].to_numpy(dtype=float))
            z_r = _zscore(mdf[RESP_CAP_PAIR[1]].to_numpy(dtype=float))
            cap_resp_raw = z_l - z_r
            cap_resp = remove_acc_artifact(
                cap_resp_raw,
                mdf["acc_mag"].to_numpy(dtype=float),
                RESP_LO,
                RESP_HI,
                FS,
            )
            _, pks_cap_resp = _detect_cap_peaks_lenient(
                cap_resp, FS, CAP_RESP_PROM, CAP_RESP_MIN_DIST_S
            )

            thx = mdf["Thorax"].to_numpy(dtype=float)
            thx_bp, pks_gt_resp = _detect_gt_peaks_updated(thx, RESP_LO, RESP_HI, FS)

            n_cap_resp = len(pks_cap_resp)
            n_gt_resp = len(pks_gt_resp)
            k_resp = n_cap_resp / n_gt_resp if n_gt_resp > 0 else np.nan
            gt_resp_bpm = _safe_rate_bpm(n_gt_resp, duration_s)
            cap_resp_raw_bpm = _safe_rate_bpm(n_cap_resp, duration_s)
            cap_resp_scaled_bpm = (
                cap_resp_raw_bpm / k_resp if np.isfinite(k_resp) and k_resp > 0 else np.nan
            )

            rows.append(
                {
                    "subject": subj,
                    "mode": mode,
                    "band": "resp",
                    "duration_s": duration_s,
                    "gt_channel": "Thorax",
                    "cap_pair": f"{RESP_CAP_PAIR[0]}-{RESP_CAP_PAIR[1]}",
                    "n_cap": n_cap_resp,
                    "n_gt": n_gt_resp,
                    "k": k_resp,
                    "gt_rate_bpm": gt_resp_bpm,
                    "cap_rate_raw_bpm": cap_resp_raw_bpm,
                    "cap_rate_scaled_bpm": cap_resp_scaled_bpm,
                    "error_raw_bpm": cap_resp_raw_bpm - gt_resp_bpm,
                    "error_scaled_bpm": (
                        cap_resp_scaled_bpm - gt_resp_bpm
                        if np.isfinite(cap_resp_scaled_bpm)
                        else np.nan
                    ),
                }
            )

            subject_signal_segments[subj]["resp"].append(
                {
                    "mode": mode,
                    "t_start_s": phase_offset_s,
                    "t_end_s": phase_offset_s + duration_s,
                    "cap": _norm(cap_resp),
                    "gt": _norm(thx_bp),
                }
            )

            resp_w = _windowed_peak_rates(
                cap_resp,
                thx,
                RESP_LO,
                RESP_HI,
                FS,
                CAP_RESP_PROM,
                CAP_RESP_MIN_DIST_S,
                WIN_SEC,
                STEP_SEC,
            )
            if not resp_w.empty:
                for rec in resp_w.to_dict("records"):
                    rec.update(
                        {
                            "subject": subj,
                            "mode": mode,
                            "band": "resp",
                            "gt_channel": "Thorax",
                            "cap_pair": f"{RESP_CAP_PAIR[0]}-{RESP_CAP_PAIR[1]}",
                            "duration_s": duration_s,
                            "t_global_s": phase_offset_s + rec["t_center_s"],
                        }
                    )
                    window_rows.append(rec)

            # Cardiac: CAP z(Cbl)-z(Cbr) vs GT Puls/Pleth
            z_l = _zscore(mdf[CARD_CAP_PAIR[0]].to_numpy(dtype=float))
            z_r = _zscore(mdf[CARD_CAP_PAIR[1]].to_numpy(dtype=float))
            cap_card_raw = z_l - z_r
            cap_card = remove_acc_artifact(
                cap_card_raw,
                mdf["acc_mag"].to_numpy(dtype=float),
                CARD_LO,
                CARD_HI,
                FS,
            )
            _, pks_cap_card = _detect_cap_peaks_lenient(
                cap_card, FS, CAP_CARD_PROM, CAP_CARD_MIN_DIST_S
            )

            gt_card_ch, gt_card_sig = _select_gt_cardiac_channel(mdf)
            if gt_card_sig is None:
                continue
            gt_card_bp, pks_gt_card = _detect_gt_peaks_updated(gt_card_sig, CARD_LO, CARD_HI, FS)

            n_cap_card = len(pks_cap_card)
            n_gt_card = len(pks_gt_card)
            k_card = n_cap_card / n_gt_card if n_gt_card > 0 else np.nan
            gt_card_bpm = _safe_rate_bpm(n_gt_card, duration_s)
            cap_card_raw_bpm = _safe_rate_bpm(n_cap_card, duration_s)
            cap_card_scaled_bpm = (
                cap_card_raw_bpm / k_card if np.isfinite(k_card) and k_card > 0 else np.nan
            )

            rows.append(
                {
                    "subject": subj,
                    "mode": mode,
                    "band": "cardiac",
                    "duration_s": duration_s,
                    "gt_channel": gt_card_ch,
                    "cap_pair": f"{CARD_CAP_PAIR[0]}-{CARD_CAP_PAIR[1]}",
                    "n_cap": n_cap_card,
                    "n_gt": n_gt_card,
                    "k": k_card,
                    "gt_rate_bpm": gt_card_bpm,
                    "cap_rate_raw_bpm": cap_card_raw_bpm,
                    "cap_rate_scaled_bpm": cap_card_scaled_bpm,
                    "error_raw_bpm": cap_card_raw_bpm - gt_card_bpm,
                    "error_scaled_bpm": (
                        cap_card_scaled_bpm - gt_card_bpm
                        if np.isfinite(cap_card_scaled_bpm)
                        else np.nan
                    ),
                }
            )

            subject_signal_segments[subj]["cardiac"].append(
                {
                    "mode": mode,
                    "t_start_s": phase_offset_s,
                    "t_end_s": phase_offset_s + duration_s,
                    "cap": _norm(cap_card),
                    "gt": _norm(gt_card_bp),
                }
            )

            card_w = _windowed_peak_rates(
                cap_card,
                gt_card_sig,
                CARD_LO,
                CARD_HI,
                FS,
                CAP_CARD_PROM,
                CAP_CARD_MIN_DIST_S,
                WIN_SEC,
                STEP_SEC,
            )
            if not card_w.empty:
                for rec in card_w.to_dict("records"):
                    rec.update(
                        {
                            "subject": subj,
                            "mode": mode,
                            "band": "cardiac",
                            "gt_channel": gt_card_ch,
                            "cap_pair": f"{CARD_CAP_PAIR[0]}-{CARD_CAP_PAIR[1]}",
                            "duration_s": duration_s,
                            "t_global_s": phase_offset_s + rec["t_center_s"],
                        }
                    )
                    window_rows.append(rec)

            phase_offset_s += duration_s

        print(f"  {subj}: done")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No layDown results generated. Check mode labels and data coverage.")

    results["mode"] = pd.Categorical(results["mode"], categories=MODE_ORDER, ordered=True)
    results = results.sort_values(["subject", "mode", "band"]).reset_index(drop=True)

    phase_csv = OUT_DIR / "val_laydown_phase_results.csv"
    results.to_csv(phase_csv, index=False)

    window_df = pd.DataFrame(window_rows)
    window_csv = OUT_DIR / "val_laydown_window_results.csv"
    if not window_df.empty:
        window_df = window_df.sort_values(["subject", "mode", "band", "t_center_s"]).reset_index(drop=True)
        window_df.to_csv(window_csv, index=False)

    subj_summary = (
        results.groupby(["subject", "band"], observed=True)
        .agg(
            n_phases=("mode", "count"),
            k_median=("k", "median"),
            k_mean=("k", "mean"),
            k_std=("k", "std"),
            mae_raw_bpm=("error_raw_bpm", lambda s: float(np.nanmean(np.abs(s)))),
            mae_scaled_bpm=("error_scaled_bpm", lambda s: float(np.nanmean(np.abs(s)))),
            bias_raw_bpm=("error_raw_bpm", "mean"),
            bias_scaled_bpm=("error_scaled_bpm", "mean"),
        )
        .reset_index()
        .sort_values(["band", "subject"])
    )
    subj_csv = OUT_DIR / "val_laydown_subject_summary.csv"
    subj_summary.to_csv(subj_csv, index=False)

    k_by_subject = (
        results.groupby(["subject", "band"], observed=True)["k"]
        .median()
        .reset_index()
        .sort_values(["band", "subject"])
    )
    k_csv = OUT_DIR / "val_laydown_scaling_by_subject.csv"
    k_by_subject.to_csv(k_csv, index=False)

    # Plot 1: scaling factors by subject for resp/cardiac
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, band in enumerate(["resp", "cardiac"]):
        ax = axes[idx]
        sub = k_by_subject[k_by_subject["band"] == band]
        ax.bar(sub["subject"], sub["k"], color="steelblue", alpha=0.75)
        if len(sub):
            ax.axhline(sub["k"].median(), color="black", ls="--", alpha=0.6,
                       label=f"median={sub['k'].median():.2f}")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.4)
        ax.set_title(f"LayDown scaling factor k by subject ({band})")
        ax.set_ylabel("k = n_cap / n_gt")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_laydown_k_by_subject.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: raw vs scaled absolute error by band
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, band in enumerate(["resp", "cardiac"]):
        ax = axes[idx]
        sub = results[results["band"] == band]
        data = [
            sub["error_raw_bpm"].abs().dropna().values,
            sub["error_scaled_bpm"].abs().dropna().values,
        ]
        bp = ax.boxplot(data, tick_labels=["raw", "scaled"], patch_artist=True)
        colors = ["coral", "steelblue"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"LayDown |error| vs GT ({band})")
        ax.set_ylabel("|error| (BPM)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_laydown_error_raw_vs_scaled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: GT vs CAP rates scatter (raw/scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=False)
    for idx, band in enumerate(["resp", "cardiac"]):
        ax = axes[idx]
        sub = results[results["band"] == band]
        gt = sub["gt_rate_bpm"].to_numpy(dtype=float)
        raw = sub["cap_rate_raw_bpm"].to_numpy(dtype=float)
        scaled = sub["cap_rate_scaled_bpm"].to_numpy(dtype=float)
        ax.scatter(gt, raw, s=26, alpha=0.7, color="coral", label="CAP raw")
        ax.scatter(gt, scaled, s=26, alpha=0.7, color="steelblue", label="CAP scaled")
        if np.isfinite(gt).any():
            lo = np.nanmin(gt)
            hi = np.nanmax(gt)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6)
        ax.set_title(f"LayDown GT vs CAP rates ({band})")
        ax.set_xlabel("GT rate (BPM)")
        ax.set_ylabel("CAP rate (BPM)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_laydown_gt_vs_cap_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: per-subject time-series (windowed) for detected CAP vs GT and scaling
    ts_dir = OUT_DIR / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    if not window_df.empty:
        for subj in subjects:
            sw = window_df[window_df["subject"] == subj]
            if sw.empty:
                continue

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            for ax, band in zip(axes, ["resp", "cardiac"]):
                sb = sw[sw["band"] == band].sort_values("t_global_s")
                if sb.empty:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", color="gray")
                    ax.set_title(f"{subj} {band} (layDown)")
                    continue

                t_min = sb["t_global_s"].to_numpy(dtype=float) / 60.0
                ax.plot(t_min, sb["gt_rate_bpm"], "k-", lw=1.4, label="GT")
                ax.plot(t_min, sb["cap_rate_raw_bpm"], color="coral", lw=1.0, alpha=0.9,
                        label="CAP raw")
                ax.plot(t_min, sb["cap_rate_scaled_bpm"], color="steelblue", lw=1.1, alpha=0.95,
                        label="CAP scaled")
                ax.set_ylabel("Rate (BPM)")
                ax.set_title(f"{subj} {band} layDown: windowed detected CAP vs GT")
                ax.grid(True, alpha=0.25)

                ax2 = ax.twinx()
                ax2.plot(t_min, sb["k"], color="gray", ls="--", lw=1.0, alpha=0.8,
                         label="k")
                ax2.set_ylabel("k", color="gray")
                ax2.tick_params(axis="y", labelcolor="gray")

                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

            axes[-1].set_xlabel("Time in layDown phases (min)")
            fig.tight_layout()
            fig.savefig(ts_dir / f"val_laydown_timeseries_{subj}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Plot 5: per-subject waveform overlays (CAP vs GT) on layDown phases
    sig_dir = OUT_DIR / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    for subj in subjects:
        segs = subject_signal_segments.get(subj, {})
        if not segs:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for ax, band in zip(axes, ["resp", "cardiac"]):
            bsegs = segs.get(band, [])
            if not bsegs:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                ax.set_title(f"{subj} {band} layDown CAP vs GT signal")
                ax.grid(True, alpha=0.25)
                continue

            for seg in bsegs:
                t0 = seg["t_start_s"]
                cap = seg["cap"]
                gt = seg["gt"]
                t = t0 + np.arange(len(cap)) / FS
                ax.plot(t / 60.0, gt, color="black", lw=0.8, alpha=0.75,
                        label="GT" if seg is bsegs[0] else None)
                ax.plot(t / 60.0, cap, color="steelblue", lw=0.7, alpha=0.65,
                        label="CAP" if seg is bsegs[0] else None)
                ax.axvline(seg["t_end_s"] / 60.0, color="gray", ls="--", lw=0.6, alpha=0.4)
                ax.text((seg["t_start_s"] + 2.0) / 60.0, 2.1, str(seg["mode"]),
                        fontsize=7, color="gray", ha="left", va="top")

            ax.set_ylabel("Normalized amplitude")
            ax.set_title(f"{subj} {band} layDown CAP vs GT signal")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time in layDown phases (min)")
        fig.tight_layout()
        fig.savefig(sig_dir / f"val_laydown_signals_{subj}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\nWrote:")
    print(f"  {phase_csv}")
    if not window_df.empty:
        print(f"  {window_csv}")
    print(f"  {subj_csv}")
    print(f"  {k_csv}")
    print(f"  {OUT_DIR / 'val_laydown_k_by_subject.png'}")
    print(f"  {OUT_DIR / 'val_laydown_error_raw_vs_scaled.png'}")
    print(f"  {OUT_DIR / 'val_laydown_gt_vs_cap_scatter.png'}")
    print(f"  {OUT_DIR / 'timeseries'}")
    print(f"  {OUT_DIR / 'signals'}")

    print("\nScaling factor summary (median k):")
    for band in ["resp", "cardiac"]:
        kb = results.loc[results["band"] == band, "k"].dropna()
        if len(kb):
            print(
                f"  {band:7s}: median={kb.median():.3f}  mean={kb.mean():.3f}  "
                f"std={kb.std():.3f}  range=[{kb.min():.2f}, {kb.max():.2f}]"
            )


if __name__ == "__main__":
    main()
