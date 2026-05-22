"""
Load and analyze the ICP validation dataset (combinedDataAnalyses_041626).

Subjects lay down (not sleeping) in various postures/modes for sensor validation.
Cap channels here are Cvl, Cvr, Cbl, Cbr (different from overnight CH, CLE, CRE).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths & constants ─────────────────────────────────────────────────────────

VALIDATION_DIR = Path(
    r"C:\Users\adity\Documents\sleep monitor\combinedDataAnalyses_041626"
)

FS = 100.0  # Hz

SENSOR_COLS = ["Cvl", "Cvr", "Cbl", "Cbr"]
ACCEL_COLS = ["aX", "aY", "aZ"]
CAP_COLS = SENSOR_COLS + ACCEL_COLS
OTHER_COLS = ["EEG", "Pleth", "Puls", "Thorax", "MAP"]
ALL_SIG_COLS = CAP_COLS + OTHER_COLS

SUBJECT_FILES = sorted(VALIDATION_DIR.glob("S*.txt"))

MODE_ORDER = [
    "layDownRest", "sit90DegRest",
    "degree0", "degree30", "degree90",
    "turnLeft", "turnMiddle", "turnRight",
    "valsavaMild", "valsavaHigh",
]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_subject(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", parse_dates=["time"])
    df["subject"] = path.name.split(" - ")[0]  # e.g. "S0001"
    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    return df


def load_all() -> pd.DataFrame:
    frames = []
    for f in SUBJECT_FILES:
        df = load_subject(f)
        print(f"{f.name}: {len(df):,} rows, {df['t_sec'].iloc[-1]:.1f}s, modes={df['experimentMode'].nunique()}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Extraction helpers ────────────────────────────────────────────────────────

def extract_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract just the 4 capacitive sensor channels + metadata."""
    return df[["subject", "time", "t_sec", "experimentMode"] + SENSOR_COLS].copy()


def extract_cap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all cap data (sensors + accelerometer) + metadata."""
    return df[["subject", "time", "t_sec", "experimentMode"] + CAP_COLS].copy()


def extract_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all signal channels (cap + accel + EEG/Pleth/Puls/Thorax/MAP) + metadata."""
    return df[["subject", "time", "t_sec", "experimentMode"] + ALL_SIG_COLS].copy()


def extract_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    return df[df["experimentMode"] == mode].copy()


def per_mode_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject, per-mode summary statistics for sensor channels."""
    rows = []
    for (subj, mode), g in df.groupby(["subject", "experimentMode"]):
        row = {"subject": subj, "mode": mode, "n_samples": len(g), "duration_s": len(g) / FS}
        for ch in SENSOR_COLS:
            row[f"{ch}_mean"] = g[ch].mean()
            row[f"{ch}_std"] = g[ch].std()
            row[f"{ch}_range"] = g[ch].max() - g[ch].min()
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats["mode"] = pd.Categorical(stats["mode"], categories=MODE_ORDER, ordered=True)
    return stats.sort_values(["subject", "mode"]).reset_index(drop=True)


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_subject_overview(df: pd.DataFrame, subject: str):
    """Plot all sensor channels for one subject, colored by experiment mode."""
    sdf = df[df["subject"] == subject]
    fig, axes = plt.subplots(len(SENSOR_COLS), 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{subject} — Capacitive Sensor Channels", fontsize=14)

    mode_colors = plt.cm.tab10(np.linspace(0, 1, len(MODE_ORDER)))
    color_map = dict(zip(MODE_ORDER, mode_colors))

    for ax, ch in zip(axes, SENSOR_COLS):
        for mode in MODE_ORDER:
            mdf = sdf[sdf["experimentMode"] == mode]
            if len(mdf) == 0:
                continue
            ax.plot(mdf["t_sec"], mdf[ch], label=mode, color=color_map[mode], alpha=0.7, lw=0.5)
        ax.set_ylabel(ch)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=7, ncol=5)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_mode_comparison(stats: pd.DataFrame, metric="mean"):
    """Box plots comparing sensor values across modes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Sensor {metric} by Experiment Mode (across subjects)", fontsize=14)

    for ax, ch in zip(axes.flat, SENSOR_COLS):
        data_by_mode = []
        labels = []
        for mode in MODE_ORDER:
            vals = stats.loc[stats["mode"] == mode, f"{ch}_{metric}"]
            if len(vals):
                data_by_mode.append(vals.values)
                labels.append(mode)
        ax.boxplot(data_by_mode, tick_labels=labels)
        ax.set_title(ch)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading validation dataset...")
    all_df = load_all()
    print(f"\nTotal: {len(all_df):,} rows, {all_df['subject'].nunique()} subjects")

    sensor_df = extract_sensor_data(all_df)
    cap_df = extract_cap_data(all_df)
    print(f"\nSensor data shape: {sensor_df.shape}")
    print(f"Cap data shape:    {cap_df.shape}")

    stats = per_mode_stats(all_df)
    print(f"\n--- Per-mode statistics ---")
    print(stats.to_string(index=False))

    print("\nGenerating plots...")
    for subj in sorted(all_df["subject"].unique()):
        fig = plot_subject_overview(all_df, subj)
        out = VALIDATION_DIR / f"{subj}_sensor_overview.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.name}")

    fig = plot_mode_comparison(stats, "mean")
    out = VALIDATION_DIR / "mode_comparison_mean.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")

    fig = plot_mode_comparison(stats, "std")
    out = VALIDATION_DIR / "mode_comparison_std.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")

    print("\nDone.")
