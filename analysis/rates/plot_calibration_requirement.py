"""Supplementary figure: rate error under three calibration strategies.

Reads the table produced by calibration_requirement.py and plots, per band, the
MAE achieved with (a) per-session k estimated over the whole night, (b) a fixed
population k, and (c) k estimated from only the first ten minutes of recording.

Outputs: analysis/rates/outputs/fig_calibration_requirement.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
STRATEGIES = [
    ("mae_persession_k", "per-session k\n(whole night)", "#2a78d6"),
    ("mae_population_k", "fixed population k", "#eb6834"),
    ("mae_first10min_k", "k from first 10 min", "#b4b2a9"),
]
BANDS = [("resp", "Respiratory", "MAE (br/min)"), ("card", "Cardiac", "MAE (BPM)")]


def main() -> None:
    df = pd.read_csv(OUT / "calibration_requirement.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (band, label, ylab) in zip(axes, BANDS):
        sub = df[df.band == band].reset_index(drop=True)
        x = np.arange(len(sub))
        for i, (col, name, color) in enumerate(STRATEGIES):
            ax.bar(x + (i - 1) * 0.27, sub[col], width=0.26, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.strategy, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(f"{label}", fontsize=11)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Rate error under three calibration strategies (per-session median)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration_requirement.png", dpi=200)
    print("wrote", OUT / "fig_calibration_requirement.png")


if __name__ == "__main__":
    main()
