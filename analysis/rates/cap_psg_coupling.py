"""How the capacitive channels relate to each PSG channel, simply.

The paper's within-night negative is mediated by a rate estimator. Before asking
whether a decoder can do better, this asks the simplest question there is: on a
common 30-second grid, does the band-limited energy of each capacitive channel
move with the band-limited energy of each PSG channel?

Two scales, both reported:

  whole-night   one value per recording, correlated across the twelve nights
  epoch         within a recording, then summarised across recordings

Two bands: respiratory 0.1-0.5 Hz and cardiac 0.5-3 Hz. The measure is the
Pearson correlation of per-epoch log RMS, which is amplitude coupling, plus the
Spearman for the same pair so a monotone-but-curved relation is not missed.

A nonlinear, non-instantaneous chain sits between what PSG measures (airflow,
circumference, cardiac potential) and what the mask measures (capacitance from
cranial and vascular displacement), so weak linear coupling is the expected
result rather than a failure. What this table is for is to say how weak, where
it is strongest, and whether anything survives at epoch scale.

Outputs
    reports/rates/coupling/cap_psg_epoch.parquet     per-epoch log RMS
    reports/rates/coupling/cap_psg_correlations.csv  the correlation table
    writeup/figures/coupling/cap_psg_matrix.png      heatmaps
    writeup/figures/coupling/cap_psg_scatter.png     the strongest pairs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from scipy import stats                 # noqa: E402

import sys                              # noqa: E402
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI  # noqa: E402
from sleep_monitor.filters import bandpass                                # noqa: E402
from sleep_monitor.loader import load_session                             # noqa: E402
from sleep_monitor.sessions import SESSION_META                           # noqa: E402

OUT = ROOT / "reports" / "rates" / "coupling"
FIG = ROOT / "writeup" / "figures" / "coupling"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

EPOCH_S = 30.0
CAP = ["CH", "CLE", "CRE", "CLE-CRE"]
PSG = ["Flow", "Thorax", "Abdomen", "Pleth", "ECG", "EEG"]
BANDS = {"resp": (RESP_LO, RESP_HI), "card": (CARD_LO, CARD_HI)}


def epoch_rms(x, lo, hi, n):
    """Log RMS of the band-limited signal in consecutive 30 s epochs."""
    b = bandpass(np.asarray(x, float), lo, hi, FS)
    win = int(EPOCH_S * FS)
    usable = (len(b) // win) * win
    b = b[:usable].reshape(-1, win)
    rms = np.sqrt((b ** 2).mean(axis=1))
    return np.log10(rms[:n] + 1e-12)


def build():
    rows = []
    for meta in SESSION_META:
        s = load_session(meta)
        cap = dict(s.cap)
        cap["CLE-CRE"] = cap["CLE"] - cap["CRE"]
        n = min(len(v) for v in list(cap.values()) + list(s.psg.values()))
        n_ep = int(n // (EPOCH_S * FS))
        rec = {"session": s.meta["label"], "subject": s.meta["subject"],
               "epoch": np.arange(n_ep)}
        for band, (lo, hi) in BANDS.items():
            for c in CAP:
                rec["%s_%s" % (c, band)] = epoch_rms(cap[c], lo, hi, n_ep)
            for p in PSG:
                if p in s.psg:
                    rec["%s_%s" % (p, band)] = epoch_rms(s.psg[p], lo, hi, n_ep)
        rows.append(pd.DataFrame(rec))
        print("  %s  %d epochs" % (s.meta["label"], n_ep))
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(OUT / "cap_psg_epoch.parquet", index=False)
    return d


def correlate(d):
    out = []
    for band in BANDS:
        for c in CAP:
            cc = "%s_%s" % (c, band)
            for p in PSG:
                pc = "%s_%s" % (p, band)
                if cc not in d or pc not in d:
                    continue
                # epoch scale: within a night, then summarised over nights
                per_night = []
                for _, g in d.groupby("session"):
                    a, b = g[cc].values, g[pc].values
                    m = np.isfinite(a) & np.isfinite(b)
                    if m.sum() < 50:
                        continue
                    per_night.append(stats.pearsonr(a[m], b[m])[0])
                per_night = np.asarray(per_night)

                # whole-night scale: one mean per recording, 12 points
                nightly = d.groupby("session")[[cc, pc]].mean().dropna()
                r_night, p_night = (stats.pearsonr(nightly[cc], nightly[pc])
                                    if len(nightly) > 4 else (np.nan, np.nan))
                rho_night = (stats.spearmanr(nightly[cc], nightly[pc])[0]
                             if len(nightly) > 4 else np.nan)

                out.append({
                    "band": band, "cap": c, "psg": p,
                    "epoch_r_median": float(np.median(per_night)),
                    "epoch_r_iqr_lo": float(np.percentile(per_night, 25)),
                    "epoch_r_iqr_hi": float(np.percentile(per_night, 75)),
                    "epoch_n_pos": int((per_night > 0).sum()),
                    "epoch_wilcoxon_p": float(stats.wilcoxon(per_night).pvalue),
                    "night_r": float(r_night), "night_p": float(p_night),
                    "night_rho": float(rho_night),
                })
    t = pd.DataFrame(out)
    t.to_csv(OUT / "cap_psg_correlations.csv", index=False)
    return t


def figures(d, t):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for i, band in enumerate(BANDS):
        for j, (col, title) in enumerate(
                [("epoch_r_median", "epoch-to-epoch (within night, median r)"),
                 ("night_r", "whole night (across 12 recordings, r)")]):
            m = (t[t.band == band].pivot(index="cap", columns="psg", values=col)
                 .reindex(index=CAP, columns=PSG))
            ax = axes[i, j]
            im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(PSG)))
            ax.set_xticklabels(PSG, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(len(CAP)))
            ax.set_yticklabels(CAP, fontsize=9)
            for y in range(m.shape[0]):
                for x in range(m.shape[1]):
                    v = m.values[y, x]
                    if np.isfinite(v):
                        ax.text(x, y, "%.2f" % v, ha="center", va="center",
                                fontsize=8,
                                color="white" if abs(v) > 0.55 else "black")
            ax.set_title("%s band — %s" % (band, title), fontsize=10, loc="left")
            fig.colorbar(im, ax=ax, fraction=0.046, label="r")
    fig.suptitle("Capacitive channels against PSG channels, band-limited log RMS",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "cap_psg_matrix.png", dpi=190)
    print("wrote %s" % (FIG / "cap_psg_matrix.png"))

    # scatter of the strongest pair in each band, at both scales
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, band in enumerate(BANDS):
        sub = t[t.band == band]
        best_ep = sub.loc[sub.epoch_r_median.abs().idxmax()]
        best_ni = sub.loc[sub.night_r.abs().idxmax()]
        for j, best in enumerate((best_ep, best_ni)):
            cc, pc = "%s_%s" % (best.cap, band), "%s_%s" % (best.psg, band)
            ax = axes[i, j]
            if j == 0:
                g = d[d.session == d.session.iloc[0]]
                ax.scatter(g[pc], g[cc], s=4, alpha=0.35, color="#2980B9")
                ax.set_title("%s band, one night (%s): %s vs %s, r = %.2f"
                             % (band, g.session.iloc[0], best.cap, best.psg,
                                best.epoch_r_median), fontsize=9, loc="left")
            else:
                nightly = d.groupby("session")[[cc, pc]].mean()
                ax.scatter(nightly[pc], nightly[cc], s=45, color="#C0392B")
                for lbl, row in nightly.iterrows():
                    ax.annotate(lbl, (row[pc], row[cc]), fontsize=7,
                                xytext=(3, 3), textcoords="offset points")
                ax.set_title("%s band, 12 recordings: %s vs %s, r = %.2f"
                             % (band, best.cap, best.psg, best.night_r),
                             fontsize=9, loc="left")
            ax.set_xlabel("%s  log RMS" % best.psg)
            ax.set_ylabel("%s  log RMS" % best.cap)
            ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "cap_psg_scatter.png", dpi=190)
    print("wrote %s" % (FIG / "cap_psg_scatter.png"))


def main():
    pq = OUT / "cap_psg_epoch.parquet"
    d = pd.read_parquet(pq) if pq.exists() else build()
    t = correlate(d)
    pd.set_option("display.width", 200)
    for band in BANDS:
        print("\n%s band" % band)
        print(t[t.band == band][
            ["cap", "psg", "epoch_r_median", "epoch_n_pos", "epoch_wilcoxon_p",
             "night_r", "night_p"]].round(3).to_string(index=False))
    figures(d, t)


if __name__ == "__main__":
    main()
