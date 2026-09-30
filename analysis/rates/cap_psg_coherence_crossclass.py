"""Coherence regrouped by PSG sensor class, and the matched-minus-unmatched margin.

Written in response to the reviewer mark on the coherence slide, which bracketed
Flow/Thorax/Abdomen apart from Pleth/ECG in the respiratory panel and Pleth/ECG
in the cardiac panel.  That bracket is a test the original figure never ran.

The original figure judged specificity as `target minus EEG`, on the stated
assumption that a scalp EEG electrode carries no respiratory or cardiac
mechanics.  It does -- respiration-linked movement, pulse artifact and
ballistocardiogram all reach the scalp -- and empirically EEG sits close to the
targets, so that margin is doing more work than it can bear.

The contrast used here needs no such assumption: within one band, compare the
capacitive channel's coherence with the sensor class that measures that
rhythm against its coherence with the class that does not.

    resp band:  median(Flow, Thorax, Abdomen)  -  median(Pleth, ECG)
    card band:  median(Pleth, ECG)             -  median(Flow, Thorax, Abdomen)

Both classes are contact sensors on the same body at the same time, so a
mask artifact that inflates one inflates the other.  Only a rhythm-specific
coupling moves the difference.

Reporting follows the project rule for n=6: the unit is the subject, not the
recording and not the epoch, and the headline number is how many subjects fall
on the same side.  A two-sided Wilcoxon over six subjects bottoms out at
p = 0.031, which is what 6/6 in one direction gives and nothing more, so the
sign count is reported first and the p-value only alongside it.

Reads   reports/rates/coupling/cap_psg_coherence.csv   (per session, per pair)
Writes  reports/rates/coupling/cap_psg_crossclass.csv
        writeup/figures/coupling/cap_psg_coherence_grouped.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
from scipy import stats               # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "rates" / "coupling" / "cap_psg_coherence.csv"
OUT = ROOT / "reports" / "rates" / "coupling"
FIG = ROOT / "writeup" / "figures" / "coupling"

CAP = ["CH", "CLE", "CRE", "CLE-CRE"]
RESP_SENS = ["Flow", "Thorax", "Abdomen"]
CARD_SENS = ["Pleth", "ECG"]
ORDER = RESP_SENS + CARD_SENS + ["EEG"]
CLASSES = {"resp": (RESP_SENS, CARD_SENS), "card": (CARD_SENS, RESP_SENS)}

COL = {"CH": "#1f77b4", "CLE": "#ff7f0e", "CRE": "#2ca02c", "CLE-CRE": "#d62728"}
BAND_TITLE = {"resp": "Respiratory band", "card": "Cardiac band"}
SHADE = "#eaf0f7"

# Source segmentation (cap_psg_coherence.py): 300 s blocks, 30 s Welch segments,
# 50% overlap -> (300-30)/15 + 1 = 19 segments.  Magnitude-squared coherence from
# N independent segments has expectation 1/N under the null of no coupling, so
# absolute values near this line carry no information.  Overlapping segments are
# not fully independent, which pushes the true floor somewhat ABOVE this line --
# it is drawn as a lower bound on the floor, not as the floor itself.
N_SEGMENTS = 19
NULL_FLOOR = 1.0 / N_SEGMENTS


def margins(d: pd.DataFrame) -> pd.DataFrame:
    """Per-subject matched-minus-unmatched margin, one row per band x channel."""
    rows = []
    for band, (match, unmatch) in CLASSES.items():
        for cap in CAP:
            s = d[(d.band == band) & (d.cap == cap)]
            piv = s.pivot_table(index=["subject", "session"], columns="psg",
                                values="coh_mean")
            diff = (piv[match].median(axis=1) - piv[unmatch].median(axis=1))
            diff = diff.reset_index().set_axis(["subject", "session", "d"], axis=1)
            # two nights per subject collapse to one value: six independent units
            subj = diff.groupby("subject")["d"].mean()
            rows.append({
                "band": band, "cap": cap,
                "matched_coh": float(piv[match].median(axis=1).median()),
                "unmatched_coh": float(piv[unmatch].median(axis=1).median()),
                "margin": float(subj.median()),
                "subjects_positive": int((subj > 0).sum()),
                "n_subjects": int(subj.size),
                "sessions_positive": int((diff.d > 0).sum()),
                "n_sessions": int(diff.d.size),
                "wilcoxon_p_n6": float(stats.wilcoxon(subj.values).pvalue),
                "per_subject": subj,
            })
    return pd.DataFrame(rows)


def figure(d: pd.DataFrame, m: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.2),
                           gridspec_kw={"height_ratios": [1.25, 1.0]})

    for j, band in enumerate(("resp", "card")):
        a = ax[0, j]
        match, _ = CLASSES[band]
        agg = (d[d.band == band].groupby(["cap", "psg"])["coh_mean"]
               .median().unstack().reindex(index=CAP, columns=ORDER))

        # shade the sensor class that measures this band's rhythm
        lo = ORDER.index(match[0]) - 0.5
        hi = ORDER.index(match[-1]) + 0.5
        a.axvspan(lo, hi, color=SHADE, lw=0, zorder=0)
        a.text((lo + hi) / 2, 0.055, "measures this rhythm", ha="center",
               va="bottom", transform=a.get_xaxis_transform(), fontsize=9,
               color="#3d5a80")

        # dividers between the three blocks
        for b in (len(RESP_SENS) - 0.5, len(RESP_SENS) + len(CARD_SENS) - 0.5):
            a.axvline(b, color="#c9ced6", lw=0.9, ls="--", zorder=1)
        a.text(len(ORDER) - 1, 0.055, "control", ha="center", va="bottom",
               transform=a.get_xaxis_transform(), fontsize=9, color="#8a8f98")

        for c in CAP:
            a.plot(range(len(ORDER)), agg.loc[c].values, "o-", color=COL[c],
                   label=c, ms=5, lw=1.6, zorder=3)
        # the level below which an absolute coherence value says nothing
        a.axhline(NULL_FLOOR, color="#b3282d", lw=1.1, ls=":", zorder=2)
        a.text(0.02, NULL_FLOOR, " null floor ≥ 1/N  (N = %d segments)" % N_SEGMENTS,
               transform=a.get_yaxis_transform(), va="bottom", ha="left",
               fontsize=8.5, color="#b3282d")

        a.set_xticks(range(len(ORDER)))
        a.set_xticklabels(ORDER, rotation=30, ha="right")
        a.set_ylim(0)                       # zero-based: the differences are small
        a.set_ylabel("median coherence with PSG channel")
        a.set_title("%s — coherence by sensor class" % BAND_TITLE[band],
                    loc="left", fontsize=11)
        if j == 0:
            a.legend(frameon=False, ncol=2, fontsize=9, loc="lower right")
        a.grid(alpha=0.25, axis="y")

        # lower row: per-subject margin, one dot per subject
        b_ax = ax[1, j]
        sub = m[m.band == band].set_index("cap")
        allv = np.concatenate([sub.loc[c, "per_subject"].values for c in CAP])
        span = allv.max() - min(allv.min(), 0.0)
        top = allv.max() + 0.30 * span          # headroom for the sign counts
        for i, c in enumerate(CAP):
            vals = sub.loc[c, "per_subject"].values
            b_ax.scatter(np.full(vals.size, i) + np.linspace(-.13, .13, vals.size),
                         vals, s=34, color=COL[c], alpha=0.85, zorder=3,
                         edgecolor="white", linewidth=0.6)
            b_ax.plot([i - .26, i + .26], [np.median(vals)] * 2, color=COL[c],
                      lw=2.4, zorder=4)
            b_ax.annotate("%d/%d" % (sub.loc[c, "subjects_positive"],
                                     sub.loc[c, "n_subjects"]),
                          (i, top), ha="center", va="top", fontsize=10,
                          color=COL[c], weight="bold")
        b_ax.set_ylim(min(allv.min(), 0.0) - 0.08 * span, top + 0.06 * span)
        b_ax.axhline(0, color="#444444", lw=1.0)
        b_ax.set_xticks(range(len(CAP)))
        b_ax.set_xticklabels(CAP)
        b_ax.set_ylabel("matched − unmatched\n(coherence)")
        b_ax.set_title("%s — margin per subject (above 0 = rhythm-specific)"
                       % BAND_TITLE[band], loc="left", fontsize=11)
        b_ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Capacitive–PSG coherence, grouped by what each PSG sensor "
                 "measures\nEach dot is one subject (two nights averaged); "
                 "the bar is the median of the six",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    print("wrote %s" % path)


def main():
    d = pd.read_csv(SRC)
    m = margins(d)
    FIG.mkdir(parents=True, exist_ok=True)
    figure(d, m, FIG / "cap_psg_coherence_grouped.png")

    out = m.drop(columns=["per_subject"])
    out.to_csv(OUT / "cap_psg_crossclass.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nMatched minus unmatched sensor class, subject as the unit\n")
    print(out.round(4).to_string(index=False))
    print("\nNote: with six subjects a two-sided Wilcoxon cannot go below "
          "p = 0.031, so read the sign count, not the p-value.")


if __name__ == "__main__":
    main()
