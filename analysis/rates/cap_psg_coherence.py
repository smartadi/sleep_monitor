"""Spectral coherence between every capacitive channel and every PSG channel.

Amplitude correlation (cap_psg_coupling.py) asks whether two signals get louder
together, which a shared driver satisfies without any shared rhythm. Coherence
asks whether they carry the *same oscillation* at the same frequency with a
stable phase, which is the question the ridge results raise: the mask clearly
contains rhythms at respiratory and cardiac frequencies, so is that rhythm the
one the PSG is measuring?

EEG is again the negative control. It sees whatever broadband disturbance the
other sensors see, but it carries no respiratory or cardiac mechanics, so
coherence above EEG is specific coupling.

Windows are 5 minutes with 30-second Welch segments, giving ~19 segments per
estimate and 0.033 Hz resolution -- coarse enough to be stable, fine enough to
separate the respiratory band from DC. Every pair uses the same segmentation, so
the upward bias of a coherence estimate is identical across the matrix and the
comparison against EEG is fair.

Outputs
    reports/rates/coupling/cap_psg_coherence.csv
    writeup/figures/coupling/cap_psg_coherence.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
from scipy import signal, stats       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI  # noqa: E402
from sleep_monitor.loader import load_session                             # noqa: E402
from sleep_monitor.sessions import SESSION_META                           # noqa: E402

OUT = ROOT / "reports" / "rates" / "coupling"
FIG = ROOT / "writeup" / "figures" / "coupling"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

CAP = ["CH", "CLE", "CRE", "CLE-CRE"]
PSG = ["Flow", "Thorax", "Abdomen", "Pleth", "ECG", "EEG"]
BANDS = {"resp": (RESP_LO, RESP_HI), "card": (CARD_LO, CARD_HI)}

BLOCK_S = 300.0
NPERSEG = int(30 * FS)


def block_coherence(a, b):
    """Mean and peak magnitude-squared coherence per band, per 5-minute block."""
    n = int(BLOCK_S * FS)
    nb = len(a) // n
    out = {k: [] for k in BANDS}
    peak = {k: [] for k in BANDS}
    for i in range(nb):
        x, y = a[i * n:(i + 1) * n], b[i * n:(i + 1) * n]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        f, cxy = signal.coherence(x, y, fs=FS, nperseg=NPERSEG,
                                  noverlap=NPERSEG // 2)
        for band, (lo, hi) in BANDS.items():
            m = (f >= lo) & (f <= hi)
            if m.any():
                out[band].append(float(np.mean(cxy[m])))
                peak[band].append(float(np.max(cxy[m])))
    return out, peak


def main():
    rows = []
    for meta in SESSION_META:
        s = load_session(meta)
        cap = dict(s.cap)
        cap["CLE-CRE"] = cap["CLE"] - cap["CRE"]
        for c in CAP:
            a = np.asarray(cap[c], float)
            a = a - np.mean(a)
            for p in PSG:
                if p not in s.psg:
                    continue
                b = np.asarray(s.psg[p], float)
                b = b - np.mean(b)
                n = min(len(a), len(b))
                mean_c, peak_c = block_coherence(a[:n], b[:n])
                for band in BANDS:
                    if not mean_c[band]:
                        continue
                    rows.append({
                        "session": s.meta["label"], "subject": s.meta["subject"],
                        "cap": c, "psg": p, "band": band,
                        "coh_mean": float(np.median(mean_c[band])),
                        "coh_peak": float(np.median(peak_c[band])),
                        "n_blocks": len(mean_c[band]),
                    })
        print("  %s done" % s.meta["label"])

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cap_psg_coherence.csv", index=False)

    # per-session values summarised across the twelve recordings
    agg = (d.groupby(["band", "cap", "psg"])
           .agg(coh=("coh_mean", "median"), peak=("coh_peak", "median"))
           .reset_index())

    pd.set_option("display.width", 200)
    for band in BANDS:
        print("\n%s band — median in-band coherence across 12 recordings" % band)
        piv = agg[agg.band == band].pivot(index="cap", columns="psg", values="coh")
        print(piv.reindex(index=CAP, columns=PSG).round(3).to_string())

    print("\nSpecificity: target minus EEG, in-band coherence, per session")
    spec_rows = []
    for band, target in (("resp", "Flow"), ("card", "ECG")):
        for c in CAP:
            tg = d[(d.band == band) & (d.cap == c) & (d.psg == target)] \
                .set_index("session").coh_mean
            eg = d[(d.band == band) & (d.cap == c) & (d.psg == "EEG")] \
                .set_index("session").coh_mean
            common = tg.index.intersection(eg.index)
            diff = (tg[common] - eg[common]).values
            w = stats.wilcoxon(diff).pvalue if len(diff) > 5 else np.nan
            spec_rows.append({"band": band, "cap": c, "target": target,
                              "target_coh": float(np.median(tg[common])),
                              "eeg_coh": float(np.median(eg[common])),
                              "specificity": float(np.median(diff)),
                              "n_sessions_positive": int((diff > 0).sum()),
                              "wilcoxon_p": float(w)})
    sp = pd.DataFrame(spec_rows)
    print(sp.round(3).to_string(index=False))
    sp.to_csv(OUT / "cap_psg_coherence_specificity.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(PSG))
    for i, band in enumerate(BANDS):
        piv = (agg[agg.band == band].pivot(index="cap", columns="psg", values="coh")
               .reindex(index=CAP, columns=PSG))
        for c in CAP:
            ax[i].plot(x, piv.loc[c].values, "o-", label=c, lw=1.5, ms=5)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(PSG, rotation=45, ha="right")
        ax[i].set_ylabel("median in-band coherence")
        ax[i].set_title("%s band" % band, loc="left", fontsize=10)
        ax[i].grid(alpha=0.3, lw=0.5)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Coherence, capacitive against PSG channels — EEG is the negative control",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "cap_psg_coherence.png", dpi=190)
    print("\nwrote %s" % (FIG / "cap_psg_coherence.png"))


if __name__ == "__main__":
    main()
