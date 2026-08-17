"""Does the decoder transfer, and is any single feature enough?

rate_decoder.py shows that a model trained on other blocks of the same night
recovers within-night rate variation. That is within-night calibration, so it
demonstrates the information exists but not that a usable method does. Two
follow-ups decide whether there is a method here.

1. SINGLE FEATURES. If one interpretable statistic carried the signal, the paper
   could recommend an estimator rather than a model. Tested: in-band spectral
   centroid, argmax peak frequency, in-band median frequency, and ACF lag, each
   k-scaled exactly like the operational estimator.

2. TRANSFER. Trained on the subject's *other* night, and trained on *other
   subjects entirely* (leave-one-subject-out). A model that transfers across
   subjects needs no calibration on the person wearing the mask, which is the
   requirement every other result in this paper fails.

Outputs -> reports/rates/decoder/single_features.csv
           reports/rates/decoder/transfer.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rates" / "decoder"

FEATURES = {0: "spectral centroid", 1: "argmax peak frequency",
            7: "in-band median frequency", 10: "ACF lag"}
CHANNELS = ["CRE", "CLE", "CH", "CLE-CRE"]


def beats_null(pred, y, rng, n=200):
    r = float(np.corrcoef(pred, y)[0, 1])
    nulls = [np.corrcoef(np.roll(pred, int(rng.integers(1, len(pred)))), y)[0, 1]
             for _ in range(n)]
    return r, bool(r > np.percentile(nulls, 97.5))


def single_features(d, band):
    rows = []
    for fi, name in FEATURES.items():
        for ch in CHANNELS:
            col = "%s_%d" % (ch, fi)
            rs, maes, beats = [], [], 0
            rng = np.random.default_rng(0)
            for _, g in d.groupby("session"):
                g = g.sort_values("epoch")
                v = g[col].to_numpy(float)
                v = np.where(v > 0, 60.0 / np.maximum(v, 1e-6), np.nan) if fi == 10 else v * 60.0
                y = g.target.to_numpy(float)
                m = np.isfinite(v) & np.isfinite(y)
                if m.sum() < 100 or np.std(v[m]) < 1e-9:
                    continue
                k = np.median(np.clip(v[m] / y[m], 0.3, 5.0))
                est = v[m] / k
                r, b = beats_null(est, y[m], rng)
                rs.append(r), maes.append(np.mean(np.abs(est - y[m])))
                beats += b
            if rs:
                rows.append({"band": band, "feature": name, "channel": ch,
                             "median_r": float(np.median(rs)),
                             "beats_null": beats, "n": len(rs),
                             "median_mae": float(np.median(maes))})
    return rows


def transfer(d, band):
    fc = [c for c in d.columns if c not in ("session", "epoch", "target", "subject")]
    rows = []
    for mode in ("cross-night", "LOSO"):
        rs, maes, beats = [], [], 0
        rng = np.random.default_rng(0)
        for sess, g in d.groupby("session"):
            g = g.sort_values("epoch")
            subj = g.subject.iloc[0]
            tr = (d[(d.subject == subj) & (d.session != sess)] if mode == "cross-night"
                  else d[d.subject != subj])
            if len(tr) < 500 or len(g) < 100:
                continue
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                              learning_rate=0.05)
            m.fit(np.nan_to_num(tr[fc].to_numpy(float)), tr.target.to_numpy(float))
            p = m.predict(np.nan_to_num(g[fc].to_numpy(float)))
            y = g.target.to_numpy(float)
            if np.std(p) < 1e-9:
                continue
            r, b = beats_null(p, y, rng)
            rs.append(r), maes.append(np.mean(np.abs(p - y)))
            beats += b
        if rs:
            rows.append({"band": band, "mode": mode, "median_r": float(np.median(rs)),
                         "beats_null": beats, "n": len(rs),
                         "median_mae": float(np.median(maes)),
                         "wilcoxon_p": float(stats.wilcoxon(rs).pvalue)})
    return rows


def main():
    sf, tr = [], []
    for band in ("resp", "card"):
        d = pd.read_parquet(OUT / ("features_%s.parquet" % band))
        d["subject"] = d.session.str[:2]
        sf += single_features(d, band)
        tr += transfer(d, band)

    sfd, trd = pd.DataFrame(sf), pd.DataFrame(tr)
    sfd.to_csv(OUT / "single_features.csv", index=False)
    trd.to_csv(OUT / "transfer.csv", index=False)

    pd.set_option("display.width", 200)
    print("Single interpretable features, k-scaled like the operational estimator\n")
    for band, g in sfd.groupby("band"):
        best = g.loc[g.median_r.idxmax()]
        print("  %-5s best is %s on %s: r %+.3f, %d/%d nights beat the null"
              % (band, best.feature, best.channel, best.median_r,
                 best.beats_null, best.n))
    print("\nTransfer of the full feature model\n")
    print(trd.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
