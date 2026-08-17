"""Evidence for the respiratory-reference quality gate (supplementary Table S2).

The consensus respiratory reference is the per-epoch median of four PSG signals
(Flow, Thorax, Abdomen, RIPSum). Before taking that median,
scripts/build_consolidated_resp_gt.py gates each signal on its correlation with
the median of the other three, keeping a signal when that correlation is >= 0.10
and always keeping at least two.

This recomputes the gate statistic from the per-signal rates already stored in
artifacts/consolidated_resp_gt.parquet, so the table in the paper shows the
number the gate actually thresholds rather than a proxy.

Outputs -> reports/rates/mask/gt_quality_gate.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "artifacts" / "consolidated_resp_gt.parquet"
OUT = ROOT / "reports" / "rates" / "mask" / "gt_quality_gate.csv"

SIGNALS = ["Flow", "Thorax", "Abdomen", "RIPSum"]
COLS = {s: "rate_" + s.lower() for s in SIGNALS}
KEEP_THRESHOLD = 0.10


def wcorr(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 20 or np.std(a[v]) < 1e-9 or np.std(b[v]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[v], b[v])[0, 1])


def main():
    df = pd.read_parquet(PARQUET)
    rows = []
    for sess, g in df.groupby("session", sort=True):
        M = np.vstack([g[COLS[s]].to_numpy(float) for s in SIGNALS])
        corr = {}
        for j, s in enumerate(SIGNALS):
            med_others = np.nanmedian(np.delete(M, j, axis=0), axis=0)
            corr[s] = wcorr(M[j], med_others)
        kept = [s for s in SIGNALS
                if np.isfinite(corr[s]) and corr[s] >= KEEP_THRESHOLD]
        if len(kept) < 2:
            kept = sorted(SIGNALS,
                          key=lambda s: corr[s] if np.isfinite(corr[s]) else -1,
                          reverse=True)[:2]
        dropped = [s for s in SIGNALS if s not in kept]
        rows.append({
            "session": sess,
            **{"r_" + s.lower(): corr[s] for s in SIGNALS},
            "n_kept": len(kept),
            "dropped": ",".join(dropped) if dropped else "-",
        })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: "%+.2f" % v))
    print("\nwrote %s" % OUT)

    neg = out[[c for c in out.columns if c.startswith("r_")]].min().min()
    print("lowest gate correlation across all sessions and signals: %+.3f" % neg)
    print("sessions with a dropped signal: %s"
          % ", ".join(out.loc[out.dropped != "-", "session"]))


if __name__ == "__main__":
    main()
