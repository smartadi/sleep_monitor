"""Regenerate the ridge stage summary from cached per-epoch features.

`band_ridge_analysis.py` computes the per-epoch ridge features from raw
recordings, which is expensive, and then summarises them. The summary carried a
tie-handling defect until 2026-08-17: per-subject direction was decided with
`'N3>' if a.median() > b.median() else 'N3<'`, so every tie -- and these are
small integer counts, so ties are the common case -- was counted as N3-lower.
That produced the "all six subjects" claims the manuscript reported.

This rebuilds the summary with the fixed `stage_summary()` from the cached
`band_ridge_epochs.parquet`, so the artifact matches the corrected manuscript
without a full recompute.

Run from the repo root:  python analysis/slow_wave/recompute_stage_summary.py
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from band_ridge_analysis import stage_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "reports" / "slow_wave" / "band_ridge_epochs.parquet"
CSV = ROOT / "reports" / "slow_wave" / "band_ridge_stage_summary.csv"


def main():
    epochs = pd.read_parquet(PARQUET)
    old = pd.read_csv(CSV) if CSV.exists() else None
    summary = stage_summary(epochs)
    summary.to_csv(CSV, index=False)

    cols = ["band", "feature", "n_subj_N3_dn", "n_subj_tied", "n_subj_N3_up",
            "n_subj_N3_dn_by_mean"]
    print(summary[cols].to_string(index=False))
    if old is not None:
        merged = summary.merge(old[["band", "feature", "n_subj_N3_dn"]],
                               on=["band", "feature"], suffixes=("", "_old"))
        changed = merged[merged.n_subj_N3_dn != merged.n_subj_N3_dn_old]
        print("\n%d of %d rows changed:" % (len(changed), len(merged)))
        for _, r in changed.iterrows():
            print("  %-5s %-18s %d/6 -> %d/6  (%d tied)"
                  % (r.band, r.feature, r.n_subj_N3_dn_old, r.n_subj_N3_dn,
                     r.n_subj_tied))
    print("\nwrote %s" % CSV)


if __name__ == "__main__":
    main()
