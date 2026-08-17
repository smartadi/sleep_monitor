"""Analyses for the reviewer pass: apnea sensitivity, ridge robustness, burden association.

B3  rate accuracy with and without apnea/hypopnea epochs
M6  ridge stage contrast on motion-free and count-matched epochs
M7  the same contrast on channels other than the post-hoc CRE
P2  what the imbalance burden tracks
P1  the motion-canceller validation already on disk
P7  delta-onset counts per night

Outputs -> reports/rates/reviewer_pass/*.csv, and everything printed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rates" / "reviewer_pass"
OUT.mkdir(parents=True, exist_ok=True)

BR = 60.0
RESULTS = {}


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ B3 apnea
def b3_apnea():
    hdr("B3  rate accuracy with and without apnea epochs")
    a = pd.read_parquet(ROOT / "artifacts" / "rate_rerun_phase_a.parquet")
    g = pd.read_parquet(ROOT / "artifacts" / "consolidated_resp_gt.parquet")

    rows = []
    for band, unit in (("resp", "br/min"), ("card", "BPM")):
        d = a[(a.band == band) & (a.channel == "CRE")].dropna(subset=["gt_hz", "peaks_loose"])
        for sess, s in d.groupby("session"):
            gg = g[g.session == sess]
            # nearest 5 s reference epoch carries the apnea flag
            idx = np.searchsorted(gg.t_hr.values, s.t_hr.values)
            idx = np.clip(idx, 0, len(gg) - 1)
            apnea = gg.apnea.values[idx] > 0

            raw = s.peaks_loose.values * BR
            gt = s.gt_hz.values * BR
            k = np.median(np.clip(raw / gt, 0.3, 5.0))
            est = raw / k
            err = np.abs(est - gt)
            rows.append({
                "band": band, "session": sess, "unit": unit,
                "pct_apnea": 100 * apnea.mean(),
                "epoch_all": np.median(err),
                "epoch_clean": np.median(err[~apnea]) if (~apnea).sum() > 20 else np.nan,
                "night_all": abs(est.mean() - gt.mean()),
                "night_clean": abs(est[~apnea].mean() - gt[~apnea].mean())
                if (~apnea).sum() > 20 else np.nan,
            })
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "apnea_sensitivity.csv", index=False)
    for band, gg in d.groupby("band"):
        print("  %-5s epochs flagged apnea: %.1f%% median (max %.1f%% in %s)"
              % (band, gg.pct_apnea.median(), gg.pct_apnea.max(),
                 gg.loc[gg.pct_apnea.idxmax(), "session"]))
        print("        per-epoch error  all %.2f   apnea-free %.2f   (delta %+.2f)"
              % (gg.epoch_all.median(), gg.epoch_clean.median(),
                 gg.epoch_clean.median() - gg.epoch_all.median()))
        print("        night error      all %.2f   apnea-free %.2f   (delta %+.2f)"
              % (gg.night_all.median(), gg.night_clean.median(),
                 gg.night_clean.median() - gg.night_all.median()))
        w = stats.wilcoxon(gg.epoch_all, gg.epoch_clean).pvalue
        print("        Wilcoxon on the twelve per-epoch pairs: p = %.3f" % w)
        RESULTS["b3_%s" % band] = dict(
            all=gg.epoch_all.median(), clean=gg.epoch_clean.median(),
            night_all=gg.night_all.median(), night_clean=gg.night_clean.median(),
            pct_med=gg.pct_apnea.median(), pct_max=gg.pct_apnea.max(), p=w)


# ------------------------------------------------------- M6 / M7 ridge checks
def m6_m7_ridges():
    hdr("M6/M7  ridge stage contrast: motion, epoch count, and channel")
    d = pd.read_parquet(ROOT / "reports" / "slow_wave" / "band_ridge_epochs.parquet")
    d = d[d.stage_label.notna()]

    rows = []
    for (band, ch), g in d.groupby(["band", "channel"]):
        for label, sub in (("all epochs", g),
                           ("motion-free", g[~g.motion_masked.astype(bool)])):
            # Figure 7 reports "mean active ridges per epoch"; per-subject medians
            # of an integer count are all equal and cannot show the effect.
            per_subj = []
            for subj, s in sub.groupby("subject"):
                n3 = s[s.stage_label == "N3"].n_ridges
                oth = s[s.stage_label.isin(["N1", "N2", "REM", "Wake"])].n_ridges
                if len(n3) < 20 or len(oth) < 20:
                    continue
                per_subj.append(n3.mean() - oth.mean())
            if not per_subj:
                continue
            rows.append({"band": band, "channel": ch, "subset": label,
                         "n_subj": len(per_subj),
                         "n3_lower": int(sum(1 for v in per_subj if v < 0)),
                         "median_delta": float(np.median(per_subj))})

            # count-matched: draw as many non-N3 epochs as N3, per subject
            if label == "motion-free":
                rng = np.random.default_rng(0)
                matched = []
                for subj, s in sub.groupby("subject"):
                    n3 = s[s.stage_label == "N3"].n_ridges.values
                    oth = s[s.stage_label.isin(["N1", "N2", "REM", "Wake"])].n_ridges.values
                    if len(n3) < 20 or len(oth) < len(n3):
                        continue
                    draws = [n3.mean() - rng.choice(oth, len(n3), replace=False).mean()
                             for _ in range(200)]
                    matched.append(float(np.mean(draws)))
                if matched:
                    rows.append({"band": band, "channel": ch, "subset": "count-matched",
                                 "n_subj": len(matched),
                                 "n3_lower": int(sum(1 for v in matched if v < 0)),
                                 "median_delta": float(np.median(matched))})

    r = pd.DataFrame(rows).sort_values(["band", "channel", "subset"])
    r.to_csv(OUT / "ridge_robustness.csv", index=False)
    print(r.to_string(index=False))
    RESULTS["ridges"] = r


# ------------------------------------------------------------- P2 the burden
def p2_burden():
    hdr("P2  what the imbalance burden tracks")
    b = pd.read_csv(ROOT / "reports" / "mean_value" / "imbalance_burden.csv")
    e = pd.read_csv(ROOT / "reports" / "mean_value" / "imbalance_epochs.csv")
    motion = e.groupby("session").motion.mean().rename("motion_frac")
    b = b.merge(motion, on="session")

    try:
        pv = pd.read_parquet(ROOT / "artifacts" / "proof_validation.parquet")
        snr = (pv.groupby("session")[["mask_resp_snr", "mask_card_snr"]]
               .median().reset_index())
        b = b.merge(snr, on="session", how="left")
    except Exception as exc:
        print("  (no SNR join: %s)" % exc)

    for col in ("motion_frac", "mask_resp_snr", "mask_card_snr"):
        if col not in b:
            continue
        v = b[["integral_abs_fFh", col]].dropna()
        if len(v) < 6:
            continue
        rho, p = stats.spearmanr(v.integral_abs_fFh, v[col])
        print("  burden vs %-15s rho = %+.2f   p = %.3f   (n = %d)"
              % (col, rho, p, len(v)))
        RESULTS["burden_%s" % col] = (rho, p, len(v))
    b.to_csv(OUT / "burden_associations.csv", index=False)


# --------------------------------------------------------- P1 canceller check
def p1_canceller():
    hdr("P1  motion-canceller validation, already on disk")
    d = pd.read_csv(ROOT / "analysis" / "rates" / "outputs" / "motion_cancel_validation.csv")
    print("  columns:", d.columns.tolist())
    for band, g in d.groupby("band") if "band" in d else []:
        num = g.select_dtypes("number").median().round(3)
        print("  %-5s %s" % (band, dict(num)))
    RESULTS["p1"] = d


# ------------------------------------------------------------ P7 onset counts
def p7_onsets():
    hdr("P7  delta-burst onset counts per night (quiet_pre = 30 s)")
    d = pd.read_csv(ROOT / "analysis" / "delta_onset" / "outputs" / "delta_onsets_summary.csv")
    q = d[d.quiet_pre_s == 30]
    print(q[["session", "n_onsets", "n_N2", "n_N3", "nrem_hr"]].to_string(index=False))
    print("  total %d onsets, range %d to %d per night"
          % (q.n_onsets.sum(), q.n_onsets.min(), q.n_onsets.max()))
    RESULTS["p7"] = dict(total=int(q.n_onsets.sum()),
                         lo=int(q.n_onsets.min()), hi=int(q.n_onsets.max()))


if __name__ == "__main__":
    b3_apnea()
    m6_m7_ridges()
    p2_burden()
    p1_canceller()
    p7_onsets()
    print("\nwrote %s" % OUT)
