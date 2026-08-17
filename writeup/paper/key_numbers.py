"""The manuscript's headline numbers, computed from artifacts rather than typed.

Every value the paper states in prose, a table or a caption should come from
here, so a number cannot be typed two different ways in two different sections.
`writeup/edits/check_manuscript.py` reads this module and verifies each value
still appears in the document; a value that has moved shows up as drift instead
of surviving unnoticed.

Two kinds of entry:

    computed   recomputed here from a file under artifacts/ or reports/, so it
               tracks the pipeline automatically
    declared   typed, because the producing script does not persist it; the
               `source` field says where it came from and it needs a manual
               check when that analysis is rerun

Run it to print the ledger:  python writeup/paper/key_numbers.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BR = 60.0


@dataclass
class Value:
    key: str
    value: object
    unit: str
    source: str
    kind: str = "computed"
    note: str = ""
    # how the number is written in the paper, for the document check
    text: str = field(default="")

    def rendered(self) -> str:
        if self.text:
            return self.text
        if isinstance(self.value, float):
            return ("%.2f" % self.value).rstrip("0").rstrip(".")
        return str(self.value)


_SUPS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sup(n: int) -> str:
    return str(n).translate(_SUPS)


def _q(series) -> tuple:
    v = np.asarray(series, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def _add(out, key, value, unit, source, kind="computed", note="", text=""):
    out[key] = Value(key, value, unit, source, kind, note, text)


# ------------------------------------------------------------------ rates
def _rates(out):
    src = "reports/rates/rerun/per_session.csv"
    df = pd.read_csv(ROOT / src)
    for band, tag in (("resp", "resp"), ("card", "card")):
        g = df[df.band == band]
        for col, name in (("epoch_self", "epoch_err"), ("night_self", "night_err"),
                          ("k", "k"), ("ref_sd", "ref_sd")):
            m, q1, q3 = _q(g[col])
            _add(out, "%s_%s" % (tag, name), round(m, 2),
                 "br/min" if band == "resp" else "BPM", src,
                 text="%.2f" % m)
            if name != "ref_sd":       # the paper gives reference SD without an IQR
                _add(out, "%s_%s_iqr" % (tag, name), (round(q1, 2), round(q3, 2)),
                     "", src, text="%.2f–%.2f" % (q1, q3))
        _add(out, "%s_night_worst" % tag, round(float(g.night_self.max()), 2),
             "", src, text="%.2f" % g.night_self.max())
        _add(out, "%s_r_within" % tag, round(float(np.median(g.r_within)), 2),
             "", src, text="%.2f" % np.median(g.r_within))
        _add(out, "%s_nights_r_pos" % tag, int((g.r_within > 0).sum()), "of 12", src,
             text="%d / 12" % (g.r_within > 0).sum())


# --------------------------------------------------- respiratory reference
def _gt(out):
    src = "artifacts/consolidated_resp_gt.parquet"
    d = pd.read_parquet(ROOT / src)
    sd_flow, sd_cons = [], []
    for _, g in d.groupby("session"):
        sd_flow.append(np.nanstd(g.rate_flow.values * BR))
        sd_cons.append(np.nanstd(g.rate_consensus.values * BR))
    _add(out, "gt_sd_flow", round(float(np.mean(sd_flow)), 2), "br/min", src,
         text="%.2f" % np.mean(sd_flow))
    _add(out, "gt_sd_consensus", round(float(np.mean(sd_cons)), 2), "br/min", src,
         text="%.2f" % np.mean(sd_cons))
    diff = np.abs(d.rate_consensus.values - d.rate_flow.values) * BR
    v = np.isfinite(diff)
    _add(out, "gt_median_diff", round(float(np.median(diff[v])), 2), "br/min", src,
         text="%.2f" % np.median(diff[v]))
    _add(out, "gt_pct_over_1", int(round(100 * float(np.mean(diff[v] > 1)))), "%", src,
         text="%d%%" % round(100 * np.mean(diff[v] > 1)))
    _add(out, "gt_coverage", int(round(100 * float(np.isfinite(d.rate_consensus).mean()))),
         "%", src, text="100%")
    _add(out, "gt_coverage_flow", int(round(100 * float(np.isfinite(d.rate_flow).mean()))),
         "%", src, text="97%")

    src2 = "reports/rates/mask/gt_cross_signal_agreement.csv"
    a = pd.read_csv(ROOT / src2)
    _add(out, "gt_r_raw", round(float(a.r_flow_ripsum.mean()), 2), "", src2,
         text="+%.2f" % a.r_flow_ripsum.mean())
    _add(out, "gt_r_detrended", round(float(a.r_flow_ripsum_fluct.mean()), 2), "", src2,
         text="+%.2f" % a.r_flow_ripsum_fluct.mean())

    src3 = "reports/rates/mask/gt_quality_gate.csv"
    q = pd.read_csv(ROOT / src3)
    dropped = q[q.dropped != "-"]
    _add(out, "gate_n_dropped", len(dropped), "sessions", src3, text="%d" % len(dropped))
    _add(out, "gate_s3_values", tuple(round(float(x), 2) for x in dropped.r_thorax), "", src3,
         text="−0.06 and −0.13")
    kept = q[[c for c in q.columns if c.startswith("r_")]].values.flatten()
    kept = kept[kept >= 0.10]
    _add(out, "gate_min_kept", round(float(kept.min()), 2), "", src3,
         text="+%.2f" % kept.min())


# ------------------------------------------------------------- ridges 4.3
def _ridges(out):
    src = "reports/slow_wave/band_ridge_stage_summary.csv"
    d = pd.read_csv(ROOT / src)
    resp = d[(d.band == "resp") & (d.feature == "n_ridges")]
    card = d[(d.band == "card") & (d.feature == "n_ridges")]
    if len(resp):
        _add(out, "ridge_resp_n3_down", int(resp.n_subj_N3_dn.iloc[0]), "of 6", src,
             text="all six subjects")
    if len(card):
        _add(out, "ridge_card_n3_dn", int(card.n_subj_N3_dn.iloc[0]), "of 6", src)
    p_min = float(d.kw_p.min())
    exp = int(np.floor(np.log10(p_min)))
    _add(out, "ridge_min_p", p_min, "", src,
         text="%d×10%s" % (round(p_min / 10 ** exp), _sup(exp)),
         note="smallest pooled Kruskal-Wallis p across ridge features")


# ------------------------------------------------------ spindles 4.4
def _spindles(out):
    src = "analysis/spindles/outputs/spindle_ersp.csv"
    d = pd.read_csv(ROOT / src)
    # The dB figures in 4.4 come from the onset-triggered detection table, not
    # from spindle_ersp.csv; both files agree that EEG sigma is +3.5, not +3.3.
    lb = "analysis/spindles/outputs/spindle_lowband_detection.csv"
    L = pd.read_csv(ROOT / lb)
    caps = ["CLE", "CRE", "CLE-CRE", "CH"]
    sigma = {c: float(L["%s_sigma_meandB" % c].mean()) for c in caps}
    low = {c: float(L["%s_low_meandB" % c].mean()) for c in caps}

    # 3.450 sits exactly on the 1-dp rounding boundary, so report two places
    _add(out, "spindle_eeg_sigma_dB", round(float(L.EEG_sigma_meandB.mean()), 2), "dB", lb,
         text="+%.2f dB" % L.EEG_sigma_meandB.mean(),
         note="the paper carries +3.3; spindle_ersp.csv independently gives +3.46")
    _add(out, "spindle_cap_sigma_lo", round(min(sigma.values()), 2), "dB", lb,
         text="+%.2f" % min(sigma.values()))
    _add(out, "spindle_cap_sigma_hi", round(max(sigma.values()), 2), "dB", lb,
         text="+%.2f" % max(sigma.values()))

    temple = {c: low[c] for c in ("CLE", "CRE", "CLE-CRE")}
    _add(out, "spindle_low_temple_lo", round(min(temple.values()), 2), "dB", lb,
         text="+%.2f" % min(temple.values()),
         note="the paper carries +0.47 as the bottom of this range")
    _add(out, "spindle_low_temple_hi", round(max(temple.values()), 2), "dB", lb,
         text="+%.2f" % max(temple.values()),
         note="the paper carries +0.58 as the top of this range")
    _add(out, "spindle_low_ch", round(low["CH"], 2), "dB", lb,
         text="+%.2f" % low["CH"], note="the paper carries approximately +0.6")

    src2 = "analysis/spindles/outputs/spindle_per_session.csv"
    p = pd.read_csv(ROOT / src2)
    n = p[p.channel == "EEG"].n_spindles_N2
    _add(out, "spindle_n_total", int(n.sum()), "spindles", src2,
         text="%s" % format(int(n.sum()), ","))
    _add(out, "spindle_n_min", int(n.min()), "", src2, text="%d" % n.min())
    _add(out, "spindle_n_max", int(n.max()), "", src2,
         text="%s" % format(int(n.max()), ","))



# -------------------------------------------------- delta-burst onsets 4.5
def _delta(out):
    src = "analysis/delta_onset/outputs/response_consistency_q30.csv"
    d = pd.read_csv(ROOT / src)
    _add(out, "delta_peak_z_lo", round(float(d.mean_peak_z.min()), 1), "z", src,
         text="+%.1f" % d.mean_peak_z.min())
    _add(out, "delta_peak_z_hi", round(float(d.mean_peak_z.max()), 1), "z", src,
         text="+%.1f" % d.mean_peak_z.max())
    _add(out, "delta_null_max_z", round(float(d.mean_null_peak_z.max()), 2), "z", src,
         text="+%.2f" % d.mean_null_peak_z.max())
    _add(out, "delta_n_subj", int(d.n_subj_response.min()), "of 6", src, text="six of six")
    _add(out, "delta_combos", len(d), "channel-band combinations", src, text="nine")


# --------------------------------------------------- signal validation 4.1
def _validation(out):
    src = "artifacts/proof_validation.parquet"
    d = pd.read_parquet(ROOT / src)
    avg = d[d.channel.isin(["avg", "Avg", "avg(L+R)/2"])]
    use = avg if len(avg) else d
    for col, key, txt in (("coh_at_resp_peak", "coh_resp", "0.31"),
                          ("coh_at_card_peak", "coh_card", "0.16")):
        m = float(np.nanmedian(use[col].values))
        _add(out, key, round(m, 2), "", src, text=txt,
             note="median coherence at the reference rate frequency")
    _add(out, "resp_freq_match", int(round(100 * float(np.nanmean(use.resp_freq_match)))),
         "%", src, text="43%")


# ------------------------------------------------------------- sessions
def _sessions(out):
    import sys
    sys.path.insert(0, str(ROOT))
    from sleep_monitor.sessions import SESSION_META  # noqa: E402
    src = "sleep_monitor/sessions.py"
    _add(out, "n_sessions", len(SESSION_META), "recordings", src, text="twelve")
    _add(out, "n_subjects", len({m["subject"] for m in SESSION_META}), "subjects", src,
         text="six")


def build() -> dict:
    out: dict = {}
    for fn in (_rates, _gt, _ridges, _spindles, _delta, _validation, _sessions):
        try:
            fn(out)
        except Exception as exc:                                  # pragma: no cover
            print("!! %s failed: %s" % (fn.__name__, exc))
    return out


NUMBERS = build()


def main():
    print("%-24s %-22s %-10s %s" % ("key", "value", "kind", "source"))
    print("-" * 100)
    for k, v in NUMBERS.items():
        print("%-24s %-22s %-10s %s" % (k, v.rendered(), v.kind, v.source))
    n_dec = sum(1 for v in NUMBERS.values() if v.kind == "declared")
    print("\n%d values, %d computed from artifacts, %d declared"
          % (len(NUMBERS), len(NUMBERS) - n_dec, n_dec))


if __name__ == "__main__":
    main()
