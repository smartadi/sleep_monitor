"""Does the delta-burst capacitive response survive an arousal control?

Section 4.5 reports a capacitive power increase time-locked to EEG delta-burst
onset, peaking 4.5 to 8.2 s afterwards in 6 of 6 subjects. That latency is also
the latency of the cardiovascular response to a micro-arousal, and 4.5 says the
onset set is dominated by K-complexes, which frequently carry autonomic
activation. The finding therefore has an untested alternative explanation: the
mask may be registering arousals that co-occur with the bursts.

This reruns the published peri-onset analysis on the subset of onsets that are
clean of scored arousals, and compares it against the full set.

It deliberately reuses `lowband_precursor_check.process` rather than
reimplementing the peri-onset maths: the whole value of a control is that only
the onset list changes. Onsets within +-10 s of any scored arousal, cortical
('Classification Arousal') or autonomic, are dropped, and `process` is run
again with the reduced list monkey-patched into `load_onsets`.

Outputs -> analysis/delta_onset/outputs/arousal_control_q30.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "delta_onset"))

import lowband_precursor_check as L                                    # noqa: E402
from sleep_monitor.loader import (load_session, load_arousals,          # noqa: E402
                                  load_autonomic_arousals)
from sleep_monitor.sessions import SESSION_META                         # noqa: E402

OUT = ROOT / "analysis" / "delta_onset" / "outputs"
GUARD_S = 10.0
TAG = "q30"


def arousal_intervals(sess):
    """(start, end) in seconds for every scored arousal, cortical or autonomic."""
    spans = []
    for loader in (load_arousals, load_autonomic_arousals):
        try:
            ev = loader(sess)
        except Exception:
            ev = None
        if not ev or "start_hr" not in ev:
            continue
        st = np.asarray(ev["start_hr"], float) * 3600.0
        du = np.asarray(ev.get("duration_s", np.full(len(st), 3.0)), float)
        spans.extend(zip(st, st + du))
    return spans


def main():
    rows = []
    for idx, meta in enumerate(SESSION_META):
        label = meta["label"]
        sess = load_session(idx)
        spans = arousal_intervals(sess)

        # onsets in analysis-rate samples, exactly as the published code loads them
        onsets = L.load_onsets(label, TAG)
        if len(onsets) < 5:
            print("  %-5s only %d onsets, skipped" % (label, len(onsets)))
            continue
        t_on = onsets / L.ANALYSIS_FS
        keep = np.ones(len(t_on), bool)
        for a, b in spans:
            keep &= ~((t_on > a - GUARD_S) & (t_on < b + GUARD_S))
        clean = onsets[keep]
        print("  %-5s onsets %3d  arousals %4d  arousal-free onsets %3d (%.0f%% removed)"
              % (label, len(onsets), len(spans), len(clean),
                 100 * (1 - len(clean) / len(onsets))))

        for subset, ons in (("all", onsets), ("arousal_free", clean)):
            if len(ons) < 5:
                rows.append({"session": label, "subject": label.split("N")[0],
                             "subset": subset, "n_onsets": len(ons)})
                continue
            orig = L.load_onsets
            L.load_onsets = lambda lb, tg, _o=ons: _o          # inject
            try:
                res = L.process(idx, TAG, np.random.default_rng(0))
            finally:
                L.load_onsets = orig
            if res is None:
                continue
            tax = res["tax"]
            post = (tax >= 0) & (tax <= 15)
            for key in res["ca"]:
                prof, null = res["ca"][key], res["ca_null"][key]
                rows.append({
                    "session": label, "subject": label.split("N")[0],
                    "subset": subset, "n_onsets": len(ons),
                    "channel": key[0], "band": key[1],
                    "peak_z": float(np.nanmax(prof[post])),
                    "peak_lat_s": float(tax[post][np.nanargmax(prof[post])]),
                    "null_peak_z": float(np.nanmax(null[post])),
                })

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "arousal_control_q30.csv", index=False)

    pd.set_option("display.width", 200)
    valid = d.dropna(subset=["peak_z"])
    print("\nPer-subject mean post-onset peak (causal estimator), z\n")
    piv = (valid.groupby(["subject", "subset"])
           .agg(peak_z=("peak_z", "mean"), null=("null_peak_z", "mean"),
                onsets=("n_onsets", "max"), combos=("peak_z", "size"))
           .reset_index()
           .pivot(index="subject", columns="subset",
                  values=["peak_z", "null", "onsets"]).round(2))
    print(piv.to_string())

    print("\nChannel-band combinations beating their null, per subject:")
    for subset in ("all", "arousal_free"):
        s = valid[valid.subset == subset]
        if not len(s):
            continue
        hits = s.groupby("subject").apply(
            lambda g: "%d/%d" % ((g.peak_z > g.null_peak_z).sum(), len(g)),
            include_groups=False)
        print("  %-13s %s" % (subset, ", ".join("%s %s" % (k, v)
                                                for k, v in hits.items())))
    print("\nwrote %s" % (OUT / "arousal_control_q30.csv"))


if __name__ == "__main__":
    main()
