"""Section 3.3: cite RIPSum, state the gate rule correctly, move its evidence to
a supplementary table, and correct three numbers in the validation paragraph.

1. The first mention of the respiratory inductance plethysmography sum (RIPSum)
   now carries reference 29 (Konno & Mead 1967), the source already cited two
   sentences later for the same quantity. It is inserted as a real superscript
   run; the plain-text "29" left in the following paragraph by an earlier edit
   pass -- the "random 29" on page 11 -- is made superscript at the same time.

2. The gate was described as excluding "sensors with net-negative correlation to
   the majority". The rule in scripts/build_consolidated_resp_gt.py:149-166 keeps
   a sensor when its correlation with the median of the other three reaches
   +0.10, and always keeps at least two. S3 Thorax happened to fall below zero,
   but the threshold is not zero.

3. Its evidence goes to the supplementary as Table S2, computed by
   analysis/rates/gt_quality_gate_table.py from the per-signal rates in
   artifacts/consolidated_resp_gt.parquet, so the table carries the statistic the
   gate actually thresholds.

4. The validation paragraph is simplified, and three of its numbers are brought
   back to what artifacts/consolidated_resp_gt.parquet currently yields:

       raw within-session r          +0.47  ->  +0.48   (mean over 12 nights)
       detrended fluctuation r       +0.27  ->  +0.28
       consensus rate-trace SD        1.61  ->   1.65 br/min
       median |consensus - Flow|      0.06  ->   0.08 br/min

   Unchanged and confirmed: Flow SD 2.26 br/min, 100% consensus epoch coverage,
   29% of epochs differing by more than 1 br/min (28.6%). Flow-alone coverage is
   97%, which the paragraph did not previously give.

Run from the repo root:  python writeup/edits/apply_gt_gate_supplement.py
"""

import copy
import shutil
import sys
from pathlib import Path

import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_GTGATE_20260817.docx")
GATE_CSV = Path("reports/rates/mask/gt_quality_gate.csv")

M33_CONSENSUS_GATE = (
    "Respiratory ground truth. Rather than relying on a single PSG respiratory sensor, we built a "
    "multi-signal consensus reference from four available channels: nasal airflow (Flow), thoracic "
    "effort (Thorax), abdominal effort (Abdomen), and respiratory inductance plethysmography sum "
    "(RIPSum). Peak detection was applied independently to each sensor, and a per-session quality "
    "gate scored every sensor by the correlation of its epoch-wise rate with the median rate of "
    "the other three across the whole recording. A sensor entered the consensus when that "
    "correlation reached +0.10, and at least two sensors were always kept. The gate removed one "
    "sensor in the cohort, the thoracic belt in both S3 nights, which showed paradoxical "
    "thoraco-abdominal motion; per-session values are in Table S2. Apnea epochs were labelled from "
    "the PSG apnea/hypopnea annotations rather than forced to a rate. The consensus rate was "
    "defined as the median across the remaining high-quality sensors at each epoch."
)

M33_VALIDATE = (
    "To validate the consensus we compared its two most physically independent members: nasal "
    "airflow, read by a pressure transducer, and RIPSum, the polarity-corrected sum of the "
    "thoracic and abdominal effort belts, which approximates thoraco-abdominal volume change. The "
    "two transduce respiration by different physical mechanisms, so where they agree on how the "
    "rate varies within a night, that variation is shared physiology rather than the estimation "
    "noise of one signal. They do agree: across the twelve nights the within-session correlation "
    "averages +0.48 on raw rates and +0.28 on detrended fluctuations. Taking the consensus also "
    "steadies the reference. The within-night standard deviation of the rate trace falls from 2.26 "
    "br/min for airflow alone to 1.65 br/min, and coverage is complete at 100% of epochs against "
    "97% for airflow alone. Consensus and airflow-only differ by a median of only 0.08 br/min, but "
    "29% of epochs differ by more than 1 br/min. That disagreement between two simultaneous "
    "reference sensors is a floor on the accuracy any external respiratory reference can claim."
)

TABLE_S2_CAPTION = (
    "Table S2. Respiratory-reference quality gate, per session. Each value is the correlation "
    "between that sensor's epoch-wise rate and the median rate of the other three across the whole "
    "recording — the statistic the gate thresholds. A sensor is kept when it reaches +0.10."
)

S2_WRITEUP = (
    "The gate removed one sensor in the cohort: the thoracic belt in both S3 nights, scoring −0.06 "
    "and −0.13 against the median of the other three. In those same two nights that belt is also "
    "anti-correlated with nasal airflow (r = −0.47 and −0.47), which is the signature of "
    "paradoxical thoraco-abdominal motion rather than a failed sensor. Every other sensor in every "
    "other night scored +0.23 or above and was kept, so eleven of the twelve nights use all four "
    "signals and the two S3 nights use three."
)


def superscript_citation(d, needle, before, marker):
    """Insert `marker` as a superscript run immediately after `before`.

    Splits only the run the anchor ends in, so formatting elsewhere in the
    paragraph survives.
    """
    el = d.body[d.find(needle)]
    runs = el.findall(W + "r")
    full = "".join(t.text or "" for r in runs for t in r.findall(W + "t"))
    if before not in full:
        raise KeyError("anchor not present: %r" % before[:60])
    at = full.index(before) + len(before)

    pos = 0
    for run in runs:
        t = run.find(W + "t")
        s = (t.text or "") if t is not None else ""
        lo, hi = pos, pos + len(s)
        pos = hi
        if t is None or not (lo < at <= hi):
            continue

        head, tail = s[: at - lo], s[at - lo:]
        t.text = head
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

        sup = copy.deepcopy(run)
        rpr = sup.find(W + "rPr")
        if rpr is None:
            rpr = etree.SubElement(sup, W + "rPr")
            sup.remove(rpr)
            sup.insert(0, rpr)
        for old in rpr.findall(W + "vertAlign"):
            rpr.remove(old)
        etree.SubElement(rpr, W + "vertAlign").set(W + "val", "superscript")
        sup.find(W + "t").text = marker

        at_idx = list(el).index(run)
        el.insert(at_idx + 1, sup)

        if tail:
            rest = copy.deepcopy(run)
            rest_rpr = rest.find(W + "rPr")
            if rest_rpr is not None:
                for bad in rest_rpr.findall(W + "vertAlign"):
                    rest_rpr.remove(bad)
            rest.find(W + "t").text = tail
            rest.find(W + "t").set(
                "{http://www.w3.org/XML/1998/namespace}space", "preserve")
            el.insert(at_idx + 2, rest)

        d.n_patch += 1
        return
    raise RuntimeError("could not place the citation run")


def main():
    if not GATE_CSV.exists():
        raise SystemExit("missing %s -- run analysis/rates/gt_quality_gate_table.py first"
                         % GATE_CSV)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ---------------------------------------------- 1. gate rule + RIPSum citation
    d.set_text("Rather than relying on a single PSG respiratory sensor", M33_CONSENSUS_GATE)
    superscript_citation(
        d, "Rather than relying on a single PSG respiratory sensor",
        "respiratory inductance plethysmography sum (RIPSum)", "29")
    print("  3.3: gate rule corrected to the +0.10 threshold; reference 29 at the first "
          "RIPSum mention; evidence pointed at Table S2")

    # ------------------------------------------- 2. validation paragraph, page 11
    d.set_text("To validate this consensus", M33_VALIDATE)
    superscript_citation(d, "To validate the consensus we compared",
                         "approximates thoraco-abdominal volume change.", "29")
    print("  3.3: validation paragraph simplified, three numbers corrected, the stray "
          "plain-text 29 now a superscript citation")

    # -------------------------------------------------------- 3. supplementary S2
    gate = pd.read_csv(GATE_CSV)
    rows = [["Session", "Flow", "Thorax", "Abdomen", "RIPSum", "Kept", "Dropped"]]
    for _, r in gate.iterrows():
        rows.append([
            str(r["session"]),
            "%+.2f" % r["r_flow"], "%+.2f" % r["r_thorax"],
            "%+.2f" % r["r_abdomen"], "%+.2f" % r["r_ripsum"],
            "%d / 4" % r["n_kept"],
            str(r["dropped"]),
        ])

    tbl_s1 = d.body[next(i for i in range(d.find("Table S1. Per-session rate accuracy"),
                                          len(d.body))
                         if d.body[i].tag == W + "tbl")]

    tail = len(d.body) - 1
    while tail > 0 and not d.text_of(d.body[tail]).strip():
        tail -= 1

    block = [
        d.para(""),
        d.para("S2. Respiratory reference: the per-signal quality gate", "Heading2"),
        d.para(TABLE_S2_CAPTION),
        d.table(rows, tbl_s1),
        d.para(S2_WRITEUP),
    ]
    for off, el in enumerate(block):
        d.body.insert(tail + 1 + off, el)
    print("  supplementary: section S2 added with Table S2 (%d rows) and its writeup"
          % (len(rows) - 1))

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Respiratory reference: citation, gate evidence, corrected numbers (2026-08-17)\n\n"
        + "- 3.3: reference 29 (Konno & Mead) at the first RIPSum mention, as a superscript run; "
          "the stray plain-text 29 on page 11 made superscript too\n"
        + "- 3.3: gate rule corrected -- threshold is +0.10 against the median of the other three "
          "sensors, not 'net-negative'\n"
        + "- 3.3: validation paragraph simplified; r +0.47->+0.48, +0.27->+0.28, consensus SD "
          "1.61->1.65, median |consensus-Flow| 0.06->0.08; Flow-alone coverage 97% added\n"
        + "- supplementary S2 / Table S2: per-session gate statistic for all four signals\n",
        encoding="utf-8")
    print("appended to writeup/edits/WRAPUP_CHANGELOG.md")


if __name__ == "__main__":
    main()
