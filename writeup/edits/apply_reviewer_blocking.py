"""Reviewer pass, blocking items B1-B4.

B1  Table S1 rebuilt from the same source as Table 3
B2  held-out calibration reported in 4.2, as 3.5 promises
B3  apnea epochs declared, with the sensitivity result
B4  ridge stage associations rewritten around what survives the tie audit

Numbers come from reports/rates/rerun/per_session.csv,
reports/rates/rerun/heldout_table.csv,
reports/rates/reviewer_pass/apnea_sensitivity.csv and
reports/rates/reviewer_pass/ridge_direction_audit.csv.

Run from the repo root:  python writeup/edits/apply_reviewer_blocking.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_REVIEWER_B_20260817.docx")
ROOT = Path(".")

# ------------------------------------------------------------------------ B2
HELDOUT_CAPTION = (
    "Table 4. Rate accuracy under four calibrations, medians across the twelve recordings. "
    "Same-night k is the calibration used everywhere else in this paper and is fitted on the "
    "recording it scores. Cross-night k is taken from the same subject's other night, population "
    "k is one constant for the cohort, and the no-sensor row predicts the leave-one-subject-out "
    "cohort median rate with no capacitive input at all."
)

HELDOUT_TEXT = (
    "Because k is fitted on the recording it scores, the figures above are in-sample. Table 4 "
    "gives the same errors under calibrations that do not use the night being scored. The "
    "conclusion is uncomfortable and belongs in the Results rather than the supplement: at the "
    "level of a night, a population k gives a cardiac error of 3.19 BPM against 2.76 BPM for "
    "predicting the cohort median with no sensor at all, and per-epoch every calibration is "
    "worse than that no-sensor constant in both bands. Only same-night calibration beats the "
    "baseline, and a factor fitted on the night it scores is a fitting residual, not a "
    "measurement. What the mask demonstrates here is therefore relative accuracy within a "
    "calibrated recording; establishing standalone accuracy needs a calibration that transfers, "
    "which this cohort does not provide."
)

# ------------------------------------------------------------------------ B3
APNEA_METHODS = (
    "Apnea and hypopnea epochs were labelled from the PSG annotations rather than forced to a "
    "rate when the reference was built, and they are retained in the rate analysis: a median of "
    "12.5% of epochs per night, reaching 34.5% in S2N2. Excluding them changes the result very "
    "little — per-epoch error moves from 3.41 to 3.39 BPM in the cardiac band (Wilcoxon p = 0.90) "
    "and from 1.79 to 1.84 br/min in the respiratory band (p = 0.02, marginally worse without "
    "them) — so the errors reported below are not driven by disordered-breathing epochs."
)

# ------------------------------------------------------------------------ B4
RIDGE_STAGES = (
    "Restricting ridge detection to each band separately allows the stage contrast to be tested "
    "per subject (Figure 6). In the respiratory band (0.1–0.5 Hz) a persistent ridge was present "
    "in 96.7% of clean epochs, and its lowest frequency sat at a median of 0.23 Hz (≈14 "
    "breaths/min) in every stage, which confirms that the ridge tracks respiration. Against that "
    "background one stage association holds and the others do not. Total respiratory ridge power "
    "is lower in N3 than in other sleep in four of six subjects, with no subject tied and two in "
    "the opposite direction, and in five of six if subject means are compared instead of medians; "
    "the frequency spread is lower in three to four of six. The ridge counts, by contrast, carry "
    "no consistent direction: they are small integers whose per-subject medians are usually "
    "identical between N3 and other sleep, and comparing subject means splits three against "
    "three. We therefore report reduced respiratory ridge power in deep sleep as a weak and "
    "partially consistent association, and make no claim about ridge counts, ridge presence, or "
    "the cardiac band, where the per-subject directions are two to three of six on every feature. "
    "Pooled Kruskal–Wallis and Mann–Whitney tests over these contrasts return very small "
    "p-values, down to 9×10⁻²⁹, but they are computed on non-independent epochs and are reported "
    "in Figure 6 as descriptive statistics only; the per-subject counts above are the evidence, "
    "and they are weak. The association is consistent with the more regular, lower-effort "
    "breathing of deep sleep transduced mechanically by the sensor, and it is nowhere near strong "
    "enough to serve as a sleep-stage classifier."
)

RIDGE_DISCUSSION = (
    "One stage association in the band-restricted ridge features survives a per-subject test: "
    "total respiratory ridge power is lower during N3, in four of six subjects by median and five "
    "of six by mean. It is consistent with the more regular, lower-effort breathing of deep sleep "
    "and it is weak. The ridge-count features that an earlier analysis reported as consistent "
    "across all six subjects do not survive: their per-subject medians are tied in most subjects, "
    "and subject means split evenly. We therefore treat the ridge structure as a descriptive "
    "property of the recording with one weak stage association, not as a staging feature."
)

FIG6_CAPTION_FIX = (
    "Kruskal–Wallis p-values across stages are shown above each panel; they are computed on "
    "pooled, non-independent epochs and are descriptive only (§3.7). The per-subject direction "
    "counts in §4.3, not these p-values, are the evidence, and only total respiratory ridge power "
    "shows a consistent direction."
)


def build_table_s1(d):
    src = ROOT / "reports" / "rates" / "rerun" / "per_session.csv"
    p = pd.read_csv(src)
    rows = [["Session", "Resp k", "Resp night err (br/min)", "Resp epoch AE (br/min)",
             "Resp r", "Card k", "Card night err (BPM)", "Card epoch AE (BPM)", "Card r"]]
    for sess in sorted(p.session.unique()):
        r = p[(p.session == sess) & (p.band == "resp")].iloc[0]
        c = p[(p.session == sess) & (p.band == "card")].iloc[0]
        rows.append([sess,
                     "%.2f" % r.k, "%.2f" % r.night_self, "%.2f" % r.epoch_self,
                     "%+.2f" % r.r_within,
                     "%.2f" % c.k, "%.2f" % c.night_self, "%.2f" % c.epoch_self,
                     "%+.2f" % c.r_within])
    med = {}
    for band in ("resp", "card"):
        g = p[p.band == band]
        med[band] = (g.k.median(), g.night_self.median(), g.epoch_self.median(),
                     g.r_within.median())
    rows.append(["median",
                 "%.2f" % med["resp"][0], "%.2f" % med["resp"][1], "%.2f" % med["resp"][2],
                 "%+.2f" % med["resp"][3],
                 "%.2f" % med["card"][0], "%.2f" % med["card"][1], "%.2f" % med["card"][2],
                 "%+.2f" % med["card"][3]])
    return rows, med


def build_table_4():
    h = pd.read_csv(ROOT / "reports" / "rates" / "rerun" / "heldout_table.csv")
    rows = [["Calibration", "Resp night err", "Resp epoch AE", "Card night err", "Card epoch AE"]]
    labels = [("night_self", "epoch_self", "same-night k (reported)"),
              ("night_cross", "epoch_cross", "the subject's other night"),
              ("night_pop", "epoch_pop", "population k"),
              ("night_nosensor", "epoch_nosensor", "no sensor, cohort median")]
    r = h[h.band == "resp"].iloc[0]
    c = h[h.band == "card"].iloc[0]
    for nk, ek, label in labels:
        rows.append([label, str(r[nk]).split(" [")[0], str(r[ek]).split(" [")[0],
                     str(c[nk]).split(" [")[0], str(c[ek]).split(" [")[0]])
    return rows


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    # ------------------------------------------------------------------- B1
    rows, med = build_table_s1(d)
    cap_i = d.find("Table S1. Per-session rate accuracy")
    tbl_i = next(i for i in range(cap_i, len(body)) if body[i].tag == W + "tbl")
    template = body[tbl_i]
    body[tbl_i:tbl_i + 1] = [d.table(rows, template)]
    d.set_text("Table S1. Per-session rate accuracy",
               "Table S1. Per-session rate accuracy, from the operational pipeline of Table 3 "
               "(loose peak counting on CRE, per-session k, causal three-epoch median filter) and "
               "rebuilt from the same source, reports/rates/rerun/per_session.csv. Sessions are "
               "labelled SxNy for subject x, night y. \"Night err\" is the absolute difference "
               "between the mean estimated and mean reference rate over the recording; \"epoch "
               "AE\" is the median absolute error across its 30-second epochs; r is the "
               "within-night Pearson correlation. The final row gives the medians quoted in "
               "Table 3.")
    print("  B1 Table S1 rebuilt (resp epoch median %.2f, card %.2f) and caption corrected"
          % (med["resp"][2], med["card"][2]))

    # ------------------------------------------------------------------- B2
    t4 = build_table_4()
    anchor = d.find("The two bands reach that limit from different directions")
    tbl3_i = next(i for i in range(d.find("Table 3. Rate agreement"), len(body))
                  if body[i].tag == W + "tbl")
    new = [d.para(HELDOUT_TEXT), d.para(HELDOUT_CAPTION),
           d.table(t4, body[tbl3_i]), d.para("")]
    for off, el in enumerate(new):
        body.insert(anchor + 1 + off, el)
    print("  B2 held-out calibration paragraph and Table 4 added to 4.2")

    # ------------------------------------------------------------------- B3
    d.set_text("All rates were computed on a common grid", APNEA_METHODS + " All rates were "
               "computed on a common grid of non-overlapping 30-second windows, aligned to the "
               "30-second epochs of the PSG technologist's American Academy of Sleep Medicine "
               "(AASM) scoring. Sleep stages were taken from that scoring.")
    print("  B3 apnea retention and sensitivity stated in 3.3")

    # ------------------------------------------------------------------- B4
    d.set_text("Restricting ridge detection to each band separately", RIDGE_STAGES)
    d.set_text("Band-restricted ridge features show physiologically interpretable stage",
               RIDGE_DISCUSSION)
    d.patch("Figure 6. Band-restricted ridge structure by sleep stage",
            "Kruskal–Wallis p-values across stages are shown above each panel; they are "
            "computed on pooled, non-independent epochs and are descriptive only (§3.7).",
            FIG6_CAPTION_FIX)
    print("  B4 4.3 and 5.1 rewritten around the association that survives")

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
