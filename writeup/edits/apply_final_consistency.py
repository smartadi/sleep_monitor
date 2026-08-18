"""Final consistency pass: nine items from the last read-through.

The one that matters is the first. Section 4.1 acquired two separate treatments
of the same question -- an at-frequency coherence comparison with shift, reverse
and EEG controls, and a channel-by-sensor matrix of in-band coherence and
amplitude. They are different statistics, and at the same five-minute window
they reported different margins with opposite orderings between the bands. As
printed a reader gets two answers to one question, and they disagree precisely
about cardiac specificity.

The fix is not to reconcile them numerically -- deciding which statistic is
right would need work neither analysis has done -- but to stop them competing:
the at-frequency comparison keeps the coherence margin, and the matrix paragraph
reports only what it alone establishes (amplitude specificity, and the ordering
among respiratory sensors), dropping its restatement of the coherence margin and
the amplitude-versus-phase reading that rests on the contested comparison.

The other eight are small: a figure reference my own renumbering broke, two
supplementary writeups still on the superseded pipeline, a promise in 3.5 that
Table 4 does not keep, a duplicated apnea sentence, an unattributed fusion claim,
a British spelling, and a Bonferroni claim that 4.3 does not display.

Run from the repo root:  python writeup/edits/apply_final_consistency.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_FINAL_20260817.docx")

MATRIX = (
    "Running the same comparison channel by channel against every PSG sensor adds two things a "
    "single reference pair cannot show. The first is that amplitude coupling and rhythm coupling "
    "are not the same measurement. Amplitude coupling — the within-night correlation of "
    "band-limited log root-mean-square (RMS) amplitude on the 30-second grid — is modest and "
    "largely non-specific in the respiratory band, at a median r of 0.38 against nasal airflow "
    "and 0.34 against the EEG control, a margin of only 0.06 to 0.09 once motion epochs are "
    "removed. In the cardiac band it is specific: 0.35 against ECG against 0.13 for EEG, a margin "
    "of 0.18 to 0.20 that survives motion removal. Respiratory amplitude therefore carries little "
    "airflow-specific information even where the rhythm itself is coherent, which is a caution "
    "against reading capacitive amplitude as an effort or volume surrogate. The second is an "
    "ordering among the respiratory sensors: the mask couples to thoracic effort at least as "
    "strongly as to nasal airflow on every capacitive channel, with a peak in-band coherence of "
    "0.404 against thoracic effort and 0.372 against airflow on the differential channel, against "
    "0.243 for the EEG control. Per-channel matrices for both measures are given in Figures S11 "
    "and S12. These are in-band values over five-minute windows and are not the at-frequency "
    "statistic of the preceding paragraphs; with 19 segments per estimate their noise floor is "
    "0.053, so absolute values close to it should not be read on their own."
)

PATTERN = (
    "Taken with the persistent ridges at both frequencies (§4.3), these comparisons establish that "
    "the two rhythms are present in the capacitive signal and are the same rhythms the "
    "polysomnograph records. They do not establish how accurately either rate can be read from "
    "them, which §4.2 reports separately, and they do not identify the transformation between the "
    "quantity each instrument measures. What they do bound is the reading of amplitude: it "
    "transfers weakly and, in the respiratory band, largely non-specifically."
)

S3_WRITEUP = (
    "Respiration tolerates a fixed calibration better than the cardiac band does. Replacing the "
    "per-session k with a population constant costs 0.29 br/min of per-epoch error (1.79 → 2.08 "
    "br/min, Table 4) and 0.70 br/min at the level of a night. The cardiac band is more exposed: "
    "the same substitution costs 2.24 BPM per epoch (3.41 → 5.65) and 1.63 BPM per night (1.56 → "
    "3.19), and a ten-minute warm-up calibration is worse than the population prior because twenty "
    "epochs give a noisy and drifting estimate. A deployment would therefore use a population "
    "prior or whole-night self-calibration rather than a short calibration period — and, as "
    "Table 4 shows, neither reaches the accuracy of same-night calibration. This bounds the "
    "calibration limitation in §6."
)

S1_OPENING = (
    "Across the estimator and channel grid the Welch spectral peak shows the lowest respiratory "
    "error and the loose peak-counting estimator the lowest cardiac error, but the respiratory "
    "figure is not usable: at this window length that estimator is the constant described in §3.5 "
    "and its apparent accuracy is the accuracy of a constant. Among the estimators that vary, no "
    "channel was reliably better than the CLE−CRE differential for respiration."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ------------------------------------------------- 1. the competing blocks
    d.set_text("How much of the polysomnographic signal the mask reproduces", MATRIX)
    d.set_text("The pattern is a shared respiratory oscillation whose amplitude does not transfer",
               PATTERN)
    print("  4.1: matrix paragraph no longer restates the coherence margin; the contested "
          "amplitude-versus-phase reading is withdrawn")

    # --------------------------------------------------- 2. figure reference
    d.patch("With the physiological content established",
            "It is shown for a representative night in Figure 2,",
            "It is shown for a representative night in Figure 3,")
    print("  4.1: representative-night reference corrected to Figure 3")

    # ------------------------------------------------ 3-4. supplementary text
    d.set_text("Respiration needs little calibration", S3_WRITEUP)
    d.patch("The spectral estimator dominated the respiratory band",
            "The spectral estimator dominated the respiratory band on every channel and the loose "
            "peak-counting estimator dominated the cardiac band; no channel was reliably better "
            "than the CLE−CRE differential for respiration.", S1_OPENING)
    print("  Figures S1 and S3 writeups brought onto the current pipeline")

    # ----------------------------------------------------- 5. broken promise
    d.patch("SEC peak counts run systematically high",
            "Uncalibrated and k-scaled accuracy are therefore reported side by side, and section "
            "4.2 separates what k supplies from what the sensor supplies.",
            "Section 4.2 therefore reports the same errors under four calibrations — fitted on "
            "the night being scored, carried from the subject's other night, a single population "
            "constant, and none at all — so that what k supplies can be read separately from what "
            "the sensor supplies.")
    print("  3.5: the side-by-side promise replaced by what Table 4 actually does")

    # ------------------------------------------------ 6. duplicated sentence
    d.patch("Rather than relying on a single PSG respiratory sensor",
            " Apnea epochs were labeled from the PSG apnea/hypopnea annotations rather than "
            "forced to a rate.", "")
    print("  3.3: duplicated apnea sentence removed")

    # -------------------------------------------------- 7. unattributed claim
    d.patch("The reported pipeline is therefore one estimator",
            "Multi-channel fusion, both quality-weighted and agreement-gated, moved per-epoch "
            "error by less than 0.1 in either band, so we report the simpler single channel.",
            "An earlier multi-channel analysis, predating the estimator rerun reported here, "
            "found that quality-weighted and agreement-gated fusion moved per-epoch error by "
            "less than 0.1 in either band; we therefore report the simpler single channel.")
    print("  3.5: fusion claim attributed as it is in the Figure S1 writeup")

    # ------------------------------------------------------- 8. British form
    for t in d.root.iter(W + "t"):
        if t.text and "neighbours" in t.text:
            t.text = t.text.replace("neighbours", "neighbors")
    print("  3.5: neighbours -> neighbors")

    # ------------------------------------------------------- 9. Bonferroni
    d.patch("Rate accuracy is reported within-session",
            "Where a family of tests is reported together — the six ridge features within a band, "
            "and the capacitive features tested against age — p-values are Bonferroni corrected "
            "within that family.",
            "Where a family of tests is reported together — the six ridge features within a band, "
            "and the capacitive features tested against age — p-values are Bonferroni corrected "
            "within that family. The pooled-epoch p-values printed on Figure 6 are the "
            "uncorrected values, shown only to indicate scale, since the per-subject direction "
            "counts and not those p-values are the evidence.")
    print("  3.8: Bonferroni statement reconciled with what Figure 6 prints")

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
