"""Fold the decoding result into 3.5, 4.2 and 5.2.

The paper's central within-night negative was stated as a property of the mask.
It is a property of rate estimators. A model reading the shape of the epoch's
spectrum, rather than collapsing it to a scalar rate, follows respiratory
variation in a held-out subject with no calibration on that subject at all --
the only result in the study that beats a no-sensor baseline without same-night
fitting. Cardiac decoding transfers within a subject and not across subjects.

Everything is stated with its bound: r is about 0.25, the mean absolute error
still sits at the reference standard deviation, and no single feature carries
the effect, so this tracks the shape of the variation rather than measuring it.

Run from the repo root:  python writeup/edits/apply_decoder_section.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_DECODER_20260817.docx")

METHODS = (
    "Rate decoding. Every estimator above reduces an epoch to a single rate, which discards the "
    "shape of the spectrum it was read from. To test whether that discarded structure carries "
    "rate information, each epoch was also described by 56 features computed from the capacitive "
    "channels alone — for each of CH, CLE, CRE and CLE−CRE, the in-band spectral centroid, peak "
    "frequency, spread, entropy, flatness and peak-to-mean ratio, the in-band spectral quartiles, "
    "the autocorrelation peak and its lag, the log band power, and the envelope mean and "
    "coefficient of variation. A gradient-boosted regression tree was trained to predict the "
    "reference rate of the epoch from those features, and scored against the same circular-shift "
    "null used for the estimators. Cross-validation is blocked: five contiguous held-out blocks "
    "per recording, so no epoch is predicted from its immediate neighbours, which random folds "
    "would allow. Transfer was then tested by training on the subject's other night, and on the "
    "five other subjects entirely."
)

RESULT_HEAD = "4.2.2 What a decoder recovers that a rate estimator does not"

RESULT_A = (
    "The failure above belongs to the estimators rather than to the signal. Trained and tested "
    "within a night under blocked cross-validation, a regressor reading the 56 spectral-shape "
    "features recovers a within-night correlation of 0.245 for respiration, exceeding the "
    "circular-shift null in 10 of 12 recordings (Wilcoxon p = 0.0002), and 0.271 for the cardiac "
    "band in 7 of 12 (p = 0.002). The operational estimator on the same recordings reaches −0.03 "
    "and −0.08 and exceeds that null in 0 and 1 of 12. A linear model on the same features "
    "reaches only 0.14, so part of the mapping is nonlinear. No single feature accounts for it: "
    "the best individual statistic, k-scaled exactly like the operational estimator, is the "
    "autocorrelation lag at r = 0.039 with 1 of 12 recordings above the null, so there is no "
    "simpler estimator that we merely failed to try."
)

RESULT_B = (
    "The respiratory result survives the transfer that nothing else in this study survives. "
    "Trained on the subject's other night it reaches r = 0.281, above the null in 11 of 11 "
    "testable recordings; trained on the five other subjects and applied to a held-out subject "
    "it reaches r = 0.237, above the null in 11 of 12 (p = 0.0002). No calibration on the wearer "
    "is involved, which distinguishes it from every accuracy figure reported earlier in this "
    "section. Cardiac decoding behaves differently: it transfers across a subject's own two "
    "nights (r = 0.385, 9 of 11) and collapses across subjects (r = −0.023, 3 of 12, p = 0.85), "
    "so it is subject-specific in the same way the calibration factor k is subject-specific, and "
    "for what is plausibly the same reason — the capacitive pulse shape depends on individual "
    "vascular anatomy while respiratory coupling does not."
)

RESULT_C = (
    "Two bounds belong with these numbers. A correlation of about 0.25 accounts for roughly six "
    "percent of the within-night variance, so the decoder follows the shape of the variation "
    "rather than measuring it; and its absolute error is no better than before, a median 1.55 "
    "br/min against a within-night reference standard deviation of 1.56. What changes is the "
    "conclusion of the previous subsection, not the accuracy of the device: the epoch carries "
    "rate information that peak counting does not reach."
)

D52 = (
    "No rate estimator follows either rate as it changes during the night. Across the seven base "
    "estimators, four channels, multi-channel fusion, CWT-ridge and STFT–Viterbi trackers, with "
    "and without calibration and smoothing, no configuration produced a within-night correlation "
    "distinguishable from a circular-shift null in either band, and averaging to windows as long "
    "as ten minutes does not change that (§5.3). The information is nonetheless partly present. A "
    "regressor reading the shape of each epoch's spectrum rather than a single rate from it "
    "recovers a correlation of about 0.25 in both bands, and for respiration it does so on a "
    "held-out subject with no calibration on that subject at all (§4.2.2). The distinction that "
    "matters for deployment is therefore not between the mask and the reference but between two "
    "ways of reading the mask: an estimator that reports a rate per epoch cannot follow the "
    "variation, while a model that reports a prediction from the whole epoch partly can. Neither "
    "is yet accurate in absolute terms, and the cardiac version does not generalize beyond the "
    "individual, so the practical statement stands: this device reports a nightly average, and a "
    "usable within-night trace would need the decoding approach developed on a larger cohort."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ------------------------------------------------------------- 3.5 methods
    at = d.find("The reported pipeline is therefore one estimator")
    d.body.insert(at + 1, d.para(METHODS))
    print("  3.5: decoding method added")

    # --------------------------------------------------------------- 4.2.2
    anchor = d.find("k is reproducible within a subject")
    for off, el in enumerate([d.para(RESULT_HEAD, "Heading3"),
                              d.para(RESULT_A), d.para(RESULT_B), d.para(RESULT_C)]):
        d.body.insert(anchor + 1 + off, el)
    print("  4.2.2 added after the k subsection")

    # ------------------------------------------------------------------- 5.2
    d.set_text("The mask does not follow either rate as it changes during the night", D52)
    print("  5.2: within-night subsection rewritten around the estimator/decoder distinction")

    # the subsection heading in 5.2 asserts the old conclusion
    d.set_text("Within-night rate variation",
               "Within-night rate variation, by rate estimation")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Decoding folded in (2026-08-17)\n\n"
        + "- 3.5 method, 4.2.2 result, 5.2 reframed from 'the mask does not follow' to 'no rate "
          "estimator follows, and the information is partly present'\n"
        + "- respiratory LOSO r = 0.237, 11/12 recordings above the null, no calibration on the "
          "wearer; cardiac transfers within subject only\n"
        + "- bounds stated: ~6% of variance, MAE 1.55 against reference SD 1.56, no single "
          "feature carries it\n", encoding="utf-8")


if __name__ == "__main__":
    main()
