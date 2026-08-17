"""Reviewer pass, major and polish items.

M1  the two within-night statements merged
M2  the two-sensor ceiling quoted one way everywhere: median [IQR]
M3  the tracker's search bands stated where the continuity claim is made
M4  S6N2's cardiac reference is photoplethysmography, noted where S6 is blamed
M6  ridge contrast robustness to motion and epoch count
M7  the same on channels other than the post-hoc CRE
M8  Bland-Altman limits declared as pooled and descriptive
M9  "screening" qualified by the calibration requirement
M10 STFT, IQR, SD, DC expanded (AASM done with B3)
M11 the Bonferroni sentence made specific
P1  the motion canceller validated in one sentence
P2  what the imbalance burden tracks
P3  cardiac error IQR with S6 excluded
P4  the peaks-per-beat exclusion rule
P6  k stability quantified by its consequence
P7  real delta-onset counts
P8  the reference floor added to the limitations

Run from the repo root:  python writeup/edits/apply_reviewer_major.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_REVIEWER_M_20260817.docx")


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ------------------------------------------------------------------- M1
    d.patch("Agreement on the level of a rate is not the same",
            "No estimator, channel or fusion strategy we tested produced a within-night "
            "correlation distinguishable from a circular-shift null (Figure S3).",
            "No estimator, channel or fusion strategy we tested produced a per-night "
            "distribution of correlations distinguishable from a circular-shift null "
            "(Figure S3); single nights do scatter above zero, and the paragraph after next "
            "puts a number on that.")
    print("  M1 within-night statements reconciled")

    # ------------------------------------------------------------------- M2
    d.patch("To validate the consensus we compared",
            "the within-session correlation averages +0.48 on raw rates and +0.28 on detrended "
            "fluctuations",
            "the within-session correlation has a median of 0.47 (interquartile range 0.33 to "
            "0.67) on raw rates and 0.26 on detrended fluctuations")
    d.patch("The negative is bounded by what we tested",
            "agree on within-night variation at r = 0.22 to 0.71",
            "agree on within-night variation at a median r of 0.47 (interquartile range 0.33 to "
            "0.67)")
    d.patch("An identical battery was applied to both bands",
            "respiratory sensors agree with each other at r = 0.47 (0.27 on detrended",
            "respiratory sensors agree with each other at a median r of 0.47 (0.26 on detrended")
    print("  M2 ceiling stated as median [IQR] in all three places")

    # ------------------------------------------------------------------- M3
    d.patch("Both physiological rhythms can be followed continuously",
            "The respiratory trace holds near 0.25–0.3 Hz and the cardiac trace near 0.9–1.2 Hz "
            "for the full recording, each with its own slow drift, and neither hops to a "
            "subharmonic.",
            "The respiratory trace holds near 0.25–0.3 Hz and the cardiac trace near 0.9–1.2 Hz "
            "for the full recording, each with its own slow drift, and neither hops to a "
            "subharmonic. Continuity is partly a property of the tracker: the Viterbi path is "
            "confined to the search bands of §3.4 (0.25–0.55 Hz and 0.85–1.45 Hz), and a "
            "reference rate outside those bands cannot be represented — which is the case for "
            "36.7% of respiratory epochs, whose true rate falls below the 15 breaths/min floor, "
            "and 14.9% of cardiac epochs. The claim here is that a continuous rhythm exists to "
            "be followed, not that the trace is an accurate rate; §4.2 reports the accuracy.")
    print("  M3 search bands and the fraction outside them stated")

    # ------------------------------------------------------------------- M4
    d.patch("k is reproducible within a subject",
            "cardiac k is stable apart from the S6 coupling anomaly",
            "cardiac k is stable apart from S6, whose second night is also the one recording "
            "where the cardiac reference is photoplethysmography rather than ECG (§3.3), so its "
            "outlying k cannot be attributed to the mask alone")
    print("  M4 S6N2's photoplethysmography reference noted where S6 is blamed")

    # --------------------------------------------------------------- M6 / M7
    d.patch("Restricting ridge detection to each band separately",
            "We therefore report reduced respiratory ridge power in deep sleep as a weak and "
            "partially consistent association,",
            "That association is not an artifact of motion or of unequal epoch counts: it is "
            "unchanged when the contrast is restricted to motion-free epochs and when the "
            "non-N3 epochs are randomly count-matched to the N3 epochs of the same subject, and "
            "it appears with the same direction on CLE and CH as on the post-hoc CRE channel. "
            "We therefore report reduced respiratory ridge power in deep sleep as a weak and "
            "partially consistent association,")
    print("  M6/M7 motion, count and channel robustness stated")

    # ------------------------------------------------------------------- M8
    d.patch("Figure S6. Bland–Altman agreement",
            "These limits are epoch-level; the night-mean errors in Table 3 are an order of "
            "magnitude smaller and are not visible at this scale.",
            "These limits are epoch-level and are computed on epochs pooled across recordings, "
            "which are not independent; they describe the spread of single-epoch differences and "
            "are not a population tolerance interval. The night-mean errors in Table 3 are an "
            "order of magnitude smaller and are not visible at this scale.")
    print("  M8 pooled limits declared descriptive")

    # ------------------------------------------------------------------- M9
    d.patch("The mask is suitable for screening-level overnight",
            "The mask is suitable for screening-level overnight respiratory and cardiac rate "
            "monitoring: accurate mean rates per session, with well-characterized calibration "
            "behavior and per-stage accuracy (Figure S2).",
            "Once calibrated against a reference on the night in question, the mask reports "
            "accurate mean respiratory and cardiac rates, with well-characterized calibration "
            "behavior and per-stage accuracy (Figure S2). That qualification is not a formality: "
            "Table 4 shows that without same-night calibration the device does not beat a "
            "no-sensor constant, so screening use is contingent on a calibration transfer that "
            "this study does not demonstrate.")
    d.patch("These findings frame the mask as a viable tool",
            "These findings frame the mask as a viable tool for unobtrusive overnight mean-rate "
            "trending",
            "These findings frame the mask as a viable tool for unobtrusive overnight mean-rate "
            "trending once a calibration that transfers between nights is established")
    print("  M9 screening and conclusion qualified by the calibration requirement")

    # ------------------------------------------------------------------ M10
    d.patch("Per-window rates were estimated",
            "STFT peak tracking with Viterbi smoothing",
            "short-time Fourier transform (STFT) peak tracking with Viterbi smoothing")
    d.patch("Table 2. Signal validation summary across channels.",
            "Table 2. Signal validation summary across channels.",
            "Table 2. Signal validation summary across channels. Coherence columns give the "
            "median with its interquartile range (IQR).")
    # "SD" is a row label inside Table 3, not caption text
    tbl3 = d.body[next(i for i in range(d.find("Table 3. Rate agreement"), len(d.body))
                       if d.body[i].tag == W + "tbl")]
    for ri, tr in enumerate(tbl3.findall(W + "tr")):
        cells = tr.findall(W + "tc")
        label = "".join(t.text or "" for t in cells[0].iter(W + "t"))
        if label.startswith("Reference rate SD"):
            d.cell_text(tbl3, ri, 0, "Reference rate standard deviation within night")
            break
    d.patch("No capacitive feature varied with subject age",
            "band SNR, DC drift", "band SNR, baseline (DC) drift")
    print("  M10 STFT, IQR, SD and DC expanded")

    # ------------------------------------------------------------------ M11
    d.patch("Rate accuracy is reported within-session",
            "Bonferroni correction is applied across the family of reported tests.",
            "Where a family of tests is reported together — the six ridge features within a "
            "band, and the capacitive features tested against age — p-values are Bonferroni "
            "corrected within that family. Single pre-specified comparisons, including the "
            "within-night correlation tests, are reported uncorrected.")
    print("  M11 Bonferroni family made specific")

    # ------------------------------------------------------------------- P1
    d.patch("Motion artifact was suppressed with an ordinary-least-squares",
            "This canceler was used for every result reported here.",
            "This canceler was used for every result reported here. It is a light touch in these "
            "bands: across the twelve recordings it removes a median of 0.1% of in-band variance, "
            "what it removes correlates with accelerometer energy at r = 1.00, and the coherence "
            "between the capacitive channel and its PSG reference is unchanged to three decimal "
            "places, so it suppresses motion without taking physiology with it.")
    print("  P1 canceller validation stated")

    # ------------------------------------------------------------------- P2
    d.patch("Integrated over a night, the imbalance separates the recordings",
            "We therefore report the magnitude burden and the left-dominant time fraction, and "
            "make no claim of a net lateralization.",
            "We therefore report the magnitude burden and the left-dominant time fraction, and "
            "make no claim of a net lateralization. What the burden does track is coupling: "
            "across the twelve nights it correlates with respiratory-band signal-to-noise ratio "
            "at ρ = +0.86 (p < 0.001) and with cardiac-band signal-to-noise ratio at ρ = +0.66 "
            "(p = 0.02), and not with the fraction of the night spent moving (ρ = +0.12, "
            "p = 0.70). The nights with the largest imbalance excursions are the nights with the "
            "strongest physiological signal, not the most restless ones.")
    print("  P2 burden association reported")

    # ------------------------------------------------------------------- P3
    d.patch("Table 3. Rate agreement, from the operational estimator",
            "per-night values are in Table S1.",
            "per-night values are in Table S1. The cardiac per-epoch interquartile range is "
            "widened by S6; excluding that subject it is 3.32 [2.95–4.37] BPM.")
    print("  P3 cardiac IQR without S6 given")

    # ------------------------------------------------------------------- P4
    d.patch("k is a waveform-morphology count",
            "across the nine sessions with usable peak detection",
            "across the nine of twelve sessions in which the R-peak-triggered average showed a "
            "countable capacitive pulse; the three excluded recordings gave no resolvable "
            "per-beat waveform")
    print("  P4 peaks-per-beat exclusion rule stated")

    # ------------------------------------------------------------------- P6
    d.patch("k is reproducible within a subject",
            "Respiratory k is therefore effectively a subject-level constant,",
            "Respiratory k is therefore effectively a subject-level constant. The cardiac band is "
            "reproducible only to about 20% of k, which is what carrying a subject's k across to "
            "their other night costs: the night-level error rises from 1.56 to 3.77 BPM.")
    print("  P6 k stability quantified by its consequence")

    # ------------------------------------------------------------------- P7
    d.patch("Cortical slow-wave (delta) activity provides a second discrete",
            "(a few dozen to roughly one hundred qualifying onsets per night; predominantly "
            "isolated N2 slow-wave and K-complex onsets, since sustained N3 slow-wave activity "
            "has no discrete quiet onset)",
            "(344 qualifying onsets in total, but distributed very unevenly: 1 to 99 per night, "
            "with three nights contributing fewer than ten; predominantly isolated N2 slow-wave "
            "and K-complex onsets, since sustained N3 slow-wave activity has no discrete quiet "
            "onset)")
    print("  P7 real onset counts given")

    # ------------------------------------------------------------------- P8
    anchor = d.find("Fourth, the respiratory consensus ground truth")
    d.patch("Fourth, the respiratory consensus ground truth",
            "Truly independent respiratory validation would require a separate sensing modality "
            "(e.g., capnography or acoustic respiration monitoring).",
            "Truly independent respiratory validation would require a separate sensing modality "
            "(e.g., capnography or acoustic respiration monitoring). The reference also sets a "
            "floor on any accuracy measured against it: two simultaneous PSG respiratory sensors "
            "differ by more than 1 br/min on 29% of epochs (§3.3), so per-epoch errors of that "
            "order cannot be attributed to the mask alone.")
    print("  P8 reference floor added to the limitations")

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
