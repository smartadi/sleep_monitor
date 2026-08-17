"""Plain-language pass over the canonical manuscript (user request, 2026-08-17).

Shorter sentences, fewer nominalisations, no restatement of a point already
made in the same paragraph. Every number, hedge and caveat is preserved --
where a sentence was carrying a caveat, the rewrite carries the same caveat in
fewer words.

Three content inconsistencies are fixed in the same pass, because they are
wording defects as much as numeric ones:

  * 3.5 described the reported pipeline as "fused across five channels by
    agreement gating". Section 4.2, 4.2.1, Table 3, Table S1 and the Figure 4
    and 5 captions all report single-channel CRE. 3.5 was stale from the
    pipeline that preceded the 2026-08-14 rewrite; fusion is now reported as
    evaluated and not adopted (it moved per-epoch error by < 0.1).
  * 3.5 still defined k over "randomly selected calibration windows", which
    the following sentence then contradicted. The phantom clause is gone.
  * The degenerate respiratory estimator returned 15 br/min in 99.98% of
    epochs (9,317 of 9,319). 3.5 and 5.3 said 99.95%.

Two garbled sentences in section 2 are also repaired -- "A few features of the
sensing mechanism is that mean ICP in the pig model showed that linear
analysis further demonstrates ..." and "To make sure the sensitivity to Cr and
Cl, additional two sensors were positioned ...".

Run from the repo root:  python writeup/edits/apply_simplify.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_SIMPLIFY_20260817.docx")

log = []


def note(msg):
    log.append(msg)
    print("  " + msg)


# ============================================================== 2. sensing

SEC2_SENSITIVITY = (
    "At the sensor's 120 mm penetration depth the cerebral vascular network geometry is taken to "
    "be stable, so with a fixed sensor-to-skin distance a change in Ctotal is read as a change in "
    "mean ICP. Sensitivity was measured against invasive pressure in a pig model: 4.8 ± 2.5 "
    "fF/mmHg for respiratory ICP oscillations and 18.0 ± 5.3 fF/mmHg for displacement-induced ICP "
    "changes, which allows capacitance fluctuations to be converted into predicted ICP amplitudes. "
    "The capacitive signal tracked invasive pressure closely in that model, with absolute "
    "correlation coefficients of 0.96 ± 0.04 for ICP and 0.93 ± 0.10 for central venous pressure "
    "(CVP)."
)

SEC2_PLACEMENT = (
    "The mask carries its sensors around the ocular region. Those at the right and left eyes (Cr "
    "and Cl) pick up eye movement as well as the corresponding r-ICP changes. Two further sensors "
    "sit to the left and right of the eyes, close to the temple, and are read as a differential "
    "channel (Ch). The channels are electrically and capacitively independent, which is what makes "
    "them useful as checks on one another: a feature that appears on independent channels is "
    "physiological rather than motion artifact or an environmental change."
)

# ============================================================== 3. methods

M35_ESTIMATORS = (
    "Per-window rates were estimated from the bandpassed SEC channel with seven base estimators: "
    "the Welch spectral peak, the dominant autocorrelation lag with parabolic interpolation, the "
    "median Hilbert instantaneous frequency, the upward zero-crossing rate, prominence-thresholded "
    "peak counting at a loose and at a strict threshold, and a spectral-guided amplitude-adaptive "
    "peak detector. The Welch estimator used 4-second segments with 50% overlap at the 100 Hz "
    "sampling rate, a frequency resolution of 0.25 Hz. Two further trackers were tried on the "
    "harder cardiac band, continuous-wavelet-transform (CWT) ridge tracking and STFT peak tracking "
    "with Viterbi smoothing, and neither entered the reported pipeline. Figure S1 compares every "
    "estimator on every channel; the one we report is described next."
)

M35_K = (
    "SEC peak counts run systematically high against the PSG reference, because the capacitive "
    "waveform does not produce one clean cycle per physiological event. We correct for this with a "
    "per-session scalar k, and report the calibrated rate as the raw estimate divided by k. k is a "
    "whole-night quantity: the median ratio of estimate to reference across every valid epoch of "
    "the recording, with ratios clipped to 0.3–5.0 to drop epochs where either value has failed. "
    "Because k is fitted against the reference, any accuracy computed after k-scaling is "
    "conditional on that calibration. Uncalibrated and k-scaled accuracy are therefore reported "
    "side by side, and section 4.2 separates what k supplies from what the sensor supplies."
)

M35_PIPELINE = (
    "The reported pipeline is therefore one estimator followed by k-scaling and a causal "
    "three-epoch median filter: loose prominence-based peak counting on the right-temple channel "
    "(CRE), in both bands. Two choices need comment. Multi-channel fusion, both quality-weighted "
    "and agreement-gated, moved per-epoch error by less than 0.1 in either band, so we report the "
    "simpler single channel. And the Welch spectral peak, which reaches a marginally lower "
    "per-epoch respiratory error, is degenerate at this window length: at 0.25 Hz resolution the "
    "0.1-0.5 Hz respiratory band holds only two usable bins, and the estimator returned 0.25 Hz -- "
    "exactly 15 breaths/min -- in 99.98% of epochs. Its apparent accuracy is that of a constant 15 "
    "br/min rescaled by a k that is itself fitted to the reference, so we treat it as a baseline "
    "rather than a measurement."
)

M37_STATS = (
    "Rate accuracy is reported within-session, so that agreement between session means cannot "
    "inflate it. Two error definitions are used throughout and kept distinct. The night-level "
    "error is the absolute difference between the mean estimated and the mean reference rate over "
    "a whole recording. The per-epoch error is the median absolute error across that recording's "
    "epochs. Cross-session summaries give the median and interquartile range of per-session "
    "values, so the unit of analysis is the recording, not the epoch. Within-session association "
    "is the Pearson correlation across a night's epochs, summarised across nights by the median "
    "and a Wilcoxon signed-rank test on the twelve values. Stage comparisons use Kruskal–Wallis "
    "across stages, with Mann–Whitney U for N3 against the rest. Where such a test is computed on "
    "pooled epochs its p-value is anti-conservative, because epochs within a night are not "
    "independent; we therefore print those p-values only beside the per-subject direction count, "
    "which takes the subject as the unit of analysis, and treat the direction count as the "
    "evidence. With six subjects a subject-level Wilcoxon test floors at p = 0.031. Correlations "
    "against subject-level markers use Spearman's rank correlation with an exact permutation "
    "p-value over all 720 orderings, since the asymptotic approximation is unreliable at n = 6. "
    "Bonferroni correction is applied across the family of reported tests."
)

# ============================================================== 4. results

R41_RECORD = (
    "The primary record the mask produces is the overnight evolution of the sensor value itself, "
    "shown for a representative night in Figure 2 beside the scored hypnogram, the head-turn angle "
    "and the tracked rhythms. The temple electrodes operate near 2 pF: session mean levels span "
    "1958–2048 fF on CLE and 1624–2353 fF on CRE, with the forehead channel CH between −719 and "
    "−1297 fF. Within a night the level does not drift smoothly. It moves in discrete steps that "
    "coincide with posture changes. The within-night excursion has a median range of 105 fF on "
    "CLE, 71 fF on CRE, 158 fF on the CLE−CRE differential and 595 fF on CH, and exceeds 1000 fF "
    "on CH in the three most mobile nights. This step structure, not sensor noise, is what makes "
    "the record non-stationary."
)

R41_POSTURE = (
    "Head position does not explain the direction of the imbalance, though it does explain its "
    "size. Pooled across 9,182 epochs we found no association between direction and head-turn "
    "angle (Spearman ρ = 0.01, p = 0.62). That is a failure to detect an association rather than "
    "evidence of independence, and because epochs within a night are not independent the test is "
    "weaker than its epoch count suggests. A simpler control points the same way: within the 7,641 "
    "supine epochs, where the head is held face-up, the direction still moves across its full "
    "range and still reverses, a median of 3.5 times per night. Magnitude behaves the opposite way "
    "and is clearly posture-scaled — a median of 9.0 fF supine and 11.3 fF left-lying against "
    "107.7 fF right-lying. We therefore report the signed direction as a signal property that "
    "posture does not explain, and the magnitude as one that must be read beside head position. "
    "Neither is offered as a measurement of intracranial fluid movement; that reading would need a "
    "zero-flow or head-position calibration this protocol does not provide."
)

R41_ENERGY = (
    "The band energy underlying those traces sits well above the sensor noise floor. The "
    "respiratory band carried 29-48% of total signal power (0-5 Hz) and the cardiac band 8-48%. We "
    "measured how far this content sits above the noise floor as a physiological-band SNR for each "
    "raw channel: the ratio of mean spectral density (power per Hz) in the 0.1-3 Hz band to that of "
    "the 10-50 Hz electronic noise floor, which an independent no-subject baseline recording "
    "confirms is spectrally white. SNR was positive on every channel in all twelve recordings, "
    "averaging +30.0 dB on CH and +18.7 and +20.7 dB on CLE and CRE, with no single value below "
    "+6.4 dB. The forehead channel was strongest in every session. The temple channels varied more "
    "from night to night, which reflects differences in sensor-skin coupling rather than added "
    "noise, since the noise floor was near-constant across sessions. Cross-subject variation was "
    "substantial: S6N1 was respiration-dominated (48% respiratory, 8% cardiac) while S3N1 was "
    "cardiac-heavy (48% cardiac)."
)

R42_INTRO = (
    "This section shows what the mask recovers of respiratory and cardiac rate over a night, "
    "against simultaneous polysomnography on twelve recordings. One thing applies to every number "
    "that follows. The capacitive waveform does not produce one deflection per physiological "
    "event, so each recording carries its own scalar calibration factor k, fitted against its own "
    "reference (§4.2.1). What the figures below show is therefore agreement after per-session "
    "calibration, not standalone accuracy, and we make no claim for an uncalibrated device. With "
    "six subjects we do not try to establish how well that calibration transfers between subjects "
    "or nights; that needs a larger cohort."
)

R43_STAGES = (
    "Restricting ridge detection to each band separately revealed physiologically interpretable "
    "stage associations (Figure 7). In the respiratory band (0.1–0.5 Hz) a persistent ridge was "
    "present in 96.7% of clean epochs, and its lowest frequency sat at a median of 0.23 Hz (≈14 "
    "breaths/min) in every stage, which confirms that the ridge tracks respiration. N3 epochs "
    "carried fewer active respiratory ridges than other sleep, in all six subjects, and lower "
    "total ridge power, in five of six — consistent with the more regular, lower-effort breathing "
    "of deep sleep. Cardiac-band ridges (0.5–3.0 Hz) were intermittent, present in only 11.7% of "
    "epochs, but more frequent during N3 and N2 than in lighter stages. When present, the lowest "
    "cardiac ridge frequency was lower in N3 than in other sleep, in five of six subjects, "
    "mirroring the sleep bradycardia of deep NREM. We report that shift as a direction rather than "
    "a calibrated frequency, because the N3 values approach the 0.5 Hz lower edge of the analysis "
    "band and the tracker's own search band begins at 0.85 Hz. The pooled Kruskal–Wallis and "
    "Mann–Whitney p-values for these contrasts are very small, down to 3×10⁻²³, but they are "
    "computed on non-independent epochs and Figure 7 reports them as descriptive statistics only; "
    "the per-subject direction counts are the evidence. Both associations reflect autonomic and "
    "respiratory change transduced mechanically by the sensor, not cortical activity, and neither "
    "is strong or consistent enough across subjects to serve as a standalone sleep-stage "
    "classifier."
)

R45_CAUSAL = (
    "The capacitive response therefore follows the cortical onset rather than preceding it. The "
    "pre-onset baseline in Figure 9 is flat under strictly causal filtering, the capacitive-to-EEG "
    "cross-correlation peaks at zero lag, and pre-onset capacitive power does not forecast an "
    "imminent onset (area under the curve 0.42 to 0.56, straddling chance). A gradual pre-onset "
    "rise does appear in the slowest band (0-0.5 Hz) under conventional zero-phase filtering, but "
    "we traced it to acausal leakage of the large post-onset response backward in time: under the "
    "causal estimator it disappears, the real-minus-null difference in the final three seconds "
    "before onset falling from +0.35 to +0.41 z in six of six subjects to approximately zero in "
    "two to three of six. As with sleep spindles (Section 4.4), what the temple sensor registers "
    "at a delta-burst onset is a mechanical process, not a direct electrical signature - the "
    "transient cardiovascular and micromotion fluctuations that follow the cortical event."
)

# ============================================================== 5. discussion

D51_RATES = (
    "The mask recovers each night's mean respiratory rate to a median of 0.24 br/min and its mean "
    "cardiac rate to 1.56 BPM after per-session k-calibration, with per-epoch errors of 1.79 "
    "br/min and 3.41 BPM. Both figures are conditional on calibration against a reference. "
    "Respiration carries a further caveat: the cohort's mean rates span only 14.4–16.8 br/min, so "
    "a constant predictor already reaches 0.78 br/min, and the respiratory figure therefore "
    "demonstrates calibration rather than measurement. The cardiac figure is the stronger result. "
    "Respiratory k is 1.18, so peak counting registers about 18% more deflections than there are "
    "breaths, and the respiratory waveform is not the clean one-peak-per-breath signal a value of "
    "1.0 would imply. Cardiac k is consistent across subjects and nights — 1.96, with an "
    "interquartile range spanning only 1.79–2.00 — and reflects the roughly 2:1 overcounting that "
    "follows from the biphasic capacitive pulse, a systolic peak plus a dicrotic notch. The "
    "calibration is stable night to night: respiratory k changed by less than 0.05 between a "
    "subject's two nights in four of six subjects. We make no direct comparison with other "
    "non-contact or wearable cardiac sensors, since most report per-epoch rather than per-night "
    "error and we are not aware of a published dataset using this window length and error "
    "definition."
)

D53_CARDIAC = (
    "What follows is our account of why within-night cardiac variation is not recovered. It is "
    "consistent with everything reported here, but these data do not test it against alternatives, "
    "and it should be read as a hypothesis. A cardiac k of about 2 across subjects says the "
    "capacitive waveform carries two inflection points per heartbeat, most likely the systolic and "
    "dicrotic pressure peaks. The dominant frequency of such a waveform is set by its morphology, "
    "which is stable, rather than by the instantaneous heart rate. As heart rate varies within a "
    "session the waveform stretches and compresses, but the peak-counting frequency stays governed "
    "by that persistent biphasic structure. Only a change large enough to alter the number of "
    "peaks, rather than their spacing, would shift the frequency the estimators see. This is why "
    "the cardiac band offers ample variation to resolve and none of it is recovered."
)

D53_RESP = (
    "The respiratory case has a different basis. At a 30-second window with 4-second Welch segments "
    "the frequency resolution is 0.25 Hz, comparable to the width of the entire 0.1–0.5 Hz "
    "respiratory band, so the spectral estimator has only two reachable output values and returned "
    "0.25 Hz in 99.98% of epochs — a constant predictor rather than a measurement. Peak-counting "
    "estimators are not resolution-limited in this way and do vary, but the variation they are "
    "asked to resolve is itself small: a median within-night standard deviation of 1.57 br/min "
    "against a per-epoch error of comparable size. Longer windows and full-window periodograms "
    "with parabolic interpolation recover resolution at the cost of temporal responsiveness, "
    "without improving the within-session correlation, so this is a genuine trade-off rather than "
    "a poor parameter choice."
)

D53_RIDGES = (
    "The subject-dependence of the weaker ridge effects may have a similar basis. The sensor sits "
    "at the temple and receives a mixture of intracranial pressure, superficial temporal artery "
    "pulsation and near-field respiratory displacement, and how much each contributes varies with "
    "anatomy, sensor placement and mask fit. During N3 cardiac output falls and respiratory "
    "mechanics change; how those changes project onto the temple sensor depends on that "
    "subject-specific coupling geometry. This would explain why the ridge effects hold their "
    "direction in most subjects but not in all."
)


def main():
    if not DOC.exists():
        raise SystemExit("missing %s" % DOC)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # -------------------------------------------------------------- section 2
    d.set_text("Considering the 120 mm penetration depth", SEC2_SENSITIVITY)
    d.set_text("In the sleep mask, the sensors were positioned", SEC2_PLACEMENT)
    note("2: two garbled sentences repaired (pig-model sensitivity, sensor placement)")

    # -------------------------------------------------------------- section 3
    d.set_text("Per-window rates were estimated", M35_ESTIMATORS)
    d.set_text("SEC peak counts are systematically scaled", M35_K)
    d.set_text("The operational pipeline was therefore", M35_PIPELINE)
    note("3.5: simplified; phantom 'randomly selected calibration windows' removed; "
         "pipeline corrected to single-channel CRE with fusion reported as not adopted; "
         "degenerate-estimator rate corrected to 99.98%")

    d.set_text("All rate accuracy metrics are reported within-session", M37_STATS)
    note("3.7: same content, shorter sentences")

    # -------------------------------------------------------------- section 4
    d.set_text("The primary record the mask produces", R41_RECORD)
    d.set_text("Head position does not account for the direction", R41_POSTURE)
    d.set_text("The band energy underlying those traces", R41_ENERGY)
    note("4.1: three paragraphs simplified, all numbers and both caveats retained")

    d.set_text("This section demonstrates what the mask recovers", R42_INTRO)
    note("4.2: opening paragraph simplified")

    d.set_text("Restricting ridge detection to each band separately", R43_STAGES)
    note("4.3: the long stage paragraph broken into shorter sentences")

    d.set_text("The capacitive response therefore follows the cortical onset", R45_CAUSAL)
    note("4.5: the causal-control sentence split into three")

    # -------------------------------------------------------------- section 5
    d.set_text("The mask recovers each night's mean respiratory rate", D51_RATES)
    note("5.1: simplified; cardiac k stated as 1.96 to match Table 3 and 4.2.1")

    d.set_text("We propose the following account", D53_CARDIAC)
    d.set_text("The respiratory case has a different basis", D53_RESP)
    d.set_text("The subject-dependence of the weaker ridge effects", D53_RIDGES)
    note("5.3: three paragraphs simplified; degenerate-estimator rate corrected to 99.98%")

    d.save(DOC)
    print("\n%d paragraphs rewritten" % d.n_patch)
    print("wrote %s" % DOC)
    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Plain-language pass (2026-08-17)\n\n"
        + "\n".join("- " + m for m in log) + "\n", encoding="utf-8")
    print("appended to writeup/edits/WRAPUP_CHANGELOG.md")


if __name__ == "__main__":
    main()
