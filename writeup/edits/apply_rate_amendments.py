"""Apply the 2026-08-06 rate-detection audit corrections to the CANONICAL manuscript.

  writeup/main/CAP_sleep_mask_manuscript_main.docx   (edited in place)

Backup taken before the first run:
  writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_RATE_AUDIT_20260806.docx

Findings and rationale: writeup/edits/RATE_AUDIT_2026-08-06.md
Numbers:                reports/rates/mask/table3_corrected.csv  (analysis/rates/corrected_rate_table.py)
Verification:           analysis/rates/audit_rate_claims.py

Run from the repo root:  python writeup/edits/apply_rate_amendments.py
"""

import sys
from pathlib import Path

import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_review_edits import Doc, W  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_RATE_AUDIT_20260806.docx")
T3 = Path("reports/rates/mask/table3_corrected.csv")


def main() -> None:
    assert BACKUP.exists(), f"refusing to edit without a backup at {BACKUP}"
    # dtype=str so the signed within-session r keeps its "+"/"−" as written
    t3 = pd.read_csv(T3, dtype=str).set_index("band")
    t3 = t3.map(lambda v: v.replace("-", "−") if isinstance(v, str) else v)
    r, c = t3.loc["resp"], t3.loc["card"]
    d = Doc(DOC)

    # ─────────────────────────────────────────────── Table 1: epoch length
    tbl1 = d.body[d.find("Table 1. Recording sessions and demographics.") + 1]
    assert etree.QName(tbl1).localname == "tbl"
    d.cell_text(tbl1, 0, 6, "Analysis epochs (30 s)")

    # ─────────────────────────────────────────────── 3.3 ground truth
    d.patch("All rates were computed on a common sliding-window grid",
            "a common sliding-window grid (60-second windows, 30-second step) aligned to the "
            "consensus 5-second epoch grid",
            "a common grid of 30-second non-overlapping windows aligned to the consensus "
            "5-second epoch grid")
    d.patch("To validate this consensus",
            "Within-session correlation was r = +0.48 on raw rates and r = +0.28 on detrended "
            "fluctuations",
            "Within-session correlation had a median of r = 0.47 (IQR 0.33–0.67) on raw rates "
            "and r = 0.27 on detrended fluctuations")

    # ─────────────────────────────────────────────── 3.5 rate estimation and k
    d.patch("Per-window rates were estimated", "six base estimators", "seven base estimators")
    d.patch("Per-window rates were estimated",
            "Of these, only the spectral peak (respiratory) and loose peak-counting (cardiac) "
            "were used for the reported rates; the remaining estimators, including the CWT-ridge "
            "and STFT-Viterbi trackers, were evaluated but not used in the final pipeline.",
            "One estimator requires comment because it constrains how the respiratory numbers "
            "must be read. The Welch spectral estimator uses 4-second segments, so at the "
            "100 Hz rate of the synchronised recordings the frequency resolution is 0.25 Hz "
            "and the 0.1–0.5 Hz respiratory band contains only two usable bins. It returned "
            "0.25 Hz — exactly 15 breaths/min — in 99.98% of all epochs, and had zero "
            "within-session variance in 10 of 12 recordings. Its output is therefore a constant, "
            "and its apparent accuracy is the accuracy of that constant after rescaling by a "
            "factor fitted to the reference. We report it only as a disclosed baseline and use "
            "the non-degenerate loose peak-counting estimator for all quoted rates; the "
            "remaining estimators, including the CWT-ridge and STFT–Viterbi trackers, were "
            "evaluated but not used in the final pipeline.")

    d.patch("SEC peak counts are systematically scaled",
            "We correct this with a per-session scalar k, defined as the median ratio between "
            "the CAP estimate and the ground-truth rate across randomly selected calibration "
            "windows, so that the calibrated rate equals the raw estimate divided by k. "
            "Calibration used 50 randomly drawn one-minute windows and was verified against the "
            "whole-night k: |k_diagnostic − k_whole| ≤ 0.04 for all sessions. Both uncalibrated "
            "and k-scaled accuracies are reported.",
            "We correct this with a per-session scalar k, the median of the ratio between the "
            "CAP estimate and the reference rate taken over every valid epoch of the recording "
            "(ratios outside 0.3–5.0 discarded), so that the calibrated rate equals the raw "
            "estimate divided by k. Because k is fitted against the reference on the same "
            "recording it is scored on, accuracy under this calibration is a fitted quantity "
            "and not a measurement. We therefore report it alongside three comparators that "
            "withhold information from the night being scored: k fitted on the same subject's "
            "other night, a single population k for the whole cohort, and a no-sensor predictor "
            "that assigns the cohort median rate to every epoch.")

    d.set_text(
        "The operational pipeline was therefore a single per-window estimator",
        "The operational pipeline was therefore agreement-gated fusion of loose "
        "prominence-based peak counting across the five available channel derivations, "
        "followed by k-scaling and a causal three-epoch median filter, applied identically to "
        "both bands. Fusion did not measurably improve either band over the single CLE−CRE "
        "differential (ΔMAE < 0.1) but avoids the single-channel selection that the earlier "
        "analysis made post hoc. Uncalibrated, cross-night-calibrated, population-calibrated "
        "and no-sensor accuracies are all reported in Table 3.")

    # ─────────────────────────────────────────────── 4.2 Results
    d.set_text(
        "The sensor was evaluated against PSG in two regimes",
        "The sensor was evaluated against PSG in two regimes that differ in the quantity "
        "certified. The aggregate regime asks whether the mask recovers a subject's average "
        "respiratory and cardiac rate over a whole recording — the quantity relevant to "
        "overnight screening and night-to-night trending — and is scored as the absolute "
        "difference between the mean estimated and the mean reference rate for that night. The "
        "within-night regime asks whether the mask follows the rate as it varies during the "
        "night, and is scored by the median absolute error across the epochs of a night and by "
        "the correlation between estimate and reference across those epochs. Both are computed "
        "on the same 30-second non-overlapping windows. They are reported together because they "
        "reach different conclusions, and because the distinction determines what the device "
        "can be used for.")

    d.set_text(
        "For the night-level regime we used the estimator that minimised error",
        "Both regimes use the single operational pipeline of §3.5. Every accuracy below is "
        "reported under four calibrations: k fitted on the night being scored, k fitted on the "
        "same subject's other night, a single population k, and a no-sensor predictor that "
        "assigns the cohort median rate (15.7 br/min, 61.4 BPM) to every epoch. The first is "
        "what a validation study can measure; the last three are what a deployment would "
        "actually have available, and the comparison between them is the main result of this "
        "section.")

    d.set_text(
        "Respiratory rate was recovered with a per-session median MAE of 0.91 br/min",
        "With k fitted on the night being scored, the mask reproduces the night mean closely in "
        f"both bands: median night-level error {r.night_self_k} br/min for respiration and "
        f"{c.night_self_k} BPM for the cardiac band. This accuracy does not survive holding the "
        "calibration out. Calibrating on the same subject's other night raises the night-level "
        f"error to {r.night_cross_night_k} br/min and {c.night_cross_night_k} BPM; a single "
        f"population k gives {r.night_population_k} br/min and {c.night_population_k} BPM. "
        "Neither improves on a predictor that uses no sensor data at all — assigning the cohort "
        f"median rate to every night gives {r.night_no_sensor} br/min and "
        f"{c.night_no_sensor} BPM. The same holds per epoch, where the operational pipeline "
        f"achieves {r.epoch_self_k} br/min and {c.epoch_self_k} BPM with same-night calibration, "
        f"{r.epoch_population_k} br/min and {c.epoch_population_k} BPM with a population k, and "
        f"the no-sensor constant achieves {r.epoch_no_sensor} br/min and {c.epoch_no_sensor} BPM. "
        "Excluding the anomalously coupled S6 subject does not change this ordering. The "
        "defensible statement is therefore narrower than a rate-measurement claim: after "
        "calibration against a reference on the same recording the sensor reproduces that "
        "recording's mean rate, and without such a reference it does not improve on a population "
        "constant. Per-session values for all twelve nights are given in Table S1 and the "
        "epoch-level agreement plots in Figure 3.")

    d.set_text(
        "Within a night, neither band's rate variation was recovered",
        "Within a night, neither band's rate variation was recovered (Table 3). Estimates "
        f"correlated with the reference at a median within-session r = {r.r_within_median} for "
        f"respiration (positive in {r.nights_r_positive} nights) and r = {c.r_within_median} for "
        f"the cardiac band (positive in {c.nights_r_positive}); a responsive detector combining "
        "loose peak counting with Hilbert instantaneous frequency traded error for temporal "
        f"resolution without recovering the variation ({r.epoch_responsive_detector} br/min, "
        f"{c.epoch_responsive_detector} BPM). Neither correlation differs from zero on a Wilcoxon "
        "signed-rank test over the twelve per-night values (p = 0.68 and p = 0.34), and no "
        "configuration in the estimator battery was distinguishable from a 200-iteration "
        "temporal-shuffle null (Figure S3). This is not entirely a limitation of the mask: two "
        "physically independent PSG respiratory sensors — nasal airflow and the respiratory "
        "inductance belt sum — agree with each other at only r = 0.47 (IQR 0.33–0.67) on the "
        "same epochs, which bounds what any respiratory sensor can achieve at this time "
        "resolution. But the mask reaches a small fraction of that bound, and for the cardiac "
        "band no comparable external bound applies.")

    d.set_text(
        "The two bands fall short of that bound for different reasons",
        "The two bands fail to track for different reasons, and the amount of variation "
        "available to be tracked distinguishes them. Within a night the reference respiratory "
        f"rate varies with a median standard deviation of {r.ref_sd_within_night} br/min against "
        f"a per-epoch error of {r.epoch_self_k} br/min, so the signal to be resolved is of the "
        "same order as the error and little variation is recoverable in principle. The cardiac "
        "band is different: reference heart rate varies with a median standard deviation of "
        f"{c.ref_sd_within_night} BPM, well above the {c.epoch_self_k} BPM per-epoch error, so "
        "the variation is in principle resolvable and is nonetheless not recovered. The cardiac "
        "failure is therefore mechanistic rather than a matter of insufficient signal, and is "
        "discussed in §5.3. An exhaustive comparison of estimators and channels in both regimes, "
        "including CWT-ridge and STFT–Viterbi trackers, is given in Figure S1 and Figure S3.")

    # ─────────────────────────────────────────────── Table 3
    d.set_text(
        "Table 3. Rate detection in both regimes.",
        "Table 3. Rate detection in both regimes, from the operational pipeline "
        "(agreement-gated five-channel fusion of loose peak counting, per-session k, causal "
        "three-epoch median filter). Values are medians across the 12 nights with the "
        "interquartile range in brackets; per-night values are in Table S1. Night-level error is "
        "|mean(estimate) − mean(reference)| for a recording; per-epoch error is the median "
        "absolute error across that recording's epochs. Cross-night k is fitted on the same "
        "subject's other night; the no-sensor row assigns the cohort median rate to every epoch "
        "and uses no sensor data.")

    old_tbl = d.body[d.find("Table 3. Rate detection in both regimes") + 1]
    assert etree.QName(old_tbl).localname == "tbl"
    rows = [
        ["", "Respiratory (br/min)", "Cardiac (BPM)"],
        ["Aggregate regime — night-level error", "", ""],
        ["k fitted on the same night", r.night_self_k, c.night_self_k],
        ["k fitted on the other night [held out]", r.night_cross_night_k, c.night_cross_night_k],
        ["population k [held out]", r.night_population_k, c.night_population_k],
        ["no sensor — cohort median rate", r.night_no_sensor, c.night_no_sensor],
        ["Within-night regime — per-epoch error", "", ""],
        ["k fitted on the same night", r.epoch_self_k, c.epoch_self_k],
        ["population k [held out]", r.epoch_population_k, c.epoch_population_k],
        ["no sensor — cohort median rate", r.epoch_no_sensor, c.epoch_no_sensor],
        ["responsive detector, same-night k", r.epoch_responsive_detector,
         c.epoch_responsive_detector],
        ["Within-night regime — tracking", "", ""],
        ["Within-session r", f"{r.r_within_median} (p = 0.68)", f"{c.r_within_median} (p = 0.34)"],
        ["Nights with r > 0", r.nights_r_positive, c.nights_r_positive],
        ["Reference rate SD within night", r.ref_sd_within_night, c.ref_sd_within_night],
        ["Independent-sensor bound", "r = 0.47 [0.33–0.67]", "not available"],
        ["Calibration", "", ""],
        ["Calibration factor k, median [IQR]", r.k_median_iqr, c.k_median_iqr],
        ["Calibration factor k, range", r.k_range, c.k_range],
    ]
    d.body.replace(old_tbl, d.table(rows, old_tbl))

    d.set_text("Night-level accuracy", "Aggregate accuracy: the per-night mean rate")
    d.set_text("Epoch-level accuracy", "Within-night accuracy: rate variation is not recovered")

    d.patch("Figure 3. Bland–Altman agreement",
            "for respiratory (left) and cardiac (right) rate in the night-level regime. Each "
            "point is one analysis epoch;",
            "for respiratory (left) and cardiac (right) rate. Each point is one 30-second "
            "analysis epoch, pooled across the twelve recordings, under same-night calibration; "
            "the spread shown is epoch-level and is not an agreement limit for the per-night "
            "means of Table 3;")

    # ─────────────────────────────────────────────── 4.2.1 the k factor
    d.patch("Rate estimates in both bands are rescaled by a per-session factor k",
            "the median ratio of the raw estimator output to the reference rate over 50 randomly "
            "selected one-minute windows",
            "the median ratio of the raw estimator output to the reference rate over every valid "
            "epoch of the recording")
    d.patch("Rate estimates in both bands are rescaled by a per-session factor k",
            "k is not a free parameter fitted to improve accuracy; it counts how many capacitive "
            "deflections the sensor produces per physiological event, and it behaves as a stable "
            "property of the subject.",
            "In the cardiac band k has a physical reading: it counts how many capacitive "
            "deflections the sensor produces per heartbeat, and that count can be measured "
            "independently of the rate pipeline. In the respiratory band it does not, for the "
            "reason given below.")

    d.patch("k is a waveform-morphology count.",
            "across the nine sessions with usable peak detection",
            "across the nine of twelve sessions with usable peak detection; S4N1 failed outright "
            "and S5N1 and S6N2, the latter having no usable ECG, were not analysed")
    d.patch("k is a waveform-morphology count.",
            "The correspondence is population-level rather than exact per subject (r = 0.50 "
            "between measured peaks-per-beat and fitted k), and one session was excluded because "
            "cardiac peak detection failed outright. The respiratory k of ≈0.97 reflects the "
            "simpler coupling: each breath produces one dominant temple displacement.",
            "The correspondence is population-level rather than exact per subject: the "
            "correlation between measured peaks-per-beat and fitted k is r = 0.50, which at "
            "n = 9 is not distinguishable from zero (p = 0.17), so the morphological account is "
            "carried by the agreement of the central values rather than by covariation across "
            "subjects. No equivalent reading is available for the respiratory band. Under the "
            "degenerate spectral estimator that the earlier analysis used, the respiratory k is "
            "identically 15 divided by the session's median reference rate — verified to within "
            "5 × 10⁻⁴ on all twelve recordings — so it carries no information about the sensor. "
            "Under the operational peak-counting pipeline the respiratory k is ≈1.00, consistent "
            "with one dominant temple displacement per breath, but it remains a fitted scale "
            "factor rather than an independently measured count.")

    d.patch("k is reproducible within a subject.",
            "Respiratory k is therefore effectively a subject-level constant, and cardiac k is "
            "stable apart from the S6 coupling anomaly. Diagnostic estimates from 50 random "
            "windows agreed with whole-night values to within 0.04.",
            "Respiratory k is therefore stable between a subject's two nights and cardiac k is "
            "stable apart from the S6 coupling anomaly, whose two nights account for the full "
            "cardiac range of 0.94–2.23 while the central half spans only 1.79–1.99. For the "
            "respiratory band this reproducibility should not be read as sensor stability: under "
            "the spectral estimator it restates the reproducibility of the subject's median "
            "breathing rate.")

    d.patch("k, age, and the calibration a deployment would require.",
            "Because k reflects the mechanical coupling between physiology and sensor, we asked "
            "whether it varies with subject age (Figure 4). Respiratory k declined across the six "
            "subjects, from 1.04 in the youngest (25 years) to 0.91 in the oldest (66 years) — "
            "Spearman ρ = −0.83, uncorrected p = 0.042.",
            "We asked whether k varies with subject age (Figure 4). Respiratory k declined across "
            "the six subjects, from 1.04 in the youngest (25 years) to 0.91 in the oldest "
            "(66 years) — Spearman ρ = −0.83. At n = 6 we evaluate this with an exact permutation "
            "test over all 720 orderings, which gives p = 0.058; the asymptotic t-approximation "
            "would give p = 0.042, and we report the exact value because the approximation is "
            "unreliable at this sample size. The result is also sensitive to how k is aggregated: "
            "computing k without the ratio clipping used in the pipeline gives ρ = −0.77 and an "
            "exact p of 0.103.")
    d.patch("k, age, and the calibration a deployment would require.",
            "Cardiac k showed no such relationship (ρ = +0.37, p = 0.47)",
            "Cardiac k showed no such relationship (ρ = +0.37, exact p = 0.47)")

    d.set_text(
        "We tested whether the respiratory relationship is strong enough to replace",
        "This relationship must be read with the identity above in mind. The respiratory k used "
        "here comes from the degenerate spectral estimator, where k is 15 divided by the "
        "session's median reference rate; the age association is therefore a restatement of the "
        "observation that median respiratory rate rose with age in this cohort, from 14.5 br/min "
        "at 25 years to 16.6 br/min at 66 years. It is an observation about breathing, not about "
        "the mechanical coupling between physiology and sensor, and it does not by itself provide "
        "a sensor calibration route. For completeness: fitting k from age on five subjects and "
        "predicting the sixth gave a mean absolute k error of 0.020, against 0.056 for k = 1.0 "
        "and 0.056 for the population mean, whereas the same test on cardiac k favoured the "
        "population mean (0.305) over an age prior (0.387) — the asymmetry a fixed pulse "
        "morphology would produce.")

    d.patch("Two cautions apply to the age result.",
            "so the association may reflect any subject characteristic that covaries with age in "
            "this sample — chest-wall compliance and respiratory displacement morphology are the "
            "plausible mechanisms, but these data cannot isolate them, and both age extremes in "
            "this cohort were male.",
            "so the association may reflect any subject characteristic that covaries with age in "
            "this sample, and both age extremes in this cohort were male.")
    d.patch("Two cautions apply to the age result.",
            "We therefore report the age relationship as a calibration result validated on "
            "held-out subjects, and as an exploratory physiological observation motivating a "
            "larger cohort.",
            "We therefore report the age relationship as an exploratory observation motivating a "
            "larger cohort, and not as a calibration route.")

    # ─────────────────────────────────────────────── 5. Discussion
    d.set_text(
        "The mask recovers per-session mean respiratory rate with MAE < 1 br/min",
        "The mask reproduces a recording's mean respiratory and cardiac rate once a per-session "
        "scale factor k has been fitted against a reference on that same recording (night-level "
        f"error {r.night_self_k} br/min and {c.night_self_k} BPM). Withholding that reference "
        "removes the advantage: calibrating from the subject's other night or from a population "
        "constant is no better than assigning the cohort median rate, in either band and in "
        "either regime (Table 3). What the sensor demonstrably provides is therefore a rate "
        "signal that a reference can be transferred onto, not a rate measurement that stands on "
        "its own. The cardiac k factor (≈1.95) is consistent across subjects and nights and "
        "reflects a ∼2:1 overcounting ratio arising from the biphasic structure of the "
        "capacitive pulse waveform (systolic peak plus dicrotic notch); the respiratory k is "
        "≈1.00 under the operational pipeline, consistent with one displacement per breath.")

    d.set_text(
        "Two further observations bear on this morphological interpretation of k.",
        "One further observation bears on this morphological interpretation of k. R-peak–"
        "triggered averaging of the capacitive pulse yields 1.70–2.43 peaks per heartbeat "
        "(median 2.02), bracketing the fitted cardiac k of 1.95 and supporting the reading that "
        "k counts capacitive peaks per cardiac cycle. This is the one part of the calibration "
        "story that is measured independently of the rate pipeline; it is an agreement of "
        "central values across nine sessions rather than a covariation (r = 0.50, p = 0.17). The "
        "respiratory k has no such independent measurement, and its relationship to age (§4.2.1) "
        "is a property of the cohort's breathing rates rather than of the sensor.")

    d.patch("The respiratory case has a different basis",
            "At 30-second windows with standard Welch parameters the frequency resolution "
            "(0.25 Hz) is comparable to the width of the entire respiratory band (0.4 Hz), which "
            "quantizes rate estimates;",
            "The Welch estimator used 4-second segments, giving a frequency resolution of "
            "0.25 Hz against a 0.4 Hz respiratory band — two usable bins, which collapses the "
            "estimate to a constant rather than merely quantizing it;")

    d.patch("The mask is suitable for screening-level overnight",
            "The mask is suitable for screening-level overnight respiratory and cardiac rate "
            "monitoring: accurate mean rates per session, with well-characterized calibration "
            "behavior and per-stage accuracy. This could support home sleep apnea screening "
            "(mean-rate anomalies, gross rate extremes) or longitudinal tracking of resting rates "
            "across nights. However, it cannot replace PSG or chest-worn sensors for applications "
            "requiring real-time or instantaneous rate monitoring.",
            "The rate results do not presently support a screening or trending application. Both "
            "uses require an estimate that stands without a same-night reference, and under "
            "cross-night or population calibration the mask does not improve on a cohort "
            "constant. Nor can it replace PSG or chest-worn sensors for real-time or "
            "instantaneous rate monitoring. What the rate analysis does establish is the "
            "morphology of the capacitive cardiac pulse — approximately two deflections per "
            "heartbeat, measured directly against ECG — which is a sensor characterisation "
            "result and constrains how any future estimator on this hardware must be built.")

    d.patch("The cardiac k ≈ 2 and the resulting",
            "The cardiac k ≈ 2 and the resulting ∼4 BPM MAE are consistent with the "
            "ballistocardiographic (BCG) literature, where waveform morphology and mechanical "
            "coupling introduce systematic overcounting that requires calibration. The "
            "respiratory k ≈ 1 reflects the simpler coupling: each breath produces a single "
            "dominant displacement of the temple sensor, unlike the complex cardiac pulse.",
            "The cardiac k ≈ 2 is consistent with the ballistocardiographic literature, in which "
            "waveform morphology and mechanical coupling introduce systematic overcounting that "
            "requires calibration. The respiratory k ≈ 1 is consistent with a single dominant "
            "displacement per breath, though it is a fitted scale factor and not an independent "
            "measurement of that coupling.")

    # ─────────────────────────────────────────────── 6. Limitations
    d.patch("Second, this is a single-site laboratory study.",
            "Second, this is a single-site laboratory study.",
            "Second, recordings were made unsupervised in participants' homes.")
    d.set_text(
        "Third, k-calibration requires a reference cardiac or respiratory rate.",
        "Third, and most consequentially, k-calibration requires a reference rate from the same "
        "recording. Every accuracy figure obtained without one — cross-night k, population k — "
        "is no better than a cohort constant in either band (§4.2, Table 3), so the rate "
        "pipeline as it stands is a validation result rather than a deployable measurement. "
        "Closing this gap requires either a k-free anchor in the capacitive signal itself or a "
        "predictor of k from observable subject characteristics; the self-supervised adaptive-k "
        "approach tested here failed for lack of such an anchor, a ten-minute warm-up "
        "calibration is worse than a population prior (Figure S4), and the respiratory k–age "
        "relationship of §4.2.1 does not supply one.")

    # ─────────────────────────────────────────────── 7. Conclusion
    d.patch("This study provides a rigorous, multi-method characterization",
            "The mask reliably recovers mean respiratory rate (MAE < 1 br/min) and cardiac rate "
            "(MAE < 4 BPM) per session with simple calibration, and its band-restricted spectral "
            "ridges carry physiologically interpretable, statistically significant sleep-stage "
            "associations.",
            "The mask reproduces a recording's mean respiratory and cardiac rate once calibrated "
            "against a reference on that recording, but not when the calibration is held out, "
            "where it does not improve on a cohort constant; within a night it recovers no rate "
            "variation in either band. What the rate analysis establishes positively is the "
            "morphology of the capacitive cardiac pulse, approximately two deflections per "
            "heartbeat measured directly against ECG. Its band-restricted spectral ridges carry "
            "physiologically interpretable, statistically significant sleep-stage associations.")

    d.save(DOC)
    print(f"applied {d.n_patch} edits -> {DOC}")


if __name__ == "__main__":
    main()
