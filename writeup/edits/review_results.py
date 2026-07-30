# -*- coding: utf-8 -*-
"""Replacement Results / Discussion prose for the reviewed manuscript.

Numbers verified against artifacts/mask_phase_c.parquet, artifacts/detB_*.parquet,
reports/rates/mask/table_s1_per_session_two_regime.csv,
analysis/swa_validation/outputs/swa_validation_results.csv and
reports/slow_wave/cap_swa/classifier/loso_ablation.csv.
"""

# ------------------------------------------------------------ 4.1 validation

R41_SURROGATE = (
    "Cross-spectral coherence at the ground-truth rate frequency confirmed physiological "
    "coupling: respiratory coherence was median 0.31 on the average channel, against a canonical "
    "bound of 0.61 measured between two genuine PSG respiratory sensors, and cardiac coherence "
    "was median 0.16 against a canonical bound of 0.27. Phase-randomized surrogate testing "
    "(200 surrogates per epoch, 8,242 of the 9,319 epochs passing the motion and coverage gate) "
    "showed that 14.7% of respiratory epochs and 9.1% of cardiac epochs exceeded the surrogate "
    "null at p < 0.05. Against the 5% of epochs expected to exceed the null by chance, the "
    "respiratory excess is roughly threefold and the cardiac excess roughly twofold, "
    "establishing that the in-band energy is not an artifact of band-limited noise. The "
    "converse is equally informative and bounds the claim: on the majority of individual epochs "
    "the coupling is not separable from the null, so per-epoch coherence should not be used as a "
    "quality gate. Coherence computed within each sleep stage retained the same ordering, with "
    "no stage falling to the null rate."
)

# ------------------------------------------------------------------ 4.2 rates

R42_INTRO = (
    "The sensor was evaluated against PSG in two regimes that differ in the estimator used and "
    "in the quantity certified. The aggregate regime asks whether the mask recovers a subject's "
    "average respiratory and cardiac rate over a whole recording — the quantity relevant to "
    "overnight screening and night-to-night trending — and is scored as the absolute difference "
    "between the mean estimated and mean reference rate for that night. The within-night regime "
    "asks whether the mask follows the rate as it varies during the night, and is scored by the "
    "per-epoch median absolute error and by the correlation between estimate and reference "
    "across the epochs of a night. Both are computed on the same non-overlapping 30-second "
    "windows. They are reported together because they reach opposite conclusions, and because "
    "the distinction determines what the device can be used for."
)

R42_PIPELINE = (
    "Both regimes use the same operational pipeline: loose prominence-based peak counting, fused "
    "across five channels by agreement gating, rescaled by the per-session factor k (§4.2.1) and "
    "smoothed with a causal three-epoch median filter. One estimator choice requires comment "
    "because it materially affects how the respiratory numbers should be read. The Welch "
    "spectral estimator attains a marginally lower per-epoch respiratory error (0.95 versus "
    "0.94 br/min) but is degenerate at this window length: with 4-second Welch segments the "
    "frequency resolution is 0.25 Hz, the 0.1–0.5 Hz respiratory band therefore contains only "
    "two usable bins, and the estimator returned 0.25 Hz — exactly 15 breaths/min — in 99.95% of "
    "all epochs. Its apparent accuracy is the accuracy of the constant 15 br/min, rescaled per "
    "session by a k that is itself fitted to the reference. We therefore report the "
    "non-degenerate fused peak-counting estimator throughout, which matches it on error while "
    "producing a genuinely varying output, and treat the spectral estimator as a baseline rather "
    "than a measurement (Figure S1)."
)

R42_AGGREGATE = (
    "Averaged over a whole night, the mask tracks both rates closely. The night-level error — "
    "the absolute difference between the mean estimated and the mean reference rate for that "
    "recording — had a median of 0.14 br/min for respiration (IQR 0.10–0.41, worst night 0.93) "
    "and 1.60 BPM for the cardiac band (IQR 1.20–2.21, worst night 10.45). Eleven of twelve "
    "nights fell below 3.1 BPM; the single outlier is S6N2, the session in which cardiac "
    "coupling was anomalous and k fell to 0.94. Per-session values for all twelve nights are "
    "given in Table S1 and the agreement plots in Figure 4. This aggregate accuracy is the "
    "quantity a screening or trending application uses, and it is the regime in which the mask "
    "performs well. Two qualifications bound it. First, it is achieved after per-session "
    "k-calibration against the reference, so it certifies that the sensor plus a calibration "
    "constant reproduces the night mean, not that the sensor alone measures it; §4.2.1 "
    "separates these. Second, respiratory rates in this cohort span a narrow range (session "
    "means 14.4–16.8 br/min, SD 0.8 across the twelve nights), so an uncalibrated constant "
    "predictor of 15 br/min would already achieve a night-level error of 0.78 br/min. The "
    "respiratory aggregate result should therefore be read as a calibration result on a "
    "homogeneous cohort; the cardiac result, where session means range more widely, is the "
    "stronger of the two."
)

R42_WITHIN = (
    "Within a night, neither band's rate variation was recovered (Table 3). The per-epoch median "
    "absolute error was 0.94 br/min (IQR 0.85–1.19) for respiration and 3.36 BPM (IQR 2.64–6.62) "
    "for the cardiac band, and a responsive detector combining loose peak counting with Hilbert "
    "instantaneous frequency traded error for temporal resolution without recovering the "
    "variation (1.33 br/min and 3.65 BPM). Estimates correlated with the reference at a median "
    "within-session r = +0.06 for respiration (Wilcoxon p = 0.64 on the twelve per-night values; "
    "positive in 8 of 12 nights) and r = −0.19 for the cardiac band (p = 0.33; positive in 5 of "
    "12). Neither differs from zero, and neither survives comparison with a temporal shuffle "
    "null (Figure S3). This is not entirely a limitation of the mask: two physically independent "
    "PSG respiratory sensors — nasal airflow and the respiratory inductance belt sum — agree "
    "with each other at only r = 0.47 (IQR 0.33–0.67) on the same epochs, which bounds what any "
    "respiratory sensor can achieve at this time resolution. But the mask reaches a small "
    "fraction of that bound, and for the cardiac band no comparable external bound applies."
)

R42_MECHANISM = (
    "The two bands fail to track for different reasons, and the amount of variation available to "
    "be tracked distinguishes them. Within a night the reference respiratory rate varies with a "
    "median standard deviation of 1.57 br/min, against a per-epoch error of 1.33 br/min for the "
    "responsive detector — the signal to be resolved is barely larger than the error, so little "
    "variation is recoverable in principle and a near-constant estimate is close to optimal. The "
    "cardiac band is different: reference heart rate varies with a median standard deviation of "
    "6.58 BPM, comfortably above the 3.65 BPM per-epoch error, so the variation is in principle "
    "resolvable and is nonetheless not recovered. We had originally expected the ratio of error "
    "to available variation to separate the two bands; it does not, being 0.85 for respiration "
    "and 0.55 for the cardiac band on the same estimator, which is why we report the underlying "
    "quantities rather than the ratio. The cardiac failure is therefore mechanistic rather than "
    "a matter of insufficient signal, and is discussed in §5.3. An exhaustive comparison of "
    "estimators and channels in both regimes, including CWT-ridge and STFT–Viterbi trackers, is "
    "given in Figure S1 and Figure S3."
)

TABLE3_CAPTION = (
    "Table 3. Rate detection in both regimes, from the operational pipeline (agreement-gated "
    "five-channel fusion of loose peak counting, per-session k, causal three-epoch median "
    "filter). Values are medians across the 12 nights with the interquartile range in brackets; "
    "per-night values are in Table S1. The night-level error is |mean(estimate) − "
    "mean(reference)| for a recording; the per-epoch error is the median absolute error across "
    "that recording's epochs. Within-session r is the Pearson correlation between estimate and "
    "reference across the epochs of a night, with a Wilcoxon signed-rank test on the twelve "
    "per-night values."
)

TABLE_3 = [
    ["", "Respiratory", "Cardiac"],
    ["Aggregate regime (per-night mean)", "", ""],
    ["Night-level error, median [IQR]", "0.14 br/min [0.10–0.41]", "1.60 BPM [1.20–2.21]"],
    ["Night-level error, worst night", "0.93 br/min", "10.45 BPM"],
    ["Within-night regime", "", ""],
    ["Per-epoch error, operational pipeline", "0.94 br/min [0.85–1.19]", "3.36 BPM [2.64–6.62]"],
    ["Per-epoch error, responsive detector", "1.33 br/min [1.11–1.51]", "3.65 BPM [2.97–7.42]"],
    ["Within-session r", "+0.06 (p = 0.64)", "−0.19 (p = 0.33)"],
    ["Nights with r > 0", "8 / 12", "5 / 12"],
    ["Reference rate SD within night", "1.57 br/min", "6.58 BPM"],
    ["Independent-sensor bound", "r = 0.47 [0.33–0.67]", "not available"],
    ["Calibration", "", ""],
    ["Calibration factor k, median [IQR]", "0.96 [0.91–1.02]", "1.95 [1.79–2.00]"],
    ["Calibration factor k, range", "0.90–1.05", "0.94–2.23"],
]

FIG4_CAPTION = (
    "Figure 4. Bland–Altman agreement for respiratory (left) and cardiac (right) rate. Each "
    "point is one 30-second analysis epoch, pooled across the twelve recordings; the solid line "
    "is the bias and the dashed lines the 95% limits of agreement (respiratory bias −0.3 br/min, "
    "limits −4.7 to +4.2; cardiac bias −0.6 BPM, limits −24.1 to +22.9). The spread shown here "
    "is epoch-level and is the subject of the within-night regime; the aggregate regime of "
    "Table 3 concerns the per-night means, whose errors are 0.14 br/min and 1.60 BPM and are "
    "not visible at this scale."
)

# ------------------------------------------------------------------- 4.2.1 k

R421_MORPHOLOGY = (
    "k is a waveform-morphology count. Averaging the capacitive signal triggered on ECG R-peaks "
    "resolves the per-beat pulse shape directly, and counting capacitive peaks against ECG beats "
    "gives 1.70–2.43 peaks per heartbeat (median 2.02 on CRE, 1.92 on CLE−CRE) across the nine "
    "of twelve sessions in which cardiac peak detection was reliable enough to count; the "
    "remaining three were excluded, including S6N2, whose ECG channel had failed and for which "
    "photoplethysmography served as the rate reference. This brackets the fitted cardiac k of "
    "1.95 and identifies its origin: the capacitive cardiac pulse is biphasic, contributing a "
    "systolic peak and a dicrotic notch to each cardiac cycle, so a peak-counting estimator "
    "reports approximately twice the true rate. The correspondence is population-level rather "
    "than exact per subject — the correlation between measured peaks-per-beat and fitted k is "
    "r = 0.50, which at n = 9 is not distinguishable from zero (p = 0.17), so the morphological "
    "account is supported by the agreement of the central values rather than by covariation "
    "across subjects. The respiratory k of ≈0.96 reflects the simpler coupling: each breath "
    "produces one dominant temple displacement."
)

R421_REPRO = (
    "k is reproducible within a subject. Across each subject's two nights the absolute change in "
    "k had a median of 0.013 for respiration (maximum 0.047) and 0.151 for the cardiac band "
    "(maximum 0.406); respiratory k changed by ≤0.03 between nights in four of six subjects and "
    "cardiac k in one of six. Respiratory k is therefore effectively a subject-level constant, "
    "and cardiac k is stable apart from the S6 coupling anomaly, whose two nights account for "
    "the full cardiac range of 0.94–2.23 while the central half of the distribution spans only "
    "1.79–2.00. Diagnostic estimates from 50 random windows agreed with whole-night values to "
    "within 0.04."
)

R421_AGE = (
    "k, age, and the calibration a deployment would require. Because k reflects the mechanical "
    "coupling between physiology and sensor, we asked whether it varies with subject age "
    "(Figure 5). Respiratory k declined across the six subjects, from 1.04 in the youngest "
    "(25 years) to 0.91 in the oldest (66 years), Spearman ρ = −0.83. At n = 6 we evaluate this "
    "with an exact permutation test over all 720 orderings, which gives p = 0.058; the "
    "asymptotic t-approximation would give p = 0.042, and we report the exact value because the "
    "approximation is unreliable at this sample size. The direction of the relationship was "
    "stable: dropping each subject in turn left ρ between −1.00 and −0.70. Cardiac k showed no "
    "such relationship (ρ = +0.37, exact p = 0.47), and unlike the respiratory case the cardiac "
    "result was not stable under the same test — leave-one-out ρ ranged from −0.10 to +0.90 — so "
    "these data neither establish nor exclude an age dependence for cardiac k. Neither factor "
    "was associated with PSQI (p > 0.19), and no other capacitive signal feature we examined "
    "varied with age (Figure S5)."
)

R421_LOSO = (
    "Because the correlation alone is not significant at this sample size, we tested the "
    "respiratory relationship in the way a deployment would use it: by fitting k from age on "
    "five subjects and predicting the sixth. Leave-one-subject-out, an age-based prior predicted "
    "respiratory k with a mean absolute error of 0.020, against 0.056 for no calibration "
    "(k = 1.0) and 0.056 for the best constant prior (the population mean). Age therefore "
    "carries information about respiratory k that a constant prior does not, and this held-out "
    "test — not the correlation coefficient — is what supports the claim. The same test on "
    "cardiac k gave the opposite result: an age prior (0.387) was worse than the population mean "
    "(0.305), consistent with cardiac k being set by a fixed pulse morphology rather than by an "
    "age-modulated variable. Translated into rate error, per-session calibration gives 0.94 "
    "br/min for respiration, a fixed k = 1.0 costs only 0.20–0.28 br/min more, and cardiac rate "
    "requires whole-night self-calibration or a population prior (3.36 versus 4.56 BPM; "
    "Figure S4)."
)

R421_CAUTION = (
    "Three cautions apply to the age result. With six subjects, age is perfectly confounded with "
    "subject identity, so the association may reflect any subject characteristic that covaries "
    "with age in this sample — chest-wall compliance and respiratory displacement morphology are "
    "the plausible mechanisms, but these data cannot isolate them, and both age extremes in this "
    "cohort were male. The exact correlation p of 0.058 does not reach the conventional "
    "threshold before correction, and does not survive correction for the four correlations "
    "tested (each factor against age and PSQI; Bonferroni p ≈ 0.23). And the leave-one-subject-"
    "out prediction, while genuinely held out, is a comparison of three numbers on six subjects. "
    "We therefore report the age relationship as a calibration result validated on held-out "
    "subjects, and as an exploratory physiological observation motivating a larger cohort, not "
    "as an established age effect."
)

FIG5_CAPTION = (
    "Figure 5. Calibration factor k and subject age. (A) Respiratory k per subject (mean of two "
    "nights; bars span the two nights; circles male, squares female) against age, with "
    "least-squares fit. k falls from 1.04 to 0.91 across the age range; the leave-one-out range "
    "of ρ is given above the panel. (B) Cardiac k against age. Values cluster near 1.95 with no "
    "stable trend; S6 is a coupling outlier (k = 1.14 averaged over its two nights) whose "
    "inclusion or exclusion reverses the sign of ρ. (C) Leave-one-subject-out prediction of "
    "respiratory k: absolute error per held-out subject using an age prior fitted on the other "
    "five, against using no calibration (k = 1.0)."
)
