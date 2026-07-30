# -*- coding: utf-8 -*-
"""Replacement prose, part 3: Discussion, Limitations, Conclusion, Supplementary."""

# ------------------------------------------------------------- 5. discussion

D51_RATES = (
    "The mask recovers each night's mean respiratory rate to a median of 0.14 br/min and its "
    "mean cardiac rate to 1.60 BPM after a per-session k-calibration, with a per-epoch error of "
    "0.94 br/min and 3.36 BPM. The respiratory k factor is near unity (k ≈ 0.96), meaning one "
    "dominant temple displacement per breath and negligible correction. The cardiac k factor "
    "(≈1.95) is consistent across subjects and nights — its interquartile range spans only "
    "1.79–2.00 — reflecting the ∼2:1 overcounting that arises from the biphasic structure of the "
    "capacitive pulse (systolic peak plus dicrotic notch). The k calibration is stable: "
    "diagnostic estimates from 50 random windows agree with whole-night values to within 0.04, "
    "and respiratory k changed by ≤0.03 between a subject's two nights in four of six subjects. "
    "Two caveats govern how these accuracies should be read. They are conditional on "
    "calibration against a reference, and for respiration the cohort's narrow range of mean "
    "rates (14.4–16.8 br/min) means a constant predictor already achieves 0.78 br/min, so the "
    "respiratory aggregate figure demonstrates calibration rather than measurement; the cardiac "
    "figure, spanning a wider range of true rates, is the stronger result. Direct comparison "
    "with other non-contact and wearable cardiac sensors is complicated by the fact that most "
    "report per-epoch rather than per-night error, and we are not aware of a published dataset "
    "using this window length and error definition against which these values could be matched."
)

D51_K = (
    "Two further observations bear on this morphological interpretation of k. First, R-peak–"
    "triggered averaging of the capacitive pulse yields 1.70–2.43 peaks per heartbeat (median "
    "2.02), bracketing the fitted cardiac k and supporting the reading that k counts capacitive "
    "peaks per cardiac cycle; the agreement is between central values, since the across-subject "
    "correlation between measured and fitted k (r = 0.50, n = 9) is not itself significant. "
    "Second, the two bands' calibration factors relate to subject age in opposite ways: "
    "respiratory k declines with age in a direction stable to leaving out any single subject, "
    "and an age-based prior predicts it better than the best constant prior on held-out "
    "subjects, whereas cardiac k shows no stable age relationship and is predicted better by a "
    "population constant than by age. Neither the respiratory correlation (exact p = 0.058) nor "
    "the cardiac null is established at six subjects — the cardiac leave-one-out ρ ranges from "
    "−0.10 to +0.90 — but the pair is what a fixed pulse morphology and an age-modulated "
    "mechanical coupling would respectively produce."
)

D51_RIDGES = (
    "Band-restricted ridge features show physiologically interpretable stage associations. An "
    "infra-slow ridge near 0.07 Hz loses power during N3 in all six subjects, respiratory-band "
    "ridges are fewer and lower in power during N3, and intermittent cardiac-band ridges slow "
    "from ≈68 to ≈60 BPM in N3, mirroring the sleep bradycardia of deep NREM. These effects "
    "track autonomic and respiratory state rather than cortical activity. The infra-slow effect "
    "is the most consistent and is the spectral expression of the same phenomenon the mechanical "
    "ISWA score captures in the time domain (§4.6); the others are weaker and, while too weak "
    "for standalone staging, the set could contribute to multi-modal approaches."
)

D52_TRACKING_HEAD = "Within-night rate variation"

D52_TRACKING = (
    "The mask does not follow either rate as it changes during the night. Across the seven base "
    "estimators, four channels, multi-channel fusion, CWT-ridge and STFT–Viterbi trackers, with "
    "and without calibration and smoothing, no configuration produced a within-session "
    "correlation distinguishable from a temporal shuffle null in either band (median r = +0.06 "
    "for respiration and −0.19 for the cardiac band). Because the battery was exhaustive, we "
    "treat this as a property of the signal at this window length rather than of a particular "
    "estimator. The consequence for deployment is concrete: the device can report a nightly "
    "average and its night-to-night trend, but must not be used where instantaneous or "
    "beat-to-beat rate matters."
)

D52_CORTICAL = (
    "The mask carries no cortical electrographic signature. Sleep spindles — the electrographic "
    "hallmark of N2 — produce no spindle-locked sigma response in any capacitive channel "
    "(trial-averaged sigma shift +0.02 to +0.03 dB), whereas the same measurement on the "
    "contact-EEG channel rises +3.3 dB at spindle centres. Capacitive slow-wave activity is "
    "likewise uncorrelated with EEG slow-wave activity (r = −0.014 ± 0.036) and discriminates N3 "
    "at chance (AUC 0.490 ± 0.041) where the same pipeline on EEG reaches 0.740. What the mask "
    "does register at these events is mechanical: a low-frequency (0–3 Hz) increase of +0.47 to "
    "+0.58 dB at spindle centres, and a sharp zero-lag rise at delta-burst onsets in all six "
    "subjects. The sensor therefore transduces mechanical and hemodynamic pulsations, not "
    "neuronal electrical activity, and EEG-like interpretations of the capacitive signal at this "
    "placement are not supportable."
)

D53_CARDIAC = (
    "The absence of within-night cardiac tracking can be understood through the k factor. The "
    "consistent k ≈ 2 across subjects indicates that the capacitive cardiac waveform contains "
    "two inflection points per heartbeat — most likely the systolic and dicrotic pressure peaks. "
    "The dominant frequency of this biphasic waveform is determined by its stable morphology "
    "rather than by the instantaneous heart rate: when heart rate varies within a session the "
    "waveform stretches or compresses, but the peak-counting frequency remains governed by the "
    "persistent biphasic structure. Only changes large enough to alter the waveform's "
    "fundamental peak structure, rather than its period, would produce a frequency shift "
    "detectable by the estimators tested. The same mechanism appears independently in §4.6, "
    "where the capacitive cardiac peak frequency rose during deep sleep in all six subjects "
    "while the photoplethysmographic peak stayed flat — the capacitive peak decoupling from true "
    "heart rate exactly as this account predicts. This is why the cardiac band shows ample "
    "variation to resolve and none of it recovered."
)

D53_RESP = (
    "The respiratory case has a different basis. At the 30-second analysis window with 4-second "
    "Welch segments the frequency resolution is 0.25 Hz, which is comparable to the width of the "
    "entire 0.1–0.5 Hz respiratory band; the spectral estimator therefore has only two reachable "
    "output values and returned 0.25 Hz in 99.95% of epochs, making it a constant predictor "
    "rather than a measurement. Peak-counting estimators are not resolution-limited in this way "
    "and do vary, but the variation they are asked to resolve is itself small — a median "
    "within-night standard deviation of 1.57 br/min against a per-epoch error of comparable "
    "magnitude. Longer windows and full-window periodograms with parabolic interpolation "
    "recover resolution at the cost of temporal responsiveness without improving the "
    "within-session correlation, so the limitation is a genuine trade-off rather than a poor "
    "parameter choice."
)

D53_COUPLING = (
    "The subject-dependence of the weaker ridge effects likely reflects individual differences "
    "in how cardiorespiratory coupling reorganizes during deep sleep. The capacitive sensor sits "
    "at the temple, receiving a mix of intracranial pressure, superficial temporal artery "
    "pulsation, and near-field respiratory displacement. The relative contribution of each "
    "component varies with individual anatomy, sensor placement, and mask fit. During N3 cardiac "
    "output decreases and respiratory mechanics change; how these changes project onto the "
    "temple sensor depends on the subject-specific coupling geometry. That the infra-slow ridge "
    "effect and the mechanical ISWA score are nonetheless consistent in all six subjects "
    "suggests they depend less on this geometry than the band-specific spectral features do."
)

D54_CLINICAL = (
    "The mask is suitable for screening-level overnight monitoring of average respiratory and "
    "cardiac rate: accurate per-night means, with well-characterized calibration behaviour. This "
    "could support longitudinal tracking of resting rates across nights, or flagging gross rate "
    "extremes. It cannot replace PSG or chest-worn sensors where real-time or instantaneous rate "
    "is required. We note explicitly that although apnea and hypopnea events were scored in "
    "these recordings, no apnea-detection analysis is reported here, and the suitability of the "
    "mask for sleep-apnea screening is therefore untested; the event-level respiratory "
    "sensitivity such an application needs is precisely the within-night resolution the mask "
    "lacks. Similarly, the ocular electrodes register eye movements, but no eye-movement or REM "
    "analysis is reported in this work."
)

D54_STAGING = (
    "The stage-associated structure, while too weak for standalone staging, could contribute "
    "features to a multi-modal system combining accelerometer-derived body position and "
    "movement, rate statistics, and slow capacitive drift. The mechanical ISWA score is the most "
    "promising of these, being the only capacitive N3 marker whose direction held in all six "
    "subjects, and it operates from mask-derived signals alone. At a per-subject AUC of ≈0.68 it "
    "is an input to a staging system, not a stager."
)

D55_PRIOR = (
    "The cardiac k ≈ 2 and the resulting per-epoch error are consistent in kind with the "
    "ballistocardiographic literature, where waveform morphology and mechanical coupling "
    "introduce systematic overcounting that requires calibration, although we make no "
    "quantitative comparison because reported error definitions and window lengths differ. The "
    "respiratory k ≈ 1 reflects the simpler coupling: each breath produces a single dominant "
    "displacement of the temple sensor, unlike the complex cardiac pulse. The "
    "resolution-versus-responsiveness trade-off that limits epoch-level respiratory precision is "
    "a known constraint in wearable respiratory monitoring."
)

# ------------------------------------------------------------ 6. limitations

L1 = (
    "First, the sample is small (6 subjects, 12 nights). While sufficient for per-session rate "
    "characterization and for the clear negative results on cortical electrical signatures, it "
    "limits between-subject generalization. Every subject-level statistic in this paper rests on "
    "six values, a Wilcoxon test on six subjects floors at p = 0.031, and the age relationship "
    "reported in §4.2.1 does not reach significance on an exact test. Confirming the "
    "ridge–stage associations, the ISWA score, and the age relationship would require a larger "
    "cohort."
)

L2 = (
    "Second, all recordings were made unattended in participants' own homes rather than in a "
    "sleep laboratory. This is the intended deployment setting and is a strength for ecological "
    "validity, but it means sensor placement was performed by participants after training rather "
    "than by a technologist, and coupling quality could not be monitored during the night. "
    "Sensor coupling may vary further with head shape, hair density, and mask fit. The S6 "
    "outlier behaviour — anomalous cardiac k of 1.35 and 0.94, and the only night-level cardiac "
    "error above 3.1 BPM — illustrates how strongly results depend on coupling quality, and a "
    "deployable system would need an automatic coupling-quality check."
)

L3 = (
    "Third, k-calibration requires a reference cardiac or respiratory rate. While k is stable "
    "within and across nights once estimated, a deployment would need either a calibration "
    "period against a reference or a population-level prior; every accuracy figure reported here "
    "is conditional on that calibration. The near-unity respiratory k makes respiratory rate "
    "effectively calibration-free, and an age-based prior for respiratory k improves on a "
    "constant prior when validated on held-out subjects (§4.2.1). The cardiac band has no such "
    "shortcut: a ten-minute warm-up calibration is worse than a population prior because twenty "
    "epochs give a noisy and drifting estimate (Figure S4), so a deployment would use a "
    "population prior or whole-night self-calibration."
)

L4 = (
    "Fourth, the respiratory consensus ground truth was derived from the same PSG system as the "
    "reference. Truly independent respiratory validation would require a separate sensing "
    "modality such as capnography or acoustic respiration monitoring. The Flow-versus-RIPSum "
    "comparison mitigates but does not remove this, since both belong to the same recorder."
)

L5 = (
    "Fifth, the 30-second analysis window was chosen for compatibility with PSG staging epochs, "
    "and it is short enough that the Welch spectral estimator is resolution-limited in the "
    "respiratory band (§5.3). Longer windows decrease per-epoch error but do not improve "
    "tracking correlation, suggesting the within-night limitation is not an artifact of window "
    "choice; nonetheless the specific numbers reported here are tied to this window length."
)

L6 = (
    "Sixth, the montage derivation of our single-channel contact EEG is not documented. It "
    "nonetheless served as a positive control in which the spindle sigma rhythm was clearly "
    "present (+3.3 dB at spindle centres) and N3 was recovered at an AUC of 0.740, so its exact "
    "derivation does not affect the capacitive findings reported here."
)

L7 = (
    "Seventh, several analysis choices were made after inspecting these data: the CRE channel "
    "was selected for the ridge analyses because it carried the dominant ridge in 9 of 12 "
    "sessions, and the cardiac estimator and fusion strategy were chosen by error on this same "
    "cohort. The mechanical ISWA definition and its 0.60 threshold were fixed a priori and "
    "evaluated leave-one-subject-out, but the ridge and rate results are descriptive of this "
    "cohort and would need out-of-sample confirmation."
)

# ------------------------------------------------------------- 7. conclusion

C1 = (
    "This study provides a multi-method characterization of a capacitive temple-sensor sleep "
    "mask during overnight sleep, and its value lies as much in the boundaries it establishes as "
    "in the capabilities it demonstrates. The mask reliably recovers per-night mean respiratory "
    "rate (0.14 br/min) and cardiac rate (1.60 BPM) with simple calibration, but does not "
    "follow either rate as it varies within the night, and we trace that failure to the biphasic "
    "morphology of the capacitive pulse rather than to insufficient signal. The sensor carries "
    "no cortical electrographic signature: sleep spindles produce no sigma-band response and "
    "capacitive slow-wave activity is uncorrelated with EEG slow-wave activity, confirming that "
    "the sensor transduces mechanical and hemodynamic pulsations rather than neuronal electrical "
    "activity."
)

C2 = (
    "What it does carry is a mechanical account of the same physiology. Low-band capacitive "
    "power rises at sleep-spindle centres and, with zero lag and in every subject, at EEG "
    "delta-burst onsets; an infra-slow spectral ridge weakens during N3 in all six subjects; and "
    "a mechanical intracranial slow-wave activity score built from slow capacitive drift under "
    "sustained quiescence discriminates N3 above chance in all six subjects, where every "
    "spectral capacitive feature we examined failed to generalize. These findings frame the mask "
    "as a viable tool for unobtrusive overnight mean-rate trending and as one input to "
    "multi-modal sleep assessment, and they identify the mechanical ISWA signature — not any "
    "EEG-like spectral quantity — as the direction in which capacitive temple sensing should be "
    "developed."
)

# ---------------------------------------------------------- supplementary S1

TABLE_S1_CAPTION = (
    "Table S1. Per-session rate accuracy. Sessions are labelled SxNy for subject x, night y. k "
    "is the per-session calibration factor. \"Night\" is the per-epoch median absolute error of "
    "the operational pipeline within that session, and \"epoch\" the same quantity for the "
    "responsive multi-channel detector; r is the within-session Pearson correlation between "
    "estimate and reference across that session's epochs. The aggregate night-level errors of "
    "Table 3 — |mean(estimate) − mean(reference)| per recording — are given in the final two "
    "columns and are a different and much smaller quantity, as explained in §4.2."
)

S1_WRITEUP = (
    "The spectral estimator attains the lowest nominal per-epoch respiratory error on every "
    "channel and the loose peak-counting estimator the lowest cardiac error, but the respiratory "
    "spectral figure is not a measurement: at 0.25 Hz resolution the estimator has two reachable "
    "values in the respiratory band and returns 0.25 Hz in 99.95% of epochs, so its error is "
    "that of a constant predictor rescaled by a fitted k. Among non-degenerate estimators no "
    "channel was reliably better than the CLE−CRE differential for respiration "
    "(oracle-over-channels 1.08 versus 1.09 br/min pooled), whereas the cardiac band did show "
    "channel-dependent headroom, which motivated the multi-channel detector. Multi-channel "
    "fusion changed the per-session median error by less than 0.1 in both bands (respiratory "
    "0.95 → 0.94 br/min; cardiac 3.42 → 3.36 BPM) but is non-degenerate, which is why it and not "
    "the spectral estimator is reported as the operational pipeline in §4.2."
)

S2_WRITEUP = (
    "Respiratory error was highest in REM (≈2.31 br/min) and cardiac error highest in Wake "
    "(≈4.73 BPM), consistent with the irregular breathing of REM and the greater heart-rate "
    "variability of wakefulness. These comparisons pool epochs across sessions and were not "
    "tested for significance, both because the six-subject sample does not support it and "
    "because pooled epochs are not independent; they are reported as descriptive context for the "
    "aggregate errors in Table 3."
)

S3_WRITEUP = (
    "An identical battery was applied to both bands: seven base estimators, multi-channel mean "
    "fusion, CWT-ridge tracking, and STFT–Viterbi tracking, each with and without k-calibration "
    "and smoothing. No configuration produced a within-session correlation distinguishable from "
    "the shuffle null in either band — 4 of 12 nights exceeded the null 95th percentile for "
    "respiration and 3 of 12 for the cardiac band, close to what the null itself produces. For "
    "scale, the two physically independent PSG respiratory sensors agree with each other at "
    "r = 0.47 (0.27 on detrended fluctuations) on these same epochs, so the achievable target in "
    "this regime is modest even for a contact sensor. The exhaustiveness of this battery is what "
    "supports treating the within-night result as a property of the signal rather than of a "
    "particular estimator."
)

S5_WRITEUP = (
    "No capacitive feature other than respiratory k varied with age. The night-to-night "
    "reproducibility of k was itself age-independent in both bands (ρ = −0.03, p = 0.96), "
    "confirming that older subjects' k values are not merely noisier estimates; band SNR, DC "
    "drift, and accelerometer activity were likewise unrelated to age (|ρ| ≤ 0.37, p ≥ 0.47). "
    "The modal cardiac peak frequency showed the largest nominal association (ρ = +0.75, exact "
    "p = 0.096), but this variable takes only four distinct values across the six subjects and "
    "sits at or just above the 0.5 Hz lower edge of the cardiac analysis band for most of them; "
    "the same edge-pinning applies to the respiratory modal peak, which equals the 0.1 Hz band "
    "floor in five of six subjects. Neither is interpretable as a physiological age effect. This "
    "panel is included as a bounding null: it establishes that the respiratory-k finding is not "
    "one of many age correlations in these data."
)

S6_CAPTION = (
    "Figure S6. Harmonic-ladder validation (CH channel). Left: per-session count of strong "
    "(≥3-member) integer-ratio ladders, split into those exceeding the per-session surrogate "
    "null (green) and those rejected (red). Middle: confidence distribution of the surviving "
    "ladders. Right: distribution of ladder fundamentals, which concentrates at the respiratory "
    "rate."
)

S6_WRITEUP = (
    "Of 6,812 windows containing a strong integer-ratio ladder, 6,783 (99.6%; 98.9–100% per "
    "session) exceeded a per-session, per-count surrogate null built by drawing 400 random "
    "frequency sets over each session's observed ridge-frequency range. The surviving ladders "
    "had a median fundamental of 0.25 Hz — the respiratory rate — a median of four members "
    "extending into the cardiac band, and a median confidence of 0.75. Integer alignment at this "
    "rate does not occur by chance, so the harmonic structure of the capacitive spectrum "
    "reflects a genuinely non-sinusoidal respiratory waveform rather than coincidental overlap "
    "of independent rhythms."
)

S7_CAPTION = (
    "Figure S7. Capacitive versus contact-EEG slow-wave activity. Per-session ROC curves for "
    "discriminating technologist-scored N3 using capacitive SWA (grey) and using the identical "
    "pipeline applied to the contact EEG (black). The EEG curves confirm the pipeline recovers "
    "N3 (AUC 0.740 ± 0.056) while the capacitive curves sit at chance (0.490 ± 0.041)."
)

S7_WRITEUP = (
    "Capacitive SWA was computed exactly as EEG SWA — band power in 0.5–4 Hz and in its four "
    "constituent sub-bands per 30-second epoch — and compared within each session. The mean "
    "within-session Pearson correlation between capacitive and EEG SWA was −0.014 ± 0.036, with "
    "every session between −0.10 and +0.05, and mean magnitude-squared coherence was 0.003 ± "
    "0.005. N3 discrimination from capacitive SWA was at chance (AUC 0.490 ± 0.041, below 0.5 in "
    "eight of twelve sessions), while the same pipeline on the contact EEG reached 0.740 ± 0.056 "
    "and exceeded the capacitive value in all six subjects. This is the control that licenses "
    "the interpretation throughout the paper: the capacitive delta-band power that a spectral "
    "analysis reports is not of cortical origin, which is why the slow-wave correlate the sensor "
    "does carry had to be defined mechanically (§4.6)."
)
