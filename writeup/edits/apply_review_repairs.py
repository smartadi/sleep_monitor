"""Apply the full repair list from the author / editor / reviewer passes.

  in  : writeup/main/CAP_sleep_mask_manuscript_main.docx   (edited in place)
  bak : writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_REPAIRS_20260813.docx

  A  contradictions      -- stale 3.5 tail, two-vs-three bands, orphan 5.3
                            paragraph, stale 5.5 number, 6-Third, mixed error
                            definitions, the unsupported common-mode clause,
                            CAP/SEC naming
  B  extrapolations      -- imbalance null stated as independence, mechanism
                            stated as fact, continuity read as accuracy,
                            band-edge cardiac frequency, pooled-epoch p-values
  C  language            -- long sentences split, ASCII placeholders replaced,
                            self-praise cut, Discussion de-duplicated
  D  numbers and figures -- k-vs-age recomputed on the fused pipeline, the
                            representative night moved off the outlier session,
                            cohort grids added as supplementary
  E  scope               -- title and Introduction aligned with what the study
                            measured

Run from the repo root:  python writeup/edits/apply_review_repairs.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_review_edits import Doc, W  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_REPAIRS_20260813.docx")

FIG_RECORD_S1N1 = Path("writeup/figures/channel_evolution/S1N1_CH_CLE-CRE.png")
FIG_GRID_CHANNELS = Path("writeup/figures/channel_evolution/ch_vs_clecre_grid.png")
FIG_GRID_IMBALANCE = Path("writeup/figures/imbalance/imbalance_grid.png")
FIG_K_AGE = Path("writeup/figures/k_biomarker/fig_k_vs_age_3panel.png")


def main() -> None:
    if not DOC.exists():
        raise SystemExit("missing: %s" % DOC)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ==================================================================== E
    # E1. The title claimed intracranial monitoring the study cannot support:
    # there is no intracranial reference measurement anywhere in this work, and
    # the paper's own conclusion is that the sensor transduces mechanical and
    # hemodynamic pulsation at the temple.
    d.set_text(
        "Noninvasive Monitoring of Sleep-Related Intracranial Physiological Dynamics",
        "Overnight Monitoring of Respiratory, Cardiac and Mechanical Signals with a "
        "Wearable Capacitive Sensor Mask")

    # E2. The Introduction promised an ISWA metric that Results never defines or
    # reports, and promised spectral signatures that "distinguish deep sleep",
    # which 4.3 explicitly says are too weak to do so.
    d.set_text(
        "In this study, we introduce intracranial slow-wave activity (ISWA)",
        "In this study we characterise what a wearable SEC sleep mask measures "
        "across a full night, and what it does not. Healthy participants underwent "
        "overnight polysomnography while wearing the mask. We first establish what "
        "the raw sensor record contains, then quantify how accurately respiratory "
        "and cardiac rate can be recovered from it, and finally ask whether the "
        "signal carries any correlate of cortical activity by testing it against "
        "two discrete, precisely timed cortical events: sleep spindles and "
        "delta-burst onsets. The answer to the last question is negative and "
        "defines the scope of the technology. We therefore report the mask as a "
        "monitor of mechanical and hemodynamic signals, and state the boundary "
        "beyond which capacitive temple sensing should not be interpreted.")

    # Section 2 asserted a measurement the Results never report.
    d.patch("The regional ICP is measured by using the single electrode",
            "The regional ICP is measured by using the single electrode capacitive "
            "sensing mechanism. 22",
            "Regional ICP is the intended target of the single-electrode capacitive "
            "sensing mechanism, which has been characterised against invasive "
            "pressure in previous work.22 No intracranial reference measurement was "
            "available in the present study, so the sections below report the "
            "capacitive signal itself rather than a derived pressure.")

    # ==================================================================== A + C
    # A8 / C. Naming: the paper calls the sensor SEC everywhere except three
    # methods sentences left over from an earlier draft.
    d.patch("Motion artifact was suppressed by regressing",
            "The accelerometer magnitude and the CAP channel",
            "The accelerometer magnitude and the SEC channel")
    d.patch("Before estimating rates, we established",
            "physically present in the CAP signal", "physically present in the SEC signal")
    d.patch("SEC peak counts are systematically scaled",
            "the median ratio between the CAP estimate", "the median ratio between the SEC estimate")

    # C. 3.4 tracker method: one 90-word sentence split into four.
    d.set_text(
        "Ridge tracking of the two rhythms.",
        "Ridge tracking of the two rhythms. Each rhythm was followed as a single "
        "continuous trace rather than estimated window by window. The trace is the "
        "Viterbi path through the spectrogram that maximises spectral amplitude "
        "minus a penalty on frequency change, searched inside a physiological band "
        "(respiration 0.25–0.55 Hz; cardiac 0.85–1.45 Hz, an upper bound of about "
        "87 BPM). The spectrogram was first background-removed: each time column "
        "had a frequency-median-filtered copy of itself subtracted, which flattens "
        "the 1/f envelope and broadband motion brightening but leaves narrow peaks "
        "intact. Because the path is global and continuous, each rhythm has exactly "
        "one value at any instant and cannot hop to a subharmonic. Each panel also "
        "reports the tracker's confidence, the fraction of windows in which the "
        "tracked band rises clearly above the column median.")

    # C. 3.4 record and imbalance paragraphs: shorter sentences, real typography.
    d.set_text(
        "Overnight evolution of the sensor value.",
        "Overnight evolution of the sensor value. Each channel was tracked across "
        "the whole night as the mean over consecutive 10-second blocks, in "
        "femtofarads, referenced to the session mean. That mean is printed on every "
        "panel so the absolute operating point is not lost. The same block grid "
        "carries the per-block variance and the head-turn angle. The angle was "
        "obtained by isolating gravity with a 0.05 Hz low-pass and computing the "
        "turn over the full ±180° range, so that rotation past pure side-lying is "
        "representable. Each night's trace was then re-wrapped about its circular "
        "median, so a brief crossing of the ±180° seam does not appear as a "
        "night-wide swing.")

    d.set_text(
        "Left-right capacitance imbalance.",
        "Left–right capacitance imbalance. The difference between the two temple "
        "electrodes carries a slowly varying component. We quantify it with a "
        "night-scale marker: a short median filter to remove spikes, then a rolling "
        "mean over τ = 30 minutes, with motion epochs excluded. Applying this to the "
        "signed difference gives the imbalance; applying it to the absolute "
        "difference gives the envelope. Both are reported in fF. The mask's "
        "placement sets a large static offset on this difference, so two controls "
        "are reported beside it: the association with head-turn angle, and the "
        "behaviour of the marker within supine epochs alone.")

    # C. 3.7 was a single 270-word paragraph. Same content, shorter sentences.
    d.set_text(
        "All rate accuracy metrics are reported within-session",
        "All rate accuracy metrics are reported within-session, to avoid inflation "
        "from between-session mean matching. Two error definitions are used and kept "
        "distinct throughout. The night-level error is the absolute difference "
        "between the mean estimated and the mean reference rate over a whole "
        "recording. The per-epoch error is the median absolute error across the "
        "epochs of a recording. Cross-session summaries use the median and "
        "interquartile range of per-session values, so the unit of analysis is the "
        "recording, not the epoch. Within-session association uses the Pearson "
        "correlation across the epochs of a night. These are summarized across "
        "nights by the median and a Wilcoxon signed-rank test on the twelve "
        "per-night values. Group comparisons use the Kruskal–Wallis test across "
        "stages, with the Mann–Whitney U test for N3-versus-rest contrasts. Where "
        "those tests are computed on pooled epochs, the epochs within a night are "
        "not independent and the p-values are anti-conservative. We therefore report "
        "such p-values only beside the per-subject direction count, which uses the "
        "subject as the unit of analysis, and we treat the direction count as the "
        "evidence. With six subjects a subject-level Wilcoxon test floors at "
        "p = 0.031. Correlations with subject-level markers use Spearman's rank "
        "correlation with an exact permutation p-value over all 720 orderings, "
        "rather than the asymptotic approximation, which is unreliable at n = 6. "
        "Bonferroni correction is applied across the family of tests reported.")

    # A1. 3.5 ended by naming estimators the next paragraph retires.
    d.patch("Per-window rates were estimated",
            "Of these, only the spectral peak (respiratory) and loose peak-counting "
            "(cardiac) were used for the reported rates; the remaining estimators, "
            "including the CWT-ridge and STFT-Viterbi trackers, were evaluated but "
            "not used in the final pipeline.",
            "None of the advanced trackers entered the reported pipeline. The "
            "estimator that did is described in the next paragraph, and the full "
            "comparison across estimators and channels is given in Figure S1.")

    # ==================================================================== 4.1
    # C. Opening sentence split. A7: the common-mode claim is contradicted by
    # this paper's own channel comparison later in the section.
    d.set_text(
        "The primary record the mask produces",
        "The primary record the mask produces is the overnight evolution of the "
        "sensor value itself. It is shown for a representative night in Figure 2, "
        "beside the scored hypnogram, the head-turn angle and the tracked rhythms. "
        "The temple electrodes operate near 2 pF: session mean levels span "
        "1958–2048 fF on CLE and 1624–2353 fF on CRE, with the forehead channel CH "
        "between −719 and −1297 fF. Within a night the level does not drift "
        "smoothly. It moves in discrete steps that coincide with posture changes. "
        "The within-night excursion has a median range of 105 fF on CLE, 71 fF on "
        "CRE, 158 fF on the CLE−CRE differential and 595 fF on CH, and exceeds "
        "1000 fF on CH in the three most mobile nights. This step structure, rather "
        "than sensor noise, is the dominant source of non-stationarity in the "
        "record.")

    # D2. The representative night moves off S6, the session the paper elsewhere
    # calls anomalously coupled and which carries the worst cardiac errors.
    d.swap_image(d.body[d.find("Figure 2. The overnight sensor record") - 1], FIG_RECORD_S1N1)
    d.set_text(
        "Figure 2. The overnight sensor record",
        "Figure 2. The overnight sensor record for a representative night (S1N1), "
        "CLE−CRE and CH. (A) PSG hypnogram. (B) Sensor value referenced to the "
        "session mean, in fF, with the subtracted mean printed on each axis; the "
        "level moves in discrete steps, not a smooth drift. (C) Left–right "
        "capacitance imbalance, signed (line; red fill left-dominant, blue fill "
        "right-dominant) within its ± envelope (dashed). (D) Per-block variance. "
        "(E) Head-turn angle over the full ±180° range, with posture marked. "
        "(F, G) Background-removed spectrograms of CLE−CRE and CH with the Viterbi "
        "respiratory (cyan) and cardiac (red) traces overlaid, and the tracker "
        "confidence printed. Stage shading is shared across rows B–E. Cohort-wide "
        "versions of rows B and C are given in Figures S6 and S7.")

    # C + typography in the imbalance paragraph.
    d.set_text(
        "The left-right capacitance imbalance is a real dynamic quantity",
        "The left–right capacitance imbalance is a dynamic quantity, not a static "
        "offset. Its magnitude has a median of 12.3 fF across the twelve nights, "
        "with session medians from 3.4 to 182.2 fF. Its sign is not fixed: the "
        "direction reverses a median of 4 times per night (range 1–8), and the "
        "fraction of the night spent left-dominant ranges from 0.40 to 0.73 across "
        "sessions. The absolute operating point behaves differently. It is dominated "
        "by a large static per-mount offset — session mean CLE−CRE levels span −374 "
        "to +356 fF — so absolute one-sidedness reflects mask placement and is not "
        "interpretable as physiology. Only the offset-invariant dynamics are "
        "reported here.")

    # B1. A null result was written as though it established independence, and it
    # pooled epochs in exactly the way 3.7 warns against.
    d.set_text(
        "The direction of the imbalance survives a head-position control",
        "Head position does not account for the direction of the imbalance, though "
        "it does account for its size. Pooled across 9,182 epochs, no association "
        "was detected between imbalance direction and head-turn angle (Spearman "
        "ρ = 0.01, p = 0.62). This is a failure to detect an association rather than "
        "evidence of independence, and because epochs within a night are not "
        "independent the test is weaker than its epoch count suggests. A second, "
        "simpler control points the same way: within the 7,641 supine epochs, where "
        "the head is held face-up, the direction still moves across its full range "
        "and still reverses, a median of 3.5 times per night. The magnitude behaves "
        "in the opposite way and is clearly posture-scaled, with a median of 9.0 fF "
        "supine and 11.3 fF in left-lying against 107.7 fF in right-lying. We "
        "therefore report the signed direction as a signal property that posture "
        "does not explain, and the magnitude as one that must be read beside head "
        "position. Neither is advanced as a measurement of intracranial fluid "
        "movement; that interpretation would need a zero-flow or head-position "
        "calibration this protocol does not provide.")

    d.patch("The forehead and differential channels are not two views",
            "Their independence is what makes agreement between them usable as "
            "evidence that a feature is physiological rather than an artifact of one "
            "mount.",
            "Their independence is what makes agreement between them usable as "
            "evidence that a feature is physiological rather than an artifact of one "
            "mount. It also means the differential channel cannot be assumed to "
            "cancel the level steps described above, since the two channels do not "
            "move together reliably.")

    # B3. Continuity of a trace is not accuracy of a trace; 4.2 reports that the
    # rate does not track, and the two sections read as a contradiction.
    d.set_text(
        "Both physiological rhythms are trackable continuously",
        "Both physiological rhythms can be followed continuously across a whole "
        "night. Rows F and G of Figure 2 carry a single Viterbi path per rhythm on "
        "each difference channel. The respiratory trace holds near 0.25–0.3 Hz and "
        "the cardiac trace near 0.9–1.2 Hz for the full recording, each with its own "
        "slow drift, and neither hops to a subharmonic. That two rhythms can be "
        "followed as continuous traces from the first hour to the last, on a channel "
        "formed by differencing two dry non-contact electrodes, is the basic "
        "signal-validation result of this study. It is a statement about continuity, "
        "not about accuracy: a trace can hold the right band all night while its "
        "moment-to-moment value tracks the reference poorly, which is what "
        "Section 4.2 goes on to report.")

    # ==================================================================== 4.2.1
    # D1. Recomputed on the operational fused pipeline by
    # analysis/rates/k_age_fused.py, so that Figure 5 and Table S1 finally use
    # one k series. Every number in this block moved.
    d.set_text(
        "k, age, and the calibration a deployment would require.",
        "k, age, and the calibration a deployment would require. Because k reflects "
        "the mechanical coupling between physiology and sensor, we asked whether it "
        "varies with subject age (Figure 5). Respiratory k declined across the six "
        "subjects, from 1.06 in the youngest (25 years) to 0.94 in the oldest "
        "(66 years), Spearman ρ = −0.81. An exact permutation test over all 720 "
        "orderings gives p = 0.072, which does not reach the conventional threshold. "
        "The direction of the relationship was nonetheless stable: dropping each "
        "subject in turn left ρ between −0.97 and −0.67. Cardiac k showed no such "
        "relationship (ρ = +0.32, exact p = 0.56), and unlike the respiratory case it "
        "was not stable under the same test, with leave-one-out ρ ranging from −0.21 "
        "to +0.82. These data neither establish nor exclude an age dependence for "
        "cardiac k. Neither factor was associated with PSQI (p > 0.19), and no other "
        "capacitive feature we examined varied with age (Figure S5).")

    d.set_text(
        "We tested whether the respiratory relationship is strong enough",
        "Because the correlation alone is not significant at this sample size, we "
        "tested the respiratory relationship the way a deployment would use it: fit "
        "k from age on five subjects, predict the sixth. Leave-one-subject-out, an "
        "age-based prior predicted respiratory k with a mean absolute error of "
        "0.026, against 0.044 for no calibration (k = 1.0) and 0.053 for the best "
        "constant prior. Age therefore carries information about respiratory k that "
        "a constant prior does not, and this held-out test, not the correlation "
        "coefficient, is what supports the claim. The same test on cardiac k gave "
        "the opposite result: an age prior (0.386) was worse than the population "
        "mean (0.304). That is what a fixed pulse morphology would produce. "
        "Translated into rate error, per-session calibration gives a per-epoch "
        "respiratory error of 0.94 br/min, a fixed k = 1.0 costs 0.20–0.28 br/min "
        "more, and the cardiac band requires whole-night self-calibration or a "
        "population prior (3.36 versus 4.56 BPM per-epoch; Figure S4).")

    d.patch("Three cautions apply to the age result",
            "The exact correlation p of 0.058 does not reach the conventional "
            "threshold before correction, and does not survive correction for the "
            "four correlations tested (each factor against age and PSQI; Bonferroni "
            "p ≈ 0.23).",
            "The exact correlation p of 0.072 does not reach the conventional "
            "threshold before correction, and does not survive correction for the "
            "four correlations tested (each factor against age and PSQI; Bonferroni "
            "p ≈ 0.29).")

    d.swap_image(d.body[d.find("Figure 5. Calibration factor k and subject age") - 1], FIG_K_AGE)
    d.set_text(
        "Figure 5. Calibration factor k and subject age.",
        "Figure 5. Calibration factor k and subject age, computed on the same "
        "operational pipeline as Tables 3 and S1. (A) Respiratory k per subject "
        "(mean of the two nights; bars span the two nights; circles male, squares "
        "female) against age, with least-squares fit. k falls from 1.06 to 0.94 "
        "across the age range; the exact permutation p and the leave-one-out range "
        "of ρ are given above the panel. (B) Cardiac k against age. Values cluster "
        "near 1.96 with no stable trend; S6 is a coupling outlier (k = 1.15) whose "
        "inclusion or exclusion reverses the sign of ρ. (C) Leave-one-subject-out "
        "prediction of respiratory k: absolute error per held-out subject using an "
        "age prior fitted on the other five, against using no calibration (k = 1.0).")

    # A6. The paragraph mixed the two error definitions 3.7 defines.
    d.patch("k is a waveform-morphology count",
            "The respiratory k of ≈0.99 reflects", "The respiratory k of ≈0.99 (median across sessions) reflects")

    # ==================================================================== 4.3
    # A2. The text described two bands; the figure shows three.
    d.set_text(
        "SEC spectrograms displayed persistent spectral ridges",
        "SEC spectrograms displayed persistent spectral ridges across full overnight "
        "recordings, concentrated in the CRE channel, which carried the dominant "
        "ridge in 9 of 12 sessions. The ridges fall into three bands: an infra-slow "
        "rhythm below respiration, near 0.07 Hz; the respiratory rate itself, near "
        "0.2–0.25 Hz; and an intermittent cardiac ridge between 0.9 and 1.8 Hz "
        "(Figure 6). Because the CRE channel was selected on these same data, the "
        "ridge results below are descriptive of this cohort.")

    # B4 + B5. The N3 cardiac frequency sits on the detector's own lower bound,
    # and the pooled-epoch p-values needed their direction counts beside them.
    d.set_text(
        "Restricting ridge detection to each band separately",
        "Restricting ridge detection to each band separately revealed "
        "physiologically interpretable stage associations (Figure 7). In the "
        "respiratory band (0.1–0.5 Hz) a persistent ridge was present in 96.7% of "
        "clean epochs, and its lowest frequency sat at a median of 0.23 Hz "
        "(≈14 breaths/min) in every stage, which confirms that the ridge tracks "
        "respiration. N3 epochs carried fewer active respiratory ridges than other "
        "sleep, in all six subjects, and lower total ridge power, in five of six. "
        "This is consistent with the more regular, lower-effort breathing of deep "
        "sleep. In the cardiac band (0.5–3.0 Hz) persistent ridges were intermittent, "
        "present in only 11.7% of epochs, but more frequent during N3 and N2 than in "
        "lighter stages. When present, the lowest cardiac ridge frequency was lower "
        "in N3 than in other sleep, in five of six subjects, mirroring the sleep "
        "bradycardia of deep NREM. We report that shift as a direction rather than "
        "as a calibrated frequency, because the N3 values approach the 0.5 Hz lower "
        "edge of the analysis band and the tracker's own search band begins at "
        "0.85 Hz. The pooled Kruskal–Wallis and Mann–Whitney p-values for these "
        "contrasts are very small (down to 3×10⁻²³), but they are computed on "
        "non-independent epochs and are reported in Figure 7 as descriptive "
        "statistics only; the per-subject direction counts given here are the "
        "evidence. Both associations reflect autonomic and respiratory changes "
        "transduced mechanically by the sensor rather than cortical activity, and "
        "neither is strong or subject-consistent enough to serve as a standalone "
        "sleep-stage classifier.")

    d.patch("Figure 7. Band-restricted ridge structure",
            "Kruskal–Wallis p-values across stages are shown above each panel.",
            "Kruskal–Wallis p-values across stages are shown above each panel; they "
            "are computed on pooled, non-independent epochs and are descriptive only "
            "(§3.7).")

    # ==================================================================== 5
    # C. The Discussion opened by praising itself.
    d.set_text(
        "This study provides a systematic, honest characterization",
        "This section sets out what the mask measured reliably, what it did not "
        "measure, and the mechanisms that account for each.")

    # C. 5.1 restated 4.1's numbers verbatim; keep the interpretation, drop the
    # repetition.
    d.set_text(
        "Both respiratory and cardiac bands carry substantial energy",
        "Both physiological bands sit well above the sensor noise floor on every "
        "channel in all twelve recordings, and the coherence and surrogate analyses "
        "show that this energy is genuine physiological coupling rather than "
        "broadband noise. That establishes the signal quality any downstream "
        "analysis depends on. It is a weaker statement than it first appears: on "
        "most individual epochs the coupling is not separable from the surrogate "
        "null, so the result licenses aggregate analysis but not per-epoch quality "
        "gating.")

    # C. 5.2 likewise repeated 4.4 in full.
    d.set_text(
        "The mask carries no cortical electrographic signature.",
        "The mask carries no cortical electrographic signature. Sleep spindles "
        "produce no spindle-locked sigma response in any capacitive channel, while "
        "the same measurement on the contact EEG rises +3.3 dB, and delta-burst "
        "onsets produce a response that follows the cortical event at zero lag "
        "rather than preceding it. What the mask registers at both events is a "
        "low-frequency mechanical increase. The sensor transduces mechanical and "
        "hemodynamic pulsations, not neuronal electrical activity, and this is the "
        "clearest boundary the study establishes.")

    # B2. The mechanism was stated as fact; nothing here tests it against
    # alternatives.
    d.patch("The absence of within-night cardiac tracking",
            "The absence of within-night cardiac tracking can be understood through "
            "the k factor.",
            "We propose the following account of the absence of within-night cardiac "
            "tracking. It is consistent with everything reported here but is not "
            "tested against alternatives by these data, and should be read as a "
            "hypothesis.")

    # A3. The paragraph explained a "harmonic direction ambiguity" that no
    # section reports. Retargeted to the subject-dependence 4.3 does report.
    d.set_text(
        "The harmonic direction ambiguity likely reflects individual differences",
        "The subject-dependence of the weaker ridge effects may have a similar "
        "basis. The capacitive sensor sits at the temple and receives a mixture of "
        "intracranial pressure, superficial temporal artery pulsation, and "
        "near-field respiratory displacement. How much each contributes varies with "
        "individual anatomy, sensor placement and mask fit. During N3 cardiac output "
        "falls and respiratory mechanics change, and how those changes project onto "
        "the temple sensor depends on the subject-specific coupling geometry. This "
        "would explain why the ridge effects hold their direction in most subjects "
        "but not all.")

    # A4 + B6. 5.5 quoted an error the paper no longer reports and claimed
    # agreement with a literature it does not cite.
    d.set_text(
        "The cardiac k ≈ 2 and the resulting ∼4 BPM MAE",
        "The cardiac k ≈ 2 is consistent in kind with the ballistocardiographic "
        "literature, where waveform morphology and mechanical coupling introduce "
        "systematic overcounting that requires calibration. We make no quantitative "
        "comparison, because reported error definitions and window lengths differ "
        "and we are not aware of a published dataset matching the night-level "
        "definition used here. The respiratory k ≈ 1 reflects the simpler coupling: "
        "each breath produces a single dominant displacement of the temple sensor, "
        "unlike the biphasic cardiac pulse. The resolution-versus-responsiveness "
        "trade-off that limits per-epoch respiratory precision is a known constraint "
        "in wearable respiratory monitoring.")

    # ==================================================================== 6
    # A5. Third contradicted 4.2 and cited a method that appears nowhere else.
    d.set_text(
        "Third, k-calibration requires a reference cardiac or respiratory rate.",
        "Third, k-calibration requires a reference cardiac or respiratory rate. k is "
        "stable within and across nights once estimated, but a deployment would need "
        "either a calibration period against a reference or a population-level "
        "prior, and every accuracy figure reported here is conditional on that "
        "calibration. Respiratory k is near unity, so a fixed k = 1.0 costs only "
        "0.20–0.28 br/min of per-epoch error, and an age-based prior improves on a "
        "constant prior when validated on held-out subjects (§4.2.1). This should "
        "not be read as the respiratory band being calibration-free in a useful "
        "sense: the cohort's mean rates span only 14.4–16.8 br/min, so a constant "
        "predictor already performs well, and a wider population would test the "
        "calibration more severely. The cardiac band has no such shortcut. A "
        "ten-minute warm-up calibration is worse than a population prior, because "
        "twenty epochs give a noisy and drifting estimate (Figure S4), so a "
        "deployment would use a population prior or whole-night self-calibration.")

    # ==================================================================== S
    # D3. The two single-session figures carried cohort claims; the cohort views
    # already exist and become supplementary panels.
    tail = d.body[d.find("No capacitive feature other than respiratory k varied with age")]
    template_img = d.body[d.find("Figure S5. Capacitive signal features versus subject age") - 1]
    at = list(d.body).index(tail) + 1
    new = [
        d.image_para(FIG_GRID_CHANNELS, template_img),
        d.para("Figure S6. Overnight sensor value for all twelve recordings, CH against "
               "CLE−CRE, each referenced to its session mean. The step structure and its "
               "coincidence with posture change are present in every night; the size of "
               "the excursion is not, and is largest in the mobile nights (S5N1, S6N1, "
               "S6N2). This is the cohort-wide version of row B of Figure 2.",
               style="Caption" if False else ""),
        d.image_para(FIG_GRID_IMBALANCE, template_img),
        d.para("Figure S7. Left–right capacitance imbalance for all twelve recordings, "
               "signed marker within its ± envelope. Sign reversals occur in every night "
               "(median 4, range 1–8) and the marker's scale varies by more than an order "
               "of magnitude between nights. This is the cohort-wide version of row C of "
               "Figure 2."),
    ]
    for off, el in enumerate(new):
        d.body.insert(at + off, el)
    d.n_patch += len(new)

    d.save(DOC)
    print("applied %d edits -> %s" % (d.n_patch, DOC))


if __name__ == "__main__":
    main()
