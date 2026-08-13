"""Repair the canonical manuscript from section 3.3 onward.

  in  : writeup/main/CAP_sleep_mask_manuscript_main.docx   (edited in place)
  bak : writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_S33_20260813.docx

Two jobs, both confined to section 3.3 and later. Sections 1-3.2, the title
block and the abstract are not touched.

  A. New primary demonstration of the data.  Section 4.1 now leads with the
     overnight channel-evolution record -- absolute sensor value in fF, the
     left-right capacitance imbalance with its head-position control, and the
     Viterbi-tracked respiratory and cardiac ridges -- in place of the plain
     two-band spectrogram that opened it before.  Figure 1 is swapped for the
     channel-evolution panel and Figure 5 for the ridge-tracker panel.
     Methods for all of it go into 3.4.

  B. The repairs carried over from the analytical review pass, restricted to
     3.3+: night-level error reported as a night quantity, the degenerate
     respiratory spectral estimator retired, epoch length made consistent,
     estimator count, k range vs IQR, exact permutation p, figure numbering.

Run from the repo root:  python writeup/edits/apply_s33_onward.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_review_edits import Doc, W  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_S33_20260813.docx")

FIG_CHANNEL_EVOLUTION = Path("writeup/figures/channel_evolution/S6N1_CH_CLE-CRE.png")
FIG_RIDGE_TRACKER = Path("writeup/figures/harmonics/ridges/ridge_tune_S1N1_CRE.png")


def main() -> None:
    if not DOC.exists():
        raise SystemExit("missing: %s" % DOC)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ================================================== 3.3 Ground truth
    # Flow-vs-RIPSum agreement: the manuscript carried rounded values that do
    # not match the reported per-session table.
    d.patch("To validate this consensus",
            "r = +0.48 on raw rates and r = +0.28 on detrended fluctuations",
            "r = +0.47 on raw rates and r = +0.27 on detrended fluctuations")

    # The analysis grid was stated three different ways across the manuscript
    # (60 s windows / 30 s step here, "30-second analysis window" in section 6,
    # "60s" in Table 1).  Epochs are non-overlapping 30 s.
    d.set_text(
        "All rates were computed on a common sliding-window grid",
        "All rates were computed on a common grid of non-overlapping 30-second "
        "windows, aligned to the 30-second epochs of the PSG technologist's AASM "
        "scoring. Sleep stages were taken from that scoring.")

    # ================================================== 3.4 Signal record
    d.set_text("3.4 Signal validation approach",
               "3.4 Overnight signal record and validation approach")

    anchor = "3.4 Overnight signal record and validation approach"
    d.insert_after(anchor, [d.para(p) for p in [
        "Overnight evolution of the sensor value. The absolute capacitance of each "
        "channel was tracked across the whole night as the mean over consecutive "
        "10-second blocks, expressed in femtofarads and referenced to the session "
        "mean, which is reported on every panel so that the absolute operating "
        "point is not lost. The same block grid carries the per-block variance and, "
        "from the accelerometer, the head-turn angle: gravity was isolated with a "
        "0.05 Hz low-pass, the turn angle computed over the full +/-180 deg range so "
        "that rotation past pure side-lying is representable, and the trace "
        "re-wrapped about each night's circular median so that a brief crossing of "
        "the +/-180 deg seam does not appear as a night-wide swing.",

        "Left-right capacitance imbalance. The difference between the two temple "
        "electrodes carries a slowly varying component that we quantify with a "
        "night-scale marker: a short median filter to de-spike, then a rolling mean "
        "over tau = 30 minutes, applied to the signed difference to give the imbalance "
        "itself and to its absolute value to give the envelope, with motion epochs "
        "excluded. We report the signed marker in fF together with that envelope. "
        "Because the quantity is a left-right difference measured through a mask "
        "whose placement sets a large static offset, two controls are reported "
        "alongside it: its association with head-turn angle, and its behaviour "
        "restricted to supine epochs only.",

        "Ridge tracking of the two rhythms. To follow the respiratory and cardiac "
        "rhythms continuously rather than window by window, each was tracked as a "
        "single Viterbi path through the spectrogram -- the globally optimal trace "
        "maximising spectral amplitude minus a penalty on frequency change, searched "
        "inside a physiological band (respiration 0.25-0.55 Hz, cardiac 0.85-1.45 Hz, "
        "the upper cardiac bound corresponding to about 87 BPM). The spectrogram was "
        "first background-removed by subtracting a frequency-median-filtered copy of "
        "each time column, which flattens the 1/f envelope and broadband motion "
        "brightening while leaving narrow peaks intact. Because the path is global "
        "and continuous, each rhythm has exactly one value at any instant and cannot "
        "hop between a rhythm and its subharmonic; the tracker's own confidence, the "
        "fraction of windows in which the tracked band is clearly peaked above the "
        "column median, is reported on each panel.",
    ]])

    # The canonical-bound and surrogate framing were asserted without saying
    # where the bound comes from or what the null rate is.
    d.set_text(
        "For each analysis epoch, magnitude-squared coherence was computed",
        "For each analysis epoch, magnitude-squared coherence was computed between "
        "the SEC channel and the corresponding PSG reference (nasal airflow for "
        "respiration, ECG for cardiac; the single-sensor references are used here "
        "rather than the multi-sensor consensus of section 3.3 because coherence "
        "requires a continuous waveform, not an epoch-wise rate), and the coherence "
        "value was read at the ground-truth rate frequency. Spectral agreement was "
        "quantified as the fraction of epochs whose SEC peak frequency fell within "
        "+/-0.05 Hz of the reference. To bound what coherence is attainable between "
        "two genuine but physically distinct measurements of the same rhythm, we "
        "computed the same statistic between two PSG channels -- Flow versus RIPSum "
        "for respiration and ECG versus PPG for cardiac -- and report it as the "
        "canonical bound. To guard against spurious coherence from band-limited "
        "noise, phase-randomized surrogates (200 per epoch) were generated that "
        "preserve the power spectrum while destroying phase structure; the fraction "
        "of epochs whose observed coherence exceeded the surrogate null at p < 0.05 "
        "was reported and compared against the 5% rate expected under the null. "
        "Coherence was evaluated within each sleep stage to test whether coupling "
        "persists beyond wake.")

    # ================================================== 3.5 Rate estimation
    d.patch("Per-window rates were estimated",
            "using six base estimators", "using seven base estimators")
    d.patch("Per-window rates were estimated",
            "and a spectral-guided amplitude-adaptive peak detector.",
            "and a spectral-guided amplitude-adaptive peak detector. The Welch "
            "estimator used 4-second segments with 50% overlap at the 100 Hz "
            "sampling rate, giving a frequency resolution of 0.25 Hz.")

    d.set_text(
        "The operational pipeline was therefore a single per-window estimator",
        "The operational pipeline was therefore a per-window estimator followed by "
        "k-scaling and a causal three-epoch median filter: loose prominence-based "
        "peak counting, fused across five channels by agreement gating, in both "
        "bands. One estimator choice requires comment because it changes how the "
        "respiratory numbers should be read. The Welch spectral peak attains a "
        "marginally lower per-epoch respiratory error but is degenerate at this "
        "window length: at 0.25 Hz resolution the 0.1-0.5 Hz respiratory band "
        "contains only two usable bins, and the estimator returned 0.25 Hz -- exactly "
        "15 breaths/min -- in 99.95% of all epochs. Its apparent accuracy is that of "
        "the constant 15 br/min rescaled by a k that is itself fitted to the "
        "reference, so we report the non-degenerate fused peak-counting estimator "
        "throughout and treat the spectral peak as a baseline rather than a "
        "measurement. Multi-channel quality-weighted and agreement-gated fusion "
        "changed per-epoch error by less than 0.1 in either band relative to the "
        "single differential channel, but the fused estimator is non-degenerate and "
        "is preferred for that reason.")

    # k is fitted against the reference; every downstream accuracy is conditional.
    d.patch("SEC peak counts are systematically scaled",
            "Both uncalibrated and k-scaled accuracies are reported.",
            "Because k is fitted against the reference, any accuracy computed after "
            "k-scaling is conditional on that calibration; both uncalibrated and "
            "k-scaled accuracies are reported, and section 4.2 separates the part of "
            "the aggregate accuracy that k supplies from the part the sensor "
            "supplies.")

    # ================================================== 3.6 Ridge detection
    d.patch("Whole-night CAP spectrograms show structured spectral ridges",
            "Whole-night CAP spectrograms", "Whole-night SEC spectrograms")
    d.patch("To separate the two physiological rhythms the sensor carries",
            "rather than as a trained classifier.",
            "rather than as a trained classifier. This persistent-ridge detector "
            "answers a different question from the single-path tracker of section "
            "3.4: it admits several concurrent ridges per band and is used for the "
            "stage statistics, whereas the tracker returns one continuous trace per "
            "rhythm and is used for the overnight records.")

    # ================================================== 3.7 Statistics
    d.set_text(
        "All rate accuracy metrics are reported within-session",
        "All rate accuracy metrics are reported within-session to avoid inflation "
        "from between-session mean matching. Two error definitions are used and are "
        "kept distinct throughout: the night-level error, the absolute difference "
        "between the mean estimated and the mean reference rate over a whole "
        "recording; and the per-epoch error, the median absolute error across the "
        "epochs of a recording. Cross-session summaries use the median and "
        "interquartile range of per-session values, so that the unit of analysis is "
        "the recording and not the epoch. Within-session association uses the "
        "Pearson correlation across the epochs of a night, summarized across nights "
        "by the median and a Wilcoxon signed-rank test on the twelve per-night "
        "values. Non-parametric group comparisons use the Kruskal-Wallis test across "
        "stages, with the Mann-Whitney U test for N3-versus-rest contrasts. Where "
        "these stage tests are computed on pooled epochs the epochs within a night "
        "are not independent and the p-values are anti-conservative; we therefore "
        "report them only alongside the per-subject direction count, which uses the "
        "subject as the unit of analysis, and treat the direction count as the "
        "evidence. With six subjects a subject-level Wilcoxon test floors at "
        "p = 0.031. Correlations with subject-level markers use Spearman's rank "
        "correlation with an exact permutation p-value (all 720 permutations at "
        "n = 6) rather than the asymptotic approximation, with Bonferroni correction "
        "across the family of tests reported.")

    # ================================================== 4.1 Signal validation
    d.set_text("4.1 Signal validation: SEC carries respiratory and cardiac band energy",
               "4.1 The overnight sensor record and its physiological content")

    # New lead paragraph: the record itself, in absolute units.
    d.set_text(
        "The capacitive temple sensor signal contained sustained energy in both the respiratory",
        "The primary record the mask produces is the overnight evolution of the "
        "sensor value itself, and it is shown for a representative night in "
        "Figure 1 alongside the scored hypnogram, the head-turn angle and the "
        "tracked rhythms. The temple electrodes operate near 2 pF -- session mean "
        "levels span 1958-2048 fF on CLE and 1624-2353 fF on CRE, with the forehead "
        "channel CH between -719 and -1297 fF -- and within a night the level does "
        "not drift smoothly but moves in discrete steps that coincide with posture "
        "changes. The within-night excursion has a median range of 105 fF on CLE and "
        "71 fF on CRE, 158 fF on the CLE-CRE differential and 595 fF on CH, with the "
        "most mobile nights reaching several thousand fF on CH. This step structure, "
        "rather than sensor noise, is the dominant source of non-stationarity in the "
        "record, and it is why the differential channel is used for rate estimation: "
        "the steps are largely common-mode.")

    d.swap_image(d.body[d.find("Figure 1. CLE−CRE spectrograms")-1], FIG_CHANNEL_EVOLUTION)
    d.set_text(
        "Figure 1. CLE−CRE spectrograms",
        "Figure 1. The overnight sensor record for a representative night (S6N1), "
        "CLE-CRE and CH. (A) PSG hypnogram. (B) Sensor value referenced to the "
        "session mean, in fF, with the subtracted mean printed on each axis; the "
        "level moves in discrete steps, not a smooth drift. (C) Left-right "
        "capacitance imbalance, signed (line, filled red for left-dominant and blue "
        "for right-dominant) within its +/-envelope (dashed). (D) Per-block variance. "
        "(E) Head-turn angle over the full +/-180 deg range, with posture called out. "
        "(F, G) Background-removed spectrograms of CLE-CRE and CH with the Viterbi "
        "respiratory (cyan) and cardiac (red) traces overlaid and the tracker "
        "confidence printed. Stage shading is shared across rows B-E.")

    d.insert_after("Figure 1. The overnight sensor record", [d.para(p) for p in [
        "The left-right capacitance imbalance is a real dynamic quantity, not a "
        "static offset. Its magnitude has a median of 12.3 fF across the twelve "
        "nights (session medians 3.4-182.2 fF), and its sign is not fixed: the "
        "direction reverses a median of 4 times per night (range 1-8), and the "
        "fraction of the night spent left-dominant ranges from 0.40 to 0.73 across "
        "sessions. The absolute operating point, by contrast, is dominated by a "
        "large static per-mount offset -- session mean CLE-CRE levels span -374 to "
        "+356 fF -- so absolute one-sidedness reflects mask placement and is not "
        "interpretable as physiology. Only the offset-invariant dynamics are "
        "reported here.",

        "The direction of the imbalance survives a head-position control, but its "
        "magnitude does not. Pooled across 9,182 epochs the imbalance direction is "
        "uncorrelated with head-turn angle (Spearman rho = 0.01, p = 0.62), and "
        "restricting the analysis to the 7,641 supine epochs -- head held face-up -- "
        "the direction still moves across its full range and still reverses, a "
        "median of 3.5 times per night. The magnitude behaves in the opposite way: "
        "it is strongly posture-scaled, with a median of 9.0 fF supine and 11.3 fF "
        "in left-lying against 107.7 fF in right-lying. We therefore treat the "
        "signed direction as a candidate physiological quantity that posture does "
        "not explain, and the magnitude as a posture-scaled quantity that must be "
        "reported alongside head position rather than on its own. Neither is "
        "advanced here as a measurement of intracranial fluid movement; that "
        "interpretation would require a zero-flow or head-position calibration the "
        "present protocol does not provide.",

        "The forehead and differential channels are not two views of one signal. CH "
        "moves 2.8 times as far as CLE-CRE at the level of the night (median ratio "
        "of standard deviations, range 1.6-11.0), and the correlation between their "
        "levels varies from -0.49 to +0.96 across sessions, with 3 of 12 sessions "
        "negative. They agree best in the slow band (median coherence 0.55) and "
        "progressively less in the respiratory (0.31) and cardiac (0.09) bands. "
        "Their independence is what makes agreement between them usable as evidence "
        "that a feature is physiological rather than an artifact of one mount.",

        "Both physiological rhythms are trackable continuously across a whole night. "
        "Rows F and G of Figure 1 carry a single Viterbi path per rhythm on each "
        "difference channel: the respiratory trace holds near 0.25-0.3 Hz and the "
        "cardiac trace near 0.9-1.2 Hz for the full recording, each with its own "
        "slow drift and neither hopping to a subharmonic. That the two rhythms can "
        "be followed as continuous traces from first to last hour, on a channel "
        "formed by differencing two dry non-contact electrodes, is the basic "
        "signal-validation result on which the rest of this paper rests.",

        "The band energy underlying those traces sits well above the sensor noise "
        "floor. The respiratory band carried 29-48% of total signal power (0-5 Hz) "
        "and the cardiac band 8-48%. To quantify how far this content sits above the "
        "noise floor we computed a physiological-band SNR for each raw channel as "
        "the ratio of mean spectral density (power per Hz) in the 0.1-3 Hz band to "
        "that of the 10-50 Hz electronic noise floor, which an independent "
        "no-subject baseline recording confirms is spectrally white. SNR was "
        "positive on every channel in all twelve recordings, averaging +30.0 dB on "
        "CH and +18.7 and +20.7 dB on CLE and CRE, with no single value below "
        "+6.4 dB. The forehead channel was strongest in every session; the temple "
        "channels varied more from night to night, reflecting differences in "
        "sensor-skin coupling rather than added noise, since the noise floor was "
        "near-constant across sessions. Cross-subject variation was substantial: "
        "S6N1 was respiration-dominated (48% respiratory, 8% cardiac) while S3N1 was "
        "cardiac-heavy (48% cardiac).",
    ]])

    # Surrogate result: state the null rate the excess is measured against.
    d.set_text(
        "Cross-spectral coherence at the ground-truth rate frequency confirmed",
        "Cross-spectral coherence at the ground-truth rate frequency confirmed "
        "physiological coupling: respiratory coherence was median 0.31 on the "
        "average channel, against a canonical bound of 0.61 measured between two "
        "genuine PSG respiratory sensors, and cardiac coherence was median 0.16 "
        "against a canonical bound of 0.27. Phase-randomized surrogate testing (200 "
        "surrogates per epoch, 8,242 of the 9,319 epochs passing the motion and "
        "coverage gate) showed that 14.7% of respiratory epochs and 9.1% of cardiac "
        "epochs exceeded the surrogate null at p < 0.05. Against the 5% expected by "
        "chance the respiratory excess is roughly threefold and the cardiac excess "
        "roughly twofold, establishing that the in-band energy is not an artifact of "
        "band-limited noise. The converse bounds the claim: on most individual "
        "epochs the coupling is not separable from the null, so per-epoch coherence "
        "should not be used as a quality gate. Coherence computed within each sleep "
        "stage retained the same ordering, with no stage falling to the null rate.")

    d.patch("Respiratory frequency agreement between the CAP spectral peak",
            "between the CAP spectral peak", "between the SEC spectral peak")

    # ================================================== 4.2 Rate detection
    d.set_text(
        "The sensor was evaluated against PSG in two regimes",
        "The sensor was evaluated against PSG in two regimes that differ in the "
        "quantity certified. The aggregate regime asks whether the mask recovers a "
        "subject's average respiratory and cardiac rate over a whole recording -- the "
        "quantity relevant to overnight screening and night-to-night trending -- and "
        "is scored as the absolute difference between the mean estimated and mean "
        "reference rate for that night. The within-night regime asks whether the "
        "mask follows the rate as it varies during the night, and is scored by the "
        "per-epoch median absolute error and by the correlation between estimate and "
        "reference across the epochs of a night. Both are computed on the same "
        "non-overlapping 30-second windows. They are reported together because they "
        "reach opposite conclusions, and because the distinction determines what the "
        "device can be used for.")

    d.set_text(
        "For the night-level regime we used the estimator that minimised error",
        "Both regimes use the same operational pipeline described in section 3.5: "
        "loose prominence-based peak counting, fused across five channels by "
        "agreement gating, rescaled by the per-session factor k (section 4.2.1) and "
        "smoothed with a causal three-epoch median filter. The degenerate Welch "
        "spectral estimator is reported only as a baseline (Figure S1).")

    d.set_text("Night-level accuracy", "Aggregate accuracy: the per-night mean rate")
    d.set_text(
        "Respiratory rate was recovered with a per-session median MAE of 0.91",
        "Averaged over a whole night, the mask tracks both rates closely. The "
        "night-level error -- the absolute difference between the mean estimated and "
        "the mean reference rate for that recording -- had a median of 0.14 br/min "
        "for respiration (IQR 0.10-0.41, worst night 0.93) and 1.60 BPM for the "
        "cardiac band (IQR 1.20-2.21, worst night 10.45). Eleven of twelve nights "
        "fell below 3.1 BPM; the single outlier is S6N2, the session in which "
        "cardiac coupling was anomalous and k fell to 0.94. Per-session values are "
        "given in Table S1 and the agreement plots in Figure 3. Two qualifications "
        "bound this result. It is achieved after per-session k-calibration against "
        "the reference, so it certifies that the sensor plus a calibration constant "
        "reproduces the night mean, not that the sensor alone measures it. And "
        "respiratory rates in this cohort span a narrow range (session means "
        "14.4-16.8 br/min), so an uncalibrated constant predictor of 15 br/min would "
        "already achieve a night-level error of 0.78 br/min; the respiratory "
        "aggregate figure should be read as a calibration result on a homogeneous "
        "cohort, and the cardiac figure, spanning a wider range of true rates, is "
        "the stronger of the two.")

    d.set_text("Epoch-level accuracy", "Within-night accuracy: rate variation is not recovered")
    d.set_text(
        "Within a night, neither band's rate variation was recovered",
        "Within a night, neither band's rate variation was recovered (Table 3). The "
        "per-epoch median absolute error was 0.94 br/min (IQR 0.85-1.19) for "
        "respiration and 3.36 BPM (IQR 2.64-6.62) for the cardiac band, and a "
        "responsive detector combining loose peak counting with Hilbert "
        "instantaneous frequency traded error for temporal resolution without "
        "recovering the variation (1.33 br/min and 3.65 BPM). Estimates correlated "
        "with the reference at a median within-session r = +0.06 for respiration "
        "(Wilcoxon p = 0.64 on the twelve per-night values; positive in 8 of 12 "
        "nights) and r = -0.19 for the cardiac band (p = 0.33; positive in 5 of 12). "
        "Neither differs from zero, and neither survives comparison with a temporal "
        "shuffle null (Figure S3). This is not entirely a limitation of the mask: "
        "two physically independent PSG respiratory sensors agree with each other at "
        "only r = 0.47 (IQR 0.33-0.67) on the same epochs, which bounds what any "
        "respiratory sensor can achieve at this time resolution. But the mask "
        "reaches a small fraction of that bound, and for the cardiac band no "
        "comparable external bound applies.")

    d.set_text(
        "The two bands fall short of that bound for different reasons",
        "The two bands fail to track for different reasons, and the amount of "
        "variation available to be tracked distinguishes them. Within a night the "
        "reference respiratory rate varies with a median standard deviation of "
        "1.57 br/min, against a per-epoch error of 1.33 br/min for the responsive "
        "detector -- the signal to be resolved is barely larger than the error, so "
        "little variation is recoverable in principle and a near-constant estimate "
        "is close to optimal. The cardiac band is different: reference heart rate "
        "varies with a median standard deviation of 6.58 BPM, comfortably above the "
        "3.65 BPM per-epoch error, so the variation is in principle resolvable and "
        "is nonetheless not recovered. The cardiac failure is therefore mechanistic "
        "rather than a matter of insufficient signal, and is discussed in section "
        "5.3. An exhaustive comparison of estimators and channels in both regimes, "
        "including CWT-ridge and STFT-Viterbi trackers, is given in Figure S1 and "
        "Figure S3.")

    # ---- Table 3 -------------------------------------------------------------
    tbl3 = d.body[d.find("Table 3. Rate detection in both regimes") + 1]
    assert tbl3.tag == W + "tbl", "Table 3 element not where expected"
    t3 = [
        ("Night-level regime", "Aggregate regime (per-night mean)"),
        ("MAE, per-session median", "Night-level error, median [IQR]"),
        ("0.91 br/min [0.81–1.19]", "0.14 br/min [0.10–0.41]"),
        ("3.41 BPM [3.06–8.38]", "1.60 BPM [1.20–2.21]"),
        ("MAE, pooled", "Night-level error, worst night"),
        ("1.09 br/min", "0.93 br/min"),
        ("3.91 BPM", "10.45 BPM"),
        ("Epoch-level regime", "Within-night regime"),
        ("Calibration factor k", "Calibration factor k, median [IQR]"),
        ("0.97 [0.91–1.04]", "0.96 [0.91–1.02]"),
        ("1.95 [0.94–2.24]", "1.95 [1.79–2.00]"),
        ("+0.06 (p = 0.68)", "+0.06 (p = 0.64)"),
        ("−0.19 (p = 0.34)", "−0.19 (p = 0.33)"),
        ("1.14 br/min", "1.57 br/min"),
        ("5.26 BPM", "6.58 BPM"),
        ("r = 0.47", "r = 0.47 [0.33–0.67]"),
    ]
    for tc in tbl3.iter(W + "tc"):
        ts = tc.findall(".//" + W + "t")
        cur = "".join(t.text or "" for t in ts).strip()
        for old, new in t3:
            if cur == old:
                for t in ts[1:]:
                    t.text = ""
                ts[0].text = new
                d.n_patch += 1
                break

    # The "MAE, per-session median" label appears twice; the second occurrence is
    # the within-night row and needs the other label.
    seen = 0
    for tc in tbl3.iter(W + "tc"):
        ts = tc.findall(".//" + W + "t")
        cur = "".join(t.text or "" for t in ts).strip()
        if cur == "Night-level error, median [IQR]":
            seen += 1
            if seen == 2:
                ts[0].text = "Per-epoch error, operational pipeline"
                for t in ts[1:]:
                    t.text = ""

    d.set_text(
        "Table 3. Rate detection in both regimes.",
        "Table 3. Rate detection in both regimes, from the operational pipeline "
        "(agreement-gated five-channel fusion of loose peak counting, per-session k, "
        "causal three-epoch median filter). Values are medians across the 12 nights "
        "with the interquartile range in brackets; per-night values are in Table S1. "
        "The night-level error is |mean(estimate) − mean(reference)| for a "
        "recording; the per-epoch error is the median absolute error across that "
        "recording's epochs. Within-session r is the Pearson correlation between "
        "estimate and reference across the epochs of a night, with a Wilcoxon "
        "signed-rank test on the twelve per-night values.")

    d.set_text(
        "Figure 3. Bland–Altman agreement",
        "Figure 3. Bland–Altman agreement for respiratory (left) and cardiac (right) "
        "rate. Each point is one 30-second analysis epoch, pooled across the twelve "
        "recordings; the solid line is the bias and the dashed lines the 95% limits "
        "of agreement (respiratory bias −0.3 br/min, limits −4.7 to +4.2; cardiac "
        "bias −0.6 BPM, limits −24.1 to +22.9). The spread shown here is epoch-level "
        "and is the subject of the within-night regime; the aggregate regime of "
        "Table 3 concerns the per-night means, whose errors are 0.14 br/min and "
        "1.60 BPM and are not visible at this scale.")

    # ================================================== 4.2.1 k
    d.patch("k is a waveform-morphology count",
            "The correspondence is population-level rather than exact per subject "
            "(r = 0.50 between measured peaks-per-beat and fitted k), and one "
            "session was excluded because cardiac peak detection failed outright.",
            "The correspondence is population-level rather than exact per subject: "
            "the correlation between measured peaks-per-beat and fitted k is "
            "r = 0.50, which at n = 9 is not distinguishable from zero (p = 0.17), so "
            "the morphological account is supported by the agreement of the central "
            "values rather than by covariation across subjects.")
    d.patch("k is a waveform-morphology count",
            "The respiratory k of ≈0.97 reflects", "The respiratory k of ≈0.96 reflects")

    d.patch("k is reproducible within a subject",
            "and cardiac k is stable apart from the S6 coupling anomaly.",
            "and cardiac k is stable apart from the S6 coupling anomaly, whose two "
            "nights account for the full cardiac range of 0.94–2.23 while the "
            "central half of the distribution spans only 1.79–2.00.")

    d.patch("k, age, and the calibration a deployment would require",
            "Spearman ρ = −0.83, uncorrected p = 0.042.",
            "Spearman ρ = −0.83. At n = 6 we evaluate this with an exact permutation "
            "test over all 720 orderings, which gives p = 0.058; the asymptotic "
            "approximation would give p = 0.042, and we report the exact value "
            "because the approximation is unreliable at this sample size.")
    d.patch("k, age, and the calibration a deployment would require",
            "(ρ = +0.37, p = 0.47)", "(ρ = +0.37, exact p = 0.47)")

    d.patch("Two cautions apply to the age result",
            "Two cautions apply to the age result.", "Three cautions apply to the age result.")
    d.patch("Three cautions apply to the age result",
            "And the respiratory correlation does not survive correction for the "
            "four correlations tested (each factor against age and PSQI; Bonferroni "
            "p ≈ 0.17).",
            "The exact correlation p of 0.058 does not reach the conventional "
            "threshold before correction, and does not survive correction for the "
            "four correlations tested (each factor against age and PSQI; Bonferroni "
            "p ≈ 0.23). And the leave-one-subject-out prediction, while genuinely "
            "held out, is a comparison of three numbers on six subjects.")

    # ================================================== 4.3 Ridges
    d.set_text("4.3 Harmonic structure and band-restricted ridge features",
               "4.3 Persistent spectral ridges and their sleep-stage associations")
    d.patch("CAP spectrograms displayed persistent spectral ridges",
            "CAP spectrograms displayed", "SEC spectrograms displayed")

    d.swap_image(d.body[d.find("Figure 5. Representative session")-1], FIG_RIDGE_TRACKER)
    d.set_text(
        "Figure 5. Representative session",
        "Figure 5. Representative session (S1N1, CRE). Top: PSG hypnogram. Middle: "
        "background-removed spectrogram, 0–3 Hz, with the Viterbi respiratory (cyan) "
        "and cardiac (red) traces; both hold continuously across the full eight-hour "
        "recording, and the tracker confidence is printed in the title. Bottom: the "
        "slow band, 0–0.3 Hz, with its persistent ridges; several slow oscillations "
        "can coexist here, so this band keeps the multi-ridge detector rather than a "
        "single path.")

    # ================================================== 4.5 figure numbering
    # Figure 8 does not exist: the delta-burst figure is numbered 9.
    for needle, old, new in [
        ("The response was present in every one of the six subjects", "Figure 9)", "Figure 8)"),
        ("The capacitive response therefore follows the cortical onset", "in Figure 9 is flat", "in Figure 8 is flat"),
        ("Figure 9. Delta-burst onsets evoke", "Figure 9. Delta-burst onsets", "Figure 8. Delta-burst onsets"),
    ]:
        d.patch(needle, old, new)

    # ================================================== 5 Discussion
    d.set_text("Accurate night-level respiratory and cardiac rates",
               "Aggregate respiratory and cardiac rates")
    d.patch("The mask recovers per-session mean respiratory rate",
            "The mask recovers per-session mean respiratory rate with MAE < 1 br/min "
            "and cardiac rate with MAE < 4 BPM after a simple per-session "
            "k-calibration.",
            "The mask recovers each night's mean respiratory rate to a median of "
            "0.14 br/min and its mean cardiac rate to 1.60 BPM after a per-session "
            "k-calibration, with a per-epoch error of 0.94 br/min and 3.36 BPM. "
            "These accuracies are conditional on calibration against a reference, "
            "and for respiration the cohort's narrow range of mean rates "
            "(14.4–16.8 br/min) means a constant predictor already achieves "
            "0.78 br/min, so the respiratory figure demonstrates calibration rather "
            "than measurement; the cardiac figure is the stronger result.")
    d.patch("Aggregate respiratory and cardiac rates" if False else
            "The mask recovers each night's mean respiratory rate",
            "The respiratory k factor is near unity (k ≈ 0.97)",
            "The respiratory k factor is near unity (k ≈ 0.96)")
    d.patch("The mask recovers each night's mean respiratory rate",
            "The cardiac k factor (≈1.95) is consistent across subjects and nights,",
            "The cardiac k factor (≈1.95) is consistent across subjects and nights "
            "— its interquartile range spans only 1.79–2.00 —")

    # 5.1 gains the overnight record; 5.2 gains the within-night null.
    d.insert_after(
        "Both respiratory and cardiac bands carry substantial energy",
        [d.para(
            "The overnight record itself is a result as well as a substrate. The "
            "sensor value is stable enough to be read in absolute femtofarads across "
            "a whole night, its excursions are step-like and posture-linked rather "
            "than a smooth instrumental drift, and both physiological rhythms can be "
            "followed as single continuous traces from the first hour to the last "
            "(Figure 1). The left–right imbalance derived from that record is a "
            "genuine dynamic quantity whose sign reverses several times a night and "
            "is not explained by head position, though its magnitude is "
            "posture-scaled and its absolute operating point is set by mask "
            "placement. It is reported here as a characterised signal property, not "
            "as a measurement of intracranial fluid movement.")])

    d.insert_after(
        "5.2 What the mask does not provide",
        [d.para("Within-night rate variation", style="Heading3"),
         d.para(
             "The mask does not follow either rate as it changes during the night. "
             "Across the seven base estimators, four channels, multi-channel fusion, "
             "CWT-ridge and STFT–Viterbi trackers, with and without calibration and "
             "smoothing, no configuration produced a within-session correlation "
             "distinguishable from a temporal shuffle null in either band. Because "
             "the battery was exhaustive, we treat this as a property of the signal "
             "at this window length rather than of a particular estimator. The "
             "consequence for deployment is concrete: the device can report a "
             "nightly average and its night-to-night trend, but must not be used "
             "where instantaneous or beat-to-beat rate matters.")])

    # 5.3 mechanism: correct the respiratory account, which blamed resolution
    # for an estimator that is no longer the one reported.
    d.patch("The absence of epoch-level cardiac tracking",
            "The absence of epoch-level cardiac tracking",
            "The absence of within-night cardiac tracking")
    d.set_text(
        "The respiratory case has a different basis, in which two factors compound.",
        "The respiratory case has a different basis. At the 30-second analysis "
        "window with 4-second Welch segments the frequency resolution is 0.25 Hz, "
        "comparable to the width of the entire 0.1–0.5 Hz respiratory band; the "
        "spectral estimator therefore has only two reachable output values and "
        "returned 0.25 Hz in 99.95% of epochs, making it a constant predictor rather "
        "than a measurement. Peak-counting estimators are not resolution-limited in "
        "this way and do vary, but the variation they are asked to resolve is itself "
        "small — a median within-night standard deviation of 1.57 br/min against a "
        "per-epoch error of comparable magnitude. Longer windows and full-window "
        "periodograms with parabolic interpolation recover resolution at the cost of "
        "temporal responsiveness without improving the within-session correlation, "
        "so the limitation is a genuine trade-off rather than a poor parameter "
        "choice.")

    # 5.4: the apnea-screening claim is not supported by anything reported.
    d.patch("The mask is suitable for screening-level overnight",
            "This could support home sleep apnea screening (mean-rate anomalies, "
            "gross rate extremes) or longitudinal tracking of resting rates across "
            "nights. However, it cannot replace PSG or chest-worn sensors for "
            "applications requiring real-time or instantaneous rate monitoring.",
            "This could support longitudinal tracking of resting rates across "
            "nights, or flagging gross rate extremes. It cannot replace PSG or "
            "chest-worn sensors where real-time or instantaneous rate is required. "
            "We note explicitly that although apnea and hypopnea events were scored "
            "in these recordings, no apnea-detection analysis is reported here, and "
            "the suitability of the mask for sleep-apnea screening is therefore "
            "untested; the event-level respiratory sensitivity such an application "
            "needs is precisely the within-night resolution the mask lacks.")

    # ================================================== 6 Limitations
    d.set_text(
        "Second, this is a single-site laboratory study.",
        "Second, all recordings were made unattended in participants' own homes "
        "rather than in a sleep laboratory. This is the intended deployment setting "
        "and is a strength for ecological validity, but it means sensor placement "
        "was performed by participants after training rather than by a technologist, "
        "and coupling quality could not be monitored during the night. Sensor "
        "coupling may vary further with head shape, hair density, and mask fit. The "
        "S6 outlier behaviour illustrates how strongly results depend on coupling "
        "quality, and a deployable system would need an automatic coupling-quality "
        "check.")

    d.set_text(
        "Fifth, the 30-second analysis window was chosen for compatibility",
        "Fifth, the 30-second analysis window was chosen for compatibility with PSG "
        "staging epochs, and it is short enough that the Welch spectral estimator is "
        "resolution-limited in the respiratory band (section 5.3). Longer windows "
        "decrease per-epoch error but do not improve tracking correlation, "
        "suggesting the within-night limitation is not an artifact of window choice; "
        "nonetheless the specific numbers reported here are tied to this window "
        "length.")

    d.insert_after(
        "Sixth, the montage derivation of our single-channel contact EEG",
        [d.para(
            "Seventh, the left–right capacitance imbalance reported in section 4.1 "
            "is characterised, not explained. Its absolute operating point is "
            "confounded by mask-mount offset and its magnitude by head position; "
            "only its direction survives the posture control, and separating a "
            "physiological lateralization from residual mount effects would require "
            "a zero-flow or head-position calibration that this protocol does not "
            "include. Several other analysis choices were also made after inspecting "
            "these data — the CRE channel was selected for the ridge analyses "
            "because it carried the dominant ridge in 9 of 12 sessions, and the "
            "cardiac estimator and fusion strategy were chosen by error on this same "
            "cohort — so these results are descriptive of this cohort and would need "
            "out-of-sample confirmation.")])

    # ================================================== 7 Conclusion
    d.set_text(
        "This study provides a rigorous, multi-method characterization",
        "This study provides a multi-method characterization of a capacitive "
        "temple-sensor sleep mask during overnight sleep, and its value lies as much "
        "in the boundaries it establishes as in the capabilities it demonstrates. "
        "The overnight sensor record is readable in absolute femtofarads across a "
        "whole night, its excursions are step-like and posture-linked, and both "
        "respiratory and cardiac rhythms can be followed as single continuous traces "
        "from the first hour to the last. From that record the mask recovers "
        "per-night mean respiratory rate to 0.14 br/min and cardiac rate to "
        "1.60 BPM with simple calibration, but does not follow either rate as it "
        "varies within the night, and we trace that failure to the biphasic "
        "morphology of the capacitive pulse rather than to insufficient signal. The "
        "sensor carries no cortical electrographic signature: sleep spindles produce "
        "no sigma-band response, confirming that it transduces mechanical and "
        "hemodynamic pulsations rather than neuronal electrical activity.")

    d.set_text(
        "These findings frame the mask as a viable tool",
        "These findings frame the mask as a viable tool for unobtrusive overnight "
        "mean-rate trending and as one input to multi-modal sleep assessment, while "
        "clarifying the boundaries of capacitive temple sensing. The characterization "
        "of both capabilities and limitations provides a foundation for future "
        "development: multi-channel cardiac fusion, multi-modal staging built on the "
        "ridge and slow-drift features, and a calibration protocol able to separate "
        "physiological lateralization from mask-mount offset.")

    # ================================================== supplementary
    d.patch("Figure S1. Estimator and channel comparison",
            "across six base estimators", "across seven base estimators")

    d.save(DOC)
    print("applied %d edits -> %s" % (d.n_patch, DOC))


if __name__ == "__main__":
    main()
