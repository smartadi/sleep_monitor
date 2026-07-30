# -*- coding: utf-8 -*-
"""Replacement prose for the reviewed manuscript.

Every number in this file was recomputed from the artifacts named beside it and
verified before being written here. Provenance:

  artifacts/mask_phase_c.parquet   -- per-epoch estimate vs reference, 6 strategies
  artifacts/mask_phase_a.parquet   -- per-channel raw estimator outputs
  artifacts/detB_{resp,card}.parquet -- responsive detector series
  reports/rates/mask/table_s1_per_session_two_regime.csv -- printed Table S1
  analysis/swa_validation/outputs/swa_validation_results.csv -- SEC vs EEG SWA
  reports/slow_wave/cap_swa/{hypothesis_summary.csv,classifier/*.csv} -- ISWA score
"""

# --------------------------------------------------------------------- abstract

ABSTRACT = (
    "Sleep-dependent intracranial fluid and pressure dynamics are increasingly implicated in "
    "brain health, but no practical technology measures them continuously during natural sleep. "
    "We evaluated a wearable single-electrode capacitance (SEC) sleep mask against full "
    "polysomnography in six healthy adults over twelve overnight recordings (9,319 thirty-second "
    "epochs), and report a systematic characterization of what the sensor does and does not "
    "measure. The SEC signal carries respiratory (0.1–0.5 Hz) and cardiac (0.5–3.0 Hz) energy "
    "well above the electronic noise floor on every channel in every recording (+18.7 to "
    "+30.0 dB), and recovers each night's mean respiratory and cardiac rate to a median of "
    "0.14 br/min and 1.60 BPM after a per-session calibration factor k. Within a night, however, "
    "neither band's rate variation is recovered by any estimator tested. We trace this to the "
    "transduction mechanism: k counts capacitive deflections per physiological event (1.70–2.43 "
    "per heartbeat, bracketing the fitted cardiac k of 1.95), and the biphasic capacitive pulse "
    "decouples peak frequency from instantaneous rate. The mask carries no cortical "
    "electrographic signature — sleep spindles produce no sigma-band response (+0.02 to "
    "+0.03 dB, against +3.3 dB on contact EEG), and SEC slow-wave activity is uncorrelated with "
    "EEG slow-wave activity (r = −0.014 ± 0.036; N3 discrimination AUC 0.490 ± 0.041, against "
    "0.740 for the EEG pipeline itself). What the sensor registers at discrete cortical events "
    "is instead a mechanical and hemodynamic accompaniment: low-band power rises at sleep-spindle "
    "centres, and rises at EEG delta-burst onsets with zero lag in all six subjects. A purely "
    "mechanical intracranial slow-wave activity (ISWA) score — slow capacitive drift under "
    "sustained quiescence — tracks N3 consistently across subjects (per-subject AUC 0.675 ± "
    "0.073, six of six above chance), where every spectral SEC feature we examined failed to "
    "generalize. Wearable capacitive sensing at the temple is therefore a mechanical and "
    "hemodynamic monitor rather than an electrographic one, suited to aggregate overnight rate "
    "trending and as one input to multi-modal sleep assessment."
)

# ----------------------------------------------------------------- introduction

INTRO_ISWA = (
    "In this study we introduce intracranial slow-wave activity (ISWA), a SEC-derived measure of "
    "the low-frequency, mechanically transduced dynamics that accompany EEG-defined slow-wave "
    "sleep. We hypothesized that wearable SEC sensors integrated into a sleep mask would be "
    "sensitive to regional intracranial mechanical and hemodynamic dynamics across sleep stages. "
    "To test this, healthy participants underwent overnight polysomnography while wearing the "
    "SEC-based sleep mask. We evaluated the relationship between SEC-derived signals and standard "
    "sleep measurements, and asked specifically whether the low-frequency content of the SEC "
    "recording reflects cortical slow-wave activity itself or its mechanical accompaniment. The "
    "answer proves to be the latter, and it defines the scope of the technology: the sensor does "
    "not reproduce the cortical slow-wave spectrum, but a mechanical ISWA signature of deep sleep "
    "is present, is time-locked to discrete cortical events, and — unlike every spectral feature "
    "we examined — is consistent across subjects. We therefore present ISWA as a mechanical and "
    "hemodynamic correlate of slow-wave sleep rather than an EEG-equivalent measure, and "
    "characterize both what wearable capacitance sensing delivers and where its boundary lies."
)

# --------------------------------------------------------- 2. sensing principles

FIG1_CAPTION = (
    "Figure 1. (a) Sensing principle reading 𝐶brain using the floating ground of the vascular "
    "network in the brain. (b) Circular sensor with high-aspect-ratio carbon nanotube-coated "
    "fibers. (c) Sleep mask composed of a plastic sensing unit attached to a cloth-based eye "
    "mask. The plastic piece is integrated with a microcontroller that collects signals for "
    "regional ICP (r-ICP), eye movements, and acceleration. Four electrodes are used: one pair "
    "over the left and right temple regions, whose difference forms the differential channel "
    "CLE−CRE, and a forehead electrode CH. Throughout the Results the two temple channels are "
    "denoted CLE and CRE and the forehead channel CH."
)

SEC2_SENSITIVITY = (
    "Considering the 120 mm penetration depth reported for this electrode geometry,26 the "
    "cerebral vascular network geometry is assumed to remain relatively consistent. Under this "
    "assumption, and with a fixed sensor-to-skin distance, Ctotal is primarily attributed to "
    "changes in mean ICP. In a pig model, linear analysis of the SEC response against invasive "
    "pressure gave sensitivities of 4.8 ± 2.5 fF/mmHg for respiratory ICP oscillations and "
    "18.0 ± 5.3 fF/mmHg for displacement-induced ICP changes, enabling conversion of capacitance "
    "fluctuations into predicted ICP amplitudes. The SEC signals also showed high correlation "
    "with invasive pressure measurements, with absolute correlation coefficients of 0.96 ± 0.04 "
    "for ICP and 0.93 ± 0.10 for central venous pressure (CVP).21"
)

SEC2_PLACEMENT = (
    "In the sleep mask, one electrode pair is positioned over the left and right temple regions "
    "and a further electrode over the forehead. The temple electrodes (CLE and CRE) are "
    "sensitive to regional ICP changes and also register eye movements; because they sit "
    "near-symmetrically about the midline, their difference CLE−CRE cancels common-mode drift "
    "while their average preserves common-mode physiological signal. The forehead electrode (CH) "
    "provides an independent measurement over a different cranial region. The channels were "
    "verified to be electrically and capacitively independent, so that agreement between them "
    "can be used to distinguish physiological signal from motion artifact and environmental "
    "drift rather than being assumed."
)

SEC2_REPRO = (
    "Importantly, the SEC sensor measures ΔC rather than bioelectric potentials, making it "
    "insensitive to intrinsic neuronal electrical activity. The protruded conductive fiber "
    "structure of the CPC electrode further enhances sensitivity compared to standard metal "
    "electrodes21,22,25, while maintaining reproducible fabrication with 2.2% variation in "
    "initial capacitance (n = 36)26. In bench characterization against conventional metal "
    "electrodes the CPC electrode achieved a signal-to-noise ratio of 12.3 (linear power ratio, "
    "≈10.9 dB), substantially higher than metal controls27; the in-band SNR measured on human "
    "overnight recordings in this study is reported separately in §4.1 and uses a different "
    "definition. Unlike impedance- or voltage-based systems, the SEC platform does not require "
    "voltage amplification, enabling stable measurement of ultra-low-frequency physiological "
    "signals (<0.1 Hz), including baseline offset changes. Frequency-counting capacitance "
    "measurements provide exceptional long-term stability (~10 fF/day drift), and operation at "
    "an excitation frequency (25 kHz, 1.2 V-amplitude) minimizes susceptibility to external "
    "electromagnetic interference. With ergonomic features such as restroom-access cutouts, low "
    "power consumption (30 mW) enabling 25-hour operation, and electric field below 4 V/m "
    "(< IEEE safety standards: 614 V/m at 25 kHz28), this system represents a clinically viable "
    "and user-friendly advancement over existing home sleep testing technologies."
)

# ------------------------------------------------------------------- 3. methods

M31_EPOCHS = (
    "Participant demographics and recording characteristics are summarized in Table 1. The six "
    "participants included four males and two females with ages ranging from 25 to 66 years and "
    "Pittsburgh Sleep Quality Index (PSQI) scores ranging from 4 to 9. Across all recording "
    "sessions, a total of 9,319 non-overlapping 30-second analysis epochs were obtained. Ethical "
    "approval covered an amendment to the original protocol (MOD00020664, Modification #1 to "
    "STUDY00018275)."
)

M32_CANCELLER = (
    "Motion artifact was suppressed by regressing band-limited accelerometer energy out of each "
    "SEC channel. The accelerometer magnitude and the SEC channel were first bandpassed to the "
    "analysis band of interest (respiratory 0.1–0.5 Hz, ≈6–30 breaths/min; cardiac 0.5–3.0 Hz, "
    "≈30–180 beats/min) so that only motion energy within that band was removed. Two cancellers "
    "were implemented: an ordinary-least-squares (OLS) projection that removes a single "
    "stationary coupling coefficient, and a normalized-LMS (NLMS) adaptive FIR canceller "
    "(16 taps, µ = 0.05) that tracks time-varying coupling when posture or sensor contact "
    "drifts. All results reported here use the OLS canceller; the NLMS variant was implemented "
    "for robustness to posture change but did not alter any reported metric and was not used. "
    "Bandpass filtering used third-order zero-phase Butterworth filters, except in the "
    "peri-onset analysis of §4.5, which uses a strictly causal envelope estimator."
)

M32_CHANNEL = (
    "Unless otherwise noted, the canonical analysis channel for rate estimation is the OLS "
    "differential CLE−CRE, which cancels common-mode drift between the two temple electrodes. "
    "This choice trades signal amplitude for drift rejection: because the two temple electrodes "
    "are near in phase, differencing also cancels part of the common-mode physiological signal "
    "and yields a lower in-band SNR than either electrode alone (Figure 3). It was retained for "
    "rate estimation because drift rejection dominates rate accuracy, but the individual "
    "channels (CH, CLE, CRE) and their average are used for the signal-validation, ridge, "
    "spindle, and delta-onset analyses, where absolute band power rather than drift stability is "
    "the quantity of interest."
)

M33_CONSENSUS = (
    "To validate this consensus, we compared its two most physically independent reference "
    "signals: nasal airflow (Flow), a pressure transducer, and the respiratory inductance "
    "plethysmography sum (RIPSum), the polarity-corrected sum of the thoracic and abdominal "
    "effort belts that approximates thoraco-abdominal volume change.29 Because airflow and "
    "inductance-belt volume transduce respiration by independent physical mechanisms, agreement "
    "between them on within-session rate variation reflects shared physiology rather than "
    "estimation noise from any single signal. Within-session correlation was r = +0.47 on raw "
    "rates and r = +0.27 on detrended fluctuations, confirming that within-session respiratory "
    "rate variation is real physiology. The consensus reduced single-sensor jitter (standard "
    "deviation 2.26 → 1.61 br/min) and provided 100% epoch coverage. The median absolute "
    "difference between consensus and Flow-only was 0.06 br/min, though 29% of epochs differed "
    "by more than 1 br/min. This inter-sensor disagreement represents a floor on the uncertainty "
    "achievable by any external respiratory rate sensor."
)

M33_GRID = (
    "All rates were computed on a common grid of non-overlapping 30-second windows aligned to "
    "the consensus epoch grid, matching the 30-second epochs of the PSG technologist's AASM "
    "scoring. Sleep stages were taken from that scoring."
)

M34_COHERENCE = (
    "For each analysis epoch, magnitude-squared coherence was computed between the SEC channel "
    "and the corresponding PSG reference (nasal airflow for respiration, ECG for cardiac; the "
    "single-sensor references are used here rather than the multi-sensor consensus of §3.3 "
    "because coherence requires a continuous waveform, not an epoch-wise rate), and the "
    "coherence value was read at the ground-truth rate frequency. Spectral agreement was "
    "quantified as the fraction of epochs whose SEC peak frequency fell within ±0.05 Hz of the "
    "reference. To bound what coherence is attainable between two genuine but physically "
    "distinct measurements of the same rhythm, we computed the same statistic between two PSG "
    "channels — Flow versus RIPSum for respiration and ECG versus PPG for cardiac — and report "
    "it as the canonical bound. To guard against spurious coherence from band-limited noise, "
    "phase-randomized surrogates (200 per epoch) were generated that preserve the power spectrum "
    "while destroying phase structure; the fraction of epochs whose observed coherence exceeded "
    "the surrogate null at p < 0.05 was reported, and compared against the 5% rate expected "
    "under the null. Coherence was evaluated within each sleep stage to test whether coupling "
    "persists beyond wake."
)

M35_ESTIMATORS = (
    "Per-window rates were estimated from the bandpassed SEC channel using seven base "
    "estimators: spectral (Welch PSD peak frequency), autocorrelation (dominant ACF lag with "
    "parabolic interpolation), Hilbert instantaneous-frequency median, upward zero-crossing "
    "rate, prominence-thresholded peak counting with loose and strict thresholds (peaks_loose, "
    "peaks_strict), and a spectral-guided amplitude-adaptive peak detector. The Welch estimator "
    "used 4-second segments with 50% overlap at the 100 Hz sampling rate, giving a frequency "
    "resolution of 0.25 Hz. Additional advanced trackers were evaluated for the harder cardiac "
    "band: continuous-wavelet-transform (CWT) ridge tracking and STFT peak tracking with Viterbi "
    "temporal smoothing. Of these, only loose peak counting entered the reported pipeline; the "
    "consequences of the spectral estimator's coarse resolution are quantified in §4.2 and §5.3."
)

M35_K = (
    "SEC peak counts are systematically scaled relative to the PSG reference because the "
    "capacitive waveform is not a clean one-cycle-per-event signal. We correct this with a "
    "per-session scalar k, defined as the median ratio between the SEC estimate and the "
    "ground-truth rate across randomly selected calibration windows, so that the calibrated rate "
    "equals the raw estimate divided by k. Calibration used 50 randomly drawn one-minute windows "
    "and was verified against the whole-night k: |k_diagnostic − k_whole| ≤ 0.04 for all "
    "sessions. Because k is fitted against the reference, any accuracy metric computed after "
    "k-scaling is conditional on that calibration; we therefore report both uncalibrated and "
    "k-scaled accuracies and, in §4.2, separate the part of the aggregate accuracy that k "
    "supplies from the part the sensor supplies."
)

M35_PIPELINE = (
    "The operational pipeline was therefore a per-window estimator followed by k-scaling and a "
    "causal three-epoch median filter. For the cardiac band this was loose prominence-based peak "
    "counting fused across five channels by agreement gating. For the respiratory band we report "
    "the same agreement-gated fusion of loose peak counting: the Welch spectral estimator "
    "attains a marginally lower per-epoch error but is degenerate at this window length (§4.2), "
    "and was therefore not adopted. Multi-channel quality-weighted and agreement-gated fusion "
    "changed per-epoch error by less than 0.1 in either band relative to the single differential "
    "channel, but the fused estimator is non-degenerate and is preferred for that reason."
)

M36_RIDGES = (
    "Whole-night SEC spectrograms show structured spectral ridges. In each analysis window we "
    "located spectral peaks and linked them across consecutive windows into persistent ridges, "
    "requiring a peak to stay within a band-dependent frequency tolerance between windows and to "
    "persist for at least 5 minutes. Each ridge is summarized by its frequency, amplitude, "
    "prominence above the local spectral background, and a flatness index (1 − coefficient of "
    "variation of its frequency over its lifetime). To prevent a single physical ridge from "
    "being reported as several short fragments, fragments separated by short dropouts were "
    "stitched together by matching their boundary-median frequencies, and each ridge's trace was "
    "interpolated across internal gaps. Flatness and drift were measured on the raw "
    "pre-interpolation trace, and ridge coverage (the fraction of a ridge's span actually "
    "detected before interpolation) is reported so that continuity is quantified rather than "
    "assumed."
)

M36_BANDS = (
    "Ridge detection was run in three physiological bands, each with an analysis window matched "
    "to the frequency resolution that band requires: a slow band, 0.02–0.15 Hz, using 240 s "
    "windows with 120 s Welch segments (≈0.008 Hz resolution), covering the infra-slow range "
    "below respiration; the respiratory band, 0.1–0.5 Hz, using a single 30 s periodogram per "
    "window (≈0.033 Hz resolution); and the cardiac band, 0.5–3.0 Hz, using 30 s windows with "
    "8 s Welch segments, which permits integer-ratio harmonics to be resolved. Per-epoch ridge "
    "features — the number of active ridges, the lowest ridge frequency, total ridge power, "
    "frequency spread, and mean flatness — were computed on the CRE channel, which carried the "
    "dominant ridge in 9 of 12 sessions. Because that channel was selected from the same data, "
    "the ridge results below are descriptive of this cohort and are reported with per-subject "
    "direction counts alongside every pooled test. Separately, because a non-sinusoidal "
    "respiratory waveform produces integer harmonics, we tested whether concurrent ridges form "
    "integer-ratio ladders on the CH channel over 0.1–3.0 Hz against a per-session, per-count "
    "surrogate null: for each number of concurrent ridges k, 400 random sets of k frequencies "
    "were drawn uniformly over the session's observed ridge-frequency range and their best "
    "integer alignment scored, taking the 95th percentile as the null threshold."
)

M37_STATS = (
    "All rate accuracy metrics are reported within-session to avoid inflation from "
    "between-session mean matching. Two error definitions are used and are kept distinct "
    "throughout: the night-level error, the absolute difference between the mean estimated and "
    "mean reference rate over a whole recording; and the per-epoch error, the median absolute "
    "error across the epochs of a recording (\"MAE\" denotes this median absolute error "
    "throughout, chosen for robustness to outliers). Cross-session summaries use the median and "
    "interquartile range of per-session values, so that the unit of analysis is the recording "
    "and not the epoch. Within-session estimate-versus-reference association uses the Pearson "
    "correlation across epochs within a night, summarized across nights by the median and a "
    "Wilcoxon signed-rank test on the twelve per-night values. Non-parametric group comparisons "
    "use the Kruskal–Wallis test across stages, with the Mann–Whitney U test for N3-versus-rest "
    "contrasts. Where these stage tests are computed on pooled epochs, the epochs within a night "
    "are not independent and the resulting p-values are anti-conservative; we therefore report "
    "them only alongside the per-subject direction count, which uses the subject as the unit of "
    "analysis, and we treat the direction count as the evidence. With six subjects a "
    "subject-level Wilcoxon test floors at p = 0.031, so subject-level effects are reported as "
    "direction counts and effect sizes rather than as p-values. Correlations with subject-level "
    "markers use Spearman's rank correlation with an exact permutation p-value (all 720 "
    "permutations at n = 6) rather than the asymptotic approximation, with Bonferroni correction "
    "across the family of tests reported."
)
