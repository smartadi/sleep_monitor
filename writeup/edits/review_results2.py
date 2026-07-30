# -*- coding: utf-8 -*-
"""Replacement prose, part 2: sections 4.3-4.6, Discussion, Limitations,
Conclusion and Supplementary.

Ridge numbers from writeup/edits/ridge_harmonic_revamp_section_draft.md and
reports/slow_wave/revamp/. SWA numbers recomputed from
analysis/swa_validation/outputs/swa_validation_results.csv and
reports/slow_wave/cap_swa/.
"""

# ------------------------------------------------------- 4.3 ridges (3 bands)

R43_HEADING = "4.3 Persistent spectral ridges track respiration, heart rate, and an infra-slow rhythm"

R43_CONTINUITY = (
    "SEC spectrograms displayed persistent spectral ridges across full overnight recordings. "
    "Linking and gap-filling the spectral peaks yields ridges that persist without interruption "
    "for long stretches of the night: on the CRE channel the mean per-ridge trace coverage was "
    "1.00, meaning the reported ridges have no internal holes, and the longest continuous ridge "
    "averaged 150 minutes in the respiratory band and 124 minutes in the slow band. A "
    "representative night (Figure 6) shows a respiratory ridge holding near 0.25 Hz across the "
    "whole recording, intermittent cardiac ridges between 0.6 and 1.8 Hz, and a continuous slow "
    "ridge near 0.07 Hz. The ridges therefore fall into three bands: an infra-slow rhythm below "
    "respiration, the respiratory rate itself, and an intermittent cardiac ridge."
)

R43_SLOW = (
    "An infra-slow ridge is near-ubiquitous and weakens in deep sleep. A persistent ridge in the "
    "0.02–0.15 Hz band was present in 88–92% of clean epochs on every capacitive channel, with a "
    "lowest frequency near 0.07 Hz (≈14 s period) — an infra-slow oscillation the sensor carries "
    "throughout the night. Its total ridge power fell markedly during N3 in all six subjects "
    "(pooled median 0.08 versus 0.37 in other stages; N3-versus-rest AUC ≈ 0.68 in the low-power "
    "direction; pooled Kruskal–Wallis p = 9×10⁻⁵³, reported here only as a descriptive statistic "
    "because pooled epochs are not independent — the six-of-six subject direction count is the "
    "evidence). This is the most subject-consistent capacitive feature in this study, and it is "
    "the same effect, measured spectrally, that the mechanical ISWA score of §4.6 captures in "
    "the time domain: deep sleep quiets the infra-slow band. The lowest slow-ridge frequency was "
    "marginally higher in N3 in four of six subjects, a weaker and less consistent effect."
)

R43_RESP_CARD = (
    "Respiratory-band ridges track breathing and thin out in N3. A respiratory ridge was present "
    "in almost every clean epoch, its lowest frequency sitting at a median of ≈0.23 Hz "
    "(≈14 breaths/min) in every stage — that is, it is the respiratory rate, which confirms that "
    "the ridge tracks respiration. N3 carried fewer active respiratory ridges (N3-lower in all "
    "six subjects) and lower total ridge power (five of six subjects), consistent with the more "
    "regular, lower-effort breathing of deep sleep. Cardiac-band ridges were intermittent; when "
    "present, their lowest frequency was lower in N3 (median 1.00 Hz, ≈60 BPM, versus 1.13 Hz, "
    "≈68 BPM, in other sleep; five of six subjects), mirroring the sleep bradycardia of deep "
    "NREM. Because the cardiac ridge frequency in N3 approaches the 0.5 Hz lower edge of the "
    "analysis band, this shift is reported as a direction rather than a calibrated frequency "
    "(Figure 7)."
)

R43_LADDER = (
    "Harmonic ladders are real, not coincidental. Across the twelve nights, 6,812 windows "
    "contained a strong (≥3-member) integer-ratio ladder on the CH channel, and 6,783 of them "
    "(99.6%; 98.9–100% per session) exceeded the per-session surrogate null — integer alignment "
    "at this rate essentially never occurs by chance. The surviving ladders had a median "
    "fundamental of 0.25 Hz, the respiratory rate, a median of four members extending into the "
    "cardiac band, and a median confidence of 0.75 (Figure S6). The harmonic structure therefore "
    "reflects a genuinely non-sinusoidal respiratory waveform transduced by the sensor, rather "
    "than coincidental overlap of independent rhythms — and it is the spectral counterpart of "
    "the biphasic pulse morphology that sets the cardiac k factor (§4.2.1)."
)

R43_SUMMARY = (
    "All of these associations reflect autonomic, respiratory, and infra-slow mechanical rhythms "
    "transduced by the sensor rather than cortical activity. The slow-band N3 effect is the most "
    "subject-consistent; none is strong enough to serve as a standalone sleep-stage classifier, "
    "and confirming their inter-individual consistency would require a larger cohort, but the "
    "slow-band power in particular could contribute to a multi-modal staging feature set."
)

FIG6_CAPTION = (
    "Figure 6. Representative session (S1N1): capacitive spectrogram (CRE) with continuous "
    "persistent ridges overlaid in three bands — respiratory (~0.25 Hz) and cardiac (0.6–1.8 Hz) "
    "on the 0–3 Hz spectrogram, and an infra-slow ridge (~0.07 Hz) on the 0.02–0.15 Hz "
    "spectrogram — aligned with the PSG hypnogram."
)

FIG7_CAPTION = (
    "Figure 7. Persistent-ridge features by sleep stage (CRE channel, 12 sessions), one row per "
    "band (slow / respiratory / cardiac). Columns: lowest ridge frequency, total ridge power, "
    "and mean flatness. Slow-band total ridge power (top-middle) drops in N3; cardiac lowest "
    "ridge frequency (bottom-left) slows in N3 and N2. Pooled Kruskal–Wallis p-values are shown "
    "above each panel and are descriptive only, since epochs within a night are not independent; "
    "per-subject direction counts are given in the text."
)

# -------------------------------------------------------------- 4.4 spindles

R44_LOWBAND = (
    "Onset-triggered averaging shows that low-band (0–3 Hz) capacitive power increases at the "
    "spindle centre across every channel. The trial-averaged increase reaches +0.47 to +0.55 dB "
    "on the temple channels and +0.58 dB on the forehead channel (CH) (Figure 8). It is "
    "reproducible across all twelve nights, is absent from randomly chosen N2 timepoints, and "
    "persists for spindles that do not coincide with a scored arousal — indicating that it "
    "tracks the spindle events themselves rather than co-occurring gross movement."
)

FIG8_CAPTION = (
    "Figure 8. The capacitive response at sleep spindles is mechanical, not electrical. "
    "(A) Onset-triggered average of capacitive low-band (0–3 Hz) power across channels; power "
    "rises at the spindle centre, strongest on the forehead channel (CH). (B) Per-spindle "
    "power-change distribution on CH for the low band (0–3 Hz) versus the sigma band (11–16 Hz): "
    "the low band shifts positive (mean +0.58 dB) while the sigma band is null (mean +0.03 dB). "
    "(C) Per-session low-band power change at spindles for each capacitive channel, showing the "
    "increase is present across all twelve nights."
)

# ---------------------------------------------------------- 4.5 delta onsets

R45_METHOD = (
    "Cortical slow-wave (delta) activity provides a second discrete cortical event at which to "
    "ask what the temple sensor reflects: the onset of a delta burst. On the contact EEG we "
    "detected delta-burst onsets as sustained rises of the 0.5–4 Hz envelope above a per-night "
    "threshold (Schmitt trigger on the Hilbert envelope, high threshold at the per-session robust "
    "NREM baseline plus two median absolute deviations, sustained for at least 4 s, with the "
    "onset walked back to the rising edge), retaining only onsets that emerged from a quiescent, "
    "motion-free baseline in NREM sleep — 6 to 108 qualifying onsets per night, predominantly "
    "isolated N2 slow-wave and K-complex onsets, since sustained N3 slow-wave activity has no "
    "discrete quiet onset. For each onset we averaged the capacitive power in three low-frequency "
    "bands (0–0.5, 0.5–1, and 1–3 Hz; the 0–0.5 Hz band high-passed at 0.03 Hz to reject DC and "
    "coupling drift) on the two temple channels (CLE, CRE) and the forehead channel (CH), "
    "relative to count-matched, motion-free NREM baseline windows, aggregating within each "
    "subject and then across the six subjects so that no analysis pools raw epochs."
)

# ------------------------------------------------------------------ 4.6 ISWA

R46_HEADING = "4.6 Slow-wave sleep: no cortical slow-wave activity, but a reproducible mechanical ISWA signature"

R46_INTRO = (
    "The spindle and delta-onset analyses establish that the sensor responds mechanically at "
    "discrete cortical events. The question the Introduction poses — whether SEC recordings "
    "carry a usable correlate of slow-wave sleep — requires a tonic rather than event-locked "
    "test, and has two parts that must be separated: whether the capacitive signal reproduces "
    "cortical slow-wave activity spectrally, and whether any capacitive quantity marks deep "
    "sleep at all."
)

R46_NEGATIVE = (
    "The capacitive signal does not carry cortical slow-wave activity. Computing a slow-wave "
    "activity (SWA) index on the capacitive channel exactly as it is computed on the contact EEG "
    "— band power in 0.5–4 Hz and in its four constituent sub-bands, per 30-second epoch — and "
    "correlating the two within each session gives a mean Pearson r of −0.014 ± 0.036 across the "
    "twelve recordings, with every individual session falling between −0.10 and +0.05. Mean "
    "magnitude-squared coherence between the two SWA time series was 0.003 ± 0.005, "
    "indistinguishable from zero. Using capacitive SWA to discriminate technologist-scored N3 "
    "from other sleep gave an AUC of 0.490 ± 0.041, at chance and below chance in eight of "
    "twelve sessions. The same pipeline applied to the contact EEG recovered N3 at an AUC of "
    "0.740 ± 0.056, above 0.65 in every session and higher than the capacitive value in all six "
    "subjects, confirming that the negative result reflects the capacitive signal rather than a "
    "failure of the analysis (Figure S7). Capacitive temple sensing therefore does not "
    "substitute for EEG in measuring slow-wave activity, and the low-frequency capacitive power "
    "that a spectral analysis reports in the delta band is not of cortical origin."
)

R46_POSITIVE = (
    "A mechanical ISWA score nonetheless marks deep sleep consistently across subjects. Because "
    "the sensor transduces mechanical and hemodynamic rather than electrical events, we defined "
    "a per-epoch ISWA score from three criteria fixed a priori, deliberately excluding heart "
    "rate, respiratory rate, and head movement so that those remained independent quantities to "
    "test: the absolute slope of the CLE−CRE DC level over a 2.5-minute rolling window (slow "
    "capacitive drift); the corresponding slow change in thoracic respiratory-effort amplitude; "
    "and quiescence, measured as low accelerometer RMS. Each criterion was converted to a "
    "per-session percentile and the three combined by geometric mean, so that failing any one "
    "criterion suppresses the score. Treated as an N3 detector and evaluated "
    "leave-one-subject-out, the score reached a per-subject AUC of 0.675 ± 0.073 (range "
    "0.575–0.759) with every one of the six subjects above chance, and a pooled AUC of 0.670 "
    "(Figure 10). This cross-subject consistency of direction is the result: it is the only "
    "capacitive N3 marker in this study that does not change sign between subjects, in contrast "
    "to the ridge features of §4.3 and to capacitive SWA above. Expanding to the eight "
    "underlying mechanical features raised the leave-one-subject-out AUC only to 0.692 ± 0.090, "
    "indicating the composite is not underfitting. At the pre-specified operating point — a "
    "score above 0.60 sustained for at least four consecutive epochs — the score recovered N3 at "
    "precision 0.22, about 2.6 times the 8.4% N3 base rate, and recall 0.38."
)

R46_MASKONLY = (
    "The score as defined uses the thoracic effort belt, which is not part of the wearable. A "
    "mask-only variant built from the two mask-derived criteria alone — slow capacitive DC drift "
    "and accelerometer quiescence — performed comparably and slightly more consistently across "
    "subjects (pooled AUC 0.666; per-subject 0.671 ± 0.042), showing that the effect is carried "
    "mainly by the mask itself. Slow capacitive drift is the dominant contributor (drift alone, "
    "pooled AUC 0.664), while quiescence alone is much weaker (0.551), so the signal is not "
    "merely a stillness detector."
)

R46_HYPOTHESES = (
    "Having fixed the definition, we tested six a priori predictions about the autonomic and "
    "respiratory behaviour of this state, reporting per-subject direction counts because a "
    "subject-level test at n = 6 floors at p = 0.031 and no contrast survives Bonferroni "
    "correction (smallest corrected p = 0.22). Five predictions were not borne out, and the "
    "pattern of failure is informative. Heart rate had been predicted to rise; it fell in all "
    "six subjects (median −1.4 BPM), the bradycardia expected of consolidated deep sleep and an "
    "independent confirmation that the mechanically defined state is physiologically deep. "
    "Respiratory rate was essentially unchanged (median +0.08 br/min, 4 of 6). The "
    "capacitive-versus-thoracic respiratory-rate deviation had been predicted to jump; it shrank "
    "in all six subjects, meaning mask and effort belt agree better during the ISWA state, not "
    "worse. The predicted divergence between photoplethysmographic and capacitive cardiac "
    "frequency reversed: the PPG cardiac peak was flat while the capacitive cardiac peak rose "
    "(+0.25 Hz, 6 of 6) — a harmonic decoupling of the biphasic capacitive pulse from true heart "
    "rate, the same mechanism that limits within-night cardiac tracking in §4.2. Distinct head "
    "movements did not precede ISWA onsets: against a matched-random null the pre-onset movement "
    "rate was below chance (median lift 0.69), so the state does not merely follow a settling "
    "transient. Only EEG delta behaved as predicted, rising during the ISWA state in five of six "
    "subjects — a weak but reassuring convergence with conventional slow-wave sleep."
)

FIG10_CAPTION = (
    "Figure 10. A mechanical ISWA score marks deep sleep. (A) Leave-one-subject-out N3 "
    "discrimination for the ISWA score, per subject; all six subjects lie above chance "
    "(per-subject AUC 0.675 ± 0.073). (B) Heart rate during ISWA epochs versus matched non-ISWA "
    "NREM epochs, per subject; the predicted increase is contradicted in all six subjects "
    "(median −1.4 BPM). (C) The six a priori hypotheses with their observed per-subject "
    "directions. (D) Precision and recall for N3 as a function of the score threshold, for raw "
    "and sustained-bout labelling; the pre-specified 0.60 sustained operating point is marked."
)
