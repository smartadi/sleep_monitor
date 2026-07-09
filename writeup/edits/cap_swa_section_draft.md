# CAP-SWA mechanical/autonomic definition — ready-to-paste manuscript text

**Placement.** This fills the parked **Part C** of the outline ("does the mask see
SWS the way EEG does?") and is the reframed *positive* that follows the two cortical
negatives: §3.5 (capacitive vs contact-EEG slow-wave activity, r = 0.015 — the mask
does not transduce cortical SWA spectrally) and §3.6 (sleep spindles absent, AUC 0.50).
Recommended framing: *the mask does not see cortical SWA electrically, but a purely
mechanical/autonomic signature of deep sleep is present and, unlike every spectral CAP
feature we tried, generalizes across subjects.* Report as an **operational, descriptive
"CAP-SWA" state**, not as EEG-equivalent SWS detection.

**Honest-negative rule (OUTLINE rule 3):** stage-wise autonomic claims are reported as
per-subject direction counts, not p-values (n = 6 floors the Wilcoxon at p = 0.031).
AUCs are discrimination metrics and are reported.

Analysis: `analysis/slow_wave/cap_swa_definition.py` (definition + hypothesis tests),
`analysis/slow_wave/swa_classifier_experiment.py` (ablation + tuning).
Figure: `writeup/figures/cap_swa/fig_cap_swa_definition.png` (`make_cap_swa_figure.py`).
Data: `reports/slow_wave/cap_swa/{all_epoch_features.parquet, hypothesis_summary.csv,
movement_initiation.csv, classifier/*.csv}`.

---

## Methods addition (new subsection, pairs with §2.8–2.9)

**2.x A mechanical CAP-SWA definition.** Because the capacitive signal does not carry
cortical slow-wave activity spectrally (§3.5), we asked whether deep sleep leaves a
*mechanical* signature the mask can see. We defined a per-epoch CAP-SWA candidate score
from three criteria we held with high confidence a priori, and — importantly —
deliberately excluded heart rate, respiratory rate, and head movement from the
definition so those remained independent quantities to test (§3.x). The three
definitional criteria were: (D1) slow change in mean capacitance, the absolute slope of
the CLE−CRE DC level over a 2.5-min rolling window; (D2) slow change in thoracic
respiratory-effort amplitude; and (D3) quiescence, low accelerometer RMS. Each criterion
was converted to a per-session percentile in [0, 1] (low raw value → high, SWA-like
score), and the three were combined by geometric mean so that failing any single
criterion suppresses the score. A binary candidate label marked epochs whose score
exceeded 0.60 in sustained runs of ≥4 epochs (≥2 min). Threshold and weighting were
examined post hoc (§3.x): the 0.60 cut coincides with the maximum F1 for recovering N3,
and equal weighting of the three criteria lies within 0.015 AUC of the best swept
weighting, so neither was tuned to the outcome.

---

## Results addition (Part C)

**3.x A mechanical CAP-SWA score marks deep sleep consistently across subjects.**
Although the capacitive signal does not reproduce the cortical slow-wave spectrum, the
mechanical CAP-SWA score aligns with technologist-scored N3. Treating the score as an N3
detector within each subject (leave-one-subject-out is unnecessary for a single
pre-specified score; we report the per-subject discrimination directly), the median
N3 vs non-N3 score separation gave an AUC of **0.68 pooled** and, critically, **0.675 ±
0.073 per subject with every one of the six subjects above chance** (range 0.575–0.759;
Figure X-A). This cross-subject *consistency of direction* is the key result: it is the
only CAP-derived N3 marker in this study that does not flip sign between subjects, in
contrast to the harmonic-ridge features (mixed direction, leave-one-subject-out AUC 0.51,
§3.7) and the raw capacitive mean level (subject-dependent, AUC 0.40–0.57). A
leave-one-subject-out classifier confirmed the score is not overfit: the composite alone
reached 0.675 per-subject AUC and the eight underlying mechanical features 0.692 ± 0.09,
both exceeding the ridge-feature (0.51) and full capacitive-feature (0.56) baselines. As
an operating point, the 0.60 sustained-bout threshold recovered N3 at precision 0.22
(≈2.6× the 8.4 % N3 base rate) and recall 0.38, with the sustained-run requirement
trading recall for precision as intended (Figure X-D).

The score partly depends on the thoracic effort belt, which is not part of the wearable.
We therefore checked a capacitive-only variant using only the two mask-derived criteria
(slow DC drift + accelerometer quiescence). It performed comparably and slightly more
consistently (pooled AUC 0.666; per-subject 0.671 ± 0.042), showing that the effect is
carried mainly by the mask itself — the slow DC-drift criterion is the dominant
contributor (DC-only AUC 0.664), while quiescence alone is weakest (0.551).

**3.x The autonomic character of the CAP-SWA state contradicts the prior hypotheses.**
Having fixed the definition, we tested six a priori hypotheses about what the autonomic
and respiratory variables do during CAP-SWA (Figure X-C; per-subject direction, n = 6).
Five were not borne out. Heart rate was predicted to *rise*; instead it *fell* during
CAP-SWA in all six subjects (median −1.4 BPM), the bradycardia expected of consolidated
deep sleep and an independent confirmation that the mechanically-defined state is
physiologically deep (Figure X-B). Respiratory rate was essentially unchanged (median
+0.08 br/min, 4/6). The capacitive-vs-thoracic respiratory-rate deviation was predicted
to jump; it slightly *shrank* in all six subjects, i.e. the mask and the effort belt
agree *better* during CAP-SWA, not worse. The predicted photoplethysmographic-vs-
capacitive cardiac-frequency divergence *reversed*: the PPG cardiac peak was flat while
the capacitive cardiac peak *rose* (+0.25 Hz, 6/6), a harmonic decoupling of the biphasic
capacitive pulse from true heart rate rather than the predicted split. Only EEG delta
behaved as expected, rising during CAP-SWA in 5/6 subjects — a reassuring but weak
convergence with conventional slow-wave sleep. Finally, distinct head movements did *not*
precede CAP-SWA onsets: against a matched-random null, the observed pre-onset movement
rate was *below* chance (median lift 0.69), so the state does not initiate from a settling
transient. None of these contrasts survives Bonferroni correction at n = 6; per-subject
direction counts are the evidence, consistent with the reporting standard used for the
harmonic and tracking results.

---

## Suggested table

**Table X. The CAP-SWA state: definition, discrimination, and tested hypotheses.**

| Quantity | Role | Result |
|---|---|---|
| CAP-SWA score (slow DC + slow thorax + quiescence) | definition | per-subject N3 AUC 0.675 [0.575–0.759], 6/6 > chance |
| Capacitive-only score (slow DC + quiescence) | definition (wearable) | pooled AUC 0.666; per-subject 0.671 ± 0.042 |
| H4 Heart rate | tested | predicted ↑; observed ↓ −1.4 BPM (6/6) — **falsified** (bradycardia) |
| H5 Respiratory rate | tested | predicted ↑; observed flat +0.08 br/min (4/6) — **not supported** |
| H6 CAP–thorax rate deviation | tested | predicted ↑ jump; observed ↓ shrinks (6/6) — **falsified** |
| H7 PPG–CAP cardiac divergence | tested | predicted diverge; observed reverses, CAP freq ↑ +0.25 Hz (6/6) — **falsified** |
| H8 EEG delta power | tested | predicted ↑; observed ↑ (5/6) — **supported (trend)** |
| H2 Movement precedes onset | tested | predicted yes; observed below-chance (lift 0.69) — **not supported** |

---

## Discussion addition (fold into the deep-sleep / limitations paragraph)

The mask does not measure cortical slow-wave activity electrically (§3.5) or sleep
spindles (§3.6), but a purely mechanical/autonomic signature of deep sleep — slow
capacitive DC drift under sustained quiescence, accompanied by bradycardia — tracks N3
consistently across subjects (per-subject AUC ≈ 0.68), where every spectral capacitive
feature we examined failed to generalize. We interpret this as an *intracranial/autonomic*
slow-wave-activity correlate rather than a cortical one: it reflects the postural and
hemodynamic stillness of consolidated deep sleep, not neuronal delta. The value is modest
and the cohort small (six subjects), and the state is defined in part by a research
effort belt, though a mask-only variant performs comparably. The professor's autonomic
predictions for this state — an elevated heart rate, a large capacitive-vs-belt
disagreement, a PPG/CAP frequency split, and a movement-triggered onset — were contradicted
in a consistent per-subject direction; the corrected picture (deep-sleep bradycardia with
*improved* mask–belt agreement) is the more useful physiological description for future
work.
