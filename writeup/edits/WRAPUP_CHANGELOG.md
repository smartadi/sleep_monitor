# Wrap-up edits applied to CAP_sleep_mask_manuscript_main.docx (2026-08-17)

- removed the duplicated author line
- abstract written (3 paragraphs) and keywords line added
- 3.1 Overnight testing styled as Heading2, matching 3.2-3.7
- Table 1: 6 doubled subject IDs fixed, 'Ages' header corrected to 'Age'
- reference 28 rejoined into a single entry
- added a References heading above the bibliography
- removed a stray empty Heading2 paragraph

## Mechanical-wording pass (2026-08-17)

- abstract: leads with the mechanical reading instead of the electrographic negative
- 1. Introduction: the question is electrical pickup, and the answer says what is picked up
- 4.4: plain statement, trailing restatement dropped
- 4.5: plain statement, trailing restatement dropped
- 5.2: rewritten to lead with the response, then say what kind of response it is
- 7: conclusion matched to the same wording

## Plain-language pass (2026-08-17)

- 2: two garbled sentences repaired (pig-model sensitivity, sensor placement)
- 3.5: simplified; phantom 'randomly selected calibration windows' removed; pipeline corrected to single-channel CRE with fusion reported as not adopted; degenerate-estimator rate corrected to 99.98%
- 3.7: same content, shorter sentences
- 4.1: three paragraphs simplified, all numbers and both caveats retained
- 4.2: opening paragraph simplified
- 4.3: the long stage paragraph broken into shorter sentences
- 4.5: the causal-control sentence split into three
- 5.1: simplified; cardiac k stated as 1.96 to match Table 3 and 4.2.1
- 5.3: three paragraphs simplified; degenerate-estimator rate corrected to 99.98%

## Canceller and channel sentence (2026-08-17)

- 3.2: states the OLS canceller as the method used; NLMS was never used for a reported result and is gone from the methods
- 3.2: 'the OLS differential CLE−CRE' corrected to the plain difference, and the canonical-channel claim replaced by what each analysis actually uses

## Respiratory reference: citation, gate evidence, corrected numbers (2026-08-17)

- 3.3: reference 29 (Konno & Mead) at the first RIPSum mention, as a superscript run; the stray plain-text 29 on page 11 made superscript too
- 3.3: gate rule corrected -- threshold is +0.10 against the median of the other three sensors, not 'net-negative'
- 3.3: validation paragraph simplified; r +0.47->+0.48, +0.27->+0.28, consensus SD 1.61->1.65, median |consensus-Flow| 0.06->0.08; Flow-alone coverage 97% added
- supplementary S2 / Table S2: per-session gate statistic for all four signals

## Formatting normalisation (2026-08-17)

- root cause fixed in apply_review_edits.Doc: para() cloned the title's bold run and set no spacing, so every paragraph and table cell any edit script created was bold and single-spaced
- body paragraphs: justified, double spaced, no first-line indent, no stray bold
- figure captions centred italic; table captions left bold
- table cells: header row bold, body rows plain, body spacing throughout
- run-in heads (Cardiac ground truth., etc.) bold on the lead phrase only

## Formatting normalisation (2026-08-17)

- root cause fixed in apply_review_edits.Doc: para() cloned the title's bold run and set no spacing, so every paragraph and table cell any edit script created was bold and single-spaced
- body paragraphs: justified, double spaced, no first-line indent, no stray bold
- figure captions centred italic; table captions left bold
- table cells: header row bold, body rows plain, body spacing throughout
- run-in heads (Cardiac ground truth., etc.) bold on the lead phrase only

## Section 4.1 reordered (2026-08-17)

- new order: SNR -> coherence and Table 2 -> rhythm continuity -> overnight record
- figures unchanged in content; SNR becomes Figure 2, the composite record figure becomes Figure 3
- 4.3 ridge stage-associations deliberately left after rates

## Formatting normalisation (2026-08-17)

- root cause fixed in apply_review_edits.Doc: para() cloned the title's bold run and set no spacing, so every paragraph and table cell any edit script created was bold and single-spaced
- body paragraphs: justified, double spaced, no first-line indent, no stray bold
- figure captions centred italic; table captions left bold
- table cells: header row bold, body rows plain, body spacing throughout
- run-in heads (Cardiac ground truth., etc.) bold on the lead phrase only

## Integrated capacitance imbalance (2026-08-17)

- 4.1: burden 15-705 fF*h across nights (46x), not a subject trait (2.0-5.8x within subject)
- the signed integral is reported as the null it is: asymmetry -0.22 to +0.09
- supplementary S3 / Figure S10

## 4.2 within-night negative, bounded (2026-08-17)

- subheading attributes the negative to the configurations tested, not to the mask
- best single night r = +0.32 resp / +0.53 card added, against a two-sensor ceiling of 0.22-0.71
- 5.2 'the battery was exhaustive' -> 'broad rather than exhaustive'

## Stale numbers found by the ledger check (2026-08-17)

- spindle low-band +0.47..+0.58 / ~+0.6 dB -> +0.45..+0.49 / +0.55 dB
- spindle EEG sigma +3.3 -> +3.45 dB (4.4, Figure 8, 5.2, limitation 6); the old value predates the 0.32 s alignment fix
- ridge Kruskal-Wallis floor 3x10^-23 -> 9x10^-29 (the old value was the second smallest p)

## Citations, orphan figures, and the S1 writeup (2026-08-17)

- superscript runs for references 22, 23,24 and 29
- Figures S2 and S5 cited from 5.4 and 4.2.1
- Figure S1 writeup no longer quotes the degenerate spectral estimator (0.95 br/min, raw_sd 0.000) as pipeline accuracy; channel spread 1.79-1.95 br/min and 3.41-4.39 BPM from the rerun instead
