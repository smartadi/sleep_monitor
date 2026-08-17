# Strict-reviewer pass, §3.2 onwards — 2026-08-17

Written against `writeup/main/CAP_sleep_mask_manuscript_main_WORKING.docx` as it
stands after today's edits. Each item is what a hostile but fair reviewer would
write, the evidence behind it, and the action proposed. Nothing here is applied
yet.

Severity: **B** blocking (a reviewer would reject or demand major revision),
**M** major (will be asked, needs a real answer), **P** polish.

---

## B1. Table S1 is a different pipeline from Table 3, and its respiratory column is the degenerate estimator

The caption says "from the same operational pipeline as Table 3". It is not.

| | Table S1 median | Table 3 |
|---|---|---|
| Resp k | 0.99 | 1.18 |
| Resp night err | 0.14 | 0.24 |
| **Resp epoch AE** | **0.94** | **1.79** |
| Resp within-night r | +0.09 | −0.03 |
| Card k | 1.95 | 1.96 |
| Card night err | 1.60 | 1.56 |
| Card epoch AE | 3.36 | 3.41 |
| Card within-night r | −0.19 | −0.08 |

Table S1 is the superseded phase-C `fused_agree` run. Its respiratory epoch error
of 0.94 br/min is the **Welch spectral estimator** — the constant 15 br/min that
§3.5 explicitly tells the reader not to treat as a measurement. The supplement
therefore reports, as per-session detail, numbers that the Methods disown, and
they differ from the main table by a factor of two in the respiratory band.

A reviewer who checks one number against the other finds this immediately, and it
casts doubt on every other number in the paper.

**Action:** rebuild Table S1 from `reports/rates/rerun/per_session.csv`, the same
source as Table 3, and re-verify that its medians reproduce Table 3 exactly. Add
that reproduction as an automated check in `check_manuscript.py`.

## B2. §4.2 never reports what happens without same-night calibration

Every headline number is produced with k fitted on the night it is scored on.
§3.5 promises the reader that "section 4.2 separates what k supplies from what
the sensor supplies". §4.2 does not do this. The comparison exists in
`reports/rates/rerun/heldout_table.csv`:

| respiratory | night err | epoch AE |
|---|---|---|
| same-night k (reported) | 0.24 | 1.79 |
| subject's other night | 0.57 | 1.90 |
| population k | 0.94 | 2.08 |
| **no sensor, cohort constant** | **1.20** | **1.29** |

| cardiac | night err | epoch AE |
|---|---|---|
| same-night k (reported) | 1.56 | 3.41 |
| subject's other night | 3.77 | 4.64 |
| population k | 3.19 | 5.65 |
| **no sensor, cohort constant** | **2.76** | **4.42** |

The cardiac population-k night error (3.19) is **worse than predicting a constant
61 BPM** (2.76), and the respiratory epoch error under any calibration is worse
than the no-sensor constant (1.29). A reviewer will find this in the supplement
(Figure S4 already alludes to it) and ask why it is not in the Results.

**Action:** add a short subsection or a Table 3 block giving the four
calibrations for both bands, and state plainly that the mask's value at present
is *per-session relative* accuracy, not standalone measurement. This is the
single most likely cause of rejection if left as is, and stating it costs the
paper less than being caught.

## B3. Apnea epochs are in every error statistic, unremarked

`analysis/rates/rerun_rate_detection.py` contains no apnea handling: all valid
epochs enter the error computation. Apnea/hypopnea fraction by night:

S2N2 36.2%, S2N1 25.1%, S4N1 20.7%, S3N1 16.3%, S1N1 12.8%, S1N2 12.1%,
S3N2 12.7%, S5N2 3.8%, S4N2 2.8%, S6N2 2.6%, S5N1 1.5%, S6N1 0.9%.

§3.3 says "apnea epochs were labelled ... rather than forced to a rate", which
describes the *reference construction*, and a reader will reasonably assume they
were then excluded from scoring. They were not. In S2N2 more than a third of the
scored epochs are periods when breathing is disordered and a single respiratory
rate is not well defined.

**Action:** state explicitly that apnea epochs are included, and add a
sensitivity analysis excluding them (a few lines on the existing parquet). If
the numbers move, report both.

---

## M1. The two within-night statements in §4.2 do not sit together

¶93 still says "No estimator, channel or fusion strategy we tested produced a
within-night correlation distinguishable from a circular-shift null", and ¶95
then says the best night reaches r = +0.32 / +0.53. Both are true — one is about
the distribution over nights, the other about single nights — but placed two
paragraphs apart they read as a contradiction.

**Action:** merge into one statement: no *configuration* separates from the null
across nights, individual nights scatter to r ≈ 0.5, and that scatter is what
n = 12 sampling noise looks like.

## M2. The two-sensor ceiling is quoted three different ways

- §3.3: within-session correlation "averages +0.48 on raw rates and +0.28 on detrended"
- §4.2: "r = 0.22 to 0.71"
- Figure S3 writeup: "respiratory sensors agree with each other at r = 0.47 (0.27 on detrended)"

The last is stale (pre-recompute). Three renderings of one quantity in one paper.

**Action:** pick one form — the mean with its range — and use it in all three
places. Add it to `key_numbers.py` so it cannot drift again.

## M3. The Viterbi tracker cannot report a rate outside its search band

§3.4 gives respiration 0.25–0.55 Hz and cardiac 0.85–1.45 Hz (≈51–87 BPM). §4.1
then reports that the traces "hold near 0.25–0.3 Hz and 0.9–1.2 Hz for the full
recording" and never hop to a subharmonic. Partly by construction: the tracker
cannot leave the band, and a genuine heart rate above 87 BPM would be clipped
rather than tracked.

**Action:** state the search bands where the continuity claim is made, and say
what fraction of reference epochs fall outside them (computable from the GT).

## M4. S6N2 uses photoplethysmography as the cardiac reference, and S6N2 is the outlier

§3.3 notes the ECG was dead for S6N2 and Pleth was substituted. S6N2 is then the
session that produces the cardiac k outlier (0.97), the full k range 0.97–2.28,
the largest imbalance burden, and the worst cardiac epoch error. Pulse rate from
Pleth is not beat rate from ECG — it is delayed and smoothed.

**Action:** report S6N2 with and without the substitution, or exclude it from the
k range and say why. At minimum, note the reference change wherever S6N2 is named
as an anomaly, so the anomaly is not attributed to the mask by default.

## M5. The delta-burst response has no arousal control

§4.5 shows a capacitive response peaking 4.5–8.2 s after cortical onset, on
motion-free baselines. A cardiovascular response at that latency is exactly what
a micro-arousal produces, and K-complexes — which §4.5 says dominate the onset
set — are frequently associated with autonomic activation.

**Action:** repeat the peri-onset average excluding onsets within ±10 s of a
scored arousal (the scoring exists; `analysis/swa_validation/cap_arousal_events.py`
already reads it). If the response survives, the section is much stronger. If it
does not, the finding is an arousal response and must be described as one.

## M6. Ridge stage effects are not controlled for motion or epoch count

§4.3 reports fewer respiratory ridges in N3 in 6/6 subjects. N3 epochs are also
the quietest and least motion-contaminated of the night, and there are fewer of
them. Neither confound is addressed.

**Action:** repeat the stage contrast on motion-matched, count-matched epoch
subsets. This is the obvious first question and the analysis is cheap.

## M7. Post-hoc channel selection is disclosed but not tested

CRE was chosen because it carried the dominant ridge in 9/12 sessions;
Limitation 7 says so. A reviewer will still ask whether the stage effects survive
on a pre-specified channel.

**Action:** rerun §4.3 on CLE−CRE and report the direction counts in the
supplement. Disclosure plus a robustness check is a much stronger position than
disclosure alone.

## M8. Bland–Altman limits pool epochs across recordings

The bias and 95% limits in Table 3 and Figure S6 are computed over all epochs of
all twelve recordings after per-session calibration. Epochs within a night are
not independent, so the limits are not a valid population statistic.

**Action:** report per-session limits summarised across nights, or state that the
pooled limits are descriptive.

## M9. "Suitable for screening" is inconsistent with the calibration requirement

§5.4 says the mask is suitable for screening-level monitoring; §6 Limitation 3
says every accuracy figure is conditional on calibration against a reference, and
B2 above shows an uncalibrated device is at or below a no-sensor constant.

**Action:** qualify §5.4 and §7 to "screening-level monitoring **once
calibrated**, with calibration transfer unsolved", or drop the screening claim.

## M10. Five abbreviations are used without ever being expanded

Caught by the new acronym audit in `check_manuscript.py`, after the user spotted
STFT by eye:

| | uses | first use |
|---|---|---|
| **STFT** | 3 | §3.5, "CWT ridge tracking and STFT peak tracking" |
| **AASM** | 1 | §3.3, "the PSG technologist's AASM scoring" |
| **IQR** | 3 | Table 2 header, before Table 3's caption defines the idea |
| **SD** | 1 | Table 3 row, "Reference rate SD within night" |
| **DC** | 1 | Figure S5 writeup, "band SNR, DC drift" |

STFT is the one a reviewer will circle: it appears three times as a named method
alongside CWT, which *is* expanded in the same sentence.

**Action:** expand at first use — short-time Fourier transform (STFT), American
Academy of Sleep Medicine (AASM), interquartile range (IQR), spell out standard
deviation in the Table 3 row, and either expand or reword DC drift. The audit now
runs on every check so this cannot recur.

## M11. Bonferroni correction is claimed but not visible

§3.7 says "Bonferroni correction is applied across the family of tests reported".
No reported p-value is identifiably corrected, and the family is not defined.

**Action:** define the family and mark corrected p-values, or remove the sentence.

---

## P1. Motion canceller is never validated in the paper
`analysis/rates/motion_cancel_validation.py` exists and tests exactly the claim
§3.2 makes ("only motion energy within that band was removed"). Its result is not
reported. One sentence, or a supplementary panel.

## P2. Imbalance burden has no interpretation
§4.1 now reports a 46× spread that is not a subject trait. A reviewer will ask
what it tracks. Test it against motion fraction and band SNR — if it tracks
coupling quality, say so; if nothing, say that too.

## P3. Cardiac per-epoch IQR is driven by one subject
Table 3 gives 3.41 [3.06–8.38] BPM; the upper quartile is 2.5× the median because
of S6. Report the IQR with S6 excluded alongside.

## P4. Peaks-per-beat uses 9 of 12 sessions
§4.2.1 says "the nine sessions with usable peak detection" without saying which
or why. Name them and the exclusion rule.

## P5. Pooled p-values printed on Figure 7 panels
§4.3 says they are descriptive only. If they cannot be interpreted, consider not
printing them on the figure.

## P6. "k behaves as a stable property of the subject" overstates the cardiac case
Cardiac |Δk| between a subject's two nights has a median of 0.146 and a maximum of
0.387 — roughly 20% of k, which is 3.77 BPM of night-level error when k is taken
from the other night. Say "reproducible within about 20% in the cardiac band".

## P7. Delta-onset counts not given per night
"a few dozen to roughly one hundred" — give the actual counts in the supplement.

## P8. Limitations omit the reference floor
§3.3 establishes that two simultaneous PSG sensors disagree by more than 1 br/min
on 29% of epochs. That is a floor on any measured accuracy and belongs in §6.

---

## Suggested order of work

1. **B1** rebuild Table S1 (mechanical, removes the worst inconsistency)
2. **B2** add the held-out calibration comparison to §4.2 (writing, numbers exist)
3. **B3** apnea sensitivity analysis (small analysis + one paragraph)
4. **M5, M6, M7** the three robustness reruns (arousal control, motion/count
   matching, pre-specified channel) — these are what turn "disclosed" into
   "tested"
5. **M1, M2, M4, M8, M9, M10, M11** wording and consistency (M10 acronyms is a five-minute fix)
6. **P1–P8** polish
