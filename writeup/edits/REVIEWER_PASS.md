# Reviewer Pass — Argument-Level Critique of the Research Sections

_Deep, second-pass peer review of Methods (§3), Results (mislabeled §3.1–3.5), and
Discussion/Limitations/Conclusion in `writeup/edits/prof_extract.txt`. Builds on
`SUBMISSION_REVIEW.md` (which handled framing/mechanics); this pass is about whether each
claim is actually supported, whether the section order serves the intended narrative arc,
what a skeptical reviewer will demand, and how to make the arguments stronger with the data
already in hand. Line numbers refer to `prof_extract.txt`. Title/Abstract/Intro are out of
scope per instructions._

---

## Overall assessment

The underlying science is genuinely publishable and, unusually, the paper's strongest
asset is its rigor: it already carries a temporal-shuffle null (tracking), phase-randomized
surrogates (coherence), an independent-sensor ceiling (Flow–RIPSum r≈0.47), and two EEG
positive controls (SWA self-AUC 0.74; spindle AUC 0.98). That is a more complete control
suite than most wearable-sensor papers ever present. The problem is that the manuscript
**does not sell this as a framework** — it reports the controls as scattered footnotes to
individual results rather than as a unified "every claim is bounded by a null or a positive
control" methodology. Reframed, the negatives become the contribution: this is a
boundary-mapping study of a wearable, not a string of failures.

Three argument-level weaknesses run through the research sections and should be fixed before
the reordering/mechanical cleanups in `SUBMISSION_REVIEW.md`:

1. **The first pillar (signal validation) is presented with its weakest evidence forward.**
   The section leads on coherence (median 0.16–0.31) and surrogate pass rates (9–15%) — soft
   numbers — while the genuinely strong evidence (in-band SNR +4 to +27 dB across all 12
   nights, and the downstream MAE < 1 br/min) is buried or siloed. The arc wants validation
   to be the confident opening; right now it reads as the shakiest section.

2. **Pseudoreplication recurs.** Pooled-epoch p-values (Kruskal–Wallis `p < 10⁻¹⁶` on ~8,000
   epochs from 6 subjects; the "8,242 epochs" surrogate line; Wilcoxon over 12 sessions
   treated as independent when there are 6 subjects × 2 nights) overstate evidence and will
   be flagged reflexively by reviewers in this space. The paper already does the *right*
   thing for harmonics (per-subject direction counts) and k-vs-age (unit = subject) — it just
   needs to apply that unit-of-analysis discipline everywhere.

3. **The k≈2 mechanistic story is asserted, never shown.** The "biphasic pulse (systolic +
   dicrotic notch)" interpretation carries a lot of narrative weight (mean-rate accuracy,
   cardiac tracking failure, harmonic ladder, k-vs-age invariance) but there is no figure of
   the waveform. This is the single highest-leverage addition available from existing data.

Detailed findings follow, tagged **[STRENGTHEN] / [OVERCLAIM] / [RIGOR] / [FLOW] /
[CONSISTENCY]**.

---

## Stage 1 — Signal validation (§3.4 Methods L121–123; Results "3.1" L143–191)

### [OVERCLAIM] The surrogate pass rate does not support "confirming physiological coupling" (L147)
> "14.7% of respiratory epochs and 9.1% of cardiac epochs exceeded the surrogate null at
> p < 0.05, confirming that the coupling persists across all sleep stages…"

Under a true null, 5% of epochs exceed a p<0.05 threshold by construction. 9.1% for cardiac
is only ~1.8× chance; 14.7% resp is ~2.9×. Reported as written, a skeptic reads "≈90% of
cardiac epochs show no significant coherence" — which *undercuts* the pillar the whole paper
is built on. Two problems compound: (a) no test that 9.1%/14.7% is even significantly above
5% (with 8,242 epochs a binomial test will clear it, but the effect is small and must be
stated as such); (b) pooling epochs across 6 subjects again inflates the denominator.

**Fix (reframe, don't retract):** State plainly that magnitude-squared coherence *between a
mechanical capacitive signal and an electrical ECG / airflow reference is expected to be
modest* — the two modalities are causally linked but differ in waveform shape and are
separated by a variable pulse-transit / mechanical-coupling delay, both of which suppress
coherence at a single frequency even when the rate is perfectly recoverable. So coherence is
a **conservative lower bound** on coupling, not the primary evidence. Then lead the section
on the evidence that *is* strong (below), and forward-reference that the decisive proof of
physiological content is the downstream mean-rate accuracy (Stage 2). Report the surrogate
result per-subject (fraction of subjects whose median epoch coherence beats their own
surrogate null) rather than as a pooled epoch percentage.

### [STRENGTHEN] Lead with SNR and band-power fraction — they are your best numbers (L144, L146)
The band-power fractions (resp 29–48%, cardiac 8–48%) and in-band SNR (+11 to +27 dB resp;
+4 to +13 dB cardiac, *positive in all 12 nights*, L146) are strong, unambiguous, and
per-session. This is the confident opening the arc needs. Reorder the subsection: (1)
band-power + SNR across all 12 (the headline), (2) coherence/surrogates as conservative
corroboration with the modality-mismatch caveat, (3) an explicit pointer that rate accuracy
(Stage 2) closes the validation loop. Add SNR confidence intervals or per-session ranges so
the "all 12 positive" claim is auditable.

### [RIGOR] A negative/physiological control would make validation bulletproof
A reviewer will ask: how do you know the in-band energy is physiology and not a
band-limited instrumental resonance? You have the control already — **apnea epochs are
labeled** (L118). Show that respiratory in-band SNR / coherence *collapses during scored
central apneas* (no airflow → no respiratory displacement) and recovers afterward. That is a
within-subject, physiologically-specific positive control that no instrumental-artifact
explanation survives. Similarly, an electrode-off or high-motion segment should null the
cardiac band. This single analysis would convert the weakest section into one of the
strongest.

### [CONSISTENCY] "Canonical (bound)" is undefined and roughly 2× every real channel (L147, L185–191)
Table 2's "Canonical (bound)" row (resp 0.606, card 0.273) is never defined — same flag as
`SUBMISSION_REVIEW.md` #8, but note the argument-level danger: it is ~2× the actual channel
values, so if a reader assumes it is an achievable channel they will read the real coherence
(0.16–0.31) as a 2× shortfall. Define it explicitly at first use (best-channel-per-epoch
oracle? peak-frequency self-coherence?) and label it as an unachievable ceiling, or cut it.

### [CONSISTENCY] Epoch count varies: 9,319 vs 8,242 vs 9,317 (L32, L147, L377)
"9,319 one-minute analysis epochs" (L32) vs "8,242 epochs tested" for surrogates (L147) vs
"9,317 of 9,319 epochs" (L377); the ANALYSIS_LOG uses 9,318. Reconcile and, where the number
differs (surrogate subset after quality gating), say *why* the denominator dropped to 8,242.

### [CONSISTENCY] Flow–RIPSum ceiling reported as three different values (L119, L131, L380)
r = +0.48/+0.28 (L119) vs +0.47/+0.27 (L380) vs ANALYSIS_LOG +0.472/+0.266. Pick one pair
and use it in both the Methods validation of the consensus GT and the tracking-ceiling
result; this number is load-bearing for the tracking argument, so it must be identical
everywhere.

---

## Stage 2 — Rate accuracy + k-calibration (§3.5 Methods L124–126; Results "3.2" L192–349)

### [STRENGTHEN] The headline result is undersold and comparative context is missing (L194, L196)
Median resp MAE 0.91 br/min with k≈0.97 (essentially calibration-free) and cardiac MAE ~3.4
BPM after a stable ~2:1 scaling are competitive with non-contact/radar/BCG cardiorespiratory
sensors — but the manuscript states this flatly and provides no benchmark. Add 1–2 sentences
citing representative non-contact resp (~1 br/min) and BCG/ballistocardiographic cardiac
error ranges so the reader can judge that this is *good*, not merely "reported." This is the
paper's main positive; it deserves an assertive topic sentence.

### [STRENGTHEN — highest leverage] Show the biphasic cardiac waveform (supports the entire k story) (L196, L503)
The k≈2 "systolic + dicrotic" claim (L196) and the cardiac-tracking mechanistic explanation
(L503, "two inflection points per heartbeat") are asserted without ever showing the
waveform. You have ECG R-peaks (L117). Produce an **R-peak-triggered ensemble average of the
CAP cardiac-band signal** — if it shows two peaks per beat, that is direct, visual proof of
k≈2 and simultaneously explains (a) the mean-rate overcount, (b) the harmonic ladder (Stage
3), and (c) why per-window dominant frequency is morphology-locked rather than rate-locked
(the tracking failure). One figure retro-validates three separate sections. This is the most
important missing analysis in the paper and it is fully doable with existing data.

### [RIGOR] The per-session cardiac r column in Table 3 is unexplained and alarming (L272–344)
Table 3 (cardiac) carries an r column that is near-zero or negative for almost every session
(e.g., S1N1 −0.203, S1N2 −0.434). Presented in the *mean-accuracy* table with no caption
context, it reads as "the method anti-correlates with truth." It is actually the
within-session tracking result arriving early. Either remove r from Table 3 and keep it in
the tracking section (Stage 3), or add one sentence: "the near-zero/negative within-session r
is the tracking limitation quantified in §X; mean-rate accuracy and within-session tracking
are distinct axes." Do not leave it unexplained.

### [OVERCLAIM / RIGOR] Median MAE hides catastrophic per-session cardiac failures (L196, L272–344)
Median cardiac MAE 3.41 is quoted as the result, but Table 3 contains S2N1 = 11.67, S6N1 =
17.96, S6N2 = 8.57 BPM — i.e., 3/12 sessions are 2.5–5× the median. The Bland–Altman LoA
[−24, +23] BPM (L196) reflect this. A reviewer will not accept "3.4 BPM" as the device
accuracy when a quarter of nights are >8 BPM. **Fix:** report the full per-session
distribution up front (median *and* range/IQR *and* n sessions >5 BPM), state that the mean
per-session value is the validated quantity while the LoA reflect epoch-level spread
(SUBMISSION_REVIEW P2#11), and either (a) attribute S6 to the coupling regime you already
flag (k=1.35/0.94, L196) with a prospective quality gate, or (b) report accuracy after a
declared coupling-quality inclusion criterion. Silent reliance on the median is the kind of
thing that gets a paper bounced.

### [CONSISTENCY] Which pipeline is Table 3 cardiac — single-channel CRE or fusion? (L196, L266)
Text says single best channel CRE median 3.41 *and* multi-channel fusion pooled 3.91 (L196);
Table 3 is headed "peaks_loose, k-calibrated" but its median (~3.36) matches neither label
cleanly. State explicitly which pipeline each number/table represents; right now the reader
cannot map Table 3 to a described method.

### [STRENGTHEN] The k-vs-age result strengthens the mechanism — integrate it, don't bolt it on (draft `k_vs_age_section_draft.md`)
The new result (cardiac k age-invariant ρ=+0.37 p=0.47; resp k declining ρ=−0.83 uncorrected
p=0.042) fits the mechanism cleanly: **cardiac k is fixed because pulse morphology is fixed
(age-invariant) — this is a genuine confirmatory prediction of the biphasic-waveform model,
not a side finding.** State it that way: "the fixed-morphology interpretation predicts
age-invariant cardiac k, which we confirm." That converts a descriptive correlation into a
mechanistic test the model passed. Keep the respiratory decline explicitly exploratory (the
draft does this correctly: 1/4 tests, Bonferroni p≈0.17, narrow dynamic range, n=6) — do not
let it drift into a claim. Also ensure the deprecated *within-session* "k(t) biomarker"
framing (killed in PAPER_FINDINGS: corr(k, rate) = −0.83, GT-free proxy −0.06) is nowhere in
the manuscript; the surviving story is "k is a stable per-*subject* morphological constant,"
which is coherent and defensible.

### [RIGOR] k-calibration uses PSG-derived truth — state the deployment consequence quantitatively (L126, L515)
Limitation 3 (L515) acknowledges k needs a reference for initial calibration. Strengthen by
quantifying the population-prior fallback: since cardiac k is age-invariant and clusters at
~1.95 (5/6 subjects), report the MAE you would get using a *fixed population k = 1.95* with
no per-subject calibration. If it is only modestly worse, that is a strong practical claim
(near-calibration-free deployment); if it is much worse, that honestly bounds the
requirement. Either way it answers the obvious "so does it need calibration or not?"
question the reviewer will ask.

### [FLOW] Per-stage MAE (L347–349) is a two-sentence orphan
"Resp worst in REM, cardiac worst in Wake" is plausible but unsupported by any statistic and
sits with no connective tissue. Either give it a per-subject consistency count (as you do for
harmonics) or fold it into the Bland–Altman/limitations discussion. As written it invites
"n=6, is this significant?" with no defense.

---

## Stage 3a — Harmonic structure + LOSO N3 (§3.7 Methods L132–135; Results "3.4" L383–427)

### [RIGOR — will be flagged hard] Kruskal–Wallis `p < 10⁻¹⁶` on pooled epochs is pseudoreplication (L387)
> "Ridge features showed statistically significant variation across sleep stages
> (Kruskal–Wallis p < 10⁻¹⁶ for all four features)."

This pools thousands of epochs from 6 subjects as if independent. In a sleep/wearable venue
this is the reflexive-rejection statistic. The honest evidence is already in the next
sentence — "consistent in 5–6 of 6 subjects." **Fix:** demote the pooled p entirely (or move
to a footnote clearly labeled "epoch-level, not corrected for subject clustering"), and lead
with (a) per-subject effect directions with a sign test across 6 subjects, and (b) a linear
mixed model with subject as random effect for the inferential claim. You already committed to
exactly this rule in `analysis/swa_validation/CLAUDE.md` ("Per-subject statistics; never
pooled raw epochs; subject as random effect") — apply it here. Report an effect size
(Cliff's delta / rank-biserial) per feature, not just significance.

### [STRENGTHEN] Tie the harmonic ladder to the pulse morphology — it is the same phenomenon (L133, L384, L503)
The harmonic ladder is currently presented as a mysterious "structured integer-ratio
spectral peak" structure. Mechanistically it is almost certainly the **frequency-domain
signature of the non-sinusoidal (biphasic) cardiac/respiratory pulse** — the same waveform
that produces k≈2. Connecting them (a) demystifies the harmonics, (b) reinforces the k story,
and (c) is testable with existing data: show the harmonic ladder is *locked to the
cardiac/respiratory fundamental* (harmonic frequencies scale with rate epoch-to-epoch). That
test also doubles as a control proving the ladder is physiological, not an instrumental comb.

### [CONSISTENCY] CRE-dominant vs CH-strongest is confusing as stated (L384)
> "concentrated in the CRE channel (dominant ridge channel in 9/12 sessions) with harmonic
> detection strongest in the CH channel (70% of windows)"

Two different "which channel wins" claims in one sentence read as a contradiction. Define the
two distinct quantities (ridge *dominance* vs harmonic *detection prevalence*) or the reader
concludes the analysis is inconsistent. The ANALYSIS_LOG distinguishes them; the manuscript
must too.

### [STRENGTHEN] Frame the AUC=0.534 as a *mechanistically explained* negative, not a shrug (L390)
The subject-dependent HER direction (S1/S2 up in N3, S3/S4 down, L387/L390) is a real,
confirmed effect that *explains why pooled discrimination cancels*. This is a much more
interesting finding than "weak classifier." State it as: harmonic structure encodes sleep
state, but the encoding is subject-specific in sign, so it cannot support a *universal*
classifier — a substantive claim about individual coupling geometry (which you develop well
at L505). Consider whether a *per-subject-normalized* or *within-subject* classifier recovers
signal; if it does, that is a positive result worth reporting ("stageable within-subject, not
across-subject") and materially strengthens the section.

---

## Stage 3b — Cortical EEG negatives: SWA + spindles (§3.8 Methods L136–138; Results "3.5" L428–483; spindle draft)

### [STRENGTHEN] Group SWA + spindle as one "not cortical EEG" result — and use the spindle control to rescue the SWA montage limitation
Follow the spindle draft's recommendation to merge these into one subsection (delta + sigma).
The argument-level payoff is specific: the SWA EEG **self-AUC is only 0.74** (L430), which a
skeptic can attack ("your own EEG N3 detector is mediocre, so a null CAP result proves
nothing about a *good* EEG signal"). The **spindle EEG positive control at AUC 0.98** (draft
L49) is the answer — it certifies the contact-EEG channel is genuine, well-placed, and
faithfully transduced. Make this explicit: *because the same electrode yields AUC 0.98 for
spindles, the near-zero CAP result is a true absence, not a weak-reference artifact.* This
also directly resolves Limitation 6 (unknown EEG montage, L518) — cross-reference it there.

### [OVERCLAIM] Explain the modest SWA self-AUC (0.74) before a reviewer weaponizes it (L430)
Don't leave 0.74 unexplained next to a "definitive" negative (L501). State that 0.74 reflects
a *single-channel power-threshold* classifier (not full multi-feature AASM staging) on an
unknown-montage derivation, and that the spindle control (0.98) is the stronger certification
of the EEG channel. Otherwise the asymmetry (0.74 vs 0.98) looks unexamined.

### [RIGOR] The "Sensitivity" column in Table 6 is misleading at chance AUC (L441–483)
CAP N3 AUC ≈ 0.44–0.55 with a "Sensitivity" of 0.85–0.99. High sensitivity at chance AUC
just means the threshold labels almost everything N3 — it is not evidence of anything.
Either drop the sensitivity column or pair it with specificity (which will be near zero) and
state that at chance AUC these operating points are meaningless. As printed, a fast reader
might mistake "sensitivity 0.90" for a capability.

### [STRENGTHEN] The 12-session visual diagnostic is good corroboration — quantify the "5 slow-modulation" sessions (L431)
The claim that 7/12 sessions are motion-spike noise and 5/12 show respiratory/mechanical (not
cortical) slow modulation (L431) is a strong qualitative argument. Strengthen it by reporting
the correlation of those 5 CAP slow-modulation traces with *respiratory* band power (should
be high) vs EEG delta (near zero) — that turns "in our judgment it's respiratory" into a
number, closing the loop on the mechanism.

### [RIGOR] Report SWA/spindle stats per subject, not pooled ± SD (L429–430)
"r = 0.015 ± 0.045" and "AUC 0.490 ± 0.040" pool across sessions. With 6 subjects × 2 nights,
report per-subject means (Table 6 already does for SWA — good) and treat subject as the unit
for any inferential statement, consistent with the SWA workspace rule.

---

## Discussion / Limitations / Conclusion (§4.1–4.5 L484–511; §5 L512–518; §6 L519–521)

### [STRENGTHEN] Open the Discussion by naming the methodological contribution
The Discussion (L485) says "honest characterization" but never claims the *method* as a
contribution. State it: this paper demonstrates a **null-bounded, positive-controlled
evaluation framework for wearable physiological claims** — temporal-shuffle null,
phase-randomized surrogates, an independent-sensor ceiling, and EEG positive controls. That
is a transferable contribution beyond this device and it reframes five negatives as a rigor
showcase.

### [OVERCLAIM] "definitive" for the SWA negative (L501) — soften by one degree
"This result… is definitive" is strong for n=6 with an unknown montage. "Definitive within
this cohort and montage, and corroborated by the spindle positive control" is defensible and
still confident. (The spindle control genuinely does most of the work here — lean on it
rather than on the adjective.)

### [STRENGTHEN] The cardiac-tracking mechanism (L503) is your best mechanistic passage — but it needs the waveform figure
The morphology-locked-frequency argument (a fixed biphasic waveform whose dominant frequency
is governed by shape, not instantaneous rate) is genuinely insightful and unifies the
mean-accuracy, tracking, and harmonic results. It is currently pure prose. The R-peak-
triggered average (Stage 2 recommendation) is the evidence that makes it a demonstrated
mechanism rather than a plausible story. Add the figure and cite it here.

### [RIGOR] State the unit-of-analysis and repeated-measures structure once, in §3.9 (L139–140)
Methods §3.9 lists tests but never states that there are 6 subjects × 2 nights and that night
is a repeated measure. The Wilcoxon over 12 sessions (L130/L354) implicitly treats nights as
independent. Add one sentence declaring subject as the unit for inferential claims (mixed
models / per-subject-then-test), and note where 12-session tests are descriptive only. This
one paragraph inoculates against most of the pseudoreplication objections at once.

### [CONSISTENCY] S3 respiratory outlier attribution is internally muddled (L118, L194)
Methods say S3 Thorax was *dropped* from the consensus GT for paradoxical breathing (L118),
yet Results attribute S3's high resp MAE to "a paradoxical thoracic effort signal that
degraded the consensus ground truth" (L194). If Thorax was already excluded, the consensus
should not still be degraded by it — clarify whether S3's error is (a) residual GT
uncertainty from losing a sensor, or (b) a genuine mask limitation. As written it reads as
double-counting the same problem.

### [STRENGTHEN] Limitation 4 (non-independent GT) is well handled — connect it to the ceiling
Limitation 4 (L516, consensus GT from the same PSG) pairs naturally with the Flow–RIPSum
ceiling: the ceiling *is* your evidence that the GT limitation does not manufacture the
tracking negative (two physically independent PSG sensors also cap out at r≈0.47). Cross-
reference so the limitation reads as "already bounded," not "open risk."

---

## Prioritized top 8 changes that most increase acceptance odds

1. **[STRENGTHEN] Add the R-peak-triggered CAP cardiac-waveform figure.** One figure from
   existing data (ECG R-peaks already computed) that shows the biphasic pulse — it converts
   the k≈2 claim, the cardiac-tracking mechanism, the harmonic ladder, and the age-invariance
   prediction from assertions into a demonstrated, unified mechanism. Highest leverage in the
   paper (L196, L503, L133).

2. **[RIGOR] Kill pseudoreplication paper-wide.** Demote the Kruskal–Wallis `p < 10⁻¹⁶`
   (L387) and the "8,242 epochs" surrogate framing (L147); replace with per-subject
   directions + sign test + mixed model, and declare subject-as-unit once in §3.9. This is the
   objection most likely to trigger rejection, and the fix is mostly reframing existing
   analyses.

3. **[STRENGTHEN] Reorder and reframe signal validation (Stage 1).** Lead with band-power +
   SNR (all 12 positive), demote coherence to conservative corroboration with the mechanical-
   vs-electrical modality-mismatch caveat, and forward-reference mean-rate accuracy as the
   decisive validation. Fixes the "weakest-evidence-first" problem in the opening pillar
   (L143–147).

4. **[STRENGTHEN] Merge SWA + spindle into one "not cortical EEG" result and use the spindle
   AUC 0.98 to certify the EEG channel.** This resolves the montage limitation (L518) and
   turns two negatives into one clean, positively-controlled boundary (draft + L430).

5. **[RIGOR] Fix the cardiac-accuracy honesty gap.** Report the full per-session distribution
   (3/12 nights >8 BPM), state that mean-per-session is the validated quantity and the LoA
   [−24,+23] reflect epoch spread, and declare a coupling-quality inclusion criterion for the
   S6-type outliers (L196, L272–344). Prevents "3.4 BPM is not credible" pushback.

6. **[STRENGTHEN] Add a physiological control to signal validation: coherence/SNR collapses
   during scored apnea.** A within-subject, artifact-proof positive control using data you
   already have (apnea labels, L118) that makes the foundational pillar unassailable.

7. **[STRENGTHEN] Integrate k-vs-age as a passed mechanistic prediction, not a side result.**
   "The fixed-morphology model predicts age-invariant cardiac k, which we confirm (ρ=+0.37,
   p=0.47)"; keep respiratory k strictly exploratory; ensure the deprecated within-session
   k(t)-biomarker framing is fully absent (draft, PAPER_FINDINGS).

8. **[CONSISTENCY] Reconcile the load-bearing numbers.** Epoch count (9,319 / 8,242 / 9,317,
   L32/L147/L377), Flow–RIPSum ceiling (0.47 vs 0.48, L119/L380), define "Canonical (bound)"
   (L147/L185), and clarify Table 3's pipeline and the S3 outlier attribution (L118/L194).
   Cheap to fix, and inconsistencies of this kind erode reviewer trust disproportionately.
