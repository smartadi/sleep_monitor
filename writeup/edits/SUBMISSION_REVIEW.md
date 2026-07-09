# CAP Sleep-Mask Manuscript — Submission-Readiness Review
_2026-07-09. Based on the current main draft (`writeup/main/CAP_sleep_mask_manuscript_main.docx`,
text extract in `writeup/edits/prof_extract.txt`)._

## Verdict
The science is solid and the honest-characterization angle is a genuine strength. The
main risk to acceptance is **not the data — it's a mismatch between the aspirational
Introduction/Title and the largely-negative Results**, plus a set of mechanical cleanups
(numbering, duplicated text, leftover author notes). Fix the framing and the mechanics
and this is a credible submission.

---

## P0 — Blocking (a reviewer will bounce on these)

### 1. Intro/Title promise something the Results disprove
The Introduction coins **"intracranial slow-wave activity (ISWA)"** and states the study
will show "SEC signals contain structured spectral signatures that distinguish deep
sleep." But §3.5 is an explicit **negative** SWA result (r≈0.015, coherence≈0.003, N3
AUC≈0.49), and the harmonic N3 classifier is near-chance (AUC 0.534). The paper's own
Discussion calls itself "a systematic, honest characterization of what a SEC-sensor sleep
mask can and cannot measure." The front matter must be rewritten to match.
- **Fix:** Drop the ISWA coinage (or explicitly frame it as a *hypothesis the study
  tests and rejects*). Recast the thesis: a wearable capacitive mask that (a) robustly
  recovers mean cardiorespiratory rates via intracranial/hemodynamic *mechanical*
  pulsation, and (b) has clearly delineated boundaries — it does **not** capture cortical
  electrical hallmarks (slow waves, spindles) or within-session rate dynamics.
- Keep the ICP-pulsation motivation (it is legitimate and pig-validated for *mechanical*
  ICP) but sever the unsupported leap from ICP-mechanical → cortical-SWS-electrical.
- **Title:** "Noninvasive Monitoring of Sleep-Related *Intracranial Physiological
  Dynamics*..." over-promises. Consider: _"A Wearable Capacitive Sensor Mask for Overnight
  Cardiorespiratory Monitoring: Capabilities and Boundaries against Polysomnography."_

### 2. Abstract is unwritten
`[TO BE WRITTEN — lead with honest characterization framing]`. Needs a full abstract. It
should lead with the positive (rate accuracy), state the boundaries plainly, and name the
sample (6 subjects / 12 nights). Draft below (§Appendix).

### 3. Section numbering is broken throughout
Methods is §3; Results is labeled "4. Results" but its subsections are numbered
**3.1–3.5** (and 3.1 appears twice). Discussion is "4. Discussion" (collides with
Results), then §4.1–4.5, then "5. Limitations", "6. Conclusion". Renumber cleanly:
1 Intro · 2 Sensing · 3 Methods · 4 Results · 5 Discussion · 6 Limitations · 7 Conclusion.

### 4. Leftover author/outline notes are still in the body
- Results opens with: _"Start with mean value (raw signal changes) in comparison to sleep
  stages."_ — delete.
- §4.1 heading stub: _"Accurate mean respiratory and cardiac rates (comparison between
  K-factor and ages)"_ — the parenthetical is an outline note; delete or turn into real
  analysis (see P2 #12).
- The entire **"OPEN ITEMS / REVIEW NOTES"** block (verification notes, "Claims revised
  from the scaffold", figure cross-reference) must be removed before submission — it's
  internal scaffolding.

### 5. Duplicated text
- **Author list** is printed twice (identical lines).
- **§3.1 Overnight testing** describes the study **twice** (recruitment, IRB, exclusion
  criteria, 111 Hz, two-night protocol appear in two near-identical passes). Collapse to
  one clean pass. This duplication is the most visible copy-paste artifact in the draft.

---

## P1 — Important (strengthens acceptance odds)

### 6. Add the spindle result — it converts a weakness into a strength
Two negatives (SWA, tracking) read as "the thing doesn't work." **Three** negatives, one
of which is a *clean, positively-controlled* spindle result (EEG AUC 0.98 vs CAP 0.50),
reframe the paper as a *rigorous boundary-mapping study*. Group SWA + spindles under one
"not cortical EEG" theme (delta band + sigma band). Ready-to-paste text and table:
`writeup/edits/spindle_section_draft.md`; figures in `analysis/spindles/outputs/`.

### 7. Pseudoreplication in the statistics
Several claims pool raw epochs (e.g. "8,242 epochs tested", Kruskal–Wallis "p < 10⁻¹⁶").
With 6 subjects, pooled-epoch p-values massively overstate evidence — reviewers in this
space reliably flag this. The SWA workspace's own rule ("subject as random effect; never
pool raw epochs") should apply paper-wide.
- **Fix:** Report per-subject statistics and effect directions (you already do this well
  for harmonics). For inferential claims use subject as the unit (linear mixed model or
  per-subject then Wilcoxon across 6). Demote the 10⁻¹⁶ p-values — they undercut
  credibility more than they help.

### 8. "Canonical (bound)" is undefined
Table 2 and the coherence text reference a "canonical upper bound" (resp 0.606, card
0.273) with no definition of what "canonical" means or how the bound was derived. Define
it at first use or remove it.

### 9. Terminology sprawl
The device/signal is variously "SEC", "CAP", "capacitive", "r-ICP", "SEC-derived",
"ISWA". Pick **one** primary term (suggest "capacitive (SEC) mask" defined once, then
"CAP signal"/"CAP channel" consistently). Define CLE, CRE, CH, CLE−CRE once with a small
schematic and never re-explain.

### 10. Figure program needs consolidation and consistent styling
14 embedded figures, many dense multi-panel. For submission:
- Add **panel letters (A/B/C)** and larger fonts (many current panels are unreadable at
  column width).
- Ensure every CAP-analysis figure includes a **spectrogram panel** for frequency-domain
  context (matches your established convention).
- The two negative-result pairs (SWA, spindles) each want a two-panel "EEG positive
  control vs CAP" layout — the spindle `fig_spindle_triggered_sigma.png` is the template.
- Consider a **Figure 1 = graphical summary** (what the mask can/can't do) to set honest
  expectations up front.
- Trim to the ~8–10 figures that carry the argument; move the rest to Supplement (you
  already list many "available but not embedded").

---

## P2 — Polish

11. **Bland–Altman LoA are very wide** (cardiac [−24, +23] BPM). State plainly that these
    reflect epoch-level spread and that the *per-session mean* is the validated quantity;
    otherwise a reader takes the LoA as the usable accuracy.
12. **"K-factor vs ages"** is teased in a heading but never analyzed. Either add the
    actual k-vs-age / k-vs-PSQI analysis (n=6, descriptive only) or drop the tease.
13. **Two operating points** (spectral = low MAE/zero tracking; FWD = higher MAE/no
    tracking) is a subtle but important point — give it a clear topic sentence so
    reviewers don't read "0.91 br/min MAE" as within-night tracking accuracy.
14. **Reproducibility:** state software (neurokit2 version, scipy), filter orders, window
    params in one Methods table. Add a data/code availability statement.
15. **Limitation on EEG montage** (§ point 6) is honest but buried — since two results now
    hinge on the EEG comparison (SWA + spindles), state the montage if recoverable; the
    spindle positive control (AUC 0.98) actually *reassures* that the EEG channel is
    genuine and well-placed — cite that as evidence the negative is real.
16. Define **HER** (harmonic energy ratio) and **FWD** (Fused Window Detection) at first
    use; both appear before definition in places.

---

## Appendix — draft Abstract (honest framing, ~200 words)

> Wearable sensors that monitor sleep without electrodes could extend sleep assessment
> beyond the laboratory, but their physiological scope is often overstated. We evaluated a
> capacitive (single-electrode) sensor mask that measures intracranial and hemodynamic
> pulsation at the temples, in six adults across twelve overnight recordings with
> simultaneous polysomnography. Using multiple estimators, five channels, and rigorous
> null tests, we mapped what the mask does and does not measure. The mask recovered
> per-night mean respiratory rate (median absolute error < 1 breath/min; calibration
> factor ≈ 1) and mean cardiac rate (< 4 beats/min after a stable ~2:1 calibration), with
> accuracy comparable to non-contact cardiorespiratory sensors. Its spectral structure
> varied systematically with sleep stage. However, the mask did not recover within-session
> rate variation (indistinguishable from a temporal-shuffle null, and only ~12% of the
> ceiling set by two independent PSG respiratory sensors), and it did not capture the
> cortical electrographic hallmarks of NREM sleep: neither slow-wave activity (delta) nor
> sleep spindles (sigma) were detectable, whereas an identical detector recovered both
> from contact EEG (AUC ≈ 0.74 and 0.98). These results delineate the capacitive mask as a
> practical tool for unobtrusive overnight cardiorespiratory monitoring, while clarifying
> that temple-placement capacitive sensing transduces mechanical, not cortical-electrical,
> signals.
