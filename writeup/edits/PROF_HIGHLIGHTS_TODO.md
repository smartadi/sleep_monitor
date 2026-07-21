# Professor Highlights → Revision To-Do

_Source: `writeup/edits/CAP_sleep_mask_manuscript_V2_prof.docx` (extracted to `prof_highlights.txt`),
54 highlighted spans. The markup is **highlighter-only — no written comments**, so each item below is an
**interpretation** of what the highlight is flagging, with a recommended action and a confidence tag._

**How to read confidence:** `[certain]` = the highlight is self-evidently a defect or the intent is
unambiguous · `[likely]` = strong inference from context · `[guess]` = plausible but confirm with the prof.

**The dominant pattern:** most highlights (HL#9–47, HL#48–54) fall on **negative / near-chance results and
strong-claim wording**. Read together, the professor is marking the same framing mismatch that
`SUBMISSION_REVIEW.md` P0#1 calls out — the paper reports many weak results in assertive language. Treat the
cluster as one editorial signal, not 40 separate edits.

---

## A. Definite cleanup (do first)

- [ ] **HL#6 — leftover outline note in the body.** Results (§4) opens with the literal instruction
  _"Start with mean value (raw signal changes) in comparison to sleep stages."_ (prof_extract line 142).
  This is a scratch note, not prose. **Action: delete it** (or, if a mean-value-vs-stage result is
  actually intended, replace with the real subsection). `[certain]` — also flagged in SUBMISSION_REVIEW P0#4.

---

## B. Methods precision (short, targeted edits)

- [ ] **HL#1 (¶145) — band-limited motion removal.** Highlighted: _"so that only motion energy within that
  band was removed."_ Prof is likely questioning whether band-limiting the accelerometer regression could
  also remove genuine in-band physiology (respiratory/cardiac motion that is real signal, not artifact).
  **Action: add one sentence** justifying that band-limited removal targets motion contamination and stating
  what protects true physiology (e.g. accelerometer has no cardiorespiratory component in-band, or the
  coupling coefficient is small). `[likely]`
- [ ] **HL#3 (¶158) — "counting systematically miscounts relative".** The word **"miscounts"** frames the
  SEC behaviour as an error. It is not an error — it is a deterministic ~2:1 waveform structure.
  **Action: reword** to "systematically differs from" / "over-counts by a fixed ratio" and make clear it is a
  predictable property, not a fault. `[likely]`
- [ ] **HL#4 (¶160) — "Mean absolute error (MAE)".** Prof marked the term. Likely wants MAE **defined at
  first use** and disambiguated from the median-based reporting (Methods says MAE is reported as *median*
  absolute error — that is a naming inconsistency worth fixing). **Action: reconcile MAE naming** across
  Methods/Results. `[likely]`
- [ ] **HL#5 (¶169) — "Lucey et al. (2019)."** Citation flagged. **Action: verify the reference is complete
  and correctly formatted** in the bibliography (and that the pipeline description matches what was actually
  replicated). `[guess]`

---

## C. Respiratory consensus caveat (¶151)

- [ ] **HL#2 — two spans in the same paragraph.**
  - _"Flow and RIPSum, which measure airflow and chest-wall expansion respectively"_ — likely a request to
    confirm these are the two most-independent sensors and that the parenthetical definitions are correct.
  - _"The median absolute difference between consensus and Flow-only was 0.06 br/min, though 29% of epochs
    differed by more than 1 br/min."_ — the **29% caveat** sits in tension with the headline "MAE < 1
    br/min." **Action: add a topic sentence** clarifying that the validated quantity is the *per-session
    mean*, and that epoch-level disagreement (29%) is expected spread, not method failure. `[likely]`
    (Same point as SUBMISSION_REVIEW P2#11/#13 — the mean-vs-epoch distinction.)

---

## D. Oracle / tracking wording

- [ ] **HL#7 (¶386) — "respiratory headroom" (oracle analysis).** Prof marked the headroom sentence
  (channel-oracle 1.08 vs method-oracle 0.54 vs full-oracle 0.16 br/min). **Action: make sure "headroom" is
  defined** and that the reader understands these are *oracle upper bounds* (best-case selection), not
  achieved accuracy. `[likely]`
- [ ] **HL#8 (¶393) — "battery" (tracking battery).** Jargon flag. **Action: rename** "tracking evaluation
  battery" → plain "set of tracking tests" or define it. `[guess]`

---

## E. The LOSO N3 classification block — HL#9 through HL#47 (Fig 12 + Table 5 + every cell)

The professor highlighted the **entire N3 result**: the paragraph (HL#10), Figure 12 caption (HL#11),
Table 5 caption (HL#12), every column header (HL#13–17), and **every cell of all six subject rows**
(HL#18–47). Highlighting a whole table end-to-end is a signal about the **result**, not the individual
numbers.

- [ ] **Decision required: does this near-chance result belong in the main text?** Pooled AUC = 0.534,
  per-subject range 0.421–0.604, mean F1 = 0.095 — statistically-significant-but-useless. Options:
  1. **Move Table 5 + Figure 12 to Supplement**, keep one honest sentence in main text ("near-chance,
     AUC 0.534; details in Supplement"). *(recommended — declutters the main results)*
  2. Keep in main text but **reframe the framing sentence** so it reads as a deliberate boundary result,
     not a failed classifier. `[likely]` — aligns with SUBMISSION_REVIEW P0#1 framing fix.
- [ ] **Verify the table numbers** against the source (`analysis/slow_wave/` outputs / Table 5 generator)
  while you're in here — the prof highlighting every cell is also an invitation to double-check them. `[guess]`
- [ ] **HL#10 specifically** — the "subject-dependent HER direction cancels pooled discrimination"
  explanation is the load-bearing sentence; keep it wherever the result lands, and **define HER at first
  use** (SUBMISSION_REVIEW P2#16). `[certain]` on the HER definition.

---

## F. Key Discussion claims — HL#48 through HL#54 (strong-claim wording)

These are the paper's headline sentences. The professor marking all of them together says: **scrutinize the
claim strength.** Recommended per-item:

- [ ] **HL#48 (¶479) — SWA negative** ("does not measure cortical slow-wave activity", r = 0.015). Keep, but
  ensure it is framed as a *tested-and-bounded* result, not a bare failure. `[likely]`
- [ ] **HL#49 (¶485) — Figure 14** (CAP near-chance vs EEG AUC≈0.74 ROC). Marked as the positive-control
  figure. **Action: confirm the EEG positive control is prominent** — it is what makes the negative
  credible. `[likely]`
- [ ] **HL#50 (¶542) — rate-recovery headline** ("MAE < 1 br/min ... MAE < 4 BPM after k-calibration",
  resp k≈0.97, card k≈1.95). This is the **main positive result**. Prof likely wants the k≈1.95 "2:1
  overcounting" mechanism kept tight and the mean-vs-tracking caveat adjacent. `[likely]`
- [ ] **HL#51 (¶546) — harmonic ridges "p < 10⁻¹⁶".** **Action: demote the pooled-epoch p-value** — with
  n=6 subjects this is pseudoreplication and reviewers will flag it. Report per-subject direction + a
  subject-level test instead. `[certain]` — SUBMISSION_REVIEW P1#7.
- [ ] **HL#52 (¶549) — within-session negative** ("indistinguishable from a temporal-shuffle null"). Strong
  but well-supported. Keep; make sure it is framed as the *most rigorously tested* boundary. `[likely]`
- [ ] **HL#53 (¶553) — "near chance (LOSO N3 AUC = 0.534)".** Consistent with §E. `[likely]`
- [ ] **HL#54 (¶555) — "definitive: the SEC signal ... detects intracranial pressure pulsations and scalp
  hemodynamics, not cortical electrical activity."** The word **"definitive"** is the strongest claim in the
  paper on n=6. **Action: soften** to "strongly indicates" / "our data indicate" unless you want to defend
  "definitive." `[likely]`

---

## Cross-reference

Every item here is consistent with `SUBMISSION_REVIEW.md`. The highlights add **no new issues** beyond that
review — they *confirm the professor's priorities*: fix the leftover note (A), tighten methods wording (B–D),
decide the fate of the near-chance N3 table (E), and calibrate claim strength / kill pooled p-values (F).
The one framing action that subsumes most of E+F is **SUBMISSION_REVIEW P0#1** (recast thesis so assertive
negatives read as deliberate boundary-mapping).

## Suggested order of work
1. A (delete stray note) — trivial, certain.
2. F HL#51 + E (pseudoreplication p-values, N3 table placement) — biggest credibility wins.
3. B + D (methods wording) — quick.
4. C + F HL#54 (caveat topic sentences, soften "definitive") — framing.
