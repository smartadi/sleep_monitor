# Claim tags — rate / SWA / staging (reporting-simple pass, 2026-08-26)

Every substantive claim in the rate, SWA, and staging sections tagged
**as-is** / **soften** / **drop** against the reporting-simple theme (n=6,
descriptive, subject as unit, no Bland-Altman, no headline p-values).

**Headline:** the manuscript already implements the review. §3.8 Statistical
Methods states the unit of analysis is the recording, discloses pooled-epoch
anticonservatism, prints those p-values only beside per-subject direction
counts, uses exact permutation for n=6, and notes the p=0.031 floor. The rate
section already reports the four-calibration table and retires k-vs-age. So most
claims are **as-is**; the real edits are few.

## Rate (§3.5, §4.2, §5.1, §5.4, §6)

| # | Claim / location | Tag | Note |
|---|---|---|---|
| R1 | Night-mean error 0.24 br/min, 1.56 BPM after per-session k (§4.2, §5.1) | **as-is** | Conditionality stated; night-mean is the right simple metric. |
| R2 | Per-epoch error 1.79 br/min, 3.41 BPM (§4.2) | **as-is** | Reported per-recording median [IQR]. |
| R3 | **Bland–Altman bias/LoA sentence, Fig S5 (§4.2 [98])** | **DROP** | Your directive: no Bland-Altman. Night-mean + per-epoch already carry agreement. Fig S5 becomes unreferenced — drop from supplement or leave as an unused figure. |
| R4 | Within-night variation not recovered; median r −0.03/−0.08 vs circular-shift null (§4.2, §5.2) | **as-is** | Circular-shift null, per-night unit, best-night caveat — exactly right. |
| R5 | Four-calibration table; nothing beats no-sensor except same-night k (§4.2, Table 4) | **as-is** | The honest core result; keep verbatim. |
| R6 | k≈2 = 1.70–2.43 peaks/beat; r=0.50 n=9 "agreement of central values not covariation" (§4.2.1, §5.1) | **as-is** | Already the correct, non-overclaiming framing. |
| R7 | Respiratory k=1.18, not ≈1 (§4.2.1, §5.1) | **as-is** | Correctly stated. |
| R8 | k-vs-age: "no relationship that survives this sample size … \|ρ\|≥0.83 to be significant" (§4.2.1, §6) | **as-is** | Dead claim already retired with the exact-n reasoning. |
| R9 | Decoder recovers r≈0.245, transfers for respiration (§4.2.2) | **as-is** | Blocked CV, circular-shift null, variance-explained caveat. |
| R10 | §5.4 "well-characterized calibration behavior and **per-stage accuracy** (Fig S2)" | **soften** | "per-stage accuracy" rests on pooled epochs (Fig S2). Keep the calibration claim; mark per-stage error as descriptive/pooled. |
| R11 | §5.4 screening use "contingent on a calibration transfer this study does not demonstrate" | **as-is** | Correctly hedged. |

## SWA / cortical (§4.4, §4.5, §4.6, §5.2)

| # | Claim / location | Tag | Note |
|---|---|---|---|
| S1 | SWA r=−0.014, sub-bands ≈0, coherence 0.003; "the absence of one" (§4.6) | **as-is** | Sound negative. Optional add: a one-line per-subject CI ("cohort mean r 95% CI excludes \|r\|>0.1") strengthens it without any Bland-Altman. |
| S2 | CAP N3 AUC 0.490 vs EEG 0.740, positive control per night (§4.6) | **as-is** | The EEG control is the power argument; keep. |
| S3 | SD reported "across recordings" (n=12) (§4.6) | **soften (minor)** | Strictly this is 12 nights not 6 subjects; a phrase ("across the twelve recordings") already present makes the unit explicit, so low priority. |
| S4 | Spindle low-band response mechanical, sigma null (§4.4) | **as-is** | Per-subject (6/6), within-subject averaging first — clean. |
| S5 | Delta-burst response follows onset; arousal confound cannot be separated at n=6 (§4.5) | **as-is** | Exemplary honest hedging. |
| S6 | Harmonic-comb ladder follows REM, "exploratory, per-subject-consistent … not a powered effect" (§4.7) | **as-is** | Already framed exploratory. |

## Staging / ridges (§3.6, §4.3, §5.1, §5.4)

| # | Claim / location | Tag | Note |
|---|---|---|---|
| T1 | Reduced N3 respiratory-ridge power, "weak and partially consistent," 4/6 (5/6 by mean) (§4.3, §5.1) | **as-is** | Per-subject direction count is the evidence; correctly called weak. |
| T2 | Ridge counts carry no consistent direction; earlier "consistent across 6" retracted (§4.3, §5.1) | **as-is** | Good — the retraction is explicit. |
| T3 | **Pooled KW/MWU "down to 9×10⁻²⁹" printed in text + Fig 6 (§4.3)** | **soften** | Already labeled descriptive/non-independent. Under reporting-simple, drop the dramatic figure from the body text; keep the "non-independent, descriptive only" sentence and the panel labels. |
| T4 | §5.4 ridge features "too weak for standalone staging," possible multimodal input | **as-is** | Correctly bounded. |
| T5 | `analysis/staging/CLAUDE.md` lists k_cardiac "p=10⁻¹³⁰" as a per-epoch feature | **DROP (doc, not paper)** | Not in the manuscript — only in the planning doc. Dead claim (pooling artifact; k is one scalar/session; violates no-k-derived rule). Fix the note so it doesn't propagate. |

## Edits applied this pass (see `apply_reporting_simplify.py`)
- **R3** — removed the Bland–Altman bias/LoA sentence in §4.2; night-mean and per-epoch errors carry agreement.
- **R10** — softened §5.4 "per-stage accuracy" to a descriptive, pooled Figure S2 reference; kept the calibration claim.
- **T3** — dropped the "down to 9×10⁻²⁹" figure from §4.3 body; kept the non-independence disclosure.

## Decisions left to you
- **Abstract** ([9] is still the deliberate `[Abstract — to be written.]` placeholder — you removed it previously via `remove_abstract.py`). Not written here without your go-ahead; say the word and I'll draft it in the reporting-simple voice.
- **Figure S5 (Bland–Altman)** now unreferenced — drop it from the supplement, or leave it unused?
- **§3.8 Methods** needs **no change** — it already states everything the review asked for.
