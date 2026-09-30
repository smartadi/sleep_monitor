# K-complexes and the capacitive sleep mask

_Written to accompany `CAP_sleep_mask_kcomplex_update.pptx` (this folder).
Rebuild the deck with `.venv/Scripts/python.exe writeup/review/_build_kcomplex_deck.py`._

This answers the September question — **"did you also compare our signals with
K-complex in EEG?"** — using the PSG's own scored K-complex marks rather than our
detector's output, and places the answer on the Fultz et al. 2019 chain.

---

## Background: what a K-complex is, and why it is the right event to ask about

A K-complex is one of the two graphoelements that define N2 sleep, the other
being the sleep spindle. Physiologically it is an **isolated cortical
down-state** — a large population of neurons falls silent together, then
resumes. It is the single-event version of the slow oscillation that fills N3.
The AASM definition is a well-delineated negative sharp wave immediately
followed by a positive component, total duration ≥ 0.5 s, maximal frontally.

It is the right event for this project because it carries a stereotyped
**autonomic and hemodynamic transient** — heart-rate and vasomotor changes
following the cortical event. A capacitive sensor measures tissue displacement,
not voltage. So a K-complex is a large, discrete, cortically-defined event with
a known mechanical consequence: exactly the test case for the paper's claim that
the mask senses the consequence rather than the cortical activity itself.

**Where the marks came from.** The PSG export carries a `Spindle  K` channel
holding both Spindle and K-Complex impulse marks. We had not used the K-Complex
side before. It holds **57 marks cohort-wide** (50 inside the capacitive
recording windows), in 11 of 12 sessions — S4N2 has no such file.

---

## Slide 1 — `fig_kcomplex_morphology.png`

**A** shows eight individual scored K-complexes; **B** the average of all 50.

The average is the textbook morphology: a sharp negative deflection to **−70 µV
at +0.1 s**, a large positive swing to **+65 µV at +0.35 s**, flat again by
+0.7 s. Median **161 µV** peak-to-peak — the largest event in healthy sleep EEG,
which is why it is visible in a single raw trace with no averaging. That the
grand average comes out clean confirms both the marks and the time alignment.

**C** is the panel that governs everything after it. Around K-complexes the
spectrum lifts in **delta (0.5–4 Hz)**; around spindles it lifts at **12–14 Hz
(sigma)**; random N2 sits between.

That difference is why the spindle test and the K-complex test are **not
interchangeable**. For spindles the question can be settled on frequency alone:
sigma sits far outside the capacitive band, so "no sigma in the mask signal"
(0.02 dB, against 3.45 dB in EEG) rules out electrical pickup outright. A
K-complex's own frequency is 0.5–4 Hz, which **overlaps the bands the mask
measures**, so that argument is unavailable here and timing has to carry it.

---

## Slide 2 — `fig_kcomplex_vs_detector.png`

This slide is a correction as much as a result.

The manuscript's delta-burst onsets are described as "isolated N2 slow waves and
K-complexes". The expectation going in was that a K-complex, being a sub-second
waveform, could not satisfy the detector's ≥ 4 s sustained-burst criterion, and
therefore that the two were different objects. Putting all 50 scored marks
through the detector's gates shows otherwise:

| | scored K-complexes |
|---|---|
| delta envelope holds above the burst threshold | median **3.5 s** (IQR 2.3–6.1) |
| pass the ≥ 4 s sustained-burst gate | 22/50 (**44%**) |
| pass the 30 s quiet-baseline gate | 25/50 (**50%**) |
| **pass both — detector-eligible** | **9/50 (18%)** |

A K-complex usually *is* a multi-second delta run, because one is typically
followed by further slow waves rather than standing alone. And the 9 that pass
both gates are **exactly the 9 our detector found** — it recovers **9/9** of the
scored K-complexes that meet its own definition.

**How the manuscript should put it:** the onsets are isolated N2 delta events,
predominantly slow waves and K-complexes, *selected* for sitting at the head of a
sustained delta run after a quiet baseline. A genuine subset of the K-complex
population, roughly a fifth of it — not the population itself, and not something
unrelated.

**One caveat on the reference set.** The scorer annotated spindles exhaustively
(21,881) and K-complexes barely at all (57), against a physiological density of
roughly 1–3 per minute in N2. These 50 marks are good *examples*; they cannot
support a detection-rate claim in either direction.

---

## Slide 3 — `fig_kcomplex_cap_response.png`

The same three questions the spindle figure asks, triggered on the scored
K-complexes. Because the trigger is the technologist's, not ours, this analysis
shares **no machinery** with the delta-burst work already in the manuscript.

Low-frequency mask power (mean of the 0–0.5, 0.5–1 and 1–3 Hz envelopes,
standardised over NREM, baselined −15 to −2 s), against a count-matched
random-NREM null:

| channel | peak | latency | null peak |
|---|---|---|---|
| **CH** | **+0.47 z** | +1.9 s | +0.06 z |
| CLE | +0.34 z | +1.9 s | +0.03 z |
| CRE | +0.32 z | +2.9 s | +0.08 z |

Five to eight times the null, with **CH largest — the same channel ordering the
spindle result gives** (CH 0.55 dB vs CLE/CRE 0.45–0.49 dB).

**Panel B carries the argument.** Per-event peak latency is **3.6–4.3 s**, while
the simultaneous EEG delta positive control in panel A peaks at **t = 0**.
Electrical coupling is instantaneous, so a response from electrical pickup would
land at zero lag. It lands three to four seconds later.

**Caveats, to state plainly.** n = 50 across ten usable sessions; S4N2 and S5N1
contribute none and S1N2 contributes one. Panel C shows per-session peaks from
0.01 to 1.77 z, so this is reported descriptively and per-session with no pooled
p-value. And the arousal confound of §3.6 is unchanged: K-complexes are
frequently arousal-associated, and an arousal produces its own hemodynamic
transient on this timescale. The *latency* is what this establishes; the
*trigger* remains ambiguous.

---

## Slide 4 — `fig_fultz_chain.png`

Fultz et al., *Science* 2019, recorded EEG and fMRI simultaneously during NREM
sleep and followed one event through three stages:

1. a **cortical slow wave** — neurons fall silent together;
2. **cortical blood volume falls** — less activity, less metabolic demand, so
   the vasculature constricts;
3. **CSF flows in** — the skull is a closed box, so the departing volume must be
   replaced.

This is a large part of why slow-wave sleep is thought to matter for glymphatic
clearance: the slow oscillation effectively pumps fluid through the brain.

**The critical detail is that their measurement is a lag, not a correlation.**
The slow-delta EEG envelope does **not** correlate with CSF flow at zero lag at
all; it **leads it by about 6.4 s** by best-fit impulse response. (This is
already encoded in the repo, in `analysis/swa_validation/fultz_eeg_cap_impulse.py`.)

That matters to us twice over.

**It explains our own earlier negative.** The SWA validation compared capacitive
delta power against EEG delta power **at zero lag** and found r ≈ 0.015. Under
Fultz that is the *expected* result and says nothing about downstream coupling,
because the coupling is displaced in time. The negative was real but it was never
the end of the question.

**It predicts where to look instead, and the prediction holds.** Our **1.9–4.3 s**
sits between the cortical event and the ventricular CSF response — which is
where a sensor measuring **cranial displacement** should sit. Fultz measure the
end of the chain, in the fourth ventricle; the mask sits one step earlier, at the
temple. Same direction, shorter latency, consistent mechanism.

This is what lets the paper name a mechanism instead of asserting "mechanical,
not electrical" on the strength of the spindle result alone.

_The schematic is drawn from scratch; the only numbers taken from Fultz are the
~6.4 s lag and the ordering of the stages._

---

## Where this leaves us

**Solid.** The mask responds to K-complexes; the response is low-frequency; and
it follows the cortical event by 3–4 s rather than coinciding with it. Shown on
an independent trigger, consistent with the 340-event delta-onset analysis
already in the manuscript, and consistent in direction with Fultz.

**Not settled.** Whether the K-complex or its accompanying arousal is the
trigger. 47–84% of onsets fall within 10 s of a scored arousal. Separating them
needs a larger cohort with enough arousal-free slow-wave onsets.

**A limitation of the reference, not the method.** The PSG's K-complex channel is
not exhaustively scored, so it supports these events as examples but cannot
support a detection-rate claim.

**Proposed for the manuscript.** Report this as a short addition to the spindle
section (§3.5), and use the Fultz chain in **§4.4** — which currently carries a
"Comparison with Existing Physiological Monitoring Methods" heading over a
paragraph that explicitly declines to compare. This would give that section real
content.

---

## Provenance

| figure | script |
|---|---|
| `fig_kcomplex_morphology.png`, `fig_kcomplex_vs_detector.png` | `analysis/delta_onset/kcomplex_morphology.py` |
| `fig_kcomplex_cap_response.png` | `analysis/delta_onset/kcomplex_cap_response.py` |
| `fig_fultz_chain.png` | `analysis/delta_onset/fultz_chain_schematic.py` |
| deck | `writeup/review/_build_kcomplex_deck.py` |

Tables: `analysis/delta_onset/outputs/{kcomplex_marks,kcomplex_criteria,kcomplex_cap_response}.csv`.
Full record in `notebooks/ANALYSIS_LOG.md`, 2026-09-30.
