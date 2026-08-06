# Delta-onset seen in CAP — consolidated story / ready-to-shape manuscript text

> **REFRAMED 2026-08-06 — observational, mechanism left open.** The earlier claim (a
> mechanical/hemodynamic co-activation of the respiratory/cardiac signals) is withdrawn: it
> predicts a capacitive response on movement-free onsets, and there is none. What replaces
> it is a description of what the mask records during delta bursts, with the movement
> co-occurrence stated as part of the observation rather than as a caveat bolted on. At
> n = 6 we cannot separate a movement artifact from K-complex-associated motor activation,
> and the section should not pretend otherwise. Superseded numbers not to quote:
> `response_consistency_q30.csv`'s "6/6 subjects" (max-of-noise statistic, pre-dates the
> motion control and the filter fix) and the "+2.4→+4.9 s" latencies (uncorrected for a
> +0.76→+2.35 s estimator delay and a −0.88 s trigger lead). §4.5 in the docx still carries
> the old mechanism text and Figure 9 — replace both.

**One-sentence claim (the story we are reporting):**
*During EEG delta bursts the capacitive temple channels show a broadband band-power
increase in all three bands and all three channels, beginning at or after cortical onset
and never before it; the great majority of these trials also carry head movement in the
same window, and on the movement-free trials no capacitive change is detectable — so what
the mask registers around a delta burst is a movement-associated event whose physiological
status this cohort cannot resolve.*

---

## Where this sits in the paper

This is the **third capacitive-vs-cortical result**, and it now resolves the same way as
the other two rather than against them:

- §3.5 — capacitive vs contact-EEG SWA: spectral r = 0.015 (mask does **not** transduce
  cortical delta electrically).
- §3.6 — sleep spindles: sigma AUC = 0.50 (no cortical sigma), **but** a small 0.1–3 Hz
  mechanical bump time-locked to N2 spindles. *Note: that bump has not had the motion
  control applied to it, and the spindle audit found it tail-driven (top 5 % of spindles
  carry 121 % of the mean) — the same signature that turned out to explain this section.*
- §3.x (CAP-SWA) — a *tonic* mechanical score marks N3 (per-subject AUC 0.675, 6/6).
  *Note: accelerometer quiescence is one of its criteria, so part of that AUC may be "N3 is
  stiller", not capacitive sensing.*
- **§3.y (this) — event-locked, and movement-confounded:** locking to individual delta-burst
  onsets shows a sharp multi-band capacitive event, which the motion control attributes to
  movement in the same window.

**Framing discipline (must hold in the text):**
1. **Descriptive, not mechanistic.** "During delta bursts the capacitive channels show X,
   and these trials also carry movement." Never "CAP measures delta", never "CAP transduces
   a hemodynamic response to delta". We report the co-occurrence and stop.
2. **The movement co-occurrence is part of the result, quantified, not a limitation.** State
   in the same paragraph as the effect that it is absent without movement. A reader must not
   be able to come away thinking movement is a partial contaminant on top of a real signal.
3. **No precursor, and that part is solid.** No band rises before onset; the apparent slow-band
   pre-onset ramp is a zero-phase-filter artifact that vanishes under causal filtering, and
   this survives a selection-matched null.
4. **n = 6 honest-negative rule:** per-subject direction counts, not p-values (Wilcoxon
   floors at p = 0.031). AUCs reported as discrimination metrics.

Analysis: `analysis/delta_onset/delta_onset_detection.py` (trigger),
`analysis/delta_onset/delta_cap_precursor.py` (peri-onset, xcorr, forecasting).
Figures (2026-08-06 set, from `analysis/delta_onset/delta_onset_figures.py`):
`fig_delta_onset_{cohort,motion_control,session,subjects,age}.png`, plus
`fig_precursor_{xcorr,auc}_q30.png` and `fig_onset{s_overview,_gallery}_S2N2_q30.png` for
the trigger. The zero-phase/q15 grids, the lowband causal check and the superseded causal
grid were removed as redundant; their numbers survive in the CSVs.
Data: `analysis/delta_onset/outputs/precursor_summary_{q15,q30}.csv`.

---

## Methods addition (pairs with §2.8–2.9, the CAP-SWA / spindle event methods)

**2.y Delta-onset triggering and the CAP peri-onset test.** To ask whether discrete
cortical slow-wave events leave a capacitive signature, we first defined an EEG
**delta-burst onset** as an event (Schmitt trigger on the 0.5–4 Hz Hilbert envelope:
per-session robust NREM baseline, high = median + 2·MAD sustained ≥ 4 s, onset walked
back to the burst's rising edge; ≥ 25 s refractory so events are near-independent). Only
onsets in NREM with a motion-clean −30→0 s pre-window were kept, and — because the whole
test hinges on a clean quiet→delta transition — we additionally required the pre-window
EEG delta to be quiescent (mean below the low threshold). Two quiescence windows are
reported (15 s / 30 s; "q15"/"q30") as a robustness check. This selects **isolated slow
wave / K-complex onsets emerging from a quiet background** (predominantly N2; sustained
N3 has no discrete quiet onset), 6–108 events per session.

For each onset we extracted three capacitive channels (CLE, CRE, CH) in three bands
(0–0.5, 0.5–1, 1–3 Hz; the 0–0.5 band high-passed at 0.03 Hz to reject DC/coupling
drift), took each band's amplitude envelope, z-scored it within NREM, and averaged the
−30→+15 s window across onsets. Aggregation is per-subject then across the six subjects
(never pooling raw epochs). Three read-outs: (i) the **peri-onset average** vs a
**random-NREM null** (count-matched, motion-clean, ≥ 60 s from any real onset); (ii) a
**CAP→EEG cross-correlation** over NREM with a circular-shift null (positive lag = CAP
leads); (iii) a **forecasting AUC** — can CAP band power in a −12→−2 s pre-onset window
separate "about to onset" from random NREM.

All envelopes are computed with a **strictly causal** estimator (forward-only Butterworth
initialised to the signal's steady state, then trailing RMS), because a zero-phase envelope
has a symmetric impulse response and leaks a large post-onset rise backward into the
pre-onset window. The cost is that a causal estimator reports events late, so its timing
bias was measured directly — on synthetic bursts that truly begin at t = 0 the estimator
reads +1.90 / +2.35 / +0.76 s late in the three bands and the zero-phase detector envelope
reads −0.88 s early — and latencies are quoted bias-corrected. Because both biases push the
measured event later, no correction can create a precursor that is not there.

**Motion control.** The detector screens the pre-onset window for movement but leaves the
post-onset window unconstrained, so the peri-onset analysis was repeated on the subset of
onsets whose −30→+15 s window is motion-free throughout, against a null carrying the
identical post-window gate (`delta_onset_figures.py`). Motion is flagged where the rolling
standard deviation of accelerometer magnitude exceeds the night's 90th percentile, and a
window counts as free if ≤ 10 % of its samples are flagged — the same criterion the detector
already applied to the pre-window.

**Aggregation and scale.** Response amplitude is the mean of the 0–10 s post-onset window,
not its peak: the maximum of a noisy per-subject curve is biased upward, and the bias grows
as onset count falls, inflating the null as much as the effect. Cross-subject amplitude
comparisons use a median/MAD normalisation rather than mean/SD, because these envelopes are
heavy-tailed (a night's peak runs ~10³× its median) so an SD-based scale is set by each
night's movement-artifact load and is not comparable across recordings.

---

## Results addition (§3.y)

**What the capacitive channels do during an EEG delta burst.** Locking the capacitive
envelopes to delta-burst onset (339 onsets, 6 subjects) shows a clear band-power increase
in every channel and every band, beginning at onset and peaking a few seconds later,
against a selection-matched null that stays flat (Figure 9). The rise is broadband — the
0–0.5, 0.5–1 and 1–3 Hz bands move together with a common shape and a common latency
rather than one band leading or dominating — and it is present in all six subjects, though
its amplitude varies about tenfold between them (per-subject peaks 0.4–13 z; Figure S1).
Response amplitude, measured as the 0–10 s mean against the matched null, is
0.38–0.91 z depending on channel and band, largest on CRE and CH in 0.5–1 and 1–3 Hz.

Corrected for the timing bias of the estimators (a strictly causal envelope reports events
+0.76→+2.35 s late; the zero-phase detector envelope fires −0.88 s early), the rise reaches
half its peak **1.2–2.7 s after cortical onset**. Nothing precedes onset in any band: the
pre-onset lead window sits at or slightly below the matched null throughout (−0.007 to
−0.171 z, 1–3 of 6 subjects positive), the CAP→EEG cross-correlation peaks at lag ≈ 0 with
its shoulder on the EEG-leads side, and pre-onset capacitive power does not forecast an
imminent onset (AUC 0.42–0.56). The apparent slow-band pre-onset ramp reported in earlier
versions of this analysis is a zero-phase-filter artifact — the estimator leaking the large
post-onset rise backward in time — and it disappears under causal filtering while the
post-onset rise is unchanged.

**These trials also carry head movement, and that accounts for the effect.** The onsets
were selected to have a motion-free 30 s *pre*-window, but nothing constrained the window
in which the capacitive rise occurs. Accelerometer-flagged motion runs at ~1 % of samples
throughout the pre-onset baseline and rises to **~30 % at +4 s**, averaging **14.6 % across
the post-onset window (3.0–31.6 % per session)** — the same time course as the capacitive
rise. Restricting to the **150 of 339 onsets whose post-onset window is also motion-free**,
and comparing them against control epochs passing that identical gate, no capacitive change
is detectable in any of the nine channel × band combinations (0–10 s mean, real minus null
**−0.045 to −0.170 z**, with 0–3 of 5 evaluable subjects positive; Figure 9b). The
contamination is unevenly distributed: one subject contributed 179 of the 339 onsets and is
the most affected night (26/99 and 9/80 onsets motion-free), so the pooled average is
weighted toward the recordings in which movement most often follows a delta burst.

The delta bursts themselves are not in question. Motion-free onsets still carry an EEG
delta peak of **2.47 z** (versus 3.13 z for all onsets), so the trigger is detecting genuine
cortical events rather than movement artifact in the EEG; it is the capacitive correlate,
not the cortical event, that is absent without movement. A broadband, simultaneous rise
across bands spanning 0.03–3 Hz is also what electrode–skin displacement produces, whereas
a modulation of the respiratory or cardiac mechanical signals the mask resolves would be
concentrated in the bands those rhythms occupy.

We therefore report this as an observation rather than a mechanism. Two readings remain
open and this cohort cannot separate them: the movement is incidental to the delta burst
and the mask is recording the movement; or the movement is K-complex-associated motor
activation, in which case the mask is registering a real physiological accompaniment of the
cortical event — but registering it *as movement*, not as a hemodynamic or respiratory
transient. Distinguishing them requires either a movement-free subsample far larger than
six subjects affords, or an independent autonomic reference (pulse or effort) recorded
through the movement.

---

## Limitations (fold into the discussion)

- **The movement confound is not resolvable here.** Excluding movement removes the effect
  entirely, so we cannot report a capacitive correlate of delta onset that is independent of
  movement; and because movement genuinely accompanies K-complexes, we equally cannot call
  the effect a pure artifact. The honest position is that the two are inseparable at n = 6.
- Onsets are **isolated N2 slow-wave / K-complex onsets** by construction (the quiescence
  gate removes sustained N3, which has no discrete quiet onset). The observation is about
  *delta-burst onset events*, not deep-sleep SWA en masse.
- **Motion is defined by a threshold, not measured absolutely** — a flag on the rolling
  standard deviation of accelerometer magnitude above the night's 90th percentile, with
  "motion-free" meaning ≤ 10 % of window samples flagged. Sub-threshold micro-movement is
  neither excluded by the gate nor measured.
- Amplitudes vary ~tenfold across subjects on the standard z scale, largely because that
  scale divides by each night's envelope standard deviation, which is itself set by that
  night's movement artifact load. Cross-subject amplitude comparisons use a median/MAD scale
  for this reason.
- n = 6; direction is reported per-subject. The motion-free analysis rests on 5 evaluable
  subjects and 150 onsets.

## Status — REOPENED 2026-08-06 (see the box at the top; the list below is what was believed on 07-30)
- [x] Exact post-onset peak table filled (q30).
- [x] 0–0.5 Hz "precursor" checked and **excluded** as a zero-phase-filter artifact
      (`lowband_precursor_check.py`). No precursor in any band.
- [x] Professor's precursor hypothesis dropped from the framing (per user).
- [x] **Per-subject response-consistency: 6/6 subjects in ALL nine channel×band combos**
      (causal post-onset peak > own random-NREM null; `response_consistency_q30.csv`).
      Per-subject peak +1.4→+3.2 z (largest CRE/CH, 0.5–1 & 1–3 Hz); null peaks ≤ +0.17 z.
- [x] **Figure 9 = strictly-causal peri-onset grid** (`fig_precursor_grid_causal_q30.png`):
      flat pre-onset baseline in all 9 panels + sharp post-onset rise → shows the response
      AND the no-precursor result in one figure, with no acausal backward-leak.
- [x] **Inserted into `writeup/main/CAP_sleep_mask_manuscript_main.docx` as §4.5** (heading
      + 3 paragraphs + embedded Figure 9 + caption), after §4.4 spindles, before §5. Validated
      (XSD passed); image rId resolves. Backup: `…BACKUP_2026-07-30.docx`. **Verify render in Word.**
- [ ] Optional: xcorr + auc panels as supplement; formal consistency table in the docx.

## Status — 2026-08-06 (current)
- [x] Motion control run; response absent on motion-free onsets. Section reframed as an
      observation with the movement co-occurrence stated as part of the result.
- [x] Timing bias measured and latencies corrected (half-rise 1.2–2.7 s).
- [x] Estimator fixed (filter startup transient), amplitude metric changed from max to a
      0–10 s window mean, cross-subject scale changed to median/MAD, null selection-matched
      on EEG-delta quiescence.
- [x] Results / Methods / Limitations text above rewritten for the new framing.
- [x] **Motion is reported alongside the bands, inside the figure.** `fig_delta_onset_cohort.png`
      now shades the head-motion profile behind the band curves in every channel panel (right
      axis, % samples with motion) and devotes panel B to the motion-free subset and panel C to
      its amplitude. The effect and its confound cannot be looked at separately. One figure
      therefore carries the whole result; `fig_delta_onset_motion_control.png` becomes
      supplementary rather than required.
- [ ] **Docx not touched.** §4.5 still carries the old mechanism paragraphs and the old
      Figure 9. Replace the prose with the Results text above and Figure 9 with
      `fig_delta_onset_cohort.png`. Caption must name the motion shading explicitly — the
      figure only protects the reader if the caption says what the red backdrop is.
- [ ] Decide where the age observation goes, if anywhere: no amplitude–age relation on the
      robust scale; half-rise latency rises with age (CH 0–0.5 Hz rho = +0.94, p = 0.005,
      8/9 combinations positive), but given the above that is movement timing, and it is
      exploratory at n = 6 with 18 uncorrected tests. Recommend supplementary or omit.
- [ ] `lowband_precursor_check.py` is superseded by `delta_onset_figures.py` and still
      carries the filter-startup bug — retire it, or fix it if its CSV is to stay cited.
