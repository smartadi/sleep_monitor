# Delta-onset CAP figures (paper)

What the capacitive channels do when an EEG delta burst starts. Trigger =
quiescence-gated EEG delta-burst onset (`analysis/delta_onset/delta_onset_detection.py`);
q30 (30 s pre-onset quiescence) is the reported set, q15 is a robustness check kept in
CSV form only.

Regenerate the current set with

    .venv/Scripts/python.exe analysis/delta_onset/delta_onset_figures.py --tag q30

## What these figures report — read this first

**During EEG delta bursts the capacitive channels show a broadband band-power rise, and
these trials also carry head movement in the same window.** Both halves belong in the
result; the second is not a footnote.

Onsets were screened for a motion-clean *pre*-onset window only; the post-onset window,
where the rise sits, was never constrained. Post-onset motion flags run at **14.6 %** of
samples on average (3.0–31.6 % per session) against ~1 % before onset, peaking at ~30 % at
+4 s — the same place the capacitive peak sits. On the **150 of 339 onsets that are also
motion-clean after t = 0**, against a null carrying the identical gate, no capacitive
change is detectable in any of the nine channel × band combinations (real − null −0.045 to
−0.170 z; 0–3 of 5 evaluable subjects positive). One subject supplied 179 of the 339 onsets
and is the most affected night (26/99 and 9/80 clean), so the pooled average is weighted
toward the recordings where movement most often follows a burst.

The delta bursts themselves are genuine: motion-clean onsets still carry an EEG delta peak
of **2.47 z** (vs 3.13 z for all onsets), so the detector is not firing on EEG movement
artifact. Real cortical bursts; no capacitive correlate without movement.

Mechanism is left open. At n = 6 these data cannot separate "movement is incidental and the
mask records the movement" from "movement is K-complex-associated motor activation and the
mask registers it as movement". The earlier §4.5 claim — a mechanical/hemodynamic
co-activation, an amplitude modulation of the ongoing respiratory/cardiac signals — is
withdrawn, because it predicts a response on movement-free onsets.

| File | What it shows |
|------|----------------|
| `fig_delta_onset_cohort.png` | **Figure 9 candidate — self-contained.** (A) EEG delta trigger. (B) CH with motion-free onsets only, faint = all 339, bold = the 150 motion-free. (C) response amplitude per channel × band, with open red circles marking the motion-free value (≈ 0). (D–F) all three bands per channel, mean ± SEM vs matched null, **with the head-motion profile shaded behind the curves on the right axis** so the band-power rise and the movement that occupies the same window are read together, not in separate panels. |
| `fig_delta_onset_motion_control.png` | The full three-channel version of panel B — all bands, all channels, all onsets vs motion-free. Supplementary; panel B carries the point in the main figure. |
| `fig_delta_onset_session.png` | The same view inside one night (S2N2, 80 onsets), motion likewise shaded behind each channel. |
| `fig_delta_onset_subjects.png` | Per subject × channel, all bands. Per-panel y-scale with the peak printed — a shared scale is unreadable because S5 (6 onsets, quietest night) peaks 5–13 z against 1–3 z elsewhere. |
| `fig_delta_onset_age.png` | Response amplitude and half-rise latency vs subject age (n = 6, exploratory). |
| `delta_onset_response_summary.csv` | Per channel × band: response amplitude vs both nulls and vs the motion-clean null, latencies raw and bias-corrected, per-subject direction counts. |
| `delta_onset_timing_calibration.csv` | Measured timing bias of each estimator (see below). |
| `delta_onset_age_correlations.csv` | Spearman rho / p for every age test. |
| `fig_onset_gallery_S2N2_q30.png` | Representative single onsets (trigger illustration). |
| `fig_onsets_overview_S2N2_q30.png` | Whole-night hypnogram / delta envelope / motion with detected onsets. |
| `fig_precursor_{xcorr,auc}_q30.png` | CAP→EEG cross-correlation and forecasting AUC (older pipeline, zero-phase envelopes). |
| `delta_onsets_summary.csv` | Per-session onset counts at both windows. |
| `precursor_summary_{q15,q30}.csv`, `lowband_precursor_check_q30.csv`, `response_consistency_q30.csv` | Older pipeline outputs. **`response_consistency_q30.csv`'s "6/6 subjects" predates the motion control and the estimator fixes below — do not quote it.** |

## Method notes that changed the numbers

**Timing bias is measured, not assumed.** A causal envelope reports an event late and the
detector's zero-phase envelope fires early, so latencies were never what they appeared.
Measured on synthetic bursts that truly start at t = 0: causal estimator **+1.90 / +2.35 /
+0.76 s** (0–0.5 / 0.5–1 / 1–3 Hz), trigger **−0.88 s**, total correction **1.6–3.2 s**.
Both biases push the measured event later, never earlier, so the no-precursor result is
unaffected — but the "+2.4→+4.9 s" peaks in the draft are upper bounds. Corrected
half-rise latencies are **1.2–2.7 s**.

**Filter startup transient (bug).** `sosfilt` was starting from zero state against a
~2000 fF DC level, so the causal envelope rang for minutes at the recording start. That
corrupted any early window and inflated the NREM standard deviation the whole trace is
z-scored by. Now initialised to the steady state for the signal's opening level, plus a
120 s warm-up excluded from the z-score reference. **The same bug is still present in
`lowband_precursor_check.py`**, which produced the current manuscript Figure 9.

**Amplitude is a window mean, not a peak.** Max-of-a-noisy-curve is biased upward, worst
for the subjects with fewest onsets, and it inflated the null as much as the signal.

**Cross-subject amplitudes use a robust (median/MAD) scale.** These envelopes are very
heavy-tailed — a night's peak runs ~3000× its median — so a standard deviation is set by a
few movement artifacts and varies several-fold between nights. On the raw z scale the
quietest night (S5) reads 5–13 z where everyone else reads 1–3.

**The null is selection-matched.** Random-NREM centres now pass the same three gates the
onsets did (NREM, motion-clean pre-window, quiescent EEG-delta pre-window). This did *not*
explain the small pre-onset dip — it survives, slightly larger, so the dip is not a
null-selection artifact.

## Still standing

No capacitive precursor in any band: pre-onset lead amplitude is −0.007 to −0.171 z
against the matched null, 1–3 of 6 subjects positive. The zero-phase estimator still shows
a spurious positive lead in the slow band (+0.10 to +0.21 z) where the causal one shows
none — the original acausal-leak finding reproduces.

Removed as redundant on 2026-08-06: `fig_lowband_causal_check_q30.png` (per request; the
zero-phase-vs-causal numbers live in `lowband_precursor_check_q30.csv`) and
`fig_precursor_grid_causal_q30.png` (superseded by `fig_delta_onset_cohort.png`, same data
and estimator). Earlier: `fig_precursor_grid_q30.png` and the three q15 figures. All
recoverable from git history.
