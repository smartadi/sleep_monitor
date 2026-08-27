# Reviewer report — methods & code, rate-estimation section

**Date:** 2026-08-26
**Scope:** the algorithms behind the manuscript's quantitative claims — rate
estimation (§3.5, §4.2), ground-truth extraction (§3.3), the k factor and its
age story (§4.2.1), and the statistics used throughout. Read as a peer review:
findings are graded, each names the file/line and, where I built one, the test
or metric that pins it.

**What I added to the repo (runnable):**
- `sleep_monitor/rate_metrics.py` — the paper-level metrics the current
  `metrics.py` is missing: frequency-sweep degeneracy/octave detectors, the
  four-regime held-out-calibration table with a no-sensor baseline, a
  circular-shift null, exact small-n Spearman, and a unit-aware Bland-Altman.
- `tests/test_rate_algorithms_review.py` — 18 guard tests encoding the failure
  modes below (C1–C6).
- `tests/test_rate_metrics.py` — 20 tests validating the metrics on synthetic
  cases with known answers.

Both new test files pass (38/38). One pre-existing failure in
`tests/test_preprocessing.py` (NLMS short-signal, `filtfilt` padlen) is
unrelated to this review.

---

## Summary judgement

The internal audit (`RATE_AUDIT_2026-08-06.md`) and the consolidated
`analysis/rates/CLAUDE.md` already reach the correct scientific conclusion: the
within-night negative result is sound, and the positive night-level rate claims
are a same-night fitting residual that does not beat a no-sensor constant. I
**confirm** that framing independently, and I add several *code-level* defects
that either (a) explain a reported number mechanistically, or (b) are latent bugs
that will corrupt results the moment they are exercised. Two of these are new
relative to the audit (C2, C5). The most important action items are C4 (the
packaged calibration function does not reproduce the paper) and adopting the
held-out-calibration metric as the reported result.

---

## A. Code defects that affect published numbers

### C1 — `rate_spectral` is degenerate in the respiratory band *(confirms audit A.1)*
`sleep_monitor/rates.py:42`. `nperseg = max(64, int(fs*4))` fixes the Welch
segment at 4 s → Δf = 0.25 Hz. Across the whole 0.1–0.5 Hz band the estimator
emits **only two distinct values** (0.25 and 0.5 Hz); a true 12 br/min breather
is reported at 15 br/min. Verified by `frequency_sweep` +
`degeneracy_report`: `distinct_fraction ≈ 0.05`, modal output 0.25 Hz for >90%
of the band. Any k fitted to this is `15/median(reference)`, so the published
resp 0.91 br/min is the error of the best constant, not a measurement.
**Guarded by** `TestSpectralDegeneracy`.

### C2 — `rate_acf` locks onto a subharmonic (reports HALF the rate) *(new)*
`sleep_monitor/rates.py:95`. The estimator selects the ACF peak of maximum
*prominence* (`k = peaks[np.argmax(props['prominences'])]`), not the shortest
qualifying lag. On sinusoids this returns half the true rate for **f ≥ 0.30 Hz
(18 br/min)** in the respiratory band and **f ≥ 2.0 Hz (120 BPM)** in the
cardiac band — verified across the band by `octave_report` (`half_fraction ≈
0.5` in the upper resp band). A 120 BPM tachycardia read as 60 BPM is a
clinically serious silent error. This estimator is in the fusion set
(`METHOD_NAMES`) and in `peaks_by_method`, so it can pull fused estimates toward
a subharmonic. **Fix:** prefer the shortest lag whose prominence exceeds a
threshold, or constrain the ACF search to `[1/f_hi, 1.5/f_lo]` and reject
lags that are ~2× a stronger shorter-lag peak. **Guarded by**
`TestAcfOctaveError`.

### C3 — `rate_peaks_scaled_resp` uses a different rate definition than `rate_peaks` *(extends audit)*
`sleep_monitor/rates.py:413` computes `(len(pks)/k)/(len(x)/fs)` — count over
the *full window*. `rate_peaks` (line 161) computes `(len(pks)-1)/peak_span` —
intervals over the *span between first and last peak*. On a clean 0.25 Hz sine
at the operational 30 s epoch, the scaled variant with k=1 reads 0.267 vs the
true 0.25 (a +6.7% overcount) while `rate_peaks` reads 0.25 exactly. The excess
is not physiology — it is the `N/T` vs `(N-1)/span` edge handling, and it
depends on where the epoch boundary falls relative to the breathing phase (the
estimate is *not* invariant to epoch phase). Since the operational respiratory
estimator is exactly this function and k = median(estimate/reference), part of
the fitted respiratory **k ≈ 1.18 is a definitional artifact of the estimator,
not a breath-to-deflection ratio.** This weakens the "peak counting registers
~18% more deflections than breaths" gloss in `analysis/rates/CLAUDE.md`.
**Fix:** make the scaled variant use the same `(N-1)/span` definition as
`rate_peaks`, then re-fit k. **Guarded by** `TestScaledRespDefinitionMismatch`.

### C4 — the packaged k-calibration does not reproduce the paper *(locates audit B/§3.5 mismatch)*
`sleep_monitor/rates.py:454` `_calibrate_k` and its public wrappers
`calibrate_k_cardiac/resp` still implement **"50 randomly drawn one-minute
windows"** (`n_windows=50, win_s=60`) and their docstrings quote **k = 1.67
(cardiac), ~1.3 (resp)**. The current pipeline and `analysis/rates/CLAUDE.md`
say k is the **whole-night median** of estimate/reference and the operational
values are **k = 1.96 / 1.18**. So:
- The manuscript's "50 randomly drawn one-minute windows … |k_diagnostic −
  k_whole| ≤ 0.04" sentence describes *this dead function*, not the pipeline
  that produced any reported number — exactly the provenance the audit could not
  find. It is inherited from a superseded pipeline.
- A reader who imports `calibrate_k_cardiac` (the public API) gets a *different*
  k than the paper, computed by a *different* method, with a *stale* docstring
  value. This is a reproducibility trap in the shipped package.
- The wrappers also calibrate against single PSG channels (`Thorax`, `Pleth`),
  not the multi-signal consensus GT the paper uses — a third inconsistency.
**Fix:** either delete the 50-window calibrators or reimplement them as thin
wrappers over the whole-night-median-vs-consensus definition, and correct the
docstrings and the manuscript sentence together.

---

## B. Latent correctness bugs (not yet in a headline number, but in the reference path)

### C5 — `_quality_filter` deletes the good beat *after* a bad interval *(new)*
`sleep_monitor/ground_truth.py:72`. Intervals are computed once from the
original peaks; when interval *i* is out of band, the filter sets
`keep[i+1]=False` — it removes the peak *after* the gap and never re-evaluates.
For a **missed beat** (a too-long interval) this discards a *valid* downstream
beat and makes the gap longer, without ever removing an impossible interval.
Verified: peaks `[0,100,200,450,550,650]` with a 2.5 s gap → the filter deletes
the beat at 450 (good) and leaves a >3 s span (still out of band). This runs on
the **cardiac and respiratory ground truth** every other result is scored
against. Impact is bounded because neurokit R-peaks are usually clean, but it is
a genuine bias in the reference and should be replaced with a symmetric rule
(drop the peak that makes the *short* interval; for long gaps, insert nothing /
flag, do not delete a neighbour). **Guarded by** `TestQualityFilterAsymmetry`.

### C6 — the existing test suite hides C1 by frequency choice *(meta)*
`tests/test_rates.py` tests every estimator at `RESP_F0 = 0.25 Hz` — the single
respiratory frequency at which the degenerate `rate_spectral` is correct.
`test_all_methods_near_correct_rate` therefore **passes because of the bug**. A
band estimator must be swept across its band, never validated at one interior
point. `test_rate_metrics.py` and `frequency_sweep` do this. **Guarded by**
`test_existing_suite_frequency_is_the_degenerate_bin`.

Minor docstring/consistency defects (fix for the camera-ready, non-blocking):
`rate_hilbert` docstring says "weighted by amplitude" but the code gates by a
25th-percentile amplitude mask and takes an unweighted median (line 129);
`sliding_rates` docstring says "all six methods" but is driven by `METHOD_NAMES`
(line 545); the module header says "Eight methods" while `estimate_rate` returns
six (+envelope). These mirror the §3.5 "six vs seven estimators" contradiction
the audit flagged (C.2).

---

## C. Methods issues (I endorse the audit and provide the metric to settle each)

| # | Issue | Metric provided |
|---|-------|-----------------|
| M1 | k is fit on the same recording it is scored on → night-level agreement is circular. Report **held-out** calibration with a **no-sensor** comparator. | `evaluate_calibration_regimes` (self / cross-night / population / no-sensor) + `paired_wilcoxon_vs_baseline`. Reproduces the audit's finding that only same-night k beats the baseline. |
| M2 | Within-night "tracking" r must be tested against a null that preserves the marginals but breaks time alignment. | `circular_shift_null` — validated to reject real tracking (p<0.05) and to hold its false-positive rate near α on two independent smooth series. |
| M3 | Bland-Altman LoA over ~90k pooled epochs from 12 nights are not night-to-night agreement limits (epochs within a night are not independent). | `bland_altman` returns `n`; if `n` is in the thousands you are on the wrong unit. Feed one value per recording. |
| M4 | n≤6 correlations: the t-approximation is invalid; the reported k–age p = 0.042 is anticonservative. | `exact_spearman_perm` — exact permutation p; also returns `min_abs_rho_for_p05 = 0.886` (the true n=6 two-sided critical value). \|−0.83\| < 0.886, so the reported correlation **cannot** reach exact p<0.05. |
| M5 | "cardiac k ≈ 2 directly confirms peaks-per-beat" overstates an r = 0.50, p = 0.17 (n=9) covariation. | Report as agreement of central values (fitted k 1.96 vs median peaks/beat 2.02), never "confirms" — matches `analysis/rates/CLAUDE.md`. |

The consolidated `analysis/rates/CLAUDE.md` already states the correct numbers
under all four regimes; the metrics module lets those numbers be **regenerated
and unit-tested** rather than trusted from a CSV.

---

## D. Recommended manuscript actions (priority order)

1. **Fix C4 first** — reconcile the "50 one-minute windows" text, the packaged
   calibrators, and the whole-night-median pipeline. Right now the Methods
   describe code that produced no reported number.
2. **Report rate accuracy only under held-out calibration**, with the no-sensor
   constant printed alongside (`evaluate_calibration_regimes`). State plainly
   that the sensor does not beat it — a clean negative consistent with the
   epoch-level result.
3. **Retire `rate_spectral` from every headline** (C1); disclose the degeneracy
   in Methods; if a spectral estimate is needed, use `rate_spectral_interp`.
4. **Fix or fence `rate_acf` (C2)** before it appears in any fused number;
   report the operating band where each estimator is valid.
5. **Recompute all small-n correlations with `exact_spearman_perm`** and drop or
   downgrade the k–age narrative (M4) — it cannot reach significance exactly.
6. **Fix `_quality_filter` (C5)** and re-extract ground truth; quantify whether
   any downstream number moves (likely small, but it is the reference).
7. Bland-Altman on per-night values only (M3); within-night r against the
   circular-shift null (M2).

---

## E. Beyond the rate section (transferable)

The same three statistical hazards apply to the SWA, spindle, staging, and
variance-vs-depth results the paper carries: (i) n = 6 → use exact/permutation
tests, not asymptotic p-values (`exact_spearman_perm`); (ii) pooling epochs
across nights inflates n and understates uncertainty — keep the recording as the
unit; (iii) any "X tracks Y over the night" claim needs a time-alignment null
(`circular_shift_null`). I did not re-audit those sections' code here; recommend
the same treatment before they are reported as positive.
