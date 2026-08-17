# Rate detection — full audit of the manuscript claims (2026-08-06)

> **SUPERSEDED for rate results (2026-08-14).** Findings A.1, A.3, B and C are now fixed in the manuscript, and A.2 was confirmed by the 2026-08-14 rerun on a non-degenerate estimator. Retained as the record of how the errors were found. Current numbers and methods: `analysis/rates/CLAUDE.md`.


**Scope.** Everything the manuscript says about rate detection: §3.3 (ground truth), §3.5
(rate estimation + k), §3.7 (statistics), §4.2 and §4.2.1 (Results, Table 3, Figures 3–4),
§5.1 / §5.3 / §5.4 / §5.5 (Discussion), Limitations 2–5, §7 (Conclusion), plus Table 1's
epoch column and the Table S1 / Figure S1–S5 supplementary writeups.

**Audited against**, not against prior review notes: `scripts/run_mask_rate_detection.py`,
`sleep_monitor/rates.py`, the cached per-window predictions in `artifacts/mask_phase_a.parquet`
(93,190 rows), and every CSV in `reports/rates/mask/` and `analysis/rates/outputs/`. Every
number below was recomputed from those sources.

**Two files exist.** `CAP_sleep_mask_manuscript_main.docx` (canonical) and
`..._main_review.docx` (the 2026-07-30 generated review copy, never merged). The canonical
file still carries every error in §A–§C. The review copy fixes most of §B and §C and
discloses the degeneracy; it does **not** fix §A.2, §A.3, or §D.

---

## Verdict

The epoch-level (within-night) negative result is sound and well supported. **The positive
night-level claims are not.** Both headline accuracy numbers are, on inspection, the residual
of a scale factor fitted to the PSG reference on the same recording, and neither beats a
predictor that uses no sensor data at all. Three findings below are blocking; the rest are
reproducibility and wording problems.

---

## A. Blocking

### A.1 The respiratory estimator is a constant. The headline 0.91 br/min contains no sensor information.

`rate_spectral()` (sleep_monitor/rates.py:32) sets `nperseg = max(64, int(fs*4))` = 4 s at
fs = 100 Hz, so Δf = 0.25 Hz. The 0.1–0.5 Hz respiratory band therefore contains exactly two
usable bins, 0.25 and 0.50 Hz. Measured over the cached predictions:

| | value |
|---|---|
| epochs returning exactly 0.25 Hz (= 15.0 br/min) | **9,317 / 9,319 (99.98%)** |
| sessions with zero within-session variance | **10 / 12** (S4N1, S4N2 have 2 distinct values) |

Consequences, all verified numerically:

- **k_resp is not a calibration factor.** k = median(estimate/reference) = 15 / median(reference RR).
  Checked per session: max deviation from that identity is 5×10⁻⁴. So the "calibrated respiratory
  rate" the paper reports is *identically* each session's median reference rate.
- **The reported MAE is the error of the best possible constant.** Recomputing per session:
  paper pipeline 0.912 br/min, best-possible constant predictor (the session's own reference
  median) 0.912 br/min — equal to three decimals in 10 of 12 sessions.
- Every §4.2.1 statement about respiratory k is therefore a statement about the reference,
  not the sensor: "k is reproducible within a subject (median |Δk| 0.013)" = *the subject's
  median breathing rate repeats across nights*. "Respiratory k declines with age" = *median RR
  rises with age in this cohort* (14.5 br/min at 25 y → 16.6 br/min at 66 y). The mechanistic
  gloss in §4.2.1 and §5.5 ("mechanical coupling", "chest-wall compliance", "each breath
  produces one dominant temple displacement", "k ≈ 1 reflects the simpler coupling") is not
  supported by anything in the data and is ruled out by the constancy.
- §3.5's "Respiration required essentially no calibration — a fixed k = 1.0 (the raw spectral
  peak) yielded a median MAE of 1.15 br/min" reads as a deployment strength. It means:
  *predicting 15 br/min for every subject and every epoch gives 1.15 br/min.* No sensor is
  involved. Stating this as a sensor capability is the most serious defensibility problem in
  the section.

§5.3 partially discloses the resolution issue ("frequency resolution 0.25 Hz ... comparable to
the width of the entire band") and §4.2 notes the lower error "is a direct consequence of its
not tracking" — but neither says the output is literally constant or that k is reference-fitted,
so the §4.2 / §5.1 / §7 headline survives unqualified.

**Fix:** the review copy's approach is right — report the non-degenerate fused peak-counting
estimator and demote the spectral estimator to a disclosed baseline. Note this raises the
respiratory error to 0.94 br/min per epoch, which is *still* not better than the constant.

### A.2 Neither band beats a no-sensor baseline once k is held out. (Not fixed in either file.)

k is fitted on the same recording it is evaluated on, so the "night-level" agreement is a
fitting residual, not a measurement. The data supports a genuine held-out test — 6 subjects ×
2 nights — which has not been run. I ran it (CRE / peaks_loose for cardiac, diff / spectral
for respiration; medians over 12 nights):

**Cardiac, night-level error |mean(estimate) − mean(reference)|, BPM**

| calibration | error |
|---|---|
| k from the same night (what the paper reports) | 1.56 |
| k from the subject's *other* night (realistic deployment) | **3.77** |
| population k = 1.96 | 2.60 |
| **no sensor** — predict 61.4 BPM for every night | **2.53** |

**Cardiac, per-epoch MAE, BPM**

| calibration | MAE |
|---|---|
| same-night k | 3.41 |
| other-night k | 4.64 |
| population k | 5.09 (paper reports 4.56 for the fused pipeline) |
| **no sensor** — constant 61.4 BPM | **4.52** |

The paper's own published population-k figure (4.56 BPM, §3.5 and Figure S4) is not better than
predicting a fixed 61 BPM (4.52). Excluding S6 does not change the conclusion (night-level:
population k 2.38 vs no-sensor 2.34). The respiratory column is the same picture and worse
(population k 0.99 vs no-sensor 0.99 — identical, as A.1 implies).

So the defensible statement is: *after per-session calibration against a reference the mask
reproduces the night mean; without that reference it does not improve on a population constant.*
That is a much weaker claim than §4.2, §5.1, §5.4 ("suitable for screening-level overnight
respiratory and cardiac rate monitoring", "longitudinal tracking of resting rates across
nights") and §7 ("reliably recovers mean respiratory rate ... and cardiac rate") currently make.
Longitudinal night-to-night trending in particular requires exactly the cross-night transfer
that fails here (3.77 BPM, worse than no sensor).

### A.3 The reported window geometry is wrong throughout.

`run_mask_rate_detection.py:68,153` — `WIN_SEC = 30.0`, `starts = np.arange(0, n-win_n+1, win_n)`.
Windows are **30 s and non-overlapping**. The manuscript says "60-second windows (30-second
step)" in §3.3, §4.2 and the Figure 3 caption, and Table 1's column header reads "Analysis
epochs (60s)". The epoch counts in Table 1 confirm the code, not the text (S1N1: 7.95 h ÷ 30 s
= 954 ✓). §5.3 and Limitation 5 correctly say 30 s, so the manuscript contradicts itself. This
also matters for §5.3's 0.25 Hz argument, which only follows from the 4-s Welch segment, not
from the window length as written.

---

## B. Numbers that do not reproduce

| Claim (canonical §4.2 / Table 3) | Recomputed | Note |
|---|---|---|
| Resp night-level MAE **0.91** [0.81–1.19], range 0.56–2.26 | median 0.91 ✓, IQR **[0.78–1.09]**, range **0.56–1.34** | **Three sources mixed in one row.** Median + min come from the whole-night-k unsmoothed diff pipeline; q3 1.19 and max 2.26 come from `per_session_summary.csv` (a different k and smoothing, median 0.945); **q1 0.81 appears in no file I can find.** |
| Card night-level MAE **3.41** [3.06–8.38] "best single channel" | ✓ exactly (CRE, whole-night k, unsmoothed) | But §4.2's method paragraph says the CLE−CRE differential and 50-window k with smoothing, one paragraph earlier. Table S1 reports the *fused* pipeline (median 3.36) so Table 3 and Table S1 disagree. |
| Resp k **0.97 [0.91–1.04]** | median **0.96**, IQR **[0.91–1.02]**, range 0.90–1.05 | q1 is right; the median is rounded the wrong way and q3 1.04 matches neither the IQR nor the range. |
| Card k **1.95 [0.94–2.24]** | median 1.95 ✓, IQR **[1.79–1.99]**, range 0.94–2.23 | The bracketed pair is the full range, not the IQR, and both endpoints are S6. |
| "k ... across randomly selected calibration windows. Calibration used 50 randomly drawn one-minute windows ... \|k_diagnostic − k_whole\| ≤ 0.04 for all sessions" (§3.5, §4.2.1, §5.1) | **the description does not match the code** | `run_mask_rate_detection.py:394–400` computes `k_full` as the median of raw/reference over **every valid epoch of the night**, with ratios clipped to 0.3–5.0. There is no 50-random-window calibration anywhere in the pipeline that produced the reported numbers (the only alternative it computes is `k_10min`, the first 20 epochs). The ≤0.04 agreement claim is therefore between the reported k and a diagnostic that no reported number uses; it is inherited from a superseded pipeline. |
| "Error-to-variation ratio 0.77 / 0.78" (Table 3 row) | 0.77 ✓ but from the **spectral** estimator; 0.78 ✓ from **detB** | **Two different estimators in one row.** Like-for-like on detB it is 1.04 vs 0.78. Worse, the respiratory 0.77 is a mathematical identity — MAE of a constant at the median ≈ 0.8 σ for any roughly-normal distribution — so it measures nothing. The review copy correctly deletes this row. |
| Independent-sensor bound r = **0.47**, detrended 0.27 (§4.2, Fig S3) vs r = **+0.48**, 0.28 (§3.3) | median 0.472 / 0.267 | Same quantity quoted with two values in two sections. |
| "nine sessions with usable peak detection ... one session was excluded" (§4.2.1) | 9 usable ✓, but **3 of 12 absent**: S4N1 failed *and* S5N1, S6N2 are not in `peaks_per_beat.csv` at all | The review copy fixes the count ("nine of twelve ... the remaining three"). |
| peaks-per-beat vs k, r = 0.50 | ✓ 0.502 on CRE (0.696 on CLE−CRE) | At n = 9, p = 0.17. The canonical calls this "directly confirming" (§5.1). |
| k–age ρ = −0.83, "uncorrected p = 0.042" | ρ = −0.829 ✓ from the 50-window k; **exact permutation p = 0.058**, not 0.042 | The t-approximation is invalid at n = 6. Also **the result is aggregation-dependent**: using whole-night k instead of 50-window k gives ρ = −0.771, exact p = 0.103. Non-significant either way, before any correction. |
| Reference SD within night 1.14 / 5.26 br/min·BPM | ✓ (`symmetric_tracking_battery.csv`) | The review copy prints 1.57 / 6.58 for the same row — one of the two files is wrong; I could not locate a source for 1.57 / 6.58. |
| All epoch-level rows (r = +0.06 p = 0.68 8/12; r = −0.19 p = 0.34 5/12; MAE 1.33 / 3.65; detrended +0.02 / −0.15) | ✓ all reproduce exactly | |
| Pooled MAE 1.09 / 3.91, bias −0.3 / −0.6, LoA (−4.7, +4.2) / (−24.1, +22.9) | ✓ `final_summary.json` | But n = 9,260 / 9,318 *epochs*, not independent observations; Bland–Altman limits over pooled epochs from 12 nights are not interpretable as agreement limits for a night-level quantity, which the Figure 3 caption implies. |
| Age-prior LOSO k errors 0.020 / 0.056 / 0.056 and 0.387 / 0.305 | ✓ `k_age_prior.csv` | See D.2. |

---

## C. Method-description errors

1. **§4.2 contradicts itself within two paragraphs** — the pipeline paragraph says both bands
   use the CLE−CRE differential; the results paragraph says cardiac is "on the best single
   channel" (it is CRE). Post-hoc channel selection per band is not declared as such.
2. **§3.5 says "six base estimators"** then lists seven (spectral, autocorrelation, Hilbert,
   zero-crossing, peaks_loose, peaks_strict, adaptive).
3. **§3.5 and §4.2 describe the night-level pipeline as smoothed with a causal 3-epoch median
   filter**, but the Table 3 numbers (0.91 resp, 3.41 card) are from unsmoothed whole-night-k
   estimates. Smoothing is inert for respiration anyway (constant input).
4. **Table 3's two "regimes" are not two levels of aggregation** — both rows are the median
   per-epoch absolute error, computed with two different estimators. Labelling the first
   "Night-level" invites the reader to read 0.91 br/min as an error in the night's mean rate.
   The true night-mean errors are 0.14 br/min and 1.60 BPM (`table_s1_review_single_pipeline.csv`);
   the review copy uses these, the canonical does not.
5. **§3.5 calls the population-k comparator "a fixed k = 1.0"** but the CSV uses the population
   mean k (0.989 / 0.995 for resp), and quotes 1.15 br/min from the *fused* row while the
   surrounding text describes the single-channel spectral pipeline (1.23 br/min).
6. **§4.2 says the epoch-level detector uses "five channels"** while §3.5/§3.2 name three
   sensors (CH, CLE, CRE) plus derived combinations; the five are never enumerated.
7. **Limitation 2 says "single-site laboratory study"**; §3.1 says the recordings were made in
   participants' homes.
8. **§3.1 contains six duplicated paragraphs** (two full descriptions of the protocol, with
   conflicting durations: "4.1 to 8.7 h" vs "approximately 9 hours"). Fixed in the review copy.

---

## D. Overclaims and interpretation

1. **§5.1 "The k calibration is remarkably stable ... 3 of 6 subjects showed night-to-night k
   variation of ≤0.03."** Three of six is not stability; and for respiration this is a claim
   about the reproducibility of the subject's breathing rate (A.1).
2. **§4.2.1's LOSO age-prior result.** Fitting a 2-parameter line on 5 subjects and predicting
   the 6th, repeated 6 times, is close to the in-sample fit; with the estimator being constant,
   "predicting k from age" *is* "predicting the subject's mean respiratory rate from age". It is
   not a sensor-calibration result and should not be cited in Limitation 3 as a deployment path.
3. **§5.3's cardiac mechanism** ("the dominant frequency is determined by stable morphology
   rather than instantaneous heart rate") is a plausible story but is asserted, not tested. A
   direct test exists and is cheap: the raw peaks_loose output has a within-session SD of 9.2 BPM
   around a mean of 118.8 — it *does* vary, it just does not vary *with* the reference. The
   morphology account predicts near-constancy; the data show uncorrelated variation. This should
   either be tested or softened.
4. **§5.5's BCG comparison** ("consistent with the BCG literature") is uncited.
5. **§5.4 "well-characterized ... per-stage accuracy"** rests on the Figure S2 writeup, which
   itself says the per-stage comparison pools epochs across sessions and was not tested for
   significance. Drop the word "well-characterized" or add the test.
6. **§7 "reliably recovers"** — see A.2. "Reliably" is not supportable for either band under any
   held-out calibration.

---

## E. What survives, unchanged

- The entire epoch-level negative result: r = +0.06 / −0.19, the Wilcoxon tests, the 8/12 and
  5/12 counts, the detrended correlations, the 200-iteration shuffle null, and the exhaustive
  estimator battery. All reproduce exactly. This is the strongest material in the section.
- The Flow-vs-RIPSum independent-sensor bound (r = 0.47, IQR 0.33–0.67; detrended 0.27) and its
  use to bound what any respiratory sensor can do at this resolution.
- The multi-signal respiratory consensus construction (§3.3) and its jitter reduction
  (2.26 → 1.61 br/min).
- **Cardiac k ≈ 2 as a morphology count.** This is the one genuinely sensor-derived positive
  result in the section: peaks-per-beat 1.70–2.43 (median 2.02 CRE) brackets the fitted k = 1.95
  independently of the rate pipeline. It should be stated as agreement of central values, not as
  covariation (r = 0.50, p = 0.17 at n = 9).
- Fusion adds nothing (ΔMAE < 0.1 both bands) — correctly reported and worth keeping.
- The oracle-headroom asymmetry (resp headroom is in the method, cardiac in the channel).

---

## F. What the section should claim instead

1. Lead with the **cardiac pulse morphology** result (k ≈ 2 = peaks per beat, R-peak-triggered
   waveform). It is sensor-derived, mechanistic, and independently corroborated.
2. Report rate accuracy **only under held-out calibration**, with the no-sensor constant
   predictor printed alongside as the comparator. If the sensor does not beat it — and on the
   present evidence it does not — say so; that is a clean, publishable negative that matches the
   epoch-level negative already in the paper and is consistent with the honest-characterization
   framing of the Introduction and §5.
3. Keep the epoch-level negative in full.
4. Drop the respiratory age-of-k narrative, or restate it as what it is: median respiratory rate
   rises with age in this cohort (ρ = −0.83 on k, exact p = 0.058; ρ = −0.77 / p = 0.10 under a
   different k aggregation), with n = 6 and age confounded with subject identity.
5. Retire the spectral respiratory estimator from all headline numbers, with the degeneracy
   disclosed in Methods.

---

## G. Immediate mechanical actions

- The review copy (`..._main_review.docx`) already fixes B (most), C.1–C.5, C.8 and the exact
  permutation p. **It has never been merged into the canonical file.** Decide that first — every
  item in §B currently sits in the primary manuscript.
- Still wrong in the review copy: (i) its Table 3 k row (0.96 [0.91–1.02], range 0.90–1.05) is
  the *spectral* estimator's k while its stated operational pipeline is fused peak counting
  (k ≈ 1.18 for resp peaks_loose); (ii) reference SD 1.57 / 6.58 has no traceable source
  (battery gives 1.14 / 5.26); (iii) §4.2.1 still frames respiratory k as mechanical coupling
  (A.1); (iv) the ≤0.04 k-stability claim (B); (v) no held-out calibration anywhere (A.2).
- Verification scripts for everything above: see `analysis/rates/` outputs plus the recomputation
  recorded in `notebooks/ANALYSIS_LOG.md` (2026-08-06).
