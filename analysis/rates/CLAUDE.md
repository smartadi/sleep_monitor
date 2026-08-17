# Rate Analysis

Respiratory and cardiac rate from the CAP temple sensors, against PSG.

**Consolidated 2026-08-14** after a code review and full rerun. Everything below is
recomputed from `artifacts/rate_rerun_phase_a.parquet`. Earlier numbers in this repo —
including previous versions of this file — are superseded; see *Superseded* at the bottom
before trusting any rate figure you find elsewhere.

## Current pipeline

`analysis/rates/rerun_rate_detection.py` — the current evaluation. Recomputes every
per-epoch estimate and scores it under four calibration regimes.

Operational estimator, **both bands**: loose peak counting (`rate_peaks`,
`prom_factor=0.05`) on the **CRE** channel, rescaled by a per-session `k`, smoothed with a
causal three-epoch median filter. Epochs are **30 s, non-overlapping**.

`scripts/run_mask_rate_detection.py` is the older pipeline that produced the published
numbers. Do not take new numbers from it: its respiratory path fuses the degenerate
spectral estimator (below) into `fused_agree`, so that output has a pooled SD of
0.46 br/min against a reference SD of 1.57 — a diluted constant.

## Results

| | respiratory | cardiac |
|---|---|---|
| per-epoch \|error\|, median [IQR] | **1.79 br/min** [1.65–2.01] | **3.41 BPM** [3.06–8.38] |
| night-mean \|error\|, median [IQR] | **0.24 br/min** [0.14–0.34] | **1.56 BPM** [1.22–2.08] |
| night-mean, worst night | 1.00 br/min | 7.41 BPM |
| Bland–Altman bias (95% LoA) | −0.10 (−5.85, +5.65) | −0.93 (−25.48, +23.62) |
| k, median [IQR] | **1.18** [1.12–1.27] | **1.96** [1.77–2.01] |
| within-night r, median | −0.03 (p = 0.34) | −0.08 (p = 0.30) |
| nights with r > 0 | 5 / 12 | 5 / 12 |
| reference SD within night | 1.57 br/min | 6.58 BPM |

All of the above use k fitted on the night being scored. **Nothing survives held-out
calibration** (`reports/rates/rerun/per_session.csv`), against a comparator that predicts
the leave-one-subject-out cohort median with no sensor input:

| night-mean error | self-k | cross-night k | population k | no sensor |
|---|---|---|---|---|
| respiratory | 0.24 (p = 0.001) | 0.57 (p = 0.110) | 0.94 (p = 0.301) | 1.20 |
| cardiac | 1.56 (p = 0.077) | 3.77 (p = 0.850) | 3.19 (p = 0.622) | 2.76 |

At epoch level every regime is *worse* than no sensor. Paired Wilcoxon on 12 nights.
Only same-night calibration beats the baseline, and that is a fitting residual.

## The two findings that hold

1. **Cardiac k ≈ 2 is a morphology count.** R-peak-triggered counting gives 1.70–2.43
   capacitive peaks per heartbeat (median 2.02 on CRE, 9 usable sessions), bracketing the
   fitted k = 1.96. Agreement of central values only — the across-subject correlation is
   r = 0.50 at n = 9, p = 0.17. Never write "directly confirms".
2. **Within-night rate variation is not recovered by any rate estimator.** Median r
   −0.03 / −0.08, 0/12 and 1/12 nights above a circular-shift null, robust across every
   estimator, channel and fusion strategy tested.

   **Qualified 2026-08-17.** The information is nonetheless partly present.
   `analysis/rates/rate_decoder.py` predicts the reference rate per epoch from 56
   spectral-shape features of the capacitive channels (blocked CV within night, scored
   against the same circular-shift null): median r **+0.245** respiratory with **10/12**
   nights beating the null (p = 0.0002) and **+0.271** cardiac with 7/12 (p = 0.002). A
   linear model reaches only +0.14, so part of the mapping is nonlinear. This is
   within-night calibration — trained on other blocks of the same night, like same-night k
   — so it shows the epoch carries rate information, not that a deployable decoder exists.
   Write the negative as a property of the estimator family, not of the signal.

## Traps

- **`rate_spectral` is degenerate for respiration.** `nperseg = max(64, int(fs*4))` fixes
  the Welch segment at 4 s, so Δf = 0.25 Hz whatever the band. The 0.1–0.5 Hz band holds
  two usable bins and the estimator returns **15.0 br/min in 99.98% of epochs**; k is then
  identically `15 / median(reference)`, so its "calibrated rate" is the session's own
  median reference rate and its 0.91 br/min error is the error of the best constant. The
  cardiac band is quantized by the same Δf to 15 BPM steps. Use `rate_spectral_interp`
  (full window, zero-padded, interpolated peak) if you need a spectral estimate — but note
  it is *less* accurate (3.9–4.4 br/min), because the respiratory rhythm is often not the
  largest peak in the band. A 30 s epoch gives Δf = 2.0 br/min at best, against a
  within-night SD of 1.57, so **30 s cannot support spectral respiratory estimation** at
  all. That is a constraint, not a bug.
- **k is session-wide**, not window-based: the median of estimate ÷ reference across every
  valid epoch of the night, ratios clipped to 0.3–5.0. The "50 randomly drawn one-minute
  windows" that older docs and manuscript drafts describe **does not exist in any pipeline
  that produced a reported number**. A genuine per-epoch k lives in
  `reports/rates/mask/k_biomarker_perwindow.csv` (`k_gt`) and belongs to the separate
  k(t)-biomarker work.
- **k vs age is dead.** ρ = −0.37 (exact p = 0.50) respiratory, +0.26 (p = 0.66) cardiac;
  leave-one-out flips the respiratory sign; a held-out age prior predicts k worse than a
  single constant. At n = 6 you need **|ρ| ≥ 0.83** for p < 0.05. The published ρ = −0.83
  was an artifact of the constant estimator. `analysis/rates/k_age_rerun.py` supersedes
  `k_vs_age.py`, `k_age_prior.py` and `k_age_fused.py` — each used a different k series.
- **Respiratory k is 1.18, not ≈1.** Peak counting registers ~18% more deflections than
  breaths, so "one dominant temple displacement per breath" is wrong.
- Pooled-epoch statistics over 12 nights are not independent observations. Report per-night
  values and treat the recording as the unit of analysis.

## Package modules

| module | role |
|---|---|
| `sleep_monitor/rates.py` | 8 estimators: spectral, **spectral_interp**, acf, hilbert, zerocross, peaks, envelope, adaptive_peaks (+ scaled variants) |
| `sleep_monitor/ground_truth.py` | ECG R-peaks; multi-sensor respiratory consensus |
| `sleep_monitor/quality.py` | per-window SNR / spectral concentration / ACF prominence |
| `sleep_monitor/preprocessing.py` | accelerometer artifact removal (OLS, NLMS), bandpass |

## Scripts and outputs

| what | where |
|---|---|
| current rerun | `analysis/rates/rerun_rate_detection.py` → `reports/rates/rerun/` |
| figures | `analysis/rates/rerun_figures.py`, `rerun_session_plots.py` → `writeup/figures/rate_rerun/` |
| k vs age | `analysis/rates/k_age_rerun.py` → `analysis/rates/outputs/k_age_rerun.csv` |
| per-epoch estimates | `artifacts/rate_rerun_phase_a.parquet` (93,190 rows) |
| audit that started this | `writeup/edits/RATE_AUDIT_2026-08-06.md` |

Ground truth: respiratory = median across nasal flow, thorax, abdomen and RIP-sum with a
per-session quality gate; cardiac = ECG R-peaks, PPG fallback for S6N2 (ECG dead).
Flow-vs-RIPSum agreement bounds what any respiratory sensor can do at this resolution:
r = 0.47 [0.33–0.67] raw, 0.27 detrended.

## Superseded — do not quote

- **Previous versions of this file** (resp MAE 2.20 br/min at k = 1.31, cardiac 4.19 BPM at
  k = 1.67; "k_cardiac tracks sleep stage"; "k_resp is a quality indicator"). June pipeline.
- **resp 0.91 br/min** — the constant estimator. **resp k 0.96–0.99** — same cause.
- **night-level 0.14 br/min / 1.60 BPM** — computed on the fused pipeline, since replaced.
- **k–age ρ = −0.83 / −0.81** and every "age prior beats a constant" claim.
- `writeup/paper/{KEY_NUMBERS,CLAIMS}.md` rate sections predating 2026-08-14, and the
  writeups in `notebooks/{k_biomarker,peak_ratio_method,validation_results,validation_methods}*.md`.
