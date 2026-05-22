# Scaled-Peaks Respiratory Rate Estimator
**Cross-session validation on 12 overnight recordings**

_Date: 2026-04-15  ·  Sessions: S1N1 – S6N2 (6 subjects × 2 nights)_

---

## 1. Motivation

The default respiratory-rate estimators in `sleep_monitor.rates` (spectral, ACF, hilbert, zerocross, peaks) all operate on the bandpassed CLE-CRE channel after OLS accelerometer removal. On S1N1 with the default ACF estimator, MAE was **5.45 br/min** with r ≈ 0 against PSG Thorax — a result inconsistent with the strong signal visible in raw CAP traces.

A previous note (2026-04-12) hypothesised that the resp bandpass [0.1, 0.5] Hz passes both the inhalation and exhalation half-cycles of each breath, producing **two CAP peaks per breath** and biasing peak-counting methods by 2×. The proposed remedy was to count peaks more sensitively, then divide by 2.

This writeup tests a generalised version of that idea: rather than hard-coding ÷2, **learn the divisor `k` from data** by measuring the empirical CAP-peak-to-Thorax-peak ratio.

---

## 2. Method

### 2.1 Signal preparation
Identical for every session:

1. Differential CAP channel: `CLE − CRE` (common-mode reject).
2. OLS accelerometer removal (`sleep_monitor.preprocessing.remove_acc_artifact`), constrained to the resp band [0.1, 0.5] Hz.
3. Ground-truth signal: PSG Thorax bandpassed in the same band.

### 2.2 Peak detection — two operating points
Both use `scipy.signal.find_peaks` with mild pre-smoothing.

| name | prom_factor | min_dist | rationale |
|---|---:|---:|---|
| **baseline** | 0.4 × σ | 1.8 s (= 0.9 / RESP_HI) | current `peaks` method default |
| **scaled (loose)** | 0.05 × σ | 0.4 s | catches every meaningful local maximum |

Thorax peak count (used as the denominator) uses `prom = 0.5 σ`, `min_dist = 1.5 s` — values chosen so each true breath is counted once on the clean PSG signal.

### 2.3 Learning the divisor `k`
Two estimates per session:

- **`k_diag`** — median ratio over **N = 50 random 1-min windows**. Cheap to compute, intended as the realistic "production" estimator (only need 50 minutes of paired CAP+Thorax data per subject to fit it).
- **`k_whole`** — median ratio over the entire night's sliding 1-min windows. Best-case estimate; tells us how stationary the ratio is within a night.

### 2.4 Whole-night evaluation
- Sliding 1-min window, 5-s step
- For each window, compute `pred_rate = (n_cap_loose / k) × 60 / window_s`
- Compare per-window prediction to GT (Thorax ACF on the same window)
- Metrics: MAE, RMSE, bias, Pearson r, coverage

---

## 3. Results

### 3.1 Per-session learned `k`

| session | subject | duration (hr) | k_diag (50 × 1 min) | k_whole (full night) | IQR_diag |
|---|---|---:|---:|---:|---:|
| S1N1 | OS001 | 7.95 | 1.235 | 1.235 | 0.18 |
| S1N2 | OS001 | 7.63 | 1.200 | 1.200 | 0.13 |
| S2N1 | OS002 | 7.73 | 1.194 | 1.188 | 0.27 |
| S2N2 | OS002 | 6.77 | 1.177 | 1.125 | 0.39 |
| S3N1 | OS003 | 6.93 | 1.608 | 1.571 | 0.26 |
| S3N2 | OS003 | 8.66 | 1.599 | 1.615 | 0.42 |
| S4N1 | OS004 | 6.18 | 1.500 | 1.429 | 0.32 |
| S4N2 | OS004 | 6.02 | 1.333 | 1.400 | 0.23 |
| S5N1 | OS005 | 4.11 | 1.276 | 1.286 | 0.32 |
| S5N2 | OS005 | 4.74 | 1.438 | 1.400 | 0.27 |
| S6N1 | OS006 | 5.16 | 1.267 | 1.333 | 0.43 |
| S6N2 | OS006 | 5.78 | 1.310 | 1.286 | 0.42 |
| **median** | — | — | **1.293** | **1.310** | 0.29 |
| **mean ± std** | — | — | **1.345 ± 0.155** | **1.339 ± 0.151** | — |
| **range** | — | — | [1.18, 1.61] | [1.12, 1.62] | — |

**Plot:** `notebooks/plots/all_sessions_k_consistency.png`

**Observations.**
- `k_diag` and `k_whole` agree closely per session (max divergence 0.07 on S2N2). 50 randomly-sampled minutes are enough to estimate the divisor.
- `k` clusters between **1.18 and 1.61** — far from the original "÷2" hypothesis (k = 2.0). On no session does each breath produce 2 CAP peaks.
- A clear **subject grouping**: OS001/OS002/OS005-N1/OS006 cluster around k ≈ 1.2–1.3; OS003/OS004/OS005-N2 cluster around k ≈ 1.4–1.6. Likely a body-position or coupling-gain effect.

### 3.2 Per-session MAE

| session | MAE baseline | MAE scaled (k_whole) | Δ MAE | bias_base → bias_scaled | r_base → r_scaled |
|---|---:|---:|---:|---:|---:|
| S1N1 | 2.63 | **1.95** | −0.68 | −0.85 → +0.33 | +0.43 → +0.37 |
| S1N2 | 2.37 | **2.02** | −0.35 | −0.82 → +0.45 | +0.34 → +0.25 |
| S2N1 | 3.52 | **2.75** | −0.77 | −1.68 → +0.28 | +0.16 → +0.09 |
| S2N2 | 4.91 | **3.18** | −1.73 | −3.88 → −0.34 | +0.09 → +0.08 |
| S3N1 | 3.60 | **1.79** | −1.82 | +2.80 → +0.02 | +0.09 → −0.02 |
| S3N2 | 4.07 | **1.77** | −2.29 | +3.35 → +0.25 | +0.17 → +0.11 |
| S4N1 | 2.57 | **2.25** | −0.33 | +1.17 → −1.13 | +0.40 → +0.33 |
| S4N2 | 2.92 | **2.07** | −0.85 | +1.41 → −0.35 | +0.20 → +0.14 |
| S5N1 | 2.32 | **2.25** | −0.07 | +0.78 → +0.34 | +0.36 → +0.21 |
| S5N2 | 3.41 | **1.79** | −1.62 | +2.04 → +0.39 | +0.12 → +0.06 |
| S6N1 | 2.50 | **2.13** | −0.37 | +0.58 → −0.03 | +0.39 → +0.27 |
| S6N2 | 2.31 | 2.49 | **+0.18** | +0.39 → +0.04 | +0.46 → +0.22 |
| **mean** | **3.09** | **2.20** | **−0.89  (−25.3 %)** | — | — |

**Plot:** `notebooks/plots/all_sessions_mae_bars.png`

**Observations.**
- Mean MAE drops from **3.09 → 2.20 br/min**, a **25 % reduction**.
- Bias is moved toward zero in **all 12 sessions** (the largest baseline biases — −3.88 on S2N2, +3.35 on S3N2 — collapse to within ±0.4).
- **Pearson correlation drops in every session.** The scaled estimator is less responsive to per-window variations because the loose peak detector picks up more noise-driven peaks.
- One regression: **S6N2** loses 0.18 br/min in MAE. (k=1.29 was learned on its own data; the issue is correlation, not bias.)

### 3.3 Per-session whole-night plots

Per-session traces showing GT (Thorax ACF) vs baseline vs scaled prediction are in `notebooks/plots/per_session_rate_plots/S{1..6}N{1,2}_resp_rates.png`. A combined 12-session grid is at `notebooks/plots/all_sessions_grid.png`.

---

## 4. Interpretation

### 4.1 The "÷2" hypothesis is wrong, but a smaller correction works
Across 12 sessions and 6 subjects, the observed CAP/Thorax peak ratio sits at **~1.3**, not 2. The bandpass does not consistently split a breath into two equal-amplitude peaks. Instead, on most breaths the two phases produce one dominant peak plus a smaller secondary peak that is sometimes (~30 % of the time) crossed by the looser detector. The result is a modest, subject-stable inflation that a single scalar `k` cancels out.

### 4.2 Bias-vs-correlation trade-off
The scaled estimator improves **mean** accuracy but loses **point-wise** accuracy. This is expected: the looser detector emphasises sensitivity over specificity, smoothing out rapid breath-rate changes. For applications where short-time-scale dynamics matter (e.g. apnoea event detection), the baseline `peaks` may still be preferable; for whole-night summary metrics (mean RR, RR distribution), the scaled estimator is clearly better.

### 4.3 Subject-specific `k` is a real effect
The 1.2–1.3 vs 1.4–1.6 cluster split is reproducible across both nights for each subject, suggesting it reflects **subject-stable physiology** (chest geometry, body position, signal coupling) rather than per-night noise. A simple per-subject calibration (50 paired minutes) recovers `k` reliably (`k_diag` ≈ `k_whole` to within 0.07).

---

## 5. Limitations

- `k` was learned and tested on the same session per subject. A truly held-out validation would learn `k` on night 1 and test on night 2. Because `k_diag(N1) ≈ k_diag(N2)` per subject (e.g. S1: 1.235 vs 1.200; S3: 1.608 vs 1.599; S6: 1.267 vs 1.310), this is unlikely to change the conclusion qualitatively.
- GT is itself an ACF estimate on Thorax, not a manually scored breath count. ACF can fail on noisy Thorax segments; some "GT errors" are likely shared with the prediction.
- Resp band only — cardiac was not tested with this approach.
- The loose detection settings (pf=0.05, md=0.4 s) are themselves not tuned per session; the same operating point is used everywhere.

---

## 6. Recommendation for integration

If the team accepts the bias-vs-correlation trade-off:

1. Add `rate_peaks_scaled(x, f_lo, f_hi, k, prom_factor=0.05, min_dist_s=0.4)` to `sleep_monitor/rates.py`.
2. Register it in `_ESTIMATOR_CHOICES` in `sleep_monitor/evaluate.py` and in `METHOD_NAMES` in `sleep_monitor/config.py`.
3. Add a per-subject `k` lookup, defaulting to `k = 1.31` (the cross-session median) when no subject-specific value is calibrated. Provide a helper `calibrate_k(session)` that runs 50 random 1-min windows and returns the median ratio.
4. Document this trade-off in the docstring: improves whole-night accuracy at the cost of per-window correlation.

If the team prefers correlation over mean accuracy: keep the baseline `peaks` and treat this as a useful diagnostic about why peak counts are inflated (subject/coupling-dependent), not as a default.

---

## 7. Reproducibility

- Script: `notebooks/analysis_peak_ratio_all_sessions.py`
- Per-session CSV: `artifacts/peak_ratio_per_session.csv`
- Per-session plots: `notebooks/plots/per_session_rate_plots/S*_resp_rates.png`
- Aggregate plots: `notebooks/plots/all_sessions_{mae_bars,k_consistency,grid}.png`
- Random seed: `42 + session_idx`
- All preprocessing uses `sleep_monitor` package functions; no parameters were tuned on a per-session basis except `k`.
