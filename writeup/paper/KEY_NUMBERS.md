# Key Numbers

All quantitative results for the manuscript, grouped by section.
**Reconciled 2026-06-18** against post-consensus GT, tracking-FAIL finding, and LOSO harmonics.

---

## Signal Validation (§3.1)

- Resp band power: 29–48% of total (0–5 Hz), SNR +11 to +27 dB vs 3.5–5 Hz noise floor
- Cardiac band power: 8–48% of total, SNR +4 to +13 dB
- All 12 sessions: positive SNR in both bands
- Resp coherence at GT peak: median 0.31 (avg ch), 0.61 (canonical upper bound)
- Cardiac coherence at GT peak: median 0.16 (avg ch), 0.27 (canonical upper bound)
- Resp freq match: 43% within ±0.05 Hz, median error 0.067 Hz
- Surrogates: 14.7% resp, 9.1% cardiac exceed null at p<0.05 (200 surrogates/epoch, 8242 epochs)

## Rate Accuracy (§4.2) — **rerun 2026-08-14**

Operational estimator both bands: loose peak counting on CRE, per-session k (whole-night
median of estimate ÷ reference, ratios clipped 0.3–5.0), causal 3-epoch median filter.
Epochs 30 s non-overlapping. Source `reports/rates/rerun/per_session.csv`.

| | respiratory | cardiac |
|---|---|---|
| per-epoch \|error\| | **1.79 br/min** [1.65–2.01] | **3.41 BPM** [3.06–8.38] |
| night-mean \|error\| | **0.24 br/min** [0.14–0.34] | **1.56 BPM** [1.22–2.08] |
| night-mean, worst | 1.00 br/min | 7.41 BPM |
| Bland–Altman bias (95% LoA) | −0.10 (−5.85, +5.65) | −0.93 (−25.48, +23.62) |
| k | **1.18** [1.12–1.27] | **1.96** [1.77–2.01] |
| reference SD within night | 1.57 br/min | 6.58 BPM |

### Held-out calibration — nothing beats a no-sensor constant
Night-mean error, paired Wilcoxon vs predicting the LOSO cohort median:

| | self-k | cross-night | population k | no sensor |
|---|---|---|---|---|
| respiratory | 0.24 (p=0.001) | 0.57 (p=0.110) | 0.94 (p=0.301) | 1.20 |
| cardiac | 1.56 (p=0.077) | 3.77 (p=0.850) | 3.19 (p=0.622) | 2.76 |

Epoch level is worse than no-sensor in every regime. Only same-night k wins, and that is a
fitting residual. Not reported in the main text; the section is framed as a demonstration.

### k
- Cardiac k ≈ 2 is a **morphology count**: R-peak-triggered counting gives 1.70–2.43
  capacitive peaks per beat (median 2.02 CRE, 9 sessions), bracketing k = 1.96. Central
  values agree; the across-subject correlation is r = 0.50 at n = 9, p = 0.17.
- Respiratory k = 1.18 — peak counting registers ~18% more deflections than breaths.
- Night-to-night |Δk|: resp median 0.027 (max 0.178), cardiac 0.146 (max 0.387).
- **k vs age is dead**: resp ρ = −0.37 exact p = 0.50 (LOO sign flips), cardiac ρ = +0.26
  p = 0.66; held-out age prior loses to a single constant in both bands. |ρ| ≥ 0.83 needed
  at n = 6.

## Within-Session Tracking — NEGATIVE (§4.2)

- Within-night r: resp median **−0.03** (p = 0.34, 5/12 positive), cardiac **−0.08**
  (p = 0.30, 5/12). 0/12 and 1/12 nights above a circular-shift null.
- Robust across every estimator, channel and fusion strategy tested.
- Achievable ceiling (Flow vs RIPSum): r = **0.47** [0.33–0.67] raw, 0.27 detrended.
- GT inter-sensor uncertainty: median diff 0.06 br/min, >1 br/min in 29% of epochs.

### The degeneracy that caused the old numbers
`rate_spectral` fixes nperseg at 4 s → Δf = 0.25 Hz. Respiratory band = 2 usable bins;
returns 15.0 br/min in **99.98%** of epochs, so k ≡ 15/median(reference) and the old
**0.91 br/min** was the error of the best constant. Cardiac quantized to 15 BPM steps. A
30 s epoch gives Δf = 2.0 br/min at best against a within-night SD of 1.57, so 30 s cannot
support spectral respiratory estimation at all.

## Harmonics (§3.4)

- KW across stages: p < 1e-16 (n_groups, min_freq, power, spread)
- Ridge feature direction (N3 vs other): fewer groups, lower min_freq (0.25 vs 0.88 Hz), less power, less spread
- Direction consistent: 5–6/6 subjects per feature
- HER direction: subject-dependent (S1/S2 N3-high, S3/S4 N3-low)
- CH dominance: 70% windows; CRE dominant ridge channel: 9/12 sessions

### LOSO N3 classification (RF, 4 ridge features)
- Pooled AUC: **0.534** [per-subject 0.421–0.604]
- Mean AUC: 0.509
- Mean F1: 0.095
- Best single-feature: prominence_score AUC=0.563

## SWA Validation — Negative (§3.5)

- CAP vs EEG SWA: r = 0.015 ± 0.045, coherence 0.003 ± 0.005
- CAP N3 AUC: **0.490 ± 0.040** (chance)
- EEG self-AUC: **0.740 ± 0.056** (pipeline validation)
- firls bug: fixed (replaced with firwin)
- Visual inspection (12 sessions): no co-movement between CAP power ratio and EEG delta

## DEPRECATED — numbers below from pre-consensus pipelines, kept for reference

### k-Biomarker (§3.5 old — CONFOUNDED)
- Stage-wise KW: H=609, p=1.64e-130 — but k(t) corr with GT rate = −0.83 (NOT independent)
- GT-free proxy corr = −0.06 (not recoverable in deployment)
- Halflife: 1.4 min (cardiac) vs 0.5 min (resp) — still valid but interpretation unclear
- Spearman (Bonferroni): SDNN r=−0.251, EEG delta r=−0.158, acc RMS r=+0.159 — confounded by rate

### Rate Consolidation Pipeline (superseded by mask pipeline)
- Best resp: spectral/diff, MAE 0.99 br/min, k≈0.98
- Best cardiac: hilbert/CRE, MAE 3.55 BPM, k≈1.66
- Best-of-both: resp 1.49, cardiac 4.11; LOSO: 1.95/5.41
- Multi-ch fusion: resp 1.82 (vs 1.90), cardiac 17.74 (vs 21.22)
