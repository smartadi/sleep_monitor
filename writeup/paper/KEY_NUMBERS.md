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

## Mean Rate Accuracy (§3.2)

### Respiratory
- Best method: spectral (any channel; channels equivalent, oracle-ch MAE 1.08 ≈ diff 1.09)
- k_resp ≈ 0.97 [range 0.90–1.05] — near unity, negligible calibration
- Per-session median MAE: **0.91 br/min** [IQR 0.81–1.19], range 0.56–2.26
- ⚠️ Spectral at 30s is a constant predictor (df=0.25 Hz, 1.6 bins across resp band)
- Pooled MAE (mask pipeline): **1.09 br/min**, bias −0.3, LoA [−4.7, +4.2]
- S3 outlier: both nights MAE >1.9 (Thorax paradox, quality-gated in consensus)

### Cardiac
- Best single-channel: peaks_loose/CRE, median MAE **3.41 BPM** [IQR 3.06–8.38]
- Multi-channel agreement fusion: pooled MAE **3.91 BPM**, bias −0.6, LoA [−24.1, +22.9]
- k_cardiac ≈ 1.95 [range 0.94–2.24] — consistent ~2:1 (biphasic pulse)
- Per-session median MAE: **3.36 BPM** [IQR 2.64–6.62], range 2.07–17.96
- S6 anomalous: k_card 1.35/0.94, MAE >8 BPM (different sensor coupling)

### Per-stage
- Resp worst: REM 2.31 br/min
- Cardiac worst: Wake 4.73 BPM

### Oracle headroom
- Resp: channel-oracle 1.08, method-oracle 0.54, full oracle 0.16
- Cardiac: channel-oracle **1.58**, method-oracle 2.55, full oracle 0.51

### k stability
- |k_diag − k_whole| ≤ 0.04 all sessions
- Night-to-night: 3/6 subjects delta_k ≤ 0.03, worst OS002 = 0.19

## Within-Session Tracking — NEGATIVE (§3.3)

### Detector B (responsive tracker)
- Methods: peaks_loose + hilbert, 5-channel mean-fusion, k-calibrated, rolling-median k=3
- Resp pooled MAE: 1.34 br/min; Cardiac pooled MAE: 4.31 BPM

### Tracking battery
- Resp within-session r: median **+0.058**, Wilcoxon p=0.34, **4/12** beat shuffle null 95th
- Cardiac within-session r: median **−0.188**, Wilcoxon p=0.85, **3/12** beat null
- Delta-tracking: resp +0.024, cardiac −0.148
- No transient-vs-steady advantage for either band

### Two operating points
- Spectral: lowest MAE (resp 0.91) but literal constant predictor (within-session r=0.00)
- Detector B: moderate MAE but no significant tracking either

### Achievable ceiling (Flow vs RIPSum)
- Within-session r: median **+0.47** (raw), +0.27 (fluctuations-only)
- Mask resp tracking (0.058) = 12% of ceiling
- GT inter-sensor uncertainty: median diff 0.06 br/min, >1 br/min in 29% of epochs

### 6 methods tested, all fail
- peaks_loose, peaks_strict, hilbert, spectral, CWT ridge, continuous Viterbi ridge
- All within-session r ≈ 0 across channels

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
