# Claims Register

Each claim maps to: data → script → figure/table → manuscript section.
Status: VERIFIED | REVISED | STALE | ADDED
**Reconciled 2026-06-18** against consensus GT, tracking-FAIL, LOSO harmonics.

---

## Signal Presence (§3.1)

### CAP carries respiratory band energy (0.1–0.5 Hz)
- **Evidence:** 29–48% of total power, SNR +11 to +27 dB above noise floor. All 12 sessions positive.
- **Figure:** fig5 (spectrograms), fig7 (SNR)
- **Script:** `writeup/figures/signal_validation/generate_band_energy.py`
- **Status:** VERIFIED

### CAP carries cardiac band energy (0.5–3.0 Hz)
- **Evidence:** 8–48% of total power, SNR +4 to +13 dB. All 12 sessions positive.
- **Figure:** fig5, fig7
- **Status:** VERIFIED

### Respiratory coupling confirmed by coherence + surrogates
- **Evidence:** Median coherence 0.31 (avg ch), 0.61 (canonical). 14.7% of epochs exceed surrogate null at p<0.05.
- **Figure:** Fig 3 (coherence + surrogates), Fig 4 (channel comparison)
- **Script:** `scripts/paper_signal_validation_figures.py`
- **Status:** VERIFIED

### Cardiac coupling confirmed
- **Evidence:** Median coherence 0.16 (avg ch), 0.27 (canonical). 9.1% exceed surrogate null.
- **Figure:** Fig 3, Fig 4
- **Status:** VERIFIED

### Respiratory frequency agreement
- **Evidence:** 43% within ±0.05 Hz, median error 0.067 Hz
- **Figure:** Fig 2
- **Status:** VERIFIED

---

## Mean Rate Accuracy (§3.2)

### Accurate mean respiratory rate recovery
- **Evidence:** Per-session median MAE 0.91 br/min [IQR 0.81–1.19]. Pooled 1.09 br/min.
- **Method:** Spectral peak frequency, any channel (channels equivalent)
- **k_resp ≈ 0.97** (near unity — negligible calibration)
- **Figure:** fig18 (MAE heatmap), fig2 (Bland-Altman)
- **Script:** `scripts/run_mask_rate_detection.py`, `scripts/evaluate_symmetric_tracking.py`
- **Data:** `reports/rates/mask/symmetric_tracking_mae_table.csv`, `reports/rates/mask/per_session_summary.csv`
- **Status:** VERIFIED

### Accurate mean cardiac rate recovery
- **Evidence:** Per-session median MAE 3.36 BPM [IQR 2.64–6.62]. Pooled 3.91 BPM.
- **Method:** peaks_loose with k-calibration, CRE best single (3.41 BPM), multi-ch fusion 3.91
- **k_cardiac ≈ 1.95** [range 0.94–2.24] — consistent ~2:1 (biphasic pulse)
- **Figure:** fig18, fig2
- **Status:** VERIFIED

### k-calibration is stable and reproducible
- **Evidence:** |k_diagnostic − k_whole| ≤ 0.04 all sessions. 3/6 subjects delta_k ≤ 0.03 night-to-night.
- **Status:** VERIFIED

### Respiratory headroom is in methods, not channels
- **Evidence:** Oracle-channel 1.08 ≈ diff 1.09. Oracle-method 0.54. Full oracle 0.16.
- **Data:** `reports/rates/mask/oracle_headroom.csv`
- **Status:** VERIFIED

### Cardiac headroom is in channels
- **Evidence:** Oracle-channel 1.58 vs fused 3.91. Win distribution ~even across channels (19–21%).
- **Data:** `reports/rates/mask/oracle_headroom.csv`, `channel_win_card.csv`
- **Status:** VERIFIED

### S6 sessions are anomalous
- **Evidence:** k_card 1.35/0.94 (vs typical 1.9–2.1), cardiac MAE >8 BPM
- **Status:** VERIFIED

---

## Within-Session Tracking — NEGATIVE (§3.3) [ADDED 2026-06-18]

### Within-session rate variation NOT recoverable for respiratory
- **Evidence:** Detector B median r=+0.058, Wilcoxon p=0.34, 4/12 beat shuffle null
- **Figure:** fig19 (tracking bars), fig23 (ceiling)
- **Script:** `scripts/evaluate_symmetric_tracking.py`
- **Data:** `reports/rates/mask/symmetric_tracking_battery.csv`
- **Status:** ADDED — the major negative finding

### Within-session rate variation NOT recoverable for cardiac
- **Evidence:** Detector B median r=−0.188, Wilcoxon p=0.85, 3/12 beat shuffle null. Also: 6 methods × channels all r≈0 (peaks, hilbert, spectral, CWT ridge, Viterbi ridge).
- **Figure:** fig19, fig12, fig13
- **Script:** `scripts/evaluate_symmetric_tracking.py`, `scripts/diagnose_cardiac_tracking.py`
- **Data:** `reports/rates/mask/symmetric_tracking_battery.csv`, `cardiac_tracking_diagnostic.csv`
- **Status:** ADDED

### Two operating points, neither tracks
- **Evidence:** Spectral = low MAE (0.91) / r=0.00 (constant predictor); Detector B = moderate MAE / r not significant.
- **Figure:** fig21 (operating points)
- **Status:** ADDED

### Achievable tracking ceiling is moderate
- **Evidence:** Flow vs RIPSum within-session r=+0.47 (raw), +0.27 (fluctuations-only). Mask resp 0.058 = 12% of ceiling.
- **Figure:** fig23
- **Data:** `reports/rates/mask/symmetric_tracking_ceiling.csv`
- **Status:** ADDED

### Spectral resp winner is a resolution artifact
- **Evidence:** nperseg=400 → df=0.25 Hz; r_spectral = 0.25 Hz in 9317/9319 epochs = constant predictor.
- **Data:** `reports/rates/mask/window_size_sweep.csv`
- **Status:** ADDED — REVISED from "spectral is best for respiratory"

---

## Harmonic Structure (§3.4)

### CAP spectrograms contain structured harmonic ladders
- **Evidence:** Integer-ratio grouping via HPS, cepstral, explicit F0
- **Figure:** paper_overlay_* (12 sessions), spectrograms/S*_spectrogram_ridges.png
- **Status:** VERIFIED

### Ridge features statistically associated with sleep stage
- **Evidence:** KW p<1e-16 for n_groups, min_freq, power, spread. Direction consistent 5–6/6 subjects.
- **Figure:** paper_quantification (6-panel)
- **Data:** `reports/slow_wave/stage3_summary.csv`
- **Status:** VERIFIED

### HER direction is subject-dependent
- **Evidence:** S1/S2 N3-high, S3/S4 N3-low. KW p<1e-23 but direction reversal cancels pooled discrimination.
- **Status:** VERIFIED

### LOSO N3 classification is near-chance [ADDED]
- **Evidence:** Pooled AUC 0.534 [0.421–0.604], mean F1=0.095. Subject-dependent direction cancels discrimination.
- **Figure:** paper_n3_loso
- **Data:** `reports/slow_wave/paper_n3_loso_metrics.csv`
- **Status:** ADDED

### CH dominates harmonic detection, CRE dominates ridges
- **Evidence:** CH 70% windows; CRE best ridge channel 9/12 sessions
- **Status:** VERIFIED

---

## SWA Validation — Negative (§3.5)

### CAP temple sensors show zero SWA correlation with contact EEG
- **Evidence:** r=0.015±0.045, coherence 0.003±0.005, N3 AUC=0.490±0.040
- **Script:** `analysis/swa_validation/run_swa_validation.py`
- **Status:** VERIFIED

### Pipeline validated via EEG self-AUC
- **Evidence:** EEG self-AUC=0.740±0.056
- **Status:** VERIFIED

---

## STALE / DEPRECATED CLAIMS (kept for provenance)

### ~~k_cardiac as independent biomarker~~
- **STALE:** k(t) corr with GT rate = −0.83 → absorbs 1/rate, NOT independent. GT-free proxy corr = −0.06.
- **Revision:** k_cardiac's stage structure is confounded by rate dependence. Not reportable as an independent biomarker without a method that tracks rate first.
- **Original evidence:** KW H=609, p=1.64e-130; SDNN r=−0.251; halflife 1.4 min

### ~~k-scaled respiratory achieves 0.99 br/min~~
- **STALE:** From rate_consolidation pipeline (spectral/diff, different k). Superseded by mask pipeline (0.91 median / 1.09 pooled).

### ~~k-scaled cardiac achieves 3.55 BPM~~
- **STALE:** From rate_consolidation (hilbert/CRE k≈1.66). Superseded by mask pipeline (3.41 median / 3.91 pooled, peaks_loose/CRE k≈1.95).

### ~~Best-of-both pipeline~~
- **STALE:** resp 1.49, cardiac 4.11. Superseded by mask pipeline.

### ~~CWT ridge outperforms all for cardiac (no k)~~
- **STALE:** From rate_consolidation. peaks_loose with k outperforms in final pipeline.

### ~~Multi-channel fusion improves~~
- **REVISED:** Resp: irrelevant (all channels equivalent). Cardiac: modest improvement (4.46→3.91) but oracle headroom (1.58) not realizable (selection-bias, per tracking diagnostic).

---

## Revision Log (2026-06-18 reconciliation)

1. **Rate tracking claims ADDED** — the major negative finding; 6 methods fail for both bands
2. **Spectral-wins-resp REVISED** — now acknowledged as constant predictor at 30s windows
3. **k-biomarker STALE** — not independent of rate (r=−0.83)
4. **Rate numbers updated** — from consolidation pipeline to mask pipeline (different methods, k values)
5. **LOSO N3 classification ADDED** — AUC=0.534, confirming harmonics are significant but weak
6. **Multi-channel fusion REVISED** — cardiac oracle not realizable per tracking diagnostic
7. **Consensus GT** — resp GT now multi-signal; ceiling r=0.47 established
