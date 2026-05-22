# Thorax Analysis Log

## 2026-05-14 — Phase 1: CAP-Thorax correlation analysis

**Question:** What is the relationship between CAP temple sensor features and thorax respiratory effort across 12 overnight sessions?

**Script:** `scripts/thorax_analysis.py`
**Artifacts:** `artifacts/thorax_cap_epochs.parquet`
**Plots:** `notebooks/plots/thorax_analysis/fullnight_*.png` (12), `corr_heatmap.png`, `scatter_best_corr.png`, `per_session_corr_bars.png`, `corr_by_stage.png`

### Method
- Per-epoch (30s) features: 5 CAP channels (CLE, CRE, CH, CLE-CRE, avg) x 4 stats (raw_mean, raw_std, resp_rms, card_rms) + 6 thorax features (raw_mean, raw_std, resp_rms, resp_p2p, dom_freq, regularity_cov) + 4 accel features
- Correlation analysis: pooled, per-session, per-sleep-stage (Pearson + Spearman)

### Findings
- Moderate pooled correlations between CAP and thorax features exist
- Correlations vary substantially across sessions and sleep stages
- Motion/position is a shared driver of both signals

---

## 2026-05-14 — Phase 2: Predictive models with thorax lags

**Question:** Can we predict thorax_resp_rms from CAP + its own history?

**Script:** `scripts/thorax_predictor.py`
**Artifacts:** `artifacts/thorax_predictor_results.csv`
**Plots:** `notebooks/plots/thorax_analysis/predictor_*.png` (14)

### Models (within-session 70/30 + LOSO)
| Model | Within-session R2 | LOSO R2 |
|-------|-------------------|---------|
| AR-only | ~0.39 | — |
| Ridge | ~0.13 | ~0.08 |
| ARIMAX(2,0,0) | ~0.26 | — |
| Stage-ARIMAX | ~0.19 | — |
| XGBoost + lags | ~0.69 | ~0.51 |

### Findings
- XGBoost dominates due to thorax lag features (autoregressive leakage)
- AR-only baseline shows thorax is highly autocorrelated at 30s resolution
- Cross-subject generalization (LOSO) substantially lower
- This motivates the CAP-only experiment: can we predict without thorax history?

---

## 2026-05-14 — Phase 3: CAP-only prediction (no thorax lags)

**Question:** Can CAP features alone predict thorax_resp_rms without thorax history or PSG labels?

**Script:** `scripts/thorax_predictor_caponly.py`
**Artifacts:** `artifacts/thorax_caponly_epochs.parquet`, `artifacts/thorax_caponly_results.csv`
**Plots:** `notebooks/plots/thorax_analysis/caponly_*.png` (17)

### Feature engineering
Enhanced epoch extraction: 52 per-epoch features (base CAP stats + spectral entropy/power ratios + resp/cardiac rates + CLE-CRE coherence/phase + position/time context) + 25 temporal (lags, rolling, deltas) = 77 total.

### Results (median R2 across 12 sessions)
| Model | Features | Within-session | LOSO |
|-------|----------|----------------|------|
| T0-Ridge | 24 base | 0.144 | 0.133 |
| T1-XGB-Base | 24 + lags | 0.201 | 0.262 |
| T2-XGB-Enh | 77 all | 0.284 | 0.368 |
| T3-XGB-Rec | 77 + pseudo-lags | 0.201 | 0.254 |
| Ref (w/ thorax lags) | 44 | 0.511 | 0.690 |

### Findings
- CAP-only best: LOSO R2=0.368, recovering ~53% of reference
- Feature importance dominated by epoch_frac (13%), cos_t (7%), roll_deg (6%), movement_rms (5%)
- Prediction driven by motion/position/time-of-night, not CAP signal
- Recursive pseudo-lags hurt (error accumulation)

---

## 2026-05-14 — Phase 4: Feature group ablation

**Question:** Which feature groups drive prediction — CAP signal or motion/context?

**Script:** `scripts/_ablation_quick.py`

### Results
| Group | N features | LOSO R2 |
|-------|-----------|---------|
| CAP signal only | 92 | 0.118 |
| Accel only | 4 | 0.279 |
| Context only | 4 | 0.297 |
| All CAP-only | 77 | 0.368 |

### Key finding
**92 CAP signal features (R2=0.118) are outperformed by 4 accelerometer features (R2=0.279).** The "prediction" is motion/position tracking, not physiological signal coupling.

---

## 2026-05-14 — Phase 5: Motion-residualized prediction (direct coupling test)

**Question:** After removing the shared motion confound, do CAP sensors have any direct predictive power on thorax effort?

**Script:** `scripts/thorax_residual_analysis.py`
**Artifacts:** `artifacts/thorax_residual_results.csv`
**Plots:** `notebooks/plots/thorax_analysis/residual_*.png` (16)

### Method
Per-session Ridge regression: accel (4 features) -> thorax_resp_rms. Replace with residual. Same for all 44 CAP features. Then predict residual thorax from residual CAP.

### Motion -> thorax R2 per session
Mean R2 = 0.333 (range 0.170-0.557). Motion/position explains ~33% of thorax variance.

### Results on residualized data (median R2)
| Model | Within-session | LOSO |
|-------|----------------|------|
| Ridge-Resid-Base | 0.007 | -0.015 |
| XGB-Resid-Base | 0.036 | -0.043 |
| XGB-Resid-All | 0.042 | -0.021 |
| XGB-Orig-CAP-Only | 0.036 | -0.030 |

### Key findings
1. After residualizing motion, LOSO R2 ~ 0 (negative = worse than mean prediction)
2. The CAP->thorax R2=0.118 from ablation was entirely motion-mediated
3. Feature importance on residuals: raw_mean (DC level / electrode coupling), not respiratory features
4. **Temple capacitive sensors do not directly measure thorax respiratory effort**

---

## 2026-05-14 — Phase 6: Per-session slow-trend models on residualized data

**Question:** Do CAP features capture any slow-moving trends in residualized thorax effort?

**Script:** `scripts/thorax_residual_persession.py`
**Artifacts:** `artifacts/thorax_residual_persession.csv`
**Plots:** `notebooks/plots/thorax_analysis/persession_*.png` (26)

### Models (within-session 70/30)
| Model | Approach |
|-------|----------|
| OLS-Top5 | Top 5 CAP features by train correlation |
| AR-only | SARIMAX(2,0,0) no exogenous |
| ARIMAX-cap | SARIMAX(2,0,0) + top 3 CAP exogenous |
| SmoothedRidge | 5-epoch rolling mean smoothing |

### AR vs ARIMAX comparison
- AR-only and ARIMAX(cap) produce similar R2 in most sessions
- CAP exogenous provides marginal AIC improvement in a subset of sessions (delta AIC < -2)
- Slow co-moving trends exist but are weak, session-specific, and do not generalize

---

## 2026-05-15 — Residualized timeseries visualization

**Script:** `scripts/thorax_residual_timeseries_plots.py`
**Plots:** `notebooks/plots/thorax_analysis/residual_timeseries/*.png` (12)

### Purpose
Visual comparison of raw vs motion-residualized signals for thorax_resp_rms + CLE/CRE/CH channels. Confirms that residualization removes the slow position-related drifts visible in the raw signals.

---

## Summary: Thorax investigation closed

The thorax prediction line of inquiry is complete. The core question — "can temple CAP sensors replace a thorax belt for respiratory effort measurement?" — is answered: **no, they cannot.** The apparent correlation was a motion/position confound.

This is a negative but informative result. It redirects effort toward what CAP sensors _can_ do: rate estimation (via direct peak/hilbert methods), sleep staging (via spectral/harmonic features), and potentially sleep event detection. The motion features identified here (roll, pitch, movement_rms) are useful inputs for those downstream tasks.
