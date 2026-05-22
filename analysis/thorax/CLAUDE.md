# Thorax Prediction Analysis

Can CAP temple sensors predict thorax respiratory effort (thorax_resp_rms) without a thorax belt?

## Conclusion (established)

**No.** After removing the shared motion/position confound, CAP features have zero cross-subject predictive power on thorax effort (LOSO R2 ~ 0). The apparent correlation (pooled R2 ~ 0.37) was entirely motion-mediated. Temple capacitive sensors do not directly measure thorax respiratory effort.

## What was tested

### Phase 1 — Correlation analysis
- Per-epoch (30s) feature extraction: 5 CAP channels x 4 stats + 6 thorax features + 4 accel features
- Pooled, per-session, and per-sleep-stage Pearson/Spearman correlations
- Full-night stacked-panel plots (12 sessions)

### Phase 2 — Predictive models (with thorax lags)
- 5 model tiers: AR-only, Ridge, ARIMAX(2,0,0), Stage-ARIMAX, XGBoost
- Within-session 70/30 temporal split + LOSO cross-validation
- XGBoost with lag features achieved best within-session R2 ~ 0.69, but relied on thorax autoregression

### Phase 3 — CAP-only prediction (no thorax lags)
- Enhanced features: 52 per-epoch (base + spectral + rate + cross-channel + context) + 25 temporal = 77 total
- 4 tiers: Ridge, XGBoost-Base, XGBoost-Enhanced, XGBoost-Recursive
- Best: T2-XGB-Enhanced, LOSO R2 = 0.368, recovering ~53% of reference with thorax lags
- Feature importance dominated by motion/position/time, not CAP signal

### Phase 4 — Feature group ablation
- 92 CAP signal features: LOSO R2 = 0.118
- 4 accelerometer features alone: LOSO R2 = 0.279
- Motion/position outperforms CAP signal features

### Phase 5 — Motion residualization (direct coupling test)
- Per-session Ridge: accel -> thorax_resp_rms, replace with residual
- Same residualization applied to all CAP features
- Motion explains ~33% of thorax variance (R2 range 0.17-0.56)
- After residualization: LOSO R2 ~ 0, within-session R2 ~ 0.04
- The CAP->thorax R2=0.118 from ablation was entirely motion-mediated

### Phase 6 — Per-session slow-trend models
- OLS, AR-only, ARIMAX(cap), Smoothed Ridge on residualized data
- AR vs ARIMAX comparison: CAP exogenous features provide marginal improvement in some sessions (delta AIC < -2 in subset)
- Slow co-moving trends exist but are weak and session-specific

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/thorax_analysis.py` | Phase 1: correlation analysis + full-night plots |
| `scripts/thorax_predictor.py` | Phase 2: 5-tier predictor with thorax lags |
| `scripts/thorax_predictor_caponly.py` | Phase 3: CAP-only predictor (4 tiers) |
| `scripts/_ablation_quick.py` | Phase 4: feature group ablation |
| `scripts/thorax_residual_analysis.py` | Phase 5: motion-residualized prediction |
| `scripts/thorax_residual_persession.py` | Phase 6: per-session slow-trend models |
| `scripts/thorax_residual_timeseries_plots.py` | Visualization: raw vs residualized timeseries |

## Artifacts

| File | Contents |
|------|----------|
| `artifacts/thorax_cap_epochs.parquet` | Phase 1 epoch table (30s epochs, all sessions) |
| `artifacts/thorax_predictor_results.csv` | Phase 2 model results |
| `artifacts/thorax_caponly_epochs.parquet` | Phase 3 enhanced epoch table (58 columns) |
| `artifacts/thorax_caponly_results.csv` | Phase 3 CAP-only results |
| `artifacts/thorax_residual_results.csv` | Phase 5 residualized results |
| `artifacts/thorax_residual_persession.csv` | Phase 6 per-session results |

## Plots

All in `notebooks/plots/thorax_analysis/`:
- `fullnight_*.png` (12) — full-night stacked panels per session
- `corr_heatmap.png`, `scatter_best_corr.png`, `per_session_corr_bars.png`, `corr_by_stage.png` — correlation summaries
- `predictor_*.png` (14) — Phase 2 model comparison + per-session timeseries
- `caponly_*.png` (17) — Phase 3 CAP-only results
- `residual_*.png` (16) — Phase 5 residualized results
- `persession_*.png` (26) — Phase 6 trend + prediction plots
- `residual_timeseries/*.png` (12) — raw vs residualized overlays

## Implications for other analysis areas
- **Rates:** thorax_resp_rms is not recoverable from CAP; rate estimation must use direct peak/hilbert methods, not effort proxies
- **Staging:** motion features (roll, pitch, movement_rms) are strong confound-free predictors; use them in staging feature sets
- **Slow wave:** CAP low-freq power and thorax effort are not directly coupled; SWS detection should rely on spectral/harmonic features, not thorax correlation
