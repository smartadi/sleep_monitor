# Projection Methods Analysis

## Status
Active. **All 12 sessions processed**, 36 pure-CAP features (no k-calibration).
Supervised UMAP consistently best (5-class sil 0.27–0.69, median ~0.45).
Next: Phase 3 GMM/DBSCAN clustering, Phase 4 cross-session pooling.
Scripts: `scripts/run_projections_v2.py` (main), `scripts/run_dsub_projections.py` (legacy v1).
Outputs → `reports/projections/<session>/`

Dimensionality reduction and manifold learning on CAP sensor features for sleep stage visualization, cluster discovery, and characterization of sleep structure.

## Methods in use

| Method | Purpose | Key files |
|--------|---------|-----------|
| **PCA** | Variance analysis, feature loadings, scree plots | `run_projections_v2.py` |
| **UMAP** | 3D nonlinear manifold embedding (nn=15,30,50) | `run_projections_v2.py` |
| **Supervised UMAP** | Stage-guided 3D embedding (nn=15,30) | `run_projections_v2.py` |
| **t-SNE** | 3D probabilistic embedding (perp=15,30,50) | `run_projections_v2.py` |
| **PHATE** | Trajectory-preserving biological embedding (knn=10,30) | `run_projections_v2.py` |
| **DMD** | Dynamic mode decomposition for cardiac frequency | `analysis_dmd_cardiac.py` (legacy) |
| **Delay-PCA** | Takens delay embedding + SVD for cardiac trajectory | `analysis_delay_pca_cardiac.py` (legacy) |

## Feature set (36 pure-CAP features, 60s window, 30s step)
All features computed from raw CLE, CRE, CH, accelerometer — NO k-calibration or PSG-derived measures.

| Group | Features | Count |
|-------|----------|-------|
| **Per-channel band powers** | infra_slow, SO, delta_low, delta_high for diff/CLE/CRE/CH | 16 |
| **Diff signal stats** | spectral_entropy, RMS, kurtosis, skewness, Hjorth mobility/complexity | 6 |
| **Respiratory band** | band_power, peak_freq, rate_acf, SNR, amp_cv, breath_interval_cv | 6 |
| **Cardiac band** | band_power, peak_freq, rate_acf, SNR | 4 |
| **Cross-channel** | CLE-CRE coherence (resp band), CLE-CRE coherence (cardiac band) | 2 |
| **Accelerometer** | acc_rms, acc_resp_power | 2 |

Coloring variables (not input features): stage, time, thorax RMS, apnea, resp/cardiac power.

## Current results (v2, all 12 sessions)

| Session | Windows | Best Method | 5-class sil | 3-class sil |
|---------|---------|-------------|-------------|-------------|
| S1N1 | 953 | UMAP_sup_nn30 | 0.325 | -0.080 |
| S1N2 | 915 | UMAP_sup_nn30 | 0.452 | -0.010 |
| S2N1 | 927 | UMAP_sup_nn30 | 0.267 | 0.210 |
| S2N2 | 811 | UMAP_sup_nn30 | 0.435 | 0.221 |
| S3N1 | 830 | UMAP_sup_nn30 | 0.539 | 0.011 |
| S3N2 | 1028 | UMAP_sup_nn30 | 0.400 | 0.303 |
| S4N1 | 740 | UMAP_sup_nn30 | 0.490 | 0.397 |
| S4N2 | 721 | UMAP_sup_nn30 | 0.692 | 0.467 |
| S5N1 | 492 | UMAP_sup_nn30 | 0.559 | 0.160 |
| S5N2 | 568 | UMAP_sup_nn30 | 0.666 | 0.437 |
| S6N1 | 614 | UMAP_sup_nn30 | -0.022 | 0.105 |
| S6N2 | 693 | UMAP_sup_nn15 | 0.025 | -0.025 |

Key observations:
- **Supervised UMAP** dominates across all sessions — stage labels guide structure
- **S4N2, S5N2** best separation (sil>0.65) — cleaner signals, stronger stage contrast
- **S6N1, S6N2** poor separation — possible signal quality or atypical sleep architecture
- Top Kruskal-Wallis features: acc_rms, breath_interval_cv, diff_hjorth_mobility/complexity, diff_SO
- Thorax RMS correlates most with diff_hjorth_complexity, resp_snr (~0.45)

## Outputs per session (reports/projections/<session>/)
- 7 interactive HTML per key method (stage, time, apnea, thorax, resp_power, cardiac_power, trajectory)
- 3 static PNGs (UMAP, supervised UMAP, PHATE panels)
- Feature boxplots, PCA scree, PCA loadings (PNG)
- Features CSV, silhouette scores CSV

## Previous results (v1, 4 sessions only)
- 15 features including k-scaled resp/cardiac rates
- PCA + UMAP + t-SNE only (no supervised UMAP, no PHATE)
- sil_3class ~0.35-0.45 (v1 used 3-class as primary metric)

## Scripts
- `scripts/run_projections_v2.py` — **current**: pure-CAP features, all methods, all 12 sessions
- `scripts/run_dsub_projections.py` — legacy v1 (k-dependent features, 4 sessions)

## Plans
- `notebooks/PROJECTION_PLAN.md` — 4-phase roadmap
- `notebooks/SLEEP_PHASE_DETECTION_PLAN.md` — supervised staging from projections

## Key parameters
- **Windows**: 60s window, 30s step
- **UMAP**: n_components=3, n_neighbors={15,30,50}, min_dist=0.1
- **Supervised UMAP**: same + target=stage_codes
- **t-SNE**: n_components=3, perplexity={15,30,50}, max_iter=1000, init='pca'
- **PHATE**: n_components=3, knn={10,30}

## Next steps
1. Phase 3: GMM/DBSCAN clustering on supervised UMAP embeddings, confusion matrix vs PSG
2. Phase 4: pooled cross-session with subject-level z-score normalization
3. Investigate S6 poor separation — signal quality issue or genuine architectural difference?
4. Consider adding temporal features (window-to-window deltas, sleep cycle phase)

## Dependencies
- scikit-learn (PCA, t-SNE, StandardScaler, silhouette_score)
- umap-learn (UMAP, supervised UMAP)
- phate (PHATE)
- plotly (interactive 3D)
- matplotlib (static panels)
