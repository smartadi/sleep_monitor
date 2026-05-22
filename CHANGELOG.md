# Code Changelog

Records all code changes to library modules, scripts, and notebooks.

---

## 2026-05-14

- **Added** `scripts/thorax_residual_analysis.py` — motion-residualized thorax prediction (tests direct CAP→thorax coupling)
- **Added** `scripts/thorax_predictor_caponly.py` — CAP-only thorax resp RMS predictor (4 tiers, no thorax lags, no stage_code)
- **Added** `scripts/_ablation_quick.py` — feature group ablation (CAP signal vs accel vs context)
- **Added** `artifacts/thorax_caponly_epochs.parquet` — enhanced 58-col epoch features (spectral, rate, cross-channel, context)
- **Added** `artifacts/thorax_caponly_results.csv` — CAP-only predictor results (4 tiers × 2 eval modes × 12 sessions)
- **Added** 17 plots in `notebooks/plots/thorax_analysis/caponly_*.png`

- **Added** `LOGGING_POLICY.md` — logging discipline for all code/analysis changes
- **Added** `CHANGELOG.md` — this file
- **Updated** `notebooks/ANALYSIS_LOG.md` — retroactive entries for ~30 previously unlogged scripts/notebooks

## 2026-04-30

- **Added** `notebooks/10_projection_cap12.ipynb` — 12 CAP-only feature 3D projection (UMAP/t-SNE)
- **Changed** `sleep_monitor/config.py` — expanded `APNEA_CODES` with Flow event types
- **Changed** `sleep_monitor/loader.py` — `_parse_flow_file` replaces `_parse_effort_file` for apnea events

## 2026-04-30

- **Added** `notebooks/analysis_k_biomarker.py` — k(t) biomarker Phases 1+2
- **Added** `notebooks/analysis_k_biomarker_phase3.py` — k(t) correlation with PSG biomarkers

## 2026-04-22

- **Added** `sleep_monitor/ground_truth.py` — ECG R-peaks (Pan-Tompkins) + Flow peak detection via neurokit2
- **Changed** `scripts/compute_rates.py` — uses `gt_sliding_rates()`, records GT signal source in metrics
- **Changed** `sleep_monitor/morphology.py` — `compute_rate_divisor()` returns continuous float, default k=1.0
- **Archived** analysis scripts from `notebooks/` to `archive/rate_exploration/`

## 2026-04-16

- **Added** `sleep_monitor/rates.py` — `rate_hilbert_scaled_cardiac`, `rate_peaks_scaled_resp`, `calibrate_k_cardiac`, `calibrate_k_resp`
- **Added** `notebooks/analysis_hilbert_scaled_all_sessions.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/analysis_card_tuned_s1n1.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/demo_scaled_estimators.py` (now in `archive/rate_exploration/`)

## 2026-04-15

- **Added** `notebooks/analysis_default_rates_s1n1.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/analysis_resp_window_methods_s1n1.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/analysis_peak_ratio_sweep_s1n1.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/analysis_peak_ratio_all_sessions.py` (now in `archive/rate_exploration/`)

## 2026-04-12

- **Added** `notebooks/analysis_br_acf_peaks.py` (now in `archive/rate_exploration/`)

## Pre-2026-04-12 (undated, retroactive)

### Infrastructure scripts
- **Added** `scripts/cap_rates.py` — CLI for rate inspection (inspect/rates/metrics modes)
- **Added** `scripts/run_rate_detection.py` — default pipeline across all 12 sessions
- **Added** `scripts/compute_rates.py` — sliding-window rates with new GT
- **Added** `scripts/compute_eeg.py` — EEG band power by sleep stage
- **Added** `scripts/sweep.py` — grid search over (channel, preproc, estimator)
- **Added** `scripts/train_classifier.py` — rate-prediction classifier LOSO CV
- **Added** `scripts/add_psg.py` — one-time notebook code generator (can archive)

### Plotting scripts
- **Added** `scripts/plot_all_sessions_timeseries.py` — full-night + window time series
- **Added** `scripts/plot_best_rates.py` — best rate methods vs GT with stage overlay
- **Added** `scripts/plot_apnea_timeseries.py` — apnea event timelines per session
- **Added** `scripts/plot_apnea_fullnight.py` — 6-row full-night apnea overview

### Validation study scripts
- **Added** `scripts/run_validation.py` — per-epoch rate estimates vs GT, all 12 sessions
- **Added** `scripts/plot_validation.py` — Bland-Altman, scatter, stage boxplots
- **Added** `scripts/generate_validation_docs.py` — DOCX report generation

### Signal validation scripts
- **Added** `scripts/signal_validation.py` — per-epoch spectral/coherence/surrogate tests
- **Added** `scripts/merge_validation.py` — join signal validation with rate validation
- **Added** `scripts/plot_validation_report.py` — signal validation figures
- **Added** `scripts/signal_validation_enhanced.py` — 5 enhanced analyses + 5 channel combos
- **Added** `scripts/signal_validation_delay_pca.py` — delay-embedding PCA for channel combination
- **Added** `scripts/signal_validation_proof.py` — publication-quality signal validation proof
- **Added** `scripts/cardiac_coherence_test.py` — 16-condition factorial cardiac coherence test

### Rate accuracy scripts
- **Added** `scripts/rate_accuracy_analysis.py` — 4-channel rate accuracy, 30s epochs with annotations
- **Added** `scripts/rate_accuracy_docx.py` — DOCX report for rate accuracy

### ICP validation dataset scripts
- **Added** `scripts/load_validation.py` — loader for ICP validation dataset
- **Added** `scripts/validation_breath_rate.py` — breath-rate k per experiment mode
- **Added** `scripts/validation_peak_analysis.py` — phase-by-phase peak/rate analysis
- **Added** `scripts/validation_laydown_rates.py` — layDown-only respiratory/cardiac validation
- **Added** `scripts/plot_validation_rates.py` — best rates on ICP validation dataset

### Analysis notebooks (Python)
- **Added** `notebooks/analysis_morphology.py` — morphological cluster pipeline
- **Added** `notebooks/analysis_pca_stacked_cle_cre.py` — delay-embedded PCA on [CLE, CRE]
- **Added** `notebooks/analysis_delay_pca_cardiac.py` — delay-embedding PCA for cardiac
- **Added** `notebooks/analysis_dmd_cardiac.py` — DMD for cardiac rate
- **Added** `notebooks/analysis_dmd_rank_sweep.py` — DMD embedding dimension sweep
- **Added** `notebooks/analysis_sws_band_ratios.py` — SWS band power ratio analysis

### Jupyter notebooks
- **Added** `notebooks/01_overview.ipynb` — raw signal inspection, hypnograms
- **Added** `notebooks/02_rate_estimation.ipynb` — sliding-window rate visualization
- **Added** `notebooks/03_eeg_analysis.ipynb` — EEG spectrograms by stage
- **Added** `notebooks/04_metrics_summary.ipynb` — cross-subject accuracy tables
- **Added** `notebooks/05_method_search.ipynb` — best pipeline from sweep leaderboard
- **Added** `notebooks/06_classifier_results.ipynb` — rate classifier LOSO results
- **Added** `notebooks/07_validation_loader.ipynb` — ICP validation dataset loader
- **Added** `notebooks/08_cap_sleep_embedding.ipynb` — PCA/t-SNE/UMAP sleep embedding
- **Added** `notebooks/08_validation_loader.ipynb` — duplicate of 07 (validation loader)
- **Added** `notebooks/09_projection_3d.ipynb` — 3D UMAP/t-SNE with 40 features
