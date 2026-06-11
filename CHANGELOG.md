# Code Changelog

Records all code changes to library modules, scripts, and notebooks.

---

## 2026-06-11

### SWA Validation
- **Added** `analysis/swa_validation/CLAUDE.md` — SWA validation workspace: Lucey et al. 2019 replication plan, Steps 0-4, deliverables, working rules
- **Added** `analysis/swa_validation/step0_inventory.py` — Step 0 data inventory script: scans all 12 sessions, reports format/channels/rates/alignment/quality

### Slow Wave / Harmonic Detection
- **Updated** `sleep_monitor/harmonics.py` — added `detect_persistent_ridges()`: temporally-continuous ridge tracking with motion masking, fragment merging, harmonic group detection, and continuous harmonic strength scoring
- **Added** `analysis/slow_wave/detect_sws.py` — LOSO N3 binary classifier from CAP features (motion, band power, spectral entropy, rate regularity, DC stability, coherence, harmonic features)
- **Added** `analysis/slow_wave/detect_trials.py` — trial-based SWS exploration: finds sleep segments matching physiological criteria (DC slope, post-movement settling, thorax smoothness)
- **Added** `analysis/slow_wave/plot_settling_events.py` — post-movement settling event visualization: ±15 min windows with stacked panels (accel, CAP, thorax, cardiac/resp rates, PPG, stage)
- **Added** `analysis/slow_wave/plot_trial_signals.py` — full-night raw CAP + PSG time series with detected trial regions highlighted (9 panels + spectrogram)
- **Added** `analysis/slow_wave/run_harmonic_ladders.py` — harmonic ladder detection via concurrent persistent ridges with integer-ratio grouping, prevalence by sleep stage
- **Added** `analysis/slow_wave/run_ridge_overlay.py` — consolidated harmonic ridge overlay: artifact removal, persistent ridges, continuous harmonic score, rich 4-row overlay plots, per-epoch parquet. Run on all 12 sessions; CRE dominant in 9/12.
- **Added** `analysis/slow_wave/NEXT_RIDGE_OVERLAY.md` — pickup spec for v2: high-res spectrogram, 5-min min ridge, median-filtered flat traces, 3-channel stacked layout
- **Added** `analysis/slow_wave/verify_harmonics_overlay.py` — harmonic detection verification: 5 figure types overlaying detected peaks/ridges on spectrograms and PSDs

### Rate Consolidation
- **Added** `scripts/run_rate_consolidation.py` — multi-channel fused rate pipeline (6 phases): method benchmark, channel confidence fusion, CWT ridge tracker, Viterbi smoothing, combined evaluation, k-calibration. Fixed stage assignment and encoding bugs; all 6 phases run to completion. 23 figures + CSV to `writeup/figures/rate_consolidation/` and `reports/rates/`
- **Updated** `writeup/PAPER_TASK.md` — added Rate Consolidation section (done) with figure inventory and key numbers
- **Added** `scripts/generate_rate_consolidation_docx.py` — Word document generation for rate consolidation section
- **Added** `writeup/CAP_rate_consolidation_section.docx` — standalone rate consolidation writeup

### Projections / Staging
- **Added** `scripts/run_clustering_phase3.py` — Phase 3 clustering: GMM (k=3,4,5) + DBSCAN on supervised UMAP embeddings for all 12 sessions
- **Added** `scripts/run_pooled_phase4.py` — Phase 4 pooled cross-session projections: subject-level z-score, pooled PCA/UMAP/t-SNE, GMM k=4, LOSO evaluation
- **Added** `scripts/run_supervised_validation.py` — supervised UMAP train/test validation: train fraction sweep (25/50/75%), GMM predict on held-out embeddings
- **Added** `scripts/run_supervised_validation_v2.py` — supervised UMAP validation v2: kNN + Random Forest in raw vs UMAP space, separating embedding value from classification ability

### Signal Validation & Paper
- **Added** `scripts/paper_signal_validation_figures.py` — paper-ready signal validation figures (4 figs + 1 table): waveform example, frequency agreement, coherence + surrogates, channel comparison
- **Added** `scripts/generate_paper_docx.py` — Word document generator for CAP sleep analysis paper (Signal Validation Methods + Results)
- **Added** `writeup/CAP_sleep_analysis_paper.docx` — main paper document (signal validation section)
- **Added** `writeup/PAPER_TASK.md` — paper writing task: scope, prerequisites, missing figures, data sources, key numbers
- **Added** `writeup/figures/` — paper-ready figures organized by topic: `signal_validation/`, `rate_consolidation/`, `rate_accuracy/`, `harmonics/`, `spectrograms/`, `supervised_umap/`, `gmm_clustering/`, `k_biomarker/`, `sfn_abstract/`

### Tests
- **Added** `tests/` — unit test suite: `test_filters.py`, `test_preprocessing.py`, `test_rates.py` with shared `conftest.py`

### Writeup
- **Added** `writeup/` — paper writeup directory with SFN abstract drafts (V4, V5, updated results v1/v2), XML export, figures

---

## 2026-05-28

- **Added** `analysis/slow_wave/run_ridge_stage3.py` — Stage 3: persistent ridge features vs sleep stage for all 12 sessions x 3 channels. Per-epoch feature extraction, KW and MW-U tests, per-subject analysis, 4 plot types.
- **Added** reports: `stage3_ridge_epochs.parquet`, `stage3_summary.csv`, `stage3_ridge_features_by_stage.png`, `stage3_ridge_features_per_subject.png`, `stage3_n3_vs_rest.png`, 12x `stage3_ridge_timeseries_*.png`

## 2026-05-22

- **Added** `analysis/thorax/CLAUDE.md` — thorax prediction analysis workspace: 6 phases of investigation, scripts/artifacts/plots inventory, conclusions and implications
- **Added** `analysis/thorax/ANALYSIS_LOG.md` — chronological log of all thorax analysis phases (correlation, prediction, CAP-only, ablation, residualization, slow-trends)
- **Updated** `CLAUDE.md` — added thorax to analysis areas listing

- **Added** `analysis/slow_wave/plot_harmonics_s1n1.py` — 3 harmonic visualization plots (full-night traces + hypnogram, stage boxplots, dominant channel breakdown)
- **Added** 3 plots in `notebooks/plots/harmonics/`: `harmonics_fullnight_s1n1.png`, `harmonics_stage_boxplots_s1n1.png`, `harmonics_dominant_channel_s1n1.png`
- **Added** `sleep_monitor/harmonics.py` — harmonic structure detection module (3 methods: HPS, cepstral, explicit f0+harmonics). Sliding-window `detect_harmonics()` and `detect_harmonics_multichannel()` returning per-window DataFrame with f0, n_harmonics, harmonic_energy_ratio, cepstral_prominence, harmonic_decay_rate, dominant_channel
- **Updated** `sleep_monitor/__init__.py` — registered `detect_harmonics`, `detect_harmonics_multichannel` in package exports
- **Added** `CLAUDE.md` — lean root context file (project identity, data paths, package API, workflow rules)
- **Added** `.claudeignore` — excludes artifacts, plots, notebooks, venv from context window
- **Added** `analysis/` workspace structure with scoped CLAUDE.md per analysis area:
  - `analysis/rates/CLAUDE.md` — rate estimation, k-biomarker, validation context
  - `analysis/slow_wave/CLAUDE.md` — SWS detection hypotheses and approach
  - `analysis/staging/CLAUDE.md` — sleep phase classification plan and feature spec
  - `analysis/projections/CLAUDE.md` — PCA, UMAP, t-SNE, DMD, delay embedding inventory
- **Added** `analysis/README.md` — index of analysis workspaces
- **Updated** `analysis/slow_wave/CLAUDE.md` — harmonic structure detection plan (HPS, cepstral, explicit f0+harmonics), 4-stage implementation, running trace spec

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
