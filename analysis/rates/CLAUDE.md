# Rate Analysis

Estimate respiratory and cardiac rates from CAP temple sensors, validated against PSG ground truth.

## Current best results
- **Respiratory:** `rate_peaks_scaled_resp / k` — MAE 2.20 br/min (k=1.31, range 1.18-1.61)
- **Cardiac:** `rate_hilbert_scaled_cardiac / k` — MAE 4.19 BPM (k=1.67, range 1.48-1.93)
- k is per-session, calibrated from 50 random 1-min windows against GT

## Key findings
- k is NOT 2.0 — continuous, physiologically meaningful, varies by subject
- k_cardiac tracks sleep stage (N1=1.71 > N2/N3=1.65 > Wake=1.61 > REM=1.58, p=10^-130)
- k_resp is primarily a quality indicator (correlates with motion r=0.29)
- CLE-CRE channel is cleanest (OLS regression differential)
- ACF method broken for cardiac (sub-harmonic lock-in, MAE 15.81)

## Package modules
- `sleep_monitor/rates.py` — 6 base estimators (spectral, acf, hilbert, zerocross, peaks, adaptive_peaks) + scaled variants
- `sleep_monitor/rates_classical.py` — VMD, CWT, STFT ridge, MUSIC (experimental)
- `sleep_monitor/rates_ml.py` — XGBoost fusion, denoising autoencoder, spectrogram CNN
- `sleep_monitor/ground_truth.py` — ECG R-peaks, Flow peaks via neurokit2
- `sleep_monitor/quality.py` — per-window quality scoring
- `sleep_monitor/morphology.py` — event-based multi-peak detection (double-peak resp, triple-peak cardiac)
- `sleep_monitor/evaluate.py` — pipeline orchestration, per-window DataFrames
- `sleep_monitor/classifier.py` — rate-prediction classifiers (Ridge, RF, HGB, MLP), LOSO CV
- `sleep_monitor/metrics.py` — MAE, RMSE, Pearson r, bias, Bland-Altman

## Scripts
- `scripts/compute_rates.py` — batch rate computation across sessions
- `scripts/run_validation.py` — full validation pipeline
- `scripts/rate_accuracy_analysis.py` — accuracy analysis + docx report
- `scripts/sweep.py` — grid search over estimators/channels/preprocessing
- `scripts/run_rate_detection.py` — event-based morphological rate pipeline
- `scripts/cap_rates.py` — CLI rate utility
- `scripts/signal_validation.py` — 4-level signal validation (coherence, surrogates)
- `scripts/merge_validation.py` — merge validation parquets
- `scripts/plot_validation_report.py` — publication-quality validation figures
- `scripts/cardiac_coherence_test.py` — cardiac coherence analysis
- `scripts/validation_breath_rate.py`, `validation_laydown_rates.py`, `validation_peak_analysis.py` — specialized validation

## Notebooks
- `notebooks/analysis_k_biomarker.py` — k(t) timeseries + characterization
- `notebooks/analysis_k_biomarker_phase3.py` — k(t) vs sleep stage, HRV, delta power
- `notebooks/analysis_morphology.py` — event-based rate detection

## Key artifacts
- `artifacts/hilbert_scaled_per_session.csv` — per-session cardiac accuracy
- `artifacts/peak_ratio_per_session.csv` — per-session respiratory accuracy
- `artifacts/k_biomarker_summary.csv` — k mean/std per session
- `artifacts/k_biomarker_correlations.csv` — k vs physiology correlations
- `artifacts/validation_windows.parquet` — per-epoch (30s) CAP vs PSG rates
- `artifacts/signal_validation.parquet` — coherence + surrogate test results

## Next steps (Phase 7)
1. Run best methods on all 12 sessions with formal metrics
2. Bland-Altman analysis per sleep stage
3. Statistical tests (paired t-test, Wilcoxon) vs baseline methods
4. S6N2 anomaly investigation (k_cardiac=0.79, Hilbert undercounting)
5. Publication-ready figures
