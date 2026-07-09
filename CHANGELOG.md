# Code Changelog

Records all code changes to library modules, scripts, and notebooks.

---

## 2026-07-09

- **Added** `analysis/slow_wave/show_ladder_examples.py` — per-window waveform + fine PSD with detected ladder members and k·f0 guides, for eyeballing whether a detection is a real ladder. Surfaced that the integer-ratio member count over-counts (grabs non-contiguous high-order cardiac ridges as respiratory harmonics).
- **Added** `analysis/slow_wave/ladder_spectrogram.py` — CH spectrogram + persistent-ridge overlay + comb-fit ladder quantifier (single session viz). `comb_fit()`: ladder = equally-spaced prominent ridges; searches Δf, scores n_rungs × span-coverage, refines Δf by least squares, flags harmonic (comb through 0) vs inharmonic. Δf search 0.15–1.6 Hz, prominence filter ≥2× floor.
- **Added** `analysis/slow_wave/ladder_quantify.py` — multichannel batch quantifier (CH/CLE/CRE, all 12 sessions). Per-window per-channel comb ladders; a window carries a ladder if any channel detects ≥3 rungs. Band split (respiratory f0<0.6 vs cardiac), harmonic/inharmonic, per-stage prevalence, per-channel comparison.
- **Output** `reports/slow_wave/ladder_quantify/{per_window_channels.parquet, per_window_combined.parquet, summary.csv, channel_ladder_counts.csv}`; per-session `reports/slow_wave/harmonic_rigor/ladder_spectrogram_<session>.png`.
- **Key result** Clean prominent ladders are intermittent (2% of non-motion windows, up to 20% S6N2), correcting the loose 85%. Channel dissociation: CH→cardiac ladder (f0~1.1 Hz), CRE→respiratory ladder (f0~0.3 Hz), CLE negligible. 76% cardiac / 24% respiratory; 63% harmonic / 37% inharmonic.

- **Added** `writeup/figures/cap_swa/make_cap_swa_figure.py` + `fig_cap_swa_definition.png` — publication figure for the CAP-SWA mechanical/autonomic definition (from cached cap_swa outputs, no recompute). Four panels: (A) per-subject swa_score→N3 AUC (6/6 above chance, mean 0.675), (B) HR non-SWA→CAP-SWA per subject (bradycardia), (C) professor-hypotheses table (predicted vs observed direction + per-subject consistency + verdict), (D) threshold sweep showing 0.60 at the F1 maximum.
- **Added** `writeup/edits/cap_swa_section_draft.md` — ready-to-paste manuscript Methods + Results (Part C) + Table + Discussion for the CAP-SWA definition. Honest-negative framing: mask does not measure cortical SWA (§3.5) or spindles (§3.6), but the mechanical/autonomic score marks N3 with consistent cross-subject direction (per-subject AUC 0.675, 6/6); five of six autonomic hypotheses contradicted (HR bradycardia not tachycardia, k-deviation shrinks, PPG/CAP divergence reverses, RR flat, movement does not precede onset), EEG delta up as a weak trend. Per-subject direction counts as evidence per OUTLINE rule 3. Manuscript docx not edited.

- **Added** `analysis/slow_wave/swa_classifier_experiment.py` — CAP-SWA classifier ablation + definition tuning (Workstream C, steps 1–2). Loads the self-consistent `reports/slow_wave/cap_swa/all_epoch_features.parquet`; PART A runs a LOSO N3 ablation (direct-score AUC + logistic + GBM) over nested feature sets (swa_score / swa_subscores / mechanical); PART B sweeps the swa_score threshold (0.40–0.85, sustained-bout vs raw) reporting precision/recall/F1; PART C sweeps the geometric-mean sub-score weighting over the simplex reporting pooled + per-subject N3 AUC. Adds a `direct_score_auc()` helper — a depth-3 GBM on one continuous feature ties its outputs and spuriously deflates AUC toward 0.5, so the composite score is reported directly (tie-free). **Key results:** per-subject `swa_score`→N3 AUC 0.675 ± 0.073 [0.575, 0.759], 6/6 subjects above chance (consistent cross-subject direction, unlike ridge/HER/mean-value); mechanical set 0.692 per-subject; CAP-only (slow-DC + quiescence, no PSG thorax belt) pooled 0.666 / per-subject 0.671 ± 0.042; the 0.60 threshold is F1-optimal (0.281); equal sub-score weighting is within 0.015 of the swept optimum. **Note:** deliberately does NOT fuse with the May-28 `sws_features.parquet` cache — that cache uses an older epoch/stage alignment (~43% stage-label disagreement) that misaligns the join and drove full_cap below chance; a valid fusion requires re-extraction (flagged in ANALYSIS_LOG). Manuscript not edited.
- **Output** `reports/slow_wave/cap_swa/classifier/{loso_ablation.csv, loso_ablation_folds.csv, threshold_sweep.csv, weighting_sweep.csv, classifier_experiment.png}`.

- **Added** `analysis/mean_value/mean_value_vs_stage.py` — professor Results-opener directive ("start with mean value / raw signal changes vs sleep stages"). Per-30s-epoch DC mean + <0.05 Hz VLF baseline for CLE/CRE/CLE-CRE across 12 sessions; per-session z-score, slow-trend-removed and motion-clean variants; Kruskal-Wallis per stage, per-subject direction-consistency check, LOSO AUC; spectrogram+hypnogram+mean-trace overlays. Finding: significant pooled association (KW p up to 1e-25, survives detrending/motion controls) but **subject-dependent in direction** (no unanimous contrast) with near-chance LOSO AUC (0.40–0.57) — not a universal biomarker; suitable only as within-subject descriptive context. Manuscript not edited.
- **Changed** `analysis/mean_value/mean_value_vs_stage.py` — extended mean-value analysis to **all channels + accelerometer**: added CLE+CRE average and acc_mag as mean-value channels (now CLE, CRE, CLE-CRE, CLE+CRE, acc); per-subject direction check and LOSO AUC now looped over every channel; per-channel boxplot + subject-direction heatmap figures. Confirms/strengthens the original verdict: every channel is significant when pooled (KW p 1e-4 → 1e-31) but **all stage contrasts are subject-dependent** across all 5 channels (only exception: CLE+CRE Wake-vs-sleep, 6/6 subjects lower, yet still near-chance LOSO AUC 0.55); LOSO AUC across all channels/contrasts spans 0.29–0.61 (chance). acc mean tracks head posture, not stage (Wake not highest). Stage ordering flips between raw and detrended for CRE/CLE+CRE, confirming overnight drift dominates.
- **Output** `reports/mean_value/{mean_value_epochs.csv, mean_value_stage_stats.csv, mean_value_subject_direction.csv, mean_value_loso_auc.csv}`; `notebooks/plots/mean_value/{spectrogram_hypno_*, meanvalue_hypno_*, boxplot_<ch>[_detr], subject_direction_<ch>}.png` for ch ∈ {CLE, CRE, CLE_CRE, CLEplusCRE, acc}.

- **Changed** `writeup/main/CAP_sleep_mask_manuscript_main.docx` — addressed professor review markup. (1) Fixed section numbering: Results subsections renumbered 3.1–3.5 → 4.1–4.5 (were duplicating Methods), Discussion 4→5, Limitations 5→6, Conclusion 6→7 (headings only; no cross-refs affected — in-text fields are EndNote citations). (2) Removed remaining informal "battery" jargon in prose ("extensive battery"→"suite"; "Tracking battery:"→"Tracking evaluation:"; figure caption "(tracking battery)"→"(tracking evaluation)"), preserving the `symmetric_tracking_battery.csv` filename. Still open: professor's directive to open Results with a raw-signal mean-value vs sleep-stage analysis (analysis in progress). Untouched original backed up in session scratchpad.
- **Added** `analysis/slow_wave/harmonic_rigor.py` — harmonic rigor (Workstream B). Shared `best_ladder()` scorer; per-session per-k surrogate null (random ridge frequencies) with 95th-pct threshold; strong-ladder criterion (≥3 members, ratio_tol 0.06) that beats the null; calibrated confidence (ratio_quality × amplitude-decay-monotonicity); non-ladder window classification (motion/quiet/single-tone/broadband/multi-non-harmonic) cross-tabbed against sleep stage; survival + confidence + otherwise-by-stage summary figure.
- **Output** `reports/slow_wave/harmonic_rigor/{ladder_windows.parquet, null_summary.csv, otherwise_crosstab.csv, harmonic_rigor.png}`.

- **Changed** `sleep_monitor/harmonics.py` — `detect_persistent_ridges()` now stashes the pre-smoothing peak trace (`freq_trace_raw`) and computes per-ridge flatness/consistency metrics (`freq_std`, `freq_cv`, `drift_slope`, `coverage`, `flatness`) in a new Step 5e. Additive keys only; existing consumers unaffected (test suite: 90 pass, 1 pre-existing unrelated preprocessing failure).
- **Added** `analysis/slow_wave/ridge_consolidation.py` — ridge consolidation (Workstream A). Baseline-vs-flat-favoring config comparison (flat-ridge yield + cross-session reproducibility); per-ridge table with flatness + dominant stage; per-window ridge features vs ALL five stages (KW + one-vs-rest AUC + per-subject direction); flatness-by-stage + AUC-heatmap summary figure.
- **Output** `reports/slow_wave/ridge_consolidation/{per_ridge.csv, per_epoch_features.parquet, stage_association.csv, retune_comparison.csv, ridge_consolidation.png}`.

- **Added** `analysis/slow_wave/cap_swa_definition.py` — CAP-SWA operational definition (Workstream C). Per-epoch mechanical/physiology/validation feature extraction; graded SWA score (percentile geometric mean of slow-DC + slow-thorax + quiescence sub-scores); binary sustained-bout candidate label; movement-initiation precursor test with matched-random null (H2); per-subject SWA-vs-non-SWA contrast for the professor's autonomic hypotheses (H4 HR, H5 RR, H6 CAP/thorax k-deviation, H7 PPG-vs-CAP cardiac freq, H8 EEG delta) with Bonferroni; SWA-score N3 AUC.
- **Output** `reports/slow_wave/cap_swa/<session>/epoch_features.csv` + `night_overview.png` (12 sessions), `all_epoch_features.parquet`, `hypothesis_summary.csv`, `movement_initiation.csv`.

## 2026-06-18

- **Added** `scripts/generate_manuscript_docx.py` — generates complete manuscript draft (Methods, Results, Discussion, Limitations, Conclusion + Open Items) as Word document. Embeds 14 key figures, 6 tables, reconciled against all post-consensus source CSVs. Title/Abstract/Intro are placeholders.
- **Output** `writeup/CAP_sleep_mask_manuscript.docx` — 18 MB, 14 embedded figures, 6 data tables, OPEN-ITEMS section listing revised claims and unverified numbers
- **Updated** `writeup/paper/KEY_NUMBERS.md` — reconciled all numbers against post-consensus GT, tracking-FAIL finding, LOSO harmonics. Deprecated pre-consensus k-biomarker and rate_consolidation numbers.
- **Updated** `writeup/paper/CLAIMS.md` — reconciled 28 claims: 7 revised/deprecated (k-biomarker, rate pipeline numbers, multi-channel), 4 added (tracking FAIL, two operating points, ceiling, LOSO N3). Revision log appended.

- **Added** `scripts/evaluate_symmetric_tracking.py` — symmetric resp+cardiac tracking evaluation from cached Phase A data. Detector B (peaks_loose + hilbert, 5-channel mean-fusion, k-calibrated, minimal smoothing) vs spectral baseline. Tracking battery: within-session r, delta-tracking, transient/steady split, 200-iteration temporal-shuffle null. Two operating points framing. Achievable ceiling (Flow vs RIPSum r=0.47).
- **Output** `writeup/figures/mask_rate_detection/fig18_mae_heatmap.png` — multichannel x multimethod MAE heatmap with per-session IQR, resp + cardiac side by side
- **Output** `writeup/figures/mask_rate_detection/fig19_tracking_r_bars.png` — per-session within-session r (DetB + spectral) with shuffle-null 5th-95th bands
- **Output** `writeup/figures/mask_rate_detection/fig20_delta_transient.png` — delta-tracking r + transient vs steady segment analysis
- **Output** `writeup/figures/mask_rate_detection/fig21_operating_points.png` — MAE vs tracking r tradeoff, per session and band
- **Output** `writeup/figures/mask_rate_detection/fig22_fullnight_traces.png` — full-night GT vs DetB vs spectral traces (4 sessions x 2 bands)
- **Output** `writeup/figures/mask_rate_detection/fig23_ceiling_comparison.png` — mask tracking r vs achievable ceiling (Flow vs RIPSum)
- **Output** `reports/rates/mask/symmetric_tracking_{mae_table,battery,ceiling}.csv` — detailed per-session results
- **Output** `artifacts/detB_{resp,card}.parquet` — Detector B fused rate estimates per epoch
- **Key results** RESP: FAIL (r=+0.058, p=0.34, 4/12 beat null). CARD: FAIL (r=-0.188, p=0.85, 3/12 beat null). Ceiling: Flow vs RIPSum r=+0.47. Mask recovers mean rate only.

- **Added** `analysis/slow_wave/paper_ridge_demo.py` — paper-ready harmonic ridge demo: (A) per-session spectrogram + ridge overlay with hypnogram, (B) pooled quantification (violin, heatmap, ROC, KW/MW-U), (C) Stage 4 LOSO N3 classifier (RF, 4 ridge features). Completes all 4 stages of the harmonic structure detection plan.
- **Output** `writeup/figures/harmonics/paper_overlay_*.png` — 12 per-session spectrogram + ridge overlay figures (CRE channel)
- **Output** `writeup/figures/harmonics/paper_quantification.png` — 6-panel pooled quantification figure
- **Output** `writeup/figures/harmonics/paper_n3_loso.png` — LOSO N3 classification results (ROC, metrics, feature importance)
- **Output** `reports/slow_wave/paper_n3_loso_metrics.csv` — per-fold LOSO metrics
- **Key result** Ridge features are statistically significant (KW p<1e-16) but near-chance N3 classifiers (LOSO AUC=0.534, mean F1=0.095). Subject-dependent direction confirmed.
- **Added** `writeup/figures/signal_validation/generate_band_energy.py` — generates fig5/6/7 demonstrating CAP mask energy in resp (0.1–0.5 Hz) and cardiac (0.5–3.0 Hz) bands. Uses `sleep_monitor/spectral.py` Welch PSD + `sleep_monitor/viz.py` for spectrograms. Three panels: annotated spectrograms (3 sessions), band-power time course vs PSG GT rate, in-band SNR summary (boxplots + time course + mean PSD).
- **Output** `writeup/figures/signal_validation/fig5_cap_spectrogram_bands.png` — CLE−CRE spectrograms (0–5 Hz) with resp/cardiac band overlays
- **Output** `writeup/figures/signal_validation/fig6_bandpower_vs_psg_rate.png` — sliding 60s band power vs PSG GT rate, dual-axis
- **Output** `writeup/figures/signal_validation/fig7_inband_snr.png` — in-band SNR boxplots, SNR time course, mean PSD with band annotations
- **Added** `sleep_monitor/ground_truth.py:gt_resp_rate_consensus()` — loads consolidated multi-signal resp GT from `artifacts/consolidated_resp_gt.parquet`, returns consensus rate on any time grid (exact sampling). Module-level cache for repeated calls.
- **Changed** `sleep_monitor/ground_truth.py:gt_sliding_rates()` — new `resp_method=` arg (default `'consensus'`). Uses multi-signal consensus for resp GT; falls back gracefully to Flow→Thorax peak detection when parquet missing or session absent (e.g. validation recordings).
- **Added** `scripts/reattach_consensus_gt.py` — re-attaches consensus resp GT to `artifacts/mask_phase_a.parquet` on the IDENTICAL grid (exact join on session+t_hr, no merge_asof). 46,595/46,595 resp rows matched (9,319 unique epochs × 5 channels), 295 previously-NaN epochs now filled. Cardiac untouched. Median |delta| = 0.06 br/min, 96.5% of epochs changed.

---

## 2026-06-17

- **Added** `scripts/analyze_adaptive_k_and_oracle.py` — cache-only follow-up analysis (no raw reprocessing): self-supervised adaptive k(t), per-epoch oracle headroom (channel/method/full), channel-win distribution. Outputs CSVs in `reports/rates/mask/` + figs 7-9.
- **Added** `CONTINUATION_RATE_DETECTION.md` — handoff doc for next CLI session (cache locations, findings, prioritized next steps)
- **Output** `writeup/figures/mask_rate_detection/fig{7,8,9}*.png` — oracle headroom, adaptive k, channel diversity
- **Key finding** Cardiac channel-diversity oracle = 1.58 BPM (vs fused 3.91) — large untapped headroom; resp headroom is method-diversity not channel (oracle-method 0.54). Self-sup adaptive k fails for cardiac (no good anchor).
- **Added** `scripts/run_mask_rate_detection.py` — paper-ready mask rate detection pipeline (6 phases: raw rates, Smart Fusion + multi-channel SQI, k-calibration + smoothing, evaluation, failure analysis, multi-channel value). Checkpointed at each phase (parquet). Generates 9 paper figures + per-session CSV.
- **Output** `writeup/figures/mask_rate_detection/` — 9 figures (pipeline progression, Bland-Altman, per-stage MAE, time series best/worst × 2 bands, failure analysis, multi-channel value)
- **Output** `reports/rates/mask/` — per-session CSV, final_summary.json, pipeline log
- **Output** `artifacts/mask_phase_{a,b,c}.parquet` — cached intermediate results (93k, 18k, 112k rows)
- **Key results** Resp: MAE=1.09 br/min, bias=-0.3, LoA=[-4.7, 4.2]. Card: MAE=3.91 BPM, bias=-0.6, LoA=[-24.1, 22.9]. Best resp: diff/spectral (k≈0.97). Best card: multi-ch agreement fusion of peaks_loose (k≈1.95).
- **Added** `writeup/paper/` — manuscript scaffold with OUTLINE.md, CLAIMS.md (28 claims with evidence chains), FIGURES.md (25 main + 48 supplementary), TABLES.md (10 tables), KEY_NUMBERS.md, DRAFT.md
- **Added** `writeup/shared/SHARED_METHODS.md` — shared methods text (participants, preprocessing, ground truth)
- **Added** `writeup/README.md` — writeup directory documentation with workflow instructions
- **Superseded** `writeup/PAPER_TASK.md` — monolithic task spec replaced by structured paper/ directory (file kept for reference)

---

## 2026-06-11 (cont.)

### Best-of-Both Rate Pipeline + Updated Documentation
- **Added** `scripts/evaluate_best_pipeline.py` — unified best pipeline: Kalman (reactive, R x0.3, Q x2.0) for resp, hilbert for cardiac, multi-channel quality-weighted fusion, configurable median temporal smoothing (SMOOTH_WIN=7), per-session + LOSO k-calibration
- **Output** `reports/rates/best_pipeline/` — 12 time-series PNGs, 12 Bland-Altman PNGs, aggregate + per-stage plots, 2 CSVs
- Resp: **1.49 br/min** (per-session k), 1.95 (LOSO) — 42% over peaks/k baseline
- Cardiac: **4.11 BPM** (per-session k), 5.41 (LOSO) — 15% over single-channel hilbert/k
- **Updated** `scripts/generate_rate_consolidation_docx.py` — now two-part document: Part 1 (original consolidation) + Part 2 (hybrid pipeline with all 12 session time-series, Bland-Altman, aggregate stats, per-stage, 4 tables)
- **Output** `writeup/CAP_rate_consolidation_section.docx` — 35 figures, 4 tables

### SWA Validation — Lucey et al. 2019 Replication (Steps 0-4)
- **Added** `analysis/swa_validation/swa_pipeline.py` — shared spectral pipeline: FIR bandpass (0.5-40 Hz via `firwin`), 6-sec epoch Welch PSD, band powers (1-4.5 Hz total, sub-bands, 20-30 Hz EMG), relative power normalization, artifact rejection (EMG 97.5th pct + accelerometer)
- **Added** `analysis/swa_validation/run_swa_validation.py` — full pipeline runner: Steps 1-2 (process all 12 sessions, both EEG and CAP), Step 3 (Pearson/Spearman correlation, Bland-Altman, coherence, ROC/AUC for N3 detection), Step 4 (per-subject summary, 5 publication plots)
- **Added** `analysis/swa_validation/step0_inventory.py` — data inventory scanner
- **Fixed** `swa_pipeline.py:bandpass_fir()` — replaced `firls` with `firwin`: `firls` with narrow 0.1 Hz transition band at 0.5 Hz produced catastrophically ill-conditioned coefficients (range ±2794, signal amplification ×26M). `firwin` window method is numerically stable (coefficients ±0.16 to 0.79)
- **Fixed** `sleep_monitor/loader.py:load_session()` — removed `unit='ms', utc=True` from `pd.to_datetime()` call (was causing `time_start` to always be None for datetime strings)
- **Fixed** `sleep_monitor/loader.py:load_sleep_profile()` — complete rewrite: now parses wall-clock timestamps from Sleep Profile epochs and computes offset from CSV `timeSM` start time, handles midnight crossing, drops out-of-range epochs. Previously assumed epoch 0 = CSV time 0, causing up to 38.5 min misalignment
- **Output** `analysis/swa_validation/outputs/` — `swa_validation_results.csv`, `swa_validation_per_subject.csv`, 5 PNGs (swa_overlay, bland_altman, coherence, roc_curves, correlation_scatter)
- **Result**: Negative — CAP temple differential does not measure cortical SWA (r≈0, coherence≈0, N3 AUC≈0.5). EEG sanity check confirms pipeline correctness (AUC=0.740).

### Hybrid Rate Pipeline — Phase 4: Streaming Demo
- **Added** `scripts/demo_realtime_rates.py` — real-time streaming rate tracker demo: `KalmanState` class for lightweight scalar Kalman filter, epoch-by-epoch processing with spectral + adaptive_peaks → Kalman fusion, live console display, summary plot
- **Output** `reports/rates/hybrid_phase4/streaming_demo_S1N1.png` — full-night time series (Kalman vs GT)
- S1N1: 954 epochs in 1.8s (16,348x real-time), resp MAE 1.88 br/min, cardiac MAE 21.29 BPM

### Hybrid Rate Pipeline — Phase 2: Multi-Channel Fusion
- **Added** `scripts/evaluate_multichannel.py` — runs spectral + adaptive_peaks → Kalman on 5 channels (CLE, CRE, CH, avg, diff) independently, quality-weighted fusion across channels, oracle (best per-window)
- **Output** `reports/rates/hybrid_phase2/` — 28 PNGs (per-session channel comparison, aggregate bars, heatmaps), 2 CSVs
- Resp: multi-ch 1.82 br/min (vs single-best 1.90, oracle 1.21) — 4% improvement
- Cardiac: multi-ch 17.74 BPM (vs single-best 21.22, oracle 8.63) — 16% improvement
- Oracle headroom shows substantial gains possible with better channel selection

### Hybrid Rate Pipeline — Phase 3: Formal Evaluation
- **Added** `scripts/evaluate_hybrid_pipeline.py` — full evaluation: per-session k-calibration for baseline (hilbert/k, peaks/k) and Kalman pipeline, LOSO k cross-validation, per-stage MAE, Bland-Altman, Wilcoxon tests
- **Output** `reports/rates/hybrid_phase3/` — 24 time-series PNGs, 3 Bland-Altman aggregate PNGs, 2 per-stage PNGs, 2 session-comparison PNGs, 3 CSVs (results, windows, k-calibration)

### Ridge Overlay v2 + Prominence Scoring
- **Updated** `sleep_monitor/harmonics.py`:
  - Step 5c: median-filter smoothing of ridge frequency traces (size=7)
  - Step 5d: per-ridge prominence traces (amp / local spectral floor ±0.3 Hz), median-filter smoothed
  - Added `compute_prominence_score()` — per-window max ridge prominence (gated at min_prominence=2.0), temporally smoothed (median filter size=15 ≈ 3.75 min), normalized to [0,1]
- **Updated** `analysis/slow_wave/run_ridge_overlay.py`:
  - `MIN_PERSIST_SEC` 180→300 (5 min minimum ridge)
  - Added `compute_fine_spectrogram()` — high-res visual background (nperseg=2048, nfft=4096)
  - Replaced harmonic-ladder scoring with prominence-based scoring throughout
  - Ridges colored by prominence (SNR vs floor) instead of raw amplitude; labels show "freq (Nx)"
  - Removed ladder dots, pick_best_channel, plot_multichannel_comparison
  - 6-row stacked layout, motion as red semi-transparent overlay, output to `reports/slow_wave/overlay/`
  - Figure size 22x20 at 200 DPI
- **Deleted** old superseded outputs: `harmonics_*.png`, `harmonic_ladders_*.png/.parquet`, `ridge_multichannel_*.png` from `reports/slow_wave/`
- **Added** `writeup/harmonics/generate_harmonics_docx.py` — Methods + Results section for harmonic ridge prominence analysis
- **Added** `writeup/harmonics/CAP_harmonic_ridge_analysis.docx` — generated Word document with 3 figures + Table 1
- **Added** `writeup/figures/harmonics/` — key figures (S1N1, S4N2 overlays, score-by-stage boxplot)

### Hybrid Rate Pipeline — Phase 1: Kalman Rate Tracker
- **Added** `kalman_rate_track()` in `sleep_monitor/rates.py` — scalar Kalman filter fusing spectral + adaptive_peaks per-window estimates with physiological rate-of-change constraints. Auto-selects Q from band (resp: 2 br/min/epoch, cardiac: 5 BPM/epoch). Handles NaN gaps, clamps to band bounds.
- **Updated** `sleep_monitor/__init__.py` — exported `kalman_rate_track`
- **Added** `tests/test_rates.py::TestKalmanRateTrack` — 6 tests (constant signal, noise smoothing, NaN gaps, all-NaN, length, bounds)
- **Added** `scripts/benchmark_kalman_tracker.py` — full benchmark: time-series, Bland-Altman, per-stage, aggregate bar chart, improvement heatmap
- **Output** `reports/rates/hybrid_phase1/` — 24 time-series PNGs, 24 Bland-Altman PNGs, 4 aggregate plots, 2 CSVs

### Hybrid Rate Pipeline — Phase 0: Adaptive Peak Detector
- **Added** `rate_adaptive_peaks()` in `sleep_monitor/rates.py` — spectral-guided, amplitude-adaptive peak detector with IPI validation. Uses spectral peak for min_distance, rolling MAD for prominence, and inter-peak-interval CV check with MAD-based outlier rejection.
- **Updated** `sleep_monitor/config.py` — added `adaptive_peaks` to METHOD_NAMES, METHOD_LABELS, METHOD_COLORS
- **Updated** `sleep_monitor/__init__.py` — exported `rate_adaptive_peaks`
- **Updated** `estimate_rate()` — now includes `adaptive_peaks` in output dict (6 methods)
- **Added** `tests/test_rates.py::TestRateAdaptivePeaks` — 7 tests (pure sine, noise robustness, short signal, erratic peaks, integration, amplitude drift)
- **Added** `scripts/benchmark_adaptive_peaks.py` — benchmark script comparing all methods on 12 sessions

---

## 2026-06-11

### SWA Validation
- **Added** `analysis/swa_validation/CLAUDE.md` — SWA validation workspace: Lucey et al. 2019 replication plan, Steps 0-4, deliverables, working rules
- **Added** `analysis/swa_validation/step0_inventory.py` — Step 0 data inventory script: scans all 12 sessions, reports format/channels/rates/alignment/quality

### Loader — staging alignment fix
- **Fixed** `sleep_monitor/loader.py` `load_session()` — `time_start` was always None due to `pd.to_datetime(val, unit='ms')` on datetime strings; removed `unit='ms'`
- **Fixed** `sleep_monitor/loader.py` `load_sleep_profile()` — now parses wall-clock timestamps from Sleep Profile epoch lines, aligns to CSV time via `session.time_start`. Drops epochs outside CSV window. Handles midnight crossing. Previously assigned epoch 0 to t=0 regardless of PSG→CSV offset (up to 38.5 min for S1-S2 sessions).

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
