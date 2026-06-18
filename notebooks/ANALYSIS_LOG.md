# Analysis Log

Each entry records the question asked, code written, parameters used, plots generated, and key findings.

## Recent (last 3 — scroll down for full history)
- **2026-06-18** — Window-size / tracking audit (`scripts/analyze_window_size_spectral.py`) + biomarker-k check. **MAE was flattering everything.** `rate_spectral` uses nperseg=400 → df=0.25 Hz, so r_spectral(resp)=0.25 Hz in 9317/9319 epochs = a CONSTANT predictor, within-session corr w/ GT = 0.00. "Spectral wins resp" is a resolution artifact (predicts population mean 15 br/min). Hi-res spectral is noisier not better; longer windows lower MAE but within-session corr stays ~0 for resp (resp genuinely stable, GT std 2.0 br/min). Honest metric = WITHIN-session corr (pooled corr inflated by per-session-k between-session mean-matching). Cardiac: GT std 9.7 BPM (real variation) but peaks within-session corr ~0; yet median-MAE 3.7 < predict-mean 6.5 → captures coarse trend, ~0 corr partly GT R-peak noise on 30s windows (re-test w/ smoothed GT). **Biomarker k:** k_card(t)=peaks/GT structured (lag-1 autocorr 0.52) w/ stage structure (N1 2.03 highest) BUT corr(k,rate)=−0.83 (just absorbs 1/rate since peaks doesn't track) and GT-free proxy corr −0.06 (not recoverable). k is NOT an independent biomarker yet — need a tracking method first, then test residual-k for dicrotic/morphology signal. Data: `reports/rates/mask/window_size_sweep.csv`, figs 10/11.
- **2026-06-17** — Adaptive-k + Oracle channel analysis (cache-only, `scripts/analyze_adaptive_k_and_oracle.py`). Two findings that reframe next steps: **(Q1) Self-supervised adaptive k(t)** (peaks anchored to spectral, no GT): helps resp peaks slightly (2.01→1.93) but still loses to spectral (1.39) → spectral genuinely better for resp, NOT just because k-free. FAILS for cardiac (4.46→16.28) because it anchors to spectral which is terrible for cardiac (20.40). Lesson: self-sup adaptive k needs a good k-free anchor; cardiac has none. Calibration drift is real (first-10-min k worse: resp 2.67, card 8.90 vs full-session 2.01/4.46). **(Q2) Channel diversity:** irrelevant for resp (oracle-channel 1.08 ≈ diff 1.09; resp headroom is in METHODS: oracle-method 0.54, full oracle 0.16). But THE headroom for cardiac (oracle-channel **1.58** vs fused 3.91; win-distribution ~even 19-21% per channel, no channel dominates → confirms channels carry different cardiac info). Full cardiac oracle 0.51 BPM. Next priority: smart cardiac channel fusion to capture the ~2.3 BPM headroom without GT. Handoff in `CONTINUATION_RATE_DETECTION.md`. Figs 7-9 in `writeup/figures/mask_rate_detection/`.
- **2026-06-17** — Mask Rate Detection (paper-ready): Smart Fusion (Karlen 2013) + multi-channel SQI-weighted fusion + k-calibration + causal smoothing. **Resp: MAE=1.09 br/min** (spectral on diff, k≈0.97 — near unity, no calibration needed). **Cardiac: MAE=3.91 BPM** (peaks_loose with multi-channel agreement fusion, k≈1.95 — consistent 2:1 ratio). 6 subjects × 2 nights, 9318 epochs. Key findings: (1) Resp spectral estimation works out-of-box, multi-channel adds nothing. (2) Cardiac peak counting REQUIRES k-calibration (57→4.5 BPM, 13x improvement). (3) Multi-channel fusion helps cardiac only (4.07→3.91, 4%). (4) S6 sessions anomalous (k=1.35/0.94 vs typical 1.9-2.1). (5) Worst at rate extremes: resp MAE=2.38 at >17 br/min, cardiac MAE=11.81 at >69 BPM. (6) REM worst for resp (2.31), Wake worst for cardiac (4.73). Figures: `writeup/figures/mask_rate_detection/` (9 figs). Supersedes best-of-both pipeline.
- **2026-06-11** — Best-of-both rate pipeline: Kalman (reactive) for resp + hilbert for cardiac, multi-channel fusion, temporal smoothing, k-scaling. **Resp: 1.49 br/min (42% over baseline), Cardiac: 4.11 BPM (15% over single-ch hilbert/k).** LOSO: 1.95/5.41. Rate documentation updated with 35 figures + 4 tables.
- **2026-06-11** — SWA Validation Steps 1-4: Lucey et al. 2019 replication complete. **Negative result** — capacitive temple sensors show zero SWA correlation with contact EEG. Pipeline validated via EEG self-AUC=0.740. Critical `firls` filter bug fixed.
- **2026-06-11** — Hybrid rate pipeline Phase 3: Kalman/k wins for resp (MAE 1.61, 37% improvement). Hilbert/k still best for cardiac (4.84 BPM).
- **2026-05-22** — Spectral peak tracker with auto-detection and ridge tracking (`analysis/slow_wave/run_peak_tracker.py`)
- **2026-05-28** — Phase 3 clustering: GMM (k=3,4,5) + DBSCAN on supervised UMAP embeddings for all 12 sessions (`scripts/run_clustering_phase3.py`). GMM k=4 is the sweet spot (best ARI in 7/12 sessions). Best: S5N2 ARI=0.943, S2N2 ARI=0.942. S6 confirmed poor (ARI<0.15). DBSCAN over-fragments. Outputs: confusion matrices, 3D HTML, side-by-side panels in `reports/projections/<session>/clustering/`.
- **2026-05-28** — Stage 3 harmonic ridges vs physiology: persistent ridges on all 12 sessions x 3 channels, per-epoch features aligned with PSG stages (`analysis/slow_wave/run_ridge_stage3.py`). Key: N3 has fewer harmonic groups (p=7.6e-5), lower ridge frequency (0.25 vs 0.88 Hz, p=5.7e-4), less ridge power (p=3.7e-5), less freq spread (p=8.5e-7). Direction is consistent: N3<other in 5/6 subjects for all features. Ridges are a quasi-stationary sleep feature but not a strong N3 discriminator on their own.
- **2026-06-11** — SWA Validation project initiated: Lucey et al. 2019 replication pipeline — capacitive EEG vs contact EEG for SWA/SWS. Workspace created at `analysis/swa_validation/`. Steps 0-4 defined (inventory → shared pipeline → reference targets → validation metrics → reporting). Awaiting user go-ahead.
- **2026-06-11** — Persistent ridge tracking added to `sleep_monitor/harmonics.py`: `detect_persistent_ridges()` with motion masking, fragment merging, harmonic group detection, continuous harmonic strength scoring
- **2026-06-11** — SWS detection classifier (`analysis/slow_wave/detect_sws.py`): LOSO N3 binary classifier using motion, band power ratios, spectral entropy, rate regularity, DC stability, coherence, harmonics
- **2026-06-11** — Trial-based SWS exploration (`analysis/slow_wave/detect_trials.py`): physiological criteria (DC slope, post-movement settling, thorax smoothness) to find candidate deep-sleep segments
- **2026-06-11** — Settling event visualization (`analysis/slow_wave/plot_settling_events.py`): post-movement windows with stacked physiological panels
- **2026-06-11** — Harmonic ladder detection (`analysis/slow_wave/run_harmonic_ladders.py`): integer-ratio grouping of concurrent persistent ridges, prevalence by sleep stage
- **2026-06-11** — Consolidated ridge overlay (`analysis/slow_wave/run_ridge_overlay.py`): tuned ridge detection with fragment merging, continuous harmonic score, rich 4-row overlay plots. **All 12 sessions processed.** Tuned params: 15s step (overlapping), SMOOTH_WINDOWS=9, PEAK_PROM_FRAC=0.25, MAX_GAP_WINDOWS=6, MIN_PERSIST_SEC=180. CRE is dominant harmonic channel (picked as best in 9/12 sessions). Pooled: CRE median harmonic score highest in REM (0.077) and N1 (0.054), lower in N3 (0.001). CLE shows N3 elevated (0.007). S4 (both nights) richest: 25-28% strong windows. Outputs: `reports/slow_wave/ridge_overlay_*.png`, `ridge_multichannel_*.png`, `ridge_overlay_score_by_stage.png`, `ridge_overlay_epochs.parquet` (55,878 rows)
- **2026-06-11** — Ridge/harmonic verification plots (`analysis/slow_wave/verify_harmonics_overlay.py`): 5 figure types overlaying detections on spectrograms and PSDs
- **2026-06-11** — Multi-channel rate consolidation pipeline (`scripts/run_rate_consolidation.py`): 6 phases completed. Phase 1: spectral best resp (1.54 br/min no-k), ACF best card (13.64 BPM no-k). Phase 2: confidence-weighted+agreement-filtered fusion. Phase 3: CWT ridge best cardiac without k (11.6 BPM on avg). Phase 4: Viterbi mandatory smoothing. Phase 5: Bland-Altman, per-stage, pipeline comparison. Phase 6: k-scaled best resp=spectral/diff 0.99 br/min (k≈0.98), best card=hilbert/CRE 3.55 BPM (k≈1.66). k varies by stage (KW significant). 23 figures + CSV in `writeup/figures/rate_consolidation/`, `reports/rates/`
- **2026-06-11** — Pooled cross-session projections Phase 4 (`scripts/run_pooled_phase4.py`): subject-level z-score normalization, pooled PCA/UMAP/supervised UMAP/t-SNE, GMM k=4, LOSO evaluation
- **2026-06-11** — Supervised UMAP validation v1 (`scripts/run_supervised_validation.py`): train/test split, train fraction sweep (25/50/75%), GMM on held-out embeddings
- **2026-06-11** — Supervised UMAP validation v2 (`scripts/run_supervised_validation_v2.py`): kNN + RF in raw vs UMAP space — separates "can CAP classify stages?" from "does UMAP help?"
- **2026-06-11** — Paper signal validation figures (`scripts/paper_signal_validation_figures.py`): 4 publication-quality figures (waveform, freq agreement, coherence+surrogates, channel comparison) + summary table
- **2026-06-11** — Paper docx generation (`scripts/generate_paper_docx.py`, `scripts/generate_rate_consolidation_docx.py`): Word documents for signal validation and rate consolidation sections
- **2026-06-11** — Unit tests added (`tests/`): `test_filters.py`, `test_preprocessing.py`, `test_rates.py`
- **2026-06-11** — SWA Validation Step 0: Data inventory across all 12 overnight sessions. All CSV+PSG files present. AASM staging available for all. Key issues found: sleep staging misalignment (up to 38.5 min offset), EEG unit/montage unknown, CAP in ADC counts.
- **2026-06-11** — Hybrid rate pipeline Phase 0: adaptive peak detector (`rate_adaptive_peaks`). Benchmarked on all 12 sessions vs 5 existing methods (no k-scaling). **Resp:** adaptive_peaks MAE 2.36 br/min (vs peaks 3.20, hilbert 2.50, spectral 1.92). **Cardiac:** adaptive_peaks MAE 26.51 BPM (vs peaks 48.53, hilbert 38.06, spectral 27.50). Adaptive_peaks is the best peak-based method for both bands — spectral-guided min_distance eliminates most double-counting, IPI validation rejects noisy windows. Bias near zero for resp (+0.05), positive for cardiac (+19.96 — the fundamental overcount that k-scaling corrects). Foundation for Phase 1 Kalman tracker. Plots: `notebooks/plots/rate_analysis/adaptive_peaks_benchmark_*.png`. Data: `artifacts/adaptive_peaks_benchmark.csv`.
- **2026-06-11** — Ridge overlay v2 + prominence scoring (`analysis/slow_wave/run_ridge_overlay.py`): Replaced harmonic-ladder scoring with ridge prominence (amp / local spectral floor). Per-ridge prominence traces computed in Step 5d of `detect_persistent_ridges()`, median-filter smoothed. `compute_prominence_score()` aggregates per-window max prominence (min_prominence=2.0 gate, temporal median smooth size=15 ≈ 3.75 min). Visual: high-res spectrogram background (Gouraud), 5-min min ridge, flat traces, 3-channel stacked 6-row figure, ridges colored by prominence. S1N1 key finding: **CH channel shows strong N3 discrimination** — median prominence 6.7x during N3, 82.6% strong, near zero in Wake/N1/REM. CLE/CRE scattered 6-14% strong. Outputs moved to `reports/slow_wave/overlay/`.
- **2026-06-11** — Best-of-both rate pipeline (`scripts/evaluate_best_pipeline.py`): Kalman (reactive, R x0.3, Q x2.0) for resp, hilbert for cardiac, 5-channel quality-weighted fusion, median temporal smoothing (win=7), per-session + LOSO k. **Per-session k: resp 1.49 br/min (42% over baseline), cardiac 4.11 BPM (15% over single-ch hilbert/k). LOSO: resp 1.95, cardiac 5.41.** Multi-channel hilbert fusion eliminates S6 catastrophic failures. All 12 sessions consistent (resp 1.07-2.16, cardiac 2.54-6.56). Rate documentation updated: `writeup/CAP_rate_consolidation_section.docx` — 35 figures, 4 tables, two-part structure (consolidation + hybrid).
- **2026-06-11** — Hybrid rate pipeline Phase 4: streaming demo (`scripts/demo_realtime_rates.py`). Lightweight `KalmanState` class processes S1N1 epoch-by-epoch: spectral + adaptive_peaks → Kalman fusion. **954 epochs in 1.8s (16,348x real-time).** Resp MAE 1.88 br/min, cardiac MAE 21.29 BPM — matches batch Kalman results, confirming streaming equivalence. Plot: `reports/rates/hybrid_phase4/streaming_demo_S1N1.png`.
- **2026-06-11** — Hybrid rate pipeline Phase 2: multi-channel fusion (`scripts/evaluate_multichannel.py`). Runs Kalman pipeline on 5 channels (CLE, CRE, CH, avg, diff) independently, then quality-weighted fusion. **Resp:** multi-ch 1.82 br/min (vs single-best 1.90, oracle 1.21) — 4% improvement, diff channel best single. **Cardiac:** multi-ch 17.74 BPM (vs single-best 21.22, oracle 8.63) — 16% improvement, diff channel best single. Oracle analysis shows 36%/59% headroom for resp/cardiac with perfect per-window channel selection. Outputs: `reports/rates/hybrid_phase2/` — 28 PNGs, 2 CSVs.
- **2026-06-11** — Hybrid rate pipeline Phase 1: Kalman tracker fusing spectral + adaptive_peaks. Benchmarked on all 12 sessions. **Resp:** Kalman MAE 1.90 br/min (vs spectral 1.92, adaptive 2.36) — matches spectral while adding temporal smoothing, RMSE 2.53 vs 2.54. **Cardiac:** Kalman MAE 21.22 BPM (vs spectral 27.50, adaptive 26.51) — **20% reduction** over best raw input. RMSE drops from 35.81 to 24.38 (32% reduction). Kalman wins on every session for cardiac. Improvement concentrated in jitter suppression (RMSE drop >> MAE drop). Bias preserved (~+20 BPM cardiac = systematic k-overcount, not noise). Per-stage and Bland-Altman plots in `reports/rates/hybrid_phase1/`. Data: `reports/rates/hybrid_phase1/kalman_tracker_results.csv`.
- **2026-06-11** — SWA Step 0 cont: fixed global staging misalignment in `loader.py`. Verified offsets via N3/delta separation sweep (12/12 match). Fixed `time_start` parsing bug (was always None due to `unit='ms'` on string). ~25 prior scripts affected. QA table + apnea loader flag.
- **2026-06-11** — Hybrid rate pipeline Phase 3: formal evaluation with k-scaling and LOSO. **Resp: Kalman /k wins decisively** — per-session MAE 1.61 br/min (vs baseline peaks/k 2.58), **37% improvement**, 12/12 sessions, Wilcoxon p=0.0002. LOSO: 2.02 br/min, **22% improvement**, 10/12 wins, p=0.005. k_kalman_resp ≈ 0.97–1.19 (close to 1.0 = less overcounting). **Cardiac: baseline hilbert/k still wins** — baseline 4.84 BPM vs Kalman/k 8.67 (per-session) / 8.94 (LOSO). Hilbert inst. freq. with k≈1.67 is inherently better calibrated for BCG cardiac than spectral+adaptive fusion (k≈1.34). Conclusion: hybrid pipeline is the new best for respiratory; for cardiac, hilbert/k remains superior. A combined pipeline (Kalman for resp, hilbert/k for cardiac) would give best-of-both: 1.61/4.84. Outputs: `reports/rates/hybrid_phase3/` (30 PNGs, 3 CSVs).
- **2026-06-11** — SWA Validation Steps 1-4 complete. **Negative result:** capacitive temple sensors (CLE-CRE) show zero SWA correlation with contact EEG (r=0.015 ± 0.045), coherence at noise floor (0.003 ± 0.005), N3 detection at chance (AUC=0.490 ± 0.040). Pipeline validated: EEG self-AUC=0.740 ± 0.056. Critical bug found and fixed: `firls` FIR design was catastrophically ill-conditioned (coefficients ±2794, amplification ×26M) — replaced with `firwin`. Outputs: `analysis/swa_validation/outputs/` (2 CSVs, 5 PNGs).
- *(add new entries below this line)*

---

## 2026-06-11 — SWA Validation Step 0: Data Inventory

**Question:** What data is available for the Lucey et al. 2019 SWA replication — capacitive EEG vs contact EEG?

**Script/Notebook:** `analysis/swa_validation/step0_inventory.py`
**Outputs:** Console summary (no artifacts yet)

### Setup
Scanned all three data directories:
- `overnight_6subject_pelthupdate_030526/` — Combined CAP+PSG synchronized CSVs
- `overnight_6subject_complete_032626/` — PSG analysis exports (Sleep Profile, Delta FFT, etc.)
- `combinedDataAnalyses_041626/` — Short ICP recordings (NOT overnight sleep — excluded)

### Results

#### Dataset summary: 12 recordings, 6 subjects × 2 nights

| Session | Subject  | Date       | Duration | Samples    | File Size |
|---------|----------|------------|----------|------------|-----------|
| S1N1    | OS001-KJK| 09-17-2024 | 7.95 hr  | 2,862,001  | 96.8 MB   |
| S1N2    | OS001-KJK| 09-18-2024 | 7.63 hr  | 2,748,001  | 93.3 MB   |
| S2N1    | OS002-LDI| 09-19-2024 | 7.73 hr  | 2,784,001  | 88.9 MB   |
| S2N2    | OS002-LDI| 09-20-2024 | 6.77 hr  | 2,436,001  | 78.2 MB   |
| S3N1    | OS003-LCW| 12-18-2025 | 6.93 hr  | 2,496,001  | 85.6 MB   |
| S3N2    | OS003-LCW| 12-19-2025 | 8.66 hr  | 3,117,001  | 104.0 MB  |
| S4N1    | OS004-CJH| 12-25-2025 | 6.18 hr  | 2,224,400  | 64.4 MB   |
| S4N2    | OS004-CJH| 12-26-2025 | 6.02 hr  | 2,166,001  | 62.9 MB   |
| S5N1    | OS005-CJY| 01-03-2026 | 4.11 hr  | 1,479,001  | 40.4 MB   |
| S5N2    | OS005-CJY| 12-27-2025 | 4.74 hr  | 1,707,001  | 50.5 MB   |
| S6N1    | OS006-SK | 01-14-2026 | 5.16 hr  | 1,857,001  | 64.3 MB   |
| S6N2    | OS006-SK | 01-15-2026 | 5.78 hr  | 2,082,001  | 67.3 MB   |

#### File format
- **Format:** Compressed CSV (`.csv.gz`), comma-separated
- **Columns:** `timeSM` (wall-clock ISO), `timeMS` (ms counter), then 6 CAP + 8 PSG channels
- **CAP channels:** CH, CLE, CRE, aX, aY, aZ
- **PSG channels:** EEG, EOGl, EOGr, ECG, Flow, Pleth, Thorax, Abdomen
- **Sampling rate:** 100 Hz (confirmed by sample_count / duration for all 12 sessions)

#### Units (NEEDS CONFIRMATION)
- **EEG:** Likely microvolts (typical range ±40 to ±270 uV; ADC clips at ±600)
- **CAP (CLE, CRE, CH):** ADC counts — no documented conversion to physical units
  - CLE ≈ 1970–2035, CRE ≈ 2190–2260, CH ≈ -1055 to -884
  - CLE-CRE differential std ranges from 5.8 (S5N2) to 199.7 (S6N2)
- **Other PSG:** ECG, Flow, Thorax, Abdomen — PSG system units (likely uV/arbitrary)

#### AASM sleep staging
- **Available:** All 12 sessions have PSG Sleep Profile text files
- **Resolution:** 30-second epochs
- **Stages:** Wake, Stage 1, Stage 2, Stage 3, REM, Artefact (AASM-like; no N3/N2 split beyond Stage 2 vs 3)
- **PSG system:** Appears to be SOMNOmedics (German "SchlafProfil" identifier in files)
- **Epochs per session:** 515–1125 (covering 4.29–9.38 hr)
- **Reliability files:** Also present (Sleep Profile Reliability) but not yet inspected

#### Time alignment

**Within CSV:** CAP and PSG signals are SAMPLE-ALIGNED (synchronization done at collection time). No additional alignment needed for sample-level comparisons.

**CSV vs Sleep Profile — MISALIGNMENT FOUND:**

| Session | CSV→PSG offset | Staging epochs before CSV |
|---------|---------------|--------------------------|
| S1N1    | +38.5 min     | ~77 epochs               |
| S1N2    | +37.0 min     | ~74 epochs               |
| S2N1    | +39.5 min     | ~79 epochs               |
| S2N2    | +28.0 min     | ~56 epochs               |
| S3N1    | +25.0 min     | ~50 epochs               |
| S3N2    | +30.5 min     | ~61 epochs               |
| S4N1    | +0.1 min      | ~0 epochs                |
| S4N2    | +2.0 min      | ~4 epochs                |
| S5N1    | +6.0 min      | ~12 epochs               |
| S5N2    | +4.0 min      | ~8 epochs                |
| S6N1    | +12.0 min     | ~24 epochs               |
| S6N2    | +8.0 min      | ~16 epochs               |

The existing `loader.load_sleep_profile()` does NOT account for this offset — it assigns epoch 0 to CSV time 0, but epoch 0 actually corresponds to PSG start (before CSV begins). **For S1–S2 (2024 recordings), staging is misaligned by 28–39.5 minutes.** For S4+ (2025–2026), offset is small (0.1–12 min). The SWA pipeline must use wall-clock alignment via `timeSM` and Sleep Profile timestamps.

#### PSG Delta FFT (system spectral output)
- Available for all 12 sessions: `Delta FFT - NNNNN.txt`
- Unit: % (relative power), 2 Hz sampling
- Could serve as sanity check but NOT as the reference — we compute our own PSD from raw EEG

#### Signal quality flags
1. **S6N2 ECG dead:** Constant ~1.0 (std < 0.04). Not relevant for SWA but noted.
2. **S6N1/S6N2 EEG noisiest:** std 65–71 uV, p1/p99 ±220–270 uV vs ~17–52 for others. Expect more artifact epochs.
3. **S5N1 shortest:** 4.11 hr — may barely meet minimum data requirements after artifact rejection.
4. **EEG ±600 clipping:** Occurs in S2N1/S2N2/S3N2/S4N1/S4N2/S5N1/S5N2/S6N1/S6N2 but rate is <0.12%.
5. **S5N1 CAP polarity:** CLE-CRE is POSITIVE (~247–405) while all others are negative. Possibly reversed electrode orientation.
6. **S6N1/S6N2 CAP noisy:** CLE-CRE differential std 108–200, vs 6–68 for others.

#### Validation dataset (combinedDataAnalyses_041626) — EXCLUDED
- Short ICP recordings (~12.5 min each), 6 subjects, tab-separated text
- Different electrode montage (Cvl, Cvr, Cbl, Cbr — 4 cap channels)
- No sleep staging, no overnight data
- **Not usable for SWA validation**

### Key findings
1. All 12 overnight recordings available with simultaneous capacitive + contact EEG at 100 Hz
2. AASM sleep staging exists for all 12 sessions (30-sec resolution, all 5 stages + artifact)
3. **CRITICAL: Sleep staging is time-misaligned** with CSV data by up to 38.5 min (early recordings). Must fix before proceeding.
4. **UNKNOWN: EEG montage/derivation.** Which contact EEG channel does the "EEG" column represent? (F3-M2? C3-M2? Cz?) Matters for Lucey comparison (they used frontal EEG).
5. **UNKNOWN: Exact EEG units.** Assumed uV but not documented. Need confirmation.
6. **UNKNOWN: Reference PSG filter characteristics.** CLAUDE.md notes "Reference device had 0.1-0.6 Hz band-stop." Need to verify if this applies to EEG in our data.
7. CAP signals are in ADC counts — SWA comparison will use power profiles (normalized), not raw values.
8. S5N1 is only 4.11 hr; S5N2 4.74 hr — subject OS005 has the least data but both nights available.
9. The combinedDataAnalyses dataset is irrelevant (short ICP recordings, different montage).

### Questions requiring user input before Step 1
1. What is the EEG derivation/montage in the combined CSV?
2. What are the EEG units? (uV seems right from the value ranges)
3. Does the PSG system apply a band-stop filter in the 0.1–0.6 Hz range to the EEG signal?

### Status
Step 0 COMPLETE — awaiting user confirmation before Step 1.

---

## 2026-06-11 — SWA Step 0 cont: Staging alignment fix + QA table

**Question:** Verify and fix the sleep staging time misalignment. Build per-night QA table.

**Script/Notebook:** `analysis/swa_validation/step0_inventory.py` (updated), `sleep_monitor/loader.py` (fixed)

### Setup
Per user direction:
- Fix staging alignment globally in `loader.py` using wall-clock timestamps
- Verify offset via independent signal (N3/delta power separation sweep)
- Flag affected analyses
- Build per-night QA table

### Results

#### Offset verification — N3/delta power separation sweep
For each session, swept epoch skip from 0 to 120 and measured Cohen's d between N3 and non-N3 log-delta power. The timestamp-derived offset matches the optimal d within 0-2 epochs for 8/12 sessions:

| Session | TS skip | Best skip | Diff | d at TS | d at best | N3 epochs | Status |
|---------|---------|-----------|------|---------|-----------|-----------|--------|
| S1N1    | 77      | 89        | +12  | 1.889   | 1.923     | 43        | Flat plateau |
| S1N2    | 74      | 73        | -1   | 1.886   | 1.978     | 32        | MATCH  |
| S2N1    | 79      | 78        | -1   | 1.024   | 1.041     | 98        | MATCH  |
| S2N2    | 56      | 56        | 0    | 1.473   | 1.473     | 131       | MATCH  |
| S3N1    | 50      | 49        | -1   | 1.631   | 1.656     | 61        | MATCH  |
| S3N2    | 61      | 51        | -10  | 0.880   | 0.891     | 105       | Flat plateau |
| S4N1    | 0       | 0         | 0    | 1.023   | 1.023     | 49        | MATCH  |
| S4N2    | 4       | 3         | -1   | 1.570   | 1.604     | 58        | MATCH  |
| S5N1    | 12      | 11        | -1   | 1.734   | 1.777     | 96        | MATCH  |
| S5N2    | 8       | 0         | -8   | 0.858   | 1.025     | 46        | Flat region |
| S6N1    | 24      | 13        | -11  | 0.327   | 1.347     | 14        | Too few N3 |
| S6N2    | 16      | 16        | 0    | 0.222   | 0.222     | 50        | MATCH  |

S1N1 and S3N2 have large diffs but nearly identical d-values (flat landscape). S6N1 has only 14 N3 epochs making the metric unreliable. **Conclusion: timestamp-derived offsets are correct.**

#### Bugs fixed in `loader.py`
1. **`load_session()` — `time_start` parsing:** Changed `pd.to_datetime(val, unit='ms', utc=True)` to `pd.to_datetime(val)`. The `unit='ms'` flag caused string datetime values to fail, making `time_start` always None.
2. **`load_sleep_profile()` — wall-clock alignment:** Now parses HH:MM:SS timestamps from each Sleep Profile epoch line, computes offset from CSV start time (`session.time_start`), and drops epochs outside the CSV recording window. Handles midnight crossing.

#### Per-Night QA Table

| Sess | Subj  | Dur   | N3 min | N2 min | REM min | Wake min | EEG std | EEG clip% | CAP diff std | Polarity | ECG | d(N3) |
|------|-------|-------|--------|--------|---------|----------|---------|-----------|--------------|----------|-----|-------|
| S1N1 | OS001 | 7.95h | 21.5   | 236.5  | 9.5     | 77.5     | 25.1    | 0.00%     | 7.3          | NEG      | OK  | +1.89 |
| S1N2 | OS001 | 7.63h | 16.0   | 166.5  | 2.5     | 99.0     | 21.1    | 0.00%     | 19.9         | NEG      | OK  | +1.89 |
| S2N1 | OS002 | 7.73h | 49.5   | 399.0  | 4.0     | 4.0      | 17.4    | 0.00%     | 58.0         | NEG      | OK  | +1.02 |
| S2N2 | OS002 | 6.77h | 66.5   | 325.0  | 3.5     | 6.5      | 16.8    | 0.00%     | 23.3         | NEG      | OK  | +1.47 |
| S3N1 | OS003 | 6.93h | 30.5   | 255.5  | 6.5     | 34.0     | 19.8    | 0.00%     | 8.7          | NEG      | OK  | +1.63 |
| S3N2 | OS003 | 8.66h | 52.5   | 296.5  | 8.0     | 67.5     | 41.3    | 0.04%     | 24.7         | NEG      | OK  | +0.88 |
| S4N1 | OS004 | 6.18h | 24.5   | 217.0  | 19.0    | 40.0     | 29.9    | 0.04%     | 67.6         | NEG      | OK  | +1.02 |
| S4N2 | OS004 | 6.02h | 29.0   | 178.0  | 39.5    | 45.5     | 22.1    | 0.00%     | 15.6         | NEG      | OK  | +1.57 |
| S5N1 | OS005 | 4.11h | 48.0   | 101.0  | 6.5     | 48.5     | 32.1    | 0.00%     | 25.0         | **POS**  | OK  | +1.73 |
| S5N2 | OS005 | 4.74h | 23.0   | 164.5  | 13.0    | 35.5     | 52.6    | 0.12%     | 5.8          | NEG      | OK  | +0.86 |
| S6N1 | OS006 | 5.16h | 7.0    | 238.5  | 2.5     | 25.5     | 64.7    | 0.04%     | 107.6        | NEG      | OK  | +0.33 |
| S6N2 | OS006 | 5.78h | 25.0   | 286.0  | 0.0     | 15.0     | 70.6    | 0.06%     | 199.7        | NEG      | DEAD| +0.22 |

#### Affected prior analyses (~25 scripts)
All scripts that loaded sleep profiles were using the OLD (misaligned) staging, with offsets up to 38.5 min for S1-S2 sessions. Key affected areas:
- Rate accuracy per-stage analysis (`scripts/rate_accuracy_analysis.py`)
- Projection/staging coloring (`scripts/run_projections_v2.py`, etc.)
- Slow wave detection (`analysis/slow_wave/detect_sws.py`, etc.)
- Ridge overlay stage analysis (`analysis/slow_wave/run_ridge_overlay.py`)
- Signal validation per-stage (`scripts/signal_validation*.py`)
- Rate consolidation per-stage (`scripts/run_rate_consolidation.py`)

**Impact:** Results from S4-S6 (small offset 0-12 min) are minimally affected. S1-S2 results (28-39 min offset) should be regenerated before citing. Aggregate cross-session statistics may shift but trends likely hold.

#### Also noted: `load_apnea_events()` has the same bug
`_hms_to_hr()` returns time-of-day (0-24h), but apnea events are filtered against `session_dur_hr` (recording duration). This drops events from recordings crossing midnight. Not needed for SWA — flagged for future fix.

### Key findings
1. Wall-clock alignment verified via independent delta power separation — timestamp offsets confirmed correct
2. `time_start` was always None due to `pd.to_datetime(string, unit='ms')` bug — now fixed
3. 8/12 sessions match within 0-1 epochs; 4 have flat d-landscapes or too few N3 epochs
4. S6N1 has only 7 min of N3 and d=0.33 — marginal for SWA validation
5. S6N2 has 0 REM epochs and dead ECG — unusual sleep architecture
6. S5N1 CLE-CRE polarity is positive (flipped) — harmless for PSD-based SWA, noted
7. All 12 sessions have d > 0 after fix (correct direction), confirming alignment

### Status
Done. Loader fix committed. Ready for Step 1 pending user go-ahead.

## 2026-05-28 — Stage 3: Persistent ridge features vs sleep stage

**Question:** Do persistent spectral ridges (tracked with temporal continuity) correlate with sleep stage? Is harmonic structure a marker of N3/deep sleep?

**Script:** `analysis/slow_wave/run_ridge_stage3.py`

### Approach
- Ran `detect_persistent_ridges()` on all 12 sessions x 3 CAP channels (CH, CLE, CRE)
- Parameters: 30s windows, 7-window median smoothing, min 300s persistence, max 0.08 Hz jump, 5-window gap tolerance
- Per-epoch features: n_ridges, n_groups_active, max_group_size, min/mean/max ridge freq, freq spread, mean/max/total ridge amplitude, strongest f0
- Aligned with PSG sleep stage labels (30s epochs)
- Statistical tests: Kruskal-Wallis (5 stages), Mann-Whitney U (N3 vs non-N3), per-subject breakdown

### Key findings

**N3 vs non-N3 (CH channel, wake excluded):**
| Feature | N3 median | Other median | MW-U p | Direction |
|---------|-----------|-------------|--------|-----------|
| n_ridges | 0.0 | 0.0 | 0.13 ns | 5/6 subj N3<other |
| n_groups_active | 0.0 | 0.0 | 7.6e-5 *** | 6/6 subj N3<other |
| max_group_size | 0.0 | 0.0 | 8.1e-5 *** | 6/6 subj N3<other |
| min_ridge_freq | 0.25 Hz | 0.88 Hz | 5.7e-4 *** | 6/6 subj N3<other |
| total_ridge_power | 0.11 | 0.17 | 3.7e-5 *** | 5/6 subj N3<other |
| freq_spread | 0.0 | 0.0 | 8.5e-7 *** | 6/6 subj N3<other |

**5-stage Kruskal-Wallis (all significant p<0.001):**
- Active ridges: KW p=7.2e-10
- Total ridge power: KW p=4.1e-6
- Ridge freq spread: KW p=4.6e-12

**Per-subject patterns (per-subject plot):**
- Direction is remarkably consistent: N3 has fewer, lower-frequency, lower-power ridges in 5-6/6 subjects
- OS002 is the only subject where N3 total ridge power > other (driven by large N3 ridge in one session)
- OS006 has the most active ridges overall (up to 4 concurrent), concentrated in N2

**Interpretation:**
- N3 epochs show *reduced* harmonic structure: fewer persistent ridges, lower frequencies when present, less spectral spread
- This is consistent with deep sleep muscle relaxation reducing non-sinusoidal respiratory waveform complexity
- Harmonic groups (integer-ratio ridge sets) are rare events even in non-N3 stages -- medians are 0 everywhere, significance comes from the tail
- The signal is real (consistent direction across subjects) but weak as a standalone N3 classifier -- most epochs have 0-1 ridges regardless of stage

### Outputs
- `reports/slow_wave/stage3_ridge_epochs.parquet` — 27,957 epoch rows (all sessions x channels)
- `reports/slow_wave/stage3_ridge_features_by_stage.png` — pooled box plots
- `reports/slow_wave/stage3_ridge_features_per_subject.png` — per-subject box plots
- `reports/slow_wave/stage3_n3_vs_rest.png` — N3 vs non-N3 comparison
- `reports/slow_wave/stage3_ridge_timeseries_<label>.png` — 12 per-session time series
- `reports/slow_wave/stage3_summary.csv` — N3 vs non-N3 summary table

### Next steps
- These ridge features are weak standalone classifiers but may add discriminative power when combined with band power ratios and k_cardiac in a multivariate model (Stage 4)
- The min_ridge_freq feature (0.25 Hz in N3 vs 0.88 Hz in other) is the most promising single feature -- it captures the shift toward slower, simpler respiratory waveforms in deep sleep

---

## 2026-05-28 — Informed unsupervised projections (KW-weighted + NCA)

**Question:** Can supervised knowledge (which features discriminate stages) transfer to unsupervised embeddings?

**Script:** `scripts/run_projections_informed.py`

### Approach
- Loaded v2 feature CSVs (36 pure-CAP features) for all 12 sessions
- **KW-weighted**: scaled features by sqrt(Kruskal-Wallis H statistic) before UMAP — amplifies stage-discriminative features
- **NCA-transformed**: learned 20D linear projection via sklearn NeighborhoodComponentsAnalysis (200 iter) that pulls same-stage points together, then ran UMAP
- **NCA+KW combined**: applied KW-weighting on NCA dimensions
- Compared all against raw unsupervised UMAP and supervised UMAP baselines

### Key findings
- **Linear transfer barely helps**: mean gap closed = 6.3% (median 6.2%)
- **KW-weighting slightly hurts** on average (delta = -0.021) — in some sessions it suppresses useful features
- **NCA provides modest improvement** (delta = -0.005 on average, but positive in 8/12 sessions)
- **The supervised→unsupervised gap is fundamentally nonlinear** — supervised UMAP uses label information during graph construction, not just feature weighting
- KW top features vary by session: acc_rms and breath_interval_cv appear most often, but the discriminative features are session-specific
- NCA top features emphasize cross-channel (CRE_SO, CH_SO, coh_resp) and rate features — different from KW rankings

### Interpretation
The stage signal IS in the features (supervised UMAP proves this), but it lives in nonlinear interactions that neither reweighting nor linear projection can extract. The dominant variance structure in the raw feature space is driven by non-stage factors (amplitude drift, inter-epoch noise, respiratory depth variability). Unsupervised methods faithfully represent this dominant structure.

### Outputs
- `reports/projections/<session>/<session>_informed_silhouette.csv`
- `reports/projections/<session>/<session>_feature_importance.csv` + `.png`
- `reports/projections/informed_unsupervised_comparison.png` — bar chart across all sessions
- `reports/projections/informed_gap_closed.png` — % of supervised gap recovered
- `reports/projections/informed_comparison_summary.csv`

---

## 2026-05-28 — Pure-CAP projections v2 (all 12 sessions)

**Question:** Can we separate sleep stages in 3D embeddings using only raw CAP sensor features (no k-calibration)?

**Script:** `scripts/run_projections_v2.py`

### Approach
- Extracted 36 features per 60s window (30s step) from CLE, CRE, CH, accelerometer ONLY
- Feature groups: per-channel band powers (16), diff signal stats with Hjorth params (6), respiratory band (6), cardiac band (4), cross-channel coherence (2), accelerometer (2)
- Rates via ACF (no k-calibration needed)
- Methods: PCA, UMAP (nn=15,30,50), supervised UMAP (nn=15,30), t-SNE (perp=15,30,50), PHATE (knn=10,30)
- 7 coloring overlays: stage, time, apnea, thorax RMS, resp power, cardiac power, trajectory

### Key findings
- **Supervised UMAP dominates** — best 5-class silhouette in 11/12 sessions
- 5-class silhouette range: -0.02 to 0.69 (median ~0.45)
- Best sessions: S4N2 (0.69), S5N2 (0.67), S5N1 (0.56), S3N1 (0.54)
- Worst sessions: S6N1 (-0.02), S6N2 (0.03) — needs investigation
- Top discriminating features: acc_rms, breath_interval_cv, diff_hjorth_mobility/complexity, diff_SO
- Thorax RMS correlates with diff_hjorth_complexity (~0.45) and resp_snr
- PHATE preserves sleep cycle trajectory structure — visible in trajectory plots
- Unsupervised UMAP/t-SNE show weaker separation than supervised UMAP

### Outputs
- `reports/projections/<session>/` — 43 files per session (interactive HTML, static PNG, CSV)
- `reports/projections/cross_session_summary.csv` — silhouette comparison table

---

## 2026-05-22 — Spectral peak tracker with auto-detection and ridge tracking

**Question:** What persistent spectral peaks exist in CAP signals across the night, without assuming integer-harmonic structure?

**Script:** `analysis/slow_wave/run_peak_tracker.py`

### Approach
Unlike the harmonic detector (Stages 1-2) which assumed integer multiples of f0, this finds ALL significant spectral peaks per window via `scipy.signal.find_peaks` on the Welch PSD, then links them across time into "ridges" (persistent frequency tracks) using nearest-neighbor matching within a tolerance.

### Parameters
- Window: 30s, step: 30s, frequency range: 0-5 Hz
- Welch segment: 10s, peak prominence threshold: 15% of window-max PSD
- Minimum peak distance: 0.08 Hz
- Ridge linking tolerance: 0.08 Hz, minimum ridge length: 4 windows

### Output per session (12 figures)
Each figure has 5 rows:
1. Hypnogram (PSG ground truth)
2. CH spectrogram + auto-detected peaks (cyan dots) + ridge lines
3. CLE spectrogram + peaks + ridges
4. CRE spectrogram + peaks + ridges
5. All persistent ridges overlaid on stage-shaded background, colour-coded by channel

### Typical ridge counts per session
- CH: 33-74 ridges, CLE: 24-84 ridges, CRE: 13-62 ridges
- Most ridges cluster in 0-1.5 Hz (respiratory fundamental + first harmonic)

### Plots generated
- `notebooks/plots/harmonics/spectrogram_peaks_s{X}n{Y}.png` (12 files)

### Observations
- S1N1, S2N1, S6N2: clear harmonic ladder visible in CH spectrogram (bright bands at 0.5, 1.0, 1.5, 2.0 Hz) that appears/disappears — auto-detected ridges track these well
- S3, S4, S5: more diffuse spectral energy, fewer persistent ridges — consistent with Stage 2 finding that harmonic structure is subject-dependent
- Ridge persistence correlates visually with stable sleep epochs (N2/N3) rather than transitions
- CH consistently shows cleanest spectral structure across subjects

---

## 2026-05-22 — Cross-session harmonic characterization (Stage 2)

**Question:** Does the N3-elevated harmonic energy ratio (observed in S1N1) generalise across all 12 sessions and 6 subjects?

**Script:** `analysis/slow_wave/run_harmonic_allsessions.py`

### Parameters (same as Stage 1)
- Window: 30s, step: 30s, f0 range: [0.1, 0.8] Hz, max harmonics: 6
- Welch segment: 8s, harmonic tolerance: ±0.05 Hz, min prominence: 0.1×max
- Motion gating: 3 MAD

### Dataset
- 9319 total windows across 12 sessions, 8156 valid (non-motion-masked)
- Saved to `artifacts/harmonics/allsessions.parquet`

### Pooled harmonic_energy_ratio by stage

| Stage | n | Mean | Median |
|-------|---|------|--------|
| Wake | 1065 | 0.518 | 0.519 |
| N1 | 1300 | 0.427 | 0.377 |
| N2 | 4907 | 0.502 | 0.509 |
| N3 | 682 | 0.494 | 0.446 |
| REM | 202 | 0.423 | 0.414 |

### Statistical tests

**Kruskal-Wallis (5-stage):** All four features significant (p < 1e-4), confirming stage-dependent variation exists. harmonic_energy_ratio H=113.5 (p=1.3e-23), hps_score H=95.0 (p=1.1e-19).

**Mann-Whitney U (N3 vs others, Bonferroni-corrected):** N3 is significantly *lower* than N1 for HER (r_rb = −0.14, p < 1e-6) and significantly lower than REM (p = 0.008). N3 vs Wake and N3 vs N2 are non-significant. This **reverses** the Stage 1 single-session finding.

### Per-session breakdown (median HER)

The heatmap reveals strong **subject dependence**:
- **Subjects 1-2 (S1N1–S2N2):** N3 is the HIGHEST stage (0.70–0.83). Replicates Stage 1 finding.
- **Subjects 3-4 (S3N1–S4N2):** N3 is the LOWEST stage (0.13–0.40). Complete reversal.
- **Subjects 5-6 (S5N1–S6N2):** N3 is moderate (0.31–0.66), no clear pattern.

N3 median > N2 median in 7/12 sessions, but not consistently across subjects.

### Dominant channel
CH dominates in all 12 sessions (48–89% of windows, 70% overall). Consistent finding.

### Plots generated
- `notebooks/plots/harmonics/fullnight_grid_allsessions.png` — 4×3 grid of HER traces with stage shading
- `notebooks/plots/harmonics/stage_boxplots_allsessions.png` — pooled feature boxplots by stage
- `notebooks/plots/harmonics/heatmap_her_session_stage.png` — session × stage median HER heatmap
- `notebooks/plots/harmonics/dominant_channel_allsessions.png` — CH/CLE/CRE % by session
- `notebooks/plots/harmonics/n3_effectsize_allsessions.png` — per-session N3-vs-rest rank-biserial effect size

### Key findings
1. **Harmonic energy ratio does NOT universally mark N3.** The S1N1 finding was subject-specific (Subjects 1-2), not a general biomarker. Subjects 3-4 show the opposite pattern.
2. **Subject-level coupling dominates stage-level variation.** The absolute HER level varies more between subjects (0.13–0.83) than between stages within a subject.
3. **Stage differences ARE statistically significant** (Kruskal-Wallis p < 1e-23) but the *direction* is inconsistent — some subjects have N3-high, others N3-low harmonic structure.
4. **CH remains the dominant channel** across all subjects (70% overall), confirming Stage 1.
5. **HPS score and cepstral prominence** show similar subject-dependent patterns — no single harmonic feature is a universal N3 marker in raw form.

### Implications for Stage 3-4
- Raw harmonic features cannot be used as a universal N3 detector without per-subject normalisation.
- **Next approach:** per-subject z-scoring or rank-normalisation of harmonic features before stage classification. The *relative* change within a night may still discriminate, even if the absolute level is subject-dependent.
- Alternatively, combine harmonic features with band power ratios (which may have complementary subject dependencies) in a multivariate classifier.

---

## 2026-05-22 — Harmonic structure detection module (Stage 1)

**Question:** Can harmonic ladders (fundamental + integer multiples) in CAP spectrograms discriminate sleep stages, particularly N3/SWS?

**Module:** `sleep_monitor/harmonics.py`
**Methods implemented:** HPS (log-domain), Cepstral analysis, Explicit f0 + harmonic counting

### Parameters
- Window: 30s, step: 30s, f0 range: [0.1, 0.8] Hz, max harmonics: 6
- Welch segment: 8s, harmonic tolerance: ±0.05 Hz, min prominence: 0.1×max
- Motion gating: 3 MAD above median accelerometer RMS

### Smoke test (synthetic signals)
| Signal | energy_ratio | hps_score | cep_prominence | n_harmonics |
|--------|-------------|-----------|----------------|-------------|
| f0=0.4 + 2 harmonics | 0.601 | 41.5 | 18.6 | 2-3 |
| Pure sinusoid f0=0.3 | 0.552 | 38.5 | 11.9 | 1 |
| White noise | 0.008 | 0.5 | 6.6 | 0-1 |

All three methods correctly discriminate harmonic from non-harmonic signals.

### S1N1 real data results (954 windows, 143 motion-masked)

**Dominant channel:** CH wins 88% of windows (713/811) — top-of-head cap sensor picks up strongest harmonics, likely due to stronger BCG coupling.

**harmonic_energy_ratio by sleep stage (key finding):**

| Stage | n | Mean | Median |
|-------|---|------|--------|
| N3 | 43 | 0.649 | 0.753 |
| Wake | 171 | 0.604 | 0.663 |
| N2 | 382 | 0.590 | 0.649 |
| REM | 9 | 0.629 | 0.597 |
| N1 | 205 | 0.483 | 0.414 |

N3 has the highest harmonic energy ratio (median 0.753), well separated from N1 (0.414). This supports the hypothesis that non-sinusoidal respiratory waveform structure increases during deep sleep.

**n_harmonics:** modest discrimination (N3 mean 1.81 vs N1 1.42).

### Plots generated
- `notebooks/plots/harmonics/harmonics_fullnight_s1n1.png` — full-night harmonic traces (energy ratio, n_harmonics, cepstral prominence) with hypnogram and stage shading
- `notebooks/plots/harmonics/harmonics_stage_boxplots_s1n1.png` — all 4 harmonic features by sleep stage (boxplots with sample counts)
- `notebooks/plots/harmonics/harmonics_dominant_channel_s1n1.png` — dominant channel overall + by-stage stacked bar

### Next steps (Stage 2)
- Run across all 12 sessions to confirm N3 discrimination generalizes
- Statistical testing (Kruskal-Wallis) for stage differences
- Overlay harmonic traces on hypnograms

---

## 2026-05-14 — CAP-only thorax respiratory effort prediction

**Question:** Can CAP temple sensors predict thorax respiratory effort (thorax_resp_rms) without a thorax belt? Which features drive prediction — CAP signal coupling or motion/position?

**Scripts:** `scripts/thorax_predictor_caponly.py`, `scripts/_ablation_quick.py`
**Artifacts:** `artifacts/thorax_caponly_epochs.parquet`, `artifacts/thorax_caponly_results.csv`
**Plots:** `notebooks/plots/thorax_analysis/caponly_*.png` (17 files)

### Feature engineering
Enhanced epoch extraction: 52 per-epoch features (base CAP stats + spectral entropy/power ratios + resp/cardiac rates + CLE-CRE coherence/phase + position/time context) + 25 temporal (lags, rolling, deltas) = 77 total.

### Model results (median R2 across 12 sessions)

| Tier | Features | Within-session R2 | LOSO R2 |
|------|----------|--------------------|---------|
| T0-Ridge | 24 base | -0.543 | 0.189 |
| T1-XGB-Base | 42 base+lags | 0.271 | 0.190 |
| T2-XGB-Enh | 77 all | 0.281 | 0.368 |
| T3-XGB-Rec | 77 + pseudo-lags | 0.201 | 0.254 |
| Ref (w/ thorax lags) | 44 | 0.511 | 0.690 |

### Feature group ablation

| Group | N | LOSO R2 |
|-------|---|---------|
| CAP signal only | 92 | 0.118 |
| Accel only | 4 | 0.279 |
| Context only | 4 | 0.148 |
| Accel+Context | 8 | 0.362 |

### Key findings
1. CAP-only best (Tier 2): LOSO R2=0.368, recovering ~53% of reference with thorax lags
2. Feature importance dominated by epoch_frac (13%), cos_t (7%), roll_deg (6%), movement_rms (5%)
3. **92 CAP signal features (LOSO R2=0.118) are outperformed by 4 accelerometer features (R2=0.279)**
4. Prediction is driven by motion/position/time-of-night, not direct CAP→thorax signal coupling
5. Recursive pseudo-lags hurt due to error accumulation
6. Next: residualize motion to test motion-independent CAP→thorax coupling

---

## 2026-05-14 — Motion-residualized thorax prediction (direct coupling test)

**Question:** After removing the shared motion confound, do CAP temple sensors have any direct predictive power on thorax respiratory effort?

**Script:** `scripts/thorax_residual_analysis.py`
**Artifacts:** `artifacts/thorax_residual_results.csv`
**Plots:** `notebooks/plots/thorax_analysis/residual_*.png` (16 files)

### Method
Per-session Ridge regression: accel (4 features) → thorax_resp_rms. Replace thorax_resp_rms with residual. Same for all 44 CAP features. Then predict residual thorax from residual CAP using XGBoost.

### Motion → thorax R2 per session
Mean R2 = 0.333 (range 0.170–0.557). Motion/position explains ~33% of thorax variance.

### Results on residualized data (median R2)

| Model | Within-session | LOSO |
|-------|----------------|------|
| Ridge-Resid-Base | -0.601 | -0.053 |
| XGB-Resid-Base | 0.045 | -0.029 |
| XGB-Resid-All | 0.040 | 0.002 |
| XGB-Orig-CAP-Only (ref) | 0.192 | 0.169 |

### Key findings
1. **After residualizing motion, CAP features have zero cross-subject predictive power on thorax effort (LOSO R2 ≈ 0)**
2. The CAP→thorax R2=0.118 found in the ablation study was entirely motion-mediated
3. Within-session R2 ~0.04 suggests minimal session-specific overfitting, not real signal
4. Feature importance on residuals dominated by raw_mean (DC level / electrode coupling), not respiratory RMS or spectral features
5. **Conclusion: temple capacitive sensors do not directly measure thorax respiratory effort. The apparent correlation was a confound from body motion/position affecting both signals.**

---

## 2026-04-30 — 3D Projection analysis: 12 CAP-only features, UMAP + t-SNE (S1N1)

**Question:** Can 12 features derived solely from the CAP sensor (CLE-CRE differential) separate PSG sleep stages and apnea events in a 3D UMAP / t-SNE embedding?

**Notebook:** `notebooks/10_projection_cap12.ipynb`
**Plan:** `notebooks/PROJECTION_PLAN.md`
**Previous:** `notebooks/09_projection_3d.ipynb` (40-feature + PCA baseline, kept as reference)

### Feature set (12 CAP-only features from CLE-CRE, 60s window / 30s step)

| # | Feature | Source |
|---|---------|--------|
| 1-4 | delta, theta, alpha, beta band power ratios | Welch PSD on CLE-CRE |
| 5 | spectral_entropy | PSD Shannon entropy |
| 6 | rms | RMS of CLE-CRE signal |
| 7 | resp_rate | `rate_peaks_scaled_resp` / k |
| 8 | resp_rate_std | std of resp rate in sub-windows |
| 9 | card_rate | `rate_hilbert_scaled_cardiac` / k |
| 10 | card_rate_std | std of cardiac rate in sub-windows |
| 11 | acc_rms | accelerometer magnitude RMS |
| 12-13 | resp_amp_cv, breath_interval_cv | amplitude and timing regularity |

Labels: PSG sleep stages (5-class: Wake/N1/N2/N3/REM; 3-class: Wake/Light/Deep+REM), PSG Flow apnea events (Normal/Apnea/Hypopnea).

### Apnea source change (project-wide)
Switched from Effort files to **Flow files** for apnea event parsing:
- `sleep_monitor/config.py`: expanded `APNEA_CODES` with Flow event types (obstructive/mixed/central apnea)
- `sleep_monitor/loader.py`: `_parse_flow_file` replaces `_parse_effort_file`, glob pattern `Flow*.txt`, regex `.+` for multi-word event types, skip non-respiratory events
- S1N1 Flow labels: 828 Normal, 112 Hypopnea, 13 Apnea (fewer than Effort-based: was 731/183/39)

### Results — Baseline (StandardScaler only, no weighting)

**Silhouette scores (UMAP n_neighbors=30):**

| Classes | Score |
|---------|------:|
| 5-class | -0.137 |
| 3-class | -0.112 |

**UMAP sweep (n_neighbors):**
| nn | 5-class sil |
|----|----------:|
| 10 | -0.148 |
| 30 | -0.137 |
| 50 | -0.126 |

**t-SNE sweep (perplexity):**
| perp | 5-class sil |
|------|----------:|
| 15 | -0.145 |
| 30 | -0.127 |
| 50 | -0.117 |

Improvement over 40-feature+PCA baseline: +0.025 to +0.057 across all settings.

### Results — KW-weighted variant (multiply StandardScaled features by normalized Kruskal-Wallis H-stat)

**Silhouette scores (UMAP nn=30):**

| Classes | Unweighted | KW-weighted |
|---------|----------:|----------:|
| 5-class | -0.137 | -0.150 |
| 3-class | -0.112 | -0.111 |

**Per-stage silhouette (UMAP nn=30, 5-class):**

| Stage | Unweighted | KW-weighted | Δ |
|-------|----------:|----------:|---:|
| REM | — | — | +0.018 |
| N3 | — | — | +0.020 |
| N2 | — | — | -0.007 |

KW weighting helped REM and N3 but hurt overall 5-class score. The weighting over-emphasizes motion/breathing irregularity features at the expense of spectral features needed for N2 separation.

### Progression summary

| Setup | UMAP 5-class sil | UMAP 3-class sil |
|-------|------------------:|------------------:|
| 40-feat + PCA (nb 09) | -0.173 | — |
| 12-feat unweighted | -0.137 | -0.112 |
| 12-feat KW-weighted | -0.150 | -0.111 |

### Plots
- `notebooks/plots/projections/cap12_S1N1_feature_boxplots.png` — feature distributions by stage
- `cap12_S1N1_umap3d_nn{10,30,50}_{stage,time}.html` — interactive 3D UMAP colored by stage/time
- `cap12_S1N1_umap3d_nn30_apnea.html` — UMAP colored by apnea events
- `cap12_S1N1_umap3d_nn30_3class.html` — UMAP 3-class coloring
- `cap12_S1N1_tsne3d_p{15,30,50}_stage.html` — interactive 3D t-SNE
- `cap12_S1N1_umap3d_nn30_static.png` / `tsne3d_p30_static.png` — static panels
- `*_kw_*` variants — KW-weighted versions of above

### Key observations
1. Negative silhouette scores across all settings indicate no clean stage clusters in CAP-only feature space — stages overlap substantially.
2. 12 CAP-only features outperform 40 features + PCA by ~0.03-0.06 silhouette, confirming that fewer, more targeted features avoid redundancy-dominated PCs.
3. KW weighting is not beneficial for overall stage separation — it trades N2 quality for marginal REM/N3 improvement.
4. Time-colored UMAP shows overnight trajectory structure, but it does not separate into discrete stage clusters.
5. Apnea events (Hypopnea + Apnea) scatter across the embedding rather than forming a distinct cluster.

### Status
Phase 1-2 complete for S1N1. Phase 3 (cluster analysis) and Phase 4 (cross-session pooled) not started.

---

## 2026-04-30 -- k(t) biomarker analysis (Phases 1-4)

**Question:** Is the scaling factor k, computed as a time series k(t) = raw_CAP_rate / GT_rate per window, a physiological biomarker or just calibration noise?

**Scripts:**
- `notebooks/analysis_k_biomarker.py` (Phase 1+2)
- `notebooks/analysis_k_biomarker_phase3.py` (Phase 3)

**Writeup:** `notebooks/k_biomarker_writeup.md`

**Plots:** `notebooks/plots/k_biomarker/` (15+ files)
**CSV:** `artifacts/k_biomarker_summary.csv`, `artifacts/k_biomarker_correlations.csv`

### Setup
- 60s window, 10s step, all 12 sessions
- k_resp(t) = rate_peaks_loose(pf=0.05, md=0.4s) / GT_Flow_rate
- k_cardiac(t) = rate_hilbert(CARD band) / GT_ECG_rate
- Quality filter: k outside [0.5, 4.0] -> NaN

### Phase 1-2: Temporal characteristics
- k_cardiac: autocorrelation halflife median 1.4 min (slow, physiological)
- k_resp: autocorrelation halflife median 0.5 min (fast, noisy)
- k_cardiac visually tracks sleep architecture on per-session detail panels

### Phase 3: Correlations

**k by sleep stage (pooled, all 12 sessions):**

| Stage | k_resp median | k_cardiac median |
|-------|--------------|-----------------|
| N1    | 1.41         | 1.71            |
| N2    | 1.37         | 1.65            |
| N3    | 1.35         | 1.65            |
| Wake  | 1.30         | 1.61            |
| REM   | 1.31         | 1.58            |

Kruskal-Wallis: resp H=246 p=5.7e-52, cardiac H=609 p=1.6e-130.

**Spearman correlations:**

| k | Biomarker | r | p |
|---|-----------|---|---|
| k_card | SDNN | -0.251 | ~0 |
| k_card | EEG delta | -0.158 | 10^-155 |
| k_card | Acc RMS | +0.159 | 10^-157 |
| k_resp | Acc RMS | +0.290 | ~0 |
| k_resp | Resp CV | +0.131 | 10^-102 |

### Phase 4: Interpretation
- **k_cardiac IS a physiological biomarker.** BCG waveform complexity (number of
  sub-peaks per heartbeat) changes with autonomic tone across sleep stages.
  N1 (autonomic instability) produces the most complex waveform; REM (muscle
  atonia, different hemodynamics) produces the simplest.
- **k_resp is primarily a noise/quality indicator.** Strongest correlation with
  movement. Useful for quality gating but not a standalone biomarker.
- **S6N2 anomaly:** k_cardiac = 0.79 (only session with Hilbert undercounting).
  Likely sensor contact issue or unique vascular anatomy for OS006-N2.
- **Practical:** add k_cardiac to sleep staging feature set; use stage-aware k
  for improved rate calibration.

---

## 2026-04-22 — Ground truth upgrade: ECG R-peaks + Flow peak detection

**Change:** Replaced the old GT pipeline (ACF on bandpassed Thorax/Pleth) with
clinically standard methods:

| Band | Old GT | New GT |
|---|---|---|
| Respiratory | ACF on Thorax bandpass | **Peak detection on Flow (nasal airflow)** via neurokit2 |
| Cardiac | ACF on Pleth bandpass | **Pan-Tompkins R-peak detection on ECG** via neurokit2 |

**New module:** `sleep_monitor/ground_truth.py`
- `gt_resp_rate(session)` → GTResult with breath-level peaks from Flow
- `gt_heart_rate(session)` → GTResult with beat-level R-peaks from ECG
- `gt_sliding_rates(session, win_sec, step_sec)` → sliding-window GT rates
- Automatic fallback to Thorax/Pleth if Flow/ECG fails
- Quality filtering: rejects physiologically impossible intervals

**Validation on S1N1 (win=30s, step=10s):**

| Band | GT signal | Method | Peaks detected | Mean rate |
|---|---|---|---|---|
| Resp | Flow | neurokit2 | 6,939 | 15.7 br/min |
| Cardiac | ECG | Pan-Tompkins | 27,418 | 57.5 BPM |

MAE comparison (CLE-CRE, best CAP channel):
| Band | Method | MAE (old GT) | MAE (new GT) |
|---|---|---|---|
| Resp | spectral | — | 2.04 br/min |
| Resp | peaks | 2.78 | 2.76 br/min |
| Resp | hilbert | — | 2.36 br/min |
| Cardiac | acf | 15.81 | 15.90 BPM |
| Cardiac | hilbert (raw) | 40.42 | 39.76 BPM |

**Why this matters:**
- Flow is the AASM gold standard for respiratory events; Thorax belt is more
  prone to postural artifact.
- ECG R-peaks give true heart rate (beat-level precision); Pleth gives pulse
  rate which diverges during arrhythmias and is delayed by pulse transit time.
- Beat/breath-level GT enables future work on HRV and breath-by-breath analysis.

**Updated scripts:** `scripts/compute_rates.py` now uses `gt_sliding_rates()`
and records which GT signal was used in the metrics parquet.

---

## 2026-04-22 — Methodology update: remove hardcoded ÷2, use per-session k + lenient detection

---

## 2026-04-16 — Cross-session validation of scaled Hilbert for cardiac (all 12 sessions)

**Question:** Validate whether `rate_hilbert / k` (the best single estimator
from the S1N1 tuning analysis) generalises across all 12 sessions. Learn
per-session `k_diag` (50 × 1-min windows) and `k_whole` (whole night) and
compare to baseline `acf` and raw `hilbert`.

**Script:** `notebooks/analysis_hilbert_scaled_all_sessions.py`
**CSV:** `artifacts/hilbert_scaled_per_session.csv`
**Plots:**
- `notebooks/plots/all_sessions_card_hilbert_mae.png`
- `notebooks/plots/all_sessions_card_hilbert_k.png`
- `notebooks/plots/all_sessions_card_hilbert_grid.png`
- per-session traces in `notebooks/plots/per_session_card_hilbert/`

### Setup
- CLE-CRE channel with OLS acc-removal, bp [0.5–3.0] Hz
- GT = ACF on PSG Pleth bandpass (prominence = 0.05)
- Whole-night sliding 1-min window, 5 s step
- k_diag: median of `rate_hilbert(cap) / rate_acf(gt)` across 50 random 1-min windows
- k_whole: same ratio over the full-night sliding grid

### Per-session results

| session | subject | dur (hr) | k_diag | k_whole | IQR | MAE_acf | MAE_hilb | MAE_scaled | bias_scaled | r_scaled |
|---------|---------|---------:|-------:|--------:|----:|--------:|---------:|-----------:|------------:|---------:|
| S1N1 | OS001 | 7.95 | 1.738 | 1.728 | 0.19 | 15.81 | 40.42 | **4.34** | −0.64 | −0.227 |
| S1N2 | OS001 | 7.63 | 1.814 | 1.832 | 0.33 | 10.35 | 41.07 | **4.38** | −1.14 | −0.353 |
| S2N1 | OS002 | 7.73 | 1.925 | 1.885 | 0.18 | 14.40 | 43.12 | **4.07** | −1.51 | −0.159 |
| S2N2 | OS002 | 6.77 | 1.740 | 1.695 | 0.33 | 13.97 | 36.12 | 6.55 | −1.21 | −0.507 |
| S3N1 | OS003 | 6.93 | 1.647 | 1.656 | 0.13 | 18.44 | 38.96 | **3.94** | −0.48 | −0.050 |
| S3N2 | OS003 | 8.66 | 1.652 | 1.669 | 0.14 | 19.99 | 39.46 | **3.59** | −0.55 | −0.371 |
| S4N1 | OS004 | 6.18 | 1.695 | 1.701 | 0.12 | 18.23 | 40.57 | **3.45** | −1.08 | +0.096 |
| S4N2 | OS004 | 6.02 | 1.583 | 1.583 | 0.11 | 20.12 | 36.86 | **3.12** | −0.21 | −0.007 |
| S5N1 | OS005 | 4.11 | 1.680 | 1.662 | 0.15 | 19.30 | 39.87 | **4.25** | −0.00 | +0.082 |
| S5N2 | OS005 | 4.74 | 1.667 | 1.654 | 0.12 | 20.26 | 39.39 | **2.93** | −0.45 | −0.094 |
| S6N1 | OS006 | 5.16 | 1.521 | 1.513 | 0.12 | 23.54 | 31.72 | **4.87** | −1.66 | −0.317 |
| S6N2 | OS006 | 5.78 | 1.495 | 1.484 | 0.18 | 25.11 | 30.96 | **4.79** | −1.47 | −0.061 |
| **median** |  |  | **1.674** | **1.666** | 0.15 | **18.34** | 39.42 | **4.16** | −0.84 | −0.126 |
| **mean ± std** |  |  | **1.680 ± 0.119** | **1.672 ± 0.115** | 0.18 | 18.29 | 38.21 | **4.19 ± 1.02** | −0.87 | — |
| **range** |  |  | [1.50, 1.93] | [1.48, 1.89] | [0.11, 0.33] | [10.4, 25.1] | [31.0, 43.1] | [2.93, 6.55] | — | — |

### Aggregate improvement
- **MAE acf → hilbert/k_whole: 18.29 → 4.19 BPM (−14.10, ~77% reduction)** across all 12 sessions.
- MAE raw hilbert → scaled: 38.21 → 4.19 BPM (−89%).
- Every session improves. Largest wins on S6N1/N2 (Δ ≈ 19–20 BPM) and S3/S4/S5 (Δ ≈ 14–17 BPM).
- Worst residual: S2N2 at 6.55 BPM (still less than half of baseline acf 13.97).

### Per-subject k stability (key for deployment)

| subject | N1 k | N2 k | Δ |
|---------|-----:|-----:|--:|
| OS001   | 1.73 | 1.83 | 0.10 |
| OS002   | 1.89 | 1.70 | 0.19 |
| OS003   | 1.66 | 1.67 | **0.01** |
| OS004   | 1.70 | 1.58 | 0.12 |
| OS005   | 1.66 | 1.65 | **0.01** |
| OS006   | 1.51 | 1.48 | **0.03** |

- 3 / 6 subjects have night-to-night Δk ≤ 0.03.
- 2 / 6 subjects (OS001, OS004) have Δk ≈ 0.10–0.12.
- 1 / 6 subjects (OS002) has Δk = 0.19 — likely body-position or coupling change.
- Overall: **per-subject k is roughly stable but not as clean as resp's bipartite split.** A 50-minute calibration per subject per night is safer than a once-per-subject calibration.

### k_diag vs k_whole
Essentially identical per session (|Δ| ≤ 0.04 on every session). **50 random 1-min windows are fully sufficient to calibrate k** — no need to process the whole night. This matches the resp finding.

### Key observations
1. **Scaled hilbert robustly delivers MAE ~3–5 BPM** (median 4.2) across every
   subject and night — 77% better than the current baseline. The one outlier
   (S2N2 at 6.5 BPM) is still sub-baseline.
2. **k is NEVER at the "÷2" value** (range 1.48–1.93). The naive "halve the
   peak count" would over-correct everywhere.
3. **Subject-stable overcount factor.** k clusters by subject (OS003, OS005
   around 1.66; OS006 at 1.50; OS002 near 1.90), mirroring the resp finding
   that coupling/geometry drives the ratio.
4. **Pearson r remains weak-to-negative on many sessions** (worst: S2N2
   r=−0.51, S3N2 r=−0.37). Scaling corrects the mean, not the window-level
   variation. GT itself varies only σ≈4 BPM across the night, so r is hard
   to raise even with a perfect estimator.
5. **ACF baseline still fails across the board** — all 12 sessions have
   acf MAE > 10 BPM with near-zero r. The sub-harmonic lock-in issue is
   not a one-session fluke.

### Recommendation
- Add `rate_hilbert_scaled_cardiac(x, k)` to `sleep_monitor/rates.py`.
- Default k = **1.67** (the cross-session median) when no calibration is
  available.
- Provide a `calibrate_k_hilbert_cardiac(session, n=50)` helper that returns
  per-session k; run it once per night before evaluation.
- Register in `_ESTIMATOR_CHOICES` and treat as the default cardiac estimator
  (replacing raw ACF).
- Caveat: correlation-heavy applications (apnea detection, HR variability)
  need a different method — this is a whole-night-average estimator.

---

## 2026-04-16 — Tuned cardiac pipeline + learned scaling factors on S1N1

**Question:** Apply the tuning suggestions from the earlier cardiac default run
(ACF prom 0.10→0.05, win 30→60 s, spectral nperseg=fs·8, fix envelope) and,
instead of a hard-coded ÷2, learn a per-method scaling factor `k` from data
(as done for resp). Evaluate on 5 windows and on the full night.

**Script:** `notebooks/analysis_card_tuned_s1n1.py`
**Plot:** `notebooks/plots/card_tuned_s1n1.png`

### Tuning applied
| Parameter | Baseline | Tuned | Reason |
|---|---|---|---|
| ACF prominence (window & envelope) | 0.10 | 0.05 | Recover NaN windows |
| Sliding window | 30 s | 60 s | More cycles → cleaner ACF |
| Welch nperseg | `fs·4` | `fs·8` | Freq res 0.125 Hz (cf. 15 BPM snap) |
| Envelope input band | [0.5, 3.0] Hz (cardiac-bp, then TK) | **[3, 20] Hz (HF-bp, then TK)** | Already-bp signal has no amplitude modulation; HF band isolates BCG pulse energy so the TK envelope oscillates at HR |
| Envelope smoothing | `fs/f_hi` samples | `0.3 s` | Tuned for HR envelope |

### Scaling factors learned on N=50 random 1-min windows
`k = median(rate_method / rate_gt_acf)` — applied only to over-counting methods.

| method    | k_median | IQR             | range           | N valid |
|-----------|---------:|-----------------|-----------------|--------:|
| spectral  | 1.127    | [0.79, 1.80]    | [0.49, 2.57]    | 50 |
| acf       | 0.849    | [0.72, 0.99]    | [0.58, 2.65]    | 49 |
| **hilbert** | **1.738** | **[1.65, 1.83]** | [0.97, 2.01] | 50 |
| **zerocross** | **1.920** | **[1.76, 1.99]** | [1.06, 2.19] | 50 |
| **peaks** | **1.930** | **[1.77, 2.08]** | [1.12, 2.19] | 50 |
| envelope  | 0.971    | [0.75, 1.19]    | [0.60, 1.78]   | 38 |

- `peaks / zerocross / hilbert` cluster at k ≈ 1.74–1.93 with **tight IQR**
  (≤0.22) — a stable, near-2× overcount that maps cleanly onto a scalar.
  Unlike resp (k≈1.3) the BCG signal really does produce ~2 detectable
  peaks per cardiac cycle (systolic + dicrotic-like bump).
- `spectral` and `envelope` have wide IQR — the rate they return is not
  always the same multiple of GT, so a single scalar won't help them.
- `acf` is close to 1.0 but with heavy tail (IQR lower bound 0.72 → half-rate
  lock-ins). Median-based k doesn't help; the failure mode is bimodal.

### 5-window aggregate MAE (BPM) with tuned + scaling
| method         | variant            | MAE   | RMSE  | bias   |
|----------------|--------------------|------:|------:|-------:|
| spectral       | tuned              |  7.98 | 12.48 |  +6.40 |
| acf            | tuned              | 20.27 | 33.13 |  +8.73 |
| hilbert        | tuned              | 43.49 | 43.71 | +43.49 |
| zerocross      | tuned              | 53.74 | 53.98 | +53.74 |
| peaks          | tuned              | 55.37 | 55.64 | +55.37 |
| envelope       | tuned (HF-band)    | 17.54 | 19.54 |  −3.07 |
| **peaks**      | **tuned / k=1.93** | **3.87** | **4.50** | **+0.68** |
| **zerocross**  | **tuned / k=1.92** | **3.76** | **4.10** | **+0.14** |
| **hilbert**    | **tuned / k=1.74** | **3.38** | **3.94** | **+0.36** |

### Whole-night sliding (win=60 s, step=5 s, N=5713 windows)
| estimator              | MAE    | RMSE   | bias    | r      | cov   |
|------------------------|-------:|-------:|--------:|-------:|------:|
| spectral (tuned)       | 23.31  | 31.74  | +12.92  | −0.056 | 100%  |
| acf (tuned)            | 15.81  | 23.98  |  −2.04  | −0.008 |  99%  |
| hilbert (tuned)        | 40.42  | 41.21  | +40.34  | −0.227 | 100%  |
| zerocross (tuned)      | 48.72  | 49.49  | +48.67  | −0.191 | 100%  |
| peaks (tuned)          | 48.82  | 50.60  | +48.33  | −0.121 | 100%  |
| **envelope (tuned-HF)**| **16.84** | 21.23 | +0.79 | −0.014 |  **75%** |
| **peaks / k=1.93**     | 5.76   |  9.03  |  −2.40  | −0.121 | 100%  |
| **zerocross / k=1.92** | **4.48** |  6.35 | −1.93 | −0.191 | 100%  |
| **hilbert / k=1.74**   | **4.33** |  6.18 | −0.96 | −0.227 | 100%  |

### Key findings

1. **Scaling factor works spectacularly for the overcounting trio.** Whole-night
   MAE drops **40 → 4 BPM** (`hilbert`, `zerocross`, `peaks`), bias collapses
   to within ±2 BPM, coverage stays 100%. Applied per-method k from just 50
   random 1-min windows held up across 5700+ windows.
2. **Best single estimator is now `hilbert / k=1.74`** at MAE = 4.33 BPM —
   a ~73% reduction vs the 15.73 BPM baseline-ACF default.
3. **Envelope fix worked.** Switching input from already-bp cardiac [0.5–3 Hz]
   to HF [3–20 Hz] changed it from 0% valid → 75% coverage and MAE 16.84 BPM
   with bias near zero. The old version ran TK on a near-sinusoid, which
   produces a ~constant envelope (no ACF period); the HF version runs TK on
   the impulsive BCG content and the resulting envelope modulates at HR.
4. **Lowering ACF prominence (0.10 → 0.05) did NOT help ACF meaningfully**:
   15.73 → 15.81 BPM. ACF's failure mode is half-period / sub-harmonic
   lock-in, not missing peaks. Needs a different fix (e.g. restrict ACF lag
   range or pick the first-peak rather than most-prominent peak).
5. **Longer window for spectral HURT whole-night** (went from no prior
   number directly to MAE 23.31 with bias +12.9 BPM). Finer frequency
   resolution lets spectral lock more precisely onto the 2× harmonic when
   it's stronger than the fundamental. On the 5-window view (MAE 8.0) it
   looked fine, so the per-window dependence matters a lot.
6. **Pearson r is still ~0 for every method** — scaling shifts bias toward
   zero but does not improve per-window responsiveness (the point-wise
   error is dominated by sub-harmonic flips, not a constant offset). Same
   bias-vs-correlation trade-off seen for resp.
7. **Per-method `k` is wildly different across methods but tight within a
   method** (IQR ≤ 0.22 for the three overcounting methods). Matches resp
   observation that the ratio is a subject/coupling-stable quantity.

### Remaining issues to address next
- **ACF sub-harmonic lock-in is the core failure mode.** Both tighter
  prominence and longer window left MAE unchanged. Options: constrain the
  ACF lag search to `[0.9/r_prior, 1.1/r_prior]` using a rolling prior,
  or detect bi-modality of ratios and flip-correct.
- **Cross-session validation of k.** The resp writeup showed `k` is
  subject-stable. Confirm the same for cardiac on the other 11 sessions
  before baking these constants into the pipeline.
- **Cardiac `peaks_scaled` estimator.** If validation holds, add
  `rate_peaks_scaled_cardiac(x, k)` to `rates.py` and register in the grid.

---

## 2026-04-16 — Per-method cardiac-rate sanity check on 5 random 1-min windows of S1N1

**Question:** Run cardiac rate detection with defaults across the whole S1N1
night and also surface results for 5 random 1-min windows to guide parameter
tuning.

**Script:** `notebooks/analysis_card_window_methods_s1n1.py`
**Plot:** `notebooks/plots/card_window_methods_s1n1.png`

### Setup
- S1N1, CLE-CRE channel with OLS acc-removal, bp [0.5–3.0] Hz
- Whole-night default pipeline: `cardiac_CLE-CRE_ols_acf_w30` (30 s / 5 s)
- 5 random 1-min windows (seed=42), starts at 0.71, 3.44, 3.48, 5.19, 6.14 hr
- GT = ACF on PSG Pleth bandpass, same window
- 6 estimators (spectral, acf, hilbert, zerocross, peaks, envelope)

### Whole-night metrics (default pipeline)
| Metric | Value |
|---|---|
| windows valid / total | 5569 / 5719 (cov 97.4%) |
| MAE  | 15.73 BPM |
| RMSE | 23.22 BPM |
| bias | −1.46 BPM |
| r    | −0.008 |
| p50/p90 \|err\| | 11.68 / 25.53 BPM |
| pred μ±σ | 55.46 ± 22.76 BPM |
| ref  μ±σ | 56.92 ±  4.15 BPM |
| quality mean / median | 0.650 / 0.640 |

### Per-window rates (BPM)
| win | t (hr) | GT   | spectral | acf   | hilbert | zerocross | peaks | envelope |
|-----|--------|------|----------|-------|---------|-----------|-------|----------|
| 1   | 0.71   | 64.3 | 120.0    | 56.4  | 96.5    | 106.0     | 107.4 | **NaN**  |
| 2   | 3.44   | 58.8 |  45.0    | 41.7  | 100.3   | 111.3     | 110.9 | **NaN**  |
| 3   | 3.48   | 58.4 | 105.0    | 55.1  | 102.2   | 114.5     | 111.5 | **NaN**  |
| 4   | 5.19   | 54.0 |  90.0    | 46.7  | 102.6   | 110.1     | 112.2 | **NaN**  |
| 5   | 6.14   | 51.0 |  75.0    | NaN   | 100.8   | 110.9     | 110.0 | **NaN**  |

### Per-method aggregate across 5 windows (BPM)
| method    | MAE   | RMSE  | bias   | valid |
|-----------|-------|-------|--------|-------|
| spectral  | 35.23 | 38.31 | +29.72 | 5 / 5 |
| **acf**   | **8.89** | **10.23** | **−8.89** | 4 / 5 |
| hilbert   | 43.19 | 43.64 | +43.19 | 5 / 5 |
| zerocross | 53.29 | 53.65 | +53.29 | 5 / 5 |
| peaks     | 53.13 | 53.44 | +53.13 | 5 / 5 |
| envelope  | NaN   | NaN   | NaN    | 0 / 5 |

### Key findings
1. **Double-counting dominates.** Hilbert / zerocross / peaks all land at
   ~100–115 BPM, roughly **2×** the ~55 BPM GT. The plots show the CLE-CRE
   bandpass has strong dicrotic / secondary bumps per cardiac cycle, so any
   time-domain peak- or crossing-based method over-counts by a factor of two,
   exactly like the double-peak issue documented for the resp band.
2. **ACF wins** (MAE ≈ 9 BPM, all negative bias ≈ −9 BPM → mild under-estimate).
   This is the only method that locks onto the fundamental period instead of
   individual peaks. However 1 / 5 windows returned NaN (prominence=0.10
   likely too high for a noisy 60 s window).
3. **Spectral is unreliable at 1-min windows.** Welch with `nperseg=min(N, 4 s)`
   gives ~0.25 Hz frequency resolution = 15 BPM steps, so the output snaps to
   45/75/90/105/120 BPM — the coarse grid shows up directly in the errors.
4. **Envelope returned NaN on every window** — `rate_envelope` is defaulting
   to `env_lo=0.5, env_hi=3.0` and running ACF with prominence=0.10 on the
   Teager-Kaiser envelope. Appears to be broken / too strict for 60 s at this
   channel — needs investigation.
5. **Whole-night pred σ (22.8) vs GT σ (4.2)** confirms the pipeline is
   frequently flipping between fundamental and half-period / noise, which
   erases correlation entirely (r ≈ 0). This is fixable.

### Tuning suggestions
- **Default to `acf` for cardiac** (already the default) but address the
  NaN-rate: lower `rate_acf` prominence from 0.10 → ~0.05, or add a fallback
  that widens the lag search when no peak clears the threshold.
- **Halve-correction layer:** for hilbert/zerocross/peaks, if pred > ~1.8× a
  robust prior (e.g. ACF or rolling median of recent windows), divide by 2.
  This would instantly bring three methods into ballpark range.
- **Longer windows for cardiac spectral.** With `win_s=30` and nperseg capped
  at 4 s, the resolution is too coarse. Either raise `win_s` to 60 s and
  `nperseg` to `fs*8`, or drop spectral from the cardiac estimator set.
- **Investigate `rate_envelope`.** Produces NaN on every tested window; needs
  a quick debug — likely the prominence on the TK envelope's ACF. This is
  specifically the cardiac-targeted estimator, so fixing it is high leverage.
- **Longer window for ACF too:** moving win_s from 30 s → 45–60 s should help
  ACF lock onto the fundamental more reliably given the noisy CLE-CRE signal.
- **Fusion with double-peak-aware weights:** since hilbert/zerocross/peaks
  reliably overcount by 2×, a ÷2 pre-correction before fusion would let them
  vote sensibly with ACF instead of being thrown out by the band-clamp.

---

## 2026-04-15 — Default rate detection on S1N1 (full night)

**Question:** Run the default rate-estimation pipeline on one full session and report statistics.

**Script:** `notebooks/analysis_default_rates_s1n1.py`
**Plot:** `notebooks/plots/default_rates_s1n1.png`

### Parameters (PipelineConfig defaults)
| Parameter | Value |
|---|---|
| Session | S1N1 (OS001-KJK 2024-09-17, 7.95 hr) |
| Channel | CLE-CRE |
| Preproc | ols (acc artifact removal) |
| Estimator | acf |
| Window / step | 30 s / 5 s |
| Quality gate | 0.0 (no gating) |
| GT reference | PSG Thorax (resp), Pleth (cardiac), ACF on same window grid |

### Results
| Band | n_used / n_total | coverage | MAE | RMSE | bias | r | p50\|err\| | p90\|err\| |
|---|---|---|---|---|---|---|---|---|
| resp (br/min)    | 5488 / 5719 | 96.0% | 5.45  |  6.38 | −4.52 |  0.009 |  5.99 |  9.36 |
| cardiac (BPM)    | 5569 / 5719 | 97.4% | 15.73 | 23.22 | −1.46 | −0.008 | 11.68 | 25.53 |

Pred vs GT distribution:
- resp: pred μ=11.18 σ=3.82 vs GT μ=15.69 σ=2.43 br/min
- cardiac: pred μ=55.46 σ=22.76 vs GT μ=56.92 σ=4.15 BPM
- Mean window quality: resp 0.857, cardiac 0.650

### Key findings
1. Defaults underperform on this session — both bands have r≈0 against PSG GT.
2. Resp prediction is biased ~4.5 br/min low and pred-σ is ~1.6× GT-σ, suggesting the
   ACF picks half-period or sub-harmonic frequently (consistent with the 2026-04-12
   note that the resp bandpass passes both inhalation and exhalation peaks).
3. Cardiac is much worse — pred-σ is ~5× GT-σ, indicating ACF is locking onto noise
   /motion artifacts rather than the BCG fundamental at 30 s window without envelope.
4. No quality gating was applied; raising the gate (or using `envelope` for cardiac /
   `÷2` correction for resp) is the obvious next step before declaring failure.

---

## 2026-04-15 — Per-method resp-rate sanity check on 5 random 2-min windows of S1N1

**Question:** Inspect raw per-method outputs on a handful of 2-min resp windows
to decide where adjustments are needed. Show whether the double-peak (÷2) issue
documented 2026-04-12 is present at the 2-min scale.

**Script:** `notebooks/analysis_resp_window_methods_s1n1.py`
**Plot:** `notebooks/plots/resp_window_methods_s1n1.png`

### Setup
- S1N1, full night, CLE-CRE channel with OLS acc-removal, bp [0.1–0.5] Hz
- 5 random 2-min windows (seed=42), starts at 0.71, 3.43, 3.47, 5.18, 6.13 hr
- GT = ACF on PSG Thorax bandpass, same window

### Per-window rates (br/min)
| win | start_hr | GT | spectral | acf | hilbert | zerocross | peaks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.71 | 16.03 | 15.0 |  NaN  | 12.97 | 14.05 | 14.05 |
| 2 | 3.43 | 18.36 | 15.0 | 11.08 | 17.30 | 19.13 | 19.98 |
| 3 | 3.47 | 18.27 | 15.0 |  8.92 | 16.38 | 17.36 | 18.94 |
| 4 | 5.18 | 14.95 | 15.0 |  7.23 | 16.11 | 16.97 | 16.65 |
| 5 | 6.13 |  8.16 | 15.0 |  8.57 | 13.40 | 14.29 |  6.75 |

### Per-method aggregate (5-window MAE / RMSE / bias, br/min)
| method | MAE | RMSE | bias | MAE÷2 | bias÷2 |
|---|---:|---:|---:|---:|---:|
| spectral  | 2.91 | 3.74 | −0.15 | 7.65 | −7.65 |
| acf       | 6.19 | 7.07 | −5.98 | 10.46| −10.46|
| hilbert   | 2.48 | 2.93 | +0.08 | 7.54 | −7.54 |
| zerocross | 2.36 | 3.06 | +1.21 | 6.98 | −6.98 |
| **peaks** | **1.47** | **1.54** | +0.12 | 7.52 | −7.52 |

### Key findings
1. **Double-peak ÷2 is NOT happening at the 2-min scale.** spectral, hilbert,
   zerocross, peaks are all near GT (bias ≈ 0). Applying ÷2 *worsens* every
   method by ~7 br/min. So the 2026-04-12 hypothesis (band passes both halves)
   does not generalise — it appears to be a window-/state-specific artifact.
2. **`spectral` is stuck at 15.0 br/min in every window.** With `nperseg = fs·4 = 400`
   samples, freq resolution = fs/nperseg = 0.25 Hz/2 = 0.0125 Hz → bin spacing
   ≈ 0.75 br/min. But the dominant bin is identical across very different GTs,
   suggesting the Welch peak is being snapped to a sub-band edge or the PSD is
   flat enough that a single bin dominates. Worth instrumenting next.
3. **`acf` is the worst** (MAE 6.2, bias −6.0). It under-estimates by 30–60% in
   4/5 windows and produces NaN once. Yet the *ground-truth pipeline uses ACF*
   on Thorax — Thorax must be cleaner / more sinusoidal than CLE-CRE for ACF
   to lock cleanly. Using ACF on CLE-CRE is the wrong default estimator for
   this band on this signal.
4. **`peaks` is the best per-window** (MAE 1.47, bias 0.12) and matches GT to
   within 2 br/min on 4/5 windows. Win 5 is the outlier (peaks 6.75 vs GT 8.16,
   −1.4) but the GT itself is anomalous (8 br/min while every other method
   says 13–14) — see plot, the segment has a long flat region with low
   variation that confuses both Thorax-ACF and the cap signal.
5. **`hilbert` and `zerocross` are competitive** with peaks (MAE ~2.4) and have
   the advantage of being parameter-free relative to peak-prominence tuning.

### Implications for the default pipeline
- Switching default `estimator` from `acf` → `peaks` (or `hilbert`/`zerocross`)
  would likely close the −4.5 br/min bias seen in the full-night run.
- The Welch `spectral` estimator needs a closer look — it's pinned at one
  value across very different signals.
- Whole-night (96 % coverage) ACF result was much worse than these 5 windows
  suggest, so window state matters: a quality gate or window-length sweep is
  the next experiment.

---

## 2026-04-15 — Peak-ratio histogram + sensitivity sweep on S1N1 (Phases 1+2+4+5)

**Question:** Test the hypothesis that each true breath produces 2 CAP peaks (in/out),
so increasing peak-detection sensitivity + dividing by a learned constant `k` should
recover GT.

**Script:** `notebooks/analysis_peak_ratio_sweep_s1n1.py`
**Plots:**
- `notebooks/plots/phase1_ratio_histogram_s1n1.png`
- `notebooks/plots/phase2_sweep_heatmaps_s1n1.png`
- `notebooks/plots/phase4_wholenight_s1n1.png`
- `notebooks/plots/phase5_visual_sanity_s1n1.png`

### Phase 1 — ratio histogram (N=50 random 2-min windows, default sensitivity)
- ratio (n_cap / n_thorax)  **median = 1.00**, mean 0.90, IQR [0.78, 1.06]
- 70% of windows in [0.8, 1.2];  **0%** near 2
- → **CAP produces ~1 peak per Thorax breath, not 2.** The double-peak hypothesis
  is *not* the dominant regime at this window scale on S1N1.

### Phase 2 — sweep on the same 50 windows (prom_factor × min_dist_s)
- Best: pf=0.05, md=0.4s, **k=1.21**, MAE=1.69 br/min, ratio IQR=0.15
- All "looser" settings (pf<=0.20) cluster around k∈[1.06, 1.21], confirming there is
  no operating point where the count doubles. Tighter sensitivity (pf=0.40, md=1.8s)
  → k≈1.0 (one peak per breath, the natural regime).

### Phase 4 — whole-night S1N1 (30 s windows / 5 s step, 5719 windows)
| method | pf | md | k | MAE | RMSE | bias | r | cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-peaks    | 0.40 | 1.80 | 1.00 | 2.78 | 3.83 | −0.26 | +0.345 | 99.9% |
| peaks-scaled (Phase-2 k) | 0.05 | 0.40 | 1.21 | 2.49 | 3.27 | +1.02 | +0.281 | 99.9% |
| peaks-scaled-kAll | 0.05 | 0.40 | 1.25 | **2.34** | — | +0.50 | — | 99.9% |

- Whole-night ratio median = 1.25 (vs 1.00 on the 2-min Phase-1 set) — k is window-length
  sensitive: shorter windows have more noise-driven extra CAP peaks.
- Modest improvement over baseline (MAE 2.78 → 2.34, ~16%); correlation actually drops
  slightly (r 0.345 → 0.281), so the scaled method tracks per-window variations *less*
  well even though its mean error is smaller.

### Phase 5 — visual sanity (5 random 2-min windows)
- Red CAP-new dots correspond ~1:1 with green Thorax dots in every window.
- No window shows a clean "two CAP peaks per Thorax peak" pattern.

### Conclusion
- The hypothesis (each breath → 2 CAP peaks → ÷2) is **empirically wrong on S1N1 at
  the 2-min and 30 s scales**. CAP produces ~1 peak per breath; tighter sensitivity
  inflates the count by a modest ~25% which a learned k=1.25 corrects.
- Net improvement on whole-night MAE is real but small (16%) and S1N1-specific;
  not yet integrated. Phase 6 awaits decision: validate on a second session before
  introducing as a new estimator.

---

## 2026-04-15 — Cross-session validation of scaled-peaks estimator (12 sessions)

**Question:** Does the scaled-peaks estimator (loose detection ÷ learned k) generalise
across all 12 sessions, and is k stable enough to be a useful per-subject constant?

**Script:** `notebooks/analysis_peak_ratio_all_sessions.py`
**Writeup:** `notebooks/peak_ratio_method_writeup.md` (presentation-ready)
**Plots:**
- `notebooks/plots/all_sessions_mae_bars.png`
- `notebooks/plots/all_sessions_k_consistency.png`
- `notebooks/plots/all_sessions_grid.png`
- `notebooks/plots/per_session_rate_plots/S*_resp_rates.png` (12 files)
- `artifacts/peak_ratio_per_session.csv`

### Setup
- 1-min windows (both diagnostic and whole-night), 5-s step
- Diag: 50 random windows → k_diag = median(n_cap_loose / n_thorax)
- Whole-night: same statistic over all sliding windows → k_whole
- Loose detector: pf=0.05, md=0.4s; baseline: pf=0.4, md=1.8s

### Results
- All 12 sessions processed. k clusters in [1.18, 1.61], median 1.31, std 0.15.
- k_diag and k_whole agree to within 0.07 per session — 50 minutes is enough
  to calibrate.
- **Mean MAE: 3.09 → 2.20 br/min (−25.3 %)**, 11/12 sessions improved.
- Bias collapses toward 0 in every session.
- Pearson r drops in every session (loose detector → less per-window specificity).
- Subjects cluster: OS001/OS002/OS005-N1/OS006 → k≈1.2-1.3;
  OS003/OS004/OS005-N2 → k≈1.4-1.6 (likely body-position / coupling-gain effect).

### Decision
Result presented in `peak_ratio_method_writeup.md`. Integration deferred until
team accepts the bias-vs-correlation trade-off. Per-subject k ≈ session-stable;
default global k = 1.31 if integrated.

## 2026-04-12 — Breathing Rate: ACF-constrained peak detection (window tuning)

**Question:** Replicate the ACF-based peak detection approach from `analysis_rates.ipynb` where every peak
in the filtered CAP signal is detected and BR = n\_peaks ÷ 2. Show tuning parameters on a short window
before applying to whole night.

**Script:** `notebooks/analysis_br_acf_peaks.py`
**Plot:** `notebooks/plots/br_acf_peaks_tuning.png`

### Parameters used
| Parameter | Value | Description |
|---|---|---|
| Subject / Night | OS001 / N1 | S1N1 recording |
| Window | 2.00–2.050 hr | 3-minute inspection window |
| Channel | CLE-CRE | OLS regression differential (CLE_bp − β·CRE_bp) |
| BP_LO / BP_HI | 0.1 / 0.5 Hz | Butterworth bandpass (6–30 br/min) |
| BP_ORDER | 3 | Filter order |
| ACF_PROM | 0.10 | Prominence threshold for dominant ACF lag |
| MIN_SEP_SCALE | 0.45 | min\_distance = T × 0.45 → catches ~2 peaks/cycle |
| PROM_FACTOR | 0.25 | Peak prominence = 0.25 × std(filtered) |
| SMOOTH\_WIN\_SEC | 0.3 | Pre-detection smoothing (seconds) |
| RATE\_DIVISOR | 2 | BR = n\_peaks / 2 per minute |
| ACC\_REMOVAL | True | OLS regress accelerometer out of each channel |

### Results (3-minute window, S1N1)
| Metric | Value |
|---|---|
| ACF dominant period T | 3.71 s |
| Implied actual breath period | ~7.42 s (= 2T, since filter captures both inhalation + exhalation peaks) |
| ACF-implied BR | ~8.1 br/min |
| CAP detected peaks | 53 in 3 min |
| CAP rate (÷2) | **8.8 br/min** |
| PSG Thorax GT peaks | 50 in 3 min |
| GT rate (÷2) | **8.3 br/min** |
| Error | 0.5 br/min (~6%) |

### Key findings
1. **Why ÷2 works:** The [0.1–0.5 Hz] bandpass passes both the inhalation and exhalation
   half-cycles as separate peaks. The ACF therefore detects T ≈ 3.71 s, which is the
   *half-period* of the true breathing cycle (~7.4 s). Setting min\_sep = 0.45T ≈ 1.67 s
   allows both peaks per cycle to be detected, and dividing by 2 recovers the correct rate.

2. **MIN\_SEP\_SCALE sensitivity (from sweep 0.30 → 0.85):**
   - 0.30: more peaks detected (risk of false positives from noise)
   - **0.45: optimal — catches both peaks, ÷2 matches GT well**
   - 0.65: starts missing the smaller (exhalation) peak — rate drops
   - 0.85: only ~1 peak per cycle → ÷2 gives ~half the true rate (incorrect)

3. **Next step:** Apply approved parameters to whole night. Pending user approval.

---

## 2026-04-12 — Multi-channel BR: CLE + CRE + CLE-CRE, more sensitive detection

**Question:** Add CLE and CRE alongside CLE-CRE. Make detection more sensitive to catch
small second peak of double-peaked crests. Determine whether additional scaling (changed
RATE_DIVISOR) is needed.

**Script:** `notebooks/analysis_br_acf_peaks.py` (updated)
**Plot:** `notebooks/plots/br_acf_peaks_multichannel.png`

### Per-channel tuning parameters

| Channel | MIN_SEP_SCALE | PROM_FACTOR | SMOOTH_WIN_SEC | Reason |
|---|---|---|---|---|
| CLE | 0.35 | 0.15 | 0.15 s | Clean signal, slightly lenient |
| CRE | **0.22** | **0.12** | **0.0** | Double peaks ~1s apart need smaller scale + no smoothing |
| CLE-CRE | 0.35 | 0.15 | 0.15 s | Regression diff is clean |

### Results (3-minute window, S1N1) — GT rate = 8.3 br/min

| Channel | n\_peaks | emp\_div | rate ÷ 2 | rate ÷ emp\_div | vs GT |
|---|---|---|---|---|---|
| CLE | 57 | 2.28 | 9.5 br/min | **8.3 br/min** | +1.2 |
| CRE | 62 | 2.48 | 10.3 br/min | **8.3 br/min** | +2.0 |
| CLE-CRE | 55 | 2.20 | 9.2 br/min | **8.3 br/min** | +0.9 |
| GT Thorax | 50 | 2.00 | 8.3 br/min | 8.3 br/min | ref |

### Key findings

1. **Does detection work?** Yes — with empirical divisor, all three channels recover exactly 8.3 br/min.

2. **Is additional scaling required?** **Yes, specifically for CRE.**
   - CRE has genuine close double peaks separated by ~1.0–1.1 s on some breath crests.
   - These are resolved with `MIN_SEP_SCALE=0.22` (min distance ≈ 0.82s at T≈3.71s).
   - With no smoothing on CRE, the ACF cannot find a clean period (T=n/a); the
     double peaks make the signal aperiodic to the ACF. Fallback to BP_HI-based
     min distance applies instead.
   - emp_div ≈ **2.48** for CRE → RATE_DIVISOR = 2.5 (or ~3 rounded) needed.

3. **Channel comparison (within this window):**
   - **CLE-CRE** best: cleanest signal, ACF stable (T=3.71s), emp_div closest to 2 (2.20).
   - **CLE** good: clean, stable ACF (T=3.74s), emp_div = 2.28.
   - **CRE** noisiest: ACF fails, double-peak structure forces higher divisor.

4. **The red-highlighted zoom panel** shows CRE double-peak pairs (peaks < 1.5s apart)
   as shaded regions — clearly different from CLE and CLE-CRE.

5. **Next step:** Decide on RATE_DIVISOR per channel, then apply to whole night.
   Pending user approval.

---

## 2026-04-16 — Library: scaled rate estimators + per-session k calibration

- Added to `sleep_monitor/rates.py` (exported from the package):
  - `rate_hilbert_scaled_cardiac(x, k, fs, f_lo, f_hi)` — cardiac HR from Hilbert inst. frequency / k.
  - `rate_peaks_scaled_resp(x, k, fs, prom_factor=0.05, min_dist_s=0.4)` — loose-peak resp rate / k.
  - `calibrate_k_cardiac(session, n_windows=50, win_s=60, seed=42)` → median ratio vs ACF on Pleth.
  - `calibrate_k_resp(session, ...)` → same, vs ACF on Thorax.
- Both estimators return `nan` for non-positive / non-finite k (guard-railed).
- Sanity run on S1N1 (`notebooks/demo_scaled_estimators.py`):
  - `k_cardiac = 1.738` (matches tuned analysis; 12-session median 1.67, range [1.48, 1.93])
  - `k_resp    = 1.244` (matches resp peak-ratio analysis; cross-session median ~1.3)
  - Mid-night test window: cardiac err +2.6 BPM, resp err +1.2 BPM.
- Usage pattern: calibrate once per night → reuse k on every window of that night.
  Per-night recommended (observed |Δk| up to 0.19 night-to-night on some subjects).

---

## 2026-04-22 — Methodology update: remove hardcoded ÷2, use per-session k + lenient detection

**Change:** Removed all hardcoded `÷2` (RATE_DIVISOR=2) assumptions from code
and analysis scripts. The original hypothesis — each breath produces exactly 2
CAP peaks — was disproven across 12 sessions (actual k ranges from 1.18 to 1.93
depending on subject, band, and coupling geometry).

**New standard approach:**
1. Use **lenient peak detection** (prom_factor=0.05, min_dist_s=0.4 for resp;
   similar for cardiac) to capture all physiological events including secondary
   peaks. Tight detection misses real events; it's better to over-detect and
   correct with k.
2. Calibrate **k per session** (or per phase within a session) using
   `calibrate_k_resp()` or `calibrate_k_cardiac()` — 50 random 1-min windows
   are sufficient.
3. Apply `rate = raw_rate / k` where k is a continuous value, not rounded to
   an integer.

**Code changes:**
- `morphology.py`: `compute_rate_divisor()` now returns a continuous float
  instead of rounding to int. `band_events_to_rates()` defaults to k=1.0
  instead of k=2 for resp.
- `analysis_br_acf_peaks.py`: replaced `RATE_DIVISOR` with `RATE_K`.
- `analysis_resp_window_methods_s1n1.py`: removed the `÷2` comparison columns.

**Rationale:** The empirical k values cluster by subject/coupling (OS001-OS002
around 1.2–1.3 for resp, 1.7–1.9 for cardiac; OS003-OS006 around 1.4–1.6 for
resp, 1.5–1.7 for cardiac). A fixed ÷2 over-corrects on every session tested.
Per-session k calibration reduced resp MAE by 25% and cardiac MAE by 77%.

---

## (Retroactive) Infrastructure scripts — undated, created pre-2026-04-30

The following scripts and notebooks were created during the project but not logged at the time. Entries are grouped by purpose; exact creation dates are not recoverable.

---

### Utility: `scripts/add_psg.py`

**Purpose:** One-time script to programmatically add PSG data paths, stage labels, and constants to `analysis_raw.ipynb`. Not an analysis — a code-generation helper.

**Status:** Used once; can be deleted or archived.

---

### CLI: `scripts/cap_rates.py`

**Purpose:** Interactive CLI for heart-rate and respiratory-rate inspection. Modes: `inspect` (single window), `rates` (whole-night sliding), `metrics` (accuracy summary). Wraps the core `sleep_monitor` pipeline with argparse.

**Output:** On-screen matplotlib plots; no saved artifacts.

---

### Pipeline: `scripts/run_rate_detection.py`

**Purpose:** Run the default rate-detection pipeline (CLE-CRE + OLS + ACF) across all 12 sessions. Writes per-session metrics CSV + 4 summary plots to `notebooks/plots/` and `artifacts/rate_detection/`.

**Status:** Superseded by `scripts/compute_rates.py` (which uses the upgraded GT pipeline).

---

### Pipeline: `scripts/compute_rates.py`

**Purpose:** Compute sliding-window rate estimates for all sessions using the new GT pipeline (Flow for resp, ECG R-peaks for cardiac via neurokit2). Outputs: `artifacts/rates/metrics.parquet`, `artifacts/rates/windows/` per-session rate time series.

**Status:** Active — uses `gt_sliding_rates()` from the 2026-04-22 GT upgrade.

---

### Pipeline: `scripts/compute_eeg.py`

**Purpose:** Compute EEG band power (delta/theta/alpha/beta) by sleep stage for all sessions. Outputs: `artifacts/eeg/band_power.parquet`, per-session spectrogram arrays.

**Status:** Active; consumed by notebook `03_eeg_analysis.ipynb`.

---

### Pipeline: `scripts/sweep.py`

**Purpose:** Grid search over (channel, preproc, estimator) combinations per band across all 12 sessions. Writes per-window parquets (`artifacts/sweep/windows/`) and leaderboard (`artifacts/sweep/leaderboard.parquet`). Per-window parquets feed the classifier pipeline.

**Status:** Active; consumed by `scripts/train_classifier.py` and `notebooks/05_method_search.ipynb`.

---

### Pipeline: `scripts/train_classifier.py`

**Purpose:** Train and evaluate rate-prediction classifiers (LOSO CV) on the per-window parquets from `sweep.py`. Outputs: `artifacts/classifier/metrics.parquet`, `artifacts/classifier/summary.csv`, per-band out-of-fold predictions.

**Status:** Active; results viewed in `notebooks/06_classifier_results.ipynb`.

---

### Plotting: `scripts/plot_all_sessions_timeseries.py`

**Purpose:** Plot full-night and 30s-window time series for all 12 sessions (cap channels + PSG). Outputs per-session PNGs to `notebooks/plots/session_timeseries/`.

**Status:** Active; useful for visual inspection.

---

### Plotting: `scripts/plot_best_rates.py`

**Purpose:** Plot the two best rate detection methods (resp: peaks_scaled/k, cardiac: hilbert_scaled/k) against GT for all 6 subjects (Night 1), with sleep-stage overlay. Outputs to `artifacts/plots/best_rates/`.

**Status:** Active.

---

### Plotting: `scripts/plot_apnea_timeseries.py`

**Purpose:** Plot apnea/hypopnea event timeseries for all 12 sessions — hypnogram, apnea timeline (colored spans), and event density (15-min bins). Outputs to `notebooks/plots/apnea/`.

**Status:** Active.

---

### Plotting: `scripts/plot_apnea_fullnight.py`

**Purpose:** Full-night apnea overview for all 12 sessions — 6-row time-aligned figures: CLE/CRE/CH means, apnea events by subtype, head movement, spectrogram. Outputs to `notebooks/plots/apnea_analysis/`.

**Status:** Active.

---

### Validation study: `scripts/run_validation.py`

**Purpose:** Compute per-epoch (30s) CAP rate estimates vs PSG GT for all 12 sessions using best estimators (peaks_scaled for resp, hilbert_scaled for cardiac). Outputs: `artifacts/validation_windows.parquet`, `artifacts/validation_session.csv`, `artifacts/validation_stage.csv`.

**Status:** Active; prerequisite for `scripts/plot_validation.py`.

---

### Validation study: `scripts/plot_validation.py`

**Purpose:** Generate validation study figures from precomputed artifacts (Bland-Altman, scatter, stage boxplots, session MAE bars, summary table). Outputs to `notebooks/plots/`.

**Requires:** `scripts/run_validation.py` first.

---

### Validation study: `scripts/generate_validation_docs.py`

**Purpose:** Generate validation study DOCX documents (`notebooks/validation_methods.docx`, `notebooks/validation_results.docx`). Formats results from the rate validation pipeline for external presentation.

---

### Signal validation: `scripts/signal_validation.py`, `scripts/merge_validation.py`, `scripts/plot_validation_report.py`

**Purpose:** Per-epoch spectral peak alignment, coherence, and cross-correlation with surrogates for all 12 sessions. Documented in `PROGRESS_LOG.md`.

**Status:** Written, syntax-verified; awaiting execution.

---

### Signal validation: `scripts/signal_validation_enhanced.py`

**Purpose:** Enhanced signal validation with 5 analyses (spectral peak agreement, peak-freq coherence, frequency tracking correlation, SNR-gated re-analysis, canonical coherence). Compares 5 L/R combination strategies (CLE-CRE, avg, CLE, CRE, PCA-1). Outputs: `artifacts/signal_validation_enhanced.parquet`, `artifacts/canonical_coherence.parquet`, `artifacts/channel_comparison_summary.csv`.

**Status:** Written; run status unknown.

---

### Signal validation: `scripts/signal_validation_delay_pca.py`

**Purpose:** Delay-embedding PCA for channel combination. Standard PCA on 2 z-scored channels degenerates to avg/diff; delay embedding lifts to higher-dimensional space. Sweeps tau and n_delays, evaluates via coherence vs GT. Outputs: `artifacts/delay_pca_validation.parquet`, `artifacts/delay_pca_sweep_summary.csv`.

**Status:** Written; run status unknown.

---

### Signal validation: `scripts/signal_validation_proof.py`

**Purpose:** Comprehensive publication-quality evidence that CAP signals contain respiratory and cardiac information. Best validated settings, canonical coherence, surrogate tests, formatted DOCX output. Outputs: `artifacts/proof_validation.parquet`, `artifacts/Signal_Validation_Proof.docx`.

**Status:** Written; run status unknown.

---

### Signal validation: `scripts/cardiac_coherence_test.py`

**Purpose:** Factorial test (2x2x2x2 = 16 conditions) for cardiac coherence improvements: ECG vs Pleth GT, wide vs narrow band, acc removal on/off, 30s vs 60s epoch. Outputs: `artifacts/cardiac_coherence_test.csv`.

**Status:** Written; run status unknown.

---

### Rate accuracy: `scripts/rate_accuracy_analysis.py`

**Purpose:** Full-night rate accuracy analysis across 4 cap channels (avg, diff, CLE, CRE). 30s non-overlapping epochs tagged with sleep stage, apnea, motion, electrode drift. Outputs: `artifacts/rate_accuracy.parquet`, `artifacts/rate_accuracy_summary.csv`, 8 figures in `notebooks/plots/rate_accuracy/`. This is task 6a from TASKS.md.

**Status:** Written; run status unknown.

---

### Rate accuracy: `scripts/rate_accuracy_docx.py`

**Purpose:** Generate DOCX report for rate accuracy analysis with embedded figures from `notebooks/plots/rate_accuracy/`. Outputs: `artifacts/Rate_Accuracy_Analysis.docx`.

**Status:** Written; depends on `rate_accuracy_analysis.py` outputs.

---

### ICP validation dataset: `scripts/load_validation.py`

**Purpose:** Load the ICP validation dataset (`combinedDataAnalyses_041626`) — subjects laying down in controlled postures/modes. Different cap channels (Cvl, Cvr, Cbl, Cbr) from overnight sessions. Utility loader used by other validation scripts.

**Status:** Active; imported by `validation_breath_rate.py`, `validation_peak_analysis.py`, `validation_laydown_rates.py`.

---

### ICP validation: `scripts/validation_breath_rate.py`

**Purpose:** Validate breath-rate scaling factor k per experiment mode on the ICP validation dataset (subject S0001). Uses z(Cvl)-z(Cvr) vs Thorax GT. Outputs plots + CSV to `notebooks/plots/`.

---

### ICP validation: `scripts/validation_peak_analysis.py`

**Purpose:** Phase-by-phase peak/rate analysis across all ICP validation subjects. Compares lenient CAP peak detection against Thorax GT peaks per subject/phase. Outputs heatmaps, scatter plots, overlays to `notebooks/plots/validation/`.

---

### ICP validation: `scripts/validation_laydown_rates.py`

**Purpose:** LayDown-only validation analysis for respiratory and cardiac rates. Restricted to layDown phases, computes raw/scaled rates and scaling factor k. Outputs to `notebooks/plots/validation/laydown/`.

---

### ICP validation: `scripts/plot_validation_rates.py`

**Purpose:** Plot best rate detection methods on the ICP validation dataset. 6 subjects, controlled posture phases. GT from Pleth (cardiac) and Thorax (resp) — no ECG/Flow available. Outputs to `notebooks/plots/validation/`.

---

### Analysis: `notebooks/analysis_morphology.py`

**Purpose:** Morphological cluster pipeline — ACF-based rate estimation (primary) + event counting with adaptive divisor (secondary). Both compared against PSG GT. Produces `morphology_signal.png` and `morphology_validation.png`.

**Status:** Exploratory; may be superseded by the scaled-peaks approach.

---

### Analysis: `notebooks/analysis_pca_stacked_cle_cre.py`

**Purpose:** PCA on delay-embedded [CLE, CRE] stack — 2 sessions, 5 random 1-min windows. Delay embedding with m=3, tau=0.25s. Produces per-window PC component plots and 3D trajectory visualizations.

**Status:** Exploratory; mentioned in TASKS.md #8 as "not validated."

---

### Analysis: `notebooks/analysis_dmd_rank_sweep.py`

**Purpose:** Hankel-DMD sweep of embedding dimension m for cardiac rate on S1N1. Tests m in {3, 6, 10, 15, 20, 30}. Per-m calibration of k, whole-night evaluation. Summary table + 3-panel plot of (MAE, coverage, k_dmd) vs m.

**Status:** Exploratory; mentioned in TASKS.md #8.

---

### Analysis: `notebooks/analysis_sws_band_ratios.py`

**Purpose:** Moving-window band power ratio analysis for all 12 sessions — EEG-band power ratios (delta, theta, alpha, beta) with motion masking + delta sub-band ratios. Outputs to `notebooks/plots/sws_band_ratios/`.

**Status:** Active; relates to SWS/slow-wave-sleep investigation thread.

---

## (Retroactive) Numbered notebooks — created pre-2026-04-30

These Jupyter notebooks were created as the interactive analysis/viewing layer. Most consume artifacts produced by scripts.

| # | Notebook | Purpose | Depends on |
|---|----------|---------|------------|
| 01 | `01_overview.ipynb` | Raw signal inspection, hypnograms, session stats | Raw data only |
| 02 | `02_rate_estimation.ipynb` | Visualize sliding-window rate estimates | `scripts/compute_rates.py` artifacts |
| 03 | `03_eeg_analysis.ipynb` | EEG spectrograms and band-power by stage | `scripts/compute_eeg.py` artifacts |
| 04 | `04_metrics_summary.ipynb` | Cross-subject accuracy tables and method comparison | `scripts/sweep.py` artifacts |
| 05 | `05_method_search.ipynb` | Best pipeline selection from sweep leaderboard | `scripts/sweep.py` leaderboard |
| 06 | `06_classifier_results.ipynb` | Rate classifier results, LOSO CV | `scripts/train_classifier.py` artifacts |
| 07 | `07_validation_loader.ipynb` | Load ICP validation dataset | `scripts/load_validation.py` |
| 08a | `08_cap_sleep_embedding.ipynb` | PCA/t-SNE/UMAP embedding of CAP features by stage | Feature extraction |
| 08b | `08_validation_loader.ipynb` | Duplicate of 07 (validation dataset loader) | `scripts/load_validation.py` |
| 09 | `09_projection_3d.ipynb` | 3D UMAP/t-SNE with 40 features + PCA baseline | Feature extraction |
| 10 | `10_projection_cap12.ipynb` | 3D projection with 12 CAP-only features | Feature extraction (logged above, 2026-04-30) |

---

## Next Steps

- Validation of cardiac and resp rates with our data
  - accuracy metric for rate detection methods

- slow wave sleep analysis
    - how do we identify events that corespond to slow wave sleep
    - thorax signal correlates to the low freq magnitude events, can we validate that low magnitude thorax corresponds to increase of low freq signal in cap data

    questions:: 
    - can we detect events like apnea
    - access sleep anpnia event in data
    - sleep staging based rates

    hypothesis
    - slow wave sleep is conected to deep sleep (N2 N3), if that is goin well then REM follows, if its short then REM may not occur


    - Compare spectrogram to the SWS analyssis, see if harmonics are observed.

    ** Projection methods


    - sleep apnea::
    Flow: gives types on apnea
    effort1
