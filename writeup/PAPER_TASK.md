# Task: Write CAP Sleep Analysis Paper (Methods, Results, Discussion)

## How to resume
Tell Claude: **"Pick up the paper writing task from `writeup/PAPER_TASK.md`"**

## Output
Single docx: `writeup/CAP_sleep_analysis_paper.docx`

## Scope
Three sections only: **Methods/Materials**, **Results**, **Discussion**.
Three topics only: **(1) Rate estimation + k-factor**, **(2) Slow-wave sleep detection**, **(3) Supervised UMAP staging**.
No Introduction, no Conclusion, no thorax coupling section.
Style: journal paper (target: *Sleep*, *IEEE TBME*, or *Sensors*). Formal, passive voice, quantitative.

## Prerequisites (generate before writing)

### Missing figures to create

1. **Bland-Altman plots (cardiac + respiratory)** — does not exist yet
   - Script needed: compute per-window (CAP_rate - GT_rate) vs mean for all 12 sessions
   - Cardiac: use `rate_hilbert_scaled_cardiac / k` vs ECG GT
   - Respiratory: use `rate_peaks_scaled_resp / k` vs Flow GT
   - Save to `writeup/figures/rate_accuracy/bland_altman_cardiac.png` and `bland_altman_resp.png`
   - Data source: `artifacts/rates/` or recompute from `scripts/compute_rates.py`

2. **Respiratory rate MAE bar chart (all 12 sessions)** — check if `notebooks/plots/all_sessions_mae_bars.png` exists and covers resp
   - If not, generate from `artifacts/peak_ratio_per_session.csv`
   - Save to `writeup/figures/rate_accuracy/all_sessions_resp_MAE.png`

3. **Copy ridge features by stage boxplot** to writeup figures
   - Source: `reports/slow_wave/stage3_ridge_features_by_stage.png`
   - Dest: `writeup/figures/harmonics/ridge_features_by_stage.png`

### Data sources for tables

| Table | Source file |
|-------|------------|
| Cardiac per-session metrics | `artifacts/hilbert_scaled_per_session.csv` |
| Respiratory per-session metrics | `artifacts/peak_ratio_per_session.csv` |
| k biomarker summary | `artifacts/k_biomarker_summary.csv` |
| k correlations | `artifacts/k_biomarker_correlations.csv` |
| GMM clustering | `analysis/projections/CLAUDE.md` lines 93-116 (inline table) |
| Supervised UMAP silhouettes | `reports/projections/cross_session_summary.csv` |
| Harmonic stats | `reports/slow_wave/stage3_summary.csv` |
| Session metadata | `sleep_monitor/sessions.py` (_RAW_META list) |

### Key numbers to use

**Cardiac rate:**
- Scaled Hilbert MAE: mean 4.19 +/- 1.02 BPM, median 4.16, range [2.93, 6.55]
- Baseline ACF MAE: mean 18.29 BPM
- Improvement: 77% reduction
- k_cardiac: median 1.67, range [1.48, 1.93]
- k_diag vs k_whole: |delta| <= 0.04 on every session
- Night-to-night delta_k: 3/6 subjects <= 0.03, worst OS002 at 0.19

**Respiratory rate:**
- Scaled peaks MAE: mean 2.20 br/min (down from 3.09, -25.3%)
- 11/12 sessions improved
- k_resp: median 1.31, range [1.18, 1.61]
- Subject clustering: OS001/OS002 at k~1.2-1.3; OS003/OS004 at k~1.4-1.6

**k_cardiac biomarker:**
- By stage: N1=1.71, N2=1.65, N3=1.65, Wake=1.61, REM=1.58
- Kruskal-Wallis H=609, p=1.64e-130
- Autocorrelation halflife: 1.4 min (cardiac) vs 0.5 min (resp)
- Spearman: SDNN r=-0.251, EEG delta r=-0.158, acc RMS r=+0.159
- k_resp: acc RMS r=+0.290 (quality/noise indicator)

**Supervised UMAP + GMM:**
- 36 pure-CAP features, 60s window / 30s step
- Supervised UMAP silhouette: range -0.02 to 0.69, median ~0.45
- GMM k=4 best in 7/12 sessions
- Best ARI: S5N2=0.943, S2N2=0.942, S4N2=0.917
- Poor: S6N1=0.150, S6N2=0.141
- Top features: acc_rms, breath_interval_cv, Hjorth mobility/complexity, diff_SO
- Unsupervised gap: NCA linear transfer closes only ~6%

**Slow wave / harmonics:**
- S1N1 HER: N3 median 0.753, N1 median 0.414
- Cross-session: subject-dependent direction (S1/S2 N3-high, S3/S4 N3-low)
- Kruskal-Wallis p < 1e-23 (significant but direction varies)
- Persistent ridges N3 vs other: min_ridge_freq 0.25 vs 0.88 Hz (p=5.7e-4)
- N3 has fewer groups (p=7.6e-5), less power (p=3.7e-5), less spread (p=8.5e-7)
- Consistent direction in 5-6/6 subjects
- CH dominates: 70% of windows across all subjects

---

## Section-by-section writing plan

### METHODS AND MATERIALS

#### 2.1 Participants and recording protocol
Write ~1 paragraph covering:
- 6 healthy subjects, age range 20s-60s, PSQI < 5
- 2 consecutive nights each = 12 recordings
- Duration range: 4.11-8.66 hr
- Concurrent PSG: EEG (standard montage), ECG, nasal airflow (Flow), photoplethysmography (Pleth), thoracic respiratory effort
- CAP sleep mask: 3 CPC capacitive sensors (CLE = left eye, CRE = right eye, CH = forehead/top), 3-axis accelerometer
- Sampling rate: 100 Hz for all channels
- Reference: `sleep_monitor/sessions.py` for session metadata

#### 2.2 Signal preprocessing
Write ~2 paragraphs:
- **Artifact removal:** OLS regression of accelerometer magnitude from each CAP channel (removes motion artifact while preserving physiological signal). Cite the approach from `sleep_monitor/preprocessing.py`.
- **Channel selection:** CLE-CRE differential (regression residual) as primary analysis channel — maximizes common-mode rejection of non-ICP signals
- **Bandpass filtering:** Butterworth order 3, respiratory band [0.1, 0.5] Hz, cardiac band [0.5, 3.0] Hz
- **Motion gating:** Epochs with accelerometer RMS > 3 MAD above session median excluded from harmonic analysis

#### 2.3 Ground truth derivation
Write ~2 paragraphs:
- **Respiratory GT:** Peak detection on nasal airflow (Flow) channel using neurokit2. Flow is the AASM gold standard for respiratory events. Fallback to thoracic belt if Flow unavailable.
- **Cardiac GT:** Pan-Tompkins R-peak detection on ECG using neurokit2. Gives true beat-level heart rate (vs Pleth which gives pulse rate, delayed by transit time).
- **Sliding-window rates:** 60s window, 5s step. Count GT peaks per window -> instantaneous rate. Quality filter: reject physiologically impossible intervals.
- **Sleep staging:** PSG-scored 30s epochs (Wake, N1, N2, N3, REM) used as ground truth labels for all staging analyses.
- Validation on S1N1: Flow detected 6,939 breaths (mean 15.7 br/min), ECG detected 27,418 R-peaks (mean 57.5 BPM).

#### 2.4 Rate estimation and k-factor calibration
Write ~3 paragraphs. This is the core methodological contribution.
- **Five base estimators:** (1) ACF — autocorrelation peak, (2) spectral — Welch PSD dominant frequency, (3) Hilbert — instantaneous frequency from analytic signal, (4) zero-crossing — rate from zero crossings, (5) peaks — local maxima counting with tunable prominence/min-distance
- **The overcounting problem:** CAP signals produce ~1.3x (resp) to ~1.7x (cardiac) more detectable events than GT because of waveform morphology (double-peaked respiratory waveform, systolic + dicrotic BCG bumps). A fixed /2 correction is too aggressive.
- **Per-session k calibration:** `k = median(rate_CAP / rate_GT)` computed from 50 randomly sampled 1-minute windows. Applied as `rate_calibrated = rate_raw / k`. The 50-window diagnostic is fully sufficient (k_diag vs k_whole differ by |delta| <= 0.04).
- **Best estimators selected:** Hilbert/k for cardiac (tightest IQR on k, 100% window coverage), peaks/k for respiratory (loose detection: prominence_factor=0.05, min_distance=0.4s).
- **k(t) as a time series:** Computed per sliding window across the full night. Autocorrelation halflife quantifies temporal smoothness. Stage-wise distributions tested via Kruskal-Wallis. Correlations computed vs SDNN (HRV), EEG delta power, accelerometer RMS (Spearman, Bonferroni-corrected).

#### 2.5 Feature extraction and supervised UMAP staging
Write ~2 paragraphs:
- **Feature set:** 36 features per 60s window (30s step overlap), computed from CLE, CRE, CH, accelerometer ONLY. No k-calibration, no PSG-derived inputs. Groups: per-channel band powers in 4 sub-bands (infra-slow, SO, delta-low, delta-high) x 4 channels = 16; CLE-CRE differential stats (spectral entropy, RMS, kurtosis, skewness, Hjorth mobility, Hjorth complexity) = 6; respiratory band (power, peak freq, ACF rate, SNR, amplitude CV, breath interval CV) = 6; cardiac band (power, peak freq, ACF rate, SNR) = 4; cross-channel coherence (resp + cardiac bands) = 2; accelerometer (RMS, resp-band power) = 2.
- **Supervised UMAP:** 3D embedding with PSG stage codes as supervision targets. n_neighbors=30, min_dist=0.1. Produces a manifold where same-stage epochs cluster.
- **GMM clustering:** Gaussian mixture models with k=3,4,5 components fitted on the 3D supervised UMAP coordinates. Cluster-to-stage mapping via Hungarian algorithm (maximize overlap). Metrics: adjusted Rand index (ARI), normalized mutual information (NMI), silhouette score.
- **Baseline comparison:** Unsupervised UMAP and NCA-transformed UMAP tested to quantify the supervised-unsupervised gap.

#### 2.6 Harmonic structure and slow-wave sleep detection
Write ~2 paragraphs:
- **Harmonic detection module:** Three complementary methods applied to each 30s window on CH, CLE, CRE channels. (1) Harmonic Product Spectrum (HPS): downsample PSD by factors 2-5, multiply, peak of product = f0. (2) Cepstral analysis: IFFT of log-PSD, peak at quefrency 1/f0 indicates harmonicity. (3) Explicit f0 search: find dominant peak in [0.1, 0.8] Hz, check for peaks at integer multiples within +/-0.05 Hz tolerance, up to 6 harmonics. Output per window: harmonic_energy_ratio, n_harmonics, cepstral_prominence, hps_score, dominant_channel.
- **Dominant channel selection:** Per window, the channel with the highest harmonic_energy_ratio is selected. Analyzed across all sessions.
- **Persistent ridge tracker:** Spectral peaks detected per window via scipy.signal.find_peaks (prominence > 15% of max PSD). Peaks linked across consecutive windows by nearest-neighbor matching (tolerance 0.08 Hz, max gap 5 windows). Ridges shorter than 4 windows discarded. Per-epoch features computed from active ridges: n_ridges, n_groups_active, max_group_size, min/mean/max ridge frequency, frequency spread, mean/max/total ridge amplitude.
- **Statistical testing:** Kruskal-Wallis for 5-stage comparison, Mann-Whitney U for N3 vs non-N3 (Bonferroni-corrected). Per-subject breakdown to assess consistency of direction.

---

### RESULTS

#### 3.1 Cardiac rate accuracy
Write ~2 paragraphs + reference Table 2 and Figures 1-2.
- Present the 12-session per-session table (copy from `artifacts/hilbert_scaled_per_session.csv`)
- Headline: MAE 4.19 BPM (77% reduction from 18.29 baseline)
- k stability: k_diag ~ k_whole, night-to-night delta_k
- Per-subject k clustering pattern
- Note: Pearson r remains weak-to-negative (scaling corrects bias, not window-level variation)
- Worst case: S2N2 at 6.55 BPM still < half baseline

#### 3.2 Respiratory rate accuracy
Write ~1.5 paragraphs + reference Table 3.
- Present 12-session table from `artifacts/peak_ratio_per_session.csv`
- Headline: 3.09 -> 2.20 br/min (-25.3%), 11/12 improved
- k range and subject clustering
- Note: correlation drops (loose detection trades per-window precision for lower bias)

#### 3.3 k_cardiac as a physiological biomarker
Write ~2 paragraphs + reference Table 5 and Figures 5-7.
- Stage-wise k: present the 5-stage table with medians
- Autocorrelation halflife comparison (cardiac vs resp)
- Correlation table: SDNN, EEG delta, acc RMS
- Interpretation: BCG waveform complexity. N1 = autonomic instability = most complex. REM = atonia = simplest.
- k_resp: motion-correlated noise indicator, not a biomarker
- S6N2 anomaly: k=0.79 (Hilbert undercounting), likely sensor contact issue

#### 3.4 Sleep stage separation via supervised UMAP and GMM
Write ~2 paragraphs + reference Table 4 and Figures 8-9.
- Present full silhouette and ARI tables
- GMM k=4 optimal (7/12 sessions). Confusion matrix interpretation for best (S5N2) and worst (S6N1).
- Feature importance: which features drive stage separation
- The supervised-unsupervised gap: stage signal is in nonlinear interactions, not recoverable by linear reweighting
- S6 outlier: discuss possible causes (poor signal quality, atypical architecture)

#### 3.5 Slow-wave sleep spectral signatures
Write ~2 paragraphs + reference Figures 10-12.
- S1N1 single-session finding: N3 HER 0.753 vs N1 0.414
- Cross-session: subject-dependent direction reversal. Present the heatmap pattern.
- Persistent ridge results: N3 has fewer, lower-frequency, lower-power ridges. Consistent direction across 5-6/6 subjects. Quote the p-values.
- CH dominance (70%)
- Note: HER is NOT a universal N3 biomarker in raw form. Needs per-subject normalization or multivariate combination.

---

### DISCUSSION

#### 4.1 The k-factor approach to rate estimation
Write ~2 paragraphs:
- Why k is not 2 — physiological basis (waveform morphology varies by coupling, anatomy, sleep state)
- Comparison with literature: wrist PPG cardiac MAE typically 3-8 BPM; our 4.19 is mid-range but from a completely different modality (capacitive vs optical)
- k stability enables deployment with brief calibration
- Limitation: initial PSG reference needed; future self-calibration approaches (e.g., use cardiac k's known stage-dependence to bootstrap)

#### 4.2 k_cardiac: a novel BCG waveform complexity biomarker
Write ~2 paragraphs:
- Autonomic tone modulates BCG morphology — established in bed-based BCG literature (cite Inan et al., Paalasmaa et al.)
- Our contribution: first demonstration from a head-mounted capacitive sensor
- N1 highest complexity (autonomic instability, frequent arousals) vs REM lowest (muscle atonia, simplified hemodynamics) — physiologically plausible
- Correlation with SDNN and EEG delta: cross-validates against established sleep biomarkers
- Clinical potential: sleep depth monitoring without EEG, particularly for home use

#### 4.3 CAP-based sleep staging
Write ~2 paragraphs:
- ARI up to 0.94 from 36 features and unsupervised GMM approaches expert-to-expert PSG agreement (~0.80 Cohen's kappa)
- But: requires supervised UMAP (labels during embedding). The features contain stage information, but in nonlinear interactions that unsupervised methods cannot extract.
- S6 poor performance: 2/12 sessions with ARI < 0.15. Investigate: poor sensor contact, low impedance, or genuinely atypical sleep architecture? Important for deployment.
- Next steps: train a supervised classifier (RF, HGB) with LOSO CV on the 36-feature space directly. This would not require UMAP at test time.
- Comparison with EEG-based automated staging (typically kappa 0.7-0.8): our approach is complementary (no scalp electrodes)

#### 4.4 Harmonic structure and slow-wave sleep
Write ~2 paragraphs:
- Harmonic ladders are real physiological signals — non-sinusoidal respiratory/BCG waveforms create integer harmonics in the spectrogram
- N3 shows reduced harmonic complexity (fewer ridges, slower, simpler) — consistent with deep sleep muscle relaxation and regularized breathing
- But the absolute HER level and its N3 direction are subject-dependent — S1/S2 show N3-high while S3/S4 show N3-low. This rules out raw HER as a universal N3 detector.
- Per-subject normalization or combination with other features (band powers, k_cardiac) is the path forward
- CH channel dominance: top-of-head sensor has strongest coupling to intracranial pulsations, consistent with proximity to the sagittal sinus

---

## Figures list (for the docx)

| Fig | Content | Location |
|-----|---------|----------|
| 1 | Per-session cardiac MAE bar chart (all 12 sessions) | `writeup/figures/rate_accuracy/all_sessions_cardiac_MAE.png` |
| 2 | Per-session cardiac rate traces (12-panel grid) | `writeup/figures/rate_accuracy/all_sessions_cardiac_traces.png` |
| 3 | Per-session resp MAE bar chart | **GENERATE** from `artifacts/peak_ratio_per_session.csv` |
| 4 | Bland-Altman (cardiac + respiratory) | **GENERATE** — script needed |
| 5 | k_cardiac by sleep stage boxplot | `writeup/figures/k_biomarker/pooled_k_by_sleep_stage.png` |
| 6 | k(t) detail panel (representative: S1N1) | `writeup/figures/k_biomarker/S1N1_k_detail.png` |
| 7 | k correlation heatmap | `writeup/figures/k_biomarker/pooled_correlation_heatmap.png` |
| 8 | Supervised UMAP (best 2-3 sessions: S5N2, S4N2, S2N2) | `writeup/figures/supervised_umap/` |
| 9 | GMM confusion matrices (S5N2 best + S6N1 worst) | `writeup/figures/gmm_clustering/` |
| 10 | Full-night spectrogram + ridges (S1N1) | `writeup/figures/spectrograms/S1N1_spectrogram_ridges.png` |
| 11 | Harmonic energy ratio grid (all 12 sessions) | `writeup/figures/harmonics/all_sessions_harmonic_energy_ratio_grid.png` |
| 12 | Ridge features by stage boxplot | `reports/slow_wave/stage3_ridge_features_by_stage.png` -> copy to `writeup/figures/harmonics/` |

## Tables list (for the docx)

| Table | Content | Data source |
|-------|---------|-------------|
| 1 | Subject demographics + session durations | `sleep_monitor/sessions.py` |
| 2 | Per-session cardiac metrics (k, MAE_acf, MAE_scaled, RMSE, bias, r) | `artifacts/hilbert_scaled_per_session.csv` |
| 3 | Per-session respiratory metrics (k, MAE_baseline, MAE_scaled) | `artifacts/peak_ratio_per_session.csv` |
| 4 | GMM clustering (session, best_k, ARI, NMI, silhouette) | `analysis/projections/CLAUDE.md` inline table |
| 5 | k by sleep stage (stage, k_resp median, k_cardiac median) | `notebooks/ANALYSIS_LOG.md` line 489-495 |

## Signal Validation Section — DONE (2026-06-04)

**Status: Figures generated, ready to integrate into paper writeup.**

Four paper-ready figures + summary table proving respiratory and cardiac signals
are present in CAP sensor data, validated against PSG ground truth.

| Figure | File | Key message |
|--------|------|-------------|
| Fig 1 — Waveform example | `writeup/figures/signal_validation/fig1_waveform_example.png` | Visual: CAP waveform tracks PSG breathing and heartbeat in time + frequency |
| Fig 2 — Frequency agreement | `writeup/figures/signal_validation/fig2_frequency_agreement.png` | Resp: 43% exact match, median error 0.067 Hz. Cardiac: systematic overcount (motivates k) |
| Fig 3 — Coherence + surrogates | `writeup/figures/signal_validation/fig3_coherence_and_surrogates.png` | Coupling persists across all sleep stages; exceeds phase-randomized null |
| Fig 4 — Channel comparison | `writeup/figures/signal_validation/fig4_channel_comparison.png` | Avg channel best; canonical upper bound shows 2x headroom for fusion |
| Table 1 | `writeup/figures/signal_validation/table1_signal_validation_summary.csv` | Per-channel coherence, freq match %, IQR |

**Key numbers for text:**
- Respiratory coherence at GT peak: median 0.31 (avg channel), 0.61 (canonical upper bound)
- Cardiac coherence at GT peak: median 0.16 (avg channel), 0.27 (canonical upper bound)
- Resp freq match: 43% within ±0.05 Hz, median error 0.067 Hz
- Surrogate significance (phase-randomized, 200/epoch): 15% resp, 9% cardiac exceed null at p<0.05
- Coherence persists across all sleep stages (Wake through REM)

**Script:** `scripts/paper_signal_validation_figures.py`
**Data sources:** `artifacts/proof_validation.parquet`, `artifacts/proof_canonical.parquet`, `artifacts/signal_validation.parquet`

**Integration note:** These should be added as an early Results subsection ("Signal Validation" or "Signal Presence") before rate estimation results. The coherence + surrogate evidence establishes that the signals exist; the rate results then show they can be quantified.

---

## Rate Consolidation Section — DONE (2026-06-11)

**Status: All 6 phases complete. Figures and tables generated, ready to integrate into paper.**

Multi-channel fused rate pipeline: 5 methods × 5 channels × 2 bands × 12 sessions,
channel fusion (confidence-weighted + agreement-filtered), CWT ridge tracker, 
Viterbi temporal smoothing, k-calibration + biomarker analysis.

### Figures generated

| Figure | File | Key message |
|--------|------|-------------|
| Phase 1 — Method × Channel heatmap | `writeup/figures/rate_consolidation/phase1_method_channel_heatmap.png` | Spectral best for resp (MAE 1.54 br/min no k); ACF best for cardiac (13.64 BPM no k) |
| Phase 2 — Fusion strategies | `writeup/figures/rate_consolidation/phase2_fusion_comparison.png` | Confidence-weighted vs agreement-filtered vs best-fixed vs oracle |
| Phase 3 — CWT cardiac | `writeup/figures/rate_consolidation/phase3_cwt_cardiac.png` | CWT ridge outperforms all methods on cardiac (avg channel 11.6 BPM no k) |
| Phase 4 — Viterbi improvement | `writeup/figures/rate_consolidation/phase4_viterbi_improvement.png` | Viterbi smoothing reduces jitter across all strategies |
| Phase 5 — Bland-Altman | `writeup/figures/rate_consolidation/phase5_bland_altman.png` | Fused + Viterbi vs PSG |
| Phase 5 — Per-stage MAE | `writeup/figures/rate_consolidation/phase5_per_stage_mae.png` | Accuracy by sleep stage |
| Phase 5 — Pipeline comparison | `writeup/figures/rate_consolidation/phase5_pipeline_comparison.png` | Full pipeline comparison: all strategies × smoothing |
| Phase 5 — Per-session summary | `writeup/figures/rate_consolidation/phase5_per_session_summary.csv` | Per-session MAE, RMSE, bias, r, coverage |
| Phase 6 — k-scaled heatmap | `writeup/figures/rate_consolidation/phase6_k_scaled_heatmap.png` | k-scaled method × channel (best: spectral/diff resp 0.99 br/min, hilbert/CRE card 3.55 BPM) |
| Phase 6 — k by stage | `writeup/figures/rate_consolidation/phase6_k_by_stage.png` | k varies by sleep stage — physiological biomarker |
| Phase 6 — k cross-channel | `writeup/figures/rate_consolidation/phase6_k_cross_channel.png` | k varies across channels for same method |
| Session timeseries (×12) | `writeup/figures/rate_consolidation/session_timeseries/*.png` | Per-session GT vs raw vs k-scaled+Viterbi |

### Key numbers for text

**Without k-scaling (GT-free):**
- Best respiratory: spectral, MAE 1.54 br/min (aggregate), session range 0.59–2.62 br/min
- Best cardiac: CWT ridge, MAE 11.6 BPM on avg channel (best single method without k)
- Viterbi smoothing: reduces jitter across all strategies

**With k-scaling:**
- Best respiratory: spectral/diff, median MAE **0.99 br/min**, k_resp ≈ 0.98
- Best cardiac: hilbert/CRE, median MAE **3.55 BPM**, k_cardiac ≈ 1.66

**k by sleep stage (Kruskal-Wallis significant):**
- Resp k: Wake 0.98, N1 1.02, N2 1.00, N3 1.00, REM 0.93
- Card k: Wake 1.63, N1 1.73, N2 1.65, N3 1.64, REM 1.58
- k stability: resp IQR [0.93, 1.05], card IQR [1.58, 1.70]

**Pipeline:** `scripts/run_rate_consolidation.py`
**Data:** `artifacts/rate_consolidation_phase1.parquet`, `artifacts/rate_consolidation_phase2.parquet`, `artifacts/rate_consolidation_phase4.parquet`
**Reports:** `reports/rates/phase*.csv`, `reports/rates/phase*.png`

**Integration note:** This replaces the single-channel rate results in sections 3.1–3.3. The multi-channel fusion + CWT ridge + Viterbi pipeline is the final rate estimation approach. k-calibration results update the k-biomarker section (3.3).

---

## Execution order

1. **Generate missing figures** (Bland-Altman, resp MAE bars, copy ridge boxplot)
2. **Read all CSV data sources** and format into tables
3. **Write Methods** sections 2.1-2.6 sequentially
4. **Write Results** sections 3.1-3.5 sequentially, embedding table and figure references
5. **Write Discussion** sections 4.1-4.4
6. **Assemble into docx** using docx-js or python-docx, save to `writeup/CAP_sleep_analysis_paper.docx`
7. **Copy all referenced figures** into `writeup/figures/` (most already there)

## Python environment
Use `.venv` at project root:
```
C:\Users\adity\Documents\sleep monitor\code\.venv\Scripts\python.exe
```
This has numpy, matplotlib, scipy, sklearn, sleep_monitor installed.
