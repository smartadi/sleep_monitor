# Tasks

## 1. Preprocessing Pipeline [DONE]

- [x] OLS accelerometer artifact removal on CLE, CRE, CLE-CRE channels
- [x] Bandpass filtering: resp [0.1-0.5 Hz], cardiac [0.5-3.0 Hz]
- [x] Multi-channel support (CLE, CRE, CLE-CRE)
- [x] Sliding window infrastructure (configurable win_sec, step_sec)

## 2. Ground Truth Pipeline [DONE]

- [x] Initial GT: ACF on bandpassed PSG Thorax (resp) and Pleth (cardiac)
- [x] Upgraded GT (2026-04-22): ECG R-peaks via Pan-Tompkins (cardiac), Flow peak detection via neurokit2 (resp)
- [x] `ground_truth.py` module with `gt_resp_rate`, `gt_heart_rate`, `gt_sliding_rates`
- [x] Automatic fallback to Thorax/Pleth if Flow/ECG fails
- [x] Quality filtering: rejects physiologically impossible intervals

## 3. Respiratory Rate Detection [DONE — integration pending]

### 3a. Method survey on S1N1
- [x] Default pipeline baseline: ACF on CLE-CRE, MAE 5.45 br/min, r~0 (2026-04-15)
- [x] Per-method comparison on 5 random 2-min windows (spectral, acf, hilbert, zerocross, peaks)
  - ACF worst (MAE 6.2, bias -6.0), peaks best (MAE 1.47)
  - Double-peak /2 hypothesis disproven at 2-min scale (ratio ~1.0, not 2.0)
- [x] ACF-constrained peak detection with RATE_DIVISOR tuning on 3-min window
  - Confirmed /2 works on short window due to bandpass passing both in/out half-cycles
  - Per-channel tuning: CLE-CRE cleanest, CRE noisiest with genuine double peaks

### 3b. Scaled-peaks estimator
- [x] Peak-ratio histogram + sensitivity sweep on S1N1 (Phases 1-5)
  - Loose detection (pf=0.05, md=0.4s) with learned k=1.25 gives MAE 2.34 br/min
  - 16% improvement over baseline but correlation drops slightly
- [x] Cross-session validation (all 12 sessions)
  - k range [1.18-1.61], median 1.31, std 0.15
  - Mean MAE: 3.09 -> 2.20 br/min (-25.3%)
  - k is subject/coupling-stable, k_diag ~ k_whole (50 windows sufficient for calibration)
  - Subjects cluster: OS001/OS002/OS005-N1/OS006 ~ k=1.2-1.3; OS003/OS004/OS005-N2 ~ k=1.4-1.6

### 3c. Hardcoded /2 removal
- [x] Replaced all RATE_DIVISOR=2 with continuous per-session k (2026-04-22)
- [x] `rate_peaks_scaled_resp` and `calibrate_k_resp` added to rates.py

### Current best: `rate_peaks_scaled_resp / k`, MAE ~2.2 br/min (new GT), k ~1.31

## 4. Cardiac Rate Detection [DONE — integration pending]

### 4a. Method survey on S1N1
- [x] Default pipeline baseline: ACF on CLE-CRE, MAE 15.73 BPM, r~0 (2026-04-15)
  - pred sigma 5x GT sigma, ACF locks onto noise/sub-harmonics
- [x] Per-method comparison on 5 random 1-min windows
  - Hilbert/zerocross/peaks all ~2x GT (~100-115 vs ~55 BPM) — systolic + dicrotic bump
  - ACF best but still MAE 8.9 BPM, 1/5 NaN
  - Envelope broken (NaN on all windows)

### 4b. Tuned pipeline + scaling factors on S1N1
- [x] Tuning: ACF prom 0.10->0.05, win 30->60s, spectral nperseg=fs*8, envelope HF-band fix
- [x] Learned per-method k from 50 random 1-min windows
  - hilbert k=1.74, zerocross k=1.92, peaks k=1.93 (tight IQR)
  - spectral/envelope have wide IQR — single scalar doesn't help
- [x] Best single estimator: hilbert/k=1.74, MAE 4.33 BPM (73% reduction vs ACF baseline)
- [x] Envelope fix: HF [3-20Hz] input -> 75% coverage, MAE 16.84 BPM
- [x] ACF prom reduction did NOT help (sub-harmonic lock-in is the failure mode)

### 4c. Cross-session validation (all 12 sessions)
- [x] Hilbert/k validated on all 12 sessions
  - k range [1.48-1.93], median 1.67
  - MAE: 18.29 -> 4.19 BPM (-77%) across all sessions
  - Every session improves; worst residual S2N2 at 6.55 BPM (still < half of baseline)
  - k_diag ~ k_whole: 50 random 1-min windows sufficient to calibrate
- [x] Per-subject k stability: 3/6 subjects Dk <= 0.03, 1/6 has Dk=0.19
  - Per-night calibration recommended over per-subject
- [x] `rate_hilbert_scaled_cardiac` and `calibrate_k_cardiac` added to rates.py
- [x] Default k=1.67 when no calibration available

### Current best: `rate_hilbert_scaled_cardiac / k`, MAE ~4.2 BPM, k ~1.67

## 5. k(t) Biomarker Analysis [DONE]

Writeup: `notebooks/k_biomarker_writeup.md`

### Phase 1+2 — k(t) time series + characterization
- [x] Computed k(t) = raw_CAP_rate / GT_rate per 60s window, 10s step, all 12 sessions
- [x] k_resp(t) = peaks_loose / Flow GT; k_cardiac(t) = Hilbert / ECG GT
- [x] Autocorrelation analysis: cardiac halflife 1-4 min (physiological), resp halflife 0.5 min (noisy)
- [x] Per-session distributions, time series overlaid on hypnogram

### Phase 3 — Correlation with PSG biomarkers
- [x] Sleep stage: k_cardiac varies by stage (N1=1.71 > N2/N3=1.65 > Wake=1.61 > REM=1.58, p=10^-130)
- [x] HRV: k_cardiac vs SDNN r=-0.251 (higher HRV -> simpler BCG waveform)
- [x] EEG delta: k_cardiac vs delta power r=-0.158 (deep sleep reduces k)
- [x] Movement: k_resp vs acc RMS r=0.290 (motion adds spurious peaks)
- [x] Resp regularity: k_resp vs breath CV r=0.131

### Phase 4 — Interpretation
- [x] k_cardiac IS a physiological biomarker (BCG waveform complexity tracks autonomic tone)
- [x] k_resp is primarily a noise/quality indicator
- [x] S6N2 anomaly flagged (k_cardiac=0.79, only session with Hilbert undercounting)

## 6. CAP → Thorax Respiratory Effort Prediction [IN PROGRESS]

Goal: predict thorax_resp_rms (30s epochs) from CAP-only features — no thorax belt at deployment.

### 6a. Correlation analysis + full-night plots [DONE]
- [x] Per-epoch feature extraction: 5 CAP channels × 4 stats + thorax + accel (35 cols)
- [x] Pooled, per-session, per-stage correlation analysis
- [x] 12 full-night plots + 4 summary plots (heatmap, scatter, bars, stage-stratified)
- [x] Scripts: `scripts/thorax_analysis.py`, output: `artifacts/thorax_cap_epochs.parquet`
- [x] Finding: pooled Pearson r between CAP and thorax features is weak (r=0.06–0.18)

### 6b. Multi-model predictor comparison [DONE]
- [x] 5 models: AR-only, Ridge, ARIMAX, Stage-ARIMAX, XGBoost
- [x] XGBoost best: within-session R2=0.511, LOSO R2=0.690 — but uses thorax lags
- [x] Scripts: `scripts/thorax_predictor.py`, output: `artifacts/thorax_predictor_results.csv`

### 6c. CAP-only predictor (no thorax lags, no stage_code) [DONE]
- [x] Enhanced features: 52 per-epoch (base + spectral + rate + cross-channel + context) + 25 temporal = 77 total
- [x] 4 tiers: Ridge, XGB-Base, XGB-Enhanced, XGB-Recursive
- [x] Best: T2-XGB-Enh within-session R2=0.281, LOSO R2=0.368
- [x] Recursive pseudo-lags hurt (error accumulation)
- [x] Scripts: `scripts/thorax_predictor_caponly.py`, output: `artifacts/thorax_caponly_results.csv`

### 6d. Feature group ablation [DONE]
- [x] CAP signal only (92 features): LOSO R2=0.118
- [x] Accel only (4 features): LOSO R2=0.279
- [x] Context only (4 features): LOSO R2=0.148
- [x] Accel+Context (8 features): LOSO R2=0.362
- [x] Finding: prediction is dominated by motion/position/time-of-night, not CAP signal coupling
- [x] Script: `scripts/_ablation_quick.py` (temporary)

### 6e. Motion-independent thorax prediction [DONE]
- [x] Residualize: Ridge(accel → thorax_resp_rms) per session, mean R2=0.333 (motion explains 33% of thorax variance)
- [x] Similarly residualized all 44 CAP features against accel
- [x] Predict residual thorax from residual CAP:
  - XGB-Resid-Base: within-session R2=0.045, LOSO R2=-0.029
  - XGB-Resid-All: within-session R2=0.040, LOSO R2=0.002
  - XGB-Orig-CAP-Only (no residualization, reference): within-session R2=0.192, LOSO R2=0.169
- [x] **Finding: after removing motion, CAP features have essentially zero cross-subject predictive power on thorax effort (LOSO R2≈0). The CAP→thorax R2=0.118 found earlier was entirely motion-mediated.**
- [x] Feature importance on residuals dominated by raw_mean channels (DC level, electrode coupling) — not respiratory signal
- [x] Script: `scripts/thorax_residual_analysis.py`, output: `artifacts/thorax_residual_results.csv`

### 6f. Per-session slow-trend models on residualized data [DONE]
- [x] Visualized slow trends: smoothed residualized thorax + top 3 CAP features per session (12 plots)
- [x] 4 models per session (70/30 temporal split): OLS-Top5, AR-only, ARIMAX(cap), SmoothedRidge
- [x] AR-only vs ARIMAX comparison with AIC/BIC
- [x] Script: `scripts/thorax_residual_persession.py`, output: `artifacts/thorax_residual_persession.csv`, 26 plots in `notebooks/plots/thorax_analysis/persession_*.png`

**Results:**
- All models have **negative median test R2** — none reliably beat the mean predictor
- AR-only: median R2=-0.017 (multi-step forecast decays to mean)
- ARIMAX(cap): median R2=-0.042, but **11/12 sessions have dAIC < -2** (median dAIC=-35)
- AIC/R2 paradox: CAP improves in-sample fit but the relationship is non-stationary — doesn't extrapolate from first 70% to last 30% of night
- 4/12 sessions show real signal (ARIMAX R2 > 0.1): S3N2=0.23, S4N1=0.15, S5N2=0.25, S6N2=0.28
- Top features are `*_raw_mean` (electrode DC levels), not respiratory signal features
- SmoothedRidge badly overfits (median R2=-1.24)

**Conclusion:** Slow co-moving trends exist in ~4/12 sessions but the coupling is non-stationary and session-specific, driven by electrode impedance changes rather than respiratory signal coupling. Confirms 6e: **after removing motion, CAP sensors do not reliably predict thorax respiratory effort amplitude.**

### 6f. Rate accuracy with 4-channel comparison
- [ ] Run rate estimation (resp: peaks/k, cardiac: hilbert/k) on all 4 channels (avg, diff, CLE, CRE) across 12 sessions, plot all night sessions and detected rates across all channels
- [ ] Calibrate k per channel per session on avg channel (not just CLE-CRE)
- [ ] Use Pleth as cardiac GT (not ECG) — validated as better reference for cap sensors
- [ ] 30s non-overlapping epochs tagged with: sleep stage, apnea status, motion level, CLE/CRE mean values + mean deltas (electrode drift)
- [ ] Per-epoch error metrics + oracle best-channel selection
- [ ] Plots: overnight rates, error by stage/apnea/motion, Bland-Altman, channel comparison, best-channel distribution
- [ ] Output: `artifacts/rate_accuracy.parquet`, 8 figures in `notebooks/plots/rate_accuracy/`

### 6b. GT-free best channel selection [NOT STARTED]
- [ ] Analyze oracle best-channel labels from 6a — which cap-signal-only features predict the best channel?
- [ ] Candidate features: SNR, spectral clarity, L/R amplitude ratio, inter-channel coherence
- [ ] Train classifier or derive heuristic rule on oracle labels
- [ ] Evaluate: adaptive channel selection MAE vs fixed avg vs oracle

### 6c. Rate improvement methods
- [ ] **Quality gating + median smoothing** — gate out windows with quality < 0.4, interpolate, apply 5-window causal median filter. Most MAE from ~5-10% noisy Wake/N1 windows. Easiest change, biggest expected gain.
- [ ] **NLMS accelerometer removal** — replace OLS with NLMS (already implemented). Tracks time-varying coupling from posture changes. One-line change in pipeline.
- [ ] **Learned fusion (Ridge/XGBoost)** — `train_fusion_model` already built in rates_ml.py. Ridge showed 4.5 BPM cardiac MAE vs 5.3 for Hilbert/k alone (~15%). Run build_fusion_dataset + train_fusion_model, evaluate LOSO.
- [ ] **Stage-aware k calibration** — now supported by k biomarker findings (k_N1=1.71, k_REM=1.58). Calibrate k per stage using sleep profile.
- [ ] **Denoising autoencoder** — self-supervised autoencoder (rates_ml.py, already built). Most effort, uncertain payoff.

## 7. Formal Validation Study [NOT STARTED]

- [ ] Define accuracy metrics for publication (MAE, RMSE, bias, Bland-Altman, coverage)
- [ ] Run best methods on all 12 sessions with new GT (ECG R-peaks, Flow peaks)
- [ ] Generate Bland-Altman plots per subject and aggregate
- [ ] Statistical tests (paired t-test or Wilcoxon on per-window errors)
- [ ] Sleep-stage stratified accuracy (how do rates perform in Wake vs N1 vs N2 vs N3 vs REM)
- [ ] Report: validation that heart rate and respiratory rate signals are present in CAP data and can be reliably estimated

## 7b. Signal Validation — Nonlinear Coupling [NOT STARTED]

- [ ] **Harmonic distortion analysis** — check if a single GT peak (cardiac/resp) produces multiple peaks in cap sensors (e.g., fundamental + harmonics from nonlinear transduction like clipping, asymmetric pulse shapes). Coherence only captures linear coupling, so harmonics would redistribute energy away from the fundamental and understate true signal content.
- [ ] Quantify harmonic-to-fundamental ratio in cap vs GT spectra per epoch
- [ ] Evaluate nonlinear coupling metrics (mutual information or cross-recurrence) that capture phase-locked harmonics coherence misses
- [ ] Reassess validation conclusions — if significant harmonic energy exists, reported coherence values are conservative lower bounds

## 8. Exploration / Advanced Methods [NOT STARTED]

- [ ] PCA / DMD projection methods for cardiac (analysis_delay_pca_cardiac.py, analysis_dmd_cardiac.py exist but not validated)
- [ ] ACF sub-harmonic lock-in fix (constrain lag search to rolling prior, or detect bimodality and flip-correct)
- [ ] Pearson r improvement — scaling fixes bias but not per-window responsiveness; need different approach for HRV/apnea applications
- [ ] Cardiac envelope method refinement (75% coverage at MAE 16.84 — gap vs hilbert/k)
- [ ] Investigate S6N2 anomaly — k_cardiac=0.79 (Hilbert undercounts). Check sensor contact, ECG GT quality, compare S6N1 vs S6N2 raw signals
- [ ] Stage-aware k calibration experiment — use k_N1=1.71, k_N2/N3=1.65, k_Wake=1.61, k_REM=1.58 instead of global k=1.67; measure MAE improvement on all 12 sessions

---

## 9. Sleep Phase Detection from CAP [NOT STARTED]

Plan: `notebooks/SLEEP_PHASE_DETECTION_PLAN.md`

### Phase 1 — Epoch-level feature extraction (30s epochs)
- [ ] Band power ratios: delta, theta, alpha, beta for CLE, CRE, CH, CLE-CRE
- [ ] Respiratory features: rate, rate variability, amplitude
- [ ] Cardiac features: heart rate, HRV, cardiac amplitude
- [ ] **k_cardiac and k_resp** — BCG waveform complexity features (validated as stage-correlated in k biomarker analysis)
- [ ] Signal stats: RMS, spectral entropy, zero-crossing rate, kurtosis
- [ ] Cross-channel: CLE-CRE coherence, amplitude ratio, phase difference
- [ ] Motion: acc RMS, acc band power, movement index
- [ ] Target: ~30-40 features per epoch across all channels

### Phase 2 — PCA & mode analysis
- [ ] Standardize + PCA on pooled feature matrix (all sessions)
- [ ] Analyze PC loadings — which features dominate each component
- [ ] Scatter PC1 vs PC2 colored by PSG sleep stage
- [ ] Per-PC stage correlation (box plots, ANOVA/Kruskal-Wallis)
- [ ] Variance explained curve

### Phase 3 — Unsupervised discovery
- [ ] GMM with k=2..6 on top PCs — does BIC pick ~5 clusters matching 5 stages?
- [ ] t-SNE / UMAP visualization colored by PSG stage
- [ ] Confusion matrix: GMM clusters vs PSG labels — which stages merge/separate?

### Phase 4 — Supervised classification
- [ ] Models: Random Forest, HistGradientBoosting, small MLP
- [ ] LOSO cross-validation (hold out 1 subject, train on 5)
- [ ] Temporal smoothing: HMM or Viterbi for realistic stage transitions
- [ ] Metrics: Cohen's kappa, per-stage F1, confusion matrix, accuracy
- [ ] Fallback: 3-class (Wake / Light[N1+N2] / Deep[N3+REM]) if 5-class is too hard

---

## Beyond Rate Detection

- [ ] Slow wave sleep analysis — identify events corresponding to SWS, validate that low-magnitude Thorax correlates with low-freq signal increase in CAP
- [ ] Sleep apnea detection — access apnea events in data, use Flow signal for typing
- [ ] Sleep harmonics analysis — compare spectrogram to SWS analysis
