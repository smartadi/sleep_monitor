# Sleep Staging Analysis

Classify sleep stages (Wake, N1, N2, N3, REM) from CAP sensor features using projections and ML.

> **CORRECTION 2026-08-26.** Earlier versions of this file listed `k_cardiac`
> as a "validated stage discriminator, p=10^-130" and a per-epoch feature. That
> is retracted on three counts: (1) k is **one scalar per session** (whole-night
> median of estimate/reference — see `analysis/rates/CLAUDE.md`), so it cannot be
> a per-epoch feature at all; (2) the p=10^-130 was a **pooled-epoch artifact** —
> ~10^5 autocorrelated epochs treated as independent (a null cohort reproduces
> p<1e-30 with no effect; see `sleep_monitor/group_stats.py` and
> `writeup/edits/REVIEWER_SWA_STAGING_REVIEW_2026-08-26.md`); (3) staging
> projections must use only raw CAP features, no k-derived ones. Do not use
> k_cardiac or k_resp as staging features.

## Feature set (~40 features per 30s epoch)
**Band powers** (via `spectral.py`): delta, theta, alpha, beta per CLE, CRE, CLE-CRE
**Rate features**: resp_rate, cardiac_rate, resp_variability, cardiac_variability (HRV proxy)
**Variance / arousal**: per-epoch <10 Hz signal variance per channel — the one CAP
feature with a demonstrated, honest stage relationship: it falls Wake→N3 (a depth
axis) and returns in REM, N3 lowest in 6/6 subjects, subject-level Friedman
p=0.0011. Separates deep sleep / wake from the middle at a modest AUC≈0.65; does
not resolve all five stages. See `analysis/staging/variance_staging.py`.
**Signal stats**: RMS, kurtosis, spectral_entropy, zero_crossing_rate per channel
**Cross-channel**: CLE-CRE coherence, amplitude ratio, phase difference
**Motion**: acc_rms, movement_index, head_orientation (roll/pitch)
**Context**: time_of_night, epoch_index (circadian proxy)

## Existing projection work
- `notebooks/analysis_pca_stacked_cle_cre.py` — stacked [CLE,CRE] PCA + instantaneous phase
- `notebooks/analysis_dmd_cardiac.py` — DMD eigenvalue spectrum, HR mode detection
- `notebooks/analysis_delay_pca_cardiac.py` — delay-embedded PCA for cardiac
- `notebooks/analysis_dmd_rank_sweep.py` — DMD rank sensitivity
- `scripts/run_dsub_projections.py` — delta sub-band projection runner
- Notebook 08: PCA/t-SNE on CAP spectral features
- Notebook 09: 3D PCA projection of sleep stages
- Notebook 10: CAP 12-feature projection
- Notebook 11: Delta sub-band projections

## Package modules
- `sleep_monitor/staging.py` — epoch-level feature extraction (raw Welch PSD 450-dim or ~33 handcrafted)
- `sleep_monitor/spectral.py` — sliding-window band power computation
- `sleep_monitor/motion.py` — accelerometer features (head orientation, dynamic acceleration)
- `sleep_monitor/classifier.py` — Ridge, RF, HGB, MLP with LOSO CV

## 4-phase implementation plan
### Phase 1: Feature extraction
- Extract ~40 features per 30s epoch across all 12 sessions
- Raw CAP features only — NO k_cardiac / k_resp (k is a per-session scalar, not a
  per-epoch feature; see correction above). Include the <10 Hz variance feature.
- Output: feature matrix (N_epochs x ~40) + PSG stage labels

### Phase 2: PCA and mode analysis
- Standardize + PCA on pooled feature matrix
- Scatter PC1 vs PC2 colored by PSG stage
- Per-stage tests take the SUBJECT as the unit (per-subject stage medians →
  Friedman / direction counts), never pooled epochs — see `sleep_monitor/group_stats.py`
- Variance explained analysis

### Phase 3: Unsupervised discovery
- GMM (k=2..6) on top PCs — does BIC select 5 clusters?
- t-SNE / UMAP colored by stage
- Confusion matrix: discovered clusters vs PSG labels

### Phase 4: Supervised classification
- Models: RandomForest, HistGradientBoosting, MLP
- LOSO CV (hold out 1 subject = 2 nights, train on 5)
- Temporal smoothing: HMM / Viterbi for realistic transitions
- Metrics: Cohen's kappa, per-stage F1, confusion matrix

## Key design decisions
- LOSO CV (leave-one-subject-out) to prevent subject leakage
- The strongest honest single CAP-derived stage signal is the <10 Hz variance
  (depth axis; `variance_staging.py`) — NOT k_cardiac (retracted, see top)
- HMM post-processing enforces realistic stage transitions (no N3→REM jumps)
- Start with handcrafted features before trying raw PSD (450-dim)
