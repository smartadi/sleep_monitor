# Sleep Phase Detection from CAP Data — Full Methods Plan

## Data Available

- **CAP channels**: CLE, CRE, CH (+ accelerometer aX, aY, aZ) at 100 Hz
- **PSG channels**: EEG, EOGl, EOGr, ECG, Flow, Pleth, Thorax, Abdomen
- **Ground truth**: PSG sleep profiles — 30s epoch labels (Wake, N1, N2, N3, REM) for 12 sessions (6 subjects × 2 nights)
- **Existing infra**: bandpass filters, acc artifact removal, Welch band-power ratios, sliding-window quality features, LOSO CV framework

## Why PCA on CAP Makes Sense

The CAP differential (CLE-CRE) carries a superposition of respiratory, cardiac, and movement signals. Each sleep stage changes these differently:

- **N3**: very regular slow breathing, slow stable HR, no movement
- **REM**: irregular breathing, variable HR, muscle atonia but eye movement artifacts
- **N2**: moderate regularity, K-complexes may show up as transients
- **Wake**: movement, irregular patterns, high-frequency muscle artifact

PCA on multi-band features should separate these modes — PC1 likely captures gross amplitude/motion, PC2-3 may split respiratory vs cardiac regularity, and later PCs may capture stage-discriminating variability patterns.

---

## Phase 1 — Epoch-Level Feature Extraction

Extract features per 30s epoch (matching PSG scoring resolution) across all channels:

| Feature group | Features | Rationale |
|---|---|---|
| **Band power ratios** | delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30) for CLE, CRE, CH, CLE-CRE | Analog to EEG band analysis — CAP picks up different frequency content per stage |
| **Respiratory** | Resp rate (Hz), resp rate variability (std over sub-windows), resp amplitude | Breathing regularity is a strong N3 marker, irregularity marks REM |
| **Cardiac** | Heart rate (Hz), HR variability, cardiac amplitude | HRV changes dramatically across stages |
| **Signal stats** | RMS, spectral entropy, zero-crossing rate, kurtosis | Capture gross signal characteristics |
| **Cross-channel** | CLE-CRE coherence, CLE/CRE amplitude ratio, phase difference | Differential vs common-mode behavior may shift with stage |
| **Motion** | Acc RMS, acc band power, movement index | Movement gates and separates Wake |

That gives ~30-40 features per epoch.

## Phase 2 — PCA & Mode Analysis

1. **Standardize** the feature matrix (zero-mean, unit-variance per feature)
2. **PCA** on the full feature matrix (all sessions pooled)
3. **Analyze PC loadings** — which features dominate each component
4. **Scatter plot PC1 vs PC2** colored by sleep stage — look for clustering
5. **Per-PC stage correlation** — box plots of each PC score by stage, ANOVA/Kruskal-Wallis test
6. **Variance explained** — how many PCs capture stage-relevant information

## Phase 3 — Unsupervised Discovery

1. **Gaussian Mixture Model (GMM)** with k=2..6 on the top PCs — does BIC pick ~5 clusters matching the 5 stages?
2. **t-SNE / UMAP** visualization colored by PSG stage — visual check of separability
3. **Confusion matrix** between GMM clusters and PSG labels — which stages merge, which separate?

## Phase 4 — Supervised Classification

1. **Models**: Random Forest, HistGradientBoosting, small MLP
2. **CV strategy**: LOSO — hold out one subject, train on 5 subjects (10 nights), test on held-out (2 nights)
3. **Temporal smoothing**: HMM or Viterbi smoothing on output to enforce realistic stage transition probabilities
4. **Metrics**: Cohen's kappa, per-stage F1, confusion matrix, accuracy
5. **Reduced staging**: If 5-class is hard, try 3-class (Wake / Light[N1+N2] / Deep[N3] / REM)

## Most Discriminative Features (Expected)

- **Resp rate variability** (N3=low, REM=high)
- **Delta-band power** in CLE-CRE (tracks slow-wave activity)
- **Movement index** (separates Wake)
- **Cardiac rate variability** (autonomic tone differs across stages)
