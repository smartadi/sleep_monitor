# 3D UMAP / t-SNE Sleep Stage & Apnea Projection Plan

## Goal

Use 3D UMAP and t-SNE to visualize whether per-minute CAP sensor features form clusters corresponding to sleep stages and apnea events across overnight sleep.

## Why 3D per-minute embedding

1. **PSD-only features miss the strongest discriminators.** k_cardiac (p=10^-130 stage effect), respiratory rate variability, HRV proxies, and motion features are absent from raw PSD.
2. **30s windows → noisy estimates.** 60s windows give ~15 breaths and ~60 heartbeats — much more stable rate/variability features.
3. **2D collapses the sleep manifold.** Sleep cycles through a trajectory in feature space (Wake→N1→N2→N3→N2→REM→repeat). A 3D projection preserves structure a 2D projection collapses.
4. **Per-session first avoids batch effects.** Subject-level coupling differences dominated the pooled 2D embedding.

## Previous work

Notebook `08_cap_sleep_embedding.ipynb` ran 2D PCA, t-SNE, UMAP on raw Welch PSD (363 dims, 30s epochs, all sessions pooled). Weak stage separation, substantial subject batch effects.

---

## Phase 1 — Per-minute feature extraction

Extract ~40-50 features per 60s window, 30s step:

| Feature group | Features | Rationale |
|---|---|---|
| **Band power ratios** | delta/theta/alpha/beta for CLE, CRE, CH, CLE-CRE | Spectral shape at 60s resolution |
| **Respiratory** | resp rate, rate std (sub-windows), breath amplitude | Resp regularity strongest N3 marker |
| **Cardiac** | heart rate, HR std, cardiac amplitude | HRV changes across stages |
| **k_cardiac, k_resp** | from existing calibration infrastructure | k_cardiac is validated stage biomarker |
| **Signal stats** | RMS, spectral entropy, zero-crossing rate, kurtosis per channel | Gross signal characteristics |
| **Cross-channel** | CLE-CRE coherence (resp band, cardiac band) | Coupling shifts with stage |
| **Motion** | acc RMS, acc band power | Separates Wake; gates noisy epochs |
| **Apnea-relevant** | resp amplitude CV, breath-to-breath interval irregularity, low-freq power ratio | Apnea causes amplitude drops and irregularity |

Each window also gets: session, subject, t_hr, stage_label, apnea_label (Normal / Apnea / Hypopnea based on overlap with PSG events).

## Phase 2 — 3D UMAP + t-SNE per session

Start with S1N1. For each session:

1. StandardScaler → PCA pre-reduction to ~20-30 PCs
2. **3D UMAP** (n_components=3, sweep n_neighbors=[10, 30, 50])
3. **3D t-SNE** (n_components=3, sweep perplexity=[15, 30, 50])
4. Interactive 3D scatter (plotly) colored by:
   - PSG sleep stage (5-class, then 3-class)
   - Time of night (color gradient — overnight trajectory)
   - Apnea events (Normal vs Apnea vs Hypopnea)
5. Static matplotlib 3D panels

Plots saved to: `notebooks/plots/projections/`

## Phase 3 — Cluster analysis

1. Visual cluster assessment in 3D
2. DBSCAN or GMM on embedding — do discovered clusters match stages?
3. Silhouette score by stage label
4. Apnea minutes: cluster together or spread across stages?

## Phase 4 — Cross-session and pooled

1. Repeat Phase 2 for all 12 sessions individually
2. Pool with subject-level normalization (z-score per subject) → 3D UMAP/t-SNE
3. Check batch effect reduction with richer features

---

## Parameters

- Window: 60s, step: 30s
- Apnea labels: per-window overlap with PSG apnea events (from `load_apnea_events`)
- Visualization: plotly (interactive 3D), matplotlib (static panels)
- Start session: S1N1 (OS001, 7.95 hr)
