# Projection Methods Analysis

Dimensionality reduction and manifold learning on CAP sensor features for sleep stage visualization, cardiac rate extraction, and cluster discovery.

## Methods in use

| Method | Purpose | Key files |
|--------|---------|-----------|
| **PCA** | Variance analysis, feature loadings, scree plots, pre-reduction for UMAP/t-SNE | `run_dsub_projections.py`, `analysis_pca_stacked_cle_cre.py` |
| **UMAP** | 3D nonlinear manifold embedding, stage/apnea cluster visualization | `run_dsub_projections.py` (nn=10,30,50) |
| **t-SNE** | 3D probabilistic embedding, stage separation comparison | `run_dsub_projections.py` (perp=15,30,50) |
| **DMD** | Dynamic mode decomposition for cardiac frequency extraction | `analysis_dmd_cardiac.py`, `analysis_dmd_rank_sweep.py` |
| **Delay-PCA** | Takens delay embedding + SVD for cardiac phase-space trajectory | `analysis_delay_pca_cardiac.py` |

## Feature set (15 features per 60s window, 30s step)
Delta sub-bands (infra_slow, SO, delta_low, delta_high), spectral entropy, RMS,
channel deviations, resp rate/std, cardiac rate/std, acc RMS, resp amplitude CV,
breath interval CV. Defined in `scripts/run_dsub_projections.py` FEAT_COLS.

Full plan calls for ~40-50 features (see PROJECTION_PLAN.md) including k_cardiac,
k_resp, cross-channel coherence, kurtosis, HRV — not yet integrated.

## Current results
- **PCA**: top 3 PCs explain ~30-40% variance; significant stage separation (Kruskal-Wallis p<0.05)
- **UMAP**: REM clusters well, Wake separable, NREM stages overlap. sil_3class ~0.35-0.45
- **t-SNE**: similar separation to UMAP, more visually clustered but less interpretable topology
- **DMD cardiac**: MAE ~4.5 BPM with k-scaling (comparable to Hilbert/k), sensitive to rank selection
- **Delay-PCA cardiac**: MAE ~6.3 BPM (benchmark, Hilbert/k at 4.3 is better)

## Scripts
- `scripts/run_dsub_projections.py` — full PCA + UMAP + t-SNE pipeline (4 sessions currently)
  - Outputs: scree plots, loadings, 3D interactive HTML (plotly), static panels, silhouette scores
  - Coloring: sleep stage, time of night, apnea events, thorax RMS

## Notebooks (analysis scripts)
- `notebooks/analysis_delay_pca_cardiac.py` — delay-embedded SVD → cardiac rate from phase rotation
  - Params: m=3, tau=0.25s, 60s windows
- `notebooks/analysis_pca_stacked_cle_cre.py` — stacked [CLE,CRE] delay PCA, 3D trajectory plots
- `notebooks/analysis_dmd_cardiac.py` — exact DMD on delay-embedded [CLE,CRE], mode selection by amplitude*stability
  - Cardiac band [0.5-3.0 Hz] mode → rate
- `notebooks/analysis_dmd_rank_sweep.py` — embedding dimension sweep m={3,6,10,15,20,30}
  - Best: m=10-15, MAE ~4.5 BPM, k_dmd varies with m

## Plans
- `notebooks/PROJECTION_PLAN.md` — 4-phase roadmap: features → 3D embedding → cluster analysis → cross-session pooling
- `notebooks/SLEEP_PHASE_DETECTION_PLAN.md` — uses projections as input to supervised staging

## Key parameters
- **Delay embedding**: m=3 default, tau=0.25s (T/4 for cardiac)
- **UMAP**: n_components=3, n_neighbors={10,30,50}, min_dist=0.1
- **t-SNE**: n_components=3, perplexity={15,30,50}, max_iter=1000, init='pca'
- **DMD rank**: best at m=10-15 (features=2m), threshold singular values at S[0]*1e-10
- **Windows**: 60s window, 30s step (per-minute resolution)

## Next steps
1. Expand feature set from 15 → ~40 (add k_cardiac, k_resp, coherence, HRV, kurtosis)
2. Run UMAP/t-SNE on all 12 sessions (currently 4)
3. Phase 3: GMM/DBSCAN clustering on 3D embeddings, confusion matrix vs PSG labels
4. Phase 4: pooled cross-session with subject-level z-score normalization
5. Evaluate whether delay-PCA or DMD adds value over Hilbert/k for cardiac (currently no)

## Dependencies
- scikit-learn (PCA, t-SNE, StandardScaler, silhouette_score)
- umap-learn (UMAP)
- plotly (interactive 3D)
- matplotlib (static panels)
