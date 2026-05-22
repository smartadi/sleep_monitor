# Analysis Workspaces

Each subdirectory is a scoped workspace for a specific analysis area.
Open Claude Code from a subdirectory to load only that area's context.

| Directory | Focus | Status |
|-----------|-------|--------|
| `rates/` | Rate estimation, k-biomarker, formal validation | Phases 3-5 done, Phase 7 pending |
| `slow_wave/` | Slow wave sleep detection from CAP sensors | Early exploration |
| `staging/` | Sleep phase classification via projections + ML | Plan written, implementation pending |
| `projections/` | PCA, UMAP, t-SNE, DMD, delay embedding | PCA+UMAP+t-SNE working (4 sessions), DMD exploratory |
| `thorax/` | CAP→thorax effort prediction | Complete — no direct coupling found |
