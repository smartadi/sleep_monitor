# Tables

| Table # | Content | Data source | Status |
|---------|---------|-------------|--------|
| 1 | Subject demographics + session durations | `sleep_monitor/sessions.py` | DONE |
| 2 | Signal validation summary (coherence, freq match, IQR) | `writeup/figures/signal_validation/table1_signal_validation_summary.csv` | DONE |
| 3 | Per-session cardiac metrics (k, MAE, RMSE, bias, r) | `writeup/figures/rate_accuracy/table3_cardiac_per_session.csv` | DONE |
| 4 | Per-session respiratory metrics (k, MAE) | `writeup/figures/rate_accuracy/table4_resp_per_session.csv` | DONE |
| 5 | Fused pipeline per-session summary | `writeup/figures/rate_consolidation/phase5_per_session_summary.csv` | DONE |
| 6 | k by sleep stage (resp + cardiac medians) | Phase 6 data | DONE |
| 7 | k correlations (Spearman, Bonferroni) | `artifacts/k_biomarker_correlations.csv` | DONE |
| 8 | Ridge features N3 vs other (p-values, effect directions) | `reports/slow_wave/stage3_summary.csv` | DONE |
| 9 | GMM clustering (session, best_k, ARI, NMI, silhouette) | `reports/projections/cross_session_summary.csv` | DONE |
| 10 | SWA validation metrics (r, coherence, AUC per session) | `analysis/swa_validation/outputs/` | DONE |
