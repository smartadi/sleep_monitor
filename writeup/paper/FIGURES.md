# Figures

| Fig # | Caption (draft) | Source | Status |
|-------|----------------|--------|--------|
| **Signal Validation** | | | |
| 1 | CAP waveform aligned with PSG breathing and heartbeat | `writeup/figures/signal_validation/fig1_waveform_example.png` | DONE |
| 2 | Respiratory and cardiac frequency agreement (CAP vs PSG) | `writeup/figures/signal_validation/fig2_frequency_agreement.png` | DONE |
| 3 | Cross-spectral coherence with phase-randomized surrogates | `writeup/figures/signal_validation/fig3_coherence_and_surrogates.png` | DONE |
| 4 | Channel comparison: coherence across all channels + canonical bound | `writeup/figures/signal_validation/fig4_channel_comparison.png` | DONE |
| **Rate Detection** | | | |
| 5 | Method × channel MAE heatmap (no k) | `writeup/figures/rate_consolidation/phase1_method_channel_heatmap.png` | DONE |
| 6 | Multi-channel fusion strategy comparison | `writeup/figures/rate_consolidation/phase2_fusion_comparison.png` | DONE |
| 7 | CWT ridge cardiac performance | `writeup/figures/rate_consolidation/phase3_cwt_cardiac.png` | DONE |
| 8 | Viterbi smoothing improvement | `writeup/figures/rate_consolidation/phase4_viterbi_improvement.png` | DONE |
| 9 | k-scaled method × channel heatmap | `writeup/figures/rate_consolidation/phase6_k_scaled_heatmap.png` | DONE |
| 10 | Bland-Altman: fused + Viterbi vs PSG | `writeup/figures/rate_consolidation/phase5_bland_altman.png` | DONE |
| 11 | Per-stage MAE comparison | `writeup/figures/rate_consolidation/phase5_per_stage_mae.png` | DONE |
| 12 | Full pipeline comparison (all strategies × smoothing) | `writeup/figures/rate_consolidation/phase5_pipeline_comparison.png` | DONE |
| **k-Biomarker** | | | |
| 13 | k_cardiac by sleep stage boxplot | `writeup/figures/k_biomarker/pooled_k_by_sleep_stage.png` | DONE |
| 14 | k(t) detail panel (S1N1 representative) | `writeup/figures/k_biomarker/S1N1_k_detail.png` | DONE |
| 15 | k correlation heatmap | `writeup/figures/k_biomarker/pooled_correlation_heatmap.png` | DONE |
| 16 | k cross-channel variation | `writeup/figures/rate_consolidation/phase6_k_cross_channel.png` | DONE |
| **Harmonics** | | | |
| 17 | Full-night spectrogram + ridges (S1N1) | `writeup/figures/spectrograms/S1N1_spectrogram_ridges.png` | DONE |
| 18 | HER across full night with hypnogram (S1N1) | `writeup/figures/harmonics/S1N1_harmonic_fullnight.png` | DONE |
| 19 | HER by stage — all 12 sessions grid | `writeup/figures/harmonics/all_sessions_harmonic_energy_ratio_grid.png` | DONE |
| 20 | HER heatmap (session × stage, shows direction reversal) | `writeup/figures/harmonics/all_sessions_HER_heatmap.png` | DONE |
| 21 | Ridge features by stage boxplots | `writeup/figures/sfn_abstract/stage3_ridge_features_by_stage.png` | DONE |
| 22 | Ridge features per-subject consistency | `writeup/figures/sfn_abstract/stage3_ridge_features_per_subject.png` | DONE |
| 23 | N3 vs rest ridge comparison | `writeup/figures/sfn_abstract/stage3_n3_vs_rest.png` | DONE |
| **Staging** | | | |
| 24 | Supervised UMAP (S5N2, S4N2, S2N2 — best 3) | `writeup/figures/supervised_umap/S*N*_UMAP_sup_nn30.png` | DONE |
| 25 | GMM confusion: best (S5N2) + worst (S6N1) | `writeup/figures/gmm_clustering/S*_GMM_k4_panel.png` | DONE |
| **Supplementary** | | | |
| S1–S12 | Per-session rate timeseries | `writeup/figures/rate_consolidation/session_timeseries/*.png` | DONE |
| S13–S24 | Per-session k(t) details | `writeup/figures/k_biomarker/S*_k_detail.png` | DONE |
| S25–S36 | Per-session harmonic full-night | `writeup/figures/harmonics/S*_harmonic_fullnight.png` | DONE |
| S37–S48 | Per-session spectrograms + ridges | `writeup/figures/spectrograms/S*_spectrogram_ridges.png` | DONE |
