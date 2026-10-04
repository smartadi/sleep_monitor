# Provenance: where every result in the manuscript comes from

Audit of **manuscript V11** (`writeup/review/final/CAP_sleep_mask_manuscript_V11.docx`) and
**supplementary V3** (`writeup/review/final/CAP_sleep_mask_manuscript supplementary V3.docx`),
2026-10-03. Every figure, table and quoted number was traced to the script that makes it.
Paths are relative to the repo root. `mv/` = `analysis/mean_value/`.

Status: **TRACED** (script + output found) · **CONFLICT** (code and text disagree) ·
**EXTERNAL** (made outside this repo) · **UNTRACED** (no source found).

`paper/stages/` replaces these scripts; this file records what each stage was ported from.

---

## §2.2–2.4, §3.1 Overnight levels → `s01_recordings`, `s02_overnight`

| item | legacy source | status | note |
|---|---|---|---|
| Table 1 durations | `load_session` sample count / fs | TRACED | no script writes them |
| Table 1 epochs (954…694) | `scripts/run_mask_rate_detection.py` → `reports/rates/mask/per_session_summary.csv` col `n` | TRACED | mean-value grid has 9,312 epochs vs 9,319: two grids |
| 111 Hz acquisition, 111→100 Hz resampling | — | EXTERNAL | files arrive at 100 Hz |
| OLS canceller | `sleep_monitor/preprocessing.remove_acc_artifact` | TRACED | |
| canceller removes 0.1% variance, r=1.00 | `analysis/rates/motion_cancel_validation.py` | CONFLICT | run on CLE−CRE only; text says each channel |
| 10-s blocks, head-turn angle | `mv/channel_evolution.py`, `sleep_monitor/motion.head_angle` | CONFLICT | only Fig 2 uses 10-s blocks; all §3.1 numbers use 30-s epochs |
| session means CLE 1958–2048, CRE 1624–2353, CH −719…−1297 fF | `mv/mean_value_vs_stage.py` → `mv/mean_centred_traces.py` | TRACED | |
| within-night ranges 105/71/158/595 fF | `mv/mean_centred_traces.py` | TRACED | "CH >1000 fF in three recordings" is wrong: five |
| Fig 2 (S4N2) | `mv/fig2_overnight_panels.py` (imports `channel_evolution`) | TRACED | redrawn by co-author; caption ages garbled (25/37/54/66) |
| ρ=0.01 p=0.62 n=9,182 | `mv/imbalance_marker.py:fig_vs_headangle` | TRACED | only in a figure title; uses sin(turn) |
| 7,641 supine epochs, 3.5 reversals | `mv/imbalance_marker.py:summarise` → `imbalance_session.csv` | TRACED | |
| posture magnitudes 9.0/11.3/107.7 fF | `mv/imbalance_marker.py` | TRACED | figure annotation only |
| 3.1–135.7 fF; 15–705 fF·h; 46×; 2.0–5.8×; asymmetry −0.22…+0.09 | `mv/imbalance_burden.py` | TRACED | text cites "Fig S5a/b"; it is S6 |
| "Fig S5c" onset/end near zero | — | UNTRACED | |
| Fig S1 24-h drift | bench test | EXTERNAL | "25-hour" in text vs "24-hour" in caption |
| Fig S2 L–R difference, all twelve | probably `mv/imbalance_marker.py:fig_grid` | UNTRACED | no image match |
| Fig S4 CH vs CLE−CRE | `mv/ch_vs_diff.py` → `mv/ch_vs_clecre_sessions.py` | CONFLICT | docx has an older render |
| Fig S6 + paragraph (224/705, median 65) | `mv/imbalance_burden.py` | TRACED | byte-identical |

Order: `mean_value_vs_stage` → `head_angle_validate` → `mean_centred_traces` → `imbalance_marker` → `imbalance_burden`; `ch_vs_diff` → `ch_vs_clecre_sessions`; `fig2_overnight_panels`.

## §3.2 Spectral, Fig S3, S5, Table S1 → `s03_signal`

| item | legacy source | status | note |
|---|---|---|---|
| resp 29–48%, cardiac 8–48% of power "<5 Hz" | `writeup/figures/signal_validation/generate_band_energy.py` (stdout) | CONFLICT | 3 sessions only, CLE−CRE, denominator 0.05–10 Hz |
| SNR CH 30.0 / CLE 18.7 / CRE 20.7 dB, min 6.4 | `signal_characterization.py --recompute` → `inband_snr.py` | TRACED | "CH highest every participant" false (S4N1, S4N2) |
| Fig S3 | `inband_snr.py` → `fig2_inband_snr.png` | TRACED | main text calls it S6 |
| unworn floor | `inband_snr_split.py`, `mv/baseline_variance_floor.py` | PARTIAL | whiteness ratio not written by any script |
| ¶180 NREM spectra, peaks 0.17/0.22 Hz; Fig 3 | — | EXTERNAL | co-author |
| aperiodic slope p=0.008/0.033; Fig 4 | — | EXTERNAL | co-author; no fit in repo |
| Fig S5 coherence (floor 0.053 = 1/19) | `analysis/rates/cap_psg_coherence.py` | TRACED | "Thorax ≥ Flow every channel" false on CH |
| Table S1 gate | `scripts/build_consolidated_resp_gt.py` → `analysis/rates/gt_quality_gate_table.py` | TRACED | all 48 values match |
| S3 thorax vs flow r=−0.47 | hard-coded in `writeup/edits/apply_gt_gate_supplement.py` | UNTRACED | reproduces from the parquet |

## Rate estimation (Discussion, supp S7–S12) → `s04_rates`

| item | legacy source | status | note |
|---|---|---|---|
| night error 0.24 br/min / 1.56 bpm; k 1.18 / 1.96 | `analysis/rates/rerun_rate_detection.py` → `artifacts/rate_rerun_phase_a.parquet`, `reports/rates/rerun/*.csv` | TRACED | |
| "2.02 peaks per cardiac cycle" | `analysis/rates/peaks_per_beat.py` | CONFLICT | method is peak counting, text says R-peak-triggered averaging |
| 14.4–16.8 br/min; "Table 3" | old pipeline | CONFLICT | rerun gives 14.21–16.98; no Table 3 in V11 |
| S7 pipeline example | `analysis/rates/supp_rate_pipeline.py` | TRACED | divides by cohort k 1.18, S2N1's own is 1.03 |
| S8 k by channel | `analysis/rates/supp_k_by_channel.py` | TRACED | |
| cardiac ref vs ECG (36%, 29%; k 1.98, 1.81–2.28) | `analysis/rates/ref_sanity_check.py` | TRACED | 1.98 hard-coded in `_build_supp_v3.py` |
| S9, S12 | `analysis/rates/supp_rate_epoch_k.py` | TRACED | |
| S10 | `analysis/rates/supp_rate_allsessions.py` | TRACED | |
| S11 calibration vs baseline | `analysis/rates/supp_rate_pipeline.py:fig_result` | TRACED | |

## §3.3 Ridges, §3.6 harmonic comb → `s05_ridges`, `s06_harmonic_comb`

| item | legacy source | status | note |
|---|---|---|---|
| Viterbi resp/cardiac traces | `analysis/slow_wave/ridge_overlay_tune.py:track_single_ridge` | TRACED | |
| per-epoch ridge features, N3 contrasts | `analysis/slow_wave/band_ridge_analysis.py` → `band_ridge_epochs.parquet`; `ridge_stage_all_channels.py` | TRACED | |
| count matching | `analysis/rates/reviewer_pass_analyses.py:m6_m7_ridges` | PARTIAL | n_ridges only, not power |
| "lower N3 resp ridge power across channels, controlling for stage duration" | `ridge_stage_all_channels.py` | CONFLICT | duration control never done for power |
| CRE dominant 9/12 | `run_ridge_overlay.py` (older detector) | LOG-ONLY | |
| 67 low-band ridges, median 6, 0.05–0.08 Hz | `ridge_overlay_tune.py` (pre-ed34139, PNG titles) | ARTIFACT | motion canceller's 0.05 Hz corner; replaced by `ridge_lowband_smooth.py` |
| Fig 5 | `ridge_overlay_tune.py --session S4N2 --channel CRE` | CONFLICT | stale render |
| comb detector, 22 events / 9 sessions / 6 subjects | `harmonic_ladder_overlay.py` → `ladder_stage_relationship.py` | TRACED | null is per session, not per event |
| Fig 8, Fig 9 | same | TRACED | Fig 8 stacks three channels; caption says CH |
| example bands 0.15/0.28/0.42/0.68/0.95 Hz | — | UNTRACED | |

## §3.4 spindles, §3.5 delta onsets, §3.9 K-complexes → `s07_spindles`, `s08_delta_kcomplex`

| item | legacy source | status | note |
|---|---|---|---|
| spindle counts 351–2,134; 14,305 | `analysis/spindles/spindle_lowband_detection.py` | TRACED | |
| +0.45–0.49 dB, CH 0.55, EEG 3.45 | `writeup/paper/key_numbers.py:_spindles` | CONFLICT | averages in the POOLED row; 12-session CH 0.542, EEG 3.44 |
| sigma 0.02–0.03 dB | same | TRACED | |
| random-N2 / arousal controls | `spindle_ersp.py` → `spindle_ersp_control.py` | TRACED | not count-matched as Methods says |
| "all 12 recordings" | `spindle_lowband_detection.py` | CONFLICT | CLE, CRE 11/12 |
| Fig 6 | `spindle_lowband_detection.py:make_figure` | EXTERNAL redraw | caption 0.55/0.02 vs pooled 0.59/0.03 |
| 344 onsets, 1–99 per recording | `analysis/delta_onset/delta_onset_detection.py` | TRACED | "three recordings <10" is four |
| peaks 1.4–3.2 z, 6/6, latency 4–8 s | `analysis/delta_onset/lowband_precursor_check.py` | TRACED | |
| AUC 0.42–0.56, xcorr at zero lag | `analysis/delta_onset/delta_cap_precursor.py` | CONFLICT | zero-phase, text says causal |
| 0.35–0.41 → ~0, 2–3 subjects | `lowband_precursor_check.py` | PARTIAL | CH 4/6 |
| "most onsets within 10 s of an arousal" | `analysis/delta_onset/arousal_control.py` | TRACED | 339 events, not 344 |
| Fig 7 | `lowband_precursor_check.py` (3×3 causal grid) | EXTERNAL redraw | |
| "K-complexes constituted most onsets" | — | CONFLICT | 9 of 344 on a scored mark |
| 50 K-complex marks, 1–10 per recording | `kcomplex_morphology.py` → `kcomplex_cap_response.py` | TRACED | |
| Fig 12 | `kcomplex_cap_response.py` | TRACED | byte-identical; latencies from zero-phase envelopes |

## §3.7 reproducibility, §3.8 associations, Fig 13 → `s09_reproducibility`

| item | legacy source | status | note |
|---|---|---|---|
| Fig 10, R² 0.63/0.71/0.68 | Excel `writeup/review/Overnight_sleep_subject_list_V3.xlsx` ("Estimated from graph"), from `mv/prof_metrics.py` | EXTERNAL + CONFLICT | exact values 0.648 / 0.717 / **0.859** |
| Fig 11, §3.8 R values | same workbook; `mv/prof_metrics.py`, `analysis/swa_validation/arousal_index.py` | EXTERNAL + CONFLICT | R=0.68 from a mis-paired column (correct 0.48); signs wrong at 0.48/0.48/0.63; R²=0.10 untraced |
| Fig 13 optical | `writeup/review/Comparison Optic and Cap(1).pptx` | EXTERNAL | |

## Cross-reference errors in V11 (text only)

Fig S6 cited as S5 (×3) · SNR figure cited as S6 (is S3) · head-angle result cited as S4 ·
spindle figure cited as Fig 7 (is 6) · delta figure cited as Fig 8 (is 7) · comb example
cited as Fig 9, stage figure as 10a/b (are 8, 9) · reproducibility cited as Fig 11 (is 10) ·
associations cited as Fig 12 (is 11) · "Table 3" does not exist · S10/S11 cited for ridges
and combs are rate figures · Table S1 called S2 in code.
