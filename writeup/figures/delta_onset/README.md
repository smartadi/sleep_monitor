# Delta-onset CAP figures (paper)

Paper-facing figures from `analysis/delta_onset/` — the test of whether a CAP
band-power event accompanies EEG delta-burst onset. Regenerate with
`.venv/Scripts/python.exe analysis/delta_onset/delta_cap_precursor.py` (peri-onset,
xcorr, forecasting) and `analysis/delta_onset/lowband_precursor_check.py` (causal
estimator + consistency); source outputs live in `analysis/delta_onset/outputs/`
(gitignored scratch).

Onset trigger = quiescence-gated EEG delta-burst onset; two windows were run
(`q15` = 15 s pre-onset quiescence, `q30` = 30 s). **q30 is the reported set**;
q15 is retained as a robustness check in CSV form only.

| File | What it shows |
|------|----------------|
| `fig_precursor_grid_causal_q30.png` | **Figure 9 (main).** Peri-onset CAP band power, strictly causal estimator (3 bands x CLE/CRE/CH), mean+/-SEM across subjects vs random-NREM null. Flat pre-onset baseline + sharp post-onset rise: the response and the no-precursor result in one figure. |
| `fig_lowband_causal_check_q30.png` | Control: the apparent 0-0.5 Hz pre-onset ramp under zero-phase filtering (top row) vanishes under a strictly causal estimator (bottom row) — the ramp is acausal backward leakage, not a precursor. |
| `fig_precursor_xcorr_q30.png` | CAP->EEG-delta cross-correlation over NREM (+lag = CAP leads); shuffle null. Peaks at lag ~0 with the shoulder on the EEG-leads side. |
| `fig_precursor_auc_q30.png` | Forecasting AUC: pre-onset CAP power vs random NREM (0.5 = chance), per-subject dots. |
| `fig_onset_gallery_S2N2_q30.png` | Representative single delta-onset events (trigger illustration). |
| `fig_onsets_overview_S2N2_q30.png` | Whole-night hypnogram / delta envelope / motion with detected onsets. |
| `precursor_summary_{q15,q30}.csv` | Per band x channel stats (lead amplitude, xcorr peak lag/r, AUC). q15 carries the robustness check. |
| `lowband_precursor_check_q30.csv` | Zero-phase vs causal pre-onset amplitudes and rise-onset times per channel. |
| `response_consistency_q30.csv` | Per-subject causal post-onset peak vs own null — 6/6 in all nine channel x band combos. |
| `delta_onsets_summary.csv` | Per-session onset counts at both windows. |

**Result:** the preceding-event hypothesis is NOT supported — no CAP precursor in any
band, forecasting AUC ~ chance. Instead a robust CAP band-power *response* AFTER onset
(all bands x channels, peak +2->+5 s, CH>CRE>CLE, 6/6 subjects); xcorr peaks at lag ~0
with the shoulder on the EEG-leads side -> CAP follows delta, doesn't lead. Robust across
the q15/q30 windows. A slow-wave/K-complex-locked mechanical-hemodynamic co-activation
that is *following*, not *preceding*.

Removed 2026-08-06 as redundant: `fig_precursor_grid_q30.png` (zero-phase grid,
superseded by the causal grid; its unique content is the top row of the lowband check)
and `fig_precursor_{grid,xcorr,auc}_q15.png` (q15 duplicates — the robustness statement
is carried by `precursor_summary_q15.csv`). Recoverable from git history.
