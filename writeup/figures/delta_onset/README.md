# Delta-onset CAP-precursor figures (paper)

Paper-facing figures from `analysis/delta_onset/` — test of the hypothesis that a
CAP band-power event *precedes* EEG delta-burst onset. Regenerate with
`.venv/Scripts/python.exe analysis/delta_onset/delta_cap_precursor.py`; source
outputs live in `analysis/delta_onset/outputs/` (gitignored scratch).

Onset trigger = quiescence-gated EEG delta-burst onset; two windows reported
(`q15` = 15 s pre-onset quiescence, `q30` = 30 s).

| File | What it shows |
|------|----------------|
| `fig_precursor_grid_{q15,q30}.png` | Peri-onset average CAP band power (3 bands × CLE/CRE/CH), mean±SEM across subjects, vs random-NREM null. **Main result.** |
| `fig_precursor_xcorr_{q15,q30}.png` | CAP→EEG-delta cross-correlation over NREM (+lag = CAP leads); shuffle null. |
| `fig_precursor_auc_{q15,q30}.png` | Forecasting AUC: pre-onset CAP power vs random NREM (0.5 = chance), per-subject dots. |
| `precursor_summary_{q15,q30}.csv` | Per band×channel stats (lead amplitude, xcorr peak lag/r, AUC). |
| `fig_onset_gallery_S2N2_q30.png` | Representative single delta-onset events (trigger illustration). |
| `fig_onsets_overview_S2N2_q30.png` | Whole-night hypnogram / delta envelope / motion with detected onsets. |
| `delta_onsets_summary.csv` | Per-session onset counts at both windows. |

**Result:** the preceding-event hypothesis is NOT supported — no CAP precursor
before onset, forecasting AUC ≈ chance. Instead a robust CAP band-power *response*
AFTER onset (all bands × channels, peak +2→+5 s, CH>CRE>CLE); xcorr peaks at lag ≈0
with the shoulder on the EEG-leads side → CAP follows delta, doesn't lead. Robust
across the q15/q30 windows. A slow-wave/K-complex-locked mechanical–hemodynamic
co-activation that is *following*, not *preceding*.
