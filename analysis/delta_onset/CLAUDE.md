# Delta-onset precursor — does CAP lead EEG delta onset?

## Hypothesis (professor)
At the **onset of a delta burst in contact EEG**, a *preceding* event appears in
specific CAP frequency bands (**0–0.5, 0.5–1, 1–3 Hz**) of CLE/CRE/CH. This is the
Fultz-style question (EEG→CSF lead/lag) turned around: here CAP is predicted to
*lead* EEG delta, not follow it.

Design mirrors `analysis/swa_validation/` (`fultz_eeg_cap_impulse.py` supplies the
lead/lag + shuffle-null machinery) and `analysis/spindles/` (`spindle_ersp.py`
supplies the event-triggered ERSP). The trigger here is a delta-onset event
instead of a spindle, and we look in a window that **precedes** t=0.

## Files
- `delta_onset_detection.py` — builds and validates the trigger (delta-burst
  onsets). Schmitt-trigger burst detector on the EEG delta (0.5–4 Hz) envelope;
  onset = rising edge (low-threshold crossing) before a sustained high crossing;
  refractory gap so trials are ~independent; kept only if in NREM with a
  motion-clean pre-window. Writes per-session onset npz + summary CSV + an
  overview figure (hypnogram/envelope/motion) and a single-event gallery for
  eyeballing. Run: `.venv/Scripts/python.exe analysis/delta_onset/delta_onset_detection.py`
- `delta_cap_precursor.py` — (TODO) peri-onset CAP band-power grid (3 bands × 3
  channels), CAP→EEG cross-correlation (test negative lag = CAP leads), and a
  forecasting AUC. Controls: random-NREM null, motion exclusion, circular-shift
  significance, per-subject reporting.

## Status
- [x] Trigger built + eyeballed (2026-07-23). N2-dominant, 100% NREM by construction.
- [x] EEG-quiescence gate added (`--quiet-pre`, default sweeps 15 & 30 s; outputs
      tagged `q15`/`q30`, side-by-side count pivot in the summary CSV). Gate yields
      clean quiet→delta trials but removes N3 (sustained delta has no quiet baseline)
      and thins low-SWA nights; 15 s recovers only modest n over 30 s and does not
      rescue S5 (genuine low-SWA/noisy-EEG). Run the precursor test on both sets.
- [x] Precursor / lead-lag analysis (`delta_cap_precursor.py`, 2026-07-23).
      **Result: hypothesis NOT supported — direction reversed.** No CAP precursor
      before delta onset (lead-window amp at/below baseline; forecasting AUC ≈ chance).
      Instead a strong CAP band-power RESPONSE after onset (all 3 bands × 3 channels,
      peak +2→+5 s, >> random-NREM null, CH>CRE>CLE); xcorr peaks at lag ≈0 with the
      shoulder on the EEG-leads side → CAP follows delta. Robust to q15/q30. A genuine
      slow-wave/K-complex-locked mechanical–hemodynamic co-activation, but *following*,
      not *preceding*. Figs `outputs/fig_precursor_{grid,xcorr,auc}_{q15,q30}.png`.
