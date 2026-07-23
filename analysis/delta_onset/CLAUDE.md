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
- [x] Trigger built + eyeballed (2026-07-23). 11–197 onsets/session, N2-dominant,
      100% NREM by construction. Clean quiet→burst transitions in most events;
      **open issue:** low-SWA sessions (S5N1 n=11) show pre-window delta
      contamination → consider a pre-window EEG-quiescence gate before precursor test.
- [ ] Precursor / lead-lag analysis.
