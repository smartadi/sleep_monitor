# Spindle Validation — does the capacitive mask sense sleep spindles?

## Goal
Test whether the SEC sleep mask detects PSG-scored sleep spindles (11–16 Hz sigma,
hallmark of N2). Design mirrors `analysis/swa_validation/`: run one detector on
contact EEG (positive control) and on the CAP channels, with PSG spindle annotations
as ground truth.

## Data
- Spindle annotations: `overnight_6subject_complete_032626/**/PSG_analysis_*/`
  - `Spindle  K - *.txt`      — one line per spindle, value = duration (ms)
  - `Spindle frequency - *.txt` — one line per spindle, value = intra-spindle freq (Hz)
  - 11/12 nights have the `K` file; **S4N2 (OS004 12-26-2025)** has only the frequency
    file — loader falls back to it for timing.
  - Line format identical to the apnea Flow files: `HH:MM:SS,mmm-HH:MM:SS,mmm; val;label`
- CAP + EEG: standard `load_session(idx)` (100 Hz, EEG + CH/CLE/CRE in same file).

## Files
- `spindle_loader.py` — `load_spindles(session)` → event start/end/center (hr from CAP
  start), duration, freq. Wall-clock alignment via `session.time_start` (same midnight
  logic as `loader.load_sleep_profile`).
- `spindle_analysis.py` — per-session sigma-envelope triggered averages + spindle-vs-
  control AUC (EEG + 4 CAP channels), and a cardiac-band autonomic probe. Writes CSVs +
  `triggered_averages.npz`.
- `plot_spindles.py` — 3 figures from the npz/CSV.

## Key result (2026-07-09)
**Negative — the mask does not detect spindles.**
- EEG positive control: sigma AUC = 0.98, ~3.4× sigma power at spindle center.
- CAP (all channels): sigma AUC = 0.50 (chance), log2 ratio ≈ 0.
- Cardiac autonomic probe: no robust spindle-locked deflection (+0.03–0.05 z, 3/12).
Extends the SWA negative result (delta band) to the sigma band: temple-placement SEC
measures mechanical/hemodynamic signals, not cortical electrical activity.

## Alignment sanity check
53–68% of spindle centers fall in N2 epochs; median freq 12.8 Hz, dur 0.50 s. Confirms
the PSG-to-CAP time alignment is correct before interpreting the EEG/CAP contrast.
