# Signal Validation Build Progress

## Plan (from sleep_mask_pathway.md)

### Script 1: `scripts/signal_validation.py`
Per-epoch (30 s) signal-level validation for all 12 sessions:
- Spectral peak alignment (mask vs GT dominant freq in resp/cardiac bands)
- Magnitude-squared coherence (mask channels vs GT references)
- Waveform cross-correlation + surrogate test (phase-randomized, 1000 surrogates)
- Left vs Right cap consistency check
- Output: `artifacts/signal_validation.parquet`

### Script 2: `scripts/merge_validation.py`
- Join `validation_windows.parquet` + `signal_validation.parquet` on (session, epoch)
- Add apnea codes, motion flags
- Output: `artifacts/merged_validation.parquet`

### Script 3: `scripts/plot_validation_report.py`
- All figures from reporting checklist
- Output: `notebooks/plots/validation_report/`

## Status
- [x] Script 1 written: `scripts/signal_validation.py` — syntax verified
- [x] Script 2 written: `scripts/merge_validation.py` — syntax verified
- [x] Script 3 written: `scripts/plot_validation_report.py` — syntax verified
- [ ] Run script 1 (longest — ~30-60 min for 12 sessions with 200 surrogates each)
- [ ] Run script 2 (fast merge, seconds)
- [ ] Run script 3 (plotting, seconds)

## How to run
```
cd "C:\Users\adity\Documents\sleep monitor\code"
C:\Users\adity\anaconda3\python.exe scripts/signal_validation.py
C:\Users\adity\anaconda3\python.exe scripts/merge_validation.py
C:\Users\adity\anaconda3\python.exe scripts/plot_validation_report.py
```

## Design Decisions
- Apnea types: no subtypes available; use code 1=Apnea, 2=Hypopnea only
- Cardiac band: using 0.7–4.0 Hz for coherence/spectral (per pathway doc), but existing rate detection used 0.5–3.0 Hz — validation script uses the pathway doc bands
- Surrogate method: phase-randomization (FFT, random phase, IFFT) — 200 surrogates (not 1000) per epoch to keep runtime reasonable across 12 sessions
- Motion flag: accelerometer RMS > median + 3*MAD (same as existing code)
