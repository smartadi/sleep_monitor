# Slow Wave Sleep Analysis

Detect slow wave sleep (N3/SWS) signatures in CAP temple sensors without EEG.

## Hypotheses
- Low-magnitude thorax periods correlate with increased low-frequency CAP power
- Narrow-band ~1 Hz features visible in CAP spectrograms during N3
- SWS events detectable from CAP alone via band power ratios
- Harmonics in spectrogram mark SWS epochs

## Frequency bands
- **Infra-slow:** <0.5 Hz (slow oscillations, if detectable through skull)
- **Delta equivalent:** 0.5-4.0 Hz in CAP (not true EEG delta, but mechanical/vascular coupling)
- **Respiratory band:** 0.1-0.5 Hz (rate slows in N3)
- **Cardiac band:** 0.5-3.0 Hz (HRV changes in N3)

## What exists
- `sleep_monitor/spectral.py` — sliding-window band power ratios (delta, theta, alpha, beta) via Welch
- `notebooks/analysis_sws_band_ratios.py` — initial band ratio exploration
- `sleep_monitor/staging.py` — epoch-level feature extraction including band powers
- k_cardiac varies by sleep stage (validated discriminator, see analysis/rates/)
- PSG sleep profile labels available (30s epochs: Wake, N1, N2, N3, REM)

## Approach (4 stages)
1. **Band power profiling** — compute CAP band powers per 30s epoch, compare N3 vs other stages
2. **Spectrogram event detection** — find narrow-band features in CAP spectrograms, correlate with N3 labels
3. **Thorax correlation** — relate CAP low-freq power to thorax effort envelope (noting: direct CAP→thorax R2≈0 after motion removal, per Phase 6 findings)
4. **SWS detector** — combine discriminative features into N3 vs non-N3 classifier

## Key questions
- Does the CAP sensor pick up any signal in the 0.5-4 Hz range that correlates with EEG delta?
- Is the correlation mechanical (skull vibration) or vascular (cerebral blood flow)?
- Can respiratory rate slowing + HRV changes alone classify N3?
- What is the minimum feature set for useful N3 detection from CAP?

## Data
- 12 sessions with PSG sleep staging (N3 epochs available as ground truth)
- CAP channels: CLE, CRE (temple capacitive), aX/aY/aZ (accelerometer)
- PSG: EEG (for comparison), Flow, ECG, Thorax
- Load with: `from sleep_monitor import load_session, load_sleep_profile`
