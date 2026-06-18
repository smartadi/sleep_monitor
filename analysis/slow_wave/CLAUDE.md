# Slow Wave Sleep Analysis

## Status
All 4 stages complete. Ridge features are statistically significant (KW p<1e-16) but
practically weak standalone N3 discriminators (LOSO AUC=0.534). Direction is subject-dependent
(some subjects N3↑, others N3↓), explaining near-chance pooled performance.

**Completed**: Stage 4 paper demo (2026-06-18) — `paper_ridge_demo.py` produces paper-ready
spectrogram + ridge overlays, pooled quantification (6-panel), and LOSO N3 classifier.
Outputs -> `writeup/figures/harmonics/paper_*.png`, `reports/slow_wave/paper_*.csv`

**Completed**: Ridge overlay v2 (2026-06-11) — `run_ridge_overlay.py`, 3-channel stacked layout.
Artifacts: `reports/slow_wave/overlay/ridge_overlay_epochs.parquet` (55,878 rows)

Detect slow wave sleep (N3/SWS) signatures in CAP temple sensors without EEG.

## Hypotheses
- Low-magnitude thorax periods correlate with increased low-frequency CAP power
- Narrow-band ~1 Hz features visible in CAP spectrograms during N3
- SWS events detectable from CAP alone via band power ratios
- Harmonics in spectrogram mark SWS epochs
- **Harmonic ladder observation:** windows with a dominant fundamental (~0.5 Hz) plus integer harmonics (1, 2, 3 Hz) appear intermittently — likely non-sinusoidal respiratory waveform. Harmonic count and strength may encode sleep stage or respiratory effort.
- **Stage 2 update (cross-session):** Harmonic energy ratio does NOT universally mark N3. Direction is subject-dependent (S1/S2: N3-high, S3/S4: N3-low). Raw HER is not a universal N3 biomarker — needs per-subject normalisation or multivariate combination.

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

---

## Harmonic Structure Detection Plan

### What we observe
In CAP spectrograms (CH, CLE, CRE), some windows show a clear harmonic ladder:
a fundamental at ~0.5 Hz with peaks at 1x, 2x, 3x, 4x, ... Sometimes only the
fundamental is present; other times 4-6 harmonics are visible. This pattern
appears and disappears across the night.

### Why it matters
- A pure sinusoid has one spectral peak. Harmonics mean the waveform is
  non-sinusoidal — the shape carries information (sharp vs smooth breathing,
  BCG pulse morphology, coupling strength).
- Harmonic count and relative amplitudes change with: sleep depth (muscle tone
  affects waveform), respiratory effort, body position, and possibly SWS.
- A running "harmonic richness" trace could be a new CAP-derived biomarker.

### Detection approach: 3 methods (implement all, compare)

#### Method 1: Harmonic Product Spectrum (HPS)
Fastest, most robust for finding f0 when harmonics exist.
```
1. Welch PSD of window (30-60s)
2. Downsample spectrum by factors 2, 3, 4, 5
3. Multiply all downsampled spectra together
4. Peak of product = fundamental frequency f0
5. Harmonic score = product_peak / median(product)
```
- Pro: O(n), no iterative fitting, naturally finds f0 even if fundamental is weak
- Con: fixed integer ratios only (fine for this case)

#### Method 2: Cepstral analysis
Best for separating "has harmonics" from "broadband noise."
```
1. Welch PSD → log(PSD)
2. IFFT of log-PSD = cepstrum (quefrency domain)
3. Peak in cepstrum at quefrency q = 1/f0 (fundamental period)
4. Cepstral peak prominence = harmonic strength indicator
5. Absence of cepstral peak = no harmonic structure
```
- Pro: single number (cepstral prominence) cleanly separates harmonic vs non-harmonic
- Con: sensitive to window length, needs careful frequency range

#### Method 3: F0 + harmonic energy ratio (explicit)
Most interpretable, gives per-harmonic amplitudes.
```
1. Welch PSD, find dominant peak in [0.1-0.8 Hz] → f0 candidate
2. For k = 2..K_max: check if peak exists within ±tolerance of k*f0
3. Count confirmed harmonics (n_harmonics)
4. Harmonic energy ratio = sum(power at f0, 2f0, ...) / total power
5. Per-harmonic amplitudes: A_k = PSD(k*f0) / PSD(f0)
```
- Pro: gives f0, n_harmonics, per-harmonic decay profile
- Con: needs peak-finding thresholds, misses weak fundamentals

### Output: running traces (per window, sliding)

| Trace | Description | Use |
|-------|-------------|-----|
| `f0_hz` | Fundamental frequency (Hz) | Track respiratory rate from harmonic structure |
| `n_harmonics` | Count of confirmed integer harmonics (1-6+) | Waveform complexity indicator |
| `harmonic_energy_ratio` | Fraction of total power in harmonic peaks | How "tonal" vs "noisy" the window is |
| `cepstral_prominence` | Height of cepstral peak above floor | Binary harmonic detector |
| `harmonic_decay_rate` | Slope of log(A_k) vs k | Waveform sharpness (fast decay = smooth, slow = sharp) |
| `dominant_channel` | Which of CH/CLE/CRE has strongest harmonics | Sensor coupling quality |

### Implementation plan

#### Stage 1: Build harmonic detector module
- New function in `sleep_monitor/spectral.py` or new module `sleep_monitor/harmonics.py`
- `detect_harmonics(sig, fs, win_sec, step_sec, f0_range, max_harmonics, method)`
- Returns DataFrame: t_s, f0_hz, n_harmonics, harmonic_energy_ratio, cepstral_prominence, per-harmonic amplitudes, dominant_channel
- Run all 3 methods, output each as separate columns

#### Stage 2: Characterize across nights
- Script: `analysis/slow_wave/run_harmonic_analysis.py`
- Run detector on all 12 sessions, all 3 CAP channels (CH, CLE, CRE)
- Overlay harmonic traces on hypnogram — do harmonics correlate with N3?
- Statistics: harmonic feature distributions by sleep stage (Kruskal-Wallis)
- Which channel shows strongest harmonics most consistently?

#### Stage 3: Correlate with physiology
- Compare harmonic traces vs: sleep stage, respiratory rate, thorax effort, motion
- Test: do harmonics disappear during motion/wake and strengthen during deep sleep?
- Test: does harmonic decay rate differ between stages (sharper BCG waveform in lighter sleep)?
- Cross-reference with k_cardiac — both encode waveform shape

#### Stage 4: Integrate as SWS feature
- Add best harmonic features to staging feature set (analysis/staging/)
- Test discriminative power for N3 vs non-N3 classification
- Combine with existing band power ratios for SWS detector

### Parameters to tune
- **Window length:** 30s (matches epoch) or 60s (more frequency resolution for close harmonics)
- **f0 search range:** [0.1, 0.8] Hz for respiratory fundamental
- **Harmonic tolerance:** ±0.05 Hz (how close peak must be to k*f0)
- **Max harmonics:** 6-8 (above 4 Hz gets into cardiac territory)
- **Welch segment:** 8-10s within window (tradeoff: frequency resolution vs variance)
- **Minimum peak prominence:** for confirming a harmonic exists vs noise

### Key questions
- Is the fundamental always respiratory (~0.5 Hz) or sometimes cardiac sub-harmonic?
- Do the harmonics come from one specific channel (CH vs CLE vs CRE)?
- Is harmonic richness a better N3 marker than simple band power?
- Does the harmonic pattern differ between subjects or is it universal?

---

## General SWS approach (4 stages)
1. **Band power profiling** — compute CAP band powers per 30s epoch, compare N3 vs other stages
2. **Harmonic structure detection** — find and characterize harmonic ladders (see plan above)
3. **Thorax correlation** — relate CAP low-freq power to thorax effort envelope (noting: direct CAP→thorax R2~0 after motion removal, per Phase 6 findings)
4. **SWS detector** — combine band powers + harmonic features + k_cardiac into N3 classifier

## Data
- 12 sessions with PSG sleep staging (N3 epochs available as ground truth)
- CAP channels: CH, CLE, CRE (temple capacitive), aX/aY/aZ (accelerometer)
- PSG: EEG (for comparison), Flow, ECG, Thorax
- Load with: `from sleep_monitor import load_session, load_sleep_profile`
