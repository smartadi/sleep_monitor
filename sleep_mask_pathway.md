# Sleep Mask Analysis Pathway

## Signals available

**Mask:** left capacitor, right capacitor, accelerometer  
**Reference:** ECG, nasal flow, thorax belt, abdomen belt, sleep stage labels, apnea event labels

---

## Phase 1: Preprocessing

- Align all signals to a common timebase and sample rate
- Detrend each channel (remove slow drift)
- Compute accelerometer magnitude from XYZ axes
- Notch filter ECG if mains interference present
- Segment into 30 s epochs, aligned to sleep stage scoring
- Flag epochs with excessive accelerometer RMS as motion-contaminated — do not silently exclude, carry the flag through all analyses

---

## Phase 2: Signal Validation

**Goal:** prove the mask signal contains cardiac and respiratory information at the waveform level, before any rate derivation. Work through the four levels in order — each is more fundamental than the last.

### Level 2 — Spectral peak alignment

For each epoch, compute the PSD of the raw mask signal. Identify the dominant frequency in the respiratory band (0.1–0.5 Hz) and the cardiac band (0.7–4.0 Hz). Compare these against the dominant frequency extracted from nasal flow and ECG respectively for the same epoch.

Key outputs:
- Scatter: mask spectral peak freq vs GT freq, per band, coloured by sleep stage
- Time series across the night: mask spectral peak overlaid on GT rate
- Spectral SNR distribution — ratio of peak power to surrounding noise floor

### Level 3 — Magnitude-squared coherence

Compute coherence between the mask signal and each GT signal across the full frequency range, per epoch. No rate derivation involved — this directly measures shared frequency content.

Pairs to compute:
- Left/right cap vs nasal flow — respiratory band
- Left/right cap vs thorax and abdomen belts — respiratory band
- Left/right cap vs ECG — cardiac band
- Accelerometer magnitude vs nasal flow — respiratory band (supplementary)

Key outputs:
- Coherence spectrogram across the night (x = time, y = frequency, colour = coherence), with sleep stage strip above
- Box plots of per-epoch band coherence grouped by sleep stage
- Box plots grouped by apnea type vs normal epochs
- Left vs right coherence comparison — internal consistency check

### Level 4 — Waveform cross-correlation and surrogate test

Bandpass both the mask signal and the GT signal into the same band, normalise, and cross-correlate the waveforms directly. Extract the peak correlation coefficient and the phase lag.

Then run the surrogate test: phase-randomise the GT signal 1000 times (preserves power spectrum, destroys phase), compute correlation for each surrogate, compare the real r against the surrogate distribution. A significant result (p < 0.05) proves the shared timing is not coincidental.

Key outputs:
- Histogram of real r vs surrogate distribution for representative epochs
- Box plots of waveform r per sleep stage and per apnea type
- Phase lag distribution — should cluster tightly near zero if signal is genuine
- Percentage of clean epochs where surrogate test is significant

### Left vs right capacitor consistency

Run waveform correlation between left and right capacitor channels in both bands. This is an internal check — both channels should agree. Systematic asymmetry is worth reporting separately as it may reflect asymmetric ICP or contact differences.

---

## Phase 3: Rate Detection Integration

Rate extraction is already complete. This phase integrates existing results with the validation outputs above.

- Merge per-epoch rate estimates with spectral, coherence, and waveform correlation outputs on epoch index
- Carry sleep stage, apnea type, and motion flag into the merged table
- Use the merged table as the basis for all reporting below

---

## Phase 4: Agreement Reporting

### Bland-Altman analysis

Run for each mask-vs-GT rate pair:
- Heart rate: mask vs ECG
- Respiratory rate: mask vs nasal flow
- Respiratory rate: mask vs thorax belt (secondary)

Stratify each Bland-Altman by sleep stage. Report bias and 95% limits of agreement. Run on clean epochs (motion flag = false) and report separately whether results change materially when flagged epochs are included.

### Apnea-stratified comparison

Compare coherence and waveform r between apnea epochs and matched normal epochs, broken down by apnea type:

| Type | Expected mask behaviour |
|---|---|
| Obstructive | Cardiac pulsatility continues, respiratory band disrupted |
| Central | Both bands attenuated |
| Hypopnea | Partial attenuation in respiratory band |

This stratification is the most clinically interesting result — it demonstrates that the mask is sensitive to the physiological changes associated with each apnea type, not just correlating during clean breathing.

---

## Reporting checklist

- [ ] Coherence spectrogram (respiratory and cardiac, one per reference pair)
- [ ] Per-epoch coherence box plots by sleep stage
- [ ] Per-epoch coherence box plots by apnea type
- [ ] Surrogate test histogram for representative epochs
- [ ] Waveform r distribution and phase lag distribution
- [ ] Spectral peak scatter and overnight time series
- [ ] Bland-Altman panels per rate pair, per sleep stage
- [ ] Apnea-stratified summary table
- [ ] All figures produced on clean epochs; flagged epoch results reported separately
