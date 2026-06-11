# SWA Validation — Capacitive EEG vs Contact EEG

## Goal
Replicate the Lucey et al. (Sci Transl Med, 2019) SWA spectral pipeline.
Apply it identically to both contact EEG and our capacitive sleep-mask channel.
Quantify agreement for slow-wave activity (SWA) and slow-wave sleep (SWS) detection.

## Hardware
- **DUT:** Sleep mask — 3 capacitive electrodes (left temple, right temple, center reference) + accelerometer.
- **Reference:** Simultaneous contact EEG, ECG, thorax/abdomen belts, PPG/pleth.
- Capacitive differential channel treated as non-contact EEG to validate against contact EEG.

## Pipeline (Steps 0-4)

### Step 0 — Inventory (BLOCKING)
Locate all recordings. Report: formats, channels, sampling rates, units (uV? ADC?),
per-night duration, N subjects/nights, AASM staging availability, time-alignment/sync.
**Do NOT proceed until inventory is confirmed by user.**

### Step 1 — Shared Processing Pipeline
One function for all signals:
1. Downsample to 128 Hz
2. Zero-phase FIR bandpass 0.5-40 Hz (least-squares design)
3. Epoch into 6-second windows (no overlap)
4. Welch PSD per epoch (Hamming, no overlap)
5. Band powers: 1-4.5 Hz (total SWA), 1-2 Hz, 2-3 Hz, 3-4 Hz, 20-30 Hz (artifact/EMG)
6. Artifact rejection: 95th-99.5th percentile threshold, expect <4%; also gate on 20-30 Hz + accelerometer
7. Exclude nights with >10% artifact; require >=2 valid nights/subject

**Sub-1 Hz note:** Reference device had 0.1-0.6 Hz band-stop. If our cap channel resolves <1 Hz cleanly, that's a novelty — but must verify it's signal, not baseline wander.

### Step 2 — Reference Targets
(a) Continuous SWA from contact EEG per sub-band.
(b) If AASM staging exists: binary N3/SWS label per 30-s epoch, aligned to 6-s power epochs.

### Step 3 — Validation Metrics
**Continuous SWA (per sub-band, 1-2 Hz primary):**
- Pearson + Spearman correlation, regression slope
- Bland-Altman: bias + limits of agreement
- Magnitude-squared coherence (0.5-4 Hz)

**Binary SWS detection (if staging available):**
- ROC/AUC sweep for threshold selection
- Sensitivity, specificity, Cohen's kappa vs N3 labels

### Step 4 — Reporting
- Per-subject statistics (never pooled raw epochs)
- Subject as random effect for inferential stats (linear mixed models)
- SWS (stage label) vs SWA (spectral power) always clearly distinguished

## Deliverables
1. Shared band-power module (documented, parameterized)
2. Per-subject results table (correlations, Bland-Altman, coherence, sens/spec/kappa/AUC)
3. Plots: SWA overlay, Bland-Altman, coherence spectrum, ROC
4. Written summary of agreement + limitations

## Working Rules
- Build and verify one stage at a time
- Show artifact-rejection % and sample processed epoch before full cohort
- Ask whenever format/unit/rate/staging is unclear — never guess

## Status
- [x] Step 0: Inventory (2026-06-11) — see ANALYSIS_LOG.md
- [ ] Step 1: Shared pipeline
- [ ] Step 2: Reference targets
- [ ] Step 3: Validation metrics
- [ ] Step 4: Reporting
