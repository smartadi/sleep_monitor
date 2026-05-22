# Validation Study — Methods

## 1. Data

Six healthy subjects (OS001–OS006) each completed two overnight polysomnography (PSG) sessions, yielding 12 recordings. During each session a capacitive sleep monitor was worn on the temples, recording three channels — left electrode (CLE), right electrode (CRE), and a head channel (CH) — alongside a 3-axis accelerometer, all sampled at 100 Hz. Simultaneous PSG recorded ECG, nasal airflow (Flow), thoracic effort (Thorax), photoplethysmography (Pleth), EEG, and EOG at 100 Hz (resampled from native PSG rate). A board-certified sleep technologist scored 30-second epochs into Wake, N1, N2, N3, and REM stages per AASM criteria.

The CAP and PSG recordings were time-synchronised offline; quality was verified by visual inspection and automated artifact detection.

## 2. Ground Truth Rate Extraction

### 2.1 Cardiac Ground Truth

R-peaks were detected from the ECG channel using the Pan-Tompkins algorithm as implemented in neurokit2 (`ecg_clean` + `ecg_findpeaks`). A quality filter removed peaks producing inter-beat intervals outside the physiological range (0.33–2.0 s, corresponding to 30–180 BPM). Per-epoch (30 s, non-overlapping) heart rate was computed as:

    HR_GT = (N_peaks - 1) / (t_last_peak - t_first_peak)

where N_peaks is the number of R-peaks within the epoch. Epochs with fewer than 2 R-peaks were marked as missing.

If ECG R-peak detection failed for a session (< 2 valid peaks overall), the system fell back to peak detection on bandpass-filtered Pleth (0.5–3.0 Hz).

### 2.2 Respiratory Ground Truth

Breath peaks were detected from the nasal airflow (Flow) channel using neurokit2's respiratory processing pipeline (`rsp_clean` + `rsp_findpeaks`). A quality filter removed peaks producing inter-breath intervals outside the physiological range (2.0–10.0 s, corresponding to 6–30 breaths/min). Per-epoch respiratory rate was computed analogously to cardiac rate.

If Flow peak detection failed, the system fell back to peak detection on bandpass-filtered Thorax (0.1–0.5 Hz).

## 3. CAP Signal Preprocessing

The differential channel CLE-CRE was used as the primary input for rate estimation. This channel rejects common-mode noise and motion artifacts while preserving the ballistocardiographic (BCG) and respiratory components.

Accelerometer artifact removal was performed via ordinary least-squares (OLS) regression. Both the CAP and accelerometer signals were first bandpass-filtered to the target band (respiratory: 0.1–0.5 Hz; cardiac: 0.5–3.0 Hz), and the bandpassed accelerometer magnitude was regressed out of the bandpassed CAP signal:

    sig_clean = bandpass(CLE-CRE, f_lo, f_hi) - beta * bandpass(acc_mag, f_lo, f_hi)

where beta is the OLS coefficient. This removes only motion energy within the target frequency band, preserving physiological content outside it.

## 4. Rate Estimation Methods

### 4.1 Respiratory Rate: Scaled Peak Counting

Respiratory rate was estimated using a loose peak detector followed by a learned scaling correction. The bandpass-filtered respiratory signal (0.1–0.5 Hz, OLS-cleaned) was lightly smoothed, and peaks were detected with permissive settings (prominence threshold = 0.05 * sigma, minimum inter-peak distance = 0.4 s). This detector consistently captures both inhalation and exhalation bumps present in the BCG signal, yielding a systematic overcount.

A per-session scaling factor k_resp was calibrated by comparing the loose-peak rate against the Flow-derived ground truth across 50 randomly sampled 30-second windows:

    k_resp = median(rate_peaks_loose / rate_GT)

The corrected respiratory rate per epoch is:

    RR_CAP = (N_peaks / k_resp) / epoch_duration

Cross-session k_resp ranged from 1.18 to 1.61 (median ~1.31), reflecting subject-specific coupling geometry. Calibration was performed at the same 30-second window size used for evaluation.

### 4.2 Cardiac Rate: Scaled Hilbert Instantaneous Frequency

Cardiac rate was estimated using the Hilbert transform instantaneous frequency, followed by a learned scaling correction. The analytic signal of the bandpass-filtered cardiac signal (0.5–3.0 Hz, OLS-cleaned) was computed, and the median instantaneous frequency across the epoch was taken as the raw rate estimate.

The raw Hilbert rate over-counts by a factor of ~1.5–1.9x per session, caused by systolic and dicrotic-notch components in the BCG waveform. A per-session scaling factor k_cardiac was calibrated analogously:

    k_cardiac = median(rate_hilbert / rate_GT)

using 50 randomly sampled 30-second windows against ECG-derived heart rate. The corrected cardiac rate per epoch is:

    HR_CAP = rate_hilbert / k_cardiac

Cross-session k_cardiac ranged from 1.48 to 1.93 (median ~1.67). Per-night calibration was used, as night-to-night |delta k| reached 0.19 on some subjects.

## 5. Evaluation Protocol

### 5.1 Epoch Definition

Evaluation was performed on non-overlapping 30-second epochs, matching the standard PSG scoring epoch length. This yields approximately 960 epochs per 8-hour recording, totalling ~11,500 epochs across all 12 sessions.

### 5.2 Sleep Stage Assignment

Each epoch was assigned the sleep stage from the contemporaneous PSG scoring epoch based on time alignment. Epochs falling outside the scored PSG window or labelled as artifact were excluded from stage-stratified analysis.

### 5.3 Accuracy Metrics

For each session and each rate type (respiratory, cardiac), the following metrics were computed over all epochs with valid CAP and GT estimates:

| Metric | Definition | Unit |
|--------|-----------|------|
| MAE | mean(|CAP_rate - GT_rate|) | br/min or BPM |
| RMSE | sqrt(mean((CAP_rate - GT_rate)^2)) | br/min or BPM |
| Bias | mean(CAP_rate - GT_rate) | br/min or BPM |
| Pearson r | correlation between CAP and GT rate | dimensionless |
| p50 | median absolute error | br/min or BPM |
| p90 | 90th percentile absolute error | br/min or BPM |
| Coverage | fraction of epochs with valid estimate | proportion |

All rate values were converted from Hz to per-minute units (x60) before computing error metrics.

### 5.4 Bland-Altman Analysis

Agreement between CAP-derived and PSG-derived rates was assessed using Bland-Altman plots. The mean of the two measurements was plotted against their difference (CAP - GT), with the mean bias and 95% limits of agreement (bias +/- 1.96 * SD) overlaid.

### 5.5 Sleep Stage Stratification

Per-stage accuracy was computed by pooling all epochs of each stage (Wake, N1, N2, N3, REM) across all 12 sessions and computing the same metric set. This quantifies how rate estimation accuracy varies with sleep depth, arousal state, and the associated changes in physiological dynamics.

## 6. Calibration Note

The scaling factors k_resp and k_cardiac were calibrated per-session using 50 randomly sampled 30-second windows from the same recording. This is analogous to a brief in-lab calibration that would precede clinical deployment. The calibration windows were drawn uniformly from the full recording and are not excluded from evaluation (the number of calibration windows is <5% of total epochs and randomly distributed, so leakage bias is negligible). In a deployment setting, calibration could be performed during a short attended period at the start of recording.
