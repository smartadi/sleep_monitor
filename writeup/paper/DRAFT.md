<!--
INTERNAL HEADER — not rendered into the docx (the builder skips HTML comments).
This file is the single source of truth for manuscript prose.
Build the deliverable with:  py build_docx.py   ->  CAP_sleep_mask_paper.docx
Status: METHODS DRAFTED · Results pending · Intro/Discussion deferred (SWS-EEG PENDING).
Standing rules live in OUTLINE.md:
  (1) Intro framing + Discussion stay stubs until the SWS-EEG analysis resolves.
  (2) Harmonics are reported as "stage-associated", never "SWS detection".
  (3) No p-values on stage-wise harmonic/ridge claims — report direction + per-subject consistency.
  (4) Sleep staging (UMAP/GMM) is out of scope (future paper).
Numbers come from KEY_NUMBERS.md; claims from CLAIMS.md.
-->

# Capacitive Temple-Sensor Sleep Mask: Signal Validation, Calibrated Rate Estimation, and Harmonic Spectral Structure

*Working manuscript draft — sections for integration into the full paper. Drafted by analysis pipeline; figures and final framing to follow.*

## 1. Introduction

> *[To be written — deferred pending the SWS-EEG analysis, which determines how the spectral half of the paper is framed.]*

## 2. Methods and Materials

### 2.1 Participants and recording protocol

Twelve overnight recordings were collected from six participants, each studied on two separate nights. Every recording paired a capacitive sensing (CAP) sleep mask with concurrent full polysomnography (PSG) as ground truth. The mask carried three capacitive electrodes — a central channel (CH) and left and right temple channels (CLE, CRE) — together with a three-axis accelerometer for motion reference. All CAP channels were sampled at 100 Hz. The simultaneous PSG provided EEG, left/right EOG, ECG, nasal airflow (Flow), photoplethysmography (Pleth), and thoracic/abdominal respiratory effort bands, along with expert 30-second sleep-stage scoring and scored apnea/hypopnea events.

### 2.2 Signal preprocessing

Motion artifact was suppressed by regressing band-limited accelerometer energy out of each CAP channel. The accelerometer magnitude and the CAP channel were first bandpassed to the analysis band of interest (respiratory 0.1–0.5 Hz, ≈6–30 breaths/min; cardiac 0.5–3.0 Hz, ≈30–180 beats/min) so that only motion energy within that band was removed. Two cancellers were available: an ordinary-least-squares (OLS) projection that removes a single stationary coupling coefficient, and a normalized-LMS (NLMS) adaptive FIR canceller (16 taps, µ = 0.05) that tracks a time-varying coupling when posture or sensor contact drifts across the night. Band-limited filtering used zero-phase Butterworth bandpass filters.

Unless otherwise noted, the canonical analysis channel is the OLS differential **CLE−CRE**, which cancels common-mode drift between the two temple electrodes. Individual channels (CH, CLE, CRE) and their average are retained for the multi-channel analyses of Sections 2.6 and 3.3.

### 2.3 Ground truth derivation

Reference cardiac and respiratory rates were derived from the PSG. Cardiac ground truth used ECG R-peak detection (neurokit2 Pan–Tompkins variant, with ECG cleaning at the native sampling rate); photoplethysmography served as a fallback when ECG quality was poor. Respiratory ground truth used peak detection on the nasal airflow signal, with the thoracic effort band as fallback. From the detected events we computed beat- and breath-level intervals and instantaneous rates, and resampled these onto a common sliding-window time base for comparison with the CAP estimates. Sleep stages were taken from the PSG technologist scoring on 30-second epochs (Wake, N1, N2, N3, REM).

### 2.4 Signal validation approach

Before estimating rates, we tested whether respiratory and cardiac rhythms are physically present in the CAP signal. For each analysis epoch we computed magnitude-squared coherence between the CAP channel and the corresponding PSG reference (Flow for respiration, ECG for cardiac), and read the coherence at the ground-truth rate frequency. Spectral agreement was quantified as the fraction of epochs whose CAP peak frequency fell within ±0.05 Hz of the reference. To guard against spurious coherence from band-limited noise, we generated phase-randomized surrogates (200 per epoch) that preserve the power spectrum while destroying phase structure, and counted the fraction of epochs whose observed coherence exceeded the surrogate null at p < 0.05. Coherence was evaluated within each sleep stage to confirm the coupling persists beyond wake.

### 2.5 Rate estimation and k-factor calibration

Per-window rates were estimated from the bandpassed CAP channel using six base estimators: spectral (Welch PSD peak frequency), autocorrelation (dominant ACF lag with parabolic interpolation), Hilbert instantaneous-frequency median, upward zero-crossing rate, prominence-thresholded peak counting, and a spectral-guided amplitude-adaptive peak detector. Four advanced trackers were added for the harder cardiac band: variational mode decomposition (VMD), continuous-wavelet-transform (CWT) ridge tracking, STFT peak tracking with Viterbi smoothing, and MUSIC.

CAP counting systematically miscounts relative to the PSG reference because the capacitive waveform is not a clean one-cycle-per-event signal. We correct this with a per-session scalar **k**, defined as the ratio between the CAP estimate and the ground-truth rate, so that the calibrated rate is the raw estimate divided by k. k was calibrated per session from 50 randomly drawn one-minute windows and applied to the whole night. We report both the uncalibrated accuracy (Sections 3.2–3.3) and the k-scaled accuracy (Section 3.4), and we verify that k estimated from the 50-window diagnostic set matches k computed over the whole night.

### 2.6 Multi-channel fusion pipeline

To exploit the three capacitive channels jointly, we evaluated quality-weighted fusion across channels. Each window received a quality score (signal-to-noise ratio, autocorrelation prominence, and spectral concentration); per-channel estimates were combined using these weights. For the cardiac band we used CWT ridge tracking as the per-channel estimator and applied Viterbi smoothing over the ridge sequence to penalize physiologically implausible jumps and reduce frame-to-frame jitter. We compared single-best-channel, multi-channel fused, and an oracle (best channel known per window) to bound the achievable headroom.

### 2.7 k(t) biomarker analysis

Because k reflects how the capacitive waveform maps onto each cardiac or respiratory cycle, we asked whether its temporal trace k(t) carries physiological information rather than being calibration noise. We computed k(t) per window over the whole night and characterized (i) its distribution across PSG sleep stages, (ii) its temporal persistence via autocorrelation half-life, and (iii) its Spearman correlation (Bonferroni-corrected) with established physiological markers — heart-rate variability (SDNN), EEG delta power, and accelerometer RMS. We contrast k_cardiac against k_resp, the latter serving as a comparison case dominated by motion and noise.

### 2.8 Harmonic detection methods

Whole-night CAP spectrograms show structured harmonic ladders rather than broadband noise. We detected this structure with three complementary methods: harmonic product spectrum (HPS) to score integer-ratio alignment, cepstral analysis to recover the fundamental period, and explicit fundamental-frequency (F0) estimation from which we computed a harmonic energy ratio (HER) — the share of spectral energy concentrated on F0 and its integer multiples. Detection was run per channel and per window across the full recording.

### 2.9 Persistent ridge tracking

To capture sustained spectral features, we linked spectrogram peaks across consecutive time frames into persistent ridges. Each ridge was characterized by its frequency, duration, power, and a prominence score relative to the local spectral background. Per window we summarized the ridge field by the number of harmonic groups, the minimum ridge frequency, total ridge power, and frequency spread. Ridge features were aggregated by PSG sleep stage for the descriptive comparison in Section 3.7; consistent with the study's limited sample size, these stage comparisons are reported directionally with per-subject consistency counts rather than significance statistics.

## 3. Results

### 3.1 Signal validation

> *[To be written — pull from KEY_NUMBERS.md §3.1: coherence, frequency match, surrogate significance.]*

### 3.2 Rate accuracy (no k-scaling)

> *[To be written — KEY_NUMBERS.md §3.2: best resp spectral, best cardiac CWT ridge.]*

### 3.3 Multi-channel fusion and temporal smoothing

> *[To be written — KEY_NUMBERS.md §3.3: fusion vs single-best vs oracle, Viterbi.]*

### 3.4 k-scaled accuracy

> *[To be written — KEY_NUMBERS.md §3.4: 0.99 br/min, 3.55 BPM, LOSO, k stability.]*

### 3.5 k_cardiac as a physiological biomarker

> *[To be written — KEY_NUMBERS.md §3.5: stage distribution, half-life, correlations.]*

### 3.6 Harmonic structure by sleep stage

> *[To be written — descriptive only, no p-values (OUTLINE rule 3).]*

### 3.7 Persistent ridge features

> *[To be written — N3 = fewer, slower, lower-power ridges; per-subject consistency, no p-values.]*

### 3.8 Channel dominance and prominence scoring

> *[To be written — CH/CRE dominance, N3 prominence.]*

## 4. Discussion

> *[To be written — deferred pending the SWS-EEG analysis.]*
