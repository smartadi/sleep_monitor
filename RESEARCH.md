# Research Notes

## Overview

We analyze data from a sleep monitor — a wearable sensor with capacitive readouts from left and right temples (CLE, CRE) and a differential channel (CLE-CRE), accompanied by accelerometer readings. A clinical bio-marker sensor suite (PSG) records simultaneously and serves as ground truth for benchmarking.

**Dataset:** 6 subjects, 2 sessions each (12 overnight recordings), ~4-9 hours per session.

## Research Goals

1. **Heart rate and respiratory rate detection from capacitive sensors** — a validation study showing both signals are present in the sleep monitor and that rates can be reliably estimated
2. **Slow wave sleep analysis** — analysis of sleep harmonics and low-frequency events
3. **Sleep apnea detection** — identify apnea events from capacitive data

---

## Signal Characteristics

### Channels
- **CLE** — left ear capacitance, clean signal, stable ACF
- **CRE** — right ear capacitance, noisier, genuine double-peak structure on some breath crests
- **CLE-CRE** — OLS regression differential, cleanest overall, best default channel
- **Accelerometer** — 3-axis, used for motion artifact removal via OLS (or NLMS) regression

### Frequency Bands
- **Respiratory:** 0.1-0.5 Hz (6-30 br/min)
- **Cardiac:** 0.5-3.0 Hz (30-180 BPM)
- **High-frequency (BCG pulse energy):** 3-20 Hz — used for Teager-Kaiser envelope cardiac method

### Key Signal Properties
- The [0.1-0.5 Hz] bandpass passes both inhalation and exhalation half-cycles as separate peaks. This is why early analysis saw ~2 peaks per breath in some windows.
- The cardiac band shows systolic + dicrotic-like bumps per cardiac cycle, causing time-domain methods (hilbert, zerocross, peaks) to overcount by a factor of ~1.7-1.9x.
- The overcount ratio k is subject/coupling-stable but NOT a clean integer. It ranges from 1.18-1.93 depending on subject, band, and sensor coupling geometry.

---

## Ground Truth

### Current (2026-04-22)
| Band | Signal | Method | Reference |
|---|---|---|---|
| Respiratory | Nasal airflow (Flow) | Peak detection via neurokit2 | AASM gold standard |
| Cardiac | ECG | Pan-Tompkins R-peak detection via neurokit2 | Beat-level precision |

Automatic fallback to Thorax (resp) and Pleth (cardiac) if primary signals fail. Quality filtering rejects physiologically impossible intervals.

### Previous (deprecated)
ACF on bandpassed Thorax/Pleth — replaced because Thorax is prone to postural artifact and Pleth gives pulse rate (not heart rate), which diverges during arrhythmias and is delayed by pulse transit time.

---

## Rate Estimation Methods

### Respiratory Rate

**Best method: Scaled loose peaks / k**
- Loose peak detection (prom_factor=0.05, min_dist_s=0.4) captures all physiological events
- Per-session k calibrated from 50 random 1-min windows (median ratio of CAP peaks to GT peaks)
- Cross-session median k = 1.31, range [1.18-1.61]
- **Performance: MAE 2.20 br/min** (25% improvement over baseline 3.09)

Method comparison (S1N1, 5 random 2-min windows):
| Method | MAE (br/min) | Notes |
|---|---|---|
| peaks | 1.47 | Best per-window accuracy |
| zerocross | 2.36 | Parameter-free |
| hilbert | 2.48 | Parameter-free |
| spectral | 2.91 | Stuck at 15.0 br/min — frequency resolution too coarse |
| acf | 6.19 | Worst — sub-harmonic lock-in on CLE-CRE |

### Cardiac Rate

**Best method: Scaled Hilbert instantaneous frequency / k**
- Hilbert envelope frequency divided by per-session k
- Per-session k calibrated from 50 random 1-min windows
- Cross-session median k = 1.67, range [1.48-1.93]
- **Performance: MAE 4.19 BPM** (77% improvement over ACF baseline 18.29)

Method comparison (S1N1, whole-night):
| Method | MAE (BPM) | Notes |
|---|---|---|
| hilbert / k=1.74 | 4.33 | Best overall |
| zerocross / k=1.92 | 4.48 | Close second |
| peaks / k=1.93 | 5.76 | Good |
| acf (tuned) | 15.81 | Sub-harmonic lock-in, not fixable with prominence tuning |
| envelope (HF-band) | 16.84 | Fixed from NaN, 75% coverage |
| spectral (tuned) | 23.31 | Locks onto 2x harmonic with finer resolution |

### Scaling Factor (k) Properties
- k is NOT 2.0 — the naive "halve the peak count" over-corrects everywhere
- k is subject/coupling-stable: clusters by subject (e.g., OS003/OS005 cardiac k ~1.66; OS006 ~1.50)
- Night-to-night stability varies: 3/6 subjects have Dk <= 0.03, but 1/6 has Dk = 0.19
- 50 random 1-min calibration windows are sufficient (k_diag ~ k_whole)
- Per-night calibration recommended over per-subject or global constant
- k is continuous, not integer — removed all hardcoded /2 assumptions (2026-04-22)

### Known Limitations
- **Pearson r remains weak-to-negative** across all methods. Scaling corrects the mean bias but not per-window responsiveness. GT itself varies only sigma ~4 BPM across the night, making r hard to raise.
- **ACF sub-harmonic lock-in** is the core cardiac failure mode. Neither tighter prominence nor longer window helps. Needs lag-range constraint with rolling prior, or bimodality detection.
- **Spectral resolution** is too coarse at short windows. Longer windows let it lock onto 2x harmonic more precisely, which makes it worse.

### k(t) as a Running Biomarker

Full writeup: `notebooks/k_biomarker_writeup.md`

k is not just a static calibration constant — computed as a time series k(t) = raw_CAP_rate / GT_rate per window, it carries physiological information:

**k_cardiac is a biomarker of BCG waveform complexity:**
- Varies systematically with sleep stage (Kruskal-Wallis p = 10^-130): N1 highest (1.71), REM lowest (1.58)
- Correlates with HRV (Spearman r = -0.25 vs SDNN) and EEG delta power (r = -0.16)
- Slow-varying: autocorrelation halflife 1-4 min — physiological timescale, not noise
- Physical mechanism: sympathetic/parasympathetic balance changes the relative amplitude of systolic vs dicrotic BCG features at the temple

**k_resp is primarily a noise/quality indicator:**
- Weaker stage effect (p = 10^-52), strongest correlation with movement (r = 0.29)
- Fast autocorrelation (0.5 min) — window-level noise dominates
- Useful as a quality gate: k_resp > 2.0 or < 0.8 flags unreliable windows

**Implications:**
- Stage-aware k calibration (k_N1=1.71, k_REM=1.58 instead of global 1.67) should reduce cardiac MAE
- k_cardiac is a direct CAP-derived correlate of sleep stage — add to sleep phase detection feature set
- S6N2 anomaly (k_cardiac = 0.79) needs investigation: only session where Hilbert undercounts

---

## Key Findings

1. **Both heart rate and respiratory rate signals are present in the capacitive data** and can be extracted with classical signal processing methods.

2. **The overcount phenomenon is consistent and correctable.** Time-domain methods (hilbert, zerocross, peaks) systematically overcount events by a subject-stable factor k. This is a coupling/geometry effect, not a flaw — the BCG waveform genuinely has multiple detectable features per physiological cycle.

3. **Per-session calibration is the key insight.** A 50-minute calibration window against any reference (even a lower-quality one) enables MAE reductions of 25% (resp) to 77% (cardiac).

4. **ACF is the wrong default estimator for CAP data** in both bands. It works on clean PSG signals (Thorax, Pleth) but locks onto sub-harmonics in the noisier capacitive signal.

5. **Channel quality matters.** CLE-CRE is best on average, but CRE has genuine double-peak structure and CLE is sometimes better. Per-window channel selection could help.

6. **k_cardiac is itself a physiological biomarker.** Its temporal variation tracks sleep stage and autonomic tone, not just sensor coupling. The BCG waveform complexity measured at the temple encodes cardiovascular state.

---

## Sleep Phase Detection from CAP

Full plan: `notebooks/SLEEP_PHASE_DETECTION_PLAN.md`

### Rationale
The CAP differential (CLE-CRE) carries a superposition of respiratory, cardiac, and movement signals. Each sleep stage modulates these differently:
- **N3 (deep sleep):** very regular slow breathing, slow stable HR, no movement
- **REM:** irregular breathing, variable HR, muscle atonia but eye movement artifacts
- **N2:** moderate regularity, K-complexes may appear as transients
- **Wake:** movement, irregular patterns, high-frequency muscle artifact

PCA on multi-band features should separate these modes — PC1 likely captures gross amplitude/motion, PC2-3 may split respiratory vs cardiac regularity, and later PCs may capture stage-discriminating variability patterns.

### Expected discriminative features
- **Resp rate variability** — N3=low, REM=high (strongest expected discriminator)
- **Delta-band power** in CLE-CRE — tracks slow-wave activity
- **Movement index** — separates Wake from all sleep stages
- **Cardiac rate variability** — autonomic tone differs across stages

### Approach
4-phase plan: (1) extract ~30-40 epoch-level features per 30s window, (2) PCA + mode analysis to understand feature structure, (3) unsupervised clustering (GMM) to see if natural clusters match sleep stages, (4) supervised classification (RF, HistGBT, MLP) with LOSO CV and temporal smoothing (HMM/Viterbi).

### Data available
- PSG sleep profiles with 30s epoch labels (Wake, N1, N2, N3, REM) for all 12 sessions
- Existing infrastructure: bandpass filters, acc artifact removal, Welch band-power ratios, sliding-window quality features, LOSO CV framework

---

## Slow Wave Sleep

### Hypotheses
- Slow wave sleep is connected to deep sleep (N2/N3); if SWS is going well then REM follows; if it's short then REM may not occur
- Low-magnitude Thorax periods correspond to increased low-frequency power in CAP data
- Sleep harmonics should be visible in spectrogram analysis

### Questions to Investigate
- Can we detect events corresponding to SWS in CAP data?
- Does Thorax signal amplitude correlate with CAP low-frequency magnitude?
- Compare spectrogram to SWS analysis — are harmonics observed?
- Projection methods (PCA, DMD) for isolating SWS components

---

## Sleep Apnea

### Data Available
- **Flow** signal gives apnea type classification (obstructive vs central)
- **Effort** channels available for effort-based detection
- PSG event annotations may be available for supervised validation

### Questions to Investigate
- Can apnea events be detected from CAP capacitive data alone?
- What signatures do apnea events leave in the resp and cardiac bands?
- Sleep-staging-based rate analysis — do rates change systematically around apnea events?

---

## Artifacts and Data

| File | Description |
|---|---|
| `artifacts/peak_ratio_per_session.csv` | Resp scaled-peaks k and MAE per session |
| `artifacts/hilbert_scaled_per_session.csv` | Cardiac Hilbert k and MAE per session |
| `notebooks/peak_ratio_method_writeup.md` | Presentation-ready resp method writeup |
| `notebooks/ANALYSIS_LOG.md` | Detailed chronological log of all analyses with parameters and results |
