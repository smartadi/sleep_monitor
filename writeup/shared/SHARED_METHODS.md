# Shared Methods

Sections written here are common to both manuscripts. Copy or adapt into each paper's draft.

---

## Participants and Recording Protocol

- 6 healthy subjects (OS001–OS006), age range 20s–60s, PSQI < 5
- 2 consecutive nights each → 12 overnight recordings
- Duration range: 4.11–8.66 hr
- Concurrent PSG: EEG (standard montage), ECG, nasal airflow (Flow), photoplethysmography (Pleth), thoracic respiratory effort
- CAP sleep mask: 3 CPC capacitive sensors (CLE = left eye, CRE = right eye, CH = forehead/top), 3-axis accelerometer
- Sampling rate: 100 Hz for all channels
- **Data ref:** `sleep_monitor/sessions.py` SESSION_META registry

## Signal Preprocessing

- **Artifact removal:** OLS regression of accelerometer magnitude from each CAP channel — removes motion artifact while preserving physiological signal (`sleep_monitor/preprocessing.py`)
- **Channel selection:** CLE-CRE differential (regression residual) as primary analysis channel — maximizes common-mode rejection of non-ICP signals. Individual channels (CLE, CRE, CH) and average channel also used in multi-channel analyses.
- **Bandpass filtering:** Butterworth order 3, respiratory band [0.1, 0.5] Hz, cardiac band [0.5, 3.0] Hz (`sleep_monitor/filters.py`)
- **Motion gating:** Epochs with accelerometer RMS > 3 MAD above session median excluded from harmonic analysis

## Ground Truth Derivation

- **Respiratory GT:** Peak detection on nasal airflow (Flow) channel using neurokit2. Flow is the AASM gold standard for respiratory events. Fallback to thoracic belt if Flow unavailable.
- **Cardiac GT:** Pan-Tompkins R-peak detection on ECG using neurokit2 → true beat-level heart rate.
- **Sliding-window rates:** 60s window, 5s step. Count GT peaks per window → instantaneous rate. Quality filter: reject physiologically impossible intervals.
- **Sleep staging:** PSG-scored 30s epochs (Wake, N1, N2, N3, REM) used as ground truth labels.
- **Code:** `sleep_monitor/ground_truth.py`

## Session Metadata

| Session | Subject   | Duration (hr) | Samples   |
|---------|-----------|---------------|-----------|
| S1N1    | OS001-KJK | 7.95          | 2,862,001 |
| S1N2    | OS001-KJK | 7.63          | 2,748,001 |
| S2N1    | OS002-LDI | 7.73          | 2,784,001 |
| S2N2    | OS002-LDI | 6.77          | 2,436,001 |
| S3N1    | OS003-LCW | 6.93          | 2,496,001 |
| S3N2    | OS003-LCW | 8.66          | 3,117,001 |
| S4N1    | OS004-CJH | 6.18          | 2,224,400 |
| S4N2    | OS004-CJH | 6.02          | 2,166,001 |
| S5N1    | OS005-CJY | 4.11          | 1,479,001 |
| S5N2    | OS005-CJY | 4.74          | 1,707,001 |
| S6N1    | OS006-SK  | 5.16          | 1,857,001 |
| S6N2    | OS006-SK  | 5.78          | 2,082,001 |
