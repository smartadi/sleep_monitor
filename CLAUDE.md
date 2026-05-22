# Sleep Monitor — CAP Sensor Analysis Platform

Capacitive temple sensors (CLE, CRE) → respiratory rate, cardiac rate, sleep staging.
12 overnight recordings, 6 subjects x 2 nights. PSG ground truth.

## Data paths
- Raw CAP+PSG: `C:\Users\adity\Documents\sleep monitor\overnight_6subject_pelthupdate_030526\`
- PSG-only: `C:\Users\adity\Documents\sleep monitor\overnight_6subject_complete_032626\`
- Validation (short): `C:\Users\adity\Documents\sleep monitor\combinedDataAnalyses_041626\`
- Artifacts output: `artifacts/`

## Package: sleep_monitor/
Core library — import with `from sleep_monitor import ...`

| Module | Role |
|--------|------|
| config | Paths, channel names, freq bands (resp 0.1-0.5 Hz, cardiac 0.5-3.0 Hz) |
| sessions | SleepSession dataclass, SESSION_META registry (12 sessions) |
| loader | CSV.GZ loading, sleep profile, apnea events |
| preprocessing | Accel artifact removal (OLS+NLMS), bandpass filtering |
| filters | Butterworth bandpass/lowpass/highpass, detrending, clipping |
| rates | 5 base estimators + scaled variants (peaks/k, hilbert/k) |
| ground_truth | PSG reference: ECG R-peaks, Flow peaks via neurokit2 |
| quality | Per-window quality scoring (SNR, ACF prominence, spectral concentration) |
| spectral | Sliding-window band powers (delta, theta, alpha, beta) |
| staging | Epoch-level feature extraction (~40 features) |
| evaluate | Pipeline orchestration, per-window DataFrames with metrics |
| viz | Hypnograms, rate time series, spectrograms, Bland-Altman |

## Analysis areas
Scoped workspaces with their own CLAUDE.md — work from these directories for focused context:
- `analysis/rates/` — rate estimation, k-biomarker, validation
- `analysis/slow_wave/` — slow wave sleep detection from CAP
- `analysis/staging/` — sleep phase classification via projections + ML
- `analysis/projections/` — PCA, UMAP, t-SNE, DMD, delay embedding

## Workflow rules
- **Log everything as it happens:** code changes → `CHANGELOG.md`, analysis findings → `notebooks/ANALYSIS_LOG.md`. See `LOGGING_POLICY.md` for format.
- **Commit and push regularly:** after each meaningful unit of work (new script, bug fix, analysis result), commit with a descriptive message and push. Don't batch multiple unrelated changes.

## Conventions
- Channel default: CLE-CRE (OLS regression differential)
- k calibration: per-session, from 50 random 1-min windows
- CV protocol: LOSO (leave-one-subject-out, not leave-one-session)
- Quality gating: compute scores, let downstream set thresholds
- Python: `from sleep_monitor import ...` — package is pip-installable via `setup.py`
