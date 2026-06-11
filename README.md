# Sleep Monitor — CAP Sensor Analysis Platform

Analysis platform for capacitive temple sensors (CLE, CRE) targeting respiratory rate, cardiac rate, and sleep staging from overnight recordings with PSG ground truth.

**12 overnight recordings, 6 subjects × 2 nights.**

---

## Session Viewer

Interactive Plotly Dash app for exploring all sessions and channels.

![Dash app](https://img.shields.io/badge/Plotly_Dash-localhost:8050-blue)

```bash
python scripts/session_viewer.py
```

Opens at **http://localhost:8050**. Features:

- **Session selector** — switch between all 12 recordings
- **Spectrograms** — CAP channels (CH, CLE, CRE) and PSG channels (EEG, EOG, ECG, Flow, Thorax, Abdomen), with global or per-column normalization
- **Hypnogram + apnea events** — color-coded sleep stages and apnea annotations from PSG
- **Band power** — EEG bands, delta sub-bands, or custom user-defined frequency bands; ratio or absolute mode with optional low-pass smoothing
- **Spectral ridges** — track power at specific frequencies over time (enter e.g. `0.5, 1.0, 1.5` or `0.5:2.5:0.25`)
- **Feature overlays** — respiratory rate, cardiac rate, head position, motion, PCA components
- **Raw signals** — any CAP or PSG channel with optional bandpass filtering and z-scoring
- **Scroll zoom** enabled for detailed inspection

All panels share a synchronized time axis (hours from recording start).

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd code
pip install -r requirements.txt
```

### 2. Install the package

```bash
pip install -e .
```

### 3. Data

Place overnight recording directories alongside the `code/` folder:

```
sleep monitor/
├── code/                          ← this repo
├── overnight_6subject_pelthupdate_030526/   ← raw CAP + PSG
├── overnight_6subject_complete_032626/      ← PSG-only
└── combinedDataAnalyses_041626/             ← validation (short recordings)
```

Data paths are configured in `sleep_monitor/config.py`.

### 4. Run the viewer

```bash
python scripts/session_viewer.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
sleep_monitor/          Core Python package (pip-installable)
├── config.py           Paths, channel names, frequency bands
├── sessions.py         Session metadata registry (12 sessions)
├── loader.py           CSV.GZ loading, sleep profiles, apnea events
├── preprocessing.py    Accelerometer artifact removal (OLS + NLMS)
├── filters.py          Butterworth bandpass/lowpass/highpass
├── rates.py            5 rate estimators + scaled variants
├── ground_truth.py     PSG reference extraction (ECG, Flow)
├── quality.py          Per-window quality scoring
├── spectral.py         Sliding-window band powers
├── staging.py          Epoch-level feature extraction (~40 features)
├── evaluate.py         Pipeline orchestration
├── harmonics.py        Harmonic ridge detection
├── motion.py           Head position and motion classification
└── viz.py              Plotting utilities

scripts/                Runnable analysis scripts
├── session_viewer.py   ← Interactive Dash viewer (main entry point)
└── ...

analysis/               Scoped analysis workspaces
├── rates/              Rate estimation and validation
├── slow_wave/          Slow wave sleep detection from CAP
├── staging/            Sleep phase classification
├── projections/        PCA, UMAP, t-SNE, DMD
└── thorax/             CAP → thorax effort analysis

tests/                  Unit tests (pytest)
reports/                Analysis outputs (HTML, PNG, CSV)
```

---

## Dependencies

- Python 3.10+
- numpy, scipy, pandas, matplotlib
- scikit-learn, umap-learn
- neurokit2 (PSG ground truth extraction)
- dash, plotly (session viewer)
- pytest (tests)

All listed in `requirements.txt`.

---

## Conventions

- **Channel default:** CLE−CRE (OLS regression differential)
- **Calibration:** per-session k from 50 random 1-min windows
- **Cross-validation:** LOSO (leave-one-subject-out)
- **Frequency bands:** respiratory 0.1–0.5 Hz, cardiac 0.5–3.0 Hz
