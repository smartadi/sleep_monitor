# Sleep Session Viewer

Interactive browser-based viewer for overnight sleep recordings captured with capacitive temple sensors (CAP) alongside polysomnography (PSG) ground truth.

Built with [Plotly Dash](https://dash.plotly.com/) — runs locally, no server needed.

---

## What You Can See

| Panel | Description |
|-------|-------------|
| **Spectrograms** | Time–frequency heatmaps for CAP channels (CH, CLE, CRE) and PSG channels (EEG, EOG, ECG, Flow, Thorax, Abdomen) |
| **Hypnogram** | Color-coded sleep stages (Wake, N1, N2, N3, REM) from PSG scoring |
| **Apnea events** | Apnea and hypopnea episodes overlaid on the timeline |
| **Band power** | EEG-band power ratios (delta/theta/alpha/beta), delta sub-bands, or custom frequency bands |
| **Spectral ridges** | Track power at user-specified frequencies over time |
| **Feature overlays** | Respiratory rate, cardiac rate, head position, motion, PCA components |
| **Raw signals** | Any CAP or PSG channel with optional bandpass filtering and z-scoring |

All panels share a synchronized time axis. Scroll-zoom is enabled for detailed inspection.

---

## Setup

### 1. Install Python dependencies

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

### 2. Install the sleep_monitor package

```bash
pip install -e .
```

### 3. Set up data

The viewer expects two dataset directories inside a common parent folder:

```
<your-data-folder>/
├── overnight_6subject_pelthupdate_030526/
│   └── overnight_6subject_pelthupdate_030526/
│       ├── S1 - AB/
│       ├── S2 - CD/
│       └── ...                 ← CAP + PSG synchronized recordings (.csv.gz)
│
└── overnight_6subject_complete_032626/
    └── overnight_6subject_complete_032626/
        ├── S1 - AB/
        ├── S2 - CD/
        └── ...                 ← PSG-only data (sleep profiles, apnea events)
```

Tell the viewer where your data lives by setting the `SLEEP_DATA_DIR` environment variable to the parent folder path:

**Windows (PowerShell):**
```powershell
$env:SLEEP_DATA_DIR = "C:\Users\you\path\to\sleep monitor"
```

**Windows (Command Prompt):**
```cmd
set SLEEP_DATA_DIR=C:\Users\you\path\to\sleep monitor
```

**macOS / Linux:**
```bash
export SLEEP_DATA_DIR="/home/you/path/to/sleep monitor"
```

> If `SLEEP_DATA_DIR` is not set, the viewer defaults to the original development path and will fail to find data.

### 4. Run

```bash
python scripts/session_viewer.py
```

Opens at **http://localhost:8050** in your browser.

---

## Controls

| Control | What it does |
|---------|-------------|
| **Session** dropdown | Switch between all 12 overnight recordings (6 subjects × 2 nights) |
| **Panels** checkboxes | Toggle hypnogram, apnea, band power, ridges panels |
| **Spectrogram channels** | Choose which CAP/PSG channels get spectrograms |
| **Band channel** | Which CAP channel to compute band power from |
| **Band mode** | Ratio (normalized 0–1) or absolute power |
| **Band set** | EEG bands, delta sub-bands, or custom (see below) |
| **Custom bands** | Define your own: `resp:0.1-0.5, cardiac:0.8-2.0` |
| **Ridge freqs** | Track specific frequencies: `0.5, 1.0, 1.5` or range `0.5:2.5:0.25` |
| **Band LP (mHz)** | Low-pass smoothing cutoff for band power / ridge traces |
| **Spec norm** | Global (default) or per-column normalization |
| **Feature overlays** | Overlay respiratory rate, cardiac rate, head position, motion, PCA |
| **CAP raw / PSG raw** | Display raw time-domain signals |
| **Z-score** | Normalize raw signals to zero mean, unit variance |
| **Filter (Hz)** | Bandpass/highpass/lowpass the raw signals |

---

## Dependencies

All in `requirements.txt`:

- numpy, scipy, pandas, matplotlib
- scikit-learn, umap-learn
- neurokit2
- dash (includes plotly)
- pytest

---

## Dataset

12 overnight recordings from 6 subjects (2 nights each). Each session contains:

- **CAP sensors**: CH (combined), CLE (left ear), CRE (right ear), plus 3-axis accelerometer
- **PSG channels**: EEG, EOGl, EOGr, ECG, respiratory Flow, Plethysmography, Thorax effort, Abdomen effort
- **Annotations**: 30-second epoch sleep staging, apnea/hypopnea events

All signals sampled at **100 Hz**, synchronized between CAP and PSG.
