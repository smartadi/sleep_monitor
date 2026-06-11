# Reports

Analysis outputs for human review. Every script that produces findings saves here
so you can open one file and verify results without touching code or notebooks.

## Structure

| Folder | What goes here | Produced by |
|---|---|---|
| `rates/` | Rate accuracy tables, Bland-Altman plots, k-biomarker summaries | `scripts/rate_accuracy_analysis.py`, `scripts/run_validation.py` |
| `validation/` | Signal coherence results, surrogate test outputs, per-session validation | `scripts/signal_validation.py`, `scripts/plot_validation_report.py` |
| `staging/` | Sleep stage classification results, confusion matrices, feature importances | staging analysis scripts |
| `slow_wave/` | Harmonic analysis results, SWS detection outputs | `analysis/slow_wave/` scripts |

## Format convention

Scripts should save reports in one of:
- **HTML** (interactive Plotly figures) — open in any browser, no Python needed
- **PNG** (static plots) — for quick visual inspection
- **CSV/parquet** — tabular results that Claude can read and summarise

## How to verify an analysis

1. Ask Claude to run the relevant script
2. Claude saves output here
3. Open the file directly to inspect the numbers / plots yourself
4. If something looks wrong, ask Claude to explain the calculation or re-run with different parameters

## Workflow

Claude runs script → saves report here → you open file → you verify → done.

No notebook required. No re-running cells. No scrolling through output.
