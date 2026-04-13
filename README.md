# Sleep Monitor Code

This repository contains the analysis code and notebooks for the sleep monitor workspace.

## Contents

- `cap_rates.py`: command-line respiratory and cardiac rate analysis utilities
- `add_psg.py`: helper script for injecting PSG-related notebook cells
- `analysis*.ipynb`: exploratory and analysis notebooks

Generated outputs and local machine state are intentionally ignored so the repository stays code-focused.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- The notebooks are treated as source files and are expected to be tracked.
- `analysis_raw_executed.ipynb`, `plots/`, virtual environments, and local tool folders are ignored as generated or machine-local artifacts.