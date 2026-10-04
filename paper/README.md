# Paper code: wearable SEC sleep mask

The code behind *Physiological Characterization of a Wearable Capacitive Sensor Mask
During Overnight Sleep*, from raw recordings to every figure, table and quoted number,
in one command.

```
python paper/run_all.py
```

It runs each stage in order, writes figures and tables under `paper/outputs/`, and
finishes with **`outputs/PAPER_NUMBERS.md`**. That report lists every number the
manuscript quotes next to what the code computes now, marked MATCH, DIFF, NEW or EXTERNAL.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r paper/requirements.txt      # Windows; bin/pip elsewhere
set SLEEPMASK_DATA=<folder holding the two data directories>   # optional, see below
python paper/run_all.py
```

Python 3.10. Versions are pinned to the ones the outputs were made with.

**Data.** Set `SLEEPMASK_DATA` to the folder that holds the two data directories. If it
is unset, the folder that contains this repository is used. Layout:

```
<SLEEPMASK_DATA>/
  overnight_6subject_pelthupdate_030526/overnight_6subject_pelthupdate_030526/
      <subject> - <initials>/<date>/Sync_<date>/SleepMask_PSG_100Hz*_combined_<date>.csv.gz
  overnight_6subject_complete_032626/overnight_6subject_complete_032626/
      <subject> - <initials>/<date>/PSG_analysis_*/       sleep stages and scored events
```

The unworn-mask recording is read from `baseline noise/SM2_33.txt` in the repository; set
`SLEEPMASK_BASELINE` to move it. Twelve recordings: six participants, two nights each.

## Layout

```
paper/
  run_all.py          entry point; --list, --only <stage...>, --numbers
  seclib/             the library the stages run on (no other repo code is imported)
  stages/             one file per manuscript section, run in order
  checks/             port-fidelity checks against the original analysis outputs
    results/          the record of the last run of each check
  PROVENANCE.md       which original script produced each manuscript item, and the problems found
  outputs/            figures/<stage>/, tables/<stage>/, numbers/, PAPER_NUMBERS.md
  _cache/             parsed recordings and products shared between stages (deletable)
```

## Stages

| stage | manuscript | produces |
|---|---|---|
| `s01_recordings` | §2.2–2.3, Table 1 | recording durations, the shared 30-s epoch grid, motion-canceller check |
| `s02_overnight` | §2.4, §3.1; Supp S2, S4, S6 | Fig. 2, session levels, posture, left–right differential and its integral |
| `s03_signal` | §3.2; Supp S3, S5, Table S1 | band fractions, in-band SNR, SEC–PSG coherence, respiratory-reference gate |
| `s04_rates` | §2.9, Discussion; Supp S7–S12 | per-epoch rates, per-recording k, held-out calibration vs no-sensor baseline |
| `s05_ridges` | §2.7, §3.3 | Fig. 5, ridge features by stage, count-matched N3 contrasts, the low band |
| `s06_harmonic_comb` | §3.6 | Figs. 8–9, comb events, stage and REM timing |
| `s07_spindles` | §2.8, §3.4 | Fig. 6, SEC low-band and sigma response at spindles, controls |
| `s08_delta_kcomplex` | §2.8, §3.5, §3.9 | Figs. 7 and 12, delta-burst onsets, K-complex response, arousal control |
| `s09_reproducibility` | §3.7–3.8 | Figs. 10–11, night-to-night reproducibility, associations with arousal/PSQI/age |
| `s99_external` | — | registers the items made outside this code (Figs. 1, 3, 4, 13, S1) |

Each stage's docstring lists the manuscript items it makes, the original scripts it was
ported from, and every place it departs from them. Each departure is marked `# CORRECTION:`
in the code.

## How this was checked

1. **Provenance.** Every figure, table and number in manuscript V11 and supplement V3 was
   traced to the script that produced it (`PROVENANCE.md`).
2. **Faithful port.** Each stage was first made to reproduce the original outputs exactly
   or to float tolerance. `checks/<stage>_vs_legacy.py` compares against the original
   analysis outputs, and `checks/results/` keeps the record. Where a match is not exact,
   the result file explains why.
3. **Corrections.** Bugs and text–code mismatches found along the way were then fixed as
   explicit, commented changes. The effect of each fix shows up as a DIFF in
   `PAPER_NUMBERS.md`, so the text can be brought in line.

Statistics follow the cohort size: six participants, so results are descriptive and per
night or per participant. Where the manuscript quotes a pooled-epoch test, it is
reproduced and flagged.

## Not reproduced here

Fig. 1 (photographs and schematic), Figs. 3–4 and their spectral-slope statistics
(co-author analysis, no code), Fig. 13 (separate optical recording) and Fig. S1 (24-h bench
test). The hand-measured oscillation durations in Fig. 11b are entered as data, not
computed. All of these are listed as EXTERNAL in the numbers report.
