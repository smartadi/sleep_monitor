# Detector B — Responsive Rate Tracker (resp AND cardiac, symmetric)

**For a fresh Claude Code CLI session.** Read `CONTINUATION_RATE_DETECTION.md` first
for full context, then execute this plan. Python: `C:\Users\adity\anaconda3\python.exe`
(NOT bare `python`). Log to CHANGELOG.md + notebooks/ANALYSIS_LOG.md, commit+push per
unit of work. Cache lives in `artifacts/`; do not reprocess raw signal more than needed.

## Goal (capacity, not accuracy)
Show whether the sleep mask CARRIES within-session rate variation — i.e. mask-derived
rate variation reliably correlates with PSG GT variation (within-session r robustly > 0,
beating a temporal-shuffle null), demonstrating the *capacity* to track. NOT precision.
Run the IDENTICAL battery on BOTH resp and cardiac so the story is symmetric (this is the
key requirement — earlier analyses treated the two bands with different methods).

## Why this differs from prior work
Prior pipeline (`run_mask_rate_detection.py` / `run_rate_consolidation.py`) was
MAE-optimized: it used the constant spectral estimator, SQI weighting that suppresses the
variation-carrying method, and temporal smoothing that kills transients — every stage
converges to the session mean. Detector B inverts all of that. Prior tracking findings
(cache-only, honest within-session correlation, both-smoothed):
- RESP: peaks/hilbert ~ +0.10 to +0.12 (weak but consistently POSITIVE); spectral ~0
  (it is a constant 0.25 Hz predictor — `nperseg=400` artifact). 47% of GT var is noise.
- CARDIAC: peaks/hilbert/spectral/CWT/Viterbi all ~0 or negative; oracle 0.63 is ~85%
  selection-bias (shuffled-GT floor 0.54). 33% of GT var is noise.
Detector B gives cardiac the SAME responsive-detector + Δ-tracking + transient battery
that resp gets, so the comparison is fair.

## Design principles
- Variation-carrying estimators ONLY: `rate_peaks(...,prom_factor=0.05)` (loose) and
  `rate_hilbert`. NO spectral (constant). Functions in `sleep_monitor/rates.py`.
- Short, OVERLAPPING windows for time resolution: sweep W in {8,10,15,20,30}s, hop 5s.
  (For cardiac, also try {6,8,10}s — HR varies faster.) Pick W* maximizing TRACKING r,
  not minimizing MAE.
- Plain multi-channel fusion: mean AND median across CLE,CRE,CH,avg,diff. NO SQI
  weighting (it suppresses variation). Compare mean vs median vs best-single.
- Minimal causal smoothing: kernel in {1 (none),3} epochs, tuned. Don't over-smooth.
- k (per-session, from GT) is IRRELEVANT to correlation (scale-invariant) — only needed
  for the secondary MAE number. So the tracking claim is calibration-free.
- Channel prep: `remove_acc_artifact(sig, acc_mag, f_lo, f_hi, fs)` from
  `sleep_monitor/preprocessing.py`. Bands from config: RESP 0.1-0.5, CARD 0.5-3.0 Hz.
  GT via `gt_sliding_rates(sess, win_sec=W, step_sec=5)` (`sleep_monitor/ground_truth`).

## Phases (checkpoint each to parquet)
1. **Window sweep for tracking** — peaks+hilbert per channel at each W (overlapping),
   fuse, within-session r vs smoothed GT. Pick W* per band. Save
   `artifacts/detB_phase1_windowsweep_{resp,card}.parquet`.
2. **Build detector at W*, full night, 12 sessions** — fused responsive rate series +
   spectral baseline for comparison. `artifacts/detB_phase2_fullnight_{resp,card}.parquet`.
3. **Tracking evaluation (core claim), identical for both bands:**
   - Within-session Pearson r vs smoothed GT (5-epoch median), per session + aggregate.
   - Δ-tracking: corr of changes (Δrate vs ΔGT) — direct "follows variation", offset-robust.
   - Transient vs steady: r on segments where |ΔGT| > threshold vs flat segments.
   - Significance: one-sample sign/Wilcoxon test that within-session r > 0 across 12
     sessions. Temporal-shuffle NULL: permute GT per session, recompute r (200x), report
     real r vs null distribution. This is the make-or-break test (mirrors the cardiac
     oracle shuffle check).
4. **Figures** (`writeup/figures/mask_rate_detection/`):
   - All-12-session full-night traces: GT vs Detector B vs spectral, BOTH bands.
   - Transient zoom-ins (~20-min segments around real GT changes).
   - Tradeoff plot: MAE vs tracking-r, spectral vs B (the two operating points).
   - Per-session r bar chart with shuffle-null band, resp and cardiac side by side.

## Success criterion (state up front, be honest)
PASS = within-session r and Δ-r reliably positive across sessions, significantly above
the shuffle null, stronger on transient segments. Expected magnitude r ~ 0.1-0.25
("loosely tracks / carries the signal", NOT "closely tracks"). Likely outcome: resp
PASSES weakly, cardiac likely FAILS — but both reported identically. If a band sits at
the null even for Δ-r and transient-r, conclusion: mask lacks the capacity there and the
mean-rate estimator is the only defensible output for that band.

## Deliverable
A short results section: "Capacity to track within-session rate variation" with the
two-operating-points framing (robust-mean spectral vs responsive Detector B), symmetric
across resp and cardiac, with the significance/null test as the headline evidence.
