# Rate Detection — Continuation Handoff

**For the next Claude Code CLI session (fresh account).** Read this first, then resume
from the cached data — do NOT reprocess raw signals.

Last updated: 2026-06-17 by Opus 4.8 session (hit ~20% budget).

---

## TL;DR of where we are

A paper-ready mask rate-detection pipeline exists and works:
- **Resp: MAE = 1.09 br/min** (spectral on diff channel, k≈0.97 — calibration-free)
- **Cardiac: MAE = 3.91 BPM** (peaks_loose, multi-channel agreement fusion, k≈1.95)

Two follow-up questions were then investigated (adaptive k + channel oracle). The
answers **reframe the next priorities** — see "What we learned" below.

---

## DO NOT reprocess raw signals

Everything needed is cached. Raw reprocessing is ~8 min/run; cache ops are seconds.

| Cache file | Contents |
|---|---|
| `artifacts/mask_phase_a.parquet` | 93,190 rows: every (session × epoch × channel × band) with `r_spectral, r_hilbert, r_peaks_loose, r_peaks_strict, gt_hz, quality, snr_db, spec_conc, acf_prom, motion_db, stage` |
| `artifacts/mask_phase_b.parquet` | Smart-fusion + multi-channel fused rates (18,638 rows) |
| `artifacts/mask_phase_c.parquet` | k-calibrated + smoothed, all strategies (111,828 rows) |

Channels: `CLE, CRE, CH, avg, diff`. Bands: `resp, card`. Methods as columns above.

**Scripts:**
- `scripts/run_mask_rate_detection.py` — full 6-phase pipeline (Phase A/B/C cached; D/E/F regenerate figures in ~1 min)
- `scripts/analyze_adaptive_k_and_oracle.py` — cache-only follow-up analysis
- `scripts/analyze_window_size_spectral.py` — window-size / spectral-resolution sweep (reprocesses diff channel; ~3 min)
- Python: `C:\Users\adity\anaconda3\python.exe` (NOT the bare `python` — that's a broken Store stub)

---

## ⚠️ CRITICAL CAVEAT (2026-06-18): MAE was flattering everything — tracking is weak

Two findings that the next session MUST account for. **Stop reporting MAE alone; report
within-session tracking correlation too.**

### The resp 'spectral win' is a window-size artifact
- `rate_spectral` uses `nperseg = max(64, fs*4) = 400` → df = 0.25 Hz. The whole resp
  band (0.1–0.5 Hz) is ~1.6 bins. Result: **r_spectral(resp) = 0.25 Hz in 9317/9319
  epochs** — a literal constant predictor (15 br/min), within-session corr with GT = **0.00**.
- It "wins" on MAE only because sleeping adults breathe near 15 br/min, so predicting
  the population mean scores well. It carries **zero** respiratory information.
- Window sweep (`window_size_sweep.csv`, figs 10/11): a high-resolution spectral
  (full-window periodogram + parabolic interp) is *noisier*, not better (resp MAE ~4).
  Longer windows lower MAE for all methods but within-session corr stays ≈0 for resp.
  Resp rate is genuinely stable (within-session GT std ≈ 2.0 br/min), so there is little
  to track — MAE-to-mean is legitimate but it is NOT "tracking respiration."

### Per-session k + stable rates means "predict the session mean" already wins
- The honest metric is **within-session** correlation. Pooled correlation is inflated
  by between-session mean-matching (per-session k makes each session's constant
  prediction ≈ that session's mean GT → looks like correlation across sessions).
- Cardiac: within-session GT std ≈ 9.7 BPM, range ≈ 58 BPM (lots to track), yet
  within-session corr of peaks ≈ 0. BUT median-MAE (3.7) < predict-the-mean (~6.5),
  so peaks captures the *coarse* HR trend; the ~0 corr is partly GT R-peak detection
  noise on 30s windows. **Next:** re-evaluate cardiac tracking with smoothed GT or
  per-stage mean HR before concluding it fails.

### Adaptive k as a biomarker — does NOT hold up as an *independent* biomarker (yet)
Tested k_card(t) = peaks/GT (supervised) and peaks/spectral (GT-free proxy), diff channel:
- k(t) is structured (lag-1 autocorr 0.52, not white noise) and has modest stage
  structure (median k: N1 2.03 > N3 1.98 > N2 1.92 ≈ Wake 1.89 ≈ REM 1.89).
- **corr(k_gt, GT_rate) within session = −0.83** → per-epoch k is mostly just absorbing
  1/rate (because peaks doesn't track HR, k is forced to carry the variation). So k(t)
  is NOT independent of rate.
- **corr(k_gt, k_selfsup) = −0.06** → the GT-free proxy can't recover true k, so it
  isn't computable in deployment.
- **For k to be a genuine biomarker you need a method that tracks rate first**; then the
  *residual* k after regressing out rate can be tested for morphology/autonomic signal.
  Mechanistic hook: k≈2 ⇒ biphasic pulse (systolic + dicrotic notch); k(t) ∝ dicrotic
  prominence ⇒ proxy for vascular tone / autonomic state. This is the real biomarker
  experiment (links to [[k-biomarker-plan]] [[project_cap_mean_drift]]).

---

## What we learned (the two follow-up questions)

### Q1. Is spectral winning for resp only because it's k-free? → NO (hypothesis refuted)
Tested self-supervised adaptive `k(t)` = causal median(peaks/spectral), no GT needed.
- Resp diff: spectral **1.39** vs peaks/k_static 2.01 vs peaks/**k(t) self-sup 1.93**.
  Adaptive k helps peaks slightly but it **still loses to spectral**. Spectral is
  genuinely better for resp, not just because it's k-free.
- Cardiac diff: peaks/k_static **4.46** vs peaks/**k(t) self-sup 16.28** (FAILS).
  Self-supervised k **poisons cardiac** because it anchors to spectral, and spectral
  is terrible for cardiac (20.40 BPM). **Lesson: self-supervised adaptive k needs a
  reliable k-free anchor; cardiac has none.**
- **Calibration drift IS real:** first-10-min k is much worse than full-session k
  (resp 2.01→2.67, card 4.46→8.90). This motivates a *better* within-session adaptive
  scheme — just not a spectral-anchored one.
- Figure: `writeup/figures/mask_rate_detection/fig8_adaptive_k.png`
- Data: `reports/rates/mask/adaptive_k_results.csv`

### Q2. Why does the single 'diff' channel win, contradicting the channel-diversity/oracle result?
- **Resp: channel diversity is irrelevant.** Oracle-over-channels = 1.08 ≈ diff 1.09.
  All channels carry the same respiratory info. The resp headroom is in **METHOD**
  diversity: oracle-over-methods (diff) = **0.54**, full oracle (ch×meth) = **0.16**.
- **Cardiac: channel diversity is THE headroom (user was right for cardiac).**
  Oracle-over-channels = **1.58 BPM** vs our fused 3.91 and diff-only 4.46.
  Win distribution is ~even (CLE 19%, CRE 21%, CH 20%, avg 20%, diff 20%) — **no
  channel dominates**, confirming channels carry different cardiac info per epoch.
  Our SQI-weighted fusion captures almost none of this (~2.3 BPM left on the table).
- Full cardiac oracle (ch×meth) = **0.51 BPM** — theoretical ceiling.
- Figures: `fig7_oracle_headroom.png`, `fig9_channel_diversity.png`
- Data: `reports/rates/mask/oracle_headroom.csv`, `channel_win_{resp,card}.csv`

---

## Next-session priorities (highest value first)

1. **Smart cardiac channel fusion** (biggest win: 3.91 → toward oracle 1.58 BPM).
   Our quality-weighting barely beats mean-fusion. Try, evaluated on cache Phase A:
   - per-epoch channel selection from SQI/features (learned weights, LOSO-CV)
   - per-stage best-channel maps (does the best channel depend on sleep stage?)
   - phase-aware combination (channels may be anti-correlated; check CLE/CRE phase)
   - redundancy/consensus weighting (down-weight outlier channels)
   *Question to answer:* what fraction of the 2.3 BPM oracle headroom is realizable
   WITHOUT GT (i.e., using only SQI/quality to pick/weight channels)?

2. **Realistic within-session adaptive k for cardiac** (calibration drift: 4.46→8.90
   with 10-min cal). Build a GT-free adaptive k that does NOT anchor to spectral:
   - cross-channel consensus (channels agree on relative scale even if absolute k drifts)
   - quality-gated slow drift tracking of the per-channel peak ratio
   *Goal:* close the 4.46 (oracle-GT k) vs 8.90 (10-min k) gap with no GT.

3. **Resp method fusion** (oracle method 0.54 vs current 1.09). Smartly combine
   spectral + peaks + hilbert per epoch (the resp headroom is in methods, not channels).

4. **Biology angle (for paper):** k_card≈2 is a stable ~2:1 ratio — likely the
   capacitive pulse waveform has two inflections per cardiac cycle (systolic +
   dicrotic/diastolic), so peak-counting doubles. Worth verifying on raw waveform
   (single example epoch) for a mechanistic figure. S6 sessions are anomalous
   (k_card = 1.35 / 0.94 vs typical 1.9–2.1) — investigate whether S6 sensor coupling
   differs (links to [[project_cap_mean_drift]]).

---

## Conventions / gotchas
- Per-session k = median(raw/gt), clipped to [0.3, 5.0], needs ≥10 valid epochs.
- MAE reported as **median** |error| × 60 (robust to outliers), not mean.
- `analyze_adaptive_k_and_oracle.py` has one harmless `All-NaN slice` warning (dead
  line ~168, `oracle_ch` computed but unused) — safe to delete.
- Oracle numbers use GT for selection → they are **upper bounds**, not achievable.
  The point is to size the headroom, not to claim the result.
- Logging: code → `CHANGELOG.md`, findings → `notebooks/ANALYSIS_LOG.md`. Commit +
  push after each unit (see [[feedback_commit_push_logging]]).
