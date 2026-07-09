# Rate-Detection Section — Thorough Review (keep what's relevant)

_Scope: the rate-estimation Methods (§3.5 L124–126, §3.6 L127–131), the mean-accuracy and
tracking Results ("§3.2" L192–349; "§3.3" L350–382), and the rate parts of the Discussion
(§4.1 L487–488, §4.2 L494–497, §4.3 L503–504, §4.5 L511; Limitations L515, L517). Grounded
in the actual pipeline (`scripts/run_mask_rate_detection.py`) and new analyses:
calibration_requirement.py, peaks_per_beat.py, rpeak_triggered_waveform.py, k_vs_age.py._

---

## 0. The biggest single gap: the pipeline that produced the numbers is never stated

§3.5 (L125) lists **six base estimators + four advanced trackers**, then the Results quote
numbers — but the manuscript never says *which* estimator, *which* channel, whether fusion
or smoothing was applied. A reviewer cannot map Table 3 to a method. The actual paper
pipeline (traced) is:

> Per 60-s window (30-s step), per channel, compute base rates → (optional per-channel
> Smart-Fusion + cross-channel SQI/agreement fusion) → **per-session scalar k** (median
> raw/GT over the night) → **causal 3-epoch median smoothing**.

- **Respiratory headline (median 0.91–0.95, pooled 1.09 br/min):** `spectral` (Welch PSD
  peak) on the CLE−CRE differential, k≈0.97, smoothed. Single method, single channel.
- **Cardiac headline (single-channel CRE 3.41; multi-channel-agreement 3.36–3.91 BPM):**
  `peaks_loose` (loose prominence peak-count), k≈1.95, smoothed.
- **No Kalman filter.** (Kalman was in the superseded "hybrid/best-of-both" pipeline.) If
  any Kalman language survives in the draft, delete it.

**Fix:** add one short, explicit "rate pipeline" paragraph + a small schematic (draft §5
below). This alone resolves the reviewer's "which pipeline is Table 3?" objection.

---

## 1. KEEP — the relevant, strong material

- **k definition + stability check** (L126): median raw/GT ratio; |k_diag − k_whole| ≤ 0.04
  from 50 random windows. Keep — it establishes k is a stable estimate.
- **Two operating points** (L376–378): spectral = lowest MAE but constant predictor;
  FWD = tracks but no significant correlation. This honest framing is a strength — keep and
  make it a labelled subsection.
- **Within-session tracking negative** (L350–382): temporal-shuffle null + Flow–RIPSum
  ceiling (r≈0.47). This is the section's rigor. Keep in full.
- **Oracle headroom** (L345–346): resp headroom is in method not channel; cardiac headroom
  is in channel. Keep — it justifies design choices.
- **Per-session Table 3** (L199–344): keep, but fix labelling/columns (see §3).
- **Mechanism paragraph** (L503–504): the "morphology-locked frequency" argument is the best
  passage in the section. Keep and *now back it with the new waveform evidence*.

---

## 2. CUT / TRIM — not relevant to the reported results

- **The estimator zoo in Methods** (L125). Six base + four advanced (CWT, Viterbi, MUSIC,
  VMD, ACF, zero-cross, strict peaks, adaptive) are listed as if used, but only **spectral**
  and **peaks_loose** produce any reported number. Trim to: "we surveyed N estimators;
  spectral (respiratory) and loose peak-counting (cardiac) were selected; the remainder,
  including advanced trackers (CWT ridge, STFT+Viterbi), are reported only for the
  within-session tracking analysis (§3.3), where all methods failed." Moving the zoo into
  the tracking section is where it is actually relevant (proving exhaustiveness of the
  negative), not the headline.
- **Redundant Smart-Fusion machinery description** if you adopt the "fusion adds nothing"
  simplification (see §3, new result) — you can describe fusion in one sentence and report
  it was not beneficial, rather than detailing Karlen/Nemati/SQI weighting.
- **Per-stage MAE orphan** (L347–349): "resp worst in REM, cardiac worst in Wake" has no
  statistic and no follow-up. Either support it (per-subject consistency count) or cut it to
  one clause in the Bland–Altman discussion.

---

## 3. FIX — errors, overclaims, inconsistencies (with new results)

### [FIX-pipeline] Specify method/channel/fusion/smoothing (L125, L192–196)
As §0. Also state the window (60 s / 30 s step) and that smoothing is a causal 3-epoch
median (matters for the "real-time capable" claim).

### [NEW-RESULT — calibration requirement] Answer "does it need calibration?" quantitatively
`analysis/rates/calibration_requirement.py` (from cached predictions):

| Band | per-session k | fixed population k | first-10-min k |
|------|--------------|--------------------|----------------|
| Resp | 0.94 br/min  | **1.15** (k=1.00)  | 2.00 br/min |
| Card | 3.36 BPM     | **4.56** (k=1.95)  | 10.1 BPM |

Interpretation to add to the paper:
- **Respiratory is effectively calibration-free** — using a fixed k=1.0 (i.e. the raw
  spectral peak, no calibration) costs only +0.2 br/min (0.94→1.15). This is a strong
  deployment claim and should be stated.
- **Cardiac genuinely requires per-session calibration** — a fixed population prior (k=1.95)
  costs +1.2 BPM, and a realistic 10-minute calibration is *not viable* (10.1 BPM, worse
  than the population prior, because 20 epochs give a noisy/drifting k). This honestly bounds
  Limitation 3 (L515): the mask needs either whole-night self-calibration or a population
  prior for cardiac, not a short warm-up.

### [NEW-RESULT — fusion adds nothing] Simplify the pipeline claim (L346, L196)
Multi-channel fusion vs single differential channel (per-session median MAE):
resp 0.95 → 0.94 (Δ0.01); cardiac 3.42 → 3.36 (Δ0.06). **Fusion is within noise.** State
plainly that the single-channel per-window estimator is the operational pipeline and
multi-channel fusion did not measurably help — an Occam simplification that strengthens the
paper. (Reconciles the current text at L196/L346, which reads as if fusion is the method.)

### [FIX] Table 3 cardiac `r` column is unexplained and alarming (L272–344)
Near-zero/negative r (S1N1 −0.203, S1N2 −0.434) in the *mean-accuracy* table reads as
"anti-correlates with truth." It's the within-session tracking result arriving early. Either
remove it from Table 3 (put in §3.3) or add one sentence: mean-rate accuracy and
within-session tracking are distinct axes.

### [OVERCLAIM] Median MAE hides catastrophic sessions (L196, L272–344)
Cardiac median 3.41 hides S2N1 11.67, S6N1 17.96, S6N2 8.57 BPM (3/12 nights >8) and LoA
[−24,+23]. Report median **and** IQR/range **and** n sessions >5 BPM, attribute S6 to the
coupling regime already flagged (k=1.35/0.94), and state the LoA reflect epoch-level spread
while the validated quantity is the per-session mean. Consider reporting accuracy under a
declared coupling-quality inclusion criterion.

### [KEEP+SUPPORT] Respiratory "spectral is best" is a constant-predictor caveat (L200/L377)
Correctly noted (df=0.25 Hz at 30 s → ~constant 15 br/min). Keep this honesty; it is the
whole point of the two-operating-points framing. Make sure the 0.91 br/min number is never
read as within-night tracking accuracy.

### [CONSISTENCY] Numbers to reconcile
- Cardiac headline appears as 3.41 (single CRE), 3.36 (fused_agree per-session median), 3.91
  (fused pooled). State which is the headline and which the operating-point comparison.
- Respiratory 0.91 (median) vs 1.09 (pooled) — label each.
- k stability "≤0.04" (L126) vs the calibration-drift reality (first-10-min k much worse) —
  not contradictory (one is whole-night reproducibility, the other is short-window
  estimation) but say so, or a reviewer sees tension.

### [FIX] "randomly selected calibration windows" (L126)
Specify 50 one-minute windows and that k is the whole-night median; note the deployment
implication from the calibration-requirement result above.

---

## 4. New results ready to fold into the rate section

1. **Cardiac k mechanism, now demonstrated** (`peaks_per_beat.py`, `rpeak_triggered_waveform.py`):
   CAP produces mean 1.78 peaks/heartbeat vs mean k 1.89 (8/10 sessions on the identity
   line); ECG R-peak-triggered CAP average shows a beat-locked, often two-bump pulse. Turns
   the k≈2 "systolic+dicrotic" claim (L196, L503) from asserted to shown. One figure
   retro-validates mean-overcount, tracking failure, harmonic ladder, and age-invariant
   cardiac k. (Draft: `rpeak_waveform_section_draft.md`.) Honest caveat: population-level, not
   a clean universal per-session proof (S4/S6 coupling deviate).
2. **k-vs-age** (`k_vs_age.py`): cardiac k age-invariant (ρ=+0.37) — a *passed prediction* of
   the fixed-morphology model; respiratory k exploratory decline (ρ=−0.83, Bonferroni n.s.).
   k reproducibility is age-independent; no other CAP feature tracks age (bounding null).
   (Drafts: `k_vs_age_section_draft.md`, `age_features_section_draft.md`.)
3. **Calibration requirement** and **fusion-adds-nothing** (§3 above).

Delete any surviving **within-session "k(t) biomarker"** framing — it was killed
(corr(k,rate)=−0.83, GT-free proxy −0.06). The surviving, defensible claim is "k is a stable
per-*subject* morphological constant."

---

## 5. Recommended Methods rewrite for §3.5 (drop-in)

> **Rate estimation.** Respiratory and cardiac rates were estimated on a 60-s sliding window
> (30-s step). We surveyed multiple per-window estimators and selected the spectral peak
> (Welch PSD) for respiration and loose prominence-based peak counting for the cardiac band;
> the remaining estimators, including CWT-ridge and STFT–Viterbi trackers, are reported only
> for the within-session tracking analysis (§3.6). Because the capacitive pulse is not a
> clean one-cycle-per-event waveform, raw counts are rescaled by a per-session scalar k
> (median of raw-estimate/reference over 50 random 1-min windows; whole-night k agreed to
> ≤0.04). Estimates were smoothed with a causal 3-epoch median filter. Multi-channel
> (SQI-weighted, agreement-gated) fusion was evaluated but did not improve accuracy over the
> single CLE−CRE differential (ΔMAE < 0.1), so the single-channel estimator is the
> operational pipeline. Respiration required essentially no calibration (fixed k=1.0 → 1.15
> br/min); cardiac required whole-night self-calibration (fixed population k → 4.56 BPM; a
> 10-min calibration was insufficient).

## 6. Recommended Results structure for §3.2–§3.3

1. Mean respiratory accuracy (spectral/diff, k≈1, near calibration-free) — lead with this.
2. Mean cardiac accuracy (peaks_loose/k≈2) — full distribution, not just median; coupling
   caveat; calibration requirement.
3. Cardiac pulse morphology figure (R-peak triggered + peaks-per-beat) → the k≈2 mechanism.
4. k as a subject-level constant: stability, age-invariance (cardiac), exploratory resp-age.
5. Two operating points (MAE vs tracking).
6. Within-session tracking negative (null + Flow–RIPSum ceiling) — the exhaustive estimator
   zoo lives here.

## 7. Recommended figures (trim to what argues)
- Pipeline schematic (window → estimator → k → smooth; two cardiac operating points).
- Bland–Altman (resp, cardiac) — keep.
- Cardiac pulse morphology (R-peak triggered average + peaks-per-beat vs k) — NEW, high value.
- k vs age (cardiac invariant / resp exploratory) — NEW.
- Two-operating-points (MAE vs within-session r) — keep.
- Tracking null bands + Flow–RIPSum ceiling — keep.
- Move all-session trace grids and the estimator-zoo comparison to Supplement.
