# Paper Workstreams — Independent CLI Session Prompts

Master checklist for finishing the paper. Each workstream below is a **self-contained
prompt** to paste into a fresh Claude Code CLI session (sessions start cold — each prompt
re-states its own context). Run order respects dependencies: **A must finish before B/C.**
D, E, F are independent and can run in parallel.

**Global conventions every session must follow** (stated in each prompt too):
- Python: `C:\Users\adity\anaconda3\python.exe` (NOT bare `python` — broken Store stub).
- Log: code→`CHANGELOG.md`, findings→`notebooks/ANALYSIS_LOG.md`; commit+push per unit.
- Don't reprocess raw signal more than needed; caches live in `artifacts/`.
- **Cardiac GT = ECG R-peaks (gold standard, unchanged). Respiratory GT = the new
  consensus (Flow+Thorax+Abdomen+RIPSum), see `CONTINUATION_RATE_DETECTION.md`.**
- **Report within-session TRACKING correlation + a temporal-shuffle null, not MAE alone**
  — MAE flatters constant predictors (resp spectral is a constant 0.25 Hz; nperseg=400).
- **Resp-rate correlations are grid-alignment fragile** — always put mask rate and GT on
  the IDENTICAL grid (no merge_asof slop).

Status legend: [ ] todo  [~] partial  [x] done

---

## A. (PREREQUISITE) Wire consensus respiratory GT as canonical  [ ]
**Depends on:** nothing. **Blocks:** B, C.

> Read the "CONSOLIDATED RESPIRATORY GT" section of `CONTINUATION_RATE_DETECTION.md`. The
> consensus resp GT is built+validated in `artifacts/consolidated_resp_gt.parquet` (Flow+
> Thorax+Abdomen+RIPSum, per-session quality gate, apnea labelled; cross-sensor agreement
> r=0.48). Cardiac GT (ECG) is unchanged. Do:
> 1. Add `gt_resp_rate_consensus()` to `sleep_monitor/ground_truth.py` and make
>    `gt_sliding_rates()` use it for resp, keeping Flow-only as a graceful legacy fallback
>    (the validation loader lacks Flow/Abdomen → fall back to Thorax). Add a `method=`
>    arg so Flow-only stays reachable.
> 2. Re-attach consensus GT to `artifacts/mask_phase_a.parquet` resp rows on the IDENTICAL
>    grid (sample consensus at exact mask epoch times; 30s is a multiple of 5s — do NOT
>    merge_asof). Save a refreshed cache.
> 3. Sanity-check: cardiac rows untouched; resp coverage now 100%.
> Use `C:\Users\adity\anaconda3\python.exe`; log to CHANGELOG + ANALYSIS_LOG; commit+push.

---

## B. Comprehensive rate detection: resp + cardiac, multichannel × multimethod, tracking  [ ]
**Depends on:** A. **Blocks:** C.

> Read `CONTINUATION_RATE_DETECTION.md` and `DETECTOR_B_PLAN.md`. Goal: a SYMMETRIC,
> comprehensive evaluation that makes the two rate-detection routes (resp, cardiac)
> comparable, characterising real-time detection — can the mask capture VARIATION, not just
> chase mean MAE. Build on cached CAP estimates in `artifacts/mask_phase_a.parquet` (CAP
> rates are GT-independent — do not recompute). Cardiac GT = ECG; resp GT = consensus.
> Produce, identically for BOTH bands:
> 1. Multichannel × multimethod matrix (CLE,CRE,CH,avg,diff × spectral,peaks_loose,hilbert,
>    cwt) — per-(channel,method) MAE AND within-session tracking r.
> 2. Detector B (responsive tracker per DETECTOR_B_PLAN.md): variation-carrying methods,
>    multichannel mean/median fusion, short overlapping windows, minimal smoothing, NO
>    spectral. Same-grid eval.
> 3. Tracking battery: within-session Pearson r vs smoothed GT; Δ-tracking (Δrate vs ΔGT);
>    transient vs steady segments; one-sample significance across 12 sessions; TEMPORAL-
>    SHUFFLE NULL (the make-or-break test — clearing the null, not r>0, is "pass").
> 4. Two-operating-points framing per band: robust-mean (low MAE, ~0 tracking) vs
>    responsive Detector B (tracks variation, higher MAE). Tradeoff plot MAE vs tracking-r.
> 5. Bound the achievable: report the reference's own ceiling (resp Flow-vs-RIPSum r≈0.48;
>    cardiac—two ECG-derived estimates or accept ECG as exact).
> Known result to confirm/extend: windowed estimators (peaks/hilbert/spectral/CWT/Viterbi)
> all give within-session cardiac r≈0; resp ≤0.12 and possibly artifact. State pass/fail
> per band against the shuffle null. Checkpoint each phase to `artifacts/`. Figures →
> `writeup/figures/mask_rate_detection/`. Log + commit per unit.

---

## C. Rate detection paper section (DOCX)  [ ]
**Depends on:** A, B.

> Use the `docx` skill. Read `writeup/paper/{OUTLINE,DRAFT,CLAIMS,KEY_NUMBERS,TABLES,
> FIGURES}.md`, the rate findings in `notebooks/ANALYSIS_LOG.md`, and figures in
> `writeup/figures/mask_rate_detection/`. `scripts/generate_rate_consolidation_docx.py` is
> an existing docx-generation template to follow. Write the rate-detection section as a
> Word doc covering: (a) the mask carries resp+cardiac bands; (b) accurate per-session MEAN
> rate recovery with calibration (resp/cardiac MAE, Bland-Altman); (c) multichannel ×
> multimethod comparison; (d) HONEST real-time tracking result — within-session correlation
> vs shuffle null, the two-operating-points (robust-mean vs responsive) framing, symmetric
> for resp and cardiac; (e) the consensus-GT methodology + its uncertainty (Flow-vs-RIPSum
> r=0.48) as a rigor point. Update `KEY_NUMBERS.md`/`TABLES.md` with final numbers first,
> then generate the docx to `writeup/`. Commit+push.

---

## D. Resp + cardiac bands present in the sleep-mask spectrogram  [ ]
**Depends on:** nothing (supports B/C).

> Goal: demonstrate, figure-first, that the CAP sleep-mask signal contains energy in the
> respiratory band (0.1–0.5 Hz) and cardiac band (0.5–3.0 Hz). Use `sleep_monitor/spectral.py`
> (sliding-window band powers) and `sleep_monitor/viz.py` spectrogram helpers; bands in
> `sleep_monitor/config.py`. For a few representative sessions (e.g. S1N1, S5N1), make
> annotated CAP spectrograms (CLE-CRE / avg) with the resp and cardiac bands marked, and
> overlay/compare the band-power time course against PSG-derived rate to show the bands are
> physiological (peaks at the breathing/heart frequency). Per project convention ALWAYS
> include spectrograms in CAP plots. Quantify: band-power SNR / concentration in each band
> vs out-of-band. Figures → `writeup/figures/`. Use `C:\Users\adity\anaconda3\python.exe`;
> log + commit.

---

## E. SWS demonstration: harmonic ridges + ladders  [ ]
**Depends on:** nothing.

> Read the `analysis/slow_wave/CLAUDE.md` and `project_slow_wave_analysis` notes. Stages 1-3
> (harmonic ridge/ladder detection in CAP spectrograms = slow-wave sleep) are done; ridge
> tracking is in `sleep_monitor/harmonics.py`. Goal: a paper-ready DEMONSTRATION that
> harmonic ridges/ladders in the CAP spectrogram correspond to N3/SWS. Produce: per-session
> spectrograms with detected ridges/ladders overlaid and the PSG hypnogram aligned, showing
> ridges concentrate in N3; quantify ridge presence vs sleep stage (e.g. ridge metric by
> stage, AUC for N3). If extending: Stage 4 = predict N3 from CAP harmonic features +
> band powers (multivariate, LOSO). Figures → `writeup/figures/` (or analysis/slow_wave
> outputs). Log + commit per unit.

---

## F. Groundwork: capacitive (mask) vs contact EEG comparison  [ ]
**Depends on:** nothing.

> Read `analysis/swa_validation/CLAUDE.md` (Steps 0-4, Lucey et al. 2019 SWA replication:
> capacitive EEG vs contact EEG). Status: workspace exists, Step 0 (data inventory) is the
> blocking first task. Do Step 0 first: inventory which sessions have usable mask-EEG and
> PSG contact-EEG channels, sampling, alignment. Then Steps 1-2: 6s epochs, Welch PSD,
> 1–4.5 Hz SWA band, artifact rejection, per-subject SWA time course from mask vs contact
> EEG, correlation/Bland-Altman. Deliverable: groundwork report establishing whether the
> mask can serve as a non-contact EEG proxy for slow-wave quantification, with the key
> agreement figures. Use `C:\Users\adity\anaconda3\python.exe`; log to the swa_validation
> ANALYSIS_LOG; commit+push.

---

## Suggested scheduling across accounts
- **Account 1 (sequential, the critical path):** A → B → C.
- **Account 2 (parallel):** D, then E.
- **Account 3 (parallel):** F.
D feeds C (spectrogram figure), so finish D before C's docx if possible.

---

## C-interim. Interim manuscript draft for review (run BEFORE rates fully finalized)
**Depends on:** A done, B's results available, D/E/F done. Produces a review copy; C
(final) reruns after rates finalize.

Status as of this entry: A done, B done (tracking FAILS both bands — see
`memory`/CHANGELOG), D/E/F done. This interim draft integrates all of that.

> Use the `docx` skill. Produce a readable INTERIM manuscript draft for the user to review;
> mark the rate section PRELIMINARY (rate workstream still finalizing).
> **Reconcile, do not transcribe:** the `writeup/paper/*.md` scaffold predates the consensus
> GT and the tracking-FAILS finding and overclaims. Read current truth from `CHANGELOG.md`
> (2026-06-18), `notebooks/ANALYSIS_LOG.md`, `CONTINUATION_RATE_DETECTION.md`,
> `analysis/swa_validation/` logs; update `CLAIMS.md` + `KEY_NUMBERS.md` to match, then write.
> Encode honest results: (1) CAP carries resp 0.1-0.5 Hz + cardiac 0.5-3 Hz bands (signal_
> validation figs 5-7); (2) consensus resp GT, Flow-vs-RIPSum ceiling r=0.47, cardiac GT=ECG;
> (3) mean-rate recovery works (resp ~1.1 br/min, card ~3.9 BPM, confirm post-consensus
> numbers from reports/rates/mask/); (4) within-session TRACKING FAILS both bands vs shuffle
> null (RESP r=+0.058 p=0.34 4/12; CARD r=-0.188 p=0.85 3/12) — two-operating-points framing,
> symmetric, "mean rate + stage structure, not instantaneous variation"; (5) harmonic ridges
> significant (KW p<1e-16) but weak N3 classifier (LOSO AUC 0.534); (6) mask-vs-contact-EEG
> groundwork per swa_validation results. Deliverable: `writeup/CAP_sleep_mask_manuscript_
> draft.docx` (Abstract, Intro, Methods, Results in that order, Discussion), key figures
> embedded, "PRELIMINARY — rate numbers pending" banner on rates, and an OPEN-ITEMS list at
> the end. Use the anaconda python; log + commit+push.

---

## C-FULL. Assemble the ENTIRE manuscript (all workstreams done)
**Depends on:** A, B, D, E, F all done. Supersedes C-interim once all analysis is final.

> Use the `docx` skill. Write the COMPLETE manuscript integrating every finished workstream
> (GT reconciliation, rate detection, signal-band validation, slow-wave harmonics, EEG
> groundwork) into one coherent paper for review. Assembly + honest reconciliation, not new
> analysis.
> THESIS: rigorous honest characterization of a capacitive temple sleep mask (CLE/CRE/CH +
> accel, 12 nights, 6x2 subjects, PSG ref) — what it can/can't measure: (1) carries resp+
> cardiac band energy; (2) accurate MEAN rate but NOT instantaneous within-session tracking;
> (3) spectral harmonics stage-associated but weak N3 classifier; (4) capacitive-vs-contact
> EEG SWA groundwork.
> RECONCILE not transcribe: scaffold `writeup/paper/*.md` predates consensus GT + tracking-
> FAILS finding and overclaims. Establish truth from CHANGELOG (2026-06-18), ANALYSIS_LOG,
> CONTINUATION_RATE_DETECTION.md, reports/rates/mask/, reports/slow_wave/, analysis/swa_
> validation/. VERIFY every number against source files (scaffold values are pre-consensus).
> Update CLAIMS.md + KEY_NUMBERS.md to reconciled truth FIRST, then write docx from them.
> STRUCTURE: Title/Abstract, Intro, Methods (participants; CAP sensors + OLS+NLMS + bands;
> GT = ECG cardiac + multi-signal consensus resp w/ Flow-vs-RIPSum r~0.47 uncertainty;
> estimators + Detector B; stats incl shuffle null + LOSO), Results [1 signal validation
> figs5-7; 2 rate detection mean-rate figs2,3,18 + honest tracking-FAILS figs19-23,14,15,
> two operating points, ceiling 0.47; 3 harmonics KW p<1e-16 / LOSO AUC 0.534 figs paper_*;
> 4 EEG groundwork per swa_validation], Discussion (capabilities vs limits + why), Limitations,
> Conclusion. TABLES: participants; per-band mean-rate MAE/bias/LoA; tracking battery
> (r, delta-r, %beat-null, p) resp+card; harmonic/N3 LOSO — from reports/ CSVs.
> DELIVERABLE: `writeup/CAP_sleep_mask_manuscript.docx`, key figures embedded, closing
> OPEN-ITEMS/REVIEW-NOTES section (unverified numbers, revised claims, still-evolving
> sections incl. B). Anaconda python; log + commit+push.
