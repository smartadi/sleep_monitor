<!--
EXECUTION PLAN (2026-07-16) — how to finish the paper analyses + reorganize the repo.
Supersedes the scheduling half of PAPER_WORKSTREAMS.md. Scope, section map, reorg
target, and the run/parallelization strategy. Source of truth for prose stays
writeup/paper/{DRAFT,OUTLINE,KEY_NUMBERS,CLAIMS}.md.
-->

# Paper execution & repo-reorg plan

## 0. The one insight that drives everything

**The heavy compute is ~80% already cached.** Nearly every result the paper needs is
sitting in `artifacts/*.parquet` and `reports/*/**.parquet` (rate estimates, GT,
harmonic/ridge/ladder features, cap-swa features, spindle triggered arrays, coherence).
The slow step — loading each 100 Hz overnight csv.gz once and preprocessing it — has
already been paid for each of these. So the remaining work is mostly **assembly**
(figure consolidation, re-pointing, prose) plus **two small new analyses**, not a big
recompute.

Consequence for "speed it up": the bottleneck is NOT CPU. Spawning parallel CLI
sessions does not make cached-figure work faster; it just lets you iterate on
independent sections at once. The real levers are (1) never reprocess raw signal that a
cache already covers, and (2) split the six independent sections across parallel tracks.

## 1. Paper scope (proposed)

Six results sections, staging + thorax out (→ supplementary or removed):

| § | Section | Analysis status | Work remaining | New compute? |
|---|---------|-----------------|----------------|--------------|
| A | Signal validation | **DONE** (figs 1–10) | reconcile KEY_NUMBERS SNR, write §3.1 | none |
| B | Rate — primary methods | **DONE** (mask pipeline) | pick primary, demote rest to suppl., re-point figs, write | none |
| C | Ridge / harmonic | mostly done | **new: resp-band vs SWA-band separation**; dedup ridge figs; write | small |
| D | Spindles (onset-level) | infra done | **new: per-spindle low-band detection rate** (1-line extension); fig; write | tiny |
| E | EEG vs CAP SWA | **DONE** (negative) | move figs to writeup, write §3.x | none |
| F | SWA definition-level | **DONE** (cap-swa) | decide framing vs C, write | none |

Only C and D need any raw loading, and each loads the 12 recordings once (~3 min). No
multiprocessing framework is warranted for that.

## 2. Section-by-section

### A. Signal validation — CLOSE OUT
- Present: figs 1–4 (coherence, freq-match, surrogates, channel); figs 5–7 (band energy);
  **figs 8–10 (added 2026-07-16: simplified SNR, mean-capacitance drift, freq characteristics)**
  + `signal_characterization_summary.csv`.
- Do: KEY_NUMBERS §3.1 still quotes the OLD SNR ("+11 to +27 dB vs 3.5–5 Hz floor"). Replace
  with the simplified-SNR numbers (resp median 0.8–15.5 dB, cardiac 0.1–2.6 dB vs 5–10 Hz
  floor) OR keep both and label them. Write §3.1. **Effort: S (writing only).**

### B. Rate — primary in main text, everything else supplementary
- **Primary (main text):** respiratory = **spectral, CLE−CRE**, pooled MAE 1.09 br/min
  (median 0.91), k≈0.97; cardiac = **agreement-fused multichannel** (peaks_loose base),
  pooled MAE 3.91 BPM (best single peaks_loose/CRE 3.41), k≈1.95. Source of truth =
  `reports/rates/mask/final_summary.json` + KEY_NUMBERS §3.2.
- **Supplementary:** the 5-method×5-channel×2-band benchmark; CWT/VMD/MUSIC/ML trackers;
  ACF cardiac failure; fusion-strategy ablations; oracle headroom; the 6-method
  within-session tracking NEGATIVE; Kalman hybrid; k-biomarker (now confounded) + k-vs-age.
- Do: (1) re-point `FIGURES.md` figs 5–16 from `rate_consolidation/` → `mask_rate_detection/`
  (numbers there are the current ones); (2) fix DRAFT §2.6 (still describes the superseded
  CWT+Viterbi cardiac story); (3) write §3.2–3.4 + a short honest "tracking not achievable"
  paragraph. **Effort: M (figure re-pointing + prose, ~no compute).**

### C. Ridge / harmonic
- **Long ridges:** `run_ridge_overlay.py` (MIN_PERSIST 5 min) + `ridge_consolidation.py` —
  DONE. Headline: long persistent ridges concentrate in **N2** (n=2573) not N3 (n=166);
  N3 discrimination weak (LOSO AUC 0.534). Report directionally (OUTLINE rule 3, no p-values).
- **Long + prominent harmonic detection:** `harmonic_rigor.py` + `ladder_quantify.py` — DONE.
  Clean comb-fit ladders intermittent (~2% non-motion, up to 20% S6N2); 76% cardiac / 24%
  resp; 63% harmonic / 37% inharmonic; channel dissociation CH→cardiac (f0≈1.1 Hz),
  CRE→resp (f0≈0.3 Hz).
- **NEW — resp-band vs SWA-band separation:** does NOT exist yet. The current "separation"
  is resp-vs-*cardiac* ladders; the SWA/delta band (0.5–4 Hz) is only handled in the Lucey
  replication. Build one script that, per session, decomposes the CAP spectrum into
  respiratory-band (0.1–0.5 Hz) vs SWA/delta-band (0.5–4 Hz) power/ridge activity over the
  night and shows how they separate (and whether the delta-band activity is really cardiac
  harmonics, not cortical SWA — ties to §E). Can reuse the PSD cache in
  `signal_characterization_cache.pkl`; ~1 new figure + table. **Effort: M (one new analysis).**
- Do: also dedup the THREE near-duplicate per-session spectrogram figure families
  (`spectrograms/S*_spectrogram_ridges.png`, `harmonics/paper_overlay_S*.png`,
  `harmonics/S*_harmonic_fullnight.png`) → pick one for main, rest supplementary.

### D. Spindles — onset-level detection rate in CAP low band
- Story: electrical sigma NEGATIVE (per-spindle AUC 0.50); **low band (0–3 Hz) POSITIVE** —
  spindle-locked mechanical bump, +0.45 dB CH, validated by random-N2 + arousal controls.
- **NEW (small):** `spindle_analysis.py::event_vs_control_auc()` currently runs only on the
  sigma envelope. The 0–3 Hz envelope `env_c` is already computed but never scored. Run the
  same per-spindle AUC / detection-rate on `env_c` → a per-spindle **detection rate** ("X% of
  N2 spindles show a resolvable low-band bump vs matched controls") + a distribution figure.
  Not stat-heavy, exactly as requested. Arrays already in `spindle_ersp.npz` /
  `triggered_averages.npz` — **no raw reload needed. Effort: S.**
- Also: update the stale `analysis/spindles/CLAUDE.md` (still says pure-negative).

### E. EEG vs CAP SWA — negative, close out
- `run_swa_validation.py` + `swa_pipeline.py` DONE. CAP vs EEG SWA r=0.015; CAP N3 AUC 0.490
  (chance) vs EEG self-AUC 0.740 (pipeline validated). Honest negative: the mask does not
  sense cortical delta.
- Do: copy the 5 headline figs from `analysis/swa_validation/outputs/` → `writeup/figures/
  swa_validation/`; write §3.x + methods. This resolves the OUTLINE "SWS-EEG PENDING" gate
  that currently blocks the Intro/Discussion. **Effort: S.**

### F. SWA definition-level
- `cap_swa_definition.py` + `swa_classifier_experiment.py` + `cap_swa_trials.py` DONE.
  Mechanical CAP-SWA score → N3 per-subject AUC 0.675 (6/6, consistent direction, unlike
  ridge); definition/threshold sweep q=0.60 F1-optimal; 59 movement-initiated trials, N3
  enriched 2.15×; autonomic hypotheses 5/6 contradicted (bradycardia, not tachycardia).
- Decision needed: C (harmonic/ridge SWS signature) and F (mechanical CAP-SWA definition)
  and E (no cortical SWA) are three angles on "does the mask see slow-wave sleep." Pick the
  narrative: E = not electrically; F = yes, mechanically (best N3 marker); C = spectral
  correlates. **Effort: M (framing + writing).**

## 3. Repo reorganization (target structure)

Problems today: 4.1 GB `reports/`, figures scattered across 5 trees
(`writeup/figures`, `reports/*`, `notebooks/plots`, `analysis/*/outputs`, top-level
`plots/`), the same topic duplicated 3× , four parallel manuscript docx efforts, 48
force-tracked binaries amid an otherwise ignore-everything policy.

Target:
```
sleep_monitor/            # library — DO NOT TOUCH (already clean)
analysis/<area>/          # code + its own CLAUDE.md; outputs/ gitignored scratch only
artifacts/                # data-of-record caches (parquet/csv) — keep, prune sweep/
writeup/
  paper/                  # THE canonical manuscript: DRAFT.md -> build_docx.py -> .docx
    figures/              #   final figures ONLY, one subfolder per section (symlink/copy
                          #   from reports, never the reverse)
  _legacy/                # every other docx effort (main/, harmonics/, root docx, SFN)
  edits/                  # professor review cycle (as-is)
reports/                  # regenerable working output — gitignored, safe to wipe
archive/                  # superseded scripts + dead analyses (already exists)
```
Concrete moves (all reversible — archive, don't delete):
- Pick `writeup/paper/` as the single manuscript; move `writeup/{main,harmonics}/`,
  root `*.docx`, and SFN docx to `writeup/_legacy/`.
- Curate `writeup/paper/figures/<section>/` as the ONLY paper-figure location; leave the
  bulk PNGs in `reports/` (regenerable) and stop mirroring into `notebooks/plots`.
- Retire to `archive/`: superseded scripts (`detect_sws.py`, `detect_trials.py`,
  `run_harmonic_ladders.py`, `run_harmonic_allsessions.py`, `plot_harmonics_s1n1.py`,
  `run_peak_tracker.py`, `run_ridge_stage3.py`, v1 rate scripts, all `thorax_*`, apnea
  plots), and the 1,880-PNG `analysis/spindles/outputs/spindle_windows_S2N1/` dump.
- Delete stray junk: top-level `plots/`, `~$P_harmonic_ridge_analysis.docx`,
  `writeup/edits/_x.py`, `artifacts_sweep.log`, empty `writeup/paper/{data,figures}` stubs,
  empty `reports/{staging,thorax}`.
- Normalize git policy: track ALL of `writeup/paper/figures/**` (force-add), nothing else
  binary. Everything in `reports/` stays regenerable-and-ignored.

## 4. How to run it — parallelization strategy (the answer)

**Do NOT reprocess raw signal.** Confirm each section reads existing caches; only C and D
touch raw data and each loads the 12 recordings once.

Two independent parallel axes — use both:

**Axis 1 — compute across the 12 recordings (only if a real re-extraction is needed).**
The nights are embarrassingly parallel. If any step needs a fresh full-night pass, do it
in ONE script with a `multiprocessing.Pool(6)` over the 12 sessions, writing per-session
caches — the pattern `signal_characterization.py` already uses. This is "across sessions
in parallel" done correctly (recording-sessions, one machine, 6 workers). Splitting the
same CPU work across chat/CLI sessions does not help.

**Axis 2 — development across the 6 paper sections (the real iteration win).**
Sections A–F are independent (shared only via read-only caches), so run them concurrently.
Two ways, pick per your workflow:
- *Parallel background agents from one session* — I launch A–F as agents; you stay in one
  place to review. Best when the work is mechanical (figure regen, re-pointing, prose from
  known numbers) — which is most of it.
- *Separate CLI sessions (the PAPER_WORKSTREAMS "across accounts" model)* — best when a
  section needs your judgment turn-by-turn (C's resp-vs-SWA framing, D's spindle framing,
  F's narrative). Paste one self-contained prompt per section.

**Recommended schedule:**
1. **Reorg first (blocking, ~30 min, one session).** Do the folder moves + git policy so
   every downstream section writes into the final structure. Reversible (archive-not-delete).
2. **Fan out A, B, E in parallel** — pure assembly/writing, no new compute, low risk →
   background agents.
3. **C and D in parallel** — each has one small new analysis; C reuses the PSD cache, D
   reuses the spindle npz. Either agents or your own sessions if you want to steer framing.
4. **F after C/E** — needs the SWS narrative decided across C/E/F.
5. **Assemble** — `writeup/paper/build_docx.py` from the updated DRAFT.md once sections land.

Net: one blocking reorg + one blocking "confirm caches" check, then 5 sections run
concurrently with essentially no CPU contention because they're I/O-light cache reads.

## 5. Decisions needed before executing
1. Canonical manuscript = `writeup/paper/` (DRAFT.md build), others → `_legacy/`? (Y/N)
2. Reorg = archive-not-delete (reversible)? (Y/N)
3. Run mode: parallel background agents here, or separate CLI sessions per section?
4. SWS narrative: keep E (no cortical SWA) + F (mechanical marker) + C (spectral) as three
   honest angles, or merge?
