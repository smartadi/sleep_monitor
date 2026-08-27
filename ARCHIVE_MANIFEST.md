# Archive manifest — stale / redundant research (2026-08-26)

Candidates for archiving into `archive/2026-08-26_superseded/`, with the evidence
for each. **Nothing has been moved** — this is the plan; execute per tier on your
say-so. All moves would be `git mv` (fully reversible); gitignored binaries
(`*.docx`) move on disk only.

Existing archive locations already in the repo: `archive/`, `writeup/_legacy/`,
`writeup/_archive/`.

---

## Tier 1 — documented-superseded, not imported by live code (safe)

Named in the "Superseded — do not quote" list of `analysis/rates/CLAUDE.md`, or
the retracted k-biomarker work; confirmed not imported by anything in
`sleep_monitor/`, `analysis/`, or `scripts/`.

| file | why stale | replaced by |
|---|---|---|
| `analysis/rates/k_vs_age.py` | k–age on the degenerate estimator (ρ=−0.83 artifact) | `analysis/rates/k_age_rerun.py` |
| `analysis/rates/k_age_prior.py` | age-prior on superseded k series | `k_age_rerun.py` |
| `analysis/rates/k_age_fused.py` | fused-k age variant | `k_age_rerun.py` |
| `notebooks/analysis_k_biomarker.py` | produced the retracted "k_cardiac tracks stage / p=10⁻¹³⁰" | — (claim retracted) |
| `notebooks/analysis_k_biomarker_phase3.py` | same k-biomarker phase work | — |
| `notebooks/k_biomarker_writeup.md` | superseded (rates/CLAUDE.md) | `analysis/rates/CLAUDE.md` |
| `notebooks/peak_ratio_method_writeup.md` | superseded (rates/CLAUDE.md) | `analysis/rates/CLAUDE.md` |
| `notebooks/validation_methods.md` | superseded (rates/CLAUDE.md) | current manuscript §3 |
| `notebooks/validation_results.md` | superseded (rates/CLAUDE.md) | current manuscript §4 / Table S1 |

Execution:
```bash
mkdir -p archive/2026-08-26_superseded
git mv analysis/rates/k_vs_age.py analysis/rates/k_age_prior.py analysis/rates/k_age_fused.py archive/2026-08-26_superseded/
git mv notebooks/analysis_k_biomarker.py notebooks/analysis_k_biomarker_phase3.py archive/2026-08-26_superseded/
git mv notebooks/k_biomarker_writeup.md notebooks/peak_ratio_method_writeup.md notebooks/validation_methods.md notebooks/validation_results.md archive/2026-08-26_superseded/
```

## Tier 2 — redundant writeup drafts

Canonical manuscript is `writeup/main/CAP_sleep_mask_manuscript_main.docx`
(project_manuscript_state). These are older/parallel drafts.

| file | note |
|---|---|
| `writeup/SFN 2026 Overnight study Sleep mask - V4.docx` | old conference draft (gitignored) |
| `writeup/SFN 2026 Overnight study Sleep mask - V5.docx` | old conference draft (gitignored) |
| `writeup/SFN 2026 Overnight study Sleep mask - updated results.docx` | old (gitignored) |
| `writeup/SFN 2026 Overnight study Sleep mask - updated results v2.docx` | old (gitignored) |
| `writeup/paper/CAP_sleep_mask_paper.docx` | separate older paper (gitignored) |
| `writeup/paper/CLAIMS.md` | SUPERSEDED banner, rate sections predate 2026-08-14 |
| `writeup/paper/DRAFT.md` | SUPERSEDED banner |
| `writeup/paper/KEY_NUMBERS.md` | SUPERSEDED banner |
| `writeup/paper/EXECUTION_PLAN.md` | SUPERSEDED banner |
| `notebooks/{rate_analysis,validation_methods,validation_results}.docx` | docx of Tier-1 superseded writeups (gitignored) |

*Caution:* `*.docx` are gitignored, so archiving them is a filesystem move only
(not visible to git). The four `writeup/paper/*.md` are tracked.

## Tier 3 — exploratory notebooks (old; judgment call)

Superseded by the scoped `analysis/` workspaces, but may hold reference
explorations. Keep the two planning docs (`PROJECTION_PLAN.md`,
`SLEEP_PHASE_DETECTION_PLAN.md`) and the active `ANALYSIS_LOG.md`.

- `notebooks/01_overview.ipynb` … `11_projection_delta_subs.ipynb` (12 notebooks, incl. duplicate `08_*`)
- `notebooks/analysis_dmd_cardiac.py`, `analysis_dmd_rank_sweep.py`,
  `analysis_delay_pca_cardiac.py`, `analysis_pca_stacked_cle_cre.py`,
  `analysis_morphology.py`, `analysis_sws_band_ratios.py`

---

## Explicitly NOT archived

- `scripts/run_mask_rate_detection.py` — **superseded but kept for provenance**:
  it produced the originally-published numbers and is the subject of
  `RATE_AUDIT_2026-08-06.md`. Historical record, not stale clutter.
- `analysis/rates/k_age_rerun.py`, `rerun_rate_detection.py` — current/live.
- `sleep_monitor/*` — the package.
- `notebooks/ANALYSIS_LOG.md`, `PROJECTION_PLAN.md`, `SLEEP_PHASE_DETECTION_PLAN.md`.
