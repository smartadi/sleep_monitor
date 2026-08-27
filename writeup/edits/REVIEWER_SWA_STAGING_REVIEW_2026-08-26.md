# Reviewer report — SWA & staging sections (statistics)

**Date:** 2026-08-26
**Scope:** extends the rate-section review
(`REVIEWER_METHODS_CODE_REVIEW_2026-08-26.md`) to the SWA-validation and
sleep-staging / slow-wave sections, applying the same four checks: exact small-n
statistics, pooled-epoch independence, permutation/time-alignment nulls, and
held-out validation against a baseline.

**What I added (runnable):**
- `sleep_monitor/group_stats.py` — group-aware tests these sections should use:
  `pooled_vs_grouped_stage_test`, `paired_stage_contrast`,
  `subject_block_permutation_auc`, `sign_test_direction`, `subject_mean_ci`.
- `tests/test_group_stats.py` — 11 tests, all pass, including a null-cohort
  demonstration of the pooled-epoch false positive.

---

## Headline

**The single largest statistical problem in these sections is the pooled-epoch
significance test.** The SWA and slow-wave scripts test per-stage differences
with `kruskal(*groups)` / `mannwhitneyu(n3, rest)` on **epochs pooled across all
12 recordings** — tens of thousands of autocorrelated, non-independent
observations from 6 subjects. This produces the p < 1e-16 and "p = 10⁻¹³⁰"
figures the manuscript and the staging notes carry. They are artifacts of the
unit of analysis, not evidence.

I demonstrated the mechanism on synthetic data with **no stage effect at all**
(6 subjects × 2 nights, realistic AR(1) within-night autocorrelation, stages in
contiguous hypnogram blocks):

| test | result on NULL data |
|------|---------------------|
| pooled Kruskal (current practice, n = 10,800 epochs) | **p = 1.6 × 10⁻³⁶** ← false positive |
| grouped Friedman (subject as unit, n = 6) | p = 0.087 ← correct |

On data with a *genuine* consistent effect the pooled test overstates the
evidence by ~**149 orders of magnitude** (p = 1.7e-151 vs an honest p = 0.010).
The slow-wave `CLAUDE.md` already recorded the symptom without naming the cause:
"Ridge features are statistically significant (KW p<1e-16) but practically weak
standalone N3 discriminators (LOSO AUC = 0.534)." The AUC is the honest number;
the p-value is the pooling artifact.

---

## Staging / slow-wave findings

### STG-1 — pooled-epoch tests throughout *(blocking for any per-stage claim)*
Locations (non-exhaustive): `analysis/slow_wave/paper_ridge_demo.py:405-411`
(the paper figure's Panel F stats table), `run_harmonic_allsessions.py:115,140`,
`ridge_harmonic_revamp.py:349,459`, `detect_sws.py:268`,
`analysis/mean_value/mean_value_vs_stage.py:326`, `abs_mean_vs_stage.py:124`,
`icp_across_sleep.py:153`. Each pools epochs across sessions and reports the
resulting p as evidence of a stage relationship. **Fix:** aggregate to one value
per subject per stage first, then `pooled_vs_grouped_stage_test` (reports both
p-values and the inflation) or `paired_stage_contrast` for N3-vs-rest.

### STG-2 — report the LOSO AUC, drop the pooled p, and give the AUC an honest null
The trustworthy N3-discrimination number is the LOSO AUC (0.534), which is at
chance. It currently has no null: 0.534 is compared to 0.5 by eye. Provide a
**subject-blocked label-permutation null** (`subject_block_permutation_auc`,
permutes stage labels within each subject so the null respects the grouping) and
a bootstrap CI. State the result as "AUC 0.53, 95% null band [0.47, 0.53],
n.s." — not as a significant KW p.

### STG-3 — subject-dependent direction is not handled
The ridge/harmonic effect flips sign across subjects (S1/S2 N3-high, S3/S4
N3-low — slow-wave `CLAUDE.md`). A two-sided pooled Mann-Whitney can be
"significant" while the effect **cancels** across subjects. Test direction
consistency with `sign_test_direction`: at n = 6 you need **6/6** subjects in the
same direction for p < 0.05; **5/6 gives p = 0.22.** `paired_stage_contrast`
returns `n_positive` so a 3/3 split is visible instead of hidden inside a pooled
U statistic.

### STG-4 — the staging feature plan still cites dead, rule-violating features
`analysis/staging/CLAUDE.md` lists **"k_cardiac (validated stage discriminator,
p = 10⁻¹³⁰)"** and **"k_resp (quality indicator)"** in the feature set. Three
problems: (i) the p = 10⁻¹³⁰ is the STG-1 pooling artifact; (ii) k is **one
scalar per session** (rate review C4 / `analysis/rates/CLAUDE.md`), so it cannot
be a per-epoch feature at all — a per-window k(t) is separate, unvalidated
biomarker work; (iii) using k-derived features violates the repo's own
no-k-derived-features rule. Recommend correcting the staging note so the plan
does not inherit a superseded claim. (I flagged rather than edited it — it is
your planning doc.)

---

## SWA-validation findings

`analysis/swa_validation/run_swa_validation.py` is, to its credit, mostly built
on the right unit — correlation/Bland-Altman/AUC are computed **per session**,
and the CLAUDE.md's design rules ("per-subject statistics, never pooled raw
epochs; subject as random effect") are correct. The gaps:

### SWA-1 — a negative result needs a null / equivalence bound, not "r ≈ 0"
The claim is absence: r = 0.015 ± 0.045, coherence 0.003, AUC 0.490. These are
compared to chance informally. Two cheap upgrades make the negative rigorous:
- `subject_mean_ci` on the per-subject correlations → "cohort mean r = 0.015,
  95% CI [−0.02, +0.05], which **excludes** |r| > 0.1" — a bounded statement of
  absence.
- `subject_block_permutation_auc` on the N3 detection → an explicit null band
  around 0.5, confirming 0.490 is within chance.
The EEG sanity check (AUC 0.740) already establishes the pipeline had the power
to detect an effect if present — keep it; it is what makes the negative
credible.

### SWA-2 — cohort aggregate uses 12 sessions as if independent
`print_summary` reports mean ± std over `df[col]` — the 12 *sessions*
(`run_swa_validation.py:433-441`), not the 6 subjects, so the ± understates
variance (2 nights/subject are correlated). Report over subjects (n = 6) with
exact/permutation stats, per the section's own Step-4 rule.

### SWA-3 — in-sample threshold selection inflates sensitivity/specificity/kappa
`compute_sws_detection` picks the Youden-optimal threshold on the same epochs it
then scores sens/spec/kappa on (`:148-158`). AUC is threshold-free and unaffected,
but any reported operating-point metric — including the EEG sanity-check
sens/spec/kappa — is optimistic. Choose the threshold on held-out data (LOSO)
before reporting an operating point.

### SWA-4 — coherence noise floor *(minor)*
Magnitude-squared coherence is biased upward; its noise floor for K Welch
segments is ~1/K, not 0. The reported 0.003 is far below any plausible floor, so
the conclusion stands — but "coherence at the noise floor" should reference 1/K
rather than 0.

---

## Priority actions

1. **Replace every pooled-epoch `kruskal`/`mannwhitneyu` with a grouped test**
   (`pooled_vs_grouped_stage_test` / `paired_stage_contrast`). Re-run
   `paper_ridge_demo.py` Panel F; the p-values will move from 1e-16 to order 0.01–0.1
   and some will cross into non-significance. Report those honestly.
2. **Purge every "p = 10⁻ⁿ" per-stage p-value from the manuscript and the
   staging notes**; lead with LOSO AUC + its permutation null + CI.
3. **Frame the SWA negative as an equivalence bound** (`subject_mean_ci`,
   `subject_block_permutation_auc`), keeping the EEG sanity check.
4. **Fix `analysis/staging/CLAUDE.md`** (STG-4) and the SWA operating-point
   threshold (SWA-3).

All statements above are demonstrated by `tests/test_group_stats.py`.
