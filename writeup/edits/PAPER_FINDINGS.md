# Decided paper findings — k biomarker (for §3.2 / rate-detection results)

These three k-factor findings are confirmed for inclusion in the manuscript.
Analysis: `analysis/rates/k_vs_age.py`; figure `analysis/rates/outputs/fig_k_vs_age.png`;
draft prose `writeup/edits/k_vs_age_section_draft.md`.

**Finding 1 — k is a reproducible subject-level trait.**
Night-to-night absolute change in k is small (median |Δk| = 0.013 respiratory, 0.15
cardiac), so each subject carries a characteristic k rather than a night-specific one.
This is the precondition for interpreting k physiologically at all.

**Finding 2 — Cardiac k is age-invariant (~1.95), consistent with fixed waveform
morphology.** No association with age (Spearman ρ = +0.37, p = 0.47; ρ = −0.10 excluding
the S6 coupling outlier, k = 1.14). Supports the paper's interpretation that cardiac k
arises from the fixed biphasic capacitive pulse (systolic peak + dicrotic notch), not an
age-modulated physiological variable.

**Finding 3 — Respiratory k shows a suggestive decline with age (EXPLORATORY).**
Respiratory k (near unity, 0.91–1.04) falls with age: youngest subject k ≈ 1.04, oldest
k ≈ 0.91 (Spearman ρ = −0.83, uncorrected p = 0.042; Pearson r = −0.95). Reported as
exploratory only — 1 of 4 correlations tested, does not survive Bonferroni (p ≈ 0.17),
narrow dynamic range, n = 6. Hypothesis: age-related chest-wall compliance / respiratory-
displacement morphology shifts the capacitive-to-breath ratio. Motivates a larger cohort.

Neither band related to PSQI (both p > 0.19).

**Finding 4 — k's reproducibility is age-independent, and no other CAP feature tracks
age (bounding null).** Night-to-night k variability shows no age relationship (ρ = −0.03,
p = 0.96 both bands). A scan of 8 CAP-derived per-session features (band fractions, in-band
SNR, spectral peak freqs, DC drift, accelerometer activity) found 0/10 tests significant at
n = 6 (strongest: cardiac peak freq ρ = +0.75 p = 0.08, but band-edge-pinned and
unreliable). This bounds the age story: respiratory k is the *only* capacitive quantity
with even a suggestive age association, reinforcing that Finding 3 is exploratory. See
`age_features_section_draft.md`.
