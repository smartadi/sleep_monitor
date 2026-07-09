# k-factor vs. age — ready-to-paste text (fills the §4.1 teased heading)

Analysis: `analysis/rates/k_vs_age.py`
Outputs: `analysis/rates/outputs/{fig_k_vs_age.png, k_vs_age_per_subject.csv, k_vs_age_stats.csv}`
Source k values: `reports/rates/mask/per_session_summary.csv`; demographics from Table 1.

**Bottom line:** k is a reproducible subject-level quantity. Cardiac k is age-invariant
(supports the fixed-waveform-morphology interpretation already in the paper); respiratory
k shows a suggestive decline with age that is exploratory at n = 6. This replaces the
current outline stub "Accurate mean respiratory and cardiac rates (comparison between
K-factor and ages)".

---

## Results — add to §3.2 (mean rate accuracy) or as a short §3.2 subsection

**Calibration factor as a subject-level quantity.** Because k relates the capacitive
peak-count estimate to the true rate through the sensor's waveform morphology, we asked
whether it behaves as a stable per-subject property and whether it varies with age. Both
factors were highly reproducible across a subject's two nights: the night-to-night
absolute change in k had a median of 0.013 (respiratory) and 0.15 (cardiac), so k is
effectively a subject-level constant rather than a night-specific artifact.

Cardiac k was consistent across subjects (median 1.95) and showed no association with age
(Spearman ρ = +0.37, p = 0.47; ρ = −0.10 after excluding the S6 coupling outlier, whose
anomalous k = 1.14 is discussed in §3.2). This age-invariance is consistent with cardiac
k arising from the fixed biphasic structure of the capacitive pulse (systolic peak plus
dicrotic notch) rather than from an age-modulated physiological variable.

Respiratory k (near unity, range 0.91–1.04) declined with age across the six subjects
(Spearman ρ = −0.83, uncorrected p = 0.042; Pearson r = −0.95), the youngest subject
showing the largest overcount (k ≈ 1.04) and the oldest the largest undercount (k ≈ 0.91)
(Figure X). We report this as an exploratory observation only: with six subjects and four
correlations tested (each factor against age and PSQI), it does not survive
multiple-comparison correction (Bonferroni p ≈ 0.17), and the dynamic range of
respiratory k is small. It is nonetheless hypothesis-generating — age-related changes in
chest-wall compliance and respiratory-displacement morphology could plausibly shift the
capacitive-to-true breath ratio — and motivates a larger cohort. Neither respiratory nor
cardiac k was associated with PSQI (both p > 0.19).

## Figure caption

**Figure X. Calibration factor k versus subject age.** Per-subject respiratory (left) and
cardiac (right) k (mean of two nights; error bars span the two nights). Points colored by
sex; dashed line is the least-squares fit. Respiratory k declines with age (ρ = −0.83,
exploratory, n = 6); cardiac k is age-invariant apart from the S6 coupling outlier
(bottom-left).

## Numbers (for a table or inline)

| Subject | Age | Sex | PSQI | Resp k | Cardiac k |
|---------|-----|-----|------|--------|-----------|
| S6 | 25 | M | 6 | 1.044 | 1.144* |
| S3 | 37 | M | 9 | 1.025 | 1.965 |
| S4 | 54 | M | 8 | 0.923 | 1.951 |
| S5 | 55 | F | 6 | 0.973 | 1.969 |
| S1 | 61 | F | 9 | 0.932 | 2.151 |
| S2 | 66 | M | 4 | 0.908 | 1.730 |

\* S6 cardiac k is a coupling outlier (see §3.2); the ~2:1 cardiac ratio holds for the
other five subjects.

Correlations (unit = subject, n = 6):
- Resp k vs age: Spearman ρ = −0.83, p = 0.042 (uncorrected); PSQI: ρ = +0.27, p = 0.61.
- Cardiac k vs age: ρ = +0.37, p = 0.47 (ρ = −0.10 without S6); PSQI: ρ = +0.62, p = 0.19.
