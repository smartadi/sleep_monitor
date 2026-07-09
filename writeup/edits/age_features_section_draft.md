# Extended age analysis — k variability & CAP-feature scan (ready-to-paste)

Analysis: `analysis/rates/age_features.py`
Outputs: `analysis/rates/outputs/{fig_age_features.png, age_features_per_subject.csv, age_features_stats.csv}`

**Purpose:** bound the k-vs-age result (§ k_vs_age_section_draft.md) by asking (a) whether
k's *reproducibility* depends on age, and (b) whether any other CAP-derived feature tracks
age. Both are honest nulls that contextualize the single exploratory respiratory-k finding.

## Results text (append to the k / rate-accuracy results)

**Reproducibility of k is age-independent.** The night-to-night change in k did not vary
with age for either band (Spearman ρ = −0.03, p = 0.96 for both respiratory and cardiac),
indicating that k is an equally stable subject-level quantity across the 25–66-year range,
not a factor whose reliability degrades in older or younger participants.

**No other capacitive feature tracked age.** As an exploratory scan, we tested eight
per-session CAP-derived features—respiratory and cardiac band-power fractions, in-band
SNR, spectral peak frequency in each band, differential-channel DC drift, and
accelerometer activity—against age at the subject level. None reached nominal significance
(all p > 0.08; 0 of 10 features at p < 0.05, versus 0.5 expected by chance). The strongest
tendency was a rise in cardiac-band spectral peak frequency with age (ρ = +0.75, p = 0.08),
but this feature is unreliable here because the cardiac peak was pinned at the 0.5-Hz band
edge in half of the recordings, so we do not interpret it. Thus, within this cohort, the
respiratory calibration factor was the only capacitive quantity showing even a suggestive
age association, and the broad null across features underscores that feature-level age
effects cannot be established at n = 6 and require a larger cohort.

## Consolidated framing (recommended for Discussion)

Combine with the main k-vs-age result into one honest paragraph: k is a reproducible
subject trait whose *stability* is age-independent; cardiac k is age-invariant (fixed
waveform morphology); respiratory k shows a lone, exploratory decline with age that a
broader feature scan does not corroborate for any other CAP feature — collectively a
hypothesis-generating signal that motivates a larger study, not a claim.

## Numbers

k night-to-night variability vs age: ρ = −0.03, p = 0.96 (both bands).
Feature-vs-age scan (n = 6, 10 features): 0 significant at p < 0.05 uncorrected;
strongest card_peak_hz ρ = +0.75 p = 0.08 (unreliable, band-edge pinned);
resp_peak_hz ρ = −0.39, card_snr ρ = −0.37, dc_drift ρ = −0.37, others |ρ| < 0.3.
