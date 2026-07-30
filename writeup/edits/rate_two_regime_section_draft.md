# Rate detection — simplified main text + supplementary statistics

Replaces the current §4.2 ("Mean rate detection accuracy"), Table 3, Figure 3, and Figure 5.
Rationale and source numbers: `RATE_SECTION_REVIEW.md`, this file's "Provenance" section at the end.

Structural change: the main text reports **both bands in both regimes** in one short section
with one table and two figures. Everything per-session, per-method, per-channel, and per-stage
moves to a new Supplementary section, each item with its own writeup.

Main figures after this edit: F1 spectrograms, F2 SNR, **F3 Bland–Altman**, **F4 k and age**,
F5 ridge spectrogram (was F6), F6 ridge-by-stage (was F7), F7 spindles (was F8).
Supplementary: Table S1, Figures S1–S5.

---

## 4.2 Rate detection

The sensor was evaluated against PSG in two regimes that differ in what is being measured and
in how it is validated. The **night-level regime** asks whether the mask recovers a subject's
average respiratory and cardiac rate over a recording — the quantity relevant to overnight
screening and night-to-night trending. The **epoch-level regime** asks whether it follows the
rate as it varies within a night, on the same 60-s windows (30-s step) used throughout. The two
are reported together because they are served by different estimators and reach different
conclusions, and because the distinction determines what the device can be used for.

For the night-level regime we used the estimator that minimised error in each band: the spectral
peak (Welch PSD) for respiration and loose prominence-based peak counting for the cardiac band,
both on the CLE−CRE differential, rescaled by a per-session factor k (§4.2.1) and smoothed with a
causal three-epoch median filter. For the epoch-level regime we used a responsive detector
combining loose peak counting with Hilbert instantaneous frequency across five channels, which
sacrifices some accuracy for temporal resolution. Multi-channel SQI-weighted and agreement-gated
fusion was evaluated but did not improve accuracy over the single differential channel in either
band (ΔMAE < 0.1; Figure S1), so the single-channel estimator is the operational pipeline.

**Night-level accuracy.** Respiratory rate was recovered with a per-session median MAE of
0.91 br/min (IQR 0.81–1.19, range 0.56–2.26) and a pooled MAE of 1.09 br/min (bias −0.3 br/min,
95% limits of agreement −4.7 to +4.2). Cardiac rate was recovered with a per-session median MAE
of 3.41 BPM (IQR 3.06–8.38) on the best single channel and a pooled MAE of 3.91 BPM (bias
−0.6 BPM, limits of agreement −24.1 to +22.9). The cardiac error distribution is right-skewed:
three of twelve nights exceeded 8 BPM (S2N1, S6N1, S6N2), and excluding the anomalously coupled
S6 subject the median was ≈3.0 BPM. The wide cardiac limits of agreement reflect epoch-level
spread; the validated quantity in this regime is the per-session mean. Per-session values for all
twelve nights are given in Table S1 and the agreement plots in Figure 3.

**Epoch-level accuracy.** Within a night, neither band's rate variation was recovered
(Table 3). Respiratory estimates correlated with the reference at a median within-session
r = +0.06 (Wilcoxon p = 0.68; positive in 8 of 12 nights; r = +0.02 on detrended fluctuations),
and cardiac estimates at r = −0.19 (p = 0.34; positive in 5 of 12; detrended r = −0.15). This is
not a limitation specific to the mask: two physically independent PSG respiratory sensors — nasal
airflow and the respiratory inductance belt sum — agree with each other at only r = 0.47
(IQR 0.33–0.67) on the same epochs, which bounds what any respiratory sensor can achieve at this
time resolution.

The two bands fall short of that bound for different reasons, and the ratio of estimation error
to the physiological variation available to be tracked separates them. Within-session respiratory
rate varies with a median standard deviation of only 1.14 br/min, which is comparable to the
epoch-level error of the best estimator (ratio 0.77) and smaller than that of the responsive
detector (1.04). For respiration there is very little variation to resolve relative to the noise
floor, and a near-constant estimate is consequently the more accurate one — the spectral
estimator's lower error (0.91 vs 1.33 br/min) is a direct consequence of its not tracking. The
cardiac band is different: within-session heart rate varies with a median standard deviation of
5.26 BPM, well above the epoch-level error (ratio 0.78), so the variation is in principle
resolvable and is nonetheless not recovered. The mechanism for this is discussed in §5.3.
An exhaustive comparison of estimators and channels in both regimes, including CWT-ridge and
STFT–Viterbi trackers, is given in Figure S1 and Figure S3.

**Table 3. Rate detection in both regimes.** Per-session medians across 12 nights (IQR in
brackets). Within-session r is the correlation between estimated and reference rate across
epochs within each night.

| | Respiratory | Cardiac |
|---|---|---|
| **Night-level regime** | | |
| MAE, per-session median | 0.91 br/min [0.81–1.19] | 3.41 BPM [3.06–8.38] |
| MAE, pooled | 1.09 br/min | 3.91 BPM |
| Bias (95% limits of agreement) | −0.3 (−4.7, +4.2) | −0.6 (−24.1, +22.9) |
| Calibration factor k | 0.97 [0.91–1.04] | 1.95 [0.94–2.24] |
| **Epoch-level regime** | | |
| MAE, per-session median | 1.33 br/min [1.11–1.51] | 3.65 BPM [2.97–7.42] |
| Within-session r | +0.06 (p = 0.68) | −0.19 (p = 0.34) |
| Nights with r > 0 | 8 / 12 | 5 / 12 |
| Reference rate SD within night | 1.14 br/min | 5.26 BPM |
| Error-to-variation ratio | 0.77 | 0.78 |
| Independent-sensor bound | r = 0.47 | not available |

*Figure 3. Bland–Altman agreement for respiratory (left) and cardiac (right) rate, night-level
regime. Each point is one analysis epoch; solid line is the bias, dashed lines the 95% limits of
agreement.*

---

### 4.2.1 The calibration factor k

Rate estimates in both bands are rescaled by a per-session factor k, the median ratio of the raw
estimator output to the reference rate over 50 randomly selected one-minute windows. k is not a
free parameter fitted to improve accuracy; it counts how many capacitive deflections the sensor
produces per physiological event, and it behaves as a stable property of the subject.

**k is a waveform-morphology count.** Averaging the capacitive signal triggered on ECG R-peaks
resolves the per-beat pulse shape directly, and counting capacitive peaks against ECG beats gives
1.70–2.43 peaks per heartbeat (median 2.02 on CRE, 1.92 on CLE−CRE, across the nine sessions with
usable peak detection). This brackets the fitted cardiac k of 1.95 and identifies its origin: the
capacitive cardiac pulse is biphasic, contributing a systolic peak and a dicrotic notch to each
cardiac cycle, so a peak-counting estimator reports approximately twice the true rate. The
correspondence is population-level rather than exact per subject (r = 0.50 between measured
peaks-per-beat and fitted k), and one session was excluded because cardiac peak detection failed
outright. The respiratory k of ≈0.97 reflects the simpler coupling: each breath produces one
dominant temple displacement.

**k is reproducible within a subject.** Across each subject's two nights the absolute change in k
had a median of 0.013 for respiration (maximum 0.047) and 0.151 for the cardiac band (maximum
0.406). Respiratory k is therefore effectively a subject-level constant, and cardiac k is stable
apart from the S6 coupling anomaly. Diagnostic estimates from 50 random windows agreed with
whole-night values to within 0.04.

**k, age, and what calibration a deployment would require.** Because k reflects the mechanical
coupling between physiology and sensor, we asked whether it varies with subject age (Figure 4).
Respiratory k declined across the six subjects, from 1.04 in the youngest (25 years) to 0.91 in
the oldest (66 years) — Spearman ρ = −0.83, uncorrected p = 0.042. The direction of this
relationship was stable: dropping each subject in turn left ρ between −1.00 and −0.70. Cardiac k
showed no such relationship (ρ = +0.37, p = 0.47), and unlike the respiratory case the cardiac
result was not stable under the same test — leave-one-out ρ ranged from −0.10 to +0.90 — so these
data neither establish nor exclude an age dependence for cardiac k. Neither factor was associated
with PSQI (p > 0.19), and no other capacitive signal feature we examined varied with age
(Figure S5).

We tested whether the respiratory relationship is strong enough to replace per-subject
calibration, by fitting k from age on five subjects and predicting the sixth. Leave-one-subject-out,
an age-based prior predicted respiratory k with a mean absolute error of 0.020, against 0.056 for
no calibration (k = 1.0) and 0.056 for the best constant prior (the population mean). Age
therefore carries information about respiratory k that a constant prior does not. The same test
on cardiac k gave the opposite result — an age prior (0.387) was worse than the population mean
(0.305) — consistent with cardiac k being set by a fixed pulse morphology rather than by an
age-modulated variable. Translated into rate error, per-session calibration gives 0.94 br/min and
0.95 br/min for respiration on the two best pipelines, a fixed k = 1.0 costs only 0.20–0.28
br/min more, and cardiac rate requires whole-night self-calibration or a population prior
(3.36 BPM vs 4.56 BPM; Figure S4).

Two cautions apply to the age result. With six subjects, age is perfectly confounded with subject
identity, so the association may reflect any subject characteristic that covaries with age in this
sample — chest-wall compliance and respiratory displacement morphology are the plausible
mechanisms, but these data cannot isolate them; both age extremes in this cohort were male. And
the respiratory correlation does not survive correction for the four correlations tested (each
factor against age and PSQI; Bonferroni p ≈ 0.17). We therefore report the age relationship as a
calibration result validated on held-out subjects, and as an exploratory physiological
observation motivating a larger cohort.

*Figure 4. Calibration factor k and subject age.* **(A)** Respiratory k per subject (mean of two
nights; bars span the two nights; circles male, squares female) against age, with least-squares
fit. k falls from 1.04 to 0.91 across the age range; the leave-one-out range of ρ is given above
the panel. **(B)** Cardiac k against age. Values cluster near 1.95 with no stable trend; S6 is a
coupling outlier (k = 1.14) whose inclusion or exclusion reverses the sign of ρ. **(C)**
Leave-one-subject-out prediction of respiratory k: absolute error per held-out subject using an
age prior fitted on the other five, against using no calibration (k = 1.0).

---

## Supplementary material — rate detection

**Table S1. Per-session rate accuracy, both regimes.** All twelve nights, with per-session k,
night-level MAE, epoch-level MAE, and within-session r for each band.
*(populate from `reports/rates/mask/per_session_summary.csv` and
`symmetric_tracking_battery.csv`)*

**Figure S1. Estimator and channel comparison.** MAE across six base estimators × four channels
for both bands, k-scaled, with per-session IQR.
> Writeup: The spectral estimator dominated the respiratory band on every channel and the loose
> peak-counting estimator dominated the cardiac band; no channel was reliably better than the
> CLE−CRE differential for respiration (oracle-over-channels 1.08 vs 1.09 br/min pooled), whereas
> the cardiac band did show channel-dependent headroom, motivating the multi-channel detector
> used in the epoch-level regime. Multi-channel fusion nonetheless changed per-session median MAE
> by less than 0.1 in both bands (respiratory 0.95 → 0.94 br/min; cardiac 3.42 → 3.36 BPM), which
> is why the single-channel estimator is reported as the operational pipeline.

**Figure S2. Per-sleep-stage MAE.** Night-level estimator error by PSG stage, both bands.
> Writeup: Respiratory error was highest in REM (≈2.31 br/min) and cardiac error highest in Wake
> (≈4.73 BPM), consistent with the irregular breathing of REM and the greater heart-rate
> variability of wakefulness. These comparisons pool epochs across sessions and were not tested
> for significance given the six-subject sample; they are reported as descriptive context for the
> aggregate errors in Table 3.

**Figure S3. Epoch-level regime: full estimator battery and the independent-sensor bound.**
Within-session r per night for each estimator in both bands, against a 200-iteration temporal
shuffle null, with the nasal-flow-versus-inductance-belt agreement shown as a reference band.
> Writeup: An identical battery was applied to both bands: six base estimators, multi-channel
> mean fusion, CWT-ridge tracking, and STFT–Viterbi tracking, each with and without k-calibration
> and smoothing. No configuration produced a within-session correlation distinguishable from the
> temporal-shuffle null in either band (respiratory: 4 of 12 nights above the null 95th
> percentile; cardiac: 3 of 12). The reference band shows that the two independent PSG
> respiratory sensors agree at r = 0.47 (0.27 on detrended fluctuations), so the achievable
> target in this regime is modest even for a contact sensor. The exhaustiveness of this battery
> is what supports treating the epoch-level result as a property of the signal rather than of a
> particular estimator.

**Figure S4. Calibration requirement.** Rate MAE under three calibration strategies — per-session
k, a fixed population k, and k estimated from the first ten minutes of recording — for both bands.
> Writeup: Respiration is close to calibration-free: replacing per-session k with a fixed
> population value costs 0.15–0.28 br/min depending on pipeline (0.94–1.09 → 1.10–1.23 br/min),
> and a fixed k = 1.0 is nearly as good. Cardiac rate genuinely requires calibration: a population
> prior costs 1.2 BPM (3.36 → 4.56 BPM), and a ten-minute warm-up calibration is worse than the
> population prior (10.1 BPM), because twenty epochs give a noisy k. A deployment would therefore
> use a population prior or whole-night self-calibration for the cardiac band, not a short
> calibration period. This bounds Limitation 3.

**Figure S5. Capacitive signal features versus age.** Per-subject night-to-night k
reproducibility, modal spectral peak frequency, band SNR, DC drift, and accelerometer activity
against age.
> Writeup: No capacitive feature other than respiratory k varied with age. The night-to-night
> reproducibility of k was itself age-independent in both bands (ρ = −0.03, p = 0.96), confirming
> that older subjects' k values are not merely noisier estimates; band SNR, DC drift, and
> accelerometer activity were likewise unrelated to age (|ρ| ≤ 0.37, p ≥ 0.47). The modal cardiac
> peak frequency showed the largest nominal association (ρ = +0.75, p = 0.084), but this variable
> takes only four distinct values across the six subjects and sits at or just above the 0.5 Hz
> lower edge of the cardiac analysis band for most of them; the same edge-pinning applies to the
> respiratory modal peak, which equals the 0.1 Hz band floor in five of six subjects. Neither is
> interpretable as a physiological age effect. This panel is included as a bounding null: it
> establishes that the respiratory-k finding is not one of many age correlations in these data.

---

## Provenance

| Quantity | Source |
|---|---|
| Night-level MAE, bias, limits of agreement | `reports/rates/mask/per_session_summary.csv`, `final_summary.json` |
| Epoch-level MAE, within-session r, reference SD | `reports/rates/mask/symmetric_tracking_battery.csv` |
| Independent-sensor bound (Flow vs RIPSum) | `reports/rates/mask/symmetric_tracking_ceiling.csv` |
| Estimator/channel comparison, fusion delta | `reports/rates/mask/single_combo_leaderboard.csv`, `channel_win_*.csv` |
| Calibration strategies | `analysis/rates/outputs/calibration_requirement.csv` |
| Peaks per heartbeat | `analysis/rates/outputs/peaks_per_beat.csv` |
| k vs age, leave-one-out ρ, LOSO age prior | `analysis/rates/k_age_prior.py` → `outputs/k_age_prior.csv` |
| CAP features vs age | `analysis/rates/outputs/age_features_stats.csv` |
| Figure 4 | `analysis/rates/outputs/fig_k_vs_age_3panel.png` |
| Figure S5 | `analysis/rates/outputs/fig_age_features.png` |
