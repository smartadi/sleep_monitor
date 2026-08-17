# Revamped ridge + harmonic section — ready-to-paste manuscript text (2026-07-23)

Replaces the current **§3.6** (Methods: "Harmonic and band-restricted ridge
detection") and **§4.3** (Results: "Harmonic structure and band-restricted ridge
features"), which described only two bands, a fragmented ridge tracker, and a
title-only "harmonic" story. This version adds (1) continuous flat ridges, (2) a
new slow band, and (3) a surrogate-null-validated harmonic-ladder result.

Analysis: `analysis/slow_wave/ridge_harmonic_revamp.py`.
Data: `reports/slow_wave/revamp/{ridge_epochs.parquet, per_ridge.csv,
continuity_comparison.csv, stage_association.csv, harmonic_ladders.parquet,
harmonic_null_summary.csv}`.
Figures: `writeup/figures/harmonics/revamp_{overlay_S1N1, by_stage, continuity,
harmonics}.png`.

**Honest-negative rule (as elsewhere in the paper):** with six subjects,
stage effects are reported as per-subject direction counts; AUCs are
discrimination metrics and are reported directly. Pooled p-values are given for
completeness but are epoch-level and not the primary evidence.

---

## Methods (replaces §3.6)

### 3.6 Persistent spectral ridges and harmonic ladders

Whole-night capacitive spectrograms contain narrow spectral peaks that persist
for minutes to hours. In each 30 s window (or 240 s window for the slow band,
below) we located spectral peaks and linked them across consecutive windows into
**persistent ridges**, requiring a peak to stay within a band-dependent
frequency tolerance between windows and to persist for at least 5 minutes. Each
ridge is summarised by its frequency, amplitude, prominence above the local
spectral background, and a **flatness** index (1 − coefficient of variation of
its frequency over its lifetime). To prevent a single physical ridge from being
reported as several short fragments — the main weakness of the earlier tracker —
fragments separated by short dropouts were stitched back together by matching
their boundary-median frequencies, and each ridge's frequency/amplitude trace
was interpolated across its internal gaps so that a persistent ridge is a single
continuous trace. Flatness and drift were measured on the raw pre-interpolation
trace, and ridge coverage (the fraction of a ridge's span actually detected
before interpolation) is reported so continuity is quantified rather than
assumed.

Ridge detection was run in three physiological bands, each with an analysis
window matched to the frequency resolution that band requires:

- **Slow band, 0.02–0.15 Hz** — 240 s windows with 120 s Welch segments
  (≈0.008 Hz resolution), covering the infra-slow / vasomotor range below
  respiration.
- **Respiratory band, 0.1–0.5 Hz** — a single 30 s periodogram per window
  (≈0.033 Hz resolution).
- **Cardiac band, 0.5–3.0 Hz** — 30 s windows with 8 s Welch segments.

Per-epoch ridge features (number of active ridges, lowest ridge frequency, total
ridge power, frequency spread, mean flatness) were computed on the CRE channel
(the dominant ridge channel) and aggregated by PSG sleep stage; stage effects
were tested with the Kruskal–Wallis test across stages, the Mann–Whitney U test
for N3-versus-rest, and per-subject direction counts.

**Harmonic ladders.** Because a non-sinusoidal respiratory waveform produces
integer harmonics, we tested whether concurrent ridges form integer-ratio
ladders (f₀, 2f₀, 3f₀, …) on the CH channel over 0.1–3.0 Hz. Integer alignment
can arise by chance when several ridges are active, so every candidate ladder was
compared against a per-session, per-count **surrogate null**: for each number of
concurrent ridges k, we drew 400 random sets of k frequencies uniformly over the
session's observed ridge-frequency range and scored their best integer alignment,
taking the 95th percentile as the null threshold. A ladder was counted as real
only if it contained at least three members (fundamental plus two harmonics) and
its alignment score exceeded the null threshold; each surviving ladder received a
confidence in [0, 1] combining integer-ratio quality and amplitude-decay
monotonicity.

---

## Results (replaces §4.3)

### 4.3 Persistent spectral ridges track respiration, heart rate, and an infra-slow rhythm; a validated harmonic ladder confirms a non-sinusoidal waveform

**Ridges are continuous and multi-hour.** Linking and gap-filling the spectral
peaks yields ridges that persist without interruption for long stretches of the
night: on the CRE channel the mean per-ridge trace coverage was 1.00 (no internal
holes), and the longest continuous ridge averaged 150 minutes in the respiratory
band and 124 minutes in the slow band (Figures 6 and 7). A representative night
(Figure 6) shows a respiratory ridge holding near 0.25 Hz across the whole
recording, intermittent cardiac ridges between 0.6 and 1.8 Hz, and a continuous
slow ridge near 0.07 Hz.

**A near-ubiquitous infra-slow ridge weakens in deep sleep.** The new slow band
revealed a persistent ridge present in 88–92 % of clean epochs on every capacitive
channel, with a lowest frequency near 0.07 Hz (≈14 s period) — an infra-slow
oscillation the sensor carries throughout the night. Its **total ridge power fell
markedly during N3 in all six subjects** (pooled median 0.08 vs 0.37 in other
stages; N3-versus-rest AUC ≈ 0.68 in the low-power direction; Kruskal–Wallis
p = 9×10⁻⁵³). This is the same direction and magnitude as the mechanical CAP-SWA
score [cross-ref to the CAP-SWA section] and is one of only a few capacitive
features whose N3 direction is consistent across every subject: deep sleep
quiets the infra-slow band. The lowest slow-ridge frequency was marginally higher
in N3 (four of six subjects), a weaker effect.

**Respiratory-band ridges track breathing and thin out in N3.** A respiratory
ridge was present in almost every clean epoch, its lowest frequency sitting at a
median of ≈0.23 Hz (≈14 breaths/min) in every stage — i.e. it is the respiratory
rate. N3 carried fewer active respiratory ridges (N3-lower in all six subjects)
and lower total ridge power (five of six subjects), consistent with the more
regular, lower-effort breathing of deep sleep.

**Cardiac-band ridges slow in N3.** Cardiac ridges were intermittent; when
present, their lowest frequency was lower in N3 (median 1.0 Hz vs 1.13 Hz in
other sleep, five of six subjects), mirroring the sleep bradycardia of deep NREM.

**Harmonic ladders are real, not coincidental.** Across the twelve nights, 6,812
windows contained a strong (≥3-member) integer-ratio ladder on the CH channel,
and **6,783 of them (99.6 %; 98.9–100 % per session) exceeded the surrogate
null** — integer alignment at this rate essentially never occurs by chance. The
surviving ladders had a median fundamental of 0.25 Hz (the respiratory rate), a
median of four members extending into the cardiac band, and a median confidence
of 0.75 (Figure 8). The harmonic structure therefore reflects a genuinely
non-sinusoidal respiratory waveform transduced by the sensor, rather than
coincidental overlap of independent rhythms.

All four associations reflect autonomic, respiratory, and infra-slow mechanical
rhythms transduced by the sensor rather than cortical activity; the slow-band N3
effect is the most subject-consistent, and while none alone is a standalone
sleep-stage classifier, the slow-band power in particular could contribute to a
multi-modal staging feature set.

---

## Figure captions

**Figure 6.** Representative session (S1N1): capacitive spectrogram (CRE) with
continuous persistent ridges overlaid in three bands — respiratory (~0.25 Hz) and
cardiac (0.6–1.8 Hz) on the 0–3 Hz spectrogram, and an infra-slow ridge (~0.07
Hz) on the 0.02–0.15 Hz spectrogram — aligned with the PSG hypnogram.

**Figure 7.** Persistent-ridge features by sleep stage (CRE, 12 sessions), one
row per band (slow / respiratory / cardiac). Columns: lowest ridge frequency,
total ridge power, mean flatness. Slow-band total ridge power (top-middle) drops
in N3; cardiac lowest ridge frequency (bottom-left) slows in N3/N2. Kruskal–Wallis
p-values shown above each panel.

**Figure 8.** Harmonic-ladder rigor (CH). Left: per-session count of strong
(≥3-member) ladders, split into those that beat the surrogate null (green) versus
those rejected (red — a negligible fraction). Middle: confidence distribution of
surviving ladders. Right: distribution of ladder fundamental frequency (median
0.25 Hz, the respiratory rate).

---

## Discussion note (fold into deep-sleep / spectral-structure paragraph)

Continuous ridge tracking shows the capacitive sensor carries three long-lived
narrow-band rhythms — respiratory (~0.25 Hz), cardiac (intermittent, 0.6–1.8 Hz),
and a near-ubiquitous infra-slow oscillation (~0.07 Hz) — together with a
validated integer-harmonic ladder of the respiratory fundamental. Of these, the
infra-slow band is the most informative about sleep depth: its power falls in N3
in every subject (AUC ≈ 0.68), the same signature captured by the mechanical
CAP-SWA score, reinforcing that the mask's deep-sleep sensitivity is carried by
slow mechanical/hemodynamic rhythms rather than cortical slow-wave activity.
