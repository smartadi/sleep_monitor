# Harmonic-comb ladder episodes — ready-to-paste manuscript subsection (2026-08-17)

**Decided framing (user, 2026-08-17):** report the honest positive —
*"the SEC signal shows sustained harmonic-comb episodes in consolidated N2 that
tend to follow REM by 10–30 min."*  This is a **distinct phenomenon from the
respiratory/cardiac rate ridges** (§4.3) and is **NOT** a cortical slow-wave /
N3 signature — the episodes cluster in N2, so it does not conflict with the
paper's thesis that the SEC signal does not transduce cortical SWA spectrally
(§3.5, r = 0.015).

**Honest-negative / small-n rule (as elsewhere):** n = 6 subjects, 22 events —
report per-subject direction counts and effect sizes vs a matched null, not
p-values; state explicitly that the REM association is exploratory.

Analysis: `analysis/slow_wave/harmonic_ladder_overlay.py` (detection),
`analysis/slow_wave/ladder_stage_relationship.py` (stage/REM timing).
Data: `reports/slow_wave/revamp/{ladder_events.csv, ladder_stage_summary.csv}`.
Figures: `writeup/figures/harmonics/ladders/ladder_<S>.png` (per-session combs),
`writeup/figures/harmonics/ladder_stage_relationship.png` (REM timing).

---

## Methods addition (new subsection, pairs with §3.6)

**3.x Harmonic-comb ladder episodes.** Separately from the respiratory and
cardiac rate ridges (§3.6), we observed transient episodes in which the
capacitive spectrogram carries a *stack* of several sustained, temporally-flat
narrow bands — a harmonic comb. To detect these we worked on each raw channel
(CH, CLE, CRE) independently (averaging channels dilutes an episode strong on a
single electrode). The spectrogram (30 s windows, 15 s step, 0–3 Hz) was
background-subtracted per time column by removing a frequency-median-filtered
envelope, so a narrow band's height is expressed in dB above its local floor. An
**episode** was defined as a sustained interval carrying a rich comb — at least
three integer-harmonic rungs, each a peak ≥ 5 dB above the local floor, with at
least three consecutive rungs from the fundamental — grown to its full extent
while a weaker comb persisted. Within each episode the **rungs** were then
recovered as the actual persistent horizontal bands: per-window spectral peaks
were linked across time into flat bands and kept only if they persisted ≥ 1.5
min and were present in ≥ 60 % of their span, on a time-smoothed spectrogram.
Rungs are reported at their true frequencies (they are quasi-harmonic, not forced
to integer multiples). Episodes overlapping in time across channels were merged
into single events for the stage analysis. Because a comb of ≥ 3 consecutive
harmonics is required, the two independent rhythms the sensor always carries
(respiration + heart rate) do not by themselves constitute an episode.

To relate events to sleep architecture we computed, per event, the fraction of
the 30 min before onset and the 30 min after offset scored REM, the minutes to
the nearest REM epoch on each side, and the stage occupancy in a ±30 min window
around onset; each REM quantity was compared with a matched random-NREM null
(200 draws per session).

---

## Results addition

**3.x Harmonic-comb episodes occur in consolidated N2 and tend to follow REM.**
Across the twelve recordings we detected **22 harmonic-comb events in 9 sessions
(6 subjects)**. Each is a stack of several flat, sustained bands at
quasi-harmonic (not exactly integer-spaced) frequencies — for example, a
representative S6N1 episode carried rungs at 0.15, 0.28, 0.42, 0.68 and 0.95 Hz
(Figure X). The events are firmly a **light-to-consolidated NREM (N2)**
phenomenon: **19 of 22 (86 %) fell in N2** (1 each in N1, N3 and Wake), and
onset-aligned stage occupancy peaked at P(N2) ≈ 0.9 at event onset (Figure Y).
They are therefore not a cortical slow-wave (N3) signature.

The events showed a consistent **temporal relationship to REM: REM precedes**
them. REM occupied a mean **0.059** of the 30 min before onset — about **3.5×**
the matched random-NREM null (0.017) — but only **0.007** of the 30 min after
offset, *below* the null (0.031); onset-aligned occupancy shows REM present from
roughly 30 to 8 min before onset and essentially absent from 8 min before through
30 min after (Figure Y). The nearest REM epoch sat a median of 30 min before an
event versus 51 min after, and **5 of 6 subjects** had REM nearer before the
event than after (the sixth reversed). Consistent with a REM→N1→N2 re-entry, N1
occupancy rose in the ~10 min immediately before onset. Taken together, the comb
episodes tend to arise in consolidated N2 roughly **10–30 min after a REM
period**, and are not themselves followed by REM. Given the small sample (22
events, 6 subjects) this REM association is reported as an exploratory,
per-subject-consistent observation rather than a powered effect.

---

## Suggested figures

**Figure X.** Representative harmonic-comb episodes (S6N1, CH): background-
enhanced spectrogram with the detected flat rungs overlaid, above the stepped
hypnogram — two episodes in consolidated N2, each a stack of quasi-harmonic bands.

**Figure Y.** (A) Stage occupancy in a ±30 min window around event onset (n = 22):
N2 peaks at onset. (B) REM occupancy around onset — elevated 30–8 min before,
absent from 8 min before onward — showing that REM precedes the episodes.

---

## Discussion note

The capacitive signal carries, beyond the respiratory and cardiac rate ridges, a
distinct transient feature: sustained stacks of quasi-harmonic flat bands during
consolidated N2 that tend to follow a REM period by 10–30 min. We interpret these
as a non-sinusoidal, quasi-periodic mechanical/hemodynamic waveform that emerges
during the stable breathing of post-REM NREM, rather than as cortical slow-wave
activity — consistent with the sensor transducing mechanical and hemodynamic
pulsations, not neuronal activity (§5.3). The REM-preceding timing is a novel but
exploratory observation (n = 6) that would merit a dedicated, larger study; it is
reported here as a characterisation of what the signal contains, not as a staging
biomarker.
