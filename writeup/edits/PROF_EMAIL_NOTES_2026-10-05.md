# Notes for the email to the professor (2026-10-05)

Everything here is a proposal. Where results live (main text or supplement) is the
professor's decision; new material has been placed in supplementary V4
(`writeup/review/final/CAP_sleep_mask_manuscript supplementary V4.docx`) so it can be
moved up if wanted.

---

## 1. Point to raise: SEC variance tracks sleep depth

The clearest stage-related result in the dataset, and V11 does not state it.

- The quietest 10% of epochs fall in N3 about twice as often as chance on all three
  channels: CH on 10/12 nights, CLE on 9/12, CRE on 11/12 (Supp. S5.1, Fig. S9).
- The noisiest 10% fall in wakefulness and almost never in N3.
- Band amplitude falls from wakefulness to N3 and returns in REM in 6/6 participants.
- The unworn mask shows that this is the participant, not the sensor. Respiratory-band
  amplitude during sleep is 1.8–9.3× the unworn level, lowest in N3 (Supp. S5.2).

Question for the professor: should this replace or lead the Discussion's opening
paragraph on SWS-related physiology? That paragraph currently rests on ridge counts and
on an association that does not hold (item 4).

## 2. New in the supplement, candidates to move into the main text

Section S5 of supplementary V4 now has five parts (Figs. S9–S15). Each states its
method plainly before its result.

| part | result | strength |
|---|---|---|
| S5.1 variance tails | low variance → N3, high → wake, on all three channels | strong, 9–11/12 nights |
| S5.2 movement-corrected slow trends + trend velocity | head position explains ~40% of the slow signal; the rest drifts at 4–157 fF/h | descriptive |
| S5.3 velocity at REM onset | rises in 5/5 participants, but only 1/5 beats its own null and it depends on how onset is defined | weak, an observation |
| S5.4 CH at N3 transitions | falls entering N3 on 10/12 nights, rises leaving on 9/11; opposite directions on 10/11 nights, never seen in 400 shuffles | consistent, small (1–2 fF) |
| S5.5 0.01–0.03 Hz band | the bright low-frequency patches are head movements; after correction, what is left is enriched in REM (7/7 nights) and absent from N3 (10/12) | clear |

## 3. §3.3 low-frequency ridges: what changed and why

- **The filter (now explained in Supp. S5.5).** The Fig. 5 bottom panel was computed
  from the signal prepared for rate estimation. That signal is band-passed at 0.05 Hz
  and has the accelerometer subtracted from it.
  - The filter edge shows up as a constant bright band just above 0.05 Hz on every
    night. That band is the "energy concentrated near 0.05–0.08 Hz" reported in §3.3.
  - Below 0.1 Hz the accelerometer adds head orientation and an instrumental tone at
    0.145 Hz.
  - The "67 ridges, median 6 per night" came from our own detector run on that signal,
    so it inherits the same problem.
- **What we see when the band is computed properly.** We started from the channel
  itself and used long windows. The visible features are short patches of 5–15 minutes
  below 0.03 Hz, not lasting ridges.
  - On the raw channel these patches are head movements: a step in level puts power at
    the lowest frequencies.
  - After the movement steps are removed, about 190 patches remain across the cohort,
    and they follow sleep state as described in item 2.
  - No persistent narrow-band oscillation at a common frequency is present.
- **Proposed main-text change (for the professor to decide).**
  - Replace the "Recurrent Low-Frequency Ridges" paragraph with two sentences pointing
    to S5.5.
  - Change "ridges around 0.1 Hz" in ¶24 to "low-frequency fluctuations".
  - Drop the low-band row from the Fig. 5 caption, or redraw Fig. 5 without that panel.
  - Soften the Conclusion's "recurrent low-frequency oscillations".

## 4. §3.8 and the SWS claims, explained

§3.8, Fig. 10 and Fig. 11 were made in a co-author's Excel workbook
(`writeup/review/Overnight_sleep_subject_list_V3.xlsx`), not in code. Rebuilding them
from the underlying data (paper code, stage s09) shows four separate problems.

1. **The reproducibility R² values were read off a graph by eye.** The workbook marks
   them "Estimated from graph".
   - The exact values are 0.64 / 0.71 / **0.86**. The paper prints 0.63 / 0.71 /
     **0.68**.
   - Panel (a) is labelled "absolute amplitude" but is actually the percent of the night
     above 10 fF².
   - With six participants, the first of the three is p = 0.06, so it is not
     "significant" as ¶230 says.
2. **Several correlations have the wrong sign.** The data give:
   - impulse frequency vs spontaneous arousals: −0.48, not +0.48
   - impulse frequency vs PSQI: −0.57, not +0.56
   - median variance vs PSQI: −0.55, not +0.48
   - mean SEC area vs age: −0.63, not +0.63

   The co-author left a note in the text, "(Check the negative signs.)", and it was
   right. R² = 0.10 for the total arousal index cannot be reproduced (0.005), and
   "Fig. Sxx" is still a placeholder.
3. **R = 0.68 (oscillation duration vs SEC area) came from mis-paired rows.** The
   duration column is ordered night 1 then night 2, while the area column is ordered by
   session. Pairing the same nights gives R = 0.48.
4. **"SWS %" in the workbook is not PSG slow-wave sleep.** It is oscillation time
   measured by hand. With PSG N3 computed from the hypnograms, the association with mean
   SEC area is R = −0.17 (12 nights), or −0.44 over two-night means.
   - The Discussion (¶248) says "mean SEC area showed a moderate association with
     PSG-measured SWS duration".
   - The Conclusion (¶259) says "the association between mean SEC area and PSG-defined
     SWS duration".
   - Neither claim is supported.

**What survives:** impulse frequency vs respiratory arousals (+0.51), impulse frequency
vs PSQI (−0.57), the percent of night above 10 fF² vs PSQI (−0.57), and the two-night
oscillation duration vs PSQI (−0.68, n = 6, hand-measured). All of these are exploratory
with n = 6–12.

**Options for the professor:**
- (a) Keep §3.8 with corrected values and signs, a single sentence on the n, and the
  SWS claims removed from the Discussion and Conclusion.
- (b) Move §3.8 to the supplement.
- (c) Cut it.

Corrected Fig. 10 and Fig. 11 are already generated by the paper code
(`paper/outputs/figures/s09_reproducibility/`).

## 5. Rate paragraph (¶253–254), rewritten from the supplementary rate section

> **Update (later 2026-10-05):** the supplement now uses the ECG cardiac reference (see
> `EMAIL_TO_PROF_2026-10-05.md` item 3). With it the cardiac figures in the paragraph below
> become: nightly error 1.19 beats/min, epoch 3.09, k 1.97–2.00 (1.84–2.32 on 11 of 12
> nights), and against the no-sensor baseline 3.89 (other night) and 2.44 (others) vs
> 4.04, i.e. heart rate also beats the baseline. The breathing numbers are unchanged.

Proposed replacement for ¶253:

> With one calibration factor k per recording (Supplementary Fig. S4), the nightly
> mean rate differed from PSG by a median of 0.24 breaths/min and 1.56 beats/min;
> epoch by epoch the differences were 1.79 breaths/min and 3.41 beats/min, and the
> estimate did not follow changes within the night (Supplementary Fig. S7). The
> night-average breathing rate was also better than a no-sensor baseline that predicts
> the cohort median (1.20 breaths/min) when k came from the same participant's other
> night (0.57) or from the other participants (0.94); the night-average heart rate was
> not (3.77 and 3.19 against 2.76 beats/min). The cardiac k was close to 2 on every
> channel (1.93–1.96; Supplementary Fig. S5): the detector counts about two
> deflections per heartbeat, and counting detected peaks per ECG R-peak gives a median
> of 2.02, consistent with a two-peaked pulse waveform. The respiratory k was 1.04–1.18.

Changes to ¶254: the respiratory range is **14.2–17.0** breaths/min (14.4–16.8 came
from an earlier version of the pipeline), and "(Table 3)" should be removed because V11
has no Table 3. The sentence "These findings demonstrate calibrated agreement with PSG"
is dropped, since a per-night calibration fitted on the night it is tested on cannot
show agreement.

**Open decision:** these cardiac numbers use the PSG pulse-oximeter (Pleth) reference,
as the supplement does. The legacy code silently used Pleth on every night even though
the paper says ECG. With ECG the nightly cardiac error is 1.19 beats/min and the
no-sensor baseline is 4.04. Which reference should the paper use?

## 6. Yellow highlights in V11

All 16 are covered in `writeup/edits/V11_HIGHLIGHTS.md`: where each passage came from,
whether it was fixed before, and a proposed replacement.

- Groups: 1 already fixed, 1 fixed earlier but lost in V11, 7 never addressed, and 4
  that need a decision.
- The most important are ¶237 (signs, see item 4) and ¶157/¶158 (low-band method and
  the N3 ridge contrast, see item 3).
- The build that produced V11 also erased the professor's highlights in ¶176. They are
  intact in V10.
