**Subject:** Supplementary V4: new slow-trend section, imbalance metric retired, cardiac reference corrected

Dear Professor,

I have updated the supplementary material (V4, attached). There are three things I would
like your advice on.

**1. Where should the new results go?**

I added a new supplementary section, S5, on how the slow part of the SEC signal behaves
across sleep. I put it in the supplement for now so that you can decide what, if
anything, should move into the main text:

- **Signal variance tracks sleep depth (S5.1, Fig. S9).** The quietest 10% of each night
  fall in N3 about twice as often as chance on all three channels (CH 10/12 nights, CLE
  9/12, CRE 11/12). The most variable epochs fall in wakefulness and almost never in N3.
  This is the clearest stage-related result we have, and it is not in the current
  manuscript.
- **Slow trends with head movements removed (S5.2, Figs. S10–S12).** Head position
  explains about 40% of the slow signal. The breathing-band amplitude during sleep is
  2–9 times that of the mask recorded unworn, so the stage differences come from the
  person, not the sensor.
- **Changes at stage transitions (S5.4, Fig. S14).**
  - CH falls when sleep gets deeper: N1→N2 on 8 of 10 nights, N2→N3 on 10 of 12.
  - CH rises when sleep gets lighter: N2→N1 on 7 of 7 nights, N3→N2 on 9 of 11.
  - CLE and CRE move the opposite way on entering N3.
- **The lowest frequencies, 0.01–0.03 Hz (S5.5, Fig. S15).**
  - The bright patches visible in the spectrogram turned out to be head movements.
  - After the movements are removed, what is left is more common in REM and rare in N3.
- **REM onset (S5.3, Fig. S13).** There is a small rise in the trend velocity at REM
  onset, as the literature on intracranial blood volume would predict. It is weak, so I
  report it only as an observation.

Would you like any of these in the main text? The variance result (S5.1) would fit the
opening of the Discussion, in my view.

**2. Why the capacitance imbalance metric has been removed**

The imbalance measure (the integrated magnitude of the CLE−CRE difference, and the
supplementary figures built on it) has been taken out. We had two reasons:

- **It was a poor metric.** It summed the size of the left–right difference over the
  night. That number is dominated by mask fit and posture: it varied 46-fold between
  nights and up to 6-fold between one person's two nights. Because the trace is
  referenced to its own mean, its sign is close to zero by construction.
- **It was filtered improperly.** The slow signal was being smoothed and motion-masked in
  a way that left head-movement steps in it and let the smoothing look ahead in time.
  Most of what looked like slow "flow" was the level jumping each time the head moved.

What the imbalance was meant to convey is the flow-like, slowly drifting behaviour of
the left–right signal. We now show that directly:

- First the CLE−CRE difference (and CH) with the head-movement steps removed.
- Then its "velocity": how fast the level is drifting, measured as the slope over the
  previous 30 minutes, in fF per hour (S5.2, Figs. S10–S12).

These show the drift without the posture jumps, and they need no summary number that
mixes the two. Please advise whether this is acceptable. If so, the §3.1 sentences that
cite the old figures (S4, S5a–c) need to be rewritten.

**3. The cardiac reference is now the ECG**

While checking the rate numbers I found that the PSG heart-rate reference had silently
been taken from the pulse oximeter on every night instead of the ECG, as the methods
state. Checked against the ECG, the pulse oximeter reads 36% and 29% too high on two
nights. The supplement now uses the ECG wherever it is usable (10 of 12 nights).

This changes the cardiac numbers:

- The calibration factor is now close to 2 on 11 of 12 nights. The exception is the one
  night without a valid reference.
- The nightly heart-rate error is 1.19 rather than 1.56 beats/min.
- Both breathing and heart-rate night averages now beat a no-sensor baseline, including
  when the calibration comes from another night or other people.

The main text (¶253 and ¶259) still quotes the old cardiac numbers. I can send the
corrected sentences.

Best regards,
Aditya
