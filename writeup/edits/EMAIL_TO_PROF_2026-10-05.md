**Subject:** Manuscript V12 and Supplementary V4: corrected numbers, new supplementary results, items for your decision

Dear Professor,

Please find attached manuscript V12 and supplementary V4.

V12 is V11 with every number checked against the analysis code and corrected where the code
disagreed. The changes are accepted into the text, so it reads cleanly. I have a list of
each change with its reason if you would like to see it. I did not touch §3.8, Figs. 3–4,
or your Fig. 2 caption.

**1. What changed in V12**

- **Heart-rate reference.** The PSG heart-rate reference had been taken from the pulse
  oximeter on every night instead of the ECG, as the methods state. On two nights the
  oximeter reads 36% and 29% too high. With the ECG:
  - nightly heart-rate error is 1.19 beats/min (was 1.56);
  - per-epoch error is 3.09 beats/min (was 3.41);
  - the cardiac calibration factor is close to 2 on 11 of 12 nights.

  The Discussion and Conclusion now give these numbers and cite the rate figures, which
  are now in the supplement (Figs. S4–S8).
- **Harmonic-comb events (§3.6, Figs. 8–9, new supplementary section S6).**
  - The old detector required three harmonics each 5 dB above background. It missed combs
    that are clearly visible and split single combs in two. I replaced it with a score
    that adapts to each night, tuned by inspecting all 12 nights on all three channels.
  - It finds 45 events on all 12 nights (previously 22 on 9).
  - The N2 association holds: 35 of 45 events are in N2.
  - The earlier claim that combs emerge 10–30 min after REM does not hold up. REM is closer
    before the event in only 2 of 6 participants (previously 5 of 6). §3.6 now says this.
- **Smaller corrections in §3.1–§3.5 and Methods.** Examples:
  - CH exceeded 1000 fF in five recordings, not three.
  - Spindle counts and dB values.
  - Only 9 of 344 delta-burst onsets coincide with a scored K-complex, so "most were
    K-complexes" was removed.
  - The causal-filtering paragraph in §3.5 is rewritten.
  - Supplementary and figure cross-references are renumbered.
- **Figure titles.** The K-complex figure (Fig. 12) and Figs. 8–9 no longer carry titles
  inside the image; the captions carry them.

**2. Where should the new supplementary results go?**

Supplementary section S5 covers how the slow part of the SEC signal behaves across sleep.
I kept it in the supplement so that you can decide what moves to the main text:

- **Signal variance tracks sleep depth (S5.1, Fig. S9).**
  - The quietest 10% of each night falls in N3 about twice as often as chance on all three
    channels (CH 10/12 nights, CLE 9/12, CRE 11/12).
  - The most variable epochs fall in wakefulness and almost never in N3.
  - This is our clearest stage-related result, and it is not in the manuscript.
- **Slow trends with head movements removed (S5.2, Figs. S10–S12).** Head position explains
  about 40% of the slow signal.
- **Stage transitions (S5.4, Fig. S14).**
  - CH falls as sleep deepens: N1→N2 on 8 of 10 nights, N2→N3 on 10 of 12.
  - CH rises as sleep lightens: N2→N1 on 7 of 7 nights, N3→N2 on 9 of 11.
  - CLE and CRE move the opposite way on entering N3.
- **0.01–0.03 Hz (S5.5, Fig. S15).** The bright low-frequency patches in the spectrograms
  are head movements. After removing them, what remains is more common in REM and rare
  in N3.
- **REM onset (S5.3, Fig. S13).** There is a small rise in trend velocity at REM onset. It
  is weak, so it is reported only as an observation.

In my view the variance result would fit the opening of the Discussion.

**3. The capacitance imbalance metric has been removed**

The imbalance measure (the integrated magnitude of the CLE−CRE difference) has been
removed, for two reasons:

- **It is a poor metric.** It is dominated by mask fit and posture. It varied 46-fold
  between nights and up to 6-fold between one person's two nights. Because the trace is
  referenced to its own mean, its sign is close to zero by construction.
- **It was filtered improperly.** The smoothing left head-movement steps in the signal and
  looked ahead in time. Much of the apparent slow "flow" was the level jumping each time
  the head moved.

The metric was meant to show the flow-like, slowly drifting behaviour of the left–right
signal. We now show that directly, in S5.2 (Figs. S10–S12):

- the CLE−CRE difference and CH, with head-movement steps removed;
- their velocity, meaning the slope over the previous 30 minutes in fF per hour.

Please advise whether this is acceptable. If it is, the §3.1 sentences that cite the old
figures (S4, S5a–c) need rewriting.

**4. Items left for your decision**

- **§3.8.**
  - "(Fig. 12)" in the first paragraph should be Fig. 11.
  - "Fig. Sxx" is still a placeholder.
  - Several correlation signs in the code differ from the text. For example, the
    impulse-frequency/PSQI value and the age value are negative in the code.
- **Fig. 2 caption.** It says "four male participants aged 54 years", but the figure now
  shows one recording.
- **§3.3 low-frequency ridges.** The 0.05–0.08 Hz ridge (67 ridges, "around 0.1 Hz") comes
  from the motion-filter corner, not physiology; S5.5 shows the corrected picture. "Fig. S10"
  in §3.3 now points to a different figure.
- **Discussion and Conclusion.** Mean SEC area against PSG N3 duration gives R = −0.17, which
  does not support "a moderate association with SWS duration".
- **§3.7.** The third R² (median variance) computes to 0.86, not 0.68.
- **Small text issues.**
  - Methods ¶53 ends mid-sentence ("mean values of").
  - Sections 2.5–2.6 are missing from the numbering.
  - A note, "Mention absolute value?", remains in §3.1.
  - The previous ICP work in §2.1 cites reference 22 (a near-infrared paper); I think it
    should be 23 or 24.
  - The "6.4 s" CSF latency in §3.9 has no citation.
  - The Fig. 13 participant is 24, which is outside our cohort; please confirm this is the
    separate optical test.

Best regards,
Aditya
