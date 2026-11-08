# V18 from §3.4 on: final text for the tracked copy

*2026-11-07. This replaces the edit list's [ask], [check] and instruction-style points from §3.4 to the end of the main text with the exact words that went into the Word copy.*

*Built 2026-11-07 as `OneDrive\CAP_sleep_mask_manuscript_V18_aditya_tracked.docx`. Everything from §3.4 on is a tracked change by "Aditya Deole". Your own revisions are kept unchanged, and the file has no comments or tags. The source `CAP_sleep_mask_manuscript_V18_aditya.docx` was not modified. The scripts are in `writeup/edits/v18_tracked/`. The paragraph-by-paragraph text below is what went in.*

*In this sheet C_LE, C_RE and C_H stand for the subscripted channel names, as Word shows them. The difference channel is written C_LE − C_RE, with a true minus and no parentheses (D3).*

---

## Decisions (2026-11-07)

| # | Point | Decision | What went in |
|---|---|---|---|
| Q1 | 77b | From the code (`paper/stages/s09_reproducibility.py`): the durations are hand-measured in the co-author workbook | "LFO ridge duration (the summed length of LFO episodes measured by hand, as % of recording time)" |
| Q2 | 77d | Professor's work, leave it | V18's "mean SEC area (fF·h/h)", no definition |
| Q3 | 78d, 78e | Professor's work, leave it | Both rms sentences kept word for word, with their R and p |
| Q4 | 81a | Unknown, leave it | V18's "from one 24-year-old participant wearing the sleep mask" |
| Q5 | 83b | Recommended | Ref 23 after both porcine sentences |
| Q6 | 82c | Recommended | "increase during REM sleep" |
| Q7 | 67b, 84b | Recommended | 6.4-s comparison in the Discussion only; §3.6 ¶2 is the event-count sentence |
| Q8 | 85 | Ignore the review question | "PSG was scored automatically by the PSG software from frontal EEG only" (no claim about review) |
| Q9 | 87d | No statement | No Data and code availability section |
| Q10 | G6 | Recommended | "Fig. Sn" without "Supporting information" from §3.4 on |
| Q11 | figures | Swap | Figs. 6–11 replaced as tracked image changes, shown uncropped |
| Q12 | 81b | Recommended | Muse is ref 32; old 32–34 are 33–35 in the list and in the text |

**Changes I'm making beyond the edit list**, because the list's own decisions require them. Say if you want any of them out.
- **§3.7 ¶2:** V18 says the other bands lie "near its multiples … consistent with a breathing-related signal whose waveform is not a pure sinusoid". R6 (next paragraph) says they are *not* at breathing multiples more than chance, and you chose "multi-band" so as to make no fundamental claim. The clause goes.
- **§3.7 ¶3:** "nonsinusoidal, quasi-periodic" → "quasi-periodic", for the same reason.
- **§3.7 ¶1:** with the base rate added, the N2 sentence says the events track N2 roughly in proportion to its share of sleep. 1.1 times chance doesn't support "associated with N2 rather than N3".
- **Fig. 6 title:** "Mechanical SEC responses" → "SEC responses", matching the softened §3.4 ¶3.
- **Discussion:** interpretation verbs go to the present tense (G10): "suggested" → "suggest", and similar.
- **Fig. 7 caption:** "green line" → "vertical line". The redrawn figure's onset line is dark, and the control is dotted.
- **Fig. 9 caption (added while building):** the new drawing puts the hypnogram above the spectrogram, so "bands above the hypnogram" became "the detected bands; the hypnogram above it places REM at the bottom".
- **Limitations (added while building):** "reported without p-values" was dropped, because the two rms sentences kept in §3.9 (Q3) still give p.
- **Discussion ¶1:** the "(C)rms vs PSG-measured SWA" sentence is still deleted (point 79). The code shows that this "SWA duration" is the hand-measured LFO duration, not PSG staging. Reject that one deletion in Word if the professor wants it back.

**Not covered here:** anything before §3.4, and the Supplement (Supp V8). Two things in your sections connect to this part. The IRB sentence is in both 2.2 and the new Ethics statement (87c). "Quiet, motion-free baseline" in 2.6 should become "quiet in EEG delta" (point 62).

---

## Final text, paragraph by paragraph

Each entry gives the V18 text and what replaces it. Unchanged sentences are shown where it helps to place the edit.

### 3.4 SEC response to sleep spindles

**P193 (55).** "Sleep spindles ~~were~~ **are** transient 11–16 Hz thalamocortical oscillations characteristic of N2 sleep and therefore ~~provided~~ **provide** precisely timed cortical events for evaluating the SEC response." The second sentence is unchanged.

**P194 (56), whole paragraph →**
> Low-frequency SEC power (0.1–3 Hz) increased at spindle centers on average, but the increase was carried by a minority of spindles. Per recording, 48–58% of spindles showed a rise, compared with 45–52% of random N2 time points, and on C_H the median spindle showed no rise in 5 of 12 recordings; the 5% of spindles with the largest rises accounted for all of the summed increase. The mean increase was 0.45–0.49 dB in the C_RE and C_LE and 0.54 dB in the C_H (mean of recording means; 0.59 dB pooled over all spindles, Fig. 6b). This mean increase was positive in all 12 recordings on C_H and C_LE − C_RE. In contrast, randomly selected N2 time points showed no comparable response in 11 of 12 recordings on C_LE − C_RE. These findings support an association, on average, between sleep spindles and increased SEC power that is not explained by scored arousals. After excluding spindles with an arousal within ±5 s (5.7% of N2 spindles), the increase remained 0.49 dB on C_H and 0.45 dB on C_LE − C_RE, against 0.05 and 0.08 dB at random N2 times.

**P195 (57), whole paragraph →**
> The response was confined to the low-frequency band (0.1–3 Hz). Sigma-band power (11–16 Hz), which contained the spindle oscillation, changed by only 0.02–0.03 dB across the SEC channels, whereas contact EEG sigma power increased by 3.45 dB at the same spindle centers. However, SEC power at 11–16 Hz was at the electronic noise floor (median within 0.4 dB of an unworn-mask recording on C_LE and C_RE), so no sigma-band change would be expected under either mechanism. The low-frequency SEC response is therefore consistent with spindle-associated mechanical processes, such as transient cardiovascular or cerebrospinal fluid-related displacement, although the sigma band alone cannot exclude direct detection of the electrical spindle oscillation.

**P197 Fig. 6 caption (58) →** (the "(c)" sentence only if Q11 = swap)
> Fig. 6 SEC responses associated with sleep spindles. (a) Event-triggered average of low-frequency SEC power (0.1–3 Hz), showing an average increase at spindle centers on all channels, with the largest increase in C_H (0.54 dB). (b) Distributions of spindle-associated power changes in C_H: the mean low-frequency change is 0.59 dB (pooled over spindles), but the median is 0.13 dB, so most spindles change little and a minority change a lot; sigma-band power changes by only 0.03 dB. (c) Mean and median low-frequency change per recording and channel.

The [check] (58c) resolves with no change: 0.03 dB is C_H, inside the text's 0.02–0.03 dB range across channels.

### 3.5 SEC response to delta-burst onsets

**P200 heading (59) →** "3.5 SEC response to delta-burst onsets"

**P201 (60) →**
> Delta-burst onsets provided a second class of discrete cortical events for evaluating the SEC response. In total, 344 qualifying onsets were identified, ranging from 1 to 99 per recording; four recordings contained fewer than 10 events. Two recordings with fewer than five onsets (S4N2, 4 onsets; S5N1, 1 onset) were excluded from averaging, so 339 onsets entered the event-triggered average. Most were isolated N2 slow waves because sustained N3 slow-wave activity generally lacked a discrete onset following a quiet baseline.

**P202 (61), last sentence →**
> The response was observed in all six participants for every channel–band combination, although S5 contributed only six onsets (Fig. 7, Fig. S7). Excluding S4 and S5, whose PSG alignment rests on a single synchronization anchor (uncertain by about ±1 s), changed no peak latency by more than 0.8 s.

Also "0.5–1 and 1–3 Hz" stays; "CRE and CH" → "C_RE and C_H".

**P204 Fig. 7 caption (62) →**
> Fig. 7 SEC band-power changes surrounding delta-burst onset. Mean SEC band power, expressed as z-scores (mean ± SEM across six participants, 339 onsets), for C_LE, C_RE and C_H, aligned to EEG delta-burst onset (vertical line, t = 0); the dotted line is the count-matched random-NREM control. Panels in each row share the y-axis. Most onsets were arousal-associated: 236 of 339 (70%) had a cortical or autonomic arousal within ±10 s.

**P206 (63):** "(AUC, 0.37–0.51)" → "(area under the receiver operating characteristic curve, AUC, 0.37–0.51; 0.5 is chance)".

**P207 (64), whole paragraph →**
> However, delta-burst onset could not be established as the cause of this response. The response peaked 4–8 s after onset, a latency also consistent with cardiovascular responses to micro-arousals, and most onsets were arousal-associated. Excluding onsets within ±10 s of a scored arousal left 103 of 339 onsets (8 recordings, 5 participants). The response shrank to 17–62% of its full size and weakened in 4–5 of 5 participants, but in the 1–3 Hz band it still exceeded the random-NREM null in every testable participant. K-complexes are also frequently accompanied by autonomic activation; only 9 of the 344 onsets fell within 5 s of a K-complex mark, but the marks are sparse (Section 3.6).

(The first V18 sentence, about sigma, is deleted: sigma was not analysed for delta bursts.)

### 3.6 SEC response at K-complexes

**P210 (65) →**
> The PSG export included 50 K-complexes marked automatically by the PSG software across ten of the twelve recordings (about one per hour of N2), providing event markers independent of our detector. Averaged on those marks, low-frequency SEC power rose after the event and exceeded a random-NREM null (Fig. 8a). The peak followed the K-complex rather than coinciding with it: the grand-average response peaked 3.9 s on C_LE, 3.9 s on C_RE, and 4.1 s on C_H after the mark (Fig. 8a). Electrical pickup of a K-complex would appear at zero lag, so the delayed peak was the key observation. Timing was used here because frequency could not separate the two: K-complexes occupied 0.5–4 Hz, which overlapped the SEC analysis bands.

**P212 Fig. 8 caption (66) →**
> Fig. 8 SEC response at automatically marked K-complexes. (a) Event-triggered average of low-frequency SEC power across the 50 marks, against a random-NREM null (dotted, C_H); the EEG delta envelope (grey, right axis) is shown as a positive control. Triangles mark each channel's peak, whose latency is given in the key. (b) Peak response per recording (number of marks in brackets).

**P214 (67), if Q7 = Discussion only →**
> Each recording contained one to ten events (Fig. 8b), so these findings were reported descriptively without statistical testing.

(The 6.4-s sentence and "shorter latencies might reflect…" move to Discussion ¶6. If Q7 = §3.6, this paragraph instead opens with the event-count sentence, keeps the comparison with ref 6 cited, and the Discussion drops the 6.4 s.)

### 3.7 Multi-band events

**P216 heading →** "3.7 Multi-band events"

**P218 Fig. 9 caption (68) →**
> Fig. 9 Representative multi-band events from the C_H channel during S6N1. The background-enhanced spectrogram (0–5 Hz) shows the detected bands; the hypnogram above it places REM at the bottom. Two events occurred during consolidated N2 sleep and were distinct from the persistent respiratory and cardiac ridges.

The [check] resolves: the redrawn image shows exactly two events.

**P220 (69) →**
> In addition to the persistent respiratory, cardiac, and LFO ridges, SEC spectrograms intermittently showed multi-band events comprising multiple sustained, temporally stable narrow-frequency bands (Fig. 9, Fig. S9). Forty-five events were identified across all 12 sessions from all six participants (Fig. S10). They lasted a median of 8.2 minutes (range 3–64). Of these, 35 occurred in N2, 4 in N3, 2 in N1, 3 in wakefulness, and 1 in REM (Fig. S11a). Event-aligned stage occupancy showed an N2 probability of 0.76 at onset, peaking at 0.82 about 4 min later, against a mean nightly N2 fraction of 0.70 of sleep (range 0.46–0.89), i.e. about 1.1 times chance, so the events occurred mainly in N2, roughly in proportion to its share of sleep. REM occupied 3.5% of the 30 min before event onset, about twice the matched random-NREM value of 1.8%, but the nearest REM epoch was a median of 85 min before an event, and REM was closer before than after the event in only two of six participants (Fig. S11b).

(The REM part is cut to two sentences, and the N1 sentence is deleted.)

**P221 (70) →**
> Counting only the bands that lasted at least half of the event, an event had a median of 4 sustained bands (range 0–13). Bands were tracked separately from event detection, so an event detected by its multi-band score could contain no band lasting half its duration. In the four examples, one sustained band lay at the breathing frequency (0.20–0.30 Hz), and the others lay above it (Fig. S12a and S12b) or below it near 0.07–0.08 Hz (Fig. S12c and S12d).

(The wrong "Fig. S11" reference for the band count is dropped; that histogram is not in the supplement.)

**P222 (71) →**
> Thus, multi-band events occurred mainly during consolidated N2 sleep. Simultaneous thoracic and abdominal belts and nasal flow showed multi-band structure above their own random-NREM baseline in at most 13 of the 45 events, and the SEC bands did not fall at multiples of the breathing rate more often than chance (median 0.20 of bands vs 0.24 expected), so the events were not a harmonic image of respiration. Because the belts were low-pass filtered at 1 Hz, only bands below 1 Hz could be compared. The pooled REM occupancy suggested a link with preceding REM, but this was not consistent across participants. Given 45 events from six participants, these patterns were considered exploratory rather than confirmatory. The events may represent quasi-periodic mechanical or hemodynamic activity associated with stable NREM sleep rather than cortical slow-wave activity.

### 3.8 SEC variance, sleep stages and reproducibility

**P225 (72): the first four sentences are unchanged; then →**
> The highest-variance epochs (the top 10%) showed the opposite pattern. They were over-represented in wakefulness, with median ratios of 1.6 to 2.0 in 8–9 of the 10 nights in which wake made up at least 1% of motion-free epochs, and almost never occurred in N3 (median ratios of 0.00 to 0.08). Because eye movements can couple into the temple sensors, we repeated the analysis after excluding each night's 20% of epochs with the most eye-movement power (anti-phase EOG). Low variance in N3 and high variance in wakefulness survived: N3 enrichment in the bottom decile was 1.43–1.98 (7–10 of 12 nights), and wake enrichment in the top decile was 2.34–2.71. With eye-movement power as a covariate, the elevated variance in REM disappeared (C_H: +0.30 → +0.03 log10 units relative to N2).

**P226 (73) →**
> This pattern paralleled the change in spectral shape from wakefulness to N3 described in Section 3.2 and, together with the eye-movement control, suggested that a simple variance measure could distinguish deep sleep from wakefulness without sleep staging. Several features of the analysis argue against an artifactual explanation. The pattern appeared on the single-ended channels C_LE and C_RE as well as on C_H, so it did not depend on the differential electrode arrangement. Head-movement epochs were also removed, so the high variance in wakefulness was unlikely to reflect motion alone. Using night-specific thresholds also avoided confounding by differences in baseline signal level between nights. Testing variance as a stage classifier in a larger sample is a natural next step.

**P228 Fig. 10 caption (74) →**
> Fig. 10 Night-to-night reproducibility of signal features derived from C_LE − C_RE: (a) percentage of sleep time with the variance exceeding 10 fF² (ρ = 0.66, ICC = 0.69). (b) Percentage of time with variance exceeding 10 times the corresponding night's median variance (ρ = 0.60, ICC = 0.86). (c) Median variance (ρ = 0.94, ICC = 0.40). Each point represents one participant, and the dashed line marks equal values on both nights.

**P229 (75) →**
> Night-to-night reproducibility was assessed using three variance-based features (Fig. 10). All three features ranked participants similarly on both nights (Spearman ρ = 0.66, 0.60 and 0.94). Absolute agreement was moderate for the percentage of time above 10 fF² (ICC = 0.69), good for the percentage above 10 times the night's median (ICC = 0.86), and poor for median variance (ICC = 0.40). Median variance therefore ranked participants consistently, but its level shifted between nights in both directions (Fig. 10c).

(The adaptation sentence is deleted: Night 2 was higher in only 3 of 6 participants.)

### 3.9 SEC-derived features and sleep-related measures

**P232 Fig. 11 caption (76) →**
> Fig. 11 Associations of SEC-derived features with arousal and PSQI (six participants, two-night means). (a) SEC impulse frequency and respiratory arousals (ρ = 0.14), (b) reported PSQI score and mean LFO ridge duration across two nights (ρ = −0.62), (c) reported PSQI score and percentage of sleep time with variance above 10 fF² (ρ = −0.35). Diamonds show each participant's two-night mean, from which ρ is computed; the two nights are the small faint markers (circle night 1, square night 2) joined to it.

**P233 (77) →**
> We explored rank correlations between SEC-derived features and sleep-related physiological and clinical measures across the six participants (two-night means). The five SEC features were impulse frequency (events/h), LFO ridge duration (the summed length of LFO episodes measured by hand, as % of recording time), percentage of the night with SEC variance above 10 fF², median SEC variance (fF²), and mean SEC area (fF·h/h). These were compared with five measures: the PSG-derived total, respiratory and spontaneous arousal indices, PSQI scores, and age (Fig. 11, Fig. S14).

**P234 (78), with Q3 = keep the rms sentences →**
> SEC impulse frequency showed a weak negative association with the total arousal index (ρ = −0.37), little association with the respiratory arousal index (ρ = 0.14), and a negative association with the spontaneous arousal index (ρ = −0.54) (Fig. 11a, Fig. S14a and S14b). Impulse frequency also decreased with increasing PSQI score (ρ = −0.44, Fig. S14c). The duration of LFO ridges was positively associated with raw (C_RE)rms (R = 0.59, p=0.041, Fig. S14 d), whereas the mean LFO ridge duration across the two nights decreased with increasing PSQI score (ρ = −0.62, Fig. 11b). Higher PSQI scores were also associated with a lower percentage of sleep time with variance above the 10 fF² threshold (ρ = −0.35, Fig. 11c). Because impulses are counted above the same 10 fF² threshold, this measure and impulse frequency overlap and are not independent evidence. The destepped (C_LE-C_RE)rms decreased with age (R = -0.58, p=0.05, Fig. S14f). These associations were weak to moderate in six participants. They are exploratory and motivate testing in a larger cohort.

(The two rms sentences are V18's own words, so they still give Pearson R with p. The "|ρ| ≤ 0.62" bracket was dropped from the second-to-last sentence because of them.)

### 4. Discussion

**P237 ¶1 (79) →**
> The SEC recordings showed several associations with sleep physiology across distinct frequency ranges. In the respiratory band (0.1–0.5 Hz), ridge power was lower during N3 in most participants. This tendency persisted across multiple SEC channels after excluding motion epochs and matching epoch counts (Section 3.3), although it followed the overall fall in band power rather than a respiration-specific change. These observations suggest that SEC captured physiological changes accompanying deep sleep, although their origin remains uncertain. LFO ridges were not specific to N3 and occurred in other stages. Ridge counts varied among participants, and most discrete delta-burst onsets were associated with N2 slow waves.

(The two unsupported sentences, about SWA coincidence and (C)rms vs SWA, are deleted.)

**P238 ¶2 (80):** "(Supporting information, Fig. S15)" → "(Fig. S15)"; "Fig S15b" → "Fig. S15b"; "These pooled comparisons described" → "These pooled comparisons describe"; last sentence →
> Since the epochs within participants were not independent, these comparisons do not establish consistent stage effects across individuals.

**P239 ¶3 (81) →**
> To explore the origin of the LFO ridges, we compared SEC with a simultaneously recorded forehead optical PPG signal from one 24-year-old participant wearing the sleep mask (Fig. S16). The 660 nm optical sensor of a Muse S headband (Interaxon, Toronto, Canada), recorded with the Mind Monitor app,32 measured local changes in blood volume in the forehead. After both signals were filtered from 0.01 to 0.3 Hz (0.01–0.1 Hz for the time-domain traces in Fig. S16b), the optical signal showed a respiratory component and broad power below 0.1 Hz, but generally lacked a distinct narrowband ridge. C_LE and C_RE, by comparison, showed ridges near 0.02 Hz, either separately or concurrently. An optical LFO ridge appeared during the same interval as SEC ridges at approximately 0.65–0.85 h, but at a different frequency. The cycles near 0.02 Hz were also more apparent in SEC than in PPG in the time domain. This single-participant comparison suggests that the two sensors capture different aspects of LFO physiology.

(81c resolves without picking a side: 0.01–0.3 Hz is the spectral comparison and 0.01–0.1 Hz the S16b traces. A respiratory component can't appear in a 0.01–0.1 Hz signal, so both bands are real.)

**P240 ¶4 (82) →**
> We tested whether the SEC slope increased at REM onset, as expected from prior invasive pressure and near-infrared spectroscopy studies showing that intracranial blood volume and pressure increase during REM sleep.16,33,34 We compared the average slope over the 15 minutes after each REM-episode onset with the 15 minutes before, averaging within each participant and using 500 random non-REM times from the same nights as a reference (Fig. S17). Only 15 onsets in 5 participants qualified. The C_LE − C_RE slope rose after onset in all 5 participants (median 8.2 fF/h), and in all 4 when onsets with head movement were excluded, whereas C_H rose in only 3 of 5. However, just 1 participant exceeded the upper 5% of their own random-time reference, and the result changed depending on how an onset was defined. Eye movements also increase at REM onset and couple into the temple sensors (Section 3.8), so they may contribute to this rise. The direction matched the expected rise, but with so few events we report it as an observation rather than a finding.

(Q6: softened. The citations are renumbered for the Muse reference, Q12.)

**P241 ¶5 (83) →**
> Our previous work in a porcine model demonstrated an association between SEC signals and measured ICP,23 supporting further investigation of cranial mechanical contributions to the signals recorded during human sleep. Although absolute SEC amplitudes remained difficult to interpret, destepping to reduce abrupt motion-related shifts showed possible associations with age and PSQI in exploratory analyses (Fig. 3), which need a larger cohort. C_LE generally varied less than C_RE across the present recordings, consistent with our previous observations in the porcine model.23 Simultaneous SEC recordings and reference measurements of cranial displacement or ICP would help quantify these relationships.

(V18 cited nothing in this paragraph, so both citations are new.)

**P242 ¶6 (84), with Q7 = Discussion →**
> SEC transients occurred near sleep spindles, whereas the responses to arousal-associated EEG delta-burst onsets peaked 4–8 s after burst onset; without arousals the response was smaller but persisted at 1–3 Hz. This delay is comparable to the approximately 6.4-s lag between slow-delta EEG envelope peaks and MRI-derived CSF signal peaks reported in an EEG–fMRI study,6 in which the hemodynamic response fell between the two. The shorter latency at K-complexes (about 4 s) might reflect sensitivity to hemodynamic changes that precede fluid flow. Such timing is consistent with a proposed mechanism in which reduced neuronal firing during sleep slow waves decreases cerebral blood volume through neurovascular coupling, followed by compensatory CSF movement.35 The delayed SEC response may therefore reflect changes in vascular volume, CSF distribution, or associated tissue displacement. Simultaneous reference measurements are needed to establish its physiological origin.

**P243 ¶7 limitations (85) →**
> The study included six participants and 12 nights, limiting assessment of consistency across individuals, and all associations were exploratory. No ICP or CSF reference was recorded, so the physiological origin of the SEC features could not be established. Eye movements couple into the temple sensors; controlling for them removed the REM elevation in variance but left the N3 and wake differences (Section 3.8). PSG was scored automatically by the PSG software from frontal EEG only, and K-complex marks were sparse. Two participants (S4, S5) had a single synchronization anchor, so their timing is uncertain by about ±1 s, although excluding them changed no latency by more than 0.8 s. Detection settings for ridges and multi-band events were tuned on the same data. Four of the six participants had a PSQI score above 5 and six of the 12 nights had an AHI of 15 or more, and the S5 recordings lasted about 4 h. Home recording reflected a realistic setting, but participant placement and overnight changes in sensor coupling were not directly monitored. Compared with other sleep wearables, such as headband EEG, forehead PPG and near-infrared spectroscopy,18,20,22 SEC senses mechanical rather than electrical or optical signals, but its physiological interpretation is less established. Larger studies with controlled coupling, participant-level analyses, and simultaneous physiological reference measurements are needed to determine the origins of the SEC features and whether they add useful information for sleep physiology.

### 5. Conclusion

**P246 (86), about half the length →**
> We evaluated a wearable SEC sleep mask in 12 overnight home recordings from six adults, using simultaneous PSG as a reference. SEC signals captured respiratory and cardiac rhythms above the electronic noise floor, recurrent LFO, and slower baseline changes. Head position contributed to abrupt baseline shifts. NREM sleep showed higher relative low-frequency spectral power and a less negative aperiodic slope than wakefulness, and SEC band power was lower in N3. Event-aligned analyses showed, on average, increased low-frequency SEC power at the centers of 14,265 N2 spindles. SEC power also rose after 344 EEG delta-burst onsets, most of them arousal-associated, and after 50 automatically marked K-complexes, peaking about 4–8 s and about 4 s later, respectively. Because the responses lagged the EEG events, they are more consistent with a mechanical or hemodynamic origin than with direct detection of cortical electrical activity. Intermittent multi-band events occurred mainly in N2, and SEC variance showed stage-dependent differences and moderate night-to-night consistency. Together, the findings support the feasibility of overnight SEC monitoring and provide a foundation for larger studies with direct physiological reference measurements.

(Dropped: the left–right channel clause, the LFO-across-stages clause, the sigma argument and the exploratory-association clause.)

### Acknowledgement and declarations

**P249 (87) →** four short sections in place of one paragraph (no Data and code availability statement, Q9):
> **Acknowledgement.** The opinions and assertions expressed herein are those of the author(s) and do not reflect the official policy or position of the Uniformed Services University of the Health Sciences or the Department of Defense.
>
> **Funding.** SK, KC, and JC acknowledge partial support from the TriService Nursing Research Program (TSNRP) and the Center for the Study of Traumatic Stress (CSTS).
>
> **Conflicts of interest.** No authors or authors' family members have a financial interest in any commercial product, service, or organization providing financial support for this research.
>
> **Ethics statement.** This research protocol was reviewed and approved by the Institutional Review Board (IRB) at the University of Washington (STUDY00018275: CPC-sleep study).


### References

| Ref (V18 no.) | Final |
|---|---|
| 3 | "Neuropsychiatric Disease and Treatment" |
| 5 | `11(8):2092` |
| 6 | `628-631` (checked: Science 366:628–631) |
| 7 | `Väyrynen T`; `123(12):e2510731123` |
| 8 | `122(41):e2509626122` |
| 9 | `8(1):1014` |
| 10 | `16(1):3630` |
| 12 | `Dec 1 2022;264:119682` |
| 13 | `560-564` (checked: Science 374:560–564) |
| 22 | `12(28):eaed2056` |
| 23 | Kim S, Cheng YJ, Li T, White N, Wang X, Ringgold K, Neidig L, Kwon Y, Chung JH. Non-Invasive Multidimensional Capacitive Sensing for In-Vivo Traumatic Brain Injury Monitoring. Advanced Materials Technologies. 2026. doi:10.1002/admt.202502415 (no volume or pages found yet) |
| 25 | `2022;7(10):2101704` |
| 26 | `Sep 2024` |
| 27 | `2022;344:113739. doi:10.1016/j.sna.2022.113739` |
| 28 | `Feb 2025;4(2). doi` |
| 29 | `Oct 16 2024;377:115701` |
| 30 | IEEE Standard for Safety Levels with Respect to Human Exposure to Electric, Magnetic, and Electromagnetic Fields, 0 Hz to 300 GHz. IEEE Std C95.1-2019. IEEE; 2019. (one line; the split second line goes) |
| 34 → 35 | `2021` |
| 32, 33 | renumbered 33, 34 |
| new 32 | Interaxon Inc. Muse S headband. Toronto, Canada: Interaxon Inc. Data recorded with Mind Monitor (mobile application), https://mind-monitor.com. |

The AASM manual (107) belongs to a Methods citation (point 34a), which is in your sections, so it isn't added here. Restyling the whole list for the journal isn't done either; only the rows above change.

### Global rows, applied from §3.4 on only

- **G2:** every "(CLE−CRE)", "CLE-CRE" etc. → C_LE − C_RE.
- **G4:** "0.5-4 Hz" → "0.5–4 Hz".
- **G6:** "Supporting information, Fig. Sn" → "Fig. Sn" (Q10).
- **G7:** "low frequency oscillation ridges" → "LFO ridges".
- **G10, G11:** Discussion tense and hedging, as written above.
- **G13:** "harmonic-comb" → "multi-band" in P216, P218, P220, P222 and P246.
- **P214:** "(Supporting Information, Fig. S8)" → "(Fig. 8b)", since Fig. 8b now holds the per-recording peaks.
