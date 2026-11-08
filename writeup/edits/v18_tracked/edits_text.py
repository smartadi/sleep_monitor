# Final text for V18 from section 3.4 on, keyed by paragraph index in word/document.xml.
# Markup: _{sub} ^{sup} *{italic} **{bold}; combinations like ^**{6}.
# Each entry: (guard = start of the old visible text, new markup).
# Decisions 2026-11-07: Q1 code (hand-measured), Q2/Q3/Q4 leave V18 wording, Q5 rec,
# Q6 rec, Q7 Discussion only, Q8 neutral wording, Q9 no data statement, Q10 "Fig. S7",
# Q11 swap figures, Q12 Muse = ref 32.

D = 'C_{LE} − C_{RE}'

EDITS = {
195: ("Sleep spindles were transient",
"Sleep spindles are transient 11–16 Hz thalamocortical oscillations characteristic of N2 sleep and therefore provide precisely timed cortical events for evaluating the SEC response. Across the 12 recordings, 351–2,130 N2 spindles were analyzed per session (14,265 total)."),

196: ("Although individual spindles",
"Low-frequency SEC power (0.1–3 Hz) increased at spindle centers on average, but the increase was carried by a minority of spindles. Per recording, 48–58% of spindles showed a rise, compared with 45–52% of random N2 time points, and on C_{H} the median spindle showed no rise in 5 of 12 recordings; the 5% of spindles with the largest rises accounted for all of the summed increase. The mean increase was 0.45–0.49 dB in the C_{RE} and C_{LE} and 0.54 dB in the C_{H} (mean of recording means; 0.59 dB pooled over all spindles, Fig. 6b). This mean increase was positive in all 12 recordings on C_{H} and " + D + ". In contrast, randomly selected N2 time points showed no comparable response in 11 of 12 recordings on " + D + ". These findings support an association, on average, between sleep spindles and increased SEC power that is not explained by scored arousals. After excluding spindles with an arousal within ±5 s (5.7% of N2 spindles), the increase remained 0.49 dB on C_{H} and 0.45 dB on " + D + ", against 0.05 and 0.08 dB at random N2 times."),

197: ("The response was confined",
"The response was confined to the low-frequency band (0.1–3 Hz). Sigma-band power (11–16 Hz), which contained the spindle oscillation, changed by only 0.02–0.03 dB across the SEC channels, whereas contact EEG sigma power increased by 3.45 dB at the same spindle centers. However, SEC power at 11–16 Hz was at the electronic noise floor (median within 0.4 dB of an unworn-mask recording on C_{LE} and C_{RE}), so no sigma-band change would be expected under either mechanism. The low-frequency SEC response is therefore consistent with spindle-associated mechanical processes, such as transient cardiovascular or cerebrospinal fluid-related displacement, although the sigma band alone cannot exclude direct detection of the electrical spindle oscillation."),

199: ("Fig. 6 Mechanical",
"**{Fig. 6} SEC responses associated with sleep spindles. (a) Event-triggered average of low-frequency SEC power (0.1–3 Hz), showing an average increase at spindle centers on all channels, with the largest increase in C_{H} (0.54 dB). (b) Distributions of spindle-associated power changes in C_{H}: the mean low-frequency change is 0.59 dB (pooled over spindles), but the median is 0.13 dB, so most spindles change little and a minority change a lot; sigma-band power changes by only 0.03 dB. (c) Mean and median low-frequency change per recording and channel. "),

202: ("3.5 SEC response to delta-burst",
"**{3.5 SEC response to delta-burst onsets}"),

203: ("Delta-burst provided",
"Delta-burst onsets provided a second class of discrete cortical events for evaluating the SEC response. In total, 344 qualifying onsets were identified, ranging from 1 to 99 per recording; four recordings contained fewer than 10 events. Two recordings with fewer than five onsets (S4N2, 4 onsets; S5N1, 1 onset) were excluded from averaging, so 339 onsets entered the event-triggered average. Most were isolated N2 slow waves because sustained N3 slow-wave activity generally lacked a discrete onset following a quiet baseline."),

204: ("Event-triggered averaging showed a sharp",
"Event-triggered averaging showed a sharp increase in low-frequency SEC power after delta-burst onset across all channels and analyzed frequency bands. The pre-onset baseline was flat and comparable to the random-NREM control, whereas power increased at onset and peaked several seconds later. The response was greatest in C_{RE} and C_{H} and within the 0.5–1 and 1–3 Hz bands. Across the nine channel–band combinations, subject-level mean post-onset peaks ranged from 1.4 to 3.2 standardized units, compared with no more than 0.17 for the random-NREM controls. The response was observed in all six participants for every channel–band combination, although S5 contributed only six onsets (Fig. 7, Fig. S7). Excluding S4 and S5, whose PSG alignment rests on a single synchronization anchor (uncertain by about ±1 s), changed no peak latency by more than 0.8 s."),

206: ("Fig. 7 SEC band-power",
"**{Fig. 7} SEC band-power changes surrounding delta-burst onset. Mean SEC band power, expressed as z-scores (mean ± SEM across six participants, 339 onsets), for C_{LE}, C_{RE} and C_{H}, aligned to EEG delta-burst onset (vertical line, t = 0); the dotted line is the count-matched random-NREM control. Panels in each row share the y-axis. Most onsets were arousal-associated: 236 of 339 (70%) had a cortical or autonomic arousal within ±10 s. "),

208: ("The SEC response followed rather",
"The SEC response followed rather than preceded delta-burst onset. With causal filtering, which used only past data, SEC power was flat before onset and did not predict an upcoming onset (area under the receiver operating characteristic curve, AUC, 0.37–0.51; 0.5 is chance). Zero-phase filtering, which also used later data, showed a small apparent rise before onset, but this was the large post-onset response spreading backward in time: switching to causal filtering reduced the pre-onset difference from the random-time control from 0.35–0.41 standardized units to approximately zero."),

209: ("As observed for sleep spindles",
"However, delta-burst onset could not be established as the cause of this response. The response peaked 4–8 s after onset, a latency also consistent with cardiovascular responses to micro-arousals, and most onsets were arousal-associated. Excluding onsets within ±10 s of a scored arousal left 103 of 339 onsets (8 recordings, 5 participants). The response shrank to 17–62% of its full size and weakened in 4–5 of 5 participants, but in the 1–3 Hz band it still exceeded the random-NREM null in every testable participant. K-complexes are also frequently accompanied by autonomic activation; only 9 of the 344 onsets fell within 5 s of a K-complex mark, but the marks are sparse (Section 3.6)."),

212: ("The PSG export included 50",
"The PSG export included 50 K-complexes marked automatically by the PSG software across ten of the twelve recordings (about one per hour of N2), providing event markers independent of our detector. Averaged on those marks, low-frequency SEC power rose after the event and exceeded a random-NREM null (Fig. 8a). The peak followed the K-complex rather than coinciding with it: the grand-average response peaked 3.9 s on C_{LE}, 3.9 s on C_{RE}, and 4.1 s on C_{H} after the mark (Fig. 8a). Electrical pickup of a K-complex would appear at zero lag, so the delayed peak was the key observation. Timing was used here because frequency could not separate the two: K-complexes occupied 0.5–4 Hz, which overlapped the SEC analysis bands."),

214: ("Fig. 8 SEC response at",
"**{Fig. 8} SEC response at automatically marked K-complexes. (a) Event-triggered average of low-frequency SEC power across the 50 marks, against a random-NREM null (dotted, C_{H}); the EEG delta envelope (grey, right axis) is shown as a positive control. Triangles mark each channel’s peak, whose latency is given in the key. (b) Peak response per recording (number of marks in brackets). "),

216: ("The direction and timescale",
"Each recording contained one to ten events (Fig. 8b), so these findings were reported descriptively without statistical testing."),

218: ("3.7 Harmonic-comb events",
"**{3.7 Multi-band events }"),

220: ("Fig. 9 Representative harmonic",
"Fig. 9 Representative multi-band events from the C_{H} channel during S6N1. The background-enhanced spectrogram (0–5 Hz) shows the detected bands; the hypnogram above it places REM at the bottom. Two events occurred during consolidated N2 sleep and were distinct from the persistent respiratory and cardiac ridges. "),

222: ("In addition to the persistent",
"In addition to the persistent respiratory, cardiac, and LFO ridges, SEC spectrograms intermittently showed multi-band events comprising multiple sustained, temporally stable narrow-frequency bands (Fig. 9, Fig. S8). Forty-five events were identified across all 12 sessions from all six participants (Fig. S9). They lasted a median of 8.2 minutes (range 3–64). Of these, 35 occurred in N2, 4 in N3, 2 in N1, 3 in wakefulness, and 1 in REM (Fig. S10a). Event-aligned stage occupancy showed an N2 probability of 0.76 at onset, peaking at 0.82 about 4 min later, against a mean nightly N2 fraction of 0.70 of sleep (range 0.46–0.89), i.e. about 1.1 times chance, so the events occurred mainly in N2, roughly in proportion to its share of sleep. REM occupied 3.5% of the 30 min before event onset, about twice the matched random-NREM value of 1.8%, but the nearest REM epoch was a median of 85 min before an event, and REM was closer before than after the event in only two of six participants (Fig. S10b)."),

223: ("Counting only the bands",
"Counting only the bands that lasted at least half of the event, an event had a median of 4 sustained bands (range 0–13). Bands were tracked separately from event detection, so an event detected by its multi-band score could contain no band lasting half its duration. In the four examples, one sustained band lay at the breathing frequency (0.20–0.30 Hz), and the others lay above it (Fig. S11a and S11b) or below it near 0.07–0.08 Hz (Fig. S11c and S11d). "),

224: ("Thus, harmonic-comb events",
"Thus, multi-band events occurred mainly during consolidated N2 sleep. Simultaneous thoracic and abdominal belts and nasal flow showed multi-band structure above their own random-NREM baseline in at most 13 of the 45 events, and the SEC bands did not fall at multiples of the breathing rate more often than chance (median 0.20 of bands vs 0.24 expected), so the events were not a harmonic image of respiration. Because the belts were low-pass filtered at 1 Hz, only bands below 1 Hz could be compared. The pooled REM occupancy suggested a link with preceding REM, but this was not consistent across participants. Given 45 events from six participants, these patterns were considered exploratory rather than confirmatory. The events may represent quasi-periodic mechanical or hemodynamic activity associated with stable NREM sleep rather than cortical slow-wave activity."),

227: ("Across 30-s epochs",
"Across 30-s epochs, the variance of the SEC signal tracked sleep depth. Each night was scaled to its own variance distribution and head-movement epochs were excluded. The comparison reflected sleep stage rather than baseline differences or motion (Fig. S12). Under these conditions, the lowest-variance epochs (the bottom 10%) were over-represented in N3, with median ratios of 1.6 to 2.3 times the night-wide frequency in 9 to 11 of 12 nights across channels, and under-represented in wakefulness, N1 and REM. The highest-variance epochs (the top 10%) showed the opposite pattern. They were over-represented in wakefulness, with median ratios of 1.6 to 2.0 in 8–9 of the 10 nights in which wake made up at least 1% of motion-free epochs, and almost never occurred in N3 (median ratios of 0.00 to 0.08). Because eye movements can couple into the temple sensors, we repeated the analysis after excluding each night’s 20% of epochs with the most eye-movement power (anti-phase EOG). Low variance in N3 and high variance in wakefulness survived: N3 enrichment in the bottom decile was 1.43–1.98 (7–10 of 12 nights), and wake enrichment in the top decile was 2.34–2.71. With eye-movement power as a covariate, the elevated variance in REM disappeared (C_{H}: +0.30 → +0.03 log_{10} units relative to N2). "),

228: ("This pattern was consistent",
"This pattern paralleled the change in spectral shape from wakefulness to N3 described in Section 3.2 and, together with the eye-movement control, suggested that a simple variance measure could distinguish deep sleep from wakefulness without sleep staging. Several features of the analysis argue against an artifactual explanation. The pattern appeared on the single-ended channels C_{LE} and C_{RE} as well as on C_{H}, so it did not depend on the differential electrode arrangement. Head-movement epochs were also removed, so the high variance in wakefulness was unlikely to reflect motion alone. Using night-specific thresholds also avoided confounding by differences in baseline signal level between nights. Testing variance as a stage classifier in a larger sample is a natural next step."),

230: ("Fig. 10 Night-to-night",
"**{Fig. 10} Night-to-night reproducibility of signal features derived from " + D + ": (a) percentage of sleep time with the variance exceeding 10 fF² (ρ = 0.66, ICC = 0.69). (b) Percentage of time with variance exceeding 10 times the corresponding night’s median variance (ρ = 0.60, ICC = 0.86). (c) Median variance (ρ = 0.94, ICC = 0.40). Each point represents one participant, and the dashed line marks equal values on both nights."),

232: ("Night-to-night reproducibility was",
"Night-to-night reproducibility was assessed using three variance-based features (Fig. 10). All three features ranked participants similarly on both nights (Spearman ρ = 0.66, 0.60 and 0.94). Absolute agreement was moderate for the percentage of time above 10 fF² (ICC = 0.69), good for the percentage above 10 times the night’s median (ICC = 0.86), and poor for median variance (ICC = 0.40). Median variance therefore ranked participants consistently, but its level shifted between nights in both directions (Fig. 10c). "),

236: ("[IMG]Fig. 11 Associations",
"Fig. 11 Associations of SEC-derived features with arousal and PSQI (six participants, two-night means). (a) SEC impulse frequency and respiratory arousals (ρ = 0.14), (b) reported PSQI score and mean LFO ridge duration across two nights (ρ = −0.62), (c) reported PSQI score and percentage of sleep time with variance above 10 fF² (ρ = −0.35). Diamonds show each participant’s two-night mean, from which ρ is computed; the two nights are the small faint markers (circle night 1, square night 2) joined to it. "),

238: ("We performed an exploratory",
"We explored rank correlations between SEC-derived features and sleep-related physiological and clinical measures across the six participants (two-night means). The five SEC features were impulse frequency (events/h), LFO ridge duration (the summed length of LFO episodes measured by hand, as % of recording time), percentage of the night with SEC variance above 10 fF², median SEC variance (fF²), and mean SEC area (fF·h/h). These were compared with five measures: the PSG-derived total, respiratory and spontaneous arousal indices, PSQI scores, and age (Fig. 11, Fig. S13)."),

239: ("SEC impulse frequency showed",
"SEC impulse frequency showed a weak negative association with the total arousal index (ρ = −0.37), little association with the respiratory arousal index (ρ = 0.14), and a negative association with the spontaneous arousal index (ρ = −0.54) (Fig. 11a, Fig. S13a and S13b). Impulse frequency also decreased with increasing PSQI score (ρ = −0.44, Fig. S13c). The duration of LFO ridges was positively associated with raw (C_{RE})_{rms} (R = 0.59, p=0.041, Fig. S13d), whereas the mean LFO ridge duration across the two nights decreased with increasing PSQI score (ρ = −0.62, Fig. 11b). Higher PSQI scores were also associated with a lower percentage of sleep time with variance above the 10 fF² threshold (ρ = −0.35, Fig. 11c). Because impulses are counted above the same 10 fF² threshold, this measure and impulse frequency overlap and are not independent evidence. The destepped (C_{LE}-C_{RE})_{rms} decreased with age (R = -0.58, p=0.05, Fig. S13f). These associations were weak to moderate in six participants. They are exploratory and motivate testing in a larger cohort. "),

242: ("The SEC recordings showed several",
"The SEC recordings showed several associations with sleep physiology across distinct frequency ranges. In the respiratory band (0.1–0.5 Hz), ridge power was lower during N3 in most participants. This tendency persisted across multiple SEC channels after excluding motion epochs and matching epoch counts (Section 3.3), although it followed the overall fall in band power rather than a respiration-specific change. These observations suggest that SEC captured physiological changes accompanying deep sleep, although their origin remains uncertain. LFO ridges were not specific to N3 and occurred in other stages. Ridge counts varied among participants, and most discrete delta-burst onsets were associated with N2 slow waves."),

243: ("The band-specific analysis",
"The band-specific analysis provided further context (Fig. S14). Respiratory-band ridge count and lowest frequency were broadly similar across stages. In pooled epochs, respiratory ridge power was highest in N2, and its median was lower in N3 than in N2, with substantial overlap between distributions. In the cardiac band (0.5–3.0 Hz, Fig. S14b), mean ridge count was highest in N3, whereas ridge power was highest and most variable in N2. The lowest ridge frequency tended to be lower in N3 and N2 than in the other stages, although distributions overlapped. These pooled comparisons describe the recorded epochs. Since the epochs within participants were not independent, these comparisons do not establish consistent stage effects across individuals. "),

244: ("To explore the origin",
"To explore the origin of the LFO ridges, we compared SEC with a simultaneously recorded forehead optical PPG signal from one 24-year-old participant wearing the sleep mask (Fig. S15). The 660 nm optical sensor of a Muse S headband (Interaxon, Toronto, Canada), recorded with the Mind Monitor app,^{32} measured local changes in blood volume in the forehead. After both signals were filtered from 0.01 to 0.3 Hz (0.01–0.1 Hz for the time-domain traces in Fig. S15b), the optical signal showed a respiratory component and broad power below 0.1 Hz, but generally lacked a distinct narrowband ridge. C_{LE} and C_{RE}, by comparison, showed ridges near 0.02 Hz, either separately or concurrently. An optical LFO ridge appeared during the same interval as SEC ridges at approximately 0.65–0.85 h, but at a different frequency. The cycles near 0.02 Hz were also more apparent in SEC than in PPG in the time domain. This single-participant comparison suggests that the two sensors capture different aspects of LFO physiology. "),

245: ("We tested if the SEC",
"We tested whether the SEC slope increased at REM onset, as expected from prior invasive pressure and near-infrared spectroscopy studies showing that intracranial blood volume and pressure increase during REM sleep.^{16,33,34} We compared the average slope over the 15 minutes after each REM-episode onset with the 15 minutes before, averaging within each participant and using 500 random non-REM times from the same nights as a reference (Fig. S16). Only 15 onsets in 5 participants qualified. The " + D + " slope rose after onset in all 5 participants (median 8.2 fF/h), and in all 4 when onsets with head movement were excluded, whereas C_{H} rose in only 3 of 5. However, just 1 participant exceeded the upper 5% of their own random-time reference, and the result changed depending on how an onset was defined. Eye movements also increase at REM onset and couple into the temple sensors (Section 3.8), so they may contribute to this rise. The direction matched the expected rise, but with so few events we report it as an observation rather than a finding."),

246: ("Our previous work in a porcine",
"Our previous work in a porcine model demonstrated an association between SEC signals and measured ICP,^{23} supporting further investigation of cranial mechanical contributions to the signals recorded during human sleep. Although absolute SEC amplitudes remained difficult to interpret, destepping to reduce abrupt motion-related shifts showed possible associations with age and PSQI in exploratory analyses (Fig. 3), which need a larger cohort. C_{LE} generally varied less than C_{RE} across the present recordings, consistent with our previous observations in the porcine model.^{23} Simultaneous SEC recordings and reference measurements of cranial displacement or ICP would help quantify these relationships. "),

247: ("SEC transients occurred near",
"SEC transients occurred near sleep spindles, whereas the responses to arousal-associated EEG delta-burst onsets peaked 4–8 s after burst onset; without arousals the response was smaller but persisted at 1–3 Hz. This delay is comparable to the approximately 6.4-s lag between slow-delta EEG envelope peaks and MRI-derived CSF signal peaks reported in an EEG–fMRI study,^**{6} in which the hemodynamic response fell between the two. The shorter latency at K-complexes (about 4 s) might reflect sensitivity to hemodynamic changes that precede fluid flow. Such timing is consistent with a proposed mechanism in which reduced neuronal firing during sleep slow waves decreases cerebral blood volume through neurovascular coupling, followed by compensatory CSF movement.^{35} The delayed SEC response may therefore reflect changes in vascular volume, CSF distribution, or associated tissue displacement. Simultaneous reference measurements are needed to establish its physiological origin."),

248: ("The study included six participants",
"The study included six participants and 12 nights, limiting assessment of consistency across individuals, and all associations were exploratory. No ICP or CSF reference was recorded, so the physiological origin of the SEC features could not be established. Eye movements couple into the temple sensors; controlling for them removed the REM elevation in variance but left the N3 and wake differences (Section 3.8). PSG was scored automatically by the PSG software from frontal EEG only, and K-complex marks were sparse. Two participants (S4, S5) had a single synchronization anchor, so their timing is uncertain by about ±1 s, although excluding them changed no latency by more than 0.8 s. Detection settings for ridges and multi-band events were tuned on the same data. Four of the six participants had a PSQI score above 5 and six of the 12 nights had an AHI of 15 or more, and the S5 recordings lasted about 4 h. Home recording reflected a realistic setting, but participant placement and overnight changes in sensor coupling were not directly monitored. Compared with other sleep wearables, such as headband EEG, forehead PPG and near-infrared spectroscopy,^{18,20,22} SEC senses mechanical rather than electrical or optical signals, but its physiological interpretation is less established. Larger studies with controlled coupling, participant-level analyses, and simultaneous physiological reference measurements are needed to determine the origins of the SEC features and whether they add useful information for sleep physiology."),

251: ("We evaluated a wearable SEC",
"We evaluated a wearable SEC sleep mask in 12 overnight home recordings from six adults, using simultaneous PSG as a reference. SEC signals captured respiratory and cardiac rhythms above the electronic noise floor, recurrent LFO, and slower baseline changes. Head position contributed to abrupt baseline shifts. NREM sleep showed higher relative low-frequency spectral power and a less negative aperiodic slope than wakefulness, and SEC band power was lower in N3. Event-aligned analyses showed, on average, increased low-frequency SEC power at the centers of 14,265 N2 spindles. SEC power also rose after 344 EEG delta-burst onsets, most of them arousal-associated, and after 50 automatically marked K-complexes, peaking about 4–8 s and about 4 s later, respectively. Because the responses lagged the EEG events, they are more consistent with a mechanical or hemodynamic origin than with direct detection of cortical electrical activity. Intermittent multi-band events occurred mainly in N2, and SEC variance showed stage-dependent differences and moderate night-to-night consistency. Together, the findings support the feasibility of overnight SEC monitoring and provide a foundation for larger studies with direct physiological reference measurements."),

# P254 (Acknowledgement) stays as V18 has it: the Funding / Conflicts of interest /
# Ethics statement split was reverted on 2026-11-08 at Aditya's request.

# References (V18 numbering)
258: ("3.\tAlhola", None, [("*{Neuropsychiatric disease and treatment}", "*{Neuropsychiatric Disease and Treatment}")]),
260: ("5.\tVoumvourakis", None, [("11(8)2092", "11(8):2092")]),
261: ("6.\tFultz", None, [("628-+", "628-631")]),
262: ("7.\tVäyrynena", None, [("Väyrynena T", "Väyrynen T"), ("123(12)2510731123", "123(12):e2510731123")]),
263: ("8.\tUji", None, [("122(41)2509626122", "122(41):e2509626122")]),
264: ("9.\tXu YB", None, [("8(1)1014", "8(1):1014")]),
265: ("10.\tBiabani", None, [("16(1)3630", "16(1):3630")]),
267: ("12.\tRuch", None, [("Dec 1 2022;264119682", "Dec 1 2022;264:119682")]),
268: ("13.\tGirardeau", None, [("560-+", "560-564")]),
278: ("22.\tBan S", None, [("12(28)eaed2056", "12(28):eaed2056")]),
279: ("23.\tShawn Kim", None, [("Shawn Kim YJC, Tianyi Li, Nathan White, Xu Wang, Kristyn Ringgold, Lauren Neidig, Younghoon Kwon, Jae‐Hyun Chung.", "Kim S, Cheng YJ, Li T, White N, Wang X, Ringgold K, Neidig L, Kwon Y, Chung JH."),
                                     ("2026:https://doi.org/10.1002/admt.202502415. ", "2026. doi:10.1002/admt.202502415")]),
281: ("25.\tLi TY", None, [("2022 2022;7(10)2101704", "2022;7(10):2101704")]),
282: ("26.\tCheng YJ", None, [("2024 Sep 2024", "Sep 2024")]),
283: ("27.\tSakthivelpathi", None, [("2022/09/01/ 2022;344:113739. doi:https://doi.org/10.1016/j.sna.2022.113739", "2022;344:113739. doi:10.1016/j.sna.2022.113739")]),
284: ("28.\tCheng YJ", None, [("Feb 2025;4(2)doi", "Feb 2025;4(2). doi")]),
285: ("29.\tSakthivelpathi", None, [("Oct 16 2024;377115701", "Oct 16 2024;377:115701")]),
286: ("30.\tCommittee ISC", "30.\tIEEE Standard for Safety Levels with Respect to Human Exposure to Electric, Magnetic, and Electromagnetic Fields, "),
287: ("0 Hz to 300 GHz", "0 Hz to 300 GHz. *{IEEE Std C95.1-2019}. IEEE; 2019."),
289: ("32.\tYokota", None, [("32.", "33.")]),
290: ("33.\tRiedel", None, [("33.", "34.")]),
291: ("34.\tLewis", None, [("34.", "35."), ("2021/10/29 2021", "2021")]),
}

# Paragraph marks deleted (paragraph joins the next one): ref 30's split line.
DELETE_MARK = [286]

# New paragraphs: (insert_after_index, template_index, markup)
NEW_PARAS = [
    (288, 289, "32.\tInteraxon Inc. Muse S headband. Toronto, Canada: Interaxon Inc. Data recorded with Mind Monitor (mobile application), https://mind-monitor.com."),
]

# Figure swaps: drawing paragraph index -> new image file (relative to the sleep_monitor repo)
FIGURES = {
    198: 'paper/outputs/figures/s07_spindles/fig6_spindles.png',
    205: 'paper/outputs/figures/s08_delta_kcomplex/fig7_delta_onset_causal_grid.png',
    213: 'paper/outputs/figures/s10_review_v18/fig8_kcomplex_response_causal.png',
    219: 'paper/outputs/figures/s06_harmonic_comb/fig8_S6N1_CH.png',
    229: 'paper/outputs/figures/s09_reproducibility/fig10_reproducibility.png',
    236: 'paper/outputs/figures/s09_reproducibility/fig11_associations.png',
}
