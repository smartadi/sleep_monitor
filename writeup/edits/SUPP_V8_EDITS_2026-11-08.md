# Supplement V8: every edit, word for word

*2026-11-08. Source: `data/CAP_sleep_mask_manuscript supplementary V8.docx` (34 pages, no tracked changes). The tracked copy will be a separate file, with every change tracked under "Aditya Deole". The source stays untouched. **Now** is the V8 text; **New** is what the copy will say. C<sub>LE</sub> etc. are real subscripts in Word. [Pn] is the paragraph number, for cross-checking only.*

**Decided 2026-11-08 (Aditya): all five recommendations (Q1a, Q2a, Q3a, Q4a, Q5a).** Built as `OneDrive\CAP_sleep_mask_manuscript supplementary V8_aditya_tracked.docx` from `v18_tracked/supp_edits.py` (182 revisions by "Aditya Deole"); the manuscript copy was rebuilt with the B20 renumbering.

---

## Part A · Five decisions (my recommendation first)

### Q1 · Fig. S6 (12 pages of C<sub>RE</sub> ridges) [P69–P96]
Your §3.3 text matches the current pages: "no corresponding slow-frequency ridge was detected during S1N1", "67 total; 2–8 per night", "C<sub>RE</sub> … dominant ridge in 9 of 12 sessions". The new detector (R7/R8) finds an LFO ridge on every night, including S1N1 (44 min, 15% of the night). It finds 25 episodes on C<sub>RE</sub> (1–5 per night), and C<sub>RE</sub> is not dominant by any rule.
- **(a) Rec: keep the current pages.** The captions change only as in B6.
- **(b) Swap** to the new figures:
  - a summary page (`figS6_lfo_ridges_all_channels`) plus 36 panels (C<sub>H</sub>, C<sub>LE</sub>, C<sub>RE</sub> for every night);
  - the captions change "on CRE" to each panel's channel;
  - §3.3 (P186, P191) must then be rewritten to the R8 numbers, otherwise text and figure disagree.

### Q2 · Fig. S8 (K-complexes) [P100–P103]
This is the old three-panel figure: (a) grand average, (b) per-event peak latency, (c) peaks per recording. You dropped panel (b) from Fig. 8 on 2026-11-07 because its medians are what noise would give. Panel (c) is now Fig. 8b, and the main text no longer cites Fig. S8.
- **(a) Rec: delete Fig. S8 and renumber S9–S17 → S8–S16.** This changes:
  - in the supplement, all the captions from S9 on;
  - in the manuscript copy, every citation of them. All of these are in §3.7 onward, which I wrote: P222, P223, P227, P238, P239, P243, P244, P245.
- **(b) Keep it**, with the caption: "**Fig. S8** SEC response at automatically marked K-complexes. (a) Event-triggered average of low-frequency SEC power. (b) Latency of the per-event peak response by channel. (c) Peak response per recording (n = number of K-complexes)."

### Q3 · Statistical analysis section [P174]
**Now:** it describes the dropped rate analysis: rate error, coherence margins, circular shifts, surrogates. It also mentions Wilcoxon tests, Bonferroni correction, and exact-permutation Spearman p-values. None of these are in the paper.
- **(a) Rec: describe what the paper does now.** New text:
  > With six participants and 12 nights, all analyses were exploratory. Wake–NREM differences in aperiodic slope and band excess were tested within recordings with paired t-tests (Section 2.4), and p-values were not corrected for multiple comparisons. Fig. 3 and Fig. S14 report correlation coefficients (r) over the plotted points with uncorrected p-values. Associations between SEC features and participant measures in Section 3.9 and Fig. 11 are Spearman's rank correlations (ρ) across the six participants, using each participant's two-night mean, without p-values. Night-to-night reproducibility (Fig. 10) is reported as ρ between nights and the two-way intraclass correlation for absolute agreement, ICC(2,1). Sleep-stage effects are summarized as the number of nights or participants showing each direction; tests on pooled epochs (Fig. S15) are descriptive only, because epochs within a recording are not independent. Event-triggered responses were compared with random time points drawn from the same recordings and sleep stages.
- **(b) The edit list's D2 rule:** "With six participants, all associations are exploratory and descriptive. For each association we report its direction and Spearman's rank correlation (ρ) across the six participants (two-night means), without p-values. Night-to-night reproducibility is reported as the two-way intraclass correlation for absolute agreement, ICC(2,1). Stage effects are summarised as the number of participants showing each direction." This contradicts the p-values that are still in §3.1–3.2 (Fig. 3, Fig. S5) and in §3.9 (Fig. S14d, S14f).
- **(c) Delete** the heading and the paragraph.

### Q4 · Figures that run over several pages (S2, S3, S6, S9)
At the moment every page repeats the same number.
- **(a) Rec:** the first caption keeps "Fig. S2", and later ones read "Fig. S2 (continued)". Each caption still names its nights.
- **(b) Letters:** S2a–S2f, S3a–S3l, S6a–S6l, S9a–S9l.

### Q5 · Figure images
- **(a) Rec: keep V8's images.** Only text changes. The pipeline's S4 (SNR) and S11 (stage occupancy) show the same data, and V8's S11 image already has (a)/(b).
- **(b) Swap S4 and S11** for the pipeline versions in the new bold style (S4 needs redrawing first).

---

## Part B · Edits ready to type (assuming the recommendations)

### B1 · Title page [P2–P6] (D7, copied from your V18 copy)
- **Author line:** Sakthivelpathi <sup>b</sup> → <sup>c</sup>; Kwon <sup>c</sup> → <sup>d</sup>; Choi <sup>d</sup> → <sup>e</sup>. Lee, Kim, Pham and Chung stay <sup>a</sup>; you stay <sup>b</sup>.
- **New line after "a Mechanical Engineering…":** "<sup>b</sup> BioEngineering Department, University of Washington, Seattle, WA"
- **Relettered lines:**
  - "<sup>b</sup> Nanosync Labs, Redmond, WA" → "<sup>c</sup> Nanosync Labs, Redmond, WA"
  - "<sup>c</sup> Division of Cardiology…" → "<sup>d</sup> Division of Cardiology…"
  - "<sup>d</sup> Department of Psychiatry…" → "<sup>e</sup> Department of Psychiatry…"

### B2 · Fig. S2 captions [P14, P17, P21, P24, P29, P32]
- **Unchanged:** the first, "Fig. S2 S1N1 and S1N2, sleep stages, SEC channels, and head turn angles".
- **The other five** get "(continued)": "Fig. S2 (continued) S2N1 and S2N2, sleep stages, SEC channels, and head turn angles", and so on up to S6N1 and S6N2.

### B3 · Fig. S3 captions [P35–P61]: the "duplicates" are wrong captions, not repeated pages
The 12 images are 12 different nights. I checked the title printed on each image. Each image has its own caption, but from the second one on, the captions are shifted or copied. **Nothing is deleted.** Every caption is corrected to the night shown above it:

| Image shows | Caption now says | New caption |
|---|---|---|
| S1N1 [P35] | S1N1 and S1N2 | Fig. S3 Destepped SEC data for S1N1, sleep stages, and SEC channels |
| S1N2 [P37] | S2N1 and S2N2 | Fig. S3 (continued) Destepped SEC data for S1N2, sleep stages, and SEC channels |
| S2N1 [P39] | S3N1 and S3N2 | … (continued) … for S2N1 … |
| S2N2 [P41] | S4N1 and S4N2 | … (continued) … for S2N2 … |
| S3N1 [P44] | S5N1 and S5N2 | … (continued) … for S3N1 … |
| S3N2 [P46] | S6N1 and S6N2 | … (continued) … for S3N2 … |
| S4N1 [P50] | S5N1 and S5N2 | … (continued) … for S4N1 … |
| S4N2 [P52] | S6N1 and S6N2 | … (continued) … for S4N2 … |
| S5N1 [P54] | S5N1 and S5N2 | … (continued) … for S5N1 … |
| S5N2 [P56] | S6N1 and S6N2 | … (continued) … for S5N2 … |
| S6N1 [P59] | S5N1 and S5N2 | … (continued) … for S6N1 … |
| S6N2 [P61] | S6N1 and S6N2 | … (continued) … for S6N2 … |

### B4 · Fig. S4 [P64]
- **Now:** "**Fig. S4.** Physiological-band SNR for each SEC channel (CH, CLE, and CRE) across the 12 recording sessions. Signal power was defined as the mean spectral density within the physiological band (0.1–3 Hz), encompassing respiratory and cardiac activity. The SNR was positive for every channel in all sessions."
- **New:** "**Fig. S4** Physiological-band SNR for each SEC channel (C<sub>H</sub>, C<sub>LE</sub>, and C<sub>RE</sub>) across the 12 recording sessions. Signal power was defined as the mean spectral density within the physiological band (0.1–3 Hz), encompassing respiratory and cardiac activity. Noise power was the mean spectral density in the 10–50 Hz band of the same night, which an unworn-mask recording showed to be white. The SNR was positive for every channel in all sessions."
- **Why:** this matches your §2 sentence (P152) and the code (`s03_signal.py`: noise band from 10 Hz to the 50 Hz Nyquist frequency of the same night). The stray full stop after "Fig. S4" goes.

### B5 · Fig. S5 [P68] (co-author's figure: wording only)
- **Now:** "**Fig. S5** Spectral slope, per night for frequency band for (a) 0.1-0.3 Hz (b) 0.01-0.1 Hz, (c) 0.1–0.5, and (d) 0.5-3 Hz. Each dot is one night with both states.  Grey lines join the wake and NREM value of the same night. Black lines join the group means."
- **New:** "**Fig. S5** Spectral slope per night in wakefulness and NREM sleep for (a) 0.1–0.3 Hz, (b) the vasomotor band (0.01–0.1 Hz), (c) the respiratory band (0.1–0.5 Hz), and (d) the cardiac band (0.5–3 Hz). Each dot is one night with both states. Grey lines join the wake and NREM values of the same night. Black lines join the group means."
- **Why:** the band names follow your §3.2 (P181), which cites S5b as vasomotor, S5c as respiratory and S5d as cardiac. Why panel (a) uses 0.1–0.3 Hz is the co-author's to say, so it isn't added.

### B6 · Fig. S6 captions [P70–P96] (if Q1 = keep)
- **First caption:** "Fig. S6 Tracked respiratory, cardiac, and LFO ridges on CRE, with the sleep stages for S1N1." → "CRE" becomes C<sub>RE</sub> (subscript), as in the other captions.
- **The other 11:** the same, plus "(continued)": "Fig. S6 (continued) Tracked respiratory, cardiac, and LFO ridges on C<sub>RE</sub>, with the sleep stages for S1N2."

### B7 · Fig. S7 [P99]
This image is byte-identical to V18's old Fig. 7, which the manuscript copy replaces with the redrawn one (rows share a y-axis).
- **Now:** "**Fig. S7** SEC band-power changes surrounding delta-burst onset. Mean SEC band power, expressed as z-scores, is aligned to delta-burst onset for C<sub>LE</sub>, C<sub>RE</sub>, and C<sub>H</sub>.  Signals are aligned to EEG delta-burst onset (green line, t = 0). Most onsets occurred within 10 s of a scored arousal."
- **New:** "**Fig. S7** SEC band-power changes surrounding delta-burst onset. Mean SEC band power, expressed as z-scores, is aligned to EEG delta-burst onset (green line, t = 0) for C<sub>LE</sub>, C<sub>RE</sub>, and C<sub>H</sub>; the grey dashed line is the random-NREM control, and each panel has its own y-axis. Most onsets were arousal-associated: 236 of 339 (70%) had a cortical or autonomic arousal within ±10 s."
- **Why:** the arousal sentence matches the new Fig. 7 caption (D6). "Its own y-axis" is what distinguishes it from Fig. 7.

### B8 · Fig. S8 [P100–P103]
- Deleted (Q2a): the page break, image and caption.
- From here on, every figure number drops by one (S9 → S8 … S17 → S16). The captions below are shown with their **current** numbers.

### B9 · Fig. S9 captions [P106–P131] (12 pages)
- **Now:** "Fig. S9 Harmonic-comb events on C<sub>H</sub>, C<sub>RE</sub>,  and C<sub>LE</sub> with the sleep stages for S1N1."
- **New:** "Fig. S8 Multi-band events on C<sub>H</sub>, C<sub>RE</sub>, and C<sub>LE</sub> with the sleep stages for S1N1."
- The other 11 get the same change plus "(continued)". The double space before "and" is fixed where it occurs.

### B10 · Fig. S10 [P134]
- **Now:** "**Fig. S10** All harmonic-comb events on the twelve nights. Each row is one night, colored by scored sleep stage; black boxes mark comb events, and the number of events per night is given next to its label."
- **New:** "**Fig. S9** All multi-band events on the twelve nights. Each row is one night, colored by scored sleep stage; black boxes mark the events, and the number of events per night is given next to its label."

### B11 · Stale paragraph under Fig. S10 [P135]: deleted
"Total 45 events were found … The timing of the events relative to REM sleep is shown in Fig. 12 of the main text." This paragraph:
- repeats §3.7 almost word for word;
- cites S8, S9 and S10 for content that is in S10, S11 and S12;
- points to a "Fig. 12" that doesn't exist.

### B12 · Fig. S11 [P139]
- **Now:** "**Fig. S11** Sleep-stage distribution of 45 harmonic-comb events from six participants. (A) Stage occupancy within ±30 min of event onset, showing that N2 predominated at onset. (B) REM occupancy surrounding event onset; REM was slightly more frequent before onset than after, but not consistently across participants."
- **New:** "**Fig. S10** Sleep-stage distribution of 45 multi-band events from six participants. (a) Stage occupancy within ±30 min of event onset, showing that N2 predominated at onset. (b) REM occupancy surrounding event onset; REM was slightly more frequent before onset than after, but not consistently across participants."

### B13 · Fig. S12 [P142]
- **Now:** "**Fig. S12** Four comb events up close, on the channel with the most bands: …"
- **New:** "**Fig. S11** Four multi-band events up close, on the channel with the most bands: …" The rest is unchanged.

### B14 · Fig. S13 [P144]
Unchanged apart from its number (→ **Fig. S12**).

### B15 · Fig. S14 [P147] (professor's figure: caption only)
- **Now:** "**Fig. S14** Associations of SEC-derived features with PSG-driven arousals, sleep quality indices (PSQI), LFO ridge duration percentage (%), and age."
- **New:** "**Fig. S13** Associations of SEC-derived features with PSG-scored arousals, sleep quality (PSQI), LFO ridge duration, and age. (a) SEC impulse frequency against the respiratory arousal index, (b) against the spontaneous arousal index, and (c) against PSQI. (d) LFO ridge duration (% of night) against raw (C<sub>RE</sub>)<sub>rms</sub>. (e) Two-night mean LFO ridge duration against PSQI. (f) Destepped (C<sub>LE</sub> − C<sub>RE</sub>)<sub>rms</sub> against age. Each point is one night, except in (e), where it is one participant; dotted lines are linear fits."
- **Why:** the main text cites the panels one by one (S14a, b, c, d, f). The descriptions are read off the axes of the image.

### B16 · Fig. S15 [P151]
Unchanged apart from its number (→ **Fig. S14**). Its sentence already says the Kruskal–Wallis p-values are descriptive only.

### B17 · Fig. S16 [P168]
- "…in the frequency band of 0.01~0.1 Hz." → "…in the frequency band of 0.01–0.1 Hz."
- Number → **Fig. S15**.
- "A 24-year-old male participant" stays. It matches the Discussion ("one 24-year-old participant").

### B18 · Fig. S17 [P170]
- "(a) Average slope of (C<sub>LE</sub>−C<sub>RE</sub>) slope from 30 min before…" → "(a) Average C<sub>LE</sub> − C<sub>RE</sub> slope from 30 min before…"
- "(b) Same as (a) for C<sub>H.</sub>" → "(b) Same as (a) for C<sub>H</sub>." (the full stop was subscripted).
- "(c) Change in (C<sub>LE</sub>−C<sub>RE</sub>) slope from…" → "(c) Change in C<sub>LE</sub> − C<sub>RE</sub> slope from…"
- Number → **Fig. S16**.

### B19 · Statistical analysis [P174]
Replaced with the Q3 text, with its figure numbers renumbered for Q2a: "Fig. S14" → "Fig. S13" (associations) and "Fig. S15" → "Fig. S14" (ridge features).

### B20 · Manuscript copy, renumbering only (Q2a)
| Paragraph | Now | New |
|---|---|---|
| P222 (§3.7) | Fig. S9; Fig. S10; Fig. S11a; Fig. S11b | Fig. S8; Fig. S9; Fig. S10a; Fig. S10b |
| P223 | Fig. S12a and S12b; Fig. S12c and S12d | Fig. S11a and S11b; Fig. S11c and S11d |
| P227 (§3.8) | Fig. S13 | Fig. S12 |
| P238, P239 (§3.9) | Fig. S14; S14a and S14b; S14c; "S14 d"; S14f | Fig. S13; S13a and S13b; S13c; S13d; S13f |
| P243 (Discussion) | Fig. S15; Fig. S15b | Fig. S14; Fig. S14b |
| P244 | Fig. S16; Fig. S16b | Fig. S15; Fig. S15b |
| P245 | Fig. S17 | Fig. S16 |

---

## Part C · Left as they are
- **Title:** "Supplementary material" stays. The manuscript's "Supporting information" mentions are in your sections before §3.4.
- **Fig. S1:** the 0.24 vs 0.25 fF/°C and 0.72 vs 3 fF question is for the hardware co-author. The caption matches your main text (P34).
- **Fig. S13 and S15 text**, and every image unless Q1b or Q5b is chosen.
- **No new supplementary figures.** The eye-movement control is described in words in §3.8. Its figure would need its layout redone in the new style, because the title and y-labels overlap.

## Part D · Old wording inside images (text edits can't reach it)
- **Fig. S9 pages:** each title reads "flat harmonic ladders".
- **Fig. S10 legend:** says "comb event".
- **Fig. S6 pages:** titled "resp/cardiac Viterbi traces".

Changing these means redrawing the figures. The pipeline draws only S6N1 for S9, so the other 11 nights would need new code.
