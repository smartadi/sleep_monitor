# V11 highlights: what each one marks, its history, and a proposed fix

Source: `writeup/review/final/CAP_sleep_mask_manuscript_V11.docx` (2 Oct 18:01, built by `writeup/ppt/_build_v11.py` from V10). Read-only audit, 2026-10-05. No .docx was modified.

**What was extracted**
- `w:highlight w:val="yellow"` runs: **16 spans in 13 paragraphs**, plus empty highlighted runs in ¶176 (explained in D3).
- There is no `word/comments.xml` and no `w:shd` fill anywhere, so the file carries no comments and no shading.
- Paragraph index `P` counts every `w:p` in the body, starting at 0. `paper/outputs/PAPER_NUMBERS.md` numbers paragraphs one higher (its ¶ = P + 1).

**Where the highlights came from**
- Every V11 highlight is already present in the reviewer's V10. None were added by `_build_v11.py`.
- Five carry over from the V8/V9 line, and all five are supplementary cross-references: S1, the low-band method, S4, S5a/b/c and S6.
- The other eight are new in V10: ¶146, ¶152, ¶153, ¶158, ¶171, ¶177, ¶186 and ¶237.

**Supplementary numbering used below** (supplement V4): S1 drift · S2 SNR · S3 coherence + Table S1 · S4 rate pipeline · S5 k by channel · S6 counts and per-epoch k · S7 all twelve recordings · S8 per-epoch k spread · S9 variance tails by stage · S10 S4N2 movement-corrected CLE−CRE/CH with velocity · S11 CLE−CRE all nights · S12 CH all nights · S13 velocity at REM onset. The old S2, S4, S6 and S11 were removed.

| group | n | entries |
|---|---|---|
| A: resolved earlier, fix survived | 1 | ¶34 |
| B: resolved earlier, lost or broken in V11, re-apply | 1 | ¶175 |
| C: never addressed, new proposal | 7 | ¶146, ¶152, ¶153, ¶171, ¶177, ¶178, ¶237 |
| D: needs a decision from the authors | 4 | ¶157, ¶158, ¶176, ¶186 |

---

## A. Resolved earlier, fix survived

### A1. ¶34, §2.1: Fig. S1 (drift test)
- **Highlighted:** "test (Supporting information,  Fig. S1)."
- **Sentence:** "The system exhibits low temperature sensitivity (0.25 fF/°C) and approximately 3 fF total drift over 24 hours with a 3°C temperature variation in our test (Supporting information,  Fig. S1)."
- **History:**
  - Highlighted in V8, V9 and V10.
  - The V6 critique (C6) said S1 was the wrong figure at the time, because the drift test was then S3.
  - In supplement V4, S1 is now the drift test, so the citation is correct and needs no change.
- **Origin:** bench test, made outside this repo (PAPER_NUMBERS EXTERNAL; caption "24-hour continuous operation").
- **Remaining fixes:**
  - Remove the double space before "Fig.".
  - Mark the source, as critique M9 asked: "...in a separate 24-hour bench test (Fig. S1)."
  - ¶171 calls the same test "25-hour" (see C4). The 25 h figure in §2.1 is battery life, not the length of the test.

---

## B. Resolved earlier, lost or broken in V11: re-apply

### B1. ¶175, §3.1: "Fig. S4" for head position
- **Highlighted:** "Supporting information, Fig. S4)."
- **Sentence:** "Head position affected the magnitude of the CLE−CRE through the brain displacement but did not explain its direction (Supporting information, Fig. S4)."
- **History:**
  - V4 cited Figure S3, which was the temperature test. `_build_v6.py:279` corrected this to S4 (then the overnight CH vs CLE−CRE figure) as an [Additional fix].
  - The reviewer's V9 kept "Fig. S4" and highlighted it. Our V7 did not touch it.
  - Supplement V4 then removed old S4, so in V11 "S4" now points at the rate-pipeline figure. The fix survived as text, but its target is gone.
  - V6 critique S6 asked for the claim to rest on the supine analysis. V10 added a hedge but kept "We therefore interpreted…".
- **Origin:** `analysis/mean_value/imbalance_marker.py`. The figures 9,182 epochs, ρ = 0.01, p = 0.62, 7,641 supine epochs, 3.5 reversals and 9.0/11.3/107.7 fF all MATCH.
- **What the pooled number hides:**
  - ρ = 0.01 is pooled over all nights and uses sin(head turn).
  - Night by night, ρ runs from −0.44 to +0.65 and is p < 0.05 in 11 of 12 nights.
  - So direction is associated with head turn within a night, with the sign changing between nights. Pooling cancels it.
- **Mechanism wording:** "through the brain displacement" is an untested mechanism.
- **Proposed text:**
  > "The size of the left–right difference depended on head position (median 9.0 fF lying on the back, 11.3 fF on the left side, 107.7 fF on the right side; one night is shown in Fig. S10). Its sign did not simply follow posture. With the head on its back (7,641 epochs), the sign still flipped a median of 3.5 times per night. Within single nights the sign was related to head turn, but in opposite directions on different nights (ρ from −0.44 to +0.65), so the pooled correlation is close to zero (ρ = 0.01). We therefore treat the sign as a property that changes during the night, and the size as something that must be read together with head position."

---

## C. Never addressed: new proposal

### C1. ¶146, a stray paragraph under an empty Heading 2 (¶145) after §2.3
- **Highlighted (the whole paragraph):** "Sleep stages and all reference and SEC-derived rates were evaluated using a common grid of non-overlapping 30-second epochs, with sleep stages scored according to American Academy of Sleep Medicine (AASM) criteria."
- **History:**
  - Present since V4 as part of "2.4 PSG-based cardiac and respiratory reference rates".
  - V10 deleted that section, because the rate methods moved to the supplement, but left this sentence behind with an empty heading. That deletion is why Methods jumps from 2.4 to 2.7.
  - It was never highlighted or edited before V10. The professor is reading a sentence about rates that the main text no longer contains.
- **Origin:**
  - The epoch grid comes from PSG scoring.
  - "9,319 epochs" in ¶140 is the rate grid (floor(N/3000) per recording). The analysis grid has 9,312 (PAPER_NUMBERS DIFF, s01).
- **Proposed fix:** delete ¶145–146 and end §2.2 with:
  > "All analyses used the 30-second epochs of the PSG scoring, in which a technologist scored sleep stages by AASM criteria. Breathing and heart rates are reported in the Supporting information (Figs. S4–S8)."
- **Also adjacent:** ¶150 is a truncated fragment, "For overview of the overnight study results, mean values of". Finish it or delete it.

### C2. ¶152, §2.4: Viterbi trajectory (whole paragraph highlighted)
- **Highlighted:** "For visualization of continuous overnight rhythms, a single respiratory and cardiac trajectory was obtained as a Viterbi path… Tracker confidence was defined as the fraction of windows in which the tracked band exceeded the column-median background by the prespecified detection criterion. These trajectories were used only to display rhythm persistence and were not the operational rate estimator."
- **History:** unchanged since V4 (V4 ¶151) and never questioned before V10.
- **Origin:** `analysis/slow_wave/ridge_overlay_tune.py:track_single_ridge`. It is used only for Fig. 5.
- **Correctness:**
  - The search bands 0.25–0.55 and 0.85–1.45 Hz MATCH.
  - The method description is wrong (PAPER_NUMBERS DIFF, s05). The median-filter subtraction belongs to the display spectrogram only. The tracker runs on log(power ÷ in-band column median) of a plain spectrogram: 30 s windows for breathing and 15 s for heart, 50% overlap.
  - "Prespecified detection criterion" is never stated. The code uses the tracked bin being more than 2× the column median, which is 3 dB.
  - **Caveat:** the 0.25 Hz floor is 15 breaths/min, but the nights' reference means are 14.2–17.0 breaths/min, so the trace can sit on the band edge. The 1.45 Hz ceiling is 87 beats/min, which excludes S6N2 (about 129 beats/min).
- **Proposed text:**
  > "To show how steady breathing and heartbeat were through the night (Fig. 5), we drew one continuous line through each spectrogram, choosing at every 30-s step the strongest frequency while penalising sudden jumps (a 'Viterbi path'). Each time column was first divided by its own median, so slow background and bursts of movement did not dominate. The search was limited to 0.25–0.55 Hz (15–33 breaths/min) for breathing and 0.85–1.45 Hz (51–87 beats/min) for heart rate. 'Tracker confidence' is the share of windows in which the tracked frequency was at least twice the column median. These lines are for display only. Rates were measured separately (Supporting information, Figs. S4–S8)."

### C3. ¶153, §2.4: SNR and band-power method (whole paragraph highlighted)
- **Highlighted:** "Physiological-band energy was quantified from sliding-window Welch power spectral density estimates. … (CLE-CRE) was excluded from SNR comparisons because subtraction of near-in-phase temporal signals attenuates common-mode physiology."
- **History:**
  - Unchanged since V4 and never questioned.
  - The V6 critique (C3) flagged "temporal" as the wrong word: these are the eye (periocular) channels. It was never fixed.
- **Origin:**
  - `writeup/figures/signal_validation/inband_snr.py` and `inband_snr_split.py`.
  - The band-fraction numbers come from `generate_band_energy.py` (stdout only).
- **Correctness:**
  - The SNR definition and the 10–50 Hz noise band MATCH.
  - "Sliding-window" is wrong. The code computes one Welch spectrum over the whole night (30-s segments, 50% overlap; PAPER_NUMBERS DIFF).
  - The whiteness check holds: the unworn mask gives −1.5 to −0.1 dB.
  - The band fractions quoted in ¶178 were computed on CLE−CRE, which contradicts "CLE−CRE excluded" (see C6).
- **Proposed text:**
  > "To check that breathing and heartbeat stand out from the electronics, we computed one power spectrum per channel over the whole night (Welch method, 30-s segments). SNR is the average power per Hz in 0.1–3 Hz divided by the average power per Hz in 10–50 Hz, where only electronic noise is present, in decibels. Using power per Hz means a wider band does not score higher just because it is wider. The same calculation on a recording with the mask not worn gave 0 dB (−1.5 to −0.1 dB) on every channel. This confirms that the electronic noise is flat across the range and that the 10–50 Hz band is a fair yardstick (Fig. S2). We also report what share of the power below 5 Hz falls in the breathing and heartbeat bands. CLE−CRE was not used for SNR because subtracting two eye channels that move together cancels much of the breathing and heartbeat."

### C4. ¶171, §3.1: "Mention absolute value?" and the session-mean sentence
- **Highlighted:**
  - "Mention absolute value?" (a working note left in the body)
  - "Because absolute capacitance values depended on sensor placement and capacitive coupling, time-varying signals were referenced to the corresponding session means."
- **History:** new in V10. Nothing earlier addresses it.
- **Origin:**
  - `analysis/mean_value/mean_value_vs_stage.py` → `mean_centred_traces.py`. CLE 1958–2048, CRE 1624–2353 and CH −719…−1297 fF all MATCH.
  - "Approximately twice the response" is a hardware statement (EXTERNAL). The measured median sd(CH)/sd(CLE−CRE) is 2.83.
  - "25-hour" conflicts with the S1 caption, which says 24 hours.
- **Answer to the note:** yes. The absolute values are already given in the first sentence, so delete the note.
- **Proposed text:**
  > "In absolute terms, CLE and CRE read close to 2000 fF (night averages 1958–2048 fF for CLE and 1624–2353 fF for CRE) and CH read between −719 and −1297 fF. These absolute levels depend mainly on how the mask sits on the face, so they change from night to night for reasons unrelated to sleep. From here on, each signal is therefore shown as its change from that night's own average. A separate 24-hour bench test showed only 3 fF of drift over a 3 °C temperature change (Fig. S1), much smaller than the overnight changes reported below."

### C5. ¶177: heading "3.2 Spectral characteristics in sleep associated changes"
- **Highlighted:** the whole heading.
- **History:**
  - The V6 critique (C17) said §3.x headings were styled as body text. Never fixed: in V11, 3.1, 3.2, 3.3 and the others are still Normal style, while 2.x headings are Heading 2.
  - The wording is new in V10, when §3.2 was rewritten around Figs. 3 and 4.
- **Proposed fix:**
  - Retitle to **"3.2 Frequency content of the SEC signal and how it changes with sleep"**.
  - Apply Heading 2 to every §3.x heading. This also repairs the navigation pane.

### C6. ¶178, §3.2: "Fig. S6" for SNR
- **Highlighted:** "Supporting Information, Fig. S6"
- **Sentence:** "SNR exceeded 6.4 dB in all recordings (Supporting Information, Fig. S6), and CH consistently yielded the highest SNR in every participant."
- **History:**
  - Highlighted in V8 ("Figure S…") and in V9 and V10. Never re-pointed.
  - V7 (`_build_v7.py`) put the SNR figure back in the main text as Fig. 3, with the stronger caption the critique (C11) asked for: "every channel exceeds that level in every session". V10 dropped it, so that caption fix is lost. The supplement's S2 caption still says only "SNR was positive".
- **Origin:** `inband_snr.py`. 30.0/18.7/20.7 dB and the 6.44 dB minimum MATCH. The rest of the paragraph is wrong in places (PAPER_NUMBERS DIFF, s03):
  - "29–48%" and "8–48%" come from 3 nights of CLE−CRE with a 0.05–10 Hz denominator. Over all 12 nights and the raw channels, the shares are 9–59% (breathing) and 3–60% (heartbeat).
  - CH is highest in 10 of 12 nights, not in every participant: it fails in S4N1 and S4N2.
  - The noise floor is not "nearly constant". It moves by up to 26 dB on CH between nights.
  - The S6N1 (48/8%) and S3N1 (48%) examples use the old definition.
- **Proposed text:**
  > "Breathing and heartbeat were visible in every recording. Depending on night and channel, the breathing band held 9–59% and the heartbeat band 3–60% of the signal power below 5 Hz. Average SNR was 30.0 dB for CH, 18.7 dB for CLE and 20.7 dB for CRE, and every channel exceeded 6.4 dB on every night, against 0 dB for the unworn mask (Fig. S2). CH had the highest SNR on 10 of 12 nights (CRE was higher on both nights of participant 4). The electronic noise level itself differed between nights, so night-to-night SNR differences are not purely physiological."
- **Also:** drop the S6N1/S3N1 sentence, and change the S2 caption to "SNR exceeded 6.4 dB on every channel and night".

### C7. ¶237, §3.8: "(Check the negative signs.)"
- **Highlighted:** "(Check the negative signs.)", a note at the end of the paragraph.
- **History:**
  - New in V10.
  - Related V6 critique items were never applied in the main text: S2 (no n), C1 ("impulse frequency" and "mean SEC area" are never defined) and C2 (amplitude vs variance).
- **Origin:**
  - The numbers come from `writeup/review/Overnight_sleep_subject_list_V3.xlsx` (values read off graphs).
  - The re-computation is in `paper/stages/s09`, using `analysis/mean_value/prof_metrics.py` and `analysis/swa_validation/arousal_index.py`.
- **What the re-computation shows:** the signs are wrong in 3–4 places, one R comes from mis-paired nights, and one value is untraced (PAPER_NUMBERS DIFF).

  | item | text | computed |
  |---|---|---|
  | total arousal | R² = 0.10 | R² = 0.005 (R = −0.07) |
  | respiratory arousal | R = 0.50 | +0.51 |
  | spontaneous arousal | "+0.48" | **−0.48** |
  | PSQI | 0.56 | **−0.57** |
  | oscillation duration vs SEC area | 0.68 | **0.48** (0.68 paired night 1 with night 2) |
  | median variance vs PSQI | 0.48 | **−0.56** |
  | area vs age | 0.63 | **−0.63** |
  | 10 fF² vs PSQI | −0.57 | MATCH |
  | duration vs PSQI | −0.68 (n = 6) | MATCH |

- **Other defects in the same paragraph:**
  - "Fig. Sxx" is a placeholder; no such figure exists.
  - Most panels use n = 12 nights from 6 people, which are not independent.
  - "Duration of low-frequency oscillations" is measured by hand from the traces. It is not PSG slow-wave sleep.
- **Proposed text:**
  > "These associations are descriptive (12 nights from 6 people, or 6 people where stated). Impulse frequency was unrelated to the total arousal index (R = −0.07). It rose with the respiratory arousal index (R = 0.51) and fell with the spontaneous arousal index (R = −0.48) and with PSQI (R = −0.57) (Fig. 11a). The hand-measured duration of low-frequency oscillations rose with mean SEC area (R = 0.48). Averaged over each person's two nights, it fell with PSQI (R = −0.68, n = 6; Fig. 11b). Higher PSQI also went with less of the night above the 10 fF² variance threshold (R = −0.57) and with lower median variance (R = −0.56). Mean SEC area fell with age (R = −0.63). Larger studies are needed to test whether any of these hold."
- **Also:** delete "Fig. Sxx". Redraw Fig. 11 from the s09 outputs, because its panels were built from the same workbook. Define impulse frequency and mean SEC area in Methods.

---

## D. Needs a decision from the professor and authors

### D1. ¶157, §2.7: low-band ridge method (two spans)
- **Highlighted:**
  - "low frequency band (0.01~0.3 Hz)."
  - "[L]ow frequency spectra were calculated using 5-minute windows with 50% overlap, yielding spectral estimates every 2.5 minutes."
- **History:**
  - The V6 critique (M13) said the slow-band ridge had no documented method.
  - The authors' line added these sentences in V8 (as "infralow"), and they were highlighted in V8, V9 and V10. So it was addressed, but with parameters that match no code.
- **Origin and correctness:**
  - The 0.01–0.3 Hz map range is fine.
  - 5 min windows at 50% overlap matches neither detector. The old `ridge_overlay_tune.py` used 5 min windows stepped every 30 s. The current `analysis/slow_wave/ridge_lowband_smooth.py` uses 10 min windows with 4 min Welch segments every 30 s, on the channel band-passed 0.005–0.5 Hz, searches 0.02–0.20 Hz, and keeps a ridge only if it stays above a gate for at least 15 min (gate from block-shuffled copies of the night).
  - **The Results built on the old detector are a filter artifact.** The motion canceller's 0.05 Hz corner produced the fixed 0.05–0.08 Hz band. This affects "67 ridges, 2–8/night, median 6", "near 0.05–0.08 Hz", "no ridge in S1N1", "≈1 h or longer" and the bottom panel of Fig. 5.
  - The current detector gives 25 episodes, 1–5 per night (median 2), at 0.02–0.20 Hz. S1N1 has 2 episodes. The longest is 54.5 min.
  - "CRE … dominant ridge in 9 of 12" is not reproduced: 0/12 by prominence and 7/12 by coverage.
- **Options:**
  - (a) Keep the low-band result. Replace this method sentence with the text below, and update ¶189/¶193, Fig. 5 (bottom panel) and ¶194/¶247 to the new numbers.
  - (b) Drop the low-band ridge from the main text and keep it as exploratory in the supplement.
- **Recommendation:** (b) if time is short. Either way, replace the "9 of 12" justification with: "CRE was used throughout, and the analyses were repeated on CLE and CH."
- **Proposed method text for (a):**
  > "The slow band was handled separately because its rhythms last minutes per cycle. Each channel was band-passed to 0.005–0.5 Hz without motion correction, which distorts this band. Spectra were computed over 10-minute windows moved every 30 seconds, and a continuous ridge was searched between 0.02 and 0.20 Hz. A ridge was kept only if it stayed above a threshold for at least 15 minutes. The threshold was set for each night so that shuffled copies of that night's own data rarely passed it."

### D2. ¶158, §2.7: N3 contrasts (whole paragraph highlighted)
- **Highlighted:** "Sleep-stage contrasts were evaluated across all stages and specifically between N3 and all other stages. … pooled Kruskal-Wallis and Mann-Whitney U tests were reported only as descriptive statistics. Ridge presence was defined as the fraction of clean epochs containing a detected persistent ridge."
- **History:**
  - Unchanged since V4.
  - V10 deleted the old Fig. 6 (ridge features by stage) and the Results paragraph that reported the direction counts and the pooled p-values.
  - That removed the paragraph V7 had repaired for critique S5, so the V7 fix is lost with it.
  - V11 therefore describes tests whose results appear nowhere. Yet ¶194 and ¶247 still claim "lower respiratory ridge power during N3 … after controlling for motion and unequal stage duration".
- **Origin:**
  - `analysis/slow_wave/band_ridge_analysis.py` and `ridge_stage_all_channels.py`.
  - The PROVENANCE CONFLICT says the duration (count) matching was never actually done for power. Paper stage s05 now does it with 200 draws per participant.
- **Counts that resulted:** respiratory ridge power lower in N3 on CH in 5/6 by mean and 6/6 by median; CLE 4/6 and 5/6; CRE 5/6 and 5/6. Ridge count lower in N3 in only 2–3 of 6.
- **Options:**
  - (a) Restore one Results sentence with these counts and simplify the Methods (text below).
  - (b) Delete ¶158 and the N3 ridge-power claims in ¶194 and ¶247.
- **Proposed text for (a):**
  > "To compare deep sleep (N3) with the other stages, we counted how many of the six participants had lower ridge power in N3. Because N3 is short and the other stages long, the comparison was repeated after removing epochs with head movement and after drawing, 200 times per participant, as many non-N3 epochs as there were N3 epochs. Counting participants avoids treating neighbouring 30-second epochs as independent, so no epoch-level p-values are given."
  >
  > Results: "Breathing-ridge power was lower in N3 in 5 of 6 participants on CH (6 of 6 by median), and in 4–5 of 6 on CLE and CRE. The number of ridges showed no consistent direction (2–3 of 6)."

### D3. ¶176, §3.1: Figs. S5a/b/c (integrated magnitude)
- **What the extraction shows:**
  - In V10 and in `address/…V11.docx`, "(", "Fig. S5a).", "(Supporting information, Fig. S5b)", "(" and "Fig. S5c)." are highlighted.
  - In `final/…V11.docx` those runs are highlighted but empty. `_build_v11.py` (the RENAMES loop, lines 198–207) copied the paragraph text into its first run and blanked the rest, which erased the professor's marks.
  - The text survives, but the highlighting does not. Future builds should edit inside runs, as `_build_v7.py:replace_text` does.
- **History:**
  - In V4 this was "(Figure S4)", highlighted. V5 and V6 corrected it to S12.
  - The reviewer's V9 renumbered it to S5. That figure became S6 in supplement V3 and was removed in supplement V4.
  - The fix is lost twice over, and the target no longer exists.
- **Origin:**
  - `analysis/mean_value/imbalance_burden.py`. The values 3.1–135.7 fF, 15–705 fF·h, 46×, 2.0–5.8× and −0.22…+0.09 all MATCH.
  - "Near zero at sleep onset and end (Fig. S5c)" is UNTRACED: no script makes it. Since the trace is referenced to its own mean, being near zero is partly true by construction.
- **Options:**
  - (a) Keep the numbers, remove the three figure citations, and delete the S5c sentence along with the "Because these signals were mean-referenced…" sentence that follows it. This is the recommended option.
  - (b) Restore the integrated-magnitude figure as a new S14.
  - (c) Cite S11 instead. This is not recommended, because S11 shows the movement-corrected trace, not the quantity these numbers describe.
- **Proposed text for (a), replacing the first two sentences:**
  > "Averaged over the night, the size of the left–right difference ranged from 3.1 to 135.7 fF across the 12 recordings. Summed over the night, it ranged from 15 to 705 fF·h, a 46-fold spread. A person's two nights differed by 2- to 6-fold, so this is mostly a property of the night (mask fit and posture), not of the person."

### D4. ¶186: Fig. 4 caption, "Spectral slope (CHECK THE Y AXIS title and unit)" and "0.1–0.5"
- **Highlighted:** the working note, and the band label for panel (c).
- **Caption:** "Fig. 4 Spectral slope (CHECK THE Y AXIS title and unit), per night for frequency band for (a) 0.1-0.3 (b) 0.01-0.1, (c) 0.1–0.5, and (d) 0.5-3 Hz."
- **History:** new in V10. This is a co-author figure: there is no code for the aperiodic fit in the repo (PAPER_NUMBERS EXTERNAL; the caption still says "CHECK THE Y AXIS").
- **Problems:**
  - Panel (a) "0.1–0.3 Hz" overlaps (c) "0.1–0.5 Hz".
  - ¶183 describes (a) as the overall slope and (b)–(d) as the vasomotor, respiratory and cardiac bands, so (a) is probably the full fit range (0.01–3 Hz, as in Fig. 3).
  - ¶183 also mixes "slope" with "band excess ratios", so it is unclear whether panels b–d show slope or excess ratio.
- **Decision needed from the co-author:**
  - The fit range for (a).
  - What b–d plot.
  - The y-axis unit. An aperiodic exponent fitted on log power against log frequency is unitless; "dB/decade" applies if it was fitted in dB.
- **Proposed caption once confirmed:**
  > "Fig. 4. Aperiodic spectral slope (unitless; fitted to log power against log frequency) in wake and NREM for each night, (a) over the full 0.01–3 Hz range, and within (b) the vasomotor band (0.01–0.1 Hz), (c) the breathing band (0.1–0.5 Hz) and (d) the heartbeat band (0.5–3 Hz). Each dot is one night; grey lines join the same night's wake and NREM values, black lines the group means."
- **Also:** delete the note.

---

## Notes for whoever applies these
- V11 still carries V10 text that V7 had fixed: "These findings demonstrate calibrated agreement with PSG", "four male participants aged 54 years", "three recordings contained fewer than 10 events" and "approximately, but not exactly, integer-spaced". `_build_v11.py` kept all of these on purpose. None of them is highlighted, but each is a V7 fix that was lost.
- Main-text figure cross-references are also off by one from Fig. 6 onward (PROVENANCE, "Cross-reference errors in V11"). They are not highlighted.
- When applying edits, edit inside runs rather than collapsing them, so that the professor's highlights survive (see D3).
