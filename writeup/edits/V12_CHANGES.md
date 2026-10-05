# V12 changes (tracked revision of V11)

Built by `writeup/ppt/_build_v12.py` from `writeup/review/final/CAP_sleep_mask_manuscript_V11.docx` (unchanged) into `CAP_sleep_mask_manuscript_V12.docx`. Every edit is a Word tracked change (author "Aditya Deole", 2026-10-05) made inside the existing runs; the rPr of the edited run is copied onto inserted text.

Paragraph index **P** counts every `w:p` in the body from 0 (as in V11_HIGHLIGHTS.md); paper_numbers.csv / the task use **¶ = P + 1**.

**47 edits** (43 insertions, 64 deletion wrappers). Highlighted runs with visible text: 60 before, 65 after. Highlighted text deleted: 'a frequency-median-filtered spectrum was subtracted', 'by the prespecified detection criterion', 'S6'. Validation: "reject all" view equals V11 text and highlighting in every paragraph.

| section | edits |
|---|---|
| Discussion | 8 |
| Conclusion | 1 |
| §3.2 | 6 |
| §3.6 | 7 |
| §3.4 | 5 |
| §3.5 | 5 |
| §3.7 | 1 |
| §3.1 | 1 |
| §2.4 Methods | 2 |
| §2.7 Methods | 2 |
| Fig. 6 caption | 4 |
| Fig. 8 caption | 1 |
| Fig. 9 caption | 2 |
| Fig. 2 caption | 2 |

## Edits

### A. Rate numbers

1. **P252 (¶253), Discussion.** "…breaths/min and **1.56 beats/min**, respectively; corresponding…" → **"1.19 beats/min"**
   - Reason: Nightly cardiac error with the ECG R-peak reference (Pleth only on S5N1, S6N2): 1.194.
   - Source: paper/outputs/paper_numbers.csv night_err_card_disc; paper/outputs/paper_numbers.csv (s04_rates); paper/outputs/tables/s04_rates/heldout_table.csv; reports/rates/k_by_channel.csv

2. **P252 (¶253), Discussion.** "…errors were 1.79 breaths/min and **3.41 beats/min**. These findings…" → **"3.09 beats/min (Supporting information, Figs. S4–S8)"**
   - Reason: Epoch-level cardiac error with the ECG reference: 3.094. Cites the supplementary rate figures (V4: S4 pipeline, S5 k by channel, S6 counts/per-epoch k, S7 all twelve recordings, S8 per-epoch k spread), since the rate method and results now live there.
   - Source: paper/outputs/paper_numbers.csv epoch_err_card; supplementary V4 captions S4–S8

3. **P252 (¶253), Discussion.** "…beats/min. **These findings demonstrate calibrated agreement with PSG. **Cardiac measurements…" → **"(deleted)"**
   - Reason: Sentence deleted: k is fitted on the same night it is evaluated on, so the per-night error cannot demonstrate agreement with PSG.
   - Source: task instruction; V11_HIGHLIGHTS.md "Notes" (lost V7 fix)

4. **P252 (¶253), Discussion.** "…nights (median, **1.96**; interquartile range…" → **"2.00"**
   - Reason: Cardiac k (CRE) median with the ECG reference.
   - Source: paper/outputs/paper_numbers.csv k_card; reports/rates/k_by_channel.csv

5. **P252 (¶253), Discussion.** "…interquartile range, **1.77–2.01**), suggesting…" → **"1.95–2.07"**
   - Reason: Cardiac k IQR with the ECG reference.
   - Source: paper/outputs/paper_numbers.csv k_card

6. **P252 (¶253), Discussion.** "…per heartbeat. **R-peak–triggered averaging yielded** a median of 2.02…" → **"Counting detected peaks per ECG R-peak gave"**
   - Reason: The 2.02 is a count ratio (CRE peaks ÷ ECG R-peaks over the asleep night), not R-peak-triggered averaging. Value 2.018 unchanged.
   - Source: paper/outputs/paper_numbers.csv peaks_per_beat; analysis/rates/peaks_per_beat.py; PROVENANCE.md (rates)

7. **P253 (¶254), Discussion.** "…nightly mean rates (**14.4–16.8** breaths/min)…" → **"14.2–17.0"**
   - Reason: Respiratory reference range across nights from the rerun pipeline: 14.21–16.98.
   - Source: paper/outputs/paper_numbers.csv resp_ref_range

8. **P253 (¶254), Discussion.** "…achieved an error of 1.20 breaths/min** (Table 3)**, underscoring…" → **"(deleted)"**
   - Reason: V11 has no Table 3. The 1.20 itself MATCHes.
   - Source: paper/outputs/paper_numbers.csv nosensor_resp; PROVENANCE.md cross-reference list

9. **P258 (¶259), Conclusion.** "…0.24 breaths/min and **1.56 beats/min**, respectively; reliable…" → **"1.19 beats/min"**
   - Reason: Nightly cardiac error with the ECG reference (same quantity as Discussion).
   - Source: paper/outputs/paper_numbers.csv night_err_card_conc

### B. Supplementary cross-references

10. **P178 (¶179), §3.2.** "…Supporting Information, Fig. **S6**), and C…" → **"S2"**
   - Reason: SNR figure is Fig. S2 in supplement V4 (old S6 / S3).
   - Source: supplementary V4 caption "Figure S2. Physiological-band SNR"; paper/outputs/paper_numbers.csv snr_fig_ref

11. **P218 (¶219), §3.6.** "…**Supporting information Fig. S11**).…" → **"Supporting information, Figs. S16–S18"**
   - Reason: Old S11 no longer exists; the harmonic-comb events are now supplementary section S6 (S16 all 45 events over the stages, S17 four events up close, S18 event properties).
   - Source: supplementary V4 section S6; analysis/slow_wave/supp_comb_figures.py

### C. Main-figure cross-references

12. **P198 (¶199), §3.4.** "… channel (Fig. **7**). This average…" → **"6"**
   - Reason: Spindle figure is captioned "Fig. 6. Mechanical SEC responses associated with sleep spindles".
   - Source: V11 caption P201; PROVENANCE.md cross-reference list

13. **P206 (¶207), §3.5.** "…channel–band combination (Fig. **8**).…" → **"7"**
   - Reason: Delta-onset figure is captioned "Fig. 7 SEC band-power changes surrounding delta-burst onset".
   - Source: V11 caption P211; PROVENANCE.md

14. **P218 (¶219), §3.6.** "…and 0.95 Hz (Fig. **9**, Supporting information…" → **"8"**
   - Reason: Comb example is captioned "Fig. 8 Representative harmonic-comb events".
   - Source: V11 caption P216; PROVENANCE.md

15. **P229 (¶230), §3.7.** "…two separate nights (Fig. **11**). Significant…" → **"10"**
   - Reason: Reproducibility figure is captioned "Fig. 10. Night-to-night reproducibility".
   - Source: V11 caption P227; paper/outputs/paper_numbers.csv fig_ref_230

### D. Factual corrections

16. **P174 (¶175), §3.1.** "…exceeded 1000 fF in the **three** most mobile recordings…" → **"five"**
   - Reason: Five recordings exceed 1000 fF within-night CH range (S2N1, S2N2, S4N1, S6N1, S6N2).
   - Source: paper/outputs/paper_numbers.csv ch_gt_1000; analysis/mean_value/mean_centred_traces.py

17. **P178 (¶179), §3.2.** "…accounted for **29–48%** of the total power below 5 Hz…" → **"9–59%"**
   - Reason: All 12 nights × CH/CLE/CRE, denominator 0 < f ≤ 5 Hz (the text already states "power below 5 Hz"). 29–48% was 3 nights of CLE−CRE with a 0.05–10 Hz denominator.
   - Source: paper/outputs/paper_numbers.csv resp_frac_range; writeup/figures/signal_validation/generate_band_energy.py

18. **P178 (¶179), §3.2.** "…contributed an additional **8–48%**. Relative to…" → **"3–60%"**
   - Reason: As above, cardiac band share.
   - Source: paper/outputs/paper_numbers.csv card_frac_range

19. **P178 (¶179), §3.2.** "…and CH **consistently **yielded the highest SNR…" → **"(deleted)"**
   - Reason: "consistently" no longer true (see next edit).
   - Source: paper/outputs/paper_numbers.csv ch_highest

20. **P178 (¶179), §3.2.** "…highest SNR in **every participant**. The greater…" → **"5 of 6 participants"**
   - Reason: CH is highest in 5/6 participants (10/12 nights); CRE is higher on both nights of participant 4.
   - Source: paper/outputs/paper_numbers.csv ch_highest; writeup/figures/signal_validation/inband_snr.py

21. **P178 (¶179), §3.2.** "…constant electronic noise floor across sessions.** The relative contributions of respiratory and cardiac activity also varied across recordings: S6N1 was predominantly respiratory (48% respiratory versus 8% cardiac power), whereas S3N1 showed a stronger cardiac contribution (48% cardiac power).**…" → **"(deleted)"**
   - Reason: Sentence cut: its percentages came from the earlier calculation (3 nights, CLE−CRE, 0.05–10 Hz denominator) and it names no channel, so it contradicts the corrected range stated two sentences earlier.
   - Source: paper/outputs/paper_numbers.csv resp_frac_S6N1, card_frac_S6N1, card_frac_S3N1

22. **P152 (¶153), §2.4 Methods.** "…In each time column, **a frequency-median-filtered spectrum was subtracted** to reduce the 1/f…" → **"power in a plain spectrogram (30-s windows for respiration and 15-s windows for cardiac activity, 50% overlap) was divided by its in-band column median"**
   - Reason: The median-filter subtraction belongs to the display spectrogram only; the tracker emission is log(PSD / in-band column median) of a plain spectrogram, 30 s / 15 s windows, 50% overlap.
   - Source: paper/outputs/paper_numbers.csv vit_method; analysis/slow_wave/ridge_overlay_tune.py:track_single_ridge

23. **P152 (¶153), §2.4 Methods.** "…column-median background **by the prespecified detection criterion**. These trajectories…" → **"by more than a factor of two (3 dB)"**
   - Reason: The confidence rule was never stated: tracked bin > 2× the in-band column median.
   - Source: paper/outputs/paper_numbers.csv vit_conf_rule; track_single_ridge

24. **P159 (¶160), §2.7 Methods.** "…random NREM draws **per event** and were interpreted…" → **"per session"**
   - Reason: The null draws 200 windows per SESSION (event durations cycled), not per event.
   - Source: paper/outputs/paper_numbers.csv null_draws; harmonic_ladder_overlay.py → ladder_stage_relationship.py

25. **P197 (¶198), §3.4.** "…Across the 12 recordings, **351–2,134** N2 spindles…" → **"351–2,130"**
   - Reason: K-complex marks (40 in N2) are no longer analysed as spindles.
   - Source: paper/outputs/paper_numbers.csv spin_n_range

26. **P197 (¶198), §3.4.** "…per session (**14,305** total)…" → **"14,265"**
   - Reason: As above, total.
   - Source: paper/outputs/paper_numbers.csv spin_n_total

27. **P198 (¶199), §3.4.** "… channels and **0.55** dB in the CH channel…" → **"0.54"**
   - Reason: CH low-band change, mean of the 12 recording means = 0.544 (0.55 was a 13-row mean that included the POOLED row). CLE/CRE 0.45–0.49 (0.448–0.492) and EEG sigma 3.45 (3.445) MATCH as recording means and are kept.
   - Source: paper/outputs/paper_numbers.csv spin_low_ch_recmean, spin_low_temple_recmean, spin_eeg_sigma_recmean

28. **P198 (¶199), §3.4.** "…response was observed in all 12 recordings**(insert)**, was absent…" → **" on CH and CLE−CRE and 11 of 12 on CLE and CRE"**
   - Reason: Positive mean low-band change in 12/12 recordings on CH and CLE−CRE but 11/12 on CLE and on CRE.
   - Source: paper/outputs/paper_numbers.csv spin_all12_CLE, spin_all12_CRE, spin_all12_CLE-CRE, spin_all12_CH

29. **P201 (¶202), Fig. 6 caption.** "…largest increase in CH (**0.55** dB). (B)…" → **"0.54"**
   - Reason: Same quantity as the main text (CH, 12-recording mean 0.544 dB; the note says the curve averaged over |t| < 1 s equals this). The single-sample curve peak would be 0.63 dB.
   - Source: paper/outputs/paper_numbers.csv fig6a_ch_peak, spin_low_ch_recmean

30. **P201 (¶202), Fig. 6 caption.** "…increases by an average of **0.55** dB, whereas…" → **"0.59"**
   - Reason: Panel B is the pooled per-spindle distribution: CH pooled mean 0.587 dB.
   - Source: paper/outputs/paper_numbers.csv fig6b_ch_low

31. **P201 (¶202), Fig. 6 caption.** "…changes by only **0.02** dB. (C)…" → **"0.03"**
   - Reason: Panel B pooled CH sigma mean 0.027 dB.
   - Source: paper/outputs/paper_numbers.csv fig6b_ch_sigma

32. **P201 (¶202), Fig. 6 caption.** "…consistent increase across all 12 recordings**(insert)**. …" → **" on CH and CLE−CRE and 11 of 12 on CLE and CRE"**
   - Reason: Panel C: CLE 11/12, CRE 11/12, CLE−CRE 12/12, CH 12/12.
   - Source: paper/outputs/paper_numbers.csv fig6c_all12

33. **P205 (¶206), §3.5.** "…1 to 99 per recording; **three** recordings contained fewer…" → **"four"**
   - Reason: Four recordings have < 10 onsets: S1N2 9, S4N2 4, S5N1 1, S5N2 6.
   - Source: paper/outputs/paper_numbers.csv n_rec_lt10; analysis/delta_onset/delta_onset_detection.py

34. **P207 (¶208), §3.5.** "…followed rather than preceded delta-burst onset. **With strictly causal filtering, the pre-onset baseline remained flat, SEC–EEG cross-correlation peaked at zero lag, and pre-onset SEC power did not predict an impending onset (area under the curve, 0.42–0.56). Zero-phase filtering produced an apparent pre-onset increase in the 0–0.5 Hz band, but this resulted from backward leakage of the large post-onset response. With causal filtering, the real-minus-control difference during the final 3 s before onset decreased from 0.35–0.41 standardized units, positive in all six participants, to approximately zero and positive in only two to three participants.**…" → **"With causal filtering, which uses only past data, SEC power was flat before onset and did not predict an upcoming onset (area under the curve, 0.37–0.51). Zero-phase filtering, which also uses later data, showed a small apparent rise before onset, but this was the large post-onset response spreading backward in time: switching to causal filtering reduced the pre-onset difference from the random-time control from 0.35–0.41 standardized units to approximately zero."**
   - Reason: Rewritten for clarity and corrected: the old text attributed the zero-lag cross-correlation and the 0.42–0.56 AUC to causal filtering, but both came from the zero-phase pipeline. With causal envelopes the AUC is 0.37–0.51. The cross-correlation and the per-participant counts are dropped as unnecessary detail.
   - Source: paper/outputs/paper_numbers.csv auc_causal, auc_zerophase, xcorr_lag_zerophase, lowband_zp_pos_*; analysis/delta_onset/lowband_precursor_check.py, delta_cap_precursor.py

35. **P208 (¶209), §3.5.** "…Moreover, K-complexes**, which constituted most detected onsets, were** frequently accompanied…" → **" are"**
   - Reason: Only 9 of 344 onsets fall within ±5 s of a scored K-complex; the claim that K-complexes made up most onsets is not supported by the scoring.
   - Source: paper/outputs/paper_numbers.csv onsets_on_kcomplex

36. **P208 (¶209), §3.5.** "…frequently accompanied by autonomic activation**(insert)**.…" → **", although only 9 of the 344 onsets fell within 5 s of a scored K-complex"**
   - Reason: As above: the computed fact replaces the unsupported "most onsets" claim.
   - Source: paper/outputs/paper_numbers.csv onsets_on_kcomplex, kc_with_onset

### F. Harmonic combs (adaptive detector; Figs. 8-9 pictures replaced, untracked)

37. **P159 (¶160), §2.7 Methods.** "…independently in each raw SEC channel **using a background-subtracted 0-3 Hz spectrogram. An episode required at least three consecutive integer-related harmonic peaks, each at least 5 dB above the local spectral floor.** Horizontal…" → **"using a background-subtracted 0–5 Hz spectrogram. Each 30-s window was scored by the mean height above the local spectral floor of the first four multiples of the best-fitting fundamental (0.15–0.55 Hz), with heights below the floor counted as zero. The score was smoothed over 1.5 min and expressed as a robust z-score within each recording and channel. An episode began where z reached 2.5, extended while z stayed above 1.0, bridged gaps of up to 7 min, and had to last at least 3 min; these settings were chosen by visual inspection of all recordings."**
   - Reason: Detector replaced: the fixed "three peaks each 5 dB" rule missed visible combs and split single combs in two. The adaptive score was tuned by eye on all 12 nights x 3 channels (analysis/slow_wave/comb_tune.py). Spectrogram now 0–5 Hz.
   - Source: paper/stages/s06_harmonic_comb.py: comb_score, _hysteresis, detect_channel

38. **P216 (¶217), Fig. 8 caption.** "…background-enhanced spectrogram (**0–3** Hz)…" → **"0–5"**
   - Reason: Figure redrawn to 5 Hz with the adaptive detector; CH still has two events, both in N2.
   - Source: paper/outputs/paper_numbers.csv fig8_events, fig8_stage; paper/outputs/figures/s06_harmonic_comb/fig8_S6N1_CH.png

39. **P218 (¶219), §3.6.** "…**Twenty-two events were identified across nine sessions** from all six…" → **"Forty-five events were identified across all 12 sessions"**
   - Reason: 45 merged events, 12 of 12 nights.
   - Source: paper/outputs/paper_numbers.csv n_events, n_sessions

40. **P218 (¶219), §3.6.** "…One representative event contained bands at **0.15, 0.28, 0.42, 0.68, and 0.95** Hz (Fig. …" → **"0.23, 0.50, and 1.10"**
   - Reason: The old example had no source. Replaced by the sustained bands of the first event in Fig. 8 (S6N1, CH, 2.27–2.73 h).
   - Source: ladder_bands.csv (S6N1 CH); paper/outputs/paper_numbers.csv example_bands

41. **P218 (¶219), §3.6.** "…consolidated NREM sleep: **19 of 22 (86%) occurred in N2, while one occurred in each of N1, N3, and wakefulness**.…" → **"35 of 45 (78%) occurred in N2, four in N3, three in wakefulness, two in N1, and one in REM"**
   - Reason: Dominant stage of each event.
   - Source: paper/outputs/paper_numbers.csv n_n2, pct_n2, n_other; ladder_events.csv

42. **P219 (¶220), §3.6.** "…**Event-aligned stage occupancy showed an N2 probability of approximately 0.9 at onset (Fig. 10a), indicating that these events were associated with N2 rather than N3 slow wave activity. The events also showed a consistent temporal relationship with preceding REM sleep. REM occupied an average of 5.9% of the 30 min before event onset, approximately 3.5 times the matched random-NREM value of 1.7%. In contrast, REM occupied only 0.7% of the 30 min after event offset, compared with 3.1% in the control. Event-aligned analysis showed elevated REM occupancy approximately 30–8 min before onset and little REM thereafter (Fig. 10b). The nearest REM epoch occurred a median of 30 min before an event and 51 min afterward; REM was closer before than after the event in five of six participants. N1 occupancy also increased during the 10 min preceding onset, consistent with a REM-to-N1-to-N2 transition.**…" → **"Event-aligned stage occupancy showed an N2 probability of 0.76 at onset, peaking at 0.82 about 4 min later (Fig. 9a), indicating that these events were associated with N2 rather than N3 slow wave activity. The relationship with REM sleep was weak and inconsistent. REM occupied an average of 3.5% of the 30 min before event onset, approximately twice the matched random-NREM value of 1.8%, and 1.3% of the 30 min after event offset, compared with 3.3% in the control (Fig. 9b). However, the nearest REM epoch occurred a median of 85 min before an event and 44 min afterward, and REM was closer before than after the event in only two of six participants. N1 occupancy was similar in the 10 min before onset and in the 20 min before that (0.15 in both)."**
   - Reason: Rewritten from the new results. With 45 events the pooled pre-onset REM excess is smaller (x2.0, was x3.4) and REM is nearer AFTER the event in 4 of 6 participants, so the claimed "consistent temporal relationship with preceding REM" no longer holds. Figure refs 10a/10b -> 9a/9b. N1 pre-onset 0.154 vs 0.151: no increase.
   - Source: paper/outputs/paper_numbers.csv n2_onset, rem_pre, rem_pre_null, rem_ratio, rem_post, rem_post_null, rem_med_before, rem_med_after, rem_side, n1_pre; ladder_onset_occupancy.csv, ladder_rem_side_by_subject.csv

43. **P221 (¶222), Fig. 9 caption.** "…distribution of **22** harmonic-comb events…" → **"45"**
   - Reason: Event count.
   - Source: paper/outputs/paper_numbers.csv n_events

44. **P221 (¶222), Fig. 9 caption.** "…REM occupancy surrounding event onset**, showing elevated occupancy approximately 30–8 min before onset and little REM thereafter**.…" → **"; REM was slightly more frequent before onset than after, but not consistently across participants"**
   - Reason: As §3.6.
   - Source: paper/outputs/paper_numbers.csv rem_pre, rem_post, rem_side

45. **P223 (¶224), §3.6.** "…**Thus, harmonic-comb events generally emerged during consolidated N2 sleep approximately 10–30 min after REM and were not typically followed by REM. Given the small sample of 22 events from six participants, this pattern was considered exploratory rather than confirmatory. The events may represent nonsinusoidal, quasi-periodic mechanical or hemodynamic activity associated with stable post-REM NREM sleep rather than cortical slow-wave activity.**…" → **"Thus, harmonic-comb events occurred mainly during consolidated N2 sleep. The pooled REM occupancy suggested a link with preceding REM, but this was not consistent across participants. Given 45 events from six participants, these patterns were considered exploratory rather than confirmatory. The events may represent nonsinusoidal, quasi-periodic mechanical or hemodynamic activity associated with stable NREM sleep rather than cortical slow-wave activity."**
   - Reason: Conclusion of §3.6 follows the new results: N2 stands, post-REM timing does not.
   - Source: paper/outputs/paper_numbers.csv rem_side, rem_10_30

### E. Fig. 2 caption

46. **P173 (¶174), Fig. 2 caption.** "…four male participants aged **(insert)**54 years…" → **"25, 37, "**
   - Reason: Panels are (a) S6N2 25 y, (b) S3N2 37 y, (c) S4N2 54 y, (d) S2N2 66 y; Table 1 lists all four as M, so "male" stays.
   - Source: paper/outputs/paper_numbers.csv fig2_caption; analysis/mean_value/fig2_overnight_panels.py PANELS; Table 1

47. **P173 (¶174), Fig. 2 caption.** "…four male participants aged 54**(insert)** years. Each panel…" → **" and 66"**
   - Reason: As above.
   - Source: paper/outputs/paper_numbers.csv fig2_caption

## Left unchanged: for the professor

Out of scope by instruction (written by the professor or awaiting a decision):
- **§3.8 (P231–P237)**, including "(Fig. 12)" in P232, which should read Fig. 11 (associations are captioned Fig. 11), "Fig. Sxx", and the R-value signs (PAPER_NUMBERS DIFF s09: R² 0.10→0.005; spontaneous +0.48→−0.48; PSQI 0.56→−0.57; 0.68→0.48; median variance 0.48→−0.56; age 0.63→−0.63).
- **Fig. 10 and Fig. 11 captions** ("absolute amplitude" is a % of night with CLE−CRE variance > 10 fF²).
- **SWS-association claims** in Discussion P247 (¶248) and Conclusion P258 (¶259): against PSG N3 the mean-SEC-area association is R = −0.17 (n = 12).
- **Figs. 3–4 and the slope text P179–P186** (co-author analysis; Fig. 4 caption "CHECK THE Y AXIS").
- **§3.3 low-band ridges**: P189 "9 of 12" (0 of 12 by prominence), "around 0.1 Hz", 0.20–0.25 Hz / 0.9–1.8 Hz; P193 (67 ridges, 2–8/night, 0.05–0.08 Hz, S1N1, "1 hour or longer"; motion-canceller filter artifact); P194; Fig. 5 caption low band; Methods P157/P158.
- **Imbalance sentences P175–P176** (¶176–177).

Supplementary figures cited in V11 that no longer exist in supplement V4 (sentence left as is):
- P175: "(Supporting information, Fig. S4)" for the head-position result (V4 S4 is the rate pipeline).
- P176: "Fig. S5a", "Fig. S5b", "Fig. S5c" (integrated magnitude; V4 S5 is k by channel; "S5c" is also UNTRACED).
- P189 and P193: "Fig. S10" for ridges (V4 S10 is the movement-corrected S4N2 night) — also inside out-of-scope §3.3.

## In scope but not applied (logged instead)

- **§2.2 P140, "9,319" epochs.** 9,319 is the rate grid and equals the Table 1 column sum; the §3.1 analysis grid has 9,312 (PSG epochs straddling recording edges dropped). Saying which grid is counted needs a new clause, not a number swap, so it is left; the two grids differ by 7 epochs.
- **§3.2 P178, S6N1/S3N1 example** ("48% respiratory versus 8% cardiac", "48% cardiac"). Under the sub-5-Hz denominator the CLE−CRE values are 20% / 3% / 42%, but on the raw channels they are CH 19/3, CLE 32/12, CRE 20/3 (S6N1) and CH 34, CLE 39, CRE 28 (S3N1). The sentence names no channel, so the replacement is ambiguous; V11_HIGHLIGHTS C6 proposes deleting the sentence.
- **§3.2 P178, "nearly constant electronic noise floor" / "greater session-to-session variability in CLE and CRE".** DIFF: the 10–50 Hz floor moves 26.2 / 3.1 / 13.6 dB (CH/CLE/CRE) between nights, and CLE varies less than CH (SD 6.4 / 5.2 / 8.3 dB). Needs a rewritten sentence (V11_HIGHLIGHTS C6 text), not a number swap.
- **Methods P153 (¶154), "sliding-window Welch"**: DIFF says one full-night Welch PSD per channel (30-s segments, 50% overlap). Not in the listed Methods paragraphs (¶153 Viterbi, ¶160, ¶163); candidate for the next pass.
- **Methods P162 (¶163) spindle method.** "low-frequency (0-3 Hz)": code drops the f = 0 bin (bins 0.78/1.56/2.34 Hz used), computed "0.1–3"; changing it would also need §3.4 and the Fig. 6 caption, and the band edges are not stated precisely. "count-matched baseline windows … spindle-free N2 windows": the detection control is all N2 points ≥ 3 s from a spindle (not count-matched) and the ERSP control is count-matched but not spindle-free, so no one-phrase replacement. "Subject-level averages … before the group average": the spindle numbers are recording/pooled means (identical to participant means here because each participant has two recordings); the same sentence also covers delta bursts, where it is true.
- **§3.5 onset count entering Fig. 7 and the arousal control**: 339, not 344 (S4N2 and S5N1 skipped, < 5 onsets). Not in the listed §3.5 items; Fig. 7 caption "For the complete onset set" is affected.
- **§3.5 P205, "Most were isolated N2 slow waves or K-complexes"**: 340 of 344 onsets are scored N2, but "K-complexes" has the same 9/344 problem as P208. Not a DIFF row; left.
- **§3.7 P229 R² values** (0.63/0.71/0.68 → 0.637/0.710/0.855) and "Significant" (p = 0.057/0.035/0.008, n = 6). §3.7 was in scope only for the figure cross-reference.
- **§3.9 K-complex latencies** (causal 3.5/5.2/5.5 s vs text 3.6/3.7/4.3 s) and "count-matched" null (it is 20 × marks, min 200). Not in scope.
