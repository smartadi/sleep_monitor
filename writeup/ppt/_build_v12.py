"""
Build manuscript V12 from V11. The edits are made as tracked changes (so the build can
verify them against V11) and then ACCEPTED: the written V12 is clean text, no red or
strike-through (author request 2026-10-05). V12_CHANGES.md is the record of every edit.

Every edit is a real tracked change (w:del / w:ins, author "Aditya Deole",
2026-10-05) made INSIDE the existing runs: the run(s) holding the target text
are split at the match boundaries, the target pieces are wrapped in w:del
(w:t -> w:delText) and the new text goes in a w:ins run that copies the rPr of
the run being edited. No paragraph is collapsed, so the professor's yellow
highlights survive on all untouched text (the V11 build destroyed them by
collapsing runs; see writeup/edits/V11_HIGHLIGHTS.md D3).

Each edit is (pre, old, post, new): pre+old+post must occur exactly once in the
whole body (original text, i.e. ignoring earlier insertions), or the build
stops. old == "" is a pure insertion at that point.

Validation (run every build):
  * zip integrity + every XML part parses
  * "reject all changes" view == V11 text, paragraph by paragraph, and the
    highlighted characters in that view == V11's highlighted characters
  * highlighted runs with non-empty visible text, before vs after
  * every new text is present in the "accept all changes" view

Sources of truth: paper/outputs/paper_numbers.csv (DIFF rows), paper/PROVENANCE.md,
writeup/edits/V11_HIGHLIGHTS.md, supplementary V4 captions.

Reads   writeup/review/final/CAP_sleep_mask_manuscript_V11.docx   (not modified)
Writes  writeup/review/final/CAP_sleep_mask_manuscript_V12.docx
        writeup/edits/V12_CHANGES.md

Usage
-----
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe writeup/ppt/_build_v12.py
"""

from __future__ import annotations

import copy
import sys
import zipfile
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / 'writeup' / 'review' / 'final'
SRC = FINAL / 'CAP_sleep_mask_manuscript_V11.docx'
OUT = FINAL / 'CAP_sleep_mask_manuscript_V12.docx'
LOG = ROOT / 'writeup' / 'edits' / 'V12_CHANGES.md'
FIGDIR = ROOT / 'paper' / 'outputs' / 'figures' / 's06_harmonic_comb'
# caption start -> new picture (the picture is the nearest earlier paragraph with one).
# Not a tracked change: Word does not track picture content.
NEW_FIGS = {'Fig. 8 Representative harmonic-comb': FIGDIR / 'fig8_S6N1_CH.png',
            'Fig. 9 Sleep-stage distribution': FIGDIR / 'fig9_stage_occupancy.png',
            'Figure 12. SEC response at technologist-scored K-complexes':
                ROOT / 'paper' / 'outputs' / 'figures' / 's08_delta_kcomplex' / 'fig12_kcomplex_response.png'}

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
W = '{%s}' % W_NS
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
AUTHOR = 'Aditya Deole'
DATE = '2026-10-05T00:00:00Z'

PN = 'paper/outputs/paper_numbers.csv'
RATE_SRC = (PN + ' (s04_rates); paper/outputs/tables/s04_rates/heldout_table.csv; '
            'reports/rates/k_by_channel.csv')
MINUS = '−'   # the document's minus sign in CLE−CRE

# ─────────────────────────────────────────────────────────────────────────────
# Edits. (group, section, pre, old, post, new, reason, source)
# ─────────────────────────────────────────────────────────────────────────────
EDITS = [
    # ── A. Rate numbers: Discussion ¶253/¶254, Conclusion ¶259 ───────────────
    ('A', 'Discussion', 'breaths/min and ', '1.56 beats/min', ', respectively; corresponding',
     '1.19 beats/min',
     'Nightly cardiac error with the ECG R-peak reference (Pleth only on S5N1, S6N2): 1.194.',
     PN + ' night_err_card_disc; ' + RATE_SRC),
    ('A', 'Discussion', 'errors were 1.79 breaths/min and ', '3.41 beats/min', '. These findings',
     '3.09 beats/min (Supporting information, Figs. S4–S8)',
     'Epoch-level cardiac error with the ECG reference: 3.094. Cites the supplementary '
     'rate figures (V4: S4 pipeline, S5 k by channel, S6 counts/per-epoch k, S7 all twelve '
     'recordings, S8 per-epoch k spread), since the rate method and results now live there.',
     PN + ' epoch_err_card; supplementary V4 captions S4–S8'),
    ('A', 'Discussion', 'beats/min. ', 'These findings demonstrate calibrated agreement with PSG. ',
     'Cardiac measurements', '',
     'Sentence deleted: k is fitted on the same night it is evaluated on, so the per-night '
     'error cannot demonstrate agreement with PSG.',
     'task instruction; V11_HIGHLIGHTS.md "Notes" (lost V7 fix)'),
    ('A', 'Discussion', 'nights (median, ', '1.96', '; interquartile range', '2.00',
     'Cardiac k (CRE) median with the ECG reference.', PN + ' k_card; reports/rates/k_by_channel.csv'),
    ('A', 'Discussion', 'interquartile range, ', '1.77–2.01', '), suggesting', '1.95–2.07',
     'Cardiac k IQR with the ECG reference.', PN + ' k_card'),
    ('A', 'Discussion', 'per heartbeat. ', 'R-peak–triggered averaging yielded',
     ' a median of 2.02', 'Counting detected peaks per ECG R-peak gave',
     'The 2.02 is a count ratio (CRE peaks ÷ ECG R-peaks over the asleep night), '
     'not R-peak-triggered averaging. Value 2.018 unchanged.',
     PN + ' peaks_per_beat; analysis/rates/peaks_per_beat.py; PROVENANCE.md (rates)'),
    ('A', 'Discussion', 'nightly mean rates (', '14.4–16.8', ' breaths/min)', '14.2–17.0',
     'Respiratory reference range across nights from the rerun pipeline: 14.21–16.98.',
     PN + ' resp_ref_range'),
    ('A', 'Discussion', 'achieved an error of 1.20 breaths/min', ' (Table 3)', ', underscoring', '',
     'V11 has no Table 3. The 1.20 itself MATCHes.', PN + ' nosensor_resp; PROVENANCE.md cross-reference list'),
    ('A', 'Conclusion', '0.24 breaths/min and ', '1.56 beats/min', ', respectively; reliable',
     '1.19 beats/min',
     'Nightly cardiac error with the ECG reference (same quantity as Discussion).',
     PN + ' night_err_card_conc'),

    # ── B. Supplementary cross-references (V4 numbering) ─────────────────────
    ('B', '§3.2', 'Supporting Information, Fig. ', 'S6', '), and C', 'S2',
     'SNR figure is Fig. S2 in supplement V4 (old S6 / S3).',
     'supplementary V4 caption "Figure S2. Physiological-band SNR"; ' + PN + ' snr_fig_ref'),

    # ── C. Main-figure cross-references (checked against V11 captions) ──────
    ('B', '§3.6', '', 'Supporting information Fig. S11', ').',
     'Supporting information, Figs. S16–S18',
     'Old S11 no longer exists; the harmonic-comb events are now supplementary section S6 '
     '(S16 all 45 events over the stages, S17 four events up close, S18 event properties).',
     'supplementary V4 section S6; analysis/slow_wave/supp_comb_figures.py'),
    ('C', '§3.4', ' channel (Fig. ', '7', '). This average', '6',
     'Spindle figure is captioned "Fig. 6. Mechanical SEC responses associated with sleep spindles".',
     'V11 caption P201; PROVENANCE.md cross-reference list'),
    ('C', '§3.5', 'channel–band combination (Fig. ', '8', ').', '7',
     'Delta-onset figure is captioned "Fig. 7 SEC band-power changes surrounding delta-burst onset".',
     'V11 caption P211; PROVENANCE.md'),
    ('C', '§3.6', 'and 0.95 Hz (Fig. ', '9', ', Supporting information', '8',
     'Comb example is captioned "Fig. 8 Representative harmonic-comb events".',
     'V11 caption P216; PROVENANCE.md'),
    ('C', '§3.7', 'two separate nights (Fig. ', '11', '). Significant', '10',
     'Reproducibility figure is captioned "Fig. 10. Night-to-night reproducibility".',
     'V11 caption P227; ' + PN + ' fig_ref_230'),

    # ── D. Factual corrections from DIFF rows ───────────────────────────────
    ('D', '§3.1', 'exceeded 1000 fF in the ', 'three', ' most mobile recordings', 'five',
     'Five recordings exceed 1000 fF within-night CH range (S2N1, S2N2, S4N1, S6N1, S6N2).',
     PN + ' ch_gt_1000; analysis/mean_value/mean_centred_traces.py'),
    ('D', '§3.2', 'accounted for ', '29–48%', ' of the total power below 5 Hz', '9–59%',
     'All 12 nights × CH/CLE/CRE, denominator 0 < f ≤ 5 Hz (the text already states '
     '"power below 5 Hz"). 29–48% was 3 nights of CLE−CRE with a 0.05–10 Hz denominator.',
     PN + ' resp_frac_range; writeup/figures/signal_validation/generate_band_energy.py'),
    ('D', '§3.2', 'contributed an additional ', '8–48%', '. Relative to', '3–60%',
     'As above, cardiac band share.', PN + ' card_frac_range'),
    ('D', '§3.2', 'and CH ', 'consistently ', 'yielded the highest SNR', '',
     '"consistently" no longer true (see next edit).', PN + ' ch_highest'),
    ('D', '§3.2', 'highest SNR in ', 'every participant', '. The greater', '5 of 6 participants',
     'CH is highest in 5/6 participants (10/12 nights); CRE is higher on both nights of participant 4.',
     PN + ' ch_highest; writeup/figures/signal_validation/inband_snr.py'),
    ('D', '§3.2', 'constant electronic noise floor across sessions.', ' The relative contributions of respiratory and cardiac activity also varied across recordings: S6N1 was predominantly respiratory (48% respiratory versus 8% cardiac power), whereas S3N1 showed a stronger cardiac contribution (48% cardiac power).', '', '',
     'Sentence cut: its percentages came from the earlier calculation (3 nights, CLE−CRE, '
     '0.05–10 Hz denominator) and it names no channel, so it contradicts the corrected range '
     'stated two sentences earlier.',
     PN + ' resp_frac_S6N1, card_frac_S6N1, card_frac_S3N1'),
    ('D', '§2.4 Methods', 'In each time column, ', 'a frequency-median-filtered spectrum was subtracted',
     ' to reduce the 1/f',
     'power in a plain spectrogram (30-s windows for respiration and 15-s windows for cardiac '
     'activity, 50% overlap) was divided by its in-band column median',
     'The median-filter subtraction belongs to the display spectrogram only; the tracker emission '
     'is log(PSD / in-band column median) of a plain spectrogram, 30 s / 15 s windows, 50% overlap.',
     PN + ' vit_method; analysis/slow_wave/ridge_overlay_tune.py:track_single_ridge'),
    ('D', '§2.4 Methods', 'column-median background ', 'by the prespecified detection criterion',
     '. These trajectories', 'by more than a factor of two (3 dB)',
     'The confidence rule was never stated: tracked bin > 2× the in-band column median.',
     PN + ' vit_conf_rule; track_single_ridge'),
    ('D', '§2.7 Methods', 'random NREM draws ', 'per event', ' and were interpreted', 'per session',
     'The null draws 200 windows per SESSION (event durations cycled), not per event.',
     PN + ' null_draws; harmonic_ladder_overlay.py → ladder_stage_relationship.py'),
    ('D', '§3.4', 'Across the 12 recordings, ', '351–2,134', ' N2 spindles', '351–2,130',
     'K-complex marks (40 in N2) are no longer analysed as spindles.', PN + ' spin_n_range'),
    ('D', '§3.4', 'per session (', '14,305', ' total)', '14,265',
     'As above, total.', PN + ' spin_n_total'),
    ('D', '§3.4', ' channels and ', '0.55', ' dB in the CH channel', '0.54',
     'CH low-band change, mean of the 12 recording means = 0.544 (0.55 was a 13-row mean that '
     'included the POOLED row). CLE/CRE 0.45–0.49 (0.448–0.492) and EEG sigma 3.45 '
     '(3.445) MATCH as recording means and are kept.',
     PN + ' spin_low_ch_recmean, spin_low_temple_recmean, spin_eeg_sigma_recmean'),
    ('D', '§3.4', 'response was observed in all 12 recordings', '', ', was absent',
     ' on CH and CLE' + MINUS + 'CRE and 11 of 12 on CLE and CRE',
     'Positive mean low-band change in 12/12 recordings on CH and CLE−CRE but 11/12 on CLE and on CRE.',
     PN + ' spin_all12_CLE, spin_all12_CRE, spin_all12_CLE-CRE, spin_all12_CH'),
    ('D', 'Fig. 6 caption', 'largest increase in CH (', '0.55', ' dB). (B)', '0.54',
     'Same quantity as the main text (CH, 12-recording mean 0.544 dB; the note says the curve '
     'averaged over |t| < 1 s equals this). The single-sample curve peak would be 0.63 dB.',
     PN + ' fig6a_ch_peak, spin_low_ch_recmean'),
    ('D', 'Fig. 6 caption', 'increases by an average of ', '0.55', ' dB, whereas', '0.59',
     'Panel B is the pooled per-spindle distribution: CH pooled mean 0.587 dB.',
     PN + ' fig6b_ch_low'),
    ('D', 'Fig. 6 caption', 'changes by only ', '0.02', ' dB. (C)', '0.03',
     'Panel B pooled CH sigma mean 0.027 dB.', PN + ' fig6b_ch_sigma'),
    ('D', 'Fig. 6 caption', 'consistent increase across all 12 recordings', '', '. ',
     ' on CH and CLE' + MINUS + 'CRE and 11 of 12 on CLE and CRE',
     'Panel C: CLE 11/12, CRE 11/12, CLE−CRE 12/12, CH 12/12.', PN + ' fig6c_all12'),
    ('D', '§3.5', '1 to 99 per recording; ', 'three', ' recordings contained fewer', 'four',
     'Four recordings have < 10 onsets: S1N2 9, S4N2 4, S5N1 1, S5N2 6.',
     PN + ' n_rec_lt10; analysis/delta_onset/delta_onset_detection.py'),
    ('D', '§3.5', 'followed rather than preceded delta-burst onset. ', 'With strictly causal filtering, the pre-onset baseline remained flat, SEC–EEG cross-correlation peaked at zero lag, and pre-onset SEC power did not predict an impending onset (area under the curve, 0.42–0.56). Zero-phase filtering produced an apparent pre-onset increase in the 0–0.5 Hz band, but this resulted from backward leakage of the large post-onset response. With causal filtering, the real-minus-control difference during the final 3 s before onset decreased from 0.35–0.41 standardized units, positive in all six participants, to approximately zero and positive in only two to three participants.', '',
     'With causal filtering, which uses only past data, SEC power was flat before onset and did not predict an upcoming onset (area under the curve, 0.37–0.51). Zero-phase filtering, which also uses later data, showed a small apparent rise before onset, but this was the large post-onset response spreading backward in time: switching to causal filtering reduced the pre-onset difference from the random-time control from 0.35–0.41 standardized units to approximately zero.',
     'Rewritten for clarity and corrected: the old text attributed the zero-lag '
     'cross-correlation and the 0.42–0.56 AUC to causal filtering, but both came from the '
     'zero-phase pipeline. With causal envelopes the AUC is 0.37–0.51. The cross-correlation '
     'and the per-participant counts are dropped as unnecessary detail.',
     PN + ' auc_causal, auc_zerophase, xcorr_lag_zerophase, lowband_zp_pos_*; '
     'analysis/delta_onset/lowband_precursor_check.py, delta_cap_precursor.py'),
    ('D', '§3.5', 'Moreover, K-complexes', ', which constituted most detected onsets, were',
     ' frequently accompanied', ' are',
     'Only 9 of 344 onsets fall within ±5 s of a scored K-complex; the claim that K-complexes '
     'made up most onsets is not supported by the scoring.',
     PN + ' onsets_on_kcomplex'),
    ('D', '§3.5', 'frequently accompanied by autonomic activation', '', '.',
     '; however, only 9 of the 344 onsets fell within 5 s of a scored K-complex, so '
     'K-complexes cannot account for most onsets',
     'As above: the computed fact replaces the unsupported "most onsets" claim.',
     PN + ' onsets_on_kcomplex, kc_with_onset'),
    # ── G. Consistency (review pass 2026-10-05) ─────────────────────────────
    ('G', '§3.5', 'Most were isolated N2 slow waves', ' or K-complexes', ' because sustained', '',
     'Only 9 of 344 onsets are within 5 s of a scored K-complex (340 of 344 are N2); '
     'same fix as the §3.5 K-complex sentence.', PN + ' onsets_on_kcomplex'),
    ('G', 'Discussion', 'onsets reflected N2 slow waves', ' or K-complexes', ' rather than', '',
     'As §3.5.', PN + ' onsets_on_kcomplex'),
    ('G', '§2.8 Methods', 'assessed from the causal pre-onset trajectory, ',
     'SEC-to-EEG cross-correlation, ', 'and the area', '',
     'The cross-correlation result was dropped from §3.5 (it came from the zero-phase '
     'pipeline), so Methods no longer announces it.', PN + ' xcorr_lag_zerophase'),

    # ── F. Harmonic combs: adaptive detector (paper/stages/s06_harmonic_comb.py) ─
    ('F', '§2.7 Methods', 'independently in each raw SEC channel ', 'using a background-subtracted 0-3 Hz spectrogram. An episode required at least three consecutive integer-related harmonic peaks, each at least 5 dB above the local spectral floor.', ' Horizontal',
     'using a background-subtracted 0–5 Hz spectrogram. Each 30-s window was scored by the mean height above the local spectral floor of the first four multiples of the best-fitting fundamental (0.15–0.55 Hz), with heights below the floor counted as zero. The score was smoothed over 1.5 min and expressed as a robust z-score within each recording and channel. An episode began where z reached 2.5, extended while z stayed above 1.0, bridged gaps of up to 7 min, and had to last at least 3 min; these settings were chosen by visual inspection of all recordings.',
     'Detector replaced: the fixed "three peaks each 5 dB" rule missed visible combs and split '
     'single combs in two. The adaptive score was tuned by eye on all 12 nights x 3 channels '
     '(analysis/slow_wave/comb_tune.py). Spectrogram now 0–5 Hz.',
     'paper/stages/s06_harmonic_comb.py: comb_score, _hysteresis, detect_channel'),
    ('F', 'Fig. 8 caption', 'background-enhanced spectrogram (', '0–3', ' Hz)', '0–5',
     'Figure redrawn to 5 Hz with the adaptive detector; CH still has two events, both in N2.',
     PN + ' fig8_events, fig8_stage; paper/outputs/figures/s06_harmonic_comb/fig8_S6N1_CH.png'),
    ('F', '§3.6', '', 'Twenty-two events were identified across nine sessions',
     ' from all six', 'Forty-five events were identified across all 12 sessions',
     '45 merged events, 12 of 12 nights.', PN + ' n_events, n_sessions'),
    ('F', '§3.6', 'One representative event contained bands at ', '0.15, 0.28, 0.42, 0.68, and 0.95',
     ' Hz (Fig. ', '0.23, 0.50, and 1.10',
     'The old example had no source. Replaced by the sustained bands of the first event in '
     'Fig. 8 (S6N1, CH, 2.27–2.73 h).', 'ladder_bands.csv (S6N1 CH); ' + PN + ' example_bands'),
    ('F', '§3.6', 'consolidated NREM sleep: ',
     '19 of 22 (86%) occurred in N2, while one occurred in each of N1, N3, and wakefulness',
     '.', '35 of 45 (78%) occurred in N2, four in N3, three in wakefulness, two in N1, and one in REM',
     'Dominant stage of each event.', PN + ' n_n2, pct_n2, n_other; ladder_events.csv'),
    ('F', '§3.6', '', 'Event-aligned stage occupancy showed an N2 probability of approximately 0.9 at onset (Fig. 10a), indicating that these events were associated with N2 rather than N3 slow wave activity. The events also showed a consistent temporal relationship with preceding REM sleep. REM occupied an average of 5.9% of the 30 min before event onset, approximately 3.5 times the matched random-NREM value of 1.7%. In contrast, REM occupied only 0.7% of the 30 min after event offset, compared with 3.1% in the control. Event-aligned analysis showed elevated REM occupancy approximately 30–8 min before onset and little REM thereafter (Fig. 10b). The nearest REM epoch occurred a median of 30 min before an event and 51 min afterward; REM was closer before than after the event in five of six participants. N1 occupancy also increased during the 10 min preceding onset, consistent with a REM-to-N1-to-N2 transition.', '', 'Event-aligned stage occupancy showed an N2 probability of 0.76 at onset, peaking at 0.82 about 4 min later (Fig. 9a), indicating that these events were associated with N2 rather than N3 slow wave activity. The relationship with REM sleep was weak and inconsistent. REM occupied an average of 3.5% of the 30 min before event onset, approximately twice the matched random-NREM value of 1.8%, and 1.3% of the 30 min after event offset, compared with 3.3% in the control (Fig. 9b). However, the nearest REM epoch occurred a median of 85 min before an event and 44 min afterward, and REM was closer before than after the event in only two of six participants. N1 occupancy was similar in the 10 min before onset and in the 20 min before that (0.15 in both).',
     'Rewritten from the new results. With 45 events the pooled pre-onset REM excess is smaller '
     '(x2.0, was x3.4) and REM is nearer AFTER the event in 4 of 6 participants, so the claimed '
     '"consistent temporal relationship with preceding REM" no longer holds. Figure refs 10a/10b '
     '-> 9a/9b. N1 pre-onset 0.154 vs 0.151: no increase.',
     PN + ' n2_onset, rem_pre, rem_pre_null, rem_ratio, rem_post, rem_post_null, rem_med_before, '
     'rem_med_after, rem_side, n1_pre; ladder_onset_occupancy.csv, ladder_rem_side_by_subject.csv'),
    ('F', 'Fig. 9 caption', 'distribution of ', '22', ' harmonic-comb events', '45',
     'Event count.', PN + ' n_events'),
    ('F', 'Fig. 9 caption', 'REM occupancy surrounding event onset',
     ', showing elevated occupancy approximately 30–8 min before onset and little REM thereafter', '.',
     '; REM was slightly more frequent before onset than after, but not consistently across participants',
     'As §3.6.', PN + ' rem_pre, rem_post, rem_side'),
    ('F', '§3.6', '', 'Thus, harmonic-comb events generally emerged during consolidated N2 sleep approximately 10–30 min after REM and were not typically followed by REM. Given the small sample of 22 events from six participants, this pattern was considered exploratory rather than confirmatory. The events may represent nonsinusoidal, quasi-periodic mechanical or hemodynamic activity associated with stable post-REM NREM sleep rather than cortical slow-wave activity.', '', 'Thus, harmonic-comb events occurred mainly during consolidated N2 sleep. The pooled REM occupancy suggested a link with preceding REM, but this was not consistent across participants. Given 45 events from six participants, these patterns were considered exploratory rather than confirmatory. The events may represent nonsinusoidal, quasi-periodic mechanical or hemodynamic activity associated with stable NREM sleep rather than cortical slow-wave activity.',
     'Conclusion of §3.6 follows the new results: N2 stands, post-REM timing does not.',
     PN + ' rem_side, rem_10_30'),

]

# Paragraph index (P, 0-based over every w:p in the body) is filled in at build time.

LEFT = r"""
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
- **§3.7 P229 R² values** (0.63/0.71/0.68 → 0.637/0.710/0.855) and "Significant" (p = 0.057/0.035/0.008, n = 6). §3.7 was in scope only for the figure cross-reference.
- **§3.9 K-complex latencies** (causal 3.5/5.2/5.5 s vs text 3.6/3.7/4.3 s) and "count-matched" null (it is 20 × marks, min 200). Not in scope.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Run-level tracked-change engine
# ─────────────────────────────────────────────────────────────────────────────
class Rev:
    def __init__(self, start=9001):
        self.n = start

    def next(self):
        self.n += 1
        return str(self.n)


REV = Rev()


def q(tag):
    return W + tag


def owner_p(el):
    a = el.getparent()
    while a is not None and a.tag != q('p'):
        a = a.getparent()
    return a


def has_anc(el, tag, stop):
    a = el.getparent()
    while a is not None and a is not stop:
        if a.tag == q(tag):
            return True
        a = a.getparent()
    return False


def orig_runs(p):
    """Runs of p's own original text (excludes inserted runs; includes deleted ones)."""
    return [r for r in p.iter(q('r'))
            if owner_p(r) is p and not has_anc(r, 'ins', p)]


def rtext(r):
    return ''.join(c.text or '' for c in r if c.tag in (q('t'), q('delText')))


def spans(p):
    out, pos = [], 0
    for r in orig_runs(p):
        n = len(rtext(r))
        out.append((r, pos, pos + n))
        pos += n
    return out


def ptext(p):
    return ''.join(rtext(r) for r in orig_runs(p))


def clean_rpr(rpr):
    if rpr is None:
        return None
    rpr = copy.deepcopy(rpr)
    for ch in rpr.findall(q('rPrChange')):
        rpr.remove(ch)
    return rpr


def split_run(r, k):
    """Split run r at text offset k (0 < k < len). Returns the new right-hand run."""
    assert r.getparent().tag != q('del'), 'refusing to split a deleted run'
    rpr = r.find(q('rPr'))
    right = etree.Element(q('r'), attrib=dict(r.attrib))
    if rpr is not None:
        right.append(copy.deepcopy(rpr))
    content = [c for c in r if c.tag != q('rPr')]
    pos, moving = 0, False
    for c in content:
        if moving:
            right.append(c)      # moves c
            continue
        if c.tag == q('t'):
            s = c.text or ''
            if pos < k < pos + len(s):
                cut = k - pos
                c.text = s[:cut]
                c.set(XML_SPACE, 'preserve')
                t2 = etree.SubElement(right, q('t'))
                t2.text = s[cut:]
                t2.set(XML_SPACE, 'preserve')
                moving = True
            elif pos + len(s) == k:
                moving = True
            pos += len(s)
    assert moving, 'split point not found'
    r.addnext(right)
    return right


def cut_at(p, b):
    for r, s, e in spans(p):
        if s < b < e:
            split_run(r, b - s)
            return


def make_run(text, rpr):
    r = etree.Element(q('r'))
    if rpr is not None:
        r.append(rpr)
    t = etree.SubElement(r, q('t'))
    t.text = text
    t.set(XML_SPACE, 'preserve')
    return r


def wrap(tag, el):
    w = etree.Element(q(tag))
    w.set(q('id'), REV.next())
    w.set(q('author'), AUTHOR)
    w.set(q('date'), DATE)
    el.addprevious(w)
    w.append(el)
    return w


def outer(r, p):
    """The element to insert after: the run, or its w:del wrapper."""
    par = r.getparent()
    return par if par.tag == q('del') else r


def apply_edit(p, start, old, new):
    end = start + len(old)
    cut_at(p, start)
    cut_at(p, end)
    sp = spans(p)
    if old:
        covered = [(r, s, e) for r, s, e in sp if s >= start and e <= end and e > s]
        assert ''.join(rtext(r) for r, _, _ in covered) == old, 'coverage mismatch'
        for r, _, _ in covered:
            assert r.getparent().tag != q('del'), 'target overlaps an earlier deletion'
            other = [c for c in r if c.tag not in (q('rPr'), q('t'), q('lastRenderedPageBreak'))]
            assert not other, f'unexpected run content in deletion: {[c.tag for c in other]}'
        rpr = clean_rpr(covered[0][0].find(q('rPr')))
        last = None
        for r, _, _ in covered:
            for t in r.findall(q('t')):
                t.tag = q('delText')
                t.set(XML_SPACE, 'preserve')
            last = wrap('del', r)
        anchor = last
    else:
        before = [(r, s, e) for r, s, e in sp if e == start and e > s]
        assert before, 'no run ends at the insertion point'
        r0 = before[-1][0]
        rpr = clean_rpr(r0.find(q('rPr')))
        anchor = outer(r0, p)
    if new:
        while anchor.getnext() is not None and anchor.getnext().tag == q('ins'):
            anchor = anchor.getnext()
        nr = make_run(new, rpr)
        anchor.addnext(nr)
        wrap('ins', nr)


# ─────────────────────────────────────────────────────────────────────────────
# Views for validation
# ─────────────────────────────────────────────────────────────────────────────
def is_hl(r):
    return r.find(q('rPr') + '/' + q('highlight')) is not None


def view(p, mode):
    """mode 'reject': original text (t + delText, no ins); 'accept': t only, no del.
    Returns (text, highlighted_text)."""
    txt, hl = [], []
    for r in p.iter(q('r')):
        if owner_p(r) is not p:
            continue
        ins, dele = has_anc(r, 'ins', p), has_anc(r, 'del', p)
        if mode == 'reject' and ins:
            continue
        if mode == 'accept' and dele:
            continue
        s = rtext(r)
        txt.append(s)
        if is_hl(r):
            hl.append(s)
    return ''.join(txt), ''.join(hl)


def hl_run_count(root):
    n = 0
    for r in root.iter(q('r')):
        if is_hl(r) and ''.join(t.text or '' for t in r.findall(q('t'))):
            n += 1
    return n


def body_ps(root):
    return list(root.find(q('body')).iter(q('p')))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    locks = [f for f in FINAL.iterdir() if f.name.startswith('~$') and 'V12' in f.name]
    if locks:
        sys.exit(f'STOP: Word lock file present: {locks[0].name}')

    zin = zipfile.ZipFile(SRC)
    v11 = etree.fromstring(zin.read('word/document.xml'))
    root = etree.fromstring(zin.read('word/document.xml'))
    ps = body_ps(root)
    v11_ps = body_ps(v11)
    hl_before = hl_run_count(v11)

    applied = []
    for grp, sec, pre, old, post, new, why, src in EDITS:
        needle = pre + old + post
        hits = [(i, j) for i, p in enumerate(ps)
                for j in _find_all(ptext(p), needle)]
        assert len(hits) == 1, f'{len(hits)} matches for {needle!r}'
        i, j = hits[0]
        apply_edit(ps[i], j + len(pre), old, new)
        applied.append((grp, sec, i, pre, old, post, new, why, src))
        print(f'  [{grp}] P{i:<3} {old!r} -> {new!r}')

    # ── validation ──────────────────────────────────────────────────────────
    assert len(ps) == len(v11_ps)
    for i, (a, b) in enumerate(zip(v11_ps, ps)):
        ta, ha = view(a, 'reject')
        tb, hb = view(b, 'reject')
        assert ta == tb, f'P{i}: reject-view text differs from V11'
        assert ha == hb, f'P{i}: reject-view highlighting differs from V11'
    for grp, sec, i, pre, old, post, new, *_ in applied:
        acc, _ = view(ps[i], 'accept')
        if new:
            assert new in acc, f'P{i}: inserted text missing from accept view: {new!r}'
        else:
            assert old not in acc, f'P{i}: deleted text still in accept view: {old!r}'
    hl_after = hl_run_count(root)
    n_ins = len(root.findall('.//' + q('ins')))
    n_del = len(root.findall('.//' + q('del')))

    # highlighted text that was itself deleted
    hl_deleted = []
    for d in root.iter(q('del')):
        for r in d.iter(q('r')):
            if is_hl(r):
                hl_deleted.append(rtext(r))
    media = swap_figures(root, zin.read('word/_rels/document.xml.rels'))
    accepted = [view(p, 'accept')[0] for p in ps]
    accept_all(root)
    assert [view(p, 'accept')[0] for p in ps] == accepted
    assert root.find('.//' + q('ins')) is None and root.find('.//' + q('del')) is None
    xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    tmp = OUT.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w') as zout:
        for info in zin.infolist():
            data = (xml if info.filename == 'word/document.xml'
                    else media.get(info.filename) or zin.read(info.filename))
            zout.writestr(info, data, compress_type=info.compress_type)
    zin.close()
    tmp.replace(OUT)

    with zipfile.ZipFile(OUT) as z:
        assert z.testzip() is None
        for n in z.namelist():
            if n.endswith('.xml') or n.endswith('.rels'):
                etree.fromstring(z.read(n))


    print(f'\nedits {len(applied)}  w:ins {n_ins}  w:del {n_del}')
    print(f'highlight runs with visible text: before {hl_before}  after {hl_after}')
    print(f'highlighted text deleted: {hl_deleted}')
    write_log(applied, hl_before, hl_after, hl_deleted, n_ins, n_del)
    print(f'wrote {OUT.relative_to(ROOT)}\nwrote {LOG.relative_to(ROOT)}')


def accept_all(root):
    """Accept every tracked change: drop w:del, unwrap w:ins."""
    for d in list(root.iter(q('del'))):
        d.getparent().remove(d)
    for ins in list(root.iter(q('ins'))):
        par = ins.getparent()
        k = par.index(ins)
        for child in list(ins):
            par.insert(k, child)
            k += 1
        par.remove(ins)


def swap_figures(root, rels_xml):
    """Point each NEW_FIGS caption's picture at a new PNG; keep width, fix height."""
    from PIL import Image
    A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
    R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
    WP = '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}'
    rels = {r.get('Id'): r.get('Target') for r in etree.fromstring(rels_xml)}
    ps = body_ps(root)
    out = {}
    for cap, png in NEW_FIGS.items():
        i = next(k for k, p in enumerate(ps) if ptext(p).strip().startswith(cap))
        j = next(k for k in range(i - 1, -1, -1) if ps[k].find('.//' + A + 'blip') is not None)
        blip = ps[j].find('.//' + A + 'blip')
        target = 'word/' + rels[blip.get(R + 'embed')]
        assert target.endswith('.png'), target
        w, h = Image.open(png).size
        for ext in [ps[j].find('.//' + WP + 'extent')] + ps[j].findall('.//' + A + 'ext'):
            if ext is not None and ext.get('cx'):
                ext.set('cy', str(int(int(ext.get('cx')) * h / w)))
        out[target] = png.read_bytes()
        print(f'  [fig] P{j} {target} <- {png.relative_to(ROOT)}')
    return out


def _find_all(s, sub):
    out, k = [], s.find(sub)
    while k >= 0:
        out.append(k)
        k = s.find(sub, k + 1)
    return out


def write_log(applied, hb, ha, hdel, n_ins, n_del):
    from collections import Counter
    c = Counter(a[1] for a in applied)
    L = ['# V12 changes (tracked revision of V11)', '',
         f'Built by `writeup/ppt/_build_v12.py` from `writeup/review/final/CAP_sleep_mask_manuscript_V11.docx` '
         f'(unchanged) into `CAP_sleep_mask_manuscript_V12.docx`. Every edit is a Word tracked change '
         f'(author "{AUTHOR}", {DATE[:10]}) made inside the existing runs; the rPr of the edited run is copied '
         f'onto inserted text.', '',
         'Paragraph index **P** counts every `w:p` in the body from 0 (as in V11_HIGHLIGHTS.md); '
         'paper_numbers.csv / the task use **¶ = P + 1**.', '',
         f'**{len(applied)} edits** ({n_ins} insertions, {n_del} deletion wrappers). '
         f'Highlighted runs with visible text: {hb} before, {ha} after. '
         f'Highlighted text deleted: {", ".join(repr(x) for x in hdel) or "none"}. '
         'Validation: "reject all" view equals V11 text and highlighting in every paragraph.', '',
         '| section | edits |', '|---|---|']
    L += [f'| {k} | {v} |' for k, v in c.items()]
    L += ['', '## Edits', '']
    names = {'A': 'A. Rate numbers', 'B': 'B. Supplementary cross-references',
             'C': 'C. Main-figure cross-references', 'D': 'D. Factual corrections',
             'G': 'G. Consistency',
             'F': 'F. Harmonic combs (adaptive detector; Figs. 8-9 pictures replaced, untracked)'}
    last = None
    for n, (grp, sec, i, pre, old, post, new, why, src) in enumerate(applied, 1):
        if grp != last:
            L += [f'### {names[grp]}', '']
            last = grp
        o = old if old else '(insert)'
        L += [f'{n}. **P{i} (¶{i + 1}), {sec}.** "…{pre}**{o}**{post}…" → **"{new or "(deleted)"}"**',
              f'   - Reason: {why}', f'   - Source: {src}', '']
    L += [LEFT.strip(), '']
    LOG.write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
