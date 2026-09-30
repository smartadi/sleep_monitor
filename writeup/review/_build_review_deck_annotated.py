"""
Build the full figure deck for the professor, with an explanation page after
every plot page.

This is `writeup/ppt/build_deck.py`'s deck — every figure at full resolution,
one per slide — with two changes:

* a new section on the K-complex work and the Fultz chain, and
* an EXPLANATION SLIDE after each plot, saying what you are looking at, what to
  look for in it, what it means, and where it stops.

One judgement call worth knowing about. Six of the sections are per-night
series, twelve slides each of the same figure for a different recording. Those
get ONE explanation slide at the end of the series rather than twelve identical
copies — the explanation does not change from S1N1 to S6N2, and repeating it
twelve times would bury the figures. Every standalone figure gets its own.

Numbers quoted in the explanations come from the manuscript and
notebooks/ANALYSIS_LOG.md. Where a result has not been verified in this pass the
explanation describes how to read the figure rather than asserting a value.

Run from the repo root:
    .venv/Scripts/python.exe writeup/review/_build_review_deck_annotated.py
Output -> writeup/review/CAP_sleep_mask_review_deck_annotated.pptx
"""

import os
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGS = ROOT / 'writeup' / 'figures'
ONSET = ROOT / 'analysis' / 'delta_onset' / 'outputs'
OUT = Path(os.environ.get('DECK_OUT',
                          HERE / 'CAP_sleep_mask_review_deck_annotated.pptx'))

W_IN, H_IN = 13.333, 7.5
INK = RGBColor(0x1B, 0x2A, 0x41)
MUTED = RGBColor(0x5A, 0x64, 0x72)
FAINT = RGBColor(0xA8, 0xB0, 0xBA)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
BODY = RGBColor(0x2C, 0x3E, 0x50)
WASH = RGBColor(0xF4, 0xF6, 0xF8)

MARGIN = 0.45
TITLE_TOP = 0.34
CAPTION_TOP = 0.86
IMG_TOP = 1.34
IMG_BOTTOM = 7.16

SESSIONS = ['S%dN%d' % (s, n) for s in range(1, 7) for n in (1, 2)]


def per_night(pattern, title, caption, explain):
    """Twelve figure slides, then one explanation slide for the series."""
    out = [(None, title % s, caption, FIGS / (pattern % s), None) for s in SESSIONS]
    out.append((None, None, None, None, explain))       # explanation-only slide
    return out


# ── explanation blocks ──────────────────────────────────────────────────────
# (heading shown on the explanation slide, [(label, text), ...])

def X(head, seeing, look, means, caveat=None):
    rows = [('What you are seeing', seeing), ('Look for', look),
            ('What it means', means)]
    if caveat:
        rows.append(('Where it stops', caveat))
    return (head, rows)


# (section, title, one-line caption, figure, explanation)
SLIDES = [
    ('What the mask records', None, None, None, None),

    (None, 'The raw capacitive signal',
     'Both rhythms are visible in the unprocessed trace, before any filtering.',
     FIGS / 'signal_validation/fig1_waveform_example.png',
     X('The raw capacitive signal',
       'An unfiltered stretch of the capacitance trace beside the simultaneous PSG.',
       'Two rhythms superimposed: a slow swing near 0.2 Hz (breathing) with a faster '
       'ripple near 1 Hz (heartbeat) riding on it.',
       'Both rhythms are present in the signal before any processing, so nothing '
       'downstream is manufacturing them.',
       'Visible is not the same as measurable epoch by epoch. That is what the rate '
       'section tests, and the answer there is more qualified.')),

    (None, 'In-band signal-to-noise',
     'Respiratory and cardiac bands stand above the local noise floor on every night.',
     FIGS / 'signal_validation/fig2_inband_snr.png',
     X('In-band signal-to-noise  (manuscript Figure 3)',
       'Physiological-band (0.1–3 Hz) power per channel per night, referenced to '
       'the electronic noise floor measured in a separate no-subject recording.',
       'Every point above zero, and CH highest in every participant.',
       'SNR 30.0 dB for CH, 20.7 dB CRE, 18.7 dB CLE; every night clears 6.4 dB. The '
       'rhythms are not instrument noise. CH is the differential channel and gives '
       'roughly twice the response of a single-ended one.',
       'CLE and CRE vary more from session to session. That is sensor-to-skin '
       'distance, not electronics — the noise floor itself is near constant.')),

    (None, 'A night in the frequency domain',
     'Both rhythms persist as continuous ridges across the whole recording.',
     FIGS / 'spectrograms/S1N1_spectrogram_ridges.png',
     X('A night in the frequency domain',
       'A whole-night spectrogram, 0–3 Hz, with the detected ridges drawn on top.',
       'Two horizontal bands running the length of the night: respiratory near '
       '0.20–0.25 Hz, cardiac between 0.9 and 1.8 Hz. A third sits near 0.07 Hz, '
       'below respiration.',
       'The rhythms are continuous rather than intermittent, so they are properties '
       'of the signal and not occasional artefacts.',
       'Brightness varies with coupling and posture, so a dim stretch is not '
       'necessarily an absent rhythm.')),

    ('Are these the rhythms the PSG records?', None, None, None, None),

    (None, 'Coherence with the PSG sensors',
     'At the breathing and heartbeat frequencies, against shift, reverse and EEG controls.',
     FIGS / 'coupling/cap_psg_coherence.png',
     X('Coherence with the PSG sensors',
       'Coherence between mask and PSG evaluated at the PSG-derived respiratory and '
       'cardiac frequencies, with circularly shifted and time-reversed references as '
       'controls and EEG as the negative control.',
       'The aligned pairs sitting above both disrupted controls, and EEG sitting low.',
       'Median coherence 0.31 respiratory and 0.16 cardiac — about half the 0.61 '
       'and 0.27 measured between two PSG sensors recording the same rhythm. Against '
       'phase-randomised surrogates, 14.7% of respiratory and 9.1% of cardiac epochs '
       'exceed the null, against the 5% expected by chance.',
       'The coupling is significant in aggregate and weak per epoch. It should not be '
       'used to accept or reject an individual epoch.')),

    (None, 'Every capacitive channel against every PSG sensor',
     'Cardiac amplitude coupling is specific; respiratory amplitude largely is not.',
     FIGS / 'coupling/cap_psg_matrix.png',
     X('Every capacitive channel against every PSG sensor',
       'Amplitude coupling — the within-night correlation of band-limited log-RMS '
       'on the 30-second grid — for every mask channel against every PSG channel.',
       'The contrast against the EEG column rather than the absolute values, which '
       'are close to uniform across sensors.',
       'Respiratory amplitude coupling is nonspecific: r = 0.38 with nasal airflow '
       'against 0.34 with EEG, a margin of only 0.06–0.09 once motion epochs are '
       'removed. Cardiac is specific: 0.35 with ECG against 0.13 with EEG, margin '
       '0.18–0.20.',
       'Mask respiratory AMPLITUDE is therefore not a surrogate for respiratory '
       'effort or volume, even on nights where the respiratory RHYTHM is coherent.')),

    ('Respiratory rate — every night', None, None, None, None),
    *per_night('rate_rerun/per_night/%s_resp.png', 'Respiration — %s',
               'Mask in green after per-session calibration, PSG reference in black.',
               X('Respiratory rate — how to read these twelve',
                 'One panel per recording: the mask estimate (prominence-thresholded '
                 'peak counting on CRE, scaled by that session’s calibration '
                 'factor k) against the multi-sensor PSG respiratory consensus.',
                 'Whether the estimate holds the right LEVEL across the night — '
                 'and, separately, whether it follows the reference’s '
                 'minute-to-minute wiggles. It holds the level; it does not follow '
                 'the wiggles.',
                 'Median epoch-level error 1.79 breaths/min; night-mean error 0.24 '
                 'breaths/min after same-night calibration.',
                 'Within-night correlation is −0.03 and no recording beats its '
                 'circular-shift null. This is a whole-night mean estimator, not a '
                 'tracker. The cohort’s mean rates span only 14.4–16.8 '
                 'breaths/min, so a constant predictor already reaches 1.20.')),

    ('Cardiac rate — every night', None, None, None, None),
    *per_night('rate_rerun/per_night/%s_card.png', 'Cardiac — %s',
               'The same presentation, cardiac band.',
               X('Cardiac rate — how to read these twelve',
                 'The same estimator and presentation in the cardiac band, against '
                 'ECG R-peaks.',
                 'Sustained excursions of 20–40 beats/min in the reference, and '
                 'whether the estimate follows them. It does not.',
                 'Median epoch-level error 3.41 beats/min, night-mean 1.56. The '
                 'calibration factor k is 1.96 (IQR 1.77–2.01) — about two '
                 'detected peaks per cardiac cycle, consistent with a biphasic '
                 'systolic-plus-dicrotic waveform, and R-peak-triggered averaging '
                 'independently gives 2.02.',
                 'Only 1 of 12 recordings exceeds its null for tracking. Cardiac rate '
                 'varies more than respiratory (SD 6.58 vs 1.57 beats/min) so the '
                 'failure to track is a real limitation, not a resolution floor.')),

    ('Where the rate error sits', None, None, None, None),

    (None, 'Accuracy by sleep stage',
     'Error is lowest in the consolidated stages and worst in wake and REM.',
     FIGS / 'mask_rate_detection/fig3_per_stage_mae.png',
     X('Accuracy by sleep stage',
       'Night-level estimator error split by PSG-scored stage, both bands.',
       'Lowest error in N2 and N3; highest in wake and REM.',
       'Accuracy tracks how still and consolidated the sleep is. Wake and REM bring '
       'movement and posture change, which is what degrades coupling.',
       'Stages differ greatly in epoch count, and epochs within a recording are not '
       'independent, so this is descriptive.')),

    (None, 'Estimator against channel',
     'No channel is reliably better than the CLE−CRE differential for respiration.',
     FIGS / 'mask_rate_detection/fig18_mae_heatmap.png',
     X('Estimator against channel',
       'Mean absolute error for every base estimator crossed with every channel, '
       'both bands, after k-scaling.',
       'The absence of a clear winner — no combination stands out for respiration.',
       'The operational choice (loose prominence-based peak counting on CRE, then a '
       'causal three-epoch median filter) was made by error on this same cohort.',
       'Two consequences. The choice is descriptive of this cohort and would need '
       'out-of-sample confirmation. And the Welch spectral estimator is DEGENERATE '
       'in the respiratory band — 4-second segments give 0.25 Hz resolution and '
       'two usable bins, returning 15.0 breaths/min in 99.98% of epochs. It is kept '
       'only as a constant baseline.')),

    ('Persistent ridges — every night', None, None, None, None),
    *per_night('harmonics/ridges/ridge_tune_%s_CRE.png', 'Ridges — %s',
               'Tracked respiratory and cardiac ridges on CRE, with the hypnogram above.',
               X('Persistent ridges — how to read these twelve',
                 'Per night: hypnogram on top, then a background-corrected 0–3 Hz '
                 'spectrogram with a single Viterbi-tracked trace per rhythm '
                 '(respiratory cyan, cardiac red) and the tracker’s own '
                 'confidence printed.',
                 'Continuity of the two traces, and the slow band below 0.1 Hz at the '
                 'bottom.',
                 'A respiratory ridge is present in 96.7% of clean epochs, lowest '
                 'frequency median 0.23 Hz (about 14 breaths/min), which is what '
                 'identifies it as respiratory. Each rhythm is tracked as ONE '
                 'continuous path, because each has exactly one value at any instant '
                 '— a generic multi-ridge detector returns fragments and lets a '
                 'ridge hop to a subharmonic.',
                 'CRE was selected because it carried the dominant ridge in 9 of 12 '
                 'sessions — chosen on this dataset, so the ridge results are '
                 'descriptive of this cohort.')),

    (None, 'Ridge structure by sleep stage',
     'Ridge count, power and lowest frequency all separate by stage.',
     FIGS / 'harmonics/band_ridge_by_stage.png',
     X('Ridge structure by sleep stage  (manuscript Figure 6)',
       'Band-specific ridge features by stage: active ridges per epoch, total ridge '
       'power, and lowest ridge frequency, respiratory band on top and cardiac below.',
       'Respiratory ridge power in N3 against the other stages.',
       'Respiratory ridge power is lower during N3 in 4 of 6 participants by '
       'subject-level median and 5 of 6 by mean. It survives excluding motion epochs '
       'and matching epoch counts, and the same direction appears in CLE and CH, so '
       'it is not specific to the post-hoc chosen CRE. Plausibly the more regular, '
       'lower-effort breathing of deep sleep.',
       'A weak, partially consistent association — not a sleep stager. The '
       'Kruskal–Wallis p-values on the panel come from pooled, non-independent '
       'epochs and are descriptive only; the subject-level direction counts are the '
       'evidence. Ridge COUNT shows no consistent stage pattern at all.')),

    ('Harmonic comb episodes — every night', None, None, None, None),
    *per_night('harmonics/ladders/ladder_%s.png', 'Harmonic comb — %s',
               'Flat, richly harmonic episodes lasting tens of minutes.',
               X('Harmonic comb episodes — how to read these twelve',
                 'Whole-night background-enhanced spectrograms with any detected '
                 'stacks of quasi-harmonic bands marked, hypnogram above.',
                 'Flat stacks of three or more sustained bands lasting minutes, '
                 'clearly distinct from the always-present respiratory and cardiac '
                 'ridges.',
                 '22 events across 9 sessions and all 6 participants. The bands sit '
                 'close to integer multiples of a common fundamental with '
                 'intermediate multiples often missing — one representative event '
                 'runs 0.15, 0.28, 0.42, 0.68, 0.95 Hz, which are the 1st, 2nd, 3rd, '
                 '5th and 7th multiples of about 0.14 Hz, deviating by up to 9%.',
                 'Rare. 22 events from 6 participants is exploratory, and the '
                 'detector requires at least three consecutive harmonics precisely so '
                 'that respiration plus heart rate cannot by themselves make one.')),

    (None, 'When the comb episodes occur',
     'They sit in N2 and tend to follow REM rather than accompany slow-wave sleep.',
     FIGS / 'harmonics/ladder_stage_relationship.png',
     X('When the comb episodes occur  (manuscript Figure 10)',
       'Stage occupancy in a ±30 minute window around event onset, and REM '
       'occupancy around the same events.',
       'N2 probability near 0.9 at onset, and REM elevated roughly 30 to 8 minutes '
       'BEFORE onset with little after.',
       '19 of 22 events (86%) fall in N2, one each in N1, N3 and wake. They emerge in '
       'consolidated N2 about 10–30 minutes after a REM period and are not '
       'themselves followed by REM, consistent with a REM→N1→N2 re-entry. '
       'So they are NOT a cortical slow-wave signature, which is what an earlier '
       'framing had assumed.',
       'Exploratory. 22 events, 6 participants, REM compared against 200 duration- '
       'and session-matched random NREM draws per event.')),

    ('The cortical events — a mechanical response', None, None, None, None),

    (None, 'Response at sleep spindles',
     'A clear low-band response, but no sigma-band signature: mechanical, not electrical.',
     FIGS / 'spindles/fig_spindle_lowband_detection.png',
     X('Response at sleep spindles  (manuscript Figure 7)',
       'Event-triggered averages around 14,305 technologist-scored N2 spindles, '
       '351–2,134 per session.',
       'The low-band trace rising while the sigma trace stays flat.',
       'Low-frequency (0–3 Hz) power rises 0.45–0.55 dB, largest in CH. '
       'Sigma (11–16 Hz) — the spindle’s OWN frequency — changes by '
       '0.02–0.03 dB, while contact EEG sigma rises 3.45 dB at the same instants. '
       'The mask responds to spindles but not at the spindle’s frequency, so it '
       'is sensing a mechanical or hemodynamic accompaniment, not the electrical '
       'oscillation. THIS IS THE PAPER’S CLEANEST RESULT.',
       'The EEG channel is the positive control that makes the negative meaningful: '
       'it proves the test could have detected sigma had sigma been there. Present in '
       'all 12 recordings, absent at random N2 points, and survives excluding '
       'spindles that coincide with scored arousals.')),

    (None, 'Response at delta-burst onset',
     'The mask responds within seconds of the EEG burst, across all three channels.',
     FIGS / 'delta_onset/fig_delta_onset_cohort.png',
     X('Response at delta-burst onset  (manuscript Figure 8)',
       'Peri-onset mask band power by frequency band and channel, aligned to EEG '
       'delta-burst onset, against count-matched random-NREM controls.',
       'A flat pre-onset baseline, then a sharp rise at onset peaking a few seconds '
       'later.',
       '344 qualifying onsets. Subject-level post-onset peaks run 1.4–3.2 '
       'standardised units against no more than 0.17 for the controls, in all six '
       'participants and every channel-band combination. Greatest in CRE and CH and '
       'in the 0.5–1 and 1–3 Hz bands.',
       'Two things. The causal envelope estimator matters — zero-phase filtering '
       'manufactures an apparent PRE-onset rise by leaking the large post-onset '
       'response backwards. And between 47% and 84% of onsets fall within 10 s of a '
       'scored arousal, so the response cannot be assigned to the delta burst rather '
       'than to the arousal.')),

    ('K-complexes, and where the response sits in time', None, None, None, None),

    (None, 'What a K-complex is, in our own recordings',
     'All 50 K-complexes the technologist scored, from the PSG’s own '
     'Spindle/K annotation channel.',
     ONSET / 'fig_kcomplex_morphology.png',
     X('What a K-complex is, in our own recordings',
       'Eight individual scored K-complexes (A), the average of all 50 (B), and mean '
       'spectra around K-complexes, around spindles and at random N2 points (C). '
       'These are the technologist’s marks, not our detections.',
       'In B, the textbook biphasic shape; in C, that the two N2 events occupy '
       'different frequency bands.',
       'The average is a sharp negative deflection to −70 µV at +0.1 s then '
       'a positive swing to +65 µV at +0.35 s, flat by +0.7 s, median 161 µV '
       'peak-to-peak. A K-complex is an isolated cortical down-state — the '
       'single-event version of the slow oscillation that fills N3.',
       'Panel C governs the next two slides: spindles lift SIGMA (11–16 Hz), '
       'far outside the capacitive band, so that test can be settled on frequency. '
       'K-complexes lift DELTA (0.5–4 Hz), which OVERLAPS the bands the mask '
       'measures — so for K-complexes, timing has to carry the argument instead.')),

    (None, 'Our delta-burst detector picks up a selective fifth of them',
     'Every scored K-complex put through the two gates the detector applies.',
     ONSET / 'fig_kcomplex_vs_detector.png',
     X('Our delta-burst detector picks up a selective fifth of them',
       'One scored K-complex (A) and one qualifying delta-burst onset (B) with the '
       'detector’s delta envelope and thresholds beneath each; then all 50 marks '
       'scored against both gates (C, D).',
       'In C, how long a real K-complex holds the envelope above the burst threshold.',
       'Median 3.5 s (IQR 2.3–6.1) — a K-complex usually IS a multi-second '
       'delta run, because one is typically followed by further slow waves rather '
       'than standing alone. 44% clear the ≥4 s gate, 50% have a quiet 30 s '
       'before them, 18% pass both — and the detector finds all nine of those, '
       '9 out of 9.',
       'So the manuscript’s onsets are genuine K-complex-like events but a '
       'SELECTED subset, roughly a fifth, chosen for sitting at the head of a '
       'sustained delta run after a quiet baseline. Separately, the scorer annotated '
       'spindles exhaustively (21,881) and K-complexes barely at all (57), so these '
       'marks are good examples and not a complete ground truth.')),

    (None, 'K-complex → mask: a response that follows the cortical event',
     'The three questions the spindle analysis asks, on a trigger independent of '
     'our own detector.',
     ONSET / 'fig_kcomplex_cap_response.png',
     X('K-complex → mask: a response that follows the cortical event',
       'Low-frequency mask power around the 50 scored K-complexes, against a '
       'count-matched random-NREM null, with the simultaneous EEG delta as a '
       'positive control.',
       'In B, where the peak sits relative to zero.',
       'CH +0.47 z at +1.9 s, CLE +0.34 z at +1.9 s, CRE +0.32 z at +2.9 s, against '
       'null peaks of 0.03–0.08 z — five to eight times the null, with CH '
       'largest, the same channel ordering the spindle result gives. Per-event peak '
       'latency is 3.6–4.3 s while the EEG delta control peaks at t = 0. '
       'Electrical coupling is instantaneous, so a response from electrical pickup '
       'would land at zero lag. It lands three to four seconds later.',
       'n = 50 over ten usable sessions; S4N2 and S5N1 contribute none and S1N2 '
       'contributes one, and per-session peaks run 0.01 to 1.77 z. Descriptive and '
       'per-session, no pooled p-value. The arousal confound of the previous section '
       'is unchanged.')),

    (None, 'How this connects to Fultz et al. 2019',
     'Their measurement is a lag, not a correlation — which is what makes our '
     'earlier zero-lag negative interpretable.',
     FIGS / 'delta_onset/fig_fultz_chain.png',
     X('How this connects to Fultz et al. 2019',
       'The chain Fultz followed with simultaneous EEG and fMRI in NREM sleep, with '
       'our measured mask latency placed on the same time axis. The schematic is '
       'drawn from scratch; only the ~6.4 s lag and the ordering come from the paper.',
       'Where our 1.9–4.3 s window sits relative to zero and to their 6.4 s.',
       'Their chain: a cortical slow wave, then cortical blood volume falls, then CSF '
       'flows into the fourth ventricle — the skull is a closed box, so departing '
       'blood must be replaced. Critically they found NO EEG–CSF correlation at '
       'zero lag; the coupling appears only once a delay is allowed. That is exactly '
       'why our own zero-lag test (capacitive delta power against EEG delta power, '
       'r ≈ 0.015) was negative, and why that negative was never the end of the '
       'question.',
       'Looking in the delayed window instead, the response is there — and its '
       'latency places the mask BETWEEN the cortical event and the ventricular CSF '
       'response, which is where a sensor measuring cranial displacement belongs. '
       'Fultz measure the end of the chain; the mask sits one step earlier.')),

    ('Overnight evolution of the sensor value — every night', None, None, None, None),
    *per_night('channel_evolution/%s_CH_CLE-CRE.png', 'Sensor value — %s',
               'Level, imbalance and band amplitude across the night, on the hypnogram.',
               X('Overnight evolution — how to read these twelve',
                 'Per night: hypnogram, the mean-referenced level, the slow left-right '
                 'imbalance, within-window variance, head-turn angle, and the rate '
                 'spectrograms for both difference channels.',
                 'Step-like changes in the level row, and whether they line up with '
                 'steps in the head-turn row below.',
                 'Median within-night ranges are 105, 71, 158 and 595 fF for CLE, CRE, '
                 'CLE−CRE and CH; CH exceeds 1000 fF on the three most mobile '
                 'nights. The nonstationarity is posture and coupling, NOT '
                 'instrumental drift — an independent 25-hour test showed only '
                 '3 fF of drift across a 3 °C swing.',
                 'Some slow cycles persist beyond individual posture transitions, so '
                 'the signal holds both posture effects and slower variation that '
                 'acceleration does not explain. Characterised, not explained.')),

    (None, 'Sensor value over the night — all twelve',
     'Each night referenced to its own session mean.',
     FIGS / 'channel_evolution/ch_vs_clecre_grid.png',
     X('Sensor value over the night — all twelve  (supplementary Figure S4)',
       'The cohort version of the level row: CH against CLE−CRE for all twelve '
       'recordings, each referenced to its own session mean.',
       'That the step structure is in every night, while the SIZE of the excursion '
       'is not.',
       'The steps and their coincidence with posture change are universal; the '
       'amplitude is night-specific and largest on the mobile nights (S5N1, S6N1, '
       'S6N2).',
       'Because each night is referenced to its own mean, these panels compare '
       'SHAPE across nights, not absolute operating point — which is confounded '
       'by mask-mount offset.')),

    ('Channels and variance', None, None, None, None),

    (None, 'CH against the CLE−CRE differential',
     'The two differ in scale and are only partly coherent — they are not interchangeable.',
     FIGS / 'mean_value/ch_vs_diff_relationship.png',
     X('CH against the CLE−CRE differential',
       'The hardware difference channel CH plotted against the arithmetic difference '
       'CLE−CRE, across the cohort.',
       'How far the relationship departs from a single fixed line.',
       'CH is the sensor’s own differential channel and is NOT the arithmetic '
       'CLE−CRE. Across the cohort the offset is about −742 fF and the gain '
       'of CH on CLE−CRE runs from −1.9 to +6.5, CHANGING SIGN between '
       'subjects.',
       'They are therefore not interchangeable, and a result stated on one channel '
       'does not transfer to the other. Anywhere polarity matters, CH cannot be '
       'assumed to share CLE−CRE’s sign convention.')),

    (None, 'Capacitance imbalance — all twelve',
     'Magnitude and direction of the left-right offset across each night.',
     FIGS / 'imbalance/imbalance_grid.png',
     X('Capacitance imbalance — all twelve',
       'The slow left-right imbalance marker per night: signed low-pass of the '
       'mean-centred difference, inside its magnitude envelope.',
       'How much the SIZE varies between nights, and how the sign behaves within a '
       'night.',
       'Time-averaged magnitude runs 3.1–135.7 fF and the overnight integral '
       '15–705 fF·h, a 46-fold spread. Two nights from the same participant '
       'differ by 2.0–5.8 times, so the burden is a NIGHT property driven by '
       'coupling and posture history, not a stable subject trait.',
       'Magnitude is posture-dependent (median 9.0 fF supine, 11.3 left-lying, '
       '107.7 right-lying) while direction is not explained by posture. The signed '
       'index stays between −0.22 and +0.09, so none of this should be read as '
       'persistent physiological lateralisation.')),

    (None, 'Band amplitude by sleep stage',
     'Amplitude falls from wake into deep sleep and returns in REM, in every subject.',
     FIGS / 'mean_value/variance_by_stage.png',
     X('Band amplitude by sleep stage',
       'Within-window capacitive variance by PSG-scored stage, per subject.',
       'The monotone fall from wake through N1, N2 to N3, and the rise again in REM.',
       'The pattern holds in 6 of 6 subjects and is carried mainly by CH. It is one '
       'of the few stage effects in this dataset that is consistent across every '
       'participant.',
       'Consistency of DIRECTION is the claim; the magnitude differs widely between '
       'subjects, and variance also responds to motion, so this is a depth-related '
       'signal rather than a clean stage marker.')),

    (None, 'Variance as a sleep-depth indicator',
     'Low variance is deep sleep, high is wake; N3 lowest in every subject.',
     FIGS / 'staging/variance_staging.png',
     X('Variance as a sleep-depth indicator',
       'The same variance recast as a classifier: how well it separates N3 from the '
       'rest, per subject and pooled.',
       'The AUC, and whether the direction is the same in every subject.',
       'N3 is lowest in every subject and the pooled AUC is about 0.65 — a real '
       'and consistent effect.',
       '0.65 is a weak classifier. This is a depth INDICATOR, not a sleep stager, and '
       'should be reported as such. The manuscript’s related N3 classifier work '
       'reaches only 0.534 under leave-one-subject-out.')),

    (None, 'Variance around REM onset',
     'It steps up at REM onset without an anticipatory rise in the minutes before.',
     FIGS / 'staging/variance_peri_rem.png',
     X('Variance around REM onset',
       'Capacitive variance averaged around every REM onset, with the minutes before '
       'and after shown.',
       'Whether anything happens BEFORE the onset line.',
       'Variance steps up at REM onset with no anticipatory rise beforehand. So it '
       'reflects the state change rather than predicting it — the same '
       'lead/lag discipline applied to the cortical events elsewhere in the deck.',
       'REM onset is scored on 30-second epochs, so the timing resolution of this '
       'figure is bounded by the staging, not by the sensor.')),

    (None, 'How much of that variance is the instrument',
     'Against an unworn recording on the same mask: respiration clears the floor, '
     'cardiac barely.',
     FIGS / 'mean_value/baseline_variance_floor.png',
     X('How much of that variance is the instrument',
       'The same variance measure computed on an UNWORN recording from the same '
       'device, laid against the on-head nights.',
       'The gap between the worn and unworn distributions, band by band.',
       'This is the control that makes the variance results interpretable: the CH '
       'swing is physical rather than chip noise, and the left-right imbalance is '
       'not an averaging artefact. The respiratory band clears the floor comfortably.',
       'The cardiac band only just clears it. Cardiac-band variance results should '
       'carry that caveat explicitly.')),

    (None, 'Where the high-variance epochs fall',
     'Top-decile variance per recording, with the hypnogram above and motion marked.',
     FIGS / 'mean_value/high_variance_traces_2col.png',
     X('Where the high-variance epochs fall',
       'For each recording, the epochs in the top decile of capacitive variance, '
       'drawn under the hypnogram with motion marked.',
       'Whether the marked epochs cluster in particular stages, and how many sit on '
       'flagged motion.',
       'The top decile is defined per recording, so every night contributes the same '
       'number by construction — this figure is about WHERE they fall, never how '
       'many.',
       'Motion is the obvious confound and is marked for exactly that reason. Read '
       'this figure together with the enrichment plot that follows.')),

    (None, 'Which stages they fall in',
     'Observed over expected occupancy — 1.0 is chance. Almost none land in N3.',
     FIGS / 'mean_value/high_variance_enrichment.png',
     X('Which stages the high-variance epochs fall in',
       'Observed-over-expected stage occupancy for the top-decile variance epochs, '
       'where 1.0 means exactly the occupancy that stage’s duration predicts.',
       'How far each stage sits from 1.0, and N3 in particular.',
       'Almost none fall in N3, which is the same result as the variance-by-stage '
       'figure seen from the other direction and is consistent with variance '
       'tracking sleep depth.',
       'Normalising by expected occupancy removes the trivial explanation that N2 '
       'wins because there is more N2 — but it does not separate depth from '
       'stillness, since deep sleep is also the stillest.')),

    ('Per-night sensor metrics', None, None, None, None),

    (None, 'What the three metrics measure',
     'Area under the mean-referenced CLE−CRE curve; time and impulse count above '
     'a variance threshold.',
     FIGS / 'prof_metrics/method.png',
     X('What the three metrics measure',
       'A worked definition of the three per-night summary metrics on a stretch of '
       'real signal.',
       'How each metric is constructed from the trace above it.',
       'Metric 1 is the area under the mean-referenced CLE−CRE curve; metric 2 '
       'is the time spent above a variance threshold; metric 3 is the number of '
       'contiguous excursions above that threshold, reported per hour.',
       'All three depend on the referencing and threshold choices spelled out in the '
       'next three slides. Read those before reading the results table.')),

    (None, 'Metric 1 — absolute area under CLE−CRE',
     'Total area scales with night length (4.1–8.7 h), so the per-hour form is '
     'on the right.',
     FIGS / 'prof_metrics/metric1_area.png',
     X('Metric 1 — absolute area under CLE−CRE',
       'The integrated magnitude of the mean-referenced difference per night, shown '
       'both as a total and normalised per hour.',
       'The difference between the two panels.',
       'Recording length in this cohort runs 4.1 to 8.7 hours, so the raw total '
       'partly measures how long the night was. The per-hour form is the comparable '
       'one.',
       'Both forms still carry the coupling and posture dependence of the underlying '
       'signal, so a between-subject comparison is not a physiological comparison.')),

    (None, 'Choosing the variance threshold',
     'A per-night percentile puts the same 10% above threshold every night, by '
     'construction.',
     FIGS / 'prof_metrics/threshold_policy.png',
     X('Choosing the variance threshold',
       'What happens to metrics 2 and 3 under a per-night percentile threshold '
       'against a single absolute threshold shared by every night.',
       'That the percentile policy puts the same fraction above threshold every '
       'night — by construction, not by measurement.',
       'A per-night percentile therefore cannot show that one night has more '
       'high-variance time than another. That is why the metrics that follow use one '
       'absolute threshold shared across nights.',
       'An absolute threshold makes nights comparable but inherits the coupling '
       'differences between them, which is the trade being made here. Worth stating '
       'plainly wherever these metrics are reported.')),

    (None, 'Metric 2 — time above a variance threshold',
     'One absolute threshold shared by every night, swept from 2 to 50 fF².',
     FIGS / 'prof_metrics/metric2_duration.png',
     X('Metric 2 — time above a variance threshold',
       'Time spent above threshold per night, with the threshold swept from 2 to '
       '50 fF² rather than fixed at one value.',
       'Whether the ordering of nights stays stable as the threshold moves.',
       'Sweeping is the honest presentation: if the ranking of nights survives the '
       'sweep, it is a property of the nights; if it flips, it was a property of the '
       'threshold.',
       'A single operating point (10 fF²) is used in the summary table, so the '
       'sweep is what justifies that choice.')),

    (None, 'Metric 3 — frequency of variance impulses',
     'An impulse is one contiguous run above threshold; reported per hour.',
     FIGS / 'prof_metrics/metric3_impulses.png',
     X('Metric 3 — frequency of variance impulses',
       'The count of contiguous excursions above threshold, per hour of recording.',
       'How this differs from metric 2: many short excursions and one long one can '
       'give the same TIME above threshold but very different COUNTS.',
       'Metric 3 captures fragmentation where metric 2 captures burden. The pair is '
       'more informative than either alone.',
       'Count is sensitive to how runs are joined — a brief dip below threshold '
       'splits one impulse into two — so it is more parameter-dependent than '
       'metric 2.')),

    (None, 'All twelve nights',
     'The three metrics at 10 fF², with age and PSQI alongside.',
     FIGS / 'prof_metrics/results_table.png',
     X('All twelve nights',
       'The three metrics at the 10 fF² operating point for every recording, '
       'with participant age and PSQI beside them.',
       'The spread between nights, and whether a participant’s two nights sit '
       'near each other.',
       'This is the table the exploratory associations in the Discussion are drawn '
       'from.',
       'Six participants. Any age or PSQI association read off this table is '
       'exploratory and cannot carry a headline claim — with n = 6 a Spearman '
       'correlation needs |ρ| ≥ 0.83 to reach p < 0.05 at all.')),

    (None, 'EEG-scored arousals per night',
     'From the PSG’s own scoring, indexed per hour of sleep rather than per hour '
     'of recording.',
     FIGS / 'prof_metrics/arousal_index.png',
     X('EEG-scored arousals per night',
       'Arousal counts from the PSG’s own scoring, expressed as an index per '
       'hour of SLEEP.',
       'The choice of denominator.',
       'Per hour of sleep rather than per hour of recording, so a night with a long '
       'quiet wake period is not credited with a low arousal rate.',
       'These are the technologist’s scored arousals and are the reference the '
       'mask-side event work is compared against — not something derived from '
       'the mask.')),

    (None, 'Arousal counts — all twelve nights',
     'Total, index, and the three subtypes, with CAP event rates alongside.',
     FIGS / 'prof_metrics/arousal_table.png',
     X('Arousal counts — all twelve nights',
       'Arousal totals and indices broken into subtype (spontaneous, respiratory, '
       'limb movement), with the mask-side event rates next to them.',
       'Whether a night high in one subtype is high overall.',
       'The subtypes have different physiology, so pooling them can hide the '
       'relationship a subtype-specific comparison would show.',
       'The mask column is placed alongside for comparison only — the two '
       'columns are separately derived and their scales differ.')),

    (None, 'Arousals hour by hour, each subject’s two nights',
     'Twelve panels, a subject to a row, so a night pair sits together. Shared axes '
     'throughout.',
     FIGS / 'prof_metrics/arousal_hourly_by_subject.png',
     X('Arousals hour by hour, each subject’s two nights',
       'Arousal rate through the night, twelve panels with a subject per row so a '
       'participant’s two nights sit side by side. Axes are shared.',
       'Whether a subject’s two nights resemble each other more than they '
       'resemble other subjects’ nights.',
       'The row layout and shared axes are there to make night-to-night '
       'reproducibility readable directly, without a summary statistic.',
       'Two nights per subject is the minimum that can show reproducibility at all, '
       'and cannot separate a stable trait from a shared recording condition.')),

    *per_night('prof_metrics/arousal_timeseries/%s.png',
               'Arousals and the mask — %s',
               'Hypnogram, event lanes with smoothed density, capacitive variance, '
               'slow mean, motion.',
               X('Arousals and the mask — how to read these twelve',
                 'Per night: hypnogram, lanes marking each scored arousal with a '
                 'smoothed density above them, then capacitive variance, the slow '
                 'mean, and flagged motion.',
                 'Whether the variance row rises where the arousal lanes cluster '
                 '— and how often it also rises where the motion row is active.',
                 'These panels are the raw material for the coincidence analysis in '
                 'the next slides. They are deliberately shown before any matching '
                 'statistic, so the reader can judge by eye first.',
                 'Motion is the competing explanation throughout. An arousal usually '
                 'involves movement, so a coincidence between mask variance and a '
                 'scored arousal is not by itself evidence of cortical sensitivity.')),

    (None, 'Do mask bumps land where EEG arousals cluster?',
     'Episodes detected in each signal on its own, then asked whether they '
     'coincide. Roughly twice chance.',
     FIGS / 'prof_metrics/bump_matching_simple.png',
     X('Do mask bumps land where EEG arousals cluster?',
       'Episodes detected independently in each signal — neither detector sees '
       'the other — then tested for coincidence against chance.',
       'The measured coincidence rate against the chance line.',
       'Roughly twice chance. Detecting in each signal separately BEFORE comparing is '
       'what keeps this from being circular; a detector tuned on the other signal '
       'would find agreement by construction.',
       'Twice chance is a real but modest association, and it does not establish '
       'direction or mechanism. Movement accompanies most arousals and is the '
       'standing alternative explanation.')),

    (None, 'The same night by night',
     'Each night’s two curves with its episodes marked — ▲ mask, '
     '▼ EEG.',
     FIGS / 'prof_metrics/bump_matching_nights.png',
     X('The same, night by night',
       'The pooled coincidence result decomposed into its twelve nights, with each '
       'night’s two curves and its marked episodes.',
       'Whether the pooled result is carried by all nights or by a few.',
       'A pooled rate can be produced by two or three nights with many events. This '
       'panel is what shows whether that has happened.',
       'With twelve nights and uneven event counts, a per-night reading is the '
       'honest one — the pooled number should not be quoted on its own.')),

    (None, 'Cortical arousal — PSG scoring against the mask',
     'Both rates per hour of sleep, night by night. Note the separate axes: the '
     'scales differ.',
     FIGS / 'prof_metrics/arousal_cap_vs_psg.png',
     X('Cortical arousal — PSG scoring against the mask',
       'Scored arousal rate and mask event rate for each night, both per hour of '
       'sleep, on SEPARATE axes.',
       'The separate axes. The two rates are not on the same scale and the figure '
       'does not claim they are.',
       'What is being compared is the night-to-night PATTERN, not the absolute '
       'rates. The mask event detector and the technologist are counting different '
       'things.',
       'This is reported as an inventory, not a detection claim. A mask event with '
       'no EEG counterpart cannot be called an arousal on the strength of the EEG’s '
       'silence — which is why the event work elsewhere corroborates with ECG '
       'and pulse rather than by EEG absence alone.')),
]

TITLE = ('Capacitive sleep mask', 'Figures for review, with notes',
         'One figure per slide at full resolution, each followed by an '
         'explanation page')


# ── slide plumbing ──────────────────────────────────────────────────────────

def textbox(slide, text, left, top, width, height, size, color,
            bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = 'Calibri'
    return tb


def rule(slide, left, top, width):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
                                 Inches(width), Inches(0.045))
    shp.fill.solid()
    shp.fill.fore_color.rgb = ACCENT
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def place_image(slide, png):
    w_px, h_px = Image.open(png).size
    box_w, box_h = W_IN - 2 * MARGIN, IMG_BOTTOM - IMG_TOP
    scale = min(box_w / w_px, box_h / h_px)
    w, h = w_px * scale, h_px * scale
    slide.shapes.add_picture(str(png), Inches((W_IN - w) / 2),
                             Inches(IMG_TOP + (box_h - h) / 2), Inches(w), Inches(h))


def explanation_slide(prs, blank, head, rows, n):
    s = prs.slides.add_slide(blank)
    textbox(s, head, MARGIN, TITLE_TOP, W_IN - 2 * MARGIN, 0.6, 22, INK, bold=True)
    rule(s, MARGIN, 1.02, 1.3)
    top = 1.42
    # heights tuned so four blocks fit without overflowing the slide
    height = (6.95 - top) / max(len(rows), 1)
    for label, text in rows:
        textbox(s, label, MARGIN, top, 2.35, 0.4, 14, ACCENT, bold=True)
        textbox(s, text, MARGIN + 2.45, top, W_IN - MARGIN - 2.9, height - 0.12,
                13.5, BODY)
        top += height
    textbox(s, str(n), W_IN - MARGIN - 1.0, 7.16, 1.0, 0.3, 9, FAINT,
            align=PP_ALIGN.RIGHT)
    return s


def main():
    missing = [str(f) for _, _, _, f, _ in SLIDES if f and not Path(f).exists()]
    if missing:
        sys.exit('figures not found:\n  ' + '\n  '.join(missing))

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W_IN), Inches(H_IN)
    blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)
    textbox(s, TITLE[0], MARGIN, 2.55, W_IN - 2 * MARGIN, 1.0, 40, INK, bold=True)
    textbox(s, TITLE[1], MARGIN, 3.45, W_IN - 2 * MARGIN, 0.6, 22, MUTED)
    rule(s, MARGIN, 4.22, 1.6)
    textbox(s, TITLE[2], MARGIN, 4.45, W_IN - 2 * MARGIN, 0.5, 13, FAINT)

    n = n_fig = n_exp = n_sec = 0
    for section, title, caption, fig, explain in SLIDES:
        if section:
            s = prs.slides.add_slide(blank)
            rule(s, MARGIN, 3.32, 1.1)
            textbox(s, section, MARGIN, 3.55, W_IN - 2 * MARGIN, 1.0, 30, INK, bold=True)
            n_sec += 1
            continue

        if fig is not None:
            s = prs.slides.add_slide(blank)
            n += 1
            n_fig += 1
            textbox(s, title, MARGIN, TITLE_TOP, W_IN - 2 * MARGIN, 0.55, 23, INK,
                    bold=True)
            textbox(s, caption, MARGIN, CAPTION_TOP, W_IN - 2 * MARGIN, 0.42, 13.5,
                    MUTED)
            place_image(s, fig)
            textbox(s, Path(fig).name, MARGIN, 7.16, 8.0, 0.3, 8, FAINT)
            textbox(s, str(n), W_IN - MARGIN - 1.0, 7.16, 1.0, 0.3, 9, FAINT,
                    align=PP_ALIGN.RIGHT)

        if explain is not None:
            n += 1
            n_exp += 1
            explanation_slide(prs, blank, explain[0], explain[1], n)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f'wrote {OUT}')
    print(f'  {len(prs.slides._sldIdLst)} slides: {n_fig} figures, '
          f'{n_exp} explanations, {n_sec} sections, '
          f'{OUT.stat().st_size / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
