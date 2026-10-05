"""
Supplementary V4: a clean rebuild of V3, with the S5 section on slow trends.

V3 had accumulated the leftovers of earlier cuts -- dozens of empty paragraphs and
stray manual page breaks between figures, which Word renders as blank pages -- and
headings in two styles with some sections unheaded. Patching that is fragile, so
V4 is assembled from scratch out of V3's parts:

    front matter              kept from V3 (title, authors, affiliations)
    S1  capacitance drift     V3 Fig S1, as is
    S2  in-band SNR           V3 Fig S3, as is
    S3  SEC vs PSG channels   V3 Fig S5 (coherence) + Table S1 + gate paragraph
    S4  rate estimation       V3 text; figures redrawn at print size, titles moved
                              into the captions; Fig S11 (calibration vs baseline)
                              dropped, its numbers kept in the prose
    S5  variance and slow     new: variance tails, movement-corrected CLE-CRE and
        trends                CH with trend velocity, velocity at REM onset

Removed (superseded by S5 or out of context): V3 Figs S2 (left-right
difference), S4 (overnight CH vs CLE-CRE), S6 and section S3 (integrated
imbalance).

Formatting rules applied throughout: one heading style per level (Heading1 for
sections, Heading2 for S5's parts); every figure paragraph is kept with its
caption (keepNext); captions open with a bold "Figure Sn." label; no empty
paragraphs and no manual page breaks. Figures are numbered in order, and the
text's figure references are resolved from the same numbering.

Text corrections against V3, from the paper-code audit (paper/outputs/PAPER_NUMBERS.md):
    Table S1 paragraph   "eleven of the twelve nights" -> ten
    coherence caption    thorax >= airflow holds on three of four channels, not every
    k paragraph          spectral spread "two to three times wider" -> up to about twice
    counts paragraph     reference "between about 55 and 80" -> about 56 and 66

Builds on   writeup/review/final/CAP_sleep_mask_manuscript supplementary V3.docx
Writes      writeup/review/final/CAP_sleep_mask_manuscript supplementary V4.docx

Usage
    .venv/Scripts/python.exe writeup/ppt/_build_supp_v4.py
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pandas as pd
from lxml import etree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _build_supp_v3 as B   # noqa: E402  (Doc and the paragraph/image helpers)

ROOT = HERE.parents[1]
SRC = B.FINAL / 'CAP_sleep_mask_manuscript supplementary V3.docx'
DST = B.FINAL / 'CAP_sleep_mask_manuscript supplementary V4.docx'
REP = ROOT / 'reports' / 'mean_value'
RATE = ROOT / 'writeup' / 'figures' / 'rate_supp'
S5 = ROOT / 'writeup' / 'figures' / 'supp_s5'
# SNR and coherence redrawn by the paper pipeline without baked-in titles;
# their values match V3's figures exactly (paper/checks/s03_signal_vs_legacy.py)
PAPER_FIG = ROOT / 'paper' / 'outputs' / 'figures'
q = B.q


def _r(x, d=1):
    return f'{x:.{d}f}'


# ── content ──────────────────────────────────────────────────────────────────
# ('h1'|'h2', text)             heading
# ('t', text)                   body paragraph; {fig:key} -> "Figure Sn"
# ('fig', key, png, caption)    figure; png None = reuse V3's image for that key
# ('table',)                    Table S1 caption + table, from V3

def rate_section():
    return [
        ('h1', 'S4. Respiratory and cardiac rate estimation'),
        ('t', 'This section describes the rate pipeline as it is run and reports '
              'what it produces. It is descriptive: no agreement limits and no '
              'inferential tests are reported.'),
        ('t', 'A rate is produced in five steps. Take one SEC channel, band-pass it '
              'to the band of interest (0.1–0.5 Hz for respiration, 0.5–3.0 Hz for '
              'cardiac activity), count peaks in each 30-second epoch with a loose '
              'detector (prominence 0.05σ, minimum spacing 0.4 s), divide the count '
              'by a scaling factor k, and report the result. The detector is loose on '
              'purpose: the capacitive waveform is not a clean sinusoid and a strict '
              'detector misses genuine cycles, so it over-counts consistently instead '
              'and k absorbs that ({fig:pipe}).'),
        ('fig', 'pipe', RATE / 'fig_rate_pipeline.png',
         'The rate pipeline worked through one 60-second stretch of CRE (S2N1). '
         '(a) The raw channel. (b) The same stretch band-passed to the respiratory '
         'band (0.1–0.5 Hz), with every peak counted by the loose detector marked '
         '(prominence 0.05σ, minimum spacing 0.4 s). The 20 peaks give 19.6 per '
         'minute, which divided by k = 1.18 gives 16.6 breaths per minute.'),
        ('t', 'k is the ratio between what the detector counts and what the PSG '
              'reference records, taken as the median of that ratio over a '
              'recording’s valid epochs with individual ratios clipped to 0.3–5.0. '
              'It is reported per night, per participant and per channel '
              '({fig:kch}). k is not a fitted correction but a measurement of what '
              'the waveform contains: with peak counting, breathing gives 1.04 on CH, '
              '1.14 on CLE and 1.18 on CRE — roughly one detected peak per breath — '
              'while cardiac activity gives 1.93, 1.95 and 1.96, two deflections per '
              'heartbeat.'),
        ('fig', 'kch', RATE / 'fig_k_by_channel.png',
         'k for every night, participant and channel. (a) Breathing, peak '
         'counting. (b) Breathing, spectral peak. (c) Heart rate, peak counting. '
         '(d) Heart rate, spectral peak. One point per night, coloured by '
         'participant age; the black bar and its value are the median of the twelve '
         'nights; the dotted line marks k = 1.'),
        ('t', 'The spectral detector takes the strongest peak of the Welch spectrum '
              'inside the band instead of counting peaks in time. For cardiac '
              'activity it lands near 1, so it finds the fundamental rather than a '
              'harmonic, but its spread across nights is up to about twice that of '
              'peak counting. For respiration its k is identical on all three '
              'channels, 0.96 with identical spread, because that estimator returns '
              'nearly the same value in every epoch; its k is therefore that constant '
              'divided by the reference and carries no channel information. It is '
              'shown so that the degeneracy is visible rather than asserted.'),
        ('t', 'A caveat applies wherever k appears. The cardiac reference was checked '
              'against R-peaks detected on the raw ECG of the same recording. It '
              'agrees on eight of the twelve, but runs 36% high on S2N1 and 29% high '
              'on S6N1, and cannot be checked on S5N1 or S6N2 where the ECG channel '
              'is unusable; S6N2’s reference of 129 beats/min median during sleep is '
              'implausible on its face. Because k is a count divided by that '
              'reference, an inflated reference reads as a deflated k, and those '
              'three recordings are exactly the three that sit below the rest. Over '
              'the eight recordings with a verified reference the cardiac k is 1.98, '
              'range 1.81–2.28. The low values should not be read as a property of '
              'the sensor until the reference is corrected.'),
        ('t', 'Before the division by k it is worth seeing what is counted '
              '({fig:counts}). The raw count, undivided, against the reference makes '
              'the gap between them — which is k — directly visible. The cardiac '
              'count sits near 120 per minute for the whole night and barely moves '
              'while the reference varies between about 56 and 66 beats/min, so after '
              'division by k the estimate is close to constant too: what brings the '
              'night average near the right value is k, not the counting. The '
              'respiratory count does follow part of the reference’s shape but with a '
              'large and varying offset. The three channels move together almost '
              'exactly in the per-epoch k, so its excursions are not channel noise.'),
        ('fig', 'counts', RATE / 'fig_rate_counts_and_k.png',
         'What is counted, before division by k, across one night (S3N2, the '
         'recording with the most usable epochs). (a, b) Peaks counted per minute on '
         'each channel against the PSG reference (black), undivided: (a) breathing, '
         '(b) heart rate. (c, d) The per-epoch k implied by those counts, with each '
         'channel’s single whole-night k as a dashed line: (c) breathing, (d) heart '
         'rate. All traces are smoothed over five epochs.'),
        ('t', 'Applying each recording’s single k epoch by epoch gives the output the '
              'pipeline would produce for a night ({fig:all}). Across all twelve '
              'recordings the estimate holds a plausible level but does not follow '
              'the reference’s excursions. Compared with a no-sensor baseline that '
              'predicts the cohort-median rate, the night-average breathing rate is '
              'better under every calibration (error 0.24, 0.57 and 0.94 breaths/min '
              'for k from the same night, the same participant’s other night, and '
              'the other participants, against 1.20 for the baseline). The '
              'night-average heart rate is not: under either calibration usable in '
              'practice the error (3.77 and 3.19 beats/min) exceeds the baseline of '
              '2.76, a comparison to be read with the reference caveat above.'),
        ('fig', 'all', RATE / 'fig_rate_allsessions.png',
         'Rate estimates for all twelve recordings, one row per recording (participant '
         'age in brackets). (a) Breathing rate, breaths/min. (b) Heart rate, '
         'beats/min. The PSG reference is black and each SEC channel is converted by '
         'its own k; traces are smoothed over five epochs. The values printed in each '
         'panel are the median absolute difference from the reference per channel.'),
        ('t', 'How much a single k per recording gets wrong is shown by the per-epoch k '
              '({fig:kep}). That quantity uses the reference, so it is not an '
              'estimator, but it measures how far the ratio moves. Within a recording '
              'it is fairly steady, with an interquartile width of about 0.18–0.20 '
              'for breathing and 0.19–0.24 for cardiac activity; between recordings '
              'it moves much more. The twelve recordings are six participants on two '
              'nights each, and nothing is averaged across them.'),
        ('fig', 'kep', RATE / 'fig_k_per_epoch.png',
         'Per-epoch k for every recording and channel. (a) Breathing. (b) Heart rate. '
         'The dot is the median and the bar the interquartile range of that '
         'recording’s per-epoch k; recordings are labelled with participant age.'),
        ('t', 'No capacitive feature varied with participant age, k included, in '
              'either band. With six participants no correlation is computed; the '
              'ages are printed so that a reader can see the spread. Cardiac k sits '
              'near 2 for the 37, 54, 55 and 61 year olds, and the two recordings '
              'furthest from that belong to the youngest and the oldest participant — '
              'the same two whose cardiac reference is in question.'),
        ('t', 'Limitations. Six participants and twelve nights; every number here is '
              'descriptive and no statistical test is reported. Reference rates come '
              'from PSG channels measuring different physical quantities from the SEC '
              'sensor, so some disagreement is expected and is not separable from '
              'sensor error, and on four recordings the cardiac reference is either '
              'demonstrably high or unverifiable. The respiratory reference range '
              'across this cohort is narrow, which is why a constant predictor '
              'performs as well as it does. k is treated as one number per '
              'recording, which the per-epoch k shows is an approximation.'),
    ]


def s5_section():
    e = pd.read_csv(REP / 'variance_tails_enrichment.csv')
    e = e[e.subset == 'motion-free']

    def tail(t, ch, stage):
        v = e[(e['tail'] == t) & (e.channel == ch) & (e.stage == stage)].enrichment
        return f'{ch} {_r(v.median())}-fold, {int((v > 1).sum())} of {len(v)} nights'

    hi_n3 = ', '.join(
        f"{c} {_r(e[(e['tail'] == 'hi') & (e.channel == c) & (e.stage == 'N3')].enrichment.median(), 2)}"
        for c in ('CH', 'CLE', 'CRE'))
    vel = pd.read_csv(REP / 'destep_velocity.csv')
    rem = pd.read_csv(REP / 'rem_onset_velocity_subjects.csv')
    sens = pd.read_csv(REP / 'rem_onset_velocity_sensitivity.csv')
    ev = pd.read_csv(REP / 'rem_onset_velocity_events.csv')
    rd, rc = rem[rem.signal == 'CLE-CRE'], rem[rem.signal == 'CH']
    still = rd.dropna(subset=['response_still_only'])
    loose = sens[(sens.rem_free_gap_min == 5) & (sens.rem_hold_min == 2)
                 & (sens.signal == 'CLE-CRE')].iloc[0]
    n_ev = int((ev.signal == 'CLE-CRE').sum())

    return [
        ('h1', 'S5. Variance and slow trends of the SEC signal across sleep'),
        ('t', 'This section collects three related observations on the slow behaviour '
              'of the SEC signal during sleep. They are reported descriptively, one '
              'value per night or per participant.'),

        ('h2', 'S5.1 Where the high- and low-variance periods fall'),
        ('t', 'For every 30-second epoch we computed the variance of each SEC channel '
              '(CH, CLE and CRE) after removing content above 10 Hz. Within each '
              'night, the 10% of epochs with the highest variance and the 10% with '
              'the lowest were marked, so every night is judged on its own scale. '
              'Epochs with head movement (the top 10% of accelerometer activity in '
              'that night) were set aside, as were 14 epochs at the end of S4N1 where '
              'the signal had stopped. For each sleep stage we then compared how '
              'often it appeared among the marked epochs with how often it appeared '
              'in the night as a whole; a ratio above 1 means the stage is '
              'over-represented.'),
        ('t', 'The two ends of the variance range fell in opposite stages, on all '
              'three channels ({fig:var}). The quietest epochs were over-represented '
              f'in N3 (median ratio {tail("lo", "CH", "N3")}; {tail("lo", "CLE", "N3")}; '
              f'{tail("lo", "CRE", "N3")}) and under-represented in wakefulness, N1 '
              'and REM. The most variable epochs showed the reverse: over-represented '
              f'in wakefulness ({tail("hi", "CH", "Wake")}; {tail("hi", "CLE", "Wake")}; '
              f'{tail("hi", "CRE", "Wake")}) and almost never in N3 (median ratio '
              f'{hi_n3}). Because the pattern holds on the two single-sided channels '
              'as well as on CH, it is not a property of the CH electrode arrangement. '
              'Low SEC variance therefore marks deep sleep and high variance marks '
              'wakefulness, in line with the fall in band amplitude from wakefulness '
              'to N3 described in the main text.'),
        ('fig', 'var', S5 / 'figS5_1_variance_tails.png',
         'Sleep stage of the highest-variance (a–c) and lowest-variance (d–f) epochs, '
         'for CH (a, d), CLE (b, e) and CRE (c, f), with head-movement epochs removed. '
         'Each point is one night: how often the stage occurs among the marked epochs '
         'divided by how often it occurs in the whole night (1 = chance, dashed line). '
         'Bars are the median across nights; the fraction above each stage is the '
         'number of nights with a ratio above 1. Points on the bottom edge had no '
         'marked epochs in that stage.'),

        ('h2', 'S5.2 Slow trends in CLE−CRE and CH'),
        ('t', 'A head movement shifts the SEC level abruptly and leaves it there, which '
              'hides slower changes. We removed these shifts as follows. The signal '
              'was averaged in 10-second blocks; the accelerometer identified the '
              'blocks in which the head moved; the change in SEC level across each of '
              'those blocks was set to zero; and the signal was rebuilt from the '
              'remaining changes. Nothing between movements is altered. For display '
              'the result was smoothed with a running median over the preceding 5 '
              'minutes. To describe how fast the level was drifting, we fitted a '
              'straight line to the preceding 30 minutes of the movement-corrected '
              'signal at every point and took its slope, in fF per hour (the trend '
              'velocity). Blocks inside a head movement were left out of each fit. '
              'Because each value uses only the past 30 minutes, it reacts about 15 '
              'minutes after a change.'),
        ('t', 'With the movement shifts removed, both signals drift slowly over the '
              'night ({fig:night}–{fig:allch}). During still periods the typical size '
              f'of the trend velocity was {_r(vel.diff_median_abs_fF_per_h.min())}–'
              f'{_r(vel.diff_median_abs_fF_per_h.max())} fF/h for CLE−CRE and '
              f'{_r(vel.CH_median_abs_fF_per_h.min())}–'
              f'{_r(vel.CH_median_abs_fF_per_h.max())} fF/h for CH, largest in '
              'participant 6. The two trends moved in the same direction on '
              f'{int((vel.corr_vdiff_vch > 0).sum())} of the 12 nights (correlation '
              f'{_r(vel.corr_vdiff_vch.min(), 2)} to {_r(vel.corr_vdiff_vch.max(), 2)}, '
              f'median {_r(vel.corr_vdiff_vch.median(), 2)}). A few large shifts were '
              'not removed because the accelerometer did not register a movement at '
              'the time (S3N1 at about 2.2 h, S2N1 at about 4.1 h, S6N1 at about '
              '3.3 h); each appears as a deep, brief excursion of the velocity and '
              'should not be read as a slow trend.'),
        ('fig', 'night', S5 / 'figS5_2_one_night_S4N2.png',
         'One night (S4N2) in full. (a) Scored sleep stage. (b) Head turn from the '
         'accelerometer. (c) CLE−CRE before (light grey) and after (grey) removal of '
         'the movement shifts, with its 5-minute running median. (d) CLE−CRE trend '
         'velocity. (e) CH after removal of the movement shifts, with its running '
         'median. (f) CH trend velocity. Trend velocity is the slope of a line fitted '
         'to the preceding 30 minutes, in fF/h.'),
        ('fig', 'alldiff', S5 / 'figS5_3_all_nights_CLE-CRE.png',
         'Movement-corrected CLE−CRE for all twelve nights. For each night, from the '
         'top: scored sleep stage (colour band), CLE−CRE with its 5-minute running '
         'median, and its trend velocity.'),
        ('fig', 'allch', S5 / 'figS5_4_all_nights_CH.png',
         'Movement-corrected CH for all twelve nights, laid out as {fig:alldiff}.'),

        ('h2', 'S5.3 The trend velocity at REM onset'),
        ('t', 'Studies with invasive pressure monitoring and with near-infrared '
              'spectroscopy report that intracranial blood volume and pressure rise '
              'within minutes of entering REM sleep, which predicts a positive change '
              'in velocity at REM onset. We took each REM-episode onset (the first REM '
              'epoch after at least 10 minutes without REM, with REM filling at least '
              'half of the next 5 minutes) and compared the average velocity over the '
              '15 minutes after onset with the average over the 15 minutes before. '
              'Here the 30-minute line was centred on each moment rather than '
              'trailing it, so that the timing of a change is not delayed; as a '
              'result a change at onset begins to show about 15 minutes earlier. '
              'Events were averaged within each participant first. As a reference, '
              'the same comparison was made at 500 sets of random times in non-REM '
              'sleep in the same nights, at least 20 minutes from any REM onset.'),
        ('t', f'{n_ev} REM onsets in {rd.shape[0]} participants met the definition; '
              "participant 6's recordings contain none, because REM there is too "
              'fragmented. The CLE−CRE velocity rose after REM onset in '
              f'{int((rd.response_fF_h > 0).sum())} of {rd.shape[0]} participants '
              f'(median {_r(rd.response_fF_h.median())} fF/h), and in '
              f'{int((still.response_still_only > 0).sum())} of {still.shape[0]} when '
              'only onsets without a head movement in the 2 minutes around them were '
              f'used ({{fig:rem}}). Only {int((rd.percentile_in_null > 95).sum())} '
              'participant exceeded the upper 5% of its own random-time reference, and '
              'the result depended on the definition: when brief returns to REM within '
              f'an episode were also counted as onsets, {int(loose.n_rise)} of '
              f'{int(loose.n_participants)} participants rose. CH rose in '
              f'{int((rc.response_fF_h > 0).sum())} of {rc.shape[0]}. The direction '
              'matches the expected rise at REM onset, but with this number of events '
              'the effect is weak and is reported as an observation, not a finding.'),
        ('fig', 'rem', S5 / 'figS5_5_rem_onset.png',
         'Trend velocity around REM-episode onset. (a, b) Average velocity of CLE−CRE '
         '(a) and CH (b) from 30 minutes before to 30 minutes after onset: thin lines '
         'are participants, the thick line and band their mean ± standard error, and '
         'the dashed line the same average at random non-REM times. (c, d) For each '
         'participant (number of onsets in brackets), the change in velocity from the '
         '15 minutes before onset to the 15 minutes after (dot), against the 5–95% '
         'range of the random-time reference (grey bar) and its median (tick).'),
    ]


def content():
    return [
        ('h1', 'S1. Capacitance drift at room temperature over 24 hours'),
        ('fig', 'drift', None, None),
        ('h1', 'S2. Signal-to-noise ratio in the physiological band'),
        ('fig', 'snr', PAPER_FIG / 's03_signal' / 'figS3_inband_snr.png', None),
        ('h1', 'S3. SEC channels against polysomnographic channels'),
        ('fig', 'coh', PAPER_FIG / 's03_signal' / 'figS5_cap_psg_coherence.png',
         'Coherence between each capacitive channel and each PSG channel in the '
         'respiratory band, 0.1–0.5 Hz (left), and the cardiac band, 0.5–3.0 Hz '
         '(right): the median '
         'in-band value over five-minute windows with 30-second Welch segments (about '
         '19 segments per estimate, noise floor 0.053). Every pair uses identical '
         'segmentation, so the upward bias of the estimator is common to the matrix '
         'and the contrast against EEG is fair. In the respiratory band thoracic '
         'effort couples at least as strongly as nasal airflow on CLE, CRE and '
         'CLE−CRE; on CH the two are within 0.003.'),
        ('table',),
        ('t', 'The gate removed one sensor in the cohort: the thoracic belt in both S3 '
              'nights, scoring −0.06 and −0.13 against the median of the other three. '
              'In those same two nights that belt is also anti-correlated with nasal '
              'airflow (r = −0.47 and −0.47), which is the signature of paradoxical '
              'thoraco-abdominal motion rather than a failed sensor. Every other '
              'sensor in every other night scored +0.23 or above and is kept, so ten '
              'of the twelve nights use all four signals and the two S3 nights use '
              'three.'),
    ] + rate_section() + s5_section()


# ── document assembly ─────────────────────────────────────────────────────────

def _text(el):
    return ''.join(t.text or '' for t in el.iter(q('t'))).strip()


def _v3_parts(doc):
    """Pick the pieces of V3 that are reused, by their caption text."""
    body = doc.root.find(q('body'))
    kids = list(body)

    def fig(cap_start):
        i = next(i for i, el in enumerate(kids) if _text(el).startswith(cap_start))
        img = next(kids[j] for j in range(i - 1, -1, -1)
                   if list(kids[j].iter('{%s}blip' % B.A)))
        return img, kids[i]

    def style(el):
        s = el.find('.//' + q('pStyle'))
        return s.get(q('val')) if s is not None else ''

    parts = {'drift': fig('Figure S1.'), 'snr': fig('Figure S3.'),
             'coh': fig('Figure S5.')}
    ti = next(i for i, el in enumerate(kids) if _text(el).startswith('Table S1.'))
    tbl = next(el for el in kids[ti:] if etree.QName(el).localname == 'tbl')
    parts['table'] = (kids[ti], tbl)
    first_h = next(i for i, el in enumerate(kids) if style(el) == 'Heading1')
    front = [el for el in kids[:first_h]
             if etree.QName(el).localname == 'p' and _text(el)]
    tmpl = {
        'h1': kids[first_h],
        'h2': next(el for el in kids if _text(el).startswith('S4. Capacitive')),
        'body': next(el for el in kids if len(_text(el)) > 300),
        'img': parts['snr'][0],
    }
    return parts, front, tmpl


def _caption(cap_tmpl, label, text):
    """Caption paragraph: bold 'Figure Sn.' then the text, in the template's style."""
    p = copy.deepcopy(cap_tmpl)
    runs = list(p.iter(q('r')))
    proto = copy.deepcopy(runs[0])
    for r in runs:
        r.getparent().remove(r)
    for child in list(proto):
        if child.tag != q('rPr'):
            proto.remove(child)
    for s, bold in ((label + ' ', True), (text, False)):
        r = copy.deepcopy(proto)
        rpr = r.find(q('rPr'))
        if rpr is None:
            rpr = etree.Element(q('rPr'))
            r.insert(0, rpr)
        for b in rpr.findall(q('b')) + rpr.findall(q('bCs')):
            rpr.remove(b)
        if bold:
            # w:b sits near the front of rPr in schema order
            pos = 1 if rpr.find(q('rFonts')) is not None else 0
            rpr.insert(pos, etree.Element(q('b')))
        t = etree.SubElement(r, q('t'))
        t.text = s
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        p.append(r)
    return p


def _keep_next(p):
    ppr = p.find(q('pPr'))
    if ppr is None:
        ppr = etree.Element(q('pPr'))
        p.insert(0, ppr)
    if ppr.find(q('keepNext')) is None:
        ppr.insert(1 if ppr.find(q('pStyle')) is not None else 0,
                   etree.Element(q('keepNext')))
    return p


def _strip_breaks(el):
    for tag in ('lastRenderedPageBreak', 'pageBreakBefore'):
        for x in list(el.iter(q(tag))):
            x.getparent().remove(x)
    for br in list(el.iter(q('br'))):
        if br.get(q('type')) == 'page':
            br.getparent().remove(br)
    return el


MAX_FIG_H_IN = 7.4      # leaves room for a caption on the same 9 in text height


def _fit_width(doc, png):
    """Full text width, unless that makes the figure too tall to share a page
    with its caption; then narrower, keeping the aspect ratio."""
    from PIL import Image
    with Image.open(png) as im:
        w, h = im.size
    cx = B.page_width_emu(doc)
    max_cy = int(MAX_FIG_H_IN * 914400)
    return cx if cx * h / w <= max_cy else int(max_cy * w / h)


def main():
    doc = B.Doc(SRC)
    parts, front, tmpl = _v3_parts(doc)
    cap_tmpl = parts['drift'][1]
    items = content()

    # number the figures in order, then resolve {fig:key} references
    order = [it[1] for it in items if it[0] == 'fig']
    num = {k: i + 1 for i, k in enumerate(order)}

    def ref(s):
        return re.sub(r'\{fig:(\w+)\}', lambda m: f'Figure S{num[m.group(1)]}', s)

    new = [copy.deepcopy(el) for el in front]
    for it in items:
        kind = it[0]
        if kind in ('h1', 'h2'):
            new.append(_keep_next(B.clone_text(tmpl[kind], it[1])))
        elif kind == 't':
            new.append(B.clone_text(tmpl['body'], ref(it[1])))
        elif kind == 'table':
            cap, tbl = parts['table']
            new.append(_keep_next(copy.deepcopy(cap)))
            new.append(copy.deepcopy(tbl))
        elif kind == 'fig':
            _, key, png, cap_text = it
            if cap_text is None:                             # V3's caption text
                cap_text = re.sub(r'^\s*Figure S\d+\.\s*', '',
                                  _text(parts[key][1])).strip()
            if png is None:                                  # reuse V3's image
                img = copy.deepcopy(parts[key][0])
            else:
                img = B.make_image_para(doc, tmpl['img'], Path(png), Path(png).name,
                                        cx=_fit_width(doc, Path(png)))
            new.append(_keep_next(img))
            new.append(_caption(cap_tmpl, f'Figure S{num[key]}.', ref(cap_text)))

    body = doc.root.find(q('body'))
    sect = body.find(q('sectPr'))
    for el in list(body):
        if el is not sect:
            body.remove(el)
    for el in new:
        sect.addprevious(_strip_breaks(el))

    B.repair_tables(doc)
    kept, freed = B.prune_orphan_media(doc)
    doc.save(DST)

    chk = B.Doc(DST)
    caps = [chk.text(p) for p in chk.paras if re.match(r'\s*Figure S\d+\.', chk.text(p))]
    nums = [int(re.match(r'\s*Figure S(\d+)\.', c).group(1)) for c in caps]
    assert nums == list(range(1, len(nums) + 1)), nums
    assert '{fig:' not in '\n'.join(chk.text(p) for p in chk.paras), 'unresolved ref'
    body = chk.root.find(q('body'))
    empties = sum(1 for p in body.findall(q('p'))
                  if not chk.text(p).strip() and not list(p.iter('{%s}blip' % B.A)))
    pbr = sum(1 for br in chk.root.iter(q('br')) if br.get(q('type')) == 'page')
    print(f'built {DST.name}: {len(nums)} figures, {kept} images, '
          f'{empties} empty body paragraphs, {pbr} manual page breaks, '
          f'{freed:.1f} MB of unused media dropped')
    for c in caps:
        print('   ', c[:95])


if __name__ == '__main__':
    main()
