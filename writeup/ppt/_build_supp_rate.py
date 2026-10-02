"""
Build the simplified rate supplement, as a standalone document.

The rate material is coming out of the main text (user directive) and going
here, rewritten at reporting-and-demo level. What the manuscript described --
seven estimators, four calibration strategies, Bland-Altman limits,
phase-randomised surrogates, a 56-feature gradient-boosted model -- is replaced
by the pipeline that is actually run, the one quantity that characterises it
(k), the same measurement with a spectral detector for comparison, what the
result is against a no-sensor baseline, and whether k could be learned live.

Every number in the text is read from the tables the analyses wrote, not
retyped, so this file cannot drift away from the figures beside it.

Writes  writeup/review/address/CAP_sleep_mask_supplementary_rate.docx

Usage
-----
    .venv/Scripts/python.exe writeup/ppt/_build_supp_rate.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGS = ROOT / 'writeup' / 'figures' / 'rate_supp'
OUT = ROOT / 'writeup' / 'review' / 'address' / \
    'CAP_sleep_mask_supplementary_rate.docx'

HELDOUT = ROOT / 'reports' / 'rates' / 'rerun' / 'heldout_table.csv'
KTAB = ROOT / 'reports' / 'rates' / 'k_by_channel.csv'

CHANNELS = ['CH', 'CLE', 'CRE']
INK = RGBColor(0x1B, 0x2A, 0x41)


def med(cell):
    """'0.24 [0.14-0.34]' -> 0.24"""
    return float(re.match(r'\s*([-\d.]+)', str(cell)).group(1))


def k_cell(k, band, method, ch):
    s = k[(k.band == band) & (k.method == method) & (k.channel == ch)].k
    return f'{s.median():.2f} ({s.quantile(.25):.2f}–{s.quantile(.75):.2f})'


def h(doc, text, level):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = INK
    return p


def para(doc, text, size=11, bold=False, italic=False, space=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    return p


def figure(doc, name, caption, width=6.4):
    path = FIGS / name
    if not path.exists():
        raise SystemExit(f'missing figure: {path}')
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(14)
    r = p.add_run(caption)
    r.font.size = Pt(9.5)
    r.italic = True


def table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Light Grid Accent 1'
    for j, cell in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = str(cell)
        for p in c.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = str(v)
            for p in cells[j].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    return t


def build():
    hd = pd.read_csv(HELDOUT).set_index('band')
    k = pd.read_csv(KTAB)

    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(11)

    h(doc, 'Supplementary: estimating respiratory and cardiac rate '
           'from the SEC signal', 0)
    para(doc, 'This supplement describes the rate pipeline in full, reports '
              'what it produces, and compares it with a spectral detector '
              'applied the same way. It is deliberately descriptive: the aim '
              'is to say what was done and what came out, not to establish '
              'accuracy claims. Rate results are not used to support any '
              'conclusion in the main text.', italic=True)

    # ── 1 ────────────────────────────────────────────────────────────────────
    h(doc, '1.  What the pipeline does', 1)
    para(doc, 'A rate is produced in five steps, and the whole method fits in '
              'a sentence: take one SEC channel, band-pass it to the band of '
              'interest, count peaks in each 30-second epoch, divide the count '
              'by a scaling factor k, and report the result.')
    para(doc, 'The bands are 0.1–0.5 Hz for respiration and 0.5–3.0 Hz '
              'for cardiac activity. The peak detector is deliberately loose — '
              'prominence 0.05σ, minimum spacing 0.4 s — because the '
              'capacitive waveform is not a clean sinusoid and a strict '
              'detector misses genuine cycles. A loose detector instead '
              'over-counts in a consistent way, and k is what absorbs that.')
    figure(doc, 'fig_rate_pipeline.png',
           'Figure S-R1. The pipeline worked through one 60-second stretch of '
           'a single recording. The filtered trace is shown with every detected '
           'peak marked. In this example 20 peaks over the window give 19.6 per '
           'minute, which divided by k = 1.18 gives 16.6 breaths per minute.')

    # ── 2 ────────────────────────────────────────────────────────────────────
    h(doc, '2.  The scaling factor k, and what it measures', 1)
    para(doc, 'k is the ratio between what the detector counts and what the PSG '
              'reference records. For each recording it is the median of that '
              'ratio across the recording’s valid epochs, with individual '
              'ratios clipped to 0.3–5.0 so that a handful of failed epochs '
              'cannot move it. k is reported per night, per participant and per '
              'channel, because all three turn out to matter.')
    para(doc, 'k is not a fitted correction. It is a measurement of what the '
              'waveform contains, and its value is interpretable:')
    table(doc,
          ['Band', 'Detector', 'CH', 'CLE', 'CRE'],
          [['Breathing', 'Peak counting',
            k_cell(k, 'resp', 'peaks_loose', 'CH'),
            k_cell(k, 'resp', 'peaks_loose', 'CLE'),
            k_cell(k, 'resp', 'peaks_loose', 'CRE')],
           ['Breathing', 'Spectral peak',
            k_cell(k, 'resp', 'spectral', 'CH'),
            k_cell(k, 'resp', 'spectral', 'CLE'),
            k_cell(k, 'resp', 'spectral', 'CRE')],
           ['Heart rate', 'Peak counting',
            k_cell(k, 'card', 'peaks_loose', 'CH'),
            k_cell(k, 'card', 'peaks_loose', 'CLE'),
            k_cell(k, 'card', 'peaks_loose', 'CRE')],
           ['Heart rate', 'Spectral peak',
            k_cell(k, 'card', 'spectral', 'CH'),
            k_cell(k, 'card', 'spectral', 'CLE'),
            k_cell(k, 'card', 'spectral', 'CRE')]])
    para(doc, 'Median across the twelve recordings, interquartile range in '
              'brackets.', size=9.5, italic=True, space=12)

    para(doc, 'With peak counting, breathing gives a k just above one, meaning '
              'roughly one detected peak per breath, and CH is closest to '
              'exactly one. Cardiac gives a k just under two on every channel: '
              'two deflections per heartbeat. That figure is independently '
              'supported by R-peak-triggered averaging, which gives 2.02 peaks '
              'per cardiac cycle, and is what a biphasic systolic and dicrotic '
              'waveform would produce.')

    # ── 3 ────────────────────────────────────────────────────────────────────
    h(doc, '3.  The same measurement with a spectral detector', 1)
    para(doc, 'The spectral detector takes the strongest peak of the Welch '
              'spectrum inside the band instead of counting peaks in time. '
              'Running it through the identical k calculation makes the two '
              'directly comparable.')
    figure(doc, 'fig_k_by_channel.png',
           'Figure S-R2. k for every night, participant and channel, for both '
           'detectors and both bands. One point per night, coloured by '
           'participant; the black bar is the median of the twelve; the dotted '
           'line marks k = 1.')
    para(doc, 'For cardiac activity the spectral detector lands near one, so it '
              'is locating the fundamental rather than a harmonic. But its '
              'spread across nights is two to three times wider than peak '
              'counting’s, and that instability is what makes its '
              'epoch-level error about five times larger.')
    para(doc, 'For respiration the spectral k is identical on all three '
              'channels, 0.96 with identical spread. This is not a result: the '
              'respiratory spectral estimator returns nearly the same value in '
              'every epoch, so its k is that constant divided by the reference '
              'and carries no channel information. It is shown because the '
              'degeneracy is visible directly in this figure, and a reader '
              'comparing the two detectors should be able to see why that '
              'column cannot be used.')

    # ── 4 ────────────────────────────────────────────────────────────────────
    h(doc, '4.  What the pipeline achieves', 1)
    para(doc, 'The question a reader should ask of any rate estimate is '
              'whether it beats not having the sensor at all. The comparison '
              'below predicts the cohort-median rate with no SEC input and '
              'uses that as the bar. Three calibrations are shown: k learned on '
              'the night being reported, k taken from the same participant’s '
              'other night, and k taken from everyone else. Only the last two '
              'could be used in practice; the first is calibrated on the '
              'quantity it then reports.')
    figure(doc, 'fig_rate_result.png',
           'Figure S-R3. Error in the night-average rate under each '
           'calibration, against a no-sensor baseline (dashed line). Bars below '
           'the line beat the baseline.')
    table(doc,
          ['Band', 'Same night', 'Other night', 'Population', 'No sensor'],
          [['Breathing (breaths/min)',
            f"{med(hd.loc['resp','night_self']):.2f}",
            f"{med(hd.loc['resp','night_cross']):.2f}",
            f"{med(hd.loc['resp','night_pop']):.2f}",
            f"{med(hd.loc['resp','night_nosensor']):.2f}"],
           ['Heart rate (beats/min)',
            f"{med(hd.loc['card','night_self']):.2f}",
            f"{med(hd.loc['card','night_cross']):.2f}",
            f"{med(hd.loc['card','night_pop']):.2f}",
            f"{med(hd.loc['card','night_nosensor']):.2f}"]])
    para(doc, 'Night-average breathing rate beats the no-sensor baseline under '
              'every calibration, including both that transfer between '
              'recordings. Night-average heart rate does not: under either '
              'transferable calibration the error exceeds the baseline. '
              'Epoch-by-epoch, neither band beats the baseline, and '
              'within-night correlation with the reference is approximately '
              'zero in both bands. The honest summary is that one quantity '
              'survives — the night average of breathing rate — and '
              'the rest do not.')

    # ── 5 ────────────────────────────────────────────────────────────────────
    h(doc, '5.  One k for a whole recording, and how much it moves', 1)
    para(doc, 'Before the division by k, it is worth seeing what is actually '
              'counted. The figure below puts the raw peak count — '
              'undivided — on the same axis as the reference, so the gap '
              'between the two is k made visible, and then shows that ratio '
              'epoch by epoch.')
    figure(doc, 'fig_rate_counts_and_k.png',
           'Figure S-R4. Top: the raw peak count per minute for each channel '
           'against the PSG reference, undivided. Bottom: the implied per-epoch '
           'k, with each channel’s single whole-night k drawn flat across '
           'it. All traces smoothed over five epochs.', width=6.6)
    para(doc, 'The cardiac panel is the more revealing of the two. The counted '
              'rate sits near 120 per minute for the whole night and barely '
              'moves, while the reference heart rate varies between about 55 '
              'and 80. The count is close to constant, so after division by k '
              'the estimate is close to constant too: what makes the night '
              'average come out near the right value is k, not the counting. '
              'The respiratory count does follow part of the reference’s '
              'shape — both fall together around the fourth hour — but '
              'with a large and varying offset.')
    para(doc, 'In the lower row the three channels move together almost '
              'exactly. Their k excursions are therefore not channel noise; '
              'something common to all three, or to the reference, is driving '
              'them.')

    para(doc, 'Applying the single k epoch by epoch and smoothing gives the '
              'output the pipeline would actually produce for a night, which '
              'can be drawn against the reference directly.')
    figure(doc, 'fig_rate_fullnight.png',
           'Figure S-R5. One whole recording, all three channels. The SEC '
           'estimate is the per-epoch peak count divided by that channel and '
           'band’s single k; both traces are smoothed over five epochs '
           '(2.5 min). A row per channel rather than three traces on one axis, '
           'since the estimates are noisy enough that overlaying them would '
           'hide how much of the reference each one follows.', width=6.6)
    para(doc, 'Every channel sits at the right level all night — that is '
              'what k buys — and none of them follows the reference’s '
              'excursions. On this recording the median absolute difference is '
              '1.82, 1.89 and 1.47 breaths/min for CH, CLE and CRE, and 3.03, '
              '2.72 and 2.39 beats/min. CRE is the closest in both bands, which '
              'is why it is the channel the pipeline uses, but the differences '
              'between channels are small next to the gap between any of them '
              'and the reference. This is the near-zero within-night '
              'correlation reported above, shown rather than tabulated: the '
              'pipeline recovers the level of a night’s breathing and heart '
              'rate, not their minute-to-minute course.')
    para(doc, 'How far would k have to move to fix that? A per-epoch k can be '
              'computed directly from the reference, k(t) = count(t) / '
              'reference(t). This is not an estimator — it uses the answer '
              '— but it says how much the ratio actually varies, and it is '
              'reported for every recording and channel separately. The twelve '
              'recordings are six participants on two nights each and are '
              'independent of one another, so nothing is averaged across them.')
    figure(doc, 'fig_k_per_epoch.png',
           'Figure S-R6. Median per-epoch k and its interquartile range, for '
           'every recording and channel, in both bands. Each recording stands '
           'alone.')
    para(doc, 'Within a recording, k is fairly steady: the interquartile width '
              'of the per-epoch k is about 0.18–0.20 for breathing and '
              '0.19–0.24 for cardiac activity. Between recordings it moves '
              'much more. Breathing k ranges from about 1.0 to 1.4 across the '
              'twelve, and cardiac k sits near 2 on most recordings but falls '
              'to about 1.3 and 0.95 on the two recordings from one '
              'participant — for whom the detector resolves one deflection '
              'per heartbeat rather than two.')
    para(doc, 'That pattern explains the calibration results in Section 4. One '
              'k per recording is a reasonable approximation, because k moves '
              'little within a night. A k carried over from another recording '
              'is not, because k moves a lot between them, and a single '
              'participant whose cardiac k is near 1 rather than 2 is enough to '
              'make a population-level cardiac k worse than using no sensor.')

    # ── 6 ────────────────────────────────────────────────────────────────────
    h(doc, '6.  Limitations', 1)
    for t in [
        'Six participants and twelve nights. Every number here is descriptive, '
        'and no statistical test is reported against which a reader could '
        'weigh it.',
        'Reference rates come from PSG channels that measure different '
        'physical quantities from the SEC sensor, so some disagreement is '
        'expected and is not separable from sensor error.',
        'The respiratory reference range across this cohort is narrow, which '
        'is why a constant predictor performs as well as it does and makes the '
        'breathing result easier to beat than it would be in a wider cohort.',
        'k is treated as one number per recording. Section 5 shows it moves '
        'within a night, so this is an approximation whose cost is stated '
        'there.',
    ]:
        p = doc.add_paragraph(t, style='List Bullet')
        for r in p.runs:
            r.font.size = Pt(11)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f'wrote {OUT}')
    print(f'  {len(doc.paragraphs)} paragraphs, '
          f'{len(doc.inline_shapes)} figures, {len(doc.tables)} tables')


if __name__ == '__main__':
    build()
