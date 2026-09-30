"""
Build the K-complex update deck for the professor.

`writeup/review/` is the home for material going out to him. This deck answers
his September question — "did you also compare our signals with K-complex in
EEG?" — using his own PSG's scored K-complex marks, and places the answer on
the Fultz et al. 2019 chain he pointed us at.

Five figure slides, each carrying the figure, a one-line caption, and a short
"what this says" block. Fuller prose is in the speaker notes of each slide and
in KCOMPLEX_UPDATE.md beside this script.

Style follows writeup/ppt/build_deck.py (16:9, same palette and margins) so the
two decks sit together.

Run from the repo root:
    .venv/Scripts/python.exe writeup/review/_build_kcomplex_deck.py
Output -> writeup/review/CAP_sleep_mask_kcomplex_update.pptx
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
ONSET_OUT = ROOT / 'analysis' / 'delta_onset' / 'outputs'
FIGS = ROOT / 'writeup' / 'figures'
OUT = Path(os.environ.get('DECK_OUT', HERE / 'CAP_sleep_mask_kcomplex_update.pptx'))

W_IN, H_IN = 13.333, 7.5
INK = RGBColor(0x1B, 0x2A, 0x41)
MUTED = RGBColor(0x5A, 0x64, 0x72)
FAINT = RGBColor(0xA8, 0xB0, 0xBA)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
BODY = RGBColor(0x2C, 0x3E, 0x50)

MARGIN = 0.45
TITLE_TOP = 0.30
CAPTION_TOP = 0.80
IMG_TOP = 1.26
IMG_BOTTOM = 5.72
TEXT_TOP = 5.84

TITLE = (
    'K-complexes and the capacitive sleep mask',
    'Answering "did you also compare our signals with K-complex in EEG?" '
    'using the PSG’s own scored K-complex marks',
    'Six participants, twelve overnight recordings · '
    'analysis/delta_onset/ · September 2026',
)

# (title, one-line caption, figure path, [what-this-says lines], speaker notes)
SLIDES = [
    (
        'What a K-complex is, in our own recordings',
        'All 50 K-complexes the sleep technologist scored across the cohort, '
        'from the PSG’s Spindle/K annotation channel.',
        ONSET_OUT / 'fig_kcomplex_morphology.png',
        ['A  eight individual marks   ·   B  the average of all 50: the textbook '
         'biphasic shape, sharp negative to −70 µV then positive to +65 µV, '
         'over by +0.7 s. Median 161 µV peak-to-peak.',
         'C  the two N2 graphoelements sit in different bands — K-complexes lift '
         'delta (0.5–4 Hz), spindles lift sigma (11–16 Hz). That difference is '
         'why the two tests are not interchangeable.'],
        'A K-complex is one of the two events that define N2 sleep, the other being '
        'the spindle. Physiologically it is an isolated cortical down-state: a large '
        'population of neurons falls silent together, then resumes. It is the single-'
        'event version of the slow oscillation that fills N3.\n\n'
        'These are not detections of ours. They are the marks the scoring '
        'technologist placed, read out of the Spindle/K channel of the PSG export. '
        'The grand average in B is the standard morphology, which confirms the marks '
        'and the alignment are sound.\n\n'
        'Panel C is the one that matters for the argument on the next slides. For '
        'spindles we can settle the question on frequency alone, because sigma sits '
        'far outside the capacitive band. A K-complex’s own frequency is 0.5–4 Hz, '
        'which overlaps the bands the mask measures, so for K-complexes the '
        'frequency argument is unavailable and timing has to carry it instead.',
    ),
    (
        'Our delta-burst detector picks up a selective fifth of them',
        'Every scored K-complex put through the two gates our detector applies.',
        ONSET_OUT / 'fig_kcomplex_vs_detector.png',
        ['C  a K-complex usually IS a multi-second delta run — median 3.5 s above '
         'the burst threshold (IQR 2.3–6.1), because one is typically followed by '
         'further slow waves rather than standing alone.',
         'D  44% clear the ≥4 s criterion, 50% have a quiet 30 s before them, '
         '18% pass both — and the detector finds all of those, 9/9. So the detected '
         'onsets are real K-complex-like events, but a selected subset.'],
        'This slide is a correction as much as a result. The expectation going in was '
        'that a K-complex, being a sub-second waveform, could not satisfy a '
        'four-second sustained-burst criterion, and therefore that our delta-burst '
        'onsets were a different kind of object. Measuring all 50 shows otherwise.\n\n'
        'What actually thins them is the pair of gates together, and the detector '
        'recovers every scored K-complex that meets its own definition.\n\n'
        'The honest description for the manuscript: the onsets are isolated N2 delta '
        'events, predominantly slow waves and K-complexes, selected for sitting at '
        'the head of a sustained delta run after a quiet baseline. Not "the '
        'K-complex population", and not something unrelated to it.\n\n'
        'One caveat on the reference set: the scorer annotated spindles exhaustively '
        '(21,881 cohort-wide) but K-complexes barely at all (57). So these 50 marks '
        'are good examples, not a complete ground truth.',
    ),
    (
        'K-complex → mask: a response that follows the cortical event',
        'The same three questions the spindle analysis asks, triggered on the '
        'scored K-complexes — a trigger independent of our own detector.',
        ONSET_OUT / 'fig_kcomplex_cap_response.png',
        ['A  low-frequency mask power rises after the K-complex in all three '
         'channels: CH +0.47 z, CLE +0.34 z, CRE +0.32 z, against a random-NREM null '
         'of +0.03 to +0.08 z. CH is largest — the same ordering the spindle result '
         'gives.',
         'B  the peak sits 3.6–4.3 s after the event while the simultaneous EEG '
         'delta peaks at t = 0. Electrical pickup would land at zero lag. It does not.'],
        'This is the direct answer to the September question, and it uses his own '
        'PSG annotations rather than our detector output, so it shares no machinery '
        'with the delta-burst analysis already in the manuscript. It reproduces that '
        'analysis’s direction and latency on an independent trigger.\n\n'
        'The reason latency carries the argument here rather than frequency: a '
        'K-complex is a delta event and the mask measures delta-band power, so a '
        'delta-band response cannot by itself distinguish electrical pickup from a '
        'mechanical response. A three-to-four second displacement can. Electrical '
        'coupling is instantaneous; a hemodynamic response is not.\n\n'
        'Caveats to state plainly. n = 50 across ten usable sessions — S4N2 and '
        'S5N1 contribute none, S1N2 contributes one. Panel C shows per-session peaks '
        'ranging 0.01 to 1.77 z, so this is descriptive and per-session, with no '
        'pooled p-value. And the arousal confound of Section 3.6 is unchanged: '
        'K-complexes are frequently arousal-associated, and an arousal produces its '
        'own hemodynamic transient on this timescale.',
    ),
    (
        'How this connects to Fultz et al. 2019',
        'Their measurement is a lag, not a correlation — and that is what makes '
        'our earlier zero-lag negative interpretable.',
        FIGS / 'delta_onset' / 'fig_fultz_chain.png',
        ['Fultz followed one event through three stages: cortical slow wave → '
         'cortical blood volume falls → CSF flows into the fourth ventricle, with '
         'EEG leading the CSF inflow by about 6.4 s.',
         'Our 1.9–4.3 s sits between the cortical event and the ventricular '
         'response — which is where a sensor measuring cranial displacement should '
         'sit. Same direction, one step earlier in the chain.'],
        'Fultz et al., Science 2019, recorded EEG and fMRI simultaneously in NREM '
        'sleep. The chain: neurons fall silent together, so metabolic demand drops, '
        'so cortical blood volume falls; the skull is a closed box, so the volume has '
        'to be replaced, and CSF flows in. This is a large part of why slow-wave '
        'sleep is thought to matter for glymphatic clearance.\n\n'
        'The critical detail is that they found NO correlation between EEG and CSF '
        'flow at zero lag. The coupling only appears once a delay is allowed. That '
        'directly explains our own earlier negative result: the SWA validation '
        'compared capacitive delta power with EEG delta power at zero lag and found '
        'r ≈ 0.015. Under Fultz that is the expected outcome and says nothing '
        'about downstream coupling.\n\n'
        'Looking in the delayed window instead, the response is there, and its '
        'latency places the mask between the cortical event and the ventricular CSF '
        'response — exactly where a mechanical cranial sensor belongs. That gives '
        'the mechanism a name instead of leaving the paper to assert "mechanical, '
        'not electrical" on the strength of the spindle result alone.\n\n'
        'The schematic is drawn from scratch. The only numbers taken from Fultz are '
        'the ~6.4 s lag and the ordering of the stages.',
    ),
]

CLOSING = (
    'Where this leaves us',
    [
        ('Solid.', 'The mask responds to K-complexes, the response is low-frequency, '
                   'and it follows the cortical event by 3–4 s rather than '
                   'coinciding with it. Shown on an independent trigger, consistent '
                   'with the 340-event delta-onset analysis already in the '
                   'manuscript, and consistent in direction with Fultz.'),
        ('Not settled.', 'Whether the K-complex or its accompanying arousal is the '
                         'trigger. Between 47% and 84% of onsets fall within 10 s of '
                         'a scored arousal, and an arousal has its own hemodynamic '
                         'transient on this timescale. Separating them needs a larger '
                         'cohort with enough arousal-free slow-wave onsets.'),
        ('A limitation of the reference, not the method.',
         'The PSG’s K-complex channel holds 57 marks cohort-wide against 21,881 '
         'spindles, so it is not an exhaustive scoring. It supports these 50 events '
         'as examples; it cannot support a detection-rate claim.'),
        ('Proposed for the manuscript.',
         'Report this as a short addition to the spindle section, and use the Fultz '
         'chain in §4.4 — which currently carries a "Comparison with Existing '
         'Methods" heading over a paragraph that declines to compare.'),
    ],
)


def textbox(slide, text, left, top, width, height, size, color,
            bold=False, align=PP_ALIGN.LEFT, space_after=0):
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
    if space_after:
        p.space_after = Pt(space_after)
    return tb


def bullets(slide, lines, left, top, width, height, size):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(5)
        r = p.add_run()
        r.text = '•  ' + line
        r.font.size = Pt(size)
        r.font.color.rgb = BODY
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


def place_image(slide, png, top, bottom):
    w_px, h_px = Image.open(png).size
    box_w, box_h = W_IN - 2 * MARGIN, bottom - top
    scale = min(box_w / w_px, box_h / h_px)
    w, h = w_px * scale, h_px * scale
    slide.shapes.add_picture(str(png), Inches((W_IN - w) / 2),
                             Inches(top + (box_h - h) / 2), Inches(w), Inches(h))


def main():
    missing = [str(f) for _, _, f, _, _ in SLIDES if not Path(f).exists()]
    if missing:
        sys.exit('figures not found:\n  ' + '\n  '.join(missing))

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W_IN), Inches(H_IN)
    blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)
    textbox(s, TITLE[0], MARGIN, 2.45, W_IN - 2 * MARGIN, 1.0, 38, INK, bold=True)
    textbox(s, TITLE[1], MARGIN, 3.35, W_IN - 2 * MARGIN, 0.9, 19, MUTED)
    rule(s, MARGIN, 4.34, 1.6)
    textbox(s, TITLE[2], MARGIN, 4.58, W_IN - 2 * MARGIN, 0.5, 13, FAINT)

    for n, (title, caption, fig, says, notes) in enumerate(SLIDES, start=1):
        s = prs.slides.add_slide(blank)
        textbox(s, title, MARGIN, TITLE_TOP, W_IN - 2 * MARGIN, 0.5, 23, INK, bold=True)
        textbox(s, caption, MARGIN, CAPTION_TOP, W_IN - 2 * MARGIN, 0.42, 13, MUTED)
        place_image(s, fig, IMG_TOP, IMG_BOTTOM)
        bullets(s, says, MARGIN, TEXT_TOP, W_IN - 2 * MARGIN, 1.3, 12.5)
        textbox(s, Path(fig).name, MARGIN, 7.18, 8.0, 0.3, 8, FAINT)
        textbox(s, str(n), W_IN - MARGIN - 1.0, 7.18, 1.0, 0.3, 9, FAINT,
                align=PP_ALIGN.RIGHT)
        s.notes_slide.notes_text_frame.text = notes

    # ── closing slide ────────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    textbox(s, CLOSING[0], MARGIN, TITLE_TOP, W_IN - 2 * MARGIN, 0.5, 26, INK, bold=True)
    rule(s, MARGIN, 0.95, 1.3)
    top = 1.35
    for head, body in CLOSING[1]:
        textbox(s, head, MARGIN, top, 3.1, 0.4, 15, ACCENT, bold=True)
        textbox(s, body, MARGIN + 3.15, top, W_IN - MARGIN - 3.6, 1.35, 13.5, BODY)
        top += 1.42
    textbox(s, str(len(SLIDES) + 1), W_IN - MARGIN - 1.0, 7.18, 1.0, 0.3, 9, FAINT,
            align=PP_ALIGN.RIGHT)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f'wrote {OUT}')
    print(f'  {len(prs.slides._sldIdLst)} slides, {OUT.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
