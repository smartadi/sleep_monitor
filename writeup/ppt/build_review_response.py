"""Build the reviewer-response deck: one marked issue per section, plain language.

The earlier review deck (`build_deck.py`) was figures-only by request -- title,
one line, plot.  This deck answers a different brief: the reviewer marked
specific slides in pen, and for each mark we say in ordinary words what he
pointed at, what was actually wrong, what we changed, and what the numbers say
afterwards.  So every figure slide here carries an explanation panel beside it.

Rule for this deck: no number appears unless it came out of a script in this
repository, and no claim appears that the data does not carry.  Where a mark has
not been decoded yet, it is listed as undecoded rather than guessed at.

Run from the repo root:  .venv/Scripts/python.exe writeup/ppt/build_review_response.py
Output -> writeup/ppt/CAP_sleep_mask_review_response_v1.pptx
"""

import os
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "writeup" / "figures"
OUT = Path(os.environ.get("DECK_OUT",
                          Path(__file__).resolve().parent /
                          "CAP_sleep_mask_review_response_v1.pptx"))

W_IN, H_IN = 13.333, 7.5
INK = RGBColor(0x1B, 0x2A, 0x41)
BODY = RGBColor(0x2E, 0x3A, 0x4A)
MUTED = RGBColor(0x5A, 0x64, 0x72)
FAINT = RGBColor(0xA8, 0xB0, 0xBA)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GOOD = RGBColor(0x1B, 0x7A, 0x43)

MARGIN = 0.5
TITLE_TOP = 0.32
CAPTION_TOP = 0.92


def textbox(slide, x, y, w, h, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = align
    return tf


def para(tf, text, size=14, color=BODY, bold=False, space_before=6,
         first=False, bullet=False, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_before = Pt(space_before)
    r = p.add_run()
    r.text = ("•  " + text) if bullet else text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return p


def title_bar(slide, title, caption=None):
    tf = textbox(slide, MARGIN, TITLE_TOP, W_IN - 2 * MARGIN, 0.6)
    para(tf, title, size=26, color=INK, bold=True, first=True, space_before=0)
    if caption:
        tf2 = textbox(slide, MARGIN, CAPTION_TOP, W_IN - 2 * MARGIN, 0.45)
        para(tf2, caption, size=13.5, color=MUTED, first=True, space_before=0)


def fit(path, box_x, box_y, box_w, box_h):
    """Largest rectangle inside the box that preserves the image aspect."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(box_w / iw, box_h / ih)
    w, h = iw * scale, ih * scale
    return box_x + (box_w - w) / 2, box_y + (box_h - h) / 2, w, h


def slide_blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def section(prs, n, title, sub):
    s = slide_blank(prs)
    tf = textbox(s, MARGIN + 0.3, 2.8, W_IN - 2 * MARGIN - 0.6, 1.8)
    para(tf, n, size=15, color=ACCENT, bold=True, first=True, space_before=0)
    para(tf, title, size=34, color=INK, bold=True, space_before=10)
    para(tf, sub, size=15, color=MUTED, space_before=12)
    return s


def bullets_slide(prs, title, caption, items):
    s = slide_blank(prs)
    title_bar(s, title, caption)
    tf = textbox(s, MARGIN + 0.1, 1.65, W_IN - 2 * MARGIN - 0.2, 5.2)
    first = True
    for it in items:
        if isinstance(it, tuple):
            head, detail = it
            para(tf, head, size=17, color=INK, bold=True,
                 space_before=0 if first else 16, first=first)
            para(tf, detail, size=14, color=BODY, space_before=3)
        else:
            para(tf, it, size=15, color=BODY, bullet=True,
                 space_before=0 if first else 8, first=first)
        first = False
    return s


def figure_slide(prs, title, caption, fig_rel, heading, lines, note=None,
                 body_size=12.5):
    """Figure on the left, plain-language explanation on the right."""
    s = slide_blank(prs)
    title_bar(s, title, caption)
    path = FIGS / fig_rel
    if not path.exists():
        raise SystemExit("missing figure: %s" % path)
    x, y, w, h = fit(path, MARGIN, 1.5, 8.1, 5.5)
    s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))

    tf = textbox(s, 8.85, 1.5, W_IN - 8.85 - MARGIN, 5.5)
    para(tf, heading, size=16, color=INK, bold=True, first=True, space_before=0)
    for ln in lines:
        para(tf, ln, size=body_size, color=BODY, space_before=8)
    if note:
        para(tf, note, size=body_size - 1, color=ACCENT, space_before=12,
             italic=True)

    tf3 = textbox(s, MARGIN, H_IN - 0.42, 8.0, 0.3)
    para(tf3, "writeup/figures/" + fig_rel, size=9, color=FAINT, first=True,
         space_before=0)
    return s


def table_slide(prs, title, caption, headers, rows, note=None):
    s = slide_blank(prs)
    title_bar(s, title, caption)
    nr, nc = len(rows) + 1, len(headers)
    shape = s.shapes.add_table(nr, nc, Inches(MARGIN + 0.4), Inches(1.75),
                               Inches(W_IN - 2 * MARGIN - 0.8),
                               Inches(0.42 * nr))
    tbl = shape.table
    for j, htxt in enumerate(headers):
        c = tbl.cell(0, j)
        c.text = htxt
        p = c.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(13)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = INK
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(val)
            p = c.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(12.5)
            p.runs[0].font.color.rgb = BODY
    if note:
        tf = textbox(s, MARGIN + 0.4, 1.9 + 0.42 * nr + 0.25,
                     W_IN - 2 * MARGIN - 0.8, 1.4)
        para(tf, note, size=13, color=ACCENT, first=True, space_before=0)
    return s


# ── content ─────────────────────────────────────────────────────────────────

def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W_IN), Inches(H_IN)

    # 1 — title
    s = slide_blank(prs)
    tf = textbox(s, MARGIN + 0.3, 2.5, W_IN - 2 * MARGIN - 0.6, 2.6)
    para(tf, "CAP sleep mask", size=40, color=INK, bold=True, first=True,
         space_before=0)
    para(tf, "What we claim, and how we test it", size=22, color=MUTED,
         space_before=14)
    para(tf, "Response to the marked review deck  ·  version 1",
         size=14, color=FAINT, space_before=22)
    para(tf, "Every number in this deck comes from a script in the repository. "
             "Marks not yet decoded are listed as such, not guessed at.",
         size=12.5, color=ACCENT, space_before=8, italic=True)

    # 2 — the claim
    section(prs, "PART ONE", "The claim",
            "One sentence, two testable halves, and an explicit scope.")

    bullets_slide(
        prs, "The simplest claim the data supports",
        "Everything else in the paper is either support for this, or a stated limit on it.",
        [("“A capacitive sensor in a sleep mask records breathing, and the "
          "strength of that signal changes with how deeply the person is asleep.”",
          "Six people, two nights each, recorded alongside full clinical PSG."),
         ("Half A — it is really breathing",
          "The rhythm in the mask signal is the same rhythm the PSG breathing "
          "sensors record, not a look-alike rhythm of the same rate."),
         ("Half B — it changes with sleep depth",
          "The size of the mask signal falls as the person goes from wake into "
          "deep sleep, and rises again in REM."),
         ("Why these two and nothing more",
          "These are the only two statements that hold in every subject we "
          "recorded, and that survive being tested against a control.")])

    bullets_slide(
        prs, "What we do not claim",
        "Stating these up front is cheaper than having them found.",
        [("Not a heart-rate measurement",
          "The cardiac evidence is weak. In the one session where the mask was "
          "worn at the same time as an optical pulse sensor, the optical sensor "
          "showed a clear pulse at 1.08 Hz and the mask showed no pulse line at all."),
         ("Not an accuracy claim",
          "Six people is too few to put numbers on agreement. No limits of "
          "agreement, no error bars presented as performance."),
         ("Not a sleep stager",
          "The signal changes with sleep depth. That is a description, not a classifier."),
         ("Not one interchangeable channel",
          "CH and CLE−CRE differ in scale and are only partly coherent. "
          "We report each channel separately rather than pooling them.")])

    bullets_slide(
        prs, "How we test it honestly",
        "Five rules applied to every result in the paper.",
        [("1.  Compare against a sensor that should see it, and one that should not",
          "Both are contact sensors on the same body at the same time, so anything "
          "that inflates one inflates the other. Only a real rhythm moves the difference."),
         ("2.  Break the timing and check the effect disappears",
          "The same comparison is repeated with the reference signal time-shifted "
          "and time-reversed. That keeps the frequency content and destroys the timing."),
         ("3.  Count people, not epochs",
          "The unit is the subject: six of them. Two nights from one person are "
          "not two independent measurements, and thousands of 30-second epochs "
          "from one night are not thousands of measurements."),
         ("4.  Compare against the mask when nobody is wearing it",
          "A recording from the same hardware, unworn, sets the floor for how much "
          "of any effect is the instrument."),
         ("5.  Report every channel separately",
          "CH, CLE, CRE and CLE−CRE are shown side by side, including where "
          "they disagree.")])

    # 3 — the first decoded mark
    section(prs, "PART TWO", "Marked slide 7 — coherence with the PSG sensors",
            "He bracketed Flow / Thorax / Abdomen apart from Pleth / ECG. "
            "That bracket is a test the figure was not running.")

    figure_slide(
        prs, "What the figure used to show",
        "The original coherence figure, as it appeared in the reviewed deck.",
        "coupling/cap_psg_coherence.png",
        "The problem he found",
        ["The six PSG sensors are drawn as one flat sequence, but they are not "
         "one kind of thing. Flow, Thorax and Abdomen measure breathing. Pleth "
         "and ECG measure the heartbeat. EEG was the intended control.",
         "The whole claim rested on the gap between a target sensor and EEG — "
         "on the assumption that a scalp electrode carries no breathing or "
         "heartbeat mechanics.",
         "It does. Breathing movement, pulse artifact and ballistocardiogram all "
         "reach the scalp. So EEG sits close to the targets and the gap is small.",
         "Tested that way, the forehead channel CH fails: its breathing-band "
         "margin over EEG is not consistent across subjects."],
        note="The y-axes also start above zero, which makes a difference of "
             "0.02 look larger than it is.")

    figure_slide(
        prs, "The same data, grouped the way he marked it",
        "Sensors grouped by what they measure; the shaded block is the group that "
        "should win in that band.",
        "coupling/cap_psg_coherence_grouped.png",
        "How to read this",
        ["Top row: the same coherence values, but the breathing sensors and the "
         "heartbeat sensors are now separated, and the axis starts at zero so the "
         "sizes are honest.",
         "The dotted red line is the level a coherence value reaches by chance, "
         "set by how many segments went into it. Anything near that line means "
         "nothing. Breathing values sit well above it; cardiac values sit just "
         "on top of it.",
         "Bottom row: for each subject we take the mask's coherence with the "
         "sensors that measure that rhythm, minus its coherence with the sensors "
         "that do not. One dot per person. Above the line means the mask is "
         "picking up that specific rhythm.",
         "Breathing band: every channel is above the line in all six people, "
         "CH included. This is the test CH failed before, and it passes here "
         "because the comparison no longer depends on EEG being clean.",
         "Heartbeat band: only CH is above the line in all six. For CRE the "
         "margin is 0.0005, which is nothing."],
        note="A difference between two coherences computed the same way is not "
             "affected by that chance floor — the bias is in both terms and "
             "cancels. That is why the bottom row is the result and the top row "
             "is context.",
        body_size=11.5)

    table_slide(
        prs, "The numbers behind that figure",
        "Matched sensor class minus unmatched sensor class. Unit = one subject, "
        "two nights averaged.",
        ["Band", "Channel", "Matched", "Unmatched", "Margin", "Subjects above 0"],
        [["Breathing", "CH", "0.127", "0.111", "+0.016", "6 / 6"],
         ["Breathing", "CLE", "0.136", "0.102", "+0.031", "6 / 6"],
         ["Breathing", "CRE", "0.109", "0.091", "+0.014", "6 / 6"],
         ["Breathing", "CLE−CRE", "0.133", "0.103", "+0.021", "6 / 6"],
         ["Heartbeat", "CH", "0.075", "0.066", "+0.011", "6 / 6"],
         ["Heartbeat", "CLE", "0.075", "0.065", "+0.006", "5 / 6"],
         ["Heartbeat", "CRE", "0.062", "0.061", "+0.000", "5 / 6"],
         ["Heartbeat", "CLE−CRE", "0.066", "0.062", "+0.003", "5 / 6"]],
        note="We deliberately do not headline a p-value: with six subjects the "
             "smallest a two-sided Wilcoxon can return is 0.031, which is exactly "
             "what “6 out of 6 in the same direction” gives and nothing more. "
             "The sign count is the honest summary.")

    bullets_slide(
        prs, "What this changes in manuscript V6",
        "Four edits from this mark — and they line up with critique item M2, "
        "which was written independently.",
        [("Replace the coherence figure",
          "The grouped version replaces the original in Section 3.2. Generated by "
          "analysis/rates/cap_psg_coherence_crossclass.py."),
         ("Drop the claim that EEG carries no respiratory or cardiac mechanics",
          "It is not defensible, and it was the load-bearing assumption of the "
          "old specificity test."),
         ("Lead with margins, not absolute coherence — this is M2's repair",
          "M2 says the absolute values sit inside their own chance floor and the "
          "margins are the real effect. The matched-minus-unmatched contrast is a "
          "third margin, and a stronger one than the EEG margin M2 settles for, "
          "because it needs no assumption about EEG."),
         ("State the cardiac result as weak, in the text",
          "Only CH is consistent across all six subjects in the heartbeat band; "
          "the margins are far smaller than in the breathing band, and the "
          "absolute values sit on the chance floor.")])

    # 4 — what is still unread
    section(prs, "PART THREE", "Marks not yet decoded",
            "Fifteen of his sixteen marked slides are pen strokes we have not read yet.")

    bullets_slide(
        prs, "Where the remaining marks are",
        "The marks are digital ink, so they carry no text we can extract — "
        "they have to be read off the rendered slides.",
        ["Slides 71–79, 81 — sensor value across the night, one per recording "
         "(level, imbalance and band amplitude on the hypnogram)",
         "Slide 88 — CH against the CLE−CRE differential",
         "Slide 89 — capacitance imbalance, all twelve recordings  "
         "(16 stroke groups: the most heavily marked slide in the deck)",
         "Slide 90 — band amplitude by sleep stage",
         "Slide 91 — how much of that variance is the instrument",
         "Slide 92 — where the high-variance epochs fall  (13 stroke groups)",
         "Together these are one theme: channel choice, capacitance imbalance, "
         "and whether the sleep-depth effect is physiology or instrument. "
         "That is the second half of the claim on slide 3."])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print("wrote %s  (%d slides)" % (OUT, len(prs.slides.__iter__.__self__._sldIdLst)))


if __name__ == "__main__":
    build()
