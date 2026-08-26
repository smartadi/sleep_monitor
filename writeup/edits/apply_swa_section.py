"""Add the slow-wave activity validation as 4.6.

Spindles (4.4) and delta bursts (4.5) test discrete events. Slow-wave activity
is the continuous measure, and it is the one a reader motivated by the
introduction will most expect a capacitive sleep sensor to deliver. The answer
is a clean null, and it is interpretable because the same pipeline run on the
contact EEG returns a clear positive: the analysis works, the sensor does not
carry the quantity.

Placed before the harmonic-comb section so the three cortical-signal tests sit
together; the comb section becomes 4.7 and Figures 9 and 10 become 10 and 11.

Numbers from analysis/swa_validation/outputs/swa_validation_results.csv.

Run from the repo root:  python writeup/edits/apply_swa_section.py
"""

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_SWA_20260817.docx")
FIG = Path("writeup/figures/swa/swa_validation_paper.png")

METHOD = (
    "Slow-wave activity. The event analyses above ask what happens at a moment; slow-wave "
    "activity asks whether a continuous cortical measure is present at all. Following the "
    "published replication design, both the capacitive and the contact-EEG signals were "
    "processed through one pipeline: 0.5 Hz high-pass, division into 6-second epochs, power "
    "spectral density per epoch, and band powers over 1–4.5 Hz and the 1–2, 2–3 and 3–4 Hz "
    "sub-bands, with epochs rejected on an amplitude criterion (4.9% of epochs on average). "
    "Capacitive slow-wave activity was then compared with EEG slow-wave activity epoch by epoch "
    "within each recording, and each was used on its own to discriminate technologist-scored N3 "
    "from all other sleep. Running the identical pipeline on the EEG makes the comparison "
    "interpretable: it is a positive control for the analysis, not a sanity check."
)

RESULT_A = (
    "Across 46,597 analysis epochs in the twelve recordings, capacitive slow-wave activity does "
    "not track the EEG's. The within-recording correlation between the two averages r = −0.014 "
    "(SD 0.036) over the 1–4.5 Hz band, and no sub-band does better: 1–2 Hz gives +0.015, 2–3 Hz "
    "+0.003 and 3–4 Hz +0.020, each with a standard deviation near 0.04 across recordings "
    "(Figure 9B). Magnitude-squared coherence between the two slow-wave envelopes averages 0.003. "
    "These are not weak correlations; they are the absence of one."
)

RESULT_B = (
    "The same holds when each signal is asked to find deep sleep on its own. Discriminating "
    "technologist-scored N3 from other sleep, capacitive slow-wave activity reaches an area under "
    "the curve of 0.490 (SD 0.040, range 0.427 to 0.567) — chance — while the contact EEG through "
    "the identical pipeline reaches 0.740 (SD 0.056, range 0.655 to 0.809). Every recording rises "
    "from the capacitive value to the EEG value (Figure 9A). The positive control therefore works "
    "in every night in which the capacitive arm fails, which places the failure in the sensor "
    "rather than in the analysis."
)

RESULT_C = (
    "One bound belongs with this. N3 is scarce in these recordings — 392 minutes in total, and "
    "between 7 and 65 minutes per night — so a single recording's area under the curve is not a "
    "precise quantity. The correlation result does not depend on how much N3 a night contains, "
    "and it is the stronger of the two. Together with the spindle and delta-burst sections, this "
    "completes the same statement in three independent ways: the mask does not carry cortical "
    "electrical activity, whether that activity is measured as a discrete event or as a "
    "continuous rhythm."
)

CAPTION = (
    "Figure 9. Slow-wave activity, capacitive against contact EEG, through one pipeline. "
    "(A) Discrimination of technologist-scored N3 from other sleep, per recording, capacitive "
    "against EEG; grey lines join the two values from the same night, and the dashed line is "
    "chance. (B) Correlation between capacitive and EEG slow-wave activity across the 6-second "
    "epochs of each recording, 1–4.5 Hz."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)

    d = Doc(DOC)
    body = d.body

    # figures 9 and 10 make room for the new figure 9
    n = 0
    for t in d.root.iter(W + "t"):
        if not t.text or "igure" not in t.text:
            continue
        new = re.sub(r"\bFigure 10\b", "\x01", t.text)
        new = re.sub(r"\bFigure 9\b", "Figure 10", new)
        new = new.replace("\x01", "Figure 11")
        if new != t.text:
            t.text = new
            n += 1
    print("  Figures 9 and 10 renumbered to 10 and 11 (%d runs)" % n)

    d.set_text("4.6 Harmonic-comb ladder episodes follow REM in consolidated N2",
               "4.7 Harmonic-comb ladder episodes follow REM in consolidated N2")

    # methods paragraph joins the cortical-event methods
    at = d.find("For both event types the capacitive power was averaged")
    body.insert(at + 1, d.para(METHOD))
    print("  3.7: slow-wave activity method added")

    # the section itself, before the comb section
    at = d.find("4.7 Harmonic-comb ladder episodes")
    img_template = body[next(i for i in range(at, 0, -1)
                             if body[i].findall(".//" + W + "drawing"))]
    block = [
        d.para("4.6 Slow-wave activity: a continuous cortical measure the mask does not carry",
               "Heading2"),
        d.para(RESULT_A), d.para(RESULT_B), d.para(RESULT_C),
        d.image_para(FIG, img_template), d.para(CAPTION), d.para(""),
    ]
    for off, el in enumerate(block):
        body.insert(at + off, el)
    print("  4.6 added with Figure 9")

    d.save(DOC)
    print("wrote %s" % DOC)


main()
