"""Reframe 4.5 around what the arousal control supports.

The response stays; the attribution of its trigger does not. Two thirds of the
delta-burst onsets carry a scored arousal within 10 s, and on the arousal-free
subset the post-onset peak attenuates in four of five testable subjects while
holding in two. With 3 to 33 onsets surviving per session, lost averaging power
and lost effect cannot be separated, so the honest statement is that this
dataset cannot say whether the capacitive response follows the cortical event or
the arousal that accompanies it.

What is unaffected, and is said so explicitly: the response is low-frequency and
carries no sigma-band component either way, so 4.5's contribution to the
mechanical-not-electrical argument does not depend on which trigger is correct.

Numbers from analysis/delta_onset/outputs/arousal_control_q30.csv.

Run from the repo root:  python writeup/edits/apply_arousal_control.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_AROUSAL_20260817.docx")

CONTROL = (
    "One alternative account has to be tested before the cortical event can be called the "
    "trigger. A capacitive response peaking four to eight seconds after an event is also the "
    "latency of the cardiovascular response to a micro-arousal, and K-complexes, which dominate "
    "this onset set, frequently carry autonomic activation. Scored arousals are available for "
    "every recording, and they are common: between 47% and 84% of the qualifying onsets, a median "
    "of about two thirds, fall within 10 seconds of a scored cortical or autonomic arousal. "
    "Repeating the whole analysis on the arousal-free onsets alone — the same causal estimator, "
    "the same count-matched null, only the onset list changed — attenuates the response in four "
    "of the five subjects that retain enough events to test, from a mean post-onset peak of 0.65 "
    "to 0.10 z in one subject and 1.54 to 0.25 z in another, while two subjects hold at or above "
    "their original values. Only 3 to 33 onsets survive per recording and two recordings fall "
    "below the minimum entirely, so lost averaging power and lost effect cannot be separated at "
    "this sample size."
)

REFRAME = (
    "We therefore report a capacitive response in the seconds after a delta burst, and do not "
    "claim that the delta burst rather than its accompanying arousal is what elicits it. These "
    "data cannot separate the two. What the section does establish is unchanged by that "
    "ambiguity: the response is confined to the low, mechanical bands, the sigma band shows "
    "nothing, and the pre-onset baseline is flat under causal filtering — so whichever event is "
    "the trigger, what the temple sensor registers is a mechanical process rather than a direct "
    "electrical signature. Separating the two triggers would need a cohort with enough "
    "arousal-free slow-wave onsets to power the comparison, which six subjects do not provide."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # the section heading asserts the trigger
    d.set_text("4.5 Delta-burst onsets evoke a mechanical response in the capacitive signal",
               "4.5 A mechanical response follows delta-burst onsets, though not "
               "necessarily because of them")

    # 6/6 strength claim, stated before the control existed
    d.patch("Onset-triggered averaging reveals a sharp increase in capacitive power",
            "The response was present in every one of the six subjects for all nine "
            "channel-band combinations (6/6 subjects; Figure 8).",
            "The response was present in every one of the six subjects for all nine "
            "channel-band combinations (6/6 subjects; Figure 8), on the full onset set.")

    anchor = d.find("The capacitive response therefore follows the cortical onset")
    d.body.insert(anchor + 1, d.para(REFRAME))
    d.body.insert(anchor + 1, d.para(CONTROL))
    print("  4.5: arousal control and the reframed attribution inserted")

    # the conclusion inherits the causal wording
    d.patch("As with sleep spindles (Section 4.4), what the temple sensor registers",
            "registers at a delta-burst onset is a mechanical process",
            "registers around a delta-burst onset is a mechanical process")
    print("  4.5: 'at a delta-burst onset' -> 'around', matching what is claimed")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## 4.5 arousal control (2026-08-17)\n\n"
        + "- heading and text no longer attribute the response to the delta burst specifically\n"
        + "- the control is reported in the Results: 47-84% of onsets carry a scored arousal, "
          "and the arousal-free subset attenuates in 4 of 5 testable subjects\n"
        + "- the mechanical conclusion is stated as independent of which trigger is correct\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
