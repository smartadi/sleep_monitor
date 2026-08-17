"""State the cortical-event finding plainly, in one consistent form.

The paper had been saying "the mask carries no cortical electrographic
signature", which reads as though the sensor shows nothing at a spindle or a
delta burst. It shows a lot. The point is what kind of thing it shows: a
mechanical process that accompanies the cortical event, not a direct
electrical signature of it.

Rewords the five places the claim is made -- introduction, 4.4, 4.5, 5.2,
conclusion -- plus the abstract paragraph that leads with it. No numbers
change; this is wording only.

Run from the repo root:  python writeup/edits/apply_mechanical_wording.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_MECHWORDING_20260817.docx")

log = []


def note(msg):
    log.append(msg)
    print("  " + msg)


ABSTRACT_3 = (
    "What the mask picks up at a cortical event is a mechanical process, not a direct electrical "
    "signature. Sleep spindles produce no sigma-band response on any capacitive channel (+0.02 to "
    "+0.03 dB) where the same measurement on contact EEG rises +3.3 dB. What rises instead, at "
    "both spindles and delta-burst onsets, is low-frequency power: +0.47 to +0.58 dB of 0–3 Hz "
    "power at spindle centres, and a sharp post-onset rise at delta-burst onsets in six of six "
    "subjects with a flat pre-onset baseline under strictly causal filtering. The sensor responds "
    "to cortical events, in other words, by way of the mechanical and hemodynamic changes that "
    "accompany them. Band-restricted spectral ridges carry interpretable but weak stage "
    "associations, respiratory-band ridges being fewer during N3 in all six subjects. The mask is "
    "therefore a transducer of mechanical and hemodynamic pulsation, suited to unobtrusive "
    "overnight mean-rate trending and to multi-modal sleep assessment as one input, and not to "
    "instantaneous rate tracking or to cortical monitoring."
)

D52 = (
    "The mask does not pick up cortical electrical activity. It does respond at cortical events, "
    "and clearly so: low-frequency power rises at spindle centres and again at delta-burst onsets. "
    "But that response is mechanical. The sigma rhythm that carries the spindle produces no change "
    "on any capacitive channel, where the same measurement on the contact EEG rises +3.3 dB, and "
    "the delta-burst response follows the cortical onset at zero lag rather than preceding it. "
    "What the sensor registers is a mechanical process that accompanies cortical activity, not a "
    "direct electrical signature of it, and this is the clearest boundary the study establishes."
)


def main():
    if not DOC.exists():
        raise SystemExit("missing %s" % DOC)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ---------------------------------------------------------------- abstract
    d.set_text("The mask carries no cortical electrographic signature. Sleep spindles produce no "
               "sigma-band response", ABSTRACT_3)
    note("abstract: leads with the mechanical reading instead of the electrographic negative")

    # ------------------------------------------------------------ introduction
    d.patch("In this study we characterise what a wearable SEC sleep mask measures",
            "whether the signal carries any correlate of cortical activity",
            "whether the signal carries any electrical correlate of cortical activity")
    d.patch("In this study we characterise what a wearable SEC sleep mask measures",
            "The answer to the last question is negative and defines the scope of the technology.",
            "It does not: at both events the sensor registers a mechanical process that accompanies "
            "the cortical activity rather than a direct electrical signature of it, and that "
            "defines the scope of the technology.")
    note("1. Introduction: the question is electrical pickup, and the answer says what is picked up")

    # ------------------------------------------------------------ 4.4 spindles
    d.patch("This response lies entirely in the low, mechanical band",
            "What the temple sensor registers at a spindle is therefore not the cortical "
            "electrical rhythm but its mechanical and hemodynamic accompaniment",
            "What the temple sensor registers at a spindle is therefore a mechanical process, "
            "not a direct electrical signature")
    d.patch("This response lies entirely in the low, mechanical band",
            "—consistent with a sensor that transduces intracranial mechanical and "
            "hemodynamic pulsations rather than neuronal electrical activity.", ".")
    note("4.4: plain statement, trailing restatement dropped")

    # -------------------------------------------------------- 4.5 delta bursts
    d.patch("The capacitive response therefore follows the cortical onset",
            "what the temple sensor registers at a delta-burst onset is therefore not the "
            "cortical electrical rhythm but its mechanical and hemodynamic accompaniment",
            "what the temple sensor registers at a delta-burst onset is therefore a mechanical "
            "process, not a direct electrical signature")
    d.patch("The capacitive response therefore follows the cortical onset",
            " - consistent with a sensor that transduces intracranial mechanical and hemodynamic "
            "pulsations rather than neuronal electrical activity.", ".")
    note("4.5: plain statement, trailing restatement dropped")

    # ----------------------------------------------------------- 5.2 discussion
    d.set_text("The mask carries no cortical electrographic signature. Sleep spindles produce no "
               "spindle-locked sigma response", D52)
    note("5.2: rewritten to lead with the response, then say what kind of response it is")

    # ------------------------------------------------------------- 7 conclusion
    d.patch("This study provides a multi-method characterization",
            "The sensor carries no cortical electrographic signature: sleep spindles produce no "
            "sigma-band response, confirming that it transduces mechanical and hemodynamic "
            "pulsations rather than neuronal electrical activity.",
            "The sensor responds at cortical events but does not pick up their electrical "
            "signature: spindles produce no sigma-band response, so what the mask measures is a "
            "mechanical process accompanying cortical activity rather than the activity itself.")
    note("7: conclusion matched to the same wording")

    d.save(DOC)
    print("\n%d edit operations applied" % d.n_patch)
    print("wrote %s" % DOC)
    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Mechanical-wording pass (2026-08-17)\n\n"
        + "\n".join("- " + m for m in log) + "\n", encoding="utf-8")
    print("appended to writeup/edits/WRAPUP_CHANGELOG.md")


if __name__ == "__main__":
    main()
