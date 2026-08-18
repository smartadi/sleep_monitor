"""Rewrite 4.1 in the voice of a physiology paper rather than a statistics one.

Nothing is dropped: every number, control and caveat survives. What changes is
what each sentence leads with. The section had grown to open paragraphs on
estimator bias, null distributions and noise floors, with the biology arriving
late or by implication. Here each paragraph opens with what the sensor does or
does not see, and the statistical apparatus follows as support.

Two paragraphs of controls are merged into one, because they were two halves of
a single argument -- that a coherence estimate is not zero under independence,
so the margin over a control is the quantity to read.

Run from the repo root:  python writeup/edits/apply_41_voice.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_41VOICE_20260817.docx")

SNR = (
    "Breathing and the heartbeat are present in the capacitive signal throughout the night, and "
    "they are not faint. Between a third and a half of the signal power below 5 Hz lies in the "
    "respiratory band (29–48%), and a further 8–48% in the cardiac band. Measured against the "
    "sensor's own electronic noise floor — taken from an independent no-subject recording and "
    "flat from 0.1 to 50 Hz — the physiological band stands 30.0 dB above it on the forehead and "
    "18.7 and 20.7 dB on the left and right temples, with no single recording falling below "
    "6.4 dB (Figure 2). The forehead was the strongest site in every subject. The temples varied "
    "more from night to night, which follows the sensor's contact with the skin rather than any "
    "change in the noise, since the floor itself was near-constant across sessions. Individual "
    "physiology shows through the variation: one night is almost entirely respiratory (S6N1, 48% "
    "respiratory against 8% cardiac) while another is dominated by the pulse (S3N1, 48% cardiac)."
)

COHERENCE = (
    "Energy in a band is not by itself evidence that the rhythm is the body's. We therefore asked "
    "whether the capacitive signal oscillates in step with the polysomnograph, at the frequency "
    "the polysomnograph reports. It does. At the reference rate the median coherence is 0.31 for "
    "breathing and 0.16 for the heartbeat, against 0.61 and 0.27 between two genuine PSG sensors "
    "measuring the same rhythm — so the mask reaches roughly half the agreement two clinical "
    "sensors reach with each other. Against phase-randomized surrogates, which keep each signal's "
    "spectrum but destroy its timing, 14.7% of respiratory and 9.1% of cardiac epochs exceed the "
    "null where 5% would be expected by chance. The coupling is therefore real in aggregate and "
    "weak epoch by epoch: on most single epochs it cannot be separated from the surrogate, so "
    "coherence should not be used to accept or reject an individual epoch. It persists in every "
    "sleep stage, with no stage falling to the null rate."
)

CONTROLS = (
    "Two controls fix how much of that 0.31 is physiology, because a coherence estimate is not "
    "zero when two signals are unrelated: from N averaged segments it returns roughly 1/N "
    "whatever the inputs, and the respiratory estimate has only three to five segments inside a "
    "30-second epoch. Pairing the same capacitive channel with a circularly shifted copy of the "
    "reference scores 0.193, and with a time-reversed copy 0.246, against 0.302 for the aligned "
    "pair. The second control is a sensor rather than a manipulation: the contact EEG is recorded "
    "simultaneously, shares the recording environment and any disturbance common to the "
    "instruments, and carries no respiratory or cardiac mechanics. It reaches 0.222 in the "
    "respiratory band and 0.065 in the cardiac band, against 0.302 and 0.126 for the "
    "physiological references. The margin over that control — 0.062 for breathing, positive in 11 "
    "of the 12 recordings (Wilcoxon p = 0.001), and 0.059 for the heartbeat, positive in all 12 "
    "(p < 0.001) — is the physiological part, and it is what should be read rather than the "
    "absolute value. Repeating the whole comparison at a five-minute window, which averages about "
    "30 segments, lowers every absolute value while preserving the direction and significance of "
    "both margins, so the result is not an artifact of the epoch length."
)

CONTINUITY = (
    "The two rhythms can be followed as single continuous traces from the first hour of a "
    "recording to the last. Rows F and G of Figure 3 carry one Viterbi path per rhythm on each "
    "difference channel: breathing holds near 0.25–0.3 Hz and the heartbeat near 0.9–1.2 Hz for "
    "the whole night, each with its own slow drift, and neither collapses onto a subharmonic. "
    "That a dry, non-contact electrode pair at the temples yields two continuously followable "
    "rhythms across a night is the basic signal-validation result of this study. It is a claim "
    "about continuity and not about accuracy: the path is confined to the search bands of §3.4 "
    "(0.25–0.55 Hz and 0.85–1.45 Hz), so a reference rate outside them cannot be represented, "
    "which is the case for 36.7% of respiratory epochs — whose true rate falls below the 15 "
    "breaths/min floor — and 14.9% of cardiac epochs. A trace can hold the right band all night "
    "while its moment-to-moment value follows the reference poorly, which is what §4.2 goes on to "
    "report."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    d.set_text("The two physiological bands carry energy well above the sensor noise floor", SNR)
    print("  SNR paragraph: opens on the physiology, numbers follow")

    d.set_text("Cross-spectral coherence at the ground-truth rate frequency confirmed",
               COHERENCE)
    print("  coherence paragraph rewritten")

    # merge the two control paragraphs into one
    d.set_text("Two controls bound that number, because a coherence estimate is not zero",
               CONTROLS)
    window_i = d.find("The 30-second epoch is used here for consistency")
    d.body.remove(d.body[window_i])
    print("  the two control paragraphs merged into one")

    d.set_text("Both physiological rhythms can be followed continuously across a whole night",
               CONTINUITY)
    print("  continuity paragraph rewritten")

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
