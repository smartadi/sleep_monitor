"""Add Results subsection 4.6 — the harmonic-comb ladder episodes (post-REM N2).

A single self-contained subsection at the end of Results (before Discussion), so
no existing subsection is renumbered and the two cross-references that point at
section numbers (to 3.7 and to Section 4.4) are untouched.  The finding is
reported as an exploratory, per-subject-consistent observation, explicitly NOT
a cortical slow-wave signature — consistent with the paper's thesis (the comb
episodes cluster in N2, not N3).

Source: analysis/slow_wave/harmonic_ladder_overlay.py (detection),
        analysis/slow_wave/ladder_stage_relationship.py (stage/REM timing).
Draft:  writeup/edits/ladder_harmonic_section_draft.md.

Run from the repo root:  python writeup/edits/apply_ladder_section.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_LADDER_20260817.docx")
FIG_COMB = Path("writeup/figures/harmonics/ladders/ladder_S6N1.png")
FIG_STAGE = Path("writeup/figures/harmonics/ladder_stage_relationship.png")

HEADING = "4.6 Harmonic-comb ladder episodes follow REM in consolidated N2"

METHOD = (
    "Separately from the persistent respiratory and cardiac rate ridges (§4.3), the "
    "capacitive spectrogram intermittently carries a stack of several sustained, temporally flat "
    "narrow bands — a harmonic comb. Working on each raw channel independently (averaging "
    "channels dilutes a comb strong on a single electrode), we background-subtracted the 0–3 "
    "Hz spectrogram per time column and defined an episode as a sustained interval carrying at "
    "least three integer-harmonic rungs, each a peak at least 5 dB above the local floor with at "
    "least three consecutive from the fundamental; within each episode the rungs were recovered as "
    "the actual persistent horizontal bands (spectral peaks tracked across time, kept if they "
    "lasted at least 1.5 minutes and were present in at least 60% of their span) and reported at "
    "their true frequencies. Because a comb of three consecutive harmonics is required, the two "
    "rhythms the sensor always carries — respiration and heart rate — do not by "
    "themselves form an episode. Episodes overlapping in time across channels were merged into "
    "single events, and each event was related to the technologist-scored hypnogram, with every "
    "REM quantity compared against a matched random-NREM null (200 draws per session)."
)

RESULT1 = (
    "Twenty-two harmonic-comb events were detected across nine sessions in all six subjects. Each "
    "is a stack of several flat, sustained bands at quasi-harmonic — not exactly "
    "integer-spaced — frequencies; a representative episode carried rungs at 0.15, 0.28, "
    "0.42, 0.68 and 0.95 Hz (Figure 9). The events are a consolidated-NREM phenomenon: 19 of the "
    "22 (86%) fell in stage N2, one each in N1, N3 and Wake, and stage occupancy aligned on event "
    "onset peaked at a probability of N2 near 0.9 (Figure 10A). They are therefore not a cortical "
    "slow-wave (N3) signature."
)

RESULT2 = (
    "The events showed a consistent temporal relationship to REM sleep, which preceded them. REM "
    "occupied a mean 0.059 of the 30 minutes before onset — about three and a half times the "
    "matched random-NREM null of 0.017 — but only 0.007 of the 30 minutes after offset, below "
    "the null of 0.031; onset-aligned occupancy shows REM present from roughly 30 to 8 minutes "
    "before onset and essentially absent from 8 minutes before onward (Figure 10B). The nearest "
    "REM epoch lay a median of 30 minutes before an event versus 51 minutes after, and REM was "
    "nearer before the event than after in five of the six subjects. Stage N1 occupancy rose in "
    "the ten minutes immediately before onset, consistent with a REM–N1–N2 re-entry. The "
    "comb episodes thus arise in consolidated N2 roughly 10 to 30 minutes after a REM period and "
    "are not themselves followed by REM. Given the small sample (22 events, six subjects), this "
    "REM association is reported as an exploratory, per-subject-consistent observation rather than "
    "a powered effect. Mechanistically we read these episodes as a non-sinusoidal, quasi-periodic "
    "mechanical or hemodynamic waveform emerging during the stable breathing of post-REM NREM, "
    "rather than as cortical slow-wave activity (§5.3)."
)

FIG9 = (
    "Figure 9. Representative harmonic-comb episodes (S6N1, forehead channel CH). "
    "Background-enhanced spectrogram (0–3 Hz) with the detected flat rungs overlaid, above "
    "the stepped hypnogram. Two episodes occur in consolidated N2, each a stack of quasi-harmonic "
    "bands distinct from the continuous respiratory and cardiac rate ridges."
)

FIG10 = (
    "Figure 10. Sleep-stage relationship of the harmonic-comb events (22 events, six subjects). "
    "(A) Stage occupancy in a ±30-minute window aligned on event onset; N2 peaks at onset. "
    "(B) REM occupancy around onset — elevated from about 30 to 8 minutes before onset and "
    "absent from 8 minutes before onward — showing that REM precedes the episodes."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    # image template: the last paragraph that contains a drawing
    img_template = body[max(i for i in range(len(body))
                            if body[i].findall(".//" + W + "drawing"))]

    anchor = d.find("5. Discussion")     # insert the whole block just before it
    block = [
        d.para(HEADING, "Heading2"),
        d.para(METHOD),
        d.para(RESULT1),
        d.para(RESULT2),
        d.image_para(FIG_COMB, img_template),
        d.para(FIG9),
        d.image_para(FIG_STAGE, img_template),
        d.para(FIG10),
    ]
    for off, el in enumerate(block):
        body.insert(anchor + off, el)
    print("  inserted 4.6 (%d paragraphs incl. Figures 9 and 10) before Discussion" % len(block))

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
