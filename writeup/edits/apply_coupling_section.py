"""Fold the CAP-PSG coupling work into 4.1, and add its matrices to the supplement.

Reports what was measured and stops there. Three things the text is careful about:

  * the EEG negative control is stated with every number, because the raw
    coupling is nearly uniform across sensors without it;
  * the coherence noise floor is stated, because the in-band values sit one to
    three times above it;
  * the existing 4.1 coherence figure (0.31 at the reference frequency, 30 s
    epochs) and the new one (peak in-band, 5-minute windows) are different
    statistics and are not merged or compared.

The interpretive sentence is marked as an observation about the pattern, not a
demonstration of a mechanism. The ridge results are cross-referenced as evidence
that the rhythms are present and shared; what that does and does not yield in
rate accuracy stays in 4.2.

Run from the repo root:  python writeup/edits/apply_coupling_section.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_COUPLING_20260817.docx")
FIG_MATRIX = Path("writeup/figures/coupling/cap_psg_matrix.png")
FIG_COH = Path("writeup/figures/coupling/cap_psg_coherence.png")

BODY = (
    "How much of the polysomnographic signal the mask reproduces was quantified channel by "
    "channel, in amplitude and in rhythm, with the contact EEG as a negative control: it is "
    "exposed to whatever broadband disturbance the other sensors see but carries no respiratory "
    "or cardiac mechanics. Amplitude coupling — the within-night correlation of band-limited log "
    "RMS on the 30-second grid — is modest and largely non-specific in the respiratory band, at a "
    "median r of 0.38 against nasal airflow and 0.34 against EEG, a margin of 0.06 to 0.09 once "
    "motion epochs are removed. In the cardiac band it is specific: 0.35 against ECG against 0.13 "
    "for EEG, a margin of 0.18 to 0.20 that survives motion removal. Coherence, computed over "
    "five-minute windows with 30-second Welch segments, shows the opposite pattern. In the "
    "respiratory band the peak in-band coherence is 0.372 against airflow, 0.404 against thoracic "
    "effort and 0.243 against EEG on the differential channel, and the mean in-band margin over "
    "EEG is +0.029 on CLE, positive in 11 of the 12 recordings (Wilcoxon p = 0.001); in the "
    "cardiac band that margin is +0.004 to +0.013. Thoracic effort, the mechanical respiratory "
    "channel, couples at least as strongly as airflow on every capacitive channel. Two limits "
    "bound these numbers. With 19 segments per estimate the coherence noise floor is 0.053, so "
    "the mean in-band values of 0.06 to 0.15 sit one to three times above it; and the EEG control "
    "itself reaches a peak in-band coherence of 0.22 to 0.30, so a peak value should not be read "
    "without it. Per-channel matrices for both measures are given in Figures S12 and S13."
)

INTERP = (
    "The pattern is a shared respiratory oscillation whose amplitude does not transfer, beside a "
    "cardiac band whose amplitude tracks the reference while its phase does not. Together with "
    "the persistent ridges at both frequencies (§4.3), this establishes that the two rhythms are "
    "present in the capacitive signal and are the same rhythms the polysomnograph records. It "
    "does not establish how accurately either rate can be read from them, which §4.2 reports "
    "separately. We report the pattern as observed; these data do not identify the transformation "
    "that produces it."
)

S4_HEADING = "S4. Capacitive channels against polysomnographic channels"

CAP_MATRIX = (
    "Figure S12. Amplitude coupling between each capacitive channel and each PSG channel, as the "
    "correlation of band-limited log RMS on the 30-second grid. Left column: median within-night "
    "correlation. Right column: correlation across the twelve recordings of their session means. "
    "Top row respiratory band, bottom row cardiac band. EEG is the negative control; the raw "
    "coupling is close to uniform across sensors, which is why the contrast against EEG rather "
    "than the absolute value is the measure reported in §4.1."
)

CAP_COH = (
    "Figure S13. Coherence between each capacitive channel and each PSG channel, median in-band "
    "value over five-minute windows with 30-second Welch segments (about 19 segments per "
    "estimate, noise floor 0.053). Every pair uses identical segmentation, so the upward bias of "
    "the estimator is common to the matrix and the contrast against EEG is fair. Thoracic effort "
    "couples at least as strongly as nasal airflow in the respiratory band on every channel."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    # ---------------------------------------------------------------- 4.1 text
    anchor = d.find("Respiratory frequency agreement between the SEC spectral peak")
    body.insert(anchor + 1, d.para(INTERP))
    body.insert(anchor + 1, d.para(BODY))
    print("  4.1: coupling paragraph and its reading added after the frequency-agreement text")

    # ------------------------------------------------------------ supplementary
    img_template = body[next(i for i in range(len(body) - 1, 0, -1)
                             if body[i].findall(".//" + W + "drawing"))]
    tail = len(body) - 1
    while tail > 0 and not d.text_of(body[tail]).strip():
        tail -= 1

    block = [
        d.para(""),
        d.para(S4_HEADING, "Heading2"),
        d.image_para(FIG_MATRIX, img_template),
        d.para(CAP_MATRIX),
        d.image_para(FIG_COH, img_template),
        d.para(CAP_COH),
    ]
    for off, el in enumerate(block):
        body.insert(tail + 1 + off, el)
    print("  supplementary S4 added with Figures S12 and S13")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## CAP-PSG coupling folded into 4.1 (2026-08-17)\n\n"
        + "- amplitude and coherence coupling reported per channel with EEG as the negative "
          "control, plus the coherence noise floor\n"
        + "- supplementary S4, Figures S12 and S13\n"
        + "- the new coherence statistic is kept separate from 4.1's existing "
          "at-reference-frequency value; they are not merged\n", encoding="utf-8")


if __name__ == "__main__":
    main()
