"""Give 4.1's coherence a negative control, a floor, and a window justification.

Three changes, all from analysis/rates/coherence_control_and_window.py.

1. NEGATIVE CONTROL. 4.1 reported coherence of 0.31 against a phase-randomized
   surrogate null and a two-sensor upper bound, but with no control channel. The
   contact EEG is a real, simultaneously recorded signal that shares the
   recording environment and carries no respiratory or cardiac mechanics. At the
   paper's own statistic it scores 0.222 in the respiratory band against the
   reference's 0.302.

2. THE FLOOR. The estimator explains most of that 0.31. Coherence in the
   respiratory band uses 10 s segments inside a 30 s epoch -- three to five
   segments -- and magnitude-squared coherence from N segments has an expected
   value near 1/N under independence. Pairing the capacitive channel with a
   circularly shifted copy of the same reference scores 0.193, and with a
   time-reversed copy 0.246. The informative quantity is the margin over a
   control, not the value.

3. THE WINDOW. Every analysis epoch in the paper is 30 s and that does not
   change. Repeating the comparison at a five-minute window (about 30 segments,
   floor 0.03) lowers every absolute value but preserves the margin and its
   significance, so the result is not an artifact of the epoch length.

Also corrects 3.4: it says the cardiac coherence reference is ECG. The code that
produced the reported values, scripts/signal_validation_proof.py:191, uses
photoplethysmography. The text is brought to the code.

Run from the repo root:  python writeup/edits/apply_coherence_control.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_COHCONTROL_20260817.docx")

CONTROL = (
    "Two controls bound that number, because a coherence estimate is not zero under "
    "independence. In the respiratory band the estimate uses 10-second segments inside the "
    "30-second epoch, which gives three to five segments, and magnitude-squared coherence from N "
    "segments has an expected value of about 1/N whatever the signals are. Pairing the same "
    "capacitive channel with a circularly shifted copy of the reference scores 0.193, and with a "
    "time-reversed copy 0.246, against the 0.302 the aligned pair reaches. The second control is a "
    "channel rather than a manipulation: the contact EEG is recorded simultaneously, shares the "
    "recording environment and any disturbance common to the sensors, and carries no respiratory "
    "or cardiac mechanics. It scores 0.222 in the respiratory band and 0.065 in the cardiac band, "
    "against 0.302 and 0.126 for the physiological references. The margin over that control is "
    "+0.062 for respiration, positive in 11 of the 12 recordings (Wilcoxon p = 0.001), and +0.059 "
    "for the cardiac band, positive in all 12 (p < 0.001). The margin, not the absolute value, is "
    "what should be read: most of a coherence of 0.3 at this window length is the estimator."
)

WINDOW = (
    "The 30-second epoch is used here for consistency with every other analysis in this paper, "
    "and it is a poor window for coherence: three segments in the respiratory band is close to the "
    "minimum at which the statistic is defined. Repeating the whole comparison at a five-minute "
    "window, which gives about 30 segments and a floor near 0.03, lowers every absolute value — "
    "the respiratory reference falls from 0.302 to 0.062 and the shifted control from 0.193 to "
    "0.012 — while preserving the direction and the significance of the margin (respiratory "
    "+0.014, 9 of 12 recordings, p = 0.034; cardiac +0.028, 10 of 12, p = 0.002). The coupling is "
    "therefore not an artifact of the epoch length, and the two window lengths are two views of "
    "one result seen through estimators with different bias."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ------------------------------------------------ 3.4: the cardiac reference
    d.patch("For each analysis epoch, magnitude-squared coherence was computed",
            "(nasal airflow for respiration, ECG for cardiac;",
            "(nasal airflow for respiration, photoplethysmography for cardiac;")
    print("  3.4: cardiac coherence reference corrected to photoplethysmography, "
          "matching signal_validation_proof.py:191")

    # ------------------------------------------------------- 4.1: the two controls
    anchor = d.find("Cross-spectral coherence at the ground-truth rate frequency")
    d.body.insert(anchor + 1, d.para(WINDOW))
    d.body.insert(anchor + 1, d.para(CONTROL))
    print("  4.1: negative control, estimator floor and window robustness added")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Coherence control and window (2026-08-17)\n\n"
        + "- 4.1: EEG negative control (0.222 resp / 0.065 card against 0.302 / 0.126), the "
          "estimator's own floor from shifted and reversed pairings, and the margin as the "
          "quantity to read\n"
        + "- 4.1: the same comparison at a 5-minute window, showing the margin survives while "
          "absolute values collapse\n"
        + "- 3.4: cardiac coherence reference corrected from ECG to photoplethysmography\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
