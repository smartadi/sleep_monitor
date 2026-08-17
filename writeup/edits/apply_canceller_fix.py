"""Section 3.2: name the canceller we used, and fix the channel sentence.

Two defects, both in 3.2.

1. "Two cancellers were available: ... OLS ... and a normalized-LMS (NLMS)
   adaptive FIR canceller (16 taps, mu = 0.05) ..." -- a menu, not a method.
   Every result in the paper used the OLS canceller (remove_acc_artifact).
   NLMS (remove_acc_artifact_nlms) appears only in the generic sweep harness
   (sleep_monitor/evaluate.py, scripts/sweep.py, scripts/train_classifier.py)
   and its unit tests; no reported number came from it. Verified across every
   analysis behind the paper:
     4.1 overnight record   analysis/mean_value/channel_evolution.py:545  OLS
     4.2 rates              analysis/rates/rerun_rate_detection.py:116    OLS
     4.3 ridges             analysis/slow_wave/band_ridge_analysis.py:295 OLS
     4.4 spindles, 4.5 delta onsets -- no canceller; these bandpass the raw
         channels and control motion with motion-free baseline windows.

2. "the OLS differential CLE-CRE" -- the differential is a plain subtraction,
   cle - cre (rerun_rate_detection.py:76, channel_evolution.py:142). No
   regression forms it. The only OLS in the pipeline is the canceller above,
   so the phrase was both wrong and a collision of two meanings in adjacent
   sentences. The sentence also claimed CLE-CRE as the canonical channel
   "unless otherwise noted", which is not what the paper does: rates and
   ridges are reported on CRE, the overnight record on CLE-CRE and CH.

Run from the repo root:  python writeup/edits/apply_canceller_fix.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_CANCELLER_20260817.docx")

M32_CANCELLER = (
    "Motion artifact was suppressed with an ordinary-least-squares (OLS) canceller: the "
    "band-limited accelerometer magnitude was regressed out of each SEC channel, removing a single "
    "stationary coupling coefficient. The accelerometer magnitude and the SEC channel were first "
    "bandpassed to the analysis band of interest (respiratory 0.1–0.5 Hz, ≈6–30 breaths/min; "
    "cardiac 0.5–3.0 Hz, ≈30–180 beats/min), so that only motion energy within that band was "
    "removed. Bandpass filtering used third-order zero-phase Butterworth filters. This canceller "
    "was used for every result reported here."
)

M32_CHANNEL = (
    "The analysis channels are the three recorded channels CH, CLE and CRE, and the difference of "
    "the two temple channels, CLE−CRE, which cancels common-mode drift between them. Which "
    "channel is used is stated with each analysis."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    d.set_text("Motion artifact was suppressed", M32_CANCELLER)
    print("  3.2: OLS canceller named as the method used; NLMS menu removed")
    d.set_text("Unless otherwise noted, the canonical analysis channel", M32_CHANNEL)
    print("  3.2: 'OLS differential' -> plain difference; channel usage stated per analysis")
    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Canceller and channel sentence (2026-08-17)\n\n"
        + "- 3.2: states the OLS canceller as the method used; NLMS was never used for a "
          "reported result and is gone from the methods\n"
        + "- 3.2: 'the OLS differential CLE−CRE' corrected to the plain difference, and the "
          "canonical-channel claim replaced by what each analysis actually uses\n", encoding="utf-8")
    print("appended to writeup/edits/WRAPUP_CHANGELOG.md")


if __name__ == "__main__":
    main()
