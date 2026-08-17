"""Three stale numbers, found by the ledger check.

Each was correct for an earlier run of its analysis and was never updated when
that analysis was rerun. All three now come from the current artifacts.

  spindle low-band dB   +0.47 to +0.58 on the temple channels, approximately
                        +0.6 on CH  ->  +0.45 to +0.49 and +0.55
                        (analysis/spindles/outputs/spindle_lowband_detection.csv,
                        per-channel means of *_low_meandB)

  spindle EEG sigma     +3.3 dB  ->  +3.45 dB, in section 4.4, section 5.2 and
                        limitation 6. Two independent files agree: the detection
                        table gives 3.450 and spindle_ersp.csv gives 3.462. The
                        paper's value predates the 0.32 s event-alignment fix
                        (commit f96e425). Reported to two places because 3.45
                        sits exactly on the one-place rounding boundary.

  ridge Kruskal-Wallis  "down to 3x10^-23" was the second smallest p in
                        reports/slow_wave/band_ridge_stage_summary.csv; the
                        smallest is 9x10^-29.

The claims these numbers support are unchanged: the capacitive sigma response
stays null at +0.02 to +0.03 dB against an EEG response two orders of magnitude
larger, and the low-band response stays clearly positive on every channel.

Run from the repo root:  python writeup/edits/apply_stale_numbers.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_STALENUM_20260817.docx")


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ---------------------------------------------------------- 4.4 low band
    d.patch("Onset-triggered averaging shows that low-band",
            "The trial-averaged increase reaches +0.47 to +0.58 dB on the temple channels and "
            "approximately +0.6 dB on the forehead channel (CH)",
            "The trial-averaged increase reaches +0.45 to +0.49 dB on the temple channels and "
            "+0.55 dB on the forehead channel (CH)")
    print("  4.4: low-band range corrected to +0.45 to +0.49 / +0.55 dB")

    # ------------------------------------------------------- 4.4 sigma vs EEG
    d.patch("This response lies entirely in the low, mechanical band",
            "the identical measurement on the contact EEG rises +3.3 dB",
            "the identical measurement on the contact EEG rises +3.45 dB")
    print("  4.4: EEG sigma corrected to +3.45 dB")

    # ------------------------------------------------------- Figure 8 caption
    d.patch("Figure 8. The capacitive response at sleep spindles",
            "strongest on the forehead channel (CH, ≈+0.6 dB)",
            "strongest on the forehead channel (CH, +0.55 dB)")
    d.patch("Figure 8. The capacitive response at sleep spindles",
            "the low band shifts positive (mean +0.58 dB) while the sigma band is null "
            "(mean +0.03 dB)",
            "the low band shifts positive (mean +0.55 dB) while the sigma band is null "
            "(mean +0.02 dB)")
    print("  Figure 8 caption: CH values corrected")

    # -------------------------------------------------------------------- 5.2
    d.patch("The mask does not pick up cortical electrical activity",
            "where the same measurement on the contact EEG rises +3.3 dB",
            "where the same measurement on the contact EEG rises +3.45 dB")
    print("  5.2: EEG sigma corrected")

    # ----------------------------------------------------------- limitation 6
    d.patch("Sixth, the montage derivation",
            "the spindle sigma rhythm was clearly present (+3.3 dB at spindle centers)",
            "the spindle sigma rhythm was clearly present (+3.45 dB at spindle centers)")
    print("  limitation 6: EEG sigma corrected")

    # ---------------------------------------------------------------- 4.3 p
    d.patch("Restricting ridge detection to each band separately",
            "very small, down to 3×10⁻²³", "very small, down to 9×10⁻²⁹")
    print("  4.3: smallest Kruskal-Wallis p corrected to 9×10⁻²⁹")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Stale numbers found by the ledger check (2026-08-17)\n\n"
        + "- spindle low-band +0.47..+0.58 / ~+0.6 dB -> +0.45..+0.49 / +0.55 dB\n"
        + "- spindle EEG sigma +3.3 -> +3.45 dB (4.4, Figure 8, 5.2, limitation 6); the old "
          "value predates the 0.32 s alignment fix\n"
        + "- ridge Kruskal-Wallis floor 3x10^-23 -> 9x10^-29 (the old value was the second "
          "smallest p)\n", encoding="utf-8")


if __name__ == "__main__":
    main()
