"""Repair figure references of the form "Figures SX and SY".

The renumbering in apply_move_bland_altman.py matched "Figures? S<n>", which
catches the first member of a pair but not the one after "and". Both members
therefore collapsed onto the first member's new number:

    Figures S6 and S7  ->  Figures S7 and S7     should be S7 and S8
    Figures S8 and S9  ->  Figures S9 and S9     should be S9 and S10
    Figures 4 and S9   ->  Figures 4 and S9      should be Figures 4 and S10

Mapping by content, under the post-move numbering:
    S9   respiratory nightly traces, all twelve recordings
    S10  cardiac nightly traces
    S7   overnight sensor value, cohort-wide
    S8   left-right imbalance, cohort-wide

Run from the repo root:  python writeup/edits/repair_figure_lists.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_FIGLIST_20260817.docx")

FIXES = [
    ("Figures S7 and S7", "Figures S7 and S8"),
    ("Figures S9 and S9", "Figures S9 and S10"),
    ("Figures 4 and S9", "Figures 4 and S10"),
]


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)

    d = Doc(DOC)
    counts = {old: 0 for old, _ in FIXES}
    for t in d.root.iter(W + "t"):
        if not t.text:
            continue
        for old, new in FIXES:
            if old in t.text:
                counts[old] += t.text.count(old)
                t.text = t.text.replace(old, new)
    for old, n in counts.items():
        print("  %-20s -> %-22s %d" % (old, dict(FIXES)[old], n))

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
