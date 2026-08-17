"""Move the Bland-Altman figure out of 4.2 and into the supplementary.

It becomes the last figure of supplementary section S1, which is where the rest
of the per-session rate material already lives, so it is numbered S6 and the
existing S6 to S10 shift up by one. In the main text Figures 6 to 9 close the
gap to 5 to 8.

The bias and limits of agreement stay in the main text: they are a row of
Table 3 and a sentence of 4.2. Only the plot moves, with its sentence rewritten
to point at the supplement.

Run from the repo root:  python writeup/edits/apply_move_bland_altman.py
"""

import re
import shutil
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_BAMOVE_20260817.docx")

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

NEW_CAPTION = (
    "Figure S6. Bland–Altman agreement between the mask and the reference for respiratory (A) and "
    "cardiac (B) rate. One point per 30-second epoch, all twelve recordings pooled, after "
    "per-session calibration. The solid line is the bias and the dashed lines the 95% limits of "
    "agreement. These limits are epoch-level; the night-mean errors in Table 3 are an order of "
    "magnitude smaller and are not visible at this scale."
)

WRITEUP = (
    "The limits of agreement are wide because they are an epoch-level statistic: they describe how "
    "far a single 30-second estimate can sit from the reference, not how far a night's mean rate "
    "does. Table 3 gives both, and the two differ by an order of magnitude."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    # ------------------------------------------------- lift the figure out of 4.2
    cap_i = d.find("Figure 5. Bland–Altman agreement")
    img_i = cap_i - 1
    if body[img_i].find(".//" + A + "blip") is None:
        raise SystemExit("expected the Bland-Altman image just above its caption")
    img_el, cap_el = body[img_i], body[cap_i]
    body.remove(cap_el)
    body.remove(img_el)
    print("  lifted the figure and caption out of 4.2")

    # ------------------------------------------------------- renumber what remains
    # Renumbering runs before the new reference and caption are written, so that
    # the S6 they name is not itself shifted.
    # main figures 6-9 close the gap to 5-8; supplementary S6-S10 make room at S6.
    def renumber(pattern, shift, lo, hi):
        n = 0
        for t in d.root.iter(W + "t"):
            if not t.text or "igure" not in t.text:
                continue
            def sub(m):
                v = int(m.group(1))
                return m.group(0) if not (lo <= v <= hi) else \
                    m.group(0).replace(m.group(1), str(v + shift))
            new = re.sub(pattern, sub, t.text)
            if new != t.text:
                t.text = new
                n += 1
        return n

    # supplementary first, upward, highest last so numbers do not collide
    for v in range(10, 5, -1):
        renumber(r"Figures? S(%d)\b" % v, +1, v, v)
    print("  supplementary S6-S10 shifted to S7-S11")

    for v in range(6, 10):
        renumber(r"(?<!S)Figures? (%d)\b" % v, -1, v, v)
    print("  main Figures 6-9 closed to 5-8")

    # the sentence in 4.2 that pointed at it
    d.patch("The mask follows both rates across a full night",
            "with 95% limits of −5.85 to +5.65 and −25.48 to +23.62 (Figure 5)",
            "with 95% limits of −5.85 to +5.65 and −25.48 to +23.62 (Figure S6)")
    print("  4.2 now points at Figure S6")

    # ------------------------------------------------- drop it into supplementary
    anchor = d.find("No capacitive feature varied with subject age")
    body.insert(anchor + 1, d.para(WRITEUP))
    body.insert(anchor + 1, d.para(NEW_CAPTION))
    body.insert(anchor + 1, img_el)
    print("  placed as Figure S6 at the end of supplementary section S1")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Bland-Altman moved to the supplement (2026-08-17)\n\n"
        + "- Figure 5 -> Figure S6, at the end of supplementary S1 with the rest of the "
          "per-session rate material\n"
        + "- main Figures 6-9 -> 5-8; supplementary S6-S10 -> S7-S11\n"
        + "- bias and limits of agreement stay in 4.2 and Table 3; only the plot moved\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
