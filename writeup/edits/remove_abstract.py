"""Pull the abstract back out of the canonical manuscript (user request, 2026-08-17).

The abstract written by apply_wrapup.py / apply_mechanical_wording.py is not
wanted yet. This removes its paragraphs and leaves a visible placeholder under
the heading, so the gap cannot be forgotten again the way the original
[TO BE WRITTEN] marker was. The keywords line is left in place.

The drafted text is not lost: paragraphs 1-2 are ABSTRACT / ABSTRACT_2 in
writeup/edits/apply_wrapup.py and paragraph 3 is ABSTRACT_3 in
writeup/edits/apply_mechanical_wording.py.

Run from the repo root:  python writeup/edits/remove_abstract.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_ABSTRACT_REMOVAL_20260817.docx")

PLACEHOLDER = "[Abstract — to be written.]"


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    i_abs = d.find("Abstract")
    if d.text_of(body[i_abs]).strip() != "Abstract":
        raise SystemExit("expected the Abstract heading, got: %r" % d.text_of(body[i_abs])[:80])

    removed = 0
    while True:
        nxt = body[i_abs + 1]
        txt = d.text_of(nxt).strip()
        style = nxt.find(".//" + W + "pStyle")
        if style is not None and (style.get(W + "val") or "").startswith("Heading"):
            break
        if txt.startswith("Keywords:") or txt.startswith("1. Introduction"):
            break
        if not txt:                      # keep the blank spacer before Keywords
            break
        body.remove(nxt)
        removed += 1

    body.insert(i_abs + 1, d.para(PLACEHOLDER))
    print("  removed %d abstract paragraph(s), inserted placeholder" % removed)

    d.save(DOC)
    print("wrote %s" % DOC)


if __name__ == "__main__":
    main()
