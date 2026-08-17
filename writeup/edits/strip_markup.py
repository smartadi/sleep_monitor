"""Remove editing markup left behind by earlier passes.

Highlight and shading were used to mark changed text while sections were being
rewritten. None of it is meaningful now and all of it prints.

Run from the repo root:  python writeup/edits/strip_markup.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_STRIPMARKUP_20260817.docx")


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)

    d = Doc(DOC)
    n_hl = n_shd = n_col = 0

    for rpr in d.root.iter(W + "rPr"):
        for tag, name in ((W + "highlight", "highlight"), (W + "shd", "shd"),
                          (W + "color", "color")):
            for el in rpr.findall(tag):
                # keep deliberate colour, drop the black-on-default redundancy
                if tag.endswith("color") and (el.get(W + "val") or "").lower() not in (
                        "000000", "auto", "windowtext"):
                    continue
                rpr.remove(el)
                if name == "highlight":
                    n_hl += 1
                elif name == "shd":
                    n_shd += 1
                else:
                    n_col += 1

    for ppr in d.root.iter(W + "pPr"):
        for el in ppr.findall(W + "shd"):
            ppr.remove(el)
            n_shd += 1

    print("  removed %d highlights, %d shadings, %d redundant black colour tags"
          % (n_hl, n_shd, n_col))
    d.save(DOC)
    print("wrote %s" % DOC)


if __name__ == "__main__":
    main()
