"""Repair the figure renumbering botched by apply_rate_regime_edit.py.

That script applied re.sub three times in sequence to the same run text, so a
reference to "Figure 8" became 7, then 6, then 5 -- collapsing figures 6, 7 and 8
all onto 5. This restores the intended mapping by matching each occurrence on its
surrounding context:

    original 6 (ridge spectrogram)   -> 5   already correct
    original 7 (ridge by stage)      -> 6   currently 5
    original 8 (spindles)            -> 7   currently 5
    original 9 (delta-burst onset)   -> 8   currently 9, never renumbered

Note: the delta-burst figure is referenced in section 4.5 but has no caption and no
image in the document. That gap predates this edit and is left as-is, renumbered.

Run from the repo root.
"""

import zipfile
from pathlib import Path

from lxml import etree

DOCX = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# (distinctive paragraph context, text to find, replacement)
FIXES = [
    ("Figure 5. Band-restricted ridge structure", "Figure 5", "Figure 6"),
    ("Figure 5. The capacitive response at sleep spindles", "Figure 5", "Figure 7"),
    ("Restricting ridge detection to each band", "(Figure 5)", "(Figure 6)"),
    ("Onset-triggered averaging shows that low-band", "(Figure 5)", "(Figure 7)"),
    ("reproduces across all six subjects", "(Figure 9)", "(Figure 8)"),
]


def main() -> None:
    with zipfile.ZipFile(DOCX) as z:
        files = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(files["word/document.xml"])
    body = root.find(W + "body")

    for context, find, replace in FIXES:
        paras = [
            p for p in body.iter(W + "p")
            if context in "".join(t.text or "" for t in p.iter(W + "t"))
        ]
        if len(paras) != 1:
            raise SystemExit(f"expected 1 paragraph for {context!r}, found {len(paras)}")

        hits = [t for t in paras[0].iter(W + "t") if t.text and find in t.text]
        if len(hits) != 1:
            raise SystemExit(
                f"{find!r} not contained in exactly one run for {context!r} "
                f"(found {len(hits)}); it may be split across runs"
            )
        hits[0].text = hits[0].text.replace(find, replace)
        print(f"{find} -> {replace}  in: {context[:52]}")

    files["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    print(f"wrote {DOCX}")


if __name__ == "__main__":
    main()
