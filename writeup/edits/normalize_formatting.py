"""One consistent look for the whole manuscript.

Successive edit passes built paragraphs and table cells with Doc.para(), whose
template run was cloned from the first unstyled paragraph carrying text -- the
title, which is bold. Everything those passes created inherited bold, and none
of it inherited the body spacing. The June original had 2 all-bold body
paragraphs and 0 missing spacing; before this pass the working copy had 42 and
40, plus three of five tables bold in every cell.

The conventions restored here are the manuscript's own, taken from the June
original and from the paragraphs that were never rewritten:

    body paragraph   justified, double spaced (after=0, line=480), no first-line
                     indent, no bold or italic
    figure caption   centred, italic, not bold
    table caption    left, bold, not italic
    table cell       header row bold, body rows plain, body spacing throughout
    run-in head      the lead phrase bold, the rest of the paragraph plain

Headings keep their Heading1/2/3 styles and are not touched. Front matter --
title, author line, affiliations -- keeps its own formatting.

Run from the repo root:  python writeup/edits/normalize_formatting.py
"""

import copy
import re
import shutil
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_FORMAT_20260817.docx")

CAPTION_RE = re.compile(r"^(Figure|Table)\s+S?\d+\.")
FRONT_MATTER_UNTIL = "Keywords:"

# Run-in heads: the lead phrase is bold, the rest of the paragraph is plain.
LEAD_INS = [
    "Cardiac ground truth.",
    "Respiratory ground truth.",
    "Overnight evolution of the sensor value.",
    "Left–right capacitance imbalance.",
    "Ridge tracking of the two rhythms.",
    "Keywords:",
]


def rpr_of(run):
    rpr = run.find(W + "rPr")
    if rpr is None:
        rpr = etree.SubElement(run, W + "rPr")
        run.remove(rpr)
        run.insert(0, rpr)
    return rpr


def set_flag(run, tag, on):
    """Turn w:b / w:i on or off for a run. Returns True if it changed."""
    rpr = rpr_of(run)
    present = rpr.find(W + tag)
    if on and present is None:
        etree.SubElement(rpr, W + tag)
        return True
    if not on and present is not None:
        rpr.remove(present)
        return True
    return False


def text_runs(p):
    return [r for r in p.findall(W + "r") if r.find(W + "t") is not None]


def ppr_of(p):
    ppr = p.find(W + "pPr")
    if ppr is None:
        ppr = etree.SubElement(p, W + "pPr")
        p.remove(ppr)
        p.insert(0, ppr)
    return ppr


def set_spacing(p):
    ppr = ppr_of(p)
    sp = ppr.find(W + "spacing")
    if sp is None:
        sp = etree.SubElement(ppr, W + "spacing")
    before = (sp.get(W + "after"), sp.get(W + "line"), sp.get(W + "lineRule"))
    sp.set(W + "after", "0")
    sp.set(W + "line", "480")
    sp.set(W + "lineRule", "auto")
    return before != ("0", "480", "auto")


def set_align(p, val):
    ppr = ppr_of(p)
    jc = ppr.find(W + "jc")
    if val is None:
        if jc is not None:
            ppr.remove(jc)
            return True
        return False
    if jc is None:
        jc = etree.SubElement(ppr, W + "jc")
    changed = jc.get(W + "val") != val
    jc.set(W + "val", val)
    return changed


def drop_indent(p):
    ppr = ppr_of(p)
    ind = ppr.find(W + "ind")
    if ind is not None:
        ppr.remove(ind)
        return True
    return False


def style_of(p):
    st = p.find(W + "pPr/" + W + "pStyle")
    return (st.get(W + "val") or "") if st is not None else ""


def apply_leadin(p, lead):
    """Bold just the lead phrase, plain for the rest."""
    runs = text_runs(p)
    if not runs:
        return False
    full = "".join(r.find(W + "t").text or "" for r in runs)
    if not full.startswith(lead):
        return False

    keep = runs[0]
    for r in runs[1:]:
        p.remove(r)
    t = keep.find(W + "t")
    t.text = lead
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    set_flag(keep, "b", True)
    set_flag(keep, "i", False)

    rest_text = full[len(lead):]
    if rest_text:
        rest = copy.deepcopy(keep)
        set_flag(rest, "b", False)
        rt = rest.find(W + "t")
        rt.text = rest_text
        rt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        p.insert(list(p).index(keep) + 1, rest)
    return True


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    front_end = d.find(FRONT_MATTER_UNTIL)
    counts = dict(body=0, figcap=0, tabcap=0, unbolded=0, leadin=0, cells=0, tables=0)

    # The two front-matter lines that are ours rather than the original's: the
    # abstract placeholder should not be bold, and Keywords is a run-in head.
    placeholder = body[d.find("[Abstract")]
    set_spacing(placeholder)
    set_align(placeholder, "both")
    for r in text_runs(placeholder):
        set_flag(r, "b", False)
        set_flag(r, "i", True)

    kw = body[front_end]
    set_spacing(kw)
    set_align(kw, "both")
    for r in text_runs(kw):
        set_flag(r, "b", False)
        set_flag(r, "i", False)
    apply_leadin(kw, "Keywords:")

    for i, el in enumerate(body):
        if etree.QName(el).localname != "p":
            continue
        txt = d.text_of(el).strip()
        if not txt or i <= front_end:
            continue
        if style_of(el).startswith("Heading"):
            continue
        if style_of(el) == "EndNoteBibliography":
            continue

        cap = CAPTION_RE.match(txt)
        runs = text_runs(el)

        if cap and cap.group(1) == "Figure":
            set_spacing(el)
            set_align(el, "center")
            for r in runs:
                set_flag(r, "b", False)
                set_flag(r, "i", True)
            drop_indent(el)
            counts["figcap"] += 1
            continue

        if cap and cap.group(1) == "Table":
            set_spacing(el)
            set_align(el, None)
            for r in runs:
                set_flag(r, "b", True)
                set_flag(r, "i", False)
            drop_indent(el)
            counts["tabcap"] += 1
            continue

        # ordinary body paragraph
        set_spacing(el)
        set_align(el, "both")
        drop_indent(el)
        lead = next((L for L in LEAD_INS if txt.startswith(L)), None)
        if lead:
            for r in runs:
                set_flag(r, "b", False)
                set_flag(r, "i", False)
            if apply_leadin(el, lead):
                counts["leadin"] += 1
        else:
            for r in runs:
                # superscript citation runs keep their vertAlign; only b/i go
                if set_flag(r, "b", False):
                    counts["unbolded"] += 1
                set_flag(r, "i", False)
        counts["body"] += 1

    # ----------------------------------------------------------------- tables
    for el in body:
        if etree.QName(el).localname != "tbl":
            continue
        counts["tables"] += 1
        rows = el.findall(W + "tr")
        for ri, tr in enumerate(rows):
            for tc in tr.findall(W + "tc"):
                for p in tc.findall(W + "p"):
                    set_spacing(p)
                    for r in text_runs(p):
                        set_flag(r, "b", ri == 0)
                        set_flag(r, "i", False)
                    counts["cells"] += 1

    d.save(DOC)
    print("\nbody paragraphs normalised   %d  (bold stripped from %d runs)"
          % (counts["body"], counts["unbolded"]))
    print("run-in heads restored        %d" % counts["leadin"])
    print("figure captions              %d  centred italic" % counts["figcap"])
    print("table captions               %d  left bold" % counts["tabcap"])
    print("table cells                  %d across %d tables" % (counts["cells"], counts["tables"]))
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Formatting normalisation (2026-08-17)\n\n"
        + "- root cause fixed in apply_review_edits.Doc: para() cloned the title's bold run and "
          "set no spacing, so every paragraph and table cell any edit script created was bold "
          "and single-spaced\n"
        + "- body paragraphs: justified, double spaced, no first-line indent, no stray bold\n"
        + "- figure captions centred italic; table captions left bold\n"
        + "- table cells: header row bold, body rows plain, body spacing throughout\n"
        + "- run-in heads (Cardiac ground truth., etc.) bold on the lead phrase only\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
