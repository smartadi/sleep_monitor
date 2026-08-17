"""Build the reviewed manuscript: writeup/main/CAP_sleep_mask_manuscript_main_review.docx

Applies the full analytical-review pass to a COPY of the canonical manuscript.
The canonical file is never modified.

Fixes applied (see REVIEW_CHANGELOG.md for the annotated list):
  Tier 1  structural  -- ISWA construct delivered (new 4.6), abstract written,
                         4.3 updated to the three-band revamp, figure numbering
  Tier 2  numeric     -- Table 3 reconciled against Table S1, k IQR, Wilcoxon p,
                         independent-sensor bound, session counts
  Tier 3  statistical -- exact permutation p, unit-of-analysis caveats, surrogate
                         framing, error-to-variation definition
  Tier 4  contradictions -- channel placement, home-vs-laboratory, window length,
                         estimator count, orphan claims
  Tier 5  editorial   -- duplicate 3.1 block, duplicate author line, Table 1 IDs,
                         broken sentences, reference 28, heading levels, units

Run from the repo root:  python writeup/edits/apply_review_edits.py
"""

import copy
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from PIL import Image

SRC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
DST = Path("writeup/main/CAP_sleep_mask_manuscript_main_review.docx")

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
IMG_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

FIG_RIDGE_OVERLAY = Path("writeup/figures/harmonics/revamp_overlay_S1N1.png")
FIG_RIDGE_STAGE = Path("writeup/figures/harmonics/revamp_by_stage.png")
FIG_CAP_SWA = Path("writeup/figures/cap_swa/fig_cap_swa_definition.png")
FIG_SWA_ROC = Path("analysis/swa_validation/outputs/roc_curves.png")
FIG_HARMONIC_LADDER = Path("writeup/figures/harmonics/revamp_harmonics.png")


# ============================================================== docx machinery

class Doc:
    def __init__(self, path):
        with zipfile.ZipFile(path) as z:
            self.files = {n: z.read(n) for n in z.namelist()}
        self.root = etree.fromstring(self.files["word/document.xml"])
        self.body = self.root.find(W + "body")
        self.rels = etree.fromstring(self.files["word/_rels/document.xml.rels"])
        self._next_rel = 1 + max(
            int(m.group(1)) for r in self.rels
            if (m := re.match(r"rId(\d+)$", r.get("Id") or "")))
        self._next_docpr = 5000
        self.n_patch = 0

    # ---------------------------------------------------------------- reading
    @staticmethod
    def text_of(el):
        return "".join(t.text or "" for t in el.findall(".//" + W + "t"))

    def find(self, needle, start=0):
        """Index of the first body element whose text contains `needle`."""
        for i in range(start, len(self.body)):
            if needle in self.text_of(self.body[i]):
                return i
        raise KeyError("not found in body: " + needle[:70])

    # ---------------------------------------------------------------- editing
    def patch(self, needle, old, new, count=1):
        """Replace `old` with `new` inside the paragraph located by `needle`.

        Operates across run boundaries while preserving the formatting of every
        run outside the replaced span: the replacement text is written into the
        first run the span touches, and the tail of the span is trimmed from the
        runs that follow.
        """
        el = self.body[self.find(needle)]
        return self._patch_el(el, old, new, count)

    def _patch_el(self, el, old, new, count=1):
        ts = el.findall(".//" + W + "t")
        full = "".join(t.text or "" for t in ts)
        if old not in full:
            raise KeyError("substring not present: %r\n  in: %r" % (old[:60], full[:160]))
        done = 0
        while done < count and old in full:
            at = full.index(old)
            end = at + len(old)
            pos, out = 0, []
            for t in ts:
                s = t.text or ""
                lo, hi = pos, pos + len(s)
                pos = hi
                if hi <= at or lo >= end:          # untouched run
                    out.append(s)
                    continue
                head = s[: max(0, at - lo)]
                tail = s[max(0, end - lo):] if hi > end else ""
                out.append(head + ("\x00" if lo <= at < hi else "") + tail)
            merged = "".join(out).replace("\x00", new, 1)
            # write back: first touched run takes everything, others emptied
            first = None
            pos = 0
            for t in ts:
                s = t.text or ""
                lo, hi = pos, pos + len(s)
                pos = hi
                if first is None and hi > at:
                    first = t
            for t in ts:
                t.text = ""
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            (first if first is not None else ts[0]).text = merged
            ts = el.findall(".//" + W + "t")
            full = "".join(t.text or "" for t in ts)
            done += 1
        self.n_patch += 1
        return el

    def set_text(self, needle, new_text):
        """Replace a paragraph's entire text, keeping its style and first run."""
        el = self.body[self.find(needle)]
        ts = el.findall(".//" + W + "t")
        for t in ts[1:]:
            t.text = ""
        ts[0].text = new_text
        ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        self.n_patch += 1
        return el

    def set_style(self, needle, style):
        el = self.body[self.find(needle)]
        ppr = el.find(W + "pPr")
        if ppr is None:
            ppr = etree.SubElement(el, W + "pPr")
            el.remove(ppr)
            el.insert(0, ppr)
        st = ppr.find(W + "pStyle")
        if st is None:
            st = etree.SubElement(ppr, W + "pStyle")
        st.set(W + "val", style)
        self.n_patch += 1

    # ------------------------------------------------------------ construction
    def _template_run(self):
        """A plain body run to clone.

        Must skip the title and any other bold/italic run: this run's formatting
        is inherited by every paragraph and table cell the edit scripts create,
        and cloning the title's bold run is how 40 body paragraphs and three
        whole tables ended up bold.
        """
        fallback = None
        for p in self.body.iter(W + "p"):
            if p.find(W + "pPr/" + W + "pStyle") is not None:
                continue
            for r in p.findall(W + "r"):
                if r.find(W + "t") is None:
                    continue
                rpr = r.find(W + "rPr")
                decorated = rpr is not None and (
                    rpr.find(W + "b") is not None or rpr.find(W + "i") is not None
                    or rpr.find(W + "vertAlign") is not None)
                if not decorated:
                    return r
                fallback = fallback or r
        if fallback is None:
            raise RuntimeError("no template run")
        return fallback

    def _body_spacing(self, ppr):
        """Match the manuscript's body spacing: double, no space after."""
        sp = ppr.find(W + "spacing")
        if sp is None:
            sp = etree.SubElement(ppr, W + "spacing")
        sp.set(W + "after", "0")
        sp.set(W + "line", "480")
        sp.set(W + "lineRule", "auto")

    def para(self, text, style=""):
        p = etree.SubElement(etree.Element("tmp"), W + "p")
        ppr = etree.SubElement(p, W + "pPr")
        if style:
            etree.SubElement(ppr, W + "pStyle").set(W + "val", style)
        else:
            self._body_spacing(ppr)
        run = copy.deepcopy(self._template_run())
        for child in list(run):
            if etree.QName(child).localname == "t":
                run.remove(child)
        t = etree.SubElement(run, W + "t")
        t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        p.append(run)
        return p

    def image_para(self, png, template):
        data = Path(png).read_bytes()
        name = "word/media/%s_%d.png" % (Path(png).stem, self._next_rel)
        self.files[name] = data
        rid = "rId%d" % self._next_rel
        self._next_rel += 1
        rel = etree.SubElement(self.rels, "{%s}Relationship" % RELS_NS)
        rel.set("Id", rid)
        rel.set("Type", IMG_TYPE)
        rel.set("Target", name.replace("word/", ""))

        p = copy.deepcopy(template)
        p.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip").set(R + "embed", rid)
        extent = p.find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent")
        cx = int(extent.get("cx"))
        w_px, h_px = Image.open(png).size
        cy = int(cx * h_px / w_px)
        extent.set("cy", str(cy))
        for ext in p.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
            ext.set("cx", str(cx))
            ext.set("cy", str(cy))
        docpr = p.find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr")
        self._next_docpr += 1
        docpr.set("id", str(self._next_docpr))
        docpr.set("name", Path(png).stem)
        return p

    def swap_image(self, img_para, png):
        """Point an existing image paragraph at a different PNG."""
        data = Path(png).read_bytes()
        name = "word/media/%s_%d.png" % (Path(png).stem, self._next_rel)
        self.files[name] = data
        rid = "rId%d" % self._next_rel
        self._next_rel += 1
        rel = etree.SubElement(self.rels, "{%s}Relationship" % RELS_NS)
        rel.set("Id", rid)
        rel.set("Type", IMG_TYPE)
        rel.set("Target", name.replace("word/", ""))
        img_para.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip").set(R + "embed", rid)
        extent = img_para.find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent")
        cx = int(extent.get("cx"))
        w_px, h_px = Image.open(png).size
        cy = int(cx * h_px / w_px)
        extent.set("cy", str(cy))
        for ext in img_para.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
            ext.set("cx", str(cx))
            ext.set("cy", str(cy))
        self.n_patch += 1

    def table(self, rows, template):
        tbl = copy.deepcopy(template)
        tr0 = copy.deepcopy(tbl.find(W + "tr"))
        tc0 = copy.deepcopy(tr0.find(W + "tc"))
        for tr in tbl.findall(W + "tr"):
            tbl.remove(tr)
        grid = tbl.find(W + "tblGrid")
        if grid is not None:
            for gc in grid.findall(W + "gridCol"):
                grid.remove(gc)
            for _ in rows[0]:
                etree.SubElement(grid, W + "gridCol").set(W + "w", str(9360 // len(rows[0])))
        for row in rows:
            tr = copy.deepcopy(tr0)
            for tc in tr.findall(W + "tc"):
                tr.remove(tc)
            for cell in row:
                tc = copy.deepcopy(tc0)
                for p in tc.findall(W + "p"):
                    tc.remove(p)
                tc.append(self.para(cell))
                tr.append(tc)
            tbl.append(tr)
        return tbl

    def cell_text(self, tbl, row, col, new):
        tc = tbl.findall(W + "tr")[row].findall(W + "tc")[col]
        ts = tc.findall(".//" + W + "t")
        if not ts:
            return
        for t in ts[1:]:
            t.text = ""
        ts[0].text = new

    def insert_after(self, needle, elements):
        at = self.find(needle) + 1
        for off, el in enumerate(elements):
            self.body.insert(at + off, el)
        return at

    def delete_range(self, first_needle, last_needle):
        a = self.find(first_needle)
        b = self.find(last_needle)
        assert b >= a, (a, b)
        for i in range(b, a - 1, -1):
            self.body.remove(self.body[i])
        return b - a + 1

    def save(self, path):
        self.files["word/document.xml"] = etree.tostring(
            self.root, xml_declaration=True, encoding="UTF-8", standalone=True)
        self.files["word/_rels/document.xml.rels"] = etree.tostring(
            self.rels, xml_declaration=True, encoding="UTF-8", standalone=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in self.files.items():
                z.writestr(name, data)
