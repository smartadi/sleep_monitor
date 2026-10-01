"""
Build manuscript V7 from the reviewer's V9.

V9 is the agreed base (user directive): it carries KH Choi's 58 tracked edits,
accepted, plus Jaehyun Chung's restructure -- §3.2 merged with §3.3, and
night-to-night reproducibility and the exploratory associations moved out of
Discussion into Results as §3.7 / §3.8, which is what Choi's comments asked for.
Our own V5/V6 edits are replayed onto it by text rather than by paragraph index,
because V9 has a different paragraph count. `_v7_replay_check.py` is the dry run
that says which of them still apply.

This script is built up one resolved decision at a time. Each operation asserts
what it expects to find, so a missed or double-applied edit fails the build
instead of shipping.

Operations so far
-----------------
1. Restore Figure 3 (in-band SNR), which V9 deleted.

   V8 had it; V9 does not, and nothing in V9 references it -- but the Methods
   paragraph that defines the SNR and the Results sentence that reports its
   numbers both survive, so V9 describes a measurement and shows nothing. The
   figure numbering was never closed up either, so V9 runs 1, 2, 4, ..., 13.
   Restoring it as Figure 3 fixes both without touching figures 4-13.

   The restored figure is the rebuilt one, not V8's: the band is split into
   respiratory and cardiac, and the unworn mask is plotted as a measured
   negative control (see inband_snr_split.py).

Usage
-----
    .venv/Scripts/python.exe writeup/ppt/_build_v7.py
"""

from __future__ import annotations

import copy
import shutil
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
RELNS = 'http://schemas.openxmlformats.org/package/2006/relationships'

SRC = ROOT / 'writeup' / 'review' / 'CAP_sleep_mask_manuscript_V9(1).docx'
DST = HERE / 'CAP_sleep_mask_manuscript_V7.docx'
FIG3 = ROOT / 'writeup' / 'figures' / 'signal_validation' / 'fig3_inband_snr_split.png'

FIG3_CAPTION = (
    'Figure 3. Physiological-band signal-to-noise ratio for each SEC channel '
    '(CH, CLE, and CRE) across the 12 recording sessions, shown separately for '
    'the respiratory (0.1–0.5 Hz) and cardiac (0.5–3.0 Hz) bands. '
    'Signal is the mean spectral density within the band and noise is the mean '
    'spectral density from 10 Hz to the Nyquist frequency, so the ratio is '
    'independent of bandwidth and of per-channel gain. Hatched bars show the '
    'same measurement applied to a recording made with the mask unworn on the '
    'same device (15.8 min), which returns −1.5 to +0.1 dB in both bands '
    'and on every channel: the physiological bands of an unworn mask carry no '
    'excess power over its own electronic floor, so this is the measured level '
    'at which the figure reports nothing. Every channel exceeds that level in '
    'every session in both bands, and CH is the strongest channel throughout.')


def q(t):
    return '{%s}%s' % (W, t)


class Doc:
    """A docx as an editable XML tree, saved by rewriting the zip."""

    def __init__(self, path):
        self.zin = zipfile.ZipFile(path)
        self.parts = {n: self.zin.read(n) for n in self.zin.namelist()}
        self.root = etree.fromstring(self.parts['word/document.xml'])
        self.rels = etree.fromstring(self.parts['word/_rels/document.xml.rels'])
        self.n_edits = 0

    @property
    def paras(self):
        return list(self.root.iter(q('p')))

    def text(self, p):
        return ''.join(t.text or '' for t in p.iter(q('t')))

    def find_para(self, needle):
        """The single paragraph containing `needle`; ambiguity is an error."""
        hits = [p for p in self.paras if needle in self.text(p)]
        assert len(hits) == 1, (f'expected 1 paragraph containing {needle!r}, '
                                f'found {len(hits)}')
        return hits[0]

    def new_rid(self):
        used = {r.get('Id') for r in self.rels}
        n = 1
        while f'rId{n}' in used:
            n += 1
        return f'rId{n}'

    def add_image(self, png: Path) -> str:
        """Add a PNG as a new media part and return its relationship id."""
        names = [n for n in self.parts if n.startswith('word/media/')]
        idx = 1
        while f'word/media/image{idx}.png' in names:
            idx += 1
        target = f'media/image{idx}.png'
        self.parts[f'word/{target}'] = png.read_bytes()
        rid = self.new_rid()
        etree.SubElement(self.rels, '{%s}Relationship' % RELNS, {
            'Id': rid,
            'Type': '%s/image' % R,
            'Target': target,
        })
        return rid

    def save(self, out: Path):
        self.parts['word/document.xml'] = etree.tostring(
            self.root, xml_declaration=True, encoding='UTF-8', standalone=True)
        self.parts['word/_rels/document.xml.rels'] = etree.tostring(
            self.rels, xml_declaration=True, encoding='UTF-8', standalone=True)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, data in self.parts.items():
                z.writestr(name, data)
        print(f'  saved {out.name}  ({self.n_edits} edits)')


def template_figure_para(doc: Doc):
    """An existing paragraph that holds one inline image, to clone.

    Cloning keeps the document's own drawing XML -- namespaces, docPr, the
    graphicFrame chain -- instead of writing a fresh element that Word may
    reject. Only the relationship id, the extents and the ids are changed.
    """
    for p in doc.paras:
        blips = list(p.iter('{%s}blip' % A))
        if len(blips) == 1 and p.find('.//' + q('t')) is None:
            return p
    raise SystemExit('no image-only paragraph found to use as a template')


def insert_figure_after(doc: Doc, anchor, png: Path, caption: str):
    """Insert an image paragraph + caption paragraph after `anchor`."""
    tmpl = template_figure_para(doc)

    # width is taken from the template so the restored figure matches the others
    ext = tmpl.find('.//{%s}extent' % WP)
    cx = int(ext.get('cx'))
    with Image.open(png) as im:
        iw, ih = im.size
    cy = int(round(cx * ih / iw))

    img_p = copy.deepcopy(tmpl)
    rid = doc.add_image(png)
    for blip in img_p.iter('{%s}blip' % A):
        blip.set('{%s}embed' % R, rid)
    for e in img_p.iter('{%s}extent' % WP):
        e.set('cx', str(cx))
        e.set('cy', str(cy))
    for e in img_p.iter('{%s}ext' % A):
        if e.get('cx') is not None:
            e.set('cx', str(cx))
            e.set('cy', str(cy))
    # give the drawing its own non-clashing id and name
    used = {int(d.get('id')) for d in doc.root.iter('{%s}docPr' % WP)
            if (d.get('id') or '').isdigit()}
    new_id = max(used) + 1 if used else 1
    for d in img_p.iter('{%s}docPr' % WP):
        d.set('id', str(new_id))
        d.set('name', 'Figure 3')

    # caption paragraph: clone the anchor so it inherits the body style, then
    # strip it back to a single run carrying the caption text
    cap_p = copy.deepcopy(anchor)
    runs = list(cap_p.iter(q('r')))
    assert runs, 'anchor paragraph has no runs to clone formatting from'
    keep = runs[0]
    for r in runs[1:]:
        r.getparent().remove(r)
    for t in list(keep.iter(q('t'))):
        t.getparent().remove(t)
    t = etree.SubElement(keep, q('t'))
    t.text = caption
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

    anchor.addnext(cap_p)
    anchor.addnext(img_p)
    doc.n_edits += 1
    return rid


def main():
    for p in (SRC, FIG3):
        if not p.exists():
            raise SystemExit(f'missing: {p}')

    doc = Doc(SRC)
    before = len(doc.paras)

    # ── 1. restore Figure 3 ──────────────────────────────────────────────────
    # It belongs immediately after the Results sentence that reports its numbers.
    anchor = doc.find_para('The respiratory band accounted for')
    assert 'Figure 3' not in ''.join(doc.text(p) for p in doc.paras), \
        'V9 already contains a Figure 3; re-check the base'
    rid = insert_figure_after(doc, anchor, FIG3, FIG3_CAPTION)
    print(f'  Figure 3 restored as {rid}, after: '
          f'{doc.text(anchor)[:70]!r}')

    doc.save(DST)
    print(f'  paragraphs {before} -> {len(doc.paras)}')

    # ── verify ───────────────────────────────────────────────────────────────
    chk = Doc(DST)
    texts = [chk.text(p) for p in chk.paras]
    caps = [t for t in texts if t.strip().startswith('Figure 3.')]
    assert len(caps) == 1, f'expected 1 Figure 3 caption, found {len(caps)}'
    media = [n for n in chk.parts if n.startswith('word/media/')]
    print(f'  verified: 1 Figure 3 caption, {len(media)} media parts')


if __name__ == '__main__':
    main()
