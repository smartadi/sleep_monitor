"""
Build manuscript V11 from V10: V10 plus the K-complex result, nothing else.

Scope is deliberate (user directive): add the K-complex section and leave every
other defect in V10 alone. So the rate claims in the Discussion and Conclusion
stay as they are, the Methods numbering keeps its 2.4 -> 2.7 jump, Figure 2's
caption keeps "four male participants aged 54 years", the delta-burst count
keeps "three recordings", and CH is not described. Those were in an earlier
draft of this script and have been removed, not merely disabled.

One consequence rather than a fix: V10's figures run 1-11 then 13, and the new
K-complex figure takes 12 at the end of Results, so the sequence closes up on
its own and the optical figure keeps its number. No figure is renumbered.

The new section is written against V10's own numbering: the event-triggered
Methods section is 2.8 in V10, and Results end at 3.8, so the new section is
3.9 and cites 2.8.

Formatting follows V10: the heading clones V10's own Heading style from §3.8
and the body paragraphs clone a V10 body paragraph, so the section renders as
part of the document rather than as an insert.

Writes  writeup/review/final/CAP_sleep_mask_manuscript_V11.docx

Usage
-----
    .venv/Scripts/python.exe writeup/ppt/_build_v11.py
"""

from __future__ import annotations

import copy
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

FINAL = ROOT / 'writeup' / 'review' / 'final'
SRC = FINAL / 'CAP_sleep_mask_manuscript_V10.docx'
DST = FINAL / 'CAP_sleep_mask_manuscript_V11.docx'
KFIG = ROOT / 'writeup' / 'figures' / 'delta_onset' / \
    'fig_kcomplex_cap_response.png'

# Terminology, on instruction: the quantity is the mean-centred differential,
# d(t) = (CLE - CRE)(t) - mu, so it is called that rather than "imbalance".
# What the text reports is its low-passed magnitude and the integral of that
# magnitude, so those are named as such rather than all collapsed into one word.
RENAMES = [
    ('The time-averaged magnitude of imbalance',
     'The time-averaged magnitude of the mean-centred differential'),
    ('indicating that the imbalance burden was primarily',
     'indicating that the integrated magnitude was primarily'),
    ('If the imbalance of CLE−CRE reflected',
     'If the mean-centred differential of CLE−CRE reflected'),
]

# the Methods sentence is appended to V10's section 2.8
METHODS_HEAD = '2.8 Cortical events and event-triggered analysis'
METHODS_ADD = (' In addition to the detector-defined delta-burst onsets, the '
               'PSG export’s technologist-scored K-complex annotations '
               'were used as an independent trigger for the same '
               'event-triggered analysis. Fifty scored marks were available '
               'across ten of the twelve recordings.')

SECTION = [
    ('h', '3.9 SEC response at technologist-scored K-complexes'),
    ('p', 'The delta-burst onsets in Section 3.5 are detected from the EEG '
          'envelope by the criteria given in Section 2.8, which makes them a '
          'proxy for a scored event class rather than the class itself. The '
          'PSG export also carries the technologist’s own K-complex '
          'marks, and these give an independent trigger — 50 marks across '
          'ten of the twelve recordings — that owes nothing to our own '
          'detector.'),
    ('p', 'Averaged on those marks, low-frequency SEC power rose after the '
          'event and exceeded a count-matched random-NREM null (Fig. 12a). The '
          'peak followed the K-complex rather than coinciding with it, with a '
          'median latency of 3.6 s on CLE, 3.7 s on CRE and 4.3 s on CH '
          '(Fig. 12b). Electrical pickup of the K-complex itself would appear '
          'at zero lag, so this delay is the discriminating observation here. '
          'It carries more weight than the frequency argument used for '
          'spindles, because a K-complex occupies 0.5–4 Hz and therefore '
          'overlaps the SEC analysis bands, which leaves timing rather than '
          'frequency as the available test.'),
    ('p', 'The direction and timescale are consistent with the sequence '
          'reported for neural, hemodynamic and cerebrospinal fluid signals in '
          'NREM sleep, in which slow-delta EEG preceded the cerebrospinal '
          'fluid peak by 6.4 s with the hemodynamic response between the two. '
          'The latencies here are shorter than that figure, which is what a '
          'measurement sensitive to the hemodynamic limb rather than to fluid '
          'flow itself would be expected to give. The comparison is one of '
          'ordering and timescale only: the modality, the measurement site and '
          'the quantity measured all differ.'),
    ('p', 'Per-recording counts range from one to ten events (Fig. 12c), so '
          'this is reported descriptively and no statistical test is attached '
          'to it.'),
    ('fig', None),
    ('c', 'Figure 12. SEC response at technologist-scored K-complexes. '
          '(a) Event-triggered average of low-frequency SEC power across the '
          '50 scored marks, against a count-matched random-NREM null; the EEG '
          'delta envelope is shown as a positive control. (b) Latency of the '
          'per-event peak response by channel; zero marks where electrical '
          'pickup would appear. (c) Peak response per recording, showing how '
          'few events each one contributes.'),
]


def q(t):
    return '{%s}%s' % (W, t)


class Doc:
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

    @property
    def full(self):
        return '\n'.join(self.text(p) for p in self.paras)

    def only(self, pred, what):
        hits = [p for p in self.paras if pred(self.text(p))]
        assert len(hits) == 1, f'{what}: expected 1 paragraph, found {len(hits)}'
        return hits[0]

    def add_image(self, png: Path) -> str:
        i = 1
        while f'word/media/image{i}.png' in self.parts:
            i += 1
        target = f'media/image{i}.png'
        self.parts[f'word/{target}'] = png.read_bytes()
        used = {r.get('Id') for r in self.rels}
        n = 1
        while f'rId{n}' in used:
            n += 1
        rid = f'rId{n}'
        etree.SubElement(self.rels, '{%s}Relationship' % RELNS,
                         {'Id': rid, 'Type': '%s/image' % R, 'Target': target})
        return rid

    def save(self, out: Path):
        self.parts['word/document.xml'] = etree.tostring(
            self.root, xml_declaration=True, encoding='UTF-8', standalone=True)
        self.parts['word/_rels/document.xml.rels'] = etree.tostring(
            self.rels, xml_declaration=True, encoding='UTF-8', standalone=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, data in self.parts.items():
                z.writestr(name, data)


def clone_text(template, text):
    """`template`'s formatting, carrying `text`. Keeps V10's own styles."""
    p = copy.deepcopy(template)
    runs = list(p.iter(q('r')))
    keep = runs[0]
    for r in runs[1:]:
        r.getparent().remove(r)
    for t in list(keep.iter(q('t'))):
        t.getparent().remove(t)
    t = etree.SubElement(keep, q('t'))
    t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return p


def main():
    for p in (SRC, KFIG):
        if not p.exists():
            raise SystemExit(f'missing: {p}')
    doc = Doc(SRC)
    n0 = len(doc.paras)
    assert '3.9 ' not in doc.full, 'V10 already has a section 3.9'
    assert 'Figure 12.' not in doc.full, 'V10 already has a Figure 12'

    # ── 0. terminology ───────────────────────────────────────────────────────
    for old, new in RENAMES:
        hits = [p for p in doc.paras if old in doc.text(p)]
        assert len(hits) == 1, f'rename {old[:40]!r}: found {len(hits)} paragraphs'
        p = hits[0]
        ts = list(p.iter(q('t')))
        joined = ''.join(t.text or '' for t in ts)
        ts[0].text = joined.replace(old, new, 1)
        for t in ts[1:]:
            t.text = ''
        doc.n_edits += 1

    # ── 1. the Methods sentence, appended to V10's 2.8 ───────────────────────
    head = doc.only(lambda t: t.strip().startswith(METHODS_HEAD), 'methods 2.8')
    body = doc.paras[doc.paras.index(head) + 1]
    ts = list(body.iter(q('t')))
    ts[-1].text = (ts[-1].text or '') + METHODS_ADD
    ts[-1].set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    doc.n_edits += 1

    # ── 2. the new section, in V10's own styles ──────────────────────────────
    head_tmpl = doc.only(lambda t: t.strip().startswith('3.8 '), 'heading 3.8')
    body_tmpl = next(p for p in doc.paras if len(doc.text(p)) > 400)
    img_tmpl = next(p for p in doc.paras
                    if len(list(p.iter('{%s}blip' % A))) == 1
                    and not doc.text(p).strip())
    disc = doc.only(lambda t: t.strip().startswith('4. Discussion'), 'discussion')
    cursor = disc.getprevious()
    while cursor is not None and cursor.tag != q('p'):
        cursor = cursor.getprevious()

    for kind, text in SECTION:
        if kind == 'fig':
            node = copy.deepcopy(img_tmpl)
            rid = doc.add_image(KFIG)
            cx = int(node.find('.//{%s}extent' % WP).get('cx'))
            with Image.open(KFIG) as im:
                iw, ih = im.size
            cy = int(round(cx * ih / iw))
            for b in node.iter('{%s}blip' % A):
                b.set('{%s}embed' % R, rid)
            for e in node.iter('{%s}extent' % WP):
                e.set('cx', str(cx))
                e.set('cy', str(cy))
            for e in node.iter('{%s}ext' % A):
                if e.get('cx') is not None:
                    e.set('cx', str(cx))
                    e.set('cy', str(cy))
            used = {int(d.get('id')) for d in doc.root.iter('{%s}docPr' % WP)
                    if (d.get('id') or '').isdigit()}
            for d in node.iter('{%s}docPr' % WP):
                d.set('id', str(max(used) + 1 if used else 1))
                d.set('name', 'Figure 12')
        else:
            node = clone_text(head_tmpl if kind == 'h' else body_tmpl, text)
        cursor.addnext(node)
        cursor = node
        doc.n_edits += 1

    doc.save(DST)

    # ── verify: the addition landed, and nothing else moved ──────────────────
    before, after = Doc(SRC), Doc(DST)
    assert '3.9 SEC response at technologist-scored K-complexes' in after.full
    assert 'Figure 12. SEC response' in after.full
    assert 'imbalance' not in after.full.lower(), 'imbalance survived the rename'
    for untouched in (
            'These findings demonstrate calibrated agreement with PSG.',
            'Representative recordings from four male participants aged 54 years.',
            'three recordings contained fewer than 10 events',
            '2.7 Persistent ridges and harmonic-comb episodes'):
        assert untouched in after.full, f'V10 text should be untouched: {untouched[:44]!r}'
    caps = sorted(int(m.group(1)) for m in
                  (__import__('re').match(r'^Fig(?:ure)?\.?\s*(\d+)\b', after.text(p))
                   for p in after.paras) if m)
    print(f'built {DST.name}  ({doc.n_edits} edits)')
    print(f'  paragraphs {n0} -> {len(after.paras)}, '
          f'media {len([n for n in after.parts if n.startswith("word/media/")])}')
    print(f'  figure captions: {caps}')
    print('  verified: section 3.9 and Figure 12 present; '
          'rate claims, Figure 2 caption, event count and Methods numbering '
          'all left as V10 had them')


if __name__ == '__main__':
    main()
