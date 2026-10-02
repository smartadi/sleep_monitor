"""
Build manuscript V11 from V10, into writeup/review/address/.

V10 is the reviewer's current version. He has already taken the rate Methods
(§2.4 PSG reference rates, §2.6 SEC rate estimation) and the rate Results
section out, but the rate claims are still standing in the Discussion and the
Conclusion -- so as it arrives, V10 asserts calibrated rate accuracy with no
method describing how rates were estimated and no results table. Both deletions
also left numbering holes: Methods runs 2.4 then 2.7, and the figures run 1-11
then 13.

This build closes those, carries over the fixes from the V7 pass that V10
predates, and adds the K-complex result.

Operations
----------
 1  Methods renumbering 2.7-2.9 -> 2.5-2.7. No in-text reference cites a 2.x
    section, so only the headings move.
 2  Optical figure 13 -> 12, closing the figure gap. The new K-complex figure
    takes 12 at the end of Results and the optical figure keeps document order
    as 13, so nothing else renumbers.
 3  Rate claims trimmed in the Discussion and Conclusion and pointed at the
    rate supplement. The Abstract is left alone (user instruction), and it
    carries none of these numbers anyway.
 4  Delta-burst event count corrected (three recordings -> four) and the
    K-complex attribution retracted: those events are detector-defined, and
    the scored K-complexes are now reported separately in their own section.
 5  Figure 2's caption gets its per-panel ages back; it had collapsed to
    "four male participants aged 54 years".
 6  CH stated as separately sensed where the channels are introduced.
 7  New Results section on the technologist-scored K-complexes, with its
    figure, plus the Methods sentence that licenses it.

Not done here
-------------
The imbalance-versus-REM section still needs its figure built, and Figure 4's
caption still carries the author's own note, "(CHECK THE Y AXIS title and
unit)", which is his to resolve rather than ours to guess.

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

SRC = ROOT / 'writeup' / 'review' / 'CAP_sleep_mask_manuscript_V10.docx'
DST = ROOT / 'writeup' / 'review' / 'address' / \
    'CAP_sleep_mask_manuscript_V11.docx'
KFIG = ROOT / 'writeup' / 'figures' / 'delta_onset' / \
    'fig_kcomplex_cap_response.png'


def q(t):
    return '{%s}%s' % (W, t)


# ── text edits, each must match exactly once ─────────────────────────────────

EDITS = [
    # 1 — methods renumbering (no in-text citation of a 2.x section exists)
    ('2.7 Persistent ridges and harmonic-comb episodes',
     '2.5 Persistent ridges and harmonic-comb episodes', 'methods 2.7->2.5'),
    ('2.8 Cortical events and event-triggered analysis',
     '2.6 Cortical events and event-triggered analysis', 'methods 2.8->2.6'),
    ('2.9 Statistical analysis',
     '2.7 Statistical analysis', 'methods 2.9->2.7'),

    # 2 — the figure gap closes by itself: the new K-complex figure takes 12 at
    # the end of Results and the optical figure keeps 13, so no figure is
    # renumbered and no cross-reference moves.

    # 5 — figure 2 ages
    ('Representative recordings from four male participants aged 54 years.',
     'Representative recordings from four male participants, aged (a) 25, '
     '(b) 37, (c) 54 and (d) 66 years.', 'figure 2 ages'),

    # 6 — CH is its own electrode
    ('three sensing channels positioned around the eyes (CH, CLE, and CRE)',
     'three sensing channels positioned around the eyes (CH, CLE, and CRE), '
     'each acquired from its own electrode — CH is sensed directly and is '
     'not computed as a difference between CLE and CRE',
     'CH separately sensed'),

    # 4 — delta-burst count and the K-complex attribution
    ('three recordings contained fewer than 10 events. Most were isolated N2 '
     'slow waves or K-complexes because sustained N3 slow-wave activity '
     'generally lacked a discrete onset following a quiet baseline.',
     'four recordings contained fewer than 10 events. Most were isolated N2 '
     'slow waves arising on a quiet baseline, because sustained N3 slow-wave '
     'activity generally lacked a discrete onset. These events are defined by '
     'the detection criteria in Section 2.6 rather than by a scoring label; '
     'the technologist’s own K-complex annotations are used as an '
     'independent trigger in Section 3.9.',
     'delta-burst count + K-complex retraction'),

    # 3 — rate claims
    ('These findings demonstrate calibrated agreement with PSG.',
     'These values come from calibration on the night being reported. Under '
     'calibrations that transfer between recordings, respiratory night-mean '
     'error remains below a no-sensor baseline (0.57–0.94 against 1.20 '
     'breaths/min) whereas cardiac night-mean error does not (3.19–3.77 '
     'against 2.76 beats/min), and within-night tracking is not recovered in '
     'either band. The rate pipeline, its calibration and these comparisons '
     'are reported in full in the accompanying rate supplement; no conclusion '
     'in this paper rests on them.',
     'rate claim, Discussion'),

    ('Session-specific PSG calibration supported estimation of nightly mean '
     'respiratory and cardiac rates, with median absolute differences of 0.24 '
     'breaths/min and 1.56 beats/min, respectively; reliable within-night '
     'tracking and transferable calibration remain to be established.',
     'Session-specific PSG calibration supported estimation of the nightly '
     'mean respiratory rate, which stayed more accurate than a no-sensor '
     'baseline under calibrations transferred between recordings; the cardiac '
     'equivalent did not, and within-night tracking was not recovered in '
     'either band.',
     'rate claim, Conclusion'),

]

KMETHOD = (' In addition to the detector-defined delta-burst onsets, the PSG '
           'export’s technologist-scored K-complex annotations were used '
           'as an independent trigger for the same event-triggered analysis. '
           'Fifty scored marks were available across ten of the twelve '
           'recordings.')

SECTION_39 = [
    ('h', '3.9 SEC response at technologist-scored K-complexes'),
    ('p', 'The delta-burst onsets in Section 3.5 are detected from the EEG '
          'envelope by the criteria given in Section 2.6, which makes them a '
          'proxy for a scored event class rather than the class itself. The '
          'PSG export also carries the technologist’s own K-complex marks, '
          'and these give an independent trigger — 50 marks across ten of '
          'the twelve recordings — that owes nothing to our own detector.'),
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
          'NREM sleep, in which slow-delta EEG preceded the CSF peak by 6.4 s '
          'with the hemodynamic response between the two. The latencies here '
          'are shorter than that figure, which is what a measurement sensitive '
          'to the hemodynamic limb rather than to fluid flow itself would be '
          'expected to give. The comparison is one of ordering and timescale '
          'only: the modality, the measurement site and the quantity measured '
          'all differ.'),
    ('p', 'Per-recording counts range from one to ten events (Fig. 12c), so '
          'this is reported descriptively and no statistical test is attached '
          'to it.'),
    ('fig', None),
    ('c', 'Figure 12. SEC response at technologist-scored K-complexes. '
          '(a) Event-triggered average of low-frequency SEC power across the 50 '
          'scored marks, against a count-matched random-NREM null; the EEG '
          'delta envelope is shown as a positive control. (b) Latency of the '
          'per-event peak response by channel; zero marks where electrical '
          'pickup would appear. (c) Peak response per recording, showing how '
          'few events each one contributes.'),
]


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

    def new_rid(self):
        used = {r.get('Id') for r in self.rels}
        n = 1
        while f'rId{n}' in used:
            n += 1
        return f'rId{n}'

    def add_image(self, png: Path) -> str:
        i = 1
        while f'word/media/image{i}.png' in self.parts:
            i += 1
        target = f'media/image{i}.png'
        self.parts[f'word/{target}'] = png.read_bytes()
        rid = self.new_rid()
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


def replace(doc: Doc, old: str, new: str, label: str):
    hits = [p for p in doc.paras if old in doc.text(p)]
    assert len(hits) == 1, f'{label}: expected 1 paragraph, found {len(hits)}'
    p = hits[0]
    ts = list(p.iter(q('t')))
    spans, pos = [], 0
    for t in ts:
        s = t.text or ''
        spans.append((pos, pos + len(s), t))
        pos += len(s)
    full = ''.join(t.text or '' for t in ts)
    at = full.find(old)
    end = at + len(old)
    first = True
    for lo, hi, t in spans:
        if hi <= at or lo >= end:
            continue
        s = t.text or ''
        a, b = max(at, lo) - lo, min(end, hi) - lo
        t.text = s[:a] + (new if first else '') + s[b:]
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        first = False
    doc.n_edits += 1


def clone_text_para(template, text):
    """A paragraph with `template`'s formatting carrying `text`."""
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


def insert_section(doc: Doc, after, body_tmpl, head_tmpl, png: Path):
    """Insert section 3.9 and its figure after paragraph `after`."""
    img_tmpl = None
    for p in doc.paras:
        if len(list(p.iter('{%s}blip' % A))) == 1 and not doc.text(p).strip():
            img_tmpl = p
            break
    assert img_tmpl is not None, 'no image-only paragraph to clone'

    cursor = after
    for kind, text in SECTION_39:
        if kind == 'fig':
            node = copy.deepcopy(img_tmpl)
            rid = doc.add_image(png)
            cx = int(node.find('.//{%s}extent' % WP).get('cx'))
            with Image.open(png) as im:
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
            tmpl = head_tmpl if kind == 'h' else body_tmpl
            node = clone_text_para(tmpl, text)
        cursor.addnext(node)
        cursor = node
        doc.n_edits += 1


def main():
    for p in (SRC, KFIG):
        if not p.exists():
            raise SystemExit(f'missing: {p}')
    doc = Doc(SRC)
    n0 = len(doc.paras)
    log = []

    for old, new, label in EDITS:
        replace(doc, old, new, label)
        log.append(label)

    # the Methods sentence goes on the paragraph after the heading
    heads = [i for i, p in enumerate(doc.paras)
             if doc.text(p).strip().startswith('2.6 Cortical events')]
    assert len(heads) == 1, 'could not locate the 2.6 heading'
    body = doc.paras[heads[0] + 1]
    ts = list(body.iter(q('t')))
    ts[-1].text = (ts[-1].text or '') + KMETHOD
    ts[-1].set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    doc.n_edits += 1
    log.append('methods sentence for scored K-complexes')

    # new section at the end of Results, just before the Discussion heading
    disc = [p for p in doc.paras if doc.text(p).strip().startswith('4. Discussion')]
    assert len(disc) == 1, 'could not locate the Discussion heading'
    results_end = disc[0].getprevious()
    while results_end is not None and results_end.tag != q('p'):
        results_end = results_end.getprevious()
    head_tmpl = [p for p in doc.paras
                 if doc.text(p).strip().startswith('3.8 ')][0]
    body_tmpl = [p for p in doc.paras
                 if len(doc.text(p)) > 400][0]
    insert_section(doc, results_end, body_tmpl, head_tmpl, KFIG)
    log.append('section 3.9 + Figure 12 inserted')

    doc.save(DST)

    # ── verify ───────────────────────────────────────────────────────────────
    chk = Doc(DST)
    txt = chk.full
    assert 'REPLACE' not in txt and 'KMETHOD' not in txt, 'staging marker left behind'
    assert 'These findings demonstrate calibrated agreement' not in txt
    assert 'aged 54 years.' not in txt
    assert '3.9 SEC response at technologist-scored K-complexes' in txt
    assert 'Figure 12. SEC response at technologist-scored K-complexes' in txt
    for want in ('2.5 Persistent ridges', '2.6 Cortical events',
                 '2.7 Statistical analysis'):
        assert want in txt, f'missing heading: {want}'
    for gone in ('2.7 Persistent ridges', '2.8 Cortical events',
                 '2.9 Statistical analysis'):
        assert gone not in txt, f'old heading survived: {gone}'
    media = [n for n in chk.parts if n.startswith('word/media/')]

    print(f'built {DST.name}  ({doc.n_edits} edits)')
    for line in log:
        print(f'  - {line}')
    print(f'  paragraphs {n0} -> {len(chk.paras)}, media {len(media)}')
    print('  verified: headings renumbered, rate claims trimmed, '
          'section 3.9 present, no staging markers')


if __name__ == '__main__':
    main()
