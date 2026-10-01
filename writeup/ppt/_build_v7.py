"""
Build manuscript V7 from the reviewer's V9.

V9 is the agreed base (user directive). It carries KH Choi's 58 tracked edits,
accepted, plus Jaehyun Chung's restructure -- §3.2 merged with §3.3, and
night-to-night reproducibility and the exploratory associations moved out of
Discussion into Results as §3.7 / §3.8, which is what Choi's comments asked for.
Our own V5/V6 edits are replayed onto it by *text*, not by paragraph index,
because V9 has a different paragraph count. `_v7_replay_check.py` is the dry run
that says which of them still apply; of 38, eight apply cleanly, three he had
already made himself, and the rest were either reworded or made obsolete by his
renumbering.

Every operation asserts what it expects to find, so a missed or double-applied
edit fails the build instead of shipping.

Operations
----------
 1  Restore Figure 3 (in-band SNR), which V9 deleted while keeping the Methods
    that define it and the Results that report its numbers. Restored as the
    rebuilt version: band split into respiratory and cardiac, with the unworn
    mask plotted as a measured negative control.
 2  Replay the V5/V6 edits that still apply cleanly to V9's text.
 3  Swap in our redrawn Figures 6-10. Figures 4 and 5 are deliberately NOT
    swapped: V9 replaced both with S4N2 recordings and retitled the captions to
    match, so our S1N1/S1N2 redraws would sit under the wrong captions.
 4  Soften the rate-performance claim (critique M1/M3), without touching the
    Abstract -- user instruction, and the Abstract carries none of these numbers.
 5  Retract the K-complex attribution (critique M7) and correct the event count.
 6  Drop the apology for p-values Figure 6 no longer prints (critique S5).
 7  Restore the per-panel ages to the Figure 2 caption.
 8  State that CH is separately sensed, where the channels are introduced.
 9  One unit per quantity: breaths/min and beats/min throughout (Choi).

Not done here, and why
----------------------
  * The grouped coherence figure belongs to the supplementary document, where
    the coherence matrices live (Figs S5 / S9), not to this file.
  * The metal-electrode S/N comparison needs a number from reference 29, which
    is not in this repository.
  * "12" vs "twelve" is left alone: a blind sweep would rewrite figures,
    p-values and years as well.

Usage
-----
    .venv/Scripts/python.exe writeup/ppt/_build_v7.py
"""

from __future__ import annotations

import ast
import copy
import re
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
BUILD_V6 = HERE / '_build_v6.py'
FIGS = ROOT / 'writeup' / 'figures'
FIG3 = FIGS / 'signal_validation' / 'fig3_inband_snr_split.png'

# figure number -> our redrawn source. 4 and 5 are absent on purpose: V9
# replaced both with S4N2 recordings and retitled to match.
SWAP = {
    6:  FIGS / 'harmonics' / 'band_ridge_by_stage.png',
    7:  FIGS / 'spindles' / 'fig_spindle_lowband_allsessions.png',
    8:  FIGS / 'delta_onset' / 'fig_delta_onset_cohort.png',
    9:  FIGS / 'harmonics' / 'ladders' / 'ladder_S6N1.png',
    10: FIGS / 'harmonics' / 'ladder_stage_relationship.png',
}

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

# ── text edits, as (old, new). Each must match exactly once in the document. ──

RATE_OLD = 'These findings demonstrate calibrated agreement with PSG.'
RATE_NEW = ('These values reflect recording-specific calibration. Under '
            'calibrations that transfer between recordings, respiratory '
            'night-mean error remained below the no-sensor baseline '
            '(0.57 breaths/min across nights and 0.94 breaths/min '
            'population-derived, against 1.20 breaths/min for the baseline), '
            'whereas cardiac night-mean error did not (3.19–3.77 '
            'beats/min against 2.76 beats/min).')

KCOMPLEX_OLD = ('three recordings contained fewer than 10 events. Most were '
                'isolated N2 slow waves or K-complexes because sustained N3 '
                'slow-wave activity generally lacked a discrete onset '
                'following a quiet baseline.')
KCOMPLEX_NEW = ('four recordings contained fewer than 10 events. Most were '
                'isolated N2 slow waves arising on a quiet baseline, because '
                'sustained N3 slow-wave activity generally lacked a discrete '
                'onset. These events were not verified against scored '
                'K-complexes: the PSG export annotated spindles exhaustively '
                'but K-complexes only sporadically (57 marks across the '
                'cohort), so that channel could not serve as a reference, and '
                'the events are therefore described here by their detection '
                'criteria rather than by a scoring label.')

PVALUE_OLD = ('Although pooled Kruskal–Wallis and Mann–Whitney tests '
              'yielded very small p-values (as low as 9 ×10-29), these '
              'results were reported as descriptive statistics in Fig. 6 '
              'because the tests treated non-independent epochs as '
              'independent observations.')
PVALUE_NEW = ('Fig. 6 therefore reports these comparisons descriptively and '
              'without test statistics: 30 s epochs within a night are not '
              'independent, so a pooled test is driven by how many epochs a '
              'recording contains rather than by the size of any effect.')

FIG2_OLD = 'Representative recordings from four male participants aged 54 years.'
FIG2_NEW = ('Representative recordings from four male participants, aged '
            '(a) 25, (b) 37, (c) 54 and (d) 66 years.')

CH_OLD = ('three sensing channels positioned around the eyes (CH, CLE, and CRE)')
CH_NEW = ('three sensing channels positioned around the eyes (CH, CLE, and '
          'CRE), each acquired from its own electrode — CH is sensed '
          'directly and is not computed as a difference between CLE and CRE')

UNITS = [('br/min', 'breaths/min'), ('BPM', 'beats/min')]


def q(t):
    return '{%s}%s' % (W, t)


class Doc:
    def __init__(self, path):
        self.zin = zipfile.ZipFile(path)
        self.parts = {n: self.zin.read(n) for n in self.zin.namelist()}
        self.root = etree.fromstring(self.parts['word/document.xml'])
        self.rels = etree.fromstring(self.parts['word/_rels/document.xml.rels'])
        self.rel_target = {r.get('Id'): r.get('Target') for r in self.rels}
        self.n_edits = 0

    @property
    def paras(self):
        return list(self.root.iter(q('p')))

    def text(self, p):
        return ''.join(t.text or '' for t in p.iter(q('t')))

    @property
    def full_text(self):
        return '\n'.join(self.text(p) for p in self.paras)

    def new_rid(self):
        used = {r.get('Id') for r in self.rels}
        n = 1
        while f'rId{n}' in used:
            n += 1
        return f'rId{n}'

    def add_image(self, png: Path) -> str:
        idx = 1
        while f'word/media/image{idx}.png' in self.parts:
            idx += 1
        target = f'media/image{idx}.png'
        self.parts[f'word/{target}'] = png.read_bytes()
        rid = self.new_rid()
        etree.SubElement(self.rels, '{%s}Relationship' % RELNS,
                         {'Id': rid, 'Type': '%s/image' % R, 'Target': target})
        self.rel_target[rid] = target
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


def replace_text(doc: Doc, old: str, new: str, where: str = '') -> bool:
    """Replace `old` with `new` in the one paragraph that contains it.

    Word splits a visible phrase over many runs, so the match is made on the
    paragraph's concatenated text and written back into the <w:t> elements the
    match spans: the first keeps the replacement, the rest lose only their
    matched characters, and formatting on untouched runs survives.
    """
    hits = [p for p in doc.paras if old in doc.text(p)]
    if len(hits) != 1:
        raise AssertionError(
            f'{where or old[:48]!r}: expected 1 paragraph, found {len(hits)}')
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
    return True


def replace_everywhere(doc: Doc, old: str, new: str) -> int:
    """Whole-word replacement across every run in the document."""
    pat = re.compile(r'(?<![\w/])%s(?![\w/])' % re.escape(old))
    n = 0
    for t in doc.root.iter(q('t')):
        if t.text and pat.search(t.text):
            t.text, k = pat.subn(new, t.text)
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            n += k
    doc.n_edits += n
    return n


def figure_rids(doc: Doc) -> dict[int, str]:
    """figure number -> relationship id of the image above its caption."""
    out, pend = {}, []
    for p in doc.paras:
        for b in p.iter('{%s}blip' % A):
            rid = b.get('{%s}embed' % R)
            if rid:
                pend.append(rid)
        m = re.match(r'^\s*Fig(?:ure)?\.?\s*(\d+)\b', doc.text(p))
        if m:
            if pend:
                out[int(m.group(1))] = pend[-1]
            pend = []
    return out


def swap_image(doc: Doc, rid: str, png: Path):
    """Replace the bytes behind `rid` and re-fit the drawing's height."""
    target = doc.rel_target[rid]
    name = 'word/' + target
    assert name in doc.parts, f'missing media part {name}'
    doc.parts[name] = png.read_bytes()
    with Image.open(png) as im:
        iw, ih = im.size
    aspect = ih / iw
    for blip in doc.root.iter('{%s}blip' % A):
        if blip.get('{%s}embed' % R) != rid:
            continue
        anc = blip
        while anc is not None and anc.tag not in ('{%s}inline' % WP,
                                                  '{%s}anchor' % WP):
            anc = anc.getparent()
        if anc is None:
            continue
        for e in anc.iter('{%s}extent' % WP):
            e.set('cy', str(int(round(int(e.get('cx')) * aspect))))
        for e in anc.iter('{%s}ext' % A):
            if e.get('cx') is not None:
                e.set('cy', str(int(round(int(e.get('cx')) * aspect))))
        break
    doc.n_edits += 1


def template_figure_para(doc: Doc):
    for p in doc.paras:
        if len(list(p.iter('{%s}blip' % A))) == 1 and not doc.text(p).strip():
            return p
    raise SystemExit('no image-only paragraph to clone')


def insert_figure_after(doc: Doc, anchor, png: Path, caption: str) -> str:
    tmpl = template_figure_para(doc)
    cx = int(tmpl.find('.//{%s}extent' % WP).get('cx'))
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
    used = {int(d.get('id')) for d in doc.root.iter('{%s}docPr' % WP)
            if (d.get('id') or '').isdigit()}
    for d in img_p.iter('{%s}docPr' % WP):
        d.set('id', str(max(used) + 1 if used else 1))
        d.set('name', 'Figure 3')

    cap_p = copy.deepcopy(anchor)
    runs = list(cap_p.iter(q('r')))
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


def v6_edits():
    """(old, new) pairs from _build_v6.py's build_main, via the AST."""
    tree = ast.parse(BUILD_V6.read_text(encoding='utf-8'))
    out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if fn.name != 'build_main':
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'replace' and len(node.args) >= 4):
                a = node.args
                if all(isinstance(x, ast.Constant) for x in (a[2], a[3])):
                    out.append((a[2].value, a[3].value))
    return out


def main():
    for p in (SRC, FIG3, BUILD_V6, *SWAP.values()):
        if not p.exists():
            raise SystemExit(f'missing: {p}')

    doc = Doc(SRC)
    n0 = len(doc.paras)
    log = []

    # 1 ── restore Figure 3 ---------------------------------------------------
    assert 'Figure 3' not in doc.full_text, 'V9 already has a Figure 3'
    hits = [p for p in doc.paras if 'The respiratory band accounted for' in doc.text(p)]
    assert len(hits) == 1, 'anchor for Figure 3 not unique'
    insert_figure_after(doc, hits[0], FIG3, FIG3_CAPTION)
    log.append('Figure 3 restored (split bands + unworn control)')

    # 2 ── replay the V5/V6 edits that still apply ---------------------------
    applied = skipped = 0
    for old, new in v6_edits():
        if doc.full_text.count(old) == 1:
            replace_text(doc, old, new)
            applied += 1
        else:
            skipped += 1
    log.append(f'replayed {applied} of {applied + skipped} V5/V6 text edits '
               f'({skipped} obsolete or reworded in V9)')

    # 3 ── swap our redrawn figures ------------------------------------------
    rids = figure_rids(doc)
    for num, png in sorted(SWAP.items()):
        assert num in rids, f'no image found for Figure {num}'
        swap_image(doc, rids[num], png)
    log.append(f'swapped Figures {sorted(SWAP)} (4 and 5 left as V9 has them)')

    # 4-9 ── the text edits ---------------------------------------------------
    replace_text(doc, RATE_OLD, RATE_NEW, 'rate claim (M1/M3)')
    log.append('rate-performance claim softened; Abstract untouched')

    replace_text(doc, KCOMPLEX_OLD, KCOMPLEX_NEW, 'K-complex (M7)')
    log.append('K-complex attribution retracted; count three -> four')

    replace_text(doc, PVALUE_OLD, PVALUE_NEW, 'pooled p-values (S5)')
    log.append('apology for Fig. 6 p-values replaced')

    replace_text(doc, FIG2_OLD, FIG2_NEW, 'Figure 2 ages')
    log.append('Figure 2 caption: per-panel ages restored')

    replace_text(doc, CH_OLD, CH_NEW, 'CH is separately sensed')
    log.append('CH stated as separately sensed, not derived')

    for old, new in UNITS:
        n = replace_everywhere(doc, old, new)
        log.append(f'units: {old} -> {new} ({n} occurrences)')

    doc.save(DST)

    # ── verify ---------------------------------------------------------------
    chk = Doc(DST)
    txt = chk.full_text
    caps = [t for t in (chk.text(p) for p in chk.paras)
            if t.strip().startswith('Figure 3.')]
    assert len(caps) == 1, f'{len(caps)} Figure 3 captions'
    # Every replacement must have landed. `old` is only required to be gone
    # when it is not a prefix of `new` -- the CH edit extends the sentence it
    # matches rather than replacing it, so its old text legitimately survives
    # inside the new text.
    for old, new in ((RATE_OLD, RATE_NEW), (KCOMPLEX_OLD, KCOMPLEX_NEW),
                     (PVALUE_OLD, PVALUE_NEW), (FIG2_OLD, FIG2_NEW),
                     (CH_OLD, CH_NEW)):
        assert new in txt, f'replacement missing: {new[:48]!r}'
        if old not in new:
            assert old not in txt, f'stale text survived: {old[:48]!r}'
    for old, _ in UNITS:
        assert not re.search(r'(?<![\w/])%s(?![\w/])' % re.escape(old), txt), \
            f'unit {old} survived'
    media = [n for n in chk.parts if n.startswith('word/media/')]

    print(f'built {DST.name}  ({doc.n_edits} edits)')
    for line in log:
        print(f'  - {line}')
    print(f'  paragraphs {n0} -> {len(chk.paras)}, media parts {len(media)}')
    print('  verified: Figure 3 present once, no stale target text, units clean')


if __name__ == '__main__':
    main()
