"""
Rebuild the supplementary: drop the old rate material, drop in the new one,
renumber every figure.

The supplement as it stands carries two sets of content appended without
reconciling, so S3, S4, S5, S6, S7, S8 and S11 are each used twice and the
sequence runs S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S4, S7, S8, S6, S3,
S5, S11. Patching that is not possible; the figures have to be renumbered in
document order once the content is settled.

What comes out (all rate, replaced wholesale by the new section):
    S2  estimator and channel comparison, seven estimators x four channels
    S3  calibration requirement
    S7  respiratory rate, twelve separate images
    S8  cardiac rate, twelve separate images
    S9  Bland-Altman, and the paragraph on limits of agreement
    S10, S11  twelve cases each
    S7, S8 again  whole-night respiratory and cardiac rate
    S3 again  per-sleep-stage accuracy
    S5 again  capacitive features versus age -- folded into the new section
    Table S2  per-session rate accuracy

What stays:
    S1  capacitance drift at room temperature over 24 hours
    S4  left-right capacitance difference, twelve recordings
    S5  integrated capacitance imbalance
    S6  physiological-band SNR
    S4 again  overnight sensor value, CH against CLE-CRE
    S6 again  coherence, capacitive against PSG
    S3. Integrated capacitance imbalance, the section
    S4. Capacitive channels against polysomnographic channels, and Table S1

What goes in: the rate pipeline as actually run, k per night and channel for
both detectors, all twelve recordings on one sheet, the comparison against a
no-sensor baseline, and how far k moves within a recording. No Bland-Altman, no
surrogate machinery, no inferential framing.

Flagged throughout: the cardiac reference is verified against ECG R-peaks on
eight of twelve recordings. It runs 36% high on S2N1 and 29% high on S6N1, and
cannot be checked on S5N1 or S6N2 where the ECG is dead. Since k is a count
divided by that reference, those three recordings are marked wherever k appears.

Writes  writeup/review/final/CAP_sleep_mask_manuscript supplementary V3.docx

Usage
-----
    .venv/Scripts/python.exe writeup/ppt/_build_supp_v3.py
"""

from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path

import pandas as pd
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
SRC = FINAL / 'CAP_sleep_mask_manuscript supplementary V2(2).docx'
DST = FINAL / 'CAP_sleep_mask_manuscript supplementary V3.docx'
FIGS = ROOT / 'writeup' / 'figures' / 'rate_supp'
HELDOUT = ROOT / 'reports' / 'rates' / 'rerun' / 'heldout_table.csv'
KTAB = ROOT / 'reports' / 'rates' / 'k_by_channel.csv'

# paragraph ranges to delete, 1-indexed inclusive, in the ORIGINAL document
CUT = [(13, 18), (35, 41),
       # the integrated-imbalance figure appears twice with the same caption
       # and the same closing paragraph: once standalone here, once under the
       # "S3. Integrated capacitance imbalance" heading later. The sectioned
       # copy is kept, so the standalone one goes.
       (45, 49),
       (91, 104), (106, 119), (120, 124), (129, 134),
       (157, 161), (168, 172), (178, 182), (290, 291)]
# Table S2's rows live in a <w:tbl>; deleting its paragraphs would leave the
# table shell behind, so the table element itself is removed by its first cell
CUT_TABLE_STARTING = 'Resp k'


def q(t):
    return '{%s}%s' % (W, t)


def med(cell):
    return float(re.match(r'\s*([-\d.]+)', str(cell)).group(1))


class Doc:
    def __init__(self, path):
        self.zin = zipfile.ZipFile(path)
        self.parts = {n: self.zin.read(n) for n in self.zin.namelist()}
        self.root = etree.fromstring(self.parts['word/document.xml'])
        self.rels = etree.fromstring(self.parts['word/_rels/document.xml.rels'])

    @property
    def paras(self):
        return list(self.root.iter(q('p')))

    def text(self, p):
        return ''.join(t.text or '' for t in p.iter(q('t')))

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


def repair_tables(doc) -> tuple[int, int]:
    """Make the table structures legal again after paragraphs were removed.

    The twelve-panel image blocks are laid out as tables, so deleting their
    paragraphs empties the cells rather than removing the grid. A <w:tc> with no
    <w:p> is invalid OOXML and Word reports the file as corrupt, which is what
    the first build of this document did.

    A table whose cells are now all empty is dropped; any cell still empty in a
    table that survives gets an empty paragraph back.
    """
    dropped = patched = 0
    for tbl in list(doc.root.iter(q('tbl'))):
        cells = list(tbl.iter(q('tc')))
        if cells and not any(c.find(q('p')) is not None for c in cells):
            tbl.getparent().remove(tbl)
            dropped += 1
    for tc in doc.root.iter(q('tc')):
        if tc.find(q('p')) is None:
            etree.SubElement(tc, q('p'))
            patched += 1
    return dropped, patched


def prune_orphan_media(doc) -> tuple[int, float]:
    """Drop images nothing references any more.

    Removing a figure removes the drawing that embeds it, but the image part and
    its relationship stay in the package. On this document that left roughly a
    third of the file as media no reader will ever see.
    """
    used = {b.get('{%s}embed' % R) for b in doc.root.iter('{%s}blip' % A)}
    used |= {b.get('{%s}link' % R) for b in doc.root.iter('{%s}blip' % A)}
    used.discard(None)
    freed = 0
    for rel in list(doc.rels):
        rid, target = rel.get('Id'), rel.get('Target') or ''
        if not target.startswith('media/') or rid in used:
            continue
        name = 'word/' + target
        if name in doc.parts:
            freed += len(doc.parts.pop(name))
        doc.rels.remove(rel)
    return len(used), freed / 1e6


def clone_text(tmpl, text):
    p = copy.deepcopy(tmpl)
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


def make_image_para(doc, tmpl, png: Path, name: str):
    node = copy.deepcopy(tmpl)
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
        d.set('name', name)
    return node


def rate_section(hd, k):
    """(kind, payload) for the new rate section. FIG numbers filled in later."""
    def kc(band, meth, ch):
        s = k[(k.band == band) & (k.method == meth) & (k.channel == ch)].k
        return f'{s.median():.2f}'

    return [
        ('h', 'Respiratory and cardiac rate estimation'),
        ('t', 'This section replaces the earlier rate material. It describes the '
              'pipeline that is actually run, reports what it produces, and '
              'compares it with a spectral detector applied the same way. It is '
              'deliberately descriptive: no agreement limits, no surrogate '
              'tests and no inferential claims. No conclusion in the main text '
              'rests on these results.'),

        ('t', 'A rate is produced in five steps. Take one SEC channel, '
              'band-pass it to the band of interest (0.1–0.5 Hz for '
              'respiration, 0.5–3.0 Hz for cardiac activity), count peaks '
              'in each 30-second epoch with a loose detector (prominence '
              '0.05σ, minimum spacing 0.4 s), divide the count by a '
              'scaling factor k, and report the result. The detector is loose '
              'on purpose: the capacitive waveform is not a clean sinusoid and '
              'a strict detector misses genuine cycles, so it over-counts '
              'consistently instead and k absorbs that.'),
        ('f', 'fig_rate_pipeline.png'),
        ('c', 'The pipeline worked through one 60-second stretch. Every '
              'detected peak is marked; 20 peaks over the window give 19.6 per '
              'minute, which divided by k = 1.18 gives 16.6 breaths per minute.'),

        ('t', 'k is the ratio between what the detector counts and what the PSG '
              'reference records, taken as the median of that ratio over a '
              'recording’s valid epochs with individual ratios clipped to '
              '0.3–5.0. It is reported per night, per participant and per '
              'channel. k is not a fitted correction but a measurement of what '
              'the waveform contains: with peak counting, breathing gives '
              f'{kc("resp","peaks_loose","CH")} on CH, '
              f'{kc("resp","peaks_loose","CLE")} on CLE and '
              f'{kc("resp","peaks_loose","CRE")} on CRE — roughly one '
              'detected peak per breath — while cardiac activity gives '
              f'{kc("card","peaks_loose","CH")}, '
              f'{kc("card","peaks_loose","CLE")} and '
              f'{kc("card","peaks_loose","CRE")}, two deflections per '
              'heartbeat.'),
        ('f', 'fig_k_by_channel.png'),
        ('c', 'k for every night, participant and channel, for both detectors '
              'and both bands. One point per night, coloured by participant '
              'age; the black bar is the median of the twelve; the dotted line '
              'marks k = 1.'),

        ('t', 'The spectral detector takes the strongest peak of the Welch '
              'spectrum inside the band instead of counting peaks in time. For '
              'cardiac activity it lands near 1, so it finds the fundamental '
              'rather than a harmonic, but its spread across nights is two to '
              'three times wider than peak counting’s. For respiration its '
              'k is identical on all three channels, 0.96 with identical '
              'spread, because that estimator returns nearly the same value in '
              'every epoch; its k is therefore that constant divided by the '
              'reference and carries no channel information. It is shown so '
              'that the degeneracy is visible rather than asserted.'),

        ('t', 'A caveat applies wherever k appears. The cardiac reference was '
              'checked against R-peaks detected on the raw ECG of the same '
              'recording. It agrees on eight of the twelve, but runs 36% high '
              'on S2N1 and 29% high on S6N1, and cannot be checked on S5N1 or '
              'S6N2 where the ECG channel is unusable; S6N2’s reference of '
              '129 beats/min median during sleep is implausible on its face. '
              'Because k is a count divided by that reference, an inflated '
              'reference reads as a deflated k, and those three recordings are '
              'exactly the three that sit below the rest. Over the eight '
              'recordings with a verified reference the cardiac k is 1.98, '
              'range 1.81–2.28. The low values should not be read as a '
              'property of the sensor until the reference is corrected.'),

        ('t', 'Before the division by k it is worth seeing what is counted. '
              'The raw count, undivided, against the reference makes the gap '
              'between them — which is k — directly visible, and the '
              'implied per-epoch k is shown below it.'),
        ('f', 'fig_rate_counts_and_k.png'),
        ('c', 'Top: the raw peak count per minute for each channel against the '
              'PSG reference, undivided. Bottom: the implied per-epoch k, with '
              'each channel’s single whole-night k drawn flat across it. '
              'All traces smoothed over five epochs.'),

        ('t', 'The cardiac panel is the more revealing. The counted rate sits '
              'near 120 per minute for the whole night and barely moves while '
              'the reference varies between about 55 and 80, so after division '
              'by k the estimate is close to constant too: what brings the '
              'night average near the right value is k, not the counting. The '
              'respiratory count does follow part of the reference’s shape '
              'but with a large and varying offset. In the lower row the three '
              'channels move together almost exactly, so the excursions in k '
              'are not channel noise.'),

        ('t', 'Applying each recording’s single k epoch by epoch gives the '
              'output the pipeline would produce for a night. Across all '
              'twelve recordings the estimate holds a plausible level and does '
              'not follow the reference’s excursions, which is the '
              'near-zero within-night correlation shown rather than tabulated.'),
        ('f', 'fig_rate_allsessions.png'),
        ('c', 'All twelve recordings. A row per recording and a column per '
              'band; the PSG reference is black and each SEC channel is '
              'converted by its own k. Traces are smoothed over five epochs, '
              'and each panel prints the median absolute difference per '
              'channel.'),

        ('t', 'The question to ask of any rate estimate is whether it beats not '
              'having the sensor. The comparison below predicts the '
              'cohort-median rate with no SEC input and uses that as the bar, '
              'under three calibrations: k learned on the night being reported, '
              'k taken from the same participant’s other night, and k '
              'taken from everyone else. Only the last two could be used in '
              'practice.'),
        ('f', 'fig_rate_result.png'),
        ('c', 'Error in the night-average rate under each calibration, against '
              'a no-sensor baseline (dashed line). Bars below the line beat the '
              'baseline.'),

        ('t', 'Night-average breathing rate beats the baseline under every '
              f'calibration ({med(hd.loc["resp","night_self"]):.2f}, '
              f'{med(hd.loc["resp","night_cross"]):.2f} and '
              f'{med(hd.loc["resp","night_pop"]):.2f} against '
              f'{med(hd.loc["resp","night_nosensor"]):.2f} breaths/min). '
              'Night-average heart rate does not: under either transferable '
              f'calibration the error ({med(hd.loc["card","night_cross"]):.2f} '
              f'and {med(hd.loc["card","night_pop"]):.2f} beats/min) exceeds '
              f'the baseline of {med(hd.loc["card","night_nosensor"]):.2f}. '
              'Epoch by epoch neither band beats the baseline. The honest '
              'summary is that one quantity survives, the night average of '
              'breathing rate, and the rest do not. The cardiac comparison '
              'should be read with the reference caveat above, since two of '
              'the recordings weighing on it have references that read high.'),

        ('t', 'Finally, how much a single k per recording gets wrong. A '
              'per-epoch k computed from the reference is not an estimator — '
              'it uses the answer — but it measures how far the ratio '
              'moves. Within a recording it is fairly steady, with an '
              'interquartile width of about 0.18–0.20 for breathing and '
              '0.19–0.24 for cardiac activity. Between recordings it moves '
              'much more. The twelve recordings are six participants on two '
              'nights each and are independent, so nothing is averaged across '
              'them.'),
        ('f', 'fig_k_per_epoch.png'),
        ('c', 'Median per-epoch k and its interquartile range, for every '
              'recording and channel, in both bands. Each recording stands '
              'alone.'),

        ('t', 'No capacitive feature varied with participant age, k included, '
              'in either band. With six participants no correlation is computed '
              'and none should be read from the figures; the ages are printed '
              'so that a reader can see the spread for themselves. Cardiac k '
              'sits near 2 for the 37, 54, 55 and 61 year olds, and the two '
              'recordings furthest from that belong to the youngest and the '
              'oldest participant — the same two whose cardiac reference '
              'is in question.'),

        ('t', 'Limitations. Six participants and twelve nights; every number '
              'here is descriptive and no statistical test is reported. '
              'Reference rates come from PSG channels measuring different '
              'physical quantities from the SEC sensor, so some disagreement is '
              'expected and is not separable from sensor error, and on four '
              'recordings the cardiac reference is either demonstrably high or '
              'unverifiable. The respiratory reference range across this cohort '
              'is narrow, which is why a constant predictor performs as well as '
              'it does. k is treated as one number per recording, which the '
              'per-epoch figure shows is an approximation.'),
    ]


def main():
    if not SRC.exists():
        raise SystemExit(f'missing: {SRC}')
    hd = pd.read_csv(HELDOUT).set_index('band')
    k = pd.read_csv(KTAB)
    k = k[k.method.isin(['peaks_loose', 'spectral'])]

    doc = Doc(SRC)
    paras = doc.paras
    n0 = len(paras)

    # templates taken before anything is removed
    head_tmpl = next(p for p in paras
                     if doc.text(p).strip().startswith('S3. Integrated'))
    body_tmpl = next(p for p in paras if len(doc.text(p)) > 300)
    cap_tmpl = next(p for p in paras
                    if doc.text(p).strip().startswith('Figure S1.'))
    img_tmpl = next(p for p in paras
                    if len(list(p.iter('{%s}blip' % A))) == 1
                    and not doc.text(p).strip())
    tail = paras[-1]

    # ── 1. remove the old rate material ──────────────────────────────────────
    doomed = []
    for lo, hi in CUT:
        doomed.extend(paras[lo - 1:hi])
    for p in doomed:
        if p.getparent() is not None:
            p.getparent().remove(p)
    # Table S2, by its table element
    for tbl in list(doc.root.iter(q('tbl'))):
        txt = ''.join(t.text or '' for t in tbl.iter(q('t')))
        if CUT_TABLE_STARTING in txt and 'Resp night err' in txt:
            tbl.getparent().remove(tbl)
            break
    dropped, patched = repair_tables(doc)
    print(f'  removed {len(doomed)} paragraphs + Table S2; '
          f'dropped {dropped} emptied tables, repaired {patched} cells')

    # ── 2. append the new rate section ───────────────────────────────────────
    cursor = tail
    n_fig = 0
    for kind, payload in rate_section(hd, k):
        if kind == 'f':
            png = FIGS / payload
            if not png.exists():
                raise SystemExit(f'missing figure: {png}')
            node = make_image_para(doc, img_tmpl, png, f'rate {payload}')
            n_fig += 1
        elif kind == 'c':
            node = clone_text(cap_tmpl, 'CAPTION_PLACEHOLDER ' + payload)
        elif kind == 'h':
            node = clone_text(head_tmpl, payload)
        else:
            node = clone_text(body_tmpl, payload)
        cursor.addnext(node)
        cursor = node

    # ── 3. renumber every figure caption in document order ───────────────────
    # the trailing \s* swallows the space before the caption text, so the
    # replacement puts one back rather than producing "Figure S2.Left-right"
    pat = re.compile(r'^\s*(Fig(?:ure)?\.?\s*S)\s*(\d+)\s*\.?\s*')
    n = 0
    for p in doc.paras:
        t = doc.text(p)
        if t.startswith('CAPTION_PLACEHOLDER'):
            n += 1
            ts = list(p.iter(q('t')))
            ts[0].text = ts[0].text.replace('CAPTION_PLACEHOLDER',
                                            f'Figure S{n}.', 1)
            continue
        m = pat.match(t)
        if m:
            n += 1
            ts = list(p.iter(q('t')))
            joined = ''.join(x.text or '' for x in ts)
            new = pat.sub(f'Figure S{n}. ', joined, count=1)
            ts[0].text = new
            for x in ts[1:]:
                x.text = ''
    print(f'  renumbered {n} figure captions in document order')

    kept, freed = prune_orphan_media(doc)
    print(f'  pruned orphan media: {freed:.1f} MB freed, {kept} images still referenced')

    doc.save(DST)

    chk = Doc(DST)
    caps = [chk.text(p) for p in chk.paras
            if re.match(r'^\s*Figure S\d+\.', chk.text(p))]
    assert 'CAPTION_PLACEHOLDER' not in '\n'.join(chk.text(p) for p in chk.paras)
    assert 'Bland' not in '\n'.join(chk.text(p) for p in chk.paras), \
        'Bland-Altman survived'
    # the check that matters for Word: every cell must still hold a paragraph
    bad = [tc for tc in chk.root.iter(q('tc')) if tc.find(q('p')) is None]
    assert not bad, f'{len(bad)} table cells have no paragraph; Word will reject this'
    ct = chk.parts['[Content_Types].xml'].decode('utf8', 'ignore')
    assert 'Extension="png"' in ct, 'png not declared in [Content_Types].xml'
    print(f'built {DST.name}')
    print(f'  paragraphs {n0} -> {len(chk.paras)}, '
          f'images {len([x for x in chk.parts if x.startswith("word/media/")])}, '
          f'{n_fig} new rate figures')
    for c in caps:
        print('   ', c[:96])


if __name__ == '__main__':
    main()
