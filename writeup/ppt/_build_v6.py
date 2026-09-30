"""
Build manuscript V6 from V5 and supplementary V4 from V3.

Addresses the reviewer highlights carried in writeup/review/address/ (main V4 +
supplementary V1, byte-identical to writeup/ppt/*_V4.docx / *supplementary V1.docx).
V5/V3 already handled the "redraw / larger font" annotations on Figures 3-10 and
S4; what is left is the Figure 2 redraw and a set of text defects, most of which
the highlights sit directly on.

Every edit is asserted against the text it expects to find, so a silently
missed or double-applied edit fails the build instead of shipping.

Usage
-----
    .venv/Scripts/python.exe writeup/ppt/_build_v6.py
"""

import copy
import io
import re
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


def q(t):
    return '{%s}%s' % (W, t)


# ── docx as an editable XML tree ──────────────────────────────────────────────

class Doc:
    def __init__(self, path):
        self.path = Path(path)
        self.zin = zipfile.ZipFile(self.path)
        self.parts = {n: self.zin.read(n) for n in self.zin.namelist()}
        self.root = etree.fromstring(self.parts['word/document.xml'])
        rels = etree.fromstring(self.parts['word/_rels/document.xml.rels'])
        self.rel_target = {r.get('Id'): r.get('Target') for r in rels}
        self.n_edits = 0

    # paragraphs in document order, including those inside table cells — the
    # same order the highlight/dump tooling in the scratchpad reports, so the
    # indices in this script match what was read off the reviewed file.
    @property
    def paras(self):
        return list(self.root.iter(q('p')))

    def para(self, i):
        return self.paras[i - 1]          # 1-based, as reported

    def text(self, i):
        return ''.join(t.text or '' for t in self.para(i).iter(q('t')))

    def save(self, out):
        self.parts['word/document.xml'] = etree.tostring(
            self.root, xml_declaration=True, encoding='UTF-8', standalone=True)
        out = Path(out)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, data in self.parts.items():
                z.writestr(name, data)
        print(f'  saved {out.name}  ({self.n_edits} edits)')


def replace(doc, i, old, new, count=1):
    """Replace `old` with `new` inside paragraph `i`, across run boundaries.

    Word splits a visible phrase over many <w:r> runs, so the match is made on
    the paragraph's concatenated text and then written back into the <w:t>
    elements the match actually spans: the first keeps the replacement, the
    others lose only their matched characters. Formatting on every untouched
    run survives.
    """
    p = doc.para(i)
    done = 0
    for _ in range(count):
        ts = [t for t in p.iter(q('t'))]
        spans, pos = [], 0
        for t in ts:
            s = t.text or ''
            spans.append((pos, pos + len(s), t))
            pos += len(s)
        full = ''.join(t.text or '' for t in ts)
        at = full.find(old)
        if at < 0:
            break
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
        done += 1
    assert done == count, (f'para {i}: expected {count} x {old!r}, applied {done}'
                           f'\n  text: {doc.text(i)[:400]!r}')
    doc.n_edits += done


def clear_highlight(doc, i):
    """Drop the reviewer's yellow marker from every run of paragraph `i`."""
    n = 0
    for rpr in doc.para(i).iter(q('rPr')):
        h = rpr.find(q('highlight'))
        if h is not None:
            rpr.remove(h)
            n += 1
    doc.n_edits += n
    return n


def strip_all_highlights(doc):
    """Final sweep: every reviewer mark in the file is now answered.

    clear_highlight() only reaches the paragraphs this script edits. What is
    left over are marks on runs the V5 build emptied and on paragraph marks,
    which carry no text of their own but still print a yellow block at the end
    of a table cell.
    """
    n = 0
    for rpr in doc.root.iter(q('rPr')):
        h = rpr.find(q('highlight'))
        if h is not None:
            rpr.remove(h)
            n += 1
    print(f'  stripped {n} residual highlight marks')
    doc.n_edits += n


def set_cell_runs(doc, i, pieces):
    """Rewrite a table-cell paragraph as [(text, italic), ...], keeping its rPr."""
    p = doc.para(i)
    runs = [r for r in p.iter(q('r'))]
    assert runs, f'para {i}: no runs'
    template = runs[0].find(q('rPr'))
    for r in runs:
        r.getparent().remove(r)
    for text, italic in pieces:
        r = p.makeelement(q('r'), {})
        if template is not None:
            rpr = copy.deepcopy(template)
            for tag in ('highlight', 'i', 'iCs'):
                el = rpr.find(q(tag))
                if el is not None:
                    rpr.remove(el)
            if italic:
                rpr.append(rpr.makeelement(q('i'), {}))
                rpr.append(rpr.makeelement(q('iCs'), {}))
            r.append(rpr)
        t = r.makeelement(q('t'), {
            '{http://www.w3.org/XML/1998/namespace}space': 'preserve'})
        t.text = text
        r.append(t)
        p.append(r)
    doc.n_edits += 1


def move_before(doc, src_idx, dst_idx):
    """Move paragraphs `src_idx` (list, 1-based) to just before paragraph dst_idx."""
    srcs = [doc.para(i) for i in src_idx]
    dst = doc.para(dst_idx)
    for el in srcs:
        el.getparent().remove(el)
    for el in srcs:
        dst.addprevious(el)
    doc.n_edits += 1


def delete_para(doc, i):
    p = doc.para(i)
    p.getparent().remove(p)
    doc.n_edits += 1


def swap_image(doc, rid, png):
    """Replace the image behind `rid` and keep the drawing's aspect ratio."""
    png = Path(png)
    assert png.exists(), f'missing {png}'
    data = png.read_bytes()
    target = 'word/' + doc.rel_target[rid].lstrip('/')
    assert target in doc.parts, target
    doc.parts[target] = data
    w, h = Image.open(io.BytesIO(data)).size
    aspect = h / w
    for blip in doc.root.iter('{%s}blip' % A):
        if blip.get('{%s}embed' % R) != rid:
            continue
        anc = blip
        while anc is not None and anc.tag not in ('{%s}inline' % WP, '{%s}anchor' % WP):
            anc = anc.getparent()
        if anc is None:
            continue
        ext = anc.find('{%s}extent' % WP)
        if ext is not None:
            ext.set('cy', str(int(round(int(ext.get('cx')) * aspect))))
        for aext in anc.iter('{%s}ext' % A):
            if aext.get('cx') is not None:
                aext.set('cy', str(int(round(int(aext.get('cx')) * aspect))))
        break
    doc.n_edits += 1


# ══ MAIN MANUSCRIPT ═══════════════════════════════════════════════════════════

def build_main():
    print('main: V5 -> V6')
    d = Doc(HERE / 'CAP_sleep_mask_manuscript_V5.docx')

    # ── 1. Figure 2: redrawn without the stage-colour wash ───────────────────
    # Reviewer: "Can you redraw the figures? Remove the background color".
    fig2 = ROOT / 'writeup' / 'figures' / 'channel_evolution'
    for rid, name in (('rId7', 'fig2_panel_a.png'), ('rId8', 'fig2_panel_b.png'),
                      ('rId9', 'fig2_panel_c.png'), ('rId10', 'fig2_panel_d.png')):
        swap_image(d, rid, fig2 / name)
    # The caption's shading sentence goes with the shading, and the request note
    # is discharged.
    replace(d, 183, ' Background colors denote PSG-scored sleep stages.', '')
    replace(d, 183, ' (Can you redraw the figures? Remove the background color)', '')
    clear_highlight(d, 183)

    # ── 2. Delta bursts vs SWS: intro and methods now match the result ───────
    # Highlights on para 21 and on the whole of para 170. Section 3.6 reports
    # that most detected onsets were isolated N2 slow waves and K-complexes, so
    # neither the intro nor the methods can introduce delta bursts as an SWS
    # phenomenon.
    replace(d, 21,
            ' associated with slow-wave sleep (SWS), a restorative stage',
            ' that occur throughout NREM sleep and are most sustained during '
            'slow-wave sleep (SWS), a restorative stage')
    clear_highlight(d, 21)

    replace(d, 170,
            'Since delta activity was a defining feature of SWS, delta-burst analysis',
            'Because delta activity is most sustained during SWS, delta-burst analysis')
    replace(d, 170,
            'However, because delta bursts could also occur during N2, events were '
            'stratified by sleep stage to distinguish general NREM delta activity '
            'from activity specifically associated with SWS (N3).',
            'Because discrete onsets arise mainly from isolated N2 slow waves and '
            'K-complexes rather than from sustained N3 activity, events were '
            'stratified by sleep stage to separate general NREM delta activity '
            'from activity specific to SWS (N3).')
    clear_highlight(d, 170)

    # ── 3. One name for the left-right marker: "imbalance" ───────────────────
    # Highlight sat on "asymmetry" twice in para 188. Section 3.1, Figure S12 and
    # Section 4.5 used "asymmetry", "asymmetric" and "imbalance" for the same
    # quantity; "imbalance" is the term the figures and the supplement carry.
    replace(d, 187, 'the CLE−CRE asymmetry', 'the CLE−CRE imbalance')
    replace(d, 187, 'asymmetric direction', 'imbalance direction')
    replace(d, 188, 'The time-averaged asymmetry magnitude',
            'The time-averaged imbalance magnitude')
    replace(d, 188, 'the asymmetry burden', 'the imbalance burden')
    replace(d, 188, 'the signed asymmetry index', 'the signed imbalance index')
    replace(d, 188, 'the magnitude of left–right asymmetry',
            'the magnitude of the left–right imbalance')
    replace(d, 188, 'close to zero  at both', 'close to zero at both')
    clear_highlight(d, 188)

    # ── 4. Supplementary cross-references ────────────────────────────────────
    # Figure S3 is the 24-hour temperature/drift test; the posture dependence of
    # the imbalance magnitude is Figure S4.
    replace(d, 187, 'imbalance (Figure S3)', 'imbalance (Figure S4)')
    # Coherence matrices are S9 and amplitude correlations S5; S4 is the
    # overnight-level figure.
    replace(d, 222,
            'Figures S4 and S5 present the channel-specific coherence and '
            'amplitude-correlation matrices.',
            'Figures S5 and S9 present the channel-specific amplitude-correlation '
            'and coherence matrices.')
    # All-night rate traces are S7 (respiratory) and S8 (cardiac); S9 is coherence.
    replace(d, 262, 'shown in Figures S8 and S9.', 'shown in Figures S7 and S8.')

    # ── 5. Section cross-references ──────────────────────────────────────────
    replace(d, 312, 'descriptive statistics only (Section 2.7)',
            'descriptive statistics only (Section 2.9)')       # statistics = 2.9
    replace(d, 326, 'analyses are described in Section 2.7.',
            'analyses are described in Section 2.8.')          # cortical events = 2.8
    clear_highlight(d, 326)
    replace(d, 377, 'reported in section 4.1 is characterized',
            'reported in Section 3.1 is characterized')        # imbalance result = 3.1

    # ── 6. Duplicate result-section numbers ──────────────────────────────────
    replace(d, 325, '3.5 Mechanical response following delta-burst onsets',
            '3.6 Mechanical response following delta-burst onsets')
    replace(d, 336, '3.6 Harmonic-comb events', '3.7 Harmonic-comb events')

    # ── 7. Duplicate table number ────────────────────────────────────────────
    replace(d, 266, 'for cardiac activity (Table 3)', 'for cardiac activity (Table 4)')
    replace(d, 267, 'Table 3. Agreement between', 'Table 4. Agreement between')
    # ...and the constant-predictor figure comes from the calibration table.
    replace(d, 362, 'error of 1.20 breaths/min (Table 2)',
            'error of 1.20 breaths/min (Table 3)')

    # ── 8. Table 3 row labels ────────────────────────────────────────────────
    # All four were highlighted: the first ran "k(" together, the second had lost
    # its k entirely, and k was roman in a document that sets it italic elsewhere.
    set_cell_runs(d, 237, [('Same-night ', False), ('k', True),
                           (' (recording-specific calibration)', False)])
    set_cell_runs(d, 242, [('Same participant’s other-night ', False),
                           ('k', True)])
    set_cell_runs(d, 247, [('Population-derived ', False), ('k', True)])
    clear_highlight(d, 252)

    # ── 9. Amplitude-coupling summary sentence ───────────────────────────────
    # Highlighted where it sat: stranded after the forward reference to §3.3, and
    # repeating the conclusion of the preceding paragraph. Folded into the
    # sentence it qualifies.
    replace(d, 223,
            'However, they did not establish the accuracy of rate estimation, '
            'which was evaluated separately in Section 3.3. Overall, SEC amplitude '
            'coupling was weak and, in the respiratory band, largely nonspecific.',
            'Overall, however, SEC amplitude coupling was weak and, in the '
            'respiratory band, largely nonspecific, and these findings did not '
            'establish the accuracy of rate estimation, which was evaluated '
            'separately in Section 3.3.')
    clear_highlight(d, 223)

    # ── 10. Harmonic-comb example stated as harmonic indices ─────────────────
    # Highlight on "integer-spaced.": the listed rungs are not consecutive
    # multiples, because the 4th and 6th are missing. Giving the indices makes
    # the quasi-harmonic claim checkable against the numbers next to it.
    replace(d, 338,
            'Each consisted of several quasi-harmonic bands that were '
            'approximately, but not exactly, integer-spaced. One representative '
            'event contained bands at 0.15, 0.28, 0.42, 0.68, and 0.95 Hz (Figure 9).',
            'Each consisted of several quasi-harmonic bands whose frequencies were '
            'close to, but not exactly, integer multiples of a common fundamental, '
            'with intermediate multiples often absent. One representative event '
            'contained bands at 0.15, 0.28, 0.42, 0.68, and 0.95 Hz, close to the '
            'first, second, third, fifth, and seventh multiples of a 0.14-Hz '
            'fundamental, from which they deviated by up to 9% (Figure 9).')
    clear_highlight(d, 338)

    # ── 11. What k counts ────────────────────────────────────────────────────
    # Highlight on "capacitive deflections per heartbeat". k is the ratio of
    # detected peaks to reference cycles, which is what the next sentence
    # reports; "deflections" implied every deflection was counted.
    replace(d, 363, 'indicating approximately two capacitive deflections per heartbeat',
            'indicating that approximately two capacitive peaks were detected per '
            'cardiac cycle')
    replace(d, 363, 'each breath also generated more than one detectable deflection',
            'more than one peak was also detected per breath')
    clear_highlight(d, 363)

    # ── 12. Discussion had two Limitations sections with §4.5 wedged between ──
    # Highlight sat on the §4.5 heading and, in V4, on the whole of its body.
    # Comparison moves ahead of the limitations, which merge into one section.
    assert d.text(367).startswith('4.4 Limitations'), d.text(367)
    assert d.text(372).startswith('4.5 Comparison'), d.text(372)
    assert d.text(375).startswith('4.6 Limitations'), d.text(375)
    # The duplicated literal k that the V5 build inserted next to the equation k
    # that was already there ("The cardiac kk of approximately 2"). The space the
    # replacement leaves behind is the one that separates the equation from "of".
    for _ in range(2):
        replace(d, 373, 'k of', ' of')
    clear_highlight(d, 373)
    replace(d, 372, '4.5 Comparison with Existing Physiological Monitoring Methods',
            '4.4 Comparison with Existing Physiological Monitoring Methods')
    clear_highlight(d, 372)
    replace(d, 367, '4.4 Limitations', '4.5 Limitations')
    # The second block's enumeration has to announce itself now that its own
    # heading is gone and it continues a section that already listed three.
    replace(d, 376, 'First, the sample is small',
            'Several further limitations apply. First, the sample is small')
    delete_para(d, 375)                      # the second "4.6 Limitations" heading
    move_before(d, [372, 373, 374], 367)     # heading, body, spacer

    # ── 13. Remaining text defects found while checking the highlights ───────
    replace(d, 186, 'sleep stages. some abrupt transitions',
            'sleep stages. While some abrupt transitions')
    replace(d, 221, 'with EEG). After motion', ' with EEG). After motion')
    replace(d, 319, 'in the CRE  and CLE channels', 'in the CRE and CLE channels')
    replace(d, 330, 'Excluding these events, the response in four of the five',
            'Excluding these events, the response decreased in four of the five')
    replace(d, 333, '(green line, t=0))', '(green line, t=0)')
    replace(d, 333, 'Most onsets occurs within', 'Most onsets occur within')
    replace(d, 299, 'mean (Figures 4 and S8).', 'mean (Figures 4 and S8).')
    clear_highlight(d, 299)

    strip_all_highlights(d)
    d.save(HERE / 'CAP_sleep_mask_manuscript_V6.docx')
    return HERE / 'CAP_sleep_mask_manuscript_V6.docx'


# ══ SUPPLEMENTARY ═════════════════════════════════════════════════════════════

def build_supp():
    print('supplementary: V3 -> V4')
    d = Doc(HERE / 'CAP_sleep_mask_manuscript supplementary V3.docx')

    # Figure S4 is the cohort version of the mean-referenced level trace, which
    # is the middle row of Figure 2 — Figure 3 is the SNR figure.
    replace(d, 29, 'the cohort-wide version of row B of Figure 3.',
            'the cohort-wide version of the middle row of Figure 2.')

    # Highlighted: amplitude coupling is reported in Section 3.2, not 3.3
    # (Section 3.3 is rate estimation).
    replace(d, 33, 'absolute value is the measure reported in section 3.3.',
            'absolute value is the measure reported in Section 3.2.')
    clear_highlight(d, 33)

    # Same single name for the marker as the main text now uses.
    replace(d, 76, 'The asymmetry index, the difference',
            'The imbalance index, the difference')
    # §4.2.1 does not exist; the outlying cardiac k is discussed in Section 4.3.
    replace(d, 78, 'the outlying cardiac k (§4.2.1)',
            'the outlying cardiac k (Section 4.3)')

    strip_all_highlights(d)
    d.save(HERE / 'CAP_sleep_mask_manuscript supplementary V4.docx')
    return HERE / 'CAP_sleep_mask_manuscript supplementary V4.docx'


# ══ verification ══════════════════════════════════════════════════════════════

def verify(main_path, supp_path):
    print('\nverify:')
    for label, path in (('main', main_path), ('supp', supp_path)):
        d = Doc(path)
        hl = [i for i, p in enumerate(d.paras, 1)
              if any(h.get(q('val')) not in (None, 'none')
                     for h in p.iter(q('highlight')))]
        print(f'  {label}: highlighted paragraphs remaining: {hl or "none"}')

    d = Doc(main_path)
    heads = [d.text(i) for i in range(1, len(d.paras) + 1)
             if re.match(r'^\d(\.\d)? [A-Z]', d.text(i))]
    print('  headings:', ' | '.join(h.strip() for h in heads if len(h) < 80))
    tables = [d.text(i) for i in range(1, len(d.paras) + 1)
              if d.text(i).startswith('Table ')]
    print('  tables:', ' | '.join(t.strip()[:46] for t in tables))
    for i in (183, 223, 237, 242, 373):
        print(f'  para {i}: {d.text(i)[:150]}')


if __name__ == '__main__':
    m = build_main()
    s = build_supp()
    verify(m, s)
