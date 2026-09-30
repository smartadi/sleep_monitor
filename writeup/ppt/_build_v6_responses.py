"""
Build the response copies: V6 + supplementary V4 with the reviewer's highlights
put back and a Word comment on each one saying how it was resolved.

The files the reviewer sent (writeup/review/address/CAP_sleep_mask_manuscript_V4.docx
and ...supplementary V1.docx) are never written to by anything in this repo and
keep every mark and note he made. This script produces a SEPARATE document:

    CAP_sleep_mask_manuscript_V6_responses.docx
    CAP_sleep_mask_manuscript supplementary V4_responses.docx

Each of the 26 highlighted spans is highlighted again at the place the change
landed, and carries a margin comment that quotes his note verbatim where he left
one, says what the mark was flagging, and says what was done. Nothing of his is
dropped: where a note was an instruction to us ("(LARGER font size)") the note
does not go back into the manuscript body, it is quoted inside the comment.

Comments marked [Additional fix] are on defects found while checking his marks
that he had not highlighted. Those are commented but NOT highlighted, so his own
marks stay distinguishable from ours in the margin.

Usage
-----
    .venv/Scripts/python.exe writeup/ppt/_build_v6_responses.py
"""

import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _build_v6 import Doc, q, W                      # noqa: E402

XSPACE = '{http://www.w3.org/XML/1998/namespace}space'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
PR = 'http://schemas.openxmlformats.org/package/2006/relationships'
COMMENTS_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments'
COMMENTS_CT = ('application/vnd.openxmlformats-officedocument.'
               'wordprocessingml.comments+xml')

AUTHOR = 'Revision response'
INITIALS = 'RR'
STAMP = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ── run surgery ───────────────────────────────────────────────────────────────

def text_runs(p):
    """Runs of this paragraph that carry exactly one w:t, in document order.

    Equation runs live in the math namespace (m:r/m:t), so they are not in this
    list and contribute nothing to the offsets below — which is what we want,
    since the anchors are written against the w:t text.
    """
    out = []
    for r in p.iter(q('r')):
        ts = r.findall(q('t'))
        if len(ts) > 1:                               # coalesce, keeping the run
            assert r.find(q('br')) is None, 'run with w:br cannot be coalesced'
            ts[0].text = ''.join(t.text or '' for t in ts)
            for t in ts[1:]:
                r.remove(t)
            ts = ts[:1]
        if ts:
            out.append(r)
    return out


def run_text(r):
    return r.find(q('t')).text or ''


def ensure_boundary(p, offset):
    """Split whichever run straddles `offset` so a run boundary exists there."""
    pos = 0
    for r in text_runs(p):
        s = run_text(r)
        if pos < offset < pos + len(s):
            new = copy.deepcopy(r)
            new.find(q('t')).text = s[:offset - pos]
            r.find(q('t')).text = s[offset - pos:]
            for el in (new.find(q('t')), r.find(q('t'))):
                el.set(XSPACE, 'preserve')
            r.addprevious(new)
            return
        pos += len(s)


def runs_covering(p, start, end):
    runs, pos, out = text_runs(p), 0, []
    for r in runs:
        s = run_text(r)
        if pos >= start and pos + len(s) <= end and s:
            out.append(r)
        pos += len(s)
    return out


def set_highlight(r, color='yellow'):
    rpr = r.find(q('rPr'))
    if rpr is None:
        rpr = r.makeelement(q('rPr'), {})
        r.insert(0, rpr)
    old = rpr.find(q('highlight'))
    if old is not None:
        rpr.remove(old)
    h = rpr.makeelement(q('highlight'), {q('val'): color})
    rpr.append(h)


# ── comments part ─────────────────────────────────────────────────────────────

class Commenter:
    def __init__(self, doc):
        self.doc = doc
        self.items = []                               # (id, body)
        self.has_style = b'"CommentReference"' in doc.parts.get('word/styles.xml', b'') \
            or b'w:styleId="CommentReference"' in doc.parts.get('word/styles.xml', b'')

    def add(self, para_idx, anchor, body, highlight=True):
        p = self.doc.para(para_idx)
        runs = text_runs(p)
        full = ''.join(run_text(r) for r in runs)
        at = full.find(anchor)
        assert at >= 0, (f'para {para_idx}: anchor not found\n  want: {anchor!r}'
                         f'\n  text: {full[:400]!r}')
        end = at + len(anchor)
        ensure_boundary(p, end)
        ensure_boundary(p, at)
        sel = runs_covering(p, at, end)
        assert sel, f'para {para_idx}: no runs selected for {anchor!r}'

        cid = str(len(self.items))
        start = p.makeelement(q('commentRangeStart'), {q('id'): cid})
        sel[0].addprevious(start)
        stop = p.makeelement(q('commentRangeEnd'), {q('id'): cid})
        sel[-1].addnext(stop)
        ref = p.makeelement(q('r'), {})
        if self.has_style:
            rpr = ref.makeelement(q('rPr'), {})
            rpr.append(rpr.makeelement(q('rStyle'), {q('val'): 'CommentReference'}))
            ref.append(rpr)
        ref.append(ref.makeelement(q('commentReference'), {q('id'): cid}))
        stop.addnext(ref)

        if highlight:
            for r in sel:
                set_highlight(r)

        self.items.append((cid, body))
        self.doc.n_edits += 1

    def write(self):
        """Emit word/comments.xml and wire up the relationship + content type."""
        nsmap = {'w': W}
        root = etree.Element(q('comments'), nsmap=nsmap)
        for cid, body in self.items:
            c = etree.SubElement(root, q('comment'), {
                q('id'): cid, q('author'): AUTHOR,
                q('date'): STAMP, q('initials'): INITIALS})
            for line in body.split('\n'):
                p = etree.SubElement(c, q('p'))
                r = etree.SubElement(p, q('r'))
                t = etree.SubElement(r, q('t'), {XSPACE: 'preserve'})
                t.text = line
        self.doc.parts['word/comments.xml'] = etree.tostring(
            root, xml_declaration=True, encoding='UTF-8', standalone=True)

        rels = etree.fromstring(self.doc.parts['word/_rels/document.xml.rels'])
        if not any(r.get('Type') == COMMENTS_REL for r in rels):
            used = {r.get('Id') for r in rels}
            rid = next(f'rId{n}' for n in range(900, 999) if f'rId{n}' not in used)
            etree.SubElement(rels, '{%s}Relationship' % PR, {
                'Id': rid, 'Type': COMMENTS_REL, 'Target': 'comments.xml'})
            self.doc.parts['word/_rels/document.xml.rels'] = etree.tostring(
                rels, xml_declaration=True, encoding='UTF-8', standalone=True)

        types = etree.fromstring(self.doc.parts['[Content_Types].xml'])
        if not any(o.get('PartName') == '/word/comments.xml' for o in types):
            etree.SubElement(types, '{%s}Override' % CT, {
                'PartName': '/word/comments.xml', 'ContentType': COMMENTS_CT})
            self.doc.parts['[Content_Types].xml'] = etree.tostring(
                types, xml_declaration=True, encoding='UTF-8', standalone=True)
        print(f'  {len(self.items)} comments')


# ── the comment bodies ────────────────────────────────────────────────────────
# (paragraph index in V6, anchor text, body, highlight?)
# Paragraph indices count every w:p in document order, table cells included —
# the same numbering _build_v6.py works in.

def fig_note(figure, note, what):
    return (f'Your note on {figure}: "{note}"\n\n{what}\n\n'
            f'The note itself is not carried into the manuscript body; it is '
            f'quoted here so nothing you wrote is lost. Your annotated V4 keeps '
            f'it in place.')


MAIN = [
    (21, 'that occur throughout NREM sleep and are most sustained during slow-wave sleep',
     'Highlighted in V4 on "...associated with slow-wave sleep".\n\n'
     'Section 3.6 reports that most detected delta-burst onsets were isolated N2 '
     'slow waves and K-complexes, because sustained N3 slow-wave activity rarely '
     'has a discrete onset after a quiet baseline. The Introduction was therefore '
     'promising an SWS-specific phenomenon that the Results contradict.\n\n'
     'Now: delta bursts occur throughout NREM and are MOST SUSTAINED during SWS. '
     'The last paragraph of Section 2.8, which you also highlighted, carries the '
     'matching change.',
     True),

    (170, 'Because delta activity is most sustained during SWS',
     'The whole of this paragraph was highlighted in V4.\n\n'
     'It read "Since delta activity was a defining feature of SWS..." and then '
     'walked that back in the next sentence ("However, because delta bursts could '
     'also occur during N2..."). The same mismatch with Section 3.6 as the '
     'Introduction.\n\n'
     'Rewritten so the N2 / K-complex origin of the detected onsets is stated up '
     'front as the reason for stratifying by stage, rather than as a caveat.',
     True),

    (183, 'Figure 2.',
     fig_note('Figure 2', 'Can you redraw the figures? Remove the background color',
              'Done, and it needed a new script: the four panels had been cropped '
              'out of the per-session evolution figures by hand, so nothing in the '
              'repository produced them. analysis/mean_value/fig2_overnight_panels.py '
              'now builds them directly for S6N2 (25 y), S3N2 (37 y), S4N2 (54 y) '
              'and S2N2 (66 y), importing every numeric step from '
              'channel_evolution.py so this figure and the per-session figures '
              'cannot disagree.\n\n'
              'Three changes: the stage-colour wash is gone from the level and '
              'head-turn rows (the hypnogram directly above already carries the '
              'stages at full saturation); type is up from 11 pt to 16 pt base; '
              'and THE TIME AXIS IS NOW DRAWN - it had been cropped off every '
              'pasted panel, so the traces carried no time reference at all.\n\n'
              'The caption sentence "Background colors denote PSG-scored sleep '
              'stages." went with the shading.'),
     True),

    (188, 'The time-averaged imbalance magnitude',
     'Highlighted in V4 on "asymmetry", twice in this paragraph.\n\n'
     'The same quantity was being called asymmetry, asymmetric direction and '
     'imbalance - sometimes within one paragraph - while the figures, the '
     'supplement (Figure S12, "Integrated capacitance imbalance") and Section 4.5 '
     'all call it imbalance.\n\n'
     'Standardised on IMBALANCE throughout Section 3.1, Section 4.5 and the '
     'Figure S12 caption. No change of meaning.',
     True),

    (188, '(Figure S12)',
     'Highlighted in V4 as "(Figure S4)".\n\n'
     'Wrong cross-reference: S4 is the overnight sensor-level figure. The '
     'imbalance integral is Figure S12. Corrected in V5, during the supplementary '
     'renumber to S1-S12.',
     True),

    (193, 'Figure 3.',
     fig_note('Figure 3', 'Redraw this with the larger font size',
              'Done in V5. writeup/figures/signal_validation/inband_snr.py: '
              'figure 12.5x5.8 -> 14.5x7.2 in, ticks 9 -> 13 pt, labels 11 -> 15, '
              'title 13 -> 17, legend 9.5 -> 13.'),
     True),

    (223, 'Overall, however, SEC amplitude coupling was weak and, in the respiratory '
          'band, largely nonspecific',
     'Highlighted in V4 where it sat, as a separate sentence at the END of the '
     'paragraph.\n\n'
     'Two problems there: it came after the forward reference to Section 3.3, so '
     'the paragraph did not end on its own point; and it repeated the conclusion '
     'the preceding paragraph had already drawn ("SEC respiratory amplitude should '
     'not be interpreted as a surrogate for respiratory effort or volume").\n\n'
     'Folded into the sentence it qualifies. Nothing dropped.',
     True),

    (237, 'Same-night k (recording-specific calibration)',
     'All four row labels of this table were highlighted in V4. Taken together '
     'they were a labelling problem, and each had a different fault:\n\n'
     '1. This one ran "k(" together with no space, and set k roman.\n'
     '2. "Same participant’s other-night" HAD LOST ITS k ENTIRELY - the label '
     'named no quantity.\n'
     '3. "Population-derived k" set k roman.\n'
     '4. "No-sensor baseline (cohort-median rate)" was correct.\n\n'
     'All four rewritten consistently, k italic as it is set everywhere else in '
     'the document.',
     True),

    (242, 'Same participant’s other-night k',
     'Highlighted in V4, where this label read "Same participant’s other-night " '
     'and stopped - the k had been lost, so the row named no quantity. Restored, '
     'italic, consistent with the other three rows.',
     True),

    (247, 'Population-derived k',
     'Highlighted in V4. k was roman here and italic elsewhere in the document; '
     'set italic.',
     True),

    (252, 'No-sensor baseline (cohort-median rate)',
     'Highlighted in V4 with the other three row labels. This label itself was '
     'correct and is unchanged.\n\n'
     'Worth a second look at the ROW, though, if that is what the mark meant: on '
     'respiratory epoch-level error this baseline (1.29 br/min) BEATS same-night k '
     'calibration (1.79 br/min). Section 4.3 explains why a constant predictor does '
     'well over a 14.4-16.8 br/min reference range, but nowhere does the text say '
     'outright that at epoch level the constant predictor wins. Left as a content '
     'decision for you - better said plainly here than found by a reviewer.',
     True),

    (262, 'Figure 4.',
     fig_note('Figure 4', 'Larger font',
              'Done in V5. analysis/rates/rerun_session_plots.py: shared rcParams '
              '7/7.5/6.5 -> 11/11.5/10 pt, representative-night figure 183x88 -> '
              '205x108 mm, panel letters 9 -> 13, in-panel text 6.5 -> 9.5.')
     + '\n\n[Additional fix] This caption also pointed the reader to "Figures S8 '
       'and S9" for the twelve per-night traces. After the supplementary renumber '
       'those are S7 (respiratory) and S8 (cardiac); S9 is the coherence figure. '
       'Corrected.',
     True),

    (299, '(Figures 4 and S8)',
     'Highlighted in V4.\n\n'
     'At the time you read it, Figure S8’s own caption referred to itself '
     '("drawn as in Figure S8"), so following this pointer led nowhere. That was '
     'fixed in the S1-S12 renumber - S8 is the cardiac all-nights figure, and this '
     'citation is correct as it stands. No change needed here.',
     True),

    (309, 'Figure 5.',
     fig_note('Figure 5', 'LARGER font size',
              'Done in V5. analysis/slow_wave/ridge_overlay_tune.py '
              '(--session S1N2 --channel CRE): figure 16x9 -> 17x10.5 in, '
              'fonts 7-12 -> 11-15 pt.')
     + '\n\n[Additional fix - needs your eye] This caption describes ONE '
       'three-row composite (hypnogram / spectrogram / slow-band spectrum), but '
       'two separate images sit under it, and only one was replaced in the V5 '
       'redraw. Which image belongs here is a content call I have not made.',
     True),

    (312, 'Figure 6.',
     fig_note('Figure 6', 'LARGER font size',
              'Done in V5. analysis/slow_wave/band_ridge_figure.py: figure '
              '15x8 -> 16.5x9 in, subplot titles/labels 9 -> 12 pt, suptitle '
              '13 -> 16, ticks 11.')
     + '\n\n[Additional fix] The caption attributed the descriptive-statistics '
       'caveat to "Section 2.7". Statistical analysis is Section 2.9; 2.7 is '
       'persistent ridges and harmonic-comb episodes. Corrected.',
     True),

    (322, 'Figure 7.',
     fig_note('Figure 7', 'LARGER font size',
              'Done in V5. analysis/spindles/plot_spindle_lowband_allsessions.py: '
              'figure 18x5.6 -> 18.5x6.6 in, titles 10.5 -> 13 pt, labels/ticks '
              '8-9 -> 10.5-12, suptitle 13 -> 15.5.'),
     True),

    (326, 'described in Section 2.8',
     'Highlighted in V4, where this read "described in Section 2.7".\n\n'
     'Wrong section: the delta-burst detection and event-triggered analyses are in '
     'Section 2.8 (Cortical events and event-triggered analysis). Section 2.7 is '
     'persistent ridges and harmonic-comb episodes. Corrected.',
     True),

    (333, 'Figure 8.',
     fig_note('Figure 8', 'LARGER font size',
              'Done in V5. analysis/delta_onset/delta_onset_figures.py: rcParams '
              'font 9 -> 12 pt, figure 11x6.4 -> 13.5x7.9 in, titles 9.5 -> 12, '
              'suptitle 11 -> 13.5.')
     + '\n\n[Additional fix] Also corrected in this caption: a doubled closing '
       'parenthesis after "t=0", and "Most onsets occurs" -> "occur".\n\n'
       '[Still open - needs your eye] The caption describes a 3 bands x 3 channels '
       'grid; the shipped figure is a 2x3 A-F layout. Caption and figure disagree, '
       'and which one is right is a content call.',
     True),

    (338, 'close to the first, second, third, fifth, and seventh multiples of a '
          '0.14-Hz fundamental',
     'Highlighted in V4 on "integer-spaced."\n\n'
     'The sentence claimed the bands were "approximately, but not exactly, '
     'integer-spaced" and then listed 0.15, 0.28, 0.42, 0.68 and 0.95 Hz - which '
     'are not consecutive multiples of anything, because the 4th and 6th rungs of '
     'the comb are absent. As written the claim did not survive contact with the '
     'numbers beside it.\n\n'
     'Now stated as the harmonic indices, so a reader can check it: least-squares '
     'fundamental over those five rungs is 0.1366 Hz, giving predicted 0.137, '
     '0.273, 0.410, 0.683, 0.956 Hz against the measured values - residuals '
     '+0.013, +0.007, +0.010, -0.003, -0.006 Hz, i.e. up to 9% on the fundamental '
     'and under 4% on the rest. The absence of intermediate multiples is now '
     'stated rather than left for the reader to notice.',
     True),

    (340, 'Figure 9.',
     fig_note('Figure 9', 'LARGER font size',
              'Done in V5. analysis/slow_wave/harmonic_ladder_overlay.py '
              '(--session S6N1): figure 16x11 -> 17x12 in, stage/frequency labels '
              '7-10 -> 11-14 pt, tick labels 12, title 11 -> 14.'),
     True),

    (345, 'Figure 10.',
     fig_note('Figure 10', 'LARGER font size, ADD (a) and (b) to the images',
              'Both done in V5. analysis/slow_wave/ladder_stage_relationship.py: '
              'figure 15x5.5 -> 16x6.2 in, titles/labels 14-15 pt, ticks 12, '
              'legend 8 -> 12 - AND (a)/(b) panel labels added to the images, '
              'which the caption already referred to.'),
     True),

    (363, 'approximately two capacitive peaks were detected per cardiac cycle',
     'Highlighted in V4 as "capacitive deflections per heartbeat".\n\n'
     'k is the ratio of DETECTED PEAKS to reference cycles, which is exactly what '
     'the next sentence reports ("a median of 2.02 peaks per cardiac cycle"). '
     '"Deflections" implied every deflection in the waveform was counted, which is '
     'not what the estimator does and not what k measures.\n\n'
     'Reworded in both bands; the respiratory sentence now reads "more than one '
     'peak was also detected per breath".',
     True),

    (367, '4.4 Comparison with Existing Physiological Monitoring Methods',
     'Highlighted in V4 - the heading, and in V4 the whole of its paragraph.\n\n'
     'The mark turned out to be structural. The Discussion had TWO sections titled '
     '"Limitations", 4.4 and 4.6, with 4.5 Comparison wedged between them, so the '
     'limitations were split in half by an unrelated section.\n\n'
     'Comparison now comes first, as 4.4; the two limitation blocks merge into a '
     'single 4.5 Limitations (its second half opens "Several further limitations '
     'apply." so the enumeration reads continuously).\n\n'
     'STILL OPEN, and a content decision rather than a copy-edit: this section is '
     'titled "Comparison with Existing Physiological Monitoring Methods" but its '
     'paragraph declines to compare - "Direct quantitative comparison with previous '
     'wearable or noncontact sensors is inappropriate...". The heading promises '
     'something the section does not deliver. The optical control session now in '
     'writeup/optical test data/ and "Comparison Optic and Cap.pptx" look like the '
     'intended material; adding that comparison is your call.',
     True),

    (375, 'estimate (Figure S2)',
     'Highlighted in V4 as "estimate (Figure S3),".\n\n'
     'Wrong cross-reference: S3 is the 24-hour temperature and drift test. The '
     'calibration-strategies figure, which is what a ten-minute warm-up estimate '
     'should point at, is S2. Corrected in V5.',
     True),

    # ── defects found while checking the marks; commented, not highlighted ────
    (186, 'While some abrupt transitions',
     '[Additional fix] This sentence had lost its opening clause: it read '
     '"...spanned multiple sleep stages. some abrupt transitions coincided with '
     'changes in head angle, the slower cycles persisted beyond..." - lower-case '
     'after a full stop, and two clauses spliced with no conjunction. Restored as '
     '"While some abrupt transitions...".',
     False),

    (187, 'imbalance (Figure S4)',
     '[Additional fix] This read "(Figure S3)", which is the 24-hour temperature '
     'and drift test - nothing to do with head position. The posture dependence of '
     'the imbalance is visible in S4 (overnight sensor value for all twelve '
     'recordings, whose caption calls out the coincidence with posture change). '
     'Corrected.',
     False),

    (222, 'Figures S5 and S9 present the channel-specific amplitude-correlation and '
          'coherence matrices',
     '[Additional fix] This read "Figures S4 and S5 present the channel-specific '
     'coherence and amplitude-correlation matrices." After the renumber, amplitude '
     'correlation is S5 and coherence is S9; S4 is the overnight level figure. '
     'Corrected, and the two nouns reordered to match the figure numbers.',
     False),

    (325, '3.6 Mechanical response following delta-burst onsets',
     '[Additional fix] There were TWO sections numbered 3.5 - "Sleep spindles" '
     'above and this one. Delta-burst onsets becomes 3.6 and harmonic-comb events '
     'becomes 3.7.',
     False),

    (330, 'the response decreased in four of the five',
     '[Additional fix] This sentence had no verb: "Excluding these events, the '
     'response in four of the five participants with sufficient remaining onsets." '
     'Restored from the numbers that follow it, which are decreases.',
     False),

    (336, '3.7 Harmonic-comb events',
     '[Additional fix] Renumbered from 3.6, following the duplicate 3.5 above.',
     False),

    (266, '(Table 4)',
     '[Additional fix] There were TWO tables numbered 3. "Agreement between '
     'SEC-derived and PSG-reference rates" becomes Table 4, and this citation with '
     'it. Table 3 remains the four-calibration table.',
     False),

    (362, 'error of 1.20 breaths/min (Table 3)',
     '[Additional fix] This cited Table 2 (signal comparison between PSG and SEC). '
     'The 1.20 br/min constant-predictor figure is the no-sensor baseline row of '
     'Table 3. Corrected.',
     False),

    (368, 'The cardiac',
     '[Additional fix - our regression, not yours] In V4 this sentence was missing '
     'its calibration factor entirely ("The cardiac of approximately 2"), which you '
     'had highlighted. The V5 build inserted a literal italic k - beside an '
     'equation k that was already there but had not rendered in our text dump. V5 '
     'therefore printed "The cardiac kk of approximately 2", in both this sentence '
     'and the respiratory one. The duplicate is removed here; the equation k '
     'remains.',
     False),

    (370, '4.5 Limitations',
     '[Additional fix] This was 4.4 Limitations. It now absorbs what was 4.6 '
     'Limitations, which had been separated from it by 4.5 Comparison. One '
     'Limitations section, ten items, in the order they were already written.',
     False),

    (376, 'reported in Section 3.1',
     '[Additional fix] This read "reported in section 4.1", which is Night-to-night '
     'reproducibility. The left-right imbalance is reported in Section 3.1. '
     'Corrected.',
     False),
]

SUPP = [
    (33, 'reported in Section 3.2',
     'Highlighted in V4 (supplementary V1) as "section 3.3."\n\n'
     'Wrong section: amplitude coupling, which this figure shows, is reported in '
     'Section 3.2 (Respiratory and cardiac components). Section 3.3 is rate '
     'estimation. Corrected.',
     True),

    (29, 'Figure S4.',
     fig_note('Figure S4',
              'Remove the background color and make the font size larger for clear view',
              'Done in supplementary V3. '
              'analysis/mean_value/ch_vs_clecre_sessions.py: stage-shading '
              'background removed, fonts 7-14.5 -> 11-17 pt, and the '
              'now-meaningless stage-colour legend patches dropped.')
     + '\n\n[Additional fix] This caption also called itself "the cohort-wide '
       'version of row B of Figure 3". Figure 3 is the SNR figure; the '
       'mean-referenced level trace is the middle row of Figure 2. Corrected.',
     True),

    (76, 'The imbalance index',
     '[Additional fix] Read "The asymmetry index". Renamed to match the single '
     'term now used for this quantity in the main text (Sections 3.1 and 4.5) and '
     'in this figure’s own title.',
     False),

    (78, '(Section 4.3)',
     '[Additional fix] This pointed at "§4.2.1", which does not exist in the '
     'main text. The outlying cardiac k is discussed in Section 4.3 (Calibrated '
     'Cardiorespiratory Performance). Corrected.',
     False),
]


def build(src, out, spec):
    print(f'{out.name}')
    d = Doc(src)
    c = Commenter(d)
    for para, anchor, body, hl in spec:
        c.add(para, anchor, body, highlight=hl)
    c.write()
    d.save(out)
    return out


if __name__ == '__main__':
    build(HERE / 'CAP_sleep_mask_manuscript_V6.docx',
          HERE / 'CAP_sleep_mask_manuscript_V6_responses.docx', MAIN)
    build(HERE / 'CAP_sleep_mask_manuscript supplementary V4.docx',
          HERE / 'CAP_sleep_mask_manuscript supplementary V4_responses.docx', SUPP)
