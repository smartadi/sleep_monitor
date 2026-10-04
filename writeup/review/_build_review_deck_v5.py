"""
Review deck V5: the reviewer's jedit deck with four sections repaired.

Built from  writeup/review/CAP_sleep_mask_review_deck jedit(1).pptx
Writes      writeup/review/CAP_sleep_mask_review_deck V5.pptx

What changes, against the jedit deck's slide numbers
----------------------------------------------------
 9-37  Rate. The 24 per-night rate plots, per-stage MAE and estimator heatmap
       are replaced by the six figures of the simplified rate section in
       supplementary V3 (S7-S12), taken out of that .docx so the slides show
       exactly what the supplement shows.
39-50  Ridges. Each night's figure is redrawn with the smoothed low band and
       its gated ridge (analysis/slow_wave/ridge_lowband_smooth.py). Two slides
       follow: why the old low band changed, and all twelve on one sheet.
69-86  Sensor value. All of it -- the per-night channel_evolution figures and
       the reviewer's cropped/annotated copies of them -- is replaced by the
       de-stepped CLE-CRE and CH with their velocity
       (analysis/mean_value/destep_velocity.py), one slide per night, then the
       two all-session sheets.
92-93  Variance. The CH-only high-variance trace and enrichment are replaced by
       both tails (top and bottom decile) on CH, CLE and CRE
       (analysis/mean_value/variance_low_high.py).

Everything else, including the reviewer's groups on slides 88-89, is kept
as is. Each new slide carries its draft text in the speaker notes; nothing is
written into the manuscript or supplement.

Usage
    .venv/Scripts/python.exe "writeup/review/_build_review_deck_v5.py"
"""

from __future__ import annotations

import copy
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.util import Emu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / 'CAP_sleep_mask_review_deck jedit(1).pptx'
DST = HERE / 'CAP_sleep_mask_review_deck V5.pptx'
SUPP = HERE / 'final' / 'CAP_sleep_mask_manuscript supplementary V3.docx'
FIG = ROOT / 'writeup' / 'figures'
REP = ROOT / 'reports'

SESSIONS = [f'S{s}N{n}' for s in range(1, 7) for n in (1, 2)]
AGE_SEX = {'S1': '61 F', 'S2': '66 M', 'S3': '37 M', 'S4': '54 M', 'S5': '55 F',
           'S6': '25 M'}                    # Table 1 of the manuscript

# picture area of a content slide, from the template
BOX_L, BOX_T, BOX_W, BOX_H = 411480, 1225296, 11368735, 5321808

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


# ── supplementary V3: rate figures and their text ────────────────────────────

def supp_rate():
    """{'S7': (png bytes, caption, [body paragraphs]), ...} from the .docx."""
    z = zipfile.ZipFile(SUPP)
    root = etree.fromstring(z.read('word/document.xml'))
    rels = {r.get('Id'): r.get('Target')
            for r in etree.fromstring(z.read('word/_rels/document.xml.rels'))}
    paras = []
    for p in root.iter('{%s}p' % W):
        blip = next(p.iter('{%s}blip' % A), None)
        text = ''.join(t.text or '' for t in p.iter('{%s}t' % W)).strip()
        paras.append((rels[blip.get('{%s}embed' % R)] if blip is not None else None, text))

    out, img = {}, None
    for target, text in paras:
        if target:
            img = z.read('word/' + target)
            continue
        m = re.match(r'Figure (S\d+)\.\s*(.*)', text)
        if m and img is not None:
            out[m.group(1)] = [img, m.group(2), []]
            img = None
    # body paragraphs, attached by their opening words
    attach = {
        'S7': ('This section replaces', 'A rate is produced'),
        'S8': ('k is the ratio', 'The spectral detector', 'A caveat applies'),
        'S9': ('Before the division', 'The cardiac panel'),
        'S10': ('Applying each recording',),
        'S11': ('The question to ask', 'Night-average breathing'),
        'S12': ('Finally, how much', 'No capacitive feature', 'Limitations.'),
    }
    for fig, starts in attach.items():
        for st in starts:
            hit = [t for _, t in paras if t.startswith(st)]
            assert len(hit) == 1, f'{fig}: {st!r} found {len(hit)} times'
            out[fig][2].append(hit[0])
    return out


RATE_SLIDES = [
    ('S7', 'The rate pipeline, one minute',
     'Band-pass, count peaks per epoch, divide by one k per recording.'),
    ('S8', 'k for every night and channel',
     'Peak counting gives about one peak per breath and two per heartbeat.'),
    ('S9', 'What is counted, before k',
     'The raw count against the reference; the gap between them is k.'),
    ('S10', 'All twelve recordings',
     'Each SEC channel converted by its own k, against the PSG reference.'),
    ('S11', 'Against a no-sensor baseline',
     'Night-average breathing beats the baseline; night-average heart rate does not.'),
    ('S12', 'How far k moves within a night',
     'Fairly steady within a recording; it moves much more between them.'),
]


# ── draft text for the new material ──────────────────────────────────────────

def lowband_text():
    t = pd.read_csv(REP / 'slow_wave' / 'low_band' / 'ridge_lowband_smooth.csv')
    pct = t.pct_night
    f = t.f_median_hz.dropna()
    sheet = (
        'DRAFT for §3.3, replacing the "Recurrent Low-Frequency Ridges" paragraph. '
        'The low-frequency band was re-examined on a smoothed spectrogram built from '
        'each channel directly: decimated to 2 Hz, band-passed at 0.005–0.5 Hz, and '
        'estimated every 30 s by Welch’s method over a 10-min window (4-min '
        'segments, about seven averages), with the 1/f trend removed per column. A '
        'single ridge was tracked through 0.02–0.20 Hz as a Viterbi path and was '
        'counted only where it stood above a per-night threshold for at least 15 min. '
        'The threshold was set so that the same night’s spectrogram, shuffled in '
        '10-min blocks, showed no more than 5% of the night as ridge. On CRE the '
        f'gated ridge covered {pct.min():.0f}–{pct.max():.0f}% of each night '
        f'(median {pct.median():.0f}%), against the 5% allowed in the shuffled maps, '
        f'and its frequency was not shared across nights (night medians '
        f'{f.min():.3f}–{f.max():.3f} Hz, several at the 0.02 Hz edge of the search '
        'band). We therefore do not find a persistent low-frequency oscillation at a '
        'common frequency. The energy near 0.05–0.08 Hz in the earlier figure came '
        'from the 0.05-Hz corner of the motion-canceller filter (next slide).\n\n'
        'NOTE: the "67 ridges, median 6 per night" in V11 §3.3 came from a co-author '
        'script we do not hold; this does not reproduce it. The ridge-by-stage slide '
        'that follows still uses the old detector.')
    check = (
        'DRAFT for §2.7 (methods). The low-frequency band was analysed on the '
        'channel itself rather than on the motion-cancelled signal used for rate '
        'estimation, for two measured reasons (S1N1 CRE shown). First, that signal '
        'is band-passed at 0.05 Hz; after the 1/f trend is removed, the filter '
        'corner appears as a constant band just above 0.05 Hz in every column of '
        'every night (B). Second, below 0.5 Hz the accelerometer magnitude is '
        'dominated by head orientation and carries an instrumental line at '
        '0.1447 Hz, with its harmonic at 0.289 Hz, on all twelve nights. Regressing '
        'it out raised the channel’s low-band power by about 6 dB and copied that '
        'line into it (A). Built from the band-passed channel alone (C), the band '
        'shows motion as broad vertical stripes and no constant feature.\n\n'
        'FLAG: the same regression runs ahead of the rate band (0.05–4 Hz). Whether '
        'it injects accelerometer content there too has not been checked.')
    per = {}
    for _, r in t.iterrows():
        per[r.session] = (
            f"{r.session}: gated ridge on {r.pct_night:.0f}% of the night, "
            f"{int(r.n_episodes)} episode(s), longest {r.longest_min:.0f} min, "
            + (f"median {r.f_median_hz:.3f} Hz" if pd.notna(r.f_median_hz) else 'no ridge')
            + f"; threshold {r.gate_db:.1f} dB from the block-shuffled map. Top and "
            "middle panels are unchanged from the previous figure.")
    return per, check, sheet


def destep_text():
    v = pd.read_csv(REP / 'mean_value' / 'destep_velocity.csv').set_index('session')
    sheet = (
        'DRAFT for §3.1 / Fig. 2. Head movements move the slow SEC level in steps, '
        'within one 10-s block, and leave it there. Those steps were removed by '
        'zeroing the first difference of the block-averaged, mean-referenced signal '
        'wherever the gravity vector changed between blocks by more than a fixed threshold '
        '(padded by 30 s either side, since the capacitance settles after the '
        'accelerometer does) and integrating back; no step size is estimated, and '
        'the level between movements is untouched. The result was smoothed with a '
        '5-min causal (trailing) median, so the trace cannot anticipate an event and '
        'lags it by about half the window. Velocity is the 2-min backward difference '
        f'of that trace. Across the twelve nights {v.pct_moving.min():.0f}–'
        f'{v.pct_moving.max():.0f}% of blocks fell inside a head movement. During '
        'still periods the median absolute velocity of CLE−CRE was '
        f'{v.diff_median_abs_fF_per_min.min():.2f}–{v.diff_median_abs_fF_per_min.max():.2f} '
        'fF/min and that of CH '
        f'{v.CH_median_abs_fF_per_min.min():.2f}–{v.CH_median_abs_fF_per_min.max():.2f} '
        'fF/min. The correlation between the two velocities was positive on every '
        'night but ranged from negligible to strong '
        f'(r = {v.corr_vdiff_vch.min():.2f}–{v.corr_vdiff_vch.max():.2f}, '
        f'median {v.corr_vdiff_vch.median():.2f}).')
    per = {}
    for s, r in v.iterrows():
        per[s] = (f'{s} ({AGE_SEX[s[:2]]}): {r.pct_moving:.0f}% of blocks inside a head '
                  f'movement (shaded). Still-period |velocity|, median: CLE−CRE '
                  f'{r.diff_median_abs_fF_per_min:.2f}, CH {r.CH_median_abs_fF_per_min:.2f} '
                  f'fF/min; r(velocity CLE−CRE, velocity CH) = {r.corr_vdiff_vch:+.2f}.')
    return per, sheet


def variance_text():
    e = pd.read_csv(REP / 'mean_value' / 'variance_tails_enrichment.csv')
    e = e[e.subset == 'motion-free']

    def s(tail, ch, st):
        v = e[(e['tail'] == tail) & (e.channel == ch) & (e.stage == st)].enrichment
        return v.median(), int((v > 1).sum()), len(v)

    lo = {ch: s('lo', ch, 'N3') for ch in ('CH', 'CLE', 'CRE')}
    hiw = {ch: s('hi', ch, 'Wake') for ch in ('CH', 'CLE', 'CRE')}
    hin3 = {ch: s('hi', ch, 'N3') for ch in ('CH', 'CLE', 'CRE')}
    enrich = (
        'DRAFT (new paragraph, after the variance-by-stage result). Within each '
        'recording, the epoch variance of the <10-Hz SEC signal was split at the '
        '10th and 90th percentiles, and the stage occupancy of each tail was '
        'compared with the night’s overall occupancy (observed/expected), after '
        'removing motion epochs (top decile of accelerometer SD) and 14 dead-channel '
        'epochs of S4N1, mostly in its final minutes. The highest-variance epochs were over-represented '
        'in wakefulness on every channel (median '
        + ', '.join(f'{c} {hiw[c][0]:.1f}× [{hiw[c][1]}/{hiw[c][2]}]' for c in hiw)
        + ' nights) and almost absent from N3 (median '
        + ', '.join(f'{c} {hin3[c][0]:.2f}' for c in hin3)
        + '; 2 of 12 nights enriched on each). The lowest-variance epochs showed the '
        'reverse: N3 was over-represented (median '
        + ', '.join(f'{c} {lo[c][0]:.1f}× [{lo[c][1]}/{lo[c][2]}]' for c in lo)
        + ' nights), and Wake, N1 and REM were under-represented. The relation held '
        'on the two single-ended channels as well as on CH, so it is not a property '
        'of the CH configuration. Low SEC variance tracked deep sleep, and high '
        'variance tracked wakefulness (and REM on CH and CRE; on CLE, REM was at '
        'chance), consistent with the band-amplitude profile. Descriptive, one value per night; no pooled test.')
    traces = {
        ch: (f'{ch} variance per 30-s epoch (<10 Hz), log scale, hypnogram above. '
             'Red: top decile of the recording; green: bottom decile; grey: either '
             'tail during movement. Which stages each tail falls in is counted on '
             'the last slide of this section.')
        for ch in ('CH', 'CLE', 'CRE')}
    return traces, enrich


# ── slide machinery ──────────────────────────────────────────────────────────

def set_text(shape, text):
    p = shape.text_frame.paragraphs[0]
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def clone(prs, tmpl, keep=('TextBox 1', 'TextBox 2', 'TextBox 4', 'TextBox 5',
                           'Rectangle 1')):
    """A new slide carrying the template's text boxes (not its picture)."""
    s = prs.slides.add_slide(tmpl.slide_layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    for shp in tmpl.shapes:
        if shp.name in keep:
            s.shapes._spTree.append(copy.deepcopy(shp._element))
    return s


def shape(s, name):
    return next(x for x in s.shapes if x.name == name)


def add_picture(s, img):
    """Fit the image into the content area, centred, aspect preserved."""
    data = img if isinstance(img, bytes) else Path(img).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        iw, ih = im.size
    scale = min(BOX_W / iw, BOX_H / ih)
    w, h = int(iw * scale), int(ih * scale)
    pic = s.shapes.add_picture(io.BytesIO(data), Emu(BOX_L + (BOX_W - w) // 2),
                               Emu(BOX_T + (BOX_H - h) // 2), Emu(w), Emu(h))
    return pic


def content(prs, tmpl, title, sub, img, path, notes):
    s = clone(prs, tmpl)
    set_text(shape(s, 'TextBox 1'), title)
    set_text(shape(s, 'TextBox 2'), sub)
    set_text(shape(s, 'TextBox 4'), path)
    add_picture(s, img)
    s.notes_slide.notes_text_frame.text = notes
    return s


def section(prs, tmpl, title):
    s = clone(prs, tmpl)
    set_text(shape(s, 'TextBox 2'), title)
    return s


def main():
    prs = Presentation(SRC)
    old = list(prs.slides)
    assert len(old) == 93, f'expected the 93-slide jedit deck, got {len(old)}'
    T_CONTENT, T_SECTION = old[38], old[37]           # slides 39 and 38

    def J(n):                                          # jedit slide n, 1-based
        return old[n - 1]

    # rate
    rate = supp_rate()
    rate_new = [section(prs, T_SECTION, 'Respiratory and cardiac rate')]
    for fig, title, sub in RATE_SLIDES:
        img, cap, body = rate[fig]
        rate_new.append(content(
            prs, T_CONTENT, title, sub, img,
            f'supplementary V3, Figure {fig}',
            '\n\n'.join(body) + f'\n\nFigure {fig}. {cap}'))

    # ridges
    per, check_note, sheet_note = lowband_text()
    ridge_new = []
    for s in SESSIONS:
        ridge_new.append(content(
            prs, T_CONTENT, f'Ridges — {s}',
            'Respiratory and cardiac traces on CRE; smoothed low band with its gated ridge.',
            FIG / 'harmonics' / 'ridges_lowband' / f'ridge_lowband_{s}_CRE.png',
            f'harmonics/ridges_lowband/ridge_lowband_{s}_CRE.png', per[s]))
    ridge_new.append(content(
        prs, T_CONTENT, 'Why the low band changed',
        'The old panel showed the canceller’s 0.05 Hz corner and the accelerometer’s line.',
        FIG / 'harmonics' / 'ridges_lowband' / 'lowband_filter_check_S1N1_CRE.png',
        'harmonics/ridges_lowband/lowband_filter_check_S1N1_CRE.png', check_note))
    ridge_new.append(content(
        prs, T_CONTENT, 'Low-band ridge — all twelve',
        'Gated against a block-shuffled null: little beyond chance, no common frequency.',
        FIG / 'harmonics' / 'ridges_lowband' / 'ridge_lowband_allsessions_CRE.png',
        'harmonics/ridges_lowband/ridge_lowband_allsessions_CRE.png', sheet_note))

    # sensor value -> de-stepped differential + velocity
    dper, dsheet = destep_text()
    mean_new = [section(prs, T_SECTION,
                        'CLE−CRE and CH, motion steps removed — every night')]
    for s in SESSIONS:
        mean_new.append(content(
            prs, T_CONTENT, f'CLE−CRE and CH — {s}  ({AGE_SEX[s[:2]]})',
            'Head turn, the de-stepped differential with CH, and their velocity.',
            FIG / 'imbalance' / f'fig_destep_velocity_{s}.png',
            f'imbalance/fig_destep_velocity_{s}.png', dper[s]))
    mean_new.append(content(
        prs, T_CONTENT, 'De-stepped CLE−CRE and CH — all twelve',
        'Jumps during head movement removed; 5-min causal median.',
        FIG / 'imbalance' / 'fig_destep_allsessions.png',
        'imbalance/fig_destep_allsessions.png', dsheet))
    mean_new.append(content(
        prs, T_CONTENT, 'Velocity — all twelve',
        'Rate of change of the smoothed CLE−CRE and CH, fF/min.',
        FIG / 'imbalance' / 'fig_destep_velocity_allsessions.png',
        'imbalance/fig_destep_velocity_allsessions.png', dsheet))

    # variance
    vtr, vnote = variance_text()
    var_new = []
    for ch in ('CH', 'CLE', 'CRE'):
        var_new.append(content(
            prs, T_CONTENT, f'Both tails of the variance — {ch}',
            'Top decile in red, bottom decile in green, with the hypnogram above.',
            FIG / 'mean_value' / f'variance_tails_traces_{ch}.png',
            f'mean_value/variance_tails_traces_{ch}.png', vtr[ch]))
    var_new.append(content(
        prs, T_CONTENT, 'Which stages the two tails fall in',
        'High variance avoids N3; low variance is enriched for N3, on all three channels.',
        FIG / 'mean_value' / 'variance_tails_enrichment.png',
        'mean_value/variance_tails_enrichment.png', vnote))

    # ── order ────────────────────────────────────────────────────────────────
    order = ([J(n) for n in range(1, 9)] + rate_new
             + [J(38)] + ridge_new + [J(n) for n in range(51, 69)]
             + mean_new + [J(n) for n in range(87, 92)] + var_new)
    keep_ids = {id(s) for s in order}

    lst = prs.slides._sldIdLst
    by_slide = {}
    for sid in list(lst):
        sl = prs.slides.get(int(sid.get('id')))
        by_slide[id(sl)] = sid
    for sid in list(lst):
        lst.remove(sid)
    for sl in old + rate_new + ridge_new + mean_new + var_new:
        if id(sl) not in keep_ids:
            prs.part.drop_rel(by_slide[id(sl)].get(
                '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'))
    for sl in order:
        lst.append(by_slide[id(sl)])

    # figure counter, bottom right, in document order
    n = 0
    for sl in order:
        if any(x.shape_type == 13 for x in sl.shapes):
            box = next((x for x in sl.shapes if x.name == 'TextBox 5'), None)
            if box is not None:
                n += 1
                set_text(box, str(n))

    prs.save(DST)
    print(f'wrote {DST.name}: {len(order)} slides ({n} figures); '
          f'rate {len(rate_new)}, ridges {len(ridge_new)}, '
          f'sensor value {len(mean_new)}, variance {len(var_new)} new')


if __name__ == '__main__':
    main()
