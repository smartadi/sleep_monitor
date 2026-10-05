"""
Supplementary V4 = V3 + a new section on variance and slow trends across sleep.

The section is written so it can move into the main text later: plain methods,
then the result, then the figure, for three analyses.

    S5.1  Where the high- and low-variance epochs fall (deck slide 73)
          analysis/mean_value/variance_low_high.py
    S5.2  CLE-CRE and CH with movement steps removed, and their slow-trend
          velocity            analysis/mean_value/destep_velocity.py (_supp figures)
    S5.3  The velocity at REM onset
          analysis/mean_value/rem_onset_velocity.py

Every number in the text is read from those scripts' CSVs at build time. The
figures are the supplement versions: white background, no head-movement shading.
New figures continue the numbering after S12 (S13-S17).

Builds on   writeup/review/final/CAP_sleep_mask_manuscript supplementary V3.docx
Writes      writeup/review/final/CAP_sleep_mask_manuscript supplementary V4.docx

Usage
    .venv/Scripts/python.exe writeup/ppt/_build_supp_v4.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _build_supp_v3 as B   # noqa: E402  (Doc and the paragraph/image helpers)

ROOT = HERE.parents[1]
SRC = B.FINAL / 'CAP_sleep_mask_manuscript supplementary V3.docx'
DST = B.FINAL / 'CAP_sleep_mask_manuscript supplementary V4.docx'
REP = ROOT / 'reports' / 'mean_value'
FIG_MV = ROOT / 'writeup' / 'figures' / 'mean_value'
FIG_IM = ROOT / 'writeup' / 'figures' / 'imbalance'


def _r(x, d=1):
    return f'{x:.{d}f}'


def variance_numbers():
    e = pd.read_csv(REP / 'variance_tails_enrichment.csv')
    e = e[e.subset == 'motion-free']

    def s(tail, ch, st):
        v = e[(e['tail'] == tail) & (e.channel == ch) & (e.stage == st)].enrichment
        return v.median(), int((v > 1).sum()), len(v)
    return {(t, c, st): s(t, c, st) for t in ('hi', 'lo') for c in ('CH', 'CLE', 'CRE')
            for st in ('Wake', 'N1', 'N2', 'N3', 'REM')}


def section():
    V = variance_numbers()
    vel = pd.read_csv(REP / 'destep_velocity.csv')
    rem = pd.read_csv(REP / 'rem_onset_velocity_subjects.csv')
    sens = pd.read_csv(REP / 'rem_onset_velocity_sensitivity.csv')
    ev = pd.read_csv(REP / 'rem_onset_velocity_events.csv')

    def n3lo(c):
        m, k, n = V[('lo', c, 'N3')]
        return f'{c} {_r(m)}-fold, {k} of {n} nights'

    def wkhi(c):
        m, k, n = V[('hi', c, 'Wake')]
        return f'{c} {_r(m)}-fold, {k} of {n} nights'

    hi_n3 = ', '.join(f"{c} {_r(V[('hi', c, 'N3')][0], 2)}" for c in ('CH', 'CLE', 'CRE'))
    rd = rem[rem.signal == 'CLE-CRE']
    rc = rem[rem.signal == 'CH']
    loose = sens[(sens.rem_free_gap_min == 5) & (sens.rem_hold_min == 2)
                 & (sens.signal == 'CLE-CRE')].iloc[0]
    n_ev = int(ev[ev.signal == 'CLE-CRE'].shape[0])
    still = rd.dropna(subset=['response_still_only'])

    return [
        ('h', 'S5. Variance and slow trends of the SEC signal across sleep'),
        ('t', 'This section collects three related observations on the slow behaviour '
              'of the SEC signal during sleep. They are reported descriptively, one '
              'value per night or per participant, and are intended for the main text '
              'once reviewed.'),

        # S5.1 ---------------------------------------------------------------
        ('t', 'Where the high- and low-variance periods fall. For every 30-second '
              'epoch we computed the variance of each SEC channel (CH, CLE and CRE) '
              'after removing content above 10 Hz. Within each night, the 10% of '
              'epochs with the highest variance and the 10% with the lowest were '
              'marked, so every night is judged on its own scale. Epochs with head '
              'movement (the top 10% of accelerometer activity in that night) were '
              'set aside, as were 14 epochs at the end of S4N1 where the signal had '
              'stopped. For each sleep stage we then asked how often it appeared among '
              'the marked epochs compared with how often it appeared in the night as a '
              'whole; a ratio above 1 means the stage is over-represented.'),
        ('t', 'The two ends of the variance range sat in opposite stages, and they did '
              'so on all three channels. The quietest epochs were over-represented in '
              f'N3 (median ratio {n3lo("CH")}; {n3lo("CLE")}; {n3lo("CRE")}). '
              'They were under-represented in wakefulness, N1 and REM. The most '
              'variable epochs showed the reverse: over-represented in wakefulness '
              f'({wkhi("CH")}; {wkhi("CLE")}; {wkhi("CRE")}) and almost never in '
              f'N3 (median ratio {hi_n3}). Because the pattern holds on the two '
              'single-sided channels as well as on CH, it is not a property of the CH '
              'electrode arrangement. Low SEC variance therefore marks deep sleep and '
              'high variance marks wakefulness, in line with the fall in band amplitude '
              'from wakefulness to N3 described in the main text (Figure S13).'),
        ('f', FIG_MV / 'variance_tails_enrichment.png'),
        ('c', 'Sleep stage of the highest-variance (top row) and lowest-variance '
              '(bottom row) epochs, for CH, CLE and CRE, with head-movement epochs '
              'removed. Each point is one night: how often the stage occurs among the '
              'marked epochs divided by how often it occurs in the whole night (1 = '
              'chance, dashed line). Bars are the median across nights; the fraction '
              'above each stage is the number of nights with a ratio above 1. Points at '
              'the bottom edge had no marked epochs in that stage.'),

        # S5.2 ---------------------------------------------------------------
        ('t', 'Slow trends in CLE−CRE and CH. A head movement shifts the SEC level '
              'abruptly and leaves it there, which hides the slower changes. We removed '
              'these shifts as follows. The signal was averaged in 10-second blocks; '
              'the accelerometer identified the blocks in which the head moved; the '
              'change in SEC level across each of those blocks was set to zero; and '
              'the signal was rebuilt from the remaining changes. Nothing between '
              'movements is altered. For display the result was smoothed with a '
              'running median over the preceding 5 minutes. To describe how fast the '
              'level was drifting, we fitted a straight line to the preceding 30 '
              'minutes of the movement-corrected signal at every point and took its '
              'slope, in fF per hour (the "trend velocity"). Blocks inside a head '
              'movement were left out of each fit. Because each value uses only the '
              'past 30 minutes, it reacts about 15 minutes after a change.'),
        ('t', 'With the movement shifts removed, both signals drift slowly over the '
              'night (Figures S14–S16). During still periods the typical size of the '
              f'trend velocity was {_r(vel.diff_median_abs_fF_per_h.min())}–'
              f'{_r(vel.diff_median_abs_fF_per_h.max())} fF/h for CLE−CRE and '
              f'{_r(vel.CH_median_abs_fF_per_h.min())}–'
              f'{_r(vel.CH_median_abs_fF_per_h.max())} fF/h for CH, largest in '
              'participant 6. The two trends moved in the same direction on '
              f'{int((vel.corr_vdiff_vch > 0).sum())} of the 12 nights (correlation '
              f'{_r(vel.corr_vdiff_vch.min(), 2)} to {_r(vel.corr_vdiff_vch.max(), 2)}, '
              f'median {_r(vel.corr_vdiff_vch.median(), 2)}). A few large shifts were '
              'not removed because the accelerometer did not register a movement at '
              'the time (S3N1 at about 2.2 h, S2N1 at about 4.1 h, S6N1 at about '
              '3.3 h); each appears as a deep, brief excursion of the velocity and '
              'should not be read as a slow trend.'),
        ('f', FIG_IM / 'fig_destep_velocity_S4N2_supp.png'),
        ('c', 'One night (S4N2) in full. From the top: scored sleep stage; head turn '
              'from the accelerometer; CLE−CRE before (grey) and after (dark grey) '
              'removal of the movement shifts, with its 5-minute running median; the '
              'CLE−CRE trend velocity; CH after removal of the movement shifts, with '
              'its running median; the CH trend velocity. Trend velocity is the slope '
              'of a line fitted to the preceding 30 minutes, in fF/h.'),
        ('f', FIG_IM / 'fig_destep_velocity_allsessions_CLE-CRE_supp.png'),
        ('c', 'CLE−CRE for all twelve nights. For each night: scored sleep stage, the '
              'movement-corrected CLE−CRE with its running median, and its trend '
              'velocity directly below.'),
        ('f', FIG_IM / 'fig_destep_velocity_allsessions_CH_supp.png'),
        ('c', 'CH for all twelve nights, laid out as Figure S15.'),

        # S5.3 ---------------------------------------------------------------
        ('t', 'The trend velocity at REM onset. Studies with invasive pressure '
              'monitoring and with near-infrared spectroscopy report that intracranial '
              'blood volume and pressure rise within minutes of entering REM sleep, '
              'which predicts a positive change in velocity at REM onset. We took '
              'each REM episode onset (the first REM epoch after at least 10 minutes '
              'without REM, with REM filling at least half of the next 5 minutes) and '
              'compared the average velocity over the 15 minutes after onset with the '
              'average over the 15 minutes before. Here the 30-minute line was '
              'centred on each moment rather than trailing it, so that the timing of '
              'a change is not delayed; as a result a change at onset begins to show '
              'about 15 minutes earlier. Events were averaged within each participant '
              'first. As a reference, the same comparison was made at 500 sets of '
              'random times in non-REM sleep in the same nights, at least 20 minutes '
              'from any REM onset.'),
        ('t', f'{n_ev} REM onsets in {rd.shape[0]} participants met the definition; '
              "participant 6's recordings contain none, because REM there is too "
              'fragmented. The CLE−CRE velocity rose after REM onset in '
              f'{int((rd.response_fF_h > 0).sum())} of {rd.shape[0]} participants '
              f'(median {_r(rd.response_fF_h.median())} fF/h), and in '
              f'{int((still.response_still_only > 0).sum())} of {still.shape[0]} when '
              'only onsets without a head movement in the 2 minutes around them were '
              'used. Only '
              f'{int((rd.percentile_in_null > 95).sum())} participant exceeded the '
              'upper 5% of its own random-time reference, and the result depended on '
              'the definition: when brief returns to REM within an episode were also '
              f'counted as onsets, {int(loose.n_rise)} of {int(loose.n_participants)} '
              'participants rose. CH rose in '
              f'{int((rc.response_fF_h > 0).sum())} of {rc.shape[0]}. The direction '
              'matches the expected rise at REM onset, but with this number of events '
              'the effect is weak and is reported as an observation, not a finding '
              '(Figure S17).'),
        ('f', FIG_IM / 'fig_rem_onset_velocity.png'),
        ('c', 'Trend velocity around REM-episode onset. (a, b) Average velocity of '
              'CLE−CRE and CH from 30 minutes before to 30 minutes after onset: thin '
              'lines are participants, the thick line and band their mean ± standard '
              'error, and the dashed line the same average at random non-REM times. '
              '(c, d) For each participant (number of onsets in brackets), the change '
              'in velocity from the 15 minutes before onset to the 15 minutes after '
              '(dot), against the 5–95% range of the random-time reference (grey bar).'),
    ]


def main():
    if not SRC.exists():
        raise SystemExit(f'missing: {SRC}')
    doc = B.Doc(SRC)
    paras = doc.paras
    head_tmpl = next(p for p in paras if doc.text(p).strip().startswith('S3. Integrated'))
    body_tmpl = next(p for p in paras if len(doc.text(p)) > 300)
    cap_tmpl = next(p for p in paras if doc.text(p).strip().startswith('Figure S1.'))
    img_tmpl = next(p for p in paras if len(list(p.iter('{%s}blip' % B.A))) == 1
                    and not doc.text(p).strip())
    n_before = max(int(m.group(1)) for p in paras
                   for m in [re.match(r'\s*Figure S(\d+)\.', doc.text(p))] if m)

    cursor = paras[-1]
    n_fig = n_before
    for kind, payload in section():
        if kind == 'f':
            if not Path(payload).exists():
                raise SystemExit(f'missing figure: {payload}')
            node = B.make_image_para(doc, img_tmpl, Path(payload), Path(payload).name,
                                     cx=B.page_width_emu(doc))
        elif kind == 'c':
            n_fig += 1
            node = B.clone_text(cap_tmpl, f'Figure S{n_fig}. {payload}')
        elif kind == 'h':
            node = B.clone_text(head_tmpl, payload)
        else:
            node = B.clone_text(body_tmpl, payload)
        cursor.addnext(node)
        cursor = node
    doc.save(DST)

    chk = B.Doc(DST)
    caps = [chk.text(p) for p in chk.paras if re.match(r'\s*Figure S\d+\.', chk.text(p))]
    nums = [int(re.match(r'\s*Figure S(\d+)\.', c).group(1)) for c in caps]
    assert nums == list(range(1, len(nums) + 1)), f'figure numbering broken: {nums}'
    bad = [tc for tc in chk.root.iter(B.q('tc')) if tc.find(B.q('p')) is None]
    assert not bad, 'empty table cell'
    print(f'built {DST.name}: figures S1-S{nums[-1]} '
          f'({nums[-1] - n_before} new, from S{n_before + 1})')
    for c in caps[n_before:]:
        print('   ', c[:100])


if __name__ == '__main__':
    main()
