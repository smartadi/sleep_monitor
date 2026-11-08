# Final text for the supplement (V8), keyed by paragraph index in word/document.xml.
# Same markup and entry forms as edits_text.py:
#   (guard, new_markup)            whole paragraph rewritten
#   (guard, None, [(old, new)])    substitutions on the paragraph's markup, each must occur once
# Decisions 2026-11-08 (writeup/edits/SUPP_V8_EDITS_2026-11-08.md, all recommendations):
# Q1 keep V8's Fig. S6 pages, Q2 delete Fig. S8 and renumber S9-S17 -> S8-S16,
# Q3 describe the statistics the paper uses now, Q4 "(continued)" on later pages, Q5 keep V8's images.

CONT = ' (continued)'

EDITS = {
# B1 title page (affiliations as in V18_aditya)
2: ("Changwoo Lee", None, [
    ('Sakthivelpathi}^**{b}', 'Sakthivelpathi}^**{c}'),
    ('Kwon}^**{c}', 'Kwon}^**{d}'),
    ('Choi}^**{d}', 'Choi}^**{e}')]),
4: ("b", None, [('^{b}  Nanosync', '^{c}  Nanosync')]),
5: ("c", None, [('^{c }Division', '^{d }Division')]),
6: ("d", None, [('^{d }Department', '^{e }Department')]),

# B4 Fig. S4
64: ("Fig. S4. Physiological-band SNR",
"**{Fig. S4 }Physiological-band SNR for each SEC channel (C_{H}, C_{LE}, and C_{RE}) across the 12 recording sessions. Signal power was defined as the mean spectral density within the physiological band (0.1–3 Hz), encompassing respiratory and cardiac activity. Noise power was the mean spectral density in the 10–50 Hz band of the same night, which an unworn-mask recording showed to be white. The SNR was positive for every channel in all sessions."),

# B5 Fig. S5
68: ("Fig. S5 Spectral slope",
"**{Fig. S5} Spectral slope per night in wakefulness and NREM sleep for (a) 0.1–0.3 Hz, (b) the vasomotor band (0.01–0.1 Hz), (c) the respiratory band (0.1–0.5 Hz), and (d) the cardiac band (0.5–3 Hz). Each dot is one night with both states. Grey lines join the wake and NREM values of the same night. Black lines join the group means. "),

# B7 Fig. S7
99: ("Fig. S7 SEC band-power",
"**{Fig. S7} SEC band-power changes surrounding delta-burst onset. Mean SEC band power, expressed as z-scores, is aligned to EEG delta-burst onset (green line, t = 0) for C_{LE}, C_{RE}, and C_{H}; the grey dashed line is the random-NREM control, and each panel has its own y-axis. Most onsets were arousal-associated: 236 of 339 (70%) had a cortical or autonomic arousal within ±10 s. "),

# B10 Fig. S10 -> S9
134: ("Fig. S10 All harmonic-comb",
"**{Fig. S9 }All multi-band events on the twelve nights. Each row is one night, colored by scored sleep stage; black boxes mark the events, and the number of events per night is given next to its label."),

# B12 Fig. S11 -> S10
139: ("Fig. S11 Sleep-stage", None, [
    ('**{Fig. S11}', '**{Fig. S10}'),
    ('45 harmonic-comb events', '45 multi-band events'),
    ('(A) Stage', '(a) Stage'),
    ('(B) REM', '(b) REM')]),

# B13 Fig. S12 -> S11
142: ("Fig. S12 Four comb", None, [
    ('**{Fig. S12 }Four comb events', '**{Fig. S11 }Four multi-band events')]),

# B14 Fig. S13 -> S12
144: ("Fig. S13 Sleep stage", None, [('**{Fig. S13 }', '**{Fig. S12 }')]),

# B15 Fig. S14 -> S13 (professor's figure: caption only)
147: ("Fig. S14 Associations",
"**{Fig. S13} Associations of SEC-derived features with PSG-scored arousals, sleep quality (PSQI), LFO ridge duration, and age. (a) SEC impulse frequency against the respiratory arousal index, (b) against the spontaneous arousal index, and (c) against PSQI. (d) LFO ridge duration (% of night) against raw (C_{RE})_{rms}. (e) Two-night mean LFO ridge duration against PSQI. (f) Destepped (C_{LE} − C_{RE})_{rms} against age. Each point is one night, except in (e), where it is one participant; dotted lines are linear fits."),

# B16 Fig. S15 -> S14
151: ("Fig. S15 Band-specific", None, [('**{Fig. S15}', '**{Fig. S14}')]),

# B17 Fig. S16 -> S15
168: ("Fig. S16 (a) Spectrograms", None, [
    ('**{Fig. S16}', '**{Fig. S15}'),
    ('0.01~0.1 Hz', '0.01–0.1 Hz')]),

# B18 Fig. S17 -> S16
170: ("Fig. S17 SEC slope", None, [
    ('**{Fig. S17 }', '**{Fig. S16 }'),
    ('Average slope of (C_{LE}−C_{RE}) slope', 'Average C_{LE} − C_{RE} slope'),
    ('for C_{H.}', 'for C_{H}.'),
    ('Change in (C_{LE}−C_{RE}) slope', 'Change in C_{LE} − C_{RE} slope')]),

# B19 Statistical analysis (Q3a; figure numbers already renumbered: associations = S13, ridge features = S14)
174: ("The recording was the primary unit",
"With six participants and 12 nights, all analyses were exploratory. Wake–NREM differences in aperiodic slope and band excess were tested within recordings with paired t-tests (Section 2.4), and p-values were not corrected for multiple comparisons. Fig. 3 and Fig. S13 report correlation coefficients (r) over the plotted points with uncorrected p-values. Associations between SEC features and participant measures in Section 3.9 and Fig. 11 are Spearman’s rank correlations (ρ) across the six participants, using each participant’s two-night mean, without p-values. Night-to-night reproducibility (Fig. 10) is reported as ρ between nights and the two-way intraclass correlation for absolute agreement, ICC(2,1). Sleep-stage effects are summarized as the number of nights or participants showing each direction; tests on pooled epochs (Fig. S14) are descriptive only, because epochs within a recording are not independent. Event-triggered responses were compared with random time points drawn from the same recordings and sleep stages."),
}

# B2 Fig. S2: later pages "(continued)"
for i in (17, 21, 24, 29, 32):
    EDITS[i] = ("Fig. S2 ", None, [('Fig. S2 ', 'Fig. S2' + CONT + ' ')])

# B3 Fig. S3: each caption names the night drawn above it (checked against the title in each image)
S3 = [(35, 'S1N1 and S1N2', 'S1N1'), (37, 'S2N1 and S2N2', 'S1N2'), (39, 'S3N1 and S3N2', 'S2N1'),
      (41, 'S4N1 and S4N2', 'S2N2'), (44, 'S5N1 and S5N2', 'S3N1'), (46, 'S6N1 and S6N2', 'S3N2'),
      (50, 'S5N1 and S5N2', 'S4N1'), (52, 'S6N1 and S6N2', 'S4N2'), (54, 'S5N1 and S5N2', 'S5N1'),
      (56, 'S6N1 and S6N2', 'S5N2'), (59, 'S5N1 and S5N2', 'S6N1'), (61, 'S6N1 and S6N2', 'S6N2')]
for k, (i, old, new) in enumerate(S3):
    EDITS[i] = ("Fig. S3 Destepped", None, [
        ('Fig. S3 Destepped SEC data for ' + old + ',',
         'Fig. S3' + (CONT if k else '') + ' Destepped SEC data for ' + new + ',')])

# B6 Fig. S6: C_RE as a subscript, later pages "(continued)"
for k, i in enumerate((70, 72, 74, 76, 79, 81, 85, 87, 89, 91, 94, 96)):
    subs = [('on CRE,', 'on C_{RE},')]
    if k: subs.append(('Fig. S6 ', 'Fig. S6' + CONT + ' '))
    EDITS[i] = ("Fig. S6 Tracked", None, subs)

# B9 Fig. S9 -> S8: "multi-band", later pages "(continued)", double space removed where present
DOUBLE_SPACE = {106, 108, 110, 112, 115}
for k, i in enumerate((106, 108, 110, 112, 115, 117, 120, 122, 124, 126, 129, 131)):
    subs = [('Fig. S9 Harmonic-comb events', 'Fig. S8' + (CONT if k else '') + ' Multi-band events')]
    if i in DOUBLE_SPACE: subs.append((',  and C_{LE}', ', and C_{LE}'))
    EDITS[i] = ("Fig. S9 Harmonic-comb", None, subs)

# Author line: track only the three superscript letters, not the names between them
NO_ABSORB = {2}

# B8 Fig. S8 (page break, image, caption, empty line) and B11 the stale paragraph under Fig. S10
DELETE_PARAS = [100, 101, 102, 103, 135]

# B1 new affiliation line after "a Mechanical Engineering ..." (P3), formatted like it
NEW_PARAS = [
    (3, 3, "^{b}BioEngineering Department, University of Washington, Seattle, WA"),
]

FIGURES = {}
