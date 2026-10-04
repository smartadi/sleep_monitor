"""Manuscript items made outside this code, listed so the numbers report is complete.

Nothing here computes anything. Each item is registered as EXTERNAL with where it
came from, so the gap is visible next to everything the code does reproduce.

    Fig. 1        sensor and mask photographs / schematic
    Fig. 3, 4     variance-preserving spectra by stage and aperiodic spectral slope
                  (co-author analysis; no code or fit in this repository)
    ¶180, ¶184    the spectral statements and slope p-values those figures carry
    Fig. 13       optical PPG vs SEC comparison (separate optical recording;
                  writeup/review/Comparison Optic and Cap(1).pptx)
    Fig. S1       24-hour bench drift / temperature test (separate experiment)
    111 Hz        acquisition rate and 111 -> 100 Hz resampling (done before the
                  merged recordings this code reads, which are already at 100 Hz)
"""

from seclib.numbers import Numbers

STAGE = 's99_external'
REQUIRES = []

ITEMS = [
    ('fig1', 'Fig. 1', 'sensor principle and mask photographs', 'figure, no data'),
    ('fig3', '§3.2 Fig. 3', 'normalized f×PSD spectra by stage, CLE and CRE',
     'co-author figure; no code in repo'),
    ('spec_nrem', '§3.2 ¶180', 'NREM > Wake at 0.04–0.1 Hz and ~0.2 Hz, peaks 0.17/0.22 Hz',
     'co-author analysis'),
    ('fig4', '§3.2 Fig. 4', 'aperiodic spectral slope per night, four bands',
     'co-author figure; caption still says CHECK THE Y AXIS'),
    ('slope_p', '§3.2 ¶184', 'slope Wake→NREM p = 0.008 (CLE), 0.033 (CLE−CRE)',
     'co-author statistics; no fit in repo'),
    ('fig13', 'Discussion Fig. 13', 'optical PPG vs SEC spectrograms',
     'separate optical recording; Comparison Optic and Cap(1).pptx'),
    ('figS1', 'Supp Fig. S1', '24-h drift, 0.25 fF/°C, 3 fF over 3 °C',
     'bench test; text says 25-hour, caption 24-hour'),
    ('fs111', '§2.2 ¶140', 'acquisition at 111 Hz, resampled to 100 Hz',
     'upstream of the merged files, which are already 100 Hz'),
]


def run():
    nb = Numbers(STAGE)
    for id_, sec, what, note in ITEMS:
        nb.add(id_, sec, what, 'external', status='EXTERNAL', note=note)
    nb.save()


if __name__ == '__main__':
    run()
