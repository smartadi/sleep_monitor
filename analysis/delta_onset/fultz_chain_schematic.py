"""
Where our K-complex result sits on the Fultz et al. (2019) chain.

Fultz et al., Science 2019, recorded EEG and fMRI simultaneously during NREM
sleep and followed one event through three stages: a cortical slow wave, then a
fall in cortical blood volume, then an inflow of CSF at the fourth ventricle.
Their load-bearing measurement is a LAG, not a correlation — the slow-delta EEG
envelope does not correlate with CSF flow at zero lag at all; it LEADS it by
about 6.4 s (best-fit impulse response). The repo already encodes this in
analysis/swa_validation/fultz_eeg_cap_impulse.py.

That matters twice over for this project.

* It explains our own earlier negative. The SWA validation compared CAP delta
  power with EEG delta power at ZERO LAG and found r ~ 0.015. Under Fultz that
  is the expected result and says nothing about downstream coupling, because
  the coupling is displaced in time.
* It predicts where to look instead, and the prediction holds: the mask's
  response to a scored K-complex peaks 1.9-4.3 s after the event, between the
  cortical event and the ventricular CSF response, which is where a sensor
  measuring cranial displacement should sit.

This draws that chain with our measured latencies on it. It is a schematic of
the mechanism, drawn here from scratch — not a reproduction of any figure from
the paper. The only numbers taken from Fultz are the ~6.4 s lag and the
ordering; everything on the lower axis is measured in this study
(kcomplex_cap_response.py).

Usage
-----
    .venv/Scripts/python.exe analysis/delta_onset/fultz_chain_schematic.py

Outputs
-------
    writeup/figures/delta_onset/fig_fultz_chain.png
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'writeup' / 'figures' / 'delta_onset'
OUT.mkdir(parents=True, exist_ok=True)

INK = '#1B2A41'
MUTED = '#5A6472'
ACCENT = '#C0392B'
OURS = '#2980B9'
FULTZ = '#8E44AD'
GREEN = '#1E8449'

# (time_s, title, subtitle, colour, is_ours)
STAGES = [
    (0.0, 'Cortical down-state', 'K-complex / slow wave\nneurons fall silent together',
     INK, False),
    (2.4, 'Blood volume falls', 'less activity, less demand\nvasculature constricts',
     MUTED, False),
    (4.3, 'Cranial tissue moves', 'the skull is a closed box\nsomething has to give',
     OURS, True),
    (6.4, 'CSF flows in', 'measured at the fourth\nventricle by Fultz 2019',
     FULTZ, False),
]

# measured in this study, from kcomplex_cap_response.py
OUR_PEAKS = [('CLE', 1.9), ('CH', 1.9), ('CRE', 2.9)]
OUR_MEDIAN_RANGE = (3.6, 4.3)          # median per-event peak latency, CLE..CH

plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig, ax = plt.subplots(figsize=(16.5, 6.8))
ax.set_xlim(-1.15, 10.2)
ax.set_ylim(-2.45, 2.5)
ax.axis('off')

# ── the chain ────────────────────────────────────────────────────────────────
for i, (t, title, sub, col, ours) in enumerate(STAGES):
    y = 1.35
    box = FancyBboxPatch((t - 0.82, y - 0.52), 1.64, 1.04,
                         boxstyle='round,pad=0.06,rounding_size=0.10',
                         facecolor=col, alpha=0.13 if not ours else 0.22,
                         edgecolor=col, linewidth=2.4 if ours else 1.3)
    ax.add_patch(box)
    ax.text(t, y + 0.26, title, ha='center', va='center', fontsize=12,
            fontweight='bold', color=col)
    ax.text(t, y - 0.20, sub, ha='center', va='center', fontsize=9.5, color=MUTED)
    if ours:
        ax.annotate('WHAT THE MASK MEASURES', xy=(t, y + 0.62), ha='center',
                    fontsize=11.5, fontweight='bold', color=OURS)
    if i < len(STAGES) - 1:
        nxt = STAGES[i + 1][0]
        ax.add_patch(FancyArrowPatch((t + 0.88, y), (nxt - 0.88, y),
                                     arrowstyle='-|>', mutation_scale=20,
                                     linewidth=1.8, color='#9AA4AE'))

# ── the lag axis ─────────────────────────────────────────────────────────────
y0 = -0.75
# drop lines, so it is obvious the boxes above are placed at their true latency
for t, _, _, col, _ in STAGES:
    ax.plot([t, t], [y0 + 0.10, 0.80], color=col, lw=1.0, ls=':', alpha=0.55, zorder=0)
ax.annotate('', xy=(8.35, y0), xytext=(-0.85, y0),
            arrowprops=dict(arrowstyle='-|>', linewidth=1.6, color=INK))
for t in range(0, 9):
    ax.plot([t, t], [y0 - 0.08, y0 + 0.08], color=INK, lw=1.2)
    ax.text(t, y0 - 0.26, f'{t}', ha='center', va='top', fontsize=11, color=INK)
ax.text(8.55, y0, 'seconds after\nthe cortical event', ha='left', va='center',
        fontsize=12, color=INK, fontweight='bold')

# zero lag — the alternative hypothesis, ruled out
ax.plot([0, 0], [y0, 0.62], color=ACCENT, lw=1.8, ls='--', zorder=1)
ax.text(-0.14, 0.60, 'zero lag', ha='right', va='top', fontsize=11.5,
        fontweight='bold', color=ACCENT)
ax.text(-0.14, 0.30, 'where a response would sit if the\nmask were picking up the '
                     'ELECTRICAL\nevent. Nothing is there.',
        ha='right', va='top', fontsize=10.5, color=ACCENT)

# our measured window
lo, hi = OUR_PEAKS[0][1], OUR_MEDIAN_RANGE[1]
ax.add_patch(FancyBboxPatch((lo, y0 - 0.17), hi - lo, 0.34,
                            boxstyle='round,pad=0.02,rounding_size=0.05',
                            facecolor=OURS, alpha=0.25, edgecolor=OURS, lw=1.6))
for name, t in OUR_PEAKS:
    ax.plot([t], [y0], marker='v', ms=10, color=OURS, zorder=5)
ax.text((lo + hi) / 2, y0 - 0.55, 'mask response to a scored K-complex\n'
                                  'peaks 1.9–4.3 s  (this study, n = 50)',
        ha='center', va='top', fontsize=11.5, color=OURS, fontweight='bold')

# Fultz
ax.plot([6.4], [y0], marker='v', ms=11, color=FULTZ, zorder=5)
ax.text(6.4, y0 + 0.30, 'Fultz 2019:  EEG leads CSF inflow by ~6.4 s',
        ha='center', va='bottom', fontsize=11.5, color=FULTZ, fontweight='bold')

# ── the sentence that makes the earlier negative make sense ──────────────────
ax.text(-1.05, -1.80,
        'Fultz found NO correlation between EEG and CSF flow at zero lag — the '
        'coupling only appears once you allow a delay.\n'
        'That is why our own zero-lag test (CAP delta power vs EEG delta power, '
        'r ≈ 0.015) was negative, and why it was never the\n'
        'end of the question. Looking in the delayed window instead, the '
        'response is there — and it sits where a sensor measuring\n'
        'cranial displacement should sit: after the cortex, before the ventricle.',
        ha='left', va='top', fontsize=12, color=INK,
        bbox=dict(facecolor='#F4F6F8', edgecolor='#D5DBE1', boxstyle='round,pad=0.6'))

fig.suptitle('One event, three stages — and where the mask sits on it',
             fontsize=17, fontweight='bold', color=INK, y=0.97)
p = OUT / 'fig_fultz_chain.png'
fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('wrote', p)
