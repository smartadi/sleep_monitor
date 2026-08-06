"""
Diagnostic: why does S5N2 have ~0 dB broadband SNR?

Short answer: it is NOT an elevated noise floor -- the >=10 Hz noise power is
completely normal (1.07x the cohort median). The physiological signal below
10 Hz has collapsed: respiratory band 0.03x and cardiac band 0.06x the cohort
median. So the sub-10 Hz signal barely exceeds an ordinary noise floor -> SNR
~ 0 dB. This is a weak-coupling / poor-fit night (quiet sensor), not a noisy one.

Reads signal_characterization_cache.pkl. Output:
    fig_s5n2_snr_diagnostic.png
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CH = 'CLE-CRE'
_trapz = getattr(np, 'trapezoid', np.trapz)

c = pickle.load(open(OUT / 'signal_characterization_cache.pkl', 'rb'))
p, labels = c['psd_by_session'], c['labels']

f0 = p[labels[0]][CH][0]
stack = np.vstack([np.interp(f0, *p[l][CH]) for l in labels])
med = np.median(stack, axis=0)

def bp(psd, f, lo, hi):
    m = (f >= lo) & (f < hi)
    return _trapz(psd[m], f[m])

f_s2, psd_s2 = p['S5N2'][CH]
f_s1, psd_s1 = p['S5N1'][CH]

bands = [('Resp\n0.1-0.5', 0.1, 0.5), ('Cardiac\n0.5-3', 0.5, 3.0),
         ('3-10 Hz', 3.0, 10.0), ('NOISE\n10-50', 10.0, 50.0)]
s2 = [bp(psd_s2, f_s2, lo, hi) for _, lo, hi in bands]
mc = [bp(med, f0, lo, hi) for _, lo, hi in bands]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.55, 1]})

# ── A: PSD overlay ────────────────────────────────────────────────────────────
m = (f0 > 0.03) & (f0 <= 50)
axA.loglog(f0[m], med[m], color='#95A5A6', lw=2.4, label='cohort median (12 sessions)')
mm = (f_s1 > 0.03) & (f_s1 <= 50)
axA.loglog(f_s1[mm], psd_s1[mm], color='#2C7FB8', lw=1.3, alpha=0.9, label='S5N1 (same subject, +6.7 dB)')
mm2 = (f_s2 > 0.03) & (f_s2 <= 50)
axA.loglog(f_s2[mm2], psd_s2[mm2], color='#E74C3C', lw=1.6, label='S5N2 (-0.3 dB)')
axA.axvspan(0.03, 10, color='#2C7FB8', alpha=0.07)
axA.axvspan(10, 50, color='#7F8C8D', alpha=0.12)
axA.axvline(10, color='#333', ls='--', lw=1.0)
axA.text(0.6, axA.get_ylim()[1] * 0.4, 'signal (<10 Hz)', color='#2C7FB8', fontsize=9, fontweight='bold')
axA.text(14, axA.get_ylim()[1] * 0.4, 'noise', color='#5D6D7E', fontsize=9, fontweight='bold')
axA.set_xlabel('Frequency (Hz)')
axA.set_ylabel('PSD')
axA.set_title('A. S5N2 collapses in the signal band, matches the pack in noise', fontsize=11, fontweight='bold')
axA.legend(fontsize=8.5, loc='lower left')
axA.grid(True, which='both', alpha=0.15)

# annotation: in the noise band the red line sits on the median
axA.annotate('noise floor:\nnormal', xy=(25, np.interp(25, f0, med)), xytext=(2.0, med.min() * 3),
             fontsize=8, color='#5D6D7E',
             arrowprops=dict(arrowstyle='->', color='#5D6D7E', lw=1))

# ── B: band power vs cohort ──────────────────────────────────────────────────
x = np.arange(len(bands))
w = 0.38
axB.bar(x - w/2, mc, w, color='#95A5A6', label='cohort median')
axB.bar(x + w/2, s2, w, color='#E74C3C', label='S5N2')
axB.set_yscale('log')
axB.set_xticks(x)
axB.set_xticklabels([b[0] for b in bands], fontsize=9)
axB.set_ylabel('Band power')
axB.set_title('B. Physiology gone, noise intact', fontsize=11, fontweight='bold')
axB.legend(fontsize=9)
axB.grid(True, axis='y', which='both', alpha=0.15)
for xi, (a, b) in enumerate(zip(s2, mc)):
    axB.text(xi + w/2, a * 1.15, f'{a/b:.2f}x', ha='center', va='bottom', fontsize=8, color='#C0392B')

fig.suptitle('Why S5N2 has ~0 dB SNR: weak-coupling night (quiet sensor), not a noisy one',
             fontsize=12.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / 'fig_s5n2_snr_diagnostic.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved fig_s5n2_snr_diagnostic.png')
print('S5N2 band power / cohort median:',
      {b[0].split(chr(10))[0]: round(a/mcx, 3) for b, a, mcx in zip(bands, s2, mc)})
