"""
Respiratory band vs SWA/delta band separation in the CAP (SEC) spectrum.
==========================================================================

Paper ridge/spectral section — honest spectral characterization.

The manuscript is framed around "intracranial slow-wave activity (ISWA)", but the
EEG-vs-CAP validation is NEGATIVE (analysis/swa_validation: r=0.015, N3 AUC 0.49).
So we characterize what the CAP delta/SWA-band energy *actually* is:

  1. Decompose the full-night CAP PSD (CLE-CRE primary; also CH/CLE/CRE) into
     the respiratory band (0.1-0.5 Hz) and the SWA/delta band (0.5-4 Hz), with
     delta sub-bands SO (0.5-1), delta_low (1-2), delta_high (2-4).  Quantify
     power fractions and how cleanly the two bands separate (spectral trough).

  2. Test whether the CAP "delta/SWA band" is respiratory-harmonic / cardiac /
     wander rather than cortical slow-wave:
       - the cardiac fundamental peak and its harmonics sit INSIDE the delta band;
       - fraction of delta-band power that lies near resp/cardiac harmonic lines;
       - across-night co-variation of delta-band power with resp / cardiac band
         power (per-window Spearman). Cortical SWA would be independent of both.
       - the EEG-vs-CAP negative (cited from swa_validation).

  3. One paper-ready figure + summary CSV.

Data reuse:
  - Full-night Welch PSDs: writeup/figures/signal_validation/signal_characterization_cache.pkl
    (keys: records / psd_by_session[label][chan]=(freqs,psd) / labels). df=0.033 Hz, 0-50 Hz.
  - Per-window band-power time courses: computed fresh from raw CLE-CRE via
    sleep_monitor.spectral.compute_band_power_ratios (~4 s/session).

Usage:
  .venv/Scripts/python.exe analysis/slow_wave/resp_vs_swa_band_separation.py
"""

from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI, CAP_COLORS, DELTA_SUB_COLORS,
)
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.loader import load_session
from sleep_monitor.spectral import compute_band_power_ratios

CACHE = ROOT / 'writeup/figures/signal_validation/signal_characterization_cache.pkl'
OUT_DIR = ROOT / 'writeup/figures/band_separation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = 'CLE-CRE'
CHANNELS = ['CH', 'CLE', 'CRE', 'CLE-CRE']

# ── Bands ─────────────────────────────────────────────────────────────────────
RESP_BAND = (RESP_LO, RESP_HI)      # 0.1-0.5 Hz
SWA_BAND  = (0.5, 4.0)              # "delta"/SWA band
SUB_BANDS = {                       # sub-bands of the SWA band
    'SO':         (0.5, 1.0),
    'delta_low':  (1.0, 2.0),
    'delta_high': (2.0, 4.0),
}
CARD_BAND = (CARD_LO, CARD_HI)      # 0.5-3.0 Hz  (cardiac reference)
LOW_TOTAL = (0.1, 4.0)             # denominator for resp-vs-SWA fractions
HARMONIC_HALFWIDTH = 0.08          # Hz +/- window around a harmonic line

_trapz = getattr(np, 'trapezoid', np.trapz)


def band_power(freqs, psd, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    return float(_trapz(psd[m], freqs[m]))


def peak_in(freqs, psd, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    fb, pb = freqs[m], psd[m]
    i = int(np.argmax(pb))
    return float(fb[i]), float(pb[i])


# ══════════════════════════════════════════════════════════════════════════════
# Load cached full-night PSDs
# ══════════════════════════════════════════════════════════════════════════════
with open(CACHE, 'rb') as f:
    cached = pickle.load(f)
labels = cached['labels']
psd_by_session = cached['psd_by_session']
records_meta = {r['session']: r for r in cached['records']}

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — spectral decomposition from cached PSDs (primary channel)
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for label in labels:
    freqs, psd = psd_by_session[label][PRIMARY]

    p_resp = band_power(freqs, psd, RESP_BAND)
    p_swa  = band_power(freqs, psd, SWA_BAND)
    p_card = band_power(freqs, psd, CARD_BAND)
    p_low  = band_power(freqs, psd, LOW_TOTAL)          # resp + SWA
    p_sub  = {n: band_power(freqs, psd, b) for n, b in SUB_BANDS.items()}

    # power fractions of the 0.1-4 Hz physiological range
    resp_frac = p_resp / p_low
    swa_frac  = p_swa / p_low
    sub_frac  = {n: p_sub[n] / p_low for n in SUB_BANDS}

    # peaks: respiratory peak (<0.5) and the dominant peak in the SWA band
    f_resp_pk, a_resp_pk = peak_in(freqs, psd, RESP_BAND)
    f_swa_pk,  a_swa_pk  = peak_in(freqs, psd, SWA_BAND)
    # cardiac fundamental: dominant peak away from the resp harmonic clutter
    f_card, a_card = peak_in(freqs, psd, (0.6, 2.5))

    # ── spectral trough between the resp peak and the SWA peak ────────────────
    lo, hi = sorted((f_resp_pk, f_swa_pk))
    mtr = (freqs >= lo) & (freqs <= hi)
    if mtr.sum() >= 3:
        itr = int(np.argmin(psd[mtr]))
        f_trough = float(freqs[mtr][itr])
        a_trough = float(psd[mtr][itr])
    else:
        f_trough, a_trough = np.nan, np.nan
    # separation ratio: how deep the trough is below the smaller flanking peak
    # (>1 => a real valley separating the two bands; ~1 => bands merge)
    sep_ratio = min(a_resp_pk, a_swa_pk) / (a_trough + 1e-30)

    # ── delta-band energy concentrated on the cardiac harmonic comb ───────────
    # comb = {n * f_card}; fraction of SWA-band power within +/- HARMONIC_HALFWIDTH
    # of a cardiac harmonic line inside 0.5-4 Hz.  (The resp peak pins to the
    # 0.1 Hz band edge, so a resp comb would trivially blanket the band; cardiac
    # harmonics give a meaningful "how much of the delta band is the cardiac comb".)
    comb = set()
    n = 1
    while f_card * n <= SWA_BAND[1]:
        if f_card * n >= SWA_BAND[0]:
            comb.add(round(f_card * n, 3))
        n += 1
    m_swa = (freqs >= SWA_BAND[0]) & (freqs <= SWA_BAND[1])
    on_line = np.zeros(m_swa.sum(), dtype=bool)
    fsw = freqs[m_swa]
    for c in comb:
        on_line |= np.abs(fsw - c) <= HARMONIC_HALFWIDTH
    psw = psd[m_swa]
    harm_frac = min(1.0, float(_trapz(psw[on_line], fsw[on_line]) /
                    (_trapz(psw, fsw) + 1e-30))) if on_line.any() else 0.0

    meta = records_meta[label]
    rows.append(dict(
        session=label, subject=meta['subject'], night=meta['night'],
        resp_frac=round(resp_frac, 4),
        swa_frac=round(swa_frac, 4),
        SO_frac=round(sub_frac['SO'], 4),
        delta_low_frac=round(sub_frac['delta_low'], 4),
        delta_high_frac=round(sub_frac['delta_high'], 4),
        card_over_swa=round(p_card / (p_swa + 1e-30), 4),
        f_resp_peak_hz=round(f_resp_pk, 3),
        f_swa_peak_hz=round(f_swa_pk, 3),
        f_card_hz=round(f_card, 3),
        f_card_bpm=round(f_card * 60, 1),
        f_trough_hz=round(f_trough, 3),
        sep_ratio=round(sep_ratio, 2),
        harm_frac_cardiac=round(harm_frac, 3),
    ))

df = pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — per-window band-power time courses from raw (across-night co-variation)
# ══════════════════════════════════════════════════════════════════════════════
WIN_SEC, STEP_SEC, SEG_SEC = 60.0, 30.0, 20.0   # 20 s Welch seg resolves 0.05 Hz
WIN_BANDS = {
    'resp': RESP_BAND, 'cardiac': CARD_BAND, 'swa': SWA_BAND,
    'delta_high': SUB_BANDS['delta_high'],
}
pooled = {k: [] for k in ['delta', 'cardiac', 'resp', 'delta_high', 'session']}
corr_rows = []
for idx in range(len(SESSION_META)):
    s = load_session(idx)
    label = s.label
    sig = (s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64))
    acc = s.cap['acc_mag'].astype(np.float64)
    res = compute_band_power_ratios(
        sig, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC,
        bands=WIN_BANDS, total_range=LOW_TOTAL, welch_seg_sec=SEG_SEC,
        acc_mag=acc, motion_thresh_mad=3.0,
    )
    d  = res['swa_abs']
    c  = res['cardiac_abs']
    r  = res['resp_abs']
    dh = res['delta_high_abs']
    good = np.isfinite(d) & np.isfinite(c) & np.isfinite(r) & np.isfinite(dh)
    d, c, r, dh = d[good], c[good], r[good], dh[good]

    rho_dc = spearmanr(d, c).statistic
    rho_dr = spearmanr(d, r).statistic
    rho_dhc = spearmanr(dh, c).statistic   # disjoint from resp band
    rho_dhr = spearmanr(dh, r).statistic
    corr_rows.append(dict(session=label,
                          rho_delta_cardiac=round(float(rho_dc), 3),
                          rho_delta_resp=round(float(rho_dr), 3),
                          rho_deltahigh_cardiac=round(float(rho_dhc), 3),
                          rho_deltahigh_resp=round(float(rho_dhr), 3),
                          n_windows=int(good.sum())))
    pooled['delta'].append(d); pooled['cardiac'].append(c)
    pooled['resp'].append(r); pooled['delta_high'].append(dh)
    pooled['session'].append(np.full(good.sum(), idx))
    del s, sig, acc

for k in pooled:
    pooled[k] = np.concatenate(pooled[k])
cdf = pd.DataFrame(corr_rows)
df = df.merge(cdf, on='session')

# ── EEG-vs-CAP negative (from swa_validation) ────────────────────────────────
swa_val = pd.read_csv(ROOT / 'analysis/swa_validation/outputs/swa_validation_per_subject.csv')
eeg_neg = dict(
    r_pearson=float(swa_val['Mean_r_pearson'].mean()),
    coherence=float(swa_val['Mean_coherence'].mean()),
    cap_auc=float(swa_val['Mean_AUC'].mean()),
)

csv_path = OUT_DIR / 'resp_vs_swa_band_separation.csv'
df.to_csv(csv_path, index=False)

# pooled overall correlations
rho_dc_all = spearmanr(pooled['delta'], pooled['cardiac']).statistic
rho_dr_all = spearmanr(pooled['delta'], pooled['resp']).statistic
rho_dhc_all = spearmanr(pooled['delta_high'], pooled['cardiac']).statistic

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.24)

# common freq grid + 12-session-mean PSD (primary)
freqs0 = psd_by_session[labels[0]][PRIMARY][0]
psd_stack = np.vstack([np.interp(freqs0, *psd_by_session[l][PRIMARY])
                       for l in labels])
psd_mean = psd_stack.mean(axis=0)

# (A) mean PSD annotated with resp band vs SWA sub-bands ------------------------
axA = fig.add_subplot(gs[0, 0])
fm = (freqs0 >= 0.05) & (freqs0 <= 6.0)
axA.semilogy(freqs0[fm], psd_mean[fm], color='#2C3E50', lw=1.6, zorder=5)
axA.axvspan(*RESP_BAND, color='#27AE60', alpha=0.22, label='Resp 0.1-0.5 Hz')
for name, b in SUB_BANDS.items():
    axA.axvspan(*b, color=DELTA_SUB_COLORS[name], alpha=0.18,
                label=f'{name} {b[0]:g}-{b[1]:g} Hz')
# mark the mean cardiac fundamental
fcard_mean = df['f_card_hz'].mean()
axA.axvline(fcard_mean, color='#C0392B', ls='--', lw=1.2)
axA.text(fcard_mean + 0.05, axA.get_ylim()[1] * 0.3,
         f'cardiac\nfund.\n{fcard_mean:.2f} Hz\n({fcard_mean*60:.0f} BPM)',
         fontsize=7.5, color='#C0392B', va='top')
axA.set_xlabel('Frequency (Hz)', fontsize=10)
axA.set_ylabel('PSD (a.u.$^2$/Hz)', fontsize=10)
axA.set_title('(A) 12-session mean CAP PSD (CLE$-$CRE): resp band vs SWA/delta sub-bands',
              fontsize=10.5, fontweight='bold')
axA.legend(fontsize=8, loc='upper right')
axA.grid(True, which='both', alpha=0.2)
axA.set_xlim(0.05, 6.0)

# (B) per-session power-fraction stacked bars ----------------------------------
axB = fig.add_subplot(gs[0, 1])
xs = np.arange(len(df))
rf = df['resp_frac'].to_numpy()
so = df['SO_frac'].to_numpy()
dl = df['delta_low_frac'].to_numpy()
dh = df['delta_high_frac'].to_numpy()
axB.bar(xs, rf, color='#27AE60', label='Resp 0.1-0.5')
axB.bar(xs, so, bottom=rf, color=DELTA_SUB_COLORS['SO'], label='SO 0.5-1')
axB.bar(xs, dl, bottom=rf+so, color=DELTA_SUB_COLORS['delta_low'], label='delta_low 1-2')
axB.bar(xs, dh, bottom=rf+so+dl, color=DELTA_SUB_COLORS['delta_high'], label='delta_high 2-4')
axB.set_xticks(xs)
axB.set_xticklabels(df['session'], rotation=45, ha='right', fontsize=8)
axB.set_ylabel('Fraction of 0.1-4 Hz power', fontsize=10)
axB.set_ylim(0, 1.0)
axB.set_title('(B) Power fractions per session (resp vs SWA sub-bands)',
              fontsize=10.5, fontweight='bold')
axB.legend(fontsize=8, loc='lower right', ncol=2)

# (C) delta-band power co-varies with cardiac-band power -----------------------
axC = fig.add_subplot(gs[1, 0])
sc = axC.scatter(pooled['cardiac'], pooled['delta'], s=4, alpha=0.15,
                 c='#C0392B', edgecolors='none')
axC.set_xscale('log'); axC.set_yscale('log')
axC.set_xlabel('Cardiac-band power 0.5-3 Hz (per 60 s window)', fontsize=10)
axC.set_ylabel('SWA/delta-band power 0.5-4 Hz', fontsize=10)
axC.set_title(f'(C) SWA/delta power tracks CARDIAC power across the night\n'
              f'pooled Spearman rho = {rho_dc_all:.2f}  (delta vs resp rho = {rho_dr_all:.2f})',
              fontsize=10.5, fontweight='bold')
axC.grid(True, which='both', alpha=0.2)

# (D) per-session co-variation correlations + EEG-vs-CAP negative ---------------
axD = fig.add_subplot(gs[1, 1])
w = 0.38
axD.bar(xs - w/2, df['rho_delta_cardiac'], w, color='#C0392B', alpha=0.85,
        label=r'$\rho$(delta, cardiac)')
axD.bar(xs + w/2, df['rho_delta_resp'], w, color='#27AE60', alpha=0.85,
        label=r'$\rho$(delta, resp)')
axD.axhline(0, color='gray', lw=0.8)
axD.set_xticks(xs)
axD.set_xticklabels(df['session'], rotation=45, ha='right', fontsize=8)
axD.set_ylabel('Per-session Spearman rho', fontsize=10)
axD.set_ylim(-0.2, 1.0)
axD.set_title('(D) CAP delta band co-varies with cardiac/resp (not cortical)',
              fontsize=10.5, fontweight='bold')
axD.legend(fontsize=8, loc='upper left')
axD.grid(True, axis='y', alpha=0.25)
# annotate the EEG-vs-CAP negative
txt = ('EEG vs CAP delta (swa_validation):\n'
       f'  Pearson r = {eeg_neg["r_pearson"]:.3f}\n'
       f'  coherence = {eeg_neg["coherence"]:.3f}\n'
       f'  CAP N3 AUC = {eeg_neg["cap_auc"]:.2f} (chance)\n'
       '=> CAP does NOT track cortical delta')
axD.text(0.98, 0.03, txt, transform=axD.transAxes, fontsize=8,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round', fc='#FDEDEC', ec='#C0392B', alpha=0.9))

fig.suptitle('Respiratory vs SWA/delta band separation in the CAP spectrum — '
             'the CAP "SWA band" is cardiac/respiratory, not cortical slow-wave',
             fontsize=13.5, fontweight='bold', y=0.975)
fig_path = OUT_DIR / 'resp_vs_swa_band_separation.png'
fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════
pd.set_option('display.width', 200, 'display.max_columns', 40)
print('\n=== Per-session decomposition + co-variation ===')
print(df.to_string(index=False))
print('\n=== Cohort means (fraction of 0.1-4 Hz power) ===')
for col in ['resp_frac', 'swa_frac', 'SO_frac', 'delta_low_frac', 'delta_high_frac']:
    print(f'  {col:16s} {df[col].mean():.3f} +/- {df[col].std():.3f}')
print(f'  card/SWA overlap {df["card_over_swa"].mean():.2f} '
      f'(cardiac 0.5-3 Hz as fraction of SWA 0.5-4 Hz power)')
print(f'  harm_frac_cardiac {df["harm_frac_cardiac"].mean():.3f} '
      f'(SWA power within +/-{HARMONIC_HALFWIDTH} Hz of cardiac harmonic lines)')
print(f'  sep_ratio        {df["sep_ratio"].mean():.2f} '
      f'(trough depth below flanking peaks; >1 = real valley at '
      f'{df["f_trough_hz"].mean():.2f} Hz)')
print(f'  f_card           {df["f_card_hz"].mean():.2f} Hz '
      f'({df["f_card_bpm"].mean():.0f} BPM) — sits INSIDE the 0.5-4 Hz SWA band')
print('\n=== Across-night co-variation (pooled) ===')
print(f'  rho(delta, cardiac) = {rho_dc_all:.2f}')
print(f'  rho(delta, resp)    = {rho_dr_all:.2f}')
print(f'  rho(delta_high 2-4, cardiac) = {rho_dhc_all:.2f}  (disjoint from resp band)')
print(f'  per-session mean rho(delta,cardiac) = {df["rho_delta_cardiac"].mean():.2f}, '
      f'rho(delta,resp) = {df["rho_delta_resp"].mean():.2f}')
print('\n=== EEG vs CAP negative (swa_validation) ===')
print(f'  Pearson r = {eeg_neg["r_pearson"]:.3f}, coherence = {eeg_neg["coherence"]:.3f}, '
      f'CAP N3 AUC = {eeg_neg["cap_auc"]:.2f}')
print(f'\nSaved: {csv_path}')
print(f'Saved: {fig_path}')
