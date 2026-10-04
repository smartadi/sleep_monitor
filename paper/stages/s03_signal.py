"""Signal characterization and coupling: band power, SNR, CAP-PSG coherence, respiratory-reference gate.

Manuscript items produced
    Methods ¶154      SNR definition, band fractions, unworn-mask whiteness check
    Results ¶179      respiratory / cardiac share of sub-5-Hz power; SNR 30.0 / 18.7 / 20.7 dB,
                      minimum 6.4 dB; "CH highest in every participant"; S6N1 / S3N1 examples
    Results ¶180-187  registered only: Fig 3, Fig 4, normalized spectra and aperiodic slope
                      are co-author work with no code here (EXTERNAL)
    Supp. Fig S3      physiological-band SNR per recording and raw channel
    Supp. Fig S5      coherence of every capacitive channel with every PSG channel
    Supp. Table S1    respiratory-reference quality gate (+ its paragraph, incl. Flow-vs-Thorax r)
    tables            snr.csv, snr_split.csv, band_fractions.csv, coherence.csv,
                      gt_quality_gate.csv, gt_cross_signal_agreement.csv
    cache             _cache/s03_signal/psd.npz (full-night PSDs),
                      _cache/s03_signal/consolidated_resp_gt.parquet (per-epoch respiratory
                      reference: four PSG rates, gated consensus) for the rate stage

Ported from:
    writeup/figures/signal_validation/signal_characterization.py --recompute
        full-night Welch PSD per channel (psd_by_session), _band_power, resp/card_pow_frac
    writeup/figures/signal_validation/inband_snr.py      band_density, snr_db, figure
    writeup/figures/signal_validation/inband_snr_split.py unworn_psds, per-band SNR
    writeup/figures/signal_validation/generate_band_energy.py  band-energy summary (stdout)
    analysis/rates/cap_psg_coherence.py                  block_coherence, main
    scripts/build_consolidated_resp_gt.py                flow_breaths, rip_breaths,
        quality_filter, sliding_rate, roll_med, wcorr, gate + consensus, agreement table
    analysis/rates/gt_quality_gate_table.py              gate statistic per session

Changes from the legacy code:
    1. CORRECTION band fractions. Legacy (generate_band_energy.py stdout) used three
       recordings (S1N1, S3N1, S6N1), the CLE-CRE differential only, and a 0.05-10 Hz
       denominator; the paper says "of the total power below 5 Hz" for the SEC channels.
       Now computed for all 12 nights and each of CH, CLE, CRE, CLE-CRE, with the
       denominator = all power above DC and up to 5 Hz. The legacy definition is kept as
       extra columns so the paper's range can be traced.
    2. CORRECTION "CH highest SNR in every participant" is checked per night and per
       participant and registered; it fails for S4 (both nights CRE > CH).
    3. CORRECTION "Thorax couples at least as strongly as Flow on every channel" (Fig S5
       caption) is checked per channel and registered; it fails on CH.
    4. CORRECTION the S3 Thorax-vs-Flow correlations (r = -0.47, -0.47) were hard-coded in
       the legacy supplement builder; they are computed here (agreement table).
    5. CORRECTION Methods says SNR comes from sliding-window Welch estimates; every SNR in
       the paper is one full-night Welch PSD per channel (30-s segments). The computation
       is kept and the discrepancy registered.
    6. Flow breath detection. Legacy `flow_breaths` tried neurokit2 (rsp_clean +
       rsp_findpeaks) and fell back to the band-pass + find_peaks detector used for the
       belts. The legacy parquet behind Table S1 matches the fallback in 99.9+% of epochs
       and the neurokit2 path in ~3% (S1N1; neurokit2 0.2.13 as installed), so the
       published table came from the fallback (neurokit2 must have raised in that run).
       The port calls the fallback directly; this reproduces the table and removes a silent
       dependency switch. Switching Flow to neurokit2 would change every Flow column of
       Table S1 (e.g. S1N1 +0.66 -> +0.43) and un-drop S3 Thorax's sign (-0.06 -> +0.08).
    Not changed: SNR, coherence and gate computations are byte-for-byte the legacy ones
    (see paper/checks/s03_signal_vs_legacy.py). Dropped: exploratory figures 5-10 of the
    signal_validation folder, the specificity Wilcoxon table, and the gate "fallback to two
    best signals" path's ranking print (the fallback itself is kept).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, resample_poly, welch

from seclib import (CACHE_DIR, FS, LABELS, RESP_LO, RESP_HI, CARD_LO, CARD_HI, TAB_DIR,
                    get_session, Numbers)
from seclib.config import BASELINE_FILE, CAP_COLORS
from seclib.filters import bandpass
from seclib.figures import save
from seclib.loader import load_apnea_events

STAGE = 's03_signal'
REQUIRES: list[str] = []

CACHE = CACHE_DIR / STAGE
TABLES = TAB_DIR / STAGE

RAW = ['CH', 'CLE', 'CRE']                 # the SNR channels (CLE-CRE excluded, ¶154)
ALL_CAP = ['CH', 'CLE', 'CRE', 'CLE-CRE']
RESP_BAND, CARD_BAND = (RESP_LO, RESP_HI), (CARD_LO, CARD_HI)
SEG_SEC = 30.0                             # Welch segment for the full-night PSD
_trapz = getattr(np, 'trapezoid', None) or np.trapz


# ═════════════════════════════════════════════════════════════════════════════
# 1. Full-night PSD per channel (signal_characterization.py)
# ═════════════════════════════════════════════════════════════════════════════

def raw_channels(s):
    """Raw capacitive channels in float64; CLE-CRE formed after the cast."""
    cle, cre = s.cap['CLE'].astype(np.float64), s.cap['CRE'].astype(np.float64)
    return {'CH': s.cap['CH'].astype(np.float64), 'CLE': cle, 'CRE': cre,
            'CLE-CRE': cle - cre}


def full_night_psd(x, fs=FS):
    n = int(SEG_SEC * fs)
    return welch(x, fs=fs, nperseg=n, noverlap=n // 2, scaling='density')


def all_psds():
    """{label: {chan: psd}} plus the common frequency grid; cached as npz."""
    path = CACHE / 'psd.npz'
    if path.exists():
        z = np.load(path)
        return z['freqs'], {lab: {c: z[f'{lab}__{c}'] for c in ALL_CAP} for lab in LABELS}
    out, freqs = {}, None
    for lab in LABELS:
        sigs = raw_channels(get_session(lab, profile=False))
        out[lab] = {}
        for c, x in sigs.items():
            freqs, out[lab][c] = full_night_psd(x)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, freqs=freqs,
             **{f'{lab}__{c}': p for lab, d in out.items() for c, p in d.items()})
    return freqs, out


# ═════════════════════════════════════════════════════════════════════════════
# 2. Band fractions (¶179)
# ═════════════════════════════════════════════════════════════════════════════

def band_power(freqs, psd, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    return float(_trapz(psd[m], freqs[m]))


def band_fractions(freqs, psds):
    rows = []
    for lab in LABELS:
        for c in ALL_CAP:
            p = psds[lab][c]
            # CORRECTION: denominator is power below 5 Hz (DC bin excluded), as the paper
            # states; legacy used 0.05-10 Hz.
            below5 = band_power(freqs, p, (1e-9, 5.0))
            legacy = band_power(freqs, p, (0.05, 10.0))
            r, k = band_power(freqs, p, RESP_BAND), band_power(freqs, p, CARD_BAND)
            rows.append(dict(session=lab, channel=c,
                             resp_frac=r / below5, card_frac=k / below5,
                             resp_frac_legacy=r / legacy, card_frac_legacy=k / legacy))
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Physiological-band SNR (inband_snr.py, inband_snr_split.py)
# ═════════════════════════════════════════════════════════════════════════════

SIG_BAND = (0.1, 3.0)          # respiration + cardiac
NOISE_LO = 10.0                # noise = 10 Hz .. Nyquist
UNWORN_FS = 1000.0 / 9.0       # unworn mask ticks every 9 ms


def band_density(freqs, psd, lo, hi):
    """Mean PSD per Hz over (lo, hi): bandwidth-independent, unlike integrated power."""
    m = (freqs > lo) & (freqs < hi)
    return float(np.mean(psd[m]))


def snr_db(freqs, psd, band=SIG_BAND):
    p_sig = band_density(freqs, psd, *band)
    p_noise = band_density(freqs, psd, NOISE_LO, freqs.max())
    return 10.0 * np.log10(p_sig / (p_noise + 1e-30) + 1e-30)


def unworn_psds():
    """PSD per channel of the mask recorded with nobody wearing it, on the 100 Hz grid."""
    d = pd.read_csv(BASELINE_FILE)
    d.columns = [c.strip() for c in d.columns]
    assert np.median(np.diff(d['timeMS'].to_numpy())) == 9, 'unworn tick is not 9 ms'
    # 1000/9 Hz -> 100 Hz exactly; also puts Nyquist at 50 Hz like the nights
    return {c: full_night_psd(resample_poly(d[c].to_numpy(float), 9, 10)) for c in RAW}


def snr_tables(freqs, psds):
    pooled = pd.DataFrame([{'session': lab, **{c: round(snr_db(freqs, psds[lab][c]), 2)
                                               for c in RAW}} for lab in LABELS])
    unworn = unworn_psds()
    split = []
    for name, band in (('respiratory', RESP_BAND), ('cardiac', CARD_BAND)):
        for lab in LABELS:
            split.append({'recording': lab, 'band': name,
                          **{c: round(snr_db(freqs, psds[lab][c], band), 2) for c in RAW}})
    for name, band in (('respiratory', RESP_BAND), ('cardiac', CARD_BAND), ('pooled', SIG_BAND)):
        split.append({'recording': 'unworn (control)', 'band': name,
                      **{c: round(snr_db(*unworn[c], band), 2) for c in RAW}})
    split = pd.DataFrame(split)
    # noise floor density (10-50 Hz) per night, to back "nearly constant noise floor"
    floor = pd.DataFrame([{'session': lab, **{c: 10 * np.log10(band_density(
        freqs, psds[lab][c], NOISE_LO, freqs.max())) for c in RAW}} for lab in LABELS])
    return pooled, split, floor


def fig_snr(pooled):
    """Fig S3: grouped bars, one group per recording, three raw channels."""
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    x = np.arange(len(LABELS))
    w = 0.26
    for i, c in enumerate(RAW):
        ax.bar(x + (i - 1) * w, pooled[c].to_numpy(), w, color=CAP_COLORS[c],
               edgecolor='white', linewidth=0.5, zorder=3,
               label=f'{c}  (mean {pooled[c].mean():.1f} dB)')
    ax.axhline(0, color='gray', lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=13)
    ax.set_ylabel('Physiological-band SNR (dB, per-Hz density)', fontsize=15)
    ax.set_xlabel('Recording', fontsize=15)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True, axis='y', alpha=0.25, zorder=0)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.95, ncol=3)
    ax.margins(y=0.14)
    fig.tight_layout()
    save(fig, 'figS3_inband_snr', STAGE)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Coherence, capacitive x PSG (cap_psg_coherence.py)
# ═════════════════════════════════════════════════════════════════════════════

PSG = ['Flow', 'Thorax', 'Abdomen', 'Pleth', 'ECG', 'EEG']   # EEG = negative control
BANDS = {'resp': RESP_BAND, 'card': CARD_BAND}
BLOCK_S = 300.0
NPERSEG = int(30 * FS)
N_SEGMENTS = int((BLOCK_S * FS - NPERSEG) // (NPERSEG // 2) + 1)   # 19 per 5-min block


def block_coherence(a, b):
    """Mean and peak magnitude-squared coherence per band, per 5-minute block."""
    n = int(BLOCK_S * FS)
    out = {k: [] for k in BANDS}
    peak = {k: [] for k in BANDS}
    for i in range(len(a) // n):
        x, y = a[i * n:(i + 1) * n], b[i * n:(i + 1) * n]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        f, cxy = signal.coherence(x, y, fs=FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        for band, (lo, hi) in BANDS.items():
            m = (f >= lo) & (f <= hi)
            out[band].append(float(np.mean(cxy[m])))
            peak[band].append(float(np.max(cxy[m])))
    return out, peak


def coherence_table():
    rows = []
    for lab in LABELS:
        s = get_session(lab, profile=False)
        cap = dict(s.cap)
        cap['CLE-CRE'] = cap['CLE'] - cap['CRE']       # float32, as legacy
        for c in ALL_CAP:
            a = np.asarray(cap[c], float)
            a = a - a.mean()
            for p in PSG:
                b = np.asarray(s.psg[p], float)
                b = b - b.mean()
                n = min(len(a), len(b))
                mean_c, peak_c = block_coherence(a[:n], b[:n])
                for band in BANDS:
                    if mean_c[band]:
                        rows.append(dict(session=lab, subject=s.subject, cap=c, psg=p,
                                         band=band, coh_mean=float(np.median(mean_c[band])),
                                         coh_peak=float(np.median(peak_c[band])),
                                         n_blocks=len(mean_c[band])))
        print(f'  coherence {lab}')
    return pd.DataFrame(rows)


def coherence_matrix(coh):
    """Median over the twelve recordings of each recording's median block coherence."""
    return coh.groupby(['band', 'cap', 'psg']).coh_mean.median()


def fig_coherence(mat):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(PSG))
    for ax, band, title in zip(axes, BANDS, ('Respiratory band (0.1–0.5 Hz)',
                                             'Cardiac band (0.5–3.0 Hz)')):
        for c in ALL_CAP:
            ax.plot(x, [mat[(band, c, p)] for p in PSG], 'o-', color=CAP_COLORS[c],
                    label=c, lw=1.5, ms=5)
        ax.axhline(1 / N_SEGMENTS, color='gray', ls='--', lw=1)
        ax.text(len(PSG) - 1, 1 / N_SEGMENTS, f'noise floor 1/{N_SEGMENTS}',
                ha='right', va='bottom', fontsize=9, color='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(PSG, rotation=45, ha='right')
        ax.set_ylabel('Median in-band coherence')
        ax.set_title(title, loc='left')
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, 'figS5_cap_psg_coherence', STAGE)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Respiratory reference and its quality gate (build_consolidated_resp_gt.py)
# ═════════════════════════════════════════════════════════════════════════════

SIGNALS = ['Flow', 'Thorax', 'Abdomen', 'RIPSum']
KEEP_THRESHOLD = 0.10
AGREE_TOL_HZ = 2.0 / 60.0      # two signals "agree" within 2 br/min


def rip_breaths(sig, fs):
    """Bandpass + peak detection for RIP / effort signals."""
    bp = bandpass(sig.astype(np.float64), RESP_LO, RESP_HI, fs)
    peaks, _ = find_peaks(bp, distance=int(fs / RESP_HI * 0.6),
                          prominence=0.05 * np.std(bp))
    return peaks


def quality_filter(pk, fs):
    """Drop a breath whose interval falls outside the respiratory band."""
    if len(pk) < 2:
        return pk
    iv = np.diff(pk) / fs
    keep = np.ones(len(pk), dtype=bool)
    keep[1:][~((iv >= 1.0 / RESP_HI) & (iv <= 1.0 / RESP_LO))] = False
    return pk[keep]


def sliding_rate(pk_times, centres_s, win_sec=30.0):
    half = win_sec / 2.0
    out = np.full(len(centres_s), np.nan)
    for i, tc in enumerate(centres_s):
        w = pk_times[(pk_times >= tc - half) & (pk_times <= tc + half)]
        if len(w) >= 2:
            out[i] = (len(w) - 1) / (w[-1] - w[0])
    return out


def roll_med(x, k=5):
    o = np.full_like(x, np.nan, float)
    h = k // 2
    for i in range(len(x)):
        s = x[max(0, i - h):i + h + 1]
        s = s[np.isfinite(s)]
        if len(s):
            o[i] = np.median(s)
    return o


def wcorr(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 20 or np.std(a[v]) < 1e-9 or np.std(b[v]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[v], b[v])[0, 1])


def gate(rates):
    """Each signal's r with the median of the other three; keep r >= 0.10, at least two."""
    M = np.vstack([rates[s] for s in SIGNALS])
    corr = {s: wcorr(M[j], np.nanmedian(np.delete(M, j, axis=0), axis=0))
            for j, s in enumerate(SIGNALS)}
    kept = [s for s in SIGNALS if np.isfinite(corr[s]) and corr[s] >= KEEP_THRESHOLD]
    if len(kept) < 2:
        kept = sorted(SIGNALS, key=lambda s: corr[s] if np.isfinite(corr[s]) else -1,
                      reverse=True)[:2]
    return corr, kept


def resp_reference():
    """Per-epoch rates of four PSG signals (30-s windows, 5-s step), gate, consensus."""
    epochs, gate_rows, agree_rows = [], [], []
    for lab in LABELS:
        s = get_session(lab, profile=False)
        s.apnea_events = load_apnea_events(s)
        fs, psg = s.fs, s.psg
        thx_bp = bandpass(psg['Thorax'].astype(np.float64), RESP_LO, RESP_HI, fs)
        abd_bp = bandpass(psg['Abdomen'].astype(np.float64), RESP_LO, RESP_HI, fs)
        # RIPSum with a global polarity correction for thoraco-abdominal asynchrony
        sign = 1.0 if np.corrcoef(thx_bp, abd_bp)[0, 1] >= 0 else -1.0
        # Flow uses the same band-pass detector as the belts: that is what produced the
        # published table (see docstring, change 6), not the neurokit2 path legacy tried first
        breaths = {'Flow': rip_breaths(psg['Flow'].astype(np.float64), fs),
                   'Thorax': rip_breaths(psg['Thorax'].astype(np.float64), fs),
                   'Abdomen': rip_breaths(psg['Abdomen'].astype(np.float64), fs),
                   'RIPSum': rip_breaths(thx_bp + sign * abd_bp, fs)}
        win_n, step_n = int(30 * fs), int(5 * fs)
        centres_s = np.array([(i + win_n / 2.0) / fs
                              for i in range(0, s.n_samples - win_n + 1, step_n)])
        t_hr = centres_s / 3600.0
        rates = {k: sliding_rate(quality_filter(v, fs) / fs, centres_s)
                 for k, v in breaths.items()}

        corr, kept = gate(rates)
        M = np.vstack([rates[k] for k in kept])
        consensus = np.nanmedian(M, axis=0)
        n_agree = np.sum(np.abs(M - consensus) <= AGREE_TOL_HZ, axis=0)
        confidence = np.clip(1.0 - np.nanstd(M, axis=0) / AGREE_TOL_HZ, 0, 1)
        dropped = [k for k in SIGNALS if k not in kept]
        epochs.append(pd.DataFrame({
            'session': lab, 't_hr': t_hr,
            'rate_flow': rates['Flow'], 'rate_thorax': rates['Thorax'],
            'rate_abdomen': rates['Abdomen'], 'rate_ripsum': rates['RIPSum'],
            'rate_consensus': consensus, 'n_agree': n_agree.astype(int),
            'confidence': confidence, 'apnea': s.apnea_at(t_hr).astype(int)}))
        gate_rows.append({'session': lab, **{'r_' + k.lower(): corr[k] for k in SIGNALS},
                          'n_kept': len(kept), 'dropped': ','.join(dropped) or '-'})
        fr, rr = rates['Flow'], rates['RIPSum']
        agree_rows.append({
            'session': lab, 'r_flow_ripsum': wcorr(fr, rr),
            # CORRECTION: computed here; legacy hard-coded the S3 values in the supplement
            'r_flow_thorax': wcorr(fr, rates['Thorax']),
            'r_flow_abdomen': wcorr(fr, rates['Abdomen']),
            'r_flow_ripsum_smooth': wcorr(roll_med(fr), roll_med(rr)),
            'r_flow_ripsum_fluct': wcorr(fr - roll_med(fr, 9), rr - roll_med(rr, 9)),
            'mean_n_agree': float(np.nanmean(n_agree)),
            'pct_apnea': float(np.mean(epochs[-1].apnea > 0) * 100),
            'dropped': ','.join(dropped) or '-'})
        print(f'  resp reference {lab}: dropped {dropped or "none"}')
    return (pd.concat(epochs, ignore_index=True), pd.DataFrame(gate_rows),
            pd.DataFrame(agree_rows))


# ═════════════════════════════════════════════════════════════════════════════
# 6. Numbers
# ═════════════════════════════════════════════════════════════════════════════

def _rng(v, pct=True, d=0):
    v = np.asarray(v) * (100 if pct else 1)
    return f'{v.min():.{d}f}–{v.max():.{d}f}'


def register(nb, frac, pooled, split, floor, coh, mat, gq, agree):
    # ── ¶154 Methods ─────────────────────────────────────────────────────────
    nb.add('snr_estimator', '¶154', 'PSD estimator behind the SNR',
           'one full-night Welch PSD per channel (30-s segments, 50% overlap)',
           paper_value='sliding-window Welch power spectral density estimates', status='DIFF',
           source='full_night_psd',
           note='CORRECTION: no sliding window anywhere in the SNR or band-fraction path; '
                'text should say one Welch estimate over the whole night.')
    nb.add('snr_noise_band', '¶154', 'electronic-noise reference band', '10 to 50',
           paper_value='10 to 50 Hz', unit='Hz', source='snr_db')
    ctrl = split[(split.recording == 'unworn (control)') & (split.band == 'pooled')]
    nb.add('unworn_whiteness', '¶154',
           'unworn mask: 0.1–3 Hz density over 10–50 Hz density (0 dB = white floor)',
           _rng(ctrl[RAW].to_numpy().ravel(), pct=False, d=2), unit='dB',
           source='unworn_psds/snr_db',
           note='The "approximately white from 0.1 to 50 Hz" check: the same SNR on the '
                'unworn recording, per channel CH/CLE/CRE. No legacy script wrote this ratio.')

    # ── ¶179 band fractions ─────────────────────────────────────────────────
    sec = frac[frac.channel.isin(RAW)]
    leg = frac[(frac.channel == 'CLE-CRE') & frac.session.isin(['S1N1', 'S3N1', 'S6N1'])]
    per_ch = '; '.join(f'{c} resp {_rng(frac[frac.channel == c].resp_frac)}%, '
                       f'card {_rng(frac[frac.channel == c].card_frac)}%' for c in ALL_CAP)
    nb.add('resp_frac_range', '¶179', 'respiratory band share of sub-5-Hz power, SEC channels, '
           '12 nights', _rng(sec.resp_frac), paper_value='29–48%', unit='%',
           source='band_fractions',
           note=f'CORRECTION: all 12 nights x CH/CLE/CRE, denominator 0<f<=5 Hz. Per channel: '
                f'{per_ch}. Legacy definition (CLE-CRE, S1N1/S3N1/S6N1, 0.05-10 Hz) gives '
                f'{_rng(leg.resp_frac_legacy)}%; all 12 nights under the legacy definition '
                f'{_rng(frac[frac.channel == "CLE-CRE"].resp_frac_legacy)}%.')
    nb.add('card_frac_range', '¶179', 'cardiac band share of sub-5-Hz power, SEC channels, '
           '12 nights', _rng(sec.card_frac), paper_value='8–48%', unit='%',
           source='band_fractions',
           note=f'CORRECTION: as resp_frac_range. Legacy definition (3 nights, CLE-CRE) gives '
                f'{_rng(leg.card_frac_legacy)}%; all 12 nights legacy definition '
                f'{_rng(frac[frac.channel == "CLE-CRE"].card_frac_legacy)}%.')
    for lab, col, pv, what in (('S6N1', 'resp_frac', '48%', 'S6N1 respiratory share'),
                               ('S6N1', 'card_frac', '8%', 'S6N1 cardiac share'),
                               ('S3N1', 'card_frac', '48%', 'S3N1 cardiac share')):
        r = frac[frac.session == lab].set_index('channel')
        nb.add(f'{lab}_{col}', '¶179', what + ' (CLE-CRE, sub-5-Hz denominator)',
               float(r.loc['CLE-CRE', col] * 100), paper_value=pv, unit='%',
               source='band_fractions',
               note=f'CORRECTION: denominator <5 Hz. Legacy definition '
                    f'{r.loc["CLE-CRE", col + "_legacy"] * 100:.1f}%. Raw channels: '
                    + ', '.join(f'{c} {r.loc[c, col] * 100:.0f}%' for c in RAW))

    # ── ¶179 SNR ────────────────────────────────────────────────────────────
    for c, pv in (('CH', '30.0 dB'), ('CLE', '18.7 dB'), ('CRE', '20.7 dB')):
        nb.add(f'snr_mean_{c}', '¶179', f'physiological-band SNR, {c}, mean of 12 nights',
               float(pooled[c].mean()), paper_value=pv, unit='dB', source='snr_tables',
               note='mean of per-night values; per-night range ' +
                    _rng(pooled[c], pct=False, d=1) + ' dB')
    mins = pooled[RAW].min()
    nb.add('snr_min', '¶179', 'lowest SNR over 12 nights x 3 channels', float(mins.min()),
           paper_value='6.4 dB', unit='dB', source='snr_tables',
           note=f'{mins.idxmin()} in {pooled.session[pooled[mins.idxmin()].idxmin()]}; text says '
                f'"exceeded 6.4 dB", true (6.44).')
    best = pooled.set_index('session')[RAW].idxmax(axis=1)
    not_ch = best[best != 'CH']
    subj_ok = sum(all(best[[f'S{i}N1', f'S{i}N2']] == 'CH') for i in range(1, 7))
    nb.add('ch_highest', '¶179', 'CH has the highest SNR in every participant',
           f'{subj_ok}/6 participants, {(best == "CH").sum()}/12 nights',
           paper_value='CH consistently yielded the highest SNR in every participant',
           status='DIFF' if len(not_ch) else 'MATCH', source='snr_tables',
           note='CORRECTION: fails in ' + ', '.join(f'{k} ({v} highest)' for k, v in not_ch.items()))
    sd = pooled[RAW].std()
    holds = sd['CH'] < min(sd['CLE'], sd['CRE'])
    nb.add('snr_variability', '¶179', 'SD of per-night SNR, CH / CLE / CRE',
           ' / '.join(f'{sd[c]:.1f}' for c in RAW), unit='dB', source='snr_tables',
           paper_value='The greater session-to-session variability in CLE and CRE',
           status='MATCH' if holds else 'DIFF',
           note='holds only if both CLE and CRE vary more than CH across nights (SD of '
                'per-night SNR); CLE varies less than CH.' if not holds else '')
    spread = floor[RAW].max() - floor[RAW].min()
    nb.add('noise_floor_spread', '¶179', 'range over nights of the 10–50 Hz noise density, '
           'CH / CLE / CRE', ' / '.join(f'{spread[c]:.1f}' for c in RAW), unit='dB',
           paper_value='the nearly constant electronic noise floor across sessions',
           status='MATCH' if spread.max() < 3 else 'DIFF', source='snr_tables',
           note='max - min over the 12 nights of 10*log10(mean 10-50 Hz density). The floor '
                'moves by more than 10 dB between nights on CH and CRE, so SNR differences '
                'are not purely signal differences.')
    nb.add('snr_fig_ref', '¶179', 'SNR figure cross-reference', 'Fig. S3',
           paper_value='Fig. S6', status='DIFF', source='-',
           note='text cites Fig. S6; the SNR figure is Fig. S3.')

    # ── ¶180-187: co-author work, no code here ─────────────────────────────
    for id_, sec_, what, pv in (
            ('nrem_lowfreq', '¶180', 'NREM > wake normalized spectra band', '0.04–0.1 Hz'),
            ('nrem_peaks', '¶180', 'NREM spectral peaks', '0.17 and 0.22 Hz'),
            ('fig3', '¶181-182', 'Fig 3 normalized power spectra (CLE, CRE)', 'Fig. 3'),
            ('slope_p', '¶184', 'aperiodic slope wake->NREM, CLE and CLE−CRE', 'p = 0.008 and 0.033'),
            ('slope_band_p', '¶184', 'band-wise slope differences, non-significant',
             'p = 0.14–0.90'),
            ('fig4', '¶186-187', 'Fig 4 spectral slope per night', 'Fig. 4')):
        nb.add(id_, sec_, what, 'no code', paper_value=pv, status='EXTERNAL', source='-',
               note='co-author analysis; no spectral-normalization or aperiodic fit in the repo.')

    # ── Supp Fig S3 ─────────────────────────────────────────────────────────
    nb.add('S3_band', 'Supp Fig S3', 'signal band of the SNR', '0.1–3', paper_value='0.1–3 Hz',
           unit='Hz', source='snr_db')
    nb.add('S3_positive', 'Supp Fig S3', 'SNR positive for every channel in all sessions',
           f'{int((pooled[RAW] > 0).to_numpy().sum())}/36 positive',
           paper_value='The SNR was positive for every channel in all sessions',
           status='MATCH' if (pooled[RAW] > 0).all().all() else 'DIFF', source='snr_tables')

    # ── Supp Fig S5 ─────────────────────────────────────────────────────────
    nb.add('S5_segments', 'Supp Fig S5', 'Welch segments per 5-min estimate', N_SEGMENTS,
           paper_value='about 19 segments', source='N_SEGMENTS')
    nb.add('S5_floor', 'Supp Fig S5', 'coherence noise floor 1/K', 1 / N_SEGMENTS,
           paper_value='0.053', source='N_SEGMENTS')
    fails = [c for c in ALL_CAP if mat[('resp', c, 'Thorax')] < mat[('resp', c, 'Flow')]]
    nb.add('S5_thorax_ge_flow', 'Supp Fig S5',
           'resp-band coherence, Thorax >= Flow on every channel',
           f'{len(ALL_CAP) - len(fails)}/{len(ALL_CAP)} channels',
           paper_value='Thoracic effort couples at least as strongly as nasal airflow in the '
                       'respiratory band on every channel',
           status='DIFF' if fails else 'MATCH', source='coherence_matrix',
           note='CORRECTION: fails on ' + ', '.join(
               f'{c} (Thorax {mat[("resp", c, "Thorax")]:.3f} < Flow {mat[("resp", c, "Flow")]:.3f})'
               for c in fails) + '. Values are medians over 12 nights (pooled across nights).')

    # ── Table S1 and its paragraph ──────────────────────────────────────────
    paper_s1 = {   # V11 supplement Table S1, Flow / Thorax / Abdomen / RIPSum
        'S1N1': '+0.66 +0.83 +0.81 +0.87', 'S1N2': '+0.64 +0.78 +0.84 +0.87',
        'S2N1': '+0.55 +0.84 +0.86 +0.90', 'S2N2': '+0.55 +0.85 +0.84 +0.86',
        'S3N1': '+0.30 -0.06 +0.71 +0.76', 'S3N2': '+0.25 -0.13 +0.53 +0.62',
        'S4N1': '+0.48 +0.85 +0.81 +0.87', 'S4N2': '+0.42 +0.77 +0.78 +0.82',
        'S5N1': '+0.68 +0.61 +0.83 +0.85', 'S5N2': '+0.66 +0.30 +0.71 +0.71',
        'S6N1': '+0.31 +0.72 +0.80 +0.84', 'S6N2': '+0.23 +0.70 +0.71 +0.77'}
    g = gq.set_index('session')
    for lab in LABELS:
        for sig, pv in zip(SIGNALS, paper_s1[lab].split()):
            nb.add(f'S1_{lab}_{sig}', 'Supp Table S1', f'gate r, {lab} {sig}',
                   float(g.loc[lab, 'r_' + sig.lower()]), paper_value=pv, source='gate')
    nb.add('S1_threshold', 'Supp Table S1', 'gate threshold', KEEP_THRESHOLD,
           paper_value='+0.10', source='KEEP_THRESHOLD')
    nb.add('S1_s3_thorax', 'Supp ¶216', 'S3 Thorax gate r (N1, N2)',
           f"{g.loc['S3N1', 'r_thorax']:+.2f}, {g.loc['S3N2', 'r_thorax']:+.2f}",
           paper_value='−0.06 and −0.13', source='gate')
    a = agree.set_index('session')
    nb.add('S1_s3_flow_thorax', 'Supp ¶216', 'S3 Thorax vs Flow rate correlation (N1, N2)',
           f"{a.loc['S3N1', 'r_flow_thorax']:+.2f}, {a.loc['S3N2', 'r_flow_thorax']:+.2f}",
           paper_value='r = −0.47 and −0.47', source='resp_reference',
           note='CORRECTION: previously hard-coded in writeup/edits/apply_gt_gate_supplement.py; '
                'now computed (within-night Pearson r of 30-s rates, 5-s step).')
    kept = gq[gq.dropped == '-']
    nb.add('S1_min_kept', 'Supp ¶216', 'lowest gate r among kept sensors in nights without '
           'a drop', float(kept[[f'r_{s.lower()}' for s in SIGNALS]].min().min()),
           paper_value='+0.23', source='gate',
           note='text: "every other sensor in every other night scored +0.23 or above"; the '
                'kept Flow in S3N1/S3N2 scores ' + ', '.join(
                    f"{g.loc[l, 'r_flow']:+.2f}" for l in ('S3N1', 'S3N2')))
    nb.add('S1_all_four', 'Supp ¶216', 'nights using all four signals',
           int((gq.n_kept == 4).sum()), paper_value='eleven of the twelve', status=None,
           source='gate', note='text: "eleven of the twelve nights use all four signals and '
                               'the two S3 nights use three" -- 12 - 2 = 10, not eleven.')


# ═════════════════════════════════════════════════════════════════════════════

def run():
    TABLES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    freqs, psds = all_psds()
    frac = band_fractions(freqs, psds)
    frac.to_csv(TABLES / 'band_fractions.csv', index=False)

    pooled, split, floor = snr_tables(freqs, psds)
    pooled.to_csv(TABLES / 'snr.csv', index=False)
    split.to_csv(TABLES / 'snr_split.csv', index=False)
    fig_snr(pooled)

    coh = coherence_table()
    coh.to_csv(TABLES / 'coherence.csv', index=False)
    mat = coherence_matrix(coh)
    fig_coherence(mat)

    epochs, gq, agree = resp_reference()
    epochs.to_parquet(CACHE / 'consolidated_resp_gt.parquet', index=False)
    gq.to_csv(TABLES / 'gt_quality_gate.csv', index=False)
    agree.to_csv(TABLES / 'gt_cross_signal_agreement.csv', index=False)

    nb = Numbers(STAGE)
    register(nb, frac, pooled, split, floor, coh, mat, gq, agree)
    nb.save()


if __name__ == '__main__':
    run()
