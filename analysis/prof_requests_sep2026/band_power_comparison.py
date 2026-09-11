"""
Request B (Prof, 2026-09-09): "Power density comparison among cardiac bands,
respiratory bands, harmonic combs, and SWS in your ridge analysis. I would like to
see the band power."

Question
--------
How much spectral power does each SEC feature carry? Put the respiratory band, the
cardiac band, the SWS (slow-wave / delta-equivalent) band, and the harmonic-comb
events on one comparison so the band power is directly visible.

Approach (descriptive)
--------
Per 30 s epoch, per channel (CLE, CRE, CH), compute a Welch PSD (linear, fF^2/Hz)
and integrate power (fF^2) over:
  * respiratory  0.10-0.50 Hz
  * cardiac      0.50-3.00 Hz
  * SWS band     0.50-2.00 Hz  (delta-equivalent; reported over N3 epochs)
Harmonic-comb events are the ladder windows from
  reports/slow_wave/ladder_quantify/per_window_channels.parquet
joined on (session, channel, epoch); their in-band power (0.10-3.0 Hz) is compared
against non-comb NREM epochs. Motion epochs dropped.

Two views:
  1. Median PSD per channel with the three bands shaded (literally "see the band power").
  2. Band power (fF^2) per category x channel, plus comb vs non-comb power.

Outputs
-------
  reports/prof_requests_sep2026/band_power_epochs.parquet
  reports/prof_requests_sep2026/band_power_summary.csv
  notebooks/plots/prof_requests_sep2026/B_psd_bands.png
  notebooks/plots/prof_requests_sep2026/B_band_power_bars.png

Usage:
    .venv/Scripts/python.exe analysis/prof_requests_sep2026/band_power_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile, FS
from sleep_monitor.config import (STAGE_LABELS, CAP_COLORS, CAP_SCALE_TO_FF,
                                  RESP_LO, RESP_HI, CARD_LO, CARD_HI, CAP_UNIT_SQ)
from sleep_monitor.sessions import SESSION_META

ROOT = Path(__file__).resolve().parents[2]
SW = ROOT / 'reports' / 'slow_wave'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'prof_requests_sep2026'
REPORT_DIR = ROOT / 'reports' / 'prof_requests_sep2026'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
CHANS = ['CLE', 'CRE', 'CH']
SWS_LO, SWS_HI = 0.5, 2.0          # delta-equivalent SWS band
COMB_LO, COMB_HI = 0.1, 3.0        # in-band range used for comb vs non-comb
PSD_FMAX = 4.0
CH_COLOR = {'CLE': CAP_COLORS['CLE'], 'CRE': CAP_COLORS['CRE'], 'CH': CAP_COLORS['CH']}


def band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapz(pxx[m], f[m]))


def process(idx, comb_lut):
    meta = SESSION_META[idx]
    s = load_session(idx)
    sp = load_sleep_profile(s)
    if sp is None:
        return None, None
    t_hr = s.time_hr.astype(np.float64)
    raw = {ch: s.cap[ch].astype(np.float64) * CAP_SCALE_TO_FF for ch in ['CLE', 'CRE', 'CH']}
    acc = s.cap['acc_mag'].astype(np.float64)
    dt_hr = EPOCH_SEC / 3600.0
    nper = int(8 * FS)                        # 8 s Welch segments

    rows, psd_accum = [], {ch: [] for ch in CHANS}
    psd_f = None
    for j in range(len(sp['t_ep_hr'])):
        t0 = sp['t_ep_hr'][j]; t1 = t0 + dt_hr
        if t0 < 0 or t1 > t_hr[-1]:
            continue
        i0 = np.searchsorted(t_hr, t0); i1 = np.searchsorted(t_hr, t1)
        if i1 - i0 < int(0.5 * FS * EPOCH_SEC):
            continue
        motion = float(acc[i0:i1].std())
        tc = t0 + dt_hr / 2.0
        row = {'session': meta['label'], 'subject': meta['subject'], 'epoch': j,
               't_hr': tc, 'stage_code': int(sp['codes'][j]),
               'stage': STAGE_LABELS.get(int(sp['codes'][j]), '?'), 'acc_std': motion}
        for ch in CHANS:
            seg = raw[ch][i0:i1]
            seg = seg - seg.mean()
            f, pxx = welch(seg, fs=FS, nperseg=min(nper, len(seg)))
            row[f'resp_{ch}'] = band_power(f, pxx, RESP_LO, RESP_HI)
            row[f'card_{ch}'] = band_power(f, pxx, CARD_LO, CARD_HI)
            row[f'sws_{ch}'] = band_power(f, pxx, SWS_LO, SWS_HI)
            row[f'inband_{ch}'] = band_power(f, pxx, COMB_LO, COMB_HI)
            # accumulate low-freq PSD for the median-PSD figure
            mff = f <= PSD_FMAX
            psd_accum[ch].append(pxx[mff])
            if psd_f is None:
                psd_f = f[mff]
            # comb flag from lookup
            key = (meta['label'], ch, j)
            row[f'comb_{ch}'] = bool(comb_lut.get(key, False))
        rows.append(row)
    df = pd.DataFrame(rows)
    # motion flag: top-decile acc within session
    thr = np.nanpercentile(df['acc_std'], 90)
    df['motion'] = df['acc_std'] > thr
    med_psd = {ch: np.median(np.array(psd_accum[ch]), axis=0) for ch in CHANS}
    return df, (psd_f, med_psd, meta['label'])


def build_comb_lut():
    """(session, channel, epoch) -> is harmonic-comb window, from ladder parquet."""
    lad = pd.read_parquet(SW / 'ladder_quantify' / 'per_window_channels.parquet')
    # epoch index is implicit by order per (session, channel); use t_hr rank instead
    lut = {}
    for (sess, ch), g in lad.groupby(['session', 'channel']):
        g = g.sort_values('t_hr').reset_index(drop=True)
        for j, isl in zip(g.index, g['combined_is_ladder'].astype(bool)):
            lut[(sess, ch, int(j))] = bool(isl)
    return lut, lad


def build_comb_lut_by_thr(lad):
    """Alt lookup keyed by (session, channel, rounded t_hr) for robust joining."""
    lut = {}
    for _, r in lad.iterrows():
        lut[(r['session'], r['channel'], round(float(r['t_hr']), 4))] = bool(r['combined_is_ladder'])
    return lut


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_psd(psd_f, med_psds, out):
    """Median PSD per channel, bands shaded — 'see the band power'."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for ch in CHANS:
        # median across sessions of the per-session median PSD
        arr = np.array([med_psds[ch][s] for s in range(len(med_psds[ch]))])
        m = np.median(arr, axis=0)
        ax.semilogy(psd_f, m, color=CH_COLOR[ch], lw=2.2, label=ch)
    for (lo, hi, c, name) in [(RESP_LO, RESP_HI, '#3498DB', 'respiratory'),
                              (SWS_LO, SWS_HI, '#2ECC71', 'SWS (delta-eq)'),
                              (CARD_LO, CARD_HI, '#E74C3C', 'cardiac')]:
        ax.axvspan(lo, hi, color=c, alpha=0.10)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], name, ha='center', va='top',
                fontsize=9, color=c, fontweight='bold')
    ax.set_xlim(0, PSD_FMAX)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel(f'PSD ({CAP_UNIT_SQ}/Hz)', fontsize=12)
    ax.set_title('B. Median SEC power spectral density with physiological bands shaded\n'
                 '(cardiac & SWS bands overlap; harmonic combs sit as quasi-integer peaks in 0.1–1 Hz)',
                 fontsize=13, fontweight='bold')
    ax.legend(title='channel', fontsize=11)
    ax.grid(True, which='both', alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def fig_band_bars(df, out):
    """Band power (fF^2) per category x channel; SWS over N3; comb vs non-comb."""
    sleep = df[(~df['motion']) & (df['stage_code'].isin([0, 1, 2, 3]))]
    n3 = df[(~df['motion']) & (df['stage_code'] == 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4),
                                   gridspec_kw={'width_ratios': [1.4, 1]})

    cats = [('respiratory\n(0.1–0.5 Hz)', 'resp', sleep),
            ('cardiac\n(0.5–3 Hz)', 'card', sleep),
            ('SWS band, N3\n(0.5–2 Hz)', 'sws', n3)]
    x = np.arange(len(cats)); w = 0.26
    for k, ch in enumerate(CHANS):
        vals = [d[f'{pref}_{ch}'].median() for (_, pref, d) in cats]
        ax1.bar(x + (k - 1) * w, vals, w, color=CH_COLOR[ch], label=ch, alpha=0.9)
    ax1.set_yscale('log')
    ax1.set_xticks(x); ax1.set_xticklabels([c[0] for c in cats], fontsize=10)
    ax1.set_ylabel(f'median band power ({CAP_UNIT_SQ})', fontsize=11)
    ax1.set_title('Band power per feature and channel', fontsize=12, fontweight='bold')
    ax1.legend(title='channel', fontsize=10); ax1.grid(True, axis='y', alpha=0.15)

    # comb vs non-comb in-band power (pool channels where combs occur: CRE, CH)
    labels, comb_v, noncomb_v, cols = [], [], [], []
    for ch in CHANS:
        d = df[(~df['motion']) & (df['stage_code'].isin([0, 1, 2, 3]))]
        c = d[d[f'comb_{ch}']][f'inband_{ch}']
        nc = d[~d[f'comb_{ch}']][f'inband_{ch}']
        if len(c) >= 3:
            labels.append(ch); comb_v.append(c.median()); noncomb_v.append(nc.median())
            cols.append(CH_COLOR[ch])
    xx = np.arange(len(labels))
    ax2.bar(xx - 0.2, comb_v, 0.4, color=cols, alpha=0.95, label='comb epochs')
    ax2.bar(xx + 0.2, noncomb_v, 0.4, color=cols, alpha=0.4, label='non-comb NREM')
    ax2.set_yscale('log')
    ax2.set_xticks(xx); ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel(f'median in-band power 0.1–3 Hz ({CAP_UNIT_SQ})', fontsize=10)
    ax2.set_title('Harmonic-comb events carry more power', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, axis='y', alpha=0.15)

    fig.suptitle('B. SEC band-power comparison — respiratory · cardiac · SWS · harmonic combs',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def main():
    print('=' * 66)
    print('B. Band-power comparison: respiratory / cardiac / SWS / harmonic combs')
    print('=' * 66)
    _, lad = build_comb_lut()
    comb_lut = build_comb_lut_by_thr(lad)

    all_df, psd_f = [], None
    med_psds = {ch: [] for ch in CHANS}
    for idx in range(len(SESSION_META)):
        # use t_hr-keyed comb lookup
        df, psd = process(idx, {})
        if df is None:
            continue
        # attach comb flags via t_hr-rounded lookup
        for ch in CHANS:
            df[f'comb_{ch}'] = [comb_lut.get((r.session, ch, round(float(r.t_hr), 4)), False)
                                for r in df.itertuples()]
        all_df.append(df)
        f, med, lbl = psd
        psd_f = f
        for ch in CHANS:
            med_psds[ch].append(med[ch])
    df = pd.concat(all_df, ignore_index=True)
    df.to_parquet(REPORT_DIR / 'band_power_epochs.parquet')

    # summary table
    sleep = df[(~df['motion']) & (df['stage_code'].isin([0, 1, 2, 3]))]
    n3 = df[(~df['motion']) & (df['stage_code'] == 1)]
    summ = []
    for ch in CHANS:
        summ.append({'channel': ch,
                     'resp_power': sleep[f'resp_{ch}'].median(),
                     'card_power': sleep[f'card_{ch}'].median(),
                     'sws_power_N3': n3[f'sws_{ch}'].median(),
                     'comb_inband': sleep[sleep[f'comb_{ch}']][f'inband_{ch}'].median(),
                     'noncomb_inband': sleep[~sleep[f'comb_{ch}']][f'inband_{ch}'].median()})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(REPORT_DIR / 'band_power_summary.csv', index=False)
    print(f'\nMedian band power ({CAP_UNIT_SQ}), motion-free sleep epochs:')
    print(sdf.to_string(index=False, float_format=lambda v: f'{v:.3g}'))

    print('\nWriting figures...')
    fig_psd(psd_f, med_psds, PLOT_DIR / 'B_psd_bands.png')
    fig_band_bars(df, PLOT_DIR / 'B_band_power_bars.png')
    print('\nDone.')


if __name__ == '__main__':
    main()
