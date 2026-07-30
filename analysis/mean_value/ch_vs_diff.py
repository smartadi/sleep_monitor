"""
Is the hardware CH channel the same quantity as the arithmetic CLE − CRE?

Motivation
----------
The source paper describes CH as the INTERHEMISPHERIC difference — "the
difference between left and right hemispheres", positive meaning higher regional
ICP on the left. Taken at face value that makes CH and (CLE − CRE) the same
physical quantity, so the mean-value analyses ought to be able to use either.

They are not the same. The front end reads FOUR sensors into THREE signals, and
the paper never writes CH = CLE − CRE. This script quantifies the discrepancy in
the mean-value (slow / DC) domain the professor's directive cares about, and asks
*where* in frequency the two channels agree, so we know which one carries the
intracranial-fluid signal and which is picking up something else.

Questions answered, per session and pooled
------------------------------------------
1. Offset      mean CH − mean(CLE−CRE), in fF. If CH were the arithmetic
               difference this is 0.
2. Gain        OLS slope of CH on (CLE−CRE) on the 10 s means. Would be 1.
3. Agreement   Pearson r of the 10 s means (level), and of the 1-min change
               (Δ, offset- and trend-robust — the honest number, because two
               slowly drifting traces correlate even when unrelated).
4. Where       Magnitude-squared coherence across 0.002–3 Hz, so we can see
               whether they agree in the respiratory/cardiac bands but diverge
               in the slow drift band (or the reverse).
5. Why         Is agreement driven by shared MOTION artifact rather than shared
               physiology? Per-session r is regressed on per-session motion, and
               r is recomputed on motion-quiet blocks only.
6. Staging     Per-stage session-centred mean of each channel — do the two rank
               the sleep stages the same way, and with the same sign?

Units: femtofarads throughout (see CAP_SCALE_TO_FF in sleep_monitor/config.py).

Outputs
-------
    reports/mean_value/ch_vs_diff_per_session.csv
    reports/mean_value/ch_vs_diff_coherence.csv
    reports/mean_value/ch_vs_diff_by_stage.csv
    writeup/figures/channel_evolution/ch_vs_diff_summary.png
    writeup/figures/channel_evolution/ch_vs_diff_trace_<SESSION>.png

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/ch_vs_diff.py
    .venv/Scripts/python.exe analysis/mean_value/ch_vs_diff.py --trace S1N1 S6N2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, coherence
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, CAP_SCALE_TO_FF, CAP_UNIT,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI,
)
from sleep_monitor.sessions import SESSION_META

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'mean_value'
PLOT_DIR = ROOT / 'writeup' / 'figures' / 'channel_evolution'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SEC = 10.0        # mean-value block (matches channel_evolution.py)
DELTA_MIN = 1.0         # lag for the change-based (Δ) correlation
SLOW_HI = 0.01          # Hz, top of the "slow drift" band
COH_FS = 10.0           # Hz, decimation rate for the coherence estimate
LADDER_ORDER = [4, 3, 2, 1, 0]      # Wake, N1, N2, N3, REM — sleep-depth order
MOTION_QUIET_PCT = 50   # blocks below this motion percentile count as "quiet"


def block_reduce(x, n, fn):
    m = len(x) // n
    return fn(x[: m * n].reshape(m, n), axis=1)


def bandpass(x, fs, lo, hi):
    sos = butter(4, [lo / (0.5 * fs), hi / (0.5 * fs)], btype='band', output='sos')
    return sosfiltfilt(sos, x)


def band_corr(a, b, fs, lo, hi):
    """Correlation restricted to a frequency band (both traces band-passed)."""
    if hi >= 0.5 * fs:
        return np.nan
    fa, fb = bandpass(a, fs, lo, hi), bandpass(b, fs, lo, hi)
    return float(np.corrcoef(fa, fb)[0, 1])


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
        return np.nan
    return float(np.corrcoef(a[good], b[good])[0, 1])


def analyse_session(idx, meta):
    s = load_session(idx)
    fs = s.fs
    cle = s.cap['CLE'].astype(np.float64) * CAP_SCALE_TO_FF
    cre = s.cap['CRE'].astype(np.float64) * CAP_SCALE_TO_FF
    ch = s.cap['CH'].astype(np.float64) * CAP_SCALE_TO_FF
    acc = s.cap['acc_mag'].astype(np.float64)
    diff = cle - cre

    # == mean-value domain: 10 s block means ==
    n = int(round(fs * BLOCK_SEC))
    m = min(len(ch), len(diff), len(acc)) // n
    t_hr = (np.arange(m) + 0.5) * BLOCK_SEC / 3600.0
    ch_b = block_reduce(ch[: m * n], n, np.mean)
    df_b = block_reduce(diff[: m * n], n, np.mean)
    mot_b = block_reduce(acc[: m * n], n, np.std)

    offset = float(np.mean(ch_b) - np.mean(df_b))
    slope, intercept = np.polyfit(df_b, ch_b, 1)
    r_level = safe_corr(ch_b, df_b)

    # Δ-correlation: change over DELTA_MIN, immune to offset and slow trend
    lag = max(1, int(round(DELTA_MIN * 60 / BLOCK_SEC)))
    r_delta = safe_corr(ch_b[lag:] - ch_b[:-lag], df_b[lag:] - df_b[:-lag])

    # motion-quiet subset (are they only agreeing when the subject moves?)
    thr = np.percentile(mot_b, MOTION_QUIET_PCT)
    quiet = mot_b <= thr
    r_level_quiet = safe_corr(ch_b[quiet], df_b[quiet])

    # == band-resolved agreement at full rate ==
    r_slow = band_corr(ch_b, df_b, 1.0 / BLOCK_SEC, 0.0008, SLOW_HI)
    nfull = min(len(ch), len(diff))
    r_resp = band_corr(ch[:nfull], diff[:nfull], fs, RESP_LO, RESP_HI)
    r_card = band_corr(ch[:nfull], diff[:nfull], fs, CARD_LO, CARD_HI)

    # == coherence on a 10 Hz decimated pair ==
    q = int(round(fs / COH_FS))
    md = nfull // q
    ch_d = block_reduce(ch[: md * q], q, np.mean)
    df_d = block_reduce(diff[: md * q], q, np.mean)
    fco, cxy = coherence(ch_d, df_d, fs=COH_FS, nperseg=int(COH_FS * 600),
                         noverlap=int(COH_FS * 300))

    def band_coh(lo, hi):
        sel = (fco >= lo) & (fco <= hi)
        return float(np.nanmean(cxy[sel])) if sel.any() else np.nan

    row = dict(
        session=meta['label'], subject=meta['label'][:2], night=meta['label'][2:],
        mean_CH=float(np.mean(ch_b)), mean_DIFF=float(np.mean(df_b)),
        sd_CH=float(np.std(ch_b)), sd_DIFF=float(np.std(df_b)),
        offset_fF=offset, slope=float(slope), intercept_fF=float(intercept),
        r_level=r_level, r_level_quiet=r_level_quiet, r_delta=r_delta,
        r_slow=r_slow, r_resp=r_resp, r_card=r_card,
        coh_slow=band_coh(0.002, SLOW_HI), coh_resp=band_coh(RESP_LO, RESP_HI),
        coh_card=band_coh(CARD_LO, CARD_HI),
        motion_median=float(np.median(mot_b)), motion_p95=float(np.percentile(mot_b, 95)),
    )

    # == per-stage, session-centred (subtract each channel's own session median) ==
    stage_rows = []
    sp = load_sleep_profile(s)
    if sp is not None:
        codes = np.asarray(sp['codes'], int)
        t_ep = np.asarray(sp['t_ep_hr'], float)
        ep_of_block = np.searchsorted(t_ep, t_hr, side='right') - 1
        ok = (ep_of_block >= 0) & (ep_of_block < len(codes))
        cb, dbb = ch_b - np.median(ch_b), df_b - np.median(df_b)
        for c in LADDER_ORDER:
            sel = ok & (codes[np.clip(ep_of_block, 0, len(codes) - 1)] == c)
            if sel.sum() < 10:
                continue
            stage_rows.append(dict(
                session=meta['label'], stage=STAGE_LABELS[c], stage_code=c,
                n_blocks=int(sel.sum()),
                CH_centred_fF=float(np.median(cb[sel])),
                DIFF_centred_fF=float(np.median(dbb[sel]))))

    coh_df = pd.DataFrame(dict(session=meta['label'], freq=fco, coherence=cxy))
    trace = dict(t_hr=t_hr, ch=ch_b, diff=df_b, motion=mot_b, sp=sp)
    return row, stage_rows, coh_df, trace


def figure_summary(per, coh, stage, out):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    # A — offset per session (would be 0 if CH == CLE-CRE)
    ax = axes[0, 0]
    ax.barh(per['session'], per['offset_fF'], color='#C0392B', alpha=0.85)
    ax.axvline(0, color='k', lw=1.0)
    ax.set_xlabel(f'CH − (CLE−CRE)  ({CAP_UNIT})')
    ax.set_title('A  Static offset\n(0 if CH were the arithmetic difference)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # B — gain
    ax = axes[0, 1]
    ax.barh(per['session'], per['slope'], color='#2980B9', alpha=0.85)
    ax.axvline(1, color='#27AE60', lw=1.4, ls='--', label='slope = 1 (identical)')
    ax.axvline(0, color='k', lw=1.0)
    ax.set_xlabel('OLS slope of CH on (CLE−CRE)')
    ax.set_title('B  Gain', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # C — agreement: level vs change vs motion-quiet
    ax = axes[0, 2]
    y = np.arange(len(per)); h = 0.27
    ax.barh(y - h, per['r_level'], h, label='level (10 s means)', color='#8E44AD', alpha=0.85)
    ax.barh(y, per['r_level_quiet'], h, label='level, motion-quiet half', color='#F39C12', alpha=0.85)
    ax.barh(y + h, per['r_delta'], h, label=f'Δ over {DELTA_MIN:g} min', color='#16A085', alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(per['session'])
    ax.axvline(0, color='k', lw=1.0); ax.set_xlim(-1, 1)
    ax.set_xlabel('Pearson r'); ax.legend(fontsize=8)
    ax.set_title('C  Agreement in the mean-value domain', fontsize=11,
                 fontweight='bold', loc='left')
    ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # D — coherence spectrum
    ax = axes[1, 0]
    piv = coh.pivot_table(index='freq', columns='session', values='coherence')
    fr = piv.index.values
    med = piv.median(axis=1).values
    q1, q3 = piv.quantile(0.25, axis=1).values, piv.quantile(0.75, axis=1).values
    ax.fill_between(fr, q1, q3, color='#2980B9', alpha=0.25, label='IQR across sessions')
    ax.plot(fr, med, color='#2980B9', lw=1.4, label='median')
    ax.axvspan(RESP_LO, RESP_HI, color='#27AE60', alpha=0.12)
    ax.axvspan(CARD_LO, CARD_HI, color='#E67E22', alpha=0.12)
    ax.text(np.sqrt(RESP_LO * RESP_HI), 0.94, 'resp', ha='center', fontsize=8.5, color='#1E8449')
    ax.text(np.sqrt(CARD_LO * CARD_HI), 0.94, 'cardiac', ha='center', fontsize=8.5, color='#B9770E')
    ax.set_xscale('log'); ax.set_xlim(0.002, 4); ax.set_ylim(0, 1)
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude-squared coherence')
    ax.set_title('D  Where the two channels agree', fontsize=11,
                 fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.2)

    # E — is agreement just shared motion / coupling swing?
    ax = axes[1, 1]
    ax.scatter(per['sd_CH'], per['r_level'], s=70, color='#C0392B',
               edgecolor='k', linewidth=0.5, zorder=3)
    for _, r in per.iterrows():
        ax.annotate(r['session'], (r['sd_CH'], r['r_level']), fontsize=7.5,
                    xytext=(4, 3), textcoords='offset points')
    rho, p = spearmanr(per['sd_CH'], per['r_level'])
    ax.axhline(0, color='k', lw=1.0)
    ax.set_xscale('log')
    ax.set_xlabel(f'CH variability over the night — sd of 10 s means ({CAP_UNIT}, log)')
    ax.set_ylabel('r (level)')
    ax.set_title(f'E  Agreement vs channel excursion\nSpearman ρ={rho:.2f}, p={p:.3f}',
                 fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.2)

    # F — per-stage session-centred level, both channels
    ax = axes[1, 2]
    order = [STAGE_LABELS[c] for c in LADDER_ORDER]
    xs = np.arange(len(order)); w = 0.36
    for k, (colname, lbl, colr) in enumerate([
            ('CH_centred_fF', 'CH (hardware)', '#2980B9'),
            ('DIFF_centred_fF', 'CLE−CRE (arithmetic)', '#E67E22')]):
        vals = [stage.loc[stage['stage'] == st, colname].values for st in order]
        pos = xs + (k - 0.5) * w
        ax.boxplot(vals, positions=pos, widths=w * 0.85, patch_artist=True,
                   boxprops=dict(facecolor=colr, alpha=0.55, lw=0.8),
                   medianprops=dict(color='k', lw=1.3), showfliers=False,
                   whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
        ax.plot([], [], color=colr, lw=6, alpha=0.55, label=lbl)
    ax.axhline(0, color='k', lw=1.0, ls=':')
    ax.set_xticks(xs); ax.set_xticklabels(order)
    ax.set_ylabel(f'Session-centred level ({CAP_UNIT})')
    ax.set_title('F  Do they rank the stages the same way?', fontsize=11,
                 fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, axis='y', alpha=0.2)

    fig.suptitle('Hardware CH vs arithmetic CLE−CRE — the paper calls CH the '
                 'interhemispheric difference; the data says they are not the same signal',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def figure_trace(label, tr, out):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.0, 1.0, 0.6]})
    t = tr['t_hr']
    ax = axes[0]
    ax.plot(t, tr['ch'], lw=0.9, color='#2980B9', label='CH (hardware)')
    ax.set_ylabel(f'CH ({CAP_UNIT})', color='#2980B9')
    ax.tick_params(axis='y', labelcolor='#2980B9')
    ax2 = ax.twinx()
    ax2.plot(t, tr['diff'], lw=0.9, color='#E67E22', label='CLE−CRE (arithmetic)')
    ax2.set_ylabel(f'CLE−CRE ({CAP_UNIT})', color='#E67E22')
    ax2.tick_params(axis='y', labelcolor='#E67E22')
    ax.set_title(f'{label} — CH and CLE−CRE on independent axes '
                 '(note the different scales and offsets)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    z = lambda x: (x - np.median(x)) / (np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-9)
    ax.plot(t, z(tr['ch']), lw=0.9, color='#2980B9', label='CH (robust z)')
    ax.plot(t, z(tr['diff']), lw=0.9, color='#E67E22', label='CLE−CRE (robust z)')
    ax.set_ylabel('Robust z')
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.15)

    ax = axes[2]
    ax.fill_between(t, 0, tr['motion'], color='#7F8C8D', alpha=0.45)
    ax.set_ylabel('Motion\n(acc. std)'); ax.set_xlabel('Time (hours)')
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', nargs='*', default=['S1N1', 'S6N2'],
                    help='sessions to draw a side-by-side trace figure for')
    args = ap.parse_args()

    rows, stage_rows, cohs, traces = [], [], [], {}
    for i, meta in enumerate(SESSION_META):
        row, srows, coh, tr = analyse_session(i, meta)
        rows.append(row); stage_rows += srows; cohs.append(coh)
        if meta['label'] in args.trace:
            traces[meta['label']] = tr
        print(f"  {row['session']}  offset={row['offset_fF']:8.1f} fF  "
              f"slope={row['slope']:6.2f}  r_level={row['r_level']:6.2f}  "
              f"r_delta={row['r_delta']:6.2f}  coh_resp={row['coh_resp']:.2f}")

    per = pd.DataFrame(rows)
    stage = pd.DataFrame(stage_rows)
    coh = pd.concat(cohs, ignore_index=True)
    per.to_csv(REPORT_DIR / 'ch_vs_diff_per_session.csv', index=False)
    stage.to_csv(REPORT_DIR / 'ch_vs_diff_by_stage.csv', index=False)
    coh.to_csv(REPORT_DIR / 'ch_vs_diff_coherence.csv', index=False)

    figure_summary(per, coh, stage, PLOT_DIR / 'ch_vs_diff_summary.png')
    for lbl, tr in traces.items():
        figure_trace(lbl, tr, PLOT_DIR / f'ch_vs_diff_trace_{lbl}.png')

    print('\n== Pooled ==')
    print(f"offset  : median {per['offset_fF'].median():.1f} fF  "
          f"[{per['offset_fF'].min():.1f}, {per['offset_fF'].max():.1f}]")
    print(f"slope   : median {per['slope'].median():.2f}  "
          f"[{per['slope'].min():.2f}, {per['slope'].max():.2f}]")
    print(f"r_level : median {per['r_level'].median():.2f}  "
          f"[{per['r_level'].min():.2f}, {per['r_level'].max():.2f}]")
    print(f"r_delta : median {per['r_delta'].median():.2f}  "
          f"[{per['r_delta'].min():.2f}, {per['r_delta'].max():.2f}]")
    for b in ('coh_slow', 'coh_resp', 'coh_card'):
        print(f"{b:9s}: median {per[b].median():.3f}  "
              f"[{per[b].min():.3f}, {per[b].max():.3f}]")
    print(f"\nReports -> {REPORT_DIR}\nFigures -> {PLOT_DIR}")


if __name__ == '__main__':
    main()
