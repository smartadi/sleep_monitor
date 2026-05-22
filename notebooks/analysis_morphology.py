# -*- coding: utf-8 -*-
"""
Morphological cluster pipeline -- end-to-end analysis script.

Produces two figures:
  morphology_signal.png     -- signal overview + cluster annotations
  morphology_validation.png -- rate timeseries + Bland-Altman plots

Rate estimation strategy (two tracks):
  Primary   : ACF of band-filtered CAP signal (decoupled from event detection)
  Secondary : event counting with adaptive divisor (morphological approach)
Both are compared against PSG GT.

Usage:
  python notebooks/analysis_morphology.py
  python notebooks/analysis_morphology.py --subject OS002 --night 1 --start 3.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os; os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib; matplotlib.use('Agg')

from sleep_monitor import SESSION_META, load_session
from sleep_monitor.morphology import (
    MorphCfg, preprocess_diff, run_pipeline,
    acf_rates_from_cap, band_events_to_rates, events_to_rates,
    gt_event_rates, gt_event_times_peaks,
    bland_altman, event_summary,
)

# ============================================================
#   PARAMETERS
# ============================================================
SUBJECT     = 'OS001'
NIGHT       = 1
START_HR    = 2.0          # window start (hours)
WIN_MIN     = 10.0         # window length (minutes)
ZOOM_S      = 30.0         # zoom panel duration (seconds)
ACC_REMOVAL = True

WIN_RATE_S  = 30.0         # sliding-window length for rate
STEP_RATE_S = 5.0          # sliding-window step

CLR = {
    'resp':    '#27AE60',   # green
    'cardiac': '#E74C3C',   # red
    'other':   '#AAAAAA',   # grey
    'gt_resp': '#2980B9',   # blue
    'gt_card': '#8E44AD',   # purple
    'acf_cap': '#E67E22',   # orange  (ACF primary)
    'morph':   '#1ABC9C',   # teal    (morphological secondary)
    'disp':    '#555555',   # dark grey (display signal)
}
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description='Morphological cluster pipeline')
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--night',   type=int, default=NIGHT, choices=[1, 2])
    p.add_argument('--start',   type=float, default=START_HR)
    p.add_argument('--win-min', type=float, default=WIN_MIN)
    return p.parse_args()


def _zsc(x): return (x - x.mean()) / (x.std() + 1e-12)


def _ba_panel(ax, means, diffs, stats, color, unit, title):
    """Draw one Bland-Altman panel."""
    if len(means) >= 3:
        ax.scatter(means, diffs, color=color, s=28, alpha=0.7, edgecolors='none')
        ax.axhline(stats['bias'],   color='black', lw=1.5, ls='-',
                   label=f"Bias = {stats['bias']:+.2f}")
        ax.axhline(stats['loa_hi'], color='grey',  lw=1.2, ls='--',
                   label=f"LoA [{stats['loa_lo']:+.2f}, {stats['loa_hi']:+.2f}]")
        ax.axhline(stats['loa_lo'], color='grey',  lw=1.2, ls='--')
        ax.axhline(0, color='black', lw=0.6, alpha=0.3)
        ax.fill_between([means.min(), means.max()],
                        stats['loa_lo'], stats['loa_hi'],
                        alpha=0.08, color=color)
        ax.legend(fontsize=7.5, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='grey')
    ax.set_xlabel(f'Mean of CAP & GT  ({unit})', fontsize=8)
    ax.set_ylabel(f'CAP - GT  ({unit})', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)


def main():
    args    = parse_args()
    win_hr  = args.win_min / 60.0

    # -- Load ----------------------------------------------------------------
    meta    = next(m for m in SESSION_META
                   if m['subject'] == args.subject and m['night'] == args.night)
    session = load_session(meta)
    print(f'\nSession : {meta["label"]}  {meta["subject"]}-{meta["initials"]}  {meta["date"]}')
    print(f'Window  : {args.start:.2f}-{args.start+win_hr:.3f} hr  ({args.win_min:.0f} min)')

    # -- Preprocess ----------------------------------------------------------
    cfg = MorphCfg(win_s=WIN_RATE_S, step_s=STEP_RATE_S)
    t_s, sig_disp, extras = preprocess_diff(session, args.start, win_hr, cfg, ACC_REMOVAL)
    t_total = float(t_s[-1])
    fs      = session.fs
    print(f'Signal  : {len(t_s):,} samples  ({t_total:.1f} s)')

    # -- Morphological event detection ----------------------------------------
    events = run_pipeline(sig_disp, t_s, fs, cfg, extras)
    event_summary(events, t_total)

    # -- PRIMARY rates: ACF of band-filtered CAP signals ----------------------
    (cap_rr_t, cap_rr_hz), (cap_hr_t, cap_hr_hz) = acf_rates_from_cap(
        extras['sig_resp'], extras['sig_card'], fs, WIN_RATE_S, STEP_RATE_S)

    # -- SECONDARY rates: event counting with adaptive divisor ----------------
    ev_rr_t, ev_rr_hz, rr_div = band_events_to_rates(
        events, 'resp',    t_total, extras['sig_resp'], fs, WIN_RATE_S, STEP_RATE_S)
    ev_hr_t, ev_hr_hz, hr_div = band_events_to_rates(
        events, 'cardiac', t_total, extras['sig_card'], fs, WIN_RATE_S, STEP_RATE_S)
    print(f'\nAdaptive divisors: resp={rr_div}  cardiac={hr_div}')

    # Rate from CLASSIFIED events only
    rr_cls_t, rr_cls_hz = events_to_rates(events, 'resp',    t_total, WIN_RATE_S, STEP_RATE_S)
    hr_cls_t, hr_cls_hz = events_to_rates(events, 'cardiac', t_total, WIN_RATE_S, STEP_RATE_S)

    # -- GT rates ---------------------------------------------------------------
    gt_rr_t, gt_rr_hz = gt_event_rates(extras['gt_thorax_raw'], fs, 'resp',    WIN_RATE_S, STEP_RATE_S)
    gt_hr_t, gt_hr_hz = gt_event_rates(extras['gt_pleth_raw'],  fs, 'cardiac', WIN_RATE_S, STEP_RATE_S)

    gt_resp_times = gt_event_times_peaks(extras['gt_thorax_raw'], fs, 'resp')
    gt_card_times = gt_event_times_peaks(extras['gt_pleth_raw'],  fs, 'cardiac')

    # -- Bland-Altman ---------------------------------------------------------
    # Primary (ACF-CAP vs GT)
    ba_rr_m,  ba_rr_d,  ba_rr_st  = bland_altman(cap_rr_t, cap_rr_hz, gt_rr_t, gt_rr_hz, 60.)
    ba_hr_m,  ba_hr_d,  ba_hr_st  = bland_altman(cap_hr_t, cap_hr_hz, gt_hr_t, gt_hr_hz, 60.)
    # Secondary (morphological event-based vs GT)
    ba_mrr_m, ba_mrr_d, ba_mrr_st = bland_altman(ev_rr_t, ev_rr_hz, gt_rr_t, gt_rr_hz, 60.)
    ba_mhr_m, ba_mhr_d, ba_mhr_st = bland_altman(ev_hr_t, ev_hr_hz, gt_hr_t, gt_hr_hz, 60.)

    # -- Print validation table -----------------------------------------------
    print('\n-- Validation summary ---------------------------------')
    print(f'{"Method":22} {"n":>5} {"MAE":>8} {"Bias":>8} {"r":>7}')
    print('-' * 50)
    for lbl, st in [
        ('RR ACF-CAP (br/min)',  ba_rr_st),
        ('RR morph-event',       ba_mrr_st),
        ('HR ACF-CAP (BPM)',     ba_hr_st),
        ('HR morph-event',       ba_mhr_st),
    ]:
        if st['n'] == 0:
            print(f'{lbl:22}   -- insufficient data')
        else:
            print(f'{lbl:22} {st["n"]:>5} {st["mae"]:>8.2f} {st["bias"]:>+8.2f} {st["r"]:>7.3f}')

    # =========================================================================
    #  FIGURE 1: Signal overview + cluster annotations
    # =========================================================================
    fig1 = plt.figure(figsize=(18, 16))
    gs1  = gridspec.GridSpec(3, 1, figure=fig1, hspace=0.55,
                              height_ratios=[1.8, 1.8, 2.8])

    def _full_band_ax(ax, sig, lbl, color, band_name):
        sig_z = _zsc(sig)
        ax.plot(t_s, sig_z, color=color, lw=0.5, alpha=0.6, label=lbl)
        seen = set()
        for ev in events:
            if ev.band != band_name:
                continue
            col  = CLR[ev.kind]
            lbl2 = ev.kind if ev.kind not in seen else None
            ax.axvspan(t_s[ev.peak_indices[0]] - 0.05,
                       t_s[ev.peak_indices[0]] + 0.05,
                       alpha=0.35, color=col, label=lbl2)
            seen.add(ev.kind)
            ax.plot(t_s[ev.peak_indices[0]], sig_z[ev.peak_indices[0]],
                    'v', color=col, ms=3, markeredgewidth=0, alpha=0.8)
        ax.set_xlim(t_s[0], t_s[-1])
        ax.grid(True, alpha=0.25)
        ax.set_ylabel('Amp (z)', fontsize=8)
        return sig_z

    # Row 0: resp band
    ax_r = fig1.add_subplot(gs1[0])
    _full_band_ax(ax_r, extras['sig_resp'], 'CLE-CRE resp band', CLR['resp'], 'resp')
    for rt in gt_resp_times[(gt_resp_times >= 0) & (gt_resp_times <= t_total)]:
        ax_r.axvline(rt, ymin=0, ymax=0.07, color=CLR['gt_resp'], lw=1.0, alpha=0.7)
    n_resp_cls = sum(e.kind == 'resp' for e in events)
    n_resp_all = sum(e.band == 'resp' for e in events)
    ax_r.set_title(
        f'Resp band (0.1-0.5 Hz)  |  '
        f'events: {n_resp_all}  |  classified "resp" (n={cfg.resp_n_min}-{cfg.resp_n_max}): {n_resp_cls}  '
        f'|  ticks = GT breath events',
        fontsize=8.5, fontweight='bold')
    ax_r.legend(fontsize=7, loc='upper right', ncol=4)

    # Row 1: cardiac band
    ax_c = fig1.add_subplot(gs1[1], sharex=ax_r)
    _full_band_ax(ax_c, extras['sig_card'], 'CLE-CRE cardiac band', CLR['cardiac'], 'cardiac')
    for ct in gt_card_times[(gt_card_times >= 0) & (gt_card_times <= t_total)]:
        ax_c.axvline(ct, ymin=0, ymax=0.07, color=CLR['gt_card'], lw=0.5, alpha=0.4)
    n_card_cls = sum(e.kind == 'cardiac' for e in events)
    n_card_all = sum(e.band == 'cardiac' for e in events)
    ax_c.set_title(
        f'Cardiac band (0.5-3.0 Hz)  |  '
        f'events: {n_card_all}  |  classified "cardiac" (n={cfg.card_n_min}-{cfg.card_n_max}): {n_card_cls}  '
        f'|  ticks = GT beat events',
        fontsize=8.5, fontweight='bold')
    ax_c.legend(fontsize=7, loc='upper right', ncol=4)
    ax_c.set_xlabel('Time (s)', fontsize=8)

    # Row 2: zoom panels (3 columns)
    zoom_lo = min(ZOOM_S, t_total * 0.2)
    zoom_hi = zoom_lo + ZOOM_S
    zm      = (t_s >= zoom_lo) & (t_s <= zoom_hi)
    t_zm    = t_s[zm]

    gs1b = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs1[2], wspace=0.35)
    axes_zm = [fig1.add_subplot(gs1b[i]) for i in range(3)]

    zoom_specs = [
        (extras['sig_resp'],  CLR['resp'],    'resp',    'Resp band'),
        (extras['sig_card'],  CLR['cardiac'], 'cardiac', 'Cardiac band'),
        (extras['gt_resp'],   CLR['gt_resp'], None,      'GT Thorax (resp)'),
    ]
    for ax_z, (sig, color, bnd, title) in zip(axes_zm, zoom_specs):
        sig_z = _zsc(sig[zm])
        ax_z.plot(t_zm, sig_z, color=color, lw=1.1, alpha=0.85)
        ax_z.axhline(0, color='grey', lw=0.4)

        if bnd is not None:
            for ev in events:
                if ev.band != bnd or ev.center_s < zoom_lo or ev.center_s > zoom_hi:
                    continue
                col = CLR[ev.kind]
                pi  = ev.peak_indices[0]
                if zm[pi]:
                    ax_z.plot(t_s[pi], sig_z[np.searchsorted(t_zm, t_s[pi])],
                              'v', color=col, ms=9, zorder=6,
                              markeredgecolor='white', markeredgewidth=0.6)
                    ax_z.text(t_s[pi], sig_z.max() * 0.88,
                              f'n={ev.n_subpeaks}|{ev.kind[0]}',
                              ha='center', va='top', fontsize=5.5,
                              color=col, fontweight='bold', clip_on=True)
        else:
            # GT
            for rt in gt_resp_times[(gt_resp_times >= zoom_lo) & (gt_resp_times <= zoom_hi)]:
                ax_z.axvline(rt, color=color, lw=1.2, ls='--', alpha=0.7)

        ax_z.set_title(f'{title}  {zoom_lo:.0f}-{zoom_hi:.0f} s', fontsize=8)
        ax_z.set_xlabel('Time (s)', fontsize=7)
        ax_z.set_ylabel('Amp (z)', fontsize=7)
        ax_z.set_xlim(zoom_lo, zoom_hi)
        ax_z.grid(True, alpha=0.25)

    fig1.suptitle(
        f'Morphological Pipeline  |  {meta["label"]} {meta["subject"]}-{meta["initials"]}  '
        f'{meta["date"]}  |  {args.start:.2f}-{args.start+win_hr:.3f} hr\n'
        f'resp classified: {n_resp_cls}/{n_resp_all}  |  '
        f'cardiac classified: {n_card_cls}/{n_card_all}',
        fontsize=10, fontweight='bold', y=1.01)

    out1 = Path(__file__).parent / 'plots' / 'morphology_signal.png'
    out1.parent.mkdir(exist_ok=True)
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f'\nSaved -> {out1}')

    # =========================================================================
    #  FIGURE 2: Rate timeseries + Bland-Altman (2x4 grid)
    # =========================================================================
    fig2 = plt.figure(figsize=(18, 10))
    gs2  = gridspec.GridSpec(2, 4, figure=fig2, hspace=0.50, wspace=0.38)

    # -- Row 0: rate timeseries ------------------------------------------------
    def _rate_ax(ax, gt_t, gt_hz, pri_t, pri_hz, sec_t, sec_hz,
                  cls_t, cls_hz, unit, title):
        ax.plot(gt_t, gt_hz * 60, color=CLR['gt_resp'] if 'br' in unit else CLR['gt_card'],
                lw=2.0, alpha=0.9, label='GT ACF', zorder=4)
        v = ~np.isnan(pri_hz)
        ax.plot(pri_t[v], pri_hz[v] * 60, color=CLR['acf_cap'],
                lw=1.4, alpha=0.85, label='CAP ACF (primary)', zorder=3)
        v2 = ~np.isnan(sec_hz)
        if v2.any():
            ax.plot(sec_t[v2], sec_hz[v2] * 60, color=CLR['morph'],
                    lw=1.2, alpha=0.75, ls='--', label='CAP morph-all (secondary)')
        v3 = ~np.isnan(cls_hz)
        if v3.any():
            ax.plot(cls_t[v3], cls_hz[v3] * 60, color=CLR['morph'],
                    lw=0.9, alpha=0.55, ls=':', label='CAP morph-classified')
        ax.set_ylabel(unit, fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.5, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.28)

    ax_rr = fig2.add_subplot(gs2[0, :2])
    _rate_ax(ax_rr,
             gt_rr_t, gt_rr_hz,
             cap_rr_t, cap_rr_hz,
             ev_rr_t,  ev_rr_hz,
             rr_cls_t, rr_cls_hz,
             'Resp rate (br/min)', 'Respiratory Rate')
    if ba_rr_st['n']:
        ax_rr.text(0.01, 0.97,
                   f"ACF: MAE={ba_rr_st['mae']:.2f}  r={ba_rr_st['r']:.3f}",
                   transform=ax_rr.transAxes, va='top', fontsize=7.5,
                   bbox=dict(facecolor='white', alpha=0.8, pad=3, edgecolor='none'))

    ax_hr = fig2.add_subplot(gs2[0, 2:])
    _rate_ax(ax_hr,
             gt_hr_t, gt_hr_hz,
             cap_hr_t, cap_hr_hz,
             ev_hr_t,  ev_hr_hz,
             hr_cls_t, hr_cls_hz,
             'Heart rate (BPM)', 'Heart Rate')
    if ba_hr_st['n']:
        ax_hr.text(0.01, 0.97,
                   f"ACF: MAE={ba_hr_st['mae']:.2f}  r={ba_hr_st['r']:.3f}",
                   transform=ax_hr.transAxes, va='top', fontsize=7.5,
                   bbox=dict(facecolor='white', alpha=0.8, pad=3, edgecolor='none'))

    # -- Row 1: Bland-Altman panels -------------------------------------------
    _ba_panel(fig2.add_subplot(gs2[1, 0]),
              ba_rr_m, ba_rr_d, ba_rr_st, CLR['acf_cap'],
              'br/min', 'BA -- RR  (ACF-CAP vs GT)')
    _ba_panel(fig2.add_subplot(gs2[1, 1]),
              ba_mrr_m, ba_mrr_d, ba_mrr_st, CLR['morph'],
              'br/min', 'BA -- RR  (morph vs GT)')
    _ba_panel(fig2.add_subplot(gs2[1, 2]),
              ba_hr_m, ba_hr_d, ba_hr_st, CLR['acf_cap'],
              'BPM', 'BA -- HR  (ACF-CAP vs GT)')
    _ba_panel(fig2.add_subplot(gs2[1, 3]),
              ba_mhr_m, ba_mhr_d, ba_mhr_st, CLR['morph'],
              'BPM', 'BA -- HR  (morph vs GT)')

    fig2.suptitle(
        f'Rate Validation  |  {meta["label"]} {meta["subject"]}  '
        f'|  {args.start:.2f}-{args.start+win_hr:.3f} hr  '
        f'| win={WIN_RATE_S:.0f}s  step={STEP_RATE_S:.0f}s  '
        f'| resp div={rr_div}  cardiac div={hr_div}',
        fontsize=10, fontweight='bold')

    out2 = Path(__file__).parent / 'plots' / 'morphology_validation.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f'Saved -> {out2}')
    print('Done.')


if __name__ == '__main__':
    main()
