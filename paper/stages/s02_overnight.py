"""
Overnight SEC levels, head posture and the left-right imbalance (Methods §2.4, Results §3.1).

Manuscript items
    §3.1 ¶172   session-mean operating points of CLE, CRE, CH
    §3.1 ¶175   median within-night ranges 105 / 71 / 158 / 595 fF; CH > 1000 fF
    §3.1 ¶176   direction vs head turn (rho = 0.01, p = 0.62, 9,182 epochs);
                7,641 supine epochs, 3.5 reversals; magnitude 9.0 / 11.3 / 107.7 fF
                by head position
    §3.1 ¶177   mean |imbalance| 3.1-135.7 fF, burden 15-705 fF·h (46-fold),
                2.0-5.8-fold within subject, asymmetry -0.22..+0.09 (11 of 12 within
                ±0.09); "Fig S5c" claim (untraced)
    Fig 2       four nights (S6N2, S3N2, S4N2, S2N2): hypnogram, CLE-CRE about the
                session mean, head turn, 10-s blocks
    Fig S2      the signed imbalance marker within its ± envelope, all twelve nights
    Fig S4      CH against CLE-CRE, each about its session mean, all twelve nights
    Fig S6      integrated imbalance per recording (+ the S6 paragraph: 224 / 705 fF·h,
                cohort median 65, 15-199 without S6, left fraction 0.41-0.75)
    cache       CACHE_DIR/s02_overnight/imbalance_epochs.parquet (columns of the legacy
                reports/mean_value/imbalance_epochs.csv), session_levels.csv,
                imbalance_session.csv, imbalance_burden.csv, ch_vs_clecre_sessions.csv

Definitions (30-s epochs of s01_recordings unless stated)
    level       mean of the raw channel over the epoch, fF
    mu          mean of the level over the sleep period (first to last scored sleep
                epoch); every trace is drawn and analysed as level - mu
    d(t)        (CLE-CRE) - mu
    LP(d)       motion epochs masked, then a 2.5-min rolling MEDIAN (removes
                electrode re-seat spikes), then a 30-min rolling MEAN
    A(t)        the same cascade applied to |d| (magnitude, fF)
    D(t)        LP(d) / A, clipped to [-1, +1] (direction, + = left)
    reversal    sign change of D, ignoring the dead band |D| <= 0.2
    burden      sum over motion-free epochs of |LP(d)| x 30 s, fF·h
    asymmetry   (burden+ - burden-) / (burden+ + burden-)
    head turn   atan2(gY, gZ) of the 0.05-Hz gravity vector (seclib.motion.head_angle),
                circular median per 30-s block counted from the first sample, then
                linearly interpolated onto the epoch centres; head position by the
                ±45/±135 deg thresholds of seclib.motion.classify_head_position
    Fig 2 only  10-s blocks of the 10-Hz low-passed channel, sleep-period mean as zero

Ported from:
    analysis/mean_value/mean_value_vs_stage.py   extract_session (mean_<ch> columns only)
    analysis/mean_value/head_angle_validate.py   epoch_reduce, circ_median_deg, the
                                                 epoch series of analyse()
    analysis/mean_value/mean_centred_traces.py   sleep_mask, the session-mean table of main()
    analysis/mean_value/imbalance_marker.py      _roll, imbalance_marker, velocity, build,
                                                 summarise (paper columns), fig_vs_headangle
                                                 statistics (pooled rho, magnitude by position)
    analysis/mean_value/imbalance_burden.py      build, figure
    analysis/mean_value/ch_vs_clecre_sessions.py per_session, fig_grid
    analysis/mean_value/fig2_overnight_panels.py draw_panel
    analysis/mean_value/channel_evolution.py     compute_features (CLE-CRE + turn only),
                                                 block_reduce, lowpass, sleep_window,
                                                 sym_zero_ylim, draw_ladder, draw_head_row

Changes from the legacy code:
    * CORRECTION: imbalance_marker's docstring, figure title and console report called
      the low pass "a 30-min rolling median". The code -- which produced every number
      in ¶176-177 -- is a 2.5-min rolling median followed by a 30-min rolling mean.
      The computation is kept; the docstring and labels now say what it does.
    * CORRECTION (numbers only): "CH excursions exceeded 1000 fF in the three most
      mobile recordings" -- five recordings exceed 1000 fF; registered as DIFF.
    * CORRECTION (numbers only): the "Fig S5c" onset/end claim has no source in the
      repo; registered EXTERNAL/untraced. The text's figure references S5a/b/c should
      be S6a/b, and the head-angle result is cited as Fig S4 (S4 is CH vs CLE-CRE).
    * Fig S2 had no traceable source script; it is drawn here from the marker as its
      caption describes (signed LP(d) inside the ±A envelope).
    * The per-epoch features are computed on the shared s01 epoch grid instead of a
      private copy of it (same grid, same values).
    * Dropped: z-scores, VLF baselines, detrended features, LOSO AUCs, the velocity
      autocorrelation and tau-sweep summaries, head-angle calibration checks and the
      ch_vs_diff coherence columns -- none is quoted in the paper. The tau-swept
      marker columns are kept in the cache because the legacy epoch table carries them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

from seclib import CACHE_DIR, TAB_DIR, iter_sessions
from seclib.config import (CAP_COLORS, CAP_UNIT, STAGE_COLORS, STAGE_LABELS, STAGE_ORDER,
                           STAGE_ROW)
from seclib.figures import save
from seclib.loader import load_position, position_at
from seclib.motion import classify_head_position, head_angle
from seclib.numbers import Numbers

STAGE = 's02_overnight'
REQUIRES = ['s01_recordings']

EPOCH_MIN = 0.5
SLEEP_CODES = [0, 1, 2, 3]
LEVEL_CHANNELS = ['CLE', 'CRE', 'CH', 'CLE-CRE']
TAU_MIN = 30.0                    # night-scale rolling mean, minutes
DESPIKE_MIN = 2.5                 # rolling median that removes re-seat spikes, minutes
TAU_SWEEP = [5.0, 15.0, 30.0, 60.0]
DEAD_BAND = 0.2                   # |D| below this is neither left nor right
EPS = 1e-9
EPOCH_HOURS = 30.0 / 3600.0

# Fig 2: (panel, session, age). Ages are from Table 1; demographics are not in the data.
FIG2_PANELS = [('a', 'S6N2', 25), ('b', 'S3N2', 37), ('c', 'S4N2', 54), ('d', 'S2N2', 66)]
FIG2_BLOCK_SEC = 10.0
FIG2_LP_HZ = 10.0                 # low-pass before 10-s block means (removes electronic noise)
HEAD_MIN_SPAN_DEG = 25.0
CH_COLOR, DIFF_COLOR, HEAD_COLOR = '#2980B9', '#E67E22', '#16A085'
POS_COLOR, NEG_COLOR = '#C0392B', '#2980B9'
CLIP_SD = 5.0                     # Fig S4 clips each trace at 5 session SDs


# ── per-epoch levels and head angle ──────────────────────────────────────────

def epoch_levels(s, ep):
    """Mean raw level of each channel over every epoch of the s01 grid, fF."""
    raw = {c: s.cap[c].astype(np.float64) for c in ('CH', 'CLE', 'CRE')}
    raw['CLE-CRE'] = raw['CLE'] - raw['CRE']
    out = ep.copy()
    for ch in LEVEL_CHANNELS:
        out[f'mean_{ch}'] = [raw[ch][a:b].mean() for a, b in zip(ep.i0, ep.i1)]
    return out


def head_blocks(s):
    """Head turn / elevation per 30-s block counted from the first sample."""
    n = int(round(s.fs * 30.0))
    m = s.n_samples // n

    def red(x, fn):
        return fn(np.asarray(x, float)[:m * n].reshape(m, n), axis=1)

    ang = head_angle(s.cap['aX'], s.cap['aY'], s.cap['aZ'], s.fs)
    r = np.radians(ang['turn_deg'][:m * n].reshape(m, n))
    # circular median: median of sin and cos, so the ±180 seam cannot pull it to 0
    turn = np.degrees(np.arctan2(np.median(np.sin(r), axis=1), np.median(np.cos(r), axis=1)))
    hb = pd.DataFrame({'t_hr': red(s.time_hr, np.median), 'turn_deg': turn,
                       'elev_deg': red(ang['elev_deg'], np.median)})
    hb['head_pos'] = classify_head_position(hb.turn_deg.to_numpy(), hb.elev_deg.to_numpy())
    pos = load_position(s)
    hb['psg_pos'] = position_at(pos, hb.t_hr.to_numpy()) if pos else 'Unknown'
    return hb


def sleep_mask(codes):
    """First to last scored sleep epoch."""
    is_sleep = np.isin(codes, SLEEP_CODES)
    if is_sleep.sum() < 20:
        return np.ones(len(codes), bool)
    on = int(np.argmax(is_sleep))
    off = len(is_sleep) - 1 - int(np.argmax(is_sleep[::-1]))
    m = np.zeros(len(codes), bool)
    m[on:off + 1] = True
    return m


def session_levels(g, sm):
    """Session means (sleep period) and the excursion about them, per channel."""
    row = {'session': g.session.iloc[0], 'subject': g.subject.iloc[0],
           'night': int(g.night.iloc[0]), 'n_epochs_sleep': int(sm.sum())}
    for ch in LEVEL_CHANNELS:
        x = g[f'mean_{ch}'].to_numpy()
        mu = float(np.nanmean(x[sm]))
        c = x[sm] - mu
        row.update({f'mean_{ch}': mu, f'sd_{ch}': float(np.nanstd(c)),
                    f'p5_{ch}': float(np.nanpercentile(c, 5)),
                    f'p95_{ch}': float(np.nanpercentile(c, 95)),
                    f'range_{ch}': float(np.nanmax(c) - np.nanmin(c))})
    return row


# ── the imbalance marker ─────────────────────────────────────────────────────

def _roll(x, win_epochs, how):
    return pd.Series(x).rolling(int(win_epochs), center=True,
                                min_periods=max(3, int(win_epochs) // 4)) \
                       .aggregate(how).to_numpy()


def imbalance_marker(d, motion, tau_min=TAU_MIN, despike_min=DESPIKE_MIN,
                     block_min=EPOCH_MIN):
    """
    Magnitude A and direction D of the mean-centred differential d.

    Motion blocks are masked, a short rolling MEDIAN removes re-seat spikes, then a
    rolling MEAN over tau low-passes. (A median at the tau scale would make D a hard
    ±1: over half an hour d is almost always one-signed, and median(d) is then
    ±median(|d|) exactly.) Returns A (fF), D in [-1, +1], and LP(d) (fF).
    """
    win = max(3, int(round(tau_min / block_min)))
    wsp = max(3, int(round(despike_min / block_min)))
    dc = _roll(np.where(motion, np.nan, d), wsp, 'median')
    A = _roll(np.abs(dc), win, 'mean')
    d_lp = _roll(dc, win, 'mean')
    return A, np.clip(d_lp / (A + EPS), -1.0, 1.0), d_lp


def imbalance_epochs(g, sm, hb):
    """Per-epoch marker for one session, joined to the head angle."""
    codes = g.stage_code.to_numpy()
    raw = g['mean_CLE-CRE'].to_numpy()
    mu = float(np.nanmean(raw[sm]))
    d = raw - mu
    motion = g.motion.to_numpy().astype(bool)
    v = np.full(len(d), np.nan)
    v[1:] = np.diff(d) / EPOCH_MIN
    A, D, d_lp = imbalance_marker(d, motion)
    rec = pd.DataFrame({'session': g.session.to_numpy(), 'subject': g.subject.to_numpy(),
                        'night': g.night.to_numpy(), 't_hr': g.t_hr.to_numpy(),
                        'stage_code': codes, 'motion': motion, 'in_sleep': sm,
                        'session_mean_fF': mu, 'd_fF': d, 'v_fF_per_min': v,
                        'imb_mag_fF': A, 'imb_dir': D, 'imb_lp_fF': d_lp})
    for tau in TAU_SWEEP:
        At, Dt, _ = imbalance_marker(d, motion, tau)
        rec[f'imb_mag_tau{int(tau)}'] = At
        rec[f'imb_dir_tau{int(tau)}'] = Dt
    # The angle lives on its own 30-s block grid; it is interpolated linearly onto the
    # epoch centres (as the legacy code did) and the labels taken from the nearest block.
    t = rec.t_hr.to_numpy()
    rec['turn_deg'] = np.interp(t, hb.t_hr, hb.turn_deg)
    j = np.abs(t[:, None] - hb.t_hr.to_numpy()[None, :]).argmin(axis=1)
    rec['psg_pos'] = hb.psg_pos.to_numpy()[j]
    rec['head_pos'] = hb.head_pos.to_numpy()[j]
    return rec


def reversals(D):
    s = np.sign(np.where(np.abs(D) > DEAD_BAND, D, 0.0))
    s = s[s != 0]
    return int(np.sum(np.diff(s) != 0)) if len(s) > 1 else 0


def summarise(rec):
    """Per-session marker summary over the sleep period."""
    g = rec[rec.in_sleep].sort_values('t_hr')
    D, A = g.imb_dir.to_numpy(), g.imb_mag_fF.to_numpy()
    ok = np.isfinite(D)
    third = len(g) // 3
    row = {'session': g.session.iloc[0], 'subject': g.subject.iloc[0],
           'night': int(g.night.iloc[0]), 'session_mean_fF': float(g.session_mean_fF.iloc[0]),
           'n_epochs': len(g),
           'imb_mag_median_fF': float(np.nanmedian(A)),
           'imb_mag_p90_fF': float(np.nanpercentile(A, 90)),
           'imb_dir_median': float(np.nanmedian(D)),
           'imb_dir_first_third': float(np.nanmedian(D[:third])),
           'imb_dir_last_third': float(np.nanmedian(D[-third:])),
           'imb_dir_reversals': reversals(D),
           'frac_left_dominant': float(np.nanmean(D[ok] > DEAD_BAND)),
           'frac_right_dominant': float(np.nanmean(D[ok] < -DEAD_BAND))}
    turn = g.turn_deg.to_numpy()
    m = np.isfinite(turn) & ok
    rho, p = spearmanr(np.sin(np.radians(turn[m])), D[m])
    row.update(rho_dir_vs_sin_turn=float(rho), p_dir_vs_sin_turn=float(p))
    sup = (g.head_pos.to_numpy() == 'Supine') & ok
    row.update(n_supine_epochs=int(sup.sum()),
               imb_dir_median_supine=float(np.nanmedian(D[sup])),
               imb_mag_median_supine_fF=float(np.nanmedian(A[sup])),
               imb_dir_reversals_supine=reversals(D[sup]))
    return row


def burden(ep):
    """Integrated |LP(d)| over the motion-free epochs of each night."""
    rows = []
    for sess, g in ep[~ep.motion.astype(bool)].groupby('session', sort=True):
        x = g.imb_lp_fF.to_numpy(float)
        x = x[np.isfinite(x)]
        pos, neg = x[x > 0].sum() * EPOCH_HOURS, -x[x < 0].sum() * EPOCH_HOURS
        rows.append({'session': sess, 'subject': g.subject.iloc[0], 'n_epochs': len(x),
                     'hours': len(x) * EPOCH_HOURS, 'mean_abs_fF': float(np.abs(x).mean()),
                     'integral_abs_fFh': float(np.abs(x).sum() * EPOCH_HOURS),
                     'integral_pos_fFh': float(pos), 'integral_neg_fFh': float(neg),
                     'asymmetry': float((pos - neg) / (pos + neg)),
                     'frac_left_dominant': float((x > 0).mean())})
    return pd.DataFrame(rows)


# ── CH against CLE-CRE ───────────────────────────────────────────────────────

def ch_vs_clecre(g, sm):
    ch, di = g.mean_CH.to_numpy(), g['mean_CLE-CRE'].to_numpy()
    mu_ch, mu_di = float(np.nanmean(ch[sm])), float(np.nanmean(di[sm]))
    c, d = ch - mu_ch, di - mu_di
    ok = sm & np.isfinite(c) & np.isfinite(d)
    lag = 2                                       # 2 epochs = 1 min
    dc, dd = c[lag:] - c[:-lag], d[lag:] - d[:-lag]
    mk = np.isfinite(dc) & np.isfinite(dd) & ok[lag:]
    row = {'session': g.session.iloc[0], 'subject': g.subject.iloc[0],
           'night': int(g.night.iloc[0]), 'n_epochs': int(ok.sum()),
           'mean_CH_fF': mu_ch, 'mean_DIFF_fF': mu_di, 'offset_fF': mu_ch - mu_di,
           'sd_CH_fF': float(np.nanstd(c[ok])), 'sd_DIFF_fF': float(np.nanstd(d[ok])),
           'sd_ratio_CH_over_DIFF': float(np.nanstd(c[ok]) / (np.nanstd(d[ok]) + 1e-9)),
           'slope_CH_on_DIFF': float(np.polyfit(d[ok], c[ok], 1)[0]),
           'r_level_30s': float(np.corrcoef(c[ok], d[ok])[0, 1]),
           'r_change_1min': float(np.corrcoef(dc[mk], dd[mk])[0, 1])}
    trace = {'t': g.t_hr.to_numpy(), 'ch': c, 'diff': d, 'mu_ch': mu_ch, 'mu_di': mu_di,
             'sd_ch': row['sd_CH_fF'], 'sd_di': row['sd_DIFF_fF']}
    return row, trace


# ── Fig 2 features (10-s blocks) ─────────────────────────────────────────────

def fig2_features(s):
    """CLE-CRE about its sleep-period mean and head turn, in 10-s blocks."""
    fs = s.fs
    n = int(round(fs * FIG2_BLOCK_SEC))
    sig = s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
    m = len(sig) // n
    t_hr = (np.arange(m) + 0.5) * FIG2_BLOCK_SEC / 3600.0
    sos = butter(4, FIG2_LP_HZ / (0.5 * fs), btype='low', output='sos')
    level = sosfiltfilt(sos, sig)[:m * n].reshape(m, n).mean(axis=1)

    ang = head_angle(s.cap['aX'], s.cap['aY'], s.cap['aZ'], fs)
    tr = np.radians(ang['turn_deg'][:m * n].reshape(m, n))
    turn = np.degrees(np.arctan2(np.sin(tr).mean(axis=1), np.cos(tr).mean(axis=1)))

    sp = s.sleep_profile
    codes = np.asarray(sp['codes'], int)
    win = sleep_mask(codes)
    j = np.searchsorted(np.asarray(sp['t_ep_hr'], float), t_hr, side='right') - 1
    ok = (j >= 0) & (j < len(codes))
    in_sleep = np.zeros(m, bool)
    in_sleep[ok] = win[np.clip(j, 0, len(codes) - 1)[ok]]
    mu = float(np.nanmean(level[in_sleep]))
    return {'t_hr': t_hr, 'centred': level - mu, 'mu': mu, 'turn_deg': turn, 'sp': sp}


# ── drawing helpers ──────────────────────────────────────────────────────────

def sym_zero_ylim(*series, k=5.0, keep_pct=85.0, pad=0.10):
    """Symmetric limits about zero following the bulk of the trace; large steps overflow."""
    y = np.concatenate([np.asarray(v, float).ravel() for v in series])
    y = y[np.isfinite(y)]
    half = max(k * 1.4826 * np.median(np.abs(y)), np.percentile(np.abs(y), keep_pct))
    half = max(min(half, float(np.max(np.abs(y)))), 1e-6) * (1 + pad)
    return -half, half


def draw_ladder(ax, sp):
    """Hypnogram as a depth-ordered ladder (Wake, N1, N2, N3, REM), runs merged."""
    t_ep = np.asarray(sp['t_ep_hr'], float)
    codes = np.asarray(sp['codes'], int)
    nep = min(len(codes), len(t_ep) - 1)
    y = np.array([STAGE_ROW.get(int(c), np.nan) for c in codes[:nep]], float)
    runs, start = [], 0
    for j in range(1, nep + 1):
        if j == nep or codes[j] != codes[start]:
            runs.append((start, j))
            start = j
    for a, b in runs:
        if np.isfinite(y[a]):
            ax.plot([t_ep[a], t_ep[b]], [y[a]] * 2, lw=4.0, solid_capstyle='butt',
                    color=STAGE_COLORS.get(int(codes[a]), '#AAA'), zorder=4)
    for (a0, b0), (a1, _) in zip(runs[:-1], runs[1:]):
        if np.isfinite(y[a0]) and np.isfinite(y[a1]):
            ax.plot([t_ep[b0]] * 2, [y[a0], y[a1]], lw=0.7, color='#2C3E50',
                    alpha=0.35, zorder=3)
    ax.set_yticks([STAGE_ROW[c] for c in STAGE_ORDER])
    ax.set_yticklabels([STAGE_LABELS[c] for c in STAGE_ORDER])
    ax.set_ylim(-0.6, len(STAGE_ORDER) - 0.4)
    for c in STAGE_ORDER:
        ax.axhline(STAGE_ROW[c], color='#CCCCCC', lw=0.5, zorder=1)


def draw_head_row(ax, t, turn):
    """Head turn, re-wrapped about the night's circular median, axis scaled to the
    sustained posture (2-98 %), brief excursions overflow."""
    turn = np.asarray(turn, float)
    fin = np.isfinite(turn)
    med = np.degrees(np.arctan2(np.median(np.sin(np.radians(turn[fin]))),
                                np.median(np.cos(np.radians(turn[fin])))))
    turn = ((turn - med + 180.0) % 360.0) - 180.0 + med
    ax.plot(t, turn, lw=1.7, color=HEAD_COLOR, zorder=4)
    lo, hi = np.percentile(turn[np.isfinite(turn)], [2, 98])
    span = max(hi - lo, HEAD_MIN_SPAN_DEG)
    mid = 0.5 * (lo + hi)
    lo, hi = max(mid - 0.62 * span, -190.0), min(mid + 0.62 * span, 190.0)
    ax.set_ylim(lo, hi)
    for lv, lb in [(180, 'prone'), (90, 'left'), (0, 'supine'), (-90, 'right'),
                   (-180, 'prone')]:
        if lo < lv < hi:
            ax.axhline(lv, color='#555', ls=':', lw=0.8, zorder=3)
            ax.annotate(lb, (0.997, lv), xycoords=('axes fraction', 'data'),
                        fontsize=11, color='#444', va='bottom', ha='right')
    step = next(s for s in (5, 10, 15, 30, 45, 90) if span / s <= 9)
    ax.set_yticks(np.arange(np.ceil(lo / step) * step, np.floor(hi / step) * step + 1, step))
    ax.set_ylabel('Head turn\n(deg)', color=HEAD_COLOR)
    ax.tick_params(axis='y', labelcolor=HEAD_COLOR)


# ── figures ──────────────────────────────────────────────────────────────────

def fig2(feats):
    fig = plt.figure(figsize=(24, 13), layout='constrained')
    subfigs = fig.subfigures(2, 2, wspace=0.03, hspace=0.04)
    for sf, (letter, label, age) in zip(subfigs.ravel(), FIG2_PANELS):
        f = feats[label]
        axes = sf.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [0.7, 1.25, 1.05]})
        draw_ladder(axes[0], f['sp'])
        axes[0].set_ylabel('Sleep stage')
        axes[0].set_title(f'({letter})  {label}, {age}-year-old male', loc='left',
                          fontsize=16, fontweight='bold')
        ax = axes[1]
        ax.axhline(0, color='#2C3E50', ls='--', lw=1.2, zorder=2)
        ax.plot(f['t_hr'], f['centred'], lw=1.2, color=CAP_COLORS['CLE-CRE'], zorder=4)
        ax.set_ylim(*sym_zero_ylim(f['centred']))
        ax.set_ylabel(f'CLE−CRE − mean\n({CAP_UNIT}; μ {f["mu"]:,.0f})',
                      color=CAP_COLORS['CLE-CRE'])
        ax.tick_params(axis='y', labelcolor=CAP_COLORS['CLE-CRE'])
        ax.grid(True, alpha=0.15)
        draw_head_row(axes[2], f['t_hr'], f['turn_deg'])
        axes[2].set_xlabel('Time (hours)')
        sf.align_ylabels(axes)
    save(fig, 'fig2_overnight', STAGE)


def fig_s2(ep, sess):
    sessions = sorted(ep.session.unique())
    fig, axes = plt.subplots(6, 2, figsize=(18, 20))
    rev = sess.set_index('session').imb_dir_reversals
    for ax, lbl in zip(axes.ravel(), sessions):
        g = ep[ep.session == lbl].sort_values('t_hr')
        t, y, A = g.t_hr.to_numpy(), g.imb_lp_fF.to_numpy(), g.imb_mag_fF.to_numpy()
        ax.fill_between(t, 0, y, where=y >= 0, interpolate=True, color=POS_COLOR, alpha=0.3, lw=0)
        ax.fill_between(t, 0, y, where=y < 0, interpolate=True, color=NEG_COLOR, alpha=0.3, lw=0)
        ax.plot(t, A, ls='--', lw=0.9, color='#555')
        ax.plot(t, -A, ls='--', lw=0.9, color='#555')
        ax.plot(t, y, lw=1.6, color=DIFF_COLOR)
        ax.axhline(0, color='#2C3E50', lw=0.8)
        ax.set_ylim(*sym_zero_ylim(y, A, -A))
        ax.set_title(f'{lbl}   reversals: {rev[lbl]}', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'LP(d)  ({CAP_UNIT})')
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)')
    handles = [plt.Line2D([], [], color=DIFF_COLOR, lw=2, label='signed marker LP(d)'),
               plt.Line2D([], [], color='#555', ls='--', label='± envelope LP|d|'),
               mpatches.Patch(color=POS_COLOR, alpha=0.4, label='left-dominant'),
               mpatches.Patch(color=NEG_COLOR, alpha=0.4, label='right-dominant')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0),
               fontsize=12)
    fig.tight_layout()
    save(fig, 'figS2_imbalance_all_nights', STAGE)


def fig_s4(traces):
    fig, axes = plt.subplots(6, 2, figsize=(20, 21))
    for ax, lbl in zip(axes.ravel(), sorted(traces)):
        tr = traces[lbl]
        lim_c = CLIP_SD * max(tr['sd_ch'], 1e-6)
        ax.plot(tr['t'], np.clip(tr['ch'], -lim_c, lim_c), lw=1.3, color=CH_COLOR, zorder=3)
        ax.axhline(0, color='#2C3E50', ls='--', lw=0.8, zorder=2)
        ax.set_ylim(-lim_c * 1.1, lim_c * 1.1)
        ax.set_ylabel(f'CH − mean ({CAP_UNIT})', color=CH_COLOR)
        ax.tick_params(axis='y', labelcolor=CH_COLOR)
        ax2 = ax.twinx()
        lim_d = CLIP_SD * max(tr['sd_di'], 1e-6)
        ax2.plot(tr['t'], np.clip(tr['diff'], -lim_d, lim_d), lw=1.3, color=DIFF_COLOR, zorder=4)
        ax2.set_ylim(-lim_d * 1.1, lim_d * 1.1)
        ax2.set_ylabel(f'CLE−CRE − mean ({CAP_UNIT})', color=DIFF_COLOR)
        ax2.tick_params(axis='y', labelcolor=DIFF_COLOR)
        ax2.spines['right'].set_visible(True)
        ax.set_title(f'{lbl}    CH μ {tr["mu_ch"]:,.0f}   CLE−CRE μ {tr["mu_di"]:,.0f} {CAP_UNIT}',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)')
    handles = [plt.Line2D([], [], color=CH_COLOR, lw=2.5, label='CH, left axis'),
               plt.Line2D([], [], color=DIFF_COLOR, lw=2.5, label='CLE−CRE, right axis')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=13,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    save(fig, 'figS4_ch_vs_clecre', STAGE)


def fig_s6(b):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(b))
    subj = sorted(b.subject.unique())
    colors = [plt.cm.tab10(subj.index(s_)) for s_ in b.subject]
    ax[0].bar(x, b.integral_abs_fFh, color=colors)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('∫|imbalance| dt  (fF·h)')
    ax[0].set_title('A  Imbalance burden per night', loc='left')
    for k, (sj, g) in enumerate(b.groupby('subject', sort=True)):
        g = g.sort_values('session')
        ax[1].plot([0, 1], g.integral_abs_fFh.values, 'o-', color=plt.cm.tab10(k),
                   label=f'S{k + 1}', lw=1.6, ms=6)
    ax[1].set_yscale('log')
    ax[1].set_xlim(-0.3, 1.3)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(['night 1', 'night 2'])
    ax[1].set_ylabel('∫|imbalance| dt  (fF·h)')
    ax[1].set_title('B  Same subject, two nights', loc='left')
    ax[1].legend(fontsize=9, frameon=False, ncol=2)
    ax[2].bar(x, b.asymmetry, color=colors)
    ax[2].axhline(0, color='k', lw=0.8)
    ax[2].set_ylim(-1, 1)
    ax[2].set_ylabel('asymmetry index')
    ax[2].set_title('C  Net lateralization (null)', loc='left')
    for a in (ax[0], ax[2]):
        a.set_xticks(x)
        a.set_xticklabels(b.session, rotation=90, fontsize=10)
    for a in ax[:2]:                     # label the log axis at readable values
        a.set_yticks([10, 20, 50, 100, 200, 500, 1000])
        a.set_yticklabels(['10', '20', '50', '100', '200', '500', '1000'])
        a.minorticks_off()
        a.set_ylim(10, 1000)
    for a in ax:
        a.grid(axis='y', alpha=0.3, lw=0.5)
    fig.tight_layout()
    save(fig, 'figS6_imbalance_burden', STAGE)


# ── run ──────────────────────────────────────────────────────────────────────

def run():
    grid = pd.read_parquet(CACHE_DIR / 's01_recordings' / 'epochs.parquet')
    out_c, out_t = CACHE_DIR / STAGE, TAB_DIR / STAGE
    out_c.mkdir(parents=True, exist_ok=True)
    out_t.mkdir(parents=True, exist_ok=True)

    levels, recs, summ, chrows, traces, f2 = [], [], [], [], {}, {}
    fig2_sessions = {p[1] for p in FIG2_PANELS}
    for s in iter_sessions():
        g = epoch_levels(s, grid[grid.session == s.label].sort_values('epoch')
                         .reset_index(drop=True))
        sm = sleep_mask(g.stage_code.to_numpy())
        levels.append(session_levels(g, sm))
        rec = imbalance_epochs(g, sm, head_blocks(s))
        recs.append(rec)
        summ.append(summarise(rec))
        row, traces[s.label] = ch_vs_clecre(g, sm)
        chrows.append(row)
        if s.label in fig2_sessions:
            f2[s.label] = fig2_features(s)
        print(f'  {s.label}: {len(g)} epochs')

    lev = pd.DataFrame(levels)
    ep = pd.concat(recs, ignore_index=True)
    sess = pd.DataFrame(summ)
    b = burden(ep)
    chv = pd.DataFrame(chrows)
    ep.to_parquet(out_c / 'imbalance_epochs.parquet', index=False)
    for name, df in [('session_levels', lev), ('imbalance_session', sess),
                     ('imbalance_burden', b), ('ch_vs_clecre_sessions', chv)]:
        df.to_csv(out_c / f'{name}.csv', index=False)
        df.to_csv(out_t / f'{name}.csv', index=False)

    fig2(f2)
    fig_s2(ep, sess)
    fig_s4(traces)
    fig_s6(b)
    register_numbers(lev, ep, sess, b, chv)


def register_numbers(lev, ep, sess, b, chv):
    nb = Numbers(STAGE)
    src = 'session_levels'
    rng = lambda c, f='{:.0f}': f'{f.format(lev[c].min())} to {f.format(lev[c].max())}'
    nb.add('mean_CLE', '§3.1 ¶172', 'session-mean CLE (sleep period)', rng('mean_CLE'),
           '1958 to 2048', unit='fF', source=src)
    nb.add('mean_CRE', '§3.1 ¶172', 'session-mean CRE (sleep period)', rng('mean_CRE'),
           '1624 to 2353', unit='fF', source=src)
    nb.add('mean_CH', '§3.1 ¶172', 'session-mean CH (sleep period)',
           f'{lev.mean_CH.max():.0f} to {lev.mean_CH.min():.0f}', '−719 to −1297',
           unit='fF', source=src)
    nb.add('drift_S1', '§3.1 ¶172', '25-h bench stability test: 3 fF drift over 3 °C',
           'bench test', '3 fF', status='EXTERNAL',
           note='Fig S1; text says 25-hour, the S1 caption says 24-hour')
    nb.add('ch_gain_2x', '§3.1 ¶172', 'CH ~twice the response of a single-ended channel',
           f'{chv.sd_ratio_CH_over_DIFF.median():.2f}', 'approximately twice',
           status='EXTERNAL', source='ch_vs_clecre',
           note='hardware statement; for reference the median sd(CH)/sd(CLE-CRE) of the '
                f'30-s levels is {chv.sd_ratio_CH_over_DIFF.median():.2f} '
                f'(range {chv.sd_ratio_CH_over_DIFF.min():.2f}-{chv.sd_ratio_CH_over_DIFF.max():.2f})')
    for ch, pv in [('CLE', '105'), ('CRE', '71'), ('CLE-CRE', '158'), ('CH', '595')]:
        nb.add(f'range_{ch}', '§3.1 ¶175', f'median within-night range, {ch}',
               float(lev[f'range_{ch}'].median()), pv, unit='fF', source=src,
               note='max - min of the 30-s level over the sleep period, median of 12')
    big = lev.loc[lev.range_CH > 1000, 'session'].tolist()
    nb.add('ch_gt_1000', '§3.1 ¶175', 'recordings with CH within-night range > 1000 fF',
           f'{len(big)} ({", ".join(big)})', 'three most mobile recordings', status='DIFF',
           source=src, note='CORRECTION: five recordings, not three')

    g = ep[ep.in_sleep & np.isfinite(ep.turn_deg) & np.isfinite(ep.imb_dir)]
    rho, p = spearmanr(np.sin(np.radians(g.turn_deg)), g.imb_dir)
    nb.add('n_pooled', '§3.1 ¶176', 'pooled sleep-period epochs, direction vs head turn',
           f'{len(g):,}', '9,182', source='imbalance_epochs')
    nb.add('rho_turn', '§3.1 ¶176', 'Spearman rho, direction D vs sin(head turn), pooled',
           float(rho), '0.01', source='fig_vs_headangle (pooled)',
           note='POOLED epochs (non-independent); uses sin(turn), not the angle; per-session '
                f'rho {sess.rho_dir_vs_sin_turn.min():+.2f} to {sess.rho_dir_vs_sin_turn.max():+.2f}, '
                f'p<0.05 in {(sess.p_dir_vs_sin_turn < 0.05).sum()}/12. Text cites Fig S4 for '
                'this result; S4 is CH vs CLE-CRE')
    nb.add('p_turn', '§3.1 ¶176', 'p of that rho', float(p), '0.62',
           source='fig_vs_headangle (pooled)', note='pooled-epoch p-value, descriptive only')
    sup = g.head_pos == 'Supine'
    nb.add('n_supine', '§3.1 ¶176', 'supine epochs (pooled)', f'{int(sup.sum()):,}', '7,641',
           source='imbalance_epochs')
    nb.add('rev_supine', '§3.1 ¶176', 'median direction reversals per night, supine only',
           float(sess.imb_dir_reversals_supine.median()), '3.5', source='summarise',
           note='reversals counted over the concatenated supine epochs of a night')
    for pos, pv in [('Supine', '9.0'), ('Left', '11.3'), ('Right', '107.7')]:
        nb.add(f'mag_{pos}', '§3.1 ¶176', f'median magnitude A, head {pos.lower()}',
               float(g.loc[g.head_pos == pos, 'imb_mag_fF'].median()), pv, unit='fF',
               source='fig_vs_headangle panel C',
               note=f'POOLED epochs (n={int((g.head_pos == pos).sum())}); right-lying is '
                    'dominated by few nights')

    nb.add('mean_abs', '§3.1 ¶177', 'time-averaged |LP(d)| per recording',
           f'{b.mean_abs_fF.min():.1f} to {b.mean_abs_fF.max():.1f}', '3.1 to 135.7',
           unit='fF', source='burden')
    nb.add('burden_range', '§3.1 ¶177', 'imbalance burden per recording',
           f'{b.integral_abs_fFh.min():.0f} to {b.integral_abs_fFh.max():.0f}', '15 to 705',
           unit='fF·h', source='burden', note='text cites Fig S5a; it is Fig S6a')
    nb.add('burden_fold', '§3.1 ¶177', 'max / min burden',
           float(b.integral_abs_fFh.max() / b.integral_abs_fFh.min()), '46', unit='fold',
           source='burden')
    within = b.groupby('subject').integral_abs_fFh.agg(lambda x: x.max() / x.min())
    nb.add('burden_within', '§3.1 ¶177', 'night-to-night burden ratio within subject',
           f'{within.min():.1f} to {within.max():.1f}', '2.0- to 5.8-fold', unit='fold',
           source='burden', note='text cites Fig S5b; it is Fig S6b')
    nb.add('asym_range', '§3.1 ¶177', 'asymmetry index range',
           f'{b.asymmetry.min():+.2f} and {b.asymmetry.max():+.2f}', '−0.22 and +0.09',
           source='burden')
    nb.add('asym_within', '§3.1 ¶177', 'recordings with |asymmetry| <= 0.09',
           f'{int((b.asymmetry.abs() <= 0.09).sum())} of {len(b)}', '11 of the 12',
           source='burden')
    nb.add('fig_s5c', '§3.1 ¶177', 'CLE-CRE near zero at sleep onset and end (Fig S5c)',
           'no source', 'Fig. S5c', status='EXTERNAL',
           note='UNTRACED: no script makes this panel or number; Fig S6c is the asymmetry '
                'index. By construction LP(d) is referenced to the sleep-period mean')

    rv = sess.imb_dir_reversals
    nb.add('s2_rev_median', 'Supp Fig S2 caption', 'direction reversals per night, median',
           float(rv.median()), '4', source='summarise')
    nb.add('s2_rev_range', 'Supp Fig S2 caption', 'direction reversals per night, range',
           f'{rv.min()}–{rv.max()}', '1–8', source='summarise',
           note='both written with an en dash so the registry parses them alike')
    ratio = sess.imb_mag_median_fF.max() / sess.imb_mag_median_fF.min()
    nb.add('s2_scale', 'Supp Fig S2 caption', 'max/min of nightly median magnitude',
           f'{ratio:.1f}', 'more than an order of magnitude',
           status='MATCH' if ratio > 10 else 'DIFF', source='summarise')
    top = chv.sort_values('sd_CH_fF', ascending=False).session.head(3).tolist()
    nb.add('s4_mobile', 'Supp Fig S4 caption', 'nights with the largest CH excursion',
           ', '.join(top), 'S5N1, S6N1, S6N2', status='DIFF', source='ch_vs_clecre',
           note='by sd of CH about its session mean; by within-night range the top three are '
                + ', '.join(lev.sort_values('range_CH', ascending=False).session.head(3)))

    bs = b.set_index('session').integral_abs_fFh
    nb.add('s6_S6N1', 'Supp S3 ¶118', 'burden S6N1', float(bs['S6N1']), '224', unit='fF·h',
           source='burden')
    nb.add('s6_S6N2', 'Supp S3 ¶118', 'burden S6N2', float(bs['S6N2']), '705', unit='fF·h',
           source='burden')
    nb.add('s6_median', 'Supp S3 ¶118', 'cohort median burden', float(bs.median()), '65',
           unit='fF·h', source='burden')
    no6 = b[b.subject != 'OS006']
    nb.add('s6_range_wo', 'Supp S3 ¶118', 'burden range without S6',
           f'{no6.integral_abs_fFh.min():.0f} to {no6.integral_abs_fFh.max():.0f}',
           '15 to 199', unit='fF·h', source='burden')
    nb.add('s6_left', 'Supp S3 ¶118', 'left-dominant time fraction range',
           f'{b.frac_left_dominant.min():.2f} to {b.frac_left_dominant.max():.2f}',
           '0.41 to 0.75', source='burden',
           note='"across the same nights" -- computed over all twelve')
    nb.add('fig2_caption', '§3.1 ¶174', 'Fig 2 caption: participants and ages',
           'S6N2 25, S3N2 37, S4N2 54, S2N2 66 (all male)', 'four male participants aged 54',
           status='EXTERNAL', note='caption garbled; panels are four participants aged 25, '
                                   '37, 54, 66 (ages from Table 1)')
    nb.save()


if __name__ == '__main__':
    run()
