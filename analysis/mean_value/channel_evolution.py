"""
Per-session channel "evolution" panels (journal figures).

The point of these figures: show how the raw sensor value evolves across a night,
and how that evolution carries a directional FLOW signal. The flow marker is still
being tuned — it is drawn here as the current best definition, not a settled one.

Two figures per session, grouped by what the channels are:

    <SESSION>_CLE_CRE.png       the two absolute temple channels
    <SESSION>_CH_CLE-CRE.png    the two difference channels

Rows (both figures)
-------------------
    A  hypnogram      PSG stages as a depth-ordered ladder — Wake, N1, N2, N3,
                      then REM — so depth reads from the geometry, not colour
                      alone. Contiguous epochs are merged into one rung with
                      faint connectors, because per-epoch step connectors at full
                      weight swamp the rungs they are meant to join.
    B  mean value     DC level referenced to the SESSION MEAN, in femtofarads,
                      with that mean printed. The y-axis is SYMMETRIC ABOUT ZERO
                      and scaled to the bulk of the trace, so the sign of an
                      excursion is read directly off the axis; nights with large
                      re-seat steps overflow the axis rather than compressing
                      everything else into a flat line.
    C  variance       within-window variance (fF^2)
    D  flow           difference figure: the night-scale flow marker — magnitude
                      A = LP(|d|) and direction D = LP(d)/A in [-1,+1], for both
                      CH and CLE-CRE on one shared dimensionless axis (the only
                      directly comparable view of the two).
                      Absolute figure: the same low pass of each level. A signed
                      flow direction needs a difference channel, so it is not
                      defined for CLE or CRE on their own.
    E  head + motion  head TURN angle from the accelerometer (0 supine, +90 left,
                      -90 right) over accelerometer activity.
    F,G  spectrograms 0-5 Hz spectrogram of each channel + dB colorbar

CH is the sensor's own hardware difference channel and is NOT the arithmetic
CLE-CRE: across the cohort the offset is ~-742 fF and the gain of CH on CLE-CRE
runs from -1.9 to +6.5, changing sign between subjects (ch_vs_clecre_sessions.py).
Row D of the difference figure prints this session's offset / gain / correlation
and how often the two flow directions agree in sign — S6N1 r=+0.93 (93% of
blocks) against S5N2 r=-0.62 (25%).

Units: the CAP columns are true capacitance in femtofarads (see CAP_SCALE_TO_FF
in sleep_monitor/config.py for the scale and its provenance/caveats).

The raw channel is low-pass filtered at 10 Hz (zero-phase Butterworth) before the
mean and variance are computed, so those series reflect physiological-band
content and are not inflated by the >10 Hz electronic noise floor.

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/channel_evolution.py --all
    .venv/Scripts/python.exe analysis/mean_value/channel_evolution.py --session S1N1

Outputs
-------
    writeup/figures/channel_evolution/<SESSION>_CLE_CRE.png
    writeup/figures/channel_evolution/<SESSION>_CH_CLE-CRE.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import butter, sosfiltfilt, spectrogram as sp_spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sleep_monitor import load_session, load_sleep_profile, head_angle
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, CAP_COLORS,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    CAP_SCALE_TO_FF, CAP_UNIT, CAP_UNIT_SQ,
)
from sleep_monitor.sessions import SESSION_META
# One definition of the flow marker, shared with flow_marker.py so the two
# figures cannot drift apart.
from flow_marker import flow_marker, TAU_MIN, DESPIKE_MIN

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'writeup' / 'figures' / 'channel_evolution'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CLE', 'CRE', 'CH', 'CLE-CRE']
DIFF_CHANNELS = ['CLE-CRE', 'CH']       # channels a signed flow direction exists for
CH_COLOR = {'CLE': CAP_COLORS['CLE'], 'CRE': CAP_COLORS['CRE'],
            'CH': CAP_COLORS['CH'], 'CLE-CRE': CAP_COLORS['CLE-CRE']}
CH_LONG = {'CLE': 'left temple (CLE)', 'CRE': 'right temple (CRE)',
           'CH': 'hardware difference (CH)',
           'CLE-CRE': 'arithmetic difference (CLE−CRE)'}
MAG_COLOR = '#B9380B'
DIR_COLOR = '#1F618D'
HEAD_COLOR = '#16A085'
MOTION_PCT = 90            # top-decile accelerometer activity counts as motion

# Hypnogram ladder: stage code -> vertical position. Ordered by STRENGTH OF SLEEP
# DEPTH, so descending the ladder is monotonically deeper sleep: Wake, N1, N2, N3,
# then REM on its own rung at the bottom. (REM is placed below N3 rather than in
# the conventional slot next to Wake, so the NREM descent reads as a clean depth
# gradient and REM does not interrupt it.)
LADDER_ORDER = [4, 3, 2, 1, 0]                                  # Wake, N1, N2, N3, REM
LADDER_Y = {c: len(LADDER_ORDER) - 1 - i for i, c in enumerate(LADDER_ORDER)}

BLOCK_SEC = 10.0            # window for mean / variance / motion (display rows)
LP_CAP_HZ = 10.0           # low-pass cap applied to raw signal before mean/variance
SPEC_FMAX = 5.0            # spectrogram top frequency (Hz)

# ── Journal styling ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'font.family': 'DejaVu Sans', 'figure.dpi': 200,
})
PANEL = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


def robust_ylim(y, pad=0.08, symmetric=False, floor0=False):
    y = np.asarray(y, float)
    good = np.isfinite(y)
    if good.sum() < 3:
        return (-1, 1)
    lo, hi = np.percentile(y[good], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo -= pad * span
    hi += pad * span
    if symmetric:
        m = max(abs(lo), abs(hi))
        return (-m, m)
    if floor0:
        lo = min(0.0, lo)
    return (lo, hi)


def sym_zero_ylim(*series, k=5.0, keep_pct=85.0, pad=0.10):
    """
    Symmetric limits about ZERO for a mean-centred trace.

    Zero is the session mean, so putting it at the centre of the axis makes the
    sign of an excursion readable straight off the axis. The half-width follows
    the bulk of the trace (k robust sigma, but at least the keep_pct percentile
    of |y|), never wider than the data. Nights with large electrode re-seat steps
    therefore OVERFLOW the axis instead of squashing every other night's
    structure onto the zero line.
    """
    y = np.concatenate([np.asarray(v, float).ravel() for v in series])
    y = y[np.isfinite(y)]
    if y.size < 3:
        return (-1.0, 1.0)
    mad0 = 1.4826 * np.median(np.abs(y))                 # spread about zero
    half = max(k * mad0, np.percentile(np.abs(y), keep_pct))
    half = min(half, float(np.max(np.abs(y))))           # never wider than data
    half = max(half, 1e-6) * (1 + pad)
    return (-half, half)


def adaptive_ylim(y, k=5.0, pad=0.08):
    """Mean-centred robust limits: median +/- k*MAD, but never zoomed OUT past the
    data. Sessions with huge coupling swings zoom in to show the bulk around the
    mean (extreme excursions clip); clean sessions show their full range."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return (-1, 1)
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826
    spread = max(k * mad, 1e-6)
    lo = max(med - spread, float(y.min()))
    hi = min(med + spread, float(y.max()))
    if hi <= lo:
        lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def block_reduce(x, n, fn):
    m = len(x) // n
    return fn(x[: m * n].reshape(m, n), axis=1)


def lowpass(x, fs, fc):
    sos = butter(4, fc / (0.5 * fs), btype='low', output='sos')
    return sosfiltfilt(sos, x)


def sleep_window(codes):
    """Mask over the sleep period (first to last scored sleep epoch)."""
    is_sleep = np.isin(codes, [0, 1, 2, 3])
    if is_sleep.sum() < 20:
        return np.ones(len(codes), bool)
    on = int(np.argmax(is_sleep))
    off = len(is_sleep) - 1 - int(np.argmax(is_sleep[::-1]))
    m = np.zeros(len(codes), bool); m[on:off + 1] = True
    return m


def compute_features(s):
    fs = s.fs
    raw = {}
    for ch in CHANNELS:
        if ch == 'CLE-CRE':
            sig = s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)
        else:
            sig = s.cap[ch].astype(np.float64)
        raw[ch] = sig * CAP_SCALE_TO_FF        # -> femtofarads
    acc = s.cap['acc_mag'].astype(np.float64)

    # ── Display rows (mean / variance / motion) at 10 s blocks ──
    n = int(round(fs * BLOCK_SEC))
    m = min(min(len(v) for v in raw.values()), len(acc)) // n
    t_hr = (np.arange(m) + 0.5) * BLOCK_SEC / 3600.0
    motion = block_reduce(acc[: m * n], n, np.std)
    is_motion = motion > np.nanpercentile(motion, MOTION_PCT)

    # ── Head angle, corrected definition (0.05 Hz gravity cutoff, full +/-180) ──
    # Block-reduce the angle through sin/cos so the +/-180 seam cannot average a
    # left-turned head into "supine".
    ang = head_angle(s.cap['aX'], s.cap['aY'], s.cap['aZ'], fs)
    tr = np.radians(ang['turn_deg'])
    turn = np.degrees(np.arctan2(block_reduce(np.sin(tr)[: m * n], n, np.mean),
                                 block_reduce(np.cos(tr)[: m * n], n, np.mean)))

    # ── Sleep-period mean, used as the zero of every level row ──
    sp = s.sleep_profile
    codes = np.asarray(sp['codes'], int)
    t_ep = np.asarray(sp['t_ep_hr'], float)
    win = sleep_window(codes)
    ep_of_block = np.searchsorted(t_ep, t_hr, side='right') - 1
    ok = (ep_of_block >= 0) & (ep_of_block < len(codes))
    in_sleep = np.zeros(len(t_hr), bool)
    in_sleep[ok] = win[np.clip(ep_of_block, 0, len(codes) - 1)[ok]]
    if in_sleep.sum() < 10:
        in_sleep[:] = True

    feats = {'t_hr': t_hr, 'motion': motion, 'is_motion': is_motion,
             'turn_deg': turn, 'in_sleep': in_sleep}
    for ch, sig in raw.items():
        filt = lowpass(sig, fs, LP_CAP_HZ)          # <10 Hz cap before mean/var
        mean_b = block_reduce(filt[: m * n], n, np.mean)
        var_b = block_reduce(filt[: m * n], n, np.var)
        mu = float(np.nanmean(mean_b[in_sleep]))
        d = mean_b - mu
        # Same marker as flow_marker.py, evaluated on this figure's 10 s blocks.
        A, D, d_lp = flow_marker(d, is_motion, TAU_MIN, DESPIKE_MIN,
                                 block_min=BLOCK_SEC / 60.0)
        feats[ch] = {'mean': mean_b, 'centred': d, 'mu': mu, 'var': var_b,
                     'flow_mag': A, 'flow_dir': D, 'flow_lp': d_lp}
    return feats, raw


def stage_shading(ax, sp, alpha=0.10):
    tep, codes = sp['t_ep_hr'], sp['codes']
    for j in range(len(tep) - 1):
        ax.axvspan(tep[j], tep[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                   alpha=alpha, lw=0, zorder=0)


def panel_letter(ax, k):
    ax.text(-0.092, 1.02, PANEL[k], transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='right')


def draw_ladder(ax, sp, title=None):
    """
    Hypnogram as a depth-ordered ladder.

    Contiguous epochs of the same stage are merged into one coloured rung and the
    connectors between rungs are drawn thin and faint. The previous version drew
    a full-weight per-epoch step trace, so on a fragmented night the vertical
    connectors were visually heavier than the rungs and the ladder read as a
    picket fence.
    """
    t_ep = np.asarray(sp['t_ep_hr'], float)
    codes = np.asarray(sp['codes'], int)
    nep = min(len(codes), len(t_ep) - 1)
    y = np.array([LADDER_Y.get(int(c), np.nan) for c in codes[:nep]], float)

    # merge contiguous same-stage runs
    runs, start = [], 0
    for j in range(1, nep + 1):
        if j == nep or codes[j] != codes[start]:
            runs.append((start, j)); start = j
    for a, b in runs:
        if not np.isfinite(y[a]):
            continue
        ax.plot([t_ep[a], t_ep[b]], [y[a]] * 2, lw=4.0, solid_capstyle='butt',
                color=STAGE_COLORS.get(int(codes[a]), '#AAA'), zorder=4)
    # faint connectors so transitions are visible without dominating the rungs
    for (a0, b0), (a1, _) in zip(runs[:-1], runs[1:]):
        if np.isfinite(y[a0]) and np.isfinite(y[a1]):
            ax.plot([t_ep[b0]] * 2, [y[a0], y[a1]], lw=0.7, color='#2C3E50',
                    alpha=0.35, zorder=3)

    ax.set_yticks(list(LADDER_Y.values()))
    ax.set_yticklabels([STAGE_LABELS[c] for c in LADDER_ORDER], fontsize=9.5)
    ax.set_ylim(-0.6, len(LADDER_ORDER) - 0.4)
    ax.set_ylabel('Sleep stage', fontsize=10)
    ax.grid(True, axis='y', alpha=0.25, lw=0.6)
    for yv in LADDER_Y.values():                       # rung guides
        ax.axhline(yv, color='#CCCCCC', lw=0.5, zorder=1)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')


def draw_flow_row(ax, t, mag, dirn, label):
    """Flow magnitude (fF, left) and direction (dimensionless, right)."""
    ax.plot(t, mag, lw=1.4, color=MAG_COLOR, zorder=3)
    ax.set_ylabel(f'Flow magnitude\nLP|Δ|  ({CAP_UNIT})', color=MAG_COLOR)
    ax.tick_params(axis='y', labelcolor=MAG_COLOR)
    ax.set_ylim(0, max(np.nanpercentile(mag, 99) * 1.15, 1e-3))
    ax2 = ax.twinx()
    ax2.axhline(0, color='#2C3E50', ls='--', lw=0.8, zorder=2)
    ax2.plot(t, dirn, lw=1.8, color=DIR_COLOR, zorder=4)
    ax2.fill_between(t, 0, dirn, where=(dirn >= 0), color='#C0392B', alpha=0.18,
                     interpolate=True, zorder=1)
    ax2.fill_between(t, 0, dirn, where=(dirn < 0), color='#2980B9', alpha=0.18,
                     interpolate=True, zorder=1)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_ylabel('Flow direction  [−1,+1]', color=DIR_COLOR)
    ax2.tick_params(axis='y', labelcolor=DIR_COLOR)
    ax.text(0.006, 0.94, f'{label} · low pass {TAU_MIN:.0f} min, motion excluded · '
            f'direction + = left-dominant, − = right-dominant',
            transform=ax.transAxes, fontsize=8, style='italic', color='#555',
            va='top', zorder=6)
    return ax2


def draw_head_row(ax, t, turn, motion):
    """
    Head-turn angle over accelerometer activity.

    The angle is the primary reading here, so it gets the full +/-180 range with
    a rung every 45 deg and the posture labels called out. Motion sits behind it
    as a shaded context band.
    """
    ax.fill_between(t, 0, motion, color='#7F8C8D', alpha=0.35, zorder=1)
    ax.plot(t, motion, lw=0.6, color='#7F8C8D', zorder=2)
    ax.set_ylabel('Motion\n(acc. std)', color='#5D6D7E', fontsize=10)
    ax.tick_params(axis='y', labelcolor='#5D6D7E', labelsize=9)
    ax.set_ylim(*robust_ylim(motion, floor0=True))
    ax2 = ax.twinx()
    ax2.plot(t, turn, lw=1.6, color=HEAD_COLOR, zorder=4)
    for lv, lb in [(180, 'prone'), (90, 'left'), (0, 'supine'), (-90, 'right'),
                   (-180, 'prone')]:
        ax2.axhline(lv, color='#555', ls=':', lw=0.7, zorder=3)
        ax2.annotate(lb, (0.997, lv), xycoords=('axes fraction', 'data'),
                     fontsize=9, color='#444', va='bottom', ha='right')
    ax2.set_ylim(-195, 195)
    ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax2.set_ylabel('Head turn (deg)', color=HEAD_COLOR, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=HEAD_COLOR, labelsize=10)
    ax2.grid(True, axis='y', alpha=0.18, lw=0.6)
    return ax2


def draw_spectrogram(ax, sig, fs, fig, label):
    """0-SPEC_FMAX Hz spectrogram of one channel with an inset dB colorbar."""
    fr, tsp, Sxx = sp_spectrogram(sig, fs=fs, nperseg=2048, noverlap=1536,
                                  nfft=4096, scaling='density')
    fmask = fr <= SPEC_FMAX
    Sdb = 10 * np.log10(Sxx[fmask] + 1e-30)
    vmin, vmax = np.nanpercentile(Sdb, [5, 97])
    pcm = ax.pcolormesh(tsp / 3600.0, fr[fmask], Sdb, shading='gouraud',
                        cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_ylabel(f'{label}\nFreq (Hz)')
    ax.set_ylim(0, SPEC_FMAX)
    trans = ax.get_yaxis_transform()
    for yb in (RESP_LO, RESP_HI, CARD_HI):
        ax.axhline(yb, color='white', ls='--', lw=0.6, alpha=0.55)
    bbox = dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1.2)
    ax.text(0.012, (RESP_LO + RESP_HI) / 2, 'Resp', transform=trans, color='white',
            fontsize=8, fontweight='bold', va='center', ha='left', bbox=bbox)
    ax.text(0.012, (CARD_LO + CARD_HI) / 2, 'Cardiac', transform=trans, color='white',
            fontsize=8, fontweight='bold', va='center', ha='left', bbox=bbox)
    cax = inset_axes(ax, width='1.4%', height='85%', loc='center left',
                     bbox_to_anchor=(1.005, 0., 1, 1), bbox_transform=ax.transAxes,
                     borderpad=0)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(f'PSD (dB re 1 {CAP_UNIT_SQ}/Hz)', fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _caption(ax, txt, y=0.95, va='top'):
    ax.text(0.006, y, txt, transform=ax.transAxes, fontsize=8, style='italic',
            color='#555', va=va,
            bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.5),
            zorder=7)


def plot_pair(s, feats, raw, chans, out, title, shared_level_axis):
    """
    One figure for a pair of channels.

    shared_level_axis
        True  -> both levels on one axis (CLE/CRE are the same physical quantity
                 at comparable scale, so a shared axis is the honest view).
        False -> independent twin axes (CH and CLE-CRE differ in scale by up to
                 11x, so sharing would flatten one of them).
    Either way ZERO -- the session mean -- sits at the centre of the axis.
    """
    sp = s.sleep_profile
    t = feats['t_hr']
    c0, c1 = chans
    f0, f1 = feats[c0], feats[c1]
    is_diff = c0 in DIFF_CHANNELS and c1 in DIFF_CHANNELS

    fig, axes = plt.subplots(
        7, 1, figsize=(14, 15.2), sharex=True,
        gridspec_kw={'height_ratios': [0.55, 1.25, 0.85, 1.25, 1.30, 0.95, 0.95]})

    # A -- hypnogram ladder
    draw_ladder(axes[0], sp, title)
    panel_letter(axes[0], 0)

    # B -- mean-centred levels, zero at the centre of the axis
    ax = axes[1]
    stage_shading(ax, sp)
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.0, zorder=2)
    if shared_level_axis:
        ax.plot(t, f0['centred'], lw=1.0, color=CH_COLOR[c0], label=c0, zorder=3)
        ax.plot(t, f1['centred'], lw=1.0, color=CH_COLOR[c1], label=c1, zorder=3)
        ax.set_ylim(*sym_zero_ylim(f0['centred'], f1['centred']))
        ax.set_ylabel(f'\u0394 from session\nmean ({CAP_UNIT})')
        ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    else:
        ax.plot(t, f0['centred'], lw=1.1, color=CH_COLOR[c0], zorder=4)
        ax.set_ylim(*sym_zero_ylim(f0['centred']))
        ax.set_ylabel(f'{c0} \u2212 mean\n({CAP_UNIT})', color=CH_COLOR[c0])
        ax.tick_params(axis='y', labelcolor=CH_COLOR[c0])
        axr = ax.twinx()
        axr.plot(t, f1['centred'], lw=1.1, color=CH_COLOR[c1], alpha=0.9, zorder=3)
        axr.set_ylim(*sym_zero_ylim(f1['centred']))
        axr.set_ylabel(f'{c1} \u2212 mean ({CAP_UNIT})', color=CH_COLOR[c1])
        axr.tick_params(axis='y', labelcolor=CH_COLOR[c1])
    _caption(ax, f'zero = session mean ({c0} {f0["mu"]:,.0f}, '
                 f'{c1} {f1["mu"]:,.0f} {CAP_UNIT}) \u00b7 axis symmetric about '
                 f'zero, scaled to the bulk \u2014 big steps overflow',
             y=0.045, va='bottom')
    ax.grid(True, alpha=0.15)
    panel_letter(ax, 1)

    # C -- variance
    ax = axes[2]
    stage_shading(ax, sp)
    ax.plot(t, f0['var'], lw=0.8, color=CH_COLOR[c0], alpha=0.9)
    ax.plot(t, f1['var'], lw=0.8, color=CH_COLOR[c1], alpha=0.9)
    ax.set_ylabel(f'Variance\n({CAP_UNIT_SQ})')
    ax.set_ylim(*robust_ylim(np.concatenate([f0['var'], f1['var']]), floor0=True))
    ax.grid(True, alpha=0.15)
    panel_letter(ax, 2)

    # D -- flow
    ax = axes[3]
    stage_shading(ax, sp)
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.0, zorder=2)
    if is_diff:
        ln0, = ax.plot(t, f0['flow_dir'], lw=2.0, color=CH_COLOR[c0],
                       label=f'{c0} direction', zorder=4)
        ln1, = ax.plot(t, f1['flow_dir'], lw=1.6, color=CH_COLOR[c1],
                       label=f'{c1} direction', alpha=0.9, zorder=3)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel('Flow direction\n[\u22121,+1]')
        axm = ax.twinx()
        axm.plot(t, f0['flow_mag'], lw=1.1, color=MAG_COLOR, ls='--', alpha=0.75,
                 zorder=2)
        ax.legend(handles=[ln0, ln1,
                           plt.Line2D([], [], color=MAG_COLOR, ls='--', lw=1.1,
                                      label=f'{c0} magnitude (right axis)')],
                  loc='lower left', fontsize=8.5, ncol=3, framealpha=0.9)
        axm.set_ylim(0, max(np.nanpercentile(f0['flow_mag'], 99) * 1.15, 1e-3))
        axm.set_ylabel(f'{c0} flow magnitude ({CAP_UNIT})', color=MAG_COLOR,
                       fontsize=9.5)
        axm.tick_params(axis='y', labelcolor=MAG_COLOR, labelsize=9)
        d0, d1 = f0['flow_dir'], f1['flow_dir']
        ok = np.isfinite(d0) & np.isfinite(d1)
        r = float(np.corrcoef(d0[ok], d1[ok])[0, 1]) if ok.sum() > 20 else np.nan
        agree = (float(np.mean(np.sign(d0[ok]) == np.sign(d1[ok])))
                 if ok.sum() > 20 else np.nan)
        lv = np.isfinite(f0['centred']) & np.isfinite(f1['centred']) & feats['in_sleep']
        gain = (float(np.polyfit(f0['centred'][lv], f1['centred'][lv], 1)[0])
                if lv.sum() > 20 else np.nan)
        _caption(ax, f'flow marker, still being tuned \u00b7 low pass '
                     f'{TAU_MIN:.0f} min, motion excluded \u00b7 + = left-dominant '
                     f'\u00b7 {c1} vs {c0}: gain {gain:+.2f} (1 if identical), '
                     f'direction r = {r:+.2f}, same sign in {agree * 100:.0f}% of blocks',
                 y=0.95, va='top')
    else:
        ax.plot(t, f0['flow_lp'], lw=1.6, color=CH_COLOR[c0], label=c0, zorder=3)
        ax.plot(t, f1['flow_lp'], lw=1.6, color=CH_COLOR[c1], label=c1, zorder=3)
        ax.set_ylim(*sym_zero_ylim(f0['flow_lp'], f1['flow_lp']))
        ax.set_ylabel(f'Slow level\nLP{TAU_MIN:.0f}min ({CAP_UNIT})')
        ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
        _caption(ax, f'{TAU_MIN:.0f} min low pass of each mean-centred level '
                     f'\u00b7 a signed flow direction needs a difference channel, '
                     f'so it is not defined for {c0} or {c1} alone \u2014 see the '
                     f'CH / CLE\u2212CRE figure')
    ax.grid(True, alpha=0.15)
    panel_letter(ax, 3)

    # E -- head turn + motion (enlarged)
    ax = axes[4]
    stage_shading(ax, sp)
    draw_head_row(ax, t, feats['turn_deg'], feats['motion'])
    ax.grid(True, alpha=0.15)
    panel_letter(ax, 4)

    # F, G -- spectrograms, one per channel
    for k, ch in enumerate((c0, c1)):
        draw_spectrogram(axes[5 + k], raw[ch], s.fs, fig, ch)
        panel_letter(axes[5 + k], 5 + k)
    axes[-1].set_xlabel('Time (hours)')

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


def run_session(label):
    idx = next((i for i, m in enumerate(SESSION_META) if m['label'] == label), None)
    if idx is None:
        print(f'  unknown session {label}'); return []
    s = load_session(idx)
    sp = load_sleep_profile(s)
    if sp is None:
        print(f'  {label}: no sleep profile, skipping'); return []
    s.sleep_profile = sp
    feats, raw = compute_features(s)
    outs = []
    for chans, tag, shared in ((('CLE', 'CRE'), 'CLE_CRE', True),
                               (('CLE-CRE', 'CH'), 'CH_CLE-CRE', False)):
        title = (f'{s.label} \u2014 {chans[0]} and {chans[1]}: overnight evolution '
                 f'of the sensor value')
        out = plot_pair(s, feats, raw, chans, PLOT_DIR / f'{label}_{tag}.png',
                        title, shared)
        outs.append(out)
        print(f'  {label} {tag:12s} -> {out.name}')
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='S1N1')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    labels = [m['label'] for m in SESSION_META] if args.all else [args.session]
    for lbl in labels:
        run_session(lbl)
    print(f'\nFigures -> {PLOT_DIR}')


if __name__ == '__main__':
    main()
