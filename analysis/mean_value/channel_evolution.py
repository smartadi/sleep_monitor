"""
Per-session channel "evolution" panels (journal figures).

For each session: one figure per channel (CLE, CRE, CH, CLE-CRE) plus a dedicated
CH-vs-CLE-CRE comparison figure.

Per-channel figure rows
-----------------------
    A  hypnogram      PSG stages as a depth-ordered ladder — Wake, N1, N2, N3,
                      then REM — so depth reads from the geometry, not colour
                      alone. Drawn as merged coloured runs with faint connectors,
                      because per-epoch step connectors at full weight swamp the
                      rungs they are meant to join.
    B  mean value     DC level referenced to the SESSION MEAN, in femtofarads,
                      with that mean printed on the panel. Nothing is normalised
                      away; zero is the session's own operating point.
    C  variance       within-window variance (fF^2)
    D  flow           for the difference channels (CLE-CRE, CH): the night-scale
                      flow marker — magnitude A = LP(|d|) and direction
                      D = LP(d)/A in [-1,+1]. For the single-ended channels
                      (CLE, CRE) a signed direction is not defined, so the row
                      shows the same low-passed level instead.
    E  head + motion  head TURN angle from the accelerometer (0 supine, +90 left,
                      -90 right) over accelerometer activity.
    F  spectrogram    0-5 Hz spectrogram of that channel + dB colorbar

Comparison figure (<SESSION>_CH_vs_CLE-CRE.png)
-----------------------------------------------
CH and CLE-CRE are NOT the same signal (offset ~-742 fF, gain -1.9 to +6.5, sign
of the gain flips between subjects — see ch_vs_clecre_sessions.py), so they get a
figure of their own rather than being read across two separate per-channel plots:
mean-centred levels on independent axes, then the two flow directions on the SAME
[-1,+1] axis where they are directly comparable, then magnitude, head angle, and
this session's offset / gain / correlation printed on the panel.

What changed from the previous version of this script
-----------------------------------------------------
* The old row B for CLE-CRE was the raw level filled about its median and called
  "CSF flow". It is now the mean-centred level, with the flow marker in its own
  row D.
* The old row D was a VELOCITY (Savitzky-Golay drift rate, fF/hr). A derivative
  turns every electrode re-seat into a spike and averages to zero over a night,
  so it carried no night-scale direction (30 min lag autocorrelation ~ -0.005 vs
  +0.52 for the marker used now — see flow_marker.py). Replaced by the flow row.
* Row E was motion only; it now carries the head-turn angle validated in
  head_angle_validate.py, using the corrected +/-180 definition and the 0.05 Hz
  gravity cutoff (the previous 0.5 Hz cutoff sat inside the respiratory band).

Units: the CAP columns are true capacitance in femtofarads (see CAP_SCALE_TO_FF
in sleep_monitor/config.py for the scale and its provenance/caveats).

The raw channel is low-pass filtered at 10 Hz (zero-phase Butterworth) before the
mean and variance are computed, so those series reflect physiological-band
content and are not inflated by the >10 Hz electronic noise floor. Every data row
is autoscaled to robust limits of ITS OWN series so the evolution fills the axis
rather than being squished by a few outliers.

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/channel_evolution.py --all
    .venv/Scripts/python.exe analysis/mean_value/channel_evolution.py --session S1N1

Outputs
-------
    writeup/figures/channel_evolution/<SESSION>_<channel>.png
    writeup/figures/channel_evolution/<SESSION>_CH_vs_CLE-CRE.png
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
PANEL = ['A', 'B', 'C', 'D', 'E', 'F']


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
    """Head-turn angle over accelerometer activity."""
    ax.fill_between(t, 0, motion, color='#7F8C8D', alpha=0.35, zorder=1)
    ax.plot(t, motion, lw=0.6, color='#7F8C8D', zorder=2)
    ax.set_ylabel('Motion\n(acc. std)', color='#5D6D7E')
    ax.tick_params(axis='y', labelcolor='#5D6D7E')
    ax.set_ylim(*robust_ylim(motion, floor0=True))
    ax2 = ax.twinx()
    ax2.plot(t, turn, lw=1.3, color=HEAD_COLOR, zorder=4)
    for lv, lb in [(90, 'left'), (0, 'supine'), (-90, 'right')]:
        ax2.axhline(lv, color='#555', ls=':', lw=0.7, zorder=3)
        ax2.annotate(lb, (0.995, lv), xycoords=('axes fraction', 'data'),
                     fontsize=7.5, color='#555', va='bottom', ha='right')
    ax2.set_ylim(-185, 185); ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.set_ylabel('Head turn (deg)', color=HEAD_COLOR)
    ax2.tick_params(axis='y', labelcolor=HEAD_COLOR)
    return ax2


def plot_channel(s, ch, feats, raw, out):
    sp = s.sleep_profile
    t = feats['t_hr']
    col = CH_COLOR[ch]
    f = feats[ch]

    fig, axes = plt.subplots(
        6, 1, figsize=(14, 12.6), sharex=True,
        gridspec_kw={'height_ratios': [0.6, 1.0, 1.0, 1.0, 0.85, 1.1]})

    # A — hypnogram ladder, depth-ordered (Wake top ... N3, REM at the bottom)
    ax = axes[0]
    draw_ladder(ax, sp, f'{s.label} — {CH_LONG[ch]}: overnight evolution')
    panel_letter(ax, 0)

    # B — mean value referenced to the session mean, in fF (mean printed, not hidden)
    ax = axes[1]; stage_shading(ax, sp)
    ax.plot(t, f['centred'], lw=0.9, color=col, zorder=3)
    ax.axhline(0, color='#2C3E50', ls='--', lw=0.9, zorder=2)
    ax.set_ylabel(f'Mean value − session mean\n({CAP_UNIT})')
    ax.set_ylim(*adaptive_ylim(f['centred']))
    ax.text(0.006, 0.93, f'session mean = {f["mu"]:,.0f} {CAP_UNIT} '
            f'(zero of this axis) · y-axis zoomed, large coupling swings clip',
            transform=ax.transAxes, fontsize=8, style='italic', color='#555', va='top', bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.5), zorder=7)
    ax.grid(True, alpha=0.15); panel_letter(ax, 1)

    # C — variance
    ax = axes[2]; stage_shading(ax, sp)
    ax.plot(t, f['var'], lw=0.8, color='#C0392B')
    ax.set_ylabel(f'Variance\n({CAP_UNIT_SQ})')
    ax.set_ylim(*robust_ylim(f['var'], floor0=True))
    ax.grid(True, alpha=0.15); panel_letter(ax, 2)

    # D — night-scale flow marker (difference channels), or the same low pass of
    #     the level where a signed direction is not defined.
    ax = axes[3]; stage_shading(ax, sp)
    if ch in DIFF_CHANNELS:
        draw_flow_row(ax, t, f['flow_mag'], f['flow_dir'], f'flow from {ch}')
    else:
        ax.plot(t, f['flow_lp'], lw=1.6, color=MAG_COLOR, zorder=3)
        ax.axhline(0, color='#2C3E50', ls='--', lw=0.9, zorder=2)
        ax.set_ylabel(f'Slow level\nLP{TAU_MIN:.0f}min ({CAP_UNIT})')
        ax.set_ylim(*adaptive_ylim(f['flow_lp']))
        ax.text(0.006, 0.93, f'{TAU_MIN:.0f} min low pass of the mean-centred level · '
                f'a signed flow direction needs a difference channel, so it is not '
                f'defined for {ch}',
                transform=ax.transAxes, fontsize=8, style='italic', color='#555',
                va='top')
    ax.grid(True, alpha=0.15); panel_letter(ax, 3)

    # E — head turn angle (corrected definition) over motion
    ax = axes[4]; stage_shading(ax, sp)
    draw_head_row(ax, t, feats['turn_deg'], feats['motion'])
    ax.grid(True, alpha=0.15); panel_letter(ax, 4)

    # F — spectrogram (0-SPEC_FMAX Hz) + inset colorbar
    ax = axes[5]
    fr, tsp, Sxx = sp_spectrogram(raw[ch], fs=s.fs, nperseg=2048, noverlap=1536,
                                  nfft=4096, scaling='density')
    fmask = fr <= SPEC_FMAX
    Sdb = 10 * np.log10(Sxx[fmask] + 1e-30)
    vmin, vmax = np.nanpercentile(Sdb, [5, 97])
    pcm = ax.pcolormesh(tsp / 3600.0, fr[fmask], Sdb, shading='gouraud',
                        cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_ylabel('Frequency\n(Hz)'); ax.set_ylim(0, SPEC_FMAX)
    ax.set_xlabel('Time (hours)')
    # band annotations: respiratory 0.1-0.5 Hz, cardiac 0.5-3.0 Hz
    trans = ax.get_yaxis_transform()
    for yb in (RESP_LO, RESP_HI, CARD_HI):
        ax.axhline(yb, color='white', ls='--', lw=0.6, alpha=0.55)
    bbox = dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1.2)
    ax.text(0.012, (RESP_LO + RESP_HI) / 2, 'Resp', transform=trans, color='white',
            fontsize=8, fontweight='bold', va='center', ha='left', bbox=bbox)
    ax.text(0.012, (CARD_LO + CARD_HI) / 2, 'Cardiac', transform=trans, color='white',
            fontsize=8, fontweight='bold', va='center', ha='left', bbox=bbox)
    panel_letter(ax, 5)
    cax = inset_axes(ax, width='1.4%', height='85%', loc='center left',
                     bbox_to_anchor=(1.005, 0., 1, 1), bbox_transform=ax.transAxes,
                     borderpad=0)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(f'PSD (dB re 1 {CAP_UNIT_SQ}/Hz)', fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_ch_vs_diff(s, feats, out):
    """
    The two difference channels side by side, on one time axis.

    They are kept apart from the per-channel figures because they are not the
    same quantity: across the cohort the static offset is ~-742 fF and the gain
    of CH on CLE-CRE runs from -1.9 to +6.5, changing sign between subjects
    (ch_vs_clecre_sessions.py). Row C is the payoff — the two flow DIRECTIONS are
    dimensionless and bounded, so unlike the raw levels they can share one axis
    and be compared directly.
    """
    sp = s.sleep_profile
    t = feats['t_hr']
    fd, fc = feats['CLE-CRE'], feats['CH']
    ok = np.isfinite(fd['centred']) & np.isfinite(fc['centred']) & feats['in_sleep']

    slope = float(np.polyfit(fd['centred'][ok], fc['centred'][ok], 1)[0]) \
        if ok.sum() > 20 else np.nan
    r_lvl = float(np.corrcoef(fc['centred'][ok], fd['centred'][ok])[0, 1]) \
        if ok.sum() > 20 else np.nan
    dd = np.diff(fd['centred'], n=6); dc = np.diff(fc['centred'], n=6)   # 1 min
    mk = np.isfinite(dd) & np.isfinite(dc)
    r_chg = float(np.corrcoef(dc[mk], dd[mk])[0, 1]) if mk.sum() > 20 else np.nan
    dsel = np.isfinite(fd['flow_dir']) & np.isfinite(fc['flow_dir'])
    r_dir = float(np.corrcoef(fc['flow_dir'][dsel], fd['flow_dir'][dsel])[0, 1]) \
        if dsel.sum() > 20 else np.nan
    agree = float(np.mean(np.sign(fc['flow_dir'][dsel]) == np.sign(fd['flow_dir'][dsel]))) \
        if dsel.sum() > 20 else np.nan

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 11.4), sharex=True,
        gridspec_kw={'height_ratios': [0.6, 1.15, 1.0, 1.0, 0.85]})

    ax = axes[0]
    draw_ladder(ax, sp, f'{s.label} — CH vs CLE−CRE: the two difference channels '
                        f'compared')
    panel_letter(ax, 0)

    # B — mean-centred levels, independent axes (they differ in scale)
    ax = axes[1]; stage_shading(ax, sp)
    ax.plot(t, fd['centred'], lw=1.1, color=CH_COLOR['CLE-CRE'], zorder=4)
    ax.axhline(0, color='#2C3E50', ls='--', lw=0.9, zorder=2)
    ax.set_ylabel(f'CLE−CRE − mean\n({CAP_UNIT})', color=CH_COLOR['CLE-CRE'])
    ax.tick_params(axis='y', labelcolor=CH_COLOR['CLE-CRE'])
    ax.set_ylim(*adaptive_ylim(fd['centred']))
    ax2 = ax.twinx()
    ax2.plot(t, fc['centred'], lw=1.1, color=CH_COLOR['CH'], alpha=0.9, zorder=3)
    ax2.set_ylabel(f'CH − mean ({CAP_UNIT})', color=CH_COLOR['CH'])
    ax2.tick_params(axis='y', labelcolor=CH_COLOR['CH'])
    ax2.set_ylim(*adaptive_ylim(fc['centred']))
    ax.text(0.006, 0.95,
            f'means: CLE−CRE {fd["mu"]:,.0f} {CAP_UNIT}, CH {fc["mu"]:,.0f} {CAP_UNIT} '
            f'(offset {fc["mu"] - fd["mu"]:,.0f}) · gain of CH on CLE−CRE = {slope:.2f} '
            f'(1 if identical) · r level {r_lvl:+.2f}, r 1-min change {r_chg:+.2f}',
            transform=ax.transAxes, fontsize=8, style='italic', color='#555', va='top', bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.5), zorder=7)
    ax.grid(True, alpha=0.15); panel_letter(ax, 1)

    # C — the two flow DIRECTIONS on one shared, dimensionless axis
    ax = axes[2]; stage_shading(ax, sp)
    ax.axhline(0, color='#2C3E50', ls='--', lw=0.9, zorder=2)
    ax.plot(t, fd['flow_dir'], lw=1.9, color=CH_COLOR['CLE-CRE'],
            label='CLE−CRE', zorder=4)
    ax.plot(t, fc['flow_dir'], lw=1.5, color=CH_COLOR['CH'], alpha=0.9,
            label='CH', zorder=3)
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel('Flow direction\n[−1,+1]')
    ax.legend(loc='lower left', fontsize=8.5, ncol=2, framealpha=0.9)
    ax.text(0.006, 0.95, f'directions are dimensionless, so they share an axis · '
            f'r = {r_dir:+.2f}, same sign in {agree*100:.0f}% of blocks · '
            f'+ = left-dominant',
            transform=ax.transAxes, fontsize=8, style='italic', color='#555', va='top', bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.5), zorder=7)
    ax.grid(True, alpha=0.15); panel_letter(ax, 2)

    # D — flow magnitudes, independent axes
    ax = axes[3]; stage_shading(ax, sp)
    ax.plot(t, fd['flow_mag'], lw=1.5, color=CH_COLOR['CLE-CRE'], zorder=4)
    ax.set_ylabel(f'CLE−CRE flow mag.\n({CAP_UNIT})', color=CH_COLOR['CLE-CRE'])
    ax.tick_params(axis='y', labelcolor=CH_COLOR['CLE-CRE'])
    ax.set_ylim(0, max(np.nanpercentile(fd['flow_mag'], 99) * 1.15, 1e-3))
    ax2 = ax.twinx()
    ax2.plot(t, fc['flow_mag'], lw=1.3, color=CH_COLOR['CH'], alpha=0.9, zorder=3)
    ax2.set_ylabel(f'CH flow mag. ({CAP_UNIT})', color=CH_COLOR['CH'])
    ax2.tick_params(axis='y', labelcolor=CH_COLOR['CH'])
    ax2.set_ylim(0, max(np.nanpercentile(fc['flow_mag'], 99) * 1.15, 1e-3))
    ax.text(0.006, 0.94, f'magnitudes keep their own axes — CH moves '
            f'{np.nanmedian(fc["flow_mag"]) / max(np.nanmedian(fd["flow_mag"]), 1e-9):.1f}x '
            f'as far on this night',
            transform=ax.transAxes, fontsize=8, style='italic', color='#555', va='top', bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=1.5), zorder=7)
    ax.grid(True, alpha=0.15); panel_letter(ax, 3)

    # E — head turn + motion
    ax = axes[4]; stage_shading(ax, sp)
    draw_head_row(ax, t, feats['turn_deg'], feats['motion'])
    ax.set_xlabel('Time (hours)')
    ax.grid(True, alpha=0.15); panel_letter(ax, 4)

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
    for ch in CHANNELS:
        out = plot_channel(s, ch, feats, raw, PLOT_DIR / f'{label}_{ch}.png')
        outs.append(out)
        print(f'  {label} {ch:8s} -> {out.name}')
    out = plot_ch_vs_diff(s, feats, PLOT_DIR / f'{label}_CH_vs_CLE-CRE.png')
    outs.append(out)
    print(f'  {label} {"compare":8s} -> {out.name}')
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
