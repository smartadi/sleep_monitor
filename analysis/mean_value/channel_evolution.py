"""
Per-session channel "evolution" panels (journal figures).

The point of these figures: show how the raw sensor value evolves across a night,
and how that evolution carries a directional capacitance imbalance. The marker is
still being tuned — it is drawn here as the current best definition, not a
settled one.

Panels carry numbers, not prose: anything that needs a sentence to explain
belongs in this docstring, not on top of the data.

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
    C  imbalance      sits directly under the mean it is derived from.
                      Difference figure: the signed 30-min low pass of the
                      mean-centred difference, LP(d), for BOTH channels. Fill
                      colour encodes SIGN (red = left-dominant, blue = right),
                      line colour encodes CHANNEL, and the dashed outline is the
                      +/-LP(|d|) envelope. Keeping the two encodings separate
                      means a sign disagreement between the channels shows up as
                      red and blue in the same column instead of being hidden.
                      Absolute figure: the same low pass of each level. A signed
                      imbalance direction needs a difference channel, so it is not
                      defined for CLE or CRE on their own.
    D  variance       within-window variance (fF^2)
    E  head angle     head TURN angle from the accelerometer (0 supine, +90 left,
                      -90 right), on an axis scaled to the movement that actually
                      happened on that night.
    F,G  rate panels   DIFFERENCE figure only, one per channel: the respiratory
                      and cardiac VITERBI rate traces over a background-removed
                      0-3 Hz spectrogram, with the tracker's own confidence
                      printed. This is the same extraction that produces
                      writeup/figures/harmonics/ridges/ridge_tune_*.png —
                      imported from analysis/slow_wave/ridge_overlay_tune.py, not
                      reimplemented, so the two agree exactly.
    F,G  spectrograms  ABSOLUTE figure: plain 0-5 Hz spectrogram per channel.

Why the rate traces are Viterbi paths
-------------------------------------
Respiration and heart rate each have exactly ONE value at any instant, so each
is tracked as a single globally-optimal continuous path (maximise spectral
amplitude minus a smoothness penalty, searched inside a physiological band). The
generic multi-ridge detector is wrong for them: it returns a scatter of
fragments, lets two "cardiac" ridges coexist at one instant, and lets a ridge
hop to a subharmonic. The slow band is NOT drawn here -- the slow physics is
already row C's subject, in fF rather than as a spectrum.

CH is the sensor's own hardware difference channel and is NOT the arithmetic
CLE-CRE: across the cohort the offset is ~-742 fF and the gain of CH on CLE-CRE
runs from -1.9 to +6.5, changing sign between subjects (ch_vs_clecre_sessions.py).
Row C of the difference figure prints how often the two difference channels
agree in sign and their correlation.

Two display choices that earlier versions got wrong
---------------------------------------------------
* The imbalance row used to show a NORMALISED direction, LP(d)/LP(|d|), in
  [-1,+1].
  It saturates flat against +/-1 whenever the sign is consistent, which reads as
  clipping rather than as "one-sided", and it needed a second axis in different
  units for the magnitude. The signed low pass carries the same information with
  one unit and no rails.
* The head-angle row used to share space with accelerometer activity on a fixed
  +/-180 axis. Motion is already implicit (every step in the angle is a
  movement), and on a night spent within a few tens of degrees of supine the
  fixed axis flattened every real posture change. Motion is gone; the axis
  follows the data with a floor of HEAD_MIN_SPAN_DEG.

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
import matplotlib.patches as mpatches
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
# One definition of the capacitance-imbalance marker, shared with
# imbalance_marker.py so the two figures cannot drift apart.
from imbalance_marker import imbalance_marker, TAU_MIN, DESPIKE_MIN
# Same for the rate tracker: the Viterbi tracker, its per-rhythm search bands
# and the background-removal are imported from the tuned ridge overlay rather
# than restated here, so the rate panels of this figure and the figures under
# writeup/figures/harmonics/ridges/ cannot disagree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'slow_wave'))
from sleep_monitor.preprocessing import remove_acc_artifact
import ridge_overlay_tune as rot
from ridge_overlay_tune import (
    TRACK, BAND_COLOR as RIDGE_COLOR, track_single_ridge, _enhance_spec,
)
rot.VERBOSE = False        # this figure reports ridge counts on the panel instead

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'writeup' / 'figures' / 'channel_evolution'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CLE', 'CRE', 'CH', 'CLE-CRE']
DIFF_CHANNELS = ['CLE-CRE', 'CH']       # channels a signed imbalance direction exists for
CH_COLOR = {'CLE': CAP_COLORS['CLE'], 'CRE': CAP_COLORS['CRE'],
            'CH': CAP_COLORS['CH'], 'CLE-CRE': CAP_COLORS['CLE-CRE']}
CH_LONG = {'CLE': 'left temple (CLE)', 'CRE': 'right temple (CRE)',
           'CH': 'hardware difference (CH)',
           'CLE-CRE': 'arithmetic difference (CLE−CRE)'}
# The imbalance row draws each channel in its OWN colour, so it needs no separate
# magnitude/direction colours -- those existed only while the two quantities were
# split across different axes.
HEAD_COLOR = '#16A085'
POS_COLOR  = '#C0392B'      # imbalance above zero: left-dominant
NEG_COLOR  = '#2980B9'      # imbalance below zero: right-dominant
MOTION_PCT = 90            # top-decile accelerometer activity counts as motion
HEAD_MIN_SPAN_DEG = 25.0   # floor on the head-angle axis span, so a still
                           # night does not zoom in on sensor noise

# Hypnogram ladder: stage code -> vertical position. Ordered by STRENGTH OF SLEEP
# DEPTH, so descending the ladder is monotonically deeper sleep: Wake, N1, N2, N3,
# then REM on its own rung at the bottom. (REM is placed below N3 rather than in
# the conventional slot next to Wake, so the NREM descent reads as a clean depth
# gradient and REM does not interrupt it.)
LADDER_ORDER = [4, 3, 2, 1, 0]                                  # Wake, N1, N2, N3, REM
LADDER_Y = {c: len(LADDER_ORDER) - 1 - i for i, c in enumerate(LADDER_ORDER)}

BLOCK_SEC = 10.0            # window for mean / variance / motion (display rows)
LP_CAP_HZ = 10.0           # low-pass cap applied to raw signal before mean/variance
SPEC_FMAX = 5.0            # plain-spectrogram top frequency (Hz)
RIDGE_FMAX = 3.0           # rate-panel top frequency (resp + cardiac)

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
        # Same marker as imbalance_marker.py, evaluated on this figure's 10 s blocks.
        A, D, d_lp = imbalance_marker(d, is_motion, TAU_MIN, DESPIKE_MIN,
                                 block_min=BLOCK_SEC / 60.0)
        feats[ch] = {'mean': mean_b, 'centred': d, 'mu': mu, 'var': var_b,
                     'imb_mag': A, 'imb_dir': D, 'imb_lp': d_lp}
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


def draw_imbalance_row(ax, t, feats, chans, shared_axis):
    """
    Imbalance, drawn identically for both channels.

    Every channel goes through the SAME computation in compute_features(): the
    same 10 s block mean, the same session-mean centring, and the same
    imbalance_marker() call with the same time constants. Nothing about CH is
    computed differently from CLE-CRE — where they differ, that is signal.

    The previous version of this row did not honour that. It drew the SIGNED
    direction for both channels but the magnitude for only one of them, in a
    third colour belonging to neither, and put direction on a normalised
    [-1,+1] scale where it sat railed at +/-1 for long stretches (the ratio
    LP(d)/LP|d| saturates whenever the sign is steady, so the rails carried no
    information and read as clipping).

    Unified presentation, one definition per channel, both in femtofarads:
        fill  = LP(d) about zero, RED above / BLUE below — the sign, i.e. which
                side leads, readable at a glance without reading the axis
        line  = LP(d) in the channel's own colour — which channel
        band  = +/-LP|d| as a dashed outline — the envelope of the imbalance
    Two encodings kept separate on purpose: colour of the FILL means sign,
    colour of the LINE means channel. Where the two channels disagree in sign
    the row shows red and blue in the same column, which is the disagreement
    made visible rather than hidden.

    POLARITY: + on CLE-CRE means the left temple reads higher. CH is the
    sensor's own difference channel and its polarity is NOT assumed to match --
    the gain of CH on CLE-CRE changes sign between subjects, so the panel
    reports the measured sign agreement rather than asserting a shared meaning.
    """
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.0, zorder=3)
    handles = []
    axes_for = {}
    for k, ch in enumerate(chans):
        f = feats[ch]
        a = ax if (shared_axis or k == 0) else ax.twinx()
        axes_for[ch] = a
        col = CH_COLOR[ch]
        y = f['imb_lp']
        a.fill_between(t, 0, y, where=(y >= 0), interpolate=True,
                       color=POS_COLOR, alpha=0.28 if k == 0 else 0.16,
                       lw=0, zorder=1)
        a.fill_between(t, 0, y, where=(y < 0), interpolate=True,
                       color=NEG_COLOR, alpha=0.28 if k == 0 else 0.16,
                       lw=0, zorder=1)
        for sgn in (1, -1):
            a.plot(t, sgn * f['imb_mag'], lw=0.9, ls='--', color=col,
                   alpha=0.55, zorder=2)
        ln, = a.plot(t, y, lw=2.2 if k == 0 else 1.7, color=col,
                     zorder=5 - k, label=ch)
        handles.append(ln)
        if not shared_axis:
            a.set_ylim(*sym_zero_ylim(f['imb_lp'], f['imb_mag'], -f['imb_mag']))
            a.set_ylabel(f'{ch} imbalance\n({CAP_UNIT})' if k == 0
                         else f'{ch} imbalance ({CAP_UNIT})', color=col)
            a.tick_params(axis='y', labelcolor=col)
    if shared_axis:
        ax.set_ylim(*sym_zero_ylim(*[feats[c]['imb_lp'] for c in chans],
                                   *[feats[c]['imb_mag'] for c in chans],
                                   *[-feats[c]['imb_mag'] for c in chans]))
        ax.set_ylabel(f'Slow imbalance\n({CAP_UNIT})')
    handles += [mpatches.Patch(color=POS_COLOR, alpha=0.35, label='left-dominant'),
                mpatches.Patch(color=NEG_COLOR, alpha=0.35, label='right-dominant'),
                plt.Line2D([], [], color='#666', ls='--', lw=0.9,
                           label='±LP|Δ|')]
    ax.legend(handles=handles, loc='lower left', fontsize=8, ncol=5,
              framealpha=0.9)
    return axes_for


def draw_head_row(ax, t, turn):
    """
    Head-turn angle alone, scaled to the movement that actually happened.

    Accelerometer activity used to share this row, but it is already implicit —
    every step in the angle is a movement — and having it there forced the angle
    onto a twin axis at fixed +/-180. Most nights the head stays within a few
    tens of degrees of supine, so a fixed +/-180 axis rendered the real posture
    changes as a nearly flat line. The axis now follows the data (with a floor,
    so a motionless night does not zoom into sensor noise) and only the posture
    guides that fall inside the view are drawn.
    """
    # Re-wrap about the night's typical posture BEFORE scaling. turn lives on a
    # circle, so a head resting near -90 that briefly crosses the seam yields
    # both -180 and +180 samples, and any spread taken over the raw values then
    # spans the whole circle. That is what pinned this axis at the full range no
    # matter how little the head actually moved (S6N1 reported a 380 deg span for
    # a night that only occupies -90 to +100).
    turn = np.asarray(turn, float)
    fin = np.isfinite(turn)
    med = np.degrees(np.arctan2(np.median(np.sin(np.radians(turn[fin]))),
                                np.median(np.cos(np.radians(turn[fin])))))
    turn = ((turn - med + 180.0) % 360.0) - 180.0 + med

    ax.plot(t, turn, lw=1.7, color=HEAD_COLOR, zorder=4)

    # Then scale to the sustained posture, letting brief excursions overflow --
    # the same treatment the raw-value rows give a re-seat step. A night that
    # genuinely spends hours on each side has a wide 2-98% range and still gets
    # a wide axis; a single roll-over does not stretch the axis and flatten the
    # real +/-30 deg changes.
    good = turn[np.isfinite(turn)]
    lo, hi = np.percentile(good, [2, 98]) if good.size > 3 else (-30.0, 30.0)
    span = max(hi - lo, HEAD_MIN_SPAN_DEG)
    mid = 0.5 * (lo + hi)
    lo, hi = mid - span / 2, mid + span / 2
    pad = 0.12 * span
    lo, hi = max(lo - pad, -190.0), min(hi + pad, 190.0)
    ax.set_ylim(lo, hi)
    n_over = int(np.sum((good < lo) | (good > hi)))

    for lv, lb in [(180, 'prone'), (90, 'left'), (0, 'supine'), (-90, 'right'),
                   (-180, 'prone')]:
        if lo < lv < hi:
            ax.axhline(lv, color='#555', ls=':', lw=0.8, zorder=3)
            ax.annotate(lb, (0.997, lv), xycoords=('axes fraction', 'data'),
                        fontsize=9, color='#444', va='bottom', ha='right')
    step = next(s for s in (5, 10, 15, 30, 45, 90) if span / s <= 9)
    ax.set_yticks(np.arange(np.ceil(lo / step) * step,
                            np.floor(hi / step) * step + 1, step))
    ax.set_ylabel('Head turn\n(deg)', color=HEAD_COLOR, fontsize=11)
    ax.tick_params(axis='y', labelcolor=HEAD_COLOR, labelsize=10)
    ax.grid(True, axis='y', alpha=0.18, lw=0.6)
    over = (f' · {n_over * BLOCK_SEC / 60:.0f} min of brief excursions off-axis'
            if n_over else '')
    return ax


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


def compute_ridges(sig, acc, fs):
    """
    The tuned ridge extraction from analysis/slow_wave/ridge_overlay_tune.py.

    Respiration and heart rate are each ONE value at any instant, so they are
    tracked as a single continuous VITERBI path per rhythm (globally optimal
    trace maximising spectral amplitude minus a smoothness penalty, searched
    inside a physiological band). The generic multi-ridge persistent-ridge
    detector is wrong for them: it returns a scatter of fragments, lets two
    "cardiac" ridges coexist at one instant, and lets a ridge hop to a
    subharmonic. The slow band keeps the multi-ridge detector, because several
    slow oscillations genuinely can coexist there, plus its prominence gate.

    Everything -- the tracker, the band definitions, the prominence gate and the
    accelerometer-regressed preprocessing -- is imported from that module rather
    than restated, so these panels stay identical to the ridge figures under
    writeup/figures/harmonics/ridges/.
    """
    clean = remove_acc_artifact(np.asarray(sig, float), np.asarray(acc, float),
                                0.05, 4.0)
    out = {}
    for rhythm in ('resp', 'card'):
        t, tr, conf = track_single_ridge(clean, fs, **TRACK[rhythm])
        out[rhythm] = {'t_hr': t, 'freq': tr, 'conf': conf}

    out['clean'] = clean
    return out


def draw_rate_panel(ax, ridges, fs, fig, label):
    """
    Background-removed 0-3 Hz spectrogram with the respiratory and cardiac
    Viterbi rate traces on top -- the primary mechanism panel.

    Two things differ from an ordinary spectrogram panel, both taken from
    ridge_overlay_tune.py:

    * the spectrogram is BACKGROUND-REMOVED (each time column minus a
      frequency-median-filtered copy of itself), which flattens the 1/f envelope
      and broadband motion brightening while leaving narrow peaks intact. The
      rate bands stand out instead of drowning in the low-frequency slope.
    * the overlay is a single continuous Viterbi trace per rhythm, not a set of
      ridge fragments. Respiration and heart rate each have exactly one value at
      any instant, so one trace is the honest representation.

    The tracker's own confidence -- the fraction of windows where the tracked
    band is clearly peaked above its column median -- is printed, so a weak
    night is visible as a number rather than having to be inferred.
    """
    f, t, db = _enhance_spec(ridges['clean'], fs, 30, 15, RIDGE_FMAX, bg_hz=0.4)
    vmax = np.percentile(db, 99.5)
    pcm = ax.pcolormesh(t / 3600.0, f, db, shading='gouraud', cmap='magma',
                        vmin=0, vmax=vmax, rasterized=True)
    for rhythm, name in (('resp', 'resp rate'), ('card', 'cardiac rate')):
        r = ridges[rhythm]
        ax.plot(r['t_hr'], r['freq'], color=RIDGE_COLOR[rhythm], lw=2.0,
                alpha=0.95, zorder=4,
                label=f'{name}  (conf {r["conf"]:.0%})')
    ax.set_ylim(0, RIDGE_FMAX)
    ax.set_ylabel(f'{label}\nFreq (Hz)')
    ax.legend(loc='upper right', fontsize=8.5, ncol=2, framealpha=0.85)
    cax = inset_axes(ax, width='1.4%', height='85%', loc='center left',
                     bbox_to_anchor=(1.005, 0., 1, 1), bbox_transform=ax.transAxes,
                     borderpad=0)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label('dB above background', fontsize=8)
    cb.ax.tick_params(labelsize=7)


def plot_pair(s, feats, raw, chans, out, title, shared_level_axis, ridges=None):
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
        7, 1, figsize=(14, 16.4), sharex=True,
        gridspec_kw={'height_ratios': [0.55, 1.25, 1.35, 0.80, 1.15, 1.40, 1.40]})

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
        ax.set_ylabel(f'\u0394 from mean\n({CAP_UNIT})')
        ax.legend([f'{c0}  \u03bc {f0["mu"]:,.0f}', f'{c1}  \u03bc {f1["mu"]:,.0f}'],
                  loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    else:
        ax.plot(t, f0['centred'], lw=1.1, color=CH_COLOR[c0], zorder=4)
        ax.set_ylim(*sym_zero_ylim(f0['centred']))
        ax.set_ylabel(f'{c0} \u2212 mean\n({CAP_UNIT};  \u03bc {f0["mu"]:,.0f})',
                      color=CH_COLOR[c0])
        ax.tick_params(axis='y', labelcolor=CH_COLOR[c0])
        # CH is deliberately not drawn here: two channels of different scale on
        # twin axes made each one's step structure harder to read. CH remains in
        # the variance row and keeps its own spectrogram panel below.
    ax.grid(True, alpha=0.15)
    panel_letter(ax, 1)

    # C -- imbalance, directly under the mean it is derived from
    #
    # Everything on this row is in fF on ONE signed axis. The previous version
    # put a normalised direction LP(d)/LP(|d|) in [-1,+1] on the left, a
    # magnitude in fF on the right, and two channels across both -- three traces,
    # two units, and a normalised trace that saturates flat against +/-1 for
    # hours at a time, which reads as clipping rather than as "consistently
    # one-sided". Replaced by the signed low-passed differential itself, filled
    # two-colour about zero, inside the +/-magnitude envelope. Sign is the
    # direction, height is the strength, and the old normalised direction is
    # still readable as how close the line sits to the envelope edge.
    ax = axes[2]
    stage_shading(ax, sp)
    draw_imbalance_row(ax, t, feats, chans, shared_axis=shared_level_axis)
    if is_diff:
        g0, g1 = f0['imb_lp'], f1['imb_lp']
        ok = np.isfinite(g0) & np.isfinite(g1)
        r = float(np.corrcoef(g0[ok], g1[ok])[0, 1]) if ok.sum() > 20 else np.nan
        agree = (float(np.mean(np.sign(g0[ok]) == np.sign(g1[ok])))
                 if ok.sum() > 20 else np.nan)
        lv = np.isfinite(f0['centred']) & np.isfinite(f1['centred']) & feats['in_sleep']
        gain = (float(np.polyfit(f0['centred'][lv], f1['centred'][lv], 1)[0])
                if lv.sum() > 20 else np.nan)
        ax.text(0.994, 0.95, f'gain {gain:+.2f} · r {r:+.2f} · '
                f'{agree * 100:.0f}% same sign',
                transform=ax.transAxes, fontsize=8.5, color='#444', va='top',
                ha='right', zorder=7,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))
    ax.grid(True, alpha=0.15)
    panel_letter(ax, 2)

    # D -- variance
    ax = axes[3]
    stage_shading(ax, sp)
    ax.plot(t, f0['var'], lw=0.8, color=CH_COLOR[c0], alpha=0.9, label=c0)
    ax.plot(t, f1['var'], lw=0.8, color=CH_COLOR[c1], alpha=0.9, label=c1)
    ax.set_ylabel(f'Variance\n({CAP_UNIT_SQ})')
    ax.set_ylim(*robust_ylim(np.concatenate([f0['var'], f1['var']]), floor0=True))
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    panel_letter(ax, 3)

    # E -- head turn angle, scaled to the night's actual movement
    ax = axes[4]
    stage_shading(ax, sp)
    draw_head_row(ax, t, feats['turn_deg'])
    panel_letter(ax, 4)

    # F, G -- one bottom panel per channel. The difference channels get the
    # tracked respiratory/cardiac ridges over their spectrogram; the absolute
    # channels keep the plain spectrogram.
    for k, ch in enumerate((c0, c1)):
        if ridges is not None:
            draw_rate_panel(axes[5 + k], ridges[ch], s.fs, fig, ch)
        else:
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
    acc = s.cap['acc_mag'].astype(np.float64)
    for chans, tag, shared in ((('CLE', 'CRE'), 'CLE_CRE', True),
                               (('CLE-CRE', 'CH'), 'CH_CLE-CRE', False)):
        title = (f'{s.label} \u2014 {chans[0]} and {chans[1]}: overnight evolution '
                 f'of the sensor value')
        # Ridge panels only on the difference figure: those are the channels the
        # imbalance story is about, and ridge detection is the expensive step here.
        ridges = ({ch: compute_ridges(raw[ch], acc, s.fs) for ch in chans}
                  if tag == 'CH_CLE-CRE' else None)
        out = plot_pair(s, feats, raw, chans, PLOT_DIR / f'{label}_{tag}.png',
                        title, shared, ridges=ridges)
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
