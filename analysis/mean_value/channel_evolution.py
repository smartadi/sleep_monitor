"""
Per-session, per-channel "evolution" panel (journal figure): mean value,
variance, smoothed velocity, motion, and a spectrogram — one figure per channel.

Reporting figure for the manuscript. For each session and each recorded channel
(CLE, CRE, CH) we draw a stacked multi-row panel so the slow evolution of the
baseline and its dynamics is visible across the night:

    A  hypnogram strip (PSG stage colour band)
    B  mean value        low-pass (<10 Hz) DC level (a.u.), 10 s windows
    C  variance          low-pass (<10 Hz) within-window variance (a.u.^2)
    D  smoothed velocity d(mean)/dt of the smoothed baseline (a.u./hour)
    E  motion            accelerometer activity (within-window std)
    F  spectrogram       0-5 Hz spectrogram of that channel + dB colorbar

The raw channel is low-pass filtered at 10 Hz (zero-phase Butterworth) before the
mean and variance are computed, so those series reflect physiological-band
content and are not inflated by the >10 Hz electronic noise floor. Every data row
is autoscaled to robust (1-99th pct) limits of ITS OWN series so the evolution
fills the axis rather than being squished by a few outliers.

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/channel_evolution.py --all
    .venv/Scripts/python.exe analysis/mean_value/channel_evolution.py --session S1N1

Outputs
-------
    writeup/figures/channel_evolution/<SESSION>_<channel>.png
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
from scipy.ndimage import uniform_filter1d, binary_dilation
from scipy.signal import butter, filtfilt, spectrogram as sp_spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, CAP_COLORS,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI,
)
from sleep_monitor.sessions import SESSION_META

ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = ROOT / 'writeup' / 'figures' / 'channel_evolution'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CLE', 'CRE', 'CH']
CH_COLOR = {'CLE': CAP_COLORS['CLE'], 'CRE': CAP_COLORS['CRE'], 'CH': CAP_COLORS['CH']}
CH_LONG = {'CLE': 'left temple (CLE)', 'CRE': 'right temple (CRE)',
           'CH': 'differential (CH)'}

BLOCK_SEC = 10.0            # window for mean / variance / motion
LP_CAP_HZ = 10.0           # low-pass cap applied to raw signal before mean/variance
BASELINE_SMOOTH_MIN = 3.0  # smoothing of the mean before differentiating (velocity)
VEL_SMOOTH_MIN = 2.0       # extra smoothing of the velocity trace
SPEC_FMAX = 5.0            # spectrogram top frequency (Hz)
MOTION_PCTL = 90.0         # accel blocks above this per-session pct are removed
MOTION_DILATE = 2          # dilate the accel-motion mask by +/- this many blocks
JUMP_K = 8.0               # robust (MAD) threshold for a baseline step artifact
JUMP_DILATE = 18           # blank velocity +/- this many blocks around a big step
                           # (covers the smoothing spread of the step, ~3 min)

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


def block_reduce(x, n, fn):
    m = len(x) // n
    return fn(x[: m * n].reshape(m, n), axis=1)


def lowpass(x, fs, fc):
    b, a = butter(4, fc / (0.5 * fs), btype='low')
    return filtfilt(b, a, x)


def compute_features(s):
    fs = s.fs
    raw = {ch: s.cap[ch].astype(np.float64) for ch in CHANNELS}
    acc = s.cap['acc_mag'].astype(np.float64)

    n = int(round(fs * BLOCK_SEC))
    m = min(min(len(v) for v in raw.values()), len(acc)) // n
    t_hr = (np.arange(m) + 0.5) * BLOCK_SEC / 3600.0

    hop_per_min = 60.0 / BLOCK_SEC
    base_win = max(3, int(round(BASELINE_SMOOTH_MIN * hop_per_min)))
    vel_win = max(3, int(round(VEL_SMOOTH_MIN * hop_per_min)))
    dt_hr = BLOCK_SEC / 3600.0

    motion = block_reduce(acc[: m * n], n, np.std)
    # accel-motion mask: top-decile per session, lightly dilated.
    accel_mask = binary_dilation(motion > np.nanpercentile(motion, MOTION_PCTL),
                                 iterations=MOTION_DILATE)

    feats = {'t_hr': t_hr, 'motion': motion, 'motion_flag': accel_mask}
    for ch, sig in raw.items():
        filt = lowpass(sig, fs, LP_CAP_HZ)          # <10 Hz cap before mean/var
        mean_b = block_reduce(filt[: m * n], n, np.mean)
        var_b = block_reduce(filt[: m * n], n, np.var)
        base = uniform_filter1d(mean_b, base_win, mode='nearest')
        vel = np.gradient(base, dt_hr)
        vel = uniform_filter1d(vel, vel_win, mode='nearest')
        # Non-artifact velocity: blank both accel-motion epochs AND large baseline
        # STEP artifacts (electrode coupling loss/regain — NOT accelerometer motion,
        # so they must be detected from the mean itself). A robust MAD threshold on
        # the block-to-block mean change flags the steps; dilate widely so the
        # smoothing bump around each step is fully removed from the display.
        dmean = np.abs(np.diff(mean_b, prepend=mean_b[:1]))
        mad = np.median(np.abs(dmean - np.median(dmean))) + 1e-9
        jump_mask = binary_dilation(dmean > JUMP_K * 1.4826 * mad,
                                    iterations=JUMP_DILATE)
        vel = vel.copy()
        vel[accel_mask | jump_mask] = np.nan
        feats[ch] = {'mean': mean_b, 'var': var_b, 'vel': vel}
    return feats, raw


def stage_shading(ax, sp, alpha=0.10):
    tep, codes = sp['t_ep_hr'], sp['codes']
    for j in range(len(tep) - 1):
        ax.axvspan(tep[j], tep[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                   alpha=alpha, lw=0, zorder=0)


def panel_letter(ax, k):
    ax.text(-0.055, 1.02, PANEL[k], transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='right')


def plot_channel(s, ch, feats, raw, out):
    sp = s.sleep_profile
    t = feats['t_hr']
    col = CH_COLOR[ch]
    f = feats[ch]

    fig, axes = plt.subplots(
        6, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={'height_ratios': [0.16, 1.0, 1.0, 1.0, 0.7, 1.1]})

    # A — hypnogram strip
    ax = axes[0]
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                   color=STAGE_COLORS.get(c, '#AAA'), alpha=0.85)
    ax.set_yticks([]); ax.set_ylabel('Stage', fontsize=10)
    ax.legend(handles=[mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
                       for c in STAGE_ORDER], loc='upper right', ncol=5,
              framealpha=0.9, handlelength=1.0, columnspacing=1.0)
    ax.set_title(f'{s.label} — {CH_LONG[ch]} channel: overnight evolution',
                 fontsize=13, fontweight='bold')
    panel_letter(ax, 0)

    # B — mean value
    ax = axes[1]; stage_shading(ax, sp)
    ax.plot(t, f['mean'], lw=0.9, color=col)
    ax.set_ylabel('Mean value\n(a.u.)'); ax.set_ylim(*robust_ylim(f['mean']))
    ax.grid(True, alpha=0.15); panel_letter(ax, 1)

    # C — variance
    ax = axes[2]; stage_shading(ax, sp)
    ax.plot(t, f['var'], lw=0.8, color='#C0392B')
    ax.set_ylabel('Variance\n(a.u.²)')
    ax.set_ylim(*robust_ylim(f['var'], floor0=True))
    ax.grid(True, alpha=0.15); panel_letter(ax, 2)

    # D — smoothed velocity (motion epochs removed before differentiation)
    ax = axes[3]; stage_shading(ax, sp)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.plot(t, f['vel'], lw=0.9, color='#2980B9')
    ax.set_ylabel('Baseline velocity\n(a.u./hr)')
    ax.set_ylim(*robust_ylim(f['vel'], symmetric=True))
    ax.text(0.006, 0.9, 'motion & step artifacts removed', transform=ax.transAxes,
            fontsize=8, style='italic', color='#555', va='top')
    ax.grid(True, alpha=0.15); panel_letter(ax, 3)

    # E — motion
    ax = axes[4]; stage_shading(ax, sp)
    ax.plot(t, feats['motion'], lw=0.7, color='#7F8C8D')
    ax.fill_between(t, 0, feats['motion'], color='#7F8C8D', alpha=0.3)
    ax.set_ylabel('Motion\n(acc. std)')
    ax.set_ylim(*robust_ylim(feats['motion'], floor0=True))
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
    cb.set_label('PSD (dB)', fontsize=8.5); cb.ax.tick_params(labelsize=7.5)

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
        print(f'  {label} {ch:4s} -> {out.name}')
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
