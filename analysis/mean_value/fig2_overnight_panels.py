"""
Manuscript Figure 2 — overnight evolution of the mean-referenced SEC signal.

Four single-night panels, one per participant, ordered by age:

    (a) S6N2  25 y    (b) S3N2  37 y    (c) S4N2  54 y    (d) S2N2  66 y

Three rows per panel, the same three the manuscript caption describes:

    hypnogram          PSG stages as the depth-ordered ladder used everywhere
                       else in this repo (Wake, N1, N2, N3, REM)
    CLE-CRE - mean     the left-right difference referenced to the session mean,
                       in femtofarads, zero at the centre of the axis
    head turn          accelerometer turn angle (0 supine, +90 left, -90 right)

Differences from the panels that were pasted into V4/V5, both requested by the
reviewer on the V4 draft:

* NO STAGE BACKGROUND SHADING. The earlier panels drew a translucent stage
  colour behind rows B and E (channel_evolution.stage_shading). The stage
  information is already in the hypnogram directly above, at full saturation,
  so the wash behind the traces added no information and made the orange and
  green traces harder to read against it. It is gone; the traces sit on white.
* Larger type, and the time axis is actually drawn. In the pasted panels the
  x-axis was cropped away, so the traces had no readable time reference at all.

The panel letter and the participant age are drawn by this script rather than
added to the image afterwards, so the four panels cannot drift out of step with
the caption.

Everything numeric -- the 10 s blocking, the low-pass cap, the sleep-period
mean used as the zero, the head-angle unwrapping and its adaptive axis -- is
imported from analysis/mean_value/channel_evolution.py, not reimplemented, so
this figure and the per-session evolution figures cannot disagree.

Usage
-----
    .venv/Scripts/python.exe analysis/mean_value/fig2_overnight_panels.py

Outputs
-------
    writeup/figures/channel_evolution/fig2_panel_a.png  .. fig2_panel_d.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import CAP_COLORS, CAP_UNIT
from sleep_monitor.sessions import SESSION_META
from channel_evolution import (
    compute_features, draw_ladder, draw_head_row, sym_zero_ylim,
)

PLOT_DIR = ROOT / 'writeup' / 'figures' / 'channel_evolution'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CHAN = 'CLE-CRE'
# (panel letter, session, age). Ordered by age, as the caption reads them.
PANELS = [('a', 'S6N2', 25),
          ('b', 'S3N2', 37),
          ('c', 'S4N2', 54),
          ('d', 'S2N2', 66)]

# Journal styling, sized up from channel_evolution's 11 pt base. These panels are
# reproduced roughly half-page-wide in the manuscript, so the per-session figure's
# type comes out too small to read on paper.
plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 15, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'axes.linewidth': 1.0, 'axes.edgecolor': '#333333',
    'font.family': 'DejaVu Sans', 'figure.dpi': 200,
})


def draw_panel(letter, label, age, out):
    idx = next(i for i, m in enumerate(SESSION_META) if m['label'] == label)
    s = load_session(idx)
    sp = load_sleep_profile(s)
    if sp is None:
        raise SystemExit(f'{label}: no sleep profile')
    s.sleep_profile = sp
    feats, _ = compute_features(s)
    t = feats['t_hr']
    f = feats[CHAN]

    fig, axes = plt.subplots(3, 1, figsize=(15, 8.2), sharex=True,
                             gridspec_kw={'height_ratios': [0.70, 1.25, 1.05]})

    # ── hypnogram ─────────────────────────────────────────────────────────────
    draw_ladder(axes[0], sp, f'{label} — overnight evolution of the sensor value')
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].set_ylabel('Sleep stage', fontsize=15)

    # ── CLE-CRE referenced to the session mean; no stage wash behind it ───────
    ax = axes[1]
    ax.axhline(0, color='#2C3E50', ls='--', lw=1.2, zorder=2)
    ax.plot(t, f['centred'], lw=1.3, color=CAP_COLORS[CHAN], zorder=4)
    ax.set_ylim(*sym_zero_ylim(f['centred']))
    ax.set_ylabel(f'CLE−CRE − mean\n({CAP_UNIT};  μ {f["mu"]:,.0f})',
                  color=CAP_COLORS[CHAN], fontsize=15)
    ax.tick_params(axis='y', labelcolor=CAP_COLORS[CHAN], labelsize=14)
    ax.grid(True, alpha=0.15)

    # ── head turn ─────────────────────────────────────────────────────────────
    draw_head_row(axes[2], t, feats['turn_deg'])
    axes[2].set_ylabel('Head turn\n(deg)', color='#16A085', fontsize=15)
    axes[2].tick_params(axis='y', labelsize=14)
    axes[2].set_xlabel('Time (hours)', fontsize=16)
    axes[2].tick_params(axis='x', labelsize=15)

    # Panel letter and participant, drawn here so they stay tied to the data.
    fig.text(0.008, 0.975, f'({letter})', fontsize=24, fontweight='bold',
             va='top', ha='left')
    axes[0].text(0.998, 1.10, f'{age} years old male', transform=axes[0].transAxes,
                 fontsize=19, va='bottom', ha='right', color='#222222')

    fig.tight_layout(rect=(0.012, 0, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {letter}  {label}  {age}y -> {out.name}')
    return out


def main():
    print('Figure 2 panels (no stage shading, larger type, time axis drawn):')
    for letter, label, age in PANELS:
        draw_panel(letter, label, age, PLOT_DIR / f'fig2_panel_{letter}.png')
    print(f'\n-> {PLOT_DIR}')


if __name__ == '__main__':
    main()
