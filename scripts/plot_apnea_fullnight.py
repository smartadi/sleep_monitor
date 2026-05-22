"""
Full-night apnea overview plots for all 12 sessions.

6-row time-aligned figure per session:
  1. CLE mean & std (30s epoch)
  2. CRE mean & std
  3. CH mean & std
  4. Apnea events (coloured spans by subtype)
  5. Head movement (roll/pitch + movement RMS)
  6. Spectrogram (CLE-CRE, 0–5 Hz)

Hypnogram strip along top.  Output: notebooks/plots/apnea_analysis/
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram as sp_spectrogram

from sleep_monitor.loader import load_session, load_sleep_profile, load_apnea_events
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import (
    FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER,
)
from sleep_monitor.motion import (
    epoch_motion, epoch_cap_stats, head_orientation, dynamic_acceleration,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "plots" / "apnea_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

APNEA_SUBTYPE_COLORS = {
    'obstructive apnea': '#E74C3C',
    'central apnea':     '#3498DB',
    'mixed apnea':       '#E67E22',
    'hypopnea':          '#F1C40F',
    'apnea':             '#E74C3C',
}
APNEA_SUBTYPE_ORDER = ['obstructive apnea', 'central apnea', 'mixed apnea', 'hypopnea']


def _hypnogram_strip(ax, sp, dur_hr):
    """Draw a thin hypnogram colour strip."""
    if sp is None:
        ax.set_visible(False)
        return
    t = sp['t_ep_hr']
    codes = sp['codes']
    for i in range(len(t) - 1):
        c = int(codes[i])
        ax.axvspan(t[i], t[i + 1], color=STAGE_COLORS.get(c, '#AAA'), alpha=0.85)
    ax.set_yticks(STAGE_ORDER)
    ax.set_yticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], fontsize=6)
    ax.set_ylim(-0.5, 4.5)
    ax.set_ylabel('Stage', fontsize=7)
    ax.tick_params(axis='x', labelbottom=False)


def _cap_row(ax, t_hr, mean, std, ch_name, color):
    """Plot cap channel mean ± std band."""
    ax.plot(t_hr, mean, color=color, lw=0.7, alpha=0.9)
    ax.fill_between(t_hr, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_ylabel(f'{ch_name}\nmean±std', fontsize=7)
    ax.tick_params(axis='x', labelbottom=False)
    ax.grid(True, alpha=0.2)


def _apnea_row(ax, ev, dur_hr):
    """Draw apnea event spans coloured by subtype."""
    if ev is None or len(ev['codes']) == 0:
        ax.text(0.5, 0.5, 'No apnea events', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='#999')
        ax.set_ylabel('Apnea', fontsize=7)
        ax.set_yticks([])
        return

    seen = set()
    for s_hr, e_hr, etype in zip(ev['start_hr'], ev['end_hr'], ev['types']):
        clr = APNEA_SUBTYPE_COLORS.get(etype, '#E74C3C')
        ax.axvspan(s_hr, e_hr, color=clr, alpha=0.75)
        seen.add(etype)

    patches = []
    for st in APNEA_SUBTYPE_ORDER:
        if st in seen:
            patches.append(mpatches.Patch(color=APNEA_SUBTYPE_COLORS[st], label=st.title()))
    if patches:
        ax.legend(handles=patches, fontsize=6, loc='upper right', ncol=len(patches))

    counts = {}
    for etype in ev['types']:
        counts[etype] = counts.get(etype, 0) + 1
    summary = '  '.join(f'{k.title()}: {v}' for k, v in sorted(counts.items()))
    ax.text(0.01, 0.92, summary, transform=ax.transAxes, va='top', fontsize=6,
            bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'))

    ax.set_ylabel('Apnea', fontsize=7)
    ax.set_yticks([])


def _movement_row(ax, motion):
    """Plot roll/pitch and movement RMS."""
    t = motion['t_hr']
    ax.plot(t, motion['roll_deg'], color='#2980B9', lw=0.6, alpha=0.8, label='Roll')
    ax.plot(t, motion['pitch_deg'], color='#27AE60', lw=0.6, alpha=0.8, label='Pitch')
    ax.set_ylabel('Angle (°)', fontsize=7, color='#2C3E50')
    ax.tick_params(axis='y', labelsize=6)

    ax2 = ax.twinx()
    ax2.fill_between(t, 0, motion['movement_rms'], color='#E74C3C', alpha=0.25, label='Move RMS')
    ax2.set_ylabel('Move RMS', fontsize=7, color='#E74C3C')
    ax2.tick_params(axis='y', labelsize=6, colors='#E74C3C')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper right', ncol=3)
    ax.grid(True, alpha=0.2)


def _spectrogram_row(ax, session, fmax=5.0):
    """Plot CLE-CRE spectrogram 0–fmax Hz."""
    cle = session.cap['CLE'].astype(np.float64)
    cre = session.cap['CRE'].astype(np.float64)
    sig = cle - cre
    nperseg = int(10.0 * session.fs)
    noverlap = int(5.0 * session.fs)
    f, t_sec, Sxx = sp_spectrogram(sig, fs=session.fs, nperseg=nperseg,
                                    noverlap=noverlap, window='hann')
    mask = f <= fmax
    Sxx_db = 10.0 * np.log10(Sxx[mask] + 1e-20)
    t_hr = t_sec / 3600.0
    ax.pcolormesh(t_hr, f[mask], Sxx_db, shading='gouraud', cmap='inferno', rasterized=True)
    ax.set_ylabel('Freq (Hz)', fontsize=7)
    ax.set_ylim(0, fmax)
    ax.tick_params(axis='y', labelsize=6)


def plot_fullnight(sess, save=True):
    """Generate the 6-row full-night apnea overview plot."""
    ev = sess.apnea_events
    sp = sess.sleep_profile
    dur_hr = sess.duration_hr

    print(f'  Computing motion + cap stats for {sess.label}...', flush=True)
    motion = epoch_motion(sess)
    cap_stats = epoch_cap_stats(sess)
    t_ep = cap_stats['t_hr']

    has_sp = sp is not None
    n_rows = 6 + int(has_sp)
    height_ratios = ([0.6] if has_sp else []) + [1, 1, 1, 0.8, 1.2, 1.5]

    fig = plt.figure(figsize=(18, 2.2 * n_rows))
    gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.35, height_ratios=height_ratios)

    row = 0

    # Hypnogram strip
    if has_sp:
        ax_hyp = fig.add_subplot(gs[row])
        _hypnogram_strip(ax_hyp, sp, dur_hr)
        row += 1

    # CLE mean ± std
    ax_cle = fig.add_subplot(gs[row])
    _cap_row(ax_cle, t_ep, cap_stats['CLE']['mean'], cap_stats['CLE']['std'], 'CLE', '#27AE60')
    row += 1

    # CRE mean ± std
    ax_cre = fig.add_subplot(gs[row], sharex=ax_cle)
    _cap_row(ax_cre, t_ep, cap_stats['CRE']['mean'], cap_stats['CRE']['std'], 'CRE', '#8E44AD')
    row += 1

    # CH mean ± std
    ax_ch = fig.add_subplot(gs[row], sharex=ax_cle)
    _cap_row(ax_ch, t_ep, cap_stats['CH']['mean'], cap_stats['CH']['std'], 'CH', '#2980B9')
    row += 1

    # Apnea events
    ax_apnea = fig.add_subplot(gs[row], sharex=ax_cle)
    _apnea_row(ax_apnea, ev, dur_hr)
    ax_apnea.tick_params(axis='x', labelbottom=False)
    row += 1

    # Head movement
    ax_move = fig.add_subplot(gs[row], sharex=ax_cle)
    _movement_row(ax_move, motion)
    ax_move.tick_params(axis='x', labelbottom=False)
    row += 1

    # Spectrogram
    ax_spec = fig.add_subplot(gs[row], sharex=ax_cle)
    _spectrogram_row(ax_spec, sess)
    ax_spec.set_xlabel('Time (hr)', fontsize=9)
    ax_spec.set_xlim(0, dur_hr)

    m = sess.meta
    fig.suptitle(
        f"{m['label']}  {m['subject']}-{m['initials']}  {m['date']}  ({dur_hr:.1f} hr)",
        fontsize=13, fontweight='bold', y=0.995,
    )

    if save:
        out = OUT_DIR / f"fullnight_{m['label']}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out.name}')
    plt.close(fig)
    return fig


if __name__ == '__main__':
    for i in range(len(SESSION_META)):
        sess = load_session(i)
        sess.sleep_profile = load_sleep_profile(sess)
        sess.apnea_events = load_apnea_events(sess)

        print(f'\n{sess.label}  ({sess.duration_hr:.1f} hr)')
        plot_fullnight(sess)

    print(f'\nAll plots saved to {OUT_DIR}')
