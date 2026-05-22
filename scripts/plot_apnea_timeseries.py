"""
Plot apnea/hypopnea event timeseries for all 12 overnight sessions.

For each session produces a 3-row figure:
  1. Hypnogram (sleep stages)
  2. Apnea event timeline (coloured spans: red=Apnea, orange=Hypopnea)
  3. Event density (events per 15-min bin)

Outputs to notebooks/plots/apnea/<label>_apnea.png
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sleep_monitor.loader import load_session, load_sleep_profile, load_apnea_events
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER,
    APNEA_LABELS, APNEA_COLORS,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "plots" / "apnea"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_MIN = 15.0


def plot_apnea_session(sess, save=True):
    ev = sess.apnea_events
    sp = sess.sleep_profile
    has_sp = sp is not None
    has_ev = ev is not None and len(ev['codes']) > 0

    n_rows = 2 + int(has_sp)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 2.4 * n_rows), sharex=True,
        gridspec_kw={"hspace": 0.35, "height_ratios": [1] * (n_rows - 1) + [1.2]},
    )
    ax_idx = 0

    # ── Row 1: Hypnogram ─────────────────────────────────────────────────────
    if has_sp:
        ax = axes[ax_idx]; ax_idx += 1
        t = sp['t_ep_hr']
        codes = sp['codes']
        for i in range(len(t) - 1):
            c = int(codes[i])
            clr = STAGE_COLORS.get(c, '#AAAAAA')
            ax.fill_between([t[i], t[i + 1]], c, c, color=clr, alpha=0.85, linewidth=0)
            ax.plot([t[i], t[i + 1]], [c, c], color=clr, lw=1.5)
        ax.set_yticks(STAGE_ORDER)
        ax.set_yticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], fontsize=7)
        ax.set_ylabel("Sleep stage", fontsize=8)
        ax.set_ylim(-0.5, 4.5)
        ax.grid(True, alpha=0.25)

    # ── Row 2: Apnea event spans ─────────────────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1
    if has_ev:
        for s_hr, e_hr, code in zip(ev['start_hr'], ev['end_hr'], ev['codes']):
            c = int(code)
            ax.axvspan(s_hr, e_hr, color=APNEA_COLORS[c], alpha=0.7)
        n_apnea = int((ev['codes'] == 1).sum())
        n_hypo = int((ev['codes'] == 2).sum())
        ax.text(
            0.01, 0.92,
            f"Apnea: {n_apnea}   Hypopnea: {n_hypo}   Total: {n_apnea + n_hypo}",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'),
        )
    else:
        ax.text(0.5, 0.5, "No apnea events", transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='#999')
    patches = [mpatches.Patch(color=APNEA_COLORS[c], label=APNEA_LABELS[c])
               for c in (1, 2)]
    ax.legend(handles=patches, fontsize=7, loc='upper right')
    ax.set_yticks([])
    ax.set_ylabel("Events", fontsize=8)
    ax.grid(True, axis='x', alpha=0.25)

    # ── Row 3: Event density histogram ───────────────────────────────────────
    ax = axes[ax_idx]
    dur_hr = sess.duration_hr
    bin_hr = BIN_MIN / 60.0
    bin_edges = np.arange(0, dur_hr + bin_hr, bin_hr)
    if has_ev:
        mid_hr = (ev['start_hr'] + ev['end_hr']) / 2.0
        is_apnea = ev['codes'] == 1
        is_hypo = ev['codes'] == 2
        ax.hist(mid_hr[is_apnea], bins=bin_edges, color=APNEA_COLORS[1],
                alpha=0.8, label='Apnea')
        ax.hist(mid_hr[is_hypo], bins=bin_edges, color=APNEA_COLORS[2],
                alpha=0.6, label='Hypopnea', bottom=np.histogram(mid_hr[is_apnea], bins=bin_edges)[0])
    ax.set_ylabel(f"Events / {int(BIN_MIN)} min", fontsize=8)
    ax.set_xlabel("Time (hr)", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.25)

    m = sess.meta
    fig.suptitle(
        f"{m['label']}  {m['subject']}-{m['initials']}  {m['date']}  "
        f"({dur_hr:.1f} hr)",
        fontsize=12, fontweight="bold",
    )

    if save:
        out = OUT_DIR / f"{m['label']}_apnea.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out.name}")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    for i in range(len(SESSION_META)):
        sess = load_session(i)
        sess.sleep_profile = load_sleep_profile(sess)
        sess.apnea_events = load_apnea_events(sess)

        print(f"\n{sess.label}  ({sess.duration_hr:.1f} hr)")
        plot_apnea_session(sess)

    print(f"\nAll plots saved to {OUT_DIR}")
