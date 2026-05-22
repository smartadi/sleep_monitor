"""
Plot full-night and short-window time series for all 12 overnight sessions.

Outputs per session:
  - Full-night overview: cap channels (CH, CLE, CRE) + PSG (Thorax, Pleth, EEG)
  - Short windows (30s each): early / mid / late night zoomed views
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sleep_monitor.loader import load_session
from sleep_monitor.config import CAP_CHANNELS, FS

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "plots" / "session_timeseries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAP_COLORS = {"CH": "#2980B9", "CLE": "#27AE60", "CRE": "#8E44AD"}
PSG_SIGS = [("Thorax", "#27AE60"), ("Pleth", "#E74C3C"), ("EEG", "#3498DB")]

WINDOW_SEC = 30.0
WINDOW_SAMPLES = int(WINDOW_SEC * FS)


def zsc(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


# ── Full-night plot ───────────────────────────────────────────────────────────

def plot_full_night(sess, save=True):
    t_hr = sess.time_hr
    ds = max(1, len(t_hr) // 8000)
    t = t_hr[::ds]

    fig, axes = plt.subplots(6, 1, figsize=(18, 14), sharex=True,
                             gridspec_kw={"hspace": 0.3})

    for ax, ch in zip(axes[:3], ["CH", "CLE", "CRE"]):
        sig = sess.cap[ch][::ds]
        ax.plot(t, sig, lw=0.3, color=CAP_COLORS[ch], alpha=0.8)
        ax.set_ylabel(ch, fontsize=9)
        ax.grid(True, alpha=0.2)
        ylo, yhi = np.nanpercentile(sig, [1, 99])
        margin = (yhi - ylo) * 0.1
        ax.set_ylim(ylo - margin, yhi + margin)

    ax_acc = axes[3]
    acc_mag = sess.cap["acc_mag"][::ds]
    ax_acc.plot(t, acc_mag, lw=0.3, color="#E67E22", alpha=0.8)
    ax_acc.set_ylabel("Acc mag", fontsize=9)
    ax_acc.grid(True, alpha=0.2)

    for i, (sig_name, color) in enumerate(PSG_SIGS[:2]):
        ax = axes[4 + i]
        sig = sess.psg[sig_name][::ds]
        ax.plot(t, sig, lw=0.3, color=color, alpha=0.8)
        ax.set_ylabel(sig_name, fontsize=9)
        ax.grid(True, alpha=0.2)
        ylo, yhi = np.nanpercentile(sig, [1, 99])
        margin = (yhi - ylo) * 0.1
        ax.set_ylim(ylo - margin, yhi + margin)

    axes[-1].set_xlabel("Time (hr)", fontsize=10)
    m = sess.meta
    fig.suptitle(
        f"{m['label']}  {m['subject']}-{m['initials']}  {m['date']}  "
        f"({sess.duration_hr:.1f} hr, {sess.n_samples:,} samples)",
        fontsize=13, fontweight="bold",
    )

    if save:
        out = OUT_DIR / f"{m['label']}_full_night.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out.name}")
    plt.close(fig)
    return fig


# ── Short window plot ─────────────────────────────────────────────────────────

def plot_short_window(sess, start_hr, save=True, tag=""):
    start_idx = int(start_hr * 3600 * FS)
    end_idx = start_idx + WINDOW_SAMPLES
    if end_idx > sess.n_samples:
        end_idx = sess.n_samples
        start_idx = max(0, end_idx - WINDOW_SAMPLES)

    t_s = np.arange(end_idx - start_idx) / FS

    cap_chans = ["CH", "CLE", "CRE"]
    psg_chans = [("Thorax", "#27AE60"), ("Pleth", "#E74C3C"), ("EEG", "#3498DB")]
    n_rows = len(cap_chans) + len(psg_chans)

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.2 * n_rows), sharex=True,
                             gridspec_kw={"hspace": 0.35})

    for i, ch in enumerate(cap_chans):
        sig = sess.cap[ch][start_idx:end_idx]
        axes[i].plot(t_s, sig, lw=0.6, color=CAP_COLORS[ch], alpha=0.9)
        axes[i].set_ylabel(ch, fontsize=9)
        axes[i].grid(True, alpha=0.25)

    for i, (sig_name, color) in enumerate(psg_chans):
        ax = axes[len(cap_chans) + i]
        sig = sess.psg[sig_name][start_idx:end_idx]
        ax.plot(t_s, sig, lw=0.6, color=color, alpha=0.9)
        ax.set_ylabel(sig_name, fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time in window (s)", fontsize=10)
    m = sess.meta
    t0_min = start_hr * 60
    t1_min = t0_min + WINDOW_SEC / 60
    fig.suptitle(
        f"{m['label']}  {m['subject']}-{m['initials']}  —  "
        f"{t0_min:.1f}–{t1_min:.1f} min ({tag})",
        fontsize=12, fontweight="bold",
    )

    if save:
        out = OUT_DIR / f"{m['label']}_window_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out.name}")
    plt.close(fig)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sleep_monitor.sessions import SESSION_META

    for i in range(len(SESSION_META)):
        sess = load_session(i)

        print(f"\n{'='*60}")
        print(f" {sess.label}  ({sess.duration_hr:.1f} hr)")
        print(f"{'='*60}")

        plot_full_night(sess)

        dur_hr = sess.duration_hr
        windows = {
            "early": dur_hr * 0.10,
            "mid":   dur_hr * 0.50,
            "late":  dur_hr * 0.85,
        }
        for tag, hr in windows.items():
            plot_short_window(sess, hr, tag=tag)

    print(f"\nAll plots saved to {OUT_DIR}")
