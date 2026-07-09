"""
Side-by-side band-isolated ladder view on the spectrogram.

For a session, shows CH and CRE spectrograms with:
  - persistent ridges (faint grey)
  - RESPIRATORY-band ladder rungs (cyan)   — comb Δf 0.12-0.50 Hz
  - CARDIAC-band ladder rungs (magenta)    — comb Δf 0.50-1.60 Hz
each drawn at its member frequencies for every window where detected, so a
sustained ladder appears as a coloured horizontal band. Bottom panel shows
when each family is present per channel.

Run:
  python ladder_bands_spectrogram.py            # default S6N2
  python ladder_bands_spectrogram.py S2N2 --zoom 4.4 5.1
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sleep_monitor import load_session, load_sleep_profile, FS
from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.harmonics import detect_persistent_ridges
from run_ridge_overlay import (
    prepare_signals, compute_fine_spectrogram, MAX_FREQ, WIN_SEC, STEP_SEC,
    SMOOTH_WINDOWS, MIN_PERSIST_SEC, MAX_FREQ_JUMP, PEAK_PROM_FRAC,
    MAX_GAP_WINDOWS, WELCH_SEG_SEC,
)
from ladder_spectrogram import comb_fit, PROM_MIN, MIN_RUNGS

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'ladder_quantify'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

_LABEL_TO_IDX = {m['label']: m['idx'] for m in SESSION_META}
PLOT_CHANNELS = ['CH', 'CRE']
RESP_BAND = (0.12, 0.50)
CARD_BAND = (0.50, 1.60)
RESP_COLOR = '#00E5FF'   # cyan
CARD_COLOR = '#FF3DDA'   # magenta


def _prominent_freqs(rr, i):
    out = []
    for r in rr['ridges']:
        f = r['freq_trace'][i]
        if not np.isfinite(f):
            continue
        pt = r.get('prominence_trace')
        prom = pt[i] if (pt is not None and np.isfinite(pt[i])) else 0.0
        if prom >= PROM_MIN:
            out.append(f)
    return out


def band_ladders(rr):
    """Return per-window member points for resp and cardiac ladders."""
    t_hr = rr['t_hr']
    resp_t, resp_f, card_t, card_f = [], [], [], []
    resp_present = np.zeros(len(t_hr), bool)
    card_present = np.zeros(len(t_hr), bool)
    for i in range(len(t_hr)):
        if rr['motion_mask'][i]:
            continue
        freqs = _prominent_freqs(rr, i)
        if len(freqs) < MIN_RUNGS:
            continue
        cr = comb_fit(freqs, df_lo=RESP_BAND[0], df_hi=RESP_BAND[1], max_min_k=2)
        cc = comb_fit(freqs, df_lo=CARD_BAND[0], df_hi=CARD_BAND[1], max_min_k=2)
        if cr['n_rungs'] >= MIN_RUNGS and cr['coverage'] > 0:
            resp_present[i] = True
            for f in cr['members']:
                resp_t.append(t_hr[i]); resp_f.append(f)
        if cc['n_rungs'] >= MIN_RUNGS and cc['coverage'] > 0:
            card_present[i] = True
            for f in cc['members']:
                card_t.append(t_hr[i]); card_f.append(f)
    return dict(t_hr=t_hr, resp_t=np.array(resp_t), resp_f=np.array(resp_f),
                card_t=np.array(card_t), card_f=np.array(card_f),
                resp_present=resp_present, card_present=card_present)


def _plot_row(ax, sig, rr, bl, ch, xlim):
    t_spec, f_spec, Sxx_db = compute_fine_spectrogram(sig, fs=FS, max_freq=MAX_FREQ)
    vmin, vmax = np.nanpercentile(Sxx_db, [5, 95])
    ax.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='inferno',
                  vmin=vmin, vmax=vmax, rasterized=True)
    for ridge in rr['ridges']:
        valid = ~np.isnan(ridge['freq_trace'])
        if valid.sum() < 2:
            continue
        ax.plot(rr['t_hr'][valid], ridge['freq_trace'][valid], '-',
                color='#888888', lw=0.5, alpha=0.35, zorder=2)
    if len(bl['card_t']):
        ax.plot(bl['card_t'], bl['card_f'], '_', color=CARD_COLOR, ms=5,
                mew=1.4, alpha=0.9, zorder=4)
    if len(bl['resp_t']):
        ax.plot(bl['resp_t'], bl['resp_f'], '_', color=RESP_COLOR, ms=5,
                mew=1.4, alpha=0.9, zorder=5)
    ax.set_ylim(0, MAX_FREQ)
    ax.set_ylabel(f'{ch}\nFreq (Hz)')
    ax.set_xlim(*xlim)
    nresp = int(bl['resp_present'].sum()); ncard = int(bl['card_present'].sum())
    ax.text(0.005, 0.97, f'{ch}: resp ladder {nresp} win | cardiac ladder {ncard} win',
            transform=ax.transAxes, color='white', fontsize=9, va='top',
            bbox=dict(fc='black', alpha=0.6, pad=0.3))


def make_figure(label, zoom=None):
    idx = _LABEL_TO_IDX[label]
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    sp = session.sleep_profile
    signals, acc = prepare_signals(session)

    det, bls = {}, {}
    for ch in PLOT_CHANNELS:
        det[ch] = detect_persistent_ridges(
            signals[ch], fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC, max_freq=MAX_FREQ,
            smooth_windows=SMOOTH_WINDOWS, min_persistence_sec=MIN_PERSIST_SEC,
            max_freq_jump=MAX_FREQ_JUMP, peak_prominence_frac=PEAK_PROM_FRAC,
            max_gap_windows=MAX_GAP_WINDOWS, welch_seg_sec=WELCH_SEG_SEC, acc_mag=acc)
        bls[ch] = band_ladders(det[ch])

    xlim = zoom if zoom else (0, session.duration_hr)
    fig, axes = plt.subplots(4, 1, figsize=(20, 13),
                             gridspec_kw={'height_ratios': [0.35, 1.7, 1.7, 0.7]})

    ax = axes[0]
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                   color=STAGE_COLORS.get(c, '#AAA'), alpha=0.6)
    ax.set_yticks([]); ax.set_ylabel('Stage'); ax.set_xlim(*xlim)
    ax.legend(handles=[mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
                       for c in STAGE_ORDER], loc='upper right', fontsize=7, ncol=5)
    ax.set_title(f'{label} — band-isolated ladders  '
                 f'(cyan = respiratory Δf 0.12–0.5 Hz, magenta = cardiac Δf 0.5–1.6 Hz)',
                 fontsize=13, fontweight='bold')

    _plot_row(axes[1], signals['CH'], det['CH'], bls['CH'], 'CH', xlim)
    _plot_row(axes[2], signals['CRE'], det['CRE'], bls['CRE'], 'CRE', xlim)
    axes[1].legend(handles=[
        mpatches.Patch(color=RESP_COLOR, label='respiratory ladder rung'),
        mpatches.Patch(color=CARD_COLOR, label='cardiac ladder rung'),
        mpatches.Patch(color='#888888', label='persistent ridge')],
        loc='upper right', fontsize=7, ncol=3)

    # occupancy
    ax = axes[3]
    for ci, ch in enumerate(PLOT_CHANNELS):
        bl = bls[ch]
        base = ci * 2
        ax.fill_between(bl['t_hr'], base, base + bl['resp_present'].astype(float) * 0.85,
                        step='mid', color=RESP_COLOR, alpha=0.8)
        ax.fill_between(bl['t_hr'], base + 1, base + 1 + bl['card_present'].astype(float) * 0.85,
                        step='mid', color=CARD_COLOR, alpha=0.8)
    ax.set_yticks([0.4, 1.4, 2.4, 3.4])
    ax.set_yticklabels(['CH resp', 'CH card', 'CRE resp', 'CRE card'], fontsize=8)
    ax.set_xlim(*xlim); ax.set_xlabel('Time (hr)'); ax.set_ylim(0, 4)
    ax.set_title('Ladder occupancy (present = filled)', fontsize=9, loc='left')

    fig.tight_layout()
    suffix = f'_{zoom[0]:.1f}-{zoom[1]:.1f}' if zoom else ''
    out = REPORT_DIR / f'ladder_bands_{label}{suffix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved -> {out}')
    for ch in PLOT_CHANNELS:
        bl = bls[ch]
        print(f'  {ch}: resp {int(bl["resp_present"].sum())} win, '
              f'cardiac {int(bl["card_present"].sum())} win')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default='S6N2')
    ap.add_argument('--zoom', nargs=2, type=float, default=None)
    args = ap.parse_args()
    make_figure(args.session, zoom=tuple(args.zoom) if args.zoom else None)


if __name__ == '__main__':
    main()
