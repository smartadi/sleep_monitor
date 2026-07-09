"""
Show ladders on the channel spectrogram + quantify them from the ridge detector.

Key idea (per discussion): a "ladder" is NOT required to be integer harmonics.
It is a set of concurrent persistent ridges that are roughly EQUALLY SPACED in
frequency (rung spacing Δf). We quantify each window's ladder with a comb fit on
the detected ridge frequencies:

  find the spacing Δf and offset `base` such that the most ridges fall on the
  grid  f_k = base + k·Δf  (within tolerance).

Outputs per window:
  n_rungs     - how many ridges lie on the best comb
  df_hz       - rung spacing Δf (the "ladder step")
  regularity  - fraction of active ridges explained by the comb (0..1)
  harmonic    - True if the comb passes through ~0 (Δf ≈ fundamental,
                i.e. the classic integer-harmonic special case); False if the
                comb is offset (an inharmonic / shifted ladder)

Figure (per session):
  Row 0  hypnogram
  Row 1  full-night CH spectrogram + detected ridges
  Row 2  zoomed spectrogram on a strong-ladder segment + comb grid overlay
  Row 3  ladder quantification traces (n_rungs, Δf, regularity)

Run:
  python ladder_spectrogram.py                 # default session S6N2
  python ladder_spectrogram.py S2N2 --zoom 4.4 5.1
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
from run_ridge_overlay import (
    prepare_signals, compute_fine_spectrogram, MAX_FREQ,
    WIN_SEC, STEP_SEC, SMOOTH_WINDOWS, MIN_PERSIST_SEC, MAX_FREQ_JUMP,
    PEAK_PROM_FRAC, MAX_GAP_WINDOWS, WELCH_SEG_SEC,
)
from sleep_monitor.harmonics import detect_persistent_ridges

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'harmonic_rigor'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

_LABEL_TO_IDX = {m['label']: m['idx'] for m in SESSION_META}

# comb-fit parameters
DF_MIN, DF_MAX, DF_STEP = 0.15, 1.6, 0.01
COMB_TOL = 0.04          # Hz — how close a ridge must sit to a comb line
MIN_RUNGS = 3
PROM_MIN = 2.0           # keep only ridges >= 2x local spectral floor (drop clutter)


def comb_fit(freqs, tol=COMB_TOL, df_lo=DF_MIN, df_hi=DF_MAX, max_min_k=99):
    """See below. max_min_k caps the lowest rung index so a comb cannot be fit
    to high harmonics only (the ladder must include a k<=max_min_k rung, i.e. a
    real low fundamental); set to 2 for band-isolated respiratory/cardiac fits."""
    return _comb_fit(freqs, tol, df_lo, df_hi, max_min_k)


def _comb_fit(freqs, tol=COMB_TOL, df_lo=DF_MIN, df_hi=DF_MAX, max_min_k=99):
    """
    Best equally-spaced comb through a set of ridge frequencies.

    A real ladder FILLS its span with consecutive rungs, so we score each
    candidate spacing by  n_rungs × coverage , where coverage = rungs / grid
    positions spanned. This rewards a comb whose rungs occupy most of the grid
    slots between its lowest and highest member, and penalises dense small-Δf
    combs that only catch scattered ridges by chance.

    ``df_lo``/``df_hi`` restrict the rung spacing Δf — use the respiratory band
    (~0.12–0.5 Hz) to isolate breathing-harmonic ladders and the cardiac band
    (~0.5–1.6 Hz) to isolate heartbeat-harmonic ladders. Rungs themselves may
    still extend across the full 0–5 Hz range.

    Returns dict(df, base, n_rungs, coverage, regularity, harmonic, members, ks).
    Δf is data-driven; integer-harmonic is the special case base≈0.
    """
    freqs = np.sort(np.asarray([f for f in freqs if f >= 0.1], dtype=float))
    out = dict(df=np.nan, base=np.nan, n_rungs=len(freqs), coverage=0.0,
               regularity=0.0, harmonic=False, fundamental=np.nan, resid=np.nan,
               members=[], ks=[])
    if len(freqs) < MIN_RUNGS:
        return out
    best_score = 0.0
    df_grid = np.arange(df_lo, df_hi + 1e-9, DF_STEP)
    for dfc in df_grid:
        residuals = np.mod(freqs, dfc)              # candidate offsets = ridge residuals
        for off in residuals:
            d = np.abs(((freqs - off + dfc / 2) % dfc) - dfc / 2)
            on = d < tol
            n = int(on.sum())
            if n < MIN_RUNGS:
                continue
            ks = np.round((freqs[on] - off) / dfc).astype(int)
            # reject high-harmonic-only fits: the ladder must include a low rung
            # (else a resp-band spacing can spuriously fit cardiac ridges at k>>1)
            if ks.min() > max_min_k:
                continue
            span_slots = int(ks.max() - ks.min()) + 1     # grid positions spanned
            coverage = n / span_slots                       # 1.0 = every rung present
            score = n * coverage
            if score > best_score:
                best_score = score
                members = freqs[on]
                # refine Δf and offset by least squares on the actual rung freqs
                # (removes grid quantization; f_k = base + k·Δf)
                slope, base0 = np.polyfit(ks.astype(float), members, 1)
                base = base0 % slope if slope > 0 else 0.0
                harm = min(base, slope - base) < tol        # comb passes through ~0
                fundamental = float(base0 + ks.min() * slope)  # lowest rung freq
                resid = float(np.sqrt(np.mean(
                    (members - (base0 + ks * slope)) ** 2)))
                out = dict(df=float(slope), base=float(base), n_rungs=n,
                           coverage=float(coverage),
                           regularity=float(n / len(freqs)), harmonic=bool(harm),
                           fundamental=fundamental, resid=resid,
                           members=members.tolist(), ks=ks.tolist())
    return out


def quantify_ladders(rr):
    """Per-window comb quantification from concurrent persistent ridges."""
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    n_win = len(t_hr)
    n_rungs = np.zeros(n_win, int)
    df_hz = np.full(n_win, np.nan)
    regularity = np.zeros(n_win)
    harmonic = np.zeros(n_win, bool)
    n_active = np.zeros(n_win, int)
    combs = [None] * n_win
    for i in range(n_win):
        if rr['motion_mask'][i]:
            continue
        # use only prominent ridges (>= PROM_MIN x local floor) — the real rungs;
        # this is the detector's own prominence trace, not an integer assumption
        active = []
        for r in ridges:
            f = r['freq_trace'][i]
            if not np.isfinite(f):
                continue
            pt = r.get('prominence_trace')
            prom = pt[i] if (pt is not None and np.isfinite(pt[i])) else 0.0
            if prom >= PROM_MIN:
                active.append(f)
        n_active[i] = len(active)
        c = comb_fit(active)
        combs[i] = c
        if c['n_rungs'] >= MIN_RUNGS and c['regularity'] > 0:
            n_rungs[i] = c['n_rungs']
            df_hz[i] = c['df']
            regularity[i] = c['regularity']
            harmonic[i] = c['harmonic']
    return dict(t_hr=t_hr, n_rungs=n_rungs, df_hz=df_hz, regularity=regularity,
                harmonic=harmonic, n_active=n_active, combs=combs)


def _plot_spec_with_ridges(ax, sig, rr, xlim=None, comb_at=None):
    t_spec, f_spec, Sxx_db = compute_fine_spectrogram(sig, fs=FS, max_freq=MAX_FREQ)
    vmin, vmax = np.nanpercentile(Sxx_db, [5, 95])
    ax.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='inferno',
                  vmin=vmin, vmax=vmax, rasterized=True)
    t_hr = rr['t_hr']
    for ridge in rr['ridges']:
        valid = ~np.isnan(ridge['freq_trace'])
        if valid.sum() < 2:
            continue
        ax.plot(t_hr[valid], ridge['freq_trace'][valid], '-', color='#00E5FF',
                lw=1.0, alpha=0.7, zorder=3)
    if comb_at is not None:
        t0, comb = comb_at
        if comb is not None and comb['n_rungs'] >= MIN_RUNGS:
            for f in comb['members']:
                ax.plot(t0, f, '>', color='#FFEB3B', ms=9, zorder=5)
    ax.set_ylim(0, MAX_FREQ)
    ax.set_ylabel('Freq (Hz)')
    if xlim:
        ax.set_xlim(*xlim)


def make_figure(label, zoom=None):
    idx = _LABEL_TO_IDX[label]
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    sp = session.sleep_profile
    signals, acc_mag = prepare_signals(session)
    sig = signals['CH']
    rr = detect_persistent_ridges(
        sig, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC, max_freq=MAX_FREQ,
        smooth_windows=SMOOTH_WINDOWS, min_persistence_sec=MIN_PERSIST_SEC,
        max_freq_jump=MAX_FREQ_JUMP, peak_prominence_frac=PEAK_PROM_FRAC,
        max_gap_windows=MAX_GAP_WINDOWS, welch_seg_sec=WELCH_SEG_SEC, acc_mag=acc_mag)
    q = quantify_ladders(rr)

    # choose zoom: the window with the most comb rungs (strongest ladder) if not given
    if zoom is None:
        wi = int(np.argmax(q['n_rungs']))
        zc = q['t_hr'][wi]
        zoom = (max(0, zc - 0.35), zc + 0.35)
    # representative comb to draw = strongest-rung window inside zoom
    inzoom = (q['t_hr'] >= zoom[0]) & (q['t_hr'] <= zoom[1])
    if inzoom.any():
        zi = np.where(inzoom)[0][np.argmax(q['n_rungs'][inzoom])]
        comb_at = (q['t_hr'][zi], q['combs'][zi])
    else:
        comb_at = None

    fig, axes = plt.subplots(4, 1, figsize=(20, 14),
                             gridspec_kw={'height_ratios': [0.35, 1.6, 1.6, 1.0]})

    # Row 0: hypnogram
    ax = axes[0]
    for j in range(len(sp['t_ep_hr']) - 1):
        c = int(sp['codes'][j])
        ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                   color=STAGE_COLORS.get(c, '#AAA'), alpha=0.6)
    ax.set_yticks([]); ax.set_ylabel('Stage')
    ax.legend(handles=[mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
                       for c in STAGE_ORDER], loc='upper right', fontsize=7, ncol=5)
    ax.set_title(f'{label} — CH spectrogram, detected ridges (cyan), and comb-fit ladder quantification',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, session.duration_hr)

    # Row 1: full-night spectrogram + ridges
    _plot_spec_with_ridges(axes[1], sig, rr, xlim=(0, session.duration_hr))
    axes[1].axvspan(zoom[0], zoom[1], color='white', alpha=0.0, ec='#FFEB3B', lw=2)
    axes[1].text(0.005, 0.97, 'full night — cyan = persistent ridges',
                 transform=axes[1].transAxes, color='white', fontsize=9, va='top',
                 bbox=dict(fc='black', alpha=0.6, pad=0.3))

    # Row 2: zoomed spectrogram + ridges + comb grid markers
    _plot_spec_with_ridges(axes[2], sig, rr, xlim=zoom, comb_at=comb_at)
    if comb_at and comb_at[1] and comb_at[1]['n_rungs'] >= MIN_RUNGS:
        c = comb_at[1]
        tag = 'harmonic (Δf≈f₀)' if c['harmonic'] else 'inharmonic (offset comb)'
        axes[2].text(0.005, 0.97,
                     f"zoom — comb: f₀={c['fundamental']:.2f} Hz, Δf={c['df']:.3f} Hz, "
                     f"{c['n_rungs']} rungs, cov={c['coverage']:.2f}, {tag}  (yellow ▶ = rungs)",
                     transform=axes[2].transAxes, color='white', fontsize=9, va='top',
                     bbox=dict(fc='black', alpha=0.6, pad=0.3))

    # Row 3: quantification traces
    ax = axes[3]
    tt = q['t_hr']
    ax.fill_between(tt, 0, q['n_rungs'], color='#2980B9', alpha=0.25, step='mid')
    ax.plot(tt, q['n_rungs'], color='#2980B9', lw=0.9, label='n rungs')
    ax.set_ylabel('n rungs', color='#2980B9'); ax.set_ylim(0, None)
    ax.set_xlim(0, session.duration_hr); ax.set_xlabel('Time (hr)')
    ax2 = ax.twinx()
    ax2.plot(tt, q['df_hz'], '.', color='#E67E22', ms=3, alpha=0.6, label='Δf (Hz)')
    # mark harmonic vs inharmonic windows
    harm = q['harmonic'] & (q['n_rungs'] >= MIN_RUNGS)
    inh = (~q['harmonic']) & (q['n_rungs'] >= MIN_RUNGS)
    ax2.plot(tt[harm], q['df_hz'][harm], 'o', color='#27AE60', ms=3, alpha=0.5)
    ax2.set_ylabel('Δf spacing (Hz)  |  green=harmonic', color='#E67E22')
    ax2.set_ylim(0, 1.2)
    ax.legend(loc='upper left', fontsize=8)

    frac_harm = harm.sum() / max((q['n_rungs'] >= MIN_RUNGS).sum(), 1)
    fig.tight_layout()
    out = REPORT_DIR / f'ladder_spectrogram_{label}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'{label}: {(q["n_rungs"]>=MIN_RUNGS).sum()} ladder windows | '
          f'{frac_harm:.0%} harmonic (df~f0), {1-frac_harm:.0%} inharmonic | '
          f'median df={np.nanmedian(q["df_hz"]):.3f} Hz | '
          f'median rungs={int(np.nanmedian(q["n_rungs"][q["n_rungs"]>=MIN_RUNGS]))}')
    print(f'saved -> {out}')
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default='S6N2')
    ap.add_argument('--zoom', nargs=2, type=float, default=None)
    args = ap.parse_args()
    make_figure(args.session, zoom=tuple(args.zoom) if args.zoom else None)


if __name__ == '__main__':
    main()
