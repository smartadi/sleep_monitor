"""
Show the low-band (0-3 Hz) CAP spindle response ON DATA, trial by trial.

For one session, every N2 spindle is treated as a trial: a +/-8 s window around
the spindle center. We recompute, per trial, the baseline-corrected low-band
power time course on the forehead channel (CH, the strongest low-band responder)
using the SAME short-time-spectrogram + own-baseline contrast as
`spindle_lowband_detection.py`, alongside the contact-EEG sigma power that marks
where the spindle actually is.

Figure (writeup/figures/spindles/fig_spindle_trials_lowband_<session>.png):
  (A) Onset-triggered AVERAGE CH low-band power +/- SEM over all N2 spindles in
      the session -- the clean population-level bump at the spindle center.
  (B..) A grid of individual spindle trials (sorted strongest-bump first),
      each showing the EEG sigma burst (grey) and the CH low-band response
      (color), with the +/-1 s detection core shaded and the per-trial dB +
      detected/not-detected verdict in the title. Honest: the effect is real on
      average but only ~52-53% of single trials clear their own baseline.

Usage:
  .venv/Scripts/python.exe -m analysis.spindles.plot_spindle_trials_lowband --session S2N1
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from analysis.spindles.spindle_loader import load_spindles

FS = 100.0
N2_CODE = 2
WIN_HALF = 8.0
CORE_HALF = 1.0
BASE_EDGE = 5.0
NPERSEG = 128
NOVERLAP = 96
SIGMA = (11.0, 16.0)
LOW_03 = (0.0, 3.0)
CH_COLOR = '#2980B9'
EEG_COLOR = '#7F8C8D'

FIGDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'writeup', 'figures', 'spindles')
os.makedirs(FIGDIR, exist_ok=True)


def stage_at(t_hr, prof):
    codes, tep = prof['codes'], prof['t_ep_hr']
    out = np.full(len(t_hr), -1, np.int8)
    for i, t in enumerate(t_hr):
        j = np.argmin(np.abs(tep - t))
        if abs(tep[j] - t) < 30.0 / 3600.0:
            out[i] = codes[j]
    return out


def event_traces(sig, centers_samp, band):
    """Per-event baseline-corrected band-power dB(t). Returns (traces[K,T], tcen, core_mask)."""
    n = len(sig)
    traces = []
    tcen = core_t = base_t = fmask = None
    for c in centers_samp:
        a, b = c - int(WIN_HALF * FS), c + int(WIN_HALF * FS) + 1
        if a < 0 or b > n:
            traces.append(None)
            continue
        f, t, Sxx = spectrogram(sig[a:b], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        dB = 10.0 * np.log10(Sxx + 1e-12)
        if tcen is None:
            tcen = t - t[-1] / 2.0
            core_t = np.abs(tcen) < CORE_HALF
            base_t = np.abs(tcen) > BASE_EDGE
            fmask = (f >= band[0]) & (f <= band[1])
        band_dB = dB[fmask].mean(axis=0)
        traces.append(band_dB - band_dB[base_t].mean())
    return traces, tcen, core_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='S2N1')
    ap.add_argument('--ntrials', type=int, default=9)
    args = ap.parse_args()

    idx = next(m['idx'] for m in SESSION_META if m['label'] == args.session)
    s = load_session(idx)
    s.sleep_profile = load_sleep_profile(s)
    sp = load_spindles(s)
    if s.sleep_profile is None or sp is None:
        raise SystemExit(f'{args.session}: missing sleep profile or spindles')

    stg = stage_at(sp['center_hr'], s.sleep_profile)
    cen_hr = sp['center_hr'][stg == N2_CODE]
    cen_samp = np.round(cen_hr * 3600.0 * FS).astype(int)
    print(f'{args.session}: {len(cen_samp)} N2 spindles')

    ch = s.cap['CH'].astype(np.float64)
    eeg = s.psg['EEG'].astype(np.float64)
    ch_tr, tcen, core_t = event_traces(ch, cen_samp, LOW_03)
    eeg_tr, _, _ = event_traces(eeg, cen_samp, SIGMA)

    # per-trial scalar dB (core minus own baseline, already baseline-subtracted -> core mean)
    valid = [i for i, tr in enumerate(ch_tr) if tr is not None]
    ch_db = np.array([ch_tr[i][core_t].mean() for i in valid])
    stack = np.vstack([ch_tr[i] for i in valid])
    det_rate = float((ch_db > 0).mean())

    # onset-triggered average +/- SEM
    avg = stack.mean(axis=0)
    sem = stack.std(axis=0) / np.sqrt(len(stack))

    # choose example trials spanning the detection distribution, strongest first,
    # but exclude motion-artifact outliers (|dB|>12: dropout/movement spikes, not the
    # ~0.5-2 dB mechanical bump) so the examples show the real physiological response.
    clean = np.array([i for i in np.argsort(ch_db)[::-1] if abs(ch_db[i]) < 12.0])
    n = args.ntrials
    picks_idx = np.linspace(0, len(clean) - 1, n).astype(int)
    picks = [valid[clean[p]] for p in picks_idx]

    # ---- figure ----
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.25, 1, 1, 1], hspace=0.55, wspace=0.25)

    # (A) onset-triggered average across the whole top row
    axA = fig.add_subplot(gs[0, :])
    axA.axhline(0, color='gray', lw=0.8, ls=':')
    axA.axvspan(-CORE_HALF, CORE_HALF, color=CH_COLOR, alpha=0.12,
                label='detection core (±1 s)')
    axA.axvspan(-WIN_HALF, -BASE_EDGE, color='gray', alpha=0.08)
    axA.axvspan(BASE_EDGE, WIN_HALF, color='gray', alpha=0.08, label='baseline (|t|>5 s)')
    axA.fill_between(tcen, avg - sem, avg + sem, color=CH_COLOR, alpha=0.25)
    axA.plot(tcen, avg, color=CH_COLOR, lw=2.0)
    axA.axvline(0, color='k', lw=0.8, alpha=0.6)
    axA.set_title(f'(A) {args.session}: CH low-band (0–3 Hz) power, onset-triggered average '
                  f'over all {len(stack)} N2 spindles — peak +{avg[core_t].mean():.2f} dB at onset  '
                  f'(per-trial detection rate {det_rate*100:.0f}%)',
                  fontsize=10.5, fontweight='bold')
    axA.set_xlabel('Time from spindle center (s)', fontsize=9)
    axA.set_ylabel('CH 0–3 Hz power\n(dB vs own baseline)', fontsize=9)
    axA.legend(fontsize=8, loc='upper right')
    axA.set_xlim(-WIN_HALF, WIN_HALF)

    # (B..) individual trials
    for k, ei in enumerate(picks):
        r, c = 1 + k // 3, k % 3
        ax = fig.add_subplot(gs[r, c])
        db = ch_tr[ei][core_t].mean()
        det = 'DETECTED' if db > 0 else 'not detected'
        # EEG sigma on twin axis (shows the spindle)
        axe = ax.twinx()
        axe.plot(tcen, eeg_tr[ei], color=EEG_COLOR, lw=1.0, alpha=0.7)
        axe.set_ylabel('EEG σ (dB)', fontsize=7, color=EEG_COLOR)
        axe.tick_params(axis='y', labelsize=6, labelcolor=EEG_COLOR)
        # CH low band
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.axvspan(-CORE_HALF, CORE_HALF, color=CH_COLOR, alpha=0.12)
        ax.plot(tcen, ch_tr[ei], color=CH_COLOR, lw=1.3, zorder=3)
        ax.axvline(0, color='k', lw=0.7, alpha=0.5)
        ax.set_zorder(axe.get_zorder() + 1); ax.patch.set_visible(False)
        col = '#1E8449' if db > 0 else '#C0392B'
        ax.set_title(f'trial {k+1}: CH {db:+.2f} dB — {det}', fontsize=8.5,
                     fontweight='bold', color=col)
        ax.tick_params(labelsize=6)
        ax.set_xlim(-WIN_HALF, WIN_HALF)
        if c == 0:
            ax.set_ylabel('CH 0–3 Hz (dB)', fontsize=7)
        if r == 3:
            ax.set_xlabel('Time from center (s)', fontsize=7)

    fig.suptitle('Sleep-spindle low-band response in the capacitive signal — per-trial view '
                 '(CH = blue, EEG sigma = grey)',
                 fontsize=13, fontweight='bold', y=0.995)
    out = os.path.join(FIGDIR, f'fig_spindle_trials_lowband_{args.session}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out)
    print(f'detection rate (CH low-band, this session): {det_rate*100:.1f}%   '
          f'mean +{ch_db.mean():.2f} dB, example trials dB: '
          f'{[round(ch_tr[p][core_t].mean(),2) for p in picks]}')


if __name__ == '__main__':
    main()
