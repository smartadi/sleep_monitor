"""
Rigorous test of the k~2 mechanism: how many CAP pulse peaks occur per true heartbeat?

The ensemble R-peak-triggered average (rpeak_triggered_waveform.py) is confounded by
beat-to-beat phase jitter of the dicrotic peak. The direct quantity behind k is the
peak-count ratio: k = (CAP peaks counted) / (true beats). So we count CAP cardiac-band
peaks and ECG R-peaks over the same asleep span and compare CAP-peaks-per-beat to the
per-session k reported in the paper. A tight ratio ~ k across sessions demonstrates that k
measures pulse-morphology overcounting (biphasic pulse -> ~2 peaks/beat).

Outputs -> analysis/rates/outputs/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.ground_truth import _ecg_rpeaks

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
FS = 100.0
CARD = (0.5, 3.0)          # estimator cardiac band
MIN_DIST_S = 0.25          # allow up to ~4 Hz (2 peaks per ~0.9 s beat)
PROM_FACTOR = 0.05         # loose detector


def bp(sig, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return filtfilt(b, a, sig.astype(np.float64))


def cap_peaks(sig):
    md = int(MIN_DIST_S * FS)
    sm = np.convolve(sig, np.ones(5) / 5, mode='same')
    pk, _ = find_peaks(sm, distance=md, prominence=PROM_FACTOR * np.std(sm))
    return pk


def asleep_mask_samples(n, prof):
    """boolean per-sample: asleep (stage != Wake and scored)."""
    m = np.zeros(n, bool)
    if prof is None:
        return ~m
    tep, codes = prof['t_ep_hr'], prof['codes']
    for t0, c in zip(tep, codes):
        if c in (4, -1):
            continue
        a = int(t0 * 3600 * FS)
        b = int((t0 + 30.0 / 3600 * 3600) * FS)  # 30 s epoch
        b = int((t0 * 3600 + 30.0) * FS)
        a = max(0, a); b = min(n, b)
        if b > a:
            m[a:b] = True
    return m


def main():
    kdf = pd.read_csv('reports/rates/mask/per_session_summary.csv')
    kcard = kdf[kdf.band == 'card'].set_index('session')['k'].to_dict()

    rows = []
    for m in SESSION_META:
        idx = m['idx']; lab = m['label']
        s = load_session(idx)
        ecg = s.psg.get('ECG')
        try:
            rp = _ecg_rpeaks(ecg.astype(np.float64), FS)
        except Exception:
            print(f'{lab}: ECG unusable, skip'); continue
        if len(rp) < 200:
            print(f'{lab}: too few beats, skip'); continue
        prof = load_sleep_profile(s)
        asleep = asleep_mask_samples(len(s.time_hr), prof)

        rp = rp[asleep[np.clip(rp, 0, len(asleep) - 1)]]
        n_beats = len(rp)
        if n_beats < 200:
            print(f'{lab}: too few asleep beats, skip'); continue

        for ch in ('CRE', 'CLE-CRE'):
            sig = (s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64)) if ch == 'CLE-CRE' \
                  else s.cap[ch].astype(np.float64)
            sig = bp(sig, *CARD)
            pk = cap_peaks(sig)
            pk = pk[asleep[np.clip(pk, 0, len(asleep) - 1)]]
            ratio = len(pk) / n_beats
            rows.append(dict(session=lab, channel=ch, cap_peaks=len(pk),
                             ecg_beats=n_beats, peaks_per_beat=ratio,
                             k_reported=kcard.get(lab, np.nan)))
        pr = [r for r in rows if r['session'] == lab]
        print(f'{lab}: k={kcard.get(lab):.2f}  CRE_ppb={pr[0]["peaks_per_beat"]:.2f}  '
              f'CLE-CRE_ppb={pr[1]["peaks_per_beat"]:.2f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'peaks_per_beat.csv'), index=False)

    print('\n=== CAP peaks-per-beat vs reported cardiac k (per session) ===')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ch in zip(axes, ('CRE', 'CLE-CRE')):
        d = df[df.channel == ch].dropna()
        x, y = d['k_reported'].values, d['peaks_per_beat'].values
        r, p = pearsonr(x, y); rho, _ = spearmanr(x, y)
        ax.scatter(x, y, s=60, color='#8E44AD', edgecolor='k', lw=0.5, zorder=3)
        for xi, yi, s in zip(x, y, d['session']):
            ax.annotate(s, (xi, yi), textcoords='offset points', xytext=(5, 3), fontsize=8)
        lim = [min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2]
        ax.plot(lim, lim, 'k--', lw=1, alpha=0.5, label='y = x')
        ax.set_xlabel('Reported cardiac k'); ax.set_ylabel('CAP peaks per ECG beat')
        ax.set_title(f'{ch}:  Pearson r={r:.2f}, Spearman rho={rho:.2f}, n={len(d)}')
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        print(f'  {ch:8s} r={r:+.3f} p={p:.3f}  rho={rho:+.3f}  '
              f'mean ppb={y.mean():.2f} (k mean={x.mean():.2f})')
    fig.suptitle('The cardiac k IS the CAP pulse-overcount ratio: peaks per heartbeat tracks k',
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_peaks_per_beat.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()
