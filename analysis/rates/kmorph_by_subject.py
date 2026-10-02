"""
What the cardiac pulse looks like per participant, on the same footing as k.

k counts peaks on a raw SEC channel band-passed to 0.5-3.0 Hz. It comes out near
2 for five participants and near 1.16 for one, the youngest. The natural reading
is that the capacitive pulse is biphasic -- a systolic deflection and a second,
later one -- and that in this participant only one of the two is resolved.

An existing R-peak-triggered figure appears to contradict that: on the CLE-CRE
differential, band-passed 0.5-8.0 Hz, this participant shows two of the
cleanest, best separated peaks in the cohort. But that figure is not comparable
with k. It uses a different channel and a band almost three times wider, and a
wider band keeps structure that the 0.5-3.0 Hz band used for k removes.

So the morphology is recomputed here on exactly what k sees: the raw channels,
0.5-3.0 Hz, ensemble-averaged on ECG R-peaks over asleep beats, one panel per
participant ordered by age, with that participant's k printed beside it. Then
the waveform and the number refer to the same thing and can be read together.

Reporting is descriptive. Six participants cannot support a correlation with
age, and none is computed.

Writes  reports/rates/kmorph_by_subject.csv
        writeup/figures/rate_supp/fig_kmorph_by_subject.png

Usage
-----
    .venv/Scripts/python.exe analysis/rates/kmorph_by_subject.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
from scipy.signal import butter, filtfilt   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.loader import load_session, load_sleep_profile   # noqa: E402
from sleep_monitor.sessions import SESSION_META                     # noqa: E402
from sleep_monitor.ground_truth import _ecg_rpeaks                  # noqa: E402

FIG = ROOT / 'writeup' / 'figures' / 'rate_supp'
TAB = ROOT / 'reports' / 'rates'
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

FS = 100.0
CARD_BAND = (0.5, 3.0)        # exactly the band k is counted in
PRE = 0.15                    # s before the R-peak
CHANNELS = ['CH', 'CLE', 'CRE']
CH_COLORS = {'CH': '#1f4e79', 'CLE': '#2e9e5b', 'CRE': '#8e44ad'}

# ages from analysis/rates/outputs/k_vs_age_per_subject.csv
AGE = {'OS006': 25, 'OS003': 37, 'OS004': 54,
       'OS005': 55, 'OS001': 61, 'OS002': 66}

plt.rcParams.update({
    'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.dpi': 200,
})


def bp(x, lo, hi, fs=FS, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, np.asarray(x, float))


def stage_codes_at(samp_idx, prof):
    """Sleep-stage code per sample index, from the 30-s epoch profile.

    Key names follow load_sleep_profile: 'codes' and 't_ep_hr'.
    """
    t_hr = samp_idx / FS / 3600.0
    codes, tep = prof['codes'], prof['t_ep_hr']
    j = np.clip(np.searchsorted(tep, t_hr) - 1, 0, len(codes) - 1)
    return np.asarray(codes)[j]


def one_session(meta):
    """Ensemble-averaged waveform per channel for one recording."""
    s = load_session(meta)
    ecg = s.psg.get('ECG')
    if ecg is None:
        return None
    rp = _ecg_rpeaks(np.asarray(ecg, float), FS)
    if rp.size < 500:
        return None

    # asleep beats only: the pulse is cleaner, and it matches how k is computed
    prof = load_sleep_profile(s)
    if prof is not None:
        st = stage_codes_at(rp, prof)
        keep_stage = (st != 4) & (st != -1)        # drop Wake and unscored
        if keep_stage.sum() > 500:
            rp = rp[keep_stage]

    rr = float(np.median(np.diff(rp)) / FS)
    if not (0.4 < rr < 1.5):
        return None
    pre, post = int(PRE * FS), int(rr * FS)
    out = {}
    for ch in CHANNELS:
        x = bp(np.asarray(s.cap[ch], float), *CARD_BAND)
        keep = rp[(rp > pre) & (rp + post < len(x))]
        seg = np.stack([x[p - pre:p + post] for p in keep])
        seg = seg - seg.mean(axis=1, keepdims=True)
        m = seg.mean(axis=0)
        rng = np.abs(m).max() or 1.0
        out[ch] = m / rng
    return dict(t=(np.arange(-pre, post) / FS), rr=rr,
                n_beats=int(len(keep)), wave=out)


def main():
    k = pd.read_csv(TAB / 'k_by_channel.csv')
    k = k[(k.method == 'peaks_loose') & (k.band == 'card')]
    kmed = k.groupby(['subject', 'channel'])['k'].median()

    # one recording per participant: the first that yields a usable ensemble
    by_subj = {}
    for meta in SESSION_META:
        sub = meta['subject']
        if sub in by_subj:
            continue
        try:
            res = one_session(meta)
        except Exception as e:
            print(f'  {meta["label"]}: skipped ({type(e).__name__}: {e})')
            continue
        if res:
            res['label'] = meta['label']
            by_subj[sub] = res
            print(f'  {meta["label"]}  {res["n_beats"]} beats, RR {res["rr"]:.2f} s')

    order = [s for s in sorted(AGE, key=AGE.get) if s in by_subj]
    fig, axes = plt.subplots(len(order), 1, figsize=(11.0, 3.0 * len(order)),
                             sharex=False)
    rows = []
    for ax, sub in zip(np.atleast_1d(axes), order):
        r = by_subj[sub]
        for ch in CHANNELS:
            ax.plot(r['t'], r['wave'][ch], lw=2.4, color=CH_COLORS[ch],
                    label=f'{ch}   k = {kmed.get((sub, ch), float("nan")):.2f}')
            rows.append(dict(subject=sub, age=AGE[sub], recording=r['label'],
                             channel=ch, k_card=float(kmed.get((sub, ch), np.nan)),
                             rr_s=r['rr'], n_beats=r['n_beats']))
        ax.axvline(0, color='#b3282d', ls=':', lw=1.6)
        ax.axvline(r['rr'], color='#888888', ls='--', lw=1.4)
        ax.set_ylabel('normalised\namplitude')
        ax.set_title(f'{sub}, age {AGE[sub]}   ·   {r["label"]}, '
                     f'{r["n_beats"]:,} beats', loc='left', fontweight='bold')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', ncol=3)
    np.atleast_1d(axes)[-1].set_xlabel(
        'seconds from the ECG R-peak   (dotted = R-peak, dashed = one median RR)')
    fig.suptitle('Capacitive cardiac pulse per participant, in the band k is '
                 'counted in\n'
                 'Raw channels, 0.5–3.0 Hz, ensemble-averaged on ECG '
                 'R-peaks over asleep beats · ordered by age',
                 fontsize=18, fontweight='bold', x=0.015, ha='left', y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = FIG / 'fig_kmorph_by_subject.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TAB / 'kmorph_by_subject.csv', index=False)
    print(f'\nwrote {p.name} and kmorph_by_subject.csv')


if __name__ == '__main__':
    main()
