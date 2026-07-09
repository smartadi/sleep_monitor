"""
What in the capacitive signal varies with subject age? (n=6, exploratory)

Two parts:
 (1) k night-to-night VARIABILITY vs age  — is k more/less reproducible with age?
 (2) A compact scan of CAP-derived per-session features vs age:
        respiratory / cardiac band fraction, in-band SNR, spectral peak frequency,
        DC drift (baseline wander), accelerometer activity.

Unit of analysis = subject (mean over the 2 nights), age from Table 1. Correlations use
Spearman; with 6 subjects and many features this is strictly exploratory — p-values are
reported raw and the number of tests is stated so nothing is over-read.

Outputs -> analysis/rates/outputs/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sleep_monitor.loader import load_session
from sleep_monitor.sessions import SESSION_META

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)
FS = 100.0
K_CSV = 'reports/rates/mask/per_session_summary.csv'

DEMO = {
    'S1': dict(age=61, sex='F', psqi=9), 'S2': dict(age=66, sex='M', psqi=4),
    'S3': dict(age=37, sex='M', psqi=9), 'S4': dict(age=54, sex='M', psqi=8),
    'S5': dict(age=55, sex='F', psqi=6), 'S6': dict(age=25, sex='M', psqi=6),
}
RESP = (0.1, 0.5); CARD = (0.5, 3.0); FLOOR = (3.5, 5.0)


def band_frac(f, p, lo, hi, tot=(0.0, 5.0)):
    return p[(f >= lo) & (f < hi)].sum() / p[(f >= tot[0]) & (f < tot[1])].sum()


def band_snr_db(f, p, lo, hi):
    sig = p[(f >= lo) & (f < hi)].mean()
    noi = p[(f >= FLOOR[0]) & (f < FLOOR[1])].mean()
    return 10 * np.log10(sig / noi)


def peak_hz(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return f[m][np.argmax(p[m])]


def session_features(idx):
    s = load_session(idx)
    diff = (s.cap['CLE'].astype(np.float64) - s.cap['CRE'].astype(np.float64))
    nper = int(30 * FS)
    f, p = welch(diff, fs=FS, nperseg=nper, noverlap=nper // 2)
    # DC drift: std of per-30s-window mean of the differential
    n = len(diff) // nper
    win_means = diff[:n * nper].reshape(n, nper).mean(1)
    dc_drift = float(np.std(win_means))
    acc_mean = float(np.mean(s.cap['acc_mag'].astype(np.float64)))
    return dict(
        resp_frac=band_frac(f, p, *RESP), card_frac=band_frac(f, p, *CARD),
        resp_snr_db=band_snr_db(f, p, *RESP), card_snr_db=band_snr_db(f, p, *CARD),
        resp_peak_hz=peak_hz(f, p, *RESP), card_peak_hz=peak_hz(f, p, *CARD),
        dc_drift=dc_drift, acc_mean=acc_mean,
    )


def subj_of(label):
    return label.split('N')[0]


def main():
    # ── per-session CAP features ──────────────────────────────────────────────
    rows = []
    for m in SESSION_META:
        idx = m['idx']
        feats = session_features(idx)
        feats.update(session=m['label'], subj=subj_of(m['label']))
        rows.append(feats)
        print(f"{m['label']}: card_frac={feats['card_frac']:.2f} card_snr={feats['card_snr_db']:.1f}dB "
              f"card_peak={feats['card_peak_hz']:.2f}Hz dc_drift={feats['dc_drift']:.1f}")
    fdf = pd.DataFrame(rows)

    # ── k night-to-night variability ─────────────────────────────────────────
    k = pd.read_csv(K_CSV)
    k['subj'] = k['session'].map(subj_of)
    kvar = []
    for band in ('resp', 'card'):
        b = k[k.band == band]
        for subj, sub in b.groupby('subj'):
            kk = sub['k'].values
            kvar.append(dict(subj=subj, band=band,
                             k_dabs=abs(kk[0] - kk[-1]),
                             k_cv=np.std(kk) / np.mean(kk)))
    kvar = pd.DataFrame(kvar)

    # ── subject-level table ───────────────────────────────────────────────────
    feat_cols = ['resp_frac', 'card_frac', 'resp_snr_db', 'card_snr_db',
                 'resp_peak_hz', 'card_peak_hz', 'dc_drift', 'acc_mean']
    subj = fdf.groupby('subj')[feat_cols].mean()
    subj['age'] = [DEMO[s]['age'] for s in subj.index]
    subj['k_dabs_resp'] = [kvar[(kvar.subj == s) & (kvar.band == 'resp')]['k_dabs'].iloc[0] for s in subj.index]
    subj['k_dabs_card'] = [kvar[(kvar.subj == s) & (kvar.band == 'card')]['k_dabs'].iloc[0] for s in subj.index]
    subj = subj.sort_values('age')
    subj.to_csv(os.path.join(OUT, 'age_features_per_subject.csv'))

    # ── correlations vs age ───────────────────────────────────────────────────
    test_cols = feat_cols + ['k_dabs_resp', 'k_dabs_card']
    print(f'\n=== Feature vs AGE (Spearman, n=6, {len(test_cols)} features tested) ===')
    res = []
    for c in test_cols:
        rho, p = spearmanr(subj['age'], subj[c])
        res.append(dict(feature=c, spearman_rho=rho, p=p))
        flag = '  <-- p<0.05 (uncorrected)' if p < 0.05 else ''
        print(f'  {c:14s} rho={rho:+.3f}  p={p:.3f}{flag}')
    res = pd.DataFrame(res).sort_values('p')
    res.to_csv(os.path.join(OUT, 'age_features_stats.csv'), index=False)
    n_sig = (res.p < 0.05).sum()
    print(f'\n{n_sig}/{len(test_cols)} features p<0.05 uncorrected '
          f'(expected by chance at alpha=0.05: {0.05*len(test_cols):.1f}). '
          f'Bonferroni alpha = {0.05/len(test_cols):.4f}.')

    # ── figure: k variability vs age + top age-correlated features ────────────
    top = res.head(4)['feature'].tolist()
    panels = ['k_dabs_resp', 'k_dabs_card'] + [c for c in top if c not in ('k_dabs_resp', 'k_dabs_card')][:2]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0))
    for ax, c in zip(axes, panels):
        colors = ['#C0392B' if DEMO[s]['sex'] == 'F' else '#2980B9' for s in subj.index]
        ax.scatter(subj['age'], subj[c], c=colors, s=60, edgecolor='k', lw=0.5, zorder=3)
        for a, y, s in zip(subj['age'], subj[c], subj.index):
            ax.annotate(s, (a, y), textcoords='offset points', xytext=(5, 3), fontsize=8)
        rho, p = spearmanr(subj['age'], subj[c])
        z = np.polyfit(subj['age'], subj[c], 1)
        xs = np.linspace(subj['age'].min() - 2, subj['age'].max() + 2, 40)
        ax.plot(xs, np.polyval(z, xs), 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('Age (years)'); ax.set_ylabel(c)
        ax.set_title(f'{c}\nrho={rho:+.2f}, p={p:.2f}', fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle('CAP features vs age (per-subject, n=6, exploratory)', y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_age_features.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()
