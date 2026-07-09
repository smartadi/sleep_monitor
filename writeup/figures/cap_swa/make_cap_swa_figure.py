"""
Manuscript figure for the CAP-SWA mechanical/autonomic definition (Workstream C).

Reads the already-computed cap_swa outputs (no recomputation) and builds a single
publication figure with four panels:
  A  Per-subject swa_score -> N3 AUC (6/6 above chance; consistent direction)
  B  Heart rate during CAP-SWA vs non-SWA, per subject (bradycardia, 6/6 down)
  C  Professor's autonomic hypotheses: predicted vs observed direction + consistency
  D  Threshold sweep — precision/recall/F1 vs swa_score cut (0.60 is F1-optimal)

Sources:
  reports/slow_wave/cap_swa/classifier/loso_ablation_folds.csv
  reports/slow_wave/cap_swa/classifier/threshold_sweep.csv
  reports/slow_wave/cap_swa/all_epoch_features.parquet
  reports/slow_wave/cap_swa/hypothesis_summary.csv
  reports/slow_wave/cap_swa/movement_initiation.csv

Out: writeup/figures/cap_swa/fig_cap_swa_definition.png
Run: .venv/Scripts/python.exe writeup/figures/cap_swa/make_cap_swa_figure.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
CAP = ROOT / 'reports' / 'slow_wave' / 'cap_swa'
CLS = CAP / 'classifier'
OUT = Path(__file__).resolve().parent / 'fig_cap_swa_definition.png'
N3_CODE = 1

# professor hypotheses: predicted direction during SWA, feature, verdict
HYP = [
    ('H4  Heart rate',      '↑ rise',  'ecg_hr_hz',            'falsified'),
    ('H5  Resp rate',       '↑ rise',  'flow_rr_hz',           'not supported'),
    ('H6  k-deviation',     '↑ jump',  'k_resp_dev',           'falsified'),
    ('H7  PPG–CAP div.',    'diverge', 'card_freq_divergence', 'falsified'),
    ('H8  EEG delta',       '↑ rise',  'eeg_delta_ratio',      'supported'),
]


def main():
    folds = pd.read_csv(CLS / 'loso_ablation_folds.csv')
    thr = pd.read_csv(CLS / 'threshold_sweep.csv')
    df = pd.read_parquet(CAP / 'all_epoch_features.parquet')
    hyp = pd.read_csv(CAP / 'hypothesis_summary.csv')
    mov = pd.read_csv(CAP / 'movement_initiation.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CAP-SWA: a mechanical/autonomic slow-wave-activity signature',
                 fontsize=15, fontweight='bold')

    # ── A: per-subject swa_score N3 AUC ──
    ax = axes[0, 0]
    d = folds[folds['feature_set'] == 'swa_score_direct'].sort_values('subject')
    subs = d['subject'].values
    aucs = d['auc'].values
    xs = np.arange(len(subs))
    ax.bar(xs, aucs, color='#16A085', alpha=0.9)
    ax.axhline(0.5, color='k', lw=0.9, alpha=0.6)
    ax.axhline(aucs.mean(), color='#2C3E50', ls='--', lw=1.2,
               label=f'mean {aucs.mean():.3f}')
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.008, f'{a:.2f}', ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(subs, fontsize=8)
    ax.set_ylim(0.4, 0.82); ax.set_ylabel('N3 AUC (per subject)')
    ax.set_title('A. CAP-SWA score discriminates N3 in every subject (6/6)')
    ax.legend(fontsize=9, loc='upper right')

    # ── B: HR during SWA vs non-SWA, per subject ──
    ax = axes[0, 1]
    for subj, g in df.groupby('subject'):
        a = g.loc[g['swa_candidate'], 'ecg_hr_hz'].dropna() * 60
        b = g.loc[~g['swa_candidate'], 'ecg_hr_hz'].dropna() * 60
        if len(a) < 5 or len(b) < 5:
            continue
        ax.plot([0, 1], [b.median(), a.median()], '-o', color='#C0392B',
                alpha=0.7, markersize=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['non-SWA', 'CAP-SWA'])
    ax.set_ylabel('Heart rate (BPM)')
    ax.set_title('B. HR falls during CAP-SWA (bradycardia, 6/6)\n'
                 'contradicts the predicted rise')
    ax.set_xlim(-0.3, 1.3)

    # ── C: hypotheses predicted vs observed ──
    ax = axes[1, 0]; ax.axis('off')
    hmap = {r['feature']: r for _, r in hyp.iterrows()}
    rows = [['Hypothesis', 'Predicted', 'Observed', 'Consist.', 'Verdict']]
    for label, pred, feat, verdict in HYP:
        r = hmap.get(feat)
        if r is None:
            continue
        delta = r['median_delta']
        if feat == 'ecg_hr_hz':
            obs = f'↓ {delta*60:+.1f} BPM'
        elif feat == 'flow_rr_hz':
            obs = f'flat {delta*60:+.2f} br/min'
        elif feat == 'k_resp_dev':
            obs = '↓ shrinks'
        elif feat == 'card_freq_divergence':
            obs = f'reverses {delta:+.2f} Hz'
        else:
            obs = f'↑ {delta:+.3f}'
        rows.append([label, pred, obs, r['consistency'], verdict])
    # movement row
    lift = np.nanmedian(mov['lift'])
    rows.append(['H2  Movement precedes', 'yes', f'no (lift {lift:.2f})',
                 '—', 'not supported'])
    tab = ax.table(cellText=rows, loc='center', cellLoc='left',
                   colWidths=[0.30, 0.16, 0.24, 0.13, 0.20])
    tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1.0, 1.55)
    for j in range(5):
        c = tab[0, j]; c.set_facecolor('#34495E')
        c.set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        verdict = rows[i][4]
        col = '#FADBD8' if verdict in ('falsified', 'not supported') else '#D5F5E3'
        for j in range(5):
            tab[i, j].set_facecolor(col if j == 4 else ('#F8F9F9' if i % 2 else 'white'))
    ax.set_title("C. Professor's autonomic hypotheses about the CAP-SWA state\n"
                 '(per-subject direction; n=6, so direction counts are the evidence)',
                 fontsize=10)

    # ── D: threshold sweep ──
    ax = axes[1, 1]
    s = thr[thr['mode'] == 'sustained'].sort_values('threshold')
    ax.plot(s['threshold'], s['precision'], '-o', color='#E74C3C', label='precision',
            markersize=4)
    ax.plot(s['threshold'], s['recall'], '-o', color='#3498DB', label='recall',
            markersize=4)
    ax.plot(s['threshold'], s['f1'], '-o', color='#2ECC71', label='F1', markersize=4)
    ax.axvline(0.60, color='k', ls=':', alpha=0.6, label='chosen cut 0.60')
    ax.set_xlabel('swa_score threshold'); ax.set_ylabel('score')
    ax.set_title('D. 0.60 threshold sits at the F1 maximum')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
