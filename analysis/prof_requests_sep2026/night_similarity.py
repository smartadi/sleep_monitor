"""Night-to-night similarity: are a subject's two nights more similar to each
other than to other people's nights?

This is the multivariate version of the manuscript's per-feature §4.1
reproducibility claim. We build one raw-CAP feature vector per session (no PSG,
no k), then ask a retrieval question: for each night, is its same-subject
partner night the nearest of the other 11 recordings?

Features per channel (CH, CLE, CRE, CLE-CRE = arithmetic differential):
  - median 30-s-window band power in slow (0.02-0.1), respiratory (0.1-0.5),
    cardiac (0.5-3.0) Hz  [log10]
  - median 30-s-window log variance (0.02-5 Hz)
  - spectral peak frequency in the respiratory and cardiac bands
=> 4 channels x 6 = 24 features.

Motion is NOT excluded: night-typical restlessness is part of what makes a
recording look like itself. Per-window medians are used so a few motion spikes
do not dominate. Reported descriptively (n=6); the permutation p is support,
not a headline.

Run from repo root:
    .venv/Scripts/python.exe analysis/prof_requests_sep2026/night_similarity.py
Outputs -> reports/prof_requests_sep2026/night_similarity_features.csv
           reports/prof_requests_sep2026/night_similarity_summary.csv
           notebooks/plots/prof_requests_sep2026/E_night_similarity.png
           notebooks/plots/prof_requests_sep2026/E_night_psd_pairs.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, sosfiltfilt
from scipy.spatial.distance import squareform, pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sleep_monitor.loader import load_session
from sleep_monitor.sessions import SESSION_META
from sleep_monitor.config import FS

REP = ROOT / 'reports' / 'prof_requests_sep2026'; REP.mkdir(parents=True, exist_ok=True)
FIG = ROOT / 'notebooks' / 'plots' / 'prof_requests_sep2026'; FIG.mkdir(parents=True, exist_ok=True)

CHANS = ['CH', 'CLE', 'CRE', 'CLE-CRE']
BANDS = {'slow': (0.02, 0.1), 'resp': (0.1, 0.5), 'card': (0.5, 3.0)}
WIN_S = 30.0
rng = np.random.default_rng(0)


def chan_signal(sess, ch):
    if ch == 'CLE-CRE':
        return (sess.cap['CLE'] - sess.cap['CRE']).astype(float)
    return sess.cap[ch].astype(float)


def session_features(sess):
    fs = sess.fs
    n = int(WIN_S * fs)
    feats = {}
    for ch in CHANS:
        x = chan_signal(sess, ch)
        nwin = len(x) // n
        if nwin < 5:
            for b in BANDS: feats[f'{ch}_{b}_logpow'] = np.nan
            feats[f'{ch}_logvar'] = np.nan
            feats[f'{ch}_resp_pk'] = np.nan; feats[f'{ch}_card_pk'] = np.nan
            continue
        bp = {b: [] for b in BANDS}
        var = []
        psd_acc = None
        for w in range(nwin):
            seg = x[w * n:(w + 1) * n]
            seg = seg - seg.mean()
            f, p = welch(seg, fs=fs, nperseg=min(len(seg), 1024))
            for b, (lo, hi) in BANDS.items():
                m = (f >= lo) & (f < hi)
                bp[b].append(np.trapz(p[m], f[m]))
            var.append(np.var(seg))
            psd_acc = p if psd_acc is None else psd_acc + p
        for b in BANDS:
            feats[f'{ch}_{b}_logpow'] = np.log10(np.median(bp[b]) + 1e-12)
        feats[f'{ch}_logvar'] = np.log10(np.median(var) + 1e-12)
        psd = psd_acc / nwin
        for b, key in [('resp', f'{ch}_resp_pk'), ('card', f'{ch}_card_pk')]:
            lo, hi = BANDS[b]; m = (f >= lo) & (f < hi)
            feats[key] = float(f[m][np.argmax(psd[m])])
    return feats, (f, psd)   # last psd returned is CLE-CRE; keep CH separately below


def main():
    rows, labels, subjects, ch_psds = [], [], [], {}
    for m in SESSION_META:
        sess = load_session(m)
        feats, _ = session_features(sess)
        rows.append(feats); labels.append(m['label']); subjects.append(m['subject'])
        # keep CH median PSD for the pair-overlay figure
        x = sess.cap['CH'].astype(float); fs = sess.fs; n = int(WIN_S * fs)
        acc = None; nwin = len(x) // n
        for w in range(nwin):
            seg = x[w * n:(w + 1) * n]; seg = seg - seg.mean()
            f, p = welch(seg, fs=fs, nperseg=min(len(seg), 1024))
            acc = p if acc is None else acc + p
        ch_psds[m['label']] = (f, acc / nwin)
        print(f"  {m['label']}: features computed")

    X = pd.DataFrame(rows, index=labels)
    X.to_csv(REP / 'night_similarity_features.csv')
    # z-score each feature across the 12 sessions
    Z = (X - X.mean()) / X.std(ddof=0)
    Z = Z.fillna(0.0)
    D = squareform(pdist(Z.values, metric='euclidean'))
    subj = np.array(subjects); labs = np.array(labels)

    # ── retrieval: rank of same-subject partner among the other 11 ──
    ranks, nn_hit, partner_d, other_d = [], 0, [], []
    for i in range(len(labs)):
        partner = np.where((subj == subj[i]) & (np.arange(len(labs)) != i))[0][0]
        order = np.argsort(D[i]); order = order[order != i]           # exclude self
        rank = int(np.where(order == partner)[0][0]) + 1               # 1 = nearest
        ranks.append(rank); nn_hit += (rank == 1)
        partner_d.append(D[i, partner])
        other_d.extend(D[i, [j for j in range(len(labs)) if j != i and j != partner]])
    ranks = np.array(ranks)

    # within vs between subject pair distances
    within = [D[i, np.where((subj == subj[i]) & (np.arange(len(labs)) != i))[0][0]]
              for i in range(0, len(labs), 2)]           # 6 unique same-subject pairs
    iu = np.triu_indices(len(labs), 1)
    between = [D[a, b] for a, b in zip(*iu) if subj[a] != subj[b]]
    obs = np.mean(within)
    # permutation: shuffle subject labels, recompute mean within-pair distance
    nperm = 20000; ge = 0
    base_pairs = [(i, i + 1) for i in range(0, len(labs), 2)]
    for _ in range(nperm):
        perm = rng.permutation(len(labs))
        wp = np.mean([D[perm[a], perm[b]] for a, b in base_pairs])
        ge += (wp <= obs)
    p_perm = (ge + 1) / (nperm + 1)

    summ = pd.DataFrame({'label': labs, 'subject': subj, 'partner_rank': ranks})
    summ.to_csv(REP / 'night_similarity_summary.csv', index=False)

    print('\n===== NIGHT-TO-NIGHT SIMILARITY =====')
    print(f'nearest-neighbour hit rate: {nn_hit}/{len(labs)} nights '
          f'(chance ~{len(labs)/11:.1f}/12)')
    print(f'median partner rank: {np.median(ranks):.1f}  (1 = nearest of 11)')
    print(f'per-subject partner rank (both nights): ')
    for s in sorted(set(subjects)):
        rr = ranks[subj == s]
        print(f'   {s}: ranks {list(rr)}')
    print(f'mean within-subject distance {obs:.2f} vs between {np.mean(between):.2f}  '
          f'(perm p={p_perm:.4f})')

    # ── Figure 1: heatmap + within/between + per-subject rank ──
    fig = plt.figure(figsize=(17, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.8, 1.05], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(D, cmap='viridis')
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=90, fontsize=9)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=9)
    # mark same-subject pairs
    for i in range(0, len(labs), 2):
        for (a, b) in [(i, i + 1), (i + 1, i)]:
            ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False,
                                       edgecolor='#E74C3C', lw=2.0))
    ax.set_title('Session-to-session distance\n(red = same-subject night pair)', fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, label='Euclidean dist (z-features)')

    ax = fig.add_subplot(gs[1])
    ax.boxplot([within, between], labels=['within\nsubject', 'between\nsubject'],
               showfliers=False, widths=0.6)
    ax.scatter(np.ones(len(within)), within, color='#E74C3C', zorder=3, label='6 same-subject pairs')
    ax.set_ylabel('distance', fontsize=11)
    ax.set_title(f'within {obs:.2f} vs between {np.mean(between):.2f}\nperm p={p_perm:.3f}', fontsize=12)
    ax.tick_params(labelsize=10)

    ax = fig.add_subplot(gs[2])
    order = np.argsort(ranks)
    ax.bar(range(len(labs)), ranks[order], color=['#2ECC71' if r == 1 else '#95A5A6' for r in ranks[order]])
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs[order], rotation=90, fontsize=9)
    ax.axhline(1.5, color='#2ECC71', ls='--', lw=1, label='rank 1 = partner is nearest')
    ax.axhline(len(labs) / 11, color='gray', ls=':', lw=1, label='chance median')
    ax.set_ylabel('partner-night rank (of 11)', fontsize=11)
    ax.set_title(f'{nn_hit}/12 nights: partner is the nearest', fontsize=12)
    ax.legend(fontsize=8); ax.tick_params(labelsize=10)
    fig.suptitle('Night-to-night similarity of the raw-CAP feature vector (24 features, 12 recordings)',
                 fontsize=14, y=1.02)
    p = FIG / 'E_night_similarity.png'; fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f'  saved {p}')

    # ── Figure 2: per-subject two-night CH PSD overlay (frequency-domain context) ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, s in zip(axes.ravel(), sorted(set(subjects))):
        nights = [l for l, ss in zip(labs, subj) if ss == s]
        for l in nights:
            f, psd = ch_psds[l]; m = (f >= 0.02) & (f <= 3.0)
            ax.semilogy(f[m], psd[m], lw=1.4, label=l)
        ax.set_title(f'{s} — CH', fontsize=12); ax.set_xlim(0, 3)
        ax.set_xlabel('Hz', fontsize=10); ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
    fig.suptitle('Same-subject two-night CH power spectra (median 30-s Welch)', fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = FIG / 'E_night_psd_pairs.png'; fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f'  saved {p}')


if __name__ == '__main__':
    main()
