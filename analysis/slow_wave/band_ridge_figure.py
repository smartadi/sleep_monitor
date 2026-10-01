"""
Paper figure + citable numbers for band-restricted ridge analysis.
Reads reports/slow_wave/band_ridge_epochs.parquet (produced by band_ridge_analysis.py).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'xtick.labelsize': 11, 'ytick.labelsize': 11})
from scipy.stats import kruskal, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]
EP = pd.read_parquet(ROOT / 'reports' / 'slow_wave' / 'band_ridge_epochs.parquet')
FIG_DIR = ROOT / 'writeup' / 'figures' / 'harmonics'
FIG_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {'resp': 'Respiratory (0.1-0.5 Hz)', 'card': 'Cardiac (0.5-3.0 Hz)'}
POOL_CH = 'CRE'


def pool(band):
    d = EP[(EP.band == band) & (EP.channel == POOL_CH)
           & (~EP.motion_masked) & (EP.stage_code >= 0)]
    return d


def per_session_mean_by_stage(ax, d, col='n_ridges'):
    """Bar of the across-recording mean, with one dot per recording.

    The unit is the recording, not the epoch. Epochs within a night are
    correlated, so a bar drawn over pooled epochs would carry a spread set by
    how many epochs a night happens to contain rather than by how much the
    quantity actually varies.
    """
    xs, hs, cols = [], [], []
    for k, sc in enumerate(STAGE_ORDER):
        per_sess = (d[d.stage_code == sc].groupby('session')[col]
                    .mean().dropna())
        if per_sess.empty:
            continue
        xs.append(k)
        hs.append(per_sess.mean())
        cols.append(STAGE_COLORS[sc])
        jitter = np.linspace(-0.17, 0.17, len(per_sess))
        ax.scatter(np.full(len(per_sess), k) + jitter, per_sess.values,
                   s=16, color='black', alpha=0.55, zorder=4, linewidths=0)
    ax.bar(xs, hs, color=cols, alpha=0.75, edgecolor='black', lw=0.5, zorder=2)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_LABELS[c] for c in STAGE_ORDER])


def box_by_stage(ax, d, col, present_only=True):
    data, labs, cols = [], [], []
    dd = d[d.ridge_present == 1] if present_only else d
    for sc in STAGE_ORDER:
        v = dd.loc[dd.stage_code == sc, col].dropna()
        if len(v) > 0:
            data.append(v.values)
            labs.append(STAGE_LABELS[sc])
            cols.append(STAGE_COLORS[sc])
    bp = ax.boxplot(data, labels=labs, patch_artist=True, widths=0.6,
                    showfliers=False, medianprops=dict(color='black', lw=1.5))
    for j, c in enumerate(cols):
        bp['boxes'][j].set_facecolor(c)
        bp['boxes'][j].set_alpha(0.7)
    return kruskal(*data)[1] if len(data) >= 2 else np.nan


fig, axes = plt.subplots(2, 3, figsize=(16.5, 9), squeeze=False)
for r, band in enumerate(BANDS):
    d = pool(band)
    # col 0: active ridges per epoch.
    #
    # The reviewer asked for medians here, to match the box-and-whiskers in the
    # other two columns. Taken literally that destroys the panel: n_ridges is a
    # small count, so the respiratory median is 2 in every stage and the cardiac
    # median is 0 in every stage -- five identical boxes, and a flat line at
    # zero. The mean is the informative statistic for a count this sparse.
    #
    # What the comment is really about is a bare mean sitting beside medians
    # with no spread shown. So the bar keeps the mean and gains the spread, as
    # one dot per recording: 12 session means, which is also the honest unit --
    # 30 s epochs within a night are not independent, and pooling them is what
    # produced the p-values this figure used to print.
    ax = axes[r, 0]
    per_session_mean_by_stage(ax, d)
    ax.set_title('Active ridges / epoch\n(bar: mean of 12 recordings; dots: each recording)',
                 fontsize=11)
    ax.set_ylabel(f'{BANDS[band]}\n({POOL_CH})', fontsize=12)
    ax.grid(True, alpha=0.15, axis='y')
    # col 1: total ridge power (present epochs)
    ax = axes[r, 1]
    box_by_stage(ax, d, 'total_ridge_power', present_only=True)
    ax.set_title('Total ridge power (ridge-present epochs)', fontsize=12)
    ax.grid(True, alpha=0.15, axis='y')
    # col 2: lowest ridge freq (present epochs)
    ax = axes[r, 2]
    box_by_stage(ax, d, 'min_ridge_freq', present_only=True)
    ax.set_title('Lowest ridge frequency (Hz)', fontsize=12)
    ax.grid(True, alpha=0.15, axis='y')

# No p-values on the panels. They would be Kruskal-Wallis over pooled 30 s
# epochs, which are not independent within a night, so the test is inflated by
# epoch count rather than by effect: printing p = 1e-29 and then disowning it in
# the caption is worse than not printing it. The comparison here is descriptive.
fig.suptitle('Band-restricted ridge structure by sleep stage (CRE, 12 recordings)\n'
             'Left column: bar is the mean of the 12 recordings, each dot one '
             'recording · middle and right: median, IQR and 1.5 x IQR '
             'whiskers over epochs\n'
             'Descriptive only — 30 s epochs within a night are not '
             'independent, so no test is reported',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = FIG_DIR / 'band_ridge_by_stage.png'
fig.savefig(out, dpi=180)
plt.close(fig)
print('wrote', out)

# ── Citable numbers ──
print('\n===== CITABLE NUMBERS =====')
for band in BANDS:
    d = pool(band)
    dn3 = d[d.stage_code == 1]
    doth = d[(d.stage_code != 1) & (d.stage_code != 4)]  # non-N3 sleep
    present_rate = d.ridge_present.mean()
    print(f'\n[{band}] pooled {POOL_CH}, n_epochs={len(d)}')
    print(f'  ridge-present rate overall: {present_rate:.2%}')
    print(f'  mean active ridges  N3={dn3.n_ridges.mean():.2f}  nonN3sleep={doth.n_ridges.mean():.2f}')
    print(f'  ridge-present rate  N3={dn3.ridge_present.mean():.2%}  nonN3sleep={doth.ridge_present.mean():.2%}')
    # present-only medians
    dp = d[d.ridge_present == 1]
    dpn3 = dp[dp.stage_code == 1]; dpoth = dp[(dp.stage_code != 1) & (dp.stage_code != 4)]
    print(f'  median lowest ridge freq (present)  N3={dpn3.min_ridge_freq.median():.3f}Hz '
          f'nonN3={dpoth.min_ridge_freq.median():.3f}Hz')
    print(f'    -> in rate units N3={dpn3.min_ridge_freq.median()*60:.0f}/min '
          f'nonN3={dpoth.min_ridge_freq.median()*60:.0f}/min')
    print(f'  median total ridge power (present)  N3={dpn3.total_ridge_power.median():.3f} '
          f'nonN3={dpoth.total_ridge_power.median():.3f}')
    # per-subject N3<nonN3 consistency for n_ridges
    ups = dns = 0
    for subj in sorted(d.subject.unique()):
        s = d[d.subject == subj]
        a = s.loc[s.stage_code == 1, 'n_ridges']
        b = s.loc[(s.stage_code != 1) & (s.stage_code != 4), 'n_ridges']
        if len(a) > 3 and len(b) > 3:
            if a.mean() < b.mean(): dns += 1
            else: ups += 1
    print(f'  n_ridges N3<nonN3 in {dns}/{dns+ups} subjects')
    if len(dn3) > 5 and len(doth) > 5:
        print(f'  MWU n_ridges N3 vs nonN3 p={mannwhitneyu(dn3.n_ridges, doth.n_ridges)[1]:.2e}')
        print(f'  MWU min_freq(present) N3 vs nonN3 p={mannwhitneyu(dpn3.min_ridge_freq, dpoth.min_ridge_freq)[1]:.2e}')
