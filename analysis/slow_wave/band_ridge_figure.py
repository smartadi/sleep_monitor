"""
Paper figure + citable numbers for band-restricted ridge analysis.
Reads reports/slow_wave/band_ridge_epochs.parquet (produced by band_ridge_analysis.py).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def bar_by_stage(ax, d, col, agg='mean'):
    xs, hs, cols = [], [], []
    for k, sc in enumerate(STAGE_ORDER):
        v = d.loc[d.stage_code == sc, col].dropna()
        if len(v) == 0:
            continue
        xs.append(k)
        hs.append(v.mean() if agg == 'mean' else v.median())
        cols.append(STAGE_COLORS[sc])
    ax.bar(xs, hs, color=cols, alpha=0.75, edgecolor='black', lw=0.5)
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


fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
for r, band in enumerate(BANDS):
    d = pool(band)
    # col 0: mean active ridges per epoch
    ax = axes[r, 0]
    bar_by_stage(ax, d, 'n_ridges', 'mean')
    kw = kruskal(*[d.loc[d.stage_code == sc, 'n_ridges'].dropna().values
                   for sc in STAGE_ORDER if (d.stage_code == sc).any()])[1]
    ax.set_title(f'Mean active ridges / epoch\nKW p={kw:.1e}', fontsize=9)
    ax.set_ylabel(f'{BANDS[band]}\n({POOL_CH})', fontsize=9)
    ax.grid(True, alpha=0.15, axis='y')
    # col 1: total ridge power (present epochs)
    ax = axes[r, 1]
    kw = box_by_stage(ax, d, 'total_ridge_power', present_only=True)
    ax.set_title(f'Total ridge power (ridge-present epochs)\nKW p={kw:.1e}', fontsize=9)
    ax.grid(True, alpha=0.15, axis='y')
    # col 2: lowest ridge freq (present epochs)
    ax = axes[r, 2]
    kw = box_by_stage(ax, d, 'min_ridge_freq', present_only=True)
    ax.set_title(f'Lowest ridge freq (Hz)\nKW p={kw:.1e}', fontsize=9)
    ax.grid(True, alpha=0.15, axis='y')

fig.suptitle('Band-restricted ridge structure by sleep stage (CRE, 12 sessions)', fontsize=13)
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
