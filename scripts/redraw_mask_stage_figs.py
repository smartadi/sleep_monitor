"""Redraw the mask pipeline's stage figure on the corrected ladder order.

`run_mask_rate_detection.py` drew per-stage MAE in the old order (Wake, N1, N2,
N3, REM). The order is now Wake, REM, N1, N2, N3, defined once in
`sleep_monitor.config.STAGE_ORDER`. Rerunning the whole pipeline to pick that up
would rewrite every report it owns, so this reads the phase-C checkpoint and the
strategy it settled on, and redraws that one figure. Values are unchanged --
same checkpoint, same estimator, same arithmetic; only the bar order moves.

Run from the repo root:  .venv/Scripts/python.exe scripts/redraw_mask_stage_figs.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import STAGE_COLORS, STAGE_LABELS, STAGE_ORDER  # noqa: E402

CHECKPOINT = ROOT / 'artifacts' / 'mask_phase_c.parquet'
SUMMARY = ROOT / 'reports' / 'rates' / 'mask' / 'final_summary.json'
FIG_DIR = ROOT / 'writeup' / 'figures' / 'mask_rate_detection'

STAGES = [STAGE_LABELS[c] for c in STAGE_ORDER]


def main():
    cdf = pd.read_parquet(CHECKPOINT)
    best_strat = json.loads(SUMMARY.read_text())['best_strat']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, band in zip(axes, ['resp', 'card']):
        unit = 'br/min' if band == 'resp' else 'BPM'
        sub = cdf[(cdf.band == band) & (cdf.strategy == best_strat[band])
                  & cdf.gt_hz.notna()]

        maes, ns = [], []
        for stage in STAGES:
            ss = sub[sub.stage == stage]
            pred = ss['rate_k_full_smooth'].values
            gt = ss['gt_hz'].values
            v = np.isfinite(pred) & np.isfinite(gt)
            maes.append(np.median(np.abs(pred[v] - gt[v])) * 60 if v.sum() > 10
                        else np.nan)
            ns.append(int(v.sum()))

        ax.bar(range(len(STAGES)), maes, alpha=0.8, edgecolor='white',
               color=[STAGE_COLORS[c] for c in STAGE_ORDER])
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES)
        ax.set_ylabel('Median MAE (%s)' % unit)
        ax.set_title('Respiratory' if band == 'resp' else 'Cardiac')
        for i, (v, n) in enumerate(zip(maes, ns)):
            if np.isfinite(v):
                ax.text(i, v + 0.1, '%.1f\nn=%d' % (v, n), ha='center', fontsize=8)
        print('  %-5s %s' % (band, ', '.join(
            '%s=%.1f (n=%d)' % (s, m, n) for s, m, n in zip(STAGES, maes, ns))))

    fig.suptitle('Rate estimation accuracy by sleep stage',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIG_DIR / 'fig3_per_stage_mae.png'
    fig.savefig(out)
    plt.close(fig)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
