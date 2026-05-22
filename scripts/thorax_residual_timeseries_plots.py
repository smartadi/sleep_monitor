"""
Plot residualized timeseries for all nights: thorax_resp_rms + CLE/CRE/CH
(all 4 stats) with raw vs residualized overlaid, plus motion and hypnogram.

Output:
  notebooks/plots/thorax_analysis/residual_timeseries/{session}.png  (x12)
"""

from __future__ import annotations
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'thorax_analysis' / 'residual_timeseries'

TARGET = 'thorax_resp_rms'
ACCEL_COLS = ['movement_rms', 'movement_peak', 'roll_deg', 'pitch_deg']

CHANNELS = ['CLE', 'CRE', 'CH']
STATS = ['raw_mean', 'raw_std']
CHANNEL_COLORS = {'CLE': '#3498DB', 'CRE': '#E67E22', 'CH': '#27AE60'}

STAT_LABELS = {
    'raw_mean': 'Raw Dev from Session Mean',
    'raw_std': 'Raw Std',
}


def residualize_per_session(df, cols_to_resid, accel_cols):
    df_out = df.copy()
    motion_r2 = {}
    for sess in sorted(df['session'].unique()):
        mask = df['session'] == sess
        idx = df.index[mask]
        X_acc = df.loc[idx, accel_cols].fillna(0).values
        for col in cols_to_resid:
            vals = df.loc[idx, col].values
            ok = np.isfinite(vals)
            if ok.sum() < 20:
                continue
            model = Ridge(alpha=1.0)
            model.fit(X_acc[ok], vals[ok])
            pred = model.predict(X_acc)
            if col == TARGET:
                motion_r2[sess] = float(r2_score(vals[ok], pred[ok]))
            df_out.loc[idx, col] = vals - pred
    return df_out, motion_r2


def plot_session(sess, df_raw, df_resid):
    raw = df_raw[df_raw['session'] == sess].sort_values('t_hr').reset_index(drop=True)
    res = df_resid[df_resid['session'] == sess].sort_values('t_hr').reset_index(drop=True)
    if len(raw) < 10:
        return

    t = raw['t_hr'].values

    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True,
                             gridspec_kw={'hspace': 0.15, 'height_ratios': [2, 2, 2, 1]})

    # Panel 1: Thorax resp_rms
    ax = axes[0]
    ax.plot(t, raw[TARGET].values, color='#95A5A6', lw=0.7, ls='--', alpha=0.4, label='raw')
    ax.plot(t, res[TARGET].values, color='#2C3E50', lw=1.0, alpha=0.9, label='residualized')
    ax.set_ylabel('Thorax resp RMS', fontsize=8)
    ax.set_title(f'{sess} — Thorax + CAP channels (raw=dashed/faint, residualized=solid)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7)

    # Panels 2-3: raw_mean (dev from session mean) and raw_std
    for panel_idx, stat in enumerate(STATS):
        ax = axes[1 + panel_idx]
        for ch in CHANNELS:
            col = f'{ch}_{stat}'
            if col not in raw.columns:
                continue
            color = CHANNEL_COLORS[ch]
            ax.plot(t, raw[col].values, color=color, lw=0.5, ls='--', alpha=0.3)
            ax.plot(t, res[col].values, color=color, lw=0.8, alpha=0.85, label=ch)
        ax.set_ylabel(STAT_LABELS[stat], fontsize=8)
        ax.legend(fontsize=7, loc='upper right', ncol=3)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=7)

    # Panel 4: Motion + hypnogram
    ax = axes[3]
    ax2 = ax.twinx()
    ax.plot(t, raw['movement_rms'].values, color='#E74C3C', lw=0.8, alpha=0.8, label='movement_rms')
    if 'roll_deg' in raw.columns:
        ax2.plot(t, raw['roll_deg'].values, color='#8E44AD', lw=0.5, alpha=0.4, label='roll')
    if 'pitch_deg' in raw.columns:
        ax2.plot(t, raw['pitch_deg'].values, color='#F39C12', lw=0.5, alpha=0.4, label='pitch')
    ax.set_ylabel('Motion RMS', fontsize=8, color='#E74C3C')
    ax2.set_ylabel('Orientation (deg)', fontsize=8, color='#8E44AD')
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax.set_xlabel('Time (hr)', fontsize=9)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper right', ncol=3)

    # Hypnogram color bar at bottom of last panel
    stages = raw['stage_code'].values
    y_bottom = ax.get_ylim()[0]
    bar_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
    w = t[1] - t[0] if len(t) > 1 else 0.01
    for code in STAGE_ORDER:
        mask = stages == code
        if mask.any():
            ax.bar(t[mask], bar_height, bottom=y_bottom, width=w,
                   color=STAGE_COLORS[code], alpha=0.6, label=STAGE_LABELS[code])

    fig.savefig(PLOT_DIR / f'{sess}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print('=' * 60)
    print('Residualized Timeseries Plots')
    print('=' * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    pq_path = ART_DIR / 'thorax_caponly_epochs.parquet'
    print(f'\nLoading {pq_path}')
    df = pd.read_parquet(pq_path)
    print(f'  {len(df)} epochs, {df.session.nunique()} sessions')

    # Columns to residualize
    cap_stat_cols = [f'{ch}_{stat}' for ch in CHANNELS for stat in STATS]
    all_resid_cols = [TARGET] + [c for c in cap_stat_cols if c in df.columns]

    # Mean-subtract per session (preserve scale so slow drifts are visible)
    for col in all_resid_cols + ACCEL_COLS:
        if col in df.columns:
            df[col] = df.groupby('session')[col].transform(lambda x: x - x.mean())

    df_raw = df.copy()

    # Residualize
    print('\nResidualizing motion...')
    df_resid, motion_r2 = residualize_per_session(df, all_resid_cols, ACCEL_COLS)
    mean_mr2 = np.nanmean(list(motion_r2.values()))
    print(f'  Mean motion R2 on thorax: {mean_mr2:.3f}')

    # Plot each session
    sessions = sorted(df['session'].unique())
    print(f'\nPlotting {len(sessions)} sessions...')
    for sess in sessions:
        plot_session(sess, df_raw, df_resid)
        print(f'  {sess} done')

    print(f'\nAll plots saved to {PLOT_DIR}')


if __name__ == '__main__':
    main()
