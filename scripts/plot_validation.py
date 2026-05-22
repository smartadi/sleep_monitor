#!/usr/bin/env python
"""
Generate validation study figures from precomputed artifacts.

Requires: run scripts/run_validation.py first.

Outputs (in notebooks/plots/):
    validation_bland_altman.png     — Bland-Altman for resp + cardiac (2×1)
    validation_scatter.png          — CAP vs GT scatter, colored by stage (2×1)
    validation_stage_boxplots.png   — Abs error by sleep stage (2×1)
    validation_session_mae.png      — Per-session MAE bar chart (2×1)
    validation_summary_table.png    — Summary metrics as a rendered table

Usage
-----
    python scripts/plot_validation.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor.viz import (
    plot_bland_altman, plot_scatter_agreement,
    plot_stage_boxplots, plot_session_bar,
)

ART_DIR  = ROOT / 'artifacts'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'rate_analysis'


def load_data():
    win = pd.read_parquet(ART_DIR / 'validation_windows.parquet')
    sess = pd.read_csv(ART_DIR / 'validation_session.csv')
    stage = pd.read_csv(ART_DIR / 'validation_stage.csv')
    return win, sess, stage


def fig_bland_altman(win: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_bland_altman(win['cap_resp_hz'].values, win['gt_resp_hz'].values,
                      axes[0], scale=60.0, unit='br/min',
                      title='Respiratory Rate — Bland-Altman',
                      color='#27AE60', stage_codes=win['stage_code'].values)
    plot_bland_altman(win['cap_card_hz'].values, win['gt_card_hz'].values,
                      axes[1], scale=60.0, unit='BPM',
                      title='Cardiac Rate — Bland-Altman',
                      color='#E74C3C', stage_codes=win['stage_code'].values)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'validation_bland_altman.png', dpi=200, bbox_inches='tight')
    print("  validation_bland_altman.png")
    plt.close(fig)


def fig_scatter(win: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_scatter_agreement(win['cap_resp_hz'].values, win['gt_resp_hz'].values,
                           axes[0], scale=60.0, unit='br/min',
                           title='Respiratory Rate — Agreement',
                           stage_codes=win['stage_code'].values)
    plot_scatter_agreement(win['cap_card_hz'].values, win['gt_card_hz'].values,
                           axes[1], scale=60.0, unit='BPM',
                           title='Cardiac Rate — Agreement',
                           stage_codes=win['stage_code'].values)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'validation_scatter.png', dpi=200, bbox_inches='tight')
    print("  validation_scatter.png")
    plt.close(fig)


def fig_stage_boxplots(win: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_stage_boxplots(win, 'resp', axes[0], scale=60.0, unit='br/min',
                        title='Respiratory Rate — Error by Sleep Stage')
    plot_stage_boxplots(win, 'card', axes[1], scale=60.0, unit='BPM',
                        title='Cardiac Rate — Error by Sleep Stage')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'validation_stage_boxplots.png', dpi=200, bbox_inches='tight')
    print("  validation_stage_boxplots.png")
    plt.close(fig)


def fig_session_mae(sess: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_session_bar(sess, 'resp_mae', axes[0], unit='br/min',
                     title='Respiratory MAE by Session', color='#27AE60')
    plot_session_bar(sess, 'card_mae', axes[1], unit='BPM',
                     title='Cardiac MAE by Session', color='#E74C3C')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'validation_session_mae.png', dpi=200, bbox_inches='tight')
    print("  validation_session_mae.png")
    plt.close(fig)


def fig_summary_table(sess: pd.DataFrame, stage: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # --- Per-session table ---
    ax = axes[0]
    ax.axis('off')
    cols = ['session', 'subject', 'k_resp', 'k_card',
            'resp_mae', 'resp_rmse', 'resp_r', 'resp_bias', 'resp_coverage',
            'card_mae', 'card_rmse', 'card_r', 'card_bias', 'card_coverage']
    headers = ['Session', 'Subject', 'k_r', 'k_c',
               'R MAE', 'R RMSE', 'R r', 'R Bias', 'R Cov',
               'C MAE', 'C RMSE', 'C r', 'C Bias', 'C Cov']
    tbl_data = sess[cols].copy()
    for c in cols[2:]:
        tbl_data[c] = tbl_data[c].apply(lambda v: f'{v:.2f}' if pd.notna(v) else '')
    table = ax.table(cellText=tbl_data.values, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.auto_set_column_width(range(len(headers)))
    # highlight aggregate row
    for j in range(len(headers)):
        table[len(sess), j].set_facecolor('#FFF3CD')
    ax.set_title('Per-Session Validation Metrics', fontsize=10, pad=10)

    # --- Per-stage table ---
    ax2 = axes[1]
    ax2.axis('off')
    scols = ['stage', 'resp_mae', 'resp_r', 'resp_n', 'card_mae', 'card_r', 'card_n']
    sheaders = ['Stage', 'R MAE', 'R r', 'R n', 'C MAE', 'C r', 'C n']
    stbl = stage[scols].copy()
    for c in scols[1:]:
        stbl[c] = stbl[c].apply(lambda v: f'{v:.2f}' if pd.notna(v) and 'n' not in c else
                                (f'{int(v)}' if pd.notna(v) else ''))
    table2 = ax2.table(cellText=stbl.values, colLabels=sheaders,
                       loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.auto_set_column_width(range(len(sheaders)))
    ax2.set_title('Per-Stage Validation Metrics (Pooled)', fontsize=10, pad=10)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'validation_summary_table.png', dpi=200, bbox_inches='tight')
    print("  validation_summary_table.png")
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading validation artifacts...")
    win, sess, stage = load_data()
    print(f"  {len(win)} windows, {len(sess)} session rows, {len(stage)} stage rows\n")

    print("Generating figures:")
    fig_bland_altman(win)
    fig_scatter(win)
    fig_stage_boxplots(win)
    fig_session_mae(sess)
    fig_summary_table(sess, stage)
    print(f"\nAll figures saved to {PLOT_DIR}/")


if __name__ == '__main__':
    main()
