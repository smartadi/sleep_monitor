"""
scripts/run_rate_detection.py — run classical rate detection across all sessions
and produce summary plots + statistics.

Uses the default pipeline (CLE-CRE + OLS + ACF) for respiratory and cardiac bands.
Writes per-session metrics CSV + four plots to notebooks/plots/.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleep_monitor import SESSION_META, load_session
from sleep_monitor.evaluate import PipelineConfig, run_pipeline, evaluate_pipeline

PLOTS_DIR = ROOT / 'notebooks' / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / 'artifacts' / 'rate_detection'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    'resp':    PipelineConfig(band='resp',    channel='CLE-CRE', preproc='ols',
                              estimator='acf', win_s=30.0, step_s=5.0),
    'cardiac': PipelineConfig(band='cardiac', channel='CLE-CRE', preproc='ols',
                              estimator='acf', win_s=30.0, step_s=5.0),
}


def run_all():
    per_session_metrics = []
    all_windows = {'resp': [], 'cardiac': []}

    for meta in SESSION_META:
        label = meta['label']
        print(f'[{label}] loading…', flush=True)
        sess = load_session(meta['idx'])
        for band, cfg in CONFIGS.items():
            print(f'  {band}: running pipeline…', flush=True)
            df = run_pipeline(sess, cfg)
            m = evaluate_pipeline(df, quality_gate=0.0)
            m.update(session=label, subject=meta['subject'], night=meta['night'],
                     band=band, tag=cfg.tag())
            per_session_metrics.append(m)
            df['session'] = label
            df['subject'] = meta['subject']
            all_windows[band].append(df)
            print(f'     MAE={m["mae"]:.2f}  RMSE={m["rmse"]:.2f}  '
                  f'r={m["r"]:.3f}  cov={m["coverage"]:.2f}', flush=True)

    metrics = pd.DataFrame(per_session_metrics)
    resp_w = pd.concat(all_windows['resp'], ignore_index=True)
    card_w = pd.concat(all_windows['cardiac'], ignore_index=True)
    return metrics, resp_w, card_w


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_per_session_mae(metrics: pd.DataFrame, path: Path):
    piv = metrics.pivot(index='session', columns='band', values='mae')
    piv = piv.reindex([f'S{i}N{n}' for i in range(1, 7) for n in (1, 2)])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(piv))
    w = 0.38
    ax.bar(x - w / 2, piv['resp'], w, label='Respiratory (br/min)', color='#1f77b4')
    ax.bar(x + w / 2, piv['cardiac'], w, label='Cardiac (BPM)', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=0)
    ax.set_ylabel('MAE')
    ax.set_title('Per-session MAE — CLE-CRE + OLS + ACF (win=30s, step=5s)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_scatter(all_w: pd.DataFrame, band: str, path: Path):
    df = all_w.dropna(subset=['rate_hz', 'gt_rate_hz'])
    if df.empty:
        return
    pred = df['rate_hz'].to_numpy() * 60
    ref = df['gt_rate_hz'].to_numpy() * 60
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(ref, pred, s=2, alpha=0.15, color='#1f77b4' if band == 'resp' else '#d62728')
    lo, hi = min(ref.min(), pred.min()), max(ref.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
    ax.set_xlabel(f'PSG GT rate ({ "br/min" if band=="resp" else "BPM" })')
    ax.set_ylabel('CAP predicted rate')
    ax.set_title(f'{band.capitalize()}: predicted vs GT (n={len(df):,} windows)')
    ax.grid(alpha=0.3)

    ax = axes[1]
    err = pred - ref
    mean_rate = (pred + ref) / 2
    bias = err.mean()
    loa = 1.96 * err.std()
    ax.scatter(mean_rate, err, s=2, alpha=0.15,
               color='#1f77b4' if band == 'resp' else '#d62728')
    ax.axhline(bias, color='k', lw=1.2, label=f'bias={bias:+.2f}')
    ax.axhline(bias + loa, color='k', lw=1, ls='--', label=f'±1.96σ={loa:.2f}')
    ax.axhline(bias - loa, color='k', lw=1, ls='--')
    ax.set_xlabel('Mean rate')
    ax.set_ylabel('Pred − GT')
    ax.set_title(f'{band.capitalize()}: Bland-Altman')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_timeseries_example(all_w: pd.DataFrame, band: str, session: str, path: Path):
    df = all_w[all_w['session'] == session].sort_values('t_s')
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 3.5))
    t_hr = df['t_s'] / 3600
    ax.plot(t_hr, df['gt_rate_hz'] * 60, color='black', lw=1.0, label='PSG GT', alpha=0.8)
    ax.plot(t_hr, df['rate_hz'] * 60,
            color='#1f77b4' if band == 'resp' else '#d62728',
            lw=0.8, label='CAP (ACF)', alpha=0.7)
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('br/min' if band == 'resp' else 'BPM')
    ax.set_title(f'{session} — {band} rate over night')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_metrics_heatmap(metrics: pd.DataFrame, path: Path):
    piv_mae = metrics.pivot(index='session', columns='band', values='mae')
    piv_r = metrics.pivot(index='session', columns='band', values='r')
    piv_cov = metrics.pivot(index='session', columns='band', values='coverage')
    order = [f'S{i}N{n}' for i in range(1, 7) for n in (1, 2)]
    piv_mae = piv_mae.reindex(order)
    piv_r = piv_r.reindex(order)
    piv_cov = piv_cov.reindex(order)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, piv, title, cmap in (
        (axes[0], piv_mae, 'MAE', 'Reds'),
        (axes[1], piv_r, 'Pearson r', 'Greens'),
        (axes[2], piv_cov, 'Coverage', 'Blues'),
    ):
        im = ax.imshow(piv.to_numpy(), aspect='auto', cmap=cmap)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else '—',
                        ha='center', va='center', fontsize=8,
                        color='black' if cmap == 'Greens' else 'black')
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle('Rate detection per-session metrics (CLE-CRE + OLS + ACF)')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    metrics, resp_w, card_w = run_all()

    metrics_path = OUT_DIR / 'per_session_metrics.csv'
    metrics.to_csv(metrics_path, index=False)
    resp_w.to_parquet(OUT_DIR / 'windows_resp.parquet', index=False)
    card_w.to_parquet(OUT_DIR / 'windows_cardiac.parquet', index=False)
    print(f'\nSaved metrics -> {metrics_path}')

    _plot_per_session_mae(metrics, PLOTS_DIR / 'rate_detection_mae_bar.png')
    _plot_metrics_heatmap(metrics, PLOTS_DIR / 'rate_detection_heatmap.png')
    _plot_scatter(resp_w, 'resp', PLOTS_DIR / 'rate_detection_scatter_resp.png')
    _plot_scatter(card_w, 'cardiac', PLOTS_DIR / 'rate_detection_scatter_cardiac.png')
    # pick the session with the median resp MAE for an example time-series
    resp_metrics = metrics[metrics['band'] == 'resp'].dropna(subset=['mae'])
    if len(resp_metrics):
        ex = resp_metrics.sort_values('mae').iloc[len(resp_metrics) // 2]['session']
        _plot_timeseries_example(resp_w, 'resp', ex,
                                 PLOTS_DIR / f'rate_detection_timeseries_{ex}_resp.png')
        _plot_timeseries_example(card_w, 'cardiac', ex,
                                 PLOTS_DIR / f'rate_detection_timeseries_{ex}_cardiac.png')

    # Aggregate stats
    agg = (metrics.groupby('band')
           .agg(mae_mean=('mae', 'mean'), mae_median=('mae', 'median'),
                rmse_mean=('rmse', 'mean'), r_mean=('r', 'mean'),
                bias_mean=('bias', 'mean'), coverage_mean=('coverage', 'mean'),
                n_sessions=('session', 'count'))
           .round(3))
    print('\n=== AGGREGATE METRICS ===')
    print(agg.to_string())
    print('\n=== PER-SESSION METRICS ===')
    cols = ['session', 'band', 'n_used', 'coverage', 'mae', 'rmse', 'r', 'bias']
    print(metrics[cols].round(3).to_string(index=False))
    agg.to_csv(OUT_DIR / 'aggregate_metrics.csv')


if __name__ == '__main__':
    main()
