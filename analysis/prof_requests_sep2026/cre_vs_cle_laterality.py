"""
Request C (Prof, 2026-09-09): "Description of CRE and CLE for SEC-based SWS (ridge
detection). I would like to see whether the ridges are occurring at both or one brain."

Question
--------
Are the SEC spectral ridges (persistent respiratory / cardiac combs and the
harmonic-comb events) bilateral, or do they sit on one side? CLE = left temple,
CRE = right temple, CH = the sensor's own interhemispheric-difference channel.

Approach (reuses already-computed ridge tables — no recompute)
--------
  * band_ridge_epochs.parquet  : per 30 s epoch, per channel, per band (resp/card),
    ridge_present + total_ridge_power + max_prominence.
  * ridge_overlay_epochs.parquet: per epoch, per channel, prominence_score / n_ridges
    (the broadband SWS ridge tracker).
  * ladder_quantify/per_window_channels.parquet: per epoch, per channel harmonic
    ladder flags (the harmonic-comb events).
Motion-masked epochs are dropped. Ridge occurrence is summarised per channel overall
and within N3 (SWS), and a per-night left/right laterality index
    LI = (CRE - CLE) / (CRE + CLE)     (+1 = right only, -1 = left only, 0 = bilateral)
is reported per subject.

Outputs
-------
  reports/prof_requests_sep2026/laterality_channel_stage.csv
  reports/prof_requests_sep2026/laterality_index_per_session.csv
  notebooks/plots/prof_requests_sep2026/C_laterality_by_channel_stage.png
  notebooks/plots/prof_requests_sep2026/C_laterality_index.png

Usage:
    .venv/Scripts/python.exe analysis/prof_requests_sep2026/cre_vs_cle_laterality.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.config import STAGE_LABELS, CAP_COLORS

ROOT = Path(__file__).resolve().parents[2]
SW = ROOT / 'reports' / 'slow_wave'
PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'prof_requests_sep2026'
REPORT_DIR = ROOT / 'reports' / 'prof_requests_sep2026'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHANS = ['CLE', 'CRE', 'CH']            # left, right, interhemispheric
CH_COLOR = {'CLE': CAP_COLORS['CLE'], 'CRE': CAP_COLORS['CRE'], 'CH': CAP_COLORS['CH']}
STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']


def _motion_free(df):
    for c in ('motion_masked', 'motion'):
        if c in df.columns:
            return df[~df[c].astype(bool)]
    return df


def load_tables():
    band = pd.read_parquet(SW / 'band_ridge_epochs.parquet')
    overlay = pd.read_parquet(SW / 'overlay' / 'ridge_overlay_epochs.parquet')
    ladder = pd.read_parquet(SW / 'ladder_quantify' / 'per_window_channels.parquet')
    return _motion_free(band), _motion_free(overlay), _motion_free(ladder)


def occurrence_table(band, overlay, ladder):
    """Ridge-occurrence rate (%) per channel x stage for each ridge type."""
    rows = []
    # persistent resp / cardiac ridges
    for bname in ('resp', 'card'):
        sub = band[band['band'] == bname]
        for ch in CHANS:
            d = sub[sub['channel'] == ch]
            for st in STAGES:
                dd = d[d['stage_label'] == st]
                if len(dd):
                    rows.append({'ridge': f'{bname}-ridge', 'channel': ch, 'stage': st,
                                 'occ_pct': 100 * dd['ridge_present'].mean(),
                                 'mean_power': dd['total_ridge_power'].mean(),
                                 'n': len(dd)})
    # harmonic-comb ladders
    for ch in CHANS:
        d = ladder[ladder['channel'] == ch]
        for st in STAGES:
            dd = d[d['stage_label'] == st]
            if len(dd):
                rows.append({'ridge': 'harmonic-comb', 'channel': ch, 'stage': st,
                             'occ_pct': 100 * dd['combined_is_ladder'].mean(),
                             'mean_power': dd['combined_n_rungs'].mean(),
                             'n': len(dd)})
    # broadband SWS ridge prominence (overlay tracker)
    for ch in CHANS:
        d = overlay[overlay['channel'] == ch]
        for st in STAGES:
            dd = d[d['stage_label'] == st]
            if len(dd):
                rows.append({'ridge': 'SWS-ridge(prom)', 'channel': ch, 'stage': st,
                             'occ_pct': 100 * (dd['n_strong_ridges'] > 0).mean(),
                             'mean_power': dd['prominence_score'].mean(),
                             'n': len(dd)})
    return pd.DataFrame(rows)


def laterality_index(band, ladder):
    """Per-session LI = (CRE-CLE)/(CRE+CLE) on ridge occupancy, overall and N3."""
    rows = []
    def occ(df, flag, ch, sess, stage=None):
        d = df[(df['channel'] == ch) & (df['session'] == sess)]
        if stage is not None:
            d = d[d['stage_label'] == stage]
        return float(d[flag].mean()) if len(d) else np.nan

    sessions = sorted(band['session'].unique())
    for sess in sessions:
        subject = band[band['session'] == sess]['subject'].iloc[0]
        for label, df, flag in [('resp-ridge', band[band['band'] == 'resp'], 'ridge_present'),
                                ('card-ridge', band[band['band'] == 'card'], 'ridge_present'),
                                ('harmonic-comb', ladder, 'combined_is_ladder')]:
            for scope, stage in [('all', None), ('N3', 'N3')]:
                l = occ(df, flag, 'CLE', sess, stage)
                r = occ(df, flag, 'CRE', sess, stage)
                li = (r - l) / (r + l) if (r + l) and np.isfinite(r + l) and (r + l) > 0 else np.nan
                rows.append({'session': sess, 'subject': subject, 'ridge': label,
                             'scope': scope, 'occ_CLE': l, 'occ_CRE': r, 'LI': li})
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_channel_stage(occ, out):
    ridge_types = ['resp-ridge', 'card-ridge', 'harmonic-comb', 'SWS-ridge(prom)']
    fig, axes = plt.subplots(1, len(ridge_types), figsize=(5.0 * len(ridge_types), 4.8),
                             sharex=True)
    x = np.arange(len(STAGES))
    w = 0.26
    for ax, rt in zip(axes, ridge_types):
        d = occ[occ['ridge'] == rt]
        for k, ch in enumerate(CHANS):
            vals = [d[(d['channel'] == ch) & (d['stage'] == st)]['occ_pct'].mean()
                    for st in STAGES]
            ax.bar(x + (k - 1) * w, vals, w, color=CH_COLOR[ch], label=ch, alpha=0.9)
        ax.set_title(rt, fontsize=12, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(STAGES, fontsize=9)
        ax.set_ylabel('epochs with ridge (%)', fontsize=10)
        ax.grid(True, axis='y', alpha=0.15)
    axes[0].legend(title='channel', fontsize=10)
    fig.suptitle('C. SEC ridge occurrence by channel and stage — left (CLE) vs right (CRE) vs interhemispheric (CH)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def fig_laterality_index(li, out):
    ridge_types = ['resp-ridge', 'card-ridge', 'harmonic-comb']
    fig, axes = plt.subplots(1, len(ridge_types), figsize=(5.0 * len(ridge_types), 5.0),
                             sharey=True)
    for ax, rt in zip(axes, ridge_types):
        d = li[(li['ridge'] == rt) & (li['scope'] == 'all')].sort_values('session')
        y = np.arange(len(d))
        colors = ['#8E44AD' if v > 0 else '#27AE60' for v in d['LI'].fillna(0)]
        ax.barh(y, d['LI'].values, color=colors, alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(d['session'], fontsize=9)
        ax.axvline(0, color='k', lw=0.9)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(rt, fontsize=12, fontweight='bold')
        ax.set_xlabel('laterality index\n(−1 left · 0 bilateral · +1 right)', fontsize=10)
        ax.grid(True, axis='x', alpha=0.15)
    axes[0].invert_yaxis()
    fig.suptitle('C. Left/right laterality of SEC ridges per night '
                 '(green = left-dominant, purple = right-dominant)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def main():
    print('=' * 66)
    print('C. CRE vs CLE laterality of SEC ridges (SWS ridge detection)')
    print('=' * 66)
    band, overlay, ladder = load_tables()

    occ = occurrence_table(band, overlay, ladder)
    occ.to_csv(REPORT_DIR / 'laterality_channel_stage.csv', index=False)

    li = laterality_index(band, ladder)
    li.to_csv(REPORT_DIR / 'laterality_index_per_session.csv', index=False)

    # console summary
    print('\nOverall ridge occurrence (%), motion-free, all stages pooled:')
    for rt in ['resp-ridge', 'card-ridge', 'harmonic-comb', 'SWS-ridge(prom)']:
        d = occ[occ['ridge'] == rt]
        cells = []
        for ch in CHANS:
            v = np.average(d[d['channel'] == ch]['occ_pct'],
                           weights=d[d['channel'] == ch]['n'])
            cells.append(f'{ch}={v:5.2f}%')
        print(f'  {rt:16s}  ' + '  '.join(cells))

    print('\nMedian laterality index (all stages), by ridge type '
          '(+1 right-only, -1 left-only):')
    for rt in ['resp-ridge', 'card-ridge', 'harmonic-comb']:
        d = li[(li['ridge'] == rt) & (li['scope'] == 'all')]['LI'].dropna()
        n_right = int((d > 0.2).sum()); n_left = int((d < -0.2).sum())
        n_bil = int((d.abs() <= 0.2).sum())
        print(f'  {rt:16s}  median LI={d.median():+.2f}  '
              f'(right-dom {n_right} / bilateral {n_bil} / left-dom {n_left} nights)')

    print('\nWriting figures...')
    fig_channel_stage(occ, PLOT_DIR / 'C_laterality_by_channel_stage.png')
    fig_laterality_index(li, PLOT_DIR / 'C_laterality_index.png')
    print('\nDone.')


if __name__ == '__main__':
    main()
