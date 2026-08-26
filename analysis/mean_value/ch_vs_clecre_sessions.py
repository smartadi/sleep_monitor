"""
CH versus CLE-CRE across all 12 sessions — are they the same signal?

The source paper describes CH as the interhemispheric difference, "the difference
between left and right hemispheres". Read literally that makes CH and the
arithmetic CLE-CRE the same physical quantity, and the mean-value analyses could
use either. They are not the same, and which one is used changes the answer, so
this compares them session by session in the mean-value domain the analyses
actually work in.

The three questions a reader has
-------------------------------
1. Do the two traces move together over the night? Both are shown mean-centred,
   in femtofarads, over the hypnogram — the same presentation as the rest of the
   mean-value figures, so the comparison is read directly rather than inferred
   from a correlation coefficient.
2. If not always, WHEN do they agree? Agreement is plotted against how far the
   channel actually moves during the night, because two nearly flat traces are
   uncorrelated for trivial reasons.
3. Does it matter for the results? Both channels are ranked against sleep stage,
   and the per-session offset and gain are reported: if CH were the arithmetic
   difference, the offset would be 0 fF and the slope 1.

Agreement is quantified three ways, deliberately: the LEVEL correlation (which
two slowly drifting traces can inflate), the CHANGE correlation over one minute
(offset- and trend-immune, the honest number), and band-resolved coherence
(where in frequency they agree).

This is the across-sessions view. The mechanism-level work — coherence spectra,
motion regression, per-session traces — is in ch_vs_diff.py, whose per-session
table this script reads.

Outputs
-------
    reports/mean_value/ch_vs_clecre_sessions.csv
    writeup/figures/channel_evolution/ch_vs_clecre_grid.png
    writeup/figures/channel_evolution/ch_vs_clecre_summary.png

Needs reports/mean_value/mean_value_epochs.csv  (mean_value_vs_stage.py)
      reports/mean_value/ch_vs_diff_per_session.csv, ch_vs_diff_by_stage.csv
                                                     (ch_vs_diff.py)

Usage:
    .venv/Scripts/python.exe analysis/mean_value/ch_vs_clecre_sessions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, CAP_UNIT,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'mean_value'
FIG_DIR = ROOT / 'writeup' / 'figures' / 'channel_evolution'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_CODES = [0, 1, 2, 3]
STAGE_LADDER = [4, 0, 3, 2, 1]
CH_COLOR = '#2980B9'
DIFF_COLOR = '#E67E22'
CLIP_SD = 5.0          # grid panels clip at this many session sd


def sleep_mask(codes):
    is_sleep = np.isin(codes, SLEEP_CODES)
    if is_sleep.sum() < 20:
        return np.ones(len(codes), bool)
    on = int(np.argmax(is_sleep))
    off = len(is_sleep) - 1 - int(np.argmax(is_sleep[::-1]))
    m = np.zeros(len(codes), bool); m[on:off + 1] = True
    return m


def per_session(df):
    rows, traces = [], {}
    for lbl, g in df.groupby('session'):
        g = g.sort_values('epoch')
        sm = sleep_mask(g['stage_code'].to_numpy())
        ch = g['mean_CH'].to_numpy()
        di = g['mean_CLE-CRE'].to_numpy()
        mu_ch, mu_di = float(np.nanmean(ch[sm])), float(np.nanmean(di[sm]))
        c, d = ch - mu_ch, di - mu_di
        ok = sm & np.isfinite(c) & np.isfinite(d)

        # 30 s level agreement and 1-min change agreement
        r_lvl = float(np.corrcoef(c[ok], d[ok])[0, 1]) if ok.sum() > 20 else np.nan
        lag = 2                                     # 2 epochs = 1 min
        dc, dd = c[lag:] - c[:-lag], d[lag:] - d[:-lag]
        mk = np.isfinite(dc) & np.isfinite(dd) & ok[lag:]
        r_chg = float(np.corrcoef(dc[mk], dd[mk])[0, 1]) if mk.sum() > 20 else np.nan

        # gain, and how far each channel actually travels
        slope = float(np.polyfit(d[ok], c[ok], 1)[0]) if ok.sum() > 20 else np.nan
        rows.append({
            'session': lbl, 'subject': g['subject'].iloc[0],
            'night': int(g['night'].iloc[0]), 'n_epochs': int(ok.sum()),
            'mean_CH_fF': mu_ch, 'mean_DIFF_fF': mu_di,
            'offset_fF': mu_ch - mu_di,
            'sd_CH_fF': float(np.nanstd(c[ok])), 'sd_DIFF_fF': float(np.nanstd(d[ok])),
            'sd_ratio_CH_over_DIFF': float(np.nanstd(c[ok]) / (np.nanstd(d[ok]) + 1e-9)),
            'slope_CH_on_DIFF': slope,
            'r_level_30s': r_lvl, 'r_change_1min': r_chg,
        })
        traces[lbl] = {'t': g['t_hr'].to_numpy(), 'ch': c, 'diff': d,
                       'codes': g['stage_code'].to_numpy(),
                       'mu_ch': mu_ch, 'mu_di': mu_di,
                       'sd_ch': np.nanstd(c[ok]), 'sd_di': np.nanstd(d[ok])}
    return pd.DataFrame(rows).sort_values('session').reset_index(drop=True), traces


def fig_grid(traces, sessions, out):
    fig, axes = plt.subplots(6, 2, figsize=(20, 21))
    for ax, lbl in zip(axes.ravel(), sessions):
        tr = traces[lbl]
        t, codes = tr['t'], tr['codes']
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                       alpha=0.13, lw=0)
        # Each channel on its own axis: they differ in scale by up to ~9x, so a
        # shared axis would hide the shape of whichever one is smaller.
        lim_c = CLIP_SD * max(tr['sd_ch'], 1e-6)
        ax.plot(t, np.clip(tr['ch'], -lim_c, lim_c), lw=1.1, color=CH_COLOR, zorder=3)
        ax.axhline(0, color='#2C3E50', ls='--', lw=0.8, zorder=2)
        ax.set_ylim(-lim_c * 1.1, lim_c * 1.1)
        ax.set_ylabel(f'CH − mean ({CAP_UNIT})', fontsize=8, color=CH_COLOR)
        ax.tick_params(axis='y', labelcolor=CH_COLOR, labelsize=7)

        ax2 = ax.twinx()
        lim_d = CLIP_SD * max(tr['sd_di'], 1e-6)
        ax2.plot(t, np.clip(tr['diff'], -lim_d, lim_d), lw=1.1, color=DIFF_COLOR,
                 zorder=4)
        ax2.set_ylim(-lim_d * 1.1, lim_d * 1.1)
        ax2.set_ylabel(f'CLE−CRE − mean ({CAP_UNIT})', fontsize=8, color=DIFF_COLOR)
        ax2.tick_params(axis='y', labelcolor=DIFF_COLOR, labelsize=7)

        ax.set_title(f'{lbl}    CH mean {tr["mu_ch"]:,.0f}   '
                     f'CLE−CRE mean {tr["mu_di"]:,.0f} {CAP_UNIT}',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel('Time (hours)', fontsize=9)
    handles = [plt.Line2D([], [], color=CH_COLOR, lw=2.5, label='CH (hardware), left axis'),
               plt.Line2D([], [], color=DIFF_COLOR, lw=2.5,
                          label='CLE−CRE (arithmetic), right axis')]
    handles += [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
                for k in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=7, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.004))
    fig.suptitle('CH and CLE−CRE over the night, each referenced to its own session '
                 f'mean — all 12 sessions.  Independent axes (the two differ in scale '
                 f'by up to 9x); traces clipped at ±{CLIP_SD:.0f} sd.',
                 fontsize=14.5, fontweight='bold', y=1.019)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)


def fig_summary(sess, band, stage, out):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    xs = np.arange(len(sess))

    # A — offset: 0 if CH were the arithmetic difference
    ax = axes[0, 0]
    ax.barh(sess['session'], sess['offset_fF'], color='#C0392B', alpha=0.85,
            edgecolor='k', lw=0.5)
    ax.axvline(0, color='#27AE60', lw=1.8, ls='--', label='0 = identical channels')
    for y, v in enumerate(sess['offset_fF']):
        ax.annotate(f'{v:,.0f}', (v, y), fontsize=7, va='center', ha='right',
                    xytext=(-4, 0), textcoords='offset points',
                    color='white', fontweight='bold')
    ax.set_xlabel(f'mean CH − mean (CLE−CRE)   ({CAP_UNIT})')
    ax.set_title('A  Static offset', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # B — gain
    ax = axes[0, 1]
    ax.barh(sess['session'], sess['slope_CH_on_DIFF'], color=CH_COLOR, alpha=0.85,
            edgecolor='k', lw=0.5)
    ax.axvline(1, color='#27AE60', lw=1.8, ls='--', label='slope = 1 (identical)')
    ax.axvline(0, color='k', lw=1.1)
    for y, v in enumerate(sess['slope_CH_on_DIFF']):
        ax.annotate(f'{v:.2f}', (v, y), fontsize=7, va='center',
                    xytext=(4 if v >= 0 else -4, 0), textcoords='offset points',
                    ha='left' if v >= 0 else 'right')
    ax.set_xlabel('OLS slope of CH on (CLE−CRE), 30 s means')
    ax.set_title('B  Gain — spans a sign change and an order of magnitude',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # C — agreement: level vs change
    ax = axes[0, 2]
    w = 0.38; y = np.arange(len(sess))
    ax.barh(y - w / 2, sess['r_level_30s'], w, color='#8E44AD', alpha=0.85,
            edgecolor='k', lw=0.4, label='level (30 s means)')
    ax.barh(y + w / 2, sess['r_change_1min'], w, color='#16A085', alpha=0.85,
            edgecolor='k', lw=0.4, label='change over 1 min')
    ax.set_yticks(y); ax.set_yticklabels(sess['session'])
    ax.axvline(0, color='k', lw=1.1); ax.set_xlim(-1, 1)
    ax.set_xlabel('Pearson r')
    ax.set_title('C  Agreement — negative on 3 of 12 nights\n(the two channels can '
                 'move in opposite directions)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # D — agreement vs excursion
    ax = axes[1, 0]
    ax.scatter(sess['sd_DIFF_fF'], sess['r_change_1min'], s=95, color='#16A085',
               edgecolor='k', lw=0.6, zorder=3)
    for _, r in sess.iterrows():
        ax.annotate(r['session'], (r['sd_DIFF_fF'], r['r_change_1min']), fontsize=7.5,
                    xytext=(5, 3), textcoords='offset points')
    m = sess[['sd_DIFF_fF', 'r_change_1min']].dropna()
    rho, p = spearmanr(m['sd_DIFF_fF'], m['r_change_1min'])
    ax.axhline(0, color='k', lw=1.1)
    ax.set_xscale('log')
    ax.set_xlabel(f'how far CLE−CRE travels — sd about its session mean '
                  f'({CAP_UNIT}, log)')
    ax.set_ylabel('r (1 min change)')
    ax.set_title(f'D  They agree on the nights with a big signal\n'
                 f'Spearman rho = {rho:.2f}, p = {p:.3f}',
                 fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.2)

    # E — where in frequency (from ch_vs_diff.py)
    ax = axes[1, 1]
    if band is not None:
        bands = [('coh_slow', 'slow drift\n<0.01 Hz'), ('coh_resp', 'respiratory\n0.1–0.5 Hz'),
                 ('coh_card', 'cardiac\n0.5–3 Hz')]
        data = [band[c].dropna().to_numpy() for c, _ in bands]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                        medianprops=dict(color='k', lw=1.7))
        for patch, c in zip(bp['boxes'], ['#7F8C8D', '#27AE60', '#E67E22']):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        for i, v in enumerate(data):
            ax.plot(np.full(len(v), i + 1) + np.linspace(-0.15, 0.15, len(v)), v, 'o',
                    ms=4, color='k', alpha=0.55)
            ax.annotate(f'{np.median(v):.2f}', (i + 1, np.median(v)), fontsize=9,
                        fontweight='bold', ha='center', xytext=(0, 8),
                        textcoords='offset points')
        ax.set_xticklabels([l for _, l in bands])
        ax.set_ylim(0, 1); ax.set_ylabel('magnitude-squared coherence')
    ax.set_title('E  Where they agree\none point = one session',
                 fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, axis='y', alpha=0.2)

    # F — do they rank the sleep stages the same way?
    ax = axes[1, 2]
    if stage is not None:
        order = [STAGE_LABELS[c] for c in STAGE_LADDER]
        xp = np.arange(len(order)); w2 = 0.36
        for k, (col, lbl, colr) in enumerate([
                ('CH_centred_fF', 'CH (hardware)', CH_COLOR),
                ('DIFF_centred_fF', 'CLE−CRE (arithmetic)', DIFF_COLOR)]):
            vals = [stage.loc[stage['stage'] == st, col].dropna().to_numpy()
                    for st in order]
            pos = xp + (k - 0.5) * w2
            bp = ax.boxplot(vals, positions=pos, widths=w2 * 0.85, patch_artist=True,
                            boxprops=dict(facecolor=colr, alpha=0.55, lw=0.8),
                            medianprops=dict(color='k', lw=1.3), showfliers=False,
                            whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
            ax.plot([], [], color=colr, lw=6, alpha=0.55, label=lbl)
        ax.axhline(0, color='k', lw=1.1, ls=':')
        ax.set_xticks(xp); ax.set_xticklabels(order)
        ax.set_ylabel(f'session-centred level ({CAP_UNIT})')
        ax.legend(fontsize=8.5)
    ax.set_title('F  Same stage ordering?', fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, axis='y', alpha=0.2)

    fig.suptitle('CH vs CLE−CRE across 12 sessions — the paper calls CH the '
                 'interhemispheric difference, but it is not the arithmetic one:\n'
                 'a large offset (A), a gain that changes sign between subjects (B), '
                 'and agreement only on the nights with a large excursion (C, D).',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=190, bbox_inches='tight'); plt.close(fig)


def main():
    mv = REPORT_DIR / 'mean_value_epochs.csv'
    if not mv.exists():
        sys.exit(f'Missing {mv}. Run mean_value_vs_stage.py first.')
    df = pd.read_csv(mv)

    band_csv = REPORT_DIR / 'ch_vs_diff_per_session.csv'
    stage_csv = REPORT_DIR / 'ch_vs_diff_by_stage.csv'
    band = pd.read_csv(band_csv) if band_csv.exists() else None
    stage = pd.read_csv(stage_csv) if stage_csv.exists() else None
    if band is None:
        print('  (no ch_vs_diff_per_session.csv — panel E will be blank; '
              'run ch_vs_diff.py for the band-resolved coherence)')

    sess, traces = per_session(df)
    if band is not None:
        sess = sess.merge(band[['session', 'coh_slow', 'coh_resp', 'coh_card',
                                'r_resp', 'r_card']], on='session', how='left')
    sess.to_csv(REPORT_DIR / 'ch_vs_clecre_sessions.csv', index=False)

    sessions = sorted(traces)
    fig_grid(traces, sessions, FIG_DIR / 'ch_vs_clecre_grid.png')
    fig_summary(sess, band, stage, FIG_DIR / 'ch_vs_clecre_summary.png')

    print('=' * 110)
    print(f'CH vs CLE-CRE across sessions  (mean-value domain, 30 s epochs, {CAP_UNIT})')
    print('=' * 110)
    show = sess[['session', 'mean_CH_fF', 'mean_DIFF_fF', 'offset_fF',
                 'sd_CH_fF', 'sd_DIFF_fF', 'sd_ratio_CH_over_DIFF',
                 'slope_CH_on_DIFF', 'r_level_30s', 'r_change_1min']].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(2)
    print(show.to_string(index=False))

    print('\nIf CH were the arithmetic CLE-CRE, offset would be 0 and slope 1. Instead:')
    print(f"  offset       median {sess['offset_fF'].median():9,.1f} {CAP_UNIT}   "
          f"[{sess['offset_fF'].min():,.1f}, {sess['offset_fF'].max():,.1f}]")
    print(f"  slope        median {sess['slope_CH_on_DIFF'].median():9.2f}       "
          f"[{sess['slope_CH_on_DIFF'].min():.2f}, {sess['slope_CH_on_DIFF'].max():.2f}]"
          f"   ({(sess['slope_CH_on_DIFF'] < 0).sum()}/12 sessions have a NEGATIVE gain)")
    print(f"  sd ratio     median {sess['sd_ratio_CH_over_DIFF'].median():9.2f}       "
          f"[{sess['sd_ratio_CH_over_DIFF'].min():.2f}, "
          f"{sess['sd_ratio_CH_over_DIFF'].max():.2f}]   "
          f"CH moves this many times as far as CLE-CRE")
    print(f"  r (level)    median {sess['r_level_30s'].median():9.2f}       "
          f"[{sess['r_level_30s'].min():.2f}, {sess['r_level_30s'].max():.2f}]"
          f"   ({(sess['r_level_30s'] < 0).sum()}/12 negative)")
    print(f"  r (1 min chg)  median {sess['r_change_1min'].median():9.2f}       "
          f"[{sess['r_change_1min'].min():.2f}, {sess['r_change_1min'].max():.2f}]"
          f"   ({(sess['r_change_1min'] < 0).sum()}/12 negative)")

    m = sess[['sd_DIFF_fF', 'r_change_1min']].dropna()
    rho, p = spearmanr(m['sd_DIFF_fF'], m['r_change_1min'])
    print(f"\n  agreement vs excursion: Spearman rho = {rho:.2f} (p = {p:.3f}) -- the two "
          f"channels\n  track each other on the nights where the differential actually "
          f"moves, and\n  decouple on the quiet nights.")

    if band is not None:
        print('\nBand-resolved coherence (from ch_vs_diff.py), median across sessions:')
        for c, nm in [('coh_slow', 'slow drift <0.01 Hz'),
                      ('coh_resp', 'respiratory 0.1-0.5 Hz'),
                      ('coh_card', 'cardiac 0.5-3 Hz')]:
            print(f'  {nm:24s} {band[c].median():.3f}  '
                  f'[{band[c].min():.3f}, {band[c].max():.3f}]')

    print(f'\nTable   -> {REPORT_DIR / "ch_vs_clecre_sessions.csv"}')
    print(f'Figures -> {FIG_DIR}')


if __name__ == '__main__':
    main()
