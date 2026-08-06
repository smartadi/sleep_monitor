"""
Is the variance of the raw CAP signal related to cortical activity?

The hypothesis under test (professor's): the variance of the raw capacitive
signal reflects cortical activity, and scored EEG does not capture cortical
arousal in full — so CAP variance might carry arousal information that the EEG
scoring misses.

That is two claims, and they are separated here.

Claim 1 — CAP variance tracks EEG band power.
    Per 30 s epoch, the variance of every CAP channel is computed band by band
    and correlated against the EEG band powers from the simultaneous PSG. The
    number that matters is not the pooled correlation but whether the SIGN is
    the same in all 6 subjects: a mask that works only on some heads is not a
    cortical readout. Every correlation is also recomputed with head motion
    partialled out, because the trivial explanation for a variance-tracks-
    arousal result is that both are movement.

Claim 2 — CAP variance sees arousals an EEG readout misses.
    The intended test was the PSG's two independent arousal scorings —
    'Classification Arousal' (cortical, from the EEG) and 'Autonomic arousals'
    (from the pleth) — split into cortical / autonomic-only / neither, with the
    hypothesis living in the autonomic-only set. That test is not available in
    this dataset: the autonomic arousals are essentially a subset of the
    cortical ones (0-7 autonomic-only epochs per night out of ~800), so there is
    no population of epochs where the body aroused and the EEG scoring saw
    nothing. That overlap is reported as a result rather than worked around.

    The claim is instead tested against the EEG READOUT rather than the scoring.
    Taking the scored arousal as truth, three single-feature detectors are
    compared per session: EEG band power, CAP variance, and head motion. Then
    the test that matters — restrict to scored arousals that the EEG SPECTRAL
    readout fails to flag (epochs whose EEG beta sits below the session median,
    i.e. an EEG power detector would call them quiet) and ask whether CAP
    variance is still elevated there. If it is, the mask is carrying
    arousal-linked information that the EEG power readout does not.

Interpretation limits, stated up front
--------------------------------------
* This is an association study in 12 nights. A CAP channel on the temples is not
  an EEG electrode, and nothing here can show that CAP variance IS cortical
  activity. What it can show is whether the variance carries arousal-linked
  information beyond motion.
* The association that survives is with EEG BETA specifically. Scalp beta power
  is the EEG band most contaminated by scalp EMG, and both the temple CAP
  channel and the EEG beta band sit over the temporalis muscle. A shared
  muscle-tone term is a live alternative explanation for a CAP-variance /
  EEG-beta correlation and is NOT excluded here. The motion control removes
  gross head movement, not muscle tone.
* The arousal effect sizes are small (a few percent in variance). They are
  reported as consistent-across-subjects rather than large.

Outputs
-------
    reports/swa_validation/cap_variance_epochs.csv
    reports/swa_validation/cap_variance_eeg_corr.csv
    reports/swa_validation/cap_variance_arousal.csv
    writeup/figures/cap_variance/cap_variance_vs_eeg.png
    writeup/figures/cap_variance/cap_variance_arousal.png
    writeup/figures/cap_variance/cap_variance_trace_<SESSION>.png

Usage:
    .venv/Scripts/python.exe analysis/swa_validation/cap_variance_vs_cortical.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import welch
from scipy.stats import spearmanr, rankdata, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import (
    load_session, load_sleep_profile, load_arousals, load_autonomic_arousals,
)
from sleep_monitor.config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER, CAP_SCALE_TO_FF, CAP_UNIT_SQ,
)
from sleep_monitor.sessions import SESSION_META

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'reports' / 'swa_validation'
FIG_DIR = ROOT / 'writeup' / 'figures' / 'cap_variance'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
CAP_CH = ['CLE', 'CRE', 'CLE-CRE', 'CH']
DEMO = 'S1N1'

# CAP variance bands. 'total' is the plain time-domain variance of the epoch —
# the quantity the hypothesis is literally about; the rest say where it lives.
CAP_BANDS = {
    'resp':  (0.1, 0.5),
    'card':  (0.5, 3.0),
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'sigma': (12.0, 16.0),
    'beta':  (13.0, 30.0),
    'hf':    (30.0, 45.0),
}
EEG_BANDS = {
    'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0),
    'sigma': (12.0, 16.0), 'beta': (13.0, 30.0),
}
# Bands compared in the correlation figure (CAP row x EEG column).
CORR_CAP = ['total', 'resp', 'card', 'delta', 'theta', 'alpha', 'sigma', 'beta', 'hf']
CORR_EEG = list(EEG_BANDS)

PERI_WIN_EP = 6          # +/- 3 min around an arousal, in 30 s epochs


_trapz = getattr(np, 'trapezoid', np.trapz)


def band_power(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(_trapz(P[m], f[m])) if m.sum() > 1 else np.nan


def extract(idx, meta):
    s = load_session(idx)
    sp = load_sleep_profile(s)
    if sp is None:
        return None
    fs = s.fs
    n = int(round(fs * EPOCH_SEC))
    t_hr = s.time_hr.astype(np.float64)

    raw = {c: s.cap[c].astype(np.float64) * CAP_SCALE_TO_FF for c in ('CLE', 'CRE', 'CH')}
    raw['CLE-CRE'] = raw['CLE'] - raw['CRE']
    eeg = s.psg['EEG'].astype(np.float64)
    acc = s.cap['acc_mag'].astype(np.float64)

    # arousal masks on the sample grid
    ar = load_arousals(s)
    au = load_autonomic_arousals(s)

    def ev_mask(ev):
        m = np.zeros(len(t_hr), bool)
        if ev is None:
            return m
        for a, b in zip(ev['start_hr'], ev['end_hr']):
            i0, i1 = np.searchsorted(t_hr, a), np.searchsorted(t_hr, b)
            m[max(i0, 0):min(i1, len(m))] = True
        return m

    m_cort, m_auto = ev_mask(ar), ev_mask(au)

    rows = []
    dt_hr = EPOCH_SEC / 3600.0
    for j, t0 in enumerate(sp['t_ep_hr']):
        t1 = t0 + dt_hr
        if t0 < 0 or t1 > t_hr[-1]:
            continue
        i0, i1 = np.searchsorted(t_hr, t0), np.searchsorted(t_hr, t1)
        if i1 - i0 < n // 2:
            continue
        row = {
            'session': meta['label'], 'subject': meta['subject'],
            'night': meta['night'], 'epoch': j, 't_hr': t0 + dt_hr / 2,
            'stage_code': int(sp['codes'][j]),
            'acc_std': float(acc[i0:i1].std()),
            'cortical_arousal': bool(m_cort[i0:i1].mean() > 0.05),
            'autonomic_arousal': bool(m_auto[i0:i1].mean() > 0.05),
        }
        # CAP variance, total and band-resolved
        for c in CAP_CH:
            seg = raw[c][i0:i1]
            row[f'cap_total_{c}'] = float(np.var(seg))
            f, P = welch(seg, fs=fs, nperseg=min(len(seg), int(fs * 8)),
                         noverlap=int(fs * 4))
            for b, (lo, hi) in CAP_BANDS.items():
                row[f'cap_{b}_{c}'] = band_power(f, P, lo, hi)
        # EEG band powers
        seg = eeg[i0:i1]
        f, P = welch(seg, fs=fs, nperseg=min(len(seg), int(fs * 8)),
                     noverlap=int(fs * 4))
        for b, (lo, hi) in EEG_BANDS.items():
            row[f'eeg_{b}'] = band_power(f, P, lo, hi)
        row['eeg_total'] = band_power(f, P, 0.5, 30.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return None
    # three-way arousal class
    df['arousal_class'] = np.where(
        df['cortical_arousal'], 'cortical',
        np.where(df['autonomic_arousal'], 'autonomic-only', 'neither'))
    return df


def partial_spearman(x, y, z):
    """Spearman of x,y with z partialled out (rank-residual method)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 30:
        return np.nan, np.nan
    rx, ry, rz = (rankdata(v[m]) for v in (x, y, z))
    rz = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - rz @ np.linalg.lstsq(rz, rx, rcond=None)[0]
    ey = ry - rz @ np.linalg.lstsq(rz, ry, rcond=None)[0]
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return np.nan, np.nan
    return spearmanr(ex, ey)


def correlations(df, cap_ch='CLE-CRE'):
    """Per-session Spearman of every CAP band against every EEG band."""
    rows = []
    for lbl, g in df.groupby('session'):
        g = g[g['stage_code'] >= 0]
        if len(g) < 60:
            continue
        mot = g['acc_std'].to_numpy()
        for cb in CORR_CAP:
            x = np.log10(g[f'cap_{cb}_{cap_ch}'].to_numpy() + 1e-12)
            for eb in CORR_EEG:
                y = np.log10(g[f'eeg_{eb}'].to_numpy() + 1e-12)
                rho, p = spearmanr(x, y, nan_policy='omit')
                prho, pp = partial_spearman(x, y, mot)
                rows.append({'session': lbl, 'subject': g['subject'].iloc[0],
                             'cap_channel': cap_ch, 'cap_band': cb, 'eeg_band': eb,
                             'rho': rho, 'p': p,
                             'rho_motion_partialled': prho, 'p_motion_partialled': pp,
                             'n': int(len(g))})
    return pd.DataFrame(rows)


def arousal_table(df, cap_ch='CLE-CRE'):
    """
    Per session: how the two arousal scorings overlap, how well each single
    feature detects a scored arousal, and the EEG-missed test.
    """
    rows = []
    for lbl, g in df.groupby('session'):
        g = g[g['stage_code'] >= 0].sort_values('epoch').reset_index(drop=True)
        if len(g) < 60:
            continue
        # log, session-centred so sessions can be pooled
        cen = lambda x: (lambda v: v - np.nanmedian(v))(np.log10(np.asarray(x, float) + 1e-12))
        v = cen(g[f'cap_total_{cap_ch}'])
        eb = cen(g['eeg_beta'])
        mot = cen(g['acc_std'])
        cort = g['cortical_arousal'].to_numpy()
        auto = g['autonomic_arousal'].to_numpy()
        cls = g['arousal_class'].to_numpy()

        row = {'session': lbl, 'subject': g['subject'].iloc[0],
               'night': int(g['night'].iloc[0]), 'n_epochs': int(len(g))}
        for c in ('cortical', 'autonomic-only', 'neither'):
            m = cls == c
            row[f'n_{c}'] = int(m.sum())
            row[f'capvar_{c}'] = float(np.nanmedian(v[m])) if m.sum() >= 5 else np.nan
            row[f'motion_{c}'] = float(np.nanmedian(mot[m])) if m.sum() >= 5 else np.nan
        # how much do the two scorings overlap?
        row['n_autonomic'] = int(auto.sum())
        row['frac_autonomic_also_cortical'] = float(
            (auto & cort).sum() / max(auto.sum(), 1))

        # single-feature detection of a scored arousal
        if 5 <= cort.sum() <= len(g) - 5:
            for nm, x in (('capvar', v), ('eegbeta', eb), ('motion', mot)):
                row[f'auc_{nm}'] = float(_auc(x, cort))

            # THE TEST: scored arousals the EEG power readout would miss.
            # "miss" = EEG beta at or below the session median, i.e. an EEG
            # spectral detector sees nothing unusual.
            missed = cort & (eb <= 0)
            base = ~cort & (eb <= 0)          # matched: same low-EEG regime
            row['n_eeg_missed_arousal'] = int(missed.sum())
            if missed.sum() >= 10 and base.sum() >= 10:
                row['capvar_eeg_missed'] = float(np.nanmedian(v[missed]))
                row['capvar_eeg_missed_base'] = float(np.nanmedian(v[base]))
                row['capvar_delta_eeg_missed'] = (row['capvar_eeg_missed']
                                                  - row['capvar_eeg_missed_base'])
                row['motion_delta_eeg_missed'] = float(
                    np.nanmedian(mot[missed]) - np.nanmedian(mot[base]))
                _, pv = mannwhitneyu(v[missed], v[base], alternative='two-sided')
                row['p_eeg_missed'] = float(pv)
                row['auc_capvar_eeg_missed'] = float(
                    _auc(v[missed | base], cort[missed | base]))
        rows.append(row)
    return pd.DataFrame(rows)


def _auc(x, y):
    """Rank AUC of feature x for binary label y (no sklearn dependency here)."""
    x = np.asarray(x, float); y = np.asarray(y, bool)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = rankdata(x)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def peri_event(df, cap_ch='CLE-CRE'):
    """Epoch-aligned CAP variance around the onset of each arousal class."""
    out = {'cortical': [], 'autonomic-only': []}
    for lbl, g in df.groupby('session'):
        g = g[g['stage_code'] >= 0].sort_values('epoch').reset_index(drop=True)
        v = np.log10(g[f'cap_total_{cap_ch}'].to_numpy() + 1e-12)
        v = v - np.nanmedian(v)
        cls = g['arousal_class'].to_numpy()
        for c in out:
            on = np.where((cls == c) & (np.roll(cls, 1) != c))[0]
            for i in on:
                a, b = i - PERI_WIN_EP, i + PERI_WIN_EP + 1
                if a < 0 or b > len(v):
                    continue
                out[c].append(v[a:b])
    return {c: np.array(v) for c, v in out.items() if v}


# ── figures ──────────────────────────────────────────────────────────────────

def fig_corr(corr, out, cap_ch='CLE-CRE'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    def heat(ax, col, title):
        M = np.full((len(CORR_CAP), len(CORR_EEG)), np.nan)
        C = np.zeros_like(M)
        for i, cb in enumerate(CORR_CAP):
            for j, eb in enumerate(CORR_EEG):
                d = corr[(corr['cap_band'] == cb) & (corr['eeg_band'] == eb)]
                if len(d):
                    M[i, j] = d[col].median()
                    # subject-level sign consistency: same sign in all 6 subjects?
                    sub = d.groupby('subject')[col].median()
                    C[i, j] = 1.0 if (len(sub) >= 5 and
                                      (np.all(sub > 0) or np.all(sub < 0))) else 0.0
        im = ax.imshow(M, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
        ax.set_xticks(range(len(CORR_EEG)))
        ax.set_xticklabels([f'EEG\n{b}' for b in CORR_EEG], fontsize=9)
        ax.set_yticks(range(len(CORR_CAP)))
        ax.set_yticklabels([f'CAP {b}' for b in CORR_CAP], fontsize=9)
        for i in range(len(CORR_CAP)):
            for j in range(len(CORR_EEG)):
                if not np.isfinite(M[i, j]):
                    continue
                mark = '*' if C[i, j] else ''
                ax.text(j, i, f'{M[i, j]:+.2f}{mark}', ha='center', va='center',
                        fontsize=8.5, fontweight='bold' if C[i, j] else 'normal',
                        color='white' if abs(M[i, j]) > 0.35 else 'black')
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        return im, C

    im, C1 = heat(axes[0], 'rho',
                  f'A  Spearman rho, median over 12 sessions\n'
                  f'CAP band variance vs EEG band power  ({cap_ch})')
    im2, C2 = heat(axes[1], 'rho_motion_partialled',
                   'B  Same, with head motion partialled out\n'
                   'what survives is not movement')
    plt.colorbar(im2, ax=axes[1], shrink=0.85, label='Spearman rho')

    # C — what is left after motion control, and is it consistent across subjects
    ax = axes[2]
    d = corr.groupby(['cap_band', 'eeg_band']).agg(
        raw=('rho', 'median'), part=('rho_motion_partialled', 'median')).reset_index()
    ax.scatter(d['raw'], d['part'], s=55, color='#2980B9', edgecolor='k', lw=0.5,
               alpha=0.85, zorder=3)
    lim = 0.7
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.0, label='motion explains nothing')
    ax.axhline(0, color='#C0392B', lw=1.2, label='motion explains everything')
    ax.axvline(0, color='k', lw=0.8)
    for _, r in d.iterrows():
        if abs(r['part']) > 0.18:
            ax.annotate(f"{r['cap_band']}/{r['eeg_band']}", (r['raw'], r['part']),
                        fontsize=7, xytext=(4, 3), textcoords='offset points')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('rho (raw)'); ax.set_ylabel('rho (motion partialled out)')
    ax.set_title('C  How much of it is just movement?', fontsize=11,
                 fontweight='bold', loc='left')
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.2)

    n_consistent = int(C2.sum())
    fig.suptitle('Claim 1 — does CAP variance track EEG band power?  '
                 f'* marks the {n_consistent} band pairs whose sign is the same in '
                 'every subject after motion control.',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=190, bbox_inches='tight'); plt.close(fig)
    return C2


def fig_arousal(ar, peri, df, out, cap_ch='CLE-CRE'):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8))

    # A — the two arousal scorings barely differ in this dataset
    ax = axes[0]
    xs = np.arange(len(ar)); w = 0.38
    ax.bar(xs - w / 2, ar['n_cortical'], w, color='#C0392B', alpha=0.85,
           edgecolor='k', lw=0.4, label='EEG-scored (cortical)')
    ax.bar(xs + w / 2, ar['n_autonomic'], w, color='#E67E22', alpha=0.85,
           edgecolor='k', lw=0.4, label='autonomic (pleth)')
    ax.plot(xs, ar['n_autonomic-only'], 'kv', ms=8, label='autonomic-only')
    ax.set_xticks(xs); ax.set_xticklabels(ar['session'], rotation=45, ha='right')
    ax.set_ylabel('epochs')
    ax.set_title(f'A  The two scorings overlap\n'
                 f'{ar["frac_autonomic_also_cortical"].median()*100:.0f}% of autonomic '
                 f'arousals are also EEG-scored,\nso there is no "EEG missed it" set '
                 f'to test on', fontsize=10.5, fontweight='bold', loc='left')
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.2)

    # B — single-feature detection of a scored arousal
    ax = axes[1]
    feats = [('auc_eegbeta', 'EEG beta', '#C0392B'),
             ('auc_capvar', 'CAP variance', '#B9380B'),
             ('auc_motion', 'head motion', '#7F8C8D')]
    data = [ar[c].dropna().to_numpy() for c, _, _ in feats]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops=dict(color='k', lw=1.8))
    for patch, (_, _, colr) in zip(bp['boxes'], feats):
        patch.set_facecolor(colr); patch.set_alpha(0.7)
    for i, v in enumerate(data):
        ax.plot(np.full(len(v), i + 1) + np.linspace(-0.15, 0.15, len(v)), v, 'o',
                ms=5, color='k', alpha=0.6)
        ax.annotate(f'{np.median(v):.3f}', (i + 1, np.median(v)), fontsize=9.5,
                    fontweight='bold', ha='center', xytext=(0, 9),
                    textcoords='offset points')
    ax.axhline(0.5, color='k', ls='--', lw=1.3, label='chance')
    ax.set_xticklabels([l for _, l, _ in feats])
    ax.set_ylabel('AUC for detecting a scored arousal')
    ax.set_title('B  Single-feature detection\none point = one session',
                 fontsize=10.5, fontweight='bold', loc='left')
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.2)

    # C — the test: arousals the EEG power readout misses
    ax = axes[2]
    d = ar.dropna(subset=['capvar_delta_eeg_missed']).sort_values('session')
    y = np.arange(len(d))
    ax.barh(y - w / 2, d['capvar_delta_eeg_missed'], w, color='#B9380B',
            alpha=0.9, edgecolor='k', lw=0.5, label='CAP variance')
    ax.barh(y + w / 2, d['motion_delta_eeg_missed'], w, color='#7F8C8D',
            alpha=0.9, edgecolor='k', lw=0.5, label='head motion (control)')
    ax.set_yticks(y); ax.set_yticklabels(d['session'])
    ax.axvline(0, color='k', lw=1.2)
    npos = int((d['capvar_delta_eeg_missed'] > 0).sum())
    ax.set_xlabel('log10 CAP variance:  arousal − no arousal,\n'
                  'both with EEG beta below the session median')
    ax.set_title(f'C  Arousals an EEG power readout would miss\n'
                 f'CAP variance still higher in {npos}/{len(d)} sessions',
                 fontsize=10.5, fontweight='bold', loc='left')
    ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(True, axis='x', alpha=0.2)

    # D — peri-event response
    ax = axes[3]
    tt = np.arange(-PERI_WIN_EP, PERI_WIN_EP + 1) * EPOCH_SEC / 60.0
    for c, colr in (('cortical', '#C0392B'), ('autonomic-only', '#E67E22')):
        if c not in peri or len(peri[c]) < 20:
            continue
        M = peri[c]
        med = np.nanmedian(M, axis=0)
        q1 = np.nanpercentile(M, 25, axis=0); q3 = np.nanpercentile(M, 75, axis=0)
        ax.fill_between(tt, q1, q3, color=colr, alpha=0.20)
        ax.plot(tt, med, lw=2.4, color=colr, label=f'{c}  (n = {len(M):,} onsets)')
    ax.axvline(0, color='k', ls='--', lw=1.3)
    ax.axhline(0, color='#555', ls=':', lw=1.0)
    ax.set_xlabel('time from arousal onset (min)')
    ax.set_ylabel('log10 CAP variance, session-centred')
    ax.set_title('D  Event-locked response\nmedian and IQR across events',
                 fontsize=10.5, fontweight='bold', loc='left')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.suptitle('Claim 2 — does CAP variance carry arousal information the EEG '
                 'readout does not?  The PSG\'s own two arousal scorings turn out to '
                 'be nearly the same set (A),\nso the comparison is made against the '
                 'EEG spectral readout instead (B, C).',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=190, bbox_inches='tight'); plt.close(fig)


def fig_trace(df, lbl, out, cap_ch='CLE-CRE'):
    g = df[df['session'] == lbl].sort_values('t_hr')
    t = g['t_hr'].to_numpy(); codes = g['stage_code'].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, .5]})

    def bands(ax):
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1], color=STAGE_COLORS.get(int(codes[j]), '#AAA'),
                       alpha=0.14, lw=0)

    ax = axes[0]; bands(ax)
    ax.plot(t, np.log10(g[f'cap_total_{cap_ch}'] + 1e-12), lw=0.9, color='#B9380B')
    ax.set_ylabel(f'log10 CAP var\n({cap_ch}, {CAP_UNIT_SQ})')
    ax.set_title(f'{lbl} — raw CAP variance against the EEG and the two arousal '
                 f'scorings', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.13)

    ax = axes[1]; bands(ax)
    ax.plot(t, np.log10(g['eeg_delta'] + 1e-12), lw=0.9, color='#C0392B',
            label='EEG delta')
    ax.plot(t, np.log10(g['eeg_beta'] + 1e-12), lw=0.9, color='#2980B9',
            label='EEG beta')
    ax.set_ylabel('log10 EEG power'); ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.13)

    ax = axes[2]; bands(ax)
    ax.plot(t, np.log10(g['acc_std'] + 1e-12), lw=0.9, color='#7F8C8D')
    ax.set_ylabel('log10 head motion\n(acc. std)')
    ax.grid(True, alpha=0.13)

    ax = axes[3]
    cort = g['cortical_arousal'].to_numpy(); auto = g['autonomic_arousal'].to_numpy()
    ax.vlines(t[cort], 0.55, 0.95, color='#C0392B', lw=1.2)
    ax.vlines(t[auto & ~cort], 0.05, 0.45, color='#E67E22', lw=1.2)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(['autonomic\nonly', 'cortical'], fontsize=8)
    ax.set_xlabel('Time (hours)')

    handles = [mpatches.Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k])
               for k in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 1.002), framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    dfs = []
    for i, meta in enumerate(SESSION_META):
        d = extract(i, meta)
        if d is None:
            print(f'  {meta["label"]}: skipped')
            continue
        n_c = int(d['cortical_arousal'].sum()); n_a = int(d['autonomic_arousal'].sum())
        print(f'  {meta["label"]}: {len(d)} epochs, {n_c} cortical-arousal, '
              f'{n_a} autonomic-arousal, '
              f'{int((d["arousal_class"] == "autonomic-only").sum())} autonomic-only')
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(REPORT_DIR / 'cap_variance_epochs.csv', index=False)

    corr = correlations(df)
    ar = arousal_table(df)
    peri = peri_event(df)
    corr.to_csv(REPORT_DIR / 'cap_variance_eeg_corr.csv', index=False)
    ar.to_csv(REPORT_DIR / 'cap_variance_arousal.csv', index=False)

    C2 = fig_corr(corr, FIG_DIR / 'cap_variance_vs_eeg.png')
    fig_arousal(ar, peri, df, FIG_DIR / 'cap_variance_arousal.png')
    demo = DEMO if (df['session'] == DEMO).any() else df['session'].iloc[0]
    fig_trace(df, demo, FIG_DIR / f'cap_variance_trace_{demo}.png')

    # ── report ──
    print('\n' + '=' * 92)
    print('CLAIM 1 - CAP variance vs EEG band power  (CLE-CRE, 30 s epochs)')
    print('=' * 92)
    piv = corr.groupby(['cap_band', 'eeg_band'])[['rho', 'rho_motion_partialled']].median()
    top = piv.reindex(piv['rho_motion_partialled'].abs().sort_values(
        ascending=False).index).head(10)
    print('Strongest band pairs after motion control (median rho across 12 sessions):')
    for (cb, eb), r in top.iterrows():
        sub = corr[(corr['cap_band'] == cb) & (corr['eeg_band'] == eb)]\
            .groupby('subject')['rho_motion_partialled'].median()
        cons = 'all 6 subjects agree' if (np.all(sub > 0) or np.all(sub < 0)) \
            else f'{int((sub>0).sum())}+/{int((sub<0).sum())}- across subjects'
        print(f'  CAP {cb:6s} vs EEG {eb:6s}  rho={r["rho"]:+.2f} -> '
              f'{r["rho_motion_partialled"]:+.2f} after motion   ({cons})')
    print(f'\n  {int(C2.sum())} of {C2.size} band pairs keep the same sign in every '
          f'subject after motion control.')
    print(f'  Median |rho| before motion control: '
          f'{piv["rho"].abs().median():.2f};  after: '
          f'{piv["rho_motion_partialled"].abs().median():.2f}.')

    print('\n' + '=' * 92)
    print('CLAIM 2 - does CAP variance carry arousal info the EEG readout does not?')
    print('=' * 92)
    tot = {c: int(ar[f'n_{c}'].sum()) for c in ('cortical', 'autonomic-only', 'neither')}
    print(f"The two PSG arousal scorings are nearly the same set:")
    print(f"  cortical (EEG-scored) {tot['cortical']:,} epochs, "
          f"autonomic {int(ar['n_autonomic'].sum()):,}, of which "
          f"{ar['frac_autonomic_also_cortical'].median()*100:.1f}% (median) are also "
          f"EEG-scored.")
    print(f"  autonomic-only: {tot['autonomic-only']:,} epochs in the whole cohort "
          f"(0-7 per night) -> the intended\n  'body aroused, EEG saw nothing' test "
          f"has no sample and is NOT reported as a result either way.")

    print('\nSingle-feature detection of a scored arousal (AUC, per session):')
    for c, nm in (('auc_eegbeta', 'EEG beta'), ('auc_capvar', 'CAP variance'),
                  ('auc_motion', 'head motion')):
        v = ar[c].dropna()
        print(f'  {nm:14s} median {v.median():.3f}  [{v.min():.3f}, {v.max():.3f}]  '
              f'({int((v > 0.55).sum())}/{len(v)} sessions above 0.55)')

    d = ar.dropna(subset=['capvar_delta_eeg_missed'])
    npos = int((d['capvar_delta_eeg_missed'] > 0).sum())
    mpos = int((d['motion_delta_eeg_missed'] > 0).sum())
    nsig = int((d['p_eeg_missed'] < 0.05).sum())
    print(f'\nArousals an EEG POWER readout would miss (scored arousal, but EEG beta '
          f'below the\nsession median), vs matched non-arousal epochs in the same '
          f'low-EEG regime:')
    print(f'  n = {int(d["n_eeg_missed_arousal"].sum()):,} such epochs across '
          f'{len(d)} sessions')
    print(f'  CAP variance higher in {npos}/{len(d)} sessions '
          f'(median {d["capvar_delta_eeg_missed"].median():+.3f} log10 = '
          f'{10**d["capvar_delta_eeg_missed"].median():.2f}x, '
          f'{nsig}/{len(d)} with p<0.05)')
    print(f'  CAP-variance AUC within this low-EEG subset: '
          f'{d["auc_capvar_eeg_missed"].median():.3f} '
          f'[{d["auc_capvar_eeg_missed"].min():.3f}, '
          f'{d["auc_capvar_eeg_missed"].max():.3f}]')
    print(f'  head motion also higher in {mpos}/{len(d)} sessions '
          f'(median {d["motion_delta_eeg_missed"].median():+.3f} log10) '
          f'-- read alongside, not after')
    for c, M in peri.items():
        pre = np.nanmedian(M[:, :PERI_WIN_EP])
        at = np.nanmedian(M[:, PERI_WIN_EP])
        print(f'  peri-event {c:15s}: {len(M):,} onsets, '
              f'baseline {pre:+.3f} -> onset {at:+.3f} log10 '
              f'({10**(at-pre):.2f}x variance)')

    print(f'\nTables  -> {REPORT_DIR}')
    print(f'Figures -> {FIG_DIR}')


if __name__ == '__main__':
    main()
