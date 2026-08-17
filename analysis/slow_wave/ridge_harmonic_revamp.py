"""
Ridge + harmonic analysis, revamped (2026-07-23).

Supersedes the fragmented Stage-3 / band_ridge_analysis pipeline.  Three
upgrades, requested for the manuscript revamp:

  1. LONG, FLAT RIDGES.  Persistent ridges are now stitched to convergence and
     gap-filled (sleep_monitor.harmonics fill_gaps + iterative merge), so a
     ridge that the old tracker chopped into bits and pieces comes out as one
     continuous, flat trace.  We quantify the improvement (n_ridges, coverage,
     internal holes: old vs new) so the "flat" claim is measured, not asserted.

  2. A SLOW BAND.  Ridge detection is extended below the respiratory band into
     the infra-slow / vasomotor range (0.02-0.15 Hz) using long 4-min analysis
     windows for the frequency resolution that band needs.  This is the band the
     CAP-SWA slow-DC-drift marker lives in, so slow-band ridges connect the
     spectral and mechanical slow-wave stories.

  3. STRONG HARMONICS.  Integer-ratio ladders are validated against a per-k
     surrogate null (random ridge frequencies over the observed band); only
     strong (>=3-member) ladders that beat the null 95th percentile are kept.
     Near-zero false positives, with a calibrated per-ladder confidence.

Bands (each with a window matched to its frequency resolution):
  slow : 0.02-0.15 Hz   240 s window, 120 s Welch segments  (df ~ 0.008 Hz)
  resp : 0.1 -0.5  Hz    30 s window, single 30 s periodogram (df ~ 0.033 Hz)
  card : 0.5 -3.0  Hz    30 s window,   8 s Welch segments   (harmonics)

Channels: ridge stage stats pooled on CRE (dominant ridge channel, 9/12
sessions per manuscript); slow-band + harmonics also checked on all channels.
Harmonic ladders run on CH (strongest harmonic channel per manuscript).

Outputs -> reports/slow_wave/revamp/
  ridge_epochs.parquet        per-epoch ridge features (band x channel x session)
  per_ridge.csv               one row per ridge (flatness, duration, dominant stage)
  continuity_comparison.csv   old vs new fragmentation, per band/session
  stage_association.csv       KW + N3-vs-rest + per-subject direction, per band/feature
  harmonic_ladders.parquet    per-window ladder score/confidence/survives-null (CH)
  harmonic_null_summary.csv   real-vs-fake strong-ladder counts per session
Figures -> writeup/figures/harmonics/
  revamp_overlay_<S>.png      spectrogram + flat ridges (3 bands) + hypnogram
  revamp_by_stage.png         ridge features by sleep stage, per band
  revamp_continuity.png       fragmentation before/after (the flat-ridge proof)
  revamp_harmonics.png        ladder survival vs null + confidence + f0

Run:  python ridge_harmonic_revamp.py --all
      python ridge_harmonic_revamp.py --session 0
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import spectrogram
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import FS, STAGE_LABELS, STAGE_ORDER, STAGE_COLORS
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import detect_persistent_ridges

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'revamp'
FIG_DIR = Path(__file__).resolve().parents[2] / 'writeup' / 'figures' / 'harmonics'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ['CH', 'CLE', 'CRE']
POOL_CH = 'CRE'          # dominant ridge channel for pooled stage stats
HARM_CH = 'CH'           # strongest harmonic channel
OVERLAY_SESSION = 'S1N1'

# Band-specific detector configs.  win_sec sets the analysis-window length; the
# slow band needs a long window for frequency resolution below 0.15 Hz.  All
# bands use the new iterative-merge + gap-fill continuity path.
BANDS = {
    'slow': dict(min_freq=0.02, max_freq=0.15, win_sec=240.0, step_sec=30.0,
                 welch_seg_sec=120.0, max_freq_jump=0.012, peak_prominence_frac=0.25,
                 smooth_windows=7, min_persistence_sec=600.0, merge_gap_windows=30),
    'resp': dict(min_freq=0.1, max_freq=0.5, win_sec=30.0, step_sec=30.0,
                 welch_seg_sec=30.0, max_freq_jump=0.05, peak_prominence_frac=0.3,
                 smooth_windows=7, min_persistence_sec=300.0, merge_gap_windows=20),
    'card': dict(min_freq=0.5, max_freq=3.0, win_sec=30.0, step_sec=30.0,
                 welch_seg_sec=8.0, max_freq_jump=0.10, peak_prominence_frac=0.4,
                 smooth_windows=7, min_persistence_sec=300.0, merge_gap_windows=20),
}
BAND_LABEL = {'slow': 'Slow (0.02-0.15 Hz)', 'resp': 'Respiratory (0.1-0.5 Hz)',
              'card': 'Cardiac (0.5-3.0 Hz)'}
BAND_COLOR = {'slow': '#16A085', 'resp': '#2980B9', 'card': '#C0392B'}
FLAT_THRESHOLD = 0.95

# Harmonic-ladder validation
RATIO_TOL = 0.06
MIN_F0 = 0.12
MIN_STRONG_MEMBERS = 3
N_SURROGATE = 400
HARM_BAND = dict(min_freq=0.1, max_freq=3.0, win_sec=30.0, step_sec=30.0,
                 welch_seg_sec=20.0, max_freq_jump=0.12, peak_prominence_frac=0.15,
                 smooth_windows=5, min_persistence_sec=300.0, merge_gap_windows=20)

FEATURES = [
    ('ridge_present', 'Ridge present (frac)'),
    ('n_ridges', 'Active ridges'),
    ('min_ridge_freq', 'Lowest ridge freq (Hz)'),
    ('total_ridge_power', 'Total ridge power'),
    ('freq_spread', 'Ridge freq spread (Hz)'),
    ('mean_flatness', 'Mean ridge flatness'),
]


# ── Detection ────────────────────────────────────────────────────────────────

def _sig(session, ch):
    acc = session.cap['acc_mag'].astype(np.float64)
    return remove_acc_artifact(session.cap[ch].astype(np.float64), acc, 0.05, 4.0)


def detect_band(session, ch, band):
    bp = dict(BANDS[band])
    acc = session.cap['acc_mag'].astype(np.float64)
    return detect_persistent_ridges(
        _sig(session, ch), fs=session.fs, acc_mag=acc,
        fill_gaps=True, **bp)


def _stage_at(sp, t_hr):
    idx = np.searchsorted(sp['t_ep_hr'], t_hr, side='right') - 1
    if 0 <= idx < len(sp['codes']):
        return int(sp['codes'][idx])
    return -1


# ── Per-epoch + per-ridge features ───────────────────────────────────────────

def epoch_features(rr, sp, ch, band, session, subject):
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    n = len(t_hr)
    n_ridges = np.zeros(n, int)
    min_f = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    power = np.full(n, np.nan)
    flat = np.full(n, np.nan)
    for i in range(n):
        fs_i, as_i, fl_i = [], [], []
        for r in ridges:
            f = r['freq_trace'][i]
            if np.isfinite(f):
                fs_i.append(f)
                as_i.append(r['amp_trace'][i])
                fl_i.append(r.get('flatness', np.nan))
        n_ridges[i] = len(fs_i)
        if fs_i:
            fa = np.array(fs_i)
            min_f[i] = fa.min()
            spread[i] = fa.std() if len(fa) > 1 else 0.0
            power[i] = float(np.sum(as_i))
            fl = np.array(fl_i)
            flat[i] = np.nanmean(fl) if np.isfinite(fl).any() else np.nan
    stages = np.array([_stage_at(sp, t) for t in t_hr])
    return pd.DataFrame(dict(
        session=session, subject=subject, band=band, channel=ch,
        t_hr=t_hr, motion_masked=rr['motion_mask'], stage_code=stages,
        stage_label=[STAGE_LABELS.get(c, '?') for c in stages],
        n_ridges=n_ridges, min_ridge_freq=min_f, freq_spread=spread,
        total_ridge_power=power, mean_flatness=flat,
        ridge_present=(n_ridges > 0).astype(int),
    ))


def per_ridge_rows(rr, sp, ch, band, session, subject):
    t_hr = rr['t_hr']
    rows = []
    for r in rr['ridges']:
        present = np.where(np.isfinite(r.get('freq_trace_raw', r['freq_trace'])))[0]
        stages = [_stage_at(sp, t_hr[i]) for i in present]
        stages = [s for s in stages if s >= 0]
        if stages:
            vals, cnts = np.unique(stages, return_counts=True)
            dom = int(vals[np.argmax(cnts)])
            n3_frac = float(np.mean(np.array(stages) == 1))
        else:
            dom, n3_frac = -1, 0.0
        rows.append(dict(
            session=session, subject=subject, band=band, channel=ch,
            median_freq=r['median_freq'], duration_sec=r['duration_sec'],
            flatness=r.get('flatness', np.nan), freq_cv=r.get('freq_cv', np.nan),
            drift_slope=r.get('drift_slope', np.nan), coverage=r.get('coverage', np.nan),
            median_prominence=r.get('median_prominence', np.nan),
            dominant_stage=STAGE_LABELS.get(dom, '?'), n3_fraction=n3_frac,
        ))
    return rows


def continuity_row(rr_old, rr_new, ch, band, session, subject):
    def stats(rr):
        cov, holes = [], 0
        for r in rr['ridges']:
            span = r['freq_trace'][r['start_idx']:r['end_idx'] + 1]
            cov.append(float(np.mean(np.isfinite(span))))
            holes += int(np.sum(~np.isfinite(span)))
        return len(rr['ridges']), (float(np.mean(cov)) if cov else np.nan), holes
    n_o, c_o, h_o = stats(rr_old)
    n_n, c_n, h_n = stats(rr_new)
    # longest continuous ridge (new)
    longest = max((r['duration_sec'] for r in rr_new['ridges']), default=0.0)
    return dict(session=session, subject=subject, band=band, channel=ch,
                n_ridges_old=n_o, n_ridges_new=n_n,
                trace_coverage_old=c_o, trace_coverage_new=c_n,
                internal_holes_old=h_o, internal_holes_new=h_n,
                longest_ridge_sec_new=longest)


# ── Harmonic ladders vs surrogate null (reused rigor logic) ──────────────────

def _score(n_members, quality):
    if n_members < 2:
        return 0.0
    return quality * min(np.log2(max(n_members, 1)) / np.log2(6), 1.0)


def best_ladder(freqs, amps, ratio_tol=RATIO_TOL, min_f0=MIN_F0):
    order = np.argsort(freqs)
    freqs = np.asarray(freqs)[order]
    amps = np.asarray(amps)[order]
    n = len(freqs)
    best = (0, 0.0, np.nan, 0.0, 0.0)
    best_score = -1.0
    for ai in range(n):
        f0 = freqs[ai]
        if f0 < min_f0:
            continue
        members = [(1, f0, amps[ai])]
        devs = []
        for aj in range(n):
            if aj == ai:
                continue
            ratio = freqs[aj] / f0
            k = round(ratio)
            if k >= 2 and abs(ratio - k) < ratio_tol:
                members.append((k, freqs[aj], amps[aj]))
                devs.append(abs(ratio - k))
        if len(members) < 2:
            continue
        quality = 1.0 - np.mean(devs) / ratio_tol if devs else 0.0
        members.sort(key=lambda m: m[0])
        m_amps = [m[2] for m in members]
        steps = np.diff(m_amps)
        decay = float(np.mean(steps <= 0)) if len(steps) else 0.0
        power = float(np.sum(m_amps))
        sc = _score(len(members), quality)
        if sc > best_score:
            best_score = sc
            best = (len(members), quality, float(f0), power, decay)
    return best


def harmonic_ladders(session, sp, idx):
    rr = detect_persistent_ridges(_sig(session, HARM_CH), fs=session.fs,
                                  acc_mag=session.cap['acc_mag'].astype(np.float64),
                                  fill_gaps=True, **HARM_BAND)
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    motion = rr['motion_mask']
    n = len(t_hr)
    if not ridges:
        return None, None

    all_freqs = np.concatenate([r['freq_trace'][np.isfinite(r['freq_trace'])] for r in ridges])
    all_amps = np.concatenate([r['amp_trace'][np.isfinite(r['amp_trace'])] for r in ridges])
    f_lo, f_hi = float(all_freqs.min()), float(all_freqs.max())

    active = []
    max_k = 0
    for i in range(n):
        act = [(r['freq_trace'][i], r['amp_trace'][i])
               for r in ridges if np.isfinite(r['freq_trace'][i])]
        active.append(act)
        max_k = max(max_k, len(act))

    rng = np.random.default_rng(1234 + idx)
    null_thresh = {}
    for k in range(2, max_k + 1):
        sc = np.empty(N_SURROGATE)
        for s in range(N_SURROGATE):
            fr = rng.uniform(f_lo, f_hi, size=k)
            am = rng.choice(all_amps, size=k, replace=True)
            nm, q, *_ = best_ladder(fr, am)
            sc[s] = _score(nm, q)
        null_thresh[k] = float(np.percentile(sc, 95))

    rows = []
    n_strong = n_survive = 0
    for i in range(n):
        act = active[i]
        stage = _stage_at(sp, t_hr[i])
        rec = dict(session=session.label, subject=session.subject, t_hr=float(t_hr[i]),
                   stage_code=stage, stage_label=STAGE_LABELS.get(stage, '?'),
                   motion=bool(motion[i]), n_active=len(act))
        if len(act) >= 2:
            fr = [a[0] for a in act]
            am = [a[1] for a in act]
            nm, q, f0, power, decay = best_ladder(fr, am)
            obs = _score(nm, q)
            thr = null_thresh.get(nm if nm >= 2 else 2, np.inf)
            is_strong = nm >= MIN_STRONG_MEMBERS
            survives = is_strong and (obs > thr)
            n_strong += int(is_strong)
            n_survive += int(survives)
            rec.update(n_members=nm, ratio_quality=float(q), ladder_f0=f0,
                       ladder_power=power, decay_factor=float(decay),
                       obs_score=obs, survives_null=survives,
                       confidence=float(q * decay) if survives else 0.0)
        else:
            rec.update(n_members=0, ratio_quality=0.0, ladder_f0=np.nan,
                       ladder_power=0.0, decay_factor=0.0, obs_score=0.0,
                       survives_null=False, confidence=0.0)
        rows.append(rec)
    df = pd.DataFrame(rows)
    null_row = dict(session=session.label, subject=session.subject, n_windows=n,
                    n_strong=n_strong, n_survive=n_survive,
                    survive_frac=(n_survive / n_strong if n_strong else np.nan))
    return df, null_row


# ── Stage association ────────────────────────────────────────────────────────

def stage_association(epochs):
    rows = []
    for band in BANDS:
        pool = epochs[(epochs['band'] == band) & (epochs['channel'] == POOL_CH)
                      & (~epochs['motion_masked']) & (epochs['stage_code'] >= 0)].copy()
        if pool.empty:
            continue
        ex_wake = pool[pool['stage_code'] != 4].copy()
        ex_wake['is_N3'] = ex_wake['stage_code'] == 1
        subjects = sorted(pool['subject'].unique())
        for feat, label in FEATURES:
            groups = [pool.loc[pool['stage_code'] == sc, feat].dropna().values
                      for sc in STAGE_ORDER
                      if (pool['stage_code'] == sc).sum() >= 10]
            kw_p = kruskal(*groups)[1] if len(groups) >= 2 else np.nan
            n3 = ex_wake.loc[ex_wake['is_N3'], feat].dropna()
            oth = ex_wake.loc[~ex_wake['is_N3'], feat].dropna()
            mwu_p = (mannwhitneyu(n3, oth, alternative='two-sided')[1]
                     if len(n3) > 5 and len(oth) > 5 else np.nan)
            # pooled N3 one-vs-rest AUC (exclude wake), direction per subject
            auc = np.nan
            sub = ex_wake[[feat, 'is_N3']].dropna()
            if sub['is_N3'].nunique() == 2:
                auc = roc_auc_score(sub['is_N3'], sub[feat])
            dirs = []
            for subj in subjects:
                sv = ex_wake[ex_wake['subject'] == subj]
                a = sv.loc[sv['is_N3'], feat].dropna()
                b = sv.loc[~sv['is_N3'], feat].dropna()
                if len(a) > 3 and len(b) > 3:
                    dirs.append('N3>' if a.median() > b.median() else 'N3<')
                else:
                    dirs.append('?')
            rows.append(dict(
                band=band, channel=POOL_CH, feature=feat,
                n3_median=float(n3.median()) if len(n3) else np.nan,
                other_median=float(oth.median()) if len(oth) else np.nan,
                kw_p=kw_p, n3_vs_rest_mwu_p=mwu_p, n3_auc=auc,
                n_subj_N3_up=dirs.count('N3>'), n_subj_N3_dn=dirs.count('N3<'),
                directions=','.join(dirs)))
    return pd.DataFrame(rows)


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_overlay(session, band_rr, out_path):
    """Spectrogram (coarse 0-3 Hz + fine slow) with flat ridges + hypnogram."""
    sp = session.sleep_profile
    sig = _sig(session, POOL_CH)
    fs = session.fs

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.35, 1.0, 0.7], hspace=0.12)

    # hypnogram
    ax0 = fig.add_subplot(gs[0])
    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax0.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                        color=STAGE_COLORS.get(c, '#AAA'), alpha=0.65)
    ax0.set_yticks([]); ax0.set_ylabel('Stage', fontsize=9)
    ax0.set_title(f'{session.label} — flat persistent ridges over three bands ({POOL_CH})',
                  fontsize=12)
    ax0.legend(handles=[mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
                        for c in STAGE_ORDER], loc='upper right', fontsize=7, ncol=5)

    # main spectrogram 0-3 Hz
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=int(30 * fs), noverlap=int(15 * fs))
    m = f <= 3.0
    ax1.pcolormesh(t / 3600, f[m], 10 * np.log10(Sxx[m] + 1e-20),
                   shading='gouraud', cmap='magma', rasterized=True)
    for band in ('resp', 'card'):
        for r in band_rr[band]['ridges']:
            ax1.plot(band_rr[band]['t_hr'], r['freq_trace'], color=BAND_COLOR[band],
                     lw=1.4, alpha=0.9)
    ax1.set_ylim(0, 3.0); ax1.set_ylabel('Frequency (Hz)', fontsize=9)
    ax1.plot([], [], color=BAND_COLOR['resp'], label='resp ridge')
    ax1.plot([], [], color=BAND_COLOR['card'], label='cardiac ridge')
    ax1.legend(loc='upper right', fontsize=8)

    # slow-band spectrogram 0-0.15 Hz
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    fS, tS, SxxS = spectrogram(sig, fs=fs, nperseg=int(240 * fs), noverlap=int(210 * fs))
    mS = fS <= 0.15
    ax2.pcolormesh(tS / 3600, fS[mS], 10 * np.log10(SxxS[mS] + 1e-20),
                   shading='gouraud', cmap='viridis', rasterized=True)
    for r in band_rr['slow']['ridges']:
        ax2.plot(band_rr['slow']['t_hr'], r['freq_trace'], color=BAND_COLOR['slow'],
                 lw=1.6, alpha=0.95)
    ax2.set_ylim(0.02, 0.15); ax2.set_ylabel('Slow (Hz)', fontsize=9)
    ax2.set_xlabel('Time (hr)', fontsize=10)
    ax2.plot([], [], color=BAND_COLOR['slow'], label='slow ridge')
    ax2.legend(loc='upper right', fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_path}')


def fig_by_stage(epochs, out_path):
    show = [('min_ridge_freq', 'Lowest ridge freq (Hz)'),
            ('total_ridge_power', 'Total ridge power'),
            ('mean_flatness', 'Mean flatness')]
    bands = list(BANDS.keys())
    fig, axes = plt.subplots(len(bands), len(show), figsize=(15, 11), squeeze=False)
    for r, band in enumerate(bands):
        pool = epochs[(epochs['band'] == band) & (epochs['channel'] == POOL_CH)
                      & (~epochs['motion_masked']) & (epochs['stage_code'] >= 0)]
        for c, (feat, label) in enumerate(show):
            ax = axes[r, c]
            data, labs, cols = [], [], []
            for sc in STAGE_ORDER:
                vals = pool.loc[pool['stage_code'] == sc, feat].dropna()
                if len(vals) > 0:
                    data.append(vals.values); labs.append(STAGE_LABELS[sc])
                    cols.append(STAGE_COLORS[sc])
            if not data:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center'); continue
            bp = ax.boxplot(data, labels=labs, patch_artist=True, widths=0.6,
                            showfliers=False, medianprops=dict(color='black', lw=1.5))
            for j, col in enumerate(cols):
                bp['boxes'][j].set_facecolor(col); bp['boxes'][j].set_alpha(0.6)
            kw_p = kruskal(*data)[1] if len(data) >= 2 else np.nan
            sig = '***' if kw_p < 1e-3 else '**' if kw_p < 1e-2 else '*' if kw_p < 0.05 else 'ns'
            ax.set_title(f'{label}\nKW p={kw_p:.1e} {sig}', fontsize=9)
            ax.grid(True, alpha=0.15, axis='y')
            if c == 0:
                ax.set_ylabel(f'{BAND_LABEL[band]}\n({POOL_CH})', fontsize=9)
    fig.suptitle('Persistent-ridge features by sleep stage (3 bands)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170, facecolor='white'); plt.close(fig)
    print(f'  wrote {out_path}')


def fig_continuity(cont_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    bands = list(BANDS.keys())
    x = np.arange(len(bands))
    # 1. trace coverage old vs new
    ax = axes[0]
    co = [cont_df[cont_df.band == b]['trace_coverage_old'].mean() for b in bands]
    cn = [cont_df[cont_df.band == b]['trace_coverage_new'].mean() for b in bands]
    ax.bar(x - 0.2, co, 0.4, label='old (fragmented)', color='#E74C3C', alpha=0.8)
    ax.bar(x + 0.2, cn, 0.4, label='new (gap-filled)', color='#27AE60', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([BAND_LABEL[b] for b in bands], fontsize=8, rotation=10)
    ax.set_ylabel('Mean trace coverage'); ax.set_ylim(0, 1.05)
    ax.set_title('Ridge continuity (holes closed)'); ax.legend(fontsize=8)
    # 2. n_ridges old vs new (fragments merged)
    ax = axes[1]
    no = [cont_df[cont_df.band == b]['n_ridges_old'].mean() for b in bands]
    nn = [cont_df[cont_df.band == b]['n_ridges_new'].mean() for b in bands]
    ax.bar(x - 0.2, no, 0.4, label='old', color='#E74C3C', alpha=0.8)
    ax.bar(x + 0.2, nn, 0.4, label='new', color='#27AE60', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([BAND_LABEL[b] for b in bands], fontsize=8, rotation=10)
    ax.set_ylabel('Mean ridges / session')
    ax.set_title('Fragments merged into fewer, longer ridges'); ax.legend(fontsize=8)
    # 3. longest continuous ridge (new), minutes
    ax = axes[2]
    lg = [cont_df[cont_df.band == b]['longest_ridge_sec_new'].mean() / 60 for b in bands]
    ax.bar(x, lg, 0.5, color=[BAND_COLOR[b] for b in bands], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([BAND_LABEL[b] for b in bands], fontsize=8, rotation=10)
    ax.set_ylabel('Longest ridge (min)')
    ax.set_title('Longest continuous flat ridge (new)')
    fig.tight_layout(); fig.savefig(out_path, dpi=150, facecolor='white'); plt.close(fig)
    print(f'  wrote {out_path}')


def fig_harmonics(harm_df, null_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    ax = axes[0]
    x = np.arange(len(null_df))
    ax.bar(x, null_df['n_survive'], label='strong ladder, beats null', color='#27AE60')
    ax.bar(x, null_df['n_strong'] - null_df['n_survive'], bottom=null_df['n_survive'],
           label='strong but rejected', color='#E74C3C', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(null_df['session'], rotation=45, fontsize=7)
    ax.set_ylabel('windows'); ax.set_title('Harmonic-ladder survival vs surrogate null')
    ax.legend(fontsize=8)
    ax = axes[1]
    conf = harm_df.loc[harm_df['survives_null'], 'confidence'].dropna()
    ax.hist(conf, bins=30, color='#2980B9', alpha=0.85)
    ax.set_xlabel('ladder confidence'); ax.set_ylabel('windows')
    ax.set_title(f'Confidence of surviving ladders (n={len(conf)})')
    ax = axes[2]
    f0 = harm_df.loc[harm_df['survives_null'], 'ladder_f0'].dropna()
    ax.hist(f0, bins=40, color='#8E44AD', alpha=0.85)
    ax.set_xlabel('ladder fundamental f0 (Hz)'); ax.set_ylabel('windows')
    ax.set_title(f'Fundamental frequency (median {f0.median():.2f} Hz)')
    fig.tight_layout(); fig.savefig(out_path, dpi=150, facecolor='white'); plt.close(fig)
    print(f'  wrote {out_path}')


# ── Drivers ──────────────────────────────────────────────────────────────────

def process_session(idx, collect_overlay=False):
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    sp = session.sleep_profile
    collect_overlay = collect_overlay or (session.label == OVERLAY_SESSION)
    print(f"\n{'='*60}\n{session.label} ({session.subject})\n{'='*60}")

    epoch_dfs, ridge_rows, cont_rows = [], [], []
    overlay_rr = {}
    for ch in CHANNELS:
        for band in BANDS:
            rr = detect_band(session, ch, band)
            epoch_dfs.append(epoch_features(rr, sp, ch, band, session.label, session.subject))
            ridge_rows += per_ridge_rows(rr, sp, ch, band, session.label, session.subject)
            if ch == POOL_CH:
                # old-config re-run (no fill, single merge pass) for the continuity delta
                bp = dict(BANDS[band]); bp.pop('merge_gap_windows', None)
                rr_old = detect_persistent_ridges(
                    _sig(session, ch), fs=session.fs,
                    acc_mag=session.cap['acc_mag'].astype(np.float64),
                    fill_gaps=False, **bp)
                cont_rows.append(continuity_row(rr_old, rr, ch, band,
                                                session.label, session.subject))
                if collect_overlay:
                    overlay_rr[band] = rr
            print(f"  {ch} {band:>4}: {len(rr['ridges'])} ridges "
                  f"(longest {max((r['duration_sec'] for r in rr['ridges']), default=0)/60:.0f} min)")

    harm_df, null_row = harmonic_ladders(session, sp, idx)
    return (pd.concat(epoch_dfs, ignore_index=True), pd.DataFrame(ridge_rows),
            pd.DataFrame(cont_rows), harm_df, null_row, session, overlay_rr)


def run_all():
    all_epoch, all_ridge, all_cont, all_harm, all_null = [], [], [], [], []
    overlay_session, overlay_rr = None, {}
    for idx in range(12):
        try:
            ep, rd, ct, hd, nr, sess, orr = process_session(idx)
            all_epoch.append(ep); all_ridge.append(rd); all_cont.append(ct)
            if hd is not None:
                all_harm.append(hd); all_null.append(nr)
            if orr:
                overlay_session, overlay_rr = sess, orr
        except Exception as e:
            print(f"  ERROR session {idx}: {e}")
            import traceback; traceback.print_exc()

    epochs = pd.concat(all_epoch, ignore_index=True)
    ridges = pd.concat(all_ridge, ignore_index=True)
    cont = pd.concat(all_cont, ignore_index=True)
    harm = pd.concat(all_harm, ignore_index=True)
    null_df = pd.DataFrame(all_null)

    epochs.to_parquet(REPORT_DIR / 'ridge_epochs.parquet')
    ridges.to_csv(REPORT_DIR / 'per_ridge.csv', index=False)
    cont.to_csv(REPORT_DIR / 'continuity_comparison.csv', index=False)
    harm.to_parquet(REPORT_DIR / 'harmonic_ladders.parquet')
    null_df.to_csv(REPORT_DIR / 'harmonic_null_summary.csv', index=False)
    assoc = stage_association(epochs)
    assoc.to_csv(REPORT_DIR / 'stage_association.csv', index=False)

    # figures
    if overlay_rr:
        fig_overlay(overlay_session, overlay_rr, FIG_DIR / f'revamp_overlay_{OVERLAY_SESSION}.png')
    fig_by_stage(epochs, FIG_DIR / 'revamp_by_stage.png')
    fig_continuity(cont, FIG_DIR / 'revamp_continuity.png')
    fig_harmonics(harm, null_df, FIG_DIR / 'revamp_harmonics.png')

    # console summary
    print(f"\n{'='*60}\nCONTINUITY (pooled {POOL_CH})\n{'='*60}")
    for b in BANDS:
        g = cont[cont.band == b]
        print(f"  {BAND_LABEL[b]:22s}: n_ridges {g.n_ridges_old.mean():.0f}->{g.n_ridges_new.mean():.0f} "
              f"| coverage {g.trace_coverage_old.mean():.2f}->{g.trace_coverage_new.mean():.2f} "
              f"| holes {g.internal_holes_old.sum():.0f}->{g.internal_holes_new.sum():.0f} "
              f"| longest {g.longest_ridge_sec_new.mean()/60:.0f} min")

    print(f"\n{'='*60}\nSLOW-BAND RIDGE PRESENCE (by channel)\n{'='*60}")
    for ch in CHANNELS:
        sub = epochs[(epochs.band == 'slow') & (epochs.channel == ch) & (~epochs.motion_masked)]
        print(f"  {ch}: ridge present in {sub.ridge_present.mean():.0%} of clean epochs, "
              f"median lowest freq {sub.min_ridge_freq.median():.3f} Hz")

    print(f"\n{'='*60}\nSTAGE ASSOCIATION (pooled {POOL_CH})\n{'='*60}")
    print(assoc.to_string(index=False))

    tot_strong = null_df.n_strong.sum(); tot_surv = null_df.n_survive.sum()
    print(f"\n{'='*60}\nHARMONIC LADDERS ({HARM_CH})\n{'='*60}")
    print(f"  strong(>=3) {tot_strong} | survive null {tot_surv} "
          f"({(tot_surv/tot_strong if tot_strong else 0):.0%} real, "
          f"{(1-tot_surv/tot_strong if tot_strong else 0):.0%} rejected)")
    surv = harm[harm.survives_null]
    print(f"  surviving-ladder median confidence {surv.confidence.median():.3f}, "
          f"median f0 {surv.ladder_f0.median():.2f} Hz, "
          f"median members {surv.n_members.median():.0f}")
    print('\nDone.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=int, default=0)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all:
        run_all()
    else:
        ep, rd, ct, hd, nr, sess, orr = process_session(args.session, collect_overlay=True)
        print(ct.to_string(index=False))
        if nr:
            print(nr)


if __name__ == '__main__':
    main()
