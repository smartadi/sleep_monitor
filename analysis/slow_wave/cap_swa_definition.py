"""
CAP-SWA operational definition and hypothesis testing (Workstream C).

Defines a CAP-derived slow-wave-activity (SWA) *candidate* state from the three
mechanical criteria the user is confident about, then treats the professor's
autonomic hypotheses as falsifiable predictions tested against that definition.

IMPORTANT — clean separation of "define" vs "test":
  DEFINITION (mechanical, high confidence):
    D1  Mean capacitance (CLE-CRE) changes slowly  (low |DC slope|)
    D3  Thorax amplitude changes slowly            (low thorax-envelope drift/variance)
    Dq  Currently quiescent                        (low accelerometer RMS)
  We deliberately do NOT put HR / RR / movement-initiation into the definition,
  so those remain independent things to test.

  TESTED HYPOTHESES (professor's — may be false, reported honestly):
    H2  SWA is more likely to *initiate* shortly after a distinct head movement
    H4  Heart rate increases during SWA
    H5  Respiratory rate increases during SWA
    H6  CAP-vs-thorax respiratory rate deviates strongly during SWA (k jumps)
    H7  PPG cardiac peak frequency rises while CAP cardiac peak frequency falls
    H8  EEG delta power is elevated during SWA (conventional SWS validation)

Outputs -> reports/slow_wave/cap_swa/
  <session>/epoch_features.csv         per-epoch feature + SWA score table
  <session>/night_overview.png         SWA score + criteria + validation traces
  all_epoch_features.parquet           pooled per-epoch table (all sessions)
  hypothesis_summary.csv               per-hypothesis effect + per-subject consistency
  movement_initiation.csv              H2 precursor test per session/subject

Run:
  python cap_swa_definition.py --session 0     # single session
  python cap_swa_definition.py --all           # all 12 + pooled stats
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, STAGE_LABELS, STAGE_COLORS,
)
from sleep_monitor.filters import bandpass, lowpass
from sleep_monitor.ground_truth import gt_heart_rate, gt_resp_rate
from sleep_monitor.sessions import SESSION_META

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'cap_swa'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = PSG_EPOCH_SEC          # 30 s
ROLL_W = 5                         # epochs in rolling window (2.5 min)
N3_CODE = 1                        # STAGE_LABELS: 1 -> 'N3'

# SWA scoring: each mechanical sub-score is a per-session percentile in [0,1]
# where 1 = most SWA-like. Definition score = geometric mean of the three.
SWA_THRESHOLD = 0.60               # graded score cut for the binary candidate label
MIN_SWA_EPOCHS = 4                 # sustained >= 2 min to count as an SWA bout


# ── Spectral helpers ─────────────────────────────────────────────────────────

def _spectral_peak_hz(seg, f_lo, f_hi, fs, seg_sec=8.0):
    """
    Dominant frequency (Hz) of seg within [f_lo, f_hi] via Welch PSD peak.

    seg_sec sets the Welch sub-segment length: use a long segment (~full epoch)
    for the respiratory band so the ~0.4 Hz-wide band is not quantized into a
    couple of coarse 0.125 Hz bins, and a shorter segment for the cardiac band
    where variance control matters more.
    """
    if len(seg) < int(4 * fs):
        return np.nan
    nperseg = min(len(seg), int(seg_sec * fs))
    freqs, psd = welch(seg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return np.nan
    return float(freqs[band][np.argmax(psd[band])])


def _delta_ratio(seg_eeg, fs):
    """EEG delta (0.5-4 Hz) power fraction of 0.5-30 Hz total."""
    nperseg = min(len(seg_eeg), int(4 * fs))
    if len(seg_eeg) < nperseg or nperseg < int(2 * fs):
        return np.nan
    freqs, psd = welch(seg_eeg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    df = freqs[1] - freqs[0]
    d = (freqs >= 0.5) & (freqs <= 4.0)
    t = (freqs >= 0.5) & (freqs <= 30.0)
    tot = float(np.trapezoid(psd[t], dx=df))
    return float(np.trapezoid(psd[d], dx=df)) / tot if tot > 0 else np.nan


def _peaks_rate_in_epoch(peak_times_s, t0_s, t1_s):
    """Rate (Hz) from pre-detected GT peaks falling inside [t0, t1]."""
    inw = peak_times_s[(peak_times_s >= t0_s) & (peak_times_s <= t1_s)]
    if len(inw) >= 2:
        return (len(inw) - 1) / (inw[-1] - inw[0])
    return np.nan


# ── Per-epoch feature extraction ─────────────────────────────────────────────

def compute_epoch_features(session, card_gt, resp_gt):
    """One row per 30 s epoch: mechanical criteria + physiology + validation."""
    sp = session.sleep_profile
    fs = session.fs
    t_hr = session.time_hr
    epoch_n = int(EPOCH_SEC * fs)
    n_epochs = len(sp['codes'])

    cle = session.cap['CLE'].astype(np.float64)
    cre = session.cap['CRE'].astype(np.float64)
    diff = cle - cre
    acc = session.cap['acc_mag'].astype(np.float64)
    thorax = session.psg['Thorax'].astype(np.float64)
    pleth = session.psg['Pleth'].astype(np.float64)
    eeg = session.psg['EEG'].astype(np.float64)

    card_peaks_s = card_gt.peak_times_s if card_gt is not None else np.array([])
    resp_peaks_s = resp_gt.peak_times_s if resp_gt is not None else np.array([])

    rows = []
    for ei in range(n_epochs):
        t_start = sp['t_ep_hr'][ei]
        t_end = t_start + EPOCH_SEC / 3600.0
        mask = (t_hr >= t_start) & (t_hr < t_end)
        if mask.sum() < epoch_n * 0.5:
            continue
        idx = np.where(mask)[0]
        seg_diff = diff[idx]
        seg_acc = acc[idx]
        seg_thorax = thorax[idx]
        seg_pleth = pleth[idx]
        seg_eeg = eeg[idx]
        t0_s, t1_s = t_start * 3600.0, t_end * 3600.0

        # ── Mechanical criteria ──
        dc_mean = float(np.mean(seg_diff))
        acc_ac = seg_acc - np.mean(seg_acc)
        acc_rms = float(np.sqrt(np.mean(acc_ac ** 2)))
        thorax_bp = bandpass(seg_thorax, RESP_LO, RESP_HI, fs)
        thorax_rms = float(np.sqrt(np.mean(thorax_bp ** 2)))

        # ── CAP rates / peak freqs (resp uses full-epoch Welch for finer bins) ──
        cap_resp_hz = _spectral_peak_hz(
            bandpass(seg_diff, RESP_LO, RESP_HI, fs), RESP_LO, RESP_HI, fs, seg_sec=EPOCH_SEC)
        cap_card_hz = _spectral_peak_hz(
            bandpass(seg_diff, CARD_LO, CARD_HI, fs), CARD_LO, CARD_HI, fs)
        thorax_resp_hz = _spectral_peak_hz(
            thorax_bp, RESP_LO, RESP_HI, fs, seg_sec=EPOCH_SEC)
        ppg_card_hz = _spectral_peak_hz(
            bandpass(seg_pleth, CARD_LO, CARD_HI, fs), CARD_LO, CARD_HI, fs)

        # ── PSG ground-truth rates (gold standard) ──
        ecg_hr_hz = _peaks_rate_in_epoch(card_peaks_s, t0_s, t1_s)
        flow_rr_hz = _peaks_rate_in_epoch(resp_peaks_s, t0_s, t1_s)

        # ── Validation ──
        eeg_delta = _delta_ratio(seg_eeg, fs)

        rows.append(dict(
            epoch_idx=ei, t_hr=float(t_start),
            stage_code=int(sp['codes'][ei]), stage_label=sp['labels'][ei],
            dc_mean=dc_mean, acc_rms=acc_rms, thorax_rms=thorax_rms,
            cap_resp_hz=cap_resp_hz, cap_card_hz=cap_card_hz,
            thorax_resp_hz=thorax_resp_hz, ppg_card_hz=ppg_card_hz,
            ecg_hr_hz=ecg_hr_hz, flow_rr_hz=flow_rr_hz,
            eeg_delta_ratio=eeg_delta,
        ))

    df = pd.DataFrame(rows).reset_index(drop=True)

    # ── Rolling-window mechanical derived features ──
    # D1: |slope| of dc_mean over rolling window (low = slow change)
    dc_slope, thx_slope, thx_var = [], [], []
    for i in range(len(df)):
        lo = max(0, i - ROLL_W // 2)
        hi = min(len(df), i + ROLL_W // 2 + 1)
        dvals = df['dc_mean'].iloc[lo:hi].values
        tvals = df['thorax_rms'].iloc[lo:hi].values
        if len(dvals) >= 3:
            x = np.arange(len(dvals), dtype=float)
            dc_slope.append(abs(float(np.polyfit(x, dvals, 1)[0])))
            thx_slope.append(abs(float(np.polyfit(x, tvals, 1)[0])))
            thx_var.append(float(np.var(tvals)))
        else:
            dc_slope.append(np.nan); thx_slope.append(np.nan); thx_var.append(np.nan)
    df['dc_abs_slope'] = dc_slope
    df['thorax_abs_slope'] = thx_slope
    df['thorax_var'] = thx_var

    # k_resp: CAP vs thorax respiratory-rate ratio (H6 deviation)
    df['k_resp'] = df['cap_resp_hz'] / df['thorax_resp_hz']
    # deviation magnitude — the actual H6 quantity ("k needs a huge change")
    df['k_resp_dev'] = (df['cap_resp_hz'] - df['thorax_resp_hz']).abs()
    # H7 divergence: PPG minus CAP cardiac peak freq (hypothesis: PPG up, CAP down)
    df['card_freq_divergence'] = df['ppg_card_hz'] - df['cap_card_hz']

    return df


# ── SWA scoring (definition) ─────────────────────────────────────────────────

def _inv_pct_score(x):
    """Map values to [0,1] where LOW raw value -> HIGH score (percentile rank inverted)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    valid = np.isfinite(x)
    if valid.sum() < 3:
        return out
    ranks = pd.Series(x[valid]).rank(pct=True).values  # 0..1, high raw -> high rank
    out[valid] = 1.0 - ranks                            # low raw -> high score
    return out


def score_swa(df):
    """Graded CAP-SWA score from D1 (slow DC) + D3 (slow thorax) + Dq (quiescent)."""
    s_dc = _inv_pct_score(df['dc_abs_slope'].values)       # slow DC -> high
    s_thx = _inv_pct_score(df['thorax_abs_slope'].values)  # slow thorax -> high
    s_still = _inv_pct_score(df['acc_rms'].values)         # still -> high
    df['swa_s_dc'] = s_dc
    df['swa_s_thorax'] = s_thx
    df['swa_s_still'] = s_still
    # geometric mean penalizes any single failing criterion
    stack = np.vstack([s_dc, s_thx, s_still])
    with np.errstate(invalid='ignore'):
        gm = np.exp(np.nanmean(np.log(np.clip(stack, 1e-6, 1)), axis=0))
    gm[np.any(~np.isfinite(stack), axis=0)] = np.nan
    df['swa_score'] = gm

    # Binary candidate label: score above threshold, sustained >= MIN_SWA_EPOCHS
    above = (df['swa_score'] >= SWA_THRESHOLD).values
    label = np.zeros(len(df), dtype=bool)
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if j - i >= MIN_SWA_EPOCHS:
                label[i:j] = True
            i = j
        else:
            i += 1
    df['swa_candidate'] = label
    return df


# ── Movement detection (for H2 precursor test) ───────────────────────────────

def detect_movements(acc_mag, fs, min_gap_sec=120, thresh_mad=4.0):
    """Distinct head-movement event sample indices (peaks in 5 s acc RMS)."""
    win = int(5.0 * fs)
    n = len(acc_mag)
    acc = acc_mag.astype(np.float64)
    rms = np.zeros(n)
    for i in range(0, n - win, win // 2):
        chunk = acc[i:i + win]
        rms[i:i + win] = np.maximum(
            rms[i:i + win],
            np.sqrt(np.mean((chunk - np.mean(chunk)) ** 2)))
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-12
    thr = med + thresh_mad * mad
    above = rms > thr
    events, in_ev, start = [], False, 0
    for i in range(n):
        if above[i] and not in_ev:
            start, in_ev = i, True
        elif not above[i] and in_ev:
            events.append(start + int(np.argmax(rms[start:i]))); in_ev = False
    if in_ev:
        events.append(start + int(np.argmax(rms[start:])))
    if len(events) < 2:
        return np.array(events, dtype=int)
    merged = [events[0]]
    gap = int(min_gap_sec * fs)
    for e in events[1:]:
        if e - merged[-1] > gap:
            merged.append(e)
        elif rms[e] > rms[merged[-1]]:
            merged[-1] = e
    return np.array(merged, dtype=int)


def test_movement_initiation(df, movement_t_hr, lookahead_min=5.0, n_null=500, rng_seed=0):
    """
    H2: is an SWA *onset* more likely within `lookahead` after a distinct movement
    than after a random time?  Compares observed onset-hit rate to a matched-random
    null (same count of anchor times drawn uniformly from scored epochs).
    """
    swa = df['swa_candidate'].values.astype(bool)
    t = df['t_hr'].values
    # SWA onsets = rising edges of the candidate label
    onsets = t[1:][swa[1:] & ~swa[:-1]]
    if swa[0]:
        onsets = np.r_[t[0], onsets]
    look_hr = lookahead_min / 60.0

    def hit_rate(anchors):
        if len(anchors) == 0:
            return np.nan
        hits = 0
        for a in anchors:
            if np.any((onsets > a) & (onsets <= a + look_hr)):
                hits += 1
        return hits / len(anchors)

    obs = hit_rate(np.asarray(movement_t_hr))

    # deterministic RNG (Date/random-free constraint irrelevant here, but seed anyway)
    rng = np.random.default_rng(rng_seed)
    n_anchor = len(movement_t_hr)
    null = np.array([hit_rate(rng.choice(t, size=n_anchor, replace=False))
                     for _ in range(n_null)]) if n_anchor > 0 else np.array([])
    null_mean = float(np.nanmean(null)) if len(null) else np.nan
    p = float(np.mean(null >= obs)) if len(null) and np.isfinite(obs) else np.nan
    return dict(n_movements=int(n_anchor), n_onsets=int(len(onsets)),
                obs_hit_rate=obs, null_hit_rate=null_mean, p_value=p,
                lift=(obs / null_mean if null_mean and null_mean > 0 else np.nan))


# ── Per-session plotting ─────────────────────────────────────────────────────

def plot_overview(df, movement_t_hr, label):
    t = df['t_hr'].values
    fig, axes = plt.subplots(7, 1, figsize=(20, 16), sharex=True)
    fig.suptitle(f'{label} — CAP-SWA definition overview '
                 f'({df["swa_candidate"].sum()} SWA-candidate epochs)',
                 fontsize=14, fontweight='bold')

    # highlight SWA-candidate epochs on every axis
    swa_t = t[df['swa_candidate'].values]
    for ax in axes:
        for tt in swa_t:
            ax.axvspan(tt, tt + EPOCH_SEC / 3600, color='#2ECC71', alpha=0.12)
    for mt in movement_t_hr:
        axes[0].axvline(mt, color='k', lw=0.5, alpha=0.4)

    ax = axes[0]
    for _, r in df.iterrows():
        ax.axvspan(r['t_hr'], r['t_hr'] + EPOCH_SEC / 3600,
                   color=STAGE_COLORS.get(r['stage_code'], '#AAA'),
                   alpha=0.8 if r['stage_code'] == N3_CODE else 0.4)
    ax.set_ylabel('Stage\n+movements'); ax.set_yticks([])

    axes[1].plot(t, df['swa_score'], color='#16A085', lw=1)
    axes[1].axhline(SWA_THRESHOLD, color='k', ls='--', alpha=0.4)
    axes[1].set_ylabel('SWA score'); axes[1].set_ylim(0, 1)

    axes[2].plot(t, df['swa_s_dc'], label='slow DC', lw=0.8)
    axes[2].plot(t, df['swa_s_thorax'], label='slow thorax', lw=0.8)
    axes[2].plot(t, df['swa_s_still'], label='quiescent', lw=0.8)
    axes[2].legend(fontsize=7, ncol=3); axes[2].set_ylabel('sub-scores')

    axes[3].plot(t, df['ecg_hr_hz'] * 60, color='#C0392B', lw=0.8, label='ECG HR')
    axes[3].plot(t, df['cap_card_hz'] * 60, color='#E67E22', lw=0.6, alpha=0.6, label='CAP card')
    axes[3].legend(fontsize=7); axes[3].set_ylabel('HR (BPM)'); axes[3].set_ylim(30, 120)

    axes[4].plot(t, df['flow_rr_hz'] * 60, color='#2980B9', lw=0.8, label='PSG RR')
    axes[4].plot(t, df['cap_resp_hz'] * 60, color='#E67E22', lw=0.6, alpha=0.6, label='CAP RR')
    axes[4].legend(fontsize=7); axes[4].set_ylabel('RR (br/min)'); axes[4].set_ylim(5, 30)

    axes[5].plot(t, df['k_resp'], color='#8E44AD', lw=0.8)
    axes[5].axhline(1.0, color='k', ls='--', alpha=0.3)
    axes[5].set_ylabel('k_resp\n(CAP/thorax)'); axes[5].set_ylim(0, 3)

    axes[6].plot(t, df['eeg_delta_ratio'], color='#C0392B', lw=0.8)
    axes[6].set_ylabel('EEG delta'); axes[6].set_xlabel('Time (hr)')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── Session driver ───────────────────────────────────────────────────────────

def process_session(idx, save=True):
    session = load_session(idx)
    session.sleep_profile = load_sleep_profile(session)
    label = session.label
    print(f"\n{'='*60}\nCAP-SWA: {label}\n{'='*60}")

    try:
        card_gt = gt_heart_rate(session)
    except Exception as e:
        print(f"  cardiac GT failed: {e}"); card_gt = None
    try:
        resp_gt = gt_resp_rate(session)
    except Exception as e:
        print(f"  resp GT failed: {e}"); resp_gt = None

    df = compute_epoch_features(session, card_gt, resp_gt)
    df = score_swa(df)
    df['session'] = label
    df['subject'] = SESSION_META[idx]['subject']
    print(f"  {len(df)} epochs, {df['swa_candidate'].sum()} SWA-candidate")

    movements = detect_movements(session.cap['acc_mag'].astype(np.float64), session.fs)
    movement_t_hr = movements / session.fs / 3600.0
    h2 = test_movement_initiation(df, movement_t_hr)
    h2.update(session=label, subject=SESSION_META[idx]['subject'])
    print(f"  H2 movement-initiation: obs={h2['obs_hit_rate']:.3f} "
          f"null={h2['null_hit_rate']:.3f} lift={h2['lift']:.2f} p={h2['p_value']:.3f}")

    if save:
        sd = REPORT_DIR / label
        sd.mkdir(parents=True, exist_ok=True)
        df.to_csv(sd / 'epoch_features.csv', index=False)
        fig = plot_overview(df, movement_t_hr, label)
        fig.savefig(sd / 'night_overview.png', dpi=110, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  saved -> {sd}")

    return df, h2


# ── Cross-session hypothesis testing ─────────────────────────────────────────

def _paired_contrast(df, feat):
    """Per-subject SWA vs non-SWA median contrast + pooled Wilcoxon on subject medians."""
    subj_rows = []
    for subj, g in df.groupby('subject'):
        a = g.loc[g['swa_candidate'], feat].dropna()
        b = g.loc[~g['swa_candidate'], feat].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        subj_rows.append((subj, float(a.median()), float(b.median())))
    if len(subj_rows) < 2:
        return None
    swa_med = np.array([r[1] for r in subj_rows])
    non_med = np.array([r[2] for r in subj_rows])
    delta = swa_med - non_med
    n_pos = int(np.sum(delta > 0))
    try:
        _, p = wilcoxon(swa_med, non_med)
    except ValueError:
        p = np.nan
    return dict(feature=feat, n_subjects=len(subj_rows),
                median_delta=float(np.median(delta)),
                n_subjects_increase=n_pos,
                consistency=f'{max(n_pos, len(subj_rows)-n_pos)}/{len(subj_rows)}',
                wilcoxon_p=float(p))


def run_all(save=True):
    all_df, all_h2 = [], []
    for idx in range(12):
        try:
            df, h2 = process_session(idx, save=save)
            all_df.append(df); all_h2.append(h2)
        except Exception as e:
            print(f"  ERROR session {idx}: {e}")
            import traceback; traceback.print_exc()

    pooled = pd.concat(all_df, ignore_index=True)
    if save:
        pooled.to_parquet(REPORT_DIR / 'all_epoch_features.parquet')
        pd.DataFrame(all_h2).to_csv(REPORT_DIR / 'movement_initiation.csv', index=False)

    # Bonferroni over the 5 autonomic/validation hypotheses tested this way
    hyp_feats = {
        'H4_HR_increase': 'ecg_hr_hz',
        'H5_RR_increase': 'flow_rr_hz',
        'H6_k_deviation': 'k_resp_dev',
        'H7_card_divergence': 'card_freq_divergence',
        'H7_cap_card_freq': 'cap_card_hz',
        'H7_ppg_card_freq': 'ppg_card_hz',
        'H8_EEG_delta': 'eeg_delta_ratio',
    }
    results = []
    for name, feat in hyp_feats.items():
        r = _paired_contrast(pooled, feat)
        if r:
            r['hypothesis'] = name
            results.append(r)
    res_df = pd.DataFrame(results)
    if len(res_df):
        res_df['bonferroni_p'] = np.clip(res_df['wilcoxon_p'] * len(res_df), 0, 1)
        cols = ['hypothesis', 'feature', 'n_subjects', 'median_delta',
                'n_subjects_increase', 'consistency', 'wilcoxon_p', 'bonferroni_p']
        res_df = res_df[cols]
    if save:
        res_df.to_csv(REPORT_DIR / 'hypothesis_summary.csv', index=False)

    # H8b: SWA score as an N3 discriminator (point-biserial via AUC)
    from sklearn.metrics import roc_auc_score
    n3 = (pooled['stage_code'] == N3_CODE).astype(int)
    sc = pooled['swa_score']
    m = sc.notna()
    auc = roc_auc_score(n3[m], sc[m]) if n3[m].nunique() == 2 else np.nan

    print(f"\n{'='*60}\nPOOLED HYPOTHESIS SUMMARY\n{'='*60}")
    print(res_df.to_string(index=False) if len(res_df) else "  (no results)")
    print(f"\n  SWA-score N3 AUC (pooled): {auc:.3f}")
    print(f"  Movement-initiation lift (median over sessions): "
          f"{np.nanmedian([h['lift'] for h in all_h2]):.2f}")

    return pooled, res_df, all_h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=int, default=0)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all:
        run_all()
    else:
        process_session(args.session)


if __name__ == '__main__':
    main()
