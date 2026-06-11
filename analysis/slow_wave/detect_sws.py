"""
SWS (N3) detection from CAP sensor features.

Extracts per-epoch features from all 12 sessions, validates discriminative power,
trains a LOSO binary classifier (N3 vs non-N3), and generates a report.

Features used (physiologically motivated from marker review):
  - Motion level (acc_rms) — stillness marks SWS
  - Band power ratios (delta/theta/alpha/beta) — spectral shape changes
  - Spectral entropy — more concentrated power in SWS
  - Respiratory rate & regularity — slower, more regular in SWS
  - Cardiac rate & regularity — lower HR, lower HRV in SWS
  - DC mean stability — slow monotonic drift during SWS vs abrupt shifts
  - CLE-CRE coherence — cross-sensor coupling
  - Harmonic features — HER, n_harmonics (subject-dependent but informative)

Output: reports/slow_wave/sws_detection_report.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import kruskal, mannwhitneyu
from scipy.signal import welch, hilbert
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_all_sessions
from sleep_monitor.config import (
    FS, EEG_BANDS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, STAGE_LABELS, STAGE_COLORS,
)
from sleep_monitor.filters import bandpass
from sleep_monitor.rates import rate_acf
from sleep_monitor.harmonics import detect_harmonics

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature extraction ────────────────────────────────────────────────────────

def _welch_band_powers(sig, fs, bands, total_range=(0.5, 30.0), nperseg=400):
    if len(sig) < nperseg:
        out = {name: np.nan for name in bands}
        out['total_power'] = np.nan
        return out
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                       scaling='density')
    df = freqs[1] - freqs[0]
    t_mask = (freqs >= total_range[0]) & (freqs <= total_range[1])
    tp = float(np.trapz(psd[t_mask], dx=df))
    out = {}
    for name, (flo, fhi) in bands.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        bp = float(np.trapz(psd[mask], dx=df))
        out[name] = bp / tp if tp > 0 else 0.0
    out['total_power'] = tp
    return out


def _spectral_entropy(sig, fs, nperseg=400):
    if len(sig) < nperseg:
        return np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    psd_norm = psd / (psd.sum() + 1e-20)
    psd_norm = psd_norm[psd_norm > 0]
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def _instantaneous_freq_variability(sig, f_lo, f_hi, fs):
    """CV of instantaneous frequency in a band — lower = more regular."""
    bp = bandpass(sig, f_lo, f_hi, fs)
    analytic = hilbert(bp)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs
    inst_freq = inst_freq[(inst_freq >= f_lo * 0.5) & (inst_freq <= f_hi * 2.0)]
    if len(inst_freq) < 10:
        return np.nan
    mean_f = np.mean(inst_freq)
    if mean_f < 1e-6:
        return np.nan
    return float(np.std(inst_freq) / mean_f)


def _dc_mean_stability(sig):
    """Slope of linear fit to raw signal — small = stable DC."""
    n = len(sig)
    if n < 10:
        return np.nan, np.nan
    t = np.arange(n, dtype=np.float64)
    mean_val = float(np.mean(sig))
    coeffs = np.polyfit(t, sig.astype(np.float64), 1)
    slope = float(coeffs[0])
    residual_std = float(np.std(sig - np.polyval(coeffs, t)))
    return slope, residual_std


def _harmonic_features_for_epoch(sig, fs, f0_range=(0.1, 0.8)):
    """Quick harmonic analysis of a single epoch."""
    nperseg = min(len(sig), int(8.0 * fs))
    if len(sig) < nperseg:
        return {'her': np.nan, 'n_harmonics': np.nan, 'hps_score': np.nan}
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                       scaling='density')
    from sleep_monitor.harmonics import _hps, _explicit_harmonics
    _, hps_score = _hps(psd, freqs, f0_range)
    explicit = _explicit_harmonics(psd, freqs, f0_range)
    return {
        'her': explicit['harmonic_energy_ratio'],
        'n_harmonics': explicit['n_harmonics'],
        'hps_score': hps_score,
    }


def extract_sws_features(session):
    """Extract per-epoch feature vectors with SWS-specific markers."""
    if session.sleep_profile is None:
        raise ValueError(f'{session.label}: no sleep profile')

    sp = session.sleep_profile
    n_epochs = len(sp['codes'])
    fs = session.fs
    epoch_n = int(PSG_EPOCH_SEC * fs)
    nperseg = min(epoch_n, int(4.0 * fs))
    t_hr = session.time_hr

    cap = session.cap
    cle = cap['CLE'].astype(np.float64)
    cre = cap['CRE'].astype(np.float64)
    ch = cap['CH'].astype(np.float64)
    diff = cle - cre
    acc = cap['acc_mag'].astype(np.float64)

    rows = []
    for ei in range(n_epochs):
        t_ep_start = sp['t_ep_hr'][ei]
        t_ep_end = t_ep_start + PSG_EPOCH_SEC / 3600.0
        mask = (t_hr >= t_ep_start) & (t_hr < t_ep_end)
        if mask.sum() < epoch_n * 0.5:
            continue

        idx = np.where(mask)[0]
        row = {
            'session': session.label,
            'subject': session.subject,
            'epoch_idx': ei,
            't_hr': float(t_ep_start),
            'stage_code': int(sp['codes'][ei]),
            'stage_label': sp['labels'][ei],
        }

        seg_diff = diff[idx]
        seg_cle = cle[idx]
        seg_cre = cre[idx]
        seg_ch = ch[idx]
        seg_acc = acc[idx]

        # 1. Band power ratios for CLE-CRE differential
        bp = _welch_band_powers(seg_diff, fs, EEG_BANDS, nperseg=nperseg)
        for bname, val in bp.items():
            row[f'diff_{bname}'] = val

        # 2. Band power ratios for CH
        bp_ch = _welch_band_powers(seg_ch, fs, EEG_BANDS, nperseg=nperseg)
        for bname, val in bp_ch.items():
            row[f'ch_{bname}'] = val

        # 3. RMS amplitudes
        row['diff_rms'] = float(np.sqrt(np.mean(seg_diff ** 2)))
        row['ch_rms'] = float(np.sqrt(np.mean(seg_ch ** 2)))

        # 4. Spectral entropy
        row['diff_spectral_entropy'] = _spectral_entropy(seg_diff, fs, nperseg)
        row['ch_spectral_entropy'] = _spectral_entropy(seg_ch, fs, nperseg)

        # 5. Motion features
        row['acc_rms'] = float(np.sqrt(np.mean((seg_acc - np.mean(seg_acc)) ** 2)))
        acc_bp_resp = bandpass(seg_acc, RESP_LO, RESP_HI, fs)
        row['acc_resp_power'] = float(np.mean(acc_bp_resp ** 2))

        # 6. Respiratory rate and regularity
        bp_resp = bandpass(seg_diff, RESP_LO, RESP_HI, fs)
        row['resp_rate_hz'] = rate_acf(bp_resp, RESP_LO, RESP_HI, fs)
        row['resp_regularity_cv'] = _instantaneous_freq_variability(
            seg_diff, RESP_LO, RESP_HI, fs)

        # 7. Cardiac rate and regularity
        bp_card = bandpass(seg_diff, CARD_LO, CARD_HI, fs)
        row['card_rate_hz'] = rate_acf(bp_card, CARD_LO, CARD_HI, fs)
        row['card_regularity_cv'] = _instantaneous_freq_variability(
            seg_diff, CARD_LO, CARD_HI, fs)

        # 8. DC mean stability
        dc_slope, dc_resid = _dc_mean_stability(seg_diff)
        row['dc_slope'] = dc_slope
        row['dc_residual_std'] = dc_resid
        row['dc_mean'] = float(np.mean(seg_diff))

        # CH DC stability too
        ch_slope, ch_resid = _dc_mean_stability(seg_ch)
        row['ch_dc_slope'] = ch_slope

        # 9. CLE-CRE coherence
        from scipy.signal import coherence as sp_coherence
        if len(seg_cle) >= nperseg:
            freqs_c, coh = sp_coherence(seg_cle, seg_cre, fs=fs,
                                         nperseg=nperseg, noverlap=nperseg // 2)
            resp_mask = (freqs_c >= RESP_LO) & (freqs_c <= RESP_HI)
            card_mask = (freqs_c >= CARD_LO) & (freqs_c <= CARD_HI)
            row['coh_resp'] = float(np.mean(coh[resp_mask])) if resp_mask.any() else np.nan
            row['coh_card'] = float(np.mean(coh[card_mask])) if card_mask.any() else np.nan
        else:
            row['coh_resp'] = np.nan
            row['coh_card'] = np.nan

        # 10. Harmonic features
        harm = _harmonic_features_for_epoch(seg_diff, fs)
        row['her'] = harm['her']
        row['n_harmonics'] = harm['n_harmonics']
        row['hps_score'] = harm['hps_score']

        # 11. Waveform shape: kurtosis, zero-crossing rate
        from scipy.stats import kurtosis as sp_kurtosis
        row['diff_kurtosis'] = float(sp_kurtosis(seg_diff, fisher=True))
        zc = np.sum(np.diff(np.sign(seg_diff - np.mean(seg_diff))) != 0)
        row['diff_zcr'] = float(zc) / (len(seg_diff) / fs)

        rows.append(row)

    return pd.DataFrame(rows)


# ── Feature columns ──────────────────────────────────────────────────────────

META_COLS = {'session', 'subject', 'epoch_idx', 't_hr', 'stage_code', 'stage_label'}

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]


# ── Discriminative power analysis ─────────────────────────────────────────────

def analyze_discriminative_power(df, feature_cols):
    """Rank features by N3 vs non-N3 discriminative power."""
    n3 = df[df['stage_code'] == 1]
    non_n3 = df[df['stage_code'] != 1]

    results = []
    for col in feature_cols:
        vals_n3 = n3[col].dropna()
        vals_other = non_n3[col].dropna()
        if len(vals_n3) < 5 or len(vals_other) < 5:
            continue

        stat, p = mannwhitneyu(vals_n3, vals_other, alternative='two-sided')
        n = len(vals_n3) + len(vals_other)
        r = 1 - (2 * stat) / (len(vals_n3) * len(vals_other))

        med_n3 = vals_n3.median()
        med_other = vals_other.median()
        pooled_std = df[col].dropna().std()
        effect_size = (med_n3 - med_other) / pooled_std if pooled_std > 0 else 0

        results.append({
            'feature': col,
            'median_N3': med_n3,
            'median_other': med_other,
            'effect_size': effect_size,
            'abs_effect_size': abs(effect_size),
            'mann_whitney_p': p,
            'rank_biserial_r': r,
        })

    return pd.DataFrame(results).sort_values('abs_effect_size', ascending=False)


# ── LOSO classification ──────────────────────────────────────────────────────

def _add_temporal_context(df, feature_cols, windows=[3, 5, 9]):
    """Add rolling mean/std features to capture temporal context per session."""
    new_cols = []
    for sess in df['session'].unique():
        sess_mask = df['session'] == sess
        sess_df = df.loc[sess_mask].copy()
        for w in windows:
            for col in feature_cols:
                mean_col = f'{col}_rm{w}'
                std_col = f'{col}_rs{w}'
                vals = sess_df[col]
                sess_df[mean_col] = vals.rolling(w, center=True, min_periods=1).mean()
                sess_df[std_col] = vals.rolling(w, center=True, min_periods=1).std()
                if mean_col not in new_cols:
                    new_cols.extend([mean_col, std_col])
            # Rate of change (first difference, smoothed)
            for col in feature_cols:
                diff_col = f'{col}_diff'
                sess_df[diff_col] = sess_df[col].diff().rolling(3, center=True, min_periods=1).mean()
                if diff_col not in new_cols:
                    new_cols.append(diff_col)
        df.loc[sess_mask, [c for c in sess_df.columns if c not in df.columns or c in new_cols]] = \
            sess_df[[c for c in sess_df.columns if c not in df.columns or c in new_cols]]
    for c in new_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df, new_cols


def _optimize_threshold(y_true, probs):
    """Find threshold that maximizes F1 score."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def run_loso_classification(df, feature_cols):
    """LOSO cross-validation: leave both nights of each subject out."""
    subjects = sorted(df['subject'].unique())
    df = df[df['stage_code'] >= 0].copy()
    df['is_n3'] = (df['stage_code'] == 1).astype(int)

    # Add temporal context features
    print("  Adding temporal context features...")
    base_cols = list(feature_cols)
    df, new_cols = _add_temporal_context(df, base_cols, windows=[3, 5])
    all_feature_cols = base_cols + new_cols
    # Drop columns that are all NaN
    all_feature_cols = [c for c in all_feature_cols if c in df.columns and df[c].notna().sum() > 100]
    print(f"  Total features with temporal context: {len(all_feature_cols)}")

    X = df[all_feature_cols].values.astype(np.float64)
    y = df['is_n3'].values

    all_preds = np.full(len(df), np.nan)
    all_probs = np.full(len(df), np.nan)
    fold_results = []

    # Compute class weight ratio for sample_weight
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    w_pos = n_neg / n_pos if n_pos > 0 else 1.0

    for subj in subjects:
        test_mask = (df['subject'] == subj).values
        train_mask = ~test_mask

        X_train, y_train = X[train_mask].copy(), y[train_mask]
        X_test, y_test = X[test_mask].copy(), y[test_mask]

        # Handle NaNs
        col_medians = np.nanmedian(X_train, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = col_medians[j]
            X_test[np.isnan(X_test[:, j]), j] = col_medians[j]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Class-weighted sample weights
        sample_weights = np.where(y_train == 1, w_pos, 1.0)

        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)

        probs = clf.predict_proba(X_test)[:, 1]

        # Optimize threshold on training fold
        train_probs = clf.predict_proba(X_train)[:, 1]
        opt_thresh, _ = _optimize_threshold(y_train, train_probs)
        preds = (probs >= opt_thresh).astype(int)

        all_preds[test_mask] = preds
        all_probs[test_mask] = probs

        n3_count = y_test.sum()
        total = len(y_test)
        acc = accuracy_score(y_test, preds)
        sens = recall_score(y_test, preds, zero_division=0)
        spec = recall_score(1 - y_test, 1 - preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs) if 0 < n3_count < total else np.nan

        fold_results.append({
            'subject': subj,
            'n_epochs': total,
            'n_n3': int(n3_count),
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec,
            'f1': f1,
            'auc': auc,
            'threshold': opt_thresh,
        })
        print(f"  {subj}: acc={acc:.3f} sens={sens:.3f} spec={spec:.3f} "
              f"F1={f1:.3f} AUC={auc:.3f} thr={opt_thresh:.2f} (N3={n3_count}/{total})")

    valid = ~np.isnan(all_preds)
    overall_acc = accuracy_score(y[valid], all_preds[valid])
    overall_sens = recall_score(y[valid], all_preds[valid], zero_division=0)
    overall_spec = recall_score(1 - y[valid], 1 - all_preds[valid], zero_division=0)
    overall_f1 = f1_score(y[valid], all_preds[valid], zero_division=0)
    overall_auc = roc_auc_score(y[valid], all_probs[valid])

    print(f"\n  OVERALL: acc={overall_acc:.3f} sens={overall_sens:.3f} "
          f"spec={overall_spec:.3f} F1={overall_f1:.3f} AUC={overall_auc:.3f}")

    # Feature importances from a model trained on all data
    X_all = X.copy()
    col_medians = np.nanmedian(X_all, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    for j in range(X_all.shape[1]):
        X_all[np.isnan(X_all[:, j]), j] = col_medians[j]

    sample_weights_all = np.where(y == 1, w_pos, 1.0)
    clf_all = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf_all.fit(X_all, y, sample_weight=sample_weights_all)
    importances = clf_all.feature_importances_

    return {
        'fold_results': pd.DataFrame(fold_results),
        'overall': {
            'accuracy': overall_acc, 'sensitivity': overall_sens,
            'specificity': overall_spec, 'f1': overall_f1, 'auc': overall_auc,
        },
        'all_preds': all_preds, 'all_probs': all_probs,
        'y_true': y,
        'importances': dict(zip(all_feature_cols, importances)),
        'feature_cols_used': all_feature_cols,
        'df': df,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_report(df, disc_df, cls_results, feature_cols, out_path):
    """Generate comprehensive multi-panel report."""
    fig = plt.figure(figsize=(28, 24))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('SWS (N3) Detection from CAP Sensors — LOSO Results', fontsize=16, y=0.98)

    # ── Panel 1: Top 15 features by effect size ──
    ax1 = fig.add_subplot(gs[0, 0:2])
    top15 = disc_df.head(15)
    colors = ['#2ECC71' if es > 0 else '#E74C3C' for es in top15['effect_size']]
    ax1.barh(range(len(top15)), top15['effect_size'].values, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top15)))
    ax1.set_yticklabels(top15['feature'].values, fontsize=8)
    ax1.set_xlabel('Effect size (Cohen d, N3 vs non-N3)')
    ax1.set_title('Top 15 Discriminative Features')
    ax1.axvline(0, color='k', linewidth=0.5)
    ax1.invert_yaxis()

    # ── Panel 2: Feature importance (GBM) ──
    ax2 = fig.add_subplot(gs[0, 2:4])
    imp = cls_results['importances']
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [x[0] for x in imp_sorted]
    vals = [x[1] for x in imp_sorted]
    ax2.barh(range(len(names)), vals, color='#3498DB', alpha=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('GBM Feature Importance')
    ax2.set_title('Top 15 Classifier Features')
    ax2.invert_yaxis()

    # ── Panel 3: Per-subject performance ──
    ax3 = fig.add_subplot(gs[1, 0:2])
    fold_df = cls_results['fold_results']
    x = range(len(fold_df))
    w = 0.2
    ax3.bar([i - w for i in x], fold_df['sensitivity'], w, label='Sensitivity', color='#2ECC71')
    ax3.bar([i for i in x], fold_df['specificity'], w, label='Specificity', color='#3498DB')
    ax3.bar([i + w for i in x], fold_df['f1'], w, label='F1', color='#E67E22')
    ax3.set_xticks(x)
    ax3.set_xticklabels(fold_df['subject'].values, fontsize=9)
    ax3.set_ylabel('Score')
    ax3.set_title('Per-Subject LOSO Performance')
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 1.05)
    overall = cls_results['overall']
    ax3.axhline(overall['f1'], color='#E67E22', linestyle='--', alpha=0.5)

    # ── Panel 4: ROC curves per subject ──
    ax4 = fig.add_subplot(gs[1, 2:4])
    result_df = cls_results['df']
    y_true = cls_results['y_true']
    all_probs = cls_results['all_probs']
    valid = ~np.isnan(all_probs)

    for subj in sorted(result_df['subject'].unique()):
        subj_mask = (result_df['subject'] == subj).values & valid
        if subj_mask.sum() < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[subj_mask], all_probs[subj_mask])
        auc_val = roc_auc_score(y_true[subj_mask], all_probs[subj_mask])
        ax4.plot(fpr, tpr, label=f'{subj} (AUC={auc_val:.2f})', alpha=0.7)

    fpr_all, tpr_all, _ = roc_curve(y_true[valid], all_probs[valid])
    ax4.plot(fpr_all, tpr_all, 'k-', linewidth=2, label=f'Overall (AUC={overall["auc"]:.2f})')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves (N3 Detection)')
    ax4.legend(fontsize=7, loc='lower right')

    # ── Panel 5: Confusion matrix ──
    ax5 = fig.add_subplot(gs[2, 0])
    cm = confusion_matrix(y_true[valid], cls_results['all_preds'][valid])
    im = ax5.imshow(cm, cmap='Blues', interpolation='nearest')
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['non-N3', 'N3'])
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['non-N3', 'N3'])
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('True')
    ax5.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)

    # ── Panel 6: Top 4 feature box plots (N3 vs non-N3) ──
    top4 = disc_df.head(4)['feature'].values
    for pi, feat in enumerate(top4):
        ax = fig.add_subplot(gs[2, 1 + pi] if pi < 3 else gs[3, 0])
        n3_vals = df[df['stage_code'] == 1][feat].dropna()
        non_n3_vals = df[df['stage_code'] != 1][feat].dropna()
        bp = ax.boxplot([non_n3_vals, n3_vals], tick_labels=['non-N3', 'N3'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#3498DB')
        bp['boxes'][1].set_facecolor('#2ECC71')
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel('Value')

    # ── Panel 7: Example hypnogram overlay (first 2 sessions) ──
    sessions_to_show = sorted(result_df['session'].unique())[:2]
    for si, sess_label in enumerate(sessions_to_show):
        ax = fig.add_subplot(gs[3, 1 + si])
        sess_df = result_df[result_df['session'] == sess_label].copy()
        sess_mask = (result_df['session'] == sess_label).values
        probs = all_probs[sess_mask]
        t = sess_df['t_hr'].values

        ax.fill_between(t, 0, 1, where=sess_df['stage_code'].values == 1,
                        alpha=0.3, color='#2ECC71', label='True N3')
        ax.plot(t, probs, color='#E74C3C', linewidth=0.8, alpha=0.8, label='P(N3)')
        ax.axhline(0.5, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (hr)')
        ax.set_ylabel('P(N3)')
        ax.set_title(f'{sess_label}: Predicted vs True N3')
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

    # ── Panel 8: Summary table ──
    ax_table = fig.add_subplot(gs[3, 3])
    ax_table.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Accuracy', f'{overall["accuracy"]:.3f}'],
        ['Sensitivity', f'{overall["sensitivity"]:.3f}'],
        ['Specificity', f'{overall["specificity"]:.3f}'],
        ['F1 Score', f'{overall["f1"]:.3f}'],
        ['AUC', f'{overall["auc"]:.3f}'],
        ['N epochs', str(len(result_df))],
        ['N3 epochs', str(int(y_true.sum()))],
        ['Features', str(len(feature_cols))],
        ['CV', 'LOSO (6-fold)'],
    ]
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for i in range(len(table_data)):
        for j in range(2):
            cell = table[i, j]
            if i == 0:
                cell.set_facecolor('#34495E')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
    ax_table.set_title('Overall LOSO Results', fontsize=11)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nReport saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SWS (N3) Detection from CAP Sensors")
    print("=" * 70)

    # Load all 12 sessions
    print("\n[1/4] Loading sessions...")
    sessions = load_all_sessions(with_sleep_profiles=True)

    # Extract features
    print("\n[2/4] Extracting SWS features per epoch...")
    all_dfs = []
    for s in sessions:
        if s.sleep_profile is None:
            print(f"  {s.label}: skipped (no sleep profile)")
            continue
        print(f"  {s.label}: ", end='', flush=True)
        feat_df = extract_sws_features(s)
        n3_count = (feat_df['stage_code'] == 1).sum()
        print(f"{len(feat_df)} epochs, {n3_count} N3")
        all_dfs.append(feat_df)

    df = pd.concat(all_dfs, ignore_index=True)
    feature_cols = get_feature_cols(df)
    print(f"\n  Total: {len(df)} epochs, {(df['stage_code'] == 1).sum()} N3, "
          f"{len(feature_cols)} features")

    # Save feature matrix
    feat_path = REPORT_DIR / 'sws_features.parquet'
    df.to_parquet(feat_path)
    print(f"  Features saved: {feat_path}")

    # Analyze discriminative power
    print("\n[3/4] Analyzing feature discriminative power...")
    disc_df = analyze_discriminative_power(df, feature_cols)
    print("\n  Top 10 features (by |effect size|):")
    for _, row in disc_df.head(10).iterrows():
        sig = '***' if row['mann_whitney_p'] < 0.001 else ('**' if row['mann_whitney_p'] < 0.01 else '*' if row['mann_whitney_p'] < 0.05 else '')
        direction = 'N3 UP' if row['effect_size'] > 0 else 'N3 DN'
        print(f"    {row['feature']:30s}  d={row['effect_size']:+.3f}  {direction}  p={row['mann_whitney_p']:.2e} {sig}")

    disc_csv = REPORT_DIR / 'sws_feature_ranking.csv'
    disc_df.to_csv(disc_csv, index=False)

    # Classification
    print("\n[4/4] LOSO classification (N3 vs non-N3)...")
    cls_results = run_loso_classification(df, feature_cols)

    # Plot report
    report_path = REPORT_DIR / 'sws_detection_report.png'
    used_cols = cls_results.get('feature_cols_used', feature_cols)
    plot_report(df, disc_df, cls_results, used_cols, report_path)

    # Also save fold results
    fold_csv = REPORT_DIR / 'sws_loso_folds.csv'
    cls_results['fold_results'].to_csv(fold_csv, index=False)
    print(f"Fold results saved: {fold_csv}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
