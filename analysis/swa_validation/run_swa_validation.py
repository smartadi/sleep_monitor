"""
SWA Validation — Full Pipeline (Steps 1-4)

Runs the Lucey et al. 2019 SWA replication on all 12 sessions:
  Step 1: Shared processing (bandpass, epoch, PSD, band powers, artifact reject)
  Step 2: Reference targets (contact EEG SWA + N3 binary labels)
  Step 3: Validation metrics (correlation, Bland-Altman, coherence, ROC/AUC)
  Step 4: Reporting (per-subject table, plots, summary)

Usage:
    python analysis/swa_validation/run_swa_validation.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import coherence
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor.loader import load_session, load_sleep_profile
from sleep_monitor.sessions import SESSION_META
from swa_pipeline import (
    process_signal, align_stages_to_epochs, bandpass_fir,
    SWA_BANDS, EPOCH_SEC,
)

OUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

BANDS_FOR_CORR = ['swa_total', 'swa_1_2', 'swa_2_3', 'swa_3_4']


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 + 2: Process all sessions
# ═══════════════════════════════════════════════════════════════════════════════

def run_steps_1_2():
    all_results = []

    for idx in range(12):
        m = SESSION_META[idx]
        s = load_session(idx)
        sp = load_sleep_profile(s)
        fs = s.fs

        cap_diff = (s.cap['CLE'] - s.cap['CRE']).astype(np.float64)
        eeg = s.psg['EEG'].astype(np.float64)
        acc_mag = s.cap['acc_mag'].astype(np.float64)

        print(f'  Processing {m["label"]}...')
        eeg_res = process_signal(eeg, fs, acc_mag)
        cap_res = process_signal(cap_diff, fs, acc_mag)

        n_ep = min(eeg_res['n_epochs'], cap_res['n_epochs'])
        combined_bad = eeg_res['artifact_mask'][:n_ep] | cap_res['artifact_mask'][:n_ep]

        stages = align_stages_to_epochs(
            sp['t_ep_hr'], sp['codes'], n_ep, fs)

        art_pct = combined_bad.sum() / n_ep * 100
        print(f'    {n_ep} epochs, artifact={art_pct:.1f}%, '
              f'N3={np.sum(stages == 1)}, N2={np.sum(stages == 2)}')

        all_results.append({
            'meta': m,
            'session': s,
            'eeg': eeg_res,
            'cap': cap_res,
            'n_epochs': n_ep,
            'artifact_mask': combined_bad,
            'stages': stages,
            'artifact_pct': art_pct,
            'fs': fs,
            'eeg_raw': eeg,
            'cap_raw': cap_diff,
        })

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Validation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_correlation_metrics(eeg_vals, cap_vals):
    valid = np.isfinite(eeg_vals) & np.isfinite(cap_vals)
    if valid.sum() < 10:
        return {'r_pearson': np.nan, 'r_spearman': np.nan,
                'slope': np.nan, 'n_valid': int(valid.sum())}
    e, c = eeg_vals[valid], cap_vals[valid]
    r_p, p_p = stats.pearsonr(e, c)
    r_s, p_s = stats.spearmanr(e, c)
    slope, intercept, _, _, _ = stats.linregress(e, c)
    return {'r_pearson': r_p, 'r_spearman': r_s,
            'slope': slope, 'n_valid': int(valid.sum())}


def compute_bland_altman(eeg_vals, cap_vals):
    valid = np.isfinite(eeg_vals) & np.isfinite(cap_vals)
    if valid.sum() < 10:
        return {'bias': np.nan, 'loa_lo': np.nan, 'loa_hi': np.nan}
    e, c = eeg_vals[valid], cap_vals[valid]
    diff = c - e
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    return {
        'bias': mean_diff,
        'loa_lo': mean_diff - 1.96 * std_diff,
        'loa_hi': mean_diff + 1.96 * std_diff,
    }


def compute_coherence_metric(eeg_raw, cap_raw, fs, nperseg=1024):
    f, coh = coherence(eeg_raw, cap_raw, fs=fs, nperseg=nperseg)
    swa_mask = (f >= 0.5) & (f <= 4.5)
    return {
        'coh_freqs': f,
        'coh_values': coh,
        'mean_coh_swa': float(np.mean(coh[swa_mask])),
    }


def compute_sws_detection(stages, cap_swa, artifact_mask):
    valid = ~artifact_mask & (stages >= 0) & (stages != -1)
    if valid.sum() < 20:
        return {'auc': np.nan, 'sensitivity': np.nan,
                'specificity': np.nan, 'kappa': np.nan,
                'best_threshold': np.nan}

    y_true = (stages[valid] == 1).astype(int)
    y_score = cap_swa[valid]

    if y_true.sum() < 5 or (1 - y_true).sum() < 5:
        return {'auc': np.nan, 'sensitivity': np.nan,
                'specificity': np.nan, 'kappa': np.nan,
                'best_threshold': np.nan}

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thresh = thresholds[best_idx]
    y_pred = (y_score >= best_thresh).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        'auc': auc, 'sensitivity': sens, 'specificity': spec,
        'kappa': kappa, 'best_threshold': best_thresh,
        'fpr': fpr, 'tpr': tpr,
    }


def run_step_3(all_results):
    rows = []
    coherence_data = []
    roc_data = []

    for res in all_results:
        m = res['meta']
        n = res['n_epochs']
        bad = res['artifact_mask']
        stages = res['stages']
        good = ~bad

        row = {
            'session': m['label'],
            'subject': m['subject'],
            'night': m['night'],
            'duration_hr': res['session'].duration_hr,
            'n_epochs': n,
            'artifact_pct': res['artifact_pct'],
            'n3_epochs': int((stages == 1).sum()),
            'n3_minutes': float((stages == 1).sum()) * EPOCH_SEC / 60,
        }

        for band in BANDS_FOR_CORR:
            eeg_rel = res['eeg']['rel_powers'].get(f'{band}_rel', np.zeros(n))[:n]
            cap_rel = res['cap']['rel_powers'].get(f'{band}_rel', np.zeros(n))[:n]
            eeg_g = eeg_rel[good]
            cap_g = cap_rel[good]

            corr = compute_correlation_metrics(eeg_g, cap_g)
            ba = compute_bland_altman(eeg_g, cap_g)

            for k, v in corr.items():
                row[f'{band}_{k}'] = v
            for k, v in ba.items():
                row[f'{band}_{k}'] = v

        min_len = min(len(res['eeg_raw']), len(res['cap_raw']))
        eeg_filt = bandpass_fir(res['eeg_raw'][:min_len], res['fs'])
        cap_filt = bandpass_fir(res['cap_raw'][:min_len], res['fs'])
        coh = compute_coherence_metric(eeg_filt, cap_filt, res['fs'])
        row['mean_coh_swa'] = coh['mean_coh_swa']
        coherence_data.append({
            'session': m['label'],
            'freqs': coh['coh_freqs'],
            'values': coh['coh_values'],
        })

        cap_swa_rel = res['cap']['rel_powers'].get('swa_1_2_rel', np.zeros(n))[:n]
        sws = compute_sws_detection(stages, cap_swa_rel, bad)
        for k, v in sws.items():
            if k not in ('fpr', 'tpr'):
                row[f'sws_{k}'] = v
        if 'fpr' in sws and not np.isnan(sws.get('auc', np.nan)):
            roc_data.append({
                'session': m['label'],
                'subject': m['subject'],
                'fpr': sws['fpr'],
                'tpr': sws['tpr'],
                'auc': sws['auc'],
                'is_eeg': False,
            })

        eeg_swa_rel = res['eeg']['rel_powers'].get('swa_1_2_rel', np.zeros(n))[:n]
        eeg_sws = compute_sws_detection(stages, eeg_swa_rel, bad)
        row['eeg_n3_auc'] = eeg_sws.get('auc', np.nan)
        if 'fpr' in eeg_sws and not np.isnan(eeg_sws.get('auc', np.nan)):
            roc_data.append({
                'session': f'{m["label"]}_EEG',
                'subject': m['subject'],
                'fpr': eeg_sws['fpr'],
                'tpr': eeg_sws['tpr'],
                'auc': eeg_sws['auc'],
                'is_eeg': True,
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, coherence_data, roc_data


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Reporting & Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_swa_overlay(all_results, out_dir):
    fig, axes = plt.subplots(3, 4, figsize=(24, 14))
    axes = axes.flatten()
    for i, res in enumerate(all_results):
        ax = axes[i]
        m = res['meta']
        n = res['n_epochs']
        bad = res['artifact_mask']
        good = ~bad
        t_min = np.arange(n) * EPOCH_SEC / 60

        eeg_swa = res['eeg']['rel_powers'].get('swa_1_2_rel', np.zeros(n))[:n]
        cap_swa = res['cap']['rel_powers'].get('swa_1_2_rel', np.zeros(n))[:n]
        eeg_swa[bad] = np.nan
        cap_swa[bad] = np.nan

        ax.plot(t_min, eeg_swa, 'b-', alpha=0.6, lw=0.5, label='EEG')
        ax.plot(t_min, cap_swa, 'r-', alpha=0.6, lw=0.5, label='CAP')

        stages = res['stages']
        n3_mask = (stages == 1) & good
        ax.fill_between(t_min, 0, 1, where=n3_mask,
                        alpha=0.15, color='green', transform=ax.get_xaxis_transform())

        ax.set_title(f'{m["label"]} ({m["subject"]})', fontsize=10)
        ax.set_xlabel('Time (min)')
        if i % 4 == 0:
            ax.set_ylabel('Relative SWA (1-2 Hz)')
        if i == 0:
            ax.legend(fontsize=7)
    plt.suptitle('SWA Overlay: Contact EEG vs Capacitive (1-2 Hz relative power)\n'
                 'Green = N3 epochs', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / 'swa_overlay_all.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_bland_altman_all(df, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, band in zip(axes.flatten(), BANDS_FOR_CORR):
        biases = df[f'{band}_bias'].dropna()
        loa_lo = df[f'{band}_loa_lo'].dropna()
        loa_hi = df[f'{band}_loa_hi'].dropna()

        x = range(len(biases))
        ax.errorbar(x, biases, yerr=[biases - loa_lo, loa_hi - biases],
                    fmt='ko', capsize=4, markersize=6)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['session'], rotation=45, fontsize=8)
        ax.set_title(f'{band} — Bland-Altman (relative power)')
        ax.set_ylabel('CAP - EEG (relative)')
    plt.tight_layout()
    plt.savefig(out_dir / 'bland_altman_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_coherence_all(coherence_data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    coh_matrix = []
    for cd in coherence_data:
        f = cd['freqs']
        swa_range = (f >= 0.5) & (f <= 10)
        ax.plot(f[swa_range], cd['values'][swa_range],
                alpha=0.4, lw=0.8, label=cd['session'])
        coh_matrix.append(cd['values'][swa_range])

    coh_mean = np.mean(coh_matrix, axis=0)
    f_plot = f[swa_range]
    ax.plot(f_plot, coh_mean, 'k-', lw=2.5, label='Mean')
    ax.axvspan(1.0, 4.5, alpha=0.1, color='blue', label='SWA band')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude-squared coherence')
    ax.set_title('EEG-CAP Coherence (0.5-10 Hz)')
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlim(0.5, 10)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / 'coherence_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_all(roc_data, out_dir):
    if not roc_data:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    for rd in roc_data:
        is_eeg = rd.get('is_eeg', False)
        style = {'ls': '--', 'alpha': 0.35, 'lw': 0.8} if is_eeg else {'ls': '-', 'alpha': 0.6, 'lw': 1}
        color = 'gray' if is_eeg else None
        ax.plot(rd['fpr'], rd['tpr'], c=color, **style,
                label=f'{rd["session"]} (AUC={rd["auc"]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('N3 Detection from SWA (1-2 Hz rel) — ROC Curves\n'
                 'Solid=CAP, Dashed gray=EEG (sanity check)')
    ax.legend(fontsize=6, loc='lower right', ncol=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_scatter(all_results, out_dir):
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    axes = axes.flatten()
    for i, res in enumerate(all_results):
        ax = axes[i]
        m = res['meta']
        n = res['n_epochs']
        bad = res['artifact_mask']
        good = ~bad

        eeg_swa = res['eeg']['rel_powers'].get('swa_1_2_rel', np.zeros(n))[:n]
        cap_swa = res['cap']['rel_powers'].get('swa_1_2_rel', np.zeros(n))[:n]

        stages = res['stages']
        colors = {1: 'green', 2: 'blue', 3: 'orange', 0: 'purple', 4: 'red'}
        stage_labels = {1: 'N3', 2: 'N2', 3: 'N1', 0: 'REM', 4: 'Wake'}

        for st, color in colors.items():
            mask = good & (stages == st)
            if mask.sum() > 0:
                ax.scatter(eeg_swa[mask], cap_swa[mask], c=color, s=4,
                           alpha=0.3, label=stage_labels.get(st, '?'))

        r_p = stats.pearsonr(eeg_swa[good], cap_swa[good])[0] if good.sum() > 10 else np.nan
        ax.set_title(f'{m["label"]} r={r_p:.3f}', fontsize=10)
        ax.set_xlabel('EEG SWA (1-2 Hz rel)')
        ax.set_ylabel('CAP SWA (1-2 Hz rel)')
        if i == 0:
            ax.legend(fontsize=6, markerscale=3)

    plt.suptitle('EEG vs CAP Relative SWA (1-2 Hz) — per epoch, colored by stage', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / 'correlation_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()


def per_subject_summary(df):
    subj_df = df.groupby('subject').agg({
        'duration_hr': 'sum',
        'n_epochs': 'sum',
        'artifact_pct': 'mean',
        'n3_minutes': 'sum',
        'swa_1_2_r_pearson': 'mean',
        'swa_1_2_r_spearman': 'mean',
        'mean_coh_swa': 'mean',
        'sws_auc': 'mean',
        'sws_sensitivity': 'mean',
        'sws_specificity': 'mean',
        'sws_kappa': 'mean',
    }).reset_index()
    subj_df.columns = [
        'Subject', 'Total_hr', 'Total_epochs', 'Mean_artifact_%',
        'Total_N3_min', 'Mean_r_pearson', 'Mean_r_spearman',
        'Mean_coherence', 'Mean_AUC', 'Mean_sensitivity',
        'Mean_specificity', 'Mean_kappa',
    ]
    return subj_df


def print_summary(df, subj_df):
    print('\n' + '=' * 90)
    print('SWA VALIDATION — RESULTS SUMMARY')
    print('=' * 90)

    print('\n--- Per-Session Results (1-2 Hz band, relative power) ---')
    cols = ['session', 'subject', 'artifact_pct', 'n3_minutes',
            'swa_1_2_r_pearson', 'swa_1_2_r_spearman',
            'mean_coh_swa', 'eeg_n3_auc', 'sws_auc',
            'sws_kappa']
    print(df[cols].to_string(index=False, float_format='%.3f'))

    print('\n--- Per-Subject Summary ---')
    print(subj_df.to_string(index=False, float_format='%.3f'))

    print('\n--- Cohort Aggregates (mean ± std) ---')
    for col in ['swa_1_2_r_pearson', 'swa_1_2_r_spearman', 'mean_coh_swa',
                'sws_auc', 'sws_sensitivity', 'sws_specificity', 'sws_kappa']:
        vals = df[col].dropna()
        print(f'  {col:30s}: {vals.mean():.3f} ± {vals.std():.3f}  '
              f'(range {vals.min():.3f}–{vals.max():.3f})')

    for band in BANDS_FOR_CORR:
        vals = df[f'{band}_r_pearson'].dropna()
        print(f'  {band:15s} r_pearson: {vals.mean():.3f} ± {vals.std():.3f}')

    excluded = df[df['artifact_pct'] > 10]
    if len(excluded) > 0:
        print(f'\n  WARNING: {len(excluded)} night(s) >10% artifact: '
              f'{excluded["session"].tolist()}')

    marginal = df[df['n3_minutes'] < 5]
    if len(marginal) > 0:
        print(f'  WARNING: {len(marginal)} night(s) <5 min N3: '
              f'{marginal["session"].tolist()}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('Step 1+2: Processing all sessions...')
    all_results = run_steps_1_2()

    print('\nStep 3: Computing validation metrics...')
    df, coherence_data, roc_data = run_step_3(all_results)

    print('\nStep 4: Generating reports and plots...')
    df.to_csv(OUT_DIR / 'swa_validation_results.csv', index=False)

    subj_df = per_subject_summary(df)
    subj_df.to_csv(OUT_DIR / 'swa_validation_per_subject.csv', index=False)

    plot_swa_overlay(all_results, OUT_DIR)
    plot_bland_altman_all(df, OUT_DIR)
    plot_coherence_all(coherence_data, OUT_DIR)
    plot_roc_all(roc_data, OUT_DIR)
    plot_correlation_scatter(all_results, OUT_DIR)

    print_summary(df, subj_df)

    print(f'\nOutputs saved to {OUT_DIR}/')
    print('Done.')


if __name__ == '__main__':
    main()
