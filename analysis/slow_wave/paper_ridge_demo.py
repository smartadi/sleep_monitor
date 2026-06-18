"""
Paper-ready harmonic ridge demo.

Part A: Per-session spectrogram + ridge overlay with aligned hypnogram (CRE channel)
Part B: Pooled quantification — ridge prominence by stage, N3 vs non-N3 ROC/AUC
Part C: Stage 4 — LOSO N3 binary classifier from ridge features alone

Outputs -> writeup/figures/harmonics/paper_*
          reports/slow_wave/paper_*

Usage:
    python paper_ridge_demo.py              # all 12 sessions
    python paper_ridge_demo.py S1N1 S2N2    # specific sessions only
    python paper_ridge_demo.py --skip-overlay  # quantification + LOSO only (fast)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.signal import spectrogram as sp_spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sleep_monitor import load_all_sessions, FS, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.preprocessing import remove_acc_artifact
from sleep_monitor.harmonics import detect_persistent_ridges, compute_prominence_score

FIG_DIR = Path(__file__).resolve().parents[2] / 'writeup' / 'figures' / 'harmonics'
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = REPORT_DIR / 'overlay' / 'ridge_overlay_epochs.parquet'

CHANNEL = 'CRE'
WIN_SEC = 30.0
STEP_SEC = 15.0
MAX_FREQ = 5.0
SMOOTH_WINDOWS = 9
MIN_PERSIST_SEC = 300.0
MAX_FREQ_JUMP = 0.10
PEAK_PROM_FRAC = 0.25
MAX_GAP_WINDOWS = 6
WELCH_SEG_SEC = 10.0


# ── Part A: Per-session spectrogram + ridge overlay ──────────────────────────

def compute_fine_spectrogram(sig, fs=100.0, max_freq=5.0):
    f, t, Sxx = sp_spectrogram(sig, fs=fs, nperseg=2048, noverlap=1920,
                                nfft=4096, scaling='density')
    mask = f <= max_freq
    return t / 3600.0, f[mask], 10 * np.log10(Sxx[mask] + 1e-30)


def plot_overlay(session, sig_clean, rr, ps, out_path):
    """2-row paper figure: hypnogram + spectrogram with ridge traces."""
    sp = session.sleep_profile
    t_hr = rr['t_hr']
    ridges = rr['ridges']
    score = ps['prominence_score']

    fig, axes = plt.subplots(2, 1, figsize=(14, 5),
                             gridspec_kw={'height_ratios': [0.25, 1.0]},
                             sharex=True)

    # ── Row 0: Hypnogram ──
    ax = axes[0]
    if sp is not None:
        for j in range(len(sp['t_ep_hr']) - 1):
            c = int(sp['codes'][j])
            ax.axvspan(sp['t_ep_hr'][j], sp['t_ep_hr'][j + 1],
                       color=STAGE_COLORS.get(c, '#AAA'), alpha=0.7)
    ax.set_yticks([])
    ax.set_ylabel('Stage', fontsize=10)
    patches = [mpatches.Patch(color=STAGE_COLORS[c], label=STAGE_LABELS[c])
               for c in STAGE_ORDER]
    ax.legend(handles=patches, loc='upper right', fontsize=8, ncol=5,
              framealpha=0.9)
    ax.set_title(f'{session.label} — Spectral Ridges in CRE Channel',
                 fontsize=12, fontweight='bold')

    # ── Row 1: Spectrogram + ridge traces ──
    ax = axes[1]
    t_spec, f_spec, Sxx_db = compute_fine_spectrogram(sig_clean, fs=FS,
                                                       max_freq=MAX_FREQ)
    vmin, vmax = np.nanpercentile(Sxx_db, [5, 95])
    ax.pcolormesh(t_spec, f_spec, Sxx_db,
                  shading='gouraud', cmap='inferno',
                  vmin=vmin, vmax=vmax, rasterized=True)

    # Motion-masked regions
    motion = rr['motion_mask']
    dt = STEP_SEC / 3600.0
    i = 0
    while i < len(motion):
        if motion[i]:
            start = t_hr[i] - dt / 2
            while i < len(motion) and motion[i]:
                i += 1
            end = t_hr[i - 1] + dt / 2
            ax.axvspan(start, end, color='red', alpha=0.12, zorder=2)
        else:
            i += 1

    # Ridge traces colored by prominence
    if ridges:
        all_prom = np.concatenate([
            r['prominence_trace'][np.isfinite(r['prominence_trace'])]
            for r in ridges if np.any(np.isfinite(r.get('prominence_trace', [])))
        ])
        prom_norm = Normalize(
            vmin=1.0,
            vmax=np.percentile(all_prom, 95) if len(all_prom) > 0 else 10.0,
        )
        cmap = plt.cm.cool

        ridge_prom = [r.get('median_prominence', 0.0) for r in ridges]
        top_n = min(15, len(ridges))
        top_idxs = set(np.argsort(ridge_prom)[-top_n:])

        for ri, ridge in enumerate(ridges):
            valid = ~np.isnan(ridge['freq_trace'])
            pt = ridge.get('prominence_trace', np.full_like(ridge['freq_trace'], np.nan))
            if valid.sum() < 2:
                continue
            t_r = t_hr[valid]
            f_r = ridge['freq_trace'][valid]
            p_r = pt[valid]
            lw = 2.5 if ri in top_idxs else 1.2
            alpha = 0.95 if ri in top_idxs else 0.5
            for si in range(len(t_r) - 1):
                pval = p_r[si] if np.isfinite(p_r[si]) else 1.0
                color = cmap(prom_norm(pval))
                ax.plot(t_r[si:si + 2], f_r[si:si + 2],
                        '-', color=color, lw=lw, alpha=alpha, zorder=3)

            if ri in top_idxs:
                mid = len(t_r) // 2
                lbl = f'{ridge["median_freq"]:.1f} Hz'
                ax.annotate(lbl, (t_r[mid], f_r[mid] + 0.1),
                            fontsize=6, color='white', fontweight='bold',
                            bbox=dict(fc='black', alpha=0.6, pad=1, lw=0),
                            zorder=4)

    ax.set_ylim(0, MAX_FREQ)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_xlabel('Time (hours)', fontsize=10)

    n_ridges = len(ridges)
    n_valid = int((~motion).sum())
    ax.text(0.005, 0.97,
            f'{n_ridges} persistent ridges detected',
            transform=ax.transAxes, fontsize=9, va='top',
            color='white', bbox=dict(fc='black', alpha=0.6, pad=2, lw=0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_overlays(sessions):
    """Run detection + generate overlay for each session."""
    for s in sessions:
        print(f'  {s.label}: artifact removal + ridge detection...')
        acc_mag = s.cap['acc_mag']
        sig = remove_acc_artifact(s.cap[CHANNEL], acc_mag, 0.05, 4.0)

        rr = detect_persistent_ridges(
            sig, fs=FS,
            win_sec=WIN_SEC, step_sec=STEP_SEC,
            max_freq=MAX_FREQ,
            smooth_windows=SMOOTH_WINDOWS,
            min_persistence_sec=MIN_PERSIST_SEC,
            max_freq_jump=MAX_FREQ_JUMP,
            peak_prominence_frac=PEAK_PROM_FRAC,
            max_gap_windows=MAX_GAP_WINDOWS,
            welch_seg_sec=WELCH_SEG_SEC,
            acc_mag=acc_mag,
        )
        ps = compute_prominence_score(rr)

        out = FIG_DIR / f'paper_overlay_{s.label}.png'
        plot_overlay(s, sig, rr, ps, out)
        print(f'    -> {out.name}  ({len(rr["ridges"])} ridges)')


# ── Part B: Pooled quantification ────────────────────────────────────────────

def quantify_pooled(df):
    """Violin plots by stage + N3 AUC from cached parquet data."""
    cre = df[(df['channel'] == CHANNEL) & (~df['motion_masked'])
             & (df['stage_code'] >= 0)].copy()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1],
                          width_ratios=[1.2, 1.0, 1.0],
                          hspace=0.35, wspace=0.35)

    # ── Panel A: Active ridges by stage (violin + box) ──
    ax = fig.add_subplot(gs[0, 0])
    stage_data_ridges = []
    stage_labels_list = []
    stage_colors_list = []
    for sc in STAGE_ORDER:
        sv = cre[cre['stage_code'] == sc]
        if len(sv) > 0:
            stage_data_ridges.append(sv['n_active_ridges'].values)
            stage_labels_list.append(STAGE_LABELS[sc])
            stage_colors_list.append(STAGE_COLORS[sc])

    vp = ax.violinplot(stage_data_ridges, positions=range(len(stage_data_ridges)),
                       showmedians=False, showextrema=False)
    for body, color in zip(vp['bodies'], stage_colors_list):
        body.set_facecolor(color)
        body.set_alpha(0.4)

    bp = ax.boxplot(stage_data_ridges, positions=range(len(stage_data_ridges)),
                    patch_artist=True, widths=0.3, showfliers=False,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], stage_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, d in enumerate(stage_data_ridges):
        med = np.median(d)
        ax.text(i, med + 0.15, f'{med:.1f}', ha='center', fontsize=8,
                fontweight='bold')

    ax.set_xticks(range(len(stage_labels_list)))
    ax.set_xticklabels(stage_labels_list, fontsize=10)
    ax.set_ylabel('Active Ridges per Window', fontsize=10)
    ax.set_title('A) Active Ridges by Sleep Stage', fontsize=11,
                 fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')

    # ── Panel B: Max prominence by stage ──
    ax = fig.add_subplot(gs[0, 1])
    stage_data_prom = []
    for sc in STAGE_ORDER:
        sv = cre[cre['stage_code'] == sc]
        if len(sv) > 0:
            vals = sv['max_prominence'].values
            stage_data_prom.append(vals[vals > 0])

    vp = ax.violinplot(stage_data_prom, positions=range(len(stage_data_prom)),
                       showmedians=False, showextrema=False)
    for body, color in zip(vp['bodies'], stage_colors_list):
        body.set_facecolor(color)
        body.set_alpha(0.4)

    bp = ax.boxplot(stage_data_prom, positions=range(len(stage_data_prom)),
                    patch_artist=True, widths=0.3, showfliers=False,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], stage_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, d in enumerate(stage_data_prom):
        med = np.median(d)
        ax.text(i, med + 0.3, f'{med:.1f}', ha='center', fontsize=8,
                fontweight='bold')

    ax.set_xticks(range(len(stage_labels_list)))
    ax.set_xticklabels(stage_labels_list, fontsize=10)
    ax.set_ylabel('Max Ridge Prominence (×floor)', fontsize=10)
    ax.set_title('B) Ridge Prominence (nonzero only)', fontsize=11,
                 fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')

    # ── Panel C: Per-subject active-ridge heatmap ──
    ax = fig.add_subplot(gs[0, 2])
    subjects = sorted(cre['subject'].unique())
    heatmap = np.full((len(subjects), len(STAGE_ORDER)), np.nan)
    for si, subj in enumerate(subjects):
        sub = cre[cre['subject'] == subj]
        for sj, sc in enumerate(STAGE_ORDER):
            sv = sub[sub['stage_code'] == sc]
            if len(sv) > 10:
                heatmap[si, sj] = sv['n_active_ridges'].mean()

    im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_LABELS[c] for c in STAGE_ORDER], fontsize=9)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f'S{i+1}' for i in range(len(subjects))], fontsize=9)
    ax.set_title('C) Mean Active Ridges per Subject', fontsize=11,
                 fontweight='bold')

    for si in range(len(subjects)):
        for sj in range(len(STAGE_ORDER)):
            val = heatmap[si, sj]
            if np.isfinite(val):
                ax.text(sj, si, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, color='black' if val < np.nanpercentile(heatmap, 70) else 'white')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Mean Active Ridges')

    # ── Panel D: N3 vs non-N3 ROC (active ridges) ──
    ax = fig.add_subplot(gs[1, 0])
    y_true = (cre['stage_code'] == 1).astype(int).values

    metrics = {}
    for feat, label, color in [
        ('n_active_ridges', 'Active ridges', '#E74C3C'),
        ('max_prominence', 'Max prominence', '#3498DB'),
        ('prominence_score', 'Prom. score', '#2ECC71'),
        ('n_strong_ridges', 'Strong ridges', '#9B59B6'),
    ]:
        y_score = cre[feat].values
        auc_pos = roc_auc_score(y_true, y_score)
        auc_neg = roc_auc_score(y_true, -y_score)
        if auc_neg > auc_pos:
            auc, y_s = auc_neg, -y_score
            direction = 'low'
        else:
            auc, y_s = auc_pos, y_score
            direction = 'high'
        fpr, tpr, _ = roc_curve(y_true, y_s)
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f'{label} ({auc:.3f}, {direction})')
        metrics[feat] = auc

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('D) N3 Detection ROC (single features)', fontsize=11,
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    n_n3 = y_true.sum()
    n_total = len(y_true)
    ax.text(0.05, 0.55,
            f'N3: {n_n3} / {n_total}\n({100*n_n3/n_total:.1f}%)',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(fc='white', alpha=0.8, pad=3))

    # ── Panel E: Per-subject AUC direction check ──
    ax = fig.add_subplot(gs[1, 1])
    subj_aucs = np.full((len(subjects), 2), np.nan)
    subj_dirs = np.full((len(subjects), 2), '', dtype=object)
    feat_names = ['n_active_ridges', 'max_prominence']
    for si, subj in enumerate(subjects):
        sub = cre[cre['subject'] == subj]
        yt = (sub['stage_code'] == 1).astype(int).values
        if yt.sum() < 5 or (1 - yt).sum() < 5:
            continue
        for fi, feat in enumerate(feat_names):
            ys = sub[feat].values
            auc_p = roc_auc_score(yt, ys)
            auc_n = roc_auc_score(yt, -ys)
            subj_aucs[si, fi] = max(auc_p, auc_n)
            subj_dirs[si, fi] = 'N3↓' if auc_n > auc_p else 'N3↑'

    x = np.arange(len(subjects))
    w = 0.35
    bars1 = ax.bar(x - w/2, subj_aucs[:, 0], w, color='#E74C3C', alpha=0.7,
                   label='Active ridges')
    bars2 = ax.bar(x + w/2, subj_aucs[:, 1], w, color='#3498DB', alpha=0.7,
                   label='Max prominence')

    for i in range(len(subjects)):
        for fi, bars in enumerate([bars1, bars2]):
            if np.isfinite(subj_aucs[i, fi]):
                offset = -w/2 if fi == 0 else w/2
                ax.text(i + offset, subj_aucs[i, fi] + 0.01,
                        subj_dirs[i, fi], ha='center', fontsize=7,
                        fontweight='bold',
                        color='red' if 'N3↑' in str(subj_dirs[i, fi]) else 'blue')

    ax.axhline(0.5, color='gray', ls='--', alpha=0.5, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(len(subjects))], fontsize=9)
    ax.set_ylabel('AUC (best direction)', fontsize=10)
    ax.set_title('E) Per-Subject AUC + Direction', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 0.85)
    ax.grid(True, alpha=0.15, axis='y')

    # ── Panel F: Kruskal-Wallis stats table ──
    from scipy.stats import kruskal, mannwhitneyu
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    stats_rows = []
    for feat, label in [
        ('n_active_ridges', 'Active ridges'),
        ('max_prominence', 'Max prominence'),
        ('prominence_score', 'Prom. score'),
    ]:
        groups = [cre[cre['stage_code'] == sc][feat].values
                  for sc in STAGE_ORDER
                  if len(cre[cre['stage_code'] == sc]) > 10]
        h, p_kw = kruskal(*groups)
        n3_vals = cre[cre['stage_code'] == 1][feat].values
        non_n3_vals = cre[cre['stage_code'] != 1][feat].values
        _, p_mw = mannwhitneyu(n3_vals, non_n3_vals, alternative='two-sided')
        stats_rows.append([label, f'{h:.1f}', f'{p_kw:.2e}', f'{p_mw:.2e}'])

    tbl = ax.table(
        cellText=stats_rows,
        colLabels=['Feature', 'KW H', 'KW p', 'MW-U p\n(N3 vs rest)'],
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
    ax.set_title('F) Statistical Tests', fontsize=11, fontweight='bold',
                 pad=20)

    fig.suptitle('Harmonic Ridge Quantification — CRE Channel (12 sessions pooled)',
                 fontsize=14, fontweight='bold')
    fig.savefig(FIG_DIR / 'paper_quantification.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    best_auc = max(metrics.values())
    best_feat = max(metrics, key=metrics.get)
    print(f'  Quantification figure -> paper_quantification.png')
    print(f'    Best single-feature N3 AUC = {best_auc:.3f} ({best_feat})')
    for feat, auc in metrics.items():
        print(f'      {feat}: {auc:.3f}')
    return best_auc


# ── Part C: Stage 4 — LOSO N3 classifier ────────────────────────────────────

def run_loso_classifier(df):
    """LOSO N3 binary classifier from ridge features."""
    cre = df[(df['channel'] == CHANNEL) & (~df['motion_masked'])
             & (df['stage_code'] >= 0)].copy()

    feature_cols = ['prominence_score', 'max_prominence',
                    'n_strong_ridges', 'n_active_ridges']
    X = cre[feature_cols].values.astype(float)
    y = (cre['stage_code'] == 1).astype(int).values
    subjects = cre['subject'].values
    unique_subjects = sorted(set(subjects))

    results = []
    all_y_true = []
    all_y_prob = []

    print(f'  LOSO N3 classifier ({len(unique_subjects)} folds, '
          f'{len(feature_cols)} features)')

    for test_subj in unique_subjects:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if y_train.sum() < 10 or y_test.sum() < 5:
            print(f'    {test_subj}: skipped (too few N3 epochs)')
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr, y_train)
        y_prob = rf.predict_proba(X_te)[:, 1]
        y_pred = rf.predict(X_te)

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        subj_idx = unique_subjects.index(test_subj) + 1
        results.append({
            'subject': test_subj,
            'subject_label': f'S{subj_idx}',
            'n_test': len(y_test),
            'n_n3': int(y_test.sum()),
            'auc': auc, 'f1': f1, 'precision': prec, 'recall': rec,
        })
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        print(f'    {test_subj} (S{subj_idx}): AUC={auc:.3f}  '
              f'F1={f1:.3f}  n_N3={y_test.sum()}/{len(y_test)}')

    if not results:
        print('  No valid LOSO folds — skipping Stage 4 figure.')
        return

    res_df = pd.DataFrame(results)
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    pooled_auc = roc_auc_score(all_y_true, all_y_prob)

    # ── Feature importance (train on all data) ──
    scaler_all = StandardScaler()
    X_all = scaler_all.fit_transform(X)
    rf_all = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    rf_all.fit(X_all, y)
    importances = rf_all.feature_importances_

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={'width_ratios': [1.0, 0.8, 0.8]})

    # Panel A: Per-fold ROC curves
    ax = axes[0]
    for _, row in res_df.iterrows():
        subj = row['subject']
        mask = subjects == subj
        yt = y[mask]
        yp = all_y_prob[all_y_true.shape[0] - sum(subjects[mask] == subj):]
    # Recompute per-fold ROC properly
    ax = axes[0]
    offset = 0
    for _, row in res_df.iterrows():
        n = row['n_test']
        yt = all_y_true[offset:offset + n]
        yp = all_y_prob[offset:offset + n]
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=1.5, alpha=0.6,
                label=f'{row["subject_label"]} ({row["auc"]:.2f})')
        offset += n

    fpr_pool, tpr_pool, _ = roc_curve(all_y_true, all_y_prob)
    ax.plot(fpr_pool, tpr_pool, 'k-', lw=2.5,
            label=f'Pooled ({pooled_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('A) LOSO ROC — N3 Detection', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    # Panel B: Per-fold metrics table
    ax = axes[1]
    ax.axis('off')
    table_data = []
    for _, row in res_df.iterrows():
        table_data.append([
            row['subject_label'],
            f'{row["auc"]:.3f}',
            f'{row["f1"]:.3f}',
            f'{row["precision"]:.3f}',
            f'{row["recall"]:.3f}',
            f'{row["n_n3"]}/{row["n_test"]}',
        ])
    table_data.append([
        'Mean',
        f'{res_df["auc"].mean():.3f}',
        f'{res_df["f1"].mean():.3f}',
        f'{res_df["precision"].mean():.3f}',
        f'{res_df["recall"].mean():.3f}',
        f'{int(res_df["n_n3"].sum())}/{int(res_df["n_test"].sum())}',
    ])
    tbl = ax.table(
        cellText=table_data,
        colLabels=['Fold', 'AUC', 'F1', 'Prec', 'Recall', 'N3/Total'],
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif i == len(table_data):
            cell.set_facecolor('#ECF0F1')
            cell.set_text_props(fontweight='bold')
    ax.set_title('B) Per-Fold Metrics', fontsize=11, fontweight='bold',
                 pad=20)

    # Panel C: Feature importance
    ax = axes[2]
    feat_names = ['Prom.\nScore', 'Max\nProm.', 'Strong\nRidges', 'Active\nRidges']
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']
    bars = ax.barh(range(len(feat_names)), importances, color=colors, alpha=0.8)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=9)
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_title('C) Feature Importance', fontsize=11, fontweight='bold')
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{imp:.3f}', va='center', fontsize=9)
    ax.grid(True, alpha=0.15, axis='x')

    fig.suptitle(f'Stage 4: LOSO N3 Classification from Ridge Features '
                 f'(pooled AUC = {pooled_auc:.3f})',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = FIG_DIR / 'paper_n3_loso.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    csv_out = REPORT_DIR / 'paper_n3_loso_metrics.csv'
    res_df.to_csv(csv_out, index=False)
    print(f'  LOSO figure  -> {out.name}')
    print(f'  LOSO metrics -> {csv_out.name}')
    print(f'  Pooled AUC = {pooled_auc:.3f}, mean AUC = {res_df["auc"].mean():.3f}')

    return pooled_auc, res_df


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    skip_overlay = '--skip-overlay' in sys.argv
    session_filter = [a for a in sys.argv[1:] if not a.startswith('--')]

    # ── Part B: Pooled quantification (fast, from cached parquet) ──
    print('\n' + '=' * 60)
    print('PART B: Pooled Ridge Quantification')
    print('=' * 60)
    if not PARQUET_PATH.exists():
        print(f'  ERROR: {PARQUET_PATH} not found. Run run_ridge_overlay.py first.')
        sys.exit(1)
    all_df = pd.read_parquet(PARQUET_PATH)
    print(f'  Loaded {len(all_df)} rows from cached parquet')
    n3_auc = quantify_pooled(all_df)

    # ── Part C: LOSO N3 classifier (fast, from cached parquet) ──
    print('\n' + '=' * 60)
    print('PART C: Stage 4 — LOSO N3 Classifier')
    print('=' * 60)
    loso_result = run_loso_classifier(all_df)

    # ── Part A: Per-session spectrogram overlays (slow, re-runs detection) ──
    if not skip_overlay:
        print('\n' + '=' * 60)
        print('PART A: Per-Session Spectrogram + Ridge Overlays')
        print('=' * 60)
        sessions = load_all_sessions(with_sleep_profiles=True)
        if session_filter:
            sessions = [s for s in sessions if s.label in session_filter]
        print(f'  Processing {len(sessions)} sessions...')
        generate_overlays(sessions)
    else:
        print('\n  Skipping overlay generation (--skip-overlay)')

    # ── Summary ──
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  N3 detection AUC (prominence alone): {n3_auc:.3f}')
    if loso_result:
        pooled_auc, res_df = loso_result
        print(f'  LOSO N3 AUC (RF, 4 ridge features): {pooled_auc:.3f}')
        print(f'  LOSO N3 mean AUC:  {res_df["auc"].mean():.3f}')
        print(f'  LOSO N3 mean F1:   {res_df["f1"].mean():.3f}')
    print(f'\n  Figures in: {FIG_DIR}')
    print(f'  Reports in: {REPORT_DIR}')
    print('\nDone.')
