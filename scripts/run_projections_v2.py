"""
Pure-CAP projection pipeline v2.

Features derived ONLY from CLE, CRE, CH, accelerometer — no k-calibration,
no PSG-derived rates. ~35 features per 60s window, 30s step.

Methods: PCA, UMAP (unsupervised + supervised), t-SNE, PHATE.
Coloring: sleep stage, time of night, thorax RMS, apnea, resp/cardiac band power.

Outputs to reports/projections/ (interactive HTML + static PNG + CSV summary).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import welch, coherence
from scipy.stats import kruskal, spearmanr, kurtosis as sp_kurtosis, skew as sp_skew

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import umap
import phate
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import load_session, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.config import (
    FS, DELTA_SUB_BANDS,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, APNEA_LABELS, APNEA_COLORS,
)
from sleep_monitor.filters import bandpass
from sleep_monitor.rates import rate_acf, detect_peaks
from sleep_monitor.loader import load_sleep_profile, load_apnea_events

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9})

REPORT_DIR = ROOT / 'reports' / 'projections'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_NAMES = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_NAME_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

CAP_BANDS = {
    'infra_slow': (0.0, 0.5),
    'SO':         (0.5, 1.0),
    'delta_low':  (1.0, 2.0),
    'delta_high': (2.0, 4.0),
    'resp':       (RESP_LO, RESP_HI),
    'cardiac':    (CARD_LO, CARD_HI),
}


# ── Feature extraction (pure CAP) ───────────────────────────────────────────

def _hjorth(seg):
    d1 = np.diff(seg)
    d2 = np.diff(d1)
    var0 = np.var(seg)
    var1 = np.var(d1)
    var2 = np.var(d2)
    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-20))
    complexity = np.sqrt(var2 / (var1 + 1e-20)) / (mobility + 1e-20)
    return activity, mobility, complexity


def _peak_freq(sig, fs, f_lo, f_hi, nperseg):
    freqs, psd = welch(sig - sig.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any() or psd[mask].sum() < 1e-20:
        return np.nan
    return float(freqs[mask][np.argmax(psd[mask])])


def _band_snr(sig, fs, f_lo, f_hi, nperseg):
    freqs, psd = welch(sig - sig.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    total_mask = freqs > 0
    if not band_mask.any():
        return np.nan
    band_power = np.mean(psd[band_mask])
    floor_mask = total_mask & ~band_mask
    if not floor_mask.any():
        return np.nan
    floor_power = np.mean(psd[floor_mask])
    return float(10 * np.log10((band_power + 1e-20) / (floor_power + 1e-20)))


FEAT_COLS = []

def extract_features(session, win_sec=60.0, step_sec=30.0):
    global FEAT_COLS
    fs = session.fs
    win_n = int(win_sec * fs)
    step_n = int(step_sec * fs)
    n_total = session.n_samples
    nperseg = int(4.0 * fs)

    cap = session.cap
    cle = cap['CLE'].astype(np.float64)
    cre = cap['CRE'].astype(np.float64)
    ch  = cap['CH'].astype(np.float64)
    diff = cle - cre
    acc = cap['acc_mag'].astype(np.float64)
    thorax = session.psg['Thorax'].astype(np.float64)
    t_hr = session.time_hr

    sp = session.sleep_profile
    ep_t_hr, ep_codes = sp['t_ep_hr'], sp['codes']

    channels = {'diff': diff, 'CLE': cle, 'CRE': cre, 'CH': ch}

    rows = []
    starts = range(0, n_total - win_n + 1, step_n)

    for i, s0 in enumerate(starts):
        s1 = s0 + win_n
        idx = slice(s0, s1)
        t_center = t_hr[s0 + win_n // 2]

        ep_mask = (ep_t_hr >= t_hr[s0] - PSG_EPOCH_SEC / 3600) & (ep_t_hr <= t_hr[s1 - 1])
        if ep_mask.any():
            codes = ep_codes[ep_mask]
            valid = codes[codes >= 0]
            stage_code = int(np.bincount(valid.astype(int)).argmax()) if len(valid) > 0 else -1
        else:
            stage_code = -1

        apnea_code = int(session.apnea_at(np.array([t_center]))[0]) if session.apnea_events else 0

        thorax_seg = thorax[idx]
        thorax_rms = float(np.sqrt(np.mean((thorax_seg - thorax_seg.mean()) ** 2)))

        row = {
            't_hr': float(t_center),
            'stage_code': stage_code,
            'stage_label': STAGE_LABELS.get(stage_code, '?'),
            'apnea_code': apnea_code,
            'apnea_label': APNEA_LABELS.get(apnea_code, 'Normal'),
            'thorax_rms': thorax_rms,
        }

        # --- Per-channel spectral band powers (diff, CLE, CRE, CH) ---
        for ch_name, ch_sig in channels.items():
            seg = ch_sig[idx]
            seg_c = seg - seg.mean()
            freqs_w, psd_w = welch(seg_c, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            freq_df = freqs_w[1] - freqs_w[0]
            total_mask = (freqs_w >= 0.0) & (freqs_w <= 4.0)
            tp = np.trapezoid(psd_w[total_mask], dx=freq_df)

            for bname in ['infra_slow', 'SO', 'delta_low', 'delta_high']:
                flo, fhi = CAP_BANDS[bname]
                bmask = (freqs_w >= flo) & (freqs_w <= fhi)
                bp = np.trapezoid(psd_w[bmask], dx=freq_df)
                row[f'{ch_name}_{bname}'] = bp / tp if tp > 0 else 0.0

        # --- Diff-channel detailed features ---
        seg_diff = diff[idx]
        seg_diff_c = seg_diff - seg_diff.mean()
        freqs_w, psd_w = welch(seg_diff_c, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

        psd_norm = psd_w / (psd_w.sum() + 1e-20)
        psd_pos = psd_norm[psd_norm > 0]
        row['diff_spectral_entropy'] = float(-np.sum(psd_pos * np.log2(psd_pos)))

        row['diff_rms'] = float(np.sqrt(np.mean(seg_diff_c ** 2)))
        row['diff_kurtosis'] = float(sp_kurtosis(seg_diff_c, fisher=True))
        row['diff_skewness'] = float(sp_skew(seg_diff_c))

        activity, mobility, complexity = _hjorth(seg_diff_c)
        row['diff_hjorth_mobility'] = float(mobility)
        row['diff_hjorth_complexity'] = float(complexity)

        # --- Resp band features (no k) ---
        bp_resp = bandpass(seg_diff, RESP_LO, RESP_HI, fs)
        row['resp_band_power'] = float(np.mean(bp_resp ** 2))
        row['resp_peak_freq'] = _peak_freq(seg_diff, fs, RESP_LO, RESP_HI, nperseg)
        row['resp_rate_acf'] = rate_acf(bp_resp, RESP_LO, RESP_HI, fs)
        row['resp_snr'] = _band_snr(seg_diff, fs, RESP_LO, RESP_HI, nperseg)

        resp_env = np.abs(bp_resp)
        row['resp_amp_cv'] = float(np.std(resp_env) / (np.mean(resp_env) + 1e-10))

        peaks_idx = detect_peaks(bp_resp, RESP_LO, RESP_HI, fs, prom_factor=0.05)
        if len(peaks_idx) >= 3:
            intervals = np.diff(peaks_idx) / fs
            row['breath_interval_cv'] = float(np.std(intervals) / (np.mean(intervals) + 1e-10))
        else:
            row['breath_interval_cv'] = np.nan

        # --- Cardiac band features (no k) ---
        bp_card = bandpass(seg_diff, CARD_LO, CARD_HI, fs)
        row['cardiac_band_power'] = float(np.mean(bp_card ** 2))
        row['cardiac_peak_freq'] = _peak_freq(seg_diff, fs, CARD_LO, CARD_HI, nperseg)
        row['cardiac_rate_acf'] = rate_acf(bp_card, CARD_LO, CARD_HI, fs)
        row['cardiac_snr'] = _band_snr(seg_diff, fs, CARD_LO, CARD_HI, nperseg)

        # --- Cross-channel coherence ---
        seg_cle = cle[idx]
        seg_cre = cre[idx]
        freqs_c, coh = coherence(seg_cle, seg_cre, fs=fs, nperseg=nperseg,
                                 noverlap=nperseg // 2)
        resp_mask = (freqs_c >= RESP_LO) & (freqs_c <= RESP_HI)
        card_mask = (freqs_c >= CARD_LO) & (freqs_c <= CARD_HI)
        row['coh_resp'] = float(np.mean(coh[resp_mask])) if resp_mask.any() else np.nan
        row['coh_cardiac'] = float(np.mean(coh[card_mask])) if card_mask.any() else np.nan

        # --- Accelerometer ---
        seg_acc = acc[idx]
        row['acc_rms'] = float(np.sqrt(np.mean((seg_acc - seg_acc.mean()) ** 2)))
        acc_bp_resp = bandpass(seg_acc, RESP_LO, RESP_HI, fs)
        row['acc_resp_power'] = float(np.mean(acc_bp_resp ** 2))

        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f'    {i + 1}/{len(starts)} windows...')

    df = pd.DataFrame(rows)

    meta_cols = {'t_hr', 'stage_code', 'stage_label', 'apnea_code', 'apnea_label', 'thorax_rms'}
    FEAT_COLS = [c for c in df.columns if c not in meta_cols]

    return df


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _make_3d_scatter(X_emb, colors, colorscale, cbar_title, names, opacities,
                     sizes, texts, title, filepath, is_categorical=False):
    fig = go.Figure()
    if is_categorical:
        for name, mask, color, opacity, size in zip(names, colors, colorscale,
                                                     opacities, sizes):
            if not mask.any():
                continue
            fig.add_trace(go.Scatter3d(
                x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
                mode='markers',
                marker=dict(size=size, color=color, opacity=opacity),
                name=name,
                text=[texts[j] for j in np.where(mask)[0]],
                hoverinfo='text+name',
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
            mode='markers',
            marker=dict(size=2.5, color=colors, colorscale=colorscale,
                        opacity=0.6, colorbar=dict(title=cbar_title)),
            text=texts,
            hoverinfo='text',
        ))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
                      legend=dict(itemsizing='constant'))
    fig.write_html(str(filepath))


def plot_by_stage(X_emb, df, stage_codes, title, filepath):
    texts = [f't={t:.2f}h' for t in df['t_hr']]
    masks = [stage_codes == sc for sc in STAGE_ORDER]
    colors_cat = [STAGE_COLORS[sc] for sc in STAGE_ORDER]
    names = [STAGE_NAMES[sc] for sc in STAGE_ORDER]
    _make_3d_scatter(X_emb, masks, colors_cat, '', names,
                     [0.6]*5, [2.5]*5, texts, title, filepath, is_categorical=True)


def plot_by_continuous(X_emb, df, values, cbar_title, colorscale, title, filepath):
    texts = [f't={t:.2f}h | {sl} | {cbar_title}={v:.3f}'
             for t, sl, v in zip(df['t_hr'], df['stage_label'], values)]
    _make_3d_scatter(X_emb, values, colorscale, cbar_title, None,
                     None, None, texts, title, filepath, is_categorical=False)


def plot_by_apnea(X_emb, df, title, filepath):
    texts = [f't={t:.2f}h | {sl}' for t, sl in zip(df['t_hr'], df['stage_label'])]
    fig = go.Figure()
    for acode, alabel in APNEA_LABELS.items():
        mask = df['apnea_code'].values == acode
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=2.5 if acode == 0 else 4.0,
                        color=APNEA_COLORS[acode],
                        opacity=0.4 if acode == 0 else 0.8),
            name=f'{alabel} (n={mask.sum()})',
            text=[texts[j] for j in np.where(mask)[0]],
            hoverinfo='text+name',
        ))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'))
    fig.write_html(str(filepath))


def plot_trajectory(X_emb, df, stage_codes, title, filepath):
    """3D scatter with lines connecting consecutive points — shows overnight trajectory."""
    t_hr = df['t_hr'].values
    texts = [f't={t:.2f}h | {sl}' for t, sl in zip(df['t_hr'], df['stage_label'])]
    stage_colors_arr = [STAGE_COLORS.get(sc, '#999999') for sc in stage_codes]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
        mode='lines',
        line=dict(color=t_hr, colorscale='Viridis', width=1.5),
        opacity=0.3,
        hoverinfo='skip',
        showlegend=False,
    ))
    for sc in STAGE_ORDER:
        mask = stage_codes == sc
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=3, color=STAGE_COLORS[sc], opacity=0.7),
            name=STAGE_NAMES[sc],
            text=[texts[j] for j in np.where(mask)[0]],
            hoverinfo='text+name',
        ))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
                      legend=dict(itemsizing='constant'))
    fig.write_html(str(filepath))


def plot_static_panels(X_3d, codes, title, filepath, elevs=[20, 60], azims=[45, 135, 225]):
    ncols, nrows = len(azims), len(elevs)
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    for ri, elev in enumerate(elevs):
        for ci, azim in enumerate(azims):
            ax = fig.add_subplot(nrows, ncols, ri * ncols + ci + 1, projection='3d')
            for sc in STAGE_ORDER:
                mask = codes == sc
                if not mask.any():
                    continue
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                           c=STAGE_COLORS[sc], s=4, alpha=0.4, label=STAGE_NAMES[sc],
                           edgecolors='none')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('D1', fontsize=7)
            ax.set_ylabel('D2', fontsize=7)
            ax.set_zlabel('D3', fontsize=7)
            ax.set_title(f'elev={elev}, azim={azim}', fontsize=8)
            ax.tick_params(labelsize=6)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=STAGE_COLORS[sc],
                      markersize=7, label=STAGE_NAMES[sc]) for sc in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    plt.suptitle(title, fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Embeddings ───────────────────────────────────────────────────────────────

def compute_embeddings(X, stage_codes):
    embeddings = {}

    # PCA
    pca = PCA()
    X_pca_full = pca.fit_transform(X)
    embeddings['PCA'] = {'3d': X_pca_full[:, :3], 'full': X_pca_full, 'model': pca}

    # UMAP unsupervised
    for nn in [15, 30, 50]:
        reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
        embeddings[f'UMAP_nn{nn}'] = {'3d': reducer.fit_transform(X)}

    # UMAP supervised (stage labels guide the embedding)
    valid_mask = stage_codes >= 0
    for nn in [15, 30]:
        reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
        X_sup = reducer.fit_transform(X, y=stage_codes)
        embeddings[f'UMAP_sup_nn{nn}'] = {'3d': X_sup}

    # t-SNE
    for perp in [15, 30, 50]:
        tsne = TSNE(n_components=3, perplexity=perp, random_state=42,
                    max_iter=1000, init='pca', learning_rate='auto')
        embeddings[f'tSNE_p{perp}'] = {'3d': tsne.fit_transform(X)}

    # PHATE
    for knn in [10, 30]:
        phate_op = phate.PHATE(n_components=3, knn=knn, random_state=42, verbose=False)
        embeddings[f'PHATE_k{knn}'] = {'3d': phate_op.fit_transform(X)}

    return embeddings


# ── All plots for one embedding ──────────────────────────────────────────────

def save_all_plots(X_emb, df, stage_codes, thorax_rms, label, method_name, out_dir):
    pfx = f'{label}_{method_name}'

    plot_by_stage(X_emb, df, stage_codes,
                  f'{method_name} — {label} — Stage', out_dir / f'{pfx}_stage.html')
    plot_by_continuous(X_emb, df, df['t_hr'].values, 'Hours', 'Viridis',
                       f'{method_name} — {label} — Time', out_dir / f'{pfx}_time.html')
    plot_by_apnea(X_emb, df,
                  f'{method_name} — {label} — Apnea', out_dir / f'{pfx}_apnea.html')
    plot_by_continuous(X_emb, df, thorax_rms, 'Thorax RMS', 'Plasma',
                       f'{method_name} — {label} — Thorax', out_dir / f'{pfx}_thorax.html')

    if 'resp_band_power' in df.columns:
        plot_by_continuous(X_emb, df, df['resp_band_power'].values, 'Resp Power', 'Blues',
                           f'{method_name} — {label} — Resp Power', out_dir / f'{pfx}_resp_power.html')
    if 'cardiac_band_power' in df.columns:
        plot_by_continuous(X_emb, df, df['cardiac_band_power'].values, 'Cardiac Power', 'Reds',
                           f'{method_name} — {label} — Cardiac Power', out_dir / f'{pfx}_cardiac_power.html')

    plot_trajectory(X_emb, df, stage_codes,
                    f'{method_name} — {label} — Trajectory', out_dir / f'{pfx}_trajectory.html')


# ── Main pipeline per session ────────────────────────────────────────────────

def run_session(sess_idx):
    print(f'\n{"=" * 60}')
    print(f'Loading session index {sess_idx}...')
    sess = load_session(sess_idx, with_profile=True, with_apnea=True)
    label = sess.label
    print(f'{label}: {sess.duration_hr:.2f} hr, {sess.n_samples} samples')

    sess_dir = REPORT_DIR / label
    sess_dir.mkdir(parents=True, exist_ok=True)

    print(f'  Extracting pure-CAP features...')
    df = extract_features(sess)
    df = df[df['stage_code'] >= 0].reset_index(drop=True)
    print(f'  {len(df)} windows after dropping unknown stages')
    print(f'  Stages: {dict(df["stage_label"].value_counts())}')
    print(f'  Apnea:  {dict(df["apnea_label"].value_counts())}')
    print(f'  Features: {len(FEAT_COLS)}')

    X_raw = df[FEAT_COLS].values.copy()
    X_raw[~np.isfinite(X_raw)] = np.nan
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X_raw[:, j] = col

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    stage_codes = df['stage_code'].values
    thorax_rms = df['thorax_rms'].values

    # --- Kruskal-Wallis feature importance ---
    print(f'\n  Kruskal-Wallis (top 10):')
    kw_results = []
    for j, fname in enumerate(FEAT_COLS):
        present = [sc for sc in STAGE_ORDER if (stage_codes == sc).sum() > 0]
        groups = [X_raw[stage_codes == sc, j] for sc in present]
        if len(groups) >= 2:
            try:
                stat, pval = kruskal(*groups)
                kw_results.append({'feature': fname, 'H': stat, 'p': pval})
            except ValueError:
                pass
    kw_df = pd.DataFrame(kw_results).sort_values('H', ascending=False).reset_index(drop=True)
    for _, r in kw_df.head(10).iterrows():
        print(f'    {r["feature"]:28s} H={r["H"]:8.1f}  p={r["p"]:.1e}')

    # --- Feature boxplots ---
    n_feat = len(FEAT_COLS)
    ncols = 5
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.ravel()
    for j, fname in enumerate(FEAT_COLS):
        ax = axes_flat[j]
        present = [sc for sc in STAGE_ORDER if (stage_codes == sc).sum() > 0]
        groups = [X_raw[stage_codes == sc, j] for sc in present]
        labels_box = [STAGE_NAMES[sc] for sc in present]
        bplot = ax.boxplot(groups, labels=labels_box, patch_artist=True,
                           flierprops={'markersize': 1.5, 'alpha': 0.2}, widths=0.6)
        for patch, sc in zip(bplot['boxes'], present):
            patch.set_facecolor(STAGE_COLORS[sc])
            patch.set_alpha(0.6)
        kw_row = kw_df[kw_df['feature'] == fname]
        if len(kw_row):
            ax.set_title(f'{fname}\nH={kw_row.iloc[0]["H"]:.0f}', fontsize=6)
        else:
            ax.set_title(fname, fontsize=6)
        ax.tick_params(axis='x', labelsize=6, rotation=45)
        ax.tick_params(axis='y', labelsize=5)
        ax.grid(True, alpha=0.2, axis='y')
    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f'Feature Distributions by Stage — {label} ({n_feat} pure-CAP features)', fontsize=11)
    plt.tight_layout()
    plt.savefig(sess_dir / f'{label}_feature_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- PCA diagnostics ---
    print(f'\n  Running PCA...')
    pca = PCA()
    X_pca_full = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    print(f'    Top 3 PCs: {evr[0]*100:.1f}%, {evr[1]*100:.1f}%, {evr[2]*100:.1f}%  |  {n95} PCs for 95%')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(evr) + 1), evr * 100, color='#3498DB', alpha=0.8)
    axes[0].set_xlabel('PC')
    axes[0].set_ylabel('Variance %')
    axes[0].set_title('Scree Plot')
    axes[1].plot(range(1, len(evr) + 1), cumvar * 100, 'o-', color='#E74C3C', ms=4)
    axes[1].axhline(95, ls='--', color='gray', alpha=0.5)
    axes[1].set_xlabel('PCs')
    axes[1].set_ylabel('Cumulative %')
    axes[1].set_title('Cumulative Variance')
    plt.suptitle(f'PCA — {label}', fontsize=11)
    plt.tight_layout()
    plt.savefig(sess_dir / f'{label}_pca_scree.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PCA loadings
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    feat_names = np.array(FEAT_COLS)
    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        order = np.argsort(np.abs(loadings))[::-1]
        top_n = min(15, len(loadings))
        top_idx = order[:top_n]
        colors = ['#E74C3C' if loadings[j] > 0 else '#3498DB' for j in top_idx]
        ax.barh(range(top_n), loadings[top_idx], color=colors, alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_names[top_idx], fontsize=7)
        ax.set_title(f'PC{i+1} ({evr[i]*100:.1f}%)', fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis='x')
    plt.suptitle(f'PCA Loadings — {label}', fontsize=11)
    plt.tight_layout()
    plt.savefig(sess_dir / f'{label}_pca_loadings.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Compute all embeddings ---
    print(f'  Computing embeddings (UMAP, supervised UMAP, t-SNE, PHATE)...')
    embeddings = compute_embeddings(X, stage_codes)

    # --- Save plots for key methods ---
    key_methods = ['UMAP_nn30', 'UMAP_sup_nn30', 'tSNE_p30', 'PHATE_k30', 'PCA']
    for method_name in key_methods:
        if method_name in embeddings:
            print(f'    Saving plots for {method_name}...')
            save_all_plots(embeddings[method_name]['3d'], df, stage_codes, thorax_rms,
                           label, method_name, sess_dir)

    # --- Static panels for key methods ---
    for method_name in ['UMAP_nn30', 'UMAP_sup_nn30', 'PHATE_k30']:
        if method_name in embeddings:
            plot_static_panels(embeddings[method_name]['3d'], stage_codes,
                               f'{method_name} — {label}',
                               sess_dir / f'{label}_{method_name}_static.png')

    # --- Silhouette scores ---
    remap_3c = {4: 0, 3: 1, 2: 1, 1: 1, 0: 2}  # Wake=0, NREM=1, REM=2
    codes_3c = np.array([remap_3c[c] for c in stage_codes])

    sil_results = []
    for method_name, emb_data in embeddings.items():
        X_emb = emb_data['3d']
        sil_5 = silhouette_score(X_emb, stage_codes)
        sil_3 = silhouette_score(X_emb, codes_3c)
        sil_samples_arr = silhouette_samples(X_emb, stage_codes)
        row = {'method': method_name, 'sil_5class': sil_5, 'sil_3class': sil_3}
        for sc in STAGE_ORDER:
            mask = stage_codes == sc
            if mask.any():
                row[STAGE_NAMES[sc]] = float(np.mean(sil_samples_arr[mask]))
        sil_results.append(row)

    sil_df = pd.DataFrame(sil_results).sort_values('sil_5class', ascending=False)
    print(f'\n  Silhouette scores ({label}):')
    print(sil_df.to_string(index=False, float_format='{:.3f}'.format))
    sil_df.to_csv(sess_dir / f'{label}_silhouette_scores.csv', index=False)

    best = sil_df.iloc[0]
    print(f'\n  Best: {best["method"]}  5-class={best["sil_5class"]:.3f}  3-class={best["sil_3class"]:.3f}')

    # --- Thorax correlations ---
    print(f'\n  Thorax RMS vs features (top 5):')
    thorax_corrs = []
    for j, fname in enumerate(FEAT_COLS):
        rho, p = spearmanr(thorax_rms, X_raw[:, j])
        thorax_corrs.append((fname, rho, p))
    thorax_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for fname, rho, p in thorax_corrs[:5]:
        print(f'    {fname:28s} rho={rho:+.3f}  p={p:.1e}')

    # --- Save feature data ---
    df.to_csv(sess_dir / f'{label}_features.csv', index=False)

    n_plots = len(key_methods) * 7 + 3  # 7 HTML per method + 3 static PNGs
    print(f'\n  DONE: {label}  |  {n_plots} plots + feature CSV + silhouette CSV')
    print(f'  Output: {sess_dir}')

    return {
        'label': label,
        'n_windows': len(df),
        'n_features': len(FEAT_COLS),
        'best_method': best['method'],
        'best_5class': best['sil_5class'],
        'best_3class': best['sil_3class'],
    }


# ── Run all 12 sessions ─────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = []

    for idx in range(12):
        try:
            result = run_session(idx)
            all_results.append(result)
        except Exception as e:
            print(f'\n  ERROR on session {idx}: {e}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 70)
    print('CROSS-SESSION SUMMARY')
    print('=' * 70)
    summary_rows = []
    for r in all_results:
        print(f'  {r["label"]:6s}: {r["n_windows"]:4d} win  |  '
              f'{r["n_features"]} feat  |  '
              f'5c={r["best_5class"]:.3f}  3c={r["best_3class"]:.3f}  |  '
              f'{r["best_method"]}')
        summary_rows.append(r)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORT_DIR / 'cross_session_summary.csv', index=False)
    print(f'\nSummary saved to {REPORT_DIR / "cross_session_summary.csv"}')
    print(f'Feature list ({len(FEAT_COLS)}): {FEAT_COLS}')
