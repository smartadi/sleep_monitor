"""
Run delta sub-band 3D projections for multiple sessions.
Same analysis as notebook 11, batch mode.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import welch
from scipy.stats import kruskal, spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import umap
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import load_session, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER
from sleep_monitor.config import (
    FS, DELTA_SUB_BANDS, DELTA_SUB_COLORS,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, APNEA_LABELS, APNEA_COLORS,
)
from sleep_monitor.filters import bandpass
from sleep_monitor.rates import (
    rate_peaks_scaled_resp, rate_hilbert_scaled_cardiac,
    calibrate_k_resp, calibrate_k_cardiac, detect_peaks,
)
from sleep_monitor.loader import load_sleep_profile, load_apnea_events

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9})

PLOT_DIR = ROOT / 'notebooks' / 'plots' / 'projections'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_NAMES = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake'}
STAGE_NAME_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

FEAT_COLS = [
    'infra_slow', 'SO', 'delta_low', 'delta_high',
    'spectral_entropy', 'rms',
    'CLE_dev', 'CRE_dev', 'CH_dev',
    'resp_rate', 'resp_rate_std',
    'card_rate', 'card_rate_std',
    'acc_rms',
    'resp_amp_cv', 'breath_interval_cv',
]

NN_LIST = [10, 30, 50]
PERP_LIST = [15, 30, 50]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(session, k_resp, k_cardiac,
                     sess_mean_cle, sess_mean_cre, sess_mean_ch,
                     win_sec=60.0, step_sec=30.0):
    fs = session.fs
    win_n = int(win_sec * fs)
    step_n = int(step_sec * fs)
    n_total = session.n_samples

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
    nperseg = int(4.0 * fs)

    rows = []
    starts = range(0, n_total - win_n + 1, step_n)

    for i, s0 in enumerate(starts):
        s1 = s0 + win_n
        idx = slice(s0, s1)
        t_center = t_hr[s0 + win_n // 2]

        ep_mask = (ep_t_hr >= t_hr[s0] - PSG_EPOCH_SEC/3600) & (ep_t_hr <= t_hr[s1-1])
        if ep_mask.any():
            codes = ep_codes[ep_mask]
            valid = codes[codes >= 0]
            stage_code = int(np.bincount(valid.astype(int)).argmax()) if len(valid) > 0 else -1
        else:
            stage_code = -1

        apnea_code = int(session.apnea_at(np.array([t_center]))[0]) if session.apnea_events else 0

        thorax_seg = thorax[idx]
        thorax_rms = float(np.sqrt(np.mean((thorax_seg - thorax_seg.mean())**2)))

        row = {
            't_hr': float(t_center),
            'stage_code': stage_code,
            'stage_label': STAGE_LABELS.get(stage_code, '?'),
            'apnea_code': apnea_code,
            'apnea_label': APNEA_LABELS.get(apnea_code, 'Normal'),
            'thorax_rms': thorax_rms,
        }

        seg = diff[idx]
        seg_c = seg - seg.mean()
        freqs_w, psd_w = welch(seg_c, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        freq_df = freqs_w[1] - freqs_w[0]
        total_mask = (freqs_w >= 0.0) & (freqs_w <= 4.0)
        tp = np.trapezoid(psd_w[total_mask], dx=freq_df)

        for bname, (flo, fhi) in DELTA_SUB_BANDS.items():
            bmask = (freqs_w >= flo) & (freqs_w <= fhi)
            bp = np.trapezoid(psd_w[bmask], dx=freq_df)
            row[bname] = bp / tp if tp > 0 else 0.0

        psd_norm = psd_w / (psd_w.sum() + 1e-20)
        psd_pos = psd_norm[psd_norm > 0]
        row['spectral_entropy'] = float(-np.sum(psd_pos * np.log2(psd_pos)))
        row['rms'] = float(np.sqrt(np.mean(seg_c**2)))

        row['CLE_dev'] = float((np.mean(np.abs(cle[idx])) - sess_mean_cle) / sess_mean_cle)
        row['CRE_dev'] = float((np.mean(np.abs(cre[idx])) - sess_mean_cre) / sess_mean_cre)
        row['CH_dev']  = float((np.mean(np.abs(ch[idx]))  - sess_mean_ch)  / sess_mean_ch)

        bp_resp = bandpass(seg, RESP_LO, RESP_HI, fs)
        row['resp_rate'] = rate_peaks_scaled_resp(bp_resp, k_resp, fs=fs)

        n_sub = 4
        sub_len = win_n // n_sub
        resp_sub = []
        for si in range(n_sub):
            sub = diff[s0 + si*sub_len : s0 + (si+1)*sub_len]
            rr = rate_peaks_scaled_resp(bandpass(sub, RESP_LO, RESP_HI, fs), k_resp, fs=fs)
            if not np.isnan(rr):
                resp_sub.append(rr)
        row['resp_rate_std'] = float(np.std(resp_sub)) if len(resp_sub) >= 2 else np.nan

        bp_card = bandpass(seg, CARD_LO, CARD_HI, fs)
        row['card_rate'] = rate_hilbert_scaled_cardiac(bp_card, k_cardiac, fs=fs)

        card_sub = []
        for si in range(n_sub):
            sub = diff[s0 + si*sub_len : s0 + (si+1)*sub_len]
            cr = rate_hilbert_scaled_cardiac(bandpass(sub, CARD_LO, CARD_HI, fs), k_cardiac, fs=fs)
            if not np.isnan(cr):
                card_sub.append(cr)
        row['card_rate_std'] = float(np.std(card_sub)) if len(card_sub) >= 2 else np.nan

        row['acc_rms'] = float(np.sqrt(np.mean((acc[idx] - acc[idx].mean())**2)))

        resp_env = np.abs(bp_resp)
        row['resp_amp_cv'] = float(np.std(resp_env) / (np.mean(resp_env) + 1e-10))

        peaks_idx = detect_peaks(bp_resp, RESP_LO, RESP_HI, fs, prom_factor=0.05)
        if len(peaks_idx) >= 3:
            intervals = np.diff(peaks_idx) / fs
            row['breath_interval_cv'] = float(np.std(intervals) / (np.mean(intervals) + 1e-10))
        else:
            row['breath_interval_cv'] = np.nan

        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f'    {i+1}/{len(starts)} windows...')

    return pd.DataFrame(rows)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_3d_stage(X_emb, df, codes, title, filepath):
    fig = go.Figure()
    for sc in STAGE_ORDER:
        mask = codes == sc
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=X_emb[mask, 0], y=X_emb[mask, 1], z=X_emb[mask, 2],
            mode='markers',
            marker=dict(size=2.5, color=STAGE_COLORS[sc], opacity=0.6),
            name=STAGE_NAMES[sc],
            text=[f't={t:.2f}h' for t in df['t_hr'].values[mask]],
            hoverinfo='text+name',
        ))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'),
                      legend=dict(itemsizing='constant'))
    fig.write_html(str(filepath))


def plot_3d_time(X_emb, df, title, filepath):
    fig = go.Figure(data=[go.Scatter3d(
        x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
        mode='markers',
        marker=dict(size=2.5, color=df['t_hr'].values, colorscale='Viridis',
                    opacity=0.6, colorbar=dict(title='Hours')),
        text=[f't={t:.2f}h | {sl}' for t, sl in zip(df['t_hr'], df['stage_label'])],
        hoverinfo='text',
    )])
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'))
    fig.write_html(str(filepath))


def plot_3d_apnea(X_emb, df, title, filepath):
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
            text=[f't={t:.2f}h | {sl}' for t, sl in
                  zip(df['t_hr'].values[mask], df['stage_label'].values[mask])],
            hoverinfo='text+name',
        ))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'))
    fig.write_html(str(filepath))


def plot_3d_thorax(X_emb, df, thorax_rms, title, filepath):
    fig = go.Figure(data=[go.Scatter3d(
        x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
        mode='markers',
        marker=dict(size=2.5, color=thorax_rms, colorscale='Plasma',
                    opacity=0.6, colorbar=dict(title='Thorax RMS')),
        text=[f't={t:.2f}h | {sl} | thor={tr:.3f}' for t, sl, tr in
              zip(df['t_hr'], df['stage_label'], thorax_rms)],
        hoverinfo='text',
    )])
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title='D1', yaxis_title='D2', zaxis_title='D3'))
    fig.write_html(str(filepath))


def plot_3d_panels(X_3d, codes, title, filepath, elevs=[20, 60], azims=[45, 135, 225]):
    ncols, nrows = len(azims), len(elevs)
    fig = plt.figure(figsize=(6*ncols, 5*nrows))
    for ri, elev in enumerate(elevs):
        for ci, azim in enumerate(azims):
            ax = fig.add_subplot(nrows, ncols, ri*ncols + ci + 1, projection='3d')
            for sc in STAGE_ORDER:
                mask = codes == sc
                if not mask.any():
                    continue
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                          c=STAGE_COLORS[sc], s=4, alpha=0.4, label=STAGE_NAMES[sc],
                          edgecolors='none')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('D1', fontsize=7); ax.set_ylabel('D2', fontsize=7); ax.set_zlabel('D3', fontsize=7)
            ax.set_title(f'elev={elev}, azim={azim}', fontsize=8)
            ax.tick_params(labelsize=6)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=STAGE_COLORS[sc],
                      markersize=7, label=STAGE_NAMES[sc]) for sc in STAGE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    plt.suptitle(title, fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main pipeline per session ─────────────────────────────────────────────────

def run_session(sess_idx):
    print(f'\n{"="*60}')
    print(f'Loading session index {sess_idx}...')
    sess = load_session(sess_idx)
    sess.sleep_profile = load_sleep_profile(sess)
    sess.apnea_events = load_apnea_events(sess)
    label = sess.label
    print(f'{label}: {sess.duration_hr:.2f} hr, {sess.n_samples} samples')

    k_resp = calibrate_k_resp(sess)
    k_cardiac = calibrate_k_cardiac(sess)
    print(f'k_resp={k_resp:.3f}, k_cardiac={k_cardiac:.3f}')

    cle_full = sess.cap['CLE'].astype(np.float64)
    cre_full = sess.cap['CRE'].astype(np.float64)
    ch_full  = sess.cap['CH'].astype(np.float64)
    sess_mean_cle = np.mean(np.abs(cle_full)) + 1e-10
    sess_mean_cre = np.mean(np.abs(cre_full)) + 1e-10
    sess_mean_ch  = np.mean(np.abs(ch_full))  + 1e-10

    # Extract features
    print(f'  Extracting features...')
    df = extract_features(sess, k_resp, k_cardiac,
                          sess_mean_cle, sess_mean_cre, sess_mean_ch)
    df = df[df['stage_code'] >= 0].reset_index(drop=True)
    print(f'  {len(df)} windows after dropping unknown stages')
    print(f'  Stages: {dict(df["stage_label"].value_counts())}')
    print(f'  Apnea: {dict(df["apnea_label"].value_counts())}')

    # Clean and scale
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

    # KW test
    print(f'\n  Kruskal-Wallis (top 5):')
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
    for _, r in kw_df.head(5).iterrows():
        print(f'    {r["feature"]:22s} H={r["H"]:.1f}  p={r["p"]:.1e}')

    # Feature boxplots
    print(f'  Saving feature boxplots...')
    n_feat = len(FEAT_COLS)
    ncols = 4
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes_flat = axes.ravel()
    for j, fname in enumerate(FEAT_COLS):
        ax = axes_flat[j]
        present = [sc for sc in STAGE_ORDER if (stage_codes == sc).sum() > 0]
        groups = [X_raw[stage_codes == sc, j] for sc in present]
        labels_box = [STAGE_NAMES[sc] for sc in present]
        bplot = ax.boxplot(groups, labels=labels_box, patch_artist=True,
                           flierprops={'markersize': 2, 'alpha': 0.3}, widths=0.6)
        for patch, sc in zip(bplot['boxes'], present):
            patch.set_facecolor(STAGE_COLORS[sc]); patch.set_alpha(0.6)
        kw_row = kw_df[kw_df['feature'] == fname]
        if len(kw_row):
            ax.set_title(f'{fname}\nH={kw_row.iloc[0]["H"]:.0f} p={kw_row.iloc[0]["p"]:.1e}', fontsize=8)
        else:
            ax.set_title(fname, fontsize=8)
        ax.tick_params(axis='x', labelsize=7, rotation=45)
        ax.grid(True, alpha=0.2, axis='y')
    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f'Feature Distributions by Stage ({label}) — Delta Sub-Bands', fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'dsub_{label}_feature_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PCA
    print(f'  Running PCA...')
    pca = PCA()
    X_pca_full = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    print(f'    Top 3 PCs: {evr[0]*100:.1f}%, {evr[1]*100:.1f}%, {evr[2]*100:.1f}%  |  {n95} PCs for 95%')

    # PCA scree
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(evr)+1), evr*100, color='#3498DB', alpha=0.8)
    axes[0].set_xlabel('PC'); axes[0].set_ylabel('Variance %'); axes[0].set_title('Scree Plot')
    axes[1].plot(range(1, len(evr)+1), cumvar*100, 'o-', color='#E74C3C', ms=4)
    axes[1].axhline(95, ls='--', color='gray', alpha=0.5)
    axes[1].set_xlabel('PCs'); axes[1].set_ylabel('Cumulative %'); axes[1].set_title('Cumulative Variance')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'dsub_{label}_pca_scree.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PCA loadings
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    feat_names = np.array(FEAT_COLS)
    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        order = np.argsort(np.abs(loadings))[::-1]
        top_n = min(12, len(loadings))
        top_idx = order[:top_n]
        colors = ['#E74C3C' if loadings[j] > 0 else '#3498DB' for j in top_idx]
        ax.barh(range(top_n), loadings[top_idx], color=colors, alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_names[top_idx], fontsize=8)
        ax.set_title(f'PC{i+1} ({evr[i]*100:.1f}%) — loadings', fontsize=9)
        ax.invert_yaxis(); ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'dsub_{label}_pca_loadings.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PCA boxplots
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    for i, ax in enumerate(axes):
        pc_scores = X_pca_full[:, i]
        present = [sc for sc in STAGE_ORDER if (stage_codes == sc).sum() > 0]
        groups = [pc_scores[stage_codes == sc] for sc in present]
        labels_box = [STAGE_NAMES[sc] for sc in present]
        bplot = ax.boxplot(groups, labels=labels_box, patch_artist=True, notch=False,
                           flierprops={'markersize': 2, 'alpha': 0.3})
        for patch, sc in zip(bplot['boxes'], present):
            patch.set_facecolor(STAGE_COLORS[sc]); patch.set_alpha(0.6)
        if len(groups) >= 2:
            stat, pval = kruskal(*groups)
            ax.set_title(f'PC{i+1}  (H={stat:.0f}, p={pval:.1e})', fontsize=8)
        ax.tick_params(axis='x', labelsize=7); ax.grid(True, alpha=0.2, axis='y')
    plt.suptitle(f'PC Scores by Sleep Stage ({label})', fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'dsub_{label}_pca_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3D UMAP
    print(f'  Running 3D UMAP...')
    umap_3d = {}
    for nn in NN_LIST:
        reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, random_state=42)
        umap_3d[nn] = reducer.fit_transform(X)
        print(f'    nn={nn} done')

    # 3D t-SNE
    print(f'  Running 3D t-SNE...')
    tsne_3d = {}
    for perp in PERP_LIST:
        tsne = TSNE(n_components=3, perplexity=perp, random_state=42,
                    max_iter=1000, init='pca', learning_rate='auto')
        tsne_3d[perp] = tsne.fit_transform(X)
        print(f'    perp={perp} done')

    # Interactive plots — UMAP
    print(f'  Saving interactive plots...')
    for nn in NN_LIST:
        X_u = umap_3d[nn]
        pfx = f'dsub_{label}_umap3d_nn{nn}'
        plot_3d_stage(X_u, df, stage_codes, f'3D UMAP (nn={nn}) — {label} — by Stage',
                      PLOT_DIR / f'{pfx}_stage.html')
        plot_3d_time(X_u, df, f'3D UMAP (nn={nn}) — {label} — by Time',
                     PLOT_DIR / f'{pfx}_time.html')
        plot_3d_apnea(X_u, df, f'3D UMAP (nn={nn}) — {label} — Apnea',
                      PLOT_DIR / f'{pfx}_apnea.html')
        plot_3d_thorax(X_u, df, thorax_rms, f'3D UMAP (nn={nn}) — {label} — Thorax RMS',
                       PLOT_DIR / f'{pfx}_thorax.html')

    # Interactive plots — t-SNE
    for perp in PERP_LIST:
        X_t = tsne_3d[perp]
        pfx = f'dsub_{label}_tsne3d_p{perp}'
        plot_3d_stage(X_t, df, stage_codes, f'3D t-SNE (perp={perp}) — {label} — by Stage',
                      PLOT_DIR / f'{pfx}_stage.html')
        plot_3d_time(X_t, df, f'3D t-SNE (perp={perp}) — {label} — by Time',
                     PLOT_DIR / f'{pfx}_time.html')
        plot_3d_apnea(X_t, df, f'3D t-SNE (perp={perp}) — {label} — Apnea',
                      PLOT_DIR / f'{pfx}_apnea.html')
        plot_3d_thorax(X_t, df, thorax_rms, f'3D t-SNE (perp={perp}) — {label} — Thorax RMS',
                       PLOT_DIR / f'{pfx}_thorax.html')

    # Static panels
    print(f'  Saving static panels...')
    plot_3d_panels(umap_3d[30], stage_codes,
                   f'3D UMAP (nn=30) — Delta Subs — {label}',
                   PLOT_DIR / f'dsub_{label}_umap3d_nn30_static.png')
    plot_3d_panels(tsne_3d[30], stage_codes,
                   f'3D t-SNE (perp=30) — Delta Subs — {label}',
                   PLOT_DIR / f'dsub_{label}_tsne3d_p30_static.png')

    # Silhouette scores
    remap_3c = {4: 'Wake', 3: 'NREM', 2: 'NREM', 1: 'NREM', 0: 'REM'}
    labels_3c = np.array([remap_3c[c] for c in stage_codes])
    codes_3c = np.array([{'Wake': 0, 'NREM': 1, 'REM': 2}[l] for l in labels_3c])

    results = []
    for method, embeddings in [('UMAP', umap_3d), ('t-SNE', tsne_3d)]:
        for param, X_emb in embeddings.items():
            sil_5 = silhouette_score(X_emb, stage_codes)
            sil_3 = silhouette_score(X_emb, codes_3c)
            sil_samples_arr = silhouette_samples(X_emb, stage_codes)
            row = {'method': method, 'param': param, 'sil_5class': sil_5, 'sil_3class': sil_3}
            for sc in STAGE_ORDER:
                mask = stage_codes == sc
                if mask.any():
                    row[STAGE_NAMES[sc]] = float(np.mean(sil_samples_arr[mask]))
            results.append(row)

    sil_df = pd.DataFrame(results)
    print(f'\n  Silhouette scores ({label}):')
    print(sil_df.to_string(index=False, float_format='{:.3f}'.format))

    best_5 = sil_df['sil_5class'].max()
    best_3 = sil_df['sil_3class'].max()
    best_row = sil_df.loc[sil_df['sil_5class'].idxmax()]

    # Thorax correlations
    print(f'\n  Thorax RMS vs UMAP dims (nn=30):')
    X_u30 = umap_3d[30]
    for d in range(3):
        rho, p = spearmanr(thorax_rms, X_u30[:, d])
        print(f'    D{d+1}: rho={rho:.3f}, p={p:.1e}')

    print(f'\n  Thorax RMS vs raw features (top 5):')
    thorax_corrs = []
    for j, fname in enumerate(FEAT_COLS):
        rho, p = spearmanr(thorax_rms, X_raw[:, j])
        thorax_corrs.append((fname, rho, p))
    thorax_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for fname, rho, p in thorax_corrs[:5]:
        print(f'    {fname:22s} rho={rho:+.3f}  p={p:.1e}')

    print(f'\n  Thorax RMS by stage:')
    for sc in STAGE_ORDER:
        mask = stage_codes == sc
        if mask.any():
            vals = thorax_rms[mask]
            print(f'    {STAGE_NAMES[sc]:6s}: median={np.median(vals):.2f}, mean={np.mean(vals):.2f}')

    print(f'\n  DONE: {label}  |  best sil 5-class={best_5:.3f} ({best_row["method"]} {best_row["param"]})  |  3-class={best_3:.3f}')
    print(f'  24 interactive HTMLs + 2 static PNGs saved to {PLOT_DIR}')

    return {
        'label': label,
        'n_windows': len(df),
        'best_5class': best_5,
        'best_3class': best_3,
        'best_method': f'{best_row["method"]} {best_row["param"]}',
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # S5N1=8, S5N2=9, S6N1=10, S6N2=11
    session_indices = [8, 9, 10, 11]
    all_results = []

    for idx in session_indices:
        result = run_session(idx)
        all_results.append(result)

    print('\n' + '='*60)
    print('CROSS-SESSION SUMMARY')
    print('='*60)
    for r in all_results:
        print(f'  {r["label"]:6s}: {r["n_windows"]} windows  |  5-class={r["best_5class"]:.3f}  |  3-class={r["best_3class"]:.3f}  |  {r["best_method"]}')
