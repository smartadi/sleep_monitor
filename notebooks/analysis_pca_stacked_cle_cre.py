"""PCA on delay-embedded [CLE, CRE] stack — 2 sessions, 5 random 1-min windows.

Pipeline per window
-------------------
1. Preprocess CLE and CRE independently with remove_acc_artifact (cardiac band).
2. Delay-embed each channel: m = 3, tau = 0.25 s  (tau ~ T/4 for T ~ 1 s).
3. Stack columns -> feature matrix X of shape (N, 2*m = 6).
4. Per-column z-score so CLE and CRE contribute comparably.
5. SVD -> top 3 principal components.

Plots per window (subplot grid):
  - Left : PC1, PC2, PC3, GT (Pleth bp) vs time (offset-stacked).
  - Right: 3D PC1-PC2-PC3 trajectory.
Delay-embedding parameters are shown in the suptitle of each figure.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for 3d projection

from sleep_monitor import (
    load_session, CARD_LO, CARD_HI,
    remove_acc_artifact, bandpass,
)

SESSIONS  = [0, 1]
N_WIN     = 5
WIN_SEC   = 60.0
TAU_S     = 0.25
EMBED_DIM = 3
SEED      = 42
OUT_DIR   = Path('notebooks/plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stacked_embedding_pca(cle_seg, cre_seg, fs, tau_s=TAU_S, m=EMBED_DIM):
    """Delay-embed CLE and CRE separately, stack, z-score, SVD → top-3 PCs."""
    tau = max(1, int(round(tau_s * fs)))
    N = len(cle_seg) - (m - 1) * tau
    X = np.empty((N, 2 * m))
    for j in range(m):
        X[:, j]         = cle_seg[j * tau : j * tau + N]
        X[:, j + m]     = cre_seg[j * tau : j * tau + N]
    X -= X.mean(axis=0)
    X /= (X.std(axis=0) + 1e-12)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pcs = U[:, :3] * S[:3]
    return pcs, S, tau


def run_session(sess_idx, seed):
    session = load_session(sess_idx)
    fs = session.fs
    print(f'\n=== idx={sess_idx}  {session} ===')

    cle_c = remove_acc_artifact(
        session.cap['CLE'].astype(np.float64),
        session.cap['acc_mag'].astype(np.float64),
        CARD_LO, CARD_HI, fs,
    )
    cre_c = remove_acc_artifact(
        session.cap['CRE'].astype(np.float64),
        session.cap['acc_mag'].astype(np.float64),
        CARD_LO, CARD_HI, fs,
    )
    gt = bandpass(session.psg['Pleth'].astype(np.float64),
                   CARD_LO, CARD_HI, fs)

    win_n     = int(round(WIN_SEC * fs))
    max_start = len(cle_c) - win_n - 1
    rng       = np.random.default_rng(seed)
    starts    = sorted(rng.integers(0, max_start, size=N_WIN).tolist())

    fig = plt.figure(figsize=(14.5, 3.6 * N_WIN))

    tau_shown = None
    for i, st in enumerate(starts):
        cle_seg = cle_c[st:st + win_n]
        cre_seg = cre_c[st:st + win_n]
        gt_seg  = gt   [st:st + win_n]

        pcs, S, tau = stacked_embedding_pca(cle_seg, cre_seg, fs)
        tau_shown = tau
        N         = pcs.shape[0]
        t_pc      = np.arange(N) / fs
        t_gt      = np.arange(len(gt_seg)) / fs

        # z-normalise for plotting
        pc_z = pcs      / (pcs.std(axis=0) + 1e-12)
        gt_z = gt_seg   / (gt_seg.std()   + 1e-12)
        var_exp = (S[:3] ** 2) / (S ** 2).sum() * 100.0

        # Left: time-plot (PC1-3 + GT)
        ax = fig.add_subplot(N_WIN, 2, 2 * i + 1)
        ax.plot(t_pc, pc_z[:, 0] + 0,   color='#E67E22', lw=0.9, label='PC1')
        ax.plot(t_pc, pc_z[:, 1] - 5,   color='#2E86C1', lw=0.9, label='PC2')
        ax.plot(t_pc, pc_z[:, 2] - 10,  color='#8E44AD', lw=0.9, label='PC3')
        ax.plot(t_gt, gt_z        - 15, color='black',   lw=0.9, alpha=0.85,
                 label='GT Pleth')
        ax.set_title(
            f'Win {i+1}  t={st/fs/3600:.2f} hr   '
            f'var_exp PC1/2/3 = {var_exp[0]:.1f}/{var_exp[1]:.1f}/{var_exp[2]:.1f} %',
            fontsize=9, loc='left',
        )
        ax.set_xlabel('time within window [s]')
        ax.set_ylabel('a.u. (z-scored, offset)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper right', ncol=4)

        # Right: 3D trajectory
        ax3 = fig.add_subplot(N_WIN, 2, 2 * i + 2, projection='3d')
        ax3.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2],
                  lw=0.5, color='#2E86C1', alpha=0.85)
        ax3.scatter(pcs[0, 0],  pcs[0, 1],  pcs[0, 2],
                     s=30, color='green', label='start', zorder=5)
        ax3.scatter(pcs[-1, 0], pcs[-1, 1], pcs[-1, 2],
                     s=30, color='red',   label='end',   zorder=5)
        ax3.set_xlabel('PC1'); ax3.set_ylabel('PC2'); ax3.set_zlabel('PC3')
        ax3.set_title('3D PC1–PC2–PC3 trajectory', fontsize=9, loc='left')
        ax3.legend(fontsize=7, loc='upper right')

    fig.suptitle(
        f'{session}   PCA on stacked [CLE, CRE] delay-embedding   '
        f'tau = {TAU_S}s ({tau_shown} samples),  m = {EMBED_DIM},  '
        f'features = 2*m = {2 * EMBED_DIM}',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = OUT_DIR / f'pca_stacked_cle_cre_sess{sess_idx}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


for s in SESSIONS:
    run_session(s, seed=SEED + s)

print('\nDone')
