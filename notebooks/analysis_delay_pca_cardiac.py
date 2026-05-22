"""Delay-embedding + PCA cardiac rate pipeline — runs on 2 sessions.

Hypothesis: the 2x overcount hitting Hilbert / zerocross / peaks comes from
BCG morphology having two bumps (systolic + dicrotic-like) per heartbeat.
A Takens delay embedding of the already-bandpassed cardiac signal, projected
to its top-2 PCs, should produce a near-circular limit cycle whose rotation
= 1 heartbeat per loop — so the instantaneous angular frequency recovers the
fundamental HR even when the waveform has harmonic structure.

Pipeline per window
-------------------
1. CLE-CRE  ->  remove_acc_artifact  ->  bandpass [0.5, 3.0] Hz   (existing)
2. Delay-embed with m=3, tau = 0.25 s          (tau ~ T/4 for T ~ 1 s)
3. SVD -> project to (PC1, PC2)
4. z(t) = PC1 + i PC2  ;  inst_freq = angle(z[n+1] * conj(z[n])) * fs / 2 pi
5. rate = median(|inst_freq|) over the window (Hz)

For each of 2 sessions we:
  A. calibrate per-session k for Hilbert  (existing helper)
  B. calibrate per-session k for Delay-PCA (same recipe, new estimator)
  C. sliding 60-s windows, 5-s step, compute rate vs Pleth-ACF GT
  D. print whole-night MAE / RMSE / bias / r / coverage for
     {ACF, Hilbert raw, Hilbert/k, DelayPCA raw, DelayPCA/k}
  E. save a diagnostic plot of one window (embedding + phase + estimate)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleep_monitor import (
    load_session, CARD_LO, CARD_HI,
    remove_acc_artifact, bandpass, rate_acf, rate_hilbert,
    calibrate_k_cardiac,
)

SESSIONS    = [0, 1]
WIN_SEC     = 60.0
STEP_SEC    = 5.0
TAU_S       = 0.25
EMBED_DIM   = 3
N_CALIB     = 50
SEED        = 42
OUT_DIR     = Path('notebooks/plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Core estimator ────────────────────────────────────────────────────────
def rate_delay_pca(x, fs, tau_s=TAU_S, m=EMBED_DIM,
                    f_lo=CARD_LO, f_hi=CARD_HI):
    """Cardiac rate (Hz) from delay-embedded + PCA-projected inst. frequency."""
    x = np.asarray(x, dtype=np.float64)
    tau = max(1, int(round(tau_s * fs)))
    N = len(x) - (m - 1) * tau
    if N < 64:
        return np.nan
    X = np.empty((N, m))
    for j in range(m):
        X[:, j] = x[j * tau : j * tau + N]
    X -= X.mean(axis=0)
    try:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan
    # Degeneracy guard: need a real 2D attractor
    if S[0] < 1e-12 or S[1] / S[0] < 0.15:
        return np.nan
    pc1 = U[:, 0] * S[0]
    pc2 = U[:, 1] * S[1]
    z = pc1 + 1j * pc2
    inst_freq = np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)
    med = float(np.median(np.abs(inst_freq)))
    if not np.isfinite(med) or med < f_lo or med > f_hi:
        return np.nan
    return med


def calibrate_k_delay_pca(session, n_windows=N_CALIB, win_s=WIN_SEC, seed=SEED):
    """Median(rate_delay_pca / rate_acf_gt) over random 1-min windows."""
    fs = session.fs
    raw = (session.cap['CLE'].astype(np.float64)
            - session.cap['CRE'].astype(np.float64))
    acc = session.cap['acc_mag'].astype(np.float64)
    sig = remove_acc_artifact(raw, acc, CARD_LO, CARD_HI, fs)
    gt  = bandpass(session.psg['Pleth'].astype(np.float64),
                    CARD_LO, CARD_HI, fs)

    win_n = int(round(win_s * fs))
    total = len(sig)
    max_start = total - win_n - 1
    rng = np.random.default_rng(seed)
    starts = sorted(rng.integers(0, max_start, size=n_windows).tolist())

    ratios = []
    for st in starts:
        seg    = sig[st:st + win_n]
        seg_gt = gt [st:st + win_n]
        r_cap = rate_delay_pca(seg, fs)
        r_gt  = rate_acf(seg_gt, CARD_LO, CARD_HI, fs, prominence=0.05)
        if np.isfinite(r_cap) and np.isfinite(r_gt) and r_gt > 0:
            ratios.append(r_cap / r_gt)
    if len(ratios) < 10:
        return float('nan')
    return float(np.median(ratios))


# ── Per-session whole-night evaluation ────────────────────────────────────
def eval_session(sess_idx):
    session = load_session(sess_idx)
    fs = session.fs
    print(f'\n======================================================================')
    print(f'Session idx={sess_idx}: {session}')
    print(f'======================================================================')

    print('  preprocessing...', end=' ', flush=True)
    raw = (session.cap['CLE'].astype(np.float64)
            - session.cap['CRE'].astype(np.float64))
    acc = session.cap['acc_mag'].astype(np.float64)
    sig = remove_acc_artifact(raw, acc, CARD_LO, CARD_HI, fs)
    gt  = bandpass(session.psg['Pleth'].astype(np.float64),
                    CARD_LO, CARD_HI, fs)
    print('done')

    # Calibrate k on both estimators
    print('  calibrating k (50 random 1-min windows)...', end=' ', flush=True)
    k_hil = calibrate_k_cardiac(session, n_windows=N_CALIB, win_s=WIN_SEC, seed=SEED)
    k_pca = calibrate_k_delay_pca(session, n_windows=N_CALIB, win_s=WIN_SEC, seed=SEED)
    print(f'done  k_hilbert={k_hil:.3f}   k_delayPCA={k_pca:.3f}')

    # Sliding windows
    win_n = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    starts = list(range(0, len(sig) - win_n + 1, step_n))
    print(f'  sliding eval: N={len(starts)} windows (win={WIN_SEC:.0f}s, step={STEP_SEC:.0f}s)...',
          end=' ', flush=True)

    arr_gt  = np.full(len(starts), np.nan)
    arr_acf = np.full(len(starts), np.nan)
    arr_hil = np.full(len(starts), np.nan)
    arr_pca = np.full(len(starts), np.nan)

    for i, st in enumerate(starts):
        seg    = sig[st:st + win_n]
        seg_gt = gt [st:st + win_n]
        arr_gt [i] = rate_acf(seg_gt, CARD_LO, CARD_HI, fs, prominence=0.05) * 60.0
        arr_acf[i] = rate_acf(seg,    CARD_LO, CARD_HI, fs, prominence=0.05) * 60.0
        arr_hil[i] = rate_hilbert  (seg, CARD_LO, CARD_HI, fs)             * 60.0
        arr_pca[i] = rate_delay_pca(seg, fs)                                * 60.0
    print('done')

    def rep(name, pred):
        ok = np.isfinite(pred) & np.isfinite(arr_gt)
        if ok.sum() < 5:
            return dict(method=name, MAE=np.nan, RMSE=np.nan, bias=np.nan,
                         r=np.nan, cov=0.0, n=int(ok.sum()))
        err = pred[ok] - arr_gt[ok]
        return dict(
            method=name,
            MAE=float(np.mean(np.abs(err))),
            RMSE=float(np.sqrt(np.mean(err ** 2))),
            bias=float(np.mean(err)),
            r=float(np.corrcoef(pred[ok], arr_gt[ok])[0, 1]),
            cov=float(ok.sum() / len(pred) * 100),
            n=int(ok.sum()),
        )

    rows = [
        rep('ACF (cap)',            arr_acf),
        rep('Hilbert raw',          arr_hil),
        rep(f'Hilbert / k={k_hil:.2f}',  arr_hil / k_hil),
        rep('DelayPCA raw',         arr_pca),
        rep(f'DelayPCA / k={k_pca:.2f}', arr_pca / k_pca),
    ]
    df = pd.DataFrame(rows)
    for c in ['MAE', 'RMSE', 'bias', 'r', 'cov']:
        df[c] = df[c].astype(float).round(2)
    pd.set_option('display.width', 200)
    print('  whole-night metrics (BPM):')
    print(df.to_string(index=False))

    return {
        'idx': sess_idx, 'name': str(session),
        'k_hil': k_hil, 'k_pca': k_pca,
        'df': df,
        'sig': sig, 'gt': gt, 'fs': fs,
    }


# ── Run both sessions ─────────────────────────────────────────────────────
results = [eval_session(i) for i in SESSIONS]


# ── Diagnostic plot: one window per session, showing the PCA limit cycle ──
print('\n== Plotting diagnostic windows ==')
fig, axes = plt.subplots(len(results), 3, figsize=(15, 4.0 * len(results)))
if len(results) == 1:
    axes = axes[None, :]

rng = np.random.default_rng(SEED + 1)
for row_i, R in enumerate(results):
    fs = R['fs']
    sig, gt = R['sig'], R['gt']
    win_n = int(round(WIN_SEC * fs))
    st = int(rng.integers(0, len(sig) - win_n - 1))
    seg    = sig[st:st + win_n]
    seg_gt = gt [st:st + win_n]

    # Raw time series
    ax = axes[row_i, 0]
    t = np.arange(win_n) / fs
    ax.plot(t, seg / (seg.std() + 1e-9), color='#E67E22', lw=0.9,
            label='CLE-CRE bp (z)')
    ax.plot(t, seg_gt / (seg_gt.std() + 1e-9) - 5, color='black', lw=0.9,
            alpha=0.85, label='Pleth bp (z, offset)')
    ax.set_title(f'{R["name"]}\n1-min window @ t={st/fs/3600:.2f} hr',
                  fontsize=9, loc='left')
    ax.set_xlabel('time [s]'); ax.set_ylabel('a.u.')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')

    # PCA embedding
    tau = max(1, int(round(TAU_S * fs)))
    N = len(seg) - (EMBED_DIM - 1) * tau
    X = np.empty((N, EMBED_DIM))
    for j in range(EMBED_DIM):
        X[:, j] = seg[j * tau : j * tau + N]
    X -= X.mean(axis=0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pc1, pc2 = U[:, 0] * S[0], U[:, 1] * S[1]

    ax = axes[row_i, 1]
    ax.plot(pc1, pc2, lw=0.7, color='#2E86C1')
    ax.scatter(pc1[0], pc2[0], s=30, color='green', zorder=5, label='start')
    ax.scatter(pc1[-1], pc2[-1], s=30, color='red', zorder=5, label='end')
    s_ratio = S[1] / S[0]
    ax.set_title(f'Delay-PCA embedding  tau={TAU_S}s  m={EMBED_DIM}\n'
                  f'sigma1/sigma0={s_ratio:.2f}   S={S[:3].round(1)}',
                  fontsize=9, loc='left')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')

    # Phase and instantaneous freq
    z = pc1 + 1j * pc2
    inst_freq = np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)
    phase_unw = np.unwrap(np.angle(z))

    ax = axes[row_i, 2]
    tN = np.arange(N) / fs
    ax.plot(tN, phase_unw / (2 * np.pi), color='#884EA0', lw=0.9,
            label='phase [cycles]')
    ax2 = ax.twinx()
    ax2.plot(tN[1:], np.abs(inst_freq) * 60.0, color='#CA6F1E', lw=0.6,
             alpha=0.6, label='|inst freq| [BPM]')
    med_bpm = np.median(np.abs(inst_freq)) * 60.0
    gt_bpm  = rate_acf(seg_gt, CARD_LO, CARD_HI, fs, prominence=0.05) * 60.0
    ax2.axhline(med_bpm, color='#CA6F1E', lw=1.2, ls='--',
                 label=f'median={med_bpm:.1f}')
    ax2.axhline(gt_bpm,  color='black', lw=1.2, ls=':',
                 label=f'GT={gt_bpm:.1f}')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('phase [cycles]', color='#884EA0')
    ax2.set_ylabel('BPM', color='#CA6F1E')
    ax.set_title(f'Phase evolution + inst. freq\n'
                  f'raw={med_bpm:.1f}  k_pca={R["k_pca"]:.2f}  '
                  f'scaled={med_bpm/R["k_pca"]:.1f}  GT={gt_bpm:.1f}',
                  fontsize=9, loc='left')
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')

fig.suptitle(
    f'Delay-embedding + PCA cardiac pipeline  '
    f'(CLE-CRE OLS-acc, bp [{CARD_LO}-{CARD_HI}] Hz, '
    f'tau={TAU_S}s, m={EMBED_DIM})',
    fontsize=11,
)
fig.tight_layout()
out = OUT_DIR / 'delay_pca_cardiac_2sessions.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'plot -> {out}')

# ── Summary ───────────────────────────────────────────────────────────────
print('\n========================================================')
print('SUMMARY across sessions')
print('========================================================')
for R in results:
    print(f'\n{R["name"]}  k_hil={R["k_hil"]:.3f}  k_pca={R["k_pca"]:.3f}')
    print(R['df'].to_string(index=False))
