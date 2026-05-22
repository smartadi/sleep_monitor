"""DMD cardiac-rate pipeline on stacked [CLE, CRE] delay-embedding — 2 sessions.

Same preprocessing + delay-embedding as analysis_pca_stacked_cle_cre.py, but
instead of taking top-PCs-then-phase, we run exact DMD on the snapshot matrix:

    X0 = X[:, :-1]       X1 = X[:, 1:]
    U, S, V = svd(X0)
    A_tilde = U* X1 V S^-1
    eig(A_tilde) = {lambda_i}    <- each encodes (freq, damping)

Frequencies: f_i = angle(lambda_i) * fs / (2 pi).
HR mode   : argmax amplitude across i with |f_i| in [0.5, 3.0] Hz.

Compared against:
  - rate_acf    (baseline)
  - rate_hilbert_scaled_cardiac  (existing best; MAE ~4.3 BPM)
  - rate_delay_pca (prior pipeline on CLE-CRE)

Diagnostic figure: for 1 random window per session, plots the DMD eigenvalue
spectrum (unit circle) and the amplitude-vs-frequency stem plot, with GT and
selected-mode markers.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleep_monitor import (
    load_session, CARD_LO, CARD_HI,
    remove_acc_artifact, bandpass,
    rate_acf, rate_hilbert, calibrate_k_cardiac,
)

SESSIONS  = [0, 1]
WIN_SEC   = 60.0
STEP_SEC  = 5.0
TAU_S     = 0.25
EMBED_DIM = 3                 # per channel -> 2*m = 6 stacked features
DMD_RANK  = 6                 # full rank of feature space
N_CALIB   = 50
SEED      = 42
OUT_DIR   = Path('notebooks/plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Build the delay-embedded stacked feature matrix ───────────────────────
def build_embed(cle_seg, cre_seg, fs, tau_s=TAU_S, m=EMBED_DIM):
    tau = max(1, int(round(tau_s * fs)))
    N = len(cle_seg) - (m - 1) * tau
    if N < 64:
        return None, tau
    X = np.empty((N, 2 * m))
    for j in range(m):
        X[:, j]     = cle_seg[j * tau : j * tau + N]
        X[:, j + m] = cre_seg[j * tau : j * tau + N]
    X -= X.mean(axis=0)
    X /= (X.std(axis=0) + 1e-12)
    return X, tau


# ── Exact DMD ─────────────────────────────────────────────────────────────
def dmd(X_embed, fs, rank=DMD_RANK):
    """Exact DMD. Input X_embed is (N_time, features). Returns dict with
    eigenvalues, frequencies (Hz), decay, amplitudes, modes."""
    X = X_embed.T                              # (features, N_time)
    X0, X1 = X[:, :-1], X[:, 1:]
    try:
        U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    r = min(rank, len(S))
    # drop near-zero singular values
    r_eff = int(np.sum(S[:r] > S[0] * 1e-10))
    if r_eff < 2:
        return None
    U, S, Vh = U[:, :r_eff], S[:r_eff], Vh[:r_eff, :]
    A_tilde  = U.conj().T @ X1 @ Vh.conj().T @ np.diag(1.0 / S)
    eigvals, W = np.linalg.eig(A_tilde)
    # exact-DMD modes
    Phi = X1 @ Vh.conj().T @ np.diag(1.0 / S) @ W
    # mode amplitudes from initial snapshot
    b, *_ = np.linalg.lstsq(Phi, X0[:, 0], rcond=None)
    # continuous-time frequency & decay: omega = log(lambda) / dt
    safe = np.abs(eigvals) > 1e-12
    omega = np.full_like(eigvals, np.nan, dtype=complex)
    omega[safe] = np.log(eigvals[safe]) * fs
    freq  = omega.imag / (2 * np.pi)
    decay = omega.real
    amp   = np.abs(b) * np.linalg.norm(Phi, axis=0)
    return dict(eigvals=eigvals, freq=freq, decay=decay, amp=amp, modes=Phi)


# ── Rate extraction ───────────────────────────────────────────────────────
def rate_dmd(cle_seg, cre_seg, fs, f_lo=CARD_LO, f_hi=CARD_HI):
    X, _ = build_embed(cle_seg, cre_seg, fs)
    if X is None:
        return np.nan
    res = dmd(X, fs)
    if res is None:
        return np.nan
    abs_freq = np.abs(res['freq'])
    in_band = np.isfinite(abs_freq) & (abs_freq >= f_lo) & (abs_freq <= f_hi)
    # penalise highly-damped modes (|decay| too negative means fast decay)
    score = res['amp'] * np.exp(res['decay'] / fs)   # small bonus for ~stable modes
    score = np.where(in_band, score, -np.inf)
    if not np.any(np.isfinite(score)) or score.max() == -np.inf:
        return np.nan
    return float(abs_freq[int(np.argmax(score))])


def calibrate_k_dmd(session, n_windows=N_CALIB, win_s=WIN_SEC, seed=SEED):
    fs  = session.fs
    cle = remove_acc_artifact(session.cap['CLE'].astype(np.float64),
                               session.cap['acc_mag'].astype(np.float64),
                               CARD_LO, CARD_HI, fs)
    cre = remove_acc_artifact(session.cap['CRE'].astype(np.float64),
                               session.cap['acc_mag'].astype(np.float64),
                               CARD_LO, CARD_HI, fs)
    gt  = bandpass(session.psg['Pleth'].astype(np.float64),
                    CARD_LO, CARD_HI, fs)
    win_n = int(round(win_s * fs))
    max_start = len(cle) - win_n - 1
    rng = np.random.default_rng(seed)
    starts = sorted(rng.integers(0, max_start, size=n_windows).tolist())
    ratios = []
    for st in starts:
        r_cap = rate_dmd(cle[st:st+win_n], cre[st:st+win_n], fs)
        r_gt  = rate_acf(gt[st:st+win_n], CARD_LO, CARD_HI, fs, prominence=0.05)
        if np.isfinite(r_cap) and np.isfinite(r_gt) and r_gt > 0:
            ratios.append(r_cap / r_gt)
    if len(ratios) < 10:
        return float('nan')
    return float(np.median(ratios))


# ── Also a delay-PCA baseline on the CLE-CRE channel (from prior script) ─
def rate_delay_pca(x, fs, tau_s=TAU_S, m=EMBED_DIM, f_lo=CARD_LO, f_hi=CARD_HI):
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
    if S[0] < 1e-12 or S[1] / S[0] < 0.15:
        return np.nan
    z = U[:, 0] * S[0] + 1j * U[:, 1] * S[1]
    inst_freq = np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)
    med = float(np.median(np.abs(inst_freq)))
    if not np.isfinite(med) or med < f_lo or med > f_hi:
        return np.nan
    return med


# ── Per-session run ───────────────────────────────────────────────────────
def eval_session(sess_idx):
    session = load_session(sess_idx)
    fs = session.fs
    print(f'\n======================================================================')
    print(f'idx={sess_idx}  {session}')
    print(f'======================================================================')

    print('  preprocessing...', end=' ', flush=True)
    cle = remove_acc_artifact(session.cap['CLE'].astype(np.float64),
                               session.cap['acc_mag'].astype(np.float64),
                               CARD_LO, CARD_HI, fs)
    cre = remove_acc_artifact(session.cap['CRE'].astype(np.float64),
                               session.cap['acc_mag'].astype(np.float64),
                               CARD_LO, CARD_HI, fs)
    diff = cle - cre
    gt  = bandpass(session.psg['Pleth'].astype(np.float64),
                    CARD_LO, CARD_HI, fs)
    print('done')

    print('  calibrating k (50 random 1-min windows)...', end=' ', flush=True)
    k_hil = calibrate_k_cardiac(session, n_windows=N_CALIB, win_s=WIN_SEC, seed=SEED)
    k_dmd = calibrate_k_dmd   (session, n_windows=N_CALIB, win_s=WIN_SEC, seed=SEED)
    print(f'done  k_hilbert={k_hil:.3f}  k_dmd={k_dmd:.3f}')

    win_n  = int(round(WIN_SEC * fs))
    step_n = int(round(STEP_SEC * fs))
    starts = list(range(0, len(cle) - win_n + 1, step_n))
    print(f'  sliding eval: N={len(starts)} windows...', end=' ', flush=True)

    gt_a  = np.full(len(starts), np.nan)
    acf_a = np.full(len(starts), np.nan)
    hil_a = np.full(len(starts), np.nan)
    pca_a = np.full(len(starts), np.nan)
    dmd_a = np.full(len(starts), np.nan)
    for i, st in enumerate(starts):
        sl = slice(st, st + win_n)
        gt_a [i] = rate_acf(gt[sl],   CARD_LO, CARD_HI, fs, prominence=0.05) * 60.0
        acf_a[i] = rate_acf(diff[sl], CARD_LO, CARD_HI, fs, prominence=0.05) * 60.0
        hil_a[i] = rate_hilbert(diff[sl], CARD_LO, CARD_HI, fs) * 60.0
        pca_a[i] = rate_delay_pca(diff[sl], fs) * 60.0
        dmd_a[i] = rate_dmd(cle[sl], cre[sl], fs) * 60.0
    print('done')

    def rep(name, pred):
        ok = np.isfinite(pred) & np.isfinite(gt_a)
        if ok.sum() < 5:
            return dict(method=name, MAE=np.nan, RMSE=np.nan, bias=np.nan,
                         r=np.nan, cov=0.0, n=int(ok.sum()))
        err = pred[ok] - gt_a[ok]
        return dict(
            method=name,
            MAE  = float(np.mean(np.abs(err))),
            RMSE = float(np.sqrt(np.mean(err ** 2))),
            bias = float(np.mean(err)),
            r    = float(np.corrcoef(pred[ok], gt_a[ok])[0, 1]),
            cov  = float(ok.sum() / len(pred) * 100),
            n    = int(ok.sum()),
        )

    rows = [
        rep('ACF (cap)',                    acf_a),
        rep('Hilbert raw',                  hil_a),
        rep(f'Hilbert / k={k_hil:.2f}',     hil_a / k_hil),
        rep('DelayPCA (CLE-CRE)',           pca_a),
        rep('DMD raw',                      dmd_a),
        rep(f'DMD / k={k_dmd:.2f}',         dmd_a / k_dmd),
    ]
    df = pd.DataFrame(rows)
    for c in ['MAE', 'RMSE', 'bias', 'r', 'cov']:
        df[c] = df[c].astype(float).round(2)
    pd.set_option('display.width', 200)
    print('  whole-night metrics (BPM):')
    print(df.to_string(index=False))

    return {'idx': sess_idx, 'name': str(session),
            'k_hil': k_hil, 'k_dmd': k_dmd, 'df': df,
            'cle': cle, 'cre': cre, 'gt': gt, 'fs': fs}


# ── Run both sessions ─────────────────────────────────────────────────────
results = [eval_session(i) for i in SESSIONS]


# ── Diagnostic plot: DMD eigenvalue spectrum + mode amp-vs-freq ───────────
print('\n== Diagnostic plots ==')
fig, axes = plt.subplots(len(results), 2, figsize=(13, 4.3 * len(results)))
if len(results) == 1:
    axes = axes[None, :]

rng = np.random.default_rng(SEED + 7)
for row_i, R in enumerate(results):
    fs = R['fs']
    win_n = int(round(WIN_SEC * fs))
    st = int(rng.integers(0, len(R['cle']) - win_n - 1))
    cle_s = R['cle'][st:st + win_n]
    cre_s = R['cre'][st:st + win_n]
    gt_s  = R['gt'] [st:st + win_n]
    X, tau = build_embed(cle_s, cre_s, fs)
    res = dmd(X, fs)
    eigvals = res['eigvals']
    freq    = res['freq']
    amp     = res['amp']
    gt_bpm  = rate_acf(gt_s, CARD_LO, CARD_HI, fs, prominence=0.05) * 60.0

    # Pick selected HR mode (same logic as rate_dmd)
    abs_freq = np.abs(freq)
    in_band = np.isfinite(abs_freq) & (abs_freq >= CARD_LO) & (abs_freq <= CARD_HI)
    score = amp * np.exp(res['decay'] / fs)
    score = np.where(in_band, score, -np.inf)
    sel = int(np.argmax(score)) if np.any(in_band) else None
    sel_bpm = abs_freq[sel] * 60.0 if sel is not None else np.nan

    # Eigenvalue spectrum (unit circle)
    ax = axes[row_i, 0]
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.6)
    ax.scatter(eigvals.real, eigvals.imag,
                s=60 + 150 * amp / (amp.max() + 1e-12),
                c='#2E86C1', alpha=0.8, edgecolor='k', linewidth=0.5,
                label='all modes')
    if sel is not None:
        ax.scatter([eigvals[sel].real], [eigvals[sel].imag],
                    s=120, facecolor='none', edgecolor='red', linewidth=1.5,
                    label='HR mode (selected)')
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('Re($\\lambda$)'); ax.set_ylabel('Im($\\lambda$)')
    ax.set_title(
        f'{R["name"]}\n'
        f'DMD eigenvalues  (window @ t={st/fs/3600:.2f} hr)   '
        f'tau={TAU_S}s ({tau} samp), m={EMBED_DIM}, 2*m=6 feats',
        fontsize=9, loc='left',
    )
    ax.legend(fontsize=7, loc='upper right')

    # Amplitude-vs-frequency stem
    ax = axes[row_i, 1]
    order = np.argsort(abs_freq)
    ax.stem(abs_freq[order] * 60.0, amp[order] / (amp.max() + 1e-12),
             linefmt='#2E86C1', markerfmt='o', basefmt=' ')
    ax.axvline(gt_bpm,  color='black', ls=':', lw=1.2, label=f'GT  = {gt_bpm:.1f} BPM')
    if sel is not None:
        ax.axvline(sel_bpm, color='red', ls='--', lw=1.2,
                    label=f'DMD = {sel_bpm:.1f} BPM')
    ax.axvspan(CARD_LO * 60, CARD_HI * 60, color='gray', alpha=0.12,
                label='cardiac band')
    ax.set_xlabel('|freq| [BPM]'); ax.set_ylabel('mode amplitude (norm.)')
    ax.set_title(
        f'Mode amplitude vs frequency   '
        f'k_dmd={R["k_dmd"]:.2f}   scaled DMD = {sel_bpm / R["k_dmd"]:.1f} BPM',
        fontsize=9, loc='left',
    )
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, 250)

fig.suptitle(
    f'DMD on stacked [CLE, CRE] delay-embedding   '
    f'bp=[{CARD_LO}, {CARD_HI}] Hz',
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT_DIR / 'dmd_cardiac_2sessions.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'plot -> {out}')

# ── Summary ───────────────────────────────────────────────────────────────
print('\n========================================================')
print('SUMMARY')
print('========================================================')
for R in results:
    print(f'\n{R["name"]}  k_hil={R["k_hil"]:.3f}  k_dmd={R["k_dmd"]:.3f}')
    print(R['df'].to_string(index=False))
