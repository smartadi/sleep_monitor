"""Hankel-DMD sweep: whole-night cardiac rate vs embedding dim m (S1N1).

Same stacked [CLE, CRE] delay-embedding + DMD pipeline, but sweep m:
  m in {3, 6, 10, 15, 20, 30}   (2*m features → rank = 2*m)

For each m:
  - calibrate per-session k  (50 random 1-min windows)
  - sliding eval over whole night (60s win, 5s step)
  - report k, MAE, RMSE, bias, r, coverage

A summary table + a 3-panel plot of (MAE, coverage, k_dmd) vs m.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleep_monitor import (
    load_session, CARD_LO, CARD_HI,
    remove_acc_artifact, bandpass, rate_acf,
)

SESS_IDX   = 0
WIN_SEC    = 60.0
STEP_SEC   = 5.0
TAU_S      = 0.25
M_SWEEP    = [3, 6, 10, 15, 20, 30]
N_CALIB    = 50
SEED       = 42
OUT_DIR    = Path('notebooks/plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_embed(cle_seg, cre_seg, fs, tau_s, m):
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


def dmd_rate(cle_seg, cre_seg, fs, tau_s, m, f_lo=CARD_LO, f_hi=CARD_HI):
    X, _ = build_embed(cle_seg, cre_seg, fs, tau_s, m)
    if X is None:
        return np.nan
    Xt = X.T
    X0, X1 = Xt[:, :-1], Xt[:, 1:]
    try:
        U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan
    r_eff = int(np.sum(S > S[0] * 1e-10))
    if r_eff < 2:
        return np.nan
    U, S, Vh = U[:, :r_eff], S[:r_eff], Vh[:r_eff, :]
    A_tilde = U.conj().T @ X1 @ Vh.conj().T @ np.diag(1.0 / S)
    eigvals, W = np.linalg.eig(A_tilde)
    Phi = X1 @ Vh.conj().T @ np.diag(1.0 / S) @ W
    b, *_ = np.linalg.lstsq(Phi, X0[:, 0], rcond=None)
    safe = np.abs(eigvals) > 1e-12
    freq  = np.full(len(eigvals), np.nan)
    decay = np.full(len(eigvals), np.nan)
    if safe.any():
        log_lam = np.log(eigvals[safe]) * fs
        freq [safe] = log_lam.imag / (2 * np.pi)
        decay[safe] = log_lam.real
    amp = np.abs(b) * np.linalg.norm(Phi, axis=0)
    abs_freq = np.abs(freq)
    in_band = np.isfinite(abs_freq) & (abs_freq >= f_lo) & (abs_freq <= f_hi)
    score = amp * np.exp(decay / fs)
    score = np.where(in_band, score, -np.inf)
    if not np.any(np.isfinite(score)) or score.max() == -np.inf:
        return np.nan
    return float(abs_freq[int(np.argmax(score))])


def calibrate_k(cle, cre, gt, fs, tau_s, m, n_windows=N_CALIB,
                 win_s=WIN_SEC, seed=SEED):
    win_n = int(round(win_s * fs))
    max_start = len(cle) - win_n - 1
    rng = np.random.default_rng(seed)
    starts = sorted(rng.integers(0, max_start, size=n_windows).tolist())
    ratios = []
    for st in starts:
        r_cap = dmd_rate(cle[st:st+win_n], cre[st:st+win_n], fs, tau_s, m)
        r_gt  = rate_acf(gt[st:st+win_n], CARD_LO, CARD_HI, fs, prominence=0.05)
        if np.isfinite(r_cap) and np.isfinite(r_gt) and r_gt > 0:
            ratios.append(r_cap / r_gt)
    if len(ratios) < 10:
        return float('nan'), 0
    return float(np.median(ratios)), len(ratios)


# ── Load session ──────────────────────────────────────────────────────────
session = load_session(SESS_IDX)
fs = session.fs
print(session)

print('preprocessing...', end=' ', flush=True)
cle = remove_acc_artifact(session.cap['CLE'].astype(np.float64),
                           session.cap['acc_mag'].astype(np.float64),
                           CARD_LO, CARD_HI, fs)
cre = remove_acc_artifact(session.cap['CRE'].astype(np.float64),
                           session.cap['acc_mag'].astype(np.float64),
                           CARD_LO, CARD_HI, fs)
gt  = bandpass(session.psg['Pleth'].astype(np.float64),
                CARD_LO, CARD_HI, fs)
print('done')

win_n  = int(round(WIN_SEC * fs))
step_n = int(round(STEP_SEC * fs))
starts = list(range(0, len(cle) - win_n + 1, step_n))
print(f'N windows = {len(starts)}   (win={WIN_SEC}s, step={STEP_SEC}s)')

# GT rates (one pass, independent of m)
print('computing GT...', end=' ', flush=True)
gt_a = np.array([rate_acf(gt[s:s+win_n], CARD_LO, CARD_HI, fs, prominence=0.05)
                   for s in starts]) * 60.0
print(f'done   (valid={np.isfinite(gt_a).sum()}/{len(gt_a)})')


# ── Sweep m ───────────────────────────────────────────────────────────────
results = []
print('\nm   | k_dmd   n_cal | MAE   RMSE   bias     r     cov%  |  time(s)')
print('----+---------------+----------------------------------+---------')
for m in M_SWEEP:
    t0 = time.time()
    k_dmd, n_cal = calibrate_k(cle, cre, gt, fs, TAU_S, m)

    preds = np.full(len(starts), np.nan)
    for i, st in enumerate(starts):
        preds[i] = dmd_rate(cle[st:st+win_n], cre[st:st+win_n],
                              fs, TAU_S, m) * 60.0

    scaled = preds / k_dmd if np.isfinite(k_dmd) else preds
    ok = np.isfinite(scaled) & np.isfinite(gt_a)
    if ok.sum() >= 5:
        err = scaled[ok] - gt_a[ok]
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        r    = float(np.corrcoef(scaled[ok], gt_a[ok])[0, 1])
        cov  = float(ok.sum() / len(preds) * 100)
    else:
        mae = rmse = bias = r = np.nan; cov = 0.0
    dt = time.time() - t0
    print(f'{m:<3d} | {k_dmd:5.3f}   {n_cal:3d}  | '
          f'{mae:5.2f} {rmse:5.2f}  {bias:+6.2f}  {r:+.3f}  {cov:5.1f} | {dt:6.1f}')
    results.append(dict(m=m, k_dmd=k_dmd, n_cal=n_cal,
                         MAE=mae, RMSE=rmse, bias=bias, r=r, cov=cov, time=dt))

df = pd.DataFrame(results)

print('\n==== Summary =============================================')
print(df.to_string(index=False))


# ── Plot trends ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))

axes[0].plot(df['m'], df['MAE'], 'o-', color='#2E86C1', lw=1.5)
axes[0].axhline(4.33, color='gray', ls='--', lw=1, label='Hilbert/k baseline (4.33)')
axes[0].axhline(6.33, color='gray', ls=':',  lw=1, label='DelayPCA baseline (6.33)')
axes[0].set_xlabel('embedding m'); axes[0].set_ylabel('MAE [BPM]')
axes[0].set_title('Whole-night MAE vs m  (after /k)')
axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

axes[1].plot(df['m'], df['cov'], 'o-', color='#27AE60', lw=1.5)
axes[1].set_xlabel('embedding m'); axes[1].set_ylabel('coverage [%]')
axes[1].set_title('Window coverage vs m')
axes[1].set_ylim(0, 105); axes[1].grid(alpha=0.3)

axes[2].plot(df['m'], df['k_dmd'], 'o-', color='#CA6F1E', lw=1.5)
axes[2].axhline(1.0, color='gray', ls=':', lw=1, label='k=1 (no scaling)')
axes[2].set_xlabel('embedding m'); axes[2].set_ylabel('learned k_dmd')
axes[2].set_title('Scaling factor k vs m')
axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

axes[3].plot(df['m'], df['r'], 'o-', color='#884EA0', lw=1.5)
axes[3].axhline(0.0, color='gray', ls=':', lw=1)
axes[3].set_xlabel('embedding m'); axes[3].set_ylabel('Pearson r vs GT')
axes[3].set_title('Per-window tracking r vs m')
axes[3].grid(alpha=0.3)

fig.suptitle(f'{session}  Hankel-DMD sweep  tau={TAU_S}s  cardiac band',
              fontsize=11)
fig.tight_layout()
out = OUT_DIR / 'dmd_rank_sweep_s1n1.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'\nplot -> {out}')
