"""
Figures for the spindle validation.
  fig_spindle_triggered_sigma.png : EEG (positive control) vs CAP sigma-envelope
                                    triggered averages -> EEG peaks, CAP flat.
  fig_spindle_auc.png             : per-session spindle-vs-control AUC by channel.
  fig_spindle_triggered_cardiac.png : CAP cardiac-envelope triggered average
                                    (autonomic-coupling probe).
Also prints a numeric test of any peri-spindle cardiac deflection.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
CHAN_COLORS = {'EEG': '#2C3E50', 'CLE-CRE': '#E67E22',
               'CLE': '#27AE60', 'CRE': '#8E44AD', 'CH': '#2980B9'}


def sem(a, axis=0):
    return np.nanstd(a, axis=axis) / np.sqrt(a.shape[axis])


def main():
    z = np.load(os.path.join(OUT, 'triggered_averages.npz'))
    t_sig, t_card = z['t_sig'], z['t_card']
    df = pd.read_csv(os.path.join(OUT, 'spindle_per_session.csv'))

    # ── Fig 1: sigma triggered average ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ch in ['EEG', 'CLE-CRE', 'CLE', 'CRE', 'CH']:
        curves = z[f'sig_{ch}']
        m, e = np.nanmean(curves, 0), sem(curves, 0)
        ax = axes[0] if ch == 'EEG' else axes[1]
        ax.plot(t_sig, m, color=CHAN_COLORS[ch], lw=1.8, label=ch)
        ax.fill_between(t_sig, m - e, m + e, color=CHAN_COLORS[ch], alpha=0.2)
    for ax, title in zip(axes, ['Contact EEG (positive control)',
                                'Capacitive mask channels']):
        ax.axvline(0, color='k', ls=':', lw=1)
        ax.set_xlabel('Time from spindle center (s)')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Sigma (11–16 Hz) envelope (z)')
    # share nothing: EEG dwarfs CAP; keep CAP axis readable
    cap_curves = np.concatenate([z[f'sig_{c}'] for c in ['CLE-CRE', 'CLE', 'CRE', 'CH']])
    axes[1].set_ylim(np.nanmean(cap_curves) - 0.15, np.nanmean(cap_curves) + 0.25)
    fig.suptitle('Spindle-triggered sigma-band envelope: EEG detects spindles, capacitive mask does not',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_spindle_triggered_sigma.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 2: AUC by channel (per-session) ───────────────────────────────────
    order = ['EEG', 'CLE-CRE', 'CLE', 'CRE', 'CH']
    fig, ax = plt.subplots(figsize=(7, 4.4))
    data = [df[df.channel == c]['sigma_auc'].values for c in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], order):
        patch.set_facecolor(CHAN_COLORS[c]); patch.set_alpha(0.5)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full_like(d, i) + np.random.uniform(-0.08, 0.08, len(d)),
                   d, color='k', s=14, zorder=3)
    ax.axhline(0.5, color='r', ls='--', lw=1, label='chance')
    ax.set_ylabel('Spindle-vs-control AUC (sigma power)')
    ax.set_title('Sigma-band spindle detectability by channel (12 sessions)')
    ax.legend(); ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_spindle_auc.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 3: cardiac-band triggered average (autonomic probe) ──────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    print('\n=== Peri-spindle cardiac-envelope deflection (CAP channels) ===')
    print('baseline = |t|>5s ; peak = |t|<=2s ; values in z-units, mean+-sd over 12 sessions')
    for ch in ['CLE-CRE', 'CLE', 'CRE', 'CH']:
        curves = z[f'card_{ch}']
        m, e = np.nanmean(curves, 0), sem(curves, 0)
        ax.plot(t_card, m, color=CHAN_COLORS[ch], lw=1.6, label=ch)
        ax.fill_between(t_card, m - e, m + e, color=CHAN_COLORS[ch], alpha=0.15)
        base = curves[:, np.abs(t_card) > 5]
        peak = curves[:, np.abs(t_card) <= 2]
        defl = np.nanmean(peak, 1) - np.nanmean(base, 1)   # per-session
        print(f'  {ch:8s} deflection = {defl.mean():+.4f} +-{defl.std():.4f} '
              f'(sessions with |d|>0.05: {np.sum(np.abs(defl) > 0.05)}/12)')
    ax.axvline(0, color='k', ls=':', lw=1)
    ax.set_xlabel('Time from spindle center (s)')
    ax.set_ylabel('Cardiac (0.5–3 Hz) envelope (z)')
    ax.set_title('Spindle-triggered CAP cardiac-pulsation envelope')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_spindle_triggered_cardiac.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved 3 figures to {OUT}')


if __name__ == '__main__':
    main()
