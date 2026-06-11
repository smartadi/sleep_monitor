"""
Visualisation helpers.

All functions accept matplotlib Axes objects so callers control
figure layout. Figure-level functions return the Figure.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import (
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER,
    CAP_CHANS, CAP_COLORS, GT_COLOR,
    RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    METHOD_NAMES, METHOD_COLORS, METHOD_LABELS,
    EEG_BANDS, BAND_COLORS,
)
from .sessions import SleepSession
from .rates import rate_acf, detect_peaks


# ── Sleep staging ──────────────────────────────────────────────────────────────

def plot_hypnogram(sp: dict, ax: plt.Axes, title: str = '') -> None:
    """
    Plot a hypnogram (sleep stages over time) onto ax.

    Parameters
    ----------
    sp    : dict from load_sleep_profile() — keys: t_ep_hr, labels, codes
    ax    : matplotlib Axes
    title : optional subplot title
    """
    t     = sp['t_ep_hr']
    codes = sp['codes']
    for i in range(len(t) - 1):
        c  = int(codes[i])
        clr = STAGE_COLORS.get(c, '#AAAAAA')
        ax.fill_between([t[i], t[i + 1]], c, c, color=clr, alpha=0.85, linewidth=0)
        ax.plot([t[i], t[i + 1]], [c, c], color=clr, lw=1.5)
    ax.set_yticks(STAGE_ORDER)
    ax.set_yticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], fontsize=7)
    ax.set_xlabel('Time (hr)', fontsize=7)
    ax.set_ylim(-0.5, 4.5)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=8)


# ── Raw signal overview ────────────────────────────────────────────────────────

def plot_session_overview(session: SleepSession, axes: np.ndarray) -> None:
    """
    Plot raw CAP channels + PSG overview for one session.

    Parameters
    ----------
    session : SleepSession
    axes    : 1-D array of at least 4 matplotlib Axes
              Row order: CH | CLE | CRE | PSG(Thorax+Pleth)
    """
    t   = session.time_hr
    cap = session.cap
    psg = session.psg

    def _zsc(x: np.ndarray) -> np.ndarray:
        mu, sd = np.nanmean(x), np.nanstd(x) + 1e-12
        return (x - mu) / sd

    ds   = max(1, len(t) // 5000)   # downsample for speed
    t_ds = t[::ds]

    for ax, ch in zip(axes[:3], ['CH', 'CLE', 'CRE']):
        ax.plot(t_ds, _zsc(cap[ch][::ds]), lw=0.4, color=CAP_COLORS[ch], alpha=0.7)
        ax.set_ylabel(ch, fontsize=7)
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)

    ax4 = axes[3]
    ax4.plot(t_ds, _zsc(psg['Thorax'][::ds]),
             lw=0.5, color='#27AE60', alpha=0.8, label='Thorax')
    ax4.plot(t_ds, _zsc(psg['Pleth'][::ds]) + 3,
             lw=0.5, color='#E74C3C', alpha=0.8, label='Pleth')
    ax4.legend(fontsize=6, loc='upper right')
    ax4.set_ylabel('PSG', fontsize=7)
    ax4.set_yticks([])
    ax4.set_xlabel('Time (hr)', fontsize=7)
    ax4.grid(True, alpha=0.2)

    axes[0].set_title(
        f"{session.label} {session.subject} ({session.duration_hr:.1f} hr)", fontsize=8
    )


def plot_all_sessions_grid(
    sessions: List[SleepSession],
    figsize: tuple = (18, 2.0),
) -> plt.Figure:
    """
    One row per session: CH | CLE | CRE | PSG overview.

    Parameters
    ----------
    sessions : list of SleepSession objects
    figsize  : (width, height_per_row)
    """
    n = len(sessions)
    fig, axes = plt.subplots(
        n, 4,
        figsize=(figsize[0], figsize[1] * n),
        sharex=False,
        gridspec_kw={'hspace': 0.6, 'wspace': 0.15},
    )
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, sess in enumerate(sessions):
        plot_session_overview(sess, axes[i])
    plt.suptitle('All Sessions — Raw Signal Overview', fontsize=12, y=1.005)
    return fig


# ── Rate estimation results ────────────────────────────────────────────────────

def plot_rates_vs_gt(
    t_hr: np.ndarray,
    cap_rates: dict,
    gt_t_hr: np.ndarray,
    gt_hz: np.ndarray,
    band: str,
    ax: plt.Axes,
    channel: str = 'CH',
    methods: Optional[List[str]] = None,
) -> None:
    """
    Overlay per-method sliding rates against PSG ground truth.

    Parameters
    ----------
    t_hr      : (K,) time axis for CAP rates in hours
    cap_rates : {method: (K,) rates in Hz}
    gt_t_hr   : (M,) PSG GT time axis in hours
    gt_hz     : (M,) PSG GT rate in Hz
    band      : 'resp' or 'card'
    ax        : matplotlib Axes
    channel   : CAP channel name (for title)
    methods   : which methods to plot (default: all)
    """
    scale = 60.0
    unit  = 'br/min' if band == 'resp' else 'BPM'
    if methods is None:
        methods = METHOD_NAMES

    ax.plot(gt_t_hr, gt_hz * scale,
            color=GT_COLOR, lw=1.5, alpha=0.9, label='PSG GT', zorder=3)
    for method in methods:
        r = cap_rates.get(method)
        if r is None:
            continue
        ax.plot(t_hr, r * scale,
                color=METHOD_COLORS[method], lw=0.8, alpha=0.7,
                label=METHOD_LABELS[method])
    ax.set_ylabel(unit, fontsize=8)
    ax.legend(fontsize=6.5, loc='upper right', ncol=3)
    ax.grid(True, alpha=0.25)
    ax.set_title(f'{channel} — {band}', fontsize=9)


# ── Window inspection ──────────────────────────────────────────────────────────

def plot_window_inspection(
    win: dict,
    t_s: np.ndarray,
    session: SleepSession,
    start_hr: float,
    method: str = 'peaks',
) -> plt.Figure:
    """
    5-row inspection plot: CH | CLE | CRE | CLE-CRE | GT
    Two columns: resp band | cardiac band.

    Parameters
    ----------
    win      : output of preprocess_window()
    t_s      : time axis in seconds within the window
    session  : SleepSession (for title)
    start_hr : window start in hours (for title)
    method   : peak detection method
    """
    from .rates import peaks_by_method, rate_acf

    def _zsc(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    ROWS = CAP_CHANS + ['GT']
    BAND_SPECS = [
        ('resp', RESP_LO, RESP_HI, win['gt_resp'], win['gt_thorax_raw'], 'GT Thorax', 'br/min'),
        ('card', CARD_LO, CARD_HI, win['gt_card'], win['gt_pleth_raw'],  'GT Pleth',  'BPM'),
    ]

    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
    fig, axes = plt.subplots(
        len(ROWS), 2,
        figsize=(15, 2.8 * len(ROWS)),
        sharex=True,
        gridspec_kw={'hspace': 0.45, 'wspace': 0.28},
    )
    raw_z = {ch: _zsc(win['raw'][ch]) for ch in CAP_CHANS}

    for ri, row_id in enumerate(ROWS):
        is_gt = (row_id == 'GT')
        color = GT_COLOR if is_gt else CAP_COLORS[row_id]

        for ci, (band, f_lo, f_hi, gt_bp, gt_raw, gt_lbl, unit) in enumerate(BAND_SPECS):
            ax  = axes[ri, ci]
            gt_pks = peaks_by_method(gt_bp, f_lo, f_hi, method)
            gt_filt_z = _zsc(gt_bp)

            if is_gt:
                ax.plot(t_s, _zsc(gt_raw),   color='#AAAAAA', lw=0.7, alpha=0.7,  label='raw (z)')
                ax.plot(t_s, gt_filt_z,       color=GT_COLOR,  lw=1.2, alpha=0.95, label=gt_lbl)
                ax.plot(t_s[gt_pks], gt_filt_z[gt_pks], '^',
                        color=GT_COLOR, ms=8, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.6,
                        label=f'GT peaks  n={len(gt_pks)}')
                r_gt  = rate_acf(gt_bp, f_lo, f_hi)
                r_str = f'{r_gt * 60:.1f} {unit}' if not np.isnan(r_gt) else 'n/a'
                ax.text(0.01, 0.97, f'ACF: {r_str}   peaks: {len(gt_pks)}',
                        transform=ax.transAxes, va='top', fontsize=7.5,
                        bbox=dict(facecolor='white', alpha=0.78, pad=2, edgecolor='none'))
                ax.set_ylabel(f'{gt_lbl}\nNorm. amp.', fontsize=8)
            else:
                cap_filt   = win['sigs'][row_id][band]
                cap_filt_z = _zsc(cap_filt)
                cap_pks    = detect_peaks(cap_filt, f_lo, f_hi)

                ax.plot(t_s, raw_z[row_id],  color='#CCCCCC', lw=0.6, alpha=0.85, label='raw (z)')
                ax.plot(t_s, cap_filt_z,     color=color,     lw=1.1, alpha=0.92,
                        label=f'{row_id} filtered')
                ax.plot(t_s[cap_pks], cap_filt_z[cap_pks], 'v',
                        color=color, ms=8, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.6,
                        label=f'cap peaks  n={len(cap_pks)}')
                for pk in gt_pks:
                    ax.axvline(t_s[pk], color=GT_COLOR, lw=0.7, alpha=0.28, zorder=1)

                r_cap = rate_acf(cap_filt, f_lo, f_hi)
                r_gt  = rate_acf(gt_bp, f_lo, f_hi)
                c_s   = f'{r_cap * 60:.1f}' if not np.isnan(r_cap) else 'n/a'
                g_s   = f'{r_gt  * 60:.1f}' if not np.isnan(r_gt)  else 'n/a'
                ax.text(0.01, 0.97,
                        f'cap ACF: {c_s} {unit}   GT ACF: {g_s} {unit}   cap peaks: {len(cap_pks)}',
                        transform=ax.transAxes, va='top', fontsize=7.5,
                        bbox=dict(facecolor='white', alpha=0.78, pad=2, edgecolor='none'))
                ax.set_ylabel(f'{row_id}\nNorm. amp.', fontsize=8)

            ax.axhline(0, color='gray', lw=0.4, alpha=0.35)
            ax.legend(fontsize=6.5, loc='lower right', ncol=3)
            if ri == 0:
                band_lbl = 'Respiratory' if ci == 0 else 'Cardiac'
                ax.set_title(f'{band_lbl} band  ({f_lo}–{f_hi} Hz)', fontsize=9, fontweight='bold')

    for ci in range(2):
        axes[-1, ci].set_xlabel('Time in window (s)', fontsize=8)

    t0m = start_hr * 60
    t1m = (start_hr + (t_s[-1] / 3600.0)) * 60
    m = session.meta
    fig.suptitle(
        f"Window inspection  |  {m['label']} {m['subject']}-{m['initials']} {m['date']}  "
        f"| {t0m:.1f}–{t1m:.1f} min  |  grey=raw  colour=filtered  "
        "\u25bd=cap peaks  \u25b3=GT peaks  dashed=GT peak times",
        fontsize=10,
    )
    plt.tight_layout()
    return fig


# ── EEG spectrogram ────────────────────────────────────────────────────────────

def plot_eeg_spectrogram(
    eeg: np.ndarray,
    fs: float,
    ax: plt.Axes,
    f_max: float = 35.0,
    title: str = '',
) -> None:
    """
    Plot EEG spectrogram (magnitude) using scipy.signal.spectrogram.
    """
    from scipy.signal import spectrogram
    f, t, Sxx = spectrogram(eeg.astype(np.float64), fs=fs,
                             nperseg=int(fs * 4), noverlap=int(fs * 3),
                             scaling='density')
    mask = f <= f_max
    ax.pcolormesh(t / 3600.0, f[mask], 10 * np.log10(Sxx[mask] + 1e-12),
                  shading='gouraud', cmap='inferno', rasterized=True)
    ax.set_ylabel('Frequency (Hz)', fontsize=7)
    ax.set_xlabel('Time (hr)', fontsize=7)
    if title:
        ax.set_title(title, fontsize=8)
    # Annotate band boundaries
    for band, (flo, fhi) in EEG_BANDS.items():
        if fhi <= f_max:
            ax.axhline(fhi, color=BAND_COLORS[band], lw=0.6, alpha=0.6, ls='--')


# ── Validation study plots ────────────────────────────────────────────────────

def plot_bland_altman(
    pred: np.ndarray,
    ref: np.ndarray,
    ax: plt.Axes,
    scale: float = 60.0,
    unit: str = 'BPM',
    title: str = '',
    color: str = '#2980B9',
    stage_codes: Optional[np.ndarray] = None,
) -> dict:
    """
    Bland-Altman plot: (pred+ref)/2 on x, pred-ref on y, with limits of agreement.

    Returns dict with bias, sd, loa_lo, loa_hi for annotation.
    """
    ok = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[ok] * scale, ref[ok] * scale
    mean_val = (p + r) / 2.0
    diff = p - r
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd

    if stage_codes is not None:
        codes = stage_codes[ok]
        for code in STAGE_ORDER:
            mask = codes == code
            if mask.any():
                ax.scatter(mean_val[mask], diff[mask], s=6, alpha=0.35,
                           color=STAGE_COLORS.get(code, '#AAAAAA'),
                           label=STAGE_LABELS.get(code, '?'), rasterized=True)
        ax.legend(fontsize=6, loc='upper right', markerscale=2, ncol=2)
    else:
        ax.scatter(mean_val, diff, s=6, alpha=0.25, color=color, rasterized=True)

    ax.axhline(bias, color='#E74C3C', lw=1.2, ls='-', label=f'Bias {bias:+.2f}')
    ax.axhline(loa_hi, color='#E74C3C', lw=0.8, ls='--', alpha=0.7)
    ax.axhline(loa_lo, color='#E74C3C', lw=0.8, ls='--', alpha=0.7)
    ax.fill_between(ax.get_xlim(), loa_lo, loa_hi, color='#E74C3C', alpha=0.05)
    ax.set_xlabel(f'Mean of CAP & GT ({unit})', fontsize=8)
    ax.set_ylabel(f'CAP − GT ({unit})', fontsize=8)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=9)
    ax.text(0.02, 0.03,
            f'Bias={bias:+.2f}  SD={sd:.2f}\nLoA=[{loa_lo:.2f}, {loa_hi:+.2f}]',
            transform=ax.transAxes, fontsize=7, va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    return dict(bias=bias, sd=sd, loa_lo=loa_lo, loa_hi=loa_hi)


def plot_scatter_agreement(
    pred: np.ndarray,
    ref: np.ndarray,
    ax: plt.Axes,
    scale: float = 60.0,
    unit: str = 'BPM',
    title: str = '',
    color: str = '#2980B9',
    stage_codes: Optional[np.ndarray] = None,
) -> None:
    """Scatter plot of CAP rate vs GT rate with identity line."""
    ok = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[ok] * scale, ref[ok] * scale

    if stage_codes is not None:
        codes = stage_codes[ok]
        for code in STAGE_ORDER:
            mask = codes == code
            if mask.any():
                ax.scatter(r[mask], p[mask], s=6, alpha=0.35,
                           color=STAGE_COLORS.get(code, '#AAAAAA'),
                           label=STAGE_LABELS.get(code, '?'), rasterized=True)
        ax.legend(fontsize=6, loc='upper left', markerscale=2, ncol=2)
    else:
        ax.scatter(r, p, s=6, alpha=0.25, color=color, rasterized=True)

    lo = min(r.min(), p.min()) * 0.95
    hi = max(r.max(), p.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'GT ({unit})', fontsize=8)
    ax.set_ylabel(f'CAP ({unit})', fontsize=8)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=9)


def plot_stage_boxplots(
    df: 'pd.DataFrame',
    value_col: str,
    ax: plt.Axes,
    scale: float = 60.0,
    unit: str = 'BPM',
    title: str = '',
) -> None:
    """
    Box plot of absolute error by sleep stage.

    df must have columns: stage_code, and the two rate columns to diff.
    value_col: 'resp' or 'card' — selects cap_{}_hz and gt_{}_hz columns.
    """
    import pandas as pd

    cap_col = f'cap_{value_col}_hz'
    gt_col = f'gt_{value_col}_hz'

    ok = df[cap_col].notna() & df[gt_col].notna() & (df['stage_code'] >= 0)
    d = df.loc[ok].copy()
    d['abs_err'] = (d[cap_col] - d[gt_col]).abs() * scale

    stage_order = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_data = []
    stage_labels_plot = []
    stage_colors_plot = []
    code_for_name = {v: k for k, v in STAGE_LABELS.items() if k >= 0}

    for name in stage_order:
        code = code_for_name.get(name)
        if code is None:
            continue
        vals = d.loc[d['stage_code'] == code, 'abs_err'].values
        if len(vals) > 0:
            stage_data.append(vals)
            n = len(vals)
            med = np.median(vals)
            stage_labels_plot.append(f'{name}\n(n={n})\nmed={med:.1f}')
            stage_colors_plot.append(STAGE_COLORS.get(code, '#AAAAAA'))

    bp = ax.boxplot(stage_data, labels=stage_labels_plot, patch_artist=True,
                    showfliers=False, widths=0.6,
                    medianprops=dict(color='black', lw=1.5))
    for patch, c in zip(bp['boxes'], stage_colors_plot):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_ylabel(f'Absolute Error ({unit})', fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)
    if title:
        ax.set_title(title, fontsize=9)


def plot_session_bar(
    sess_df: 'pd.DataFrame',
    metric: str,
    ax: plt.Axes,
    unit: str = '',
    title: str = '',
    color: str = '#3498DB',
) -> None:
    """Bar chart of a per-session metric (e.g. MAE) with aggregate dashed line."""
    data = sess_df[sess_df['session'] != 'ALL']
    agg = sess_df[sess_df['session'] == 'ALL']

    x = np.arange(len(data))
    ax.bar(x, data[metric], color=color, alpha=0.7, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(data['session'], fontsize=7, rotation=45, ha='right')

    if len(agg) > 0:
        agg_val = agg[metric].iloc[0]
        ax.axhline(agg_val, color='#E74C3C', lw=1.2, ls='--',
                   label=f'Mean={agg_val:.2f}')
        ax.legend(fontsize=7)

    ax.set_ylabel(f'{metric} ({unit})' if unit else metric, fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)
    if title:
        ax.set_title(title, fontsize=9)
