"""
Moving-window band power ratio analysis for all 12 CAP sessions.

1) EEG-band power ratios (delta, theta, alpha, beta) with motion masking
2) Delta sub-band ratios (infra-slow, SO, delta-low, delta-high) within delta

Output: notebooks/plots/sws_band_ratios/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.signal import spectrogram as scipy_spectrogram

from sleep_monitor import (
    load_session, SESSION_META, FS,
    EEG_BANDS, BAND_COLORS,
    DELTA_SUB_BANDS, DELTA_SUB_COLORS,
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER,
    APNEA_LABELS, APNEA_COLORS,
)
from sleep_monitor.loader import load_apnea_events
from sleep_monitor.spectral import compute_band_power_ratios

OUT_DIR = Path(__file__).parent / 'plots' / 'sws_band_ratios'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAP_CHS = ['CH', 'CLE', 'CRE']
BAND_NAMES = list(EEG_BANDS.keys())
DELTA_SUB_NAMES = list(DELTA_SUB_BANDS.keys())

SMOOTH_PTS = 12  # 12 × 10s step = 2-min causal moving average


def _smooth(arr, n=SMOOTH_PTS):
    """Causal moving average, NaN-aware."""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        window = arr[max(0, i - n + 1):i + 1]
        valid = window[np.isfinite(window)]
        out[i] = np.mean(valid) if len(valid) > 0 else np.nan
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────


def _plot_hypnogram(ax, sp, title):
    t_ep = sp['t_ep_hr']
    codes = sp['codes']
    epoch_dur = t_ep[1] - t_ep[0] if len(t_ep) > 1 else 30.0 / 3600.0
    for i in range(len(t_ep)):
        c = int(codes[i])
        clr = STAGE_COLORS.get(c, '#AAAAAA')
        ax.barh(c, epoch_dur, left=t_ep[i], height=0.8, color=clr, alpha=0.85,
                edgecolor='none')
    ax.set_yticks(STAGE_ORDER)
    ax.set_yticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], fontsize=8)
    ax.set_ylim(-0.5, 4.5)
    ax.invert_yaxis()
    ax.set_ylabel('Sleep stage', fontsize=9)
    ax.grid(True, axis='x', alpha=0.25)
    ax.set_title(title, fontsize=11, fontweight='bold')
    return epoch_dur



def _plot_apnea_row(ax, apnea_events):
    """Draw apnea/hypopnea events as colored bars on a dedicated row."""
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Apnea', 'Hypo'], fontsize=7)
    ax.set_ylim(0.4, 2.6)
    ax.invert_yaxis()
    ax.set_ylabel('Resp\nevents', fontsize=8)
    ax.grid(True, axis='x', alpha=0.25)
    if apnea_events is None:
        return
    for s, e, code in zip(apnea_events['start_hr'], apnea_events['end_hr'],
                          apnea_events['codes']):
        clr = APNEA_COLORS.get(code, '#E74C3C')
        ax.barh(code, e - s, left=s, height=0.7, color=clr, alpha=0.85,
                edgecolor='none')


SPEC_NPERSEG = 30 * 100  # 30s segments for spectrogram
SPEC_FMAX = 5.0


def _plot_spectrogram_row(ax, sig, fs, xlim_hr):
    """Plot a 0–5 Hz spectrogram of sig on ax, x-axis in hours.
    Each time column is independently normalized so low-signal
    periods still show spectral shape instead of going black."""
    f, t, Sxx = scipy_spectrogram(sig.astype(np.float64), fs=fs,
                                  nperseg=SPEC_NPERSEG,
                                  noverlap=SPEC_NPERSEG // 2,
                                  scaling='density')
    fmask = f <= SPEC_FMAX
    t_hr = t / 3600.0
    Sxx_sub = Sxx[fmask]
    Sxx_db = 10 * np.log10(Sxx_sub + 1e-20)
    col_min = np.nanmin(Sxx_db, axis=0, keepdims=True)
    col_max = np.nanmax(Sxx_db, axis=0, keepdims=True)
    col_range = col_max - col_min
    col_range[col_range < 1e-6] = 1.0
    Sxx_norm = (Sxx_db - col_min) / col_range
    ax.pcolormesh(t_hr, f[fmask], Sxx_norm, shading='gouraud',
                  cmap='inferno', vmin=0, vmax=1, rasterized=True)
    ax.set_ylabel(f'Freq\n(0–{SPEC_FMAX:.0f} Hz)', fontsize=8)
    ax.set_ylim(0, SPEC_FMAX)
    ax.grid(False)


def _collect_ratios_by_stage(t_hr, ratios_dict, band_names, sp, motion_mask):
    """Map each window to its PSG epoch and collect per-stage band ratios."""
    t_ep = sp['t_ep_hr']
    codes = sp['codes']
    out = {s: {b: [] for b in band_names} for s in STAGE_ORDER}
    for i, tw in enumerate(t_hr):
        if motion_mask[i]:
            continue
        ep_idx = int(np.clip(np.searchsorted(t_ep, tw, side='right') - 1,
                             0, len(codes) - 1))
        stage = int(codes[ep_idx])
        if stage not in out:
            continue
        for b in band_names:
            v = ratios_dict[b][i]
            if np.isfinite(v):
                out[stage][b].append(v)
    return out


# ── Per-session EEG band plot ─────────────────────────────────────────────────

def plot_eeg_bands(session, band_data, sp, apnea_events, cap_sig, out_path):
    n_rows = 6  # hypnogram, apnea, spectrogram, CH, CLE, CRE
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={
                                 'height_ratios': [0.8, 0.5, 1.2, 1.2, 1.2, 1.2],
                                 'hspace': 0.10})
    n_apnea = len(apnea_events['codes']) if apnea_events else 0
    title = (f'{session.label}  ({session.subject}-{session.meta["initials"]}  '
             f'{session.meta["date"]}  {session.duration_hr:.1f} hr)  '
             f'[2-min smooth, {n_apnea} resp events]')
    t_ep = sp['t_ep_hr']
    epoch_dur = _plot_hypnogram(axes[0], sp, title)
    _plot_apnea_row(axes[1], apnea_events)
    _plot_spectrogram_row(axes[2], cap_sig, FS,
                          (t_ep[0], t_ep[-1] + epoch_dur))

    for ci, ch in enumerate(CAP_CHS):
        ax = axes[ci + 3]
        d = band_data[ch]
        t_hr = d['t_hr']
        for band in BAND_NAMES:
            ax.plot(t_hr, _smooth(d[band]), color=BAND_COLORS[band], lw=1.0,
                    alpha=0.85,
                    label=f'{band} ({EEG_BANDS[band][0]}-{EEG_BANDS[band][1]} Hz)')
        pct = d['motion_mask'].sum() / len(d['motion_mask']) * 100
        ax.set_ylabel(f'{ch}\nband ratio', fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.text(0.99, 0.97, f'motion-masked: {pct:.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        if ci == 0:
            ax.legend(fontsize=7, loc='upper left', ncol=4, framealpha=0.8)

    axes[-1].set_xlabel('Time (hr)', fontsize=10)
    axes[-1].set_xlim(t_ep[0], t_ep[-1] + epoch_dur)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


# ── Per-session delta sub-band plot ───────────────────────────────────────────

def plot_delta_subbands(session, delta_data, sp, apnea_events, cap_sig, out_path):
    n_rows = 6
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={
                                 'height_ratios': [0.8, 0.5, 1.2, 1.2, 1.2, 1.2],
                                 'hspace': 0.10})
    n_apnea = len(apnea_events['codes']) if apnea_events else 0
    title = (f'{session.label}  delta sub-bands  '
             f'({session.subject}-{session.meta["initials"]}  '
             f'{session.meta["date"]}  {session.duration_hr:.1f} hr)  '
             f'[2-min smooth, {n_apnea} resp events]')
    t_ep = sp['t_ep_hr']
    epoch_dur = _plot_hypnogram(axes[0], sp, title)
    _plot_apnea_row(axes[1], apnea_events)
    _plot_spectrogram_row(axes[2], cap_sig, FS,
                          (t_ep[0], t_ep[-1] + epoch_dur))

    for ci, ch in enumerate(CAP_CHS):
        ax = axes[ci + 3]
        d = delta_data[ch]
        t_hr = d['t_hr']
        for band in DELTA_SUB_NAMES:
            flo, fhi = DELTA_SUB_BANDS[band]
            ax.plot(t_hr, _smooth(d[band]), color=DELTA_SUB_COLORS[band],
                    lw=1.0, alpha=0.85, label=f'{band} ({flo}-{fhi} Hz)')
        ax.set_ylabel(f'{ch}\nsub-band ratio', fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        if ci == 0:
            ax.legend(fontsize=7, loc='upper left', ncol=4, framealpha=0.8)

    axes[-1].set_xlabel('Time (hr)', fontsize=10)
    axes[-1].set_xlim(t_ep[0], t_ep[-1] + epoch_dur)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


# ── Summary bar chart ────────────────────────────────────────────────────────

def plot_summary_bar(stage_data, band_names, band_colors, ylabel, title, out_path):
    n_bands = len(band_names)
    n_stages = len(STAGE_ORDER)
    width = 0.8 / n_bands
    x = np.arange(n_stages)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for bi, band in enumerate(band_names):
        means, stds = [], []
        for s in STAGE_ORDER:
            vals = stage_data[s][band]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        offset = (bi - n_bands / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, color=band_colors[band],
               alpha=0.8, capsize=3, edgecolor='white', label=band)

    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('Sleep stage', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_eeg_by_stage = {s: {b: [] for b in BAND_NAMES} for s in STAGE_ORDER}
    all_delta_by_stage = {s: {b: [] for b in DELTA_SUB_NAMES} for s in STAGE_ORDER}

    for idx in range(len(SESSION_META)):
        session = load_session(idx, with_profile=True)
        sp = session.sleep_profile
        if sp is None:
            print(f'  skipping {session.label} — no sleep profile')
            continue

        apnea = load_apnea_events(session)
        n_ev = len(apnea['codes']) if apnea else 0
        print(f'  {session.label}: {n_ev} apnea/hypopnea events')

        acc = session.cap['acc_mag']

        # EEG-band ratios (0.5–30 Hz denominator)
        eeg_data = {}
        for ch in CAP_CHS:
            eeg_data[ch] = compute_band_power_ratios(
                session.cap[ch], fs=FS, acc_mag=acc)
        cap_sig = session.cap['CH']

        plot_eeg_bands(session, eeg_data, sp, apnea, cap_sig,
                       OUT_DIR / f'{session.label}_band_ratios.png')

        # Delta sub-band ratios (0–4 Hz denominator)
        delta_data = {}
        for ch in CAP_CHS:
            delta_data[ch] = compute_band_power_ratios(
                session.cap[ch], fs=FS, acc_mag=acc,
                bands=DELTA_SUB_BANDS, total_range=(0.0, 4.0))
        plot_delta_subbands(session, delta_data, sp, apnea, cap_sig,
                            OUT_DIR / f'{session.label}_delta_subbands.png')

        # Accumulate CH ratios by stage
        ch_eeg = eeg_data['CH']
        stage_eeg = _collect_ratios_by_stage(
            ch_eeg['t_hr'], ch_eeg, BAND_NAMES, sp, ch_eeg['motion_mask'])
        for s in STAGE_ORDER:
            for b in BAND_NAMES:
                all_eeg_by_stage[s][b].extend(stage_eeg[s][b])

        ch_delta = delta_data['CH']
        stage_delta = _collect_ratios_by_stage(
            ch_delta['t_hr'], ch_delta, DELTA_SUB_NAMES, sp, ch_delta['motion_mask'])
        for s in STAGE_ORDER:
            for b in DELTA_SUB_NAMES:
                all_delta_by_stage[s][b].extend(stage_delta[s][b])

    # Summary plots
    plot_summary_bar(
        all_eeg_by_stage, BAND_NAMES, BAND_COLORS,
        'Mean power ratio (CH)',
        'EEG band power ratios by sleep stage — all sessions (motion-masked)',
        OUT_DIR / 'summary_eeg_bands_by_stage.png')

    plot_summary_bar(
        all_delta_by_stage, DELTA_SUB_NAMES, DELTA_SUB_COLORS,
        'Mean sub-band ratio within 0–4 Hz (CH)',
        'Delta sub-band ratios by sleep stage — all sessions (motion-masked)',
        OUT_DIR / 'summary_delta_subbands_by_stage.png')

    print('Done.')


if __name__ == '__main__':
    main()
