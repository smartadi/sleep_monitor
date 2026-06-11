"""
Plot raw CAP + PSG time-series for the full night with detected trials highlighted.

One plot per session showing the entire recording with trial regions shaded:
  - Sleep stage ribbon (PSG ground truth)
  - CLE-CRE differential (raw + DC mean overlay)
  - CLE, CRE individual DC traces
  - Accelerometer magnitude (RMS envelope)
  - PSG Thorax (resp band)
  - PSG EEG (raw + delta envelope)
  - PSG Flow (resp band)
  - PSG Pleth / PPG (cardiac envelope)
  - CLE-CRE spectrogram (0-5 Hz)

Output: reports/slow_wave/trials/<session>/fullnight_signals.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sleep_monitor import load_session, load_sleep_profile
from sleep_monitor.config import (
    FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI,
    PSG_EPOCH_SEC, STAGE_LABELS, STAGE_COLORS,
)
from sleep_monitor.filters import bandpass, lowpass
from sleep_monitor.sessions import SESSION_META

from detect_trials import compute_epoch_traces, apply_criteria, extract_trials

REPORT_DIR = Path(__file__).resolve().parents[2] / 'reports' / 'slow_wave' / 'trials'


def _get_stage_blocks(sleep_profile):
    if sleep_profile is None:
        return []
    t_ep = sleep_profile['t_ep_hr']
    codes = sleep_profile['codes']
    epoch_hr = PSG_EPOCH_SEC / 3600.0
    return [(float(t_ep[i]), float(t_ep[i]) + epoch_hr, int(codes[i]))
            for i in range(len(codes))]


def plot_fullnight(session, trials):
    """Full-night raw signal plot with trial regions highlighted."""
    fs = session.fs
    t_hr = session.time_hr.astype(np.float64)
    n = len(t_hr)
    label = session.label

    # Heavy downsample for raw waveforms (100 Hz → ~2 Hz)
    ds = 50
    t_ds = t_hr[::ds]

    cle = session.cap['CLE'].astype(np.float64)
    cre = session.cap['CRE'].astype(np.float64)
    diff = cle - cre
    acc = session.cap['acc_mag'].astype(np.float64)
    thorax = session.psg['Thorax'].astype(np.float64)
    eeg = session.psg['EEG'].astype(np.float64)
    flow = session.psg['Flow'].astype(np.float64)
    pleth = session.psg['Pleth'].astype(np.float64)

    # Precompute envelopes / filtered signals
    diff_dc = lowpass(diff, 0.02, fs)
    cle_dc = lowpass(cle, 0.02, fs)
    cre_dc = lowpass(cre, 0.02, fs)
    thorax_resp = bandpass(thorax, RESP_LO, RESP_HI, fs)
    flow_resp = bandpass(flow, RESP_LO, RESP_HI, fs)
    eeg_delta = bandpass(eeg, 0.5, 4.0, fs)
    pleth_card = bandpass(pleth, CARD_LO, CARD_HI, fs)

    # RMS envelopes (30s windows, step 15s — matches epoch scale)
    env_win = int(30 * fs)
    env_step = int(15 * fs)
    env_centers = np.arange(env_win // 2, n - env_win // 2, env_step)
    t_env = t_hr[env_centers]

    def rms_envelope(sig):
        out = np.empty(len(env_centers))
        for j, c in enumerate(env_centers):
            chunk = sig[c - env_win // 2: c + env_win // 2]
            out[j] = np.sqrt(np.mean(chunk ** 2))
        return out

    acc_ac = acc - np.mean(acc)
    acc_env = rms_envelope(acc_ac)
    eeg_delta_env = rms_envelope(eeg_delta)
    pleth_env = rms_envelope(pleth_card)
    thorax_env = rms_envelope(thorax_resp)

    # ── Figure ──
    n_panels = 9
    fig = plt.figure(figsize=(28, 26))
    gs = GridSpec(n_panels, 1, figure=fig, hspace=0.22,
                  height_ratios=[0.3, 1.0, 0.6, 0.6, 0.8, 0.9, 0.6, 0.6, 1.2])

    trial_count = len(trials)
    n3_trials = sum(1 for t in trials if t['n3_fraction'] > 0)
    fig.suptitle(
        f'{label} — Full Night  |  {trial_count} trials (>= 5 min)  |  '
        f'{n3_trials} with N3 overlap  |  '
        f'duration {t_hr[-1]:.1f} hr',
        fontsize=14, fontweight='bold',
    )

    def shade_trials(ax):
        for i, trial in enumerate(trials):
            color = '#2ECC71' if trial['n3_fraction'] > 0.3 else '#3498DB'
            ax.axvspan(trial['t_start_hr'], trial['t_end_hr'],
                       color=color, alpha=0.18)
            # Label only on the top panel
            if ax == axes[0]:
                mid = (trial['t_start_hr'] + trial['t_end_hr']) / 2
                ax.text(mid, 1.15, f'T{i+1}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=color,
                        clip_on=False)

    axes = []

    # ── 0: Stage ribbon ──
    ax = fig.add_subplot(gs[0])
    axes.append(ax)
    blocks = _get_stage_blocks(session.sleep_profile)
    for bs, be, sc in blocks:
        color = STAGE_COLORS.get(sc, '#AAAAAA')
        alpha = 0.85 if sc == 1 else 0.45
        ax.axvspan(bs, be, color=color, alpha=alpha)
    shade_trials(ax)
    ax.set_ylabel('Stage', fontsize=9)
    ax.set_yticks([])

    # ── 1: CLE-CRE raw + DC ──
    ax = fig.add_subplot(gs[1])
    axes.append(ax)
    ax.plot(t_ds, diff[::ds], color='#BDC3C7', linewidth=0.08, alpha=0.4, rasterized=True)
    ax.plot(t_ds, diff_dc[::ds], color='#E67E22', linewidth=1.5, label='DC (0.02 Hz LP)')
    shade_trials(ax)
    ax.set_ylabel('CLE-CRE', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('CAP differential', fontsize=10, loc='left')

    # ── 2: CLE, CRE DC ──
    ax = fig.add_subplot(gs[2])
    axes.append(ax)
    ax.plot(t_ds, cle_dc[::ds], color='#27AE60', linewidth=1, label='CLE DC')
    ax.plot(t_ds, cre_dc[::ds], color='#8E44AD', linewidth=1, label='CRE DC')
    shade_trials(ax)
    ax.set_ylabel('CLE / CRE', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')

    # ── 3: Accelerometer RMS ──
    ax = fig.add_subplot(gs[3])
    axes.append(ax)
    ax.plot(t_env, acc_env, color='#E74C3C', linewidth=0.8)
    shade_trials(ax)
    ax.set_ylabel('Acc RMS', fontsize=9)
    ax.set_title('Accelerometer (motion level)', fontsize=10, loc='left')

    # ── 4: PSG Thorax ──
    ax = fig.add_subplot(gs[4])
    axes.append(ax)
    ax.plot(t_ds, thorax_resp[::ds], color='#8E44AD', linewidth=0.3, alpha=0.6, rasterized=True)
    ax2 = ax.twinx()
    ax2.plot(t_env, thorax_env, color='#C0392B', linewidth=1.2, alpha=0.8, label='Thorax RMS (30s)')
    ax2.set_ylabel('RMS', fontsize=8, color='#C0392B')
    ax2.tick_params(axis='y', labelcolor='#C0392B')
    shade_trials(ax)
    shade_trials(ax2)
    ax.set_ylabel('Thorax', fontsize=9)
    ax2.legend(fontsize=8, loc='upper right')
    ax.set_title('PSG Thorax (resp band + RMS envelope)', fontsize=10, loc='left')
    axes.append(ax2)

    # ── 5: PSG EEG ──
    ax = fig.add_subplot(gs[5])
    axes.append(ax)
    ax.plot(t_ds, eeg[::ds], color='#2C3E50', linewidth=0.1, alpha=0.5, rasterized=True)
    ax2 = ax.twinx()
    ax2.plot(t_env, eeg_delta_env, color='#C0392B', linewidth=1.2, alpha=0.8, label='Delta env')
    ax2.set_ylabel('Delta env', fontsize=8, color='#C0392B')
    ax2.tick_params(axis='y', labelcolor='#C0392B')
    shade_trials(ax)
    shade_trials(ax2)
    ax.set_ylabel('EEG', fontsize=9)
    ax2.legend(fontsize=8, loc='upper right')
    ax.set_title('PSG EEG (raw + delta 0.5-4 Hz envelope)', fontsize=10, loc='left')
    axes.append(ax2)

    # ── 6: PSG Flow ──
    ax = fig.add_subplot(gs[6])
    axes.append(ax)
    ax.plot(t_ds, flow_resp[::ds], color='#2980B9', linewidth=0.2, alpha=0.6, rasterized=True)
    shade_trials(ax)
    ax.set_ylabel('Flow', fontsize=9)
    ax.set_title('PSG Flow (resp band)', fontsize=10, loc='left')

    # ── 7: PSG Pleth / PPG ──
    ax = fig.add_subplot(gs[7])
    axes.append(ax)
    ax.plot(t_env, pleth_env, color='#1ABC9C', linewidth=1, label='PPG cardiac env')
    shade_trials(ax)
    ax.set_ylabel('Pleth', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('PSG Pleth / PPG (cardiac band envelope)', fontsize=10, loc='left')

    # ── 8: CLE-CRE spectrogram ──
    ax = fig.add_subplot(gs[8])
    axes.append(ax)
    nperseg_spec = int(30 * fs)
    noverlap_spec = int(25 * fs)
    f_spec, t_spec_s, Sxx = spectrogram(
        diff, fs=fs, nperseg=nperseg_spec, noverlap=noverlap_spec, scaling='density',
    )
    t_spec_hr = t_spec_s / 3600.0
    f_mask = f_spec <= 5.0
    Sxx_db = 10 * np.log10(Sxx[f_mask, :] + 1e-20)
    vmin, vmax = np.percentile(Sxx_db, [5, 95])
    ax.pcolormesh(t_spec_hr, f_spec[f_mask], Sxx_db, shading='gouraud',
                  cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
    shade_trials(ax)
    ax.set_ylabel('Freq (Hz)', fontsize=9)
    ax.set_xlabel('Time (hr)', fontsize=10)
    ax.set_title('CLE-CRE spectrogram (0-5 Hz)', fontsize=10, loc='left')

    for a in axes:
        a.set_xlim(0, t_hr[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def process_session(session_idx):
    session = load_session(session_idx)
    session.sleep_profile = load_sleep_profile(session)
    label = session.label

    print(f"\n{'='*60}")
    print(f"Full-night signal plot: {label}")
    print(f"{'='*60}")

    print("  Computing epoch traces + trials...")
    df = compute_epoch_traces(session)
    df, _ = apply_criteria(df)
    trials = extract_trials(df)
    print(f"  {len(trials)} trials (>= 5 min)")

    if not trials:
        print("  No trials — skipping")
        return 0

    sess_dir = REPORT_DIR / label
    sess_dir.mkdir(parents=True, exist_ok=True)

    print("  Plotting full-night signals...")
    fig = plot_fullnight(session, trials)
    out = sess_dir / 'fullnight_signals.png'
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out}")
    return len(trials)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    print("Full-Night Trial Signal Plots — Raw CAP + PSG")
    print("=" * 60)

    if args.session is not None:
        indices = [args.session]
    else:
        indices = range(12)

    total = 0
    for idx in indices:
        try:
            total += process_session(idx)
        except Exception as e:
            print(f"  ERROR on session {idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. {total} trials plotted across {len(indices)} sessions.")


if __name__ == '__main__':
    main()
