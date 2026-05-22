"""
Compute EEG band power by sleep stage for all sessions.

Outputs:
    artifacts/eeg/band_power.parquet   — per-epoch band power with stage labels
    artifacts/eeg/spectrograms/        — per-session spectrogram arrays
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch, spectrogram

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sleep_monitor import (
    load_all_sessions,
    FS, EEG_BANDS, STAGE_LABELS, outlier_clip,
)


def compute_session_eeg(session):
    if session.sleep_profile is None:
        print(f'  Skipping {session.label}: no sleep profile')
        return [], None

    eeg = outlier_clip(session.psg['EEG'].astype(np.float64))
    sp = session.sleep_profile
    t_ep = sp['t_ep_hr']
    codes = sp['codes']
    t_hr = session.time_hr
    epoch_n = int(30.0 * FS)

    rows = []
    for ep_i, (t_ep_start, stage_code) in enumerate(zip(t_ep, codes)):
        mask = (t_hr >= t_ep_start) & (t_hr < t_ep_start + 30 / 3600)
        if mask.sum() < epoch_n // 2:
            continue
        seg = eeg[mask]
        freqs, psd = welch(seg, fs=FS, nperseg=min(len(seg), int(FS * 4)))
        row = {
            'session': session.label,
            'subject': session.subject,
            'epoch': ep_i,
            'stage': STAGE_LABELS.get(int(stage_code), '?'),
            'stage_code': int(stage_code),
            't_hr': float(t_ep_start),
        }
        for band, (flo, fhi) in EEG_BANDS.items():
            bmask = (freqs >= flo) & (freqs <= fhi)
            row[band] = float(np.mean(psd[bmask])) if bmask.any() else np.nan
        rows.append(row)

    f_spec, t_spec, Sxx = spectrogram(eeg, fs=FS, nperseg=int(FS * 4),
                                       noverlap=int(FS * 2))
    spec_data = {'f': f_spec, 't_hr': t_spec / 3600, 'Sxx': Sxx}

    return rows, spec_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-spectrograms', action='store_true',
                        help='Skip saving spectrogram arrays')
    args = parser.parse_args()

    out_dir = ROOT / 'artifacts' / 'eeg'
    spec_dir = out_dir / 'spectrograms'
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_spectrograms:
        spec_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for session in load_all_sessions(with_sleep_profiles=True):
        print(f'── {session.label} ──')
        rows, spec_data = compute_session_eeg(session)
        all_rows.extend(rows)

        if spec_data is not None and not args.no_spectrograms:
            np.savez_compressed(
                spec_dir / f'{session.label}_spectrogram.npz',
                **spec_data)

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_dir / 'band_power.parquet', index=False)

    print(f'\nWrote {len(df)} epoch rows to {out_dir / "band_power.parquet"}')
    if not args.no_spectrograms:
        print(f'Spectrograms in {spec_dir}/')

    print('\nMean band power by stage:')
    band_cols = list(EEG_BANDS.keys())
    print(df.groupby('stage')[band_cols].mean().round(6).to_string())


if __name__ == '__main__':
    main()
