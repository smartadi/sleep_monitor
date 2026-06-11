"""
SWA Validation — Step 0: Data Inventory

Scans all overnight recordings and reports: file format, channels, sampling rate,
units, duration, sleep staging availability, and time alignment between CAP and PSG.

Usage:
    python analysis/swa_validation/step0_inventory.py
"""

import glob
import re
from datetime import datetime

import numpy as np
import pandas as pd

from sleep_monitor.config import CAP_CHANNELS, PSG_CHANNELS
from sleep_monitor.sessions import SESSION_META


def inventory():
    print("=" * 90)
    print("SWA Validation — Step 0: Data Inventory")
    print("=" * 90)

    # ── 1. CSV file check ─────────────────────────────────────────────────────
    print("\n--- Combined CSV Files (CAP + PSG @ 100 Hz) ---")
    print(f"{'Session':5s}  {'Subject':10s}  {'Date':12s}  {'CSV':3s}  {'Size_MB':>8s}  "
          f"{'Samples':>10s}  {'Dur_hr':>7s}  {'Fs':>5s}")
    print("-" * 80)

    for m in SESSION_META:
        csv_path = m['csv']
        exists = csv_path.exists()
        size_mb = csv_path.stat().st_size / 1e6 if exists else 0

        if exists:
            tms = pd.read_csv(csv_path, compression='gzip', usecols=['timeMS'])
            n_samples = len(tms)
            t_range_ms = tms['timeMS'].iloc[-1] - tms['timeMS'].iloc[0]
            dur_hr = t_range_ms / 3_600_000
            fs = n_samples / (t_range_ms / 1000) if t_range_ms > 0 else 0
        else:
            n_samples = 0
            dur_hr = 0
            fs = 0

        print(f"{m['label']:5s}  {m['subject']+'-'+m['initials']:10s}  {m['date']:12s}  "
              f"{'Y' if exists else 'N':3s}  {size_mb:>7.1f}M  {n_samples:>10,}  "
              f"{dur_hr:>6.2f}h  {fs:>5.1f}")

    # ── 2. PSG staging check ──────────────────────────────────────────────────
    print("\n--- PSG Sleep Profile (AASM staging) ---")
    print(f"{'Session':5s}  {'Profile':7s}  {'Epochs':>7s}  {'PSG_dur':>8s}  "
          f"{'PSG_start':>23s}  {'CSV_start':>23s}  {'Offset_min':>11s}")
    print("-" * 100)

    data_re = re.compile(r'^\d{2}:\d{2}:\d{2},\d{3};\s*(.+)$')

    for m in SESSION_META:
        pattern = str(m['psg_dir'] / 'PSG_analysis_*' / 'Sleep Profile*.txt')
        sp_files = [f for f in glob.glob(pattern) if 'reliability' not in f.lower()]
        has_profile = len(sp_files) > 0

        psg_start = None
        n_epochs = 0
        if has_profile:
            with open(sp_files[0], 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if line.startswith('Start Time:'):
                        psg_start = datetime.strptime(
                            line.strip().split(': ', 1)[1], '%m/%d/%Y %I:%M:%S %p')
                    if data_re.match(line.strip()):
                        n_epochs += 1

        psg_dur_hr = n_epochs * 30 / 3600

        csv_start = None
        offset_min = None
        if m['csv'].exists():
            df_head = pd.read_csv(m['csv'], compression='gzip', nrows=1)
            csv_start = datetime.strptime(
                df_head['timeSM'].iloc[0], '%Y-%m-%d %H:%M:%S.%f')
            if psg_start:
                offset_min = (csv_start - psg_start).total_seconds() / 60

        print(f"{m['label']:5s}  {'Y' if has_profile else 'N':7s}  {n_epochs:>7d}  "
              f"{psg_dur_hr:>7.2f}h  {str(psg_start):>23s}  {str(csv_start):>23s}  "
              f"{f'{offset_min:>+10.1f}' if offset_min is not None else 'N/A':>11s}")

    # ── 3. EEG quality ────────────────────────────────────────────────────────
    print("\n--- EEG Signal Quality ---")
    print(f"{'Session':5s}  {'EEG_min':>8s}  {'EEG_max':>8s}  {'EEG_std':>8s}  "
          f"{'Clip%':>7s}  {'P01':>8s}  {'P99':>8s}")
    print("-" * 60)

    for m in SESSION_META:
        eeg = pd.read_csv(m['csv'], compression='gzip', usecols=['EEG'])['EEG'].values
        emin, emax = eeg.min(), eeg.max()
        estd = eeg.std()
        clip_thresh = 0.01 * (emax - emin)
        clip_pct = ((eeg <= emin + clip_thresh) | (eeg >= emax - clip_thresh)).sum() / len(eeg) * 100
        p01, p99 = np.percentile(eeg, [1, 99])
        print(f"{m['label']:5s}  {emin:>8.1f}  {emax:>8.1f}  {estd:>8.1f}  "
              f"{clip_pct:>6.2f}%  {p01:>8.1f}  {p99:>8.1f}")

    # ── 4. CAP differential quality ───────────────────────────────────────────
    print("\n--- CAP Differential (CLE - CRE) ---")
    print(f"{'Session':5s}  {'Diff_std':>8s}  {'P01':>8s}  {'P99':>8s}")
    print("-" * 35)

    for m in SESSION_META:
        df = pd.read_csv(m['csv'], compression='gzip', usecols=['CLE', 'CRE'])
        diff = df['CLE'].values - df['CRE'].values
        p01, p99 = np.percentile(diff, [1, 99])
        print(f"{m['label']:5s}  {diff.std():>8.1f}  {p01:>8.1f}  {p99:>8.1f}")

    # ── 5. Channel dead check ─────────────────────────────────────────────────
    print("\n--- Dead/Flat Channel Check ---")
    key_chans = ['EEG', 'ECG', 'Flow', 'CLE', 'CRE']
    for m in SESSION_META:
        df = pd.read_csv(m['csv'], compression='gzip', usecols=key_chans)
        issues = []
        for ch in key_chans:
            if df[ch].std() < 0.1:
                issues.append(f"{ch}(FLAT)")
        status = ', '.join(issues) if issues else 'OK'
        print(f"  {m['label']:5s}: {status}")

    print("\n" + "=" * 90)
    print("INVENTORY COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    inventory()
