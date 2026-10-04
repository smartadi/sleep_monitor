"""
Recordings, the shared 30-s epoch grid, and the motion-canceller check (Methods §2.2-2.3).

Manuscript items
    Table 1        duration and analysis epochs per recording (demographics are EXTERNAL)
    §2.2 ¶140-141  4.1-8.7 h recordings, 9,319 analysis epochs, 111 Hz -> 100 Hz (EXTERNAL)
    §2.3 ¶144      OLS motion canceller: median 0.1 % of in-band variance removed,
                   r = 1.00 with accelerometer energy, coherence with the PSG
                   reference unchanged to three decimals
    cache          CACHE_DIR/s01_recordings/epochs.parquet -- the ONE 30-s epoch
                   grid every later stage uses (see `epoch_grid`)

The epoch grid
    The paper's epochs are the PSG sleep-profile epochs, placed on the SEC time
    axis by the wall-clock alignment in seclib.loader.load_sleep_profile. An
    epoch is kept when it lies wholly inside the SEC recording and holds at
    least half an epoch of samples. This is the grid of the overnight
    mean-value / imbalance analyses (9,312 epochs). The 9,319 quoted in the
    text and the per-recording counts of Table 1 come from a different grid:
    the rate pipeline cut each recording into floor(N / 3000) back-to-back
    30-s windows from the first sample, ignoring where the PSG epochs fall.
    (Legacy Table 1 took the count of windows with a finite rate and
    reference, per band; the larger of the two bands equals floor(N / 3000)
    for every recording.) Both counts are reported.

Ported from:
    scripts/run_mask_rate_detection.py   phase_a window grid (starts = arange(0, n-win+1, win)),
                                         per_session_summary.csv column n
    analysis/mean_value/mean_value_vs_stage.py   extract_session: epoch loop, acc_std,
                                                 top-decile motion flag
    analysis/rates/motion_cancel_validation.py   run_session, mean_band_coh, win_energy

Changes from the legacy code:
    * CORRECTION: the canceller validation is run on each SEC channel (CH, CLE,
      CRE), as §2.3 says, and on CLE-CRE as before. The legacy script ran it on
      CLE-CRE only, so the quoted 0.1 % / r = 1.00 described a channel the
      sentence does not name.
    * Table 1 durations were never written by a script; they are computed here
      as samples / fs.
    * The epoch grid is cached once (with sample indices i0, i1) instead of being
      re-derived in each analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import coherence

from seclib import CACHE_DIR, TAB_DIR, RESP_LO, RESP_HI, CARD_LO, CARD_HI, iter_sessions
from seclib.config import STAGE_LABELS
from seclib.filters import bandpass
from seclib.numbers import Numbers
from seclib.preprocessing import remove_acc_artifact

STAGE = 's01_recordings'
REQUIRES: list[str] = []

EPOCH_SEC = 30.0
MOTION_PCT = 90                       # top decile of per-epoch accelerometer SD = motion
CANCEL_CHANNELS = ['CH', 'CLE', 'CRE', 'CLE-CRE']
BANDS = {'resp': (RESP_LO, RESP_HI, 'Flow'), 'card': (CARD_LO, CARD_HI, 'Pleth')}


# ── the epoch grid ───────────────────────────────────────────────────────────

def epoch_grid(s) -> pd.DataFrame:
    """PSG sleep-profile epochs that lie inside the SEC recording, with motion flag."""
    fs = s.fs
    t_hr = s.time_hr.astype(np.float64)
    acc = s.cap['acc_mag'].astype(np.float64)
    sp = s.sleep_profile
    dt_hr = EPOCH_SEC / 3600.0
    rows = []
    for j, (t0, code) in enumerate(zip(sp['t_ep_hr'], sp['codes'])):
        t1 = t0 + dt_hr
        if t0 < 0 or t1 > t_hr[-1]:
            continue
        i0, i1 = np.searchsorted(t_hr, t0), np.searchsorted(t_hr, t1)
        if i1 - i0 < int(0.5 * fs * EPOCH_SEC):        # need >= half an epoch of samples
            continue
        rows.append({'session': s.label, 'subject': s.subject, 'night': s.meta['night'],
                     'epoch': j, 't_hr': t0 + dt_hr / 2.0,
                     'stage': STAGE_LABELS.get(int(code), '?'), 'stage_code': int(code),
                     'acc_sd': float(acc[i0:i1].std()), 'i0': int(i0), 'i1': int(i1)})
    ep = pd.DataFrame(rows)
    ep['motion'] = ep['acc_sd'] > np.nanpercentile(ep['acc_sd'], MOTION_PCT)
    return ep


def rate_grid_count(s) -> int:
    """Back-to-back 30-s windows from the first sample: the rate pipeline's grid."""
    win = int(EPOCH_SEC * s.fs)
    return len(np.arange(0, s.n_samples - win + 1, win))


# ── motion-canceller validation ──────────────────────────────────────────────

def mean_band_coh(x, y, lo, hi, fs):
    f, cxy = coherence(x, y, fs=fs, nperseg=int(60 * fs))
    m = (f >= lo) & (f <= hi)
    return float(np.mean(cxy[m]))


def win_energy(sig, win):
    n = len(sig) // win
    return np.var(sig[:n * win].reshape(n, win), axis=1)


def cancel_validation(s) -> list[dict]:
    """Variance removed, removed-vs-motion r, and reference coherence before/after."""
    fs = s.fs
    win = int(EPOCH_SEC * fs)
    cle, cre = s.cap['CLE'].astype(np.float64), s.cap['CRE'].astype(np.float64)
    chans = {'CH': s.cap['CH'].astype(np.float64), 'CLE': cle, 'CRE': cre,
             'CLE-CRE': cle - cre}
    acc = s.cap['acc_mag'].astype(np.float64)
    rows = []
    for band, (lo, hi, refname) in BANDS.items():
        acc_bp = bandpass(acc, lo, hi, fs)
        ref_bp = bandpass(s.psg[refname].astype(np.float64), lo, hi, fs)
        for ch in CANCEL_CHANNELS:
            before = bandpass(chans[ch], lo, hi, fs)
            after = remove_acc_artifact(chans[ch], acc, lo, hi, fs)
            removed = before - after
            e_rem, e_acc = win_energy(removed, win), win_energy(acc_bp, win)
            n = min(len(e_rem), len(e_acc))
            cb = mean_band_coh(before, ref_bp, lo, hi, fs)
            ca = mean_band_coh(after, ref_bp, lo, hi, fs)
            rows.append(dict(session=s.label, channel=ch, band=band, ref=refname,
                             frac_var_removed=1.0 - np.var(after) / (np.var(before) + 1e-20),
                             corr_removed_vs_motion=np.corrcoef(e_rem[:n], e_acc[:n])[0, 1],
                             coh_ref_before=cb, coh_ref_after=ca, coh_delta=ca - cb))
    return rows


# ── run ──────────────────────────────────────────────────────────────────────

def run():
    out_c = CACHE_DIR / STAGE
    out_t = TAB_DIR / STAGE
    out_c.mkdir(parents=True, exist_ok=True)
    out_t.mkdir(parents=True, exist_ok=True)

    grids, table, cancel = [], [], []
    for s in iter_sessions():
        ep = epoch_grid(s)
        grids.append(ep)
        table.append({'session': s.label, 'subject': s.subject,
                      'duration_h': s.n_samples / s.fs / 3600.0,
                      'epochs_rate_grid': rate_grid_count(s),
                      'epochs_profile_grid': len(ep),
                      'epochs_profile_scored': int((ep['stage_code'] >= 0).sum())})
        cancel += cancel_validation(s)
        print(f'  {s.label}: {s.duration_hr:.2f} h, {len(ep)} profile epochs')

    epochs = pd.concat(grids, ignore_index=True)
    epochs.to_parquet(out_c / 'epochs.parquet', index=False)
    t1 = pd.DataFrame(table)
    t1.to_csv(out_t / 'table1_recordings.csv', index=False)
    cv = pd.DataFrame(cancel)
    cv.to_csv(out_t / 'motion_cancel_validation.csv', index=False)
    register_numbers(t1, cv, epochs)


def register_numbers(t1, cv, epochs):
    nb = Numbers(STAGE)
    paper_dur = dict(S1N1='7.95', S1N2='7.63', S2N1='7.73', S2N2='6.77', S3N1='6.93',
                     S3N2='8.66', S4N1='6.18', S4N2='6.02', S5N1='4.11', S5N2='4.74',
                     S6N1='5.16', S6N2='5.78')
    paper_ep = dict(S1N1='954', S1N2='916', S2N1='928', S2N2='812', S3N1='832',
                    S3N2='1039', S4N1='741', S4N2='722', S5N1='493', S5N2='569',
                    S6N1='619', S6N2='694')
    for _, r in t1.iterrows():
        nb.add(f'dur_{r.session}', '§2.2 Table 1', f'{r.session} duration', r.duration_h,
               paper_dur[r.session], unit='h', source='samples / fs')
        nb.add(f'ep_{r.session}', '§2.2 Table 1', f'{r.session} analysis epochs (30 s)',
               int(r.epochs_rate_grid), paper_ep[r.session], source='rate_grid_count',
               note='Table 1 = floor(N/3000) windows from the first sample (rate-pipeline '
                    f'grid); the PSG-aligned paper grid has {int(r.epochs_profile_grid)}')
    nb.add('dur_range', '§2.2 ¶140', 'recording durations',
           f'{t1.duration_h.min():.1f} to {t1.duration_h.max():.1f}', '4.1 to 8.7',
           unit='h', source='samples / fs')
    nb.add('epochs_total', '§2.2 ¶141', 'total 30-s analysis epochs (rate-pipeline grid)',
           f'{int(t1.epochs_rate_grid.sum()):,}', '9,319', source='rate_grid_count',
           note='legacy per_session_summary.csv n = max(resp, card) valid windows; equals '
                'floor(N/3000) per recording')
    nb.add('epochs_total_profile', '§2.2 ¶141',
           'total 30-s epochs on the PSG-aligned grid (epochs.parquet, used by §3.1)',
           f'{len(epochs):,}', '9,319', source='epoch_grid',
           note='the paper grid; 7 fewer than the rate grid because PSG epochs straddling '
                'the recording edges are dropped. Text should say which grid it counts')
    nb.add('n_recordings', '§2.2 ¶140', 'overnight recordings', len(t1), '12')
    nb.add('fs_111', '§2.2 ¶140', '111 Hz acquisition, linear resampling to 100 Hz',
           'files arrive at 100 Hz', '111 Hz', status='EXTERNAL',
           note='done before the synchronised CSVs; no code in the repo')
    nb.add('demographics', '§2.2 ¶141 Table 1', 'ages 25-66, 4 M / 2 F, PSQI 4-9',
           'not in the data files', '25 to 66', status='EXTERNAL', note='Table 1 demographics')

    # §2.3 canceller. Paper quotes a median over the 12 recordings; the legacy number
    # pooled both bands of CLE-CRE (24 values).
    leg = cv[cv.channel == 'CLE-CRE']
    sec = cv[cv.channel.isin(['CH', 'CLE', 'CRE'])]
    nb.add('cancel_frac_clecre', '§2.3 ¶144', 'median in-band variance removed, CLE-CRE '
           '(legacy channel, both bands pooled)', 100 * leg.frac_var_removed.median(),
           '0.1%', unit='%', source='cancel_validation')
    nb.add('cancel_frac_sec', '§2.3 ¶144', 'median in-band variance removed, CH/CLE/CRE '
           'pooled, both bands', 100 * sec.frac_var_removed.median(), '0.1%', unit='%',
           source='cancel_validation', note='CORRECTION: per SEC channel, as the text says')
    for ch in ['CH', 'CLE', 'CRE']:
        for band in BANDS:
            g = cv[(cv.channel == ch) & (cv.band == band)]
            nb.add(f'cancel_frac_{ch}_{band}', '§2.3 ¶144',
                   f'median variance removed, {ch} {band}', 100 * g.frac_var_removed.median(),
                   unit='%', source='cancel_validation',
                   note=f'range {100*g.frac_var_removed.min():.3f}-{100*g.frac_var_removed.max():.2f}%')
    nb.add('cancel_r_clecre', '§2.3 ¶144', 'median r(removed energy, accel energy), CLE-CRE',
           leg.corr_removed_vs_motion.median(), '1.00', source='cancel_validation')
    nb.add('cancel_r_sec', '§2.3 ¶144', 'min r(removed energy, accel energy), CH/CLE/CRE',
           sec.corr_removed_vs_motion.min(), '1.00', source='cancel_validation',
           note='CORRECTION: per SEC channel. r = 1 is by construction: the removed '
                'component is beta x bandpassed accel, so its energy is exactly proportional')
    # The legacy script printed the per-band MEDIAN coherence before and after at 3 d.p.;
    # that is the comparison the sentence describes. Per-recording agreement is in the note.
    for nm, g in [('clecre', leg), ('sec', sec)]:
        med = g.groupby('band')[['coh_ref_before', 'coh_ref_after']].median()
        same_med = bool((med.coh_ref_before.round(3) == med.coh_ref_after.round(3)).all())
        same = (g.coh_ref_before.round(3) == g.coh_ref_after.round(3))
        nb.add(f'cancel_coh3dp_{nm}', '§2.3 ¶144',
               f'median reference coherence unchanged to 3 d.p. '
               f'({"CLE-CRE" if nm == "clecre" else "CH/CLE/CRE"})',
               '; '.join(f'{b} {r.coh_ref_before:.3f} -> {r.coh_ref_after:.3f}'
                         for b, r in med.iterrows()),
               'unchanged to three decimal places', status='MATCH' if same_med else 'DIFF',
               source='cancel_validation',
               note=f'per recording and band: {int(same.sum())} of {len(g)} equal at 3 d.p., '
                    f'max |delta| {g.coh_delta.abs().max():.4f}; the coherences themselves '
                    f'are {g.coh_ref_before.min():.5f}-{g.coh_ref_before.max():.3f}, near the '
                    'estimator floor, so "unchanged" carries little information')
    nb.save()


if __name__ == '__main__':
    run()
