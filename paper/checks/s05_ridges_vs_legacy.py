"""Does s05_ridges reproduce the legacy ridge outputs?

Run after the stage:  .venv/Scripts/python.exe paper/checks/s05_ridges_vs_legacy.py

Two levels for each table:
  logic  the stage's function applied to the LEGACY per-epoch parquet must give the
         legacy table exactly (tests the ported summary code on identical input);
  data   the stage's own output must equal the legacy table (tests the whole chain).
Writes paper/checks/results/s05_ridges_vs_legacy.txt.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(PAPER))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

from seclib import CACHE_DIR, TAB_DIR   # noqa: E402
from stages import s05_ridges as S      # noqa: E402

LEG = ROOT / 'reports'
OUT = TAB_DIR / S.STAGE


def same(a: pd.DataFrame, b: pd.DataFrame, keys=None, rtol=1e-9):
    """Exact for text/int columns, rtol for floats. Returns (ok, message)."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if keys:
        a = a.sort_values(keys).reset_index(drop=True)
        b = b.sort_values(keys).reset_index(drop=True)
    if list(a.columns) != list(b.columns):
        return False, f'columns differ: {list(a.columns)} vs {list(b.columns)}'
    if len(a) != len(b):
        return False, f'rows {len(a)} vs {len(b)}'
    bad = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
            xv, yv = x.astype(float).values, y.astype(float).values
            ok = np.isclose(xv, yv, rtol=rtol, atol=1e-12, equal_nan=True)
        else:
            ok = (x.astype(str).values == y.astype(str).values)
        if not ok.all():
            bad.append(f'{c}: {int((~ok).sum())} of {len(ok)} differ')
    return (not bad), ('identical' if not bad else '; '.join(bad))


def _merge_jul16(ridges, n_win, step_sec, max_freq_jump, merge_gap_windows):
    """sleep_monitor.harmonics._merge_ridge_fragments as of 94b817d (2026-07-16), the
    version that wrote the legacy parquet: boundary frequency = the single end/start
    sample, not the median of three."""
    if len(ridges) < 2:
        return ridges
    ridges = sorted(ridges, key=lambda r: (r['start_idx'], r['median_freq']))
    merged = [ridges[0]]
    for cand in ridges[1:]:
        did_merge = False
        for mi, base in enumerate(merged):
            gap = cand['start_idx'] - base['end_idx']
            if gap < 1 or gap > merge_gap_windows:
                continue
            freq_base = base['freq_trace'][base['end_idx']]
            freq_cand = cand['freq_trace'][cand['start_idx']]
            if np.isnan(freq_base) or np.isnan(freq_cand):
                continue
            if abs(freq_base - freq_cand) > max_freq_jump * 2:
                continue
            new_freq = base['freq_trace'].copy()
            new_amp = base['amp_trace'].copy()
            mask_c = ~np.isnan(cand['freq_trace'])
            new_freq[mask_c] = cand['freq_trace'][mask_c]
            new_amp[mask_c] = cand['amp_trace'][mask_c]
            end_idx = max(base['end_idx'], cand['end_idx'])
            start_idx = min(base['start_idx'], cand['start_idx'])
            median_freq = float(np.nanmedian(new_freq))
            merged[mi] = {'freq_trace': new_freq, 'amp_trace': new_amp,
                          'start_idx': start_idx, 'end_idx': end_idx,
                          'duration_sec': (end_idx - start_idx + 1) * step_sec,
                          'median_freq': median_freq,
                          'n_present': int(np.sum(~np.isnan(new_freq))),
                          'label': f'{median_freq:.2f}Hz'}
            did_merge = True
            break
        if not did_merge:
            merged.append(cand)
    merged.sort(key=lambda r: r['median_freq'])
    return merged


def single_pass_retest(cells, leg_ep):
    """Re-run the given band/channel/night cells with the July (one-pass,
    endpoint-frequency) fragment merge."""
    import seclib.harmonics as H
    from seclib import get_session
    from seclib.preprocessing import remove_acc_artifact
    orig, calls = H._merge_ridge_fragments, {'n': 0}

    def one_pass(ridges, *a):
        calls['n'] += 1
        return _merge_jul16(ridges, *a) if calls['n'] == 1 else ridges

    H._merge_ridge_fragments = one_pass
    n_ok, cache = 0, {}
    try:
        for band, ch, sess in cells:
            s = cache.get(sess) or cache.setdefault(sess, get_session(sess))
            sig = remove_acc_artifact(s.cap[ch], s.cap['acc_mag'], 0.05, 4.0)
            bp = S.BANDS[band]
            calls['n'] = 0
            rr = H.detect_persistent_ridges(
                sig, fs=S.FS, win_sec=S.WIN_SEC, step_sec=S.STEP_SEC,
                min_freq=bp['min_freq'], max_freq=bp['max_freq'],
                smooth_windows=S.SMOOTH_WINDOWS, min_persistence_sec=S.MIN_PERSIST_SEC,
                max_freq_jump=bp['max_freq_jump'],
                peak_prominence_frac=bp['peak_prominence_frac'],
                welch_seg_sec=bp['welch_seg_sec'], max_gap_windows=S.MAX_GAP_WINDOWS,
                acc_mag=s.cap['acc_mag'])
            df = S.align_sleep_stages(S.ridge_epoch_features(rr, ch, band), s.sleep_profile)
            df['session'], df['subject'] = s.label, s.subject
            ref = leg_ep[(leg_ep.band == band) & (leg_ep.channel == ch)
                         & (leg_ep.session == sess)]
            n_ok += same(df, ref)[0]
    finally:
        H._merge_ridge_fragments = orig
    return f'{n_ok} of {len(cells)} cells now identical to the legacy parquet'


def main():
    lines = []
    say = lines.append

    leg_ep = pd.read_parquet(LEG / 'slow_wave' / 'band_ridge_epochs.parquet')
    new_ep = pd.read_parquet(CACHE_DIR / S.STAGE / 'band_ridge_epochs.parquet')

    # ── per-epoch features ──
    ok, msg = same(new_ep, leg_ep)
    say(f'[data ] band_ridge_epochs.parquet ({len(new_ep)} rows): {msg}')
    if not ok:
        key = ['band', 'channel', 'session']
        d = new_ep.merge(leg_ep, on=key + ['t_hr'], suffixes=('', '_leg'))
        for c in ['n_ridges', 'total_ridge_power', 'min_ridge_freq', 'ridge_present',
                  'motion_masked', 'stage_code']:
            diff = ~np.isclose(d[c].astype(float), d[c + '_leg'].astype(float), equal_nan=True)
            by = d[diff].groupby(key).size()
            say(f'        {c}: {int(diff.sum())} epochs differ in {len(by)} '
                f'band/channel/night cells')
        cols = ['n_groups_active', 'min_ridge_freq', 'mean_ridge_freq', 'freq_spread',
                'max_prominence']
        d['any'] = np.zeros(len(d), bool)
        for c in cols:
            d['any'] |= ~np.isclose(d[c].astype(float), d[c + '_leg'].astype(float),
                                    equal_nan=True)
        cells = sorted(d[d['any']].groupby(key).size().index)
        say(f'        differing cells: {len(cells)} of {d.groupby(key).ngroups}: '
            + ', '.join('/'.join(c) for c in cells))
        say('        cause test: the vendored detector merges ridge fragments iteratively and '
            'matches them on a 3-sample boundary median (sleep_monitor commit 52cc5f3, '
            '2026-08-17); the legacy parquet was written 2026-07-16 with ONE merge pass on '
            'single endpoint samples. Re-running the differing cells with the July merge:')
        say('        ' + single_pass_retest(cells, leg_ep))

    # ── stage summary (CRE) ──
    leg = pd.read_csv(LEG / 'slow_wave' / 'band_ridge_stage_summary.csv')
    say(f'[logic] band_ridge_stage_summary.csv: {same(S.stage_summary(leg_ep), leg)[1]}')
    new = pd.read_csv(OUT / 'band_ridge_stage_summary.csv')
    say(f'[data ] band_ridge_stage_summary.csv: {same(new, leg)[1]}')
    m = new.merge(leg, on=['band', 'feature'], suffixes=('', '_leg'))
    for _, r in m.iterrows():
        if (not np.isclose(r.kw_p, r.kw_p_leg, rtol=1e-9, equal_nan=True)
                or r.directions_by_mean != r.directions_by_mean_leg):
            say(f'        {r.band} {r.feature}: kw_p {r.kw_p:.3g} (legacy {r.kw_p_leg:.3g}), '
                f'N3 lower by mean {r.n_subj_N3_dn_by_mean} (legacy {r.n_subj_N3_dn_by_mean_leg})')

    # ── all channels ──
    leg = pd.read_csv(LEG / 'slow_wave' / 'ridge_stage_all_channels.csv')
    keys = ['band', 'channel', 'feature', 'statistic']
    say(f'[logic] ridge_stage_all_channels.csv: '
        f'{same(S.all_channel_directions(leg_ep), leg, keys)[1]}')
    new = pd.read_csv(OUT / 'ridge_stage_all_channels.csv')
    say(f'[data ] ridge_stage_all_channels.csv: {same(new, leg, keys)[1]}')
    m = new.merge(leg, on=keys, suffixes=('', '_leg'))
    for _, r in m[m.directions != m.directions_leg].iterrows():
        say(f'        {r.band} {r.channel} {r.feature} {r.statistic}: {r.directions} '
            f'(legacy {r.directions_leg})')
    hl = m[(m.band == 'resp') & (m.feature == 'total_ridge_power')]
    for _, r in hl.iterrows():
        say(f'        resp power {r.channel:<3} {r.statistic:<6} N3 lower '
            f'{r.n3_lower}/6 (legacy {r.n3_lower_leg}/6)')

    # ── robustness (m6_m7_ridges) ──
    leg = pd.read_csv(LEG / 'rates' / 'reviewer_pass' / 'ridge_robustness.csv')
    keys = ['band', 'channel', 'subset']
    say(f'[logic] ridge_robustness.csv: {same(S.ridge_robustness(leg_ep), leg, keys)[1]}')
    say(f'[data ] ridge_robustness.csv: '
        f'{same(pd.read_csv(OUT / "ridge_robustness.csv"), leg, keys)[1]}')

    # ── low band ──
    leg = pd.read_csv(LEG / 'slow_wave' / 'low_band' / 'ridge_lowband_smooth.csv')
    say(f'[data ] ridge_lowband_smooth.csv: '
        f'{same(pd.read_csv(OUT / "lowband_nights.csv"), leg)[1]}')
    leg = pd.read_csv(LEG / 'slow_wave' / 'low_band' / 'ridge_lowband_smooth_episodes.csv')
    say(f'[data ] ridge_lowband_smooth_episodes.csv: '
        f'{same(pd.read_csv(OUT / "lowband_episodes.csv"), leg)[1]}')

    text = '\n'.join(lines)
    print(text)
    res = PAPER / 'checks' / 'results'
    res.mkdir(parents=True, exist_ok=True)
    (res / 's05_ridges_vs_legacy.txt').write_text(text + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
