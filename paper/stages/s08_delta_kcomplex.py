"""Delta-burst onsets and scored K-complexes: the SEC response that follows a cortical event.

Manuscript items produced
    Methods ¶162-164  onset detector, count-matched nulls, causal estimator, AUC /
                      cross-correlation, ±10 s arousal control; 50 K-complex marks in
                      10 of 12 recordings
    Results ¶206      344 onsets, 1-99 per recording, recordings with < 10 events
    Results ¶207      post-onset peaks 1.4-3.2 z vs null <= 0.17, 6/6 participants
    Results ¶208      AUC 0.42-0.56, cross-correlation lag, zero-phase vs causal
                      pre-onset difference (0.35-0.41 -> ~0), participants positive
    Results ¶209      latency 4-8 s; "K-complexes constituted most detected onsets"
    Fig. 7 (+ ¶212)   causal 3x3 peri-onset grid; "most onsets within 10 s of an arousal"
    Results ¶241-244  50 scored K-complexes, latencies 3.6 / 3.7 / 4.3 s, 1-10 per recording
    Fig. 12 (+ ¶246)  SEC response at scored K-complexes
    tables            delta_onsets.csv, delta_onsets_summary.csv, precursor_summary.csv,
                      lowband_precursor_check.csv, response_consistency.csv,
                      arousal_control.csv, arousal_control_counts.csv,
                      kcomplex_marks.csv, kcomplex_criteria.csv, kcomplex_onset_overlap.csv,
                      kcomplex_cap_response.csv, kcomplex_latencies.csv
    cache             _cache/s08_delta_kcomplex/{detector,zerophase,fullrate}_<rec>.npz

Ported from:
    analysis/delta_onset/delta_onset_detection.py   _bandpass, _smooth, _rolling_std,
        stage_code_per_sample, motion_mask, load_features, detect_onsets, run_session
        (summary row); q30 only
    analysis/delta_onset/delta_cap_precursor.py     load_features, _env, _zscore_on,
        peri_stack, random_nrem_centers, nrem_concat, nrem_xcorr, _lagwin, auc,
        build_base, process_tag, aggregate, run_tag (summary table)
    analysis/delta_onset/lowband_precursor_check.py env_zerophase, env_causal,
        _trail_ma, process, per_subject, rise_onset_time, run (check table,
        response-consistency table, causal 3x3 grid = Fig. 7)
    analysis/delta_onset/arousal_control.py         arousal_intervals, main
    analysis/delta_onset/kcomplex_morphology.py     collect (mark table only),
        longest_run_s, score_criteria
    analysis/delta_onset/kcomplex_cap_response.py   scored_kcomplexes_20hz, lowband,
        baselined, collect, figure (= Fig. 12)
    Dropped: overview/gallery figures, the q15 quiescence window, the zero-phase grid,
    xcorr and AUC figures, the 0-0.5 Hz zero-phase-vs-causal figure, the K-complex
    morphology and detector-gate figures, and delta_onset_figures.py.

Changes from the legacy code:
    1. CORRECTION ¶206 "three recordings contained fewer than 10 events": it is four
       (S1N2 9, S4N2 4, S5N1 1, S5N2 6); registered.
    2. CORRECTION ¶208 "0.35-0.41 ... positive in all six participants" holds for CLE
       and CRE only; on CH 4 of 6 participants are positive. Registered per channel.
    3. CORRECTION ¶208 attributes the AUC and the cross-correlation to causal
       filtering, but legacy computed both with the zero-phase (decimate-first)
       pipeline. Both are now also computed on the strictly causal envelopes (causal
       EEG delta envelope as the cross-correlation target) and registered side by side.
    4. CORRECTION the K-complex latencies (¶242, Fig. 12b) come from zero-phase
       envelopes, against Methods ¶163. They are recomputed per event on the causal
       envelopes too (same marks, same baseline) and both are written to
       kcomplex_latencies.csv; legacy only printed them on the figure.
    5. CORRECTION ¶209 "K-complexes, which constituted most detected onsets": only 9 of
       the 344 onsets fall within ±5 s of a scored K-complex mark, and only 57 marks
       were scored cohort-wide (50 inside the SEC recording windows). Registered.
    6. CORRECTION the Fig. 7 average and the arousal control skip recordings with < 5
       onsets (S4N2, S5N1), so they rest on 339 onsets, not 344. Registered with a note.
    7. The K-complex null (¶242, ¶246 "count-matched") draws max(20 n, 200) random-NREM
       windows per recording, as in legacy. Kept; registered as a text discrepancy.
    8. Legacy drew the AUC's random-NREM null from one generator (seed 0) that had
       first fed a 150-shift circular null for every cross-correlation (used only by a
       dropped figure) and then the whole q15 run (not in the paper). Neither is kept:
       the shift null is not computed (saves ~40 min) and the AUC null is drawn from a
       fresh seed-0 generator. The pooled AUCs move by up to 0.085 (0.419-0.557 ->
       0.432-0.531): the AUC is that sensitive to which random-NREM windows are drawn.
       paper/checks/s08_delta_kcomplex_vs_legacy.py replays the legacy draw order and
       reproduces the legacy table exactly.
    9. The 20 Hz full-rate envelopes are cached as float32 (legacy recomputed them in
       float64); effect <1e-5 z, see the check.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from seclib import LABELS, TAB_DIR, get_session, Numbers
from seclib.figures import save
from seclib.loader import (_INTERVAL_RE, load_arousals, load_autonomic_arousals,
                           load_interval_events)

from stages._s08_delta_kcomplex_signal import (
    AFS, BANDS, CHANNELS, FS, KEYS, LEAD_WIN, MIN_BURST_S, MIN_ONSETS, NREM_CODES,
    POST_S, PRE_S, Q, K_HIGH, K_LOW, auc, detect_onsets, detector_features,
    fullrate_features, longest_run_s, nrem_xcorr, peri_stack, per_subject,
    random_nrem_centers, zerophase_features, zscore_on)
from stages._s08_delta_kcomplex_figures import fig7_causal_grid, fig12_kcomplex

STAGE = 's08_delta_kcomplex'
REQUIRES: list[str] = []

TABLES = TAB_DIR / STAGE
PRE, POST = int(PRE_S * AFS), int(POST_S * AFS)     # peri-onset window on the 20 Hz grid
EPS = 0.05                     # z threshold for the "rise-onset" crossing
AROUSAL_GUARD_S = 10.0         # arousal control: drop onsets within ±10 s of an arousal
KC_MATCH_S = 5.0               # onset "on" a scored K-complex: within ±5 s of its mark
KC_PRE_S, KC_POST_S = 15.0, 15.0
KC_PRE, KC_POST = int(KC_PRE_S * AFS), int(KC_POST_S * AFS)
KC_BASE_END = KC_PRE - int(2 * AFS)    # K-complex baseline: -15 s .. -2 s
KC_PEAK_WIN = (0.0, 10.0)              # where a K-complex response peak is looked for
KC_STEM = 'Spindle  K'                 # PSG export file holding Spindle + K-Complex marks


def subject_of(label):
    return label.split('N')[0]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Delta-burst onsets (delta_onset_detection.py)
# ═════════════════════════════════════════════════════════════════════════════
def detect_all():
    """q30 onsets per recording: event table, per-recording summary, {label: samples}."""
    events, summary, onsets = [], [], {}
    for lab in LABELS:
        F = detector_features(lab)
        env, codes = F['env'], F['codes']
        on, _ = detect_onsets(env, codes, F['motion'])
        onsets[lab] = on['onset_samp'].to_numpy() if len(on) else np.array([], int)
        if len(on):
            events.append(on.assign(session=lab))
        nrem_hr = np.isin(codes, NREM_CODES).sum() / FS / 3600.0
        mix = on['stage'].value_counts().to_dict() if len(on) else {}
        iei = np.diff(np.sort(on['onset_hr'].to_numpy())) * 3600.0 if len(on) > 1 else []
        n = len(on)
        n_nrem = int(on['stage_code'].isin(NREM_CODES).sum()) if n else 0
        summary.append({
            'session': lab, 'quiet_pre_s': 30, 'n_onsets': n,
            'dur_hr': round(len(env) / FS / 3600.0, 2), 'nrem_hr': round(nrem_hr, 2),
            'onsets_per_nrem_hr': round(n / max(nrem_hr, 1e-6), 2),
            'pct_in_nrem': round(100.0 * n_nrem / max(n, 1), 1),
            'n_N3': int(mix.get('N3', 0)), 'n_N2': int(mix.get('N2', 0)),
            'median_iei_s': round(float(np.median(iei)), 1) if len(iei) else np.nan})
    ev = pd.concat(events, ignore_index=True)
    ev = ev[['session'] + [c for c in ev.columns if c != 'session']]
    return ev, pd.DataFrame(summary), onsets


def to_20hz(onsets):
    return {lab: np.round(np.asarray(s) / Q).astype(int) if len(s) else np.array([], int)
            for lab, s in onsets.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Zero-phase and causal precursor tests: AUC, cross-correlation
#    (delta_cap_precursor.py; causal variant = CORRECTION 3)
# ═════════════════════════════════════════════════════════════════════════════
def load_zerophase(lab):
    F = zerophase_features(lab)
    envs = {(ch, b): F[f'cap_{ch}_{b}'] for ch, b in KEYS}
    return envs, F['eeg_delta'], F['nrem'], F['motion']


def load_causal(lab):
    F = fullrate_features(lab)
    envs = {(ch, b): F[f'ca_{ch}_{b}'] for ch, b in KEYS}
    return envs, F['ca_eeg_delta'], F['nrem'], F['motion']


def xcorr_pass(labels, load):
    """SEC -> EEG-delta cross-correlation per recording (build_base)."""
    out = {}
    for lab in labels:
        envs, eeg, nrem, _ = load(lab)
        eeg_z = zscore_on(eeg.astype(np.float64), nrem)
        out[lab] = {key: nrem_xcorr(zscore_on(envs[key].astype(np.float64), nrem),
                                    eeg_z, nrem) for key in KEYS}
    return out


def onset_pass(labels, onsets20, load, xcorr, rng):
    """Peri-onset curves, count-matched null and forecasting AUC (process_tag)."""
    lw0, lw1 = int((LEAD_WIN[0] + PRE_S) * AFS), int((LEAD_WIN[1] + PRE_S) * AFS)
    sessions = []
    for lab in labels:
        ons = onsets20[lab]
        if len(ons) < MIN_ONSETS:
            continue
        envs, _, nrem, motion = load(lab)
        rand = random_nrem_centers(nrem, motion, ons, PRE, POST, len(ons), rng)
        res = {'label': lab, 'subject': subject_of(lab), 'curves': {}, 'null': {},
               'auc': {}, 'xcorr': xcorr[lab]}
        for key in KEYS:
            env_z = zscore_on(envs[key].astype(np.float64), nrem)
            stk = peri_stack(env_z, ons, PRE, POST)
            nstk = peri_stack(env_z, rand, PRE, POST) if len(rand) else None
            res['curves'][key] = stk.mean(0) if stk is not None else np.full(PRE + POST, np.nan)
            res['null'][key] = nstk.mean(0) if nstk is not None else np.full(PRE + POST, np.nan)
            pos = stk[:, lw0:lw1].mean(1) if stk is not None else np.array([])
            neg = nstk[:, lw0:lw1].mean(1) if nstk is not None else np.array([])
            res['auc'][key] = auc(pos, neg)
        sessions.append(res)
    return sessions


def precursor_summary(sessions, estimator):
    """Per channel x band: post-onset peak, lead amplitude, xcorr peak, AUC (run_tag)."""
    tax = np.arange(-PRE, POST) / AFS
    subj = sorted({s['subject'] for s in sessions})
    _, C = per_subject(sessions, 'curves')
    _, N = per_subject(sessions, 'null')
    pre_mask = (tax >= LEAD_WIN[0]) & (tax <= LEAD_WIN[1])
    post = tax >= 0
    rows = []
    for key in KEYS:
        auc_subj = np.array([np.nanmean([s['auc'][key] for s in sessions
                                         if s['subject'] == sb]) for sb in subj])
        m, nm = np.nanmean(C[key], 0), np.nanmean(N[key], 0)
        xcs = [s['xcorr'][key] for s in sessions if s['xcorr'][key] is not None]
        M = np.nanmean([x[1] for x in xcs], 0)
        lags = xcs[0][0]
        rows.append({
            'estimator': estimator, 'channel': key[0], 'band_hz': key[1],
            'post_peak_z': round(float(np.nanmax(m[post])), 3),
            'post_peak_t_s': round(float(tax[post][np.nanargmax(m[post])]), 1),
            'null_peak_z': round(float(np.nanmax(nm[post])), 3),
            'lead_amp_z': round(float(np.nanmean(m[pre_mask])), 3),
            'peak_pre_t_s': round(float(tax[:len(tax) // 2][np.nanargmax(m[:len(tax) // 2])]), 1),
            'xcorr_peak_lag_s': round(float(lags[np.argmax(np.abs(M))]), 1),
            'xcorr_peak_r': round(float(M[np.argmax(np.abs(M))]), 3),
            'auc_pooled': round(float(np.nanmean(auc_subj)), 3),
            'auc_subj_min': round(float(np.nanmin(auc_subj)), 3),
            'auc_subj_max': round(float(np.nanmax(auc_subj)), 3)})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Zero-phase vs causal peri-onset curves, Fig. 7, response consistency
#    (lowband_precursor_check.py)
# ═════════════════════════════════════════════════════════════════════════════
def onset_curves(F, ons, rng):
    """Onset-locked and null mean curves for both estimators (lowband process)."""
    nrem, motion = F['nrem'], F['motion']
    rand = random_nrem_centers(nrem, motion, ons, PRE, POST, len(ons), rng)
    out = {'zp': {}, 'ca': {}, 'zp_null': {}, 'ca_null': {}}
    for ch, b in KEYS:
        for est in ('zp', 'ca'):
            env = zscore_on(F[f'{est}_{ch}_{b}'].astype(np.float64), nrem)
            s_on = peri_stack(env, ons, PRE, POST)
            s_rn = peri_stack(env, rand, PRE, POST) if len(rand) else None
            out[est][(ch, b)] = s_on.mean(0) if s_on is not None else np.full(PRE + POST, np.nan)
            out[f'{est}_null'][(ch, b)] = (s_rn.mean(0) if s_rn is not None
                                           else np.full(PRE + POST, np.nan))
    return out


def rise_onset_time(tax, mean_diff):
    """First time mean_diff crosses +EPS and stays >= 0 for >= 1 s afterwards."""
    hold = int(1.0 * AFS)
    for i in range(len(tax)):
        if mean_diff[i] >= EPS and np.all(mean_diff[i:i + hold] >= 0):
            return float(tax[i])
    return np.nan


def lowband_pass(onsets20):
    """Per-recording curves (one generator across recordings, as legacy run)."""
    rng = np.random.default_rng(0)
    sessions = []
    for lab in LABELS:
        ons = onsets20[lab]
        if len(ons) < MIN_ONSETS:
            continue
        r = onset_curves(fullrate_features(lab), ons, rng)
        r.update(label=lab, subject=subject_of(lab), n_onsets=len(ons))
        sessions.append(r)
    return sessions


def lowband_tables(sessions):
    tax = np.arange(-PRE, POST) / AFS
    subj, ZP = per_subject(sessions, 'zp')
    _, ZPn = per_subject(sessions, 'zp_null')
    _, CA = per_subject(sessions, 'ca')
    _, CAn = per_subject(sessions, 'ca_null')
    near, mid = (tax >= -3) & (tax <= 0), (tax >= -8) & (tax < -3)
    rows = []
    for key in KEYS:
        for est, R, Rn in (('zerophase', ZP, ZPn), ('causal', CA, CAn)):
            diff = R[key] - Rn[key]
            near_ps, mid_ps = diff[:, near].mean(1), diff[:, mid].mean(1)
            rows.append({
                'channel': key[0], 'band_hz': key[1], 'estimator': est,
                'near[-3,0]_realMinusNull_z': round(float(np.nanmean(near_ps)), 3),
                'near_n_subj_pos': int(np.sum(near_ps > 0)),
                'mid[-8,-3]_realMinusNull_z': round(float(np.nanmean(mid_ps)), 3),
                'mid_n_subj_pos': int(np.sum(mid_ps > 0)),
                'rise_onset_t_s': round(rise_onset_time(tax, np.nanmean(diff, 0)), 1)})
    post = tax >= 0
    crows = []
    for key in KEYS:
        pk = np.nanmax(CA[key][:, post], axis=1)
        pkn = np.nanmax(CAn[key][:, post], axis=1)
        lat = tax[post][np.nanargmax(np.nanmean(CA[key], 0)[post])]
        crows.append({'channel': key[0], 'band_hz': key[1],
                      'n_subj_response': int(np.sum(pk > pkn)), 'n_subj': CA[key].shape[0],
                      'mean_peak_z': round(float(np.nanmean(pk)), 3),
                      'mean_null_peak_z': round(float(np.nanmean(pkn)), 3),
                      'grand_peak_latency_s': round(float(lat), 1)})
    return tax, subj, CA, CAn, pd.DataFrame(rows), pd.DataFrame(crows)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Arousal control (arousal_control.py)
# ═════════════════════════════════════════════════════════════════════════════
def arousal_intervals(sess):
    """(start, end) in seconds for every scored arousal, cortical or autonomic."""
    spans = []
    for loader in (load_arousals, load_autonomic_arousals):
        ev = loader(sess)
        if not ev or 'start_hr' not in ev:
            continue
        st = np.asarray(ev['start_hr'], float) * 3600.0
        du = np.asarray(ev.get('duration_s', np.full(len(st), 3.0)), float)
        spans.extend(zip(st, st + du))
    return spans


def arousal_control(onsets20):
    rows, counts = [], []
    tax = np.arange(-PRE, POST) / AFS
    post = (tax >= 0) & (tax <= 15)
    for lab in LABELS:
        ons = onsets20[lab]
        spans = arousal_intervals(get_session(lab, profile=False))
        t_on = ons / AFS
        keep = np.ones(len(t_on), bool)
        for a, b in spans:
            keep &= ~((t_on > a - AROUSAL_GUARD_S) & (t_on < b + AROUSAL_GUARD_S))
        counts.append({'session': lab, 'n_onsets': len(ons), 'n_arousals': len(spans),
                       'n_arousal_free': int(keep.sum()), 'analysed': len(ons) >= MIN_ONSETS})
        if len(ons) < MIN_ONSETS:
            continue
        for subset, sub in (('all', ons), ('arousal_free', ons[keep])):
            if len(sub) < MIN_ONSETS:
                rows.append({'session': lab, 'subject': subject_of(lab), 'subset': subset,
                             'n_onsets': len(sub)})
                continue
            res = onset_curves(fullrate_features(lab), sub, np.random.default_rng(0))
            for key in KEYS:
                prof, null = res['ca'][key], res['ca_null'][key]
                rows.append({'session': lab, 'subject': subject_of(lab), 'subset': subset,
                             'n_onsets': len(sub), 'channel': key[0], 'band': key[1],
                             'peak_z': float(np.nanmax(prof[post])),
                             'peak_lat_s': float(tax[post][np.nanargmax(prof[post])]),
                             'null_peak_z': float(np.nanmax(null[post]))})
    return pd.DataFrame(rows), pd.DataFrame(counts)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Scored K-complex marks (kcomplex_morphology.py)
# ═════════════════════════════════════════════════════════════════════════════
def raw_kcomplex_count(sess):
    """K-Complex marks in the PSG file before clipping to the SEC recording window."""
    files = sorted(glob.glob(str(sess.meta['psg_dir'] / 'PSG_analysis_*' / f'{KC_STEM}*.txt')))
    if not files:
        return 0
    with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
        return sum(1 for line in f
                   if (m := _INTERVAL_RE.match(line.strip()))
                   and m.group(10).strip().lower().startswith('k'))


def score_criteria(env, codes, k_samp, low, high, lab):
    """Put every scored K-complex through the detector's own two gates."""
    q = int(PRE_S * FS)
    out = []
    for c in k_samp:
        c = int(c)
        a, b = c - q, c + int(10 * FS)
        if a < 0 or b > len(env):
            continue
        out.append({'session': lab, 'stage': int(codes[c]),
                    'burst_s': longest_run_s(env[c - int(FS):b] > high),
                    'quiet_ok': bool(env[a:c].mean() < low)})
    return out


def kcomplex_marks():
    """Mark table, detector-gate table, {label: mark times (s)}, raw cohort count."""
    rows, crit, marks, raw = [], [], {}, 0
    for lab in LABELS:
        s = get_session(lab, profile=False)
        raw += raw_kcomplex_count(s)
        ev = load_interval_events(s, KC_STEM)
        if ev is None:
            rows.append({'session': lab, 'n_kcomplex': 0, 'n_spindle': 0,
                         'note': f'no {KC_STEM} file'})
            marks[lab] = np.array([])
            continue
        types = np.array([t.strip().lower() for t in ev['types']])
        is_k = np.char.startswith(types, 'k')
        dur = np.asarray(ev['duration_s'])
        marks[lab] = np.asarray(ev['start_hr'])[is_k] * 3600.0
        rows.append({'session': lab, 'n_kcomplex': int(is_k.sum()),
                     'n_spindle': int((~is_k).sum()),
                     'k_dur_ms_median': float(np.median(dur[is_k])) if is_k.any() else np.nan,
                     'note': ''})
        if is_k.any():
            F = detector_features(lab)
            env, codes = np.asarray(F['env'], float), F['codes']
            nrem = np.isin(codes, NREM_CODES)
            med = np.median(env[nrem])
            mad = np.median(np.abs(env[nrem] - med)) * 1.4826 + 1e-12
            crit += score_criteria(env, codes, marks[lab] * FS,
                                   med + K_LOW * mad, med + K_HIGH * mad, lab)
    return pd.DataFrame(rows), pd.DataFrame(crit), marks, raw


def onset_kcomplex_overlap(onsets, marks):
    """Onsets within ±KC_MATCH_S of a scored K-complex mark, per recording."""
    rows = []
    for lab in LABELS:
        t_on = np.asarray(onsets[lab], float) / FS
        t_k = marks[lab]
        on_mark = np.array([np.any(np.abs(t_k - t) <= KC_MATCH_S) for t in t_on], bool)
        mark_hit = np.array([np.any(np.abs(t_on - t) <= KC_MATCH_S) for t in t_k], bool)
        rows.append({'session': lab, 'n_onsets': len(t_on), 'n_kcomplex': len(t_k),
                     'onsets_on_kcomplex': int(on_mark.sum()),
                     'kcomplexes_with_onset': int(mark_hit.sum())})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 6. SEC response at scored K-complexes (kcomplex_cap_response.py; causal = CORRECTION 4)
# ═════════════════════════════════════════════════════════════════════════════
def lowband_trace(envs, ch, nrem):
    """Mean of the three NREM-standardised band envelopes of one channel."""
    return np.mean([zscore_on(envs[(ch, b)].astype(np.float64), nrem) for b in BANDS], axis=0)


def baselined(stack):
    return stack - stack[:, :KC_BASE_END].mean(axis=1, keepdims=True)


def kcomplex_response(marks):
    """Event stacks (zero-phase and causal), null, EEG control, per-recording table,
    per-event latency table. RNG seeded 11 across recordings, as legacy."""
    rng = np.random.default_rng(11)
    lags = np.arange(-KC_PRE, KC_POST) / AFS
    w = (lags >= KC_PEAK_WIN[0]) & (lags <= KC_PEAK_WIN[1])
    ev = {est: {c: [] for c in CHANNELS} for est in ('zerophase', 'causal')}
    null = {c: [] for c in CHANNELS}
    eeg_stacks, rows, lat_rows = [], [], []
    for lab in LABELS:
        k = np.round(marks[lab] * AFS).astype(int)
        if k.size == 0:
            continue
        envs_zp, eeg_delta, nrem, motion = load_zerophase(lab)
        n = len(eeg_delta)
        k = k[(k >= KC_PRE) & (k + KC_POST < n)]
        if k.size == 0:
            continue
        nulls = random_nrem_centers(nrem, motion, k, KC_PRE, KC_POST,
                                    max(k.size * 20, 200), rng)
        st = peri_stack(zscore_on(eeg_delta, nrem), k, KC_PRE, KC_POST)
        if st is not None:
            eeg_stacks.append(baselined(st))
        envs_ca, _, nrem_ca, _ = load_causal(lab)

        row = {'session': lab, 'n_kcomplex': int(k.size)}
        per_event = {}
        for est, envs, nr in (('zerophase', envs_zp, nrem), ('causal', envs_ca, nrem_ca)):
            for ch in CHANNELS:
                tr = lowband_trace(envs, ch, nr)
                st = baselined(peri_stack(tr, k, KC_PRE, KC_POST))
                ev[est][ch].append(st)
                if est == 'zerophase' and nulls.size:
                    ns = peri_stack(tr, nulls, KC_PRE, KC_POST)
                    if ns is not None:
                        null[ch].append(baselined(ns))
                m = st.mean(axis=0)
                sfx = '' if est == 'zerophase' else '_causal'
                row[f'{ch}_peak_z{sfx}'] = float(m[w].max())
                row[f'{ch}_peak_lat_s{sfx}'] = float(lags[w][np.argmax(m[w])])
                per_event[(est, ch)] = (lags[w][np.argmax(st[:, w], axis=1)], st[:, w].max(1))
        rows.append(row)
        for i, ki in enumerate(k):
            r = {'session': lab, 'mark_s': ki / AFS}
            for est in ('zerophase', 'causal'):
                for ch in CHANNELS:
                    lat, pk = per_event[(est, ch)]
                    r[f'{ch}_lat_s_{est}'] = float(lat[i])
                    r[f'{ch}_peak_z_{est}'] = float(pk[i])
            lat_rows.append(r)

    cat = lambda d: {c: (np.vstack(v) if v else np.zeros((0, KC_PRE + KC_POST)))
                     for c, v in d.items()}
    eeg = np.vstack(eeg_stacks) if eeg_stacks else np.zeros((0, KC_PRE + KC_POST))
    return (lags, {e: cat(d) for e, d in ev.items()}, cat(null), eeg,
            pd.DataFrame(rows), pd.DataFrame(lat_rows), rng)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Numbers
# ═════════════════════════════════════════════════════════════════════════════
def _word(nb, id_, section, what, computed, paper_value, ok, **kw):
    """A number the manuscript states in words (or as a claim): status set by `ok`."""
    nb.add(id_, section, what, computed, paper_value=paper_value,
           status='MATCH' if ok else 'DIFF', **kw)


def _rng(a, b, sep='–', d=3):
    return f'{a:.{d}f}{sep}{b:.{d}f}'


def register(nb, summ, overlap, kc_marks, kc_crit, raw_k, prec, lowchk, resp,
             ar_counts, kc_tab, kc_lat):
    S = '§2.8 ¶162'
    n_on = int(summ.n_onsets.sum())
    per = summ.n_onsets
    analysed = summ[summ.n_onsets >= MIN_ONSETS]
    n_analysed = int(analysed.n_onsets.sum())

    # ── Methods ──────────────────────────────────────────────────────────────
    n_k = int(kc_marks.n_kcomplex.sum())
    n_k_rec = int((kc_marks.n_kcomplex > 0).sum())
    _word(nb, 'kc_marks_methods', S, 'scored K-complex marks inside the SEC recordings',
          n_k, 'Fifty scored marks', n_k == 50, source='kcomplex_marks')
    _word(nb, 'kc_recordings_methods', S, 'recordings with at least one scored K-complex',
          n_k_rec, 'ten of the twelve recordings', n_k_rec == 10, source='kcomplex_marks',
          note='S4N2 has no Spindle K file; S5N1 has the file but no K-complex mark')
    _word(nb, 'delta_null_count', '§2.8 ¶163', 'random-NREM null windows per delta onset',
          '1 per onset (same recording)', 'count-matched baseline windows', True,
          source='random_nrem_centers')
    nb.add('arousal_guard', '§2.8 ¶164', 'arousal-exclusion window', AROUSAL_GUARD_S,
           paper_value='±10 seconds', unit='s', source='arousal_control')
    n_af_ok = int(sum(1 for _, r in ar_counts[ar_counts.analysed].iterrows()
                      if r.n_arousal_free >= MIN_ONSETS))
    nb.add('arousal_testable', '§2.8 ¶164',
           'recordings with >= 5 arousal-free onsets (independently testable)',
           f'{n_af_ok} of {int(ar_counts.analysed.sum())}', source='arousal_control',
           note='recordings with < 5 onsets in total (S4N2, S5N1) are not analysed at all')

    # ── §3.5 ¶206 ────────────────────────────────────────────────────────────
    S = '§3.5 ¶206'
    nb.add('n_onsets', S, 'qualifying delta-burst onsets (q30)', n_on, paper_value='344',
           source='detect_onsets')
    nb.add('onsets_range', S, 'onsets per recording, min to max',
           f'{int(per.min())} to {int(per.max())}', paper_value='1 to 99',
           source='detect_onsets')
    lt10 = summ[summ.n_onsets < 10]
    _word(nb, 'n_rec_lt10', S, 'recordings with fewer than 10 onsets', len(lt10),
          'three recordings contained fewer than 10 events', len(lt10) == 3,
          source='detect_onsets',
          note='CORRECTION: ' + ', '.join(f'{r.session} {r.n_onsets}' for r in lt10.itertuples()))
    nb.add('n_onsets_N2', S, 'onsets scored N2 (of all onsets)',
           f'{int(summ.n_N2.sum())} of {n_on}', source='detect_onsets',
           note=f'{int(summ.n_N3.sum())} in N3; "Most were isolated N2 slow waves"')

    # ── §3.5 ¶207 (causal estimator, response_consistency) ──────────────────
    S = '§3.5 ¶207'
    nb.add('resp_peak_range', S, 'subject-mean post-onset peak across 9 channel-band combos',
           _rng(resp.mean_peak_z.min(), resp.mean_peak_z.max(), ' to '),
           paper_value='1.4 to 3.2', unit='z', source='lowband_tables (causal)')
    nb.add('resp_null_max', S, 'largest subject-mean null post-onset peak',
           float(resp.mean_null_peak_z.max()), paper_value='0.17', unit='z',
           source='lowband_tables (causal)')
    m6 = int(resp.n_subj_response.min())
    _word(nb, 'resp_n_subj', S, 'participants whose peak beats their own null, worst combo',
          f'{m6} of {int(resp.n_subj.max())}', 'all six participants for every channel–band '
          'combination', m6 == 6, source='lowband_tables (causal)')
    nb.add('resp_n_events', S, 'onsets entering the Fig. 7 average', n_analysed,
           paper_value='344', source='lowband_pass',
           note='CORRECTION: recordings with < 5 onsets (S4N2 4, S5N1 1) are skipped, '
                'so the average rests on 339 onsets; still 6 participants')

    # ── §3.5 ¶208 ────────────────────────────────────────────────────────────
    S = '§3.5 ¶208'
    for est in ('zerophase', 'causal'):
        p = prec[prec.estimator == est]
        nb.add(f'auc_{est}', S, f'forecasting AUC, across-subject mean, {est} envelopes',
               _rng(p.auc_pooled.min(), p.auc_pooled.max()), paper_value='0.42–0.56',
               source=f'precursor_summary ({est})',
               note=('CORRECTION: the paper value came from this zero-phase pipeline, '
                     'though ¶208 attributes it to causal filtering. The DIFF is the random '
                     'null draw alone: replaying the legacy draw order (checks/'
                     's08_delta_kcomplex_vs_legacy.py) gives the paper\'s 0.419–0.557'
                     if est == 'zerophase' else
                     'CORRECTION: the causal estimator the text names; per-subject '
                     f'range {p.auc_subj_min.min():.2f}–{p.auc_subj_max.max():.2f}'))
        lags = p.xcorr_peak_lag_s
        _word(nb, f'xcorr_lag_{est}', S,
              f'SEC→EEG-delta cross-correlation peak lag, 9 combos, {est} envelopes',
              _rng(lags.min(), lags.max(), ' to ', 1), 'cross-correlation peaked at zero lag',
              bool(np.all(np.abs(lags) <= 0.5)), unit='s (+ = SEC leads)',
              source=f'precursor_summary ({est})',
              note=('MATCH = every peak within ±0.5 s of zero. ' +
                    ('CORRECTION: this is the zero-phase pipeline the paper value came from'
                     if est == 'zerophase' else
                     'causal: both envelopes carry the filters\' group delay, which differs '
                     'by band (largest for 0.03–0.5 Hz)')
                    + f'; peak r {p.xcorr_peak_r.min():.3f}–{p.xcorr_peak_r.max():.3f}'))
    lo = lowchk[lowchk.band_hz == '0-0.5']
    zp, ca = lo[lo.estimator == 'zerophase'], lo[lo.estimator == 'causal']
    col = 'near[-3,0]_realMinusNull_z'
    nb.add('lowband_zp_near', S, '0–0.5 Hz real−null in [−3,0] s, zero-phase, CLE/CRE/CH range',
           _rng(zp[col].min(), zp[col].max(), d=3), paper_value='0.35–0.41', unit='z',
           source='lowband_tables', note=', '.join(f'{r.channel} {r[col]:.3f}'
                                                    for _, r in zp.iterrows()))
    for _, r in zp.iterrows():
        _word(nb, f'lowband_zp_pos_{r.channel}', S,
              f'participants with real−null > 0 in [−3,0] s, zero-phase, {r.channel}',
              f'{int(r.near_n_subj_pos)} of 6', 'positive in all six participants',
              int(r.near_n_subj_pos) == 6, source='lowband_tables',
              note='CORRECTION: holds for CLE and CRE, not CH' if r.channel == 'CH' else '')
    _word(nb, 'lowband_ca_near', S, '0–0.5 Hz real−null in [−3,0] s, causal, CLE/CRE/CH',
          ', '.join(f'{r.channel} {r[col]:+.3f}' for _, r in ca.iterrows()),
          'to approximately zero', bool(np.all(np.abs(ca[col]) < 0.05)), unit='z',
          source='lowband_tables', note='MATCH = every channel |value| < 0.05 z')
    npos = ca.near_n_subj_pos
    _word(nb, 'lowband_ca_pos', S, 'participants with real−null > 0 in [−3,0] s, causal',
          f'{int(npos.min())}–{int(npos.max())} of 6', 'positive in only two to three '
          'participants', int(npos.min()) == 2 and int(npos.max()) == 3,
          source='lowband_tables',
          note=', '.join(f'{r.channel} {int(r.near_n_subj_pos)}' for _, r in ca.iterrows()))

    # ── §3.5 ¶209 ────────────────────────────────────────────────────────────
    S = '§3.5 ¶209'
    nb.add('resp_latency', S, 'grand-mean post-onset peak latency, 9 combos (causal)',
           _rng(resp.grand_peak_latency_s.min(), resp.grand_peak_latency_s.max(), d=1),
           paper_value='4–8', unit='s', source='lowband_tables (causal)')
    n_on_k = int(overlap.onsets_on_kcomplex.sum())
    _word(nb, 'onsets_on_kcomplex', S, f'onsets within ±{KC_MATCH_S:.0f} s of a scored '
          'K-complex mark', f'{n_on_k} of {n_on}', 'K-complexes, which constituted most '
          'detected onsets', n_on_k > n_on / 2, source='onset_kcomplex_overlap',
          note=f'CORRECTION: only {raw_k} K-complexes were scored cohort-wide ({n_k} inside '
               'the SEC recordings), against thousands expected at 1–3/min in N2: the '
               'reference cannot show what most onsets are')
    nb.add('kc_raw_total', S, 'K-complex marks in the PSG export, cohort-wide (before '
           'clipping to the SEC recording)', raw_k, source='raw_kcomplex_count')
    nb.add('kc_with_onset', S, 'scored K-complexes matched by an onset (±5 s)',
           f'{int(overlap.kcomplexes_with_onset.sum())} of {n_k}',
           source='onset_kcomplex_overlap')
    both = int((kc_crit.quiet_ok & (kc_crit.burst_s >= MIN_BURST_S)).sum())
    nb.add('kc_detector_eligible', S, 'scored K-complexes passing both detector gates '
           '(≥4 s burst, 30 s quiet)', f'{both} of {len(kc_crit)}', source='score_criteria',
           note=f'burst-gate median {kc_crit.burst_s.median():.1f} s')

    # ── Fig. 7 caption ¶212 ──────────────────────────────────────────────────
    S = 'Fig. 7 ¶212'
    _word(nb, 'fig7_n_subj', S, 'participants in the Fig. 7 average', int(resp.n_subj.max()),
          'across six participants', int(resp.n_subj.max()) == 6, source='lowband_tables')
    a = ar_counts[ar_counts.analysed]
    pct = 100 * (1 - a.n_arousal_free.sum() / a.n_onsets.sum())
    per_rec = 100 * (1 - a.n_arousal_free / a.n_onsets)
    _word(nb, 'arousal_within10', S, 'onsets within ±10 s of a scored arousal (pooled)',
          f'{pct:.1f}% ({int(a.n_onsets.sum() - a.n_arousal_free.sum())} of '
          f'{int(a.n_onsets.sum())})', 'Most onsets occurred within 10 s of a scored arousal',
          pct > 50, unit='%', source='arousal_control',
          note=f'per recording {per_rec.min():.0f}–{per_rec.max():.0f}%')
    allp = 100 * (1 - ar_counts.n_arousal_free.sum() / ar_counts.n_onsets.sum())
    nb.add('arousal_within10_all344', S, 'onsets within ±10 s of an arousal, all onsets '
           'incl. S4N2/S5N1', f'{allp:.1f}% of {int(ar_counts.n_onsets.sum())}', unit='%',
           source='arousal_control')
    nb.add('arousal_n_events', S, 'onsets entering the arousal control',
           int(a.n_onsets.sum()), paper_value='344', source='arousal_control',
           note='CORRECTION: S4N2 (4) and S5N1 (1) are skipped (< 5 onsets), so the control '
                'is run on 339 onsets, not the complete set of 344')

    # ── §3.9 ¶241-246 ────────────────────────────────────────────────────────
    S = '§3.9 ¶241'
    _word(nb, 'kc_marks', S, 'scored K-complex marks', n_k, '50 marks', n_k == 50,
          source='kcomplex_marks')
    _word(nb, 'kc_recordings', S, 'recordings contributing marks', n_k_rec,
          'ten of the twelve recordings', n_k_rec == 10, source='kcomplex_marks')
    S = '§3.9 ¶242'
    for ch, pv in (('CLE', '3.6'), ('CRE', '3.7'), ('CH', '4.3')):
        for est in ('zerophase', 'causal'):
            nb.add(f'kc_latency_{ch}_{est}', S,
                   f'median per-event peak latency, {ch}, {est} envelopes',
                   float(np.median(kc_lat[f'{ch}_lat_s_{est}'])), paper_value=pv, unit='s',
                   source=f'kcomplex_response ({est})',
                   note=('CORRECTION: the paper value came from zero-phase envelopes, against '
                         'Methods ¶163' if est == 'zerophase' else
                         'CORRECTION: causal envelopes, as Methods ¶163 specifies; the '
                         'causal filters add their own group delay'))
    for est in ('zerophase', 'causal'):
        for ch in CHANNELS:
            mean_peak = float(np.nanmean(kc_lat[f'{ch}_peak_z_{est}']))
            nb.add(f'kc_event_peak_{ch}_{est}', S, f'mean per-event peak, {ch}, {est}',
                   mean_peak, unit='z', source='kcomplex_response')
    S = '§3.9 ¶244'
    kc_per = kc_tab.n_kcomplex
    _word(nb, 'kc_per_recording', S, 'scored K-complexes per contributing recording',
          f'{int(kc_per.min())} to {int(kc_per.max())}', 'from one to ten events',
          int(kc_per.min()) == 1 and int(kc_per.max()) == 10, source='kcomplex_response')
    S = 'Fig. 12 ¶246'
    nb.add('kc_marks_fig12', S, 'scored marks in the Fig. 12 average', int(kc_per.sum()),
           paper_value='50', source='kcomplex_response')
    _word(nb, 'kc_null_count', S, 'random-NREM null windows per recording',
          'max(20 × n marks, 200)', 'count-matched random-NREM null', False,
          source='kcomplex_response',
          note='legacy null is 20× the event count (min 200), not count-matched; '
               'kept as in legacy, the text should say so')


# ═════════════════════════════════════════════════════════════════════════════
def run():
    TABLES.mkdir(parents=True, exist_ok=True)
    nb = Numbers(STAGE)

    print('  detecting delta-burst onsets (q30)')
    ev_tab, summ, onsets = detect_all()
    onsets20 = to_20hz(onsets)
    ev_tab.to_csv(TABLES / 'delta_onsets.csv', index=False)
    summ.to_csv(TABLES / 'delta_onsets_summary.csv', index=False)
    print(f'    {int(summ.n_onsets.sum())} onsets')

    print('  scored K-complex marks')
    kc_marks, kc_crit, marks, raw_k = kcomplex_marks()
    overlap = onset_kcomplex_overlap(onsets, marks)
    kc_marks.to_csv(TABLES / 'kcomplex_marks.csv', index=False)
    kc_crit.to_csv(TABLES / 'kcomplex_criteria.csv', index=False)
    overlap.to_csv(TABLES / 'kcomplex_onset_overlap.csv', index=False)

    print('  precursor tests: AUC + cross-correlation, zero-phase and causal')
    prec = []
    for est, load in (('zerophase', load_zerophase), ('causal', load_causal)):
        xc = xcorr_pass(LABELS, load)
        prec.append(precursor_summary(
            onset_pass(LABELS, onsets20, load, xc, np.random.default_rng(0)), est))
    prec = pd.concat(prec, ignore_index=True)
    prec.to_csv(TABLES / 'precursor_summary.csv', index=False)

    print('  peri-onset curves, Fig. 7')
    low_sessions = lowband_pass(onsets20)
    tax, subj, CA, CAn, lowchk, resp = lowband_tables(low_sessions)
    lowchk.to_csv(TABLES / 'lowband_precursor_check.csv', index=False)
    resp.to_csv(TABLES / 'response_consistency.csv', index=False)
    n_events = sum(s['n_onsets'] for s in low_sessions)
    save(fig7_causal_grid(tax, CA, CAn, len(subj), n_events), 'fig7_delta_onset_causal_grid',
         STAGE)

    print('  arousal control')
    ar, ar_counts = arousal_control(onsets20)
    ar.to_csv(TABLES / 'arousal_control.csv', index=False)
    ar_counts.to_csv(TABLES / 'arousal_control_counts.csv', index=False)

    print('  K-complex response, Fig. 12')
    lags, ev, null, eeg, kc_tab, kc_lat, rng = kcomplex_response(marks)
    kc_tab.to_csv(TABLES / 'kcomplex_cap_response.csv', index=False)
    kc_lat.to_csv(TABLES / 'kcomplex_latencies.csv', index=False)
    save(fig12_kcomplex(lags, ev['zerophase'], null, eeg, kc_tab, KC_PEAK_WIN, rng),
         'fig12_kcomplex_response', STAGE)

    register(nb, summ, overlap, kc_marks, kc_crit, raw_k, prec, lowchk, resp, ar_counts,
             kc_tab, kc_lat)
    nb.save()


if __name__ == '__main__':
    run()
