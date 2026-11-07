"""Signal origin (review of V18, Major 11): do the SEC low-frequency ridges and harmonic
combs also appear in the PSG's own peripheral signals?

    LFO ridges   the s05 low-band detector (_s05_lowband.night) run unchanged on the PSG
                 finger PPG (Pleth) and on SEC C_RE, on the same column grid: share of the
                 night with a gated ridge, share of SEC ridge time that also has a Pleth
                 ridge (against the Pleth base rate), and frequency agreement where both
                 have one.
    Combs        the s06 comb score (background-corrected spectrogram, mean height of the
                 first four harmonics, per-night robust z) computed on thoracic effort,
                 abdominal effort and nasal flow. For each SEC comb event (s06 merged
                 events): the PSG channel's mean z during the event against 200 random
                 NREM spans of the same length from the same night; the breathing rate
                 (thorax spectral peak, 0.1-0.5 Hz) during the event against the event's
                 comb fundamental; and whether the comb's own tracked bands (s06
                 ladder_bands) sit within 0.03 Hz of a multiple of that breathing rate
                 more often than chance (2 x 0.03 / breathing rate).

Caveats: the belts are band-limited at acquisition (0.1-1 Hz; flow low-pass 1 Hz), so comb
harmonics above 1 Hz cannot appear in them. A PSG channel is skipped on a night where it is
flat in more than 30 % of 30-s epochs. Every number is NEW.

Reads   outputs/tables/s06_harmonic_comb/{ladder_events,ladder_bands}.csv, the sessions
Writes  outputs/tables/s11_psg_origin/*.csv, outputs/numbers/s11_psg_origin.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

from seclib import TAB_DIR, Numbers, iter_sessions
from stages import _s05_lowband as LB
import stages.s06_harmonic_comb as S6

STAGE = 's11_psg_origin'
REQUIRES = ['s05_ridges', 's06_harmonic_comb']

OUT = TAB_DIR / STAGE
SEC_LFO_CH = 'CRE'                       # the channel of the V18 LFO results
COMB_PSG = ['Thorax', 'Abdomen', 'Flow']
NREM = (1, 2, 3)
FLAT_REL = 0.01                          # epoch SD < 1 % of the night's median = flat
FLAT_MAX = 0.30
N_NULL = 200
BREATH = (0.1, 0.5)
HARM_TOL = 0.03                          # Hz: a band "at" k x breathing rate
F0_FLOOR_STEPS = 2                       # fitted f0 within 2 grid steps of S6.F0_LO


def flat_fraction(x, fs):
    n = int(30 * fs)
    m = len(x) // n
    sd = x[: m * n].reshape(m, n).std(1)
    return float(np.mean(sd < FLAT_REL * np.median(sd[sd > 0]))) if (sd > 0).any() else 1.0


def comb_z(sig, fs):
    """s06.detect_channel's z series, without the accelerometer regression (PSG signal)."""
    f, t_hr, enh = S6._enhance_spec(sig, fs)
    sc, f0 = S6.comb_score(f, enh)
    sm = uniform_filter1d(sc, max(1, int(round(S6.SMOOTH_MIN * 60 / S6.STEP_SEC))))
    med = np.median(sm)
    z = (sm - med) / (1.4826 * np.median(np.abs(sm - med)) + 1e-9)
    return t_hr, z, f0


def lfo_compare(s):
    sec = LB.night(s.cap[SEC_LFO_CH].astype(np.float64), s.fs)
    ppg_raw = np.nan_to_num(s.psg['Pleth'].astype(np.float64))
    flat = flat_fraction(ppg_raw, s.fs)
    row = dict(session=s.label, subject=s.subject, pleth_flat_frac=flat)
    if flat > FLAT_MAX:
        return row
    ppg = LB.night(ppg_raw, s.fs)
    n = min(len(sec['keep']), len(ppg['keep']))
    ks, kp = sec['keep'][:n], ppg['keep'][:n]
    both = ks & kp
    row.update(sec_ridge_frac=float(ks.mean()), pleth_ridge_frac=float(kp.mean()),
               p_pleth_given_sec=float(both.sum() / ks.sum()) if ks.any() else np.nan,
               jaccard=float(both.sum() / (ks | kp).sum()) if (ks | kp).any() else np.nan,
               n_sec_episodes=len(sec['episodes']), n_pleth_episodes=len(ppg['episodes']))
    if both.any():
        d = np.abs(sec['fr'][:n][both] - ppg['fr'][:n][both])
        row.update(freq_absdiff_med_hz=float(np.median(d)),
                   freq_within_0p01=float(np.mean(d <= 0.01)),
                   sec_f_med=float(np.median(sec['fr'][:n][both])),
                   pleth_f_med=float(np.median(ppg['fr'][:n][both])))
    return row


def event_f0(bands, lab, t0, t1):
    b = bands[(bands.session == lab) & (bands.ep_t1_hr >= t0) & (bands.ep_t0_hr <= t1)]
    return float(b.drop_duplicates(['channel', 'episode']).ep_f0_hz.median()) if len(b) else np.nan


def event_bands(bands, lab, t0, t1):
    """Frequencies of the s06 tracked bands of every channel episode overlapping the event."""
    b = bands[(bands.session == lab) & (bands.ep_t1_hr >= t0) & (bands.ep_t0_hr <= t1)]
    return b.drop_duplicates(['channel', 'episode', 'band_hz']).band_hz.to_numpy(float)


def comb_compare(s, ev, bands, rng):
    sp = s.sleep_profile
    rows, qual = [], []
    zs = {}
    for ch in COMB_PSG:
        x = np.nan_to_num(s.psg[ch].astype(np.float64))
        flat = flat_fraction(x, s.fs)
        qual.append(dict(session=s.label, channel=ch, flat_frac=flat, used=flat <= FLAT_MAX))
        if flat <= FLAT_MAX:
            zs[ch] = comb_z(x, s.fs)
    thorax = np.nan_to_num(s.psg['Thorax'].astype(np.float64))
    t_end = len(thorax) / s.fs / 3600.0
    for _, e in ev[ev.session == s.label].iterrows():
        r = dict(session=s.label, subject=s.subject, t0_hr=e.t0_hr, t1_hr=e.t1_hr,
                 dur_min=e.dur_min, dom_stage=e.dom_stage,
                 sec_f0_hz=event_f0(bands, s.label, e.t0_hr, e.t1_hr))
        a, b = int(e.t0_hr * 3600 * s.fs), int(e.t1_hr * 3600 * s.fs)
        if b - a > int(60 * s.fs):
            f, p = welch(thorax[a:b], fs=s.fs, nperseg=int(60 * s.fs))
            m = (f >= BREATH[0]) & (f <= BREATH[1])
            br = float(f[m][np.argmax(p[m])])
            r['thorax_breath_hz'] = br
            # do the comb's own tracked bands sit at multiples of the breathing rate?
            fb = event_bands(bands, s.label, e.t0_hr, e.t1_hr)
            if fb.size:
                k = np.maximum(np.round(fb / br), 1)
                r['n_bands'] = int(fb.size)
                r['bands_at_breath_multiple'] = float(np.mean(np.abs(fb - k * br) <= HARM_TOL))
                r['bands_chance'] = min(1.0, 2 * HARM_TOL / br)
        dur = e.t1_hr - e.t0_hr
        for ch, (t_hr, z, _) in zs.items():
            ins = (t_hr >= e.t0_hr) & (t_hr <= e.t1_hr)
            if not ins.any():
                continue
            ze = float(z[ins].mean())
            null = []
            tries = 0
            while len(null) < N_NULL and tries < 50 * N_NULL:
                tries += 1
                t0 = rng.uniform(0, max(t_end - dur, 1e-6))
                if S6._stage_at(sp, t0 + dur / 2) not in NREM:
                    continue
                w = (t_hr >= t0) & (t_hr <= t0 + dur)
                if w.any():
                    null.append(z[w].mean())
            null = np.asarray(null)
            r[f'{ch}_z_event'] = ze
            r[f'{ch}_z_null_med'] = float(np.median(null)) if null.size else np.nan
            r[f'{ch}_z_null_p95'] = float(np.percentile(null, 95)) if null.size else np.nan
            r[f'{ch}_event_pct'] = float(np.mean(null < ze) * 100) if null.size else np.nan
        rows.append(r)
    return rows, qual


def register(nb, lfo, comb, qual, bands):
    ok = lfo.dropna(subset=['sec_ridge_frac'])
    nb.add('lfo_nights_tested', 'V18 §3.3 / Discussion', 'nights with usable Pleth (flat <= 30 %)',
           f'{len(ok)} of {len(lfo)}')
    nb.add('lfo_sec_frac', 'V18 §3.3', 'share of night with an SEC C_RE LFO ridge, median (range)',
           f'{ok.sec_ridge_frac.median():.2f} ({ok.sec_ridge_frac.min():.2f}–'
           f'{ok.sec_ridge_frac.max():.2f})')
    nb.add('lfo_pleth_frac', 'V18 §3.3', 'share of night with a Pleth LFO ridge (same detector), '
           'median (range)', f'{ok.pleth_ridge_frac.median():.2f} ({ok.pleth_ridge_frac.min():.2f}–'
           f'{ok.pleth_ridge_frac.max():.2f})')
    has = ok.dropna(subset=['p_pleth_given_sec'])
    nb.add('lfo_overlap', 'V18 §3.3', 'P(Pleth ridge | SEC ridge) vs P(Pleth ridge), median over '
           'nights with an SEC ridge', f'{has.p_pleth_given_sec.median():.2f} vs '
           f'{has.pleth_ridge_frac.median():.2f} (n = {len(has)})')
    nb.add('lfo_overlap_above_chance', 'V18 §3.3', 'nights where P(Pleth | SEC) > P(Pleth)',
           f'{int((has.p_pleth_given_sec > has.pleth_ridge_frac).sum())} of {len(has)}')
    fq = ok.dropna(subset=['freq_absdiff_med_hz'])
    if len(fq):
        nb.add('lfo_freq_agree', 'V18 §3.3', 'where both have a ridge: median |f_SEC - f_Pleth| '
               '(median over nights); share of columns within 0.01 Hz',
               f'{fq.freq_absdiff_med_hz.median():.3f} Hz; {fq.freq_within_0p01.median():.2f}')
    for ch in COMB_PSG:
        col = f'{ch}_z_event'
        if col not in comb:
            continue
        c = comb.dropna(subset=[col])
        nb.add(f'comb_{ch}_z', 'V18 §3.7', f'{ch} comb z during SEC comb events vs random NREM '
               'spans: median event z vs median null z (events)',
               f'{c[col].median():.2f} vs {c[f"{ch}_z_null_med"].median():.2f} (n = {len(c)})')
        nb.add(f'comb_{ch}_above_p95', 'V18 §3.7', f'SEC comb events where {ch} z exceeds the '
               "night's 95th percentile of random NREM spans",
               f'{int((c[col] > c[f"{ch}_z_null_p95"]).sum())} of {len(c)}')
    b = comb.dropna(subset=['thorax_breath_hz', 'sec_f0_hz'])
    if len(b):
        ratio = b.sec_f0_hz / b.thorax_breath_hz
        nb.add('comb_f0_vs_breath', 'V18 §3.7', 'SEC comb fundamental / thorax breathing rate '
               'during the event: median (IQR); events within ±10 %',
               f'{ratio.median():.2f} ({ratio.quantile(.25):.2f}–{ratio.quantile(.75):.2f}); '
               f'{int((ratio.sub(1).abs() <= 0.10).sum())} of {len(b)}')
    h = comb.dropna(subset=['bands_at_breath_multiple'])
    if len(h):
        nb.add('comb_bands_at_breath', 'V18 §3.7', 'share of an event\'s comb bands within '
               f'±{HARM_TOL} Hz of a multiple of the thorax breathing rate: median vs chance; '
               'events above chance',
               f'{h.bands_at_breath_multiple.median():.2f} vs {h.bands_chance.median():.2f}; '
               f'{int((h.bands_at_breath_multiple > h.bands_chance).sum())} of {len(h)}')
    ep = bands.drop_duplicates(['session', 'channel', 'episode'])
    floor = S6.F0_LO + F0_FLOOR_STEPS * S6.F0_STEP + 1e-9
    nb.add('comb_f0_at_floor', 'V18 §2.5', f'channel episodes whose fitted f0 is within '
           f'{F0_FLOOR_STEPS} grid steps of the {S6.F0_LO} Hz search floor',
           f'{int((ep.ep_f0_hz <= floor).sum())} of {len(ep)}')
    q = qual.groupby('channel').used.agg(['sum', 'count'])
    for ch, r in q.iterrows():
        nb.add(f'psg_used_{ch}', 'methods', f'nights with usable {ch} (flat <= 30 %)',
               f'{int(r["sum"])} of {int(r["count"])}')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    nb = Numbers(STAGE)
    src = TAB_DIR / 's06_harmonic_comb'
    ev = pd.read_csv(src / 'ladder_events.csv')
    bands = pd.read_csv(src / 'ladder_bands.csv')
    rng = np.random.default_rng(0)
    lfo, comb, qual = [], [], []
    for s in iter_sessions():
        lfo.append(lfo_compare(s))
        r, q = comb_compare(s, ev, bands, rng)
        comb += r
        qual += q
        print(f'  {s.label}: LFO + comb done')
    lfo, comb, qual = pd.DataFrame(lfo), pd.DataFrame(comb), pd.DataFrame(qual)
    lfo.to_csv(OUT / 'lfo_sec_vs_pleth.csv', index=False)
    comb.to_csv(OUT / 'comb_events_psg.csv', index=False)
    qual.to_csv(OUT / 'psg_channel_quality.csv', index=False)
    register(nb, lfo, comb, qual, bands)
    nb.save()


if __name__ == '__main__':
    run()
