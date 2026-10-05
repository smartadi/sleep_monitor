"""
Supplement section S6 (harmonic-comb events): figures at print size.

Built on the paper pipeline's comb detector (paper/stages/s06_harmonic_comb.py),
so the events here are exactly the events that the manuscript (§3.6, Figs. 8-9)
reports. Same print style as the other supplementary figures
(analysis/rates/_supp_style.py: 9 in wide for a 6 in text width, no titles,
panel letters).

    S6 fig 1  all events on the twelve nights, over the scored stages
    S6 fig 2  four events up close: spectrogram 0-5 Hz with the detected bands
    S6 fig 3  per-event properties: duration, number of bands, band spacing

Reads   paper/outputs/tables/s06_harmonic_comb/ladder_events.csv, ladder_bands.csv
Writes  writeup/figures/supp_s6/*.png

Usage
    .venv/Scripts/python.exe paper/run_all.py --only s06_harmonic_comb   (if tables missing)
    .venv/Scripts/python.exe analysis/slow_wave/supp_comb_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
sys.path.insert(0, str(ROOT / 'analysis' / 'rates'))

import seclib                                         # noqa: E402
from seclib.config import STAGE_COLORS, STAGE_LABELS, STAGE_ORDER   # noqa: E402
from stages import s06_harmonic_comb as C             # noqa: E402
import _supp_style as st                              # noqa: E402

st.apply()
TAB = ROOT / 'paper' / 'outputs' / 'tables' / 's06_harmonic_comb'
OUT = ROOT / 'writeup' / 'figures' / 'supp_s6'
OUT.mkdir(parents=True, exist_ok=True)
STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']
SCOL = {STAGE_LABELS[c]: STAGE_COLORS[c] for c in STAGE_ORDER}
EVENT_C = '#111111'
# four events chosen to span participants and durations
PERSIST = 0.5    # a band counts as a rung if it lasts >= this share of the event
EXAMPLES = [('S6N1', 2.271), ('S3N2', 4.079), ('S2N1', 0.346), ('S5N2', 0.496)]


def save(fig, name):
    p = OUT / f'{name}.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p.relative_to(ROOT))


def stage_band(ax, sp, xmax):
    t = np.asarray(sp['t_ep_hr'], float)
    codes = np.asarray(sp['codes'])
    for c in STAGE_ORDER:
        ax.fill_between(t, 0, 1, where=codes == c, step='post', color=STAGE_COLORS[c], lw=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for s in ('left', 'top', 'right'):
        ax.spines[s].set_visible(False)


def fig_overview(ev):
    labels = seclib.LABELS
    fig, axes = plt.subplots(len(labels), 1, figsize=(st.WIDTH_IN, 5.6), sharex=True)
    for ax, lab in zip(axes, labels):
        s = seclib.get_session(lab)
        stage_band(ax, s.sleep_profile, 9.0)
        for _, e in ev[ev.session == lab].iterrows():
            ax.add_patch(plt.Rectangle((e.t0_hr, -0.35), e.t1_hr - e.t0_hr, 1.7,
                                       fc='none', ec=EVENT_C, lw=1.6, clip_on=False))
        ax.text(-0.012, 0.5, f'{lab}  ({int((ev.session == lab).sum())})',
                transform=ax.transAxes, ha='right', va='center', fontsize=10.5)
        ax.tick_params(labelbottom=False, bottom=False)
    axes[-1].tick_params(labelbottom=True, bottom=True)
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel('time (h)')
    h = [plt.Rectangle((0, 0), 1, 1, color=SCOL[x]) for x in STAGES]
    h.append(plt.Rectangle((0, 0), 1, 1, fc='none', ec=EVENT_C, lw=1.6))
    fig.legend(h, STAGES + ['comb event'], loc='lower center', ncol=6, frameon=False,
               bbox_to_anchor=(0.55, 0.90), fontsize=10.5)
    fig.subplots_adjust(hspace=1.1, left=0.16, top=0.86)
    save(fig, 'figS6_1_events_overview')


def _best_channel(bands, lab, t0):
    b = bands[(bands.session == lab) & (np.abs(bands.ep_t0_hr - t0) < 0.25)]
    return b.groupby('channel').size().idxmax(), b


def fig_examples(bands):
    fig, axes = plt.subplots(2, 2, figsize=(st.WIDTH_IN, 6.6))
    for ax, (lab, t0), l in zip(axes.ravel(), EXAMPLES, 'abcd'):
        ch, b = _best_channel(bands, lab, t0)
        s = seclib.get_session(lab)
        f, t_hr, enh, _, episodes = C.detect_channel(s, ch)
        ep = min(episodes, key=lambda e_: abs(t_hr[e_['lo']] - t0))
        a, z = t_hr[ep['lo']], t_hr[min(ep['hi'], len(t_hr) - 1)]
        pad = 0.25
        m = (t_hr >= a - pad) & (t_hr <= z + pad)
        tm = (t_hr[m] - a) * 60
        ax.pcolormesh(tm, f, enh[:, m], shading='gouraud', cmap='magma', vmin=0,
                      vmax=np.percentile(enh[:, m], 99.5), rasterized=True)
        span = ep['hi'] - ep['lo']
        n_rung = 0
        for fr, s0, s1 in ep['bands']:
            keep = (s1 - s0 + 1) >= PERSIST * span
            n_rung += keep
            if keep:     # brief fragments are not drawn
                ax.plot([(t_hr[s0] - a) * 60, (t_hr[s1] - a) * 60], [fr, fr],
                        color='#00E5FF', lw=2.0)
        ax.axvline(0, color='w', ls=':', lw=0.8)
        ax.axvline((z - a) * 60, color='w', ls=':', lw=0.8)
        ax.set_ylim(0, C.FMAX)
        ax.set_xlim(tm[0], tm[-1])
        ax.set_xlabel('minutes from event onset')
        ax.set_ylabel('frequency (Hz)')
        stg = C._stage_at(s.sleep_profile, (a + z) / 2)
        ax.set_title(f'{lab}, {ch}, {STAGE_LABELS.get(stg, "?")}: {n_rung} sustained bands',
                     loc='right', fontsize=10.5)
        st.letter(ax, l, x=-0.2)
    fig.tight_layout()
    save(fig, 'figS6_2_examples')


def fig_properties(ev, bands):
    per = []
    for _, e in ev.iterrows():
        b = bands[(bands.session == e.session) & (bands.ep_t0_hr < e.t1_hr + 1 / 60)
                  & (bands.ep_t1_hr > e.t0_hr - 1 / 60)]
        if b.empty:
            per.append((e.dur_min, 0, np.nan))
            continue
        ch = b.groupby('channel').size().idxmax()
        bb = b[b.channel == ch]
        dur = (e.t1_hr - e.t0_hr) * 60
        fr = np.sort(bb[bb.band_min >= PERSIST * dur].band_hz.round(3).unique())
        sp = np.median(np.diff(fr)) if len(fr) > 1 else np.nan
        per.append((e.dur_min, len(fr), sp))
    per = np.array(per, float)
    fig, axes = plt.subplots(1, 3, figsize=(st.WIDTH_IN, 3.0))
    for ax, (vals, xl, bins), l in zip(axes, (
            (per[:, 0], 'event duration (min)', np.arange(0, 67, 6)),
            (per[:, 1], 'sustained bands per event', np.arange(-0.5, np.nanmax(per[:, 1]) + 1.5, 1)),
            (per[:, 2], 'median band spacing (Hz)', np.arange(0.0, 0.65, 0.05))), 'abc'):
        v = vals[np.isfinite(vals)]
        ax.hist(v, bins=bins, color='#5B6B7F', edgecolor='white')
        ax.axvline(np.median(v), color='k', ls='--', lw=1)
        ax.set_xlabel(xl)
        st.letter(ax, l, x=-0.25)
    axes[0].set_ylabel('events')
    fig.tight_layout()
    save(fig, 'figS6_3_properties')
    return per


def main():
    ev = pd.read_csv(TAB / 'ladder_events.csv')
    bands = pd.read_csv(TAB / 'ladder_bands.csv')
    fig_overview(ev)
    fig_examples(bands)
    per = fig_properties(ev, bands)
    pd.DataFrame(per, columns=['dur_min', 'n_bands', 'band_spacing_hz']).assign(
        session=ev.session.values, t0_hr=ev.t0_hr.values).to_csv(
        TAB / 'ladder_event_properties.csv', index=False)
    print('duration median', np.nanmedian(per[:, 0]), 'range', np.nanmin(per[:, 0]),
          np.nanmax(per[:, 0]))
    print('bands median', np.nanmedian(per[:, 1]), 'range', np.nanmin(per[:, 1]),
          np.nanmax(per[:, 1]))
    print('spacing median', round(np.nanmedian(per[:, 2]), 3), 'IQR',
          np.nanpercentile(per[:, 2], [25, 75]).round(3))


if __name__ == '__main__':
    main()
