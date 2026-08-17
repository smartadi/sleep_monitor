"""
Interactive CALIBRATION GUI for the harmonic-ladder band detector.

Drag the sliders and watch, live on the spectrogram, which horizontal bands the
detector keeps — so you can see exactly what each parameter does and pick good
values.  Episode time-extents are fixed (computed once); the sliders tune the
band tracker (the sensitivity knobs).

Pipeline the sliders expose (per channel):
  1. spectrogram, each column minus its smooth frequency background  -> "enhanced"
  2. TSMOOTH   : time-smooth so steady rungs survive, transient noise averages down
  3. BAND_DB   : keep peaks at least this many dB above the local floor
  4. BAND_PROM : ... and at least this prominent (a real rung is a clear peak)
  5. link surviving peaks across time within BAND_JUMP Hz  -> flat bands
  6. MIN_BAND_SEC + BAND_COVER : keep only bands that PERSIST and are CONSISTENT

Run:  python ladder_calibrate.py --session S6N1 --channel CH
      python ladder_calibrate.py --test     # headless plumbing check -> PNG
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib

SCRATCH = Path(r"C:/Users/adity/AppData/Local/Temp/claude/"
               r"C--Users-adity-Documents-sleep-monitor-code/"
               r"331097c7-ff4f-45af-9a10-5cec26a81f37/scratchpad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='S6N1')
    ap.add_argument('--channel', default='CH')
    ap.add_argument('--test', action='store_true', help='headless build -> PNG')
    args = ap.parse_args()

    matplotlib.use('Agg' if args.test else 'TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sleep_monitor import load_session, load_sleep_profile
    import harmonic_ladder_overlay as H

    session = None
    for i in range(12):
        s = load_session(i)
        if s.label == args.session:
            s.sleep_profile = load_sleep_profile(s)
            session = s
            break
    if session is None:
        print(f'session {args.session} not found')
        return

    state = {'ch': args.channel if args.channel in H.CHANNELS else 'CH'}

    def compute(ch):
        f, t_hr, enh, active, episodes = H.detect_channel(session, ch)
        state.update(f=f, t_hr=t_hr, enh=enh,
                     extents=[(e['lo'], e['hi']) for e in episodes])
    compute(state['ch'])

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([0.30, 0.32, 0.66, 0.60])
    lines = []

    def draw_spec():
        ax.clear()
        ax.pcolormesh(state['t_hr'], state['f'], state['enh'], cmap='magma',
                      vmin=0, vmax=np.percentile(state['enh'], 99.5), shading='gouraud')
        ax.set_ylim(0, H.FMAX)
        ax.set_xlabel('Time (hr)')
        ax.set_ylabel('Frequency (Hz)')

    def redraw(event=None):
        H.BAND_DB = s_db.val
        H.BAND_PROM = s_prom.val
        H.MIN_BAND_SEC = s_min.val
        H.BAND_COVER = s_cov.val
        H.TSMOOTH = int(round(s_ts.val))
        for ln in lines:
            ln.remove()
        lines.clear()
        f, t_hr, enh = state['f'], state['t_hr'], state['enh']
        n = 0
        for lo, hi in state['extents']:
            for fr, s0, s1 in H.track_bands(enh, f, lo, hi):
                ln, = ax.plot([t_hr[s0], t_hr[s1]], [fr, fr], color='#00E5FF', lw=2.2)
                lines.append(ln)
                n += 1
        ax.set_title(f"{session.label}  {state['ch']}  —  "
                     f"{len(state['extents'])} episode(s), {n} bands  "
                     f"(dB{H.BAND_DB:.1f} prom{H.BAND_PROM:.1f} "
                     f"min{H.MIN_BAND_SEC:.0f}s cov{H.BAND_COVER:.2f} ts{H.TSMOOTH})",
                     fontsize=11)
        fig.canvas.draw_idle()

    def sax(y):
        return fig.add_axes([0.34, y, 0.55, 0.025])
    s_db = Slider(sax(0.22), 'BAND_DB (dB)', 0.0, 8.0, valinit=H.BAND_DB)
    s_prom = Slider(sax(0.18), 'BAND_PROM', 0.0, 4.0, valinit=H.BAND_PROM)
    s_min = Slider(sax(0.14), 'MIN_BAND_SEC', 15.0, 300.0, valinit=H.MIN_BAND_SEC)
    s_cov = Slider(sax(0.10), 'BAND_COVER', 0.3, 1.0, valinit=H.BAND_COVER)
    s_ts = Slider(sax(0.06), 'TSMOOTH (win)', 1, 15, valinit=H.TSMOOTH, valstep=1)
    for sl in (s_db, s_prom, s_min, s_cov, s_ts):
        sl.on_changed(redraw)

    rax = fig.add_axes([0.04, 0.62, 0.16, 0.16])
    rax.set_title('channel', fontsize=9)
    radio = RadioButtons(rax, tuple(H.CHANNELS),
                         active=H.CHANNELS.index(state['ch']))

    def setch(label):
        state['ch'] = label
        compute(label)
        draw_spec()
        redraw()
    radio.on_clicked(setch)

    fig.text(0.04, 0.45,
             "algorithm\n"
             "----------\n"
             "1 enhance: column minus\n  smooth freq background\n"
             "2 TSMOOTH: time-smooth\n  (steady rungs survive)\n"
             "3 BAND_DB / PROM:\n  keep bright, peaky bands\n"
             "4 link peaks over time\n  (flat rungs)\n"
             "5 MIN_BAND_SEC + COVER:\n  keep persistent,\n  consistent bands",
             fontsize=8, va='top', family='monospace')

    draw_spec()
    redraw()

    if args.test:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        out = SCRATCH / 'calib_test.png'
        fig.savefig(out, dpi=100)
        print(f'test build OK -> {out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
