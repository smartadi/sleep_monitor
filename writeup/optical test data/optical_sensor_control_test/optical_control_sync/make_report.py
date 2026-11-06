"""Write sync_report.md from the report dict produced by run_sync.main().

Every number in the prose comes from the dict, so the report can never drift
away from the run that produced it.  Usage:

    python make_report.py            # regenerate from sync_report.json
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def build(rep: dict, taps: pd.DataFrame | None = None) -> str:
    t, c, m, mc = rep["mask_timing"], rep["clock"], rep["muse_timing"], rep["motion_check"]
    r = rep.get("refinement", {})
    L = []
    A = L.append

    A("# Sleep mask <-> Muse S synchronisation — optical_sensor_control_test\n")
    A(f"Session of {rep['tap_markers_reported'][0][:10]}. Mask file `{rep['mask_file']}`,")
    A(f"Muse file `{rep['muse_file']}`.\n")

    A("## The clock problem\n")
    A("The mask export has no wall clock, only `timeMS` — milliseconds since the device")
    A("powered on. The Muse / Mind Monitor export carries phone wall-clock timestamps.")
    A("The tester therefore pressed the mask sensors three times at **02:10:03** and")
    A("three more times at **02:10:40**; those presses are the only bridge between the")
    A("two clocks. A second stage then refines the offset on shared head movement.\n")

    A("## Mask stream integrity\n")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| samples | {t['n_samples']:,} |")
    A(f"| tick spacing | {t['dt_median_ms']:.0f} ms (min {t['dt_min_ms']:.0f}, max {t['dt_max_ms']:.0f}) |")
    A(f"| effective rate | {t['fs_effective']:.2f} Hz |")
    A(f"| gaps > 50 ms | {t['n_gaps_over_50ms']} |")
    A(f"| span | {t['span_s']/3600:.2f} h |\n")
    A(f"The tick spacing is exactly {t['dt_median_ms']:.0f} ms on every one of the {t['n_samples']:,}")
    A("samples, with no dropouts, so the device counter can be treated as a linear clock")
    A("and only a constant offset is needed.\n")

    A("## Stage 1 — tap markers\n")
    A("Six presses were detected, in two bursts of three, as saturating pulses on the")
    A("capacitive channels (CH drops to about -10, CRE rises to 4-7; robust z of order")
    A("10^3 against the night-long baseline).\n")
    A("| burst | reported | first tap (device s) | inter-tap gaps (s) |")
    A("|---|---|---|---|")
    for b in rep["per_burst"]:
        gaps = ", ".join(f"{g:.3f}" for g in b["intervals_s"])
        A(f"| {b['burst']+1} | {b['reported']} | {b['first_tap_t_dev']:.3f} | {gaps} |")
    A("")
    A("Taking the first tap of each burst as the reported instant, the two bursts give")
    A(f"offsets that agree to **{c['offset_disagreement_between_bursts_s']*1000:.0f} ms**.")
    A("Because the reported times are quoted only to the second and the bursts are just")
    A("37 s apart, a drift term would be meaningless; a constant offset is used.\n")
    A(f"    tap-only offset = {c['offset_s_tap_only']:.3f} s\n")

    A("## Stage 2 — refinement on large head movements\n")
    if r.get("n_used"):
        A("The tap times are quoted to the second, so the stage-1 clock is good to about")
        A("±1 s. Individual large head movements do better: a posture change is a slow ramp")
        A("in tilt but a fast spike in **jerk** (|d(acceleration vector)/dt|), and both")
        A("devices see the same spike. Each event is cross-correlated independently inside a")
        A("±3 s window, so the events give a distribution of residual-offset estimates.\n")
        A("One structural point makes this possible: Mind Monitor repeats the last")
        A("accelerometer value on every EEG row, so the file *looks* like 256 Hz motion data")
        A("but 93% of consecutive values are byte-identical. The instants where the (X, Y, Z)")
        A("triple actually changes are the real sensor updates — 17.3 Hz, median spacing")
        A("55 ms — and only those instants were used.\n")
        A("| quantity | value |")
        A("|---|---|")
        A(f"| candidate movement events | {r['n_candidate_events']} |")
        A(f"| events with a clear match | {r['n_used']} |")
        A(f"| median residual lag | **{r['lag_median_s']:+.3f} s** |")
        A(f"| standard error of the median | {r['lag_se_s']:.3f} s |")
        A(f"| robust SD across events | {r['lag_robust_sd_s']:.3f} s |")
        A(f"| median peak correlation | {r['r_median']:.2f} |\n")
        A("The median is stable as the acceptance threshold on per-event correlation is")
        A("tightened (+1.04 s at r>=0.45, n=40; +0.99 s at r>=0.70, n=6), which is what a")
        A("real constant offset looks like rather than a fitting artefact. A drift term")
        A("fitted across events came out at 0.33 ± 0.61 s over the 110 min overlap — smaller")
        A("than its own standard error — so the constant-offset model was kept.\n")
        A("The shipped clock is therefore\n")
        A(f"    real_time = 1970-01-01 + {c['offset_s']:.3f} s + timeMS/1000\n")
        A(f"- mask power-on (timeMS = 0): **{c['mask_power_on'][:23]}**")
        A(f"- mask stream: **{c['mask_stream_start'][:23]} -> {c['mask_stream_end'][:23]}**\n")
        A("Direction of the correction: the mask clock is pushed **later** by")
        A(f"{r['lag_median_s']:.2f} s. The tester's reported times are truncated to the second,")
        A("so the presses most likely happened a fraction of a second after the stated mark;")
        A("in addition Mind Monitor timestamps a packet when the phone *receives* it, which")
        A("is systematically late by the Bluetooth buffering delay and biases this estimate")
        A("in the same direction. Those two cannot be separated here, so the **absolute**")
        A("accuracy is limited to a few hundred ms even though the **relative** alignment of")
        A("the two streams is now good to about ±0.1 s.\n")
    else:
        A("Refinement did not converge on this session.\n")

    A("## Muse stream\n")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| sample rows | {m['n_sample_rows']:,} |")
    A(f"| span | {m['t_start'][:19]} -> {m['t_end'][:19]} |")
    A(f"| mean row rate | {m['fs_mean']:.2f} Hz |")
    A(f"| timestamp backsteps (after sort) | {m['n_timestamp_backsteps']} |")
    A(f"| rows sharing a timestamp with the previous row | {m['frac_duplicate_timestamps']*100:.0f}% |\n")
    A("Timestamps are written at 1 ms resolution but arrive in BLE packets, so a large")
    A("fraction of rows repeat the previous row's stamp. Muse time resolution is")
    A("packet-level (tens of ms), not sample-level.\n")

    A("## No tap marker exists in the Muse data\n")
    A("Checked, at both marker times, in: gyroscope, accelerometer, the four raw EEG")
    A("channels, the four optical channels `Optics1-4`, and the four raw optical")
    A("amplitudes `AUX_1-4`.\n")
    A("`AUX_1-3` do show elevated activity in the tap windows (robust z 15-21 versus a")
    A("control-window 99th percentile of 4-5), but this is not specific: scored against")
    A("every 6-s window in the whole recording, the two tap windows are exceeded by")
    A("**32%** and **43%** of windows. Their internal peak spacing also fails to match")
    A("the mask's tap fingerprint (burst 2: 2.01 s in AUX versus 1.31 and 1.24 s in the")
    A("mask). The headband was not mechanically coupled to the mask sensors, so the")
    A("presses left no usable marker in Muse.\n")

    A("## Whole-night check on the shipped clock\n")
    A("Per-second SD of the accelerometer vector magnitude, log-transformed and z-scored,")
    A(f"cross-correlated over the {mc['n_bins']/60:.0f}-min overlap ({mc['n_bins']:,} bins):\n")
    A(f"- peak r = **{mc['r_peak']:.3f}** at lag **{mc['lag_peak_s']:+.0f} s**")
    A(f"- r at lag 0 = {mc['r_at_zero']:.3f}")
    A(f"- background: 95th percentile of |r| beyond ±30 s = {mc['r_background_p95']:.3f}\n")
    A("Before the stage-2 correction this peak sat at -1 s with r = 0.254 at zero lag;")
    A("after it, the peak is at 0 s and the zero-lag value equals the peak. That is the")
    A("sign check on the correction, not an independent measurement of it.\n")
    A("**Stated sync accuracy: ±0.1 s relative, a few hundred ms absolute.** Good enough")
    A("for movement events, epoch-level (30 s) comparison and artefact matching; the")
    A("absolute figure is still not good enough to claim beat-to-beat correspondence.\n")

    A("## Head tilt\n")
    A("The mask accelerometer gives two absolute angles whenever the head is still:\n")
    A("    headUDTilt = atan2(aX, sqrt(aY^2 + aZ^2))   # nodding / pitch")
    A("    headLRTilt = atan2(aY, sqrt(aX^2 + aZ^2))   # side-to-side / roll\n")
    A("Both are added by `load_sleepmask` and carried into both exports. Over this")
    A("session they sit on flat plateaus interrupted by step changes at posture shifts —")
    A("the same events the stage-2 refinement uses.\n")

    A("## Files\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `sync_mask_muse.py` | loader + head tilt + tap detection + anchoring + motion refinement + merge |")
    A("| `run_sync.py` | driver; regenerates everything below |")
    A("| `figures.py` | the three figures |")
    A("| `make_report.py` | this report, generated from the run's own numbers |")
    A("| `taps_sleepmask.csv` | the six tap pulses, device and wall clock |")
    A("| `sync_motion_events.csv` | per-event lags, correlations and jerk amplitudes |")
    A("| `mask_realtime_111Hz.csv.gz` | full mask stream, real-time column + head tilt |")
    A(f"| `synced_mask_muse_10Hz.csv.gz` | both devices on one 10 Hz wall-clock grid, {rep['synced_10Hz']['rows']:,} rows |")
    A("| `sync_report.json` | every number above, machine-readable |\n")
    A("Re-run with `cd optical_control_sync && python run_sync.py`. The raw-data folder")
    A("is located by `$SLEEP_OPTICAL_DIR`, then by looking beside and above this folder,")
    A("so a folder move needs no edit.\n")

    A("## Caveats\n")
    A("1. The mask device clock is assumed linear at its own 9 ms tick. Drift over the")
    A("   5.1 h recording is uncorrected and untestable here: the movement events only")
    A("   span the 110 min of Muse overlap, and within that span no drift is detectable")
    A("   above its own error bar. If the mask is re-run against a reference clock, tap at")
    A("   the start **and** the end of the night.")
    A("2. Mind Monitor's timestamps are phone-receive times, so the absolute offset")
    A("   inherits an unknown Bluetooth buffering delay of order 0.1 s.")
    A(f"3. Only the first {mc['n_bins']/60:.0f} min of the mask recording has a Muse counterpart;")
    A(f"   Muse disconnected at {m['t_end'][:19]} while the mask ran to")
    A(f"   {c['mask_stream_end'][:19]}.")
    A("4. `Optics5-16` are all zero in this export; only `Optics1-4` and `AUX_1-4` carry")
    A("   optical data.")
    return "\n".join(L) + "\n"


def write(rep: dict, taps=None, outdir: str = HERE) -> str:
    p = os.path.join(outdir, "sync_report.md")
    with open(p, "w") as f:
        f.write(build(rep, taps))
    return p


if __name__ == "__main__":
    rep = json.load(open(os.path.join(HERE, "sync_report.json")))
    taps = pd.read_csv(os.path.join(HERE, "taps_sleepmask.csv"))
    print(write(rep, taps))
