# Sleep mask <-> Muse S synchronisation — optical_sensor_control_test

Session of 2026-09-11. Mask file `SM2_4_20260910_Giang.txt`,
Muse file `mindMonitor_2026-09-11--02-07-30.csv`.

## The clock problem

The mask export has no wall clock, only `timeMS` — milliseconds since the device
powered on. The Muse / Mind Monitor export carries phone wall-clock timestamps.
The tester therefore pressed the mask sensors three times at **02:10:03** and
three more times at **02:10:40**; those presses are the only bridge between the
two clocks. A second stage then refines the offset on shared head movement.

## Mask stream integrity

| quantity | value |
|---|---|
| samples | 2,042,881 |
| tick spacing | 9 ms (min 9, max 9) |
| effective rate | 111.11 Hz |
| gaps > 50 ms | 0 |
| span | 5.11 h |

The tick spacing is exactly 9 ms on every one of the 2,042,881
samples, with no dropouts, so the device counter can be treated as a linear clock
and only a constant offset is needed.

## Stage 1 — tap markers

Six presses were detected, in two bursts of three, as saturating pulses on the
capacitive channels (CH drops to about -10, CRE rises to 4-7; robust z of order
10^3 against the night-long baseline).

| burst | reported | first tap (device s) | inter-tap gaps (s) |
|---|---|---|---|
| 1 | 2026-09-11 02:10:03 | 163.967 | 1.674, 0.801 |
| 2 | 2026-09-11 02:10:40 | 200.669 | 1.305, 1.242 |

Taking the first tap of each burst as the reported instant, the two bursts give
offsets that agree to **298 ms**.
Because the reported times are quoted only to the second and the bursts are just
37 s apart, a drift term would be meaningless; a constant offset is used.

    tap-only offset = 1789092439.182 s

## Stage 2 — refinement on large head movements

The tap times are quoted to the second, so the stage-1 clock is good to about
±1 s. Individual large head movements do better: a posture change is a slow ramp
in tilt but a fast spike in **jerk** (|d(acceleration vector)/dt|), and both
devices see the same spike. Each event is cross-correlated independently inside a
±3 s window, so the events give a distribution of residual-offset estimates.

One structural point makes this possible: Mind Monitor repeats the last
accelerometer value on every EEG row, so the file *looks* like 256 Hz motion data
but 93% of consecutive values are byte-identical. The instants where the (X, Y, Z)
triple actually changes are the real sensor updates — 17.3 Hz, median spacing
55 ms — and only those instants were used.

| quantity | value |
|---|---|
| candidate movement events | 133 |
| events with a clear match | 40 |
| median residual lag | **+1.040 s** |
| standard error of the median | 0.104 s |
| robust SD across events | 0.659 s |
| median peak correlation | 0.57 |

The median is stable as the acceptance threshold on per-event correlation is
tightened (+1.04 s at r>=0.45, n=40; +0.99 s at r>=0.70, n=6), which is what a
real constant offset looks like rather than a fitting artefact. A drift term
fitted across events came out at 0.33 ± 0.61 s over the 110 min overlap — smaller
than its own standard error — so the constant-offset model was kept.

The shipped clock is therefore

    real_time = 1970-01-01 + 1789092440.222 s + timeMS/1000

- mask power-on (timeMS = 0): **2026-09-11 02:07:20.222**
- mask stream: **2026-09-11 02:09:25.129 -> 2026-09-11 07:15:51.049**

Direction of the correction: the mask clock is pushed **later** by
1.04 s. The tester's reported times are truncated to the second,
so the presses most likely happened a fraction of a second after the stated mark;
in addition Mind Monitor timestamps a packet when the phone *receives* it, which
is systematically late by the Bluetooth buffering delay and biases this estimate
in the same direction. Those two cannot be separated here, so the **absolute**
accuracy is limited to a few hundred ms even though the **relative** alignment of
the two streams is now good to about ±0.1 s.

## Muse stream

| quantity | value |
|---|---|
| sample rows | 1,727,954 |
| span | 2026-09-11 02:07:28 -> 2026-09-11 04:00:04 |
| mean row rate | 255.76 Hz |
| timestamp backsteps (after sort) | 0 |
| rows sharing a timestamp with the previous row | 58% |

Timestamps are written at 1 ms resolution but arrive in BLE packets, so a large
fraction of rows repeat the previous row's stamp. Muse time resolution is
packet-level (tens of ms), not sample-level.

## No tap marker exists in the Muse data

Checked, at both marker times, in: gyroscope, accelerometer, the four raw EEG
channels, the four optical channels `Optics1-4`, and the four raw optical
amplitudes `AUX_1-4`.

`AUX_1-3` do show elevated activity in the tap windows (robust z 15-21 versus a
control-window 99th percentile of 4-5), but this is not specific: scored against
every 6-s window in the whole recording, the two tap windows are exceeded by
**32%** and **43%** of windows. Their internal peak spacing also fails to match
the mask's tap fingerprint (burst 2: 2.01 s in AUX versus 1.31 and 1.24 s in the
mask). The headband was not mechanically coupled to the mask sensors, so the
presses left no usable marker in Muse.

## Whole-night check on the shipped clock

Per-second SD of the accelerometer vector magnitude, log-transformed and z-scored,
cross-correlated over the 111-min overlap (6,640 bins):

- peak r = **0.336** at lag **+0 s**
- r at lag 0 = 0.336
- background: 95th percentile of |r| beyond ±30 s = 0.061

Before the stage-2 correction this peak sat at -1 s with r = 0.254 at zero lag;
after it, the peak is at 0 s and the zero-lag value equals the peak. That is the
sign check on the correction, not an independent measurement of it.

**Stated sync accuracy: ±0.1 s relative, a few hundred ms absolute.** Good enough
for movement events, epoch-level (30 s) comparison and artefact matching; the
absolute figure is still not good enough to claim beat-to-beat correspondence.

## Head tilt

The mask accelerometer gives two absolute angles whenever the head is still:

    headUDTilt = atan2(aX, sqrt(aY^2 + aZ^2))   # nodding / pitch
    headLRTilt = atan2(aY, sqrt(aX^2 + aZ^2))   # side-to-side / roll

Both are added by `load_sleepmask` and carried into both exports. Over this
session they sit on flat plateaus interrupted by step changes at posture shifts —
the same events the stage-2 refinement uses.

## Files

| file | contents |
|---|---|
| `sync_mask_muse.py` | loader + head tilt + tap detection + anchoring + motion refinement + merge |
| `run_sync.py` | driver; regenerates everything below |
| `figures.py` | the three figures |
| `make_report.py` | this report, generated from the run's own numbers |
| `taps_sleepmask.csv` | the six tap pulses, device and wall clock |
| `sync_motion_events.csv` | per-event lags, correlations and jerk amplitudes |
| `mask_realtime_111Hz.csv.gz` | full mask stream, real-time column + head tilt |
| `synced_mask_muse_10Hz.csv.gz` | both devices on one 10 Hz wall-clock grid, 66,393 rows |
| `sync_report.json` | every number above, machine-readable |

Re-run with `cd optical_control_sync && python run_sync.py`. The raw-data folder
is located by `$SLEEP_OPTICAL_DIR`, then by looking beside and above this folder,
so a folder move needs no edit.

## Caveats

1. The mask device clock is assumed linear at its own 9 ms tick. Drift over the
   5.1 h recording is uncorrected and untestable here: the movement events only
   span the 110 min of Muse overlap, and within that span no drift is detectable
   above its own error bar. If the mask is re-run against a reference clock, tap at
   the start **and** the end of the night.
2. Mind Monitor's timestamps are phone-receive times, so the absolute offset
   inherits an unknown Bluetooth buffering delay of order 0.1 s.
3. Only the first 111 min of the mask recording has a Muse counterpart;
   Muse disconnected at 2026-09-11 04:00:04 while the mask ran to
   2026-09-11 07:15:51.
4. `Optics5-16` are all zero in this export; only `Optics1-4` and `AUX_1-4` carry
   optical data.
