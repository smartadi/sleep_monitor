# Spindle section — corrected manuscript text (2026-08-06)

Supersedes the earlier draft in this file, which was written for the pure-negative
framing ("the mask does not detect spindles") and quoted pre-fix numbers.

Every number below comes from the re-run after the 2026-08-06 code fixes (0.32 s
event-alignment bug, 0.1–3 Hz low band, robust statistics — see CHANGELOG). Do not
mix these with the numbers currently in the docx; they are not the same analysis.

Sources: `analysis/spindles/outputs/spindle_lowband_detection.csv` (POOLED row),
`spindle_ersp_control.csv`, `spindle_lowband_subbands.npz`.
Figure: `writeup/figures/spindles/fig_spindle_lowband_detection.png`.

**Two things changed in substance, not just in wording:**

1. The sigma band (11–16 Hz) lies inside the range the paper itself defines as the
   electronic noise floor ("10 Hz to the Nyquist frequency, 50 Hz", §Methods and the
   Figure 2 caption). A null measured there cannot support a claim about what the
   sensor transduces. It is reframed as a bound.
2. The low-band effect is carried by a minority of spindles (median +0.13 dB against
   a mean of +0.59 dB on CH; 51.8 % of spindles exceed their own baseline against a
   50 % chance rate; 5 of 12 nights below chance). It is a group-level effect, not a
   per-spindle response, and is now described as such.

---

## Replace para 111 (section heading)

> **4.4 Sleep spindles: a low-frequency mechanical correlate, and the bandwidth limit
> on testing for an electrical one**

Rationale: the old heading ("a mechanical, not electrical, reflection") asserts the
conclusion that §4.4 can no longer carry on its own. The delta-onset result (§4.5),
measured inside the sensor's passband, is what supports the mechanism claim.

## Replace para 112 (setup and methods)

> Sleep spindles—11–16 Hz (sigma) thalamocortical bursts that are the electrographic
> hallmark of N2 sleep—offer a discrete, precisely timed cortical event at which to ask
> what the temple sensor reflects. Spindle annotations were taken from the automatic
> spindle detection exported by the PSG analysis software (per-event start, end and
> duration; intra-spindle frequency for a subset), and aligned to the capacitive
> recording using the same wall-clock offset applied to sleep staging. Event boundaries
> are reported on a 250 ms grid, so event centres carry ±125 ms of timing quantisation.
> Alignment was confirmed by the stage distribution of the annotated events: no spindle
> fell in REM and 0.0–1.6 % fell in Wake, against Wake base rates of up to 10.9 % per
> night. Across the twelve recordings, 351–2,134 N2 spindles per night were analysed
> (14,305 in total). For each spindle we compared capacitive power in a ±1 s window at
> the spindle centre against that same spindle's own surrounding baseline (|t| > 5 s
> within a ±8 s window), and separately against matched spindle-free N2 control windows.

Also note in Methods or the supplement: one night (S4N2) lacks the primary duration
annotation file and falls back to the frequency file, contributing 554 events where
comparable nights contribute 1,800–2,000; and intra-spindle frequency is available for
27–96 % of events depending on night.

## Replace para 113 (low-band result)

> Centre-triggered averaging shows that low-band (0.1–3 Hz) capacitive power increases
> at the spindle centre on every channel. The trial-averaged increase is +0.48 to
> +0.51 dB on the temple channels and +0.59 dB on the forehead channel (CH), positive
> on every channel in 11 or 12 of the twelve nights (Figure 7A, 7C). The effect is
> spindle-specific: the identical measurement at randomly chosen N2 timepoints is flat
> (+0.05 dB), scored arousals alone produce no positive deflection (−0.33 dB), and
> spindles with no arousal within ±5 s still show +0.51 dB, so arousal co-occurrence
> amplifies the response (+1.16 dB) without causing it. Within the low band the
> increase is broadly distributed rather than concentrated at one rhythm: +0.73 dB at
> 0.1–0.5 Hz, +0.68 dB at 0.5–1.5 Hz and +0.47 dB at 1.5–3 Hz, positive in 12 of 12
> nights in all three sub-bands.
>
> This is a group-level effect carried by a minority of events, and we report it as
> such. The per-spindle distribution is strongly right-skewed: on CH the mean of
> +0.59 dB sits against a median of +0.13 dB and a 10 %-trimmed mean of +0.18 dB, the
> strongest 5 % of spindles account for more than the whole of the mean (the remaining
> 95 % sum slightly negative), and 51.8 % of individual spindles exceed their own
> baseline against a 50 % chance rate and a 48.2 % matched-control rate (Figure 7B).
> Five of the twelve nights fall below chance on this per-spindle measure. The contrast
> with the contact EEG is instructive: the EEG low-band response to the same events has
> a median (+1.25 dB) close to its mean (+1.83 dB) and is detectable on 60.5 % of
> individual spindles, i.e. it is distributed across events in a way the capacitive
> response is not. The mask therefore registers a reliable average low-frequency
> disturbance around spindles, but not one that can be resolved spindle by spindle.

## Replace para 114 (sigma result and mechanism)

> The sigma band that carries the spindle oscillation itself shows no detectable change
> on any capacitive channel. Pooled across 14,305 N2 spindles the trial-averaged shift
> is bounded at |Δ| < 0.05–0.06 dB (< 1.4 % power, 95 % confidence) on all four
> channels, against +3.56 dB (+127 % power) for the identical measurement on the
> contact EEG.
>
> This bound must be read against the sensor's bandwidth, and it is a bound rather than
> a mechanistic result. The capacitive spectral density at 11–16 Hz is only 1.8–2.0
> times the 40–49 Hz electronic noise floor in every session and on every channel, and
> no capacitive channel exceeds three times its own floor above 6.9 Hz (median 1.5 Hz
> across sessions), whereas the contact EEG retains structure to 35 Hz and sits
> 460–1,200 times above its floor in the sigma band. The spindle band therefore falls
> inside the range we characterise elsewhere as the sensor's white noise floor
> (§Methods, Figure 2). The correct reading is that no spindle-locked sigma signature is
> recoverable at this sensor's noise floor — not that cortical electrical fields are
> absent from the transduction path, which this measurement has no power to decide.
> What the mask does register at a spindle lies entirely in the low, mechanical band,
> and the question of whether the capacitive response to cortical events is driven
> mechanically or electrically is settled instead by the delta-burst analysis of §4.5,
> which is measured at 0.5–4 Hz, inside the sensor's passband.

## Replace para 116 (Figure 7 caption)

> **Figure 7. The capacitive response at sleep spindles is low-frequency, small, and
> carried by a minority of events.** (A) Centre-triggered average of capacitive
> low-band (0.1–3 Hz) power across channels; power rises at the spindle centre,
> strongest on the forehead channel (CH, +0.59 dB). Shading is ±1 SEM across the twelve
> nights. (B) Per-spindle power-change distribution on CH for the low band versus the
> sigma band (11–16 Hz). Solid lines mark means, dotted lines medians; the gap between
> them is the point — the low-band mean of +0.59 dB sits against a median of +0.13 dB,
> and only 51.8 % of individual spindles exceed their own baseline (chance 50 %). The
> sigma band is bounded at |Δ| < 0.06 dB. (C) Per-session low-band power change for
> each capacitive channel (bars, means; ticks, medians), showing the average increase
> is present across the twelve nights while the median remains small throughout.

## Replace para 135 (§5 summary)

> The mask carries no recoverable cortical electrographic signature. Sleep spindles—the
> electrographic hallmark of N2—produce no detectable spindle-locked sigma response in
> any capacitive channel (bounded at |Δ| < 0.06 dB, < 1.4 % power), whereas the same
> measurement on the contact-EEG channel rises +3.56 dB; this bound is set by the
> sensor's noise floor, which the 11–16 Hz band falls inside. What the mask does
> register at a spindle is a low-frequency (0.1–3 Hz) increase of +0.48 to +0.59 dB,
> reliable in the session average but carried by a minority of events (median +0.13 dB
> on CH, 51.8 % of spindles above their own baseline). Together with the delta-burst
> onset analysis, which is measured inside the sensor's passband and shows the
> capacitive response following rather than preceding the cortical event, this
> establishes that the sensor transduces mechanical and hemodynamic pulsations.

## Replace para 143 (scope boundary)

> Within the sensor's passband, which ends between 2 and 7 Hz, the capacitive signal
> tracks mechanical and hemodynamic events rather than cortical electrical activity;
> above it the signal is at the electronic noise floor. Both facts point the same way
> for development: EEG-like interpretations of the capacitive signal are not supported
> at this placement, and the sigma and higher bands are not merely uninformative but
> unmeasured.

## Add to §6 Limitations (new item)

> The capacitive sensor's usable bandwidth ends between 2 and 7 Hz depending on channel
> and night, above which its spectral density is within a factor of two of the
> electronic noise floor. Claims about capacitive content at higher frequencies —
> including the sleep-spindle sigma band — are therefore bounds set by the instrument,
> not measurements of physiology, and a sensor with a lower noise floor would be
> required to test them.

## Amend para 153 (limitation 6, EEG montage)

Change "in which the spindle sigma rhythm was clearly present (+3.3 dB at spindle
centers)" to "(+3.6 dB at spindle centres)".

---

## Numbers table (for checking the docx after the edit)

| quantity | value |
|---|---|
| N2 spindles analysed | 14,305 (351–2,134/night) |
| low-band mean, CAP | CLE +0.51, CRE +0.48, CLE−CRE +0.49, CH +0.59 dB |
| low-band median, CAP | +0.13 to +0.20 dB |
| per-spindle detection rate | 0.518–0.531 (control null 0.481–0.490) |
| CH nights below chance | 5/12 |
| top 5 % share of the mean | 1.13–1.26 (EEG 0.42) |
| EEG low band | mean +1.83, median +1.25 dB, detection 0.605 |
| sigma, CAP | means +0.017 to +0.028 dB; bounds < 0.046–0.058 dB (< 1.1–1.3 % power) |
| sigma, EEG | +3.555 dB |
| CAP 11–16 Hz vs 40–49 Hz floor | 1.76–2.01× (all sessions, all channels) |
| CAP corner (3× floor) | 0.0–6.9 Hz, median 1.5 Hz (EEG 35.4 Hz) |
| controls | randN2 +0.05, arousal −0.33, spindle-no-arousal +0.51, spindle+arousal +1.16 dB |
| sub-bands | 0.1–0.5 Hz +0.73, 0.5–1.5 Hz +0.68, 1.5–3 Hz +0.47 (12/12 each) |

## Open item before the section is final

The peri-event baseline is not flat. The corrected grand-mean curve rises from about
−0.3 dB at −6 s to its peak at t = 0 and decays with a long positive tail past +6 s, so
the |t| > 5 s baseline sits inside the response rather than outside it. Spindles occur
in trains (≈1 per 7.5 s in N2), so each window's baseline contains other spindles. This
biases the measured bump downward and so does not threaten the positive finding, but a
±8 s window cannot separate a spindle-specific transient from a slower N2 process that
the spindles ride on. A wider-window re-run would settle it.
