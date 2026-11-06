# Signal processing - control-session time, spectrogram and power-spectrum figures

Session 2026-09-11 02:09:25 to 2026-09-11 04:00:04 local (110.7 min), the window where the sleep mask and the Muse S overlap after the tap-anchored clock correction. Every panel in `fig8_time_domain.png`, `fig9_spectrograms.png` and `fig10_power_spectra.png` is on that one wall clock.

This chain follows the session viewer (`smartadi/sleep_monitor`, `filters.py` / `preprocessing.py` / `motion.py`) so the numbers here are comparable with the viewer's own output. TP9, TP10 and AUX_1-4 are excluded: TP9/TP10 carry more broadband noise than the frontal pair in this rig, and the AUX inputs carry no physiology (flat DC plus a comb of fixed tones, i.e. floating analog inputs).

## Chain, in order

**1. Outlier clip.** Capacitive channels clipped at median +/- 4 sigma (viewer `outlier_clip`); EEG at +/- 8 sigma, looser because real EEG arousals are large and clipping them would flatten the feature of interest. The three finger taps at the start of the record exceed any of these limits and survive as clipped transients, which is why they are handled again by the motion gate below.

**2. Bandpass, order-3 Butterworth, zero-phase.** `sosfiltfilt`, so no phase distortion and no group delay - important because these figures are read for cross-device timing. Capacitive and optical channels 0.03-5.0 Hz; EEG 0.5-30.0 Hz. The 0.03 Hz high-pass removes the overnight capacitive drift (tens of minutes) that would otherwise dominate every panel; the 5.0 Hz low-pass is the band the mask and optics were asked to be shown in and sits above the second cardiac harmonic.

**3. Accelerometer artifact removal, capacitive channels only.** Viewer OLS variant: accelerometer magnitude is bandpassed to the *same* band as the capacitive channel, then one coefficient beta = <acc,cap>/<acc,acc> is fitted over the whole window and beta*acc subtracted. Fitted values this session: `CLE` +0.00234, `CRE` +0.00322, `CH` -0.00667. These are small, so in-band motion energy is a minor contributor outside the gated epochs - the gate, not the regression, is doing the work here. A single global beta cannot track a coupling that changes with head position; the viewer's 16-tap NLMS variant is the option if that becomes a limit.

**4. Head tilt.** Gravity is the 0.5 Hz low-pass of each accelerometer axis; tilt then uses the two formulas as given:

```
headUDTilt = degrees(arctan2(aX, sqrt(aY^2 + aZ^2)))
headLRTilt = degrees(arctan2(aY, sqrt(aX^2 + aZ^2)))
```

Applying them to the gravity vector rather than to raw acceleration is the only deviation from the formulas as written, and it is the viewer's `motion.py` convention: without it every head movement injects a transient of tens of degrees into an angle that is meant to describe posture.

**5. Motion gate.** Dynamic acceleration = 0.5 Hz high-pass per axis, magnitude; RMS over non-overlapping 30 s epochs; an epoch is flagged when RMS > median + 3*MAD (viewer `motion_thresh_mad=3.0`). This session: 221 epochs, threshold 0.00942, 18.1% flagged in 14 spans. Flagged spans are shaded grey in figure 8, left in place in figure 9, and excluded from figure 10.

## Figure-specific parameters

| | fig 8 time domain | fig 9 spectrogram | fig 10 power spectrum |
|---|---|---|---|
| capacitive | per-pixel min/max envelope, 1600 px | 60 s Hann, 50% overlap, 2-window average -> 60 s/column | Welch 120 s Hann, 50% overlap |
| EEG | same envelope | 30 s Hann, 50% overlap, 4-window average -> 60 s/column | Welch 8 s Hann, 50% overlap |
| optics | same envelope | as capacitive | as capacitive |
| y range | median +/- 10 robust sigma (1.4826*MAD) | 5 Hz span; EEG 0.5-30 Hz | same |
| colour range | - | 10-97 pct (mask, optics), 5-99.5 pct (EEG) | - |

**Why these windows.** 60 s gives 0.017 Hz resolution, enough to separate a 1.08 Hz cardiac line from its 0.05 Hz respiratory side-structure, and two-window averaging halves the chi-square(2) speckle that otherwise buries a narrow-dynamic-range channel. EEG uses 30 s windows because that is the scoring epoch and 0.033 Hz resolution is already far finer than any EEG band edge. The 120 s Welch segment in figure 10 is the longest that still yields several independent segments inside the shortest motion-free run.

**Normalisation.** All spectra are power spectral density, `scaling='density'`, 10*log10, referenced to nothing external - dB here is relative within a channel, and the absolute level of a capacitive or optical channel is in arbitrary sensor units. Compare shapes and time-course, not absolute levels, between channels of different type. Each spectrogram column is linearly detrended before the transform.

**Envelope, not decimation.** Figure 8 draws the per-pixel min and max of every sample in that pixel, so a single-sample transient cannot be lost to downsampling; the thin line inside the band is the per-pixel median.

**Motion gating in figure 10.** The Welch average runs only over contiguous motion-free runs of at least 60 s, after discarding the first 120 s (the tap burst and filter settling): 12 runs covering 82% of the window. Without this the tap transient alone raised apparent capacitive band power by roughly 20 dB and put a leakage ripple across the whole 0-5 Hz span, which is what the first render of this figure showed.

## What this session's spectra say

| channel | respiratory-band peak (Hz) | cardiac-band peak (Hz) |
|---|---|---|
| CLE | 0.100 | 1.050 |
| CRE | 0.100 | 0.608 |
| CH | 0.117 | 0.608 |
| Optics 1 | 0.100 | 1.083 |
| Optics 2 | 0.100 | 1.083 |
| Optics 3 | 0.100 | 1.083 |
| Optics 4 | 0.100 | 1.083 |

The optical channels carry a sharp cardiac fundamental near 1.08 Hz with visible first and second harmonics, and all four agree to the frequency bin - they are seeing one pulsatile source. The capacitive channels show no resolvable cardiac line over this window: their band 'peaks' are the maximum of a smooth falling spectrum, not a feature, and should not be read as a rate estimate. Respiratory-band entries sit at or near the lower band edge for the same reason, i.e. the 1/f rise, except for the optical peak visible near 0.27 Hz.

## Interpretive limits

* A whole-window Welch average over 110 min mixes sleep stages and postures; a line that is present only part of the night is attenuated here but visible in figure 9.
* The gate is derived from the mask accelerometer only, and is applied to the Muse streams as well. It therefore removes head movement, not Muse-specific electrode artifacts.
* The optical channels are reconstructed from a forward-filled export: true update rate 21.44 Hz, linearly interpolated onto a uniform 20 Hz grid before filtering. Content above about 11 Hz is not trustworthy.
* Figure 8 y limits are robust (median +/- 10 sigma from the MAD), so the tap transients and the largest movement artifacts run off-panel by design. Their amplitude is not readable from this figure.
* Absolute clock accuracy is that of the tap anchor, about 1 s; relative alignment between the two devices is better than that. Nothing at these window lengths is sensitive to it.

## Tunable constants

| constant | value | cost of changing it |
|---|---|---|
| capacitive/optical band | 0.03-5.0 Hz | a higher high-pass removes real vasomotion; a lower one lets overnight drift back in |
| EEG band | 0.5-30.0 Hz | raising the top edge admits EMG; lowering it hides spindle/beta activity |
| gravity split | 0.5 Hz | higher puts slow head motion into the tilt trace; lower smears posture steps |
| clip | 4 sigma (cap), 8 sigma (EEG) | tighter flattens real arousals |
| motion MAD factor | 3.0 | lower gates more of the night out of figure 10 |
| gate epoch | 30 s | shorter tracks brief movements, at the cost of fragmenting the motion-free runs figure 10 needs |
| Welch segment | 120 s (slow), 8 s (EEG) | longer sharpens lines and reduces the number of independent segments |

Not done, deliberately: no notch filter (no mains line is visible in these channels), no interpolation across the gated epochs, no CLE-CRE differential channel (the viewer's primary channel, but the request was for CLE and CRE separately), no per-epoch normalisation, no artifact rejection beyond the clip and the motion gate.
