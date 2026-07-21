# Results §4.1 (draft) — Absolute mean value (DC baseline) vs sleep stage

_Purpose: this is the Results opener the professor asked for ("Start with mean value (raw signal
changes) in comparison to sleep stages"), written as a real result in **native a.u.** so it complements —
rather than repeats — the z-scored slow-mean analyses used elsewhere. It replaces the leftover outline note
flagged as HL#6._

_Analysis code: `analysis/mean_value/abs_mean_vs_stage.py`.
Figures: `notebooks/plots/mean_value/abs_{scale_comparison, stage_boxplot_au, baseline_by_session, trace_*}.png`.
Numbers: `reports/mean_value/abs_mean_{scale, stage_au, subject_direction}.csv`._

---

## Draft text

**4.1 The absolute capacitive baseline is dominated by sensor coupling, not sleep stage**

We first asked whether the raw mean value — the absolute DC level of each capacitive channel, in native
sensor units (a.u.) — carries sleep-stage information directly, before any per-session normalization. It
does not, in any transferable way, and understanding why motivates the normalization used throughout the
remainder of the Results.

The absolute baseline is set primarily by the sensor–skin coupling of that night's mask fit. Across the
twelve recordings, the per-session median DC level varied over a wide range that dwarfs any within-night
variation: the between-session standard deviation of the baseline was 30 a.u. for CLE, 175 a.u. for CRE,
170 a.u. for the CLE−CRE differential, and 93 a.u. for the CLE+CRE common mode (Figure Ax, per-session
baselines). Superimposed on this offset, the baseline drifts slowly across the night — a monotonic
time-of-night change of 4–25 a.u. (median absolute last-30-min minus first-30-min) that is unrelated to
sleep stage and is itself comparable in size to any stage effect. By comparison, the total within-session
excursion of the mean value, after removing the between-session offset, had a median interquartile range of
only 11–25 a.u. In short, the variance of the absolute mean value lives overwhelmingly in the coupling
offset (≫), then the overnight drift (≳), and only last in sleep stage (Figure Bx, scale comparison). On the
raw a.u. axis the channel baselines are near-flat lines across the entire night, with the stage-linked
excursion amounting to less than ~0.3% of the ~2000 a.u. baseline (Figure Cx, representative session).

When the between-session offset is removed by centering each channel on its own within-session Wake median
(a subtraction that preserves the native a.u. scale, unlike z-scoring, which additionally divides out the
amplitude), a small stage-graded excursion does emerge on the temple-referenced channels: relative to Wake,
the median mean value in N3 was +7.0 a.u. for CLE and +4.6 a.u. for CLE−CRE, with intermediate stages
falling in between and REM near or below Wake (Figure Dx, per-stage boxplots in a.u.). This N3-elevated
direction was present in 5 of 6 subjects for both CLE and CLE−CRE. It was absent on the right channel (CRE,
+0.7 a.u., 2 of 6 subjects) and on the common mode (CLE+CRE, direction split 3/3), so it does not generalize
to a channel-independent baseline signature. Pooled across all epochs the stage difference is formally
significant (Kruskal–Wallis p < 10⁻⁷ for every channel), but with six subjects this pooled test reflects
epoch-level replication rather than independent evidence; we therefore report the per-subject effect sizes
and directions above as the honest summary. [cross-ref: same subject-as-unit convention applied paper-wide,
per Methods.]

The direction of the effect — a few a.u. of increased capacitive baseline in deep sleep on the left/temple
channels — is qualitatively consistent with the intracranial/hemodynamic reading of the SEC signal
(sleep-stage-dependent changes in cerebral blood volume, CSF distribution, and intracranial compliance alter
the mean coupling). But its magnitude sits at the level of the overnight coupling drift and does not hold
across channels or all subjects. We therefore characterize the absolute mean value as a coupling-dominated
baseline that is not, on its own, a usable sleep-stage marker, and we normalize each channel per session
(removing offset and drift) for all subsequent analyses. This mirrors the subject-dependent directionality
seen later in the harmonic-ridge features (HER), and is the first of several places where the SEC signal
reflects sleep-state changes only after per-subject referencing.

---

## Suggested Table (native a.u.) — optional, or fold into the figure captions

| Channel | Between-session offset SD (a.u.) | Within-night drift \|median\| (a.u.) | Within-session stage IQR (a.u.) | N3 − Wake median (a.u.) | N3↑ subjects |
|---------|---------------------------------:|-------------------------------------:|--------------------------------:|------------------------:|:------------:|
| CLE     | 30  | 25 | 16 | **+7.0** | 5/6 |
| CRE     | 175 | 4  | 11 | +0.7 | 2/6 |
| CLE−CRE | 170 | 22 | 25 | **+4.6** | 5/6 |
| CLE+CRE | 93  | 17 | 13 | +4.4 | 3/6 |

---

## Notes for the author (not for the manuscript)

- **Message:** this is an honest-characterization / mildly-negative result — the absolute mean value is
  coupling-offset + drift dominated; a real but small (~a few a.u.) N3-elevated excursion exists on CLE and
  CLE−CRE in 5/6 subjects, not universal. It *earns its place* by (a) answering the professor's directive,
  (b) justifying the per-session normalization the rest of the paper relies on, and (c) being another honest
  boundary. Confirm you want it framed this way vs. leading harder on the 5/6 N3 trend.
- **Pseudoreplication:** I deliberately demoted the KW p < 10⁻⁷ to a parenthetical and led with per-subject
  5/6. This is the same fix as HL#51 / SUBMISSION_REVIEW P1#7 — keep it consistent.
- **Units:** kept everything in native a.u. No a.u.→fF/ICP conversion, since we have no per-recording
  a.u.→fF calibration for the human mask (the 4.8 fF/mmHg pig figure is a different setup). If you have that
  conversion, I can restate the a.u. effect sizes in fF / predicted-mmHg.
- **Figure letters (Ax–Dx)** are placeholders — assign real numbers once section placement is fixed.
- **Placement:** goes first in Results (before "Signal validation: SEC carries respiratory and cardiac band
  energy"), and the HL#6 stray note gets deleted.
