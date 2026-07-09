# Spindle validation — ready-to-paste manuscript text

Placement: this pairs with the existing SWA subsection (§3.5 "SEC vs contact EEG
slow-wave sleep"). Recommend grouping both under one Results theme — e.g.
**"3.6 Boundaries of capacitive sensing: cortical EEG hallmarks (slow waves and
spindles) are absent"** — so the two negatives reinforce each other instead of
reading as two separate disappointments.

Figures produced by `analysis/spindles/plot_spindles.py`:
- `fig_spindle_triggered_sigma.png` — EEG (positive control) vs CAP sigma-envelope triggered averages
- `fig_spindle_auc.png` — per-session spindle-vs-control AUC by channel
- `fig_spindle_triggered_cardiac.png` — CAP cardiac-envelope autonomic probe
Data: `analysis/spindles/outputs/spindle_per_session.csv`, `spindle_summary.csv`

---

## Methods addition (insert after §3.8 "SEC vs. contact EEG slow-wave validation")

**3.9 Sleep-spindle validation.** To test whether the mask detects sleep spindles—the
11–16 Hz (sigma) thalamocortical bursts that are the electrographic hallmark of N2
sleep—we used the technologist-scored spindle annotations exported by the PSG software
(per-event start, end, duration, and intra-spindle frequency). Event times were aligned
to the capacitive recording by the same wall-clock offset used for sleep staging. As an
alignment check, we confirmed that 53–68% of scored spindle centers fell within PSG N2
epochs, with a median intra-spindle frequency of 12.8 Hz and median duration 0.50 s,
consistent with normal spindle morphology.

We applied one detector identically to the contact EEG channel (positive control) and to
each capacitive channel (CLE−CRE, CLE, CRE, CH). Each channel was band-pass filtered to
11–16 Hz (4th-order Butterworth, zero-phase) and its analytic (Hilbert) envelope was
z-scored across the night. Two quantities were computed per session, restricted to N2
spindles for a physiology-matched comparison: (i) a spindle-triggered average of the
sigma envelope in a ±4 s window; and (ii) a spindle-versus-control discriminability AUC,
where the mean sigma-envelope power in a ±0.25 s core window at each spindle was compared
against an equal number of control windows drawn from spindle-free N2 timepoints (≥3 s
from any spindle). To probe an indirect autonomic correlate, we also computed the
spindle-triggered average of the capacitive cardiac-band (0.5–3 Hz) envelope in a ±10 s
window.

## Results addition (insert after §3.5 SWA, as §3.6)

**3.6 Sleep spindles are not detectable in the capacitive signal.** Sleep spindles
provide a second, independent test of whether the temple sensor captures cortical
electrical activity, at a higher frequency (sigma, 11–16 Hz) than slow-wave activity.
Across the twelve recordings, 351–2134 N2 spindles per night were analyzed (≈14,300
spindles in total). The identical detector applied to the contact EEG channel recovered
spindles cleanly: the spindle-triggered EEG sigma envelope rose sharply at the spindle
center (peak ≈ 2.4 z above a flat baseline), and sigma power separated spindle from
control windows with a median AUC of 0.98 [range 0.96–0.99], corresponding to ≈3.4×
greater sigma power during spindles (median log₂ ratio +1.79). This confirms that the
detector and the spindle-to-capacitive time alignment are correct.

In contrast, no capacitive channel showed any spindle-locked sigma activity. The
spindle-triggered sigma envelope was flat (within ±0.05 z) for every channel, and
spindle-versus-control AUC was at chance for all channels (0.50; CLE−CRE 0.502, CLE
0.498, CRE 0.503, CH 0.504), with a median log₂ power ratio of ≈0.00 (Figure X). Because
the same analysis recovers spindles from contact EEG at AUC 0.98, this is a true absence
of a spindle signature in the capacitive signal, not a limitation of the method.

We also asked whether spindles leave an indirect autonomic footprint that the mask could
sense, since spindles are accompanied by transient cardiovascular fluctuations. The
spindle-triggered capacitive cardiac-band (0.5–3 Hz) envelope showed only a small,
non-robust deflection (+0.03 to +0.05 z; exceeding 0.05 z in just 3 of 12 sessions),
providing no reliable spindle-locked hemodynamic surrogate.

Together with the slow-wave result (§3.5), this establishes that the capacitive temple
sensor does not capture the cortical electrographic hallmarks of NREM sleep at either the
delta (slow-wave) or sigma (spindle) band—consistent with a sensor that transduces
mechanical and hemodynamic intracranial pulsations rather than neuronal electrical
activity.

## Suggested table (parallels Table 6 SWA)

**Table 7. Sleep-spindle validation per session.** Spindle-vs-control sigma-power AUC for
the contact EEG positive control and the canonical capacitive channel.

| Session | N2 spindles | EEG AUC | CLE−CRE AUC |
|---------|------------|---------|-------------|
| S1N1 | 1047 | 0.961 | 0.512 |
| S1N2 | 684  | 0.961 | 0.502 |
| S2N1 | 1674 | 0.992 | 0.488 |
| S2N2 | 1371 | 0.994 | 0.503 |
| S3N1 | 1785 | 0.973 | 0.515 |
| S3N2 | 2134 | 0.957 | 0.509 |
| S4N1 | 632  | 0.977 | 0.491 |
| S4N2 | 351  | 0.987 | 0.506 |
| S5N1 | 518  | 0.986 | 0.478 |
| S5N2 | 807  | 0.992 | 0.495 |
| S6N1 | 1447 | 0.979 | 0.502 |
| S6N2 | 1855 | 0.985 | 0.491 |
| **Median** | | **0.982** | **0.502** |

## Discussion addition (fold into §4.2 "Cortical EEG slow-wave activity")

Extend the existing paragraph: "...The same conclusion holds for sleep spindles: the
capacitive sigma-band envelope carries no spindle-locked activity (AUC 0.50) even though
the identical detector recovers spindles from contact EEG at AUC 0.98. The mask therefore
does not detect the two defining cortical rhythms of NREM sleep—slow waves and
spindles—reinforcing that temple-placement SEC senses intracranial mechanical/hemodynamic
pulsation, not cortical electrical fields. Future development should not pursue EEG-like
interpretations of the capacitive signal at this placement."
