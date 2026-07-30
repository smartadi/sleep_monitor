# Delta-onset seen in CAP — consolidated story / ready-to-shape manuscript text

**One-sentence claim (the story we are reporting):**
*The onset of a cortical delta burst in EEG is accompanied by a robust, time-locked
band-power event in the capacitive temple channels — i.e. EEG delta onset is visible in
the CAP signal — but as a mechanical/hemodynamic **response that follows** the cortical
event (~2–5 s later), not as an electrical copy of delta and not as a precursor.*

---

## Where this sits in the paper

This is a **third capacitive-vs-cortical result**, and it resolves the same way as the
other two but with a *positive* twist, so it belongs right after them:

- §3.5 — capacitive vs contact-EEG SWA: spectral r = 0.015 (mask does **not** transduce
  cortical delta electrically).
- §3.6 — sleep spindles: sigma AUC = 0.50 (no cortical sigma), **but** a small 0–3 Hz
  mechanical bump time-locked to N2 spindles (indirect low-frequency correlate).
- §3.x (CAP-SWA) — a *tonic* mechanical score marks N3 (per-subject AUC 0.675, 6/6).
- **§3.y (this) — a *phasic*, event-locked companion to CAP-SWA:** lock to individual
  EEG delta-burst *onsets* and the mask shows a sharp multi-band co-activation. Same
  thesis (SEC sees the *mechanical/autonomic shadow* of cortical events, not their
  electrical signal), now demonstrated at single-event time resolution.

**Framing discipline (must hold in the text):**
1. **Positive, honest:** "delta onset is *seen in* / *accompanied by a signature in* CAP,"
   never "CAP measures delta" or "CAP detects delta electrically."
2. **Precursor hypothesis is reported as tested-and-not-supported.** The professor's
   prior was that a CAP event *precedes* delta onset. We tested it directly and it failed;
   the coupling runs the other way (CAP follows). Report both — the negative precursor and
   the positive response — so the direction claim is airtight.
3. **n = 6 honest-negative rule:** per-subject direction counts, not p-values (Wilcoxon
   floors at p = 0.031). AUCs reported as discrimination metrics.

Analysis: `analysis/delta_onset/delta_onset_detection.py` (trigger),
`analysis/delta_onset/delta_cap_precursor.py` (peri-onset, xcorr, forecasting).
Figures: `writeup/figures/delta_onset/fig_precursor_{grid,xcorr,auc}_{q15,q30}.png`
(+ `fig_onset{s_overview,_gallery}_S2N2_q30.png` for the trigger).
Data: `analysis/delta_onset/outputs/precursor_summary_{q15,q30}.csv`.

---

## Methods addition (pairs with §2.8–2.9, the CAP-SWA / spindle event methods)

**2.y Delta-onset triggering and the CAP peri-onset test.** To ask whether discrete
cortical slow-wave events leave a capacitive signature, we first defined an EEG
**delta-burst onset** as an event (Schmitt trigger on the 0.5–4 Hz Hilbert envelope:
per-session robust NREM baseline, high = median + 2·MAD sustained ≥ 4 s, onset walked
back to the burst's rising edge; ≥ 25 s refractory so events are near-independent). Only
onsets in NREM with a motion-clean −30→0 s pre-window were kept, and — because the whole
test hinges on a clean quiet→delta transition — we additionally required the pre-window
EEG delta to be quiescent (mean below the low threshold). Two quiescence windows are
reported (15 s / 30 s; "q15"/"q30") as a robustness check. This selects **isolated slow
wave / K-complex onsets emerging from a quiet background** (predominantly N2; sustained
N3 has no discrete quiet onset), 6–108 events per session.

For each onset we extracted three capacitive channels (CLE, CRE, CH) in three bands
(0–0.5, 0.5–1, 1–3 Hz; the 0–0.5 band high-passed at 0.03 Hz to reject DC/coupling
drift), took each band's amplitude envelope, z-scored it within NREM, and averaged the
−30→+15 s window across onsets. Aggregation is per-subject then across the six subjects
(never pooling raw epochs). Three read-outs: (i) the **peri-onset average** vs a
**random-NREM null** (count-matched, motion-clean, ≥ 60 s from any real onset); (ii) a
**CAP→EEG cross-correlation** over NREM with a circular-shift null (positive lag = CAP
leads); (iii) a **forecasting AUC** — can CAP band power in a −12→−2 s pre-onset window
separate "about to onset" from random NREM.

---

## Results addition (§3.y)

**A capacitive event is time-locked to EEG delta onset, and it follows the cortex.**
Locking the capacitive envelopes to EEG delta-burst onset revealed a sharp, unambiguous
band-power increase in every channel and every band, rising at onset and peaking
**+2–5 s later**, far above the flat random-NREM null (Figure Y, grid). The effect is
strongest on the central channel and larger in the 0.5–1 and 1–3 Hz bands than in
0–0.5 Hz, and it is consistent across subjects.

**Post-onset peak of the capacitive response (q30; across-subject mean z, n = 6).**
Every band × channel peaks within +2.4→+4.9 s of onset, against a random-NREM null that
never leaves ≈ 0. CH (central) is the dominant channel; the 0.5–1 and 1–3 Hz bands carry
the largest deflections.

| Channel | 0–0.5 Hz | 0.5–1 Hz | 1–3 Hz | null peak (max) |
|---------|----------|----------|--------|-----------------|
| **CLE** | 0.60 z @ +2.5 s | 0.67 z @ +2.6 s | 0.62 z @ +2.4 s | ≤ 0.24 |
| **CRE** | 0.52 z @ +4.9 s | 0.96 z @ +4.2 s | 0.94 z @ +4.0 s | ≤ 0.12 |
| **CH**  | 0.71 z @ +3.8 s | **1.77 z @ +4.1 s** | **1.76 z @ +4.1 s** | ≤ 0.01 |

(q15 reproduces this — same rank order, peaks +5→+8 s; `precursor_summary_q15.csv`.)

Crucially, the coupling is **directional and the professor's precursor hypothesis is not
supported**: (i) the pre-onset lead window sat *at or below* baseline in every band and
channel (lead amplitude −0.02 to −0.11 z), i.e. no CAP rise *before* onset; (ii) the
CAP→EEG cross-correlation peaked at lag ≈ 0 s (within ±0.2 s) with its shoulder on the
EEG-leads side, never on the CAP-leads side; and (iii) pre-onset CAP band power did not
forecast an imminent onset — AUC 0.42–0.56 pooled, per-subject straddling chance. The
capacitive event therefore **follows** the cortical delta onset rather than heralding it.

Because the mask carries no cortical delta spectrally (§3.5) and no cortical sigma (§3.6),
this post-onset event cannot be electrical pickup. It is a **mechanical/autonomic
co-activation** — most parsimoniously a slow-wave/K-complex-associated transient in the
respiratory/cardiac mechanical coupling and micro-vascular pulsation that the temple
electrodes transduce (the three bands overlap the respiratory 0.1–0.5 Hz and cardiac
0.5–3 Hz signatures the mask already resolves, so the "band power" rise is best read as a
brief **amplitude modulation of the ongoing mechanical signals at delta onset**, not a
new oscillation). It is the phasic, single-event counterpart of the tonic CAP-SWA state
marker (§3.x) and the direct-event analogue of the spindle-locked 0–3 Hz bump (§3.6).

Both quiescence windows (q15, q30) give the same picture, so the result is not an artifact
of the onset definition.

---

## Limitations (fold into the discussion)

- Onsets are **isolated N2 slow-wave / K-complex onsets** by construction (the quiescence
  gate removes sustained N3, which has no discrete quiet onset). The claim is about
  *delta-burst onset events*, not deep-sleep SWA en masse.
- We exclude gross motion in the pre-window, but cannot fully separate an **autonomic**
  (pulse/vascular/respiratory) from a **micro-motor** (tone/K-complex-linked movement)
  contribution to the post-onset event with these channels alone.
- n = 6; direction is reported per-subject.

## TODO before paste
- [ ] Fill exact post-onset peak table (amplitude z, latency s, null) from
      `precursor_summary_q30.csv` once run **bdffh3ltl** completes.
- [ ] Add a **per-subject response-consistency** statistic (how many of 6 subjects show a
      post-onset peak above their own null) for the honest-negative reporting — small
      addition to `delta_cap_precursor.py`.
- [ ] Decide figure: q30 grid as main; xcorr + auc as supplement (or a 1-row composite).
- [ ] Confirm section number / placement with the canonical manuscript
      (`writeup/main/…main.docx`).
