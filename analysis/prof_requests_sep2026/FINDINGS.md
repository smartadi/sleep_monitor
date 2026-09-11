# Prof requests — Sep 2026 (for the Friday 2pm meeting)

Four analyses answering Jae's Sep 7–9 emails. Descriptive, per-subject (n=6), no
headline p-values. All scripts in this folder; plots in
`notebooks/plots/prof_requests_sep2026/`, tables in `reports/prof_requests_sep2026/`.

Run all four:
```
.venv/Scripts/python.exe analysis/prof_requests_sep2026/mean_cap_vs_rem.py
.venv/Scripts/python.exe analysis/prof_requests_sep2026/band_power_comparison.py
.venv/Scripts/python.exe analysis/prof_requests_sep2026/cre_vs_cle_laterality.py
.venv/Scripts/python.exe analysis/prof_requests_sep2026/kcomplex_comparison.py
```

---

## A. Mean cap value vs REM / sleep cycles  (`mean_cap_vs_rem.py`)
> *"Correlate the mean Cap value changes (positive and negative) with REM … it is coupled with sleep cycles."*

**What the data shows — partial support, stated honestly.**
- The slow mean **does** swing up and down over the night with large positive and
  negative excursions (the prof's observation is real — see the 12-panel overlay).
- But those excursions are **step-like and dominated by posture/coupling shifts**,
  not a smooth ultradian oscillation tightly phase-locked to the NREM–REM cycle.
  Correlation of the slow mean with a smoothed REM-occupancy cycle is weak
  (median |r| ≈ 0.10–0.15 across nights, every channel).
- There **is** a modest, consistent-ish direction: **CH is higher in REM than NREM in
  9/12 nights** (median +0.19 z); **CLE−CRE is lower in REM in 8/12 nights** (median
  −0.33 z). The REM-onset-triggered average is weak pooled but present per subject.

**Takeaway for the meeting:** the mean-cap swings are real; their locking to REM is
weak/step-like and subject-variable. Worth discussing whether he's seeing specific
nights (the overlay grid makes per-night coupling visible) vs a cohort rule.

Figures: `A_meancap_rem_overlay_grid.png`, `A_meancap_rem_triggered.png`,
`A_meancap_rem_direction.png` · table `mean_cap_vs_rem_session.csv`

## B. Band-power comparison  (`band_power_comparison.py`)
> *"Power density comparison among cardiac bands, respiratory bands, harmonic combs, and SWS … I would like to see the band power."*

Median band power (fF², motion-free sleep epochs):

| channel | resp (0.1–0.5) | cardiac (0.5–3) | SWS band N3 (0.5–2) | comb epochs (0.1–3) | non-comb NREM |
|--------|-----|-----|-----|-----|-----|
| CLE | 0.022 | 0.065 | 0.037 | 0.259 | 0.094 |
| CRE | 0.012 | 0.060 | 0.034 | 0.401 | 0.078 |
| CH  | **0.153** | **0.107** | **0.060** | **2.92** | 0.27 |

- **CH carries by far the most power** in every band (it's the interhemispheric-
  difference channel with the largest swings).
- Power falls with frequency (1/f): respiratory band > cardiac/SWS. Respiratory is
  strongly CH-dominant; cardiac is roughly balanced CLE≈CRE.
- **Harmonic-comb events carry ~3–5× the in-band power of ordinary NREM epochs** on
  every channel — they are genuine high-power episodes, not spectral noise.

Figures: `B_psd_bands.png` (median PSD with bands shaded — literally "see the band
power"), `B_band_power_bars.png` · table `band_power_summary.csv`

## C. CRE vs CLE laterality of SWS ridges  (`cre_vs_cle_laterality.py`)
> *"… whether the ridges are occurring at both or one brain."*

**Different ridge types lateralize differently — it is not one uniform answer.**

| ridge type | CLE (L) | CRE (R) | CH | verdict |
|---|---|---|---|---|
| respiratory ridge | 96% | 97% | 94% | **bilateral** (12/12 nights, LI≈0) |
| cardiac ridge | 21% | 12% | 20% | **left-dominant** (8/12 nights) |
| harmonic combs | 0.06% | 1.2% | 1.4% | **right/CH**, absent on left (rare, N2, concentrated in a few nights) |
| SWS-prominence ridge (N3) | ~18% | ~4% | ~10% | **left-dominant in N3** |

- The persistent **respiratory** ridge is bilateral — both temples see it nearly
  always.
- The **cardiac** ridge and the **N3 broadband SWS ridge** are **left(CLE)-dominant**.
- The **harmonic combs** are essentially **one-sided (right / CH), almost never on
  CLE** — but they are rare and per-night laterality is noisy (only ~4 nights carry
  enough combs to score; pooled occupancy is CRE/CH-heavy, driven by S6).

Figures: `C_laterality_by_channel_stage.png`, `C_laterality_index.png` · tables
`laterality_channel_stage.csv`, `laterality_index_per_session.csv`

## D. K-complex comparison  (`kcomplex_comparison.py`)
> *"Did you also compare our signals with K-complex in EEG?"*

**Yes.** The delta-burst onset set is N2-dominant — it *is* the K-complex / isolated-
slow-wave population. Re-cutting to N2 onsets (n = **340** pooled, all 6 subjects):
- The SEC band-power response **follows the K-complex**, peaking **+2 to +5 s** after
  it, in all three bands (0–0.5, 0.5–1, 1–3 Hz) and all three channels, clearly above
  the random-NREM null. **No pre-onset precursor.**
- CLE leads (~+2 s), CRE lags (~+5 s), CH ~+4 s.

**This is the Fultz connection he flagged:** EEG leads the downstream mechanical/
hemodynamic response — same direction as Fultz et al. 2019 (EEG advances the CSF
response by ~6 s). Our SEC-follows-K-complex latency (~2–5 s) is the mask-side version
of that lead/lag. (Consistent with the earlier delta-onset precursor result:
hypothesis of a CAP *precursor* not supported; a CAP *response* after the cortical
event is.)

Figure: `D_kcomplex_triggered_grid.png` · table `kcomplex_response_peak.csv`

---

## Still open (document work he also asked for — not analysis)
- **Redraw figures with larger fonts:** manuscript Fig 3, 4, 8, 9, 10 (Fig 10 also
  needs (a)/(b) panel labels).
- **Supplementary:** Fig S4 remove background color + larger font; **fix the figure
  numbering** — S3/S5/S6 each appear twice.
- **Fultz paper** ("Coupled electrophysiological, hemodynamic, and CSF oscillations",
  2019) — not in the repo; fold the EEG→response lead framing into §4.5 / delta-onset.
