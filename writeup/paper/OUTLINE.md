# CAP Sleep Mask Paper — Outline

**Target journals:** Sleep, IEEE TBME, Sensors
**Scope:** Validation of respiratory/cardiac signals in capacitive temple sensors, per-window rate
estimation with k-calibration, the k(t) biomarker, and whole-night harmonic/ridge spectral structure.

## Standing rules (apply while drafting)
1. **SWS-EEG analysis is PENDING.** Do not write the Intro framing, the Discussion synthesis, or any
   SWS-vs-EEG-delta section until that analysis resolves. Leave them as stubs.
2. **Harmonics are reported as *stage-associated*, never as *SWS detection*,** until rule 1 resolves.
3. **No p-values for stage-wise harmonic/ridge claims** — sample size is insufficient. Report direction
   and effect (fewer / slower / lower-power ridges in N3) qualitatively, with per-subject consistency
   counts, not significance numbers.
4. Sleep staging (UMAP/GMM) is **out of scope** — deferred to a future paper.

---

## Section Status

| # | Section | Status | Notes |
|---|---------|--------|-------|
| 1 | Introduction | STUB — pending SWS-EEG | Wearable gap, CAP sensor concept, contributions. Framing deferred. |
| 2.1 | Participants & protocol | NOT STARTED | 6 subjects × 2 nights, CAP mask + PSG |
| 2.2 | Signal preprocessing | NOT STARTED | OLS/NLMS accel artifact removal, channel selection, filtering |
| 2.3 | Ground truth derivation | NOT STARTED | ECG R-peaks, Flow peaks (neurokit2), PSG staging |
| 2.4 | Signal validation approach | NOT STARTED | Coherence, frequency matching, phase-randomized surrogates |
| 2.5 | Rate estimation & k-factor | NOT STARTED | 6 base + advanced methods, overcounting, per-session k |
| 2.6 | Multi-channel fusion | NOT STARTED | CWT ridge, Viterbi smoothing, quality-weighted fusion |
| 2.7 | k(t) biomarker analysis | NOT STARTED | Stage-wise k, autocorrelation, physiological correlations |
| 2.8 | Harmonic detection | NOT STARTED | HPS, cepstral, explicit F0 + HER |
| 2.9 | Persistent ridge tracking | NOT STARTED | Peak linking, prominence scoring |
| 3.1 | Signal validation results | NOT STARTED | Coherence, freq match, surrogates |
| 3.2 | Rate accuracy (no k) | NOT STARTED | Method × channel, CWT ridge |
| 3.3 | Multi-channel fusion & smoothing | NOT STARTED | Fusion strategies, Viterbi |
| 3.4 | k-scaled accuracy | NOT STARTED | Best resp + cardiac, per-session, LOSO |
| 3.5 | k_cardiac as biomarker | NOT STARTED | Stage distributions, halflife, correlations |
| 3.6 | Harmonic structure by stage | NOT STARTED | Stage-associated ridge features (no p-values, rule 3) |
| 3.7 | Persistent ridge features | NOT STARTED | N3 = fewer, slower, lower-power ridges (directional) |
| 3.8 | Channel dominance & prominence | NOT STARTED | CH/CRE dominance, N3 prominence |
| 4 | Discussion | STUB — pending SWS-EEG | k-factor, BCG complexity, harmonic origin, limitations. Deferred. |

### Parked (do not draft yet)
| # | Section | Status | Notes |
|---|---------|--------|-------|
| — | SWS vs EEG delta | PENDING ANALYSIS | Item 3 in journal. Under active analysis; framing not yet decided. |

### Cut from this paper (future work)
- Feature extraction & supervised UMAP staging (was §2.10 / §3.9) — separate paper.

---

## Narrative Arc (current scope)

**Part A — Rate Detection (§2.5–2.7, §3.1–3.5)**
1. Signal is there — coherence + surrogates prove respiratory & cardiac coupling
2. Raw counting overcounts — per-session k-calibration fixes bias
3. Multi-channel fusion + CWT ridge + Viterbi improves precision
4. k(t) is itself informative — tracks sleep stage via BCG complexity

**Part B — Spectral Signatures (§2.8–2.9, §3.6–3.8)**
5. CAP spectrograms show structured harmonic ladders (not noise)
6. Harmonic/ridge features are stage-associated — N3 = fewer, slower, simpler ridges (directional)
7. CH/CRE channels carry the strongest signal

**(Deferred) Part C — does the mask see SWS the way EEG does?** Held pending the SWS-EEG analysis.

---

## Deliverable & sources

- **Deliverable:** `writeup/paper/CAP_sleep_mask_paper.docx` — built from `DRAFT.md` via `build_docx.py`.
  Edit DRAFT.md, never the docx directly.
- Archived prior drafts (reference only) in `writeup/_legacy/`:
  `CAP_rate_consolidation_section.docx` (35 figs, 4 tables), `CAP_sleep_analysis_paper.docx`,
  `CAP_harmonic_ridge_analysis.docx`, `document_v5.xml`.
- `writeup/SFN 2026 *.docx` — conference abstracts (separate deliverable, kept in place).
- Source material: `notebooks/validation_methods.md`, `validation_results.md`,
  `k_biomarker_writeup.md`, `peak_ratio_method_writeup.md`.
