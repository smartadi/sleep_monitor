"""End-to-end consistency pass over Results and the Discussion text that restates it.

Eighteen issues, found by reading §4.1 to §7 against each other after the day's
edits. Grouped by kind.

ORDER
  Table 4 was printed before Table 3.

CONTRADICTIONS INTRODUCED BY 4.2.2
  the 4.2 subheading said no configuration recovers within-night variation;
  4.2's opening said transfer was not attempted; 4.2 told the reader that
  "the paragraph after next" carries a number, which an inserted table broke;
  4.2's scope sentence and 7's conclusion both said the mask cannot follow
  within-night variation at all.

BROKEN SENTENCES
  4.2.1 had two of the day's edits collide mid-sentence; 4.5's method trim left
  "yields (344 qualifying onsets" dangling and a duplicated method sentence.

STALE NUMBERS
  5.1 carried cardiac k IQR 1.79-2.00 where Table 3 has 1.77-2.01, and a
  constant-predictor figure of 0.78 br/min that Table 4 now gives as 1.20.

CAPTIONS ASSERTING WHAT THE TEXT RETRACTS
  Figure 8 still said onsets "evoke" the response and that it is present in all
  six subjects, both of which 4.5 now qualifies.

DISCUSSION AGAINST RESULTS
  5.2's delta-burst sentence lacked the arousal caveat; 5.4 called the ridge
  power "consistently low across subjects" where it is four of six; 5.5 said
  respiratory k is about 1 and each breath gives a single displacement, which
  4.2.1 explicitly denies at k = 1.18; 6's third limitation called k stable
  across nights without the cardiac 20% figure.

STRUCTURE
  4.6 still carries its whole detection method inside Results, which is what
  3.7 was created to stop for 4.4 and 4.5. Its method moves to 3.6, whose
  subject is exactly harmonic and ridge detection.

Run from the repo root:  python writeup/edits/apply_results_consistency.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_CONSISTENCY_20260817.docx")

K_REPRO = (
    "k is reproducible within a subject. Across each subject's two nights the absolute change in "
    "k had a median of 0.027 for respiration (maximum 0.178) and 0.146 for the cardiac band "
    "(maximum 0.387). Respiratory k is therefore effectively a subject-level constant. The "
    "cardiac band is reproducible only to about 20% of k, which is what carrying a subject's k "
    "across to their other night costs: the night-level error rises from 1.56 to 3.77 BPM. "
    "Cardiac k is otherwise stable, apart from S6, whose two nights account for the full cardiac "
    "range of 0.97–2.28 while the central half of the distribution spans only 1.77–2.01; S6's "
    "second night is also the one recording whose cardiac reference is photoplethysmography "
    "rather than ECG (§3.3), so its outlying k cannot be attributed to the mask alone. k is a "
    "whole-night quantity: the median of estimate over reference across every valid epoch of the "
    "recording. We examined whether it varies with subject age, and with any other capacitive "
    "feature we could form, and found no relationship in either band that survives this sample "
    "size (Figure S4) — with six subjects a Spearman correlation must reach |ρ| ≥ 0.83 to be "
    "significant, and on held-out subjects an age-based prior for k performs worse than a single "
    "population constant. We therefore report k as a per-subject calibration constant and make no "
    "claim about what sets it."
)

DELTA_INTRO = (
    "Cortical slow-wave (delta) activity provides a second discrete cortical event at which to ask "
    "what the temple sensor reflects: the onset of a delta burst. The detection and the "
    "event-triggered procedure are described in §3.7. They yield 344 qualifying onsets in total, "
    "distributed very unevenly — 1 to 99 per night, with three nights contributing fewer than ten "
    "— and predominantly isolated N2 slow-wave and K-complex onsets, since sustained N3 slow-wave "
    "activity has no discrete quiet onset."
)

FIG8_CAPTION = (
    "Figure 8. The capacitive response around delta-burst onsets. Peri-onset average of capacitive "
    "band power (rows: 0-0.5, 0.5-1, 1-3 Hz; columns: left temple CLE, right temple CRE, forehead "
    "CH) time-locked to EEG delta-burst onset (green line, t = 0), computed with a strictly causal "
    "envelope estimator so that no post-onset power can leak backward in time. Solid lines and "
    "shading are the mean and standard error across the six subjects; gray dashed lines are the "
    "count-matched random-NREM null. The pre-onset baseline is flat in every panel and a sharp "
    "band-power increase follows onset, peaking within several seconds, on the full onset set in "
    "all six subjects. Most of these onsets carry a scored arousal within 10 s; §4.5 reports what "
    "survives when those are excluded."
)

COMB_METHOD = (
    "Harmonic-comb episodes. Separately from the persistent rate ridges, the spectrogram "
    "intermittently carries a stack of sustained, temporally flat narrow bands. Each raw channel "
    "was worked independently, since averaging channels dilutes a comb strong on a single "
    "electrode. The 0–3 Hz spectrogram was background-subtracted per time column, and an episode "
    "defined as a sustained interval carrying at least three integer-harmonic rungs, each a peak "
    "at least 5 dB above the local floor with at least three consecutive from the fundamental. "
    "Within an episode the rungs were recovered as the persistent horizontal bands themselves — "
    "spectral peaks tracked across time, kept if they lasted at least 1.5 minutes and were present "
    "in at least 60% of their span — and reported at their true frequencies. Requiring three "
    "consecutive harmonics means the two rhythms the sensor always carries, respiration and heart "
    "rate, cannot by themselves form an episode. Episodes overlapping in time across channels were "
    "merged into single events, and each event was related to the scored hypnogram with every REM "
    "quantity compared against a matched random-NREM null of 200 draws per session."
)

COMB_RESULT = (
    "Separately from the persistent respiratory and cardiac rate ridges (§4.3), the capacitive "
    "spectrogram intermittently carries a stack of several sustained, temporally flat narrow bands "
    "— a harmonic comb, detected as described in §3.6."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    # ------------------------------------------------------------------ ORDER
    cap4 = d.find("Table 4. Rate accuracy under four calibrations")
    tbl4 = next(i for i in range(cap4, len(body)) if body[i].tag == W + "tbl")
    block = [body[cap4], body[tbl4]]
    for el in block:
        body.remove(el)
    cap3 = d.find("Table 3. Rate agreement, from the operational estimator")
    tbl3 = next(i for i in range(cap3, len(body)) if body[i].tag == W + "tbl")
    for off, el in enumerate(block):
        body.insert(tbl3 + 1 + off, el)
    print("  Table 4 moved after Table 3")

    # -------------------------------------------------- 4.2 vs 4.2.2 conflicts
    d.set_text("Variation within the night, which no configuration tested recovers",
               "Variation within the night, which no rate estimator recovers")
    d.patch("This section shows what the mask recovers of respiratory and cardiac rate",
            "With six subjects we do not try to establish how well that calibration transfers "
            "between subjects or nights; that needs a larger cohort.",
            "How far that calibration transfers between nights and between subjects is reported "
            "in Table 4 and §4.2.2.")
    d.patch("Agreement on the level of a rate is not the same",
            "single nights do scatter above zero, and the paragraph after next puts a number on "
            "that", "single nights do scatter above zero, and §4.2.2 puts a number on that")
    d.patch("The two bands reach that limit from different directions",
            "the mask can report a nightly average, and should not be used where instantaneous "
            "rate matters.",
            "read through a rate estimator the mask reports a nightly average and should not be "
            "used where instantaneous rate matters. §4.2.2 shows that this is a limit of rate "
            "estimation rather than of the recording.")
    print("  4.2: heading, transfer sentence, cross-reference and scope statement reconciled")

    # -------------------------------------------------------- broken sentences
    d.set_text("k is reproducible within a subject", K_REPRO)
    d.set_text("Cortical slow-wave (delta) activity provides a second discrete", DELTA_INTRO)
    print("  4.2.1 and 4.5 opening sentences repaired")

    # ----------------------------------------------------------- stale numbers
    d.patch("The mask recovers each night's mean respiratory rate",
            "an interquartile range spanning only 1.79–2.00", "an interquartile range spanning "
            "only 1.77–2.01")
    d.patch("The mask recovers each night's mean respiratory rate",
            "so a constant predictor already reaches 0.78 br/min",
            "so a no-sensor constant already reaches 1.20 br/min (Table 4)")
    print("  5.1: cardiac k IQR and the constant-predictor figure aligned with Tables 3 and 4")

    # ------------------------------------------------------------- Figure 8
    d.set_text("Figure 8. Delta-burst onsets evoke a mechanical response", FIG8_CAPTION)
    print("  Figure 8 caption no longer asserts what 4.5 withdraws")

    # -------------------------------------------------- Discussion vs Results
    d.patch("The mask does not pick up cortical electrical activity",
            "and the delta-burst response follows the cortical onset at zero lag rather than "
            "preceding it",
            "and the delta-burst response follows its onset at zero lag rather than preceding it, "
            "whether the burst or the arousal that usually accompanies it is the trigger (§4.5)")
    d.patch("The stage-associated spectral structure, while too weak for standalone staging",
            "The consistently low N3 respiratory-ridge power across subjects would add value as "
            "one input among many",
            "The lower N3 respiratory-ridge power, which holds in four of six subjects, might add "
            "value as one input among many")
    d.patch("The cardiac k ≈ 2 is consistent in kind with the ballistocardiographic literature",
            "The respiratory k ≈ 1 reflects the simpler coupling: each breath produces a single "
            "dominant displacement of the temple sensor, unlike the biphasic cardiac pulse.",
            "The respiratory k of 1.18 reflects a simpler coupling than the biphasic cardiac "
            "pulse, though its departure from unity shows that a breath does not produce exactly "
            "one capacitive deflection either.")
    d.patch("Third, k-calibration requires a reference cardiac or respiratory rate",
            "k is stable within and across nights once estimated",
            "k is stable within a night and, for respiration, across a subject's nights; the "
            "cardiac factor reproduces only to about 20% (§4.2.1)")
    print("  5.2, 5.4, 5.5 and limitation 3 aligned with the Results as they now stand")

    # ---------------------------------------------------------------- 7
    d.patch("This study provides a multi-method characterization",
            "but does not follow either rate as it varies within the night",
            "but no rate estimator follows either rate as it varies within the night, though a "
            "decoder reading the whole epoch partly does (§4.2.2)")
    print("  7: conclusion matched to 4.2.2")

    # ------------------------------------------------------------ 4.6 method
    at = d.find("To separate the two physiological rhythms the sensor carries")
    body.insert(at + 1, d.para(COMB_METHOD))
    d.patch("Separately from the persistent respiratory and cardiac rate ridges",
            "Separately from the persistent respiratory and cardiac rate ridges (§4.3), the "
            "capacitive spectrogram intermittently carries a stack of several sustained, "
            "temporally flat narrow bands — a harmonic comb. Working on each raw channel "
            "independently (averaging channels dilutes a comb strong on a single electrode), we "
            "background-subtracted the 0–3 Hz spectrogram per time column and defined an episode "
            "as a sustained interval carrying at least three integer-harmonic rungs, each a peak "
            "at least 5 dB above the local floor with at least three consecutive from the "
            "fundamental; within each episode the rungs were recovered as the actual persistent "
            "horizontal bands (spectral peaks tracked across time, kept if they lasted at least "
            "1.5 minutes and were present in at least 60% of their span) and reported at their "
            "true frequencies. Because a comb of three consecutive harmonics is required, the two "
            "rhythms the sensor always carries — respiration and heart rate — do not by "
            "themselves form an episode. Episodes overlapping in time across channels were merged "
            "into single events, and each event was related to the technologist-scored hypnogram, "
            "with every REM quantity compared against a matched random-NREM null (200 draws per "
            "session).", COMB_RESULT)
    print("  4.6 method moved into 3.6, matching what 3.7 did for 4.4 and 4.5")

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
