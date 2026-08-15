"""Rewrite section 4.2 as a demonstration, on the recomputed pipeline.

  in  : writeup/main/CAP_sleep_mask_manuscript_main.docx   (edited in place)
  bak : writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_42DEMO_20260814.docx

The section is reframed from a validation claim to a demonstration: the mask's
rate estimate is shown against PSG on twelve nights with per-session calibration,
and the conditionality of that calibration is stated once, plainly, rather than
carried by a statistics apparatus that six subjects cannot support.

Three substantive changes come with it.

  1. The respiratory estimator changes. The published figure of 0.91 br/min came
     from a Welch estimator whose 4 s segment gives two usable bins across
     0.1-0.5 Hz; it returned 15.0 br/min in 99.98% of epochs, so k was
     identically 15/median(reference) and the "calibrated rate" was the session's
     own median reference rate. Loose peak counting replaces it: a real estimate
     that varies, at 1.79 br/min. No framing rescues a constant, so this is not
     optional.
  2. Every number in Table 3 is recomputed on that estimator (CRE channel,
     per-session k) from reports/rates/rerun/per_session.csv.
  3. The k-versus-age analysis is deleted. It reached significance only under the
     degenerate estimator; on the operational one rho = -0.37, exact p = 0.50,
     the sign flips under leave-one-out, and a held-out age prior is worse than a
     single constant. Figure 5 goes with it, replaced by Bland-Altman.

Figures: 4 becomes the representative-night trace, 5 becomes Bland-Altman.
Figures 6-9 are untouched, since one figure is removed and one added.

Run from the repo root:  python writeup/edits/apply_section_42_demo.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_review_edits import Doc, W  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_42DEMO_20260814.docx")

FIG_NIGHT = Path("writeup/figures/rate_rerun/fig_representative_night.png")
FIG_BA = Path("writeup/figures/rate_rerun/fig_bland_altman.png")

TABLE3 = [
    ("Respiratory", "Cardiac"),
    ("Per-epoch error, median [IQR]", "1.79 br/min [1.65–2.01]", "3.41 BPM [3.06–8.38]"),
    ("Night-mean error, median [IQR]", "0.24 br/min [0.14–0.34]", "1.56 BPM [1.22–2.08]"),
    ("Night-mean error, worst night", "1.00 br/min", "7.41 BPM"),
    ("Bias (95% limits of agreement)", "−0.10 (−5.85, +5.65) br/min", "−0.93 (−25.48, +23.62) BPM"),
    ("Calibration factor k, median [IQR]", "1.18 [1.12–1.27]", "1.96 [1.77–2.01]"),
    ("Within-night r, median", "−0.03 (p = 0.34)", "−0.08 (p = 0.30)"),
    ("Nights with r > 0", "5 / 12", "5 / 12"),
    ("Reference rate SD within night", "1.57 br/min", "6.58 BPM"),
]


def main() -> None:
    if not DOC.exists():
        raise SystemExit("missing: %s" % DOC)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ─────────────────────────────────────────────────── 4.2 opening
    d.set_text(
        "The sensor was evaluated against PSG in two regimes",
        "This section demonstrates what the mask recovers of respiratory and "
        "cardiac rate over a night, against simultaneous polysomnography on twelve "
        "recordings. One point should be read into every number that follows: the "
        "capacitive waveform does not produce one deflection per physiological "
        "event, so each recording carries a scalar calibration factor k fitted "
        "against its own reference (§4.2.1). The figures below therefore show "
        "agreement after per-session calibration, not standalone accuracy, and we "
        "do not claim performance for an uncalibrated device. With six subjects we "
        "do not attempt to establish how well that calibration transfers between "
        "subjects or nights; that requires a larger cohort.")

    d.set_text(
        "Both regimes use the same operational pipeline described in section 3.5",
        "Both bands use the same estimator: prominence-thresholded peak counting on "
        "the right-temple channel, rescaled by k and smoothed with a causal "
        "three-epoch median filter. The Welch spectral estimator is reported only "
        "as a baseline and is not used, for the reason given in §3.5 — at a 4-second "
        "segment its frequency resolution is 0.25 Hz, which leaves two usable bins "
        "across the respiratory band, and it returned 15.0 breaths/min in 99.98% of "
        "epochs. A constant cannot demonstrate rate detection whatever its error, "
        "and the comparison across estimators and channels is given in Figure S1.")

    # ─────────────────────────────────────────────────── the two subsections
    d.set_text("Aggregate accuracy: the per-night mean rate",
               "Agreement with the polysomnographic reference")
    d.set_text(
        "Averaged over a whole night, the mask tracks both rates closely",
        "The mask follows both rates across a full night (Figure 4). Pooled over "
        "the twelve recordings, the per-epoch median absolute error is 1.79 br/min "
        "for respiration and 3.41 BPM for the cardiac band, and the Bland–Altman "
        "bias is −0.10 br/min and −0.93 BPM with 95% limits of −5.85 to +5.65 and "
        "−25.48 to +23.62 (Figure 5). Averaged over a whole recording the agreement "
        "is closer, as the epoch-level errors are largely uncorrelated and average "
        "out: the difference between the mean estimated and mean reference rate has "
        "a median of 0.24 br/min and 1.56 BPM, and its worst case across the twelve "
        "nights is 1.00 br/min and 7.41 BPM. Per-session values are in Table S1 and "
        "the full set of nightly traces in Figures S8 and S9. Two caveats bound "
        "these numbers. They follow per-session calibration against the reference, "
        "and the respiratory rates in this cohort span a narrow range — session "
        "means 14.4 to 16.8 br/min — so the night-mean figure in that band is a "
        "modest test.")

    d.set_text("Within-night accuracy: rate variation is not recovered",
               "What the mask does not recover: variation within the night")
    d.set_text(
        "Within a night, neither band's rate variation was recovered",
        "Agreement on the level of a rate is not the same as following it as it "
        "changes, and the mask does not do the latter (Table 3). Within a "
        "recording, the estimate and the reference correlate at a median r of "
        "−0.03 for respiration and −0.08 for the cardiac band across epochs, "
        "positive in 5 of 12 nights in both bands and not different from zero "
        "(Wilcoxon p = 0.34 and 0.30 on the twelve per-night values). No estimator, "
        "channel or fusion strategy we tested produced a within-night correlation "
        "distinguishable from a circular-shift null (Figure S3). The failure is "
        "visible directly in the traces: in the cardiac panels of Figures 4 and S9 "
        "the reference makes sustained excursions of 20 to 40 BPM that the estimate "
        "does not follow, while holding the correct overall level.")

    d.set_text(
        "The two bands fail to track for different reasons",
        "The two bands reach that limit from different directions. Within a night "
        "the reference respiratory rate varies with a median standard deviation of "
        "only 1.57 br/min, comparable to the per-epoch error itself, so there is "
        "little variation available to resolve. The cardiac reference varies by "
        "6.58 BPM, comfortably more than its 3.41 BPM error, so that variation is "
        "in principle resolvable and is nonetheless not recovered; §5.3 offers an "
        "account of why. Either way the practical consequence is the same, and it "
        "is a scope statement rather than a tuning problem: the mask can report a "
        "nightly average, and should not be used where instantaneous rate matters.")

    # ─────────────────────────────────────────────────── Table 3
    tbl = d.body[d.find("Table 3. Rate detection in both regimes") + 1]
    assert tbl.tag == W + "tbl", "Table 3 not where expected"
    rows = tbl.findall(W + "tr")
    flat = []
    for tr in rows:
        for tc in tr.findall(W + "tc"):
            flat.append(tc)
    # rewrite as a flat label/value/value grid, blanking anything left over
    wanted = [TABLE3[0][0], TABLE3[0][1]] + [x for row in TABLE3[1:] for x in row]
    for i, tc in enumerate(flat):
        ts = tc.findall(".//" + W + "t")
        if not ts:
            continue
        for t in ts[1:]:
            t.text = ""
        ts[0].text = wanted[i] if i < len(wanted) else ""
    d.n_patch += 1

    d.set_text(
        "Table 3. Rate detection in both regimes,",
        "Table 3. Rate agreement, from the operational estimator (loose peak "
        "counting on CRE, per-session k, causal three-epoch median filter). Values "
        "are medians across the twelve recordings with the interquartile range in "
        "brackets; per-night values are in Table S1. The per-epoch error is the "
        "median absolute error across a recording's 30-second epochs; the "
        "night-mean error is |mean(estimate) − mean(reference)| for that recording. "
        "Within-night r is the Pearson correlation between estimate and reference "
        "across a recording's epochs, tested across nights with a Wilcoxon "
        "signed-rank test on the twelve values.")

    # ─────────────────────────────────────────────────── Figures 4 and 5
    d.swap_image(d.body[d.find("Figure 4. Bland–Altman agreement") - 1], FIG_NIGHT)
    d.set_text(
        "Figure 4. Bland–Altman agreement",
        "Figure 4. Respiratory (A) and cardiac (B) rate across one night (S1N1). "
        "The capacitive estimate is loose peak counting on CRE, rescaled by that "
        "session's k; the reference is the multi-sensor respiratory consensus and "
        "ECG R-peaks respectively. The estimate holds the correct level across the "
        "whole recording in both bands, while the sustained cardiac excursions near "
        "hour 2 are not followed. All twelve recordings are shown in Figures S8 "
        "and S9.")

    template_img = d.body[d.find("Figure 4. Respiratory (A) and cardiac (B) rate") - 1]
    d.insert_after("Figure 4. Respiratory (A) and cardiac (B) rate", [
        d.image_para(FIG_BA, template_img),
        d.para("Figure 5. Bland–Altman agreement between the mask and the reference "
               "for respiratory (A) and cardiac (B) rate. One point per 30-second "
               "epoch, all twelve recordings pooled, after per-session calibration. "
               "The solid line is the bias and the dashed lines the 95% limits of "
               "agreement. These limits are epoch-level; the night-mean errors in "
               "Table 3 are an order of magnitude smaller and are not visible at "
               "this scale."),
    ])

    # ─────────────────────────────────────────────────── 4.2.1: drop the age story
    n = d.delete_range(
        "k, age, and the calibration a deployment would require",
        "Figure 5. Calibration factor k and subject age")
    print("removed %d paragraphs of age analysis (incl. old Figure 5)" % n)
    d.n_patch += n

    d.patch("k is reproducible within a subject",
            "Diagnostic estimates from 50 random windows agreed with whole-night "
            "values to within 0.04.",
            "k is a whole-night quantity: the median of estimate over reference "
            "across every valid epoch of the recording. We examined whether it "
            "varies with subject age and found no relationship in either band that "
            "survives this sample size — with six subjects a Spearman correlation "
            "must reach |ρ| ≥ 0.83 to be significant, and on held-out subjects an "
            "age-based prior for k performs worse than a single population "
            "constant. We therefore report k as a per-subject calibration constant "
            "and make no claim about what sets it.")

    # respiratory k is no longer near unity, so the morphology gloss must go
    d.patch("k is a waveform-morphology count",
            "The respiratory k of ≈0.99 (median across sessions) reflects the "
            "simpler coupling: each breath produces one dominant temple "
            "displacement.",
            "The respiratory k of ≈1.18 is closer to unity but not equal to it: "
            "peak counting registers about 18% more deflections than there are "
            "breaths, so the respiratory waveform is not the clean "
            "one-peak-per-breath signal a value of 1.0 would imply.")

    d.save(DOC)
    print("applied %d edits -> %s" % (d.n_patch, DOC))


if __name__ == "__main__":
    main()
