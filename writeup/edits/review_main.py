# -*- coding: utf-8 -*-
"""Entry point: build CAP_sleep_mask_manuscript_main_review.docx.

Run from the repo root:  python writeup/edits/review_main.py
The canonical manuscript is opened read-only and never written to.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import (Doc, SRC, DST, W, FIG_RIDGE_OVERLAY, FIG_RIDGE_STAGE,
                                FIG_CAP_SWA, FIG_SWA_ROC, FIG_HARMONIC_LADDER)
import review_content as C
import review_results as R
import review_results2 as R2
import review_results3 as R3

# per-session values, all verified against artifacts/mask_phase_c.parquet
TABLE_S1 = [
    ["Session", "Resp k", "Resp night err\n(br/min)", "Resp epoch AE\n(br/min)", "Resp r",
     "Card k", "Card night err\n(BPM)", "Card epoch AE\n(BPM)", "Card r"],
    ["S1N1", "0.91", "0.01", "0.97", "+0.32", "2.07", "1.68", "3.46", "−0.21"],
    ["S1N2", "0.95", "0.08", "0.87", "+0.17", "2.23", "1.62", "2.75", "−0.50"],
    ["S2N1", "0.91", "0.10", "0.91", "+0.10", "1.62", "0.27", "11.67", "+0.03"],
    ["S2N2", "0.90", "0.13", "1.25", "+0.06", "1.84", "2.26", "5.97", "+0.28"],
    ["S3N1", "1.03", "0.93", "1.89", "−0.34", "1.94", "2.20", "4.45", "−0.31"],
    ["S3N2", "1.02", "0.87", "2.25", "−0.08", "1.99", "1.42", "2.76", "−0.67"],
    ["S4N1", "0.90", "0.17", "1.10", "−0.09", "2.02", "1.27", "2.07", "+0.22"],
    ["S4N2", "0.95", "0.49", "1.17", "+0.11", "1.88", "0.55", "2.31", "−0.17"],
    ["S5N1", "0.97", "0.14", "0.65", "+0.01", "1.97", "0.98", "3.26", "−0.46"],
    ["S5N2", "0.97", "0.38", "0.80", "−0.34", "1.96", "1.58", "2.14", "−0.26"],
    ["S6N1", "1.05", "0.08", "0.73", "+0.05", "1.35", "3.05", "17.96", "+0.03"],
    ["S6N2", "1.04", "0.10", "0.87", "+0.09", "0.94", "10.45", "8.57", "+0.53"],
]

log = []


def note(msg):
    log.append(msg)
    print("  " + msg)


def main():
    d = Doc(SRC)
    body = d.body

    # ---------------------------------------------------------------- 1. figures
    # Renumber Figure 1-7 -> 2-8 in a single pass so the sensing schematic keeps
    # Figure 1 and the delta-onset figure keeps Figure 9. "Fig. 1a/1c" references
    # in sections 2 and 3.1 use "Fig." and are deliberately untouched.
    n = 0
    for t in body.iter(W + "t"):
        if t.text and "Figure" in t.text:
            new = re.sub(r"Figure ([1-7])\b", lambda m: "Figure %d" % (int(m.group(1)) + 1), t.text)
            if new != t.text:
                t.text = new
                n += 1
    note("renumbered Figure 1-7 -> 2-8 in %d runs (Figure 9 unchanged, gap at 8 closed)" % n)

    # ------------------------------------------------------------ 2. front matter
    i = d.find("Changwoo Lee")
    if d.text_of(body[i + 1]).startswith("Changwoo Lee"):
        body.remove(body[i + 1])
        note("removed duplicated author line")

    d.set_text("[TO BE WRITTEN", C.ABSTRACT)
    note("abstract written")

    d.set_text("we introduce intracranial slow-wave activity", C.INTRO_ISWA)
    note("introduction: ISWA reframed as a mechanical measure that the Results deliver")

    # ------------------------------------------------------- 3. sensing principles
    d.set_style("2. Sleep Mask and Sensing Principles", "Heading1")
    d.set_text("Sensing principle reading", C.FIG1_CAPTION)
    d.set_text("Considering the 120 mm penetration depth", C.SEC2_SENSITIVITY)
    d.set_text("In the sleep mask, the sensors were positioned", C.SEC2_PLACEMENT)
    d.set_text("Importantly, the SEC sensor measures", C.SEC2_REPRO)
    d.patch("A roll-to-roll fabrication process", "with only 2% variation between sensors",
            "with approximately 2% variation between sensors")
    note("section 2: heading level, panel labels, channel placement, broken sentences, SNR units")

    # ------------------------------------------------------------------ 4. methods
    d.set_style("3.1 Overnight testing", "Heading2")
    d.set_text("Participant demographics and recording characteristics", C.M31_EPOCHS)
    removed = d.delete_range("For the sleep study, we used the sleep mask in Fig. 1c",
                             "The overnight study was conducted without calibration")
    note("deleted %d duplicated paragraphs from 3.1 (MOD number preserved)" % removed)

    d.set_text("Motion artifact was suppressed", C.M32_CANCELLER)
    d.set_text("Unless otherwise noted, the canonical analysis channel", C.M32_CHANNEL)
    d.set_text("To validate this consensus", C.M33_CONSENSUS)
    d.set_text("All rates were computed on a common", C.M33_GRID)
    d.set_text("For each analysis epoch, magnitude-squared coherence", C.M34_COHERENCE)
    d.set_text("Per-window rates were estimated", C.M35_ESTIMATORS)
    d.set_text("SEC peak counts are systematically scaled", C.M35_K)
    d.set_text("The operational pipeline was therefore", C.M35_PIPELINE)
    d.set_text("Whole-night CAP spectrograms show structured", C.M36_RIDGES)
    d.set_text("To separate the two physiological rhythms", C.M36_BANDS)
    d.set_text("All rate accuracy metrics are reported within-session", C.M37_STATS)
    note("methods 3.2-3.7 rewritten: canceller used, channel trade-off, 30-s grid, "
         "canonical bound defined, 7 estimators + Welch parameters, unit of analysis")

    # Table 1: doubled subject IDs, header labels
    tbl1 = body[d.find("Session", d.find("Table 1. Recording sessions"))]
    d.cell_text(tbl1, 0, 2, "Age")
    d.cell_text(tbl1, 0, 6, "Analysis epochs (30 s)")
    fixed = 0
    for r_i in range(1, len(tbl1.findall(W + "tr"))):
        tc = tbl1.findall(W + "tr")[r_i].findall(W + "tc")[1]
        txt = "".join(t.text or "" for t in tc.findall(".//" + W + "t"))
        m = re.match(r"^(OS\d{3})\1$", txt.strip())
        if m:
            d.cell_text(tbl1, r_i, 1, m.group(1))
            fixed += 1
    note("Table 1: %d doubled subject IDs fixed, headers corrected to Age / 30 s epochs" % fixed)

    # ------------------------------------------------------------------ 5. results
    d.patch("The capacitive temple sensor signal contained sustained energy",
            "across all twelve recordings (Figure 2)", "across all twelve recordings (Figure 2)")
    d.set_text("Cross-spectral coherence at the ground-truth", R.R41_SURROGATE)
    d.patch("Respiratory frequency agreement", "between the CAP spectral peak",
            "between the SEC spectral peak")
    note("4.1: surrogate result reframed against the 5% null, per-stage claim bounded, CAP->SEC")

    # -- 4.2 rate detection
    d.set_text("The sensor was evaluated against PSG in two regimes", R.R42_INTRO)
    d.set_text("For the night-level regime we used the estimator", R.R42_PIPELINE)
    d.set_text("Respiratory rate was recovered with a per-session median", R.R42_AGGREGATE)
    d.set_text("Within a night, neither band's rate variation", R.R42_WITHIN)
    d.set_text("The two bands fall short of that bound", R.R42_MECHANISM)
    d.set_text("Table 3. Rate detection in both regimes", R.TABLE3_CAPTION)
    d.set_text("Night-level accuracy", "Aggregate accuracy: the per-night mean rate")
    d.set_text("Epoch-level accuracy", "Within-night accuracy: rate variation is not recovered")

    cap_i = d.find("Table 3. Rate detection in both regimes")
    tbl3_i = next(i for i in range(cap_i, len(body))
                  if body[i].tag == W + "tbl")
    template_tbl = body[tbl3_i]
    body[tbl3_i:tbl3_i + 1] = [d.table(R.TABLE_3, template_tbl)]
    note("4.2 rewritten; Table 3 rebuilt with night-level vs per-epoch error separated, "
         "true IQR for k, ratio row removed")

    d.set_text("Bland–Altman agreement for respiratory", R.FIG4_CAPTION)

    # -- 4.2.1 calibration factor k
    d.set_text("k is a waveform-morphology count", R.R421_MORPHOLOGY)
    d.set_text("k is reproducible within a subject", R.R421_REPRO)
    d.set_text("k, age, and the calibration a deployment", R.R421_AGE)
    d.set_text("We tested whether the respiratory relationship", R.R421_LOSO)
    d.set_text("Two cautions apply to the age result", R.R421_CAUTION)
    d.set_text("Calibration factor k and subject age", R.FIG5_CAPTION)
    note("4.2.1: exact permutation p (0.058) replaces t-approximation, session-exclusion "
         "count fixed, r=0.50 no longer described as confirmation")

    # -- 4.3 ridges: three-band revamp
    d.set_text("4.3 Harmonic structure and band-restricted ridge features", R2.R43_HEADING)
    d.set_text("CAP spectrograms displayed persistent spectral ridges", R2.R43_CONTINUITY)
    d.set_text("Representative session (S1N1): CRE spectrogram", R2.FIG6_CAPTION)
    d.set_text("Band-restricted ridge features by sleep stage",
               "Ridge features by band and sleep stage")
    d.set_text("Restricting ridge detection to each band separately", R2.R43_SLOW)
    stage_cap = d.find("Band-restricted ridge structure by sleep stage")
    # the stage figure sits immediately above its caption: text must go before the image
    body.insert(stage_cap - 1, d.para(R2.R43_RESP_CARD))
    d.set_text("Band-restricted ridge structure by sleep stage", R2.FIG7_CAPTION)
    after = d.find(R2.FIG7_CAPTION[:40])
    for off, p in enumerate([d.para(R2.R43_LADDER), d.para(R2.R43_SUMMARY)]):
        body.insert(after + 1 + off, p)

    # swap in the three-band figures
    ov_i = next(i for i in range(d.find(R2.R43_CONTINUITY[:40]), len(body))
                if body[i].findall(".//" + W + "drawing"))
    d.swap_image(body[ov_i], FIG_RIDGE_OVERLAY)
    st_i = next(i for i in range(d.find(R2.R43_SLOW[:40]), len(body))
                if body[i].findall(".//" + W + "drawing"))
    d.swap_image(body[st_i], FIG_RIDGE_STAGE)
    note("4.3 replaced with the three-band revamp (infra-slow ridge, 6/6 subjects; "
         "harmonic-ladder validation); both figures swapped to the July 23 three-band versions")

    # -- 4.4 spindles
    d.set_text("Onset-triggered averaging shows that low-band", R2.R44_LOWBAND)
    d.set_text("The capacitive response at sleep spindles is mechanical", R2.FIG8_CAPTION)
    note("4.4: dB attribution made consistent between text and figure caption")

    # -- 4.5 delta onsets
    d.set_text("Cortical slow-wave (delta) activity provides a second", R2.R45_METHOD)
    note("4.5: onset count corrected to 6-108 per night, 0.03 Hz high-pass stated")

    # -- 4.6 new section: ISWA
    fig9_cap = d.find("Delta-burst onsets evoke a mechanical response in the capacitive signal. "
                      "Peri-onset")
    img_template = body[next(i for i in range(fig9_cap, 0, -1)
                             if body[i].findall(".//" + W + "drawing"))]
    new46 = [d.para(R2.R46_HEADING, "Heading2"),
             d.para(R2.R46_INTRO),
             d.para(R2.R46_NEGATIVE),
             d.para(R2.R46_POSITIVE),
             d.para(R2.R46_MASKONLY),
             d.para(R2.R46_HYPOTHESES),
             d.image_para(FIG_CAP_SWA, img_template),
             d.para(R2.FIG10_CAPTION),
             d.para("")]
    for off, el in enumerate(new46):
        body.insert(fig9_cap + 1 + off, el)
    note("NEW section 4.6 inserted: SEC-vs-EEG SWA negative (r=-0.014, AUC 0.490 vs 0.740) "
         "plus the mechanical ISWA score (per-subject AUC 0.675, 6/6) + Figure 10")

    # --------------------------------------------------------------- 6. discussion
    d.set_text("The mask recovers per-session mean respiratory rate", R3.D51_RATES)
    d.set_text("Two further observations bear on this morphological", R3.D51_K)
    d.set_text("Band-restricted ridge features show physiologically", R3.D51_RIDGES)
    d.set_text("The mask carries no cortical electrographic signature", R3.D52_CORTICAL)

    # add the missing "what it does not provide" subsection for within-night tracking
    anchor = d.find("Cortical electrical activity")
    for off, el in enumerate([d.para(R3.D52_TRACKING_HEAD, "Heading3"),
                              d.para(R3.D52_TRACKING)]):
        body.insert(anchor + off, el)
    note("5.2: within-night tracking added as a stated limitation (was only in 5.3)")

    d.set_text("The absence of epoch-level cardiac tracking", R3.D53_CARDIAC)
    d.set_text("The respiratory case has a different basis", R3.D53_RESP)
    d.set_text("The harmonic direction ambiguity likely reflects", R3.D53_COUPLING)
    d.set_text("The mask is suitable for screening-level overnight", R3.D54_CLINICAL)
    d.set_text("The stage-associated spectral structure, while too weak", R3.D54_STAGING)
    d.set_text("The cardiac k ≈ 2 and the resulting", R3.D55_PRIOR)
    note("5.3-5.5: orphan 'harmonic direction ambiguity' resolved, apnea/eye-movement "
         "claims withdrawn, uncited comparisons qualified")

    # -------------------------------------------------------------- 7. limitations
    d.set_text("First, the sample is small", R3.L1)
    d.set_text("Second, this is a single-site laboratory study", R3.L2)
    d.set_text("Third, k-calibration requires a reference", R3.L3)
    d.set_text("Fourth, the respiratory consensus ground truth", R3.L4)
    d.set_text("Fifth, the 30-second analysis window", R3.L5)
    d.set_text("Sixth, the montage derivation", R3.L6)
    body.insert(d.find("Sixth, the montage derivation") + 1, d.para(R3.L7))
    note("6: laboratory->home contradiction fixed; new seventh limitation on post-hoc "
         "channel/estimator selection")

    # --------------------------------------------------------------- 8. conclusion
    d.set_text("This study provides a rigorous, multi-method characterization", R3.C1)
    d.set_text("These findings frame the mask as a viable tool", R3.C2)
    note("7: conclusion aligned with the corrected numbers and the ISWA result")

    # ---------------------------------------------------------------- 9. references
    i28 = d.find("28.Committee ISC. IEEE Standards")
    tail = d.text_of(body[i28 + 1])
    if tail.startswith("0 Hz to 300 GHz"):
        d._patch_el(body[i28], "Electric, Magnetic, and Electromagnetic Fields,",
                    "Electric, Magnetic, and Electromagnetic Fields, 0 Hz to 300 GHz. "
                    "IEEE Std C95.1-2019. 2019.")
        body.remove(body[i28 + 1])
        note("reference 28 rejoined into a single entry")

    # ------------------------------------------------------------- 10. supplementary
    d.set_text("Table S1. Per-session rate accuracy", R3.TABLE_S1_CAPTION)
    cap_i = d.find("Table S1. Per-session rate accuracy")
    ts1_i = next(i for i in range(cap_i, len(body)) if body[i].tag == W + "tbl")
    body[ts1_i:ts1_i + 1] = [d.table(TABLE_S1, body[ts1_i])]
    note("Table S1 rebuilt with night-level error columns; now agrees with Table 3 exactly")

    d.patch("Estimator and channel comparison", "across six base estimators",
            "across seven base estimators")
    d.set_text("The spectral estimator dominated the respiratory band", R3.S1_WRITEUP)
    d.set_text("Respiratory error was highest in REM", R3.S2_WRITEUP)
    d.set_text("An identical battery was applied to both bands", R3.S3_WRITEUP)
    d.set_text("No capacitive feature other than respiratory k varied", R3.S5_WRITEUP)
    note("supplementary writeups corrected (estimator count, degeneracy disclosure, "
         "pooled-epoch caveat, exact p)")

    # new supplementary figures S6 (harmonic ladder) and S7 (SWA ROC)
    end = d.find("No capacitive feature other than respiratory k varied")
    img_t = body[next(i for i in range(end, 0, -1) if body[i].findall(".//" + W + "drawing"))]
    supp = [d.para(""),
            d.image_para(FIG_HARMONIC_LADDER, img_t), d.para(R3.S6_CAPTION), d.para(R3.S6_WRITEUP),
            d.para(""),
            d.image_para(FIG_SWA_ROC, img_t), d.para(R3.S7_CAPTION), d.para(R3.S7_WRITEUP)]
    for off, el in enumerate(supp):
        body.insert(end + 1 + off, el)
    note("added Figure S6 (harmonic-ladder validation) and Figure S7 (SEC vs EEG SWA ROC)")

    # ------------------------------------------------------------- 11. final polish
    # "Fig. 1" is the only figure cited in the old style; normalise so every
    # reference and caption in the paper reads "Figure N".
    nf = 0
    for t in body.iter(W + "t"):
        if t.text and "Fig." in t.text:
            new = re.sub(r"\bFig\. (\d)", r"Figure \1", t.text)
            if new != t.text:
                t.text = new
                nf += 1
    note("normalised %d 'Fig. 1' references to 'Figure 1'" % nf)

    # 5.1 subheading asserted the conclusion; make it descriptive and consistent
    # with the aggregate/within-night vocabulary used in 4.2.
    d.set_text("Accurate night-level respiratory and cardiac rates",
               "Aggregate respiratory and cardiac rates")

    # supplementary captions still used the retired regime names
    d.patch("Per-sleep-stage accuracy", "Night-level estimator error by PSG-scored sleep stage",
            "Per-epoch estimator error by PSG-scored sleep stage")
    d.patch("within-session correlation per night against a temporal shuffle",
            "Figure S3. Epoch-level regime:", "Figure S3. Within-night regime:")
    note("5.1 heading and supplementary captions aligned with the corrected regime names")

    # stray empty Heading2 left over from an earlier edit pass
    for i in range(len(body) - 1, -1, -1):
        el = body[i]
        st = el.find(".//" + W + "pStyle")
        if (st is not None and st.get(W + "val") == "Heading2"
                and not d.text_of(el).strip() and not el.findall(".//" + W + "drawing")):
            body.remove(el)
            note("removed a stray empty Heading2 paragraph")

    d.save(DST)
    print("\n%d edit operations applied" % d.n_patch)
    print("wrote %s" % DST)
    Path("writeup/edits/REVIEW_CHANGELOG.md").write_text(
        "# Review edits applied to CAP_sleep_mask_manuscript_main_review.docx\n\n"
        + "\n".join("- " + m for m in log) + "\n", encoding="utf-8")
    print("wrote writeup/edits/REVIEW_CHANGELOG.md")


if __name__ == "__main__":
    main()
