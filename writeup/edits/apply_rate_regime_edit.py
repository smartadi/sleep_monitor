"""Apply the two-regime rate-detection restructure to the canonical manuscript.

Rewrites section 4.2 to report both bands in both regimes with one table and two
figures, moves the per-session tables, estimator sweep, and per-stage breakdown
into a new Supplementary section (each with its own writeup), adds the k-versus-age
analysis, and renumbers figures 6-8 down to 5-7.

Content source: writeup/edits/rate_two_regime_section_draft.md
Run from the repo root. Back up the .docx first -- it is git-ignored.
"""

import copy
import re
import shutil
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

DOCX = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
IMG_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

# Section 4.2 currently occupies body[SEC42_START:SEC42_END] (verified before running).
SEC42_START, SEC42_END = 83, 105
IDX_HEATMAP_IMG = 88      # Figure 3 image paragraph -> Supplementary S1
IDX_BLANDALTMAN_IMG = 90  # Figure 4 image paragraph -> stays, becomes Figure 3
IDX_PERSTAGE_IMG = 103    # Figure 5 image paragraph -> Supplementary S2

NEW_FIGURES = {
    "k_age": Path("analysis/rates/outputs/fig_k_vs_age_3panel.png"),
    "tracking": Path("writeup/figures/mask_rate_detection/fig19_tracking_r_bars.png"),
    "calibration": Path("analysis/rates/outputs/fig_calibration_requirement.png"),
    "age_features": Path("analysis/rates/outputs/fig_age_features.png"),
}


# --------------------------------------------------------------------------- text

SEC_42 = [
    ("Heading2", "4.2 Rate detection"),
    ("", "The sensor was evaluated against PSG in two regimes that differ in what is being "
     "measured and in how it is validated. The night-level regime asks whether the mask recovers "
     "a subject's average respiratory and cardiac rate over a recording — the quantity relevant "
     "to overnight screening and night-to-night trending. The epoch-level regime asks whether it "
     "follows the rate as it varies within a night, on the same 60-second windows (30-second "
     "step) used throughout. The two are reported together because they are served by different "
     "estimators and reach different conclusions, and because the distinction determines what "
     "the device can be used for."),
    ("", "For the night-level regime we used the estimator that minimised error in each band: "
     "the spectral peak (Welch PSD) for respiration and loose prominence-based peak counting for "
     "the cardiac band, both on the CLE−CRE differential, rescaled by a per-session factor k "
     "(§4.2.1) and smoothed with a causal three-epoch median filter. For the epoch-level regime "
     "we used a responsive detector combining loose peak counting with Hilbert instantaneous "
     "frequency across five channels, which sacrifices some accuracy for temporal resolution. "
     "Multi-channel SQI-weighted and agreement-gated fusion was evaluated but did not improve "
     "accuracy over the single differential channel in either band (ΔMAE < 0.1; Figure S1), so "
     "the single-channel estimator is the operational pipeline."),
    ("Heading3", "Night-level accuracy"),
    ("", "Respiratory rate was recovered with a per-session median MAE of 0.91 br/min "
     "(IQR 0.81–1.19, range 0.56–2.26) and a pooled MAE of 1.09 br/min (bias −0.3 br/min, 95% "
     "limits of agreement −4.7 to +4.2). Cardiac rate was recovered with a per-session median "
     "MAE of 3.41 BPM (IQR 3.06–8.38) on the best single channel and a pooled MAE of 3.91 BPM "
     "(bias −0.6 BPM, limits of agreement −24.1 to +22.9). The cardiac error distribution is "
     "right-skewed: three of twelve nights exceeded 8 BPM (S2N1, S6N1, S6N2), and excluding the "
     "anomalously coupled S6 subject the median was ≈3.0 BPM. The wide cardiac limits of "
     "agreement reflect epoch-level spread; the validated quantity in this regime is the "
     "per-session mean. Per-session values for all twelve nights are given in Table S1 and the "
     "agreement plots in Figure 3."),
    ("Heading3", "Epoch-level accuracy"),
    ("", "Within a night, neither band's rate variation was recovered (Table 3). Respiratory "
     "estimates correlated with the reference at a median within-session r = +0.06 (Wilcoxon "
     "p = 0.68; positive in 8 of 12 nights; r = +0.02 on detrended fluctuations), and cardiac "
     "estimates at r = −0.19 (p = 0.34; positive in 5 of 12; detrended r = −0.15). This is not a "
     "limitation specific to the mask: two physically independent PSG respiratory sensors — nasal "
     "airflow and the respiratory inductance belt sum — agree with each other at only r = 0.47 "
     "(IQR 0.33–0.67) on the same epochs, which bounds what any respiratory sensor can achieve "
     "at this time resolution."),
    ("", "The two bands fall short of that bound for different reasons, and the ratio of "
     "estimation error to the physiological variation available to be tracked separates them. "
     "Within-session respiratory rate varies with a median standard deviation of only "
     "1.14 br/min, which is comparable to the epoch-level error of the best estimator "
     "(ratio 0.77) and smaller than that of the responsive detector (1.04). For respiration "
     "there is very little variation to resolve relative to the noise floor, and a near-constant "
     "estimate is consequently the more accurate one — the spectral estimator's lower error "
     "(0.91 versus 1.33 br/min) is a direct consequence of its not tracking. The cardiac band is "
     "different: within-session heart rate varies with a median standard deviation of 5.26 BPM, "
     "well above the epoch-level error (ratio 0.78), so the variation is in principle resolvable "
     "and is nonetheless not recovered. The mechanism for this is discussed in §5.3. An "
     "exhaustive comparison of estimators and channels in both regimes, including CWT-ridge and "
     "STFT–Viterbi trackers, is given in Figure S1 and Figure S3."),
    ("", "Table 3. Rate detection in both regimes. Per-session medians across 12 nights "
     "(interquartile range in brackets). Within-session r is the correlation between estimated "
     "and reference rate across epochs within each night."),
]

TABLE_3 = [
    ["", "Respiratory", "Cardiac"],
    ["Night-level regime", "", ""],
    ["MAE, per-session median", "0.91 br/min [0.81–1.19]", "3.41 BPM [3.06–8.38]"],
    ["MAE, pooled", "1.09 br/min", "3.91 BPM"],
    ["Bias (95% limits of agreement)", "−0.3 (−4.7, +4.2) br/min", "−0.6 (−24.1, +22.9) BPM"],
    ["Calibration factor k", "0.97 [0.91–1.04]", "1.95 [0.94–2.24]"],
    ["Epoch-level regime", "", ""],
    ["MAE, per-session median", "1.33 br/min [1.11–1.51]", "3.65 BPM [2.97–7.42]"],
    ["Within-session r", "+0.06 (p = 0.68)", "−0.19 (p = 0.34)"],
    ["Nights with r > 0", "8 / 12", "5 / 12"],
    ["Reference rate SD within night", "1.14 br/min", "5.26 BPM"],
    ["Error-to-variation ratio", "0.77", "0.78"],
    ["Independent-sensor bound", "r = 0.47", "not available"],
]

FIG3_CAPTION = ("Figure 3. Bland–Altman agreement for respiratory (left) and cardiac (right) "
                "rate in the night-level regime. Each point is one analysis epoch; the solid "
                "line is the bias and the dashed lines the 95% limits of agreement.")

SEC_421 = [
    ("Heading3", "4.2.1 The calibration factor k"),
    ("", "Rate estimates in both bands are rescaled by a per-session factor k, the median ratio "
     "of the raw estimator output to the reference rate over 50 randomly selected one-minute "
     "windows. k is not a free parameter fitted to improve accuracy; it counts how many "
     "capacitive deflections the sensor produces per physiological event, and it behaves as a "
     "stable property of the subject."),
    ("", "k is a waveform-morphology count. Averaging the capacitive signal triggered on ECG "
     "R-peaks resolves the per-beat pulse shape directly, and counting capacitive peaks against "
     "ECG beats gives 1.70–2.43 peaks per heartbeat (median 2.02 on CRE, 1.92 on CLE−CRE, across "
     "the nine sessions with usable peak detection). This brackets the fitted cardiac k of 1.95 "
     "and identifies its origin: the capacitive cardiac pulse is biphasic, contributing a "
     "systolic peak and a dicrotic notch to each cardiac cycle, so a peak-counting estimator "
     "reports approximately twice the true rate. The correspondence is population-level rather "
     "than exact per subject (r = 0.50 between measured peaks-per-beat and fitted k), and one "
     "session was excluded because cardiac peak detection failed outright. The respiratory k of "
     "≈0.97 reflects the simpler coupling: each breath produces one dominant temple displacement."),
    ("", "k is reproducible within a subject. Across each subject's two nights the absolute "
     "change in k had a median of 0.013 for respiration (maximum 0.047) and 0.151 for the "
     "cardiac band (maximum 0.406). Respiratory k is therefore effectively a subject-level "
     "constant, and cardiac k is stable apart from the S6 coupling anomaly. Diagnostic estimates "
     "from 50 random windows agreed with whole-night values to within 0.04."),
    ("", "k, age, and the calibration a deployment would require. Because k reflects the "
     "mechanical coupling between physiology and sensor, we asked whether it varies with subject "
     "age (Figure 4). Respiratory k declined across the six subjects, from 1.04 in the youngest "
     "(25 years) to 0.91 in the oldest (66 years) — Spearman ρ = −0.83, uncorrected p = 0.042. "
     "The direction of this relationship was stable: dropping each subject in turn left ρ "
     "between −1.00 and −0.70. Cardiac k showed no such relationship (ρ = +0.37, p = 0.47), and "
     "unlike the respiratory case the cardiac result was not stable under the same test — "
     "leave-one-out ρ ranged from −0.10 to +0.90 — so these data neither establish nor exclude "
     "an age dependence for cardiac k. Neither factor was associated with PSQI (p > 0.19), and "
     "no other capacitive signal feature we examined varied with age (Figure S5)."),
    ("", "We tested whether the respiratory relationship is strong enough to replace "
     "per-subject calibration, by fitting k from age on five subjects and predicting the sixth. "
     "Leave-one-subject-out, an age-based prior predicted respiratory k with a mean absolute "
     "error of 0.020, against 0.056 for no calibration (k = 1.0) and 0.056 for the best constant "
     "prior (the population mean). Age therefore carries information about respiratory k that a "
     "constant prior does not. The same test on cardiac k gave the opposite result — an age "
     "prior (0.387) was worse than the population mean (0.305) — consistent with cardiac k being "
     "set by a fixed pulse morphology rather than by an age-modulated variable. Translated into "
     "rate error, per-session calibration gives 0.94 br/min for respiration, a fixed k = 1.0 "
     "costs only 0.20–0.28 br/min more, and cardiac rate requires whole-night self-calibration "
     "or a population prior (3.36 versus 4.56 BPM; Figure S4)."),
    ("", "Two cautions apply to the age result. With six subjects, age is perfectly confounded "
     "with subject identity, so the association may reflect any subject characteristic that "
     "covaries with age in this sample — chest-wall compliance and respiratory displacement "
     "morphology are the plausible mechanisms, but these data cannot isolate them, and both age "
     "extremes in this cohort were male. And the respiratory correlation does not survive "
     "correction for the four correlations tested (each factor against age and PSQI; Bonferroni "
     "p ≈ 0.17). We therefore report the age relationship as a calibration result validated on "
     "held-out subjects, and as an exploratory physiological observation motivating a larger "
     "cohort."),
]

FIG4_CAPTION = ("Figure 4. Calibration factor k and subject age. (A) Respiratory k per subject "
                "(mean of two nights; bars span the two nights; circles male, squares female) "
                "against age, with least-squares fit. k falls from 1.04 to 0.91 across the age "
                "range; the leave-one-out range of ρ is given above the panel. (B) Cardiac k "
                "against age. Values cluster near 1.95 with no stable trend; S6 is a coupling "
                "outlier (k = 1.14) whose inclusion or exclusion reverses the sign of ρ. "
                "(C) Leave-one-subject-out prediction of respiratory k: absolute error per "
                "held-out subject using an age prior fitted on the other five, against using no "
                "calibration (k = 1.0).")

TABLE_S1_HEADER = ["Session", "Resp k", "Resp MAE\nnight", "Resp MAE\nepoch", "Resp r",
                   "Card k", "Card MAE\nnight", "Card MAE\nepoch", "Card r"]

SUPP_INTRO = [
    ("Heading1", "Supplementary material"),
    ("Heading2", "S1. Rate detection: per-session results, estimator comparison, and regime statistics"),
    ("", "Table S1. Per-session rate accuracy in both regimes. Night-level MAE is the "
     "k-calibrated single-channel estimator; epoch-level MAE and within-session r are the "
     "responsive multi-channel detector. Sessions are labelled SxNy for subject x, night y."),
]

SUPP_FIGURES = [
    ("heatmap_moved",
     "Figure S1. Estimator and channel comparison. Mean absolute error across six base "
     "estimators and four channels for both bands, k-scaled, with per-session interquartile "
     "range.",
     "The spectral estimator dominated the respiratory band on every channel and the loose "
     "peak-counting estimator dominated the cardiac band; no channel was reliably better than "
     "the CLE−CRE differential for respiration (oracle-over-channels 1.08 versus 1.09 br/min "
     "pooled), whereas the cardiac band did show channel-dependent headroom, which motivated the "
     "multi-channel detector used in the epoch-level regime. Multi-channel fusion nonetheless "
     "changed per-session median MAE by less than 0.1 in both bands (respiratory 0.95 → "
     "0.94 br/min; cardiac 3.42 → 3.36 BPM), which is why the single-channel estimator is "
     "reported as the operational pipeline in §4.2."),
    ("perstage_moved",
     "Figure S2. Per-sleep-stage accuracy. Night-level estimator error by PSG-scored sleep "
     "stage, for both bands.",
     "Respiratory error was highest in REM (≈2.31 br/min) and cardiac error highest in Wake "
     "(≈4.73 BPM), consistent with the irregular breathing of REM and the greater heart-rate "
     "variability of wakefulness. These comparisons pool epochs across sessions and were not "
     "tested for significance given the six-subject sample; they are reported as descriptive "
     "context for the aggregate errors in Table 3."),
    ("tracking",
     "Figure S3. Epoch-level regime: within-session correlation per night against a temporal "
     "shuffle null. Bars show the within-session Pearson r for the responsive detector and the "
     "spectral estimator in each band; the grey band is the 5th–95th percentile of a "
     "200-iteration temporal shuffle null.",
     "An identical battery was applied to both bands: six base estimators, multi-channel mean "
     "fusion, CWT-ridge tracking, and STFT–Viterbi tracking, each with and without k-calibration "
     "and smoothing. No configuration produced a within-session correlation distinguishable from "
     "the shuffle null in either band — 4 of 12 nights exceeded the null 95th percentile for "
     "respiration and 3 of 12 for the cardiac band, close to what the null itself produces. For "
     "scale, the two physically independent PSG respiratory sensors agree with each other at "
     "r = 0.47 (0.27 on detrended fluctuations) on these same epochs, so the achievable target "
     "in this regime is modest even for a contact sensor. The exhaustiveness of this battery is "
     "what supports treating the epoch-level result as a property of the signal rather than of a "
     "particular estimator."),
    ("calibration",
     "Figure S4. Calibration requirement. Rate error under three calibration strategies — "
     "per-session k estimated over the whole night, a fixed population k, and k estimated from "
     "the first ten minutes of recording — across the estimator pipelines evaluated, for both "
     "bands.",
     "Respiration is close to calibration-free: replacing per-session k with a fixed population "
     "value costs 0.15–0.28 br/min depending on pipeline (0.94–1.09 → 1.10–1.23 br/min), and a "
     "fixed k = 1.0 is nearly as good. Cardiac rate genuinely requires calibration: a population "
     "prior costs about 1.2 BPM (3.36 → 4.56 BPM), and a ten-minute warm-up calibration is worse "
     "than the population prior (10.1 BPM), because twenty epochs give a noisy and drifting k "
     "estimate. A deployment would therefore use a population prior or whole-night "
     "self-calibration for the cardiac band, not a short calibration period. This bounds the "
     "calibration limitation in §6."),
    ("age_features",
     "Figure S5. Capacitive signal features versus subject age. Per-subject night-to-night k "
     "reproducibility (|Δk| between the two nights) for both bands, and the modal spectral peak "
     "frequency in each band, against age. Spearman ρ and uncorrected p are shown above each "
     "panel.",
     "No capacitive feature other than respiratory k varied with age. The night-to-night "
     "reproducibility of k was itself age-independent in both bands (ρ = −0.03, p = 0.96), "
     "confirming that older subjects' k values are not merely noisier estimates; band SNR, DC "
     "drift, and accelerometer activity were likewise unrelated to age (|ρ| ≤ 0.37, p ≥ 0.47). "
     "The modal cardiac peak frequency showed the largest nominal association (ρ = +0.75, "
     "p = 0.084), but this variable takes only four distinct values across the six subjects and "
     "sits at or just above the 0.5 Hz lower edge of the cardiac analysis band for most of them; "
     "the same edge-pinning applies to the respiratory modal peak, which equals the 0.1 Hz band "
     "floor in five of six subjects. Neither is interpretable as a physiological age effect. "
     "This panel is included as a bounding null: it establishes that the respiratory-k finding "
     "is not one of many age correlations in these data."),
]


# ---------------------------------------------------------------------- xml helpers

class Doc:
    def __init__(self, path: Path):
        self.path = path
        with zipfile.ZipFile(path) as z:
            self.files = {n: z.read(n) for n in z.namelist()}
        self.root = etree.fromstring(self.files["word/document.xml"])
        self.body = self.root.find(W + "body")
        self.rels = etree.fromstring(self.files["word/_rels/document.xml.rels"])
        self._next_rel = 1 + max(
            int(m.group(1))
            for r in self.rels
            if (m := re.match(r"rId(\d+)$", r.get("Id") or ""))
        )
        self._next_docpr = 1000

    # -- paragraph construction, cloning run formatting from an existing body paragraph
    def _template_run(self):
        for p in self.body.iter(W + "p"):
            if p.find(W + "pPr/" + W + "pStyle") is None:
                r = p.find(W + "r")
                if r is not None and r.find(W + "t") is not None:
                    return r
        raise RuntimeError("no template run found")

    def para(self, text: str, style: str = "") -> etree._Element:
        p = etree.SubElement(etree.Element("tmp"), W + "p")
        if style:
            ppr = etree.SubElement(p, W + "pPr")
            etree.SubElement(ppr, W + "pStyle").set(W + "val", style)
        run = copy.deepcopy(self._template_run())
        for child in list(run):
            if etree.QName(child).localname == "t":
                run.remove(child)
        t = etree.SubElement(run, W + "t")
        t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        p.append(run)
        return p

    # -- image insertion, cloning the drawing wrapper of an existing figure
    def image_para(self, png: Path, template_img_para: etree._Element) -> etree._Element:
        data = png.read_bytes()
        name = f"word/media/{png.stem}_{self._next_rel}.png"
        self.files[name] = data

        rid = f"rId{self._next_rel}"
        self._next_rel += 1
        rel = etree.SubElement(self.rels, f"{{{RELS_NS}}}Relationship")
        rel.set("Id", rid)
        rel.set("Type", IMG_TYPE)
        rel.set("Target", name.replace("word/", ""))

        p = copy.deepcopy(template_img_para)
        blip = p.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
        blip.set(R + "embed", rid)

        # keep the template's width; scale height to this image's aspect ratio
        extent = p.find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent")
        cx = int(extent.get("cx"))
        w_px, h_px = Image.open(png).size
        cy = int(cx * h_px / w_px)
        extent.set("cy", str(cy))
        for ext in p.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
            ext.set("cx", str(cx))
            ext.set("cy", str(cy))

        docpr = p.find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr")
        self._next_docpr += 1
        docpr.set("id", str(self._next_docpr))
        docpr.set("name", png.stem)
        return p

    # -- tables, cloning formatting from an existing table
    def table(self, rows: list[list[str]], template_tbl: etree._Element) -> etree._Element:
        tbl = copy.deepcopy(template_tbl)
        template_tr = copy.deepcopy(tbl.find(W + "tr"))
        template_tc = copy.deepcopy(template_tr.find(W + "tc"))
        for tr in tbl.findall(W + "tr"):
            tbl.remove(tr)

        grid = tbl.find(W + "tblGrid")
        if grid is not None:
            for gc in grid.findall(W + "gridCol"):
                grid.remove(gc)
            for _ in rows[0]:
                etree.SubElement(grid, W + "gridCol").set(W + "w", str(9360 // len(rows[0])))

        for row in rows:
            tr = copy.deepcopy(template_tr)
            for tc in tr.findall(W + "tc"):
                tr.remove(tc)
            for cell in row:
                tc = copy.deepcopy(template_tc)
                for p in tc.findall(W + "p"):
                    tc.remove(p)
                for i, line in enumerate(cell.split("\n")):
                    tc.append(self.para(line))
                if not cell:
                    tc.append(self.para(""))
                tr.append(tc)
            tbl.append(tr)
        return tbl

    def save(self, path: Path) -> None:
        self.files["word/document.xml"] = etree.tostring(
            self.root, xml_declaration=True, encoding="UTF-8", standalone=True)
        self.files["word/_rels/document.xml.rels"] = etree.tostring(
            self.rels, xml_declaration=True, encoding="UTF-8", standalone=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in self.files.items():
                z.writestr(name, data)


# ---------------------------------------------------------------------------- main

def main() -> None:
    import pandas as pd

    doc = Doc(DOCX)
    body = doc.body

    heatmap_img = copy.deepcopy(body[IDX_HEATMAP_IMG])
    bland_img = copy.deepcopy(body[IDX_BLANDALTMAN_IMG])
    perstage_img = copy.deepcopy(body[IDX_PERSTAGE_IMG])
    template_tbl = body[81]  # Table 2

    # ---- build the replacement for section 4.2
    new = [doc.para(text, style) for style, text in SEC_42]
    new.append(doc.table(TABLE_3, template_tbl))
    new.append(doc.para(""))
    new.append(bland_img)
    new.append(doc.para(FIG3_CAPTION))
    new.extend(doc.para(text, style) for style, text in SEC_421)
    new.append(doc.image_para(NEW_FIGURES["k_age"], bland_img))
    new.append(doc.para(FIG4_CAPTION))

    for idx in range(SEC42_END - 1, SEC42_START - 1, -1):
        body.remove(body[idx])
    for offset, el in enumerate(new):
        body.insert(SEC42_START + offset, el)
    print(f"section 4.2 rebuilt: {SEC42_END - SEC42_START} elements -> {len(new)}")

    # ---- renumber figures 6,7,8,9 -> 5,6,7,8
    # Must be a single pass: applying re.sub once per mapping would rewrite an
    # already-rewritten number (8 -> 7 -> 6 -> 5) regardless of the order used.
    n_renum = 0
    for t in body.iter(W + "t"):
        if not t.text or "Figure" not in t.text:
            continue
        original = t.text
        t.text = re.sub(r"Figure ([6-9])\b",
                        lambda m: f"Figure {int(m.group(1)) - 1}", t.text)
        if t.text != original:
            n_renum += 1
    print(f"renumbered figure references in {n_renum} runs")

    # ---- supplementary section, appended before the final sectPr
    supp = [doc.para(text, style) for style, text in SUPP_INTRO]

    s1 = pd.read_csv("reports/rates/mask/table_s1_per_session_two_regime.csv")
    supp.append(doc.table([TABLE_S1_HEADER] + s1.astype(str).values.tolist(), template_tbl))
    supp.append(doc.para(""))

    images = {"heatmap_moved": heatmap_img, "perstage_moved": perstage_img}
    for key, caption, writeup in SUPP_FIGURES:
        img = images.get(key) or doc.image_para(NEW_FIGURES[key], bland_img)
        supp.extend([img, doc.para(caption), doc.para(writeup), doc.para("")])

    sect_pr = body.find(W + "sectPr")
    anchor = list(body).index(sect_pr) if sect_pr is not None else len(body)
    for offset, el in enumerate(supp):
        body.insert(anchor + offset, el)
    print(f"supplementary section added: {len(supp)} elements")

    doc.save(DOCX)
    print(f"wrote {DOCX}")


if __name__ == "__main__":
    main()
