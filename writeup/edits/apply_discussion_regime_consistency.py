"""Align the Discussion with the two-regime rate framing applied to section 4.2.

Three fixes:
  1. Section 5.1 asserted cardiac k is "age-invariant" on the strength of a null at
     n = 6. That null is not stable to leaving out a single subject, so it is restated
     as a null, and the leave-one-subject-out calibration result is added.
  2. Section 5.3 called the epoch-level result a "tracking failure" and explained the
     respiratory case by spectral resolution alone; it now also carries the
     error-to-variation argument from section 4.2.
  3. The calibration limitation gains the age-prior alternative.

Paragraphs are matched by a unique text prefix, not by index. Run from the repo root.
"""

import zipfile
from pathlib import Path

from lxml import etree

DOCX = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

REPLACEMENTS = [
    (
        "Accurate mean respiratory and cardiac rates",
        "Accurate night-level respiratory and cardiac rates",
    ),
    (
        "Two further observations support this morphological interpretation of k.",
        "Two further observations bear on this morphological interpretation of k. First, "
        "R-peak–triggered averaging of the capacitive pulse yields 1.70–2.43 peaks per heartbeat "
        "(median 2.02), matching the fitted cardiac k and directly confirming that k measures the "
        "number of capacitive peaks produced per cardiac cycle. Second, the two bands' "
        "calibration factors relate to subject age in opposite ways: respiratory k declines with "
        "age in a direction stable to leaving out any single subject, and an age-based prior "
        "predicts it better than the best constant prior on held-out subjects, whereas cardiac k "
        "shows no stable age relationship and is predicted better by a population constant than "
        "by age. The cardiac result is a null at six subjects and should not be read as "
        "established age-invariance — leave-one-out ρ ranges from −0.10 to +0.90 — but it is what "
        "a fixed pulse morphology would produce, while the respiratory result is what an "
        "age-modulated mechanical coupling would produce.",
    ),
    (
        "The cardiac tracking failure can be understood through the k factor.",
        "The absence of epoch-level cardiac tracking can be understood through the k factor. The "
        "consistent k ≈ 2 across subjects indicates that the capacitive cardiac waveform contains "
        "two inflection points per heartbeat — most likely the systolic and dicrotic pressure "
        "peaks. The dominant frequency of this biphasic waveform is determined by its stable "
        "morphology rather than by the instantaneous heart rate. When heart rate varies within a "
        "session the waveform stretches or compresses, but the peak-counting frequency remains "
        "governed by the persistent biphasic structure. Only changes large enough to alter the "
        "waveform's fundamental peak structure, rather than its period, would produce a frequency "
        "shift detectable by the estimators tested. This is why the cardiac band shows ample "
        "variation to resolve and none of it recovered.",
    ),
    (
        "The respiratory tracking limitation has a simpler explanation: spectral resolution.",
        "The respiratory case has a different basis, in which two factors compound. At 30-second "
        "windows with standard Welch parameters the frequency resolution (0.25 Hz) is comparable "
        "to the width of the entire respiratory band (0.4 Hz), which quantizes rate estimates; "
        "and the within-session variation being sought is itself small, a median standard "
        "deviation of 1.14 br/min against an epoch-level estimation error of comparable "
        "magnitude. A near-constant estimate is therefore the more accurate one, and attempts to "
        "increase resolution — longer windows, full-window periodograms with parabolic "
        "interpolation — trade temporal responsiveness for resolution without improving the "
        "within-session correlation.",
    ),
    (
        "Third, k-calibration requires a reference cardiac or respiratory rate for initial calibration.",
        "Third, k-calibration requires a reference cardiac or respiratory rate. While k is stable "
        "within and across nights once estimated, a deployment would need either a calibration "
        "period against a reference or a population-level prior. The near-unity respiratory k "
        "makes respiratory rate effectively calibration-free, and an age-based prior for "
        "respiratory k improves on a constant prior when validated on held-out subjects "
        "(§4.2.1); the cardiac band has no such shortcut, the self-supervised adaptive-k approach "
        "tested here failed for lack of a reliable k-free cardiac anchor, and a ten-minute "
        "warm-up calibration is worse than a population prior (Figure S4).",
    ),
]


def set_paragraph_text(para: etree._Element, text: str) -> None:
    """Replace a paragraph's text, keeping the formatting of its first run."""
    runs = para.findall(W + "r")
    keep = runs[0]
    for run in runs[1:]:
        para.remove(run)
    for t in keep.findall(W + "t"):
        keep.remove(t)
    t = etree.SubElement(keep, W + "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def main() -> None:
    with zipfile.ZipFile(DOCX) as z:
        files = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(files["word/document.xml"])
    body = root.find(W + "body")

    for prefix, new_text in REPLACEMENTS:
        hits = [
            p for p in body.iter(W + "p")
            if "".join(t.text or "" for t in p.iter(W + "t")).strip().startswith(prefix)
        ]
        if len(hits) != 1:
            raise SystemExit(f"expected 1 match, got {len(hits)}: {prefix[:60]!r}")
        set_paragraph_text(hits[0], new_text)
        print(f"patched: {prefix[:64]}")

    files["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    print(f"wrote {DOCX}")


if __name__ == "__main__":
    main()
