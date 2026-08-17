"""Legitimate citation markers, the two orphan supplementary figures, and a
supplementary sentence that was quoting a retired pipeline.

1. Citation markers. Three are plain body text rather than superscript runs:
   "previous work.22" and "electrodes23,24" in section 2, and "(RIPSum)29" in
   3.3. The last one was a proper superscript until normalize_formatting's
   run-in-head handler collapsed the paragraph's runs; that helper is fixed in
   the same commit so it splits the first run instead of flattening all of them.

2. Figures S2 and S5 had captions and writeups but no reference from the main
   text. Both are cited now from the sentence that already makes the claim they
   support: per-stage accuracy in 5.4, and the age analysis in 4.2.1. Neither
   figure needed to be dropped -- the text was simply not pointing at them.

3. The Figure S1 writeup read

       Multi-channel fusion nonetheless changed per-session median MAE by less
       than 0.1 in both bands (respiratory 0.95 -> 0.94 br/min; cardiac
       3.42 -> 3.36 BPM)

   Those numbers are from the superseded phase-C analysis, and the respiratory
   pair is the *degenerate spectral estimator* -- reports/rates/rerun/
   estimator_table.csv gives resp/spectral epoch error 0.91 br/min with
   raw_sd exactly 0.000, the constant that 3.5 tells the reader not to treat as
   a measurement. Quoting it here as the pipeline's accuracy contradicts 3.5 and
   understates the real per-epoch error by half (1.79 br/min).

   The rerun contains no fused variant, so the fusion comparison cannot be
   restated from it at all. The sentence now reports the channel spread the
   rerun does support, and attributes the fusion result to the earlier
   multi-channel analysis without borrowing its numbers.

Run from the repo root:  python writeup/edits/apply_citations_and_s1.py
"""

import copy
import shutil
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_CITATIONS_20260817.docx")

S1_WRITEUP = (
    "The spectral estimator dominated the respiratory band on every channel and the loose "
    "peak-counting estimator dominated the cardiac band; no channel was reliably better than the "
    "CLE−CRE differential for respiration. Channel choice matters little in the respiratory band, "
    "where the operational estimator spans 1.79 to 1.95 br/min per epoch across the five "
    "channels, and more in the cardiac band, where it spans 3.41 to 4.39 BPM and the right temple "
    "is best. An earlier multi-channel analysis found that fusing channels changed per-epoch "
    "error by less than 0.1 in either band; that analysis predates the estimator rerun reported "
    "here and its figures are not quoted, but its conclusion is why the simpler single-channel "
    "estimator is the operational pipeline in §4.2."
)


def superscript(d, needle, before, marker):
    """Turn text already present into a superscript run, splitting one run."""
    el = d.body[d.find(needle)]
    runs = el.findall(W + "r")
    full = "".join(t.text or "" for r in runs for t in r.findall(W + "t"))
    at = full.index(before) + len(before)
    pos = 0
    for run in runs:
        t = run.find(W + "t")
        s = (t.text or "") if t is not None else ""
        lo, hi = pos, pos + len(s)
        pos = hi
        if t is None or not (lo < at <= hi):
            continue
        cut = at - lo
        if not s[cut:].startswith(marker):
            raise KeyError("expected %r after %r, found %r" % (marker, before, s[cut:cut + 8]))
        head, tail = s[:cut], s[cut + len(marker):]
        t.text = head
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

        sup = copy.deepcopy(run)
        rpr = sup.find(W + "rPr")
        if rpr is None:
            rpr = etree.SubElement(sup, W + "rPr")
            sup.remove(rpr)
            sup.insert(0, rpr)
        for old in rpr.findall(W + "vertAlign"):
            rpr.remove(old)
        etree.SubElement(rpr, W + "vertAlign").set(W + "val", "superscript")
        sup.find(W + "t").text = marker
        idx = list(el).index(run)
        el.insert(idx + 1, sup)

        if tail:
            rest = copy.deepcopy(run)
            rr = rest.find(W + "rPr")
            if rr is not None:
                for bad in rr.findall(W + "vertAlign"):
                    rr.remove(bad)
            rest.find(W + "t").text = tail
            rest.find(W + "t").set(
                "{http://www.w3.org/XML/1998/namespace}space", "preserve")
            el.insert(idx + 2, rest)
        d.n_patch += 1
        return
    raise RuntimeError("could not place %r" % marker)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ------------------------------------------------------- 1. citations
    superscript(d, "Regional ICP is the intended target",
                "characterised against invasive pressure in previous work.", "22")
    superscript(d, "Regional ICP is the intended target",
                "made of carbon nanotube-paper composite (CPC) electrodes", "23,24")
    superscript(d, "Rather than relying on a single PSG respiratory sensor",
                "respiratory inductance plethysmography sum (RIPSum)", "29")
    print("  3 citation markers converted to superscript runs")

    # ------------------------------------------------- 2. orphan S2 and S5
    d.patch("The mask is suitable for screening-level overnight",
            "with well-characterized calibration behavior and per-stage accuracy",
            "with well-characterized calibration behavior and per-stage accuracy (Figure S2)")
    print("  Figure S2 now cited from 5.4")

    d.patch("k is reproducible within a subject",
            "We examined whether it varies with subject age and found no relationship in either "
            "band that survives this sample size",
            "We examined whether it varies with subject age, and with any other capacitive "
            "feature we could form, and found no relationship in either band that survives this "
            "sample size (Figure S5)")
    print("  Figure S5 now cited from 4.2.1")

    # --------------------------------------- 3. the S1 writeup's stale numbers
    d.set_text("The spectral estimator dominated the respiratory band", S1_WRITEUP)
    print("  Figure S1 writeup: degenerate-estimator numbers removed, channel spread from "
          "the rerun stated")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Citations, orphan figures, and the S1 writeup (2026-08-17)\n\n"
        + "- superscript runs for references 22, 23,24 and 29\n"
        + "- Figures S2 and S5 cited from 5.4 and 4.2.1\n"
        + "- Figure S1 writeup no longer quotes the degenerate spectral estimator (0.95 br/min, "
          "raw_sd 0.000) as pipeline accuracy; channel spread 1.79-1.95 br/min and 3.41-4.39 BPM "
          "from the rerun instead\n", encoding="utf-8")


if __name__ == "__main__":
    main()
