"""Add the per-session integrated capacitance imbalance to 4.1 and the supplement.

A per-session integral of the *signed* imbalance is not reportable: the marker is
referenced to each session's own mean -- deliberately, because the absolute
CLE-CRE level is set by mask placement -- so its positive and negative parts
cancel and the asymmetry index sits within +-0.09 for eleven of twelve nights.
What survives the reference is the magnitude burden, and that separates the
recordings 46-fold.

Adds a paragraph to 4.1, a supplementary section S3 with Figure S10, and
retitles 4.1 to match its new order (physiological content first).

Numbers come from analysis/mean_value/imbalance_burden.py ->
reports/mean_value/imbalance_burden.csv; run that first.

Run from the repo root:  python writeup/edits/apply_imbalance_burden.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_BURDEN_20260817.docx")
CSV = Path("reports/mean_value/imbalance_burden.csv")
FIG = Path("writeup/figures/mean_value/imbalance_burden.png")


def main():
    if not CSV.exists() or not FIG.exists():
        raise SystemExit("run analysis/mean_value/imbalance_burden.py first")
    d0 = pd.read_csv(CSV)
    lo, hi = d0.integral_abs_fFh.min(), d0.integral_abs_fFh.max()
    med = float(np.median(d0.integral_abs_fFh))
    mlo, mhi = d0.mean_abs_fF.min(), d0.mean_abs_fF.max()
    alo, ahi = d0.asymmetry.min(), d0.asymmetry.max()
    within = d0.groupby("subject").integral_abs_fFh.agg(lambda s: s.max() / s.min())
    no_s6 = d0[d0.subject != "OS006"].integral_abs_fFh
    n_small = int((d0.asymmetry.abs() <= 0.09).sum())

    para = (
        "Integrated over a night, the imbalance separates the recordings sharply. The "
        "time-average magnitude of the marker runs from %.1f to %.1f fF across the twelve "
        "nights and its integral from %.0f to %.0f fF·h, a %.0f-fold spread (Figure S10). That "
        "spread is not a subject characteristic: within a subject the two nights differ by %.1f- "
        "to %.1f-fold, so the burden is a property of a night — most plausibly of that night's "
        "coupling and posture history — rather than of the person. The signed integral carries "
        "no comparable information. Because the marker is referenced to each session's own mean, "
        "its positive and negative parts cancel by construction: the asymmetry index spans only "
        "%+.2f to %+.2f, with %d of twelve nights inside ±0.09. We therefore report the magnitude "
        "burden and the left-dominant time fraction, and make no claim of a net lateralization."
        % (mlo, mhi, lo, hi, hi / lo, within.min(), within.max(), alo, ahi, n_small)
    )

    caption = (
        "Figure S10. Integrated capacitance imbalance per recording. (A) The imbalance burden, "
        "the integral of the marker's magnitude over the night's motion-free epochs, on a log "
        "scale; colour marks the subject. (B) The same quantity with each subject's two nights "
        "joined — a subject trait would give flat lines. (C) The asymmetry index, the difference "
        "between the positive and negative parts of the integral over their sum, which is near "
        "zero by construction because the marker is referenced to each session's own mean. It is "
        "shown so that the magnitude result in (A) is not read as net lateralization."
    )

    writeup = (
        "The burden is dominated by S6, whose two nights carry %.0f and %.0f fF·h against a "
        "cohort median of %.0f fF·h; S6N2 is the coupling anomaly that also produces the "
        "outlying cardiac k (§4.2.1). Excluding that subject the burden still spans %.0f to %.0f "
        "fF·h, so the range is not one outlier. The left-dominant time fraction stays within "
        "%.2f to %.2f across the same nights, which is the sense in which the direction of the "
        "imbalance is dynamic while its size is not comparable between recordings."
        % (d0[d0.subject == "OS006"].integral_abs_fFh.min(),
           d0[d0.subject == "OS006"].integral_abs_fFh.max(), med,
           no_s6.min(), no_s6.max(),
           d0.frac_left_dominant.min(), d0.frac_left_dominant.max())
    )

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    d.set_text("4.1 The overnight sensor record and its physiological content",
               "4.1 Physiological content of the sensor record")
    print("  4.1 retitled for the new order")

    anchor = d.find("Head position does not explain")
    body.insert(anchor + 1, d.para(para))
    print("  4.1: imbalance-burden paragraph added")

    img_template = body[next(i for i in range(len(body) - 1, 0, -1)
                             if body[i].findall(".//" + W + "drawing"))]
    tail = len(body) - 1
    while tail > 0 and not d.text_of(body[tail]).strip():
        tail -= 1
    block = [
        d.para(""),
        d.para("S3. Integrated capacitance imbalance", "Heading2"),
        d.image_para(FIG, img_template),
        d.para(caption),
        d.para(writeup),
    ]
    for off, el in enumerate(block):
        body.insert(tail + 1 + off, el)
    print("  supplementary S3 added with Figure S10")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Integrated capacitance imbalance (2026-08-17)\n\n"
        + "- 4.1: burden %.0f-%.0f fF*h across nights (%.0fx), not a subject trait "
          "(%.1f-%.1fx within subject)\n" % (lo, hi, hi / lo, within.min(), within.max())
        + "- the signed integral is reported as the null it is: asymmetry %+.2f to %+.2f\n"
          % (alo, ahi)
        + "- supplementary S3 / Figure S10\n", encoding="utf-8")


if __name__ == "__main__":
    main()
