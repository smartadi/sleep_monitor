"""Bound the within-night negative to the configurations actually tested.

The subheading asserted a property of the mask -- "What the mask does not
recover: variation within the night" -- where the evidence supports the
narrower claim that nothing we tested recovers it. The battery in
reports/rates/mask/symmetric_tracking_battery.csv:

    respiratory, detB   median r +0.06, 8/12 positive, Wilcoxon p = 0.68
    cardiac, detB       median r -0.19, 5/12 positive, Wilcoxon p = 0.34
    best single night   +0.32 respiratory, +0.53 cardiac

so individual nights do scatter well above zero; it is the per-night
distribution that is indistinguishable from zero. The reference ceiling -- two
independent PSG sensors measuring the same breathing -- is r = 0.33 to 0.71,
so the target is real and reachable in principle. Adding these three facts
turns an assertion into a bounded result, and lets a reader see that a single
promising session is what sampling noise looks like at n = 12.

Run from the repo root:  python writeup/edits/apply_42_tracking_framing.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_42FRAMING_20260817.docx")
BATTERY = Path("reports/rates/mask/symmetric_tracking_battery.csv")
CEILING = Path("reports/rates/mask/symmetric_tracking_ceiling.csv")

HEADING_OLD = "What the mask does not recover: variation within the night"
HEADING_NEW = "Variation within the night, which no configuration tested recovers"


def main():
    b = pd.read_csv(BATTERY)
    c = pd.read_csv(CEILING)
    stat = {}
    for band in ("resp", "card"):
        v = b[b.band == band].r_detB.dropna().values
        stat[band] = dict(med=float(np.median(v)), best=float(v.max()),
                          npos=int((v > 0).sum()), n=len(v),
                          p=float(stats.wilcoxon(v).pvalue))
    ceil_lo = float(c.r_flow_ripsum.min())
    ceil_hi = float(c.r_flow_ripsum.max())

    bounded = (
        "The negative is bounded by what we tested rather than by principle, and two facts fix "
        "its scale. First, single nights do scatter well above zero: across the tracking battery "
        "the best night reaches r = %+.2f for respiration and r = %+.2f for the cardiac band, so "
        "an encouraging session is what sampling noise looks like at twelve nights. It is the "
        "distribution over nights that is indistinguishable from zero — median r = %+.2f "
        "(%d of %d nights positive, Wilcoxon p = %.2f) for respiration and %+.2f (%d of %d, "
        "p = %.2f) for the cardiac band. Second, the target is real and reachable: two "
        "independent PSG sensors measuring the same breathing agree on within-night variation at "
        "r = %.2f to %.2f, so the ceiling is not near zero and the mask is not being asked for "
        "something the reference cannot itself demonstrate."
        % (stat["resp"]["best"], stat["card"]["best"],
           stat["resp"]["med"], stat["resp"]["npos"], stat["resp"]["n"], stat["resp"]["p"],
           stat["card"]["med"], stat["card"]["npos"], stat["card"]["n"], stat["card"]["p"],
           ceil_lo, ceil_hi)
    )

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    d.set_text(HEADING_OLD, HEADING_NEW)
    print("  4.2 subheading no longer asserts a property of the mask")

    # 5.2 makes the same assertion in its heading and its inference
    d.patch("The mask does not follow either rate as it changes during the night",
            "Because the battery was exhaustive, we treat this as a property of the signal at "
            "this window length rather than of a particular estimator.",
            "The battery was broad rather than exhaustive, so we report this as a property of "
            "every configuration tested at this window length; the best single night reached "
            "r = %+.2f, against a two-sensor reference ceiling of %.2f to %.2f."
            % (max(stat["resp"]["best"], stat["card"]["best"]), ceil_lo, ceil_hi))
    print("  5.2 'exhaustive' softened to 'broad', with the ceiling stated")

    anchor = d.find("Agreement on the level of a rate is not the same")
    d.body.insert(anchor + 2, d.para(bounded))
    print("  4.2: bounded-negative paragraph added")

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## 4.2 within-night negative, bounded (2026-08-17)\n\n"
        + "- subheading attributes the negative to the configurations tested, not to the mask\n"
        + "- best single night r = %+.2f resp / %+.2f card added, against a two-sensor ceiling "
          "of %.2f-%.2f\n" % (stat["resp"]["best"], stat["card"]["best"], ceil_lo, ceil_hi)
        + "- 5.2 'the battery was exhaustive' -> 'broad rather than exhaustive'\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
