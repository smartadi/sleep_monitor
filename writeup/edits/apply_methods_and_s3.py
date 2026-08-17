"""Two structural items.

A. A Methods section for the cortical-event analyses. 4.4 and 4.5 are the only
   analyses in the paper that carry their method inside Results. Both run the
   same experiment at two different cortical events, and describing that
   machinery once makes the shared design visible -- which is what carries the
   mechanical-not-electrical argument. New 3.7; statistics moves to 3.8.

B. Figure S3 is retired. It plots the within-night correlation against a
   circular-shift null for "the responsive detector and the spectral estimator";
   the spectral estimator is now reported only as a baseline, and its caption
   also misstates the null band as the 5th-95th percentile where the code uses
   2.5/97.5. The result it carries is better given as a row of Table 3, where
   the other within-night numbers live: 0 of 12 respiratory nights and 1 of 12
   cardiac nights exceed the null band. Supplementary figures S4 onward shift
   down by one.

Run from the repo root:  python writeup/edits/apply_methods_and_s3.py
"""

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_METHODS37_20260817.docx")

M37 = (
    "Two analyses ask what the capacitive signal does at a discrete, precisely timed cortical "
    "event, and both use the same procedure at two different events. Sleep spindles were taken "
    "from the technologist's per-event annotations (start, end, duration and intra-spindle "
    "frequency), aligned to the capacitive recording with the same wall-clock offset applied to "
    "the sleep staging. Delta-burst onsets were detected on the contact EEG as sustained rises of "
    "the 0.5–4 Hz envelope above a per-night threshold, each event timestamped at the rising edge "
    "of the burst, retaining only onsets that emerged from a quiescent, motion-free baseline in "
    "NREM sleep."
)

M37_B = (
    "For both event types the capacitive power was averaged in a window centered on the event, "
    "against baseline windows drawn from the same recording and matched in number to the events "
    "— spindle-free N2 windows for spindles, motion-free NREM windows for delta-burst onsets — "
    "so that the null carries the same averaging and the same stage composition as the signal. "
    "Envelopes were computed with a strictly causal estimator, a forward-only Butterworth filter "
    "followed by a trailing root-mean-square, so that no post-event power can leak backward in "
    "time and produce an apparent precursor; a zero-phase estimator is shown alongside where that "
    "distinction matters. Spindle analysis used the low band (0–3 Hz) and the sigma band "
    "(11–16 Hz), the latter as the test of electrical pickup; delta-burst analysis used three low "
    "bands (0–0.5, 0.5–1 and 1–3 Hz) on the two temple channels and the forehead channel. "
    "Averaging was performed within each subject first and then across subjects, so that a "
    "recording with many events cannot dominate. For the delta-burst onsets a further control "
    "restricted the analysis to onsets with no scored arousal, cortical or autonomic, within "
    "10 seconds either side; that comparison is reported in section 4.5."
)

TABLE3_NULL_NOTE = (
    " The final row counts recordings whose within-night correlation exceeds a circular-shift "
    "null: the estimate is shifted against the reference 200 times per recording, which destroys "
    "the time alignment while preserving each series' own distribution and autocorrelation, and "
    "the 95% band of the pooled result is −0.12 to +0.11 for respiration and −0.21 to +0.17 for "
    "the cardiac band."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    # ------------------------------------------------------------------ A. 3.7
    d.set_text("3.7 Statistical methods", "3.8 Statistical methods")
    at = d.find("3.8 Statistical methods")
    for off, el in enumerate([d.para("3.7 Cortical events and event-triggered analysis",
                                     "Heading2"),
                              d.para(M37), d.para(M37_B)]):
        body.insert(at + off, el)
    print("  3.7 added, statistics renumbered to 3.8")

    # trim the method sentences now duplicated in Results
    d.patch("Sleep spindles—11–16 Hz (sigma) thalamocortical bursts",
            "Technologist-scored spindle annotations (per-event start, end, duration, and "
            "intra-spindle frequency) were aligned to the capacitive recording using the same "
            "wall-clock offset applied to sleep staging; across",
            "Across")
    d.patch("Sleep spindles—11–16 Hz (sigma) thalamocortical bursts",
            "For each spindle we averaged the capacitive power in a window centered on the "
            "spindle center, relative to matched spindle-free N2 baseline windows.", "")
    d.patch("Cortical slow-wave (delta) activity provides a second discrete",
            "On the contact EEG we detected delta-burst onsets as sustained rises of the "
            "0.5-4 Hz envelope above a per-night threshold, timestamping each event at the "
            "rising edge of the burst and retaining only onsets that emerged from a quiescent, "
            "motion-free baseline in NREM sleep", "The detection is described in §3.7 and yields")
    print("  4.4 and 4.5 trimmed of the method text now in 3.7")

    # ------------------------------------------------------------------ B. S3
    cap_i = d.find("Figure S3.")
    writeup_i = cap_i + 1
    img_i = cap_i - 1
    if body[img_i].find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip") is None:
        raise SystemExit("expected the Figure S3 image above its caption")
    for i in sorted([img_i, cap_i, writeup_i], reverse=True):
        body.remove(body[i])
    print("  Figure S3, its caption and its writeup removed")

    # supplementary S4.. shift down one, ascending so numbers cannot collide
    for old in range(4, 14):
        for pat, rep in ((r"Figures? S%d\b" % old, None), (r"and S%d\b" % old, None)):
            for t in d.root.iter(W + "t"):
                if not t.text:
                    continue
                t.text = re.sub(pat,
                                lambda m: m.group(0).replace("S%d" % old, "S%d" % (old - 1)),
                                t.text)
    print("  supplementary figures S4-S13 renumbered to S3-S12")

    # the result S3 carried, as a Table 3 row
    tbl3 = body[next(i for i in range(d.find("Table 3. Rate agreement"), len(body))
                     if body[i].tag == W + "tbl")]
    template = tbl3.findall(W + "tr")[-1]
    import copy
    row = copy.deepcopy(template)
    cells = row.findall(W + "tc")
    for cell, text in zip(cells, ["Recordings above the circular-shift null", "0 / 12", "1 / 12"]):
        for p in cell.findall(W + "p")[1:]:
            cell.remove(p)
        ts = cell.findall(".//" + W + "t")
        if ts:
            ts[0].text = text
            for extra in ts[1:]:
                extra.text = ""
    tbl3.append(row)
    d.patch("Table 3. Rate agreement, from the operational estimator",
            "tested across nights with a Wilcoxon signed-rank test on the twelve values.",
            "tested across nights with a Wilcoxon signed-rank test on the twelve values."
            + TABLE3_NULL_NOTE)
    print("  Table 3: circular-shift null row added, with the band defined in the caption")

    # the sentence that pointed at Figure S3
    d.patch("Agreement on the level of a rate is not the same",
            "distinguishable from a circular-shift null (Figure S3);",
            "distinguishable from a circular-shift null (Table 3);")
    print("  4.2 now points at Table 3 instead of the retired figure")

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
