"""The three upstream defects left by the section-3.3-onward pass.

  in  : writeup/main/CAP_sleep_mask_manuscript_main.docx   (edited in place)
  bak : writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_UPSTREAM_20260813.docx

  1. Figure numbering.  Section 2's sensor schematic was cited as "Fig. 1",
     colliding with the Figure 1 that section 4.1 now uses for the overnight
     channel record.  The schematic is genuinely the document's first figure, so
     it takes Figure 1 and every later figure shifts up by one: channel record 2,
     SNR 3, Bland-Altman 4, k-vs-age 5, ridge tracker 6, ridge-by-stage 7,
     spindles 8, delta-burst 9.  Renumbering runs highest-first so that no
     intermediate state has two figures sharing a number.

  2. Table 1's column header said "Analysis epochs (60s)" and 3.1 called them
     one-minute epochs, contradicting the non-overlapping 30 s grid stated in
     3.3.  The counts themselves are already 30 s counts (S1N1: 7.95 h = 954),
     so only the labels were wrong.

  3. Section 3.1 carried the same study description twice, in two drafts of
     different quality.  The second block is removed and the one fact it held
     that the first lacked -- the IRB modification number -- is folded in.

Run from the repo root:  python writeup/edits/apply_upstream_three.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_review_edits import Doc, W  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_UPSTREAM_20260813.docx")

# (needle locating the paragraph, old text, new text) -- highest figure first
RENUMBER = [
    ("Figure 8. Delta-burst onsets evoke", "Figure 8. Delta-burst onsets", "Figure 9. Delta-burst onsets"),
    ("The response was present in every one of the six subjects", "Figure 8)", "Figure 9)"),
    ("The capacitive response therefore follows the cortical onset", "in Figure 8 is flat", "in Figure 9 is flat"),

    ("Figure 7. The capacitive response at sleep spindles", "Figure 7. The capacitive", "Figure 8. The capacitive"),
    ("Onset-triggered averaging shows that low-band", "(Figure 7)", "(Figure 8)"),

    ("Figure 6. Band-restricted ridge structure", "Figure 6. Band-restricted", "Figure 7. Band-restricted"),
    ("Restricting ridge detection to each band separately", "(Figure 6)", "(Figure 7)"),

    ("Figure 5. Representative session", "Figure 5. Representative session", "Figure 6. Representative session"),
    ("The ridges fall into the two physiological bands", "(Figure 5)", "(Figure 6)"),

    ("Figure 4. Calibration factor k and subject age", "Figure 4. Calibration factor k", "Figure 5. Calibration factor k"),
    ("k, age, and the calibration a deployment would require", "with subject age (Figure 4)", "with subject age (Figure 5)"),

    ("Figure 3. Bland–Altman agreement", "Figure 3. Bland–Altman", "Figure 4. Bland–Altman"),
    ("Averaged over a whole night, the mask tracks both rates closely", "the agreement plots in Figure 3", "the agreement plots in Figure 4"),

    ("Figure 2. Physiological-band SNR per session", "Figure 2. Physiological-band", "Figure 3. Physiological-band"),

    ("Figure 1. The overnight sensor record", "Figure 1. The overnight sensor record", "Figure 2. The overnight sensor record"),
    ("The primary record the mask produces", "shown for a representative night in Figure 1", "shown for a representative night in Figure 2"),
    ("Both physiological rhythms are trackable continuously", "Rows F and G of Figure 1", "Rows F and G of Figure 2"),
    ("The overnight record itself is a result as well as a substrate", "(Figure 1).", "(Figure 2)."),
]


def main() -> None:
    if not DOC.exists():
        raise SystemExit("missing: %s" % DOC)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    # ---------------------------------------------------- 1. figure numbering
    for needle, old, new in RENUMBER:
        d.patch(needle, old, new)

    # The schematic becomes Figure 1 and is cited as "Figure", not "Fig.".
    d.patch("Fig. 1 (a) Sensing principle", "Fig. 1 (a) Sensing principle", "Figure 1. (a) Sensing principle")
    d.patch("The regional ICP is measured by using the single electrode", "(Fig. 1a inset)", "(Figure 1a inset)")
    d.patch("The regional ICP is measured by using the single electrode", "floating ground (Fig. 1a)", "floating ground (Figure 1a)")
    d.patch("A prospective observational sleep study was conducted", "shown in Fig. 1c", "shown in Figure 1c")

    # The schematic caption labels two different panels "(b)".
    d.patch("Figure 1. (a) Sensing principle",
            "(b) Sleep mask composed of a plastic sensing unit",
            "(c) Sleep mask composed of a plastic sensing unit")

    # ---------------------------------------------------- 2. epoch length
    tbl1 = d.body[d.find("Table 1. Recording sessions and demographics") + 1]
    assert tbl1.tag == W + "tbl", "Table 1 element not where expected"
    for tc in tbl1.iter(W + "tc"):
        ts = tc.findall(".//" + W + "t")
        cur = "".join(t.text or "" for t in ts).strip()
        if cur == "Analysis epochs (60s)":
            for t in ts[1:]:
                t.text = ""
            ts[0].text = "Analysis epochs (30 s)"
            d.n_patch += 1

    d.patch("Participant demographics and recording characteristics are summarized",
            "a total of 9,319 one-minute analysis epochs were obtained for further analysis.",
            "a total of 9,319 non-overlapping 30-second analysis epochs were obtained "
            "for further analysis. Ethical approval covered an amendment to the "
            "original protocol (MOD00020664, Modification #1 to STUDY00018275).")

    # ---------------------------------------------------- 3. duplicated 3.1
    n = d.delete_range(
        "For the sleep study, we used the sleep mask in Fig. 1c.",
        "The overnight study was conducted without calibration and the sampling rate was 111 Hz")
    print("removed %d duplicated paragraphs from 3.1" % n)
    d.n_patch += n

    d.save(DOC)
    print("applied %d edits -> %s" % (d.n_patch, DOC))


if __name__ == "__main__":
    main()
