"""Reorder section 4.1 so it argues in the order a reader needs.

Old order led with the overnight fF record -- the most exploratory material --
and buried the evidence that there is a signal at all. New order:

    1. band energy and SNR                 the signal sits above the noise floor
    2. coherence, surrogates, Table 2      that energy is physiological
    3. the two rhythms tracked all night   the bands carry continuous rhythms
    4. the overnight record and imbalance  what the slow record then shows

Figures are unchanged in content. They move with their prose, so the SNR figure
becomes Figure 2 and the composite overnight-record figure becomes Figure 3,
keeping figure numbers ascending in reading order. Section 4.3's ridge
stage-associations stay where they are: those are an inferential result and
belong beside the staging discussion, not in signal validation.

One sentence needed rewriting rather than moving: the SNR paragraph opened
"The band energy underlying those traces", which referred forward to Viterbi
traces that now come after it.

Run from the repo root:  python writeup/edits/apply_41_reorder.py
"""

import re
import shutil
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_41REORDER_20260817.docx")

# The SNR paragraph led with a forward reference to the Viterbi traces.
SNR_OPENING = (
    "The two physiological bands carry energy well above the sensor noise floor. The respiratory "
    "band carried 29-48% of total signal power (0-5 Hz) and the cardiac band 8-48%."
)
SNR_OLD_OPENING = (
    "The band energy underlying those traces sits well above the sensor noise floor. The "
    "respiratory band carried 29-48% of total signal power (0-5 Hz) and the cardiac band 8-48%."
)

# The record block needs its own lead now that it no longer opens the section.
RECORD_OPENING = (
    "With the physiological content established, the remaining record is the overnight evolution "
    "of the sensor value itself. It is shown for a representative night in Figure 3, beside the "
    "scored hypnogram, the head-turn angle and the tracked rhythms."
)
RECORD_OLD_OPENING = (
    "The primary record the mask produces is the overnight evolution of the sensor value itself, "
    "shown for a representative night in Figure 2 beside the scored hypnogram, the head-turn "
    "angle and the tracked rhythms."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    head = d.find("4.1 The overnight sensor record")
    end = d.find("4.2 Rate detection")

    # locate the blocks by their opening text, as offsets from the heading
    def at(needle):
        return d.find(needle, head)

    i_record = at("The primary record the mask produces")
    i_fig2_img = i_record + 1
    i_fig2_cap = i_record + 2
    i_imbalance = at("The left–right capacitance imbalance is a dynamic")
    i_posture = at("Head position does not explain")
    i_channels = at("The forehead and differential channels")
    i_traces = at("Both physiological rhythms can be followed")
    i_snr = at("The band energy underlying those traces")
    i_fig3_img = i_snr + 1
    i_fig3_cap = i_snr + 2
    i_coh = at("Cross-spectral coherence at the ground-truth")
    i_freq = at("Respiratory frequency agreement")
    i_tab2_cap = at("Table 2. Signal validation summary")
    i_tab2 = i_tab2_cap + 1

    for name, idx in [("Figure 2 image", i_fig2_img), ("Figure 3 image", i_fig3_img)]:
        if body[idx].find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip") is None:
            raise SystemExit("expected an image at %s (index %d)" % (name, idx))
    if etree.QName(body[i_tab2]).localname != "tbl":
        raise SystemExit("expected Table 2 at index %d" % i_tab2)

    new_order = [
        i_snr, i_fig3_img, i_fig3_cap,          # 1. SNR
        i_coh, i_freq, i_tab2_cap, i_tab2,      # 2. coherence and Table 2
        i_traces,                               # 3. the rhythms hold all night
        i_record, i_fig2_img, i_fig2_cap,       # 4. the overnight record
        i_imbalance, i_posture, i_channels,
    ]
    old_span = sorted(new_order)
    if old_span != list(range(min(old_span), max(old_span) + 1)):
        raise SystemExit("section 4.1 is not contiguous; aborting rather than guessing")

    elements = [body[i] for i in new_order]
    for el in elements:
        body.remove(el)
    for off, el in enumerate(elements):
        body.insert(min(old_span) + off, el)
    print("  reordered %d elements in 4.1" % len(elements))

    # the two openings that referred to their old position
    d.patch("The band energy underlying those traces",
            "The band energy underlying those traces sits well above the sensor noise floor.",
            "The two physiological bands carry energy well above the sensor noise floor.")
    d.patch("The primary record the mask produces", RECORD_OLD_OPENING, RECORD_OPENING)
    print("  rewrote the two paragraph openings that pointed at the old order")

    # Figure 2 <-> Figure 3, so numbering ascends in reading order
    swaps = 0
    for t in d.root.iter(W + "t"):
        if not t.text or "igure" not in t.text:
            continue
        new = re.sub(r"\bFigure 2\b", "\x01", t.text)
        new = re.sub(r"\bFigure 3\b", "Figure 2", new)
        new = new.replace("\x01", "Figure 3")
        if new != t.text:
            t.text = new
            swaps += 1
    print("  swapped Figure 2 and Figure 3 across %d runs" % swaps)

    d.save(DOC)
    print("\nwrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## Section 4.1 reordered (2026-08-17)\n\n"
        + "- new order: SNR -> coherence and Table 2 -> rhythm continuity -> overnight record\n"
        + "- figures unchanged in content; SNR becomes Figure 2, the composite record figure "
          "becomes Figure 3\n"
        + "- 4.3 ridge stage-associations deliberately left after rates\n", encoding="utf-8")


if __name__ == "__main__":
    main()
