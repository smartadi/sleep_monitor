"""Figures 2 and 5 regenerated, and two captions.

Figure 2  the SNR definition box is off the axes -- it repeated the caption
          inside the panel.
Figure 5  the slow-band panel is now a smoothed power spectrum with the ridge
          overlay removed. The ridge detector is unchanged and still supplies
          the stage statistics of 4.3; only its overlay is gone, because a
          0.10 Hz median kernel inside a 0.30 Hz band left mostly speckle and
          the traces then invited the eye to follow lines through it.
Table 3   caption cut from 120 words to the definitions a reader needs, with
          the rest moved to where it belongs or dropped.

Run from the repo root:  python writeup/edits/apply_figure_updates.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_FIGUPDATE_20260817.docx")

FIG2_PNG = Path("writeup/figures/signal_validation/fig2_inband_snr.png")
FIG5_PNG = Path("writeup/figures/harmonics/ridges/ridge_tune_S1N1_CRE.png")

FIG5_CAPTION = (
    "Figure 5. Representative session (S1N1, CRE). Top: PSG hypnogram. Middle: background-removed "
    "spectrogram, 0–3 Hz, with the Viterbi respiratory (cyan) and cardiac (red) traces; both hold "
    "continuously across the full eight-hour recording, and the tracker confidence is printed in "
    "the title. Bottom: the slow band, 0–0.3 Hz, as a smoothed power spectrum with the 1/f trend "
    "removed, showing that slow power sits mainly between 0.05 and 0.10 Hz and persists through "
    "the night."
)

TABLE3_CAPTION = (
    "Table 3. Rate agreement from the operational estimator, as medians across the twelve "
    "recordings with the interquartile range in brackets. The per-epoch error is the median "
    "absolute error across a recording's 30-second epochs; the night-mean error is the absolute "
    "difference between the mean estimated and mean reference rate. Within-night r is the Pearson "
    "correlation across a recording's epochs, and the last row counts recordings whose r exceeds a "
    "circular-shift null (95% band −0.12 to +0.11 respiratory, −0.21 to +0.17 cardiac). "
    "Per-session values are in Table S1."
)


def swap(d, caption_start, png):
    """Point the image above a caption at a new file."""
    cap = d.find(caption_start)
    img = cap - 1
    if d.body[img].find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip") is None:
        raise SystemExit("no image above %r" % caption_start[:40])
    d.swap_image(d.body[img], png)


def main():
    for p in (FIG2_PNG, FIG5_PNG):
        if not p.exists():
            raise SystemExit("missing %s" % p)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)

    swap(d, "Figure 2. Physiological-band SNR per session", FIG2_PNG)
    print("  Figure 2 replaced, in-panel definition box gone")

    swap(d, "Figure 5. Representative session (S1N1, CRE)", FIG5_PNG)
    d.set_text("Figure 5. Representative session (S1N1, CRE)", FIG5_CAPTION)
    print("  Figure 5 replaced and its caption rewritten for the new slow panel")

    d.set_text("Table 3. Rate agreement, from the operational estimator", TABLE3_CAPTION)
    print("  Table 3 caption shortened")

    d.save(DOC)
    print("\nwrote %s" % DOC)


if __name__ == "__main__":
    main()
