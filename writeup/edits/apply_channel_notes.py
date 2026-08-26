"""Figure 3 without CH on the level row, and the channel relationship reported properly.

Two of the co-author's notes.

* The level row carried CLE-CRE and CH on twin axes of different scale, which
  made each one's step structure harder to read. CH is dropped from that row and
  stays in the variance row and its own spectrogram panel.

* Section 4.1 asserted the CLE-CRE / CH relationship from level statistics alone
  and concluded the channels "are not two views of one signal". Measured on the
  30-second grid the two are strongly coupled in band amplitude -- median within
  night r = 0.86 respiratory and 0.85 cardiac, with 0 of 12 sessions negative --
  while their slow levels behave as previously described. Both are true and they
  say something sharper together: the mounts see the same rhythms, and their
  slow drift is mount-specific.

Numbers from reports/mean_value/cle_cre_vs_ch.csv.

Run from the repo root:  python writeup/edits/apply_channel_notes.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_CHNOTES_20260817.docx")
FIG = Path("writeup/figures/channel_evolution/S1N1_CH_CLE-CRE.png")

CAPTION = (
    "Figure 3. The overnight sensor record for a representative night (S1N1). (A) PSG hypnogram. "
    "(B) The CLE−CRE sensor value referenced to the session mean, in fF, with the subtracted mean "
    "printed on the axis; the level moves in discrete steps, not a smooth drift. (C) Left–right "
    "capacitance imbalance, signed (line; red fill left-dominant, blue fill right-dominant) "
    "within its ± envelope (dashed). (D) Per-block variance of both channels. (E) Head-turn angle "
    "over the full ±180° range, with posture marked. (F, G) Background-removed spectrograms of "
    "CLE−CRE and CH with the Viterbi respiratory (cyan) and cardiac (red) traces overlaid, and "
    "the tracker confidence printed. Stage shading is shared across rows B–E. Cohort-wide "
    "versions of rows B and C are in Figures S6 and S7."
)

CHANNELS = (
    "The forehead and differential channels agree about the body and disagree about the mount. "
    "On the 30-second grid their band amplitudes move together closely: the within-night "
    "correlation of band-limited amplitude has a median of 0.86 in the respiratory band and 0.85 "
    "in the cardiac band, and is positive in every one of the twelve recordings (range 0.61 to "
    "0.99). Their slow levels do not: CH moves 2.8 times as far as CLE−CRE across a night (median "
    "ratio of standard deviations, range 1.6–11.0), and the correlation between the levels "
    "themselves ranges from −0.49 to +0.96, negative in 3 of 12 recordings. Coherence follows the "
    "same split, strongest in the slow band (median 0.55) and weaker in the respiratory (0.31) "
    "and cardiac (0.09) bands. The two mounts therefore see the same physiological rhythms while "
    "each carries its own slow drift, which is why agreement between them is usable as evidence "
    "that a feature is physiological rather than an artifact of one mount, and why the "
    "differential cannot be assumed to cancel the level steps described above."
)


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)

    d = Doc(DOC)
    cap_i = d.find("Figure 3. The overnight sensor record")
    img = d.body[cap_i - 1]
    if img.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip") is None:
        raise SystemExit("expected the Figure 3 image above its caption")
    d.swap_image(img, FIG)
    d.set_text("Figure 3. The overnight sensor record", CAPTION)
    print("  Figure 3 replaced; caption now describes a single-channel level row")

    d.set_text("The forehead and differential channels are not two views of one signal",
               CHANNELS)
    print("  4.1: channel relationship reported in amplitude and in level")

    d.save(DOC)
    print("wrote %s" % DOC)
main()
