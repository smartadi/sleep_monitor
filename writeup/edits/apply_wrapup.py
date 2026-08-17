"""Final wrap-up pass on the canonical manuscript.

Closes the completeness and front-matter gaps that remained after the
2026-08-14 rate rewrite, so the paper reads end to end:

  1. Abstract written (it was empty; the placeholder had been deleted, never filled)
  2. Keywords line added
  3. Duplicated author line removed
  4. "3.1 Overnight testing" given Heading2 style like every other subsection
  5. Table 1: doubled subject IDs (OS001OS001) fixed, "Ages" -> "Age"
  6. Stray empty Heading2 removed
  7. Reference 28 rejoined (it was split across two bibliography paragraphs)
  8. "References" heading added above the bibliography

Numbers in the abstract are taken from the current text of sections 4.1-4.5,
Table 3 and section 7 -- nothing new is computed here.

Run from the repo root:  python writeup/edits/apply_wrapup.py
"""

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_WRAPUP_20260817.docx")

log = []


def note(msg):
    log.append(msg)
    print("  " + msg)


ABSTRACT = (
    "Wearable sensors that report brain-adjacent physiology without scalp electrodes are "
    "attractive for home sleep monitoring, but what such a sensor actually measures across a "
    "whole night is rarely established against a full reference. Here we characterise a sleep "
    "mask carrying single-electrode capacitance (SEC) sensors at both temples and the forehead, "
    "worn for twelve overnight recordings by six healthy adults aged 25–66 years, two nights "
    "each, with simultaneous polysomnography in the participants' own homes. We report both what "
    "the mask measures and what it does not."
)

ABSTRACT_2 = (
    "The overnight capacitance record is readable in absolute terms from the first hour of a "
    "night to the last. The temple electrodes operate near 2 pF and move in discrete, "
    "posture-linked steps rather than drifting smoothly, and the left–right imbalance between "
    "them is dynamic, with a median magnitude of 12.3 fF and a sign that reverses a median of "
    "four times per night. Respiratory and cardiac energy sits far above the electronic noise "
    "floor on every channel of every recording (physiological-band SNR +18.7 to +30.0 dB by "
    "channel, minimum +6.4 dB), exceeds a phase-randomised surrogate null about threefold in the "
    "respiratory band and twofold in the cardiac band, and can be followed as two continuous "
    "spectrogram traces across a full recording. After a per-session calibration factor k, the "
    "mask recovers each night's mean respiratory rate to a median of 0.24 br/min and its mean "
    "cardiac rate to 1.56 BPM, with per-epoch errors of 1.79 br/min and 3.41 BPM. It does not "
    "follow either rate as it varies within the night: the within-recording correlation with the "
    "reference is indistinguishable from zero for every estimator, channel and fusion strategy "
    "tested. The calibration factor is a waveform-morphology count rather than a fitted "
    "parameter — R-peak-triggered averaging gives 1.70 to 2.43 capacitive peaks per heartbeat, "
    "bracketing the fitted cardiac k of 1.96 — and it reproduces across a subject's two nights."
)

ABSTRACT_3 = (
    "The mask carries no cortical electrographic signature. Sleep spindles produce no sigma-band "
    "response on any capacitive channel (+0.02 to +0.03 dB) where the same measurement on contact "
    "EEG rises +3.3 dB. What appears instead, at both spindles and delta-burst onsets, is a "
    "low-frequency mechanical response that follows the cortical event: +0.47 to +0.58 dB of "
    "0–3 Hz power at spindle centres, and a sharp post-onset rise at delta-burst onsets in six of "
    "six subjects with a flat pre-onset baseline under strictly causal filtering. Band-restricted "
    "spectral ridges carry interpretable but weak stage associations, respiratory-band ridges "
    "being fewer during N3 in all six subjects. The mask is therefore a transducer of mechanical "
    "and hemodynamic pulsation, suited to unobtrusive overnight mean-rate trending and to "
    "multi-modal sleep assessment as one input, and not to instantaneous rate tracking or to "
    "cortical monitoring."
)

KEYWORDS = (
    "Keywords: capacitive sensing; single-electrode capacitance; wearable sleep monitoring; "
    "respiratory rate; heart rate; ballistocardiography; polysomnography; sleep spindles; "
    "slow-wave activity"
)


def main():
    if not DOC.exists():
        raise SystemExit("missing %s" % DOC)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    body = d.body

    # ------------------------------------------------------------- front matter
    i = d.find("Changwoo Lee")
    if d.text_of(body[i + 1]).startswith("Changwoo Lee"):
        body.remove(body[i + 1])
        note("removed the duplicated author line")

    i_abs = d.find("Abstract")
    after = d.text_of(body[i_abs + 1]).strip()
    if after and not after.startswith("1. Introduction"):
        raise SystemExit("abstract is not empty -- refusing to insert over: %r" % after[:80])
    for off, txt in enumerate([ABSTRACT, ABSTRACT_2, ABSTRACT_3, "", KEYWORDS, ""]):
        body.insert(i_abs + 1 + off, d.para(txt))
    note("abstract written (3 paragraphs) and keywords line added")

    # ------------------------------------------------------------------ methods
    d.set_style("3.1 Overnight testing", "Heading2")
    note("3.1 Overnight testing styled as Heading2, matching 3.2-3.7")

    tbl1 = body[d.find("Session", d.find("Table 1. Recording sessions"))]
    d.cell_text(tbl1, 0, 2, "Age")
    fixed = 0
    for r_i in range(1, len(tbl1.findall(W + "tr"))):
        tc = tbl1.findall(W + "tr")[r_i].findall(W + "tc")[1]
        txt = "".join(t.text or "" for t in tc.findall(".//" + W + "t")).strip()
        m = re.match(r"^(OS\d{3})\1$", txt)
        if m:
            d.cell_text(tbl1, r_i, 1, m.group(1))
            fixed += 1
    note("Table 1: %d doubled subject IDs fixed, 'Ages' header corrected to 'Age'" % fixed)

    # --------------------------------------------------------------- references
    i28 = d.find("28.Committee ISC. IEEE Standards")
    if d.text_of(body[i28 + 1]).startswith("0 Hz to 300 GHz"):
        d._patch_el(body[i28], "Electric, Magnetic, and Electromagnetic Fields,",
                    "Electric, Magnetic, and Electromagnetic Fields, 0 Hz to 300 GHz. "
                    "IEEE Std C95.1-2019. 2019.")
        body.remove(body[i28 + 1])
        note("reference 28 rejoined into a single entry")

    i_ref1 = d.find("1.Carley DW, Farabi SS")
    prev = d.text_of(body[i_ref1 - 1]).strip()
    if not prev.lower().startswith("reference"):
        body.insert(i_ref1, d.para("References", "Heading1"))
        note("added a References heading above the bibliography")

    # -------------------------------------------------------------- final sweep
    for i in range(len(body) - 1, -1, -1):
        el = body[i]
        st = el.find(".//" + W + "pStyle")
        if (st is not None and st.get(W + "val") == "Heading2"
                and not d.text_of(el).strip() and not el.findall(".//" + W + "drawing")):
            body.remove(el)
            note("removed a stray empty Heading2 paragraph")

    d.save(DOC)
    print("\n%d edit operations applied" % d.n_patch)
    print("wrote %s" % DOC)
    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        "# Wrap-up edits applied to CAP_sleep_mask_manuscript_main.docx (2026-08-17)\n\n"
        + "\n".join("- " + m for m in log) + "\n", encoding="utf-8")
    print("wrote writeup/edits/WRAPUP_CHANGELOG.md")


if __name__ == "__main__":
    main()
