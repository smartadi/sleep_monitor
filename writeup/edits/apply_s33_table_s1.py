"""Rebuild Table S1 from the single-pipeline source, and reconcile the k values.

Follow-up to apply_s33_onward.py.  Table S1 and Table 3 previously came from
different pipelines, so no per-session value in S1 summed to the headline in 3.
Both are now populated from one file:

    reports/rates/mask/table_s1_review_single_pipeline.csv

Checked against that file, every Table 3 cell reproduces except the respiratory
calibration factor, which is 0.99 [0.95-1.05] on this pipeline rather than the
0.96 [0.91-1.02] carried over from the retired spectral pipeline.  The
respiratory k is corrected here wherever it is quoted as an operational value.

NOT touched: the k series behind Figure 4 and the age analysis of 4.2.1, which
comes from analysis/rates/k_age_prior.py and is a different pipeline again.
That mismatch is left visible rather than papered over.

Run from the repo root:  python writeup/edits/apply_s33_table_s1.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_review_edits import Doc, W  # noqa: E402

DOC = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
SRC = Path("reports/rates/mask/table_s1_review_single_pipeline.csv")

HEADER = ["Session", "Resp k", "Resp night err\n(br/min)", "Resp epoch AE\n(br/min)",
          "Resp r", "Card k", "Card night err\n(BPM)", "Card epoch AE\n(BPM)", "Card r"]


def main() -> None:
    df = pd.read_csv(SRC)
    d = Doc(DOC)

    # ---- Table S1 ------------------------------------------------------------
    tbl = d.body[d.find("Table S1. Per-session rate accuracy") + 1]
    assert tbl.tag == W + "tbl", "Table S1 element not where expected"
    rows = tbl.findall(W + "tr")
    assert len(rows) == 1 + len(df), "unexpected Table S1 shape: %d rows" % len(rows)

    for col, label in enumerate(HEADER):
        d.cell_text(tbl, 0, col, label.replace("\n", " "))
    fmt = ["%s", "%.2f", "%.2f", "%.2f", "%+.2f", "%.2f", "%.2f", "%.2f", "%+.2f"]
    for i in range(len(df)):
        row = df.iloc[i]
        for col, f in enumerate(fmt):
            d.cell_text(tbl, i + 1, col, f % row.iloc[col])
    d.n_patch += 1

    d.set_text(
        "Table S1. Per-session rate accuracy in both regimes.",
        "Table S1. Per-session rate accuracy, from the same operational pipeline as "
        "Table 3. Sessions are labelled SxNy for subject x, night y. k is the "
        "per-session calibration factor. \"Night err\" is the night-level error, "
        "|mean(estimate) − mean(reference)| for that recording, and is the quantity "
        "summarized in Table 3's aggregate regime; \"epoch AE\" is the median "
        "absolute error across that recording's epochs; r is the within-session "
        "Pearson correlation between estimate and reference across those epochs. "
        "Because both tables are computed from one pipeline, the medians of these "
        "columns reproduce Table 3 by construction.")

    # ---- k reconciliation ----------------------------------------------------
    for tc in tbl.getparent().iter(W + "tc"):
        pass  # no-op; kept for clarity that only Table 3 below is edited

    tbl3 = d.body[d.find("Table 3. Rate detection in both regimes") + 1]
    for tc in tbl3.iter(W + "tc"):
        ts = tc.findall(".//" + W + "t")
        cur = "".join(t.text or "" for t in ts).strip()
        if cur == "0.96 [0.91–1.02]":
            for t in ts[1:]:
                t.text = ""
            ts[0].text = "0.99 [0.95–1.05]"
            d.n_patch += 1

    d.patch("k is a waveform-morphology count",
            "The respiratory k of ≈0.96 reflects", "The respiratory k of ≈0.99 reflects")

    d.patch("k is reproducible within a subject",
            "a median of 0.013 for respiration (maximum 0.047) and 0.151 for the "
            "cardiac band (maximum 0.406)",
            "a median of 0.020 for respiration (maximum 0.050) and 0.150 for the "
            "cardiac band (maximum 0.410)")

    # 5.1 still described k through the retired spectral estimator, and closed on
    # a literature comparison that no cited source supports.
    d.patch("The mask recovers each night's mean respiratory rate",
            "The respiratory k factor is near unity (k ≈ 0.96), meaning the spectral "
            "peak frequency directly reflects the breathing rate with negligible "
            "correction.",
            "The respiratory k factor is near unity (k ≈ 0.99), meaning one dominant "
            "temple displacement per breath and negligible correction.")
    d.patch("The mask recovers each night's mean respiratory rate",
            "The k calibration is remarkably stable: diagnostic estimates from 50 "
            "random windows agree with whole-night values within 0.04, and 3 of 6 "
            "subjects showed night-to-night k variation of ≤0.03. These accuracies "
            "are comparable to other non-contact and wearable cardiac rate sensors "
            "in the literature.",
            "The k calibration is stable: diagnostic estimates from 50 random "
            "windows agree with whole-night values to within 0.04, and respiratory k "
            "changed by ≤0.03 between a subject's two nights in four of six "
            "subjects. Direct comparison with other non-contact and wearable cardiac "
            "sensors is complicated by the fact that most report per-epoch rather "
            "than per-night error, and we are not aware of a published dataset using "
            "this window length and error definition against which these values "
            "could be matched.")

    d.save(DOC)
    print("applied %d edits -> %s" % (d.n_patch, DOC))


if __name__ == "__main__":
    main()
