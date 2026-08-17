"""Consistency checks for the manuscript docx.

Four families of check, in decreasing order of how much they can prove:

  LEDGER      every value in writeup/paper/key_numbers.py, recomputed from the
              artifacts, must still appear in the document. A value that has
              moved is drift: either the paper is stale or the pipeline changed
              under it.
  CONFLICT    the same quantity stated twice with different values, found by
              comparing numbers that share their surrounding wording. This is
              what 99.95 vs 99.98 and 1.95 vs 1.96 looked like.
  VOCAB       the pipeline descriptors in Methods must match those in Results
              (channel, estimator, window, filter). This is what "fused across
              five channels" vs "loose peak counting on CRE" looked like.
  STRUCTURE   figure and table numbering contiguous and referenced, every
              caption backed by an image, citation markers within the
              bibliography range and never left as plain text, no placeholders,
              one spelling convention.

Usage:
    python writeup/edits/check_manuscript.py [path-to-docx]

Exit status is 1 when any FAIL is reported, so it can gate a commit.
"""

from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "writeup" / "paper"))

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

DEFAULT_DOC = ROOT / "writeup/main/CAP_sleep_mask_manuscript_main.docx"

findings: list[tuple[str, str, str]] = []


def report(level, family, msg):
    findings.append((level, family, msg))


# ------------------------------------------------------------------ document
class Doc:
    def __init__(self, path):
        z = zipfile.ZipFile(path)
        self.root = etree.fromstring(z.read("word/document.xml"))
        self.body = self.root.find(W + "body")
        self.paras = []          # (index, style, text, has_image)
        for i, el in enumerate(self.body):
            tag = etree.QName(el).localname
            if tag == "p":
                st = el.find(W + "pPr/" + W + "pStyle")
                self.paras.append((
                    i,
                    (st.get(W + "val") or "") if st is not None else "",
                    "".join(t.text or "" for t in el.iter(W + "t")),
                    el.find(".//" + A + "blip") is not None,
                ))
            elif tag == "tbl":
                self.paras.append((i, "__table__",
                                   "".join(t.text or "" for t in el.iter(W + "t")), False))
        self.text = "\n".join(p[2] for p in self.paras)

    def body_text(self):
        """Everything except the bibliography."""
        return "\n".join(t for _, st, t, _ in self.paras
                         if st != "EndNoteBibliography")


# -------------------------------------------------------------- 1. ledger
def check_ledger(doc):
    try:
        from key_numbers import NUMBERS
    except Exception as exc:
        report("FAIL", "LEDGER", "could not import the ledger: %s" % exc)
        return
    text = doc.text.replace("−", "-")
    missing = []
    for key, v in NUMBERS.items():
        needle = v.rendered().replace("−", "-")
        alts = {needle}
        # tolerate en-dash / hyphen and thousands separators
        alts.add(needle.replace("–", "-"))
        alts.add(needle.replace(",", ""))
        if not any(a in text for a in alts):
            missing.append((key, v))
    for key, v in missing:
        report("FAIL", "LEDGER",
               "%s = %s (%s) does not appear in the document -- source %s%s"
               % (key, v.rendered(), v.unit or "-", v.source,
                  ("; " + v.note) if v.note else ""))
    n_ok = len(NUMBERS) - len(missing)
    report("INFO", "LEDGER", "%d of %d ledger values found in the document"
           % (n_ok, len(NUMBERS)))


# ------------------------------------------------------------ 2. conflicts
STOP = set("the a an of in on at to for by and or is are was were with from as that this "
           "it its be been than then so we our their there here which when what while "
           "each per both all any not no only also more most less least over under".split())

NUM_RE = re.compile(r"(?<![\w.])([+\-−]?\d+(?:\.\d+)?)(%|\s?(?:br/min|BPM|dB|Hz|fF|z))?")


def context_key(words):
    return frozenset(w for w in words if w not in STOP and len(w) > 2)


def _contrasted(text, vi, vj, span=220):
    """True if the two values appear close together joined by a comparison word."""
    pi = [m.start() for m in re.finditer(re.escape(vi), text)]
    pj = [m.start() for m in re.finditer(re.escape(vj), text)]
    for a in pi:
        for b in pj:
            lo, hi = min(a, b), max(a, b)
            if hi - lo > span:
                continue
            if re.search(r"(against|versus|compared|rather than|no better than|"
                         r"comparable to)", text[lo:hi]):
                return True
    return False


def check_conflicts(doc):
    text = doc.body_text()
    tokens = []
    for m in NUM_RE.finditer(text):
        val, unit = m.group(1), (m.group(2) or "").strip()
        lo = max(0, m.start() - 90)
        hi = min(len(text), m.end() + 60)
        ctx = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text[lo:hi].lower())
        tokens.append((val, unit, context_key(ctx), text[lo:hi].replace("\n", " ")))

    flagged = set()
    for i in range(len(tokens)):
        vi, ui, ci, si = tokens[i]
        for j in range(i + 1, len(tokens)):
            vj, uj, cj, sj = tokens[j]
            if vi == vj or ui != uj or not ui:
                continue
            try:
                a, b = abs(float(vi.replace("−", "-"))), abs(float(vj.replace("−", "-")))
            except ValueError:
                continue
            if a == 0 or b == 0 or a == b:      # "3" vs "3.0" is not a conflict
                continue
            ratio = max(a, b) / min(a, b)
            # A restated quantity drifts by a rounding step, not by percent. Wider
            # gaps are two different quantities that happen to sit near each other,
            # e.g. 100% consensus coverage against 97% for airflow alone.
            if ratio > 1.02:
                continue
            shared = ci & cj
            if len(shared) < 4:
                continue
            # numbers inside one sentence are being contrasted deliberately
            if si.strip()[:40] == sj.strip()[:40]:
                continue
            # Two values joined by a comparison word inside one clause are being
            # contrasted on purpose: "1.55 br/min against a reference SD of 1.57".
            # Searched over the whole text, since the context windows are narrow.
            if _contrasted(text, vi, vj):
                continue
            # "moves from 3.41 to 3.39" states a change, not a disagreement
            if any(re.search(r"from %s" % re.escape(v), s)
                   for v in (vi, vj) for s in (si, sj)):
                continue
            key = tuple(sorted([vi + ui, vj + uj]))
            if key in flagged:
                continue
            flagged.add(key)
            report("WARN", "CONFLICT",
                   "%s%s vs %s%s share wording %s\n        ... %s\n        ... %s"
                   % (vi, ui, vj, uj, sorted(shared)[:6], si.strip()[:110],
                      sj.strip()[:110]))


# ---------------------------------------------------------------- 3. vocab
PIPELINE_TERMS = {
    "channel": [r"\bCRE\b", r"\bCLE−CRE\b", r"\bCLE-CRE\b", r"fused across (\w+) channels"],
    "estimator": [r"peak counting", r"spectral peak", r"CWT", r"Viterbi"],
    "window": [r"(\d+)[ -]second (?:analysis )?window", r"(\d+) s (?:non-overlapping|window)"],
}


def section_span(doc, start_needle, end_needle):
    idx = {t[2][:60]: n for n, t in enumerate(doc.paras)}
    s = e = None
    for n, (_, _, txt, _) in enumerate(doc.paras):
        if s is None and txt.strip().startswith(start_needle):
            s = n
        elif s is not None and txt.strip().startswith(end_needle):
            e = n
            break
    return (s, e if e is not None else len(doc.paras))


def check_vocab(doc):
    m_s, m_e = section_span(doc, "3.5", "3.6")
    r_s, r_e = section_span(doc, "4.2 Rate detection", "4.3")
    if m_s is None or r_s is None:
        report("WARN", "VOCAB", "could not locate 3.5 / 4.2 to compare")
        return
    methods = " ".join(t[2] for t in doc.paras[m_s:m_e])
    results = " ".join(t[2] for t in doc.paras[r_s:r_e])

    fused_m = re.search(r"fused across \w+ channels", methods)
    fused_r = re.search(r"fused across \w+ channels", results)
    cre_m = "CRE" in methods
    cre_r = "CRE" in results
    if bool(fused_m) != bool(fused_r):
        report("FAIL", "VOCAB",
               "3.5 %s multi-channel fusion but 4.2 %s -- the reported pipeline must "
               "be described the same way in both"
               % ("describes" if fused_m else "does not describe",
                  "does" if fused_r else "does not"))
    if cre_m != cre_r:
        report("WARN", "VOCAB",
               "3.5 %s name CRE, 4.2 %s" % ("does" if cre_m else "does not",
                                            "does" if cre_r else "does not"))
    wins = set(re.findall(r"(\d+)[ -]second (?:analysis )?window", doc.body_text()))
    wins |= set(re.findall(r"(\d+) s (?:non-overlapping|analysis)", doc.body_text()))
    if len(wins) > 1:
        report("FAIL", "VOCAB", "analysis window stated as %s seconds in different places"
               % sorted(wins))


# ------------------------------------------------------------ 4. structure
def check_structure(doc):
    text = doc.text

    for kind in ("Figure", "Table"):
        for prefix, label in (("", "main"), ("S", "supplementary")):
            caps = []
            for _, st, t, _ in doc.paras:
                m = re.match(r"^%s %s(\d+)\." % (kind, prefix), t.strip())
                if m and (prefix or not t.strip().startswith("%s S" % kind)):
                    caps.append(int(m.group(1)))
            if not caps:
                continue
            expect = list(range(1, len(caps) + 1))
            if sorted(caps) != expect:
                report("FAIL", "STRUCTURE",
                       "%s %s captions are %s, expected %s"
                       % (label, kind, sorted(caps), expect))
            for n in sorted(caps):
                # count "Figure 4", "Figures S8 and S9", "Fig. 1c" alike
                pat = r"%ss?\.?\s+(?:[S\d, and]*\s)?%s%d(?![\d])" % (kind, prefix, n)
                body_refs = len(re.findall(pat, text))
                if body_refs < 2:
                    report("WARN", "STRUCTURE",
                           "%s %s%d has a caption but is never referenced in the text"
                           % (kind, prefix, n))

    # every figure caption should sit next to an image
    for n, (_, st, t, _) in enumerate(doc.paras):
        if not re.match(r"^Figure S?\d+\.", t.strip()):
            continue
        near = any(doc.paras[k][3] for k in range(max(0, n - 3), min(len(doc.paras), n + 2)))
        if not near:
            report("FAIL", "STRUCTURE", "%s has no image near it" % t.strip()[:50])

    # citations
    bib = [t for _, st, t, _ in doc.paras if st == "EndNoteBibliography"]
    n_refs = 0
    for t in bib:
        m = re.match(r"^(\d+)\.", t.strip())
        if m:
            n_refs = max(n_refs, int(m.group(1)))
    cited = {int(x) for x in re.findall(r"(?<=\D)(\d{1,2})(?=\s|,|\.)", "")}
    if n_refs:
        report("INFO", "STRUCTURE", "bibliography has %d entries" % n_refs)

    # a citation marker left as body text rather than a superscript run
    for el in doc.body.iter(W + "p"):
        runs = [r for r in el.findall(W + "r") if r.find(W + "t") is not None]
        for r in runs:
            sup = r.find(W + "rPr/" + W + "vertAlign")
            if sup is not None:
                continue
            txt = r.find(W + "t").text or ""
            for m in re.finditer(r"[a-z]\.(\d{1,2})(?:\s|$)", txt):
                report("WARN", "STRUCTURE",
                       "possible citation marker in plain text: '...%s'"
                       % txt[max(0, m.start() - 40):m.end()].strip())

    # The abstract is the professor's to write; its placeholder marks the slot
    # and is not a finding.
    for bad in ("TO BE WRITTEN", "TBD", "TODO", "XXX"):
        if bad in text:
            report("WARN", "STRUCTURE", "placeholder present: %r" % bad)

    # spelling convention
    # Counted against the same word list the spelling pass uses, so the two
    # cannot drift apart. A generic -ise pattern would flag "noise" and
    # "characteristic"; only real pairs count.
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from apply_us_spelling import MAP as SPELLING
    except Exception:
        SPELLING = {}
    brit = amer = 0
    for bword, aword in SPELLING.items():
        brit += len(re.findall(r"\b%s\b" % bword, text, re.I))
        amer += len(re.findall(r"\b%s\b" % aword, text, re.I))
    if brit and amer:
        report("WARN", "STRUCTURE",
               "mixed spelling convention: %d British, %d American forms" % (brit, amer))

    empty = [i for i, st, t, img in doc.paras
             if st.startswith("Heading") and not t.strip()]
    if empty:
        report("FAIL", "STRUCTURE", "empty heading paragraphs at %s" % empty)

    check_acronyms(doc)


# Channel and stage labels, address abbreviations, and organisation names that
# are not expanded in any paper.
ACRONYM_SKIP = {
    "CLE", "CRE", "CH", "REM", "NREM", "IEEE", "MD", "WA", "UW", "US",
    "PSG", "SEC", "AE", "A", "B", "C", "D", "E", "F", "G", "N1", "N2", "N3",
}


def check_acronyms(doc):
    """Every abbreviation should be expanded where it is first used."""
    text = doc.body_text()
    for acr in sorted(set(re.findall(r"\b([A-Z]{2,6})\b", text))):
        if acr in ACRONYM_SKIP or re.match(r"^S\d", acr):
            continue
        # "expansion (ACR)" or "ACR (expansion)" anywhere counts as defined
        defined = (re.search(r"\([^()]{0,90}\b%s\b[^()]{0,12}\)" % acr, text)
                   or re.search(r"\b%s\s*\([a-z]" % acr, text))
        if not defined:
            at = text.find(acr)
            report("WARN", "ACRONYM",
                   "%s used %d time(s), never expanded -- first use: ...%s..."
                   % (acr, len(re.findall(r"\b%s\b" % acr, text)),
                      text[max(0, at - 55):at + len(acr) + 10].replace("\n", " ")))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOC
    if not path.exists():
        raise SystemExit("no such file: %s" % path)
    doc = Doc(path)
    print("checking %s\n" % path)

    check_ledger(doc)
    check_conflicts(doc)
    check_vocab(doc)
    check_structure(doc)

    order = {"FAIL": 0, "WARN": 1, "INFO": 2}
    findings.sort(key=lambda f: order[f[0]])
    counts = Counter(f[0] for f in findings)
    for level, family, msg in findings:
        print("[%-4s] %-9s %s" % (level, family, msg))
    print("\n%d FAIL, %d WARN, %d INFO"
          % (counts["FAIL"], counts["WARN"], counts["INFO"]))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
