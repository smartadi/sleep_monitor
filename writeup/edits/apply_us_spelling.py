"""Normalise the manuscript to American spelling.

The bibliography is skipped: reference titles are quoted material and must keep
the spelling they were published with.

Two words that look British and are not, and are therefore left alone:

  analyses   plural of analysis, correct in American English. Both occurrences
             are nouns ("surrogate analyses", "ridge analyses"), not the verb.
  canceller  changed to "canceler" on 2026-08-17 at the user's direction
             ("americanize everything"). The signal-processing literature
             commonly writes "canceller" even in US journals, so this is a
             house-style choice rather than a correction.

Run from the repo root:  python writeup/edits/apply_us_spelling.py
"""

import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apply_review_edits import Doc, W  # noqa: E402
from target_doc import DOC  # noqa: E402

BACKUP = Path("writeup/_archive/CAP_sleep_mask_manuscript_main_PRE_USSPELL2_20260817.docx")

# British -> American, applied as whole words, case preserved on the first letter.
MAP = {
    "characterise": "characterize", "characterised": "characterized",
    "characterises": "characterizes", "characterising": "characterizing",
    "characterisation": "characterization",
    "summarise": "summarize", "summarised": "summarized",
    "summarises": "summarizes", "summarising": "summarizing",
    "normalise": "normalize", "normalised": "normalized",
    "normalises": "normalizes", "normalising": "normalizing",
    "normalisation": "normalization",
    "minimise": "minimize", "minimised": "minimized", "minimising": "minimizing",
    "maximise": "maximize", "maximised": "maximized", "maximises": "maximizes",
    "maximising": "maximizing",
    "quantise": "quantize", "quantised": "quantized", "quantises": "quantizes",
    "randomise": "randomize", "randomised": "randomized", "randomising": "randomizing",
    "synchronise": "synchronize", "synchronised": "synchronized",
    "recognise": "recognize", "recognised": "recognized",
    "organise": "organize", "organised": "organized",
    "emphasise": "emphasize", "emphasised": "emphasized",
    "generalise": "generalize", "generalised": "generalized",
    "generalisation": "generalization",
    "utilise": "utilize", "utilised": "utilized",
    "standardise": "standardize", "standardised": "standardized",
    "hypothesise": "hypothesize", "hypothesised": "hypothesized",
    "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing",
    "behaviour": "behavior", "behaviours": "behaviors", "behavioural": "behavioral",
    "colour": "color", "colours": "colors", "coloured": "colored",
    "favour": "favor", "favoured": "favored", "labour": "labor",
    "labelled": "labeled", "labelling": "labeling",
    "modelled": "modeled", "modelling": "modeling",
    "cancelled": "canceled", "cancelling": "canceling",
    "canceller": "canceler", "cancellers": "cancelers",
    "signalled": "signaled", "signalling": "signaling",
    "travelled": "traveled", "totalled": "totaled",
    "centre": "center", "centres": "centers", "centred": "centered",
    "metre": "meter", "metres": "meters",
    "fibre": "fiber", "fibres": "fibers",
    "grey": "gray", "greyscale": "grayscale",
    "artefact": "artifact", "artefacts": "artifacts",
    "defence": "defense", "practise": "practice", "licence": "license",
    "ageing": "aging", "judgement": "judgment",
    "whilst": "while", "amongst": "among",
    "programme": "program", "sulphur": "sulfur", "aluminium": "aluminum",
}

PATTERN = re.compile(r"\b(%s)\b" % "|".join(sorted(MAP, key=len, reverse=True)),
                     re.IGNORECASE)


def convert(text):
    hits = Counter()

    def sub(m):
        word = m.group(0)
        repl = MAP[word.lower()]
        if word[0].isupper():
            repl = repl[0].upper() + repl[1:]
        hits[word] += 1
        return repl

    return PATTERN.sub(sub, text), hits


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("target -> %s" % DOC)
    print("backup -> %s" % BACKUP)

    d = Doc(DOC)
    total = Counter()
    skipped_bib = 0

    for el in d.body:
        style = el.find(W + "pPr/" + W + "pStyle")
        if style is not None and style.get(W + "val") == "EndNoteBibliography":
            skipped_bib += 1
            continue
        for t in el.iter(W + "t"):
            if not t.text:
                continue
            new, hits = convert(t.text)
            if new != t.text:
                t.text = new
                total.update(hits)

    for word, n in sorted(total.items()):
        print("  %-16s -> %-16s %d" % (word, MAP[word.lower()], n))
    print("\n%d substitutions, %d bibliography paragraphs left untouched"
          % (sum(total.values()), skipped_bib))

    d.save(DOC)
    print("wrote %s" % DOC)

    Path("writeup/edits/WRAPUP_CHANGELOG.md").write_text(
        Path("writeup/edits/WRAPUP_CHANGELOG.md").read_text(encoding="utf-8")
        + "\n## American spelling (2026-08-17)\n\n"
        + "- %d substitutions in the body; bibliography untouched so reference titles keep "
          "their published spelling\n" % sum(total.values())
        + "- 'analyses' left alone (plural noun, correct in American English); 'canceller' "
          "left as the signal-processing term of art\n", encoding="utf-8")


if __name__ == "__main__":
    main()
