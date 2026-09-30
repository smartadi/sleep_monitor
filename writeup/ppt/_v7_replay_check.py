"""Dry run: which of our V5/V6 edits still apply to the reviewer's V9?

The two manuscript lines diverged. Ours is V4 -> V5 -> V6, built by script.
His is V4 -> V8 (58 tracked insertions and 26 deletions by KH Choi) -> V9,
edited by hand. V9 is now the agreed base, so our scripted edits have to be
replayed onto text that has moved underneath them.

`_build_v5.py` addresses paragraphs by index (`P[85].runs[8]`) and `_build_v6.py`
by index plus an expected string. Indices are worthless against V9 -- it has a
different paragraph count -- but the expected strings are not: they say what the
edit was looking for, independently of where it sat. This script pulls every
`replace(d, i, old, new)` call out of `_build_v6.py` with `ast` (so multi-line
calls parse correctly) and asks V9 one question per edit:

    already applied   `new` is present and `old` is gone -- he made the same fix
    applies cleanly   `old` occurs exactly once -- replay it, retargeted by text
    ambiguous         `old` occurs more than once -- needs a human to pick
    conflict          neither `old` nor `new` -- he rewrote the sentence

Nothing is written. The output is the conflict list that has to be resolved
before a V7 build can be trusted.

Usage
-----
    .venv/Scripts/python.exe writeup/ppt/_v7_replay_check.py
"""

from __future__ import annotations

import ast
import zipfile
from collections import Counter
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
BUILD_V6 = HERE / '_build_v6.py'
BASE = ROOT / 'writeup' / 'review' / 'CAP_sleep_mask_manuscript_V9(1).docx'
OURS = HERE / 'CAP_sleep_mask_manuscript_V6.docx'


def q(t):
    return '{%s}%s' % (W, t)


def paragraphs(path: Path) -> list[str]:
    """Paragraph text in document order, tables included -- the same order
    `_build_v6.py` indexes, so a surviving index can be reported for context."""
    z = zipfile.ZipFile(path)
    root = etree.fromstring(z.read('word/document.xml'))
    return [''.join(t.text or '' for t in p.iter(q('t')))
            for p in root.iter(q('p'))]


def edits_from_build(path: Path) -> list[tuple[str, int, str, str]]:
    """Every replace(d, i, old, new) in the build script, via the AST.

    Returns (function, para_index, old, new). Calls whose old/new are not plain
    string literals are skipped and counted separately by the caller.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'))
    out, skipped = [], 0
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'replace'):
                continue
            a = node.args
            if (len(a) >= 4
                    and isinstance(a[1], ast.Constant) and isinstance(a[1].value, int)
                    and isinstance(a[2], ast.Constant) and isinstance(a[2].value, str)
                    and isinstance(a[3], ast.Constant) and isinstance(a[3].value, str)):
                out.append((fn.name, a[1].value, a[2].value, a[3].value))
            else:
                skipped += 1
    return out, skipped


def clip(s: str, n: int = 110) -> str:
    s = ' '.join(s.split())
    return s if len(s) <= n else s[:n] + '...'


def nearest(paras: list[str], old: str) -> tuple[float, int, str]:
    """Best fuzzy home for `old` in the base document.

    Scores each paragraph by the longest common substring it shares with `old`,
    as a fraction of `old`. A high score means the sentence is still there in
    altered wording -- the edit can be merged. A low score means the passage the
    edit was aimed at is not in this document at all.
    """
    import difflib
    best = (0.0, 0, '')
    for k, para in enumerate(paras, start=1):
        if not para.strip():
            continue
        m = difflib.SequenceMatcher(None, old, para, autojunk=False)\
                   .find_longest_match(0, len(old), 0, len(para))
        score = m.size / max(1, len(old))
        if score > best[0]:
            lo = max(0, m.b - 30)
            best = (score, k, para[lo:m.b + m.size + 30])
    return best


def classify(full: str, old: str, new: str) -> str:
    n_old = full.count(old)
    n_new = full.count(new) if new else 0
    if n_old == 1:
        return 'applies cleanly'
    if n_old > 1:
        return 'ambiguous'
    # old is absent
    if new and n_new:
        return 'already applied'
    return 'conflict'


def main():
    for p in (BASE, OURS, BUILD_V6):
        if not p.exists():
            raise SystemExit('missing: %s' % p)

    edits, skipped = edits_from_build(BUILD_V6)
    main_edits = [e for e in edits if e[0] == 'build_main']
    supp_edits = [e for e in edits if e[0] == 'build_supp']

    base = paragraphs(BASE)
    ours = paragraphs(OURS)
    full_base, full_ours = '\n'.join(base), '\n'.join(ours)

    print('base (his)  %-44s %4d paragraphs' % (BASE.name, len(base)))
    print('ours        %-44s %4d paragraphs' % (OURS.name, len(ours)))
    print('edits parsed from _build_v6.py: %d main, %d supplementary, '
          '%d skipped (non-literal)\n' % (len(main_edits), len(supp_edits), skipped))

    buckets = {}
    for fn, i, old, new in main_edits:
        buckets.setdefault(classify(full_base, old, new), []).append((i, old, new))

    order = ['applies cleanly', 'already applied', 'ambiguous', 'conflict']
    counts = Counter({k: len(v) for k, v in buckets.items()})
    print('MAIN TEXT EDITS vs V9')
    for k in order:
        print('  %-16s %3d' % (k, counts.get(k, 0)))

    for k in ('ambiguous',):
        rows = buckets.get(k, [])
        if rows:
            print('\n%s -- occurs more than once, needs a human to pick:' % k.upper())
            for i, old, new in rows:
                print('  [p%-4d] %s' % (i, old[:96]))

    rows = buckets.get('conflict', [])
    if rows:
        print('\nCONFLICTS, split by whether the sentence survives in V9')
        reworded, gone = [], []
        for i, old, new in rows:
            score, where, snippet = nearest(base, old)
            (reworded if score >= 0.60 else gone).append(
                (i, old, new, score, where, snippet))

        print('\n  REWORDED in V9 (%d) -- our edit and his rewrite overlap; '
              'merge by hand' % len(reworded))
        for i, old, new, score, where, snip in sorted(reworded, key=lambda r: -r[3]):
            print('    [p%-4d match %.0f%% at V9 p%d]' % (i, score * 100, where))
            print('        ours wanted : %s' % clip(old))
            print('        V9 now says : %s' % clip(snip))
            print('        our new text: %s' % (clip(new) if new else '(delete)'))

        print('\n  NOT IN V9 (%d) -- the target text is absent; check whether the '
              'whole passage was cut' % len(gone))
        for i, old, new, score, where, snip in sorted(gone, key=lambda r: -r[3]):
            print('    [p%-4d best match only %.0f%%]  %s' % (i, score * 100, clip(old)))

    # V5's edits are index-addressed, so check them by their content signature
    print('\nV5 EDITS (index-addressed -- checked by content)')
    v5 = [('Figure S4 -> S12 cross-ref', 'Figure S12', 'Figure S4'),
          ('Figure S3 -> S2 cross-ref', 'Figure S2', 'Figure S3'),
          ('editorial annotation "(LARGER font size)"', None, 'LARGER font size'),
          ('editorial annotation "(Larger font)"', None, 'Larger font'),
          ('editorial annotation "Redraw this"', None, 'Redraw this'),
          ('editorial annotation "redraw the figures"', None, 'redraw the figures')]
    for label, want, unwanted in v5:
        has_want = (want in full_base) if want else None
        has_bad = unwanted in full_base
        if want:
            state = 'already applied' if has_want and not has_bad else (
                'needs replay' if has_bad else 'target text absent')
        else:
            state = 'still present -- needs removal' if has_bad else 'already removed'
        print('  %-46s %s' % (label, state))

    print('\nFIGURE SWAPS (V5 swapped 8 images by rId; V6 rebuilt Figure 2)')
    print('  image count  his V9: %d   ours V6: %d'
          % (n_images(BASE), n_images(OURS)))
    print('  -> image identity cannot be checked by count alone; compare blobs '
          'before assuming his figures are the redrawn ones.')

    print('\nNothing was written. Resolve the conflicts above, then build V7.')


def n_images(path: Path) -> int:
    z = zipfile.ZipFile(path)
    return sum(1 for n in z.namelist() if n.startswith('word/media/'))


if __name__ == '__main__':
    main()
