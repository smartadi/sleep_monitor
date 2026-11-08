"""Check the tracked copy: accepted view == intended markup, rejected view == original.

Usage: python verify.py <orig_unpacked> <out_unpacked>
"""
import sys
from pathlib import Path
from lxml import etree
sys.path.insert(0, str(Path(__file__).parent))
from edits_text import EDITS, NEW_PARAS

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
q = lambda t: f'{{{W}}}{t}'
LN = lambda e: etree.QName(e).localname

def fmt(r):
    rp = r.find(q('rPr'))
    def on(t):
        e = rp.find(q(t)) if rp is not None else None
        return e is not None and e.get(q('val')) not in ('0', 'false', 'off')
    va = ''
    if rp is not None and rp.find(q('vertAlign')) is not None:
        va = rp.find(q('vertAlign')).get(q('val'))
    return (va, on('i'), on('b'))

def view(p, mode):
    """mode 'new' = all changes accepted, 'old' = all changes rejected. Returns markup."""
    seq = []
    for r in p.iter(q('r')):
        anc = {LN(a) for a in r.iterancestors()}
        if mode == 'new' and anc & {'del', 'moveFrom'}: continue
        if mode == 'old' and anc & {'ins', 'moveTo'}: continue
        f = fmt(r)
        for x in r:
            n = LN(x)
            if n in ('t', 'delText'):
                seq += [(c, f) for c in (x.text or '')]
            elif n == 'tab':
                seq.append(('\t', f))
            elif n == 'drawing':
                seq.append(('[IMG]', ('', False, False)))
    s, cur = [], ('', False, False)
    for c, f in seq:
        if f != cur:
            if cur != ('', False, False): s.append('}')
            if f != ('', False, False):
                s.append({'subscript': '_', 'superscript': '^'}.get(f[0], '') + ('**' if f[2] else '') + ('*' if f[1] else '') + '{')
            cur = f
        s.append(c)
    if cur != ('', False, False): s.append('}')
    return ''.join(s)

def para_marks(p):
    ppr = p.find(q('pPr'))
    rpr = ppr.find(q('rPr')) if ppr is not None else None
    if rpr is None: return set()
    return {LN(c) for c in rpr if LN(c) in ('ins', 'del') and c.get(q('date')) == '2026-11-08T04:10:00Z'}

o = etree.parse(sys.argv[1] + '/word/document.xml').getroot().find(q('body'))
n = etree.parse(sys.argv[2] + '/word/document.xml').getroot().find(q('body'))
OP = list(o.iter(q('p'))); NP = list(n.iter(q('p')))
# map: new paragraphs that carry an inserted mark are additions
added = [p for p in NP if 'ins' in para_marks(p) and not any(LN(a) == 'ins' for a in [])]
kept = [p for p in NP if 'ins' not in para_marks(p)]
assert len(kept) == len(OP), (len(kept), len(OP))
bad = 0
for i, (po, pn) in enumerate(zip(OP, kept)):
    old_o, old_n = view(po, 'old'), view(pn, 'old')
    if old_o != old_n:
        bad += 1; print(f'REJECTED VIEW DIFFERS P{i}\n  orig {old_o[:200]!r}\n  now  {old_n[:200]!r}')
    if i in EDITS:
        new = view(pn, 'new')
        spec = EDITS[i]
        if spec[1] is not None:
            want = spec[1]
        else:
            want = view(po, 'new')
            for a, b in spec[2]: want = want.replace(a, b)
        if i == 236 or i in (198, 205, 213, 219, 229):
            new = new.replace('[IMG]', '', 2) if new.count('[IMG]') else new
            want = want.replace('[IMG]', '')
        if new.replace('[IMG]', '') != want.replace('[IMG]', ''):
            bad += 1
            import difflib
            sm = difflib.SequenceMatcher(None, want, new)
            print(f'ACCEPTED VIEW DIFFERS P{i}')
            for t, a1, a2, b1, b2 in sm.get_opcodes():
                if t != 'equal': print('   ', t, repr(want[max(0,a1-20):a2+20]), '->', repr(new[max(0,b1-20):b2+20]))
    elif i not in EDITS and view(po, 'new') != view(pn, 'new'):
        bad += 1; print(f'UNEDITED PARAGRAPH CHANGED P{i}')
# figures: accepted view has one image, rejected has one image
for i in (198, 205, 213, 219, 229, 236):
    pn = kept[i]
    assert view(pn, 'new').count('[IMG]') == 1 and view(pn, 'old').count('[IMG]') == 1, i
ins_paras = [p for p in NP if 'ins' in para_marks(p)]
print('inserted paragraphs:', [view(p, 'new')[:40] for p in ins_paras])
print('deleted marks:', sum('del' in para_marks(p) for p in NP))
print('problems:', bad)
