"""Build a tracked-changes copy of the V18 manuscript from edits_text.py.

Usage: python build.py <unpacked_dir_in> <unpacked_dir_out> <repo_root>
The input directory is the user's docx unzipped, untouched. Only the listed paragraphs
are rebuilt; every other byte of document.xml stays as it was.
"""
import copy, difflib, re, shutil, struct, sys
from pathlib import Path
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from edits_text import EDITS, DELETE_MARK, NEW_PARAS, FIGURES

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
XML = 'http://www.w3.org/XML/1998/namespace'
q = lambda t: f'{{{W}}}{t}'
LN = lambda e: etree.QName(e).localname
AUTHOR = 'Aditya Deole'
DATE = '2026-11-08T04:10:00Z'

src, dst, repo = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
docpath = dst / 'word' / 'document.xml'
tree = etree.parse(str(docpath))
root = tree.getroot()
body = root.find(q('body'))
PS = list(body.iter(q('p')))

_ids = [int(x) for x in re.findall(r'w:id="(-?\d+)"', docpath.read_text(encoding='utf8'))]
_next = [max(_ids) + 1000]
def nid():
    _next[0] += 1
    return str(_next[0])

def track(tag):
    e = etree.Element(q(tag))
    e.set(q('id'), nid()); e.set(q('author'), AUTHOR); e.set(q('date'), DATE)
    return e

# ---------------------------------------------------------------- formatting keys
def on(rp, tag):
    if rp is None: return False
    e = rp.find(q(tag))
    return e is not None and e.get(q('val')) not in ('0', 'false', 'off')

def fmtkey_rpr(rp):
    va = ''
    if rp is not None:
        v = rp.find(q('vertAlign'))
        if v is not None: va = v.get(q('val'))
    return (va, on(rp, 'i'), on(rp, 'b'))

RPR_ORDER = ['ins', 'del', 'moveFrom', 'moveTo', 'rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps',
             'smallCaps', 'strike', 'dstrike', 'outline', 'shadow', 'emboss', 'imprint', 'noProof',
             'snapToGrid', 'vanish', 'webHidden', 'color', 'spacing', 'w', 'kern', 'position', 'sz',
             'szCs', 'highlight', 'u', 'effect', 'bdr', 'shd', 'fitText', 'vertAlign', 'rtl', 'cs',
             'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath']

def set_child(rp, tag, val):
    """Set or remove a toggle/val child of rPr, keeping schema order."""
    old = rp.find(q(tag))
    if old is not None: rp.remove(old)
    if val is None: return
    e = etree.Element(q(tag))
    if val is not True: e.set(q('val'), val)
    pos = RPR_ORDER.index(tag)
    for k, c in enumerate(rp):
        n = LN(c)
        if n in RPR_ORDER and RPR_ORDER.index(n) > pos:
            rp.insert(k, e); return
    rp.append(e)

def rpr_for(template_rpr, fmt):
    rp = copy.deepcopy(template_rpr) if template_rpr is not None else etree.Element(q('rPr'))
    for t in ('ins', 'del', 'rPrChange'):
        for e in rp.findall(q(t)): rp.remove(e)
    va, it, bd = fmt
    if fmtkey_rpr(rp) != fmt:
        set_child(rp, 'vertAlign', va or None)
        set_child(rp, 'i', True if it else None); set_child(rp, 'iCs', True if it else None)
        set_child(rp, 'b', True if bd else None); set_child(rp, 'bCs', True if bd else None)
    assert fmtkey_rpr(rp) == fmt, (fmt, etree.tostring(rp))
    return rp

# ---------------------------------------------------------------- atoms
class Atom:
    __slots__ = ('kind', 'ch', 'elem', 'rpr', 'fmt', 'cont', 'field', 'ftype', 'state')
    def __init__(s, kind, ch=None, elem=None, rpr=None, cont=None):
        s.kind, s.ch, s.elem, s.rpr, s.cont = kind, ch, elem, rpr, cont
        s.fmt = fmtkey_rpr(rpr); s.field = None; s.ftype = None; s.state = 'keep'

def atoms_of(p):
    out = []
    def runs(container, cont):
        for c in container:
            n = LN(c)
            if n == 'pPr': continue
            if n == 'r':
                rp = c.find(q('rPr'))
                for x in c:
                    m = LN(x)
                    if m == 'rPr' or m == 'lastRenderedPageBreak': continue
                    if m == 't':
                        for ch in (x.text or ''): out.append(Atom('c', ch, None, rp, cont))
                    elif m == 'tab':
                        out.append(Atom('c', '\t', None, rp, cont))
                    elif m in ('fldChar', 'instrText', 'drawing'):
                        out.append(Atom('o', None, x, rp, cont))
                    else:
                        raise ValueError(f'unhandled run child {m}')
            elif n == 'hyperlink':
                runs(c, c)
            elif n in ('proofErr',):
                continue
            else:
                raise ValueError(f'unhandled paragraph child {n}')
    runs(p, None)
    # field bookkeeping
    stack, fid = [], 0
    for a in out:
        if a.kind == 'o' and LN(a.elem) == 'fldChar':
            t = a.elem.get(q('fldCharType')); a.ftype = t
            if t == 'begin':
                fid += 1; stack.append([fid, 'code']); a.field = fid
            elif t == 'separate':
                stack[-1][1] = 'result'; a.field = stack[-1][0]
            elif t == 'end':
                a.field = stack.pop()[0]
        elif a.kind == 'c':
            res = [f for f, st in stack if st == 'result']
            a.field = res[-1] if res else None
        else:
            a.field = stack[-1][0] if stack else None
    return out

def markup_of(chars):
    s, cur = [], ('', False, False)
    for a in chars:
        if a.fmt != cur:
            if cur != ('', False, False): s.append('}')
            if a.fmt != ('', False, False):
                va, it, bd = a.fmt
                s.append({'subscript': '_', 'superscript': '^'}.get(va, '') + ('**' if bd else '') + ('*' if it else '') + '{')
            cur = a.fmt
        s.append(a.ch)
    if cur != ('', False, False): s.append('}')
    return ''.join(s)

MK = re.compile(r'([_^]?)(\*\*)?(\*)?\{([^{}]*)\}')
def parse_markup(m):
    out, pos = [], 0
    for x in MK.finditer(m):
        for ch in m[pos:x.start()]: out.append((ch, ('', False, False)))
        va = {'_': 'subscript', '^': 'superscript'}.get(x.group(1), '')
        f = (va, bool(x.group(3)), bool(x.group(2)))
        for ch in x.group(4): out.append((ch, f))
        pos = x.end()
    for ch in m[pos:]: out.append((ch, ('', False, False)))
    assert '{' not in ''.join(c for c, _ in out) and '}' not in ''.join(c for c, _ in out), m[:80]
    return out

TOK = re.compile(r'\w+|\s+|[^\w\s]')
def tokens(seq):
    """seq: list of (ch, fmt). Returns list of (key, start, end) over seq indices."""
    out, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j][1] == seq[i][1]: j += 1
        s = ''.join(c for c, _ in seq[i:j])
        for m in TOK.finditer(s):
            out.append(((m.group(0), seq[i][1]), i + m.start(), i + m.end()))
        i = j
    return out

def opcodes(a_keys, b_keys, protected=frozenset()):
    ops = [list(o) for o in difflib.SequenceMatcher(None, a_keys, b_keys, autojunk=False).get_opcodes()]
    changed = True
    while changed:
        changed = False
        # absorb short equal islands between changes
        for k in range(1, len(ops) - 1):
            o = ops[k]
            if o[0] == 'equal' and ops[k-1][0] != 'equal' and ops[k+1][0] != 'equal':
                words = [t for t in a_keys[o[1]:o[2]] if t[0].strip()]
                if len(words) <= 3 and not any(t in protected for t in range(o[1], o[2])):
                    o[0] = 'replace'; changed = True
        merged = []
        for o in ops:
            if merged and o[0] != 'equal' and merged[-1][0] != 'equal':
                m = merged[-1]; merged[-1] = ['replace', m[1], o[2], m[3], o[4]]
            else:
                merged.append(o)
        ops = merged
    for o in ops:
        if o[0] == 'replace' and o[1] == o[2]: o[0] = 'insert'
        if o[0] == 'replace' and o[3] == o[4]: o[0] = 'delete'
    return ops

def groups_of(idx_list, keyf):
    """Consecutive runs of token indices sharing keyf -> list of (start, end)."""
    out = []
    for k in sorted(idx_list):
        if out and out[-1][1] == k and keyf(out[-1][1] - 1) == keyf(k):
            out[-1][1] = k + 1
        else:
            out.append([k, k + 1])
    return out

def anchored_opcodes(ot, nt, prot, chars):
    """Diff with old citation-field results pinned to the new superscript groups, in order."""
    if not prot:
        return opcodes([t[0] for t in ot], [t[0] for t in nt])
    og = groups_of(prot, lambda k: chars[ot[k][1]].field)
    nsup = [k for k, t in enumerate(nt) if t[0][1][0] == 'superscript']
    ng = groups_of(nsup, lambda k: 1)
    assert len(og) == len(ng), ('citation groups differ', len(og), len(ng))
    ops, pa, pb = [], 0, 0
    def seg(a1, a2, b1, b2, protect):
        if a1 == a2 and b1 == b2: return
        sub = opcodes([t[0] for t in ot[a1:a2]], [t[0] for t in nt[b1:b2]],
                      frozenset(range(a2 - a1)) if protect else frozenset())
        for tag, i1, i2, j1, j2 in sub:
            ops.append([tag, i1 + a1, i2 + a1, j1 + b1, j2 + b1])
    for (oa, ob), (na, nb) in zip(og, ng):
        seg(pa, oa, pb, na, False)
        seg(oa, ob, na, nb, True)
        pa, pb = ob, nb
    seg(pa, len(ot), pb, len(nt), False)
    return ops

# ---------------------------------------------------------------- paragraph rewrite
def rewrite(p, new_markup, guard):
    atoms = atoms_of(p)
    chars = [a for a in atoms if a.kind == 'c']
    old_m = markup_of(chars)
    assert old_m.replace('**{', '').startswith(guard) or ''.join(a.ch for a in chars).startswith(guard.replace('[IMG]', '')), (guard, old_m[:60])
    newseq = parse_markup(new_markup)
    oldseq = [(a.ch, a.fmt) for a in chars]
    ot, nt = tokens(oldseq), tokens(newseq)
    prot = {k for k, t in enumerate(ot) if any(chars[c].field is not None for c in range(t[1], t[2]))}
    ops = anchored_opcodes(ot, nt, prot, chars)
    after = {}      # char index -> list of new (ch, fmt) emitted right after that char
    bound = {}      # boundary char index -> list of new (ch, fmt)
    for tag, i1, i2, j1, j2 in ops:
        if tag in ('delete', 'replace'):
            for a in chars[ot[i1][1]:ot[i2-1][2]]: a.state = 'del'
        if tag in ('insert', 'replace'):
            ins = newseq[nt[j1][1]:nt[j2-1][2]]
            if tag == 'replace':
                after.setdefault(ot[i2-1][2] - 1, []).extend(ins)
            else:
                b = ot[i1][1] if i1 < len(ot) else len(chars)
                bound.setdefault(b, []).extend(ins)
    # no field may lose its whole result
    for f in {a.field for a in chars if a.field}:
        res = [a for a in chars if a.field == f]
        assert any(a.state == 'keep' for a in res) or any(1 for k in after if chars[k].field == f), f'field {f} result fully deleted'
    cidx = {id(a): k for k, a in enumerate(chars)}
    # build output stream
    out = []   # tuples: ('atom', atom) or ('ins', ch, fmt, template_rpr, cont)

    def neighbour_kept(k, step):
        while 0 <= k < len(chars):
            if chars[k].state == 'keep': return chars[k]
            k += step
        return None

    def template(k_before, k_after, fmt):
        cands = []
        for k in (k_before, k_after):
            if k is not None and 0 <= k < len(chars): cands.append(chars[k])
        for c in cands:
            if c.fmt == fmt: return c.rpr
        best = None
        for d in range(1, len(chars) + 1):
            for k in (k_before - d if k_before is not None else -1, (k_after or 0) + d):
                if 0 <= k < len(chars) and chars[k].fmt == fmt:
                    best = chars[k].rpr; break
            if best is not None: break
        if best is not None: return best
        return cands[0].rpr if cands else None

    def emit_ins(seq, kb, ka):
        kb_keep = neighbour_kept(kb, -1) if kb is not None and kb >= 0 else None
        ka_keep = neighbour_kept(ka, +1) if ka is not None and ka < len(chars) else None
        cont = kb_keep.cont if (kb_keep is not None and ka_keep is not None and kb_keep.cont is ka_keep.cont) else None
        for ch, f in seq:
            out.append(('ins', ch, f, template(kb, ka, f), cont))

    # boundary insertions need placement relative to opaque atoms between chars
    order = list(atoms)
    pending_bound = dict(bound)
    def flush_boundary(b):
        if b in pending_bound:
            emit_ins(pending_bound.pop(b), b - 1, b)

    # boundary 0 before any char: after leading drawings / field ends
    k_char = 0
    i = 0
    n = len(order)
    while i < n:
        a = order[i]
        if a.kind == 'c':
            b = cidx[id(a)]
            # insertion at boundary b goes before this char unless already placed
            flush_boundary(b)
            out.append(('atom', a))
            if b in after:
                emit_ins(after.pop(b), b, b + 1)
            # boundary b+1: decide placement now if the inserted text continues this char's field/format
            nb = b + 1
            if nb in pending_bound:
                seq = pending_bound[nb]
                if a.field is not None and all(f == a.fmt for _, f in seq):
                    emit_ins(pending_bound.pop(nb), b, nb)
                else:
                    # skip over following field ends / drawings, then place
                    j = i + 1
                    while j < n and order[j].kind == 'o' and (order[j].ftype == 'end' or LN(order[j].elem) == 'drawing'):
                        out.append(('atom', order[j])); j += 1
                    emit_ins(pending_bound.pop(nb), b, nb)
                    i = j; continue
            i += 1
        else:
            if 0 in pending_bound and k_char == 0 and not any(x.kind == 'c' for x in order[:i]):
                # leading opaque atoms: keep drawings/ends before a boundary-0 insertion
                if not (a.ftype == 'end' or LN(a.elem) == 'drawing'):
                    flush_boundary(0)
            out.append(('atom', a)); i += 1
    for b in sorted(pending_bound):
        emit_ins(pending_bound[b], b - 1, b)
    pending_bound.clear()
    assert not after, after

    # ---- serialise: group by container, then state, then run formatting
    for c in list(p):
        if LN(c) != 'pPr': p.remove(c)

    def item_state(it):
        return 'ins' if it[0] == 'ins' else it[1].state
    def item_cont(it):
        return it[4] if it[0] == 'ins' else it[1].cont
    def item_rpr_key(it):
        if it[0] == 'ins':
            return ('ins', id(it[3]), it[2])
        a = it[1]
        return ('old', id(a.rpr), a.fmt) if a.kind == 'c' else ('opaque', id(a))

    def make_run(items, state):
        it0 = items[0]
        if it0[0] == 'ins':
            rp = rpr_for(it0[3], it0[2])
        else:
            rp = copy.deepcopy(it0[1].rpr) if it0[1].rpr is not None else None
        r = etree.Element(q('r'))
        if rp is not None and len(rp): r.append(rp)
        buf = []
        def flush():
            if buf:
                t = etree.SubElement(r, q('delText' if state == 'del' else 't'))
                t.text = ''.join(buf); t.set(f'{{{XML}}}space', 'preserve'); buf.clear()
        for it in items:
            if it[0] == 'ins':
                ch = it[1]
            else:
                a = it[1]
                if a.kind == 'o':
                    flush()
                    e = copy.deepcopy(a.elem)
                    if state == 'del' and LN(e) == 'instrText': e.tag = q('delInstrText')
                    r.append(e); continue
                ch = a.ch
            if ch == '\t':
                flush(); etree.SubElement(r, q('tab'))
            else:
                buf.append(ch)
        flush()
        return r

    groups = []   # (cont, state, [items]) with run sub-grouping later
    for it in out:
        key = (id(item_cont(it)) if item_cont(it) is not None else None, item_state(it))
        if groups and groups[-1][0] == key:
            groups[-1][2].append(it)
        else:
            groups.append((key, item_cont(it), [it]))
    cont_el, cont_src = None, None
    for key, cont, items in groups:
        state = key[1]
        runs = []
        cur = []
        for it in items:
            if cur and item_rpr_key(cur[-1]) != item_rpr_key(it):
                runs.append(cur); cur = []
            cur.append(it)
        if cur: runs.append(cur)
        rel = [make_run(rr, state) for rr in runs]
        if state in ('ins', 'del'):
            w = track(state)
            for r in rel: w.append(r)
            rel = [w]
        if cont is not None:
            if cont_src is not cont:
                cont_el = etree.Element(cont.tag, attrib=dict(cont.attrib)); p.append(cont_el); cont_src = cont
            for e in rel: cont_el.append(e)
        else:
            cont_src = None
            for e in rel: p.append(e)
    return old_m

def apply_subs(p, subs, guard):
    chars = [a for a in atoms_of(p) if a.kind == 'c']
    m = markup_of(chars)
    assert m.startswith(guard), (guard, m[:40])
    for a, b in subs:
        assert m.count(a) == 1, (a, m)
        m = m.replace(a, b)
    return m

def mark_para(p, kind):
    ppr = p.find(q('pPr'))
    if ppr is None:
        ppr = etree.Element(q('pPr')); p.insert(0, ppr)
    rpr = ppr.find(q('rPr'))
    if rpr is None:
        rpr = etree.Element(q('rPr'))
        tail = [c for c in ppr if LN(c) in ('sectPr', 'pPrChange')]
        if tail: tail[0].addprevious(rpr)
        else: ppr.append(rpr)
    rpr.insert(0, track(kind))

def new_para(template_p, markup):
    np_ = etree.Element(q('p'))
    tp = template_p.find(q('pPr'))
    if tp is not None:
        ppr = copy.deepcopy(tp)
        for e in ppr.findall(q('rPr')): ppr.remove(e)
        for e in ppr.findall(q('pPrChange')): ppr.remove(e)
        np_.append(ppr)
    mark_para(np_, 'ins')
    trpr = None
    for r in template_p.iter(q('r')):
        if r.find(q('t')) is not None:
            trpr = r.find(q('rPr')); break
    seq = parse_markup(markup)
    w = track('ins')
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j][1] == seq[i][1]: j += 1
        r = etree.SubElement(w, q('r'))
        r.append(rpr_for(trpr, seq[i][1]))
        buf = ''
        for ch, _ in seq[i:j]:
            if ch == '\t':
                if buf:
                    t = etree.SubElement(r, q('t')); t.text = buf; t.set(f'{{{XML}}}space', 'preserve'); buf = ''
                etree.SubElement(r, q('tab'))
            else:
                buf += ch
        if buf:
            t = etree.SubElement(r, q('t')); t.text = buf; t.set(f'{{{XML}}}space', 'preserve')
        i = j
    np_.append(w)
    return np_

# ---------------------------------------------------------------- figures
def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    assert head[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', head[16:24])

rels_path = dst / 'word' / '_rels' / 'document.xml.rels'
rels = etree.parse(str(rels_path))
rel_ids = {e.get('Id') for e in rels.getroot()}
doc_pr_ids = {int(e.get('id')) for e in root.iter(f'{{{WP}}}docPr')}

def swap_figure(p, img):
    runs = [r for r in p.iter(q('r')) if r.find(q('drawing')) is not None]
    assert len(runs) == 1
    r = runs[0]
    parent = r.getparent()
    assert LN(parent) == 'p', LN(parent)
    # new media + relationship
    n = len(list((dst / 'word' / 'media').iterdir())) + 1
    name = f'image_v18r{n}.png'
    shutil.copy(img, dst / 'word' / 'media' / name)
    rid = f'rIdV18r{n}'
    assert rid not in rel_ids
    etree.SubElement(rels.getroot(), '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship',
                     Id=rid, Type=f'{R}/image', Target=f'media/{name}')
    newr = copy.deepcopy(r)
    for e in newr.findall(q('lastRenderedPageBreak')): newr.remove(e)
    for t in newr.findall(q('t')): newr.remove(t)
    d = newr.find(q('drawing'))
    wpx, hpx = png_size(img)
    ext = d.find(f'.//{{{WP}}}extent')
    cx = int(ext.get('cx')); cy = round(cx * hpx / wpx)
    ext.set('cy', str(cy))
    for e in d.iter(f'{{{A}}}ext'):
        if e.get('cx') is not None:
            e.set('cx', str(cx)); e.set('cy', str(cy))
    did = max(doc_pr_ids) + 1; doc_pr_ids.add(did)
    dp = d.find(f'.//{{{WP}}}docPr'); dp.set('id', str(did)); dp.set('name', f'Picture {did}')
    for e in d.iter(f'{{{PIC}}}cNvPr'): e.set('name', name)
    for e in d.iter(f'{{{A}}}blip'): e.set(f'{{{R}}}embed', rid)
    for e in d.iter(f'{{{A}}}srcRect'): e.attrib.clear()   # new file is shown whole, no crop
    # strip the drawing run of any text (text atoms were in their own runs after rewrite)
    assert r.find(q('t')) is None and r.find(q('delText')) is None
    # wrap old in del, put new in ins right after
    idx = parent.index(r)
    dw = track('del'); parent.remove(r)
    old = copy.deepcopy(r)
    for e in old.findall(q('lastRenderedPageBreak')): old.remove(e)
    dw.append(old)
    iw = track('ins'); iw.append(newr)
    parent.insert(idx, dw); parent.insert(idx + 1, iw)
    return cx, cy, wpx, hpx

# ---------------------------------------------------------------- run
log = []
for i, spec in sorted(EDITS.items()):
    p = PS[i]
    guard = spec[0]
    if spec[1] is None:
        new_m = apply_subs(p, spec[2], guard)
    else:
        new_m = spec[1]
    old_m = rewrite(p, new_m, guard)
    log.append((i, old_m == new_m))
for i in DELETE_MARK:
    mark_para(PS[i], 'del')
# new paragraphs, grouped by anchor so order is preserved
by_anchor = {}
for after, tmpl, m in NEW_PARAS:
    by_anchor.setdefault(after, []).append((tmpl, m))
for after, items in by_anchor.items():
    anchor = PS[after]
    for tmpl, m in items:
        np_ = new_para(PS[tmpl], m)
        anchor.addnext(np_); anchor = np_
for i, rel in FIGURES.items():
    print('figure', i, swap_figure(PS[i], repo / rel))

tree.write(str(docpath), xml_declaration=True, encoding='UTF-8', standalone=True)
rels.write(str(rels_path), xml_declaration=True, encoding='UTF-8', standalone=True)
for i, same in log:
    if same: print('WARNING: no change in paragraph', i)
print('edited paragraphs:', len(log), '| new paragraphs:', len(NEW_PARAS), '| figures:', len(FIGURES))
