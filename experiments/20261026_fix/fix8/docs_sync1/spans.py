"""読むだけ: 今の文書の中で、前の文書の台本(試験のある fix7 docs r1・r2・仕上げ・fix8 docs-decided・fix8 docs r1)の直し(new。
あとの台本が書き換えたあとの形)が占める所を出す。足す所(old の終わり・始め)が前の new の中なら、その new の持ち主を出す
(前の new の中を変えると、その台本の試験の「new がちょうど 1 回」が落ちる = SUPERSEDE にして revert・effective を足す要がある)。

使い方: python3 -I -B spans.py '<文書>' '<文の頭>' ...(その文の始め・終わりの位置を持つ new を出す)
"""
import importlib.util
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
E = JF / 'experiments' / '20261026_fix'
REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
SCRIPTS = {'r1': E / 'fix7/docs_r1/docs_fix7_r1.py', 'r2': E / 'fix7/docs_r2/docs_fix7_r2.py',
           'final': E / 'fix7/docs_final/docs_fix7_final.py', 'decided': E / 'fix8/docs_decided/docs_fix8_decided.py',
           'r81': E / 'fix8/fix_docs_r1/docs_fix8_r1.py'}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MODS = {k: _load('sp_' + k, p) for k, p in SCRIPTS.items()}
DEC, R81 = MODS['decided'], MODS['r81']


def eff_new(k, f, i, new):
    """前の台本 k の (f, i) の new の、今の形(あとの台本の書き換えのあと)"""
    if k in ('r1', 'r2', 'final'):
        new = DEC.effective(k, f, i, new)
    if k in ('final', 'decided'):
        new = R81.effective(k, f, i, new)
    return new


def spans(texts):
    """{文書: [(start, end, 台本, 何番目)]}。new が今の文書にちょうど 1 回ある所だけ(無い・2 回以上は bad に)"""
    out, bad = {}, []
    for k, m in MODS.items():
        for f, edits in m.ALL:
            if f not in texts:
                continue
            for i, (_, new) in enumerate(edits):
                n = eff_new(k, f, i, new)
                c = texts[f].count(n)
                if c != 1:
                    bad.append((k, f, i, c))
                    continue
                s = texts[f].index(n)
                out.setdefault(f, []).append((s, s + len(n), k, i))
    return out, bad


def inside(sp, f, pos):
    """pos が前の new の中(端は含まない)なら、その (台本, 何番目) の一覧"""
    return [(k, i) for s, e, k, i in sp.get(f, []) if s < pos < e]


if __name__ == '__main__':
    files = ('runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix3/disk.md', 'fix7/README.md',
             'fix6/README.md')
    texts = {f: (REPORTS / f).read_text(encoding='utf-8') for f in files}
    sp, bad = spans(texts)
    print('new が 1 回でない:', bad)
    args = sys.argv[1:]
    for f, x in zip(args[::2], args[1::2]):
        t = texts[f]
        if '‸' in x:                    # 足す所の印: 印の位置が前の new の中か
            y = x.replace('‸', '')
            if t.count(y) != 1:
                print(f, '|', y[:40], '→ 文書に %d 回' % t.count(y))
                continue
            pos = t.index(y) + x.index('‸')
            print(f, '|', x[:50].replace('\n', '⏎'), '→ 足す所', inside(sp, f, pos) or 'どの new の中でもない')
            continue
        if t.count(x) != 1:
            print(f, '|', x[:40], '→ 文書に %d 回' % t.count(x))
            continue
        a = t.index(x)
        print(f, '|', x[:40].replace('\n', '⏎'), '→ 始め', inside(sp, f, a), '・終わり', inside(sp, f, a + len(x)))


def broken_by(texts, f, old, new):
    """文書 f の old(ちょうど 1 回)を new にすると、前の new(今の形)がちょうど 1 回でなくなる (台本, 何番目) の一覧"""
    t = texts[f]
    assert t.count(old) == 1, (f, old[:40], t.count(old))
    t2 = t.replace(old, new, 1)
    out = []
    for k, m in MODS.items():
        for ff, edits in m.ALL:
            if ff != f:
                continue
            for i, (_, n0) in enumerate(edits):
                n = eff_new(k, ff, i, n0)
                if t.count(n) == 1 and t2.count(n) != 1:
                    out.append((k, i))
    return out
