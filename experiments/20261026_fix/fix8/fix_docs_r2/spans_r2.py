"""読むだけ: 今の文書の中で、前の文書の台本(試験のある fix7 docs r1・r2・仕上げ・fix8 docs-decided・fix8 docs r1・fix8 docs 同期 1)の
直し(new。あとの台本が書き換えたあとの今の形)が占める所を出す(fix8/docs_sync1/spans.py に同期 1 を足した版)。
直したい文(old)の始め・終わりが前の new の中なら、その持ち主を出す(前の new の中を変えると、その台本の試験の「new がちょうど 1 回」が
落ちる = SUPERSEDE にして revert・effective の鎖に足す)。

使い方: python3 -I -B spans_r2.py '<文書>' '<文>' ...(文に ‸ があれば、その位置が前の new の中か)
"""
import importlib.util
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
E = JF / 'experiments' / '20261026_fix'
REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
SCRIPTS = {'r1': E / 'fix7/docs_r1/docs_fix7_r1.py', 'r2': E / 'fix7/docs_r2/docs_fix7_r2.py',
           'final': E / 'fix7/docs_final/docs_fix7_final.py', 'decided': E / 'fix8/docs_decided/docs_fix8_decided.py',
           'r81': E / 'fix8/fix_docs_r1/docs_fix8_r1.py', 's1': E / 'fix8/docs_sync1/docs_fix8_sync1.py'}
FILES = ('runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix3/disk.md', 'fix7/README.md',
         'fix6/README.md')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MODS = {k: _load('sp2_' + k, p) for k, p in SCRIPTS.items()}


def eff_new(k, f, i, new):
    """前の台本 k の (f, i) の new の、今の形(あとの台本の書き換えのあと。同期 1 の prev_new → effective と同じ鎖)"""
    if k in ('r1', 'r2', 'final'):
        new = MODS['decided'].effective(k, f, i, new)
    if k in ('final', 'decided'):
        new = MODS['r81'].effective(k, f, i, new)
    if k != 's1':
        new = MODS['s1'].effective(k, f, i, new)
    return new


def spans(texts):
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
    return [(k, i) for s, e, k, i in sp.get(f, []) if s < pos < e]


def owners(texts, f, old):
    """old(文書にちょうど 1 回)の始め・終わり・中が、どの前の new に掛かるか"""
    sp, _ = spans(texts)
    t = texts[f]
    a = t.index(old)
    b = a + len(old)
    return sorted({(k, i) for s, e, k, i in sp.get(f, []) if s < b and a < e})


if __name__ == '__main__':
    texts = {f: (REPORTS / f).read_text(encoding='utf-8') for f in FILES}
    sp, bad = spans(texts)
    print('new が 1 回でない:', bad)
    args = sys.argv[1:]
    for f, x in zip(args[::2], args[1::2]):
        t = texts[f]
        if '‸' in x:
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
        print(f, '|', x[:50].replace('\n', '⏎'), '→ 掛かる new', owners(texts, f, x) or 'なし')
