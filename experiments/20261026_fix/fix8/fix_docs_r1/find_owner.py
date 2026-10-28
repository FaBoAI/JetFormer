"""読むだけ: 文書の文 X が、前の文書の台本(fix7 docs・r1・r2・仕上げ・fix8 docs-decided)のどの直し(new)の中にあるかを出す。
fix7 の台本の new は fix8 docs-decided の書き換え(effective)のあとの形で見る。"""
import importlib.util
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
E = JF / 'experiments' / '20261026_fix'
SCRIPTS = {'docs7': E / 'fix7/docs/docs_fix7.py', 'r1': E / 'fix7/docs_r1/docs_fix7_r1.py', 'r2': E / 'fix7/docs_r2/docs_fix7_r2.py',
           'final': E / 'fix7/docs_final/docs_fix7_final.py', 'decided': E / 'fix8/docs_decided/docs_fix8_decided.py'}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MODS = {k: _load('fo_' + k, p) for k, p in SCRIPTS.items()}
DEC = MODS['decided']


def owners(f, x):
    out = []
    for k, m in MODS.items():
        for ff, edits in m.ALL:
            if ff != f:
                continue
            for i, (old, new) in enumerate(edits):
                n = DEC.effective(k, ff, i, new) if k in ('r1', 'r2', 'final') else new
                if x in n:
                    out.append((k, i, 'whole' if n == x else 'in'))
    return out


if __name__ == '__main__':
    import json
    for f, x in json.load(open(sys.argv[1], encoding='utf-8')):
        print(f, '|', x[:50].replace('\n', '⏎'), '→', owners(f, x))
