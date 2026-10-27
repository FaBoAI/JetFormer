"""fix_safety_r2.md 4 章の 2 つ目('gap' の惰性を下がった時間に数える見張り + 見積もり → 後退が短い。fr_rev の 2 組)を、
fix6 r2 の exp2(その係の場面・値そのまま)で ON・OFF・fix5 のあと(before)と並べる(読むだけ)。出力 out/gapcoast.json"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'fix6' / 'safety_r2'))
os.environ.setdefault('R2_TAG', 'fix7_rv')
import exp2 as E   # noqa: E402
E.TMP = HERE.parents[1] / 'fix7' / 'tmp_review_safety_r1' / 'gapcoast'
CELLS = [((2.09, 2.3), (2.36, 2.53)), ((2.09, 2.3), (2.4, 2.57)), ((2.12, 2.29), (2.29, 2.49))]
out = {}
try:
    for a, b in CELLS:
        for rear in (None, 1.2, 1.4):
            sc = E.scene('rev', skips=[b], drops=[a], rear=rear)
            row = {}
            for v, mod, p in (('on', 'now', E.P1()), ('off', 'now', E.P1(rec_back_guard=0, rec_hold_keep=0)),
                              ('before_on', 'before', E.P1())):
                m = E.metrics(E.run(mod, p, sc), p)
                row[v] = {k: m[k] for k in ('back', 'push', 'front', 'rear', 'retry', 'guards')} if m else None
            key = 'A=%s B=%s rear=%s' % (a, b, rear)
            out[key] = row
            print(key, {k: (v['back'], v['push'], v['rear']) if v else None for k, v in row.items()}, flush=True)
finally:
    shutil.rmtree(E.TMP, ignore_errors=True)
(HERE / 'out' / 'gapcoast.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
