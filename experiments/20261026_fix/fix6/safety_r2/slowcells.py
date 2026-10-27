"""中立 0.12 秒・後退側から 0.10 秒の偽物の ESC で、指摘 1 の形の 15 組(反証の probe_false_ready_slow で fix6 r1 が「0 m・壁押し」に
なった組)を fix5 のあと(before)・fix6 r1(r1)・今(final)で流す(読むだけ)"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2 as E   # noqa: E402
CELLS = [((2.11, 2.26), (2.26, 2.43)), ((2.11, 2.26), (2.26, 2.45)), ((2.11, 2.26), (2.26, 2.47)), ((2.11, 2.26), (2.26, 2.49)),
         ((2.11, 2.26), (2.29, 2.46)), ((2.11, 2.26), (2.29, 2.48)), ((2.11, 2.26), (2.29, 2.5)), ((2.11, 2.26), (2.29, 2.52)),
         ((2.11, 2.28), (2.28, 2.45)), ((2.11, 2.28), (2.28, 2.47)), ((2.11, 2.28), (2.28, 2.49)), ((2.14, 2.29), (2.29, 2.46)),
         ((2.14, 2.29), (2.29, 2.48)), ((2.14, 2.29), (2.29, 2.5)), ((2.14, 2.29), (2.29, 2.52))]
out = {}
try:
    p = E.P1()
    out['ng'] = E.metrics(E.run('final', p, E.scene('rev'), esc=(0.12, 0.10)), p)['back']
    for a, b in CELLS:
        sc = E.scene('rev', skips=[b], drops=[a])
        row = {v: E.metrics(E.run(v, p, sc, esc=(0.12, 0.10)), p) for v in ('before', 'r1', 'final')}
        out['A=%s B=%s' % (a, b)] = {v: (m['back'], m['push']) for v, m in row.items()}
        print(a, b, out['A=%s B=%s' % (a, b)], flush=True)
finally:
    E.cleanup()
(HERE / 'out' / 'slowcells.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print('ng', out['ng'])
