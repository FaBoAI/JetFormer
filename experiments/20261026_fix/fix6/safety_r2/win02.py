"""rev_s 0.2・rev_us 300 の fix6 r1 の指摘の窓(16 場面 × rev・brake)を fix5 のあと(before)・fix6 r1(r1)・今(final)で(読むだけ)"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2 as E   # noqa: E402
T_ON = {'rev': 2.60, 'brake': 2.73}
out = {}
try:
    for kind in ('rev', 'brake'):
        for rs, ru in ((0.2, 300.0), (0.3, 300.0), (0.4, 200.0)):
            p = E.P1(rev_s=rs, rev_us=ru)
            ref = E.metrics(E.run('final', p, E.scene(kind)), p)['back']
            for d0 in (-0.06, -0.04, -0.02, 0.0):
                for L in (0.45, 0.5, 0.55, 0.6):
                    t0 = T_ON[kind] + d0
                    sc = E.scene(kind, skips=[(round(t0, 3), round(t0 + L, 3))])
                    row = {v: E.metrics(E.run(v, p, sc), p) for v in ('before', 'r1', 'final')}
                    k = '%s rev_s=%.1f rev_us=%d d0=%+.2f L=%.2f' % (kind, rs, ru, d0, L)
                    out[k] = {'ref': ref, **{v: (m['back'], m['push'], m['guards'][:2]) for v, m in row.items()}}
                    print(k, 'ref', ref, {v: (m['back'], m['push']) for v, m in row.items()}, flush=True)
finally:
    E.cleanup()
(HERE / 'out' / 'win02.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
