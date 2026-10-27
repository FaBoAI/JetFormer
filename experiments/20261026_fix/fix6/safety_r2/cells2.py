"""決め手になる数場面を、候補ごとに流して下がった距離・壁押しを並べる(読むだけ)。
    python -I cells2.py variant [variant ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2 as E   # noqa: E402

T_ON = {'rev': 2.60, 'brake': 2.73}
CELLS = {
    'f1': ('rev', None, [(2.26, 2.45)], [(2.11, 2.26)]),
    'f1_c40': ('rev', None, [(2.32, 2.49)], [(2.09, 2.26)]),
    'f1_c_0.55': ('rev', None, [(2.4, 2.7)], [(2.09, 2.3)]),
    'f2': ('rev', (0.2, 0.08), [(2.52, 2.72)], []),
    'brake_true1': ('brake', None, [(2.87, 3.17)], [(2.68, 2.81)]),
    'brake_true2': ('brake', None, [(2.63, 3.08)], []),
}
for d0, L in [(-0.06, 0.45), (-0.06, 0.5), (-0.06, 0.55), (-0.04, 0.55), (-0.06, 0.4), (-0.04, 0.45), (-0.02, 0.5),
              (0.0, 0.55)]:
    t0 = T_ON['brake'] + d0
    CELLS['w61b_%+.2f_%.2f' % (d0, L)] = ('brake', None, [(round(t0, 3), round(t0 + L, 3))], [])
    t0 = T_ON['rev'] + d0
    CELLS['w61r_%+.2f_%.2f' % (d0, L)] = ('rev', None, [(round(t0, 3), round(t0 + L, 3))], [])


def main(vs):
    out = {}
    try:
        p = E.P1()
        for k, (kind, esc, sk, dr) in CELLS.items():
            row = {}
            for v in vs:
                name, fl = (v.split(':', 1) + [''])[:2] if ':' in v else (v, '')
                m = E.metrics(E.run(name, p, E.scene(kind, skips=sk, drops=dr), esc=esc, flags=fl), p)
                row[v] = (m['back'], m['push'])
            out[k] = row
            print('%-16s' % k, '  '.join('%s=%s/%s' % (v, *row[v]) for v in vs), flush=True)
    finally:
        E.cleanup()
    (HERE / 'out' / 'cells2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
