"""全部 OFF(rec_back_guard 0・rec_hold_keep 0)の今の版が、写し f5b(fix5 の前)・f6b(fix6 の前)・f7b(fix7 の前)・
r1(fix7 r2 の前)の OFF とビット単位で同じか(PCA9685 に書いた値・記録の行・出来事)を、exp11 の fz の場面で照らす。
発進の前進・天井・自動合わせは場面の形(回 1〜4)のまま。
    python -I -B idcheck11.py <seed> <count> [i/n]   → out/idcheck11_<seed>_<part>.json
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', 'id_' + '_'.join(a.replace('/', 'of') for a in sys.argv[1:4]))
sys.path.insert(0, str(HERE))
import h11  # noqa: E402
import exp11  # noqa: E402

VERS = ('f7b', 'r1', 'f6b', 'f5b')


def main(seed, count, part='0/1'):
    h11.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    out = {'seed': seed, 'count': count, 'part': part, 'rows': []}
    t0 = time.time()
    try:
        for i, sd in enumerate(exp11.gen_fz(int(seed), int(count))):
            if i % pn != pi:
                continue
            p = h11.OFF(exp11.params_of(sd))
            r_now = h11.run('now', p, exp11.build(sd), wrap=False)
            row = {'i': i, 'form': sd['form']}
            for v in VERS:
                ok, why = h11.same_run(r_now, h11.run(v, p, exp11.build(sd), wrap=False))
                row[v] = ok
                if not ok:
                    row[v + '_why'] = str(why)[:300]
            out['rows'].append(row)
            if not all(row[v] for v in VERS):
                print(i, row, flush=True)
    finally:
        h11.cleanup()
        (HERE / 'out' / ('idcheck11_%s_%s.json' % (seed, part.replace('/', 'of')))).write_text(
            json.dumps(out, ensure_ascii=False))
    n = len(out['rows'])
    print('done', n, {v: sum(1 for r in out['rows'] if r[v]) for v in VERS}, round(time.time() - t0), 's')


if __name__ == '__main__':
    main(*sys.argv[1:4])
