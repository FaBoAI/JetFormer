"""1 つの場面を ON / OFF で流し、段の移り・書いた値・車の位置を並べる(調べ用)。
    python -I -B trace11.py '<場面の JSON>' [t_from] [t_to] [版 now|f7b|r1]
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', 'trace')
sys.path.insert(0, str(HERE))
import h11  # noqa: E402
import exp11  # noqa: E402

N, C, A = h11.N, h11.C, h11.A


def trace(sd, t_from=0.0, t_to=99.0, ver='now', variants=('on', 'off')):
    out = {}
    for v in variants:
        prm = h11.ON(exp11.params_of(sd)) if v.startswith('on') else h11.OFF(exp11.params_of(sd))
        res = h11.run(ver, prm, exp11.build(sd))
        sim, rec, pca, ph, stats = res
        m = h11.metrics(res, sd.get('ai_from', 1.0))
        out[v] = (res, m)
        print('====', v, json.dumps(exp11.slim(m), ensure_ascii=False, default=str))
        print('  guards', [(round(float(e[1]['cam_epoch_us']) / 1e6 - h11.clock.to_epoch_us(h11.T0) / 1e6, 3), e[1]['what'],
                           e[1]['phase'], e[1]['backed_s'], e[1].get('hold_s'))
                          for e in rec.events if e[0] == 'recovery_guard'])
        last = None
        for x in ph:
            if not (t_from <= x[0] <= t_to):
                continue
            key = (x[1], x[2], x[3])
            if x[1] != x[2] or x[3] or x[9] or abs(x[6]) > 300 or key != last:
                lg = next((r for r in sim.log if r[0] >= x[0] - 100 * 0 - 1e-9), None)
                print('   ph t=%.3f %s->%s hold=%s hb=%.3f backed=%.3f erpm=%.0f esc_rev=%s imp=%.2f hg=%.2f' % x[:10])
            last = key
        w = [(round(t, 3), round(th, 1)) for t, s, th in pca.w if t_from <= t <= t_to]
        # 書いた値の変わり目だけ
        prev = None
        seq = []
        for t, th in w:
            if th != prev:
                seq.append((t, th))
                prev = th
        print('  writes', seq[:80])
        xs = [(round(r[0], 2), round(r[1], 3), round(r[2], 2), r[4], r[5], r[6]) for r in sim.log
              if t_from <= r[0] <= t_to and int(round(r[0] * 100)) % 10 == 0]
        print('  car', xs[:120])
        print('  cam_timeouts', [t for t in sim.cam_timeouts if t_from <= t <= t_to][:30])
    return out


if __name__ == '__main__':
    sd = json.loads(sys.argv[1])
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    b = float(sys.argv[3]) if len(sys.argv) > 3 else 99.0
    ver = sys.argv[4] if len(sys.argv) > 4 else 'now'
    try:
        trace(sd, a, b, ver)
    finally:
        h11.cleanup()
