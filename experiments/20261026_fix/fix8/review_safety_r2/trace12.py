"""1 つの場面を ON / OFF で流し、段の移り・書いた値・車の位置・出来事を並べる(調べ用)。
    python -I -B trace12.py '<場面の JSON(exp12 の sd)>' [t_from] [t_to] [variants: on,off,base_off]
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS82_TAG', 'trace')
sys.path.insert(0, str(HERE))
import h12  # noqa: E402
import exp12 as E  # noqa: E402

h11 = h12.h11
N, C, A, T0 = h12.N, h12.C, h12.A, h12.T0


def trace(sd, t_from=0.0, t_to=99.0, variants=('on', 'off'), quiet=False, builder=None, params=None):
    build = builder or E.build
    out = {}
    for v in variants:
        sdv = dict(sd, pert=[]) if v.startswith('base') else sd
        p = params or E.params_of(sd)
        prm = h12.ON(p) if v.endswith('on') else h12.OFF(p)
        res = h12.run(prm, build(sdv))
        sim, rec, pca, ph, stats = res
        m = E.metr(res, dict(sdv, human_any=True))
        out[v] = (res, m)
        if quiet:
            continue
        print('====', v, json.dumps({k: m[k] for k in ('front_push_s', 'zero_push', 'giveup_t', 'recoveries', 'holds', 'back', 'vback',
                                                  'guards', 'rows_bad', 'marks')}, ensure_ascii=False, default=str))
        print('  prof', m['prof'])
        print('  hits', sim.car.hits[:6])
        print('  guards', [(round(float(e[1]['cam_epoch_us']) / 1e6 - h12.h11.clock.to_epoch_us(T0) / 1e6, 3), e[1]['what'],
                           e[1]['phase'], e[1]['backed_s'], e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard'])
        for x in ph:
            if t_from <= x[0] <= t_to:
                print('   ph t=%.3f %s->%s hold=%s hb=%.3f backed=%.3f erpm=%.0f esc_rev=%s imp=%.2f hg=%.2f' % x[:10])
        w = [(round(t, 3), round(th, 1)) for t, s, th in pca.w if t_from <= t <= t_to]
        print('  writes', w)
        lg = [(r[0], round(r[1], 3), round(r[2], 2), round(r[3]), r[4], r[5], r[6]) for r in sim.log
              if t_from <= r[0] <= t_to and abs(round(r[0] * 100) % 3) == 0]
        print('  car(t,x,v,thr,contact,rev_on,sel)', lg)
        print('  cam_timeouts', [t for t in sim.cam_timeouts if t_from <= t <= t_to])
    return out


if __name__ == '__main__':
    sd = json.loads(sys.argv[1])
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    b = float(sys.argv[3]) if len(sys.argv) > 3 else 99.0
    vs = tuple(sys.argv[4].split(',')) if len(sys.argv) > 4 else ('on', 'off')
    try:
        trace(sd, a, b, vs)
    finally:
        h12.cleanup()
