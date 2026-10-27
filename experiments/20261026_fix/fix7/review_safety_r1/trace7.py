"""1 つの場面の段の並び・出した値・車の速さを ON・OFF で出す(読むだけ)。
  python -I trace7.py <kind rev|brake> <skips "a,b;c,d"> <drops "a,b"> [rear] [t_from] [t_to] [extra-json]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
from grid_onoff import scene, first_rec  # noqa: E402

N = rv7.N


def parse(s):
    if not s or s == '-':
        return []
    return [tuple(float(x) for x in seg.split(',')) for seg in s.split(';')]


def main(kind, skips='-', drops='-', rear='None', t_from='2.0', t_to='4.5', extra='{}'):
    p = rv7.base_params()
    sc = scene(kind, skips=parse(skips), drops=parse(drops), rear=None if rear == 'None' else float(rear))
    sc.update(json.loads(extra))
    t_from, t_to = float(t_from), float(t_to)
    try:
        for nm, pp in (('ON', rv7.ON(p)), ('OFF', rv7.OFF(p))):
            res = rv7.run('now', pp, sc)
            sim, rec, pca, ph = res
            print('=====', nm, first_rec(res))
            rows = {round((t - rv7.T0) / 1e9, 3): r for t, r in rec.rows}
            phs = {}
            for t, p0, p1, hs, hr, es, bs, e in ph:
                phs[round(t, 3)] = (p0, p1, hs, hr, es, bs, e)
            for t, s, th in pca.w:
                if t_from <= t <= t_to:
                    x, v = next(((x, v) for tt, x, v, *_ in sim.log if tt >= t - 1e-9), (None, None))
                    k = min(phs, key=lambda q: abs(q - t)) if phs else None
                    info = phs[k] if k is not None and abs(k - t) < 0.02 else ''
                    print('%.3f thr %.0f x %.3f v %+.2f %s' % (t, th, x, v, info))
    finally:
        rv7.cleanup()


if __name__ == '__main__':
    main(*sys.argv[1:])
