"""fuzz_onoff の 1 場面を作り直して、ON・OFF(・ON の止まりなし)の段の並びと後ろの当たりを出す(読むだけ)。
  python -I replay_fz.py <種> <番号> [t_from] [t_to] [variants on,off,onc,on_f6b,on_f6r2b]"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
import fuzz_onoff as FZ  # noqa: E402

N = rv7.N


def scene_of(seed, idx):
    rng = random.Random(seed)
    for i in range(idx + 1):
        sc = FZ.FC.rand_scene(rng)
        p = FZ.rand_params(rng)
    return sc, p


def phases(ph):
    out, last = [], None
    for t, p0, p1, hs, hr, es, bs, e in ph:
        if p1 != last or hs:
            out.append((t, p0, p1, 'H' if hs else '', hr, es, bs, e))
            last = p1
    return out


def main(seed, idx, t_from='0', t_to='99', variants='on,off,onc'):
    seed, idx, t_from, t_to = int(seed), int(idx), float(t_from), float(t_to)
    sc, p = scene_of(seed, idx)
    print('params', {k: p[k] for k in ('launch_boost', 'speed_ceiling', 'speed_auto', 'ceil_left_mps', 'throttle_gain')})
    print('scene', {k: v for k, v in sc.items() if k != 'model'})
    try:
        for v in variants.split(','):
            mod = 'now'
            s2 = sc
            if v == 'on':
                pp = rv7.ON(p)
            elif v == 'off':
                pp = rv7.OFF(p)
            elif v == 'onc':
                pp, s2 = rv7.ON(p), FZ.clean(sc)
            elif v.startswith('on_'):
                pp, mod = rv7.ON(p), v[3:]
            sim, rec, pca, ph = rv7.run(mod, pp, s2)
            m = rv7.metrics(sim, rec, pca)
            print('=====', v, 'rear', m['rear'], 'front', m['front'][:6], 'front_push', m['front_push_s'], 'giveup', m['giveup_t'],
                  'guards', m['guards'][:8])
            holds = [((t - rv7.T0) / 1e9) for t, r in rec.rows if r['hold'] and r['select']]
            # 中立止めの区間
            segs = []
            for h in holds:
                if segs and h - segs[-1][1] < 0.12:
                    segs[-1][1] = h
                else:
                    segs.append([h, h])
            print('   holds', [(round(a, 3), round(b, 3)) for a, b in segs if t_from - 1 <= a <= t_to])
            for x in phases(ph):
                if t_from <= x[0] <= t_to:
                    xx = next((q for q in sim.log if q[0] >= x[0] - 1e-9), sim.log[-1])
                    print('   %.3f %s->%s %s hr=%.3f est=%.3f back=%.3f erpm=%.0f | x=%.3f v=%+.2f' % (x[0], x[1], x[2], x[3], x[4], x[5],
                                                                                             x[6], x[7], xx[1], xx[2]))
    finally:
        rv7.cleanup()


if __name__ == '__main__':
    main(*sys.argv[1:])
