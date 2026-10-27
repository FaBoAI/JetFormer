"""fix5 係 safety: K1 の本の中身(段の移り・残した中立止めのあとの最初のフレームに渡ったもの・後ろの当たり)を今と直す前の写しで並べる。
    ... python -I k1_detail.py 種 i   出力: out/k1_detail_<種>_<i>.json"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fuzz_fix5 as F          # noqa: E402
R, FC = F.R, F.FC
LOG = []


def wrap(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0, b0, tl = self.phase, self.backed_s, self.t_last
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        if p0 != self.phase or kw.get('hold_since') is not None:
            LOG.append((round(now - 100.0, 3), p0, self.phase, round(float(erpm)), None if tl is None else round(now - tl, 3),
                        kw.get('hold_since') is not None, round(float(kw.get('hold_back_s', 0.0)), 3),
                        round(float(kw.get('hold_unseen_s', 0.0)), 3), round(float(kw.get('hold_g', 0.0)), 2),
                        round(b0, 3), round(self.backed_s, 3), r[1]))
        return r
    return step


def main(seed, i0):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix5_safety_before')
    for i in range(i0 + 1):
        sc = FC.rand_scene(rng)
        p1 = FC.rand_params(rng)
    out = {'scene': {k: v for k, v in sc.items() if k != 'model'}, 'params': {k: p1[k] for k in (
        'launch_boost', 'launch_no_kick', 'start_kick', 'tof_guard', 'rev_s', 'rev_us', 'throttle_gain')}}
    orig = R.CL.Sim
    R.CL.Sim = FC.FSim
    try:
        for nm, mod in (('now1', None), ('bef1', BEF)):
            del LOG[:]
            if mod is None:
                s, rc, pc, _ = R.run_closed(p1, sc, wrap=wrap)
            else:
                s, rc, pc, _ = R.run_closed(p1, sc, mod=mod[0], amod=mod[1], wrap=wrap)
            out[nm] = {'hits': s.car.hits, 'log': list(LOG),
                       'x_v': [(round(t, 2), round(x, 3), round(v, 2), round(th)) for t, x, v, th, *_ in s.log[::5]]}
    finally:
        R.CL.Sim = orig
        R.cleanup()
    (HERE / 'out' / ('k1_detail_%d_%d.json' % (seed, i0))).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(out['scene'], out['params'])
    for nm in ('now1', 'bef1'):
        print(nm, out[nm]['hits'])
        for x in out[nm]['log']:
            print('   ', x)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
