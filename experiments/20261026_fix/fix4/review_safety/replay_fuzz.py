"""反証 4 回目・安全: fuzz_closed.py の 種 S の i 本目を同じ場面で流し直し、now1(keep 1)・now0(keep 0)・bef1 の
時間の流れ(リカバリーの段・出力・車の速さと位置・出来事)を並べる。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix4/review_safety/replay_fuzz.py 種 i [t0 t1]
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as R          # noqa: E402
from rs_common import A, C, N, T0   # noqa: E402
import fuzz_closed as F        # noqa: E402

PH = []


def wrap(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, round(float(kw.get('hold_g', 0.0)), 2),
                   kw.get('hold_since') is not None, round(float(kw.get('hold_back_s', 0.0)), 3), int(erpm)))
        return r
    return step


def scene_of(seed, i):
    rng = random.Random(seed)
    for k in range(i + 1):
        sc = F.rand_scene(rng)
        p1 = F.rand_params(rng)
    return sc, p1


def timeline(sim, rec, ph, t0, t1):
    out = []
    rows = {round((t - T0) / 1e9, 3): r for t, r in rec.rows}
    log = {round(t, 2): (x, v, thr, con, rev_on) for t, x, v, thr, con, rev_on, sl in sim.log}
    phm = {}
    for x in ph:
        phm.setdefault(round(x[0] + 0.0, 3), x)
    for s in sorted(rows):
        if not (t0 <= s <= t1):
            continue
        r = rows[s]
        lg = log.get(round(s, 2)) or log.get(round(s + 0.01, 2))
        out.append((s, 'H' if r['hold'] else '-', int(r['assist']), round(float(r['out'][0])), round(float(r['out'][1])),
                    None if lg is None else (round(lg[0], 2), round(lg[1], 2), lg[3])))
    return out


def main(seed, i, t0=0.0, t1=99.0):
    sc, p1 = scene_of(seed, i)
    BEF = R.load_snapshot('fix3_r3_1_before')
    orig = R.CL.Sim
    R.CL.Sim = F.FSim
    res = {}
    try:
        for nm, p, mod in (('now1', p1, None), ('now0', dict(p1, rec_hold_keep=0), None), ('bef1', p1, BEF)):
            del PH[:]
            if mod is None:
                sim, rec, pca, d = R.run_closed(p, sc, wrap=wrap)
            else:
                sim, rec, pca, d = R.run_closed(p, sc, mod=mod[0], amod=mod[1], wrap=wrap)
            res[nm] = {'hits': sim.car.hits,
                       'events': [(round(e[1].get('cam_epoch_us', 0) / 1e6 - 1_800_000_000, 3) if 'cam_epoch_us' in e[1] else None,
                                   e[0], e[1].get('what'), e[1].get('phase'), e[1].get('backed_s'), e[1].get('hold_s'))
                                  for e in rec.events if e[0] in ('recovery_guard',)],
                       'phases': [x for x in PH if x[1] != x[2] or x[4]],
                       'timeline': timeline(sim, rec, PH, t0, t1)}
    finally:
        R.CL.Sim = orig
        R.cleanup()
    print('scene', {k: v for k, v in sc.items() if k != 'model'})
    print('params', {k: p1[k] for k in ('launch_boost', 'launch_no_kick', 'start_kick', 'speed_ceiling', 'zone_caps',
                                         'tof_guard', 'rev_s', 'rev_us', 'throttle_gain')})
    for nm, v in res.items():
        print('==', nm, 'hits', v['hits'])
        print('  events', v['events'])
        print('  phases', [x for x in v['phases'] if t0 <= x[0] <= t1])
        for row in v['timeline']:
            print('   ', row)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('replay_%d_%d.json' % (seed, i))).write_text(json.dumps(
        {'scene': {k: v for k, v in sc.items() if k != 'model'}, 'params': p1, 'res': res}, ensure_ascii=False, default=str))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), *(float(x) for x in a[2:4]))
