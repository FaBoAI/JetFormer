"""反証 6-1・安全: fuzz_on の場面を同じ種で作り直し、決まりに引っかかった本を 5 通りで比べる
(今 keep 1・今 keep 0・今 keep 1 で車両の値の止まりを抜く・今 keep 1 でカメラの止まりを抜く・直す前の写し keep 1)。

    python -I check_runs.py 種 本 [本 ...]
出力: out/check_runs_<種>.json
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_common as K          # noqa: E402
from r6_common import R, CL, PH   # noqa: E402
import fuzz_closed as FC       # noqa: E402


def summ(res):
    sim, rec, pca, _ = res
    t_hit = next((h[0] for h in sim.car.hits if h[1] == 'front'), None)
    xs = [x for t, x, *_ in sim.log if t_hit is not None and t >= t_hit]
    return {'hits': sim.car.hits[:6], 'min_x': round(min(xs), 3) if xs else None,
            'events': [(e[1].get('what'), e[1].get('phase'), e[1].get('backed_s')) for e in rec.events
                       if e[0] == 'recovery_guard'][:6]}


def main(seed, runs):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix5_safety_before')
    scenes = {}
    for i in range(max(runs) + 1):
        sc = FC.rand_scene(rng)
        p = FC.rand_params(rng)
        if i in runs:
            scenes[i] = (sc, p)
    out = {}
    orig = CL.Sim
    CL.Sim = FC.FSim
    try:
        for i, (sc, p) in scenes.items():
            o = {'scene': {k: v for k, v in sc.items() if k != 'model'},
                 'params': {k: p[k] for k in ('launch_boost', 'launch_no_kick', 'start_kick', 'tof_guard', 'rev_s', 'rev_us')}}
            del PH[:]
            o['now1'] = summ(R.run_closed(p, sc, wrap=K.wrap))
            o['now1_after_hold'] = [x for x in PH if x[3]][:6]
            o['now0'] = summ(R.run_closed(dict(p, rec_hold_keep=0), sc))
            o['now1_no_veh_skip'] = summ(R.run_closed(p, dict(sc, veh_skip=[])))
            o['now1_no_cam_drop'] = summ(R.run_closed(p, dict(sc, drops=[], hit_drop=None)))
            o['before1'] = summ(R.run_closed(p, sc, mod=BEF[0], amod=BEF[1]))
            out[i] = o
            print(i, json.dumps(o, ensure_ascii=False, default=str), flush=True)
    finally:
        CL.Sim = orig
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('check_runs_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main(int(sys.argv[1]), [int(x) for x in sys.argv[2:]])
