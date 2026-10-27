"""fix5 係 safety: fuzz_fix5.py の K1(残した中立止めのあとの後ろの当たりが keep 0 より悪い)の本を、今(now1)・直す前の写し
(bef1)・keep 0(now0)・中立止めなし(カメラと車両の値の止まりを抜いた同じ場面)で流し直して、後ろの当たりを比べる。
    ... python -I k1_check.py 種 i [i ...]   出力: out/k1_check_<種>.json"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fuzz_fix5 as F          # noqa: E402
R, FC = F.R, F.FC


def main(seed, idx):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix5_safety_before')
    want = set(idx)
    out = {}
    orig = R.CL.Sim
    R.CL.Sim = FC.FSim
    try:
        for i in range(max(want) + 1):
            sc = FC.rand_scene(rng)
            p1 = FC.rand_params(rng)
            if i not in want:
                continue
            res = {}
            for nm, p, mod in (('now1', p1, None), ('bef1', p1, BEF), ('now0', dict(p1, rec_hold_keep=0), None)):
                del FC.PH[:]
                if mod is None:
                    s, rc, pc, _ = R.run_closed(p, sc, wrap=FC.wrap)
                else:
                    s, rc, pc, _ = R.run_closed(p, sc, mod=mod[0], amod=mod[1], wrap=FC.wrap)
                ph = [x for x in FC.PH if x[1] != x[2] or x[3]]
                res[nm] = {'hits': s.car.hits, 'phases': ph[:60]}
            nh = dict(sc, drops=[], veh_skip=[])
            nh.pop('hit_drop', None)
            s, rc, pc, _ = R.run_closed(p1, nh)
            res['now1_no_holds'] = {'hits': s.car.hits}
            out[i] = res
            print(i, {k: [h for h in v['hits'] if h[1] == 'rear'] for k, v in res.items()}, flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
    (HERE / 'out' / ('k1_check_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main(int(sys.argv[1]), [int(x) for x in sys.argv[2:]])
