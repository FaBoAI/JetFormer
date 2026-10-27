"""fix5 係 safety: K1 の本を、車両の値の止まりを抜いた場面(カメラの止まりは残す)・カメラの止まりを抜いた場面で、今と直す前の
写しの両方で流し、後ろの当たりを比べる(当たりが s3 の直しのせいか、前からの「中立止めの間の惰性」かを分ける)。
    ... python -I k1_variants.py 種 i [i ...]   出力: out/k1_variants_<種>.json"""
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
            v_noveh = dict(sc, veh_skip=[])
            v_nocam = dict(sc, drops=[])
            v_nocam.pop('hit_drop', None)
            for vn, s_ in (('as_is', sc), ('no_veh_skip', v_noveh), ('no_cam_drop', v_nocam)):
                for nm, mod in (('now1', None), ('bef1', BEF)):
                    if mod is None:
                        s, rc, pc, _ = R.run_closed(p1, s_)
                    else:
                        s, rc, pc, _ = R.run_closed(p1, s_, mod=mod[0], amod=mod[1])
                    res['%s %s' % (vn, nm)] = [h for h in s.car.hits if h[1] == 'rear']
                    res['%s %s front' % (vn, nm)] = [h for h in s.car.hits if h[1] == 'front']
            out[i] = res
            print(i, json.dumps(res, ensure_ascii=False), flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
    (HERE / 'out' / ('k1_variants_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main(int(sys.argv[1]), [int(x) for x in sys.argv[2:]])
