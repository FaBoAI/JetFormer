"""fuzz_closed.py の K1(残した中立止めのあとの後ろの当たりで keep 0 に同じ当たりが無い)の本を、当たりの前 2 秒の
カメラ・車両の値の止まりを抜いた同じ場面(中立止めなし)と比べる。中立止めなしでも同じ強さで当たるなら、
当たりはリカバリーそのものの後退(rev_us・rev_s)で、中立止めのあとに続けたことのせいではない。

    ... python -B -I k1_nohold_cmp.py 種 i [i ...]
出力: out/k1_nohold_cmp_<種>.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as R          # noqa: E402
import fuzz_closed as F        # noqa: E402
import replay_fuzz as RF       # noqa: E402

seed = int(sys.argv[1])
out = {}
orig = R.CL.Sim
R.CL.Sim = F.FSim
try:
    for i in (int(x) for x in sys.argv[2:]):
        sc, p1 = RF.scene_of(seed, i)
        s1, r1, _, _ = R.run_closed(p1, sc)
        rear = [h for h in s1.car.hits if h[1] == 'rear']
        res = {'with_holds': rear}
        for h in rear[:1]:
            sc2 = dict(sc, drops=[d for d in sc['drops'] if not (h[0] - 2.0 <= d[1] and d[0] <= h[0])],
                       veh_skip=[d for d in sc['veh_skip'] if not (h[0] - 2.0 <= d[1] and d[0] <= h[0])])
            s2, r2, _, _ = R.run_closed(p1, sc2)
            res['without_holds_near_hit'] = [x for x in s2.car.hits if x[1] == 'rear']
            res['removed'] = [d for d in sc['drops'] + sc['veh_skip'] if d not in sc2['drops'] + sc2['veh_skip']]
        out[i] = res
        print(i, res)
finally:
    R.CL.Sim = orig
    R.cleanup()
(HERE / 'out' / ('k1_nohold_cmp_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
