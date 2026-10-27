"""fuzz_closed 種 4404 の 107 本目: 後ろの当たりの前の 0.12 秒のカメラの止まり(4.87〜4.99 秒)だけを抜いた場面と、今の場面の
時間の流れ(段・出力・速さ)を並べる。出力: out/k1_detail_4404_107.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common as R          # noqa: E402
from rs_common import A, C, N, T0   # noqa: E402
import fuzz_closed as F        # noqa: E402
import replay_fuzz as RF       # noqa: E402

sc, p1 = RF.scene_of(4404, 107)
out = {}
orig = R.CL.Sim
R.CL.Sim = F.FSim
try:
    for nm, scx, keep in (('with_drop_keep1', sc, 1),
                          ('without_0.12s_drop_keep1', dict(sc, drops=[d for d in sc['drops'] if not 4.8 < d[0] < 4.9]), 1)):
        del RF.PH[:]
        sim, rec, pca, d = R.run_closed(dict(p1, rec_hold_keep=keep), scx, wrap=RF.wrap)
        rows = [(round((t - T0) / 1e9, 3), bool(r['hold']), int(r['assist']), round(float(r['out'][1]))) for t, r in rec.rows
                if 4.2 <= (t - T0) / 1e9 <= 5.2]
        log = [(round(t, 2), round(x, 3), round(v, 2), round(thr), con) for t, x, v, thr, con, *_ in sim.log if 4.2 <= t <= 5.2][::3]
        out[nm] = {'hits': sim.car.hits, 'phases': [x for x in RF.PH if (x[1] != x[2] or x[4]) and 3.0 <= x[0] <= 5.3],
                   'events': [(e[1]['what'], e[1].get('phase'), e[1].get('backed_s')) for e in rec.events if e[0] == 'recovery_guard'],
                   'rows': rows, 'log': log}
        print('==', nm, out[nm]['hits'])
        print('  phases', out[nm]['phases'])
        print('  events', out[nm]['events'])
        for r in rows:
            print('   ', r)
finally:
    R.CL.Sim = orig
    R.cleanup()
(HERE / 'out' / 'k1_detail_4404_107.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
