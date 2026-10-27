"""repro.py の主な場面(neu 0.2・'reverse' の始め 2.52 秒から車両の値 0.2 秒なし)を 10 秒まで回し、最初の失敗のあとどうなるか
(2 回目のリカバリー・中止・前後の当たり・下がった距離)を見る(読むだけ)。  python -I after.py"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX6 = HERE.parent
sys.path.insert(0, str(FIX6 / 'refute62_safety'))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
E = K.E
R = K.R
TMP = FIX6 / 'tmp_refute_w2s2'
R.TMP = TMP / 'rs'
R.CL.TMP = R.TMP
E.TMP = TMP / 'exp'
N = T.N

try:
    T.ESC.update(neu=0.2, rdelay=0.08)
    p = E.P1()
    for nm, mod, vs in (('now1', 'now', [(2.52, 2.72)]), ('bef1', 'fix6_before', [(2.52, 2.72)]), ('now1_nogap', 'now', [])):
        sc = dict(T.scene('rev', 2.0, 120.0, 0.5), veh_skip=vs, dur=10.0)
        sim, rec, pca = T.run(mod, p, sc)
        esc = [t for t, s, th in pca.w if abs(th - (N - 130.0)) < 1e-6]
        esc_starts = [t for i, t in enumerate(esc) if i == 0 or t - esc[i - 1] > 0.1]
        rev_on = [t for i, (t, r, c) in enumerate(sim.car.revlog) if r and (i == 0 or not sim.car.revlog[i - 1][1])]
        giveup = any(e[0] == 'recovery_guard' and e[1].get('what') == 'hold_long' for e in rec.events)
        minx = min(x for t, x, v, th in sim.log if t >= 2.0)
        print(nm, 'escape starts', [round(t, 2) for t in esc_starts], 'fake ESC reverse starts', [round(t, 2) for t in rev_on],
              'hits', sim.car.hits[:6], 'min x', round(minx, 3), 'xf', sim.s['xf'],
              'events', sorted({e[0] for e in rec.events}), flush=True)
finally:
    K.cleanup()
    shutil.rmtree(TMP, ignore_errors=True)
