"""X10b: X10 を中立の惰性の減速 1.5・3.0・4.5 m/s² で。後ろの物を「中立止めなしのリカバリーでは届かない」所に置き、
明けたあと中止(中立)で惰性のまま当たるか(keep=1)、モデルの前進で止まるか(keep=0)"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl3                                    # noqa: E402
import closed_loop as CL                      # noqa: E402
from cl3 import C, N, PARAMS                  # noqa: E402

ORIG = CL.Car.step
DEC = [1.5]


def step(self, t, thr):
    """closed_loop.Car.step の中立の惰性 1.5 m/s² だけを DEC[0] にした写し"""
    s = self.s
    if not (thr < N - 20.0 or thr > N + 40.0):
        v_before = self.v
        r = ORIG(self, t, thr)
        # ORIG は 1.5 で減らした。差を足し直す(止まりを越えない)
        dv = (DEC[0] - 1.5) * CL.DT
        if self.contact is None:
            if v_before > 0:
                self.v = max(0.0, self.v - dv)
            elif v_before < 0:
                self.v = min(0.0, self.v + dv)
        return r
    return ORIG(self, t, thr)


CL.Car.step = step
XF = 2.0


def one(keep, drop, back):
    sc_ = {'dur': 8.0, 'xf': XF, 'xr': XF - back, 'model': (C, N - 120.0), 'drops': [drop] if drop else []}
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep)
    sim, rec, pca, ph = cl3.run(p, sc_)
    return {'rear_hits': [h for h in sim.car.hits if h[1] == 'rear'], 'min_x': round(min(r[1] for r in sim.log if r[0] > 1.6), 3),
            'ev': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']}


def main():
    res = {}
    for dec in (1.5, 3.0, 4.5):
        DEC[0] = dec
        nh = one(1, None, 9.0)
        reach = round(XF - nh['min_x'], 3)                   # 中立止めなしで下がる距離
        base = cl3.run(dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1),
                       {'dur': 8.0, 'xf': XF, 'model': (C, N - 120.0)})
        tr = next(t for t, p0, p1 in cl3.phases(base[3]) if p1 == 'reverse')
        for back in (reach + 0.15, reach + 0.3, reach + 0.5):
            for o, L in ((0.4, 0.55), (0.45, 0.6), (0.5, 0.55), (0.5, 0.7)):
                for keep in (0, 1):
                    r = one(keep, (tr + o, tr + o + L), back)
                    res['dec=%.1f back=%.2f(+%.2f) rev+%.2f L=%.2f keep=%d' % (dec, back, back - reach, o, L, keep)] = r
        res['dec=%.1f nohold' % dec] = {'reach_m': reach}
    for k, v in res.items():
        print(k, v)
    (Path(__file__).resolve().parent / 'out' / 'x10b_coast_decel.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    shutil.rmtree(cl3.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
