"""反証 6-2・安全: probe_false_ready_orig の 1 場面(A = カメラ 2.11〜2.26・B = 車両の値 2.26〜2.45)を、元の偽物の ESC で
段・EscTrack の見積もり・書いた値・偽物の ESC の状態と並べる(読むだけ)。  python -I trace_false_ready.py [now|fix6_before]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
E = K.E
import test_fix3_r2_1 as R2    # noqa: E402

PH = []


def main(name='now'):
    mod, amod = E.load(name)
    orig = amod.Recovery.step

    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0, b0, r0 = self.phase, self.backed_s, self.retries
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_back_s', 0.0)), 3), round(float(kw.get('hold_rev_s', 0.0)), 3),
                   round(b0, 3), round(self.backed_s, 3), round(float(erpm)), self.retries,
                   None if r[0] is None else round(r[0][1])))
        return r
    amod.Recovery.step = step
    orig_car_step = R2.Car.step
    CAR = []

    def car_step(self, t, thr):
        res = orig_car_step(self, t, thr)
        CAR.append((round(t, 2), round(thr), self.ready, self.braked, self.rev_on, round(self.neu_t, 3), round(self.v, 2)))
        return res
    R2.Car.step = car_step
    try:
        sc = dict(T.scene('rev', 2.0, 120.0, 0.5), drops=[(2.11, 2.26)], veh_skip=[(2.26, 2.45)])
        sim, rec, pca = E.run(name, E.P1(), sc)
    finally:
        amod.Recovery.step = orig
        R2.Car.step = orig_car_step
        K.cleanup()
    print('guards', E.guards(rec))
    for x in PH:
        if 2.0 <= x[0] <= 4.2:
            print('rec', x)
    for x in CAR:
        if 2.05 <= x[0] <= 2.75 or (3.4 <= x[0] <= 4.2 and abs((x[0] * 100) % 5) < 1e-6):
            print('esc', x)
    print('writes', [(round(t, 3), round(th)) for t, s, th in pca.w if 2.05 <= t <= 4.2])
    print('front hits', sim.car.hits[:4])


if __name__ == '__main__':
    main(*sys.argv[1:2])
