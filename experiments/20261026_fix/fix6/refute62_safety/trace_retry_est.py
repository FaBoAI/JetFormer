"""反証 6-2・安全: probe_retry_est の 1 場面(ESC は中立 0.2 秒で下がれる・'reverse' の始め 2.52 秒から車両の値が L 秒来ない)を
段・backed_s・hold_rev_s・書いた値・偽物の ESC の後退と並べる(読むだけ)。  python -I trace_retry_est.py <now|fix6_before> [d0] [L]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
E = K.E

PH = []


def main(name='now', d0=0.0, L=0.2):
    T.ESC.update(neu=0.2, rdelay=0.08)
    mod, amod = E.load(name)
    orig = amod.Recovery.step

    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0, b0 = self.phase, self.backed_s
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_back_s', 0.0)), 3), round(float(kw.get('hold_rev_s', 0.0)), 3),
                   round(b0, 3), round(self.backed_s, 3), round(float(erpm)), self.retries,
                   None if r[0] is None else round(r[0][1])))
        return r
    amod.Recovery.step = step
    try:
        b = (round(2.52 + float(d0), 3), round(2.52 + float(d0) + float(L), 3))
        sc = dict(T.scene('rev', 2.0, 120.0, 0.5), veh_skip=[b])
        sim, rec, pca = T.run(name, E.P1(), sc)
    finally:
        amod.Recovery.step = orig
        K.cleanup()
    print('veh_skip', b, 'guards', E.guards(rec))
    for x in PH:
        if 2.3 <= x[0] <= 4.2:
            print('rec', x)
    print('fake ESC reverse first', next((t for t, r, c in sim.car.revlog if t > 2.0 and r), None))
    print('front hits', sim.car.hits[:4])


if __name__ == '__main__':
    main(*sys.argv[1:4])
