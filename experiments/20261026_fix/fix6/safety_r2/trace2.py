"""1 場面を流して、Recovery の段の動き・書いた値・偽物の ESC の様子を出す(読むだけ)。
    python -I trace2.py <variant> <kind> <esc|orig> "<drops>" "<skips>" [t_from t_to]
    例: python -I trace2.py cand:so,oc,dr rev orig "2.09,2.26" "2.32,2.49"
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2 as E   # noqa: E402
import grid2 as G  # noqa: E402


def pairs(s):
    v = [float(x) for x in s.split(',') if x]
    return [(v[i], v[i + 1]) for i in range(0, len(v), 2)]


def main(variant, kind, esc, drops, skips, t0=2.0, t1=4.4):
    name, fl = (variant.split(':', 1) + [''])[:2] if ':' in variant else (variant, '')
    mod, amod = E.setup(name, fl)
    PH = []
    orig = amod.Recovery.step

    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0, b0 = self.phase, self.backed_s
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_back_s', 0.0)), 3), round(float(kw.get('hold_rev_s', 0.0)), 3),
                   round(b0, 3), round(self.backed_s, 3), round(getattr(self, 'est_s', 0.0), 3), int(erpm),
                   self.retries, None if r[0] is None else round(r[0][1])))
        return r
    amod.Recovery.step = step
    try:
        p = E.P1()
        res = E.run(name, p, E.scene(kind, skips=pairs(skips), drops=pairs(drops)), esc=G.ESC[esc], flags=fl)
    finally:
        amod.Recovery.step = orig
        E.cleanup()
    sim, rec, pca = res
    for x in PH:
        if t0 <= x[0] <= t1:
            print('rec t=%s %s->%s held=%s hb=%s hrev=%s backed %s->%s est=%s erpm=%s retr=%s out=%s' % x)
    w = [(round(t, 3), round(th)) for t, s, th in pca.w if t0 <= t <= t1]
    print('writes', w)
    on = [t for t, r in sim.car.revlog if r and t0 <= t <= t1]
    print('esc rev_on', (on[0], on[-1], len(on)) if on else None)
    print('metrics', E.metrics(res, p))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], a[4], *[float(x) for x in a[5:7]])
