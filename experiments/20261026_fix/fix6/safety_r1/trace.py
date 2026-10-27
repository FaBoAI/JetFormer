"""1 場面の Recovery の段・中立止めのあとの数・偽物の車の動きを出す(読むだけ)
    python -I trace.py rev|brake name d0 L [rev_s] [t_from] [t_to]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_common as E    # noqa: E402
import exp_grid as G      # noqa: E402

kind, name, d0, L = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rev_s = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
t_from = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
t_to = float(sys.argv[7]) if len(sys.argv) > 7 else 4.2
base = name.split('+')[0]
mod, amod = E.load(base)
calls = []
orig = amod.Recovery.step


def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
    p0, b0 = self.phase, self.backed_s
    r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
    calls.append((round(now - 100.0, 3), p0, self.phase, round(float(erpm)), kw.get('hold_since') is not None,
                  round(float(kw.get('hold_back_s', 0.0)), 3), round(b0, 3), round(self.backed_s, 3),
                  None if r[0] is None else round(float(r[0][1]))))
    return r


amod.Recovery.step = step
try:
    p = E.P1(rev_s=rev_s)
    ref = G.one(name, p, kind)
    t_on = ref['t_rev_on']
    sk = (round(t_on + d0, 3), round(t_on + d0 + L, 3)) if L > 0 else None
    del calls[:]
    sc, xf = G.scene(kind, sk)
    sim, rec, pca = E.run(name, p, sc)
    print('skip', sk, E.first_rec(sim, pca, xf), E.guards(rec)[:4])
    for c in calls:
        if t_from <= c[0] <= t_to and (c[1] != c[2] or c[4] or abs(c[0] - (sk[0] if sk else 0)) < 0.25):
            print('REC t=%.3f %s->%s erpm=%d held=%s hb=%.3f backed %.3f->%.3f out=%s' % c)
    hold = [round((t - E.R2.T0) / 1e9, 3) for t, r in rec.rows if r['hold']]
    print('hold rows', hold[:3], '...', hold[-2:], len(hold))
    for (t, x, v, th), (_, ron, con) in zip(sim.log, sim.car.revlog):
        if t_from <= t <= t_to and int(round(t * 100)) % 4 == 0:
            print('%.2f x=%.3f v=%.2f thr=%.0f rev_on=%s' % (t, x, v, th, ron))
finally:
    amod.Recovery.step = orig
    E.cleanup()
