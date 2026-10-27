"""反証 6-2・安全: tf62 の出力の 1 場面を組み直して、段・出力・車の動きを並べる(読むだけ)。
    python -I trace62.py <tf62 の json> <run> [esc] [names] [t0] [t1]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
E = K.E

PH = []


def wrap(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0, b0 = self.phase, self.backed_s
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_back_s', 0.0)), 3), round(float(kw.get('hold_rev_s', 0.0)), 3),
                   round(b0, 3), round(self.backed_s, 3), round(float(erpm), 1), r[0] and round(r[0][1], 1)))
        return r
    return step


def main():
    f, run = sys.argv[1], int(sys.argv[2])
    esc = sys.argv[3] if len(sys.argv) > 3 else 'fake'
    names = (sys.argv[4] if len(sys.argv) > 4 else 'now1,ng1,now0,bef1').split(',')
    t0 = float(sys.argv[5]) if len(sys.argv) > 5 else None
    t1 = float(sys.argv[6]) if len(sys.argv) > 6 else None
    if esc == 'slow':
        T.ESC.update(neu=0.12, rdelay=0.10)
    ex = next(x for x in json.load(open(f))['examples'] if x['run'] == run)
    p = E.P1(**ex['params'])
    sc0 = T.scene(ex['kind'], ex['xf'], ex['mthr'], ex['bd'])
    if ex['rear'] is not None:
        sc0['xr'] = ex['xf'] - ex['rear']
    sc = dict(sc0, veh_skip=[tuple(x) for x in ex['veh_skip']], drops=[tuple(x) for x in ex['drops']])
    print('scene', {k: v for k, v in ex.items() if k not in ('res',)})
    try:
        for nm in names:
            mod = {'now1': 'now', 'ng1': 'now', 'now0': 'now', 'bef1': 'fix6_before', 'ng0': 'now'}[nm]
            pp = dict(p, rec_hold_keep=0) if nm in ('now0', 'ng0') else p
            s = sc0 if nm in ('ng1', 'ng0') else sc
            del PH[:]
            mm, amod = E.load(mod)
            orig = amod.Recovery.step
            amod.Recovery.step = wrap(orig)
            try:
                sim, rec, pca = T.run(mod, pp, s)
            finally:
                amod.Recovery.step = orig
            m = T.win(sim, pca, pp)
            print('=====', nm, {k: m[k] for k in ('t_brake', 't_escape', 'back_m', 'esc_push_s', 'rear_win', 't_rev_on')}
                  if m else None)
            print('  guards', E.guards(rec)[:6])
            lo = t0 if t0 is not None else (m['t_brake'] - 0.3 if m else 1.5)
            hi = t1 if t1 is not None else (m['end'] if m else lo + 3)
            rows = {round((t - K.T0) / 1e9, 3): r for t, r in rec.rows}
            ph = {x[0]: x for x in PH}
            for t, x, v, th in sim.log:
                if lo <= t <= hi and abs((t * 100) % 2) < 1e-6:
                    rv = next((r for tt, r, c in sim.car.revlog if abs(tt - t) < 1e-6), None)
                    near = [x2 for k2, x2 in ph.items() if t - 0.02 < k2 <= t]
                    print('  t=%.2f x=%.3f v=%+.2f thr=%.0f rev_on=%s %s' % (t, x, v, th, rv,
                                                                       near[-1][1:] if near else ''))
    finally:
        K.cleanup()


if __name__ == '__main__':
    main()
