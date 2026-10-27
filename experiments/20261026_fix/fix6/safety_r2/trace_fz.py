"""fuzz_fix6r2 の 1 場面(種・番号)を、今(fix6 r2)と直す前(fix6 r1)で流し、Recovery の段・見積もり・書いた値・後ろの当たりを並べる(読むだけ)。
    python -I trace_fz.py <種> <番号> [t_from t_to]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], '1', sys.argv[1]] + sys.argv[2:]
import fuzz_fix6r2 as Z   # noqa: E402
import random             # noqa: E402

F = Z.F
R, FC, A = F.R, F.FC, F.A


def main(seed, idx, t0=2.0, t1=6.0):
    rng = random.Random(seed)
    for i in range(idx + 1):
        sc = FC.rand_scene(rng)
        p1 = FC.rand_params(rng)
    print('scene', sc)
    print('params', {k: p1[k] for k in ('rev_s', 'rev_us', 'tof_guard', 'launch_boost', 'start_kick')})
    BEF = R.load_snapshot('fix5_safety_before')
    orig = R.CL.Sim
    R.CL.Sim = FC.FSim
    try:
        for nm, mod, amod in (('now', None, None), ('r1', BEF[0], BEF[1])):
            am = amod or A
            PH = []
            o = am.Recovery.step

            def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
                p0, b0 = self.phase, self.backed_s
                r = o(self, model_us, erpm, now, impact_g, *a, **kw)
                PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None,
                           round(float(kw.get('hold_back_s', 0.0)), 3), round(float(kw.get('hold_rev_s', 0.0)), 3),
                           round(b0, 3), round(self.backed_s, 3), round(getattr(self, 'est_s', 0.0), 3), int(erpm),
                           self.retries, None if r[0] is None else round(r[0][1]), self.guard_event))
                return r
            am.Recovery.step = step
            try:
                s, rc, pc, _ = R.run_closed(p1, sc, mod=mod, amod=amod)
            finally:
                am.Recovery.step = o
            print('=====', nm, 'rear', [h for h in s.car.hits if h[1] == 'rear'])
            for x in PH:
                if t0 <= x[0] <= t1 and (x[1] is not None or x[2] is not None):
                    print('  t=%s %s->%s held=%s hb=%s hrev=%s backed %s->%s est=%s erpm=%s retr=%s out=%s ev=%s' % x)
    finally:
        R.CL.Sim = orig
        R.cleanup()


if __name__ == '__main__':
    a = sys.argv[2:]
    main(int(a[0]), int(a[1]), *[float(x) for x in a[2:4]])
