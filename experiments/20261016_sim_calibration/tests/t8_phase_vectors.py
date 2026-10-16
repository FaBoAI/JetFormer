"""試験 T8 の入力: jetformer/safety_caps.py の phase_step と combine_zone を、作った列に通した答え(JS の写しと比べる)。
python tests/t8_phase_vectors.py <出力 JSON>"""
import json
import math
import random
import sys

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.safety_caps import PhaseState, phase_step, combine_zone  # noqa: E402


def make_seq(rng, n=900, dt=1 / 30):
    """周回に似た列: 直線・左の U ターン(LH)・右の U ターン(R1)・左(GM)…を、雑音・止まり・衝撃つきで"""
    t, out = 0.0, []
    plan = []
    for _ in range(12):
        plan += [('s', rng.uniform(.5, 1.5), 0), ('L', rng.uniform(.6, 1.2), rng.uniform(150, 320)), ('s', rng.uniform(.1, .4), 0),
                 ('R', rng.uniform(.5, 1.0), -rng.uniform(120, 330)), ('L', rng.uniform(.6, 1.4), rng.uniform(100, 260))]
        if rng.random() < .2:
            plan.append(('stop', rng.uniform(.3, 2.0), 0))
    for kind, dur, rate in plan:
        k = int(dur / dt)
        for _ in range(k):
            v = 0.0 if kind == 'stop' else rng.uniform(1.2, 3.2)
            gz = rate + rng.gauss(0, 8) if kind != 'stop' else rng.gauss(0, 1.5)
            if rng.random() < .01:
                gz = rng.choice([-1000.0, 1000.0])
            out.append((round(t, 6), gz, v))
            t += dt
            if len(out) >= n:
                return out
    return out


def main():
    rng = random.Random(20261016)
    cases = []
    for c in range(4):
        seq = make_seq(rng)
        st = PhaseState()
        res = []
        for t, gz, v in seq:
            st, o = phase_step(st, t, gz, v)
            res.append({'zone': o.zone, 'rel': None if o.rel_psi is None else round(o.rel_psi, 9), 'tr': o.turning_right, 'conf': o.r1_confirmed,
                        'comb': combine_zone(o.zone, o.turning_right, False, v)})
        cases.append({'seq': seq, 'res': res})
    json.dump({'cases': cases}, open(sys.argv[1], 'w'))
    zones = sorted({r['zone'] for c in cases for r in c['res']})
    print('T8 の入力:', sum(len(c['seq']) for c in cases), 'サンプル。区間', zones)


if __name__ == '__main__':
    main()
