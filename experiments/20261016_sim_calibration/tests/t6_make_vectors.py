#!/usr/bin/env python3
"""試験 T6 の入力と期待値: jetformer/safety_caps.py の cap_step に乱数の入力列を流し、JS の governorStep と比べる材料を書く

    python tests/t6_make_vectors.py → /home/jetson/jetformer_reports/sim_calibration_20261016/tests/t6_vectors.json
"""
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import safety_caps as sc  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/tests/t6_vectors.json')


def enc(x):
    return 'inf' if x == math.inf else ('-inf' if x == -math.inf else x)


def main():
    rng = random.Random(20261016)
    sets = {'sim_v1.1': sc.params_sim(), 'sim_v1': sc.params_v1(cap_command='identity'), 'real_v1.1': sc.params_real_car(),
            'r1_275': sc.params_r1_275(cap_command='identity')}
    out = {'schema': 'jf-calib-t6/v1', 'sets': {}}
    for name, p in sets.items():
        st = sc.CapState()
        t, zone, v = 0.0, 'none', 2.0
        rows = []
        for i in range(3000):
            t += rng.choice((1 / 30, 1 / 30, 0.01, 0.05, 0.2))
            if rng.random() < 0.06:
                zone = rng.choice(sc.ZONES)
            v = max(0.0, min(6.0, v + rng.gauss(0, 0.3)))
            u = rng.uniform(-0.8, 7.24)
            steer = None if rng.random() < 0.2 else rng.uniform(-1, 1)
            st, o = sc.cap_step(st, t, zone, v, u, p, steer)
            rows.append([t, zone, v, u, steer, o.u, o.mode, enc(o.cap_eff), o.brake_start])
        pd = p._asdict()
        pd['brake_zones'] = list(pd['brake_zones'])
        out['sets'][name] = {'params': pd, 'rows': rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(OUT, sum(len(s['rows']) for s in out['sets'].values()))


if __name__ == '__main__':
    main()
