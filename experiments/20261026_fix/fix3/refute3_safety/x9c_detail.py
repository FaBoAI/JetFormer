import json, random, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'refute2_safety'))
import cl3
from cl3 import A, C, N, T0
import fuzz_cl as F
import x9_fuzz as X
want = set(int(a) for a in sys.argv[1:])
rng = random.Random(1103)
for i in range(max(want) + 1):
    sc_ = F.scene_of(rng)
    p = X.params_on(rng)
    if i not in want:
        continue
    sim, rec, pca, ph = cl3.run(p, sc_)
    print('== i', i, {k: v for k, v in sc_.items() if k != 'model'})
    print('params', {k: p[k] for k in ('start_kick', 'launch_boost', 'launch_no_kick', 'tof_guard', 'throttle_gain', 'rev_us', 'rev_s')})
    print('hits', sim.car.hits)
    print('phases', cl3.phases(ph))
    print('events', [(e[0], e[1].get('what') or e[1].get('event'), e[1].get('hold_s')) for e in rec.events])
shutil.rmtree(cl3.TMP, ignore_errors=True)
