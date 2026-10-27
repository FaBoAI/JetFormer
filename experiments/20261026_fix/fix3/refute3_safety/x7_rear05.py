import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl3
from cl3 import C, N, PARAMS
for keep, guard in ((0, 0), (0, 1), (1, 1)):
    for back in (0.5, 0.8):
        sc_ = {'dur': 8.0, 'xf': 2.0, 'xr': 2.0 - back, 'model': (C, N - 120.0)}
        sim, rec, pca, ph = cl3.run(dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=guard, rec_hold_keep=keep), sc_)
        print(keep, guard, back, sim.car.hits, [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard'], cl3.phases(ph)[:16])
shutil.rmtree(cl3.TMP, ignore_errors=True)
