"""反証 6-1・安全: probe_s3_reverse_start の 1 場面の段・出力・車の動きを細かく出す(読むだけの確かめ)。
    python -I trace_one.py d0 L [keep] [before]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_s3_reverse_start as P   # noqa: E402
import r6_common as K                 # noqa: E402
from r6_common import PARAMS, R, PH, N, T0   # noqa: E402

d0, L = float(sys.argv[1]), float(sys.argv[2])
keep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
bef = len(sys.argv) > 4 and sys.argv[4] == 'before'
p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=keep, tof_guard=1, rev_s=0.4, rev_us=300.0)
try:
    mod = R.load_snapshot('fix5_safety_before') if bef else None
    t0 = 2.6
    skip = (round(t0 + d0, 3), round(t0 + d0 + L, 3)) if L > 0 else None
    del PH[:]
    sc = {'dur': 5.0, 'xf': P.XF, 'model': (P.C, N - 120.0), 'drops': [], 'veh_skip': [skip] if skip else []}
    kw = {} if mod is None else {'mod': mod[0], 'amod': mod[1]}
    sim, rec, pca, d = R.run_closed(p1, sc, wrap=K.wrap, **kw)
    print('skip', skip)
    for x in PH:
        if 2.0 <= x[0] <= 4.2 and (x[1] != x[2] or x[3]):
            print('PH', x)
    rows = {round((t - T0) / 1e9, 3): r for t, r in rec.rows}
    for t, x, v, thr, con, rv, sl in sim.log:
        if 2.4 <= t <= 4.2 and abs(t * 100 - round(t * 100)) < 1e-6 and int(round(t * 100)) % 3 == 0:
            print('%.2f x=%.3f v=%.2f thr=%.0f rev_on=%s con=%s' % (t, x, v, thr, rv, con))
    print('rear hits', [h for h in sim.car.hits])
finally:
    R.cleanup()
