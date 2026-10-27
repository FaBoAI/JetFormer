"""動くかの確かめ: 場面 rev(前の壁 2.0 m・モデルは前進 120 µs)を ON・OFF で。fix6 の数(値の止まりなし 1.272 m)と照らす"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rv7  # noqa: E402

rv7.check_now()
P = dict(rv7.PARAMS, recovery=1, start_kick=0, rev_s=0.4, rev_us=300.0)
sc = {'dur': 7.0, 'xf': 2.0, 'model': (rv7.C, rv7.N - 120.0), 'drops': [], 'veh_skip': []}
for nm, p in (('ON', rv7.ON(P)), ('OFF', rv7.OFF(P))):
    t = time.time()
    sim, rec, pca, ph = rv7.run('now', p, sc)
    xs = [x for t_, x, *_ in sim.log if t_ > 2.0]
    m = rv7.metrics(sim, rec, pca)
    print(nm, 'run %.2fs' % (time.time() - t), 'back', round(2.0 - min(xs), 3), {k: m[k] for k in ('front_push_s', 'rev_side_s', 'guards')})
rv7.cleanup()
