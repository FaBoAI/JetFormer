"""反証 8-1 docs: dip_norm(ふつうの後退の途中の途切れ)を ON・OFF で clean と比べる。読むだけ。"""
import json
import sys
import numpy as np
d = json.load(open(sys.argv[1]))
cells = d['cells']
cl = {k[:-len(' clean')]: r for k, r in cells.items() if k.endswith('clean')}
for v in d['variants']:
    chg, longer, shorter, n = [], [], [], 0
    for k, r in cells.items():
        if k.endswith('clean'):
            continue
        n += 1
        pre = next(x for x in cl if k.startswith(x + ' '))
        c, b = cl[pre][v], r[v]
        db = (b['back'] or 0) - (c['back'] or 0)
        if b['n_retry'] != c['n_retry']:
            chg.append(db)
        elif db > 0.05:
            longer.append(db)
        elif db < -0.05:
            shorter.append(db)
    q = lambda xs: ('%+.3f〜%+.3f(中央 %+.3f)' % (min(xs), max(xs), float(np.median(xs)))) if xs else '—'
    print('%s: 組 %d・決めが変わった %d(差 %s)・決めは同じで長く %d(差 %s)・短く %d' % (v, n, len(chg), q(chg), len(longer), q(longer), len(shorter)))
