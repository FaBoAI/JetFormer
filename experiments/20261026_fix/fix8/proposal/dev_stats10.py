"""fix8 係 safety-proposal: clean のある くみで、決めが変わった組(やり直し・early_none の数が clean と違う)の、
下がる距離・最大の後ろへの速さ・escape までの秒の clean との差(案 − clean)の分布と、後ろの当たり。python -I dev_stats10.py くみ ..."""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for kind in sys.argv[1:]:
    d = json.loads((HERE / 'out' / ('exp10_%s.json' % kind)).read_text(encoding='utf-8'))
    cells = d['cells']
    cl = {k[:-5]: r for k, r in cells.items() if k.endswith('clean')}
    for v in [x for x in d['variants'] if x in ('on', 'pA', 'pB', 'pC')]:
        db, dv, dt, rh, rv = [], [], [], 0, 0
        for k, r in cells.items():
            if k.endswith('clean'):
                continue
            c = cl[next(p for p in cl if k.startswith(p))][v]
            b = r[v]
            if b['n_retry'] == c['n_retry']:
                continue
            db.append((b['back'] or 0) - (c['back'] or 0))
            dv.append((b['max_back_v'] or 0) - (c['max_back_v'] or 0))
            if b['t_to_escape'] is not None and c['t_to_escape'] is not None:
                dt.append(b['t_to_escape'] - c['t_to_escape'])
            rh += len(b['rear_hits'] or []) > len(c['rear_hits'] or [])
            rv += (b['rear_max_v'] or 0) > (c['rear_max_v'] or 0) + 0.05
        if not db:
            print(kind, v, '決めが変わった 0')
            continue
        q = lambda xs: '%+.2f / %+.2f / %+.2f' % (min(xs), float(np.median(xs)), max(xs)) if xs else '—'
        print('%s %s 決めが変わった %d: 下がる距離の差(最小/中央/最大)%s m・長く下がった %d・短く %d・最大の速さの差 %s m/s・'
              'escape までの差 %s 秒・後ろの当たりが増えた %d・速い %d' % (
                  kind, v, len(db), q(db), sum(x > 0.05 for x in db), sum(x < -0.05 for x in db), q(dv), q(dt), rh, rv))
