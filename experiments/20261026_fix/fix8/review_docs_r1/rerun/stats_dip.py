"""反証 8-1 docs: 自分で流した閉じたループ(dip_skip・今の ON = 11/3 の形)で、'gap' を飛ばした後退の途中の途切れの組を
途切れの無い同じ場面(clean)と比べる。決め(やり直し・early_none の数)が変わった組の下がる距離の差と、決めが変わらず長く下がった組。読むだけ。"""
import json
import sys
from pathlib import Path
import numpy as np
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding='utf-8'))
cells = d['cells']
cl = {k[:-len(' clean')]: r for k, r in cells.items() if k.endswith('clean')}
v = 'on'
chg, same_long, same_short = [], [], []
for k, r in cells.items():
    if k.endswith('clean'):
        continue
    pre = next(x for x in cl if k.startswith(x + ' '))
    c, b = cl[pre][v], r[v]
    db = (b['back'] or 0) - (c['back'] or 0)
    en_b = b['all_guards'].count('early_none'); en_c = c['all_guards'].count('early_none')
    row = (k, round(db, 3), b['n_retry'], c['n_retry'], en_b, en_c, b['back'], c['back'], len(b['rear_hits'] or []), len(c['rear_hits'] or []))
    if b['n_retry'] != c['n_retry']:
        chg.append(row)
    elif db > 0.05:
        same_long.append(row)
    elif db < -0.05:
        same_short.append(row)
print('くみ', d['kind'], '組', sum(1 for k in cells if not k.endswith('clean')), 'clean', len(cl))
for pre, r in cl.items():
    print('  clean', pre, '下がる', r[v]['back'], 'やり直し', r[v]['n_retry'], r[v]['all_guards'])
dbs = [x[1] for x in chg]
print('決め(やり直し・early_none の数)が clean と変わった組: %d' % len(chg))
if dbs:
    print('  下がる距離の差(今 − clean): 最小 %+.3f・中央 %+.3f・最大 %+.3f m。長く(+0.05 m より)%d・短く(−0.05 m より)%d' % (
        min(dbs), float(np.median(dbs)), max(dbs), sum(x > 0.05 for x in dbs), sum(x < -0.05 for x in dbs)))
    print('  後ろの当たりが clean より増えた組: %d' % sum(1 for x in chg if x[8] > x[9]))
    for x in sorted(chg, key=lambda x: -x[1])[:5]:
        print('   いちばん長い側:', x)
print('決めは変わらず clean より長く下がった組(+0.05 m より): %d' % len(same_long))
for x in sorted(same_long, key=lambda x: -x[1])[:5]:
    print('   ', x)
print('決めは変わらず clean より短く下がった組: %d' % len(same_short))
