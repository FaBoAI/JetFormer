"""読むだけ: 反証の係 docs の流し直し(review_docs_r1/rerun/out/exp10_dip_skip.json = fix8/proposal/exp10.py の写しの閉じたループ dip_skip。
今の ON)から、やり直し・early_none の数が clean と変わった組の、ESC の下がり始め(rd)ごとの下がった距離と escape が前の壁を押した秒の最大を出す。"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / 'review_docs_r1/rerun/out/exp10_dip_skip.json').read_text(encoding='utf-8'))
cells = d['cells']
cl = {k[:-len(' clean')]: r for k, r in cells.items() if k.endswith('clean')}
by = collections.defaultdict(list)
for k, r in cells.items():
    if k.endswith('clean'):
        continue
    pre = next(x for x in cl if k.startswith(x + ' '))
    c, b = cl[pre]['on'], r['on']
    if b['n_retry'] != c['n_retry']:
        by[pre.split()[0]].append(((b['back'] or 0) - (c['back'] or 0), b['back'] or 0, b['esc_push'] or 0,
                                   len(b['rear_hits'] or []) - len(c['rear_hits'] or [])))
alld = [x for v in by.values() for x in v]
print('変わった組 %d: 差 %+.3f〜%+.3f m・長く下がった(> +0.005 m)%d・後ろの当たりが増えた %d' % (
    len(alld), min(x[0] for x in alld), max(x[0] for x in alld), sum(x[0] > 0.005 for x in alld), sum(x[3] > 0 for x in alld)))
for rd, v in sorted(by.items()):
    pushes = [x[2] for x in v]
    print('%s: 組 %d・下がった距離 %.3f〜%.3f m・差 %+.3f〜%+.3f m・前の壁を 0.05 秒より押した組 %d(最大 %.3f 秒)' % (
        rd, len(v), min(x[1] for x in v), max(x[1] for x in v), min(x[0] for x in v), max(x[0] for x in v),
        sum(p > 0.05 for p in pushes), max(pushes)))
for pre, r in sorted(cl.items()):
    if pre.endswith('rear=None'):
        print('clean %s: 下がった距離 %.3f m・前の壁を押した秒 %s' % (pre, r['on']['back'], r['on']['esc_push']))
