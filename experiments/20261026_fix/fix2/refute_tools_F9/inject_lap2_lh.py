"""tools:F9 の確かめ(読むだけ。data/raw・labels は変えない): 
1) 発進ごとに [1 周目の線(standing_3lap の line_ends[0]), A2) の場所(le.locate)を 0.05 秒ごとに数える。
2) 1 本の記録(既定 20261024-183826。もとは衝突 0)の、その区間の LH の真ん中に 3.0G の当たりを 1 つ足して、
   collect.py の collisions_first_lap と standing_3lap の collisions_by_lap を比べる。"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/diag_launch')
sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import collect as C  # noqa
import standing_3lap as S3  # noqa
import label_events as le  # noqa
OUT = Path(__file__).resolve().parent / 'out'

def pairs(day):
    L = json.loads((OUT / ('launches_%s.json' % day)).read_text())['launches']
    S = json.loads((OUT / ('s3_%s.json' % day)).read_text())['results']
    sidx = {(r['log'], round(s['t_ai'], 2)): s for r in S for s in r.get('standing') or []}
    for l in L:
        s = sidx.get((l['log'], round(l['t_ai'], 2)))
        if l.get('to_r1_s') is None or not s or not s.get('line_ends') or len(s.get('anchors') or []) < 2:
            continue
        if not (5.0 <= s['anchors'][1] - s['anchors'][0] <= 16.0) or s['d_pre_m'] > 12:
            continue                      # 発進が周の途中(d_pre 22〜26 m)などは外す
        yield l, s

# 1) 区間の場所
tot = Counter()
for day in ('20261024', '20261023'):
    for l, s in pairs(day):
        d = C.load(l['log'], True, {}, {})
        e1, a2 = s['line_ends'][0], float(l['first_lap_end_s'])
        if abs(a2 - s['anchors'][1]) > 0.01:
            continue
        T = np.arange(e1, a2, 0.05)
        k = le.locate(T, d['anchors'], d['gm'], d['it'], d['hd'], d['gzs'])
        c = Counter(le.LOC_DETAIL[int(x)] for x in k)
        tot.update(c)
        lh = [t for t, x in zip(T, k) if le.LOC_DETAIL[int(x)] == 'LH']
        print('%s t_ai=%7.2f 線1=%7.2f A2=%7.2f 長さ %.2f 秒 d_pre %.2f m 場所 %s LH %s〜%s' % (
            l['log'], l['t_ai'], e1, a2, a2 - e1, s['d_pre_m'], dict(c),
            '%.2f' % lh[0] if lh else '-', '%.2f' % lh[-1] if lh else '-'))
print('区間の場所(0.05 秒ごと、全部):', dict(tot))

# 2) 当たりを 1 つ足す
log = sys.argv[1] if len(sys.argv) > 1 else '20261024-183826'
d = C.load(log, True, {}, {})
res0 = S3.analyze_log(log)
s = [x for x in res0['standing'] if x.get('line_ends')][0]
e1, A2 = s['line_ends'][0], s['anchors'][1]
T = np.arange(e1, A2, 0.05)
k = le.locate(T, d['anchors'], d['gm'], d['it'], d['hd'], d['gzs'])
lh = [t for t, x in zip(T, k) if le.LOC_DETAIL[int(x)] == 'LH']
tq = round(float(lh[len(lh) // 2]), 3)
print('\n足す当たり: %s t=%.3f(線1 %.3f < t < A2 %.3f)場所 %s' % (log, tq, e1, A2,
      le.LOC_DETAIL[int(le.locate([tq], d['anchors'], d['gm'], d['it'], d['hd'], d['gzs'])[0])]))
for inject in (False, True):
    dd = dict(d)
    dd['hits'] = sorted(list(d['hits']) + ([(tq, 3.0)] if inject else []))
    rows = [r for r in C.launch_rows(dd, [], 'none') if r.get('to_r1_s') is not None]
    r = [x for x in rows if abs(x['t_ai'] - s['t_ai']) < 0.02][0]
    orig = S3.ss.score_log
    def patched(*a, **kw):
        sc = orig(*a, **kw)
        if inject:
            sc['collision_events'] = sorted((sc.get('collision_events') or []) + [{'t_s': tq, 'g': 3.0, 'strong': False, 'loc': 'LH'}],
                                            key=lambda c: c['t_s'])
        return sc
    S3.ss.score_log = patched
    try:
        res = S3.analyze_log(log)
    finally:
        S3.ss.score_log = orig
    s1 = [x for x in res['standing'] if abs(x['t_ai'] - s['t_ai']) < 0.02][0]
    print('足す=%s collect: collisions_first_lap=%s(終わり %.3f)→ first_lap_collisions_0=%s | standing_3lap: collisions_by_lap=%s' % (
        inject, r['collisions_first_lap'], r['first_lap_end_s'], len(r['collisions_first_lap']) == 0, s1['collisions_by_lap']))
