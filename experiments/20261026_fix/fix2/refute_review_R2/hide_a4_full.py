#!/usr/bin/env python3
"""R2 の確かめ 2: A4 から後の起点を記録から消す(1 周の距離の中央値にも入れない)+ AI の区間を 3 周目の線の 0.5 秒後で切る。
= 「3 周目の線を越えた HS の上で人側 → 止める」をまねる。読むだけ"""
import json, sys
from pathlib import Path
sys.path.insert(0, '/home/jetson/jetformer'); sys.path.insert(0, '/home/jetson/jetformer/tools')
import numpy as np
import standing_3lap as s3
ss = s3.ss
orig_score = ss.score_log
CUT = {'a4': None}

def score_patched(*a, **k):
    sc = orig_score(*a, **k)
    ex = k.get('extras')
    if CUT['a4'] is not None and ex is not None:
        an = np.asarray(ex['anchors'], float)
        ex['anchors'] = an[an < CUT['a4'] - 1e-3]
    return sc
s3.ss.score_log = score_patched

logs = []
for day in ('20261023', '20261024'):
    logs += s3.day_logs(day)[0]
names = s3.candidate_names()
rows = []
for lg in logs:
    CUT['a4'] = None
    r0 = s3.analyze_log(lg, names=names)
    for i, a in enumerate(r0.get('standing') or []):
        if a.get('method') != 'anchor' or not a.get('valid'):
            continue
        CUT['a4'] = a['anchors'][3]
        r1 = s3.analyze_log(lg, names=names)
        CUT['a4'] = None
        b = [x for x in r1['standing'] if x['t_ai'] == a['t_ai']][0]
        d = None if b.get('from_move_s') is None else round(b['from_move_s'] - a['from_move_s'], 3)
        rows.append({'log': lg, 'ai_no': r0.get('ai_no'), 'from_move_a4': a['from_move_s'], 'Lrun_a4': r0['lap_dist_median_m'],
                     'Lrun_cut': r1['lap_dist_median_m'], 'method_cut': b.get('method'), 'valid_cut': b['valid'],
                     'from_move_cut': b.get('from_move_s'), 'diff': d, 'dist_anchor_m': a['dist_anchor_m'], 'note': a.get('note')})
        print(rows[-1])
ds = [r['diff'] for r in rows if r['diff'] is not None and r['valid_cut'] and not (r['log'] == '20261023-211828')]
print('n', len(ds), 'min', min(ds), 'max', max(ds), 'abs median', float(np.median(np.abs(ds))), 'neg', sum(1 for x in ds if x < 0))
Path(sys.argv[1]).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
