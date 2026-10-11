#!/usr/bin/env python3
"""DE(緑マット出口 tg の 0.2〜0.8秒後)のステア: 人はスティック(VEHICLE pwm)、AI は out(ゲイン後)と raw(ゲイン前)。
µs は中心 1443 からの差(+ = 左)。読み取りのみ、CPU。  nice -n 15 python de_steer.py > de_steer.json"""
import json
import sys

import numpy as np

import lap_quality as q

res = {}
for log in list(q.HUMAN) + ['20261010-172910', '20261010-173445']:
    d = q.load(log)
    q.heading(d)
    q.impacts(d)
    an, passes, laps = q.lap_rows(d, log in q.AI)
    rows = []
    for P in passes:
        tg = P['tg']
        if tg is None:
            continue
        if log in q.AI:
            if not P.get('ai_de'):
                continue
            m = (d['at'] >= tg + 0.2) & (d['at'] < tg + 0.8)
            st = float(np.mean(d['out'][m, 0]) - 1443)
            raw = float(np.mean(d['raw'][m, 0]) - 1443)
            gain = float(np.median(d['gains'][m, 0]))
        else:
            m = (d['vt'] >= tg + 0.2) & (d['vt'] < tg + 0.8)
            st = float(np.mean(d['pwm'][m, 0]) - 1443)
            raw, gain = None, None
        rows.append(dict(st=round(st, 1), raw=None if raw is None else round(raw, 1), gain=gain, hit=bool(P.get('de_hits')),
                         yaw=round(P['de_yaw'], 1) if P.get('de_yaw') is not None else None))
    res[log] = rows
out = {}
for g, logs in {'human_afternoon': [k for k, v in q.HUMAN.items() if v[0] == 'afternoon'],
                'human_evening': [k for k, v in q.HUMAN.items() if v[0] == 'evening'],
                'ai_006_007': ['20261010-172910', '20261010-173445']}.items():
    rr = [r for lg in logs for r in res[lg]]
    st = [r['st'] for r in rr]
    o = {'n': len(rr), 'steer_out_median': float(np.median(st)), 'steer_out_p25': float(np.percentile(st, 25)),
         'steer_out_p75': float(np.percentile(st, 75)),
         'yaw_median': float(np.median([r['yaw'] for r in rr if r['yaw'] is not None]))}
    if g.startswith('ai'):
        o['steer_raw_median'] = float(np.median([r['raw'] for r in rr]))
        o['gain_median'] = float(np.median([r['gain'] for r in rr]))
    out[g] = o
json.dump({'groups': out, 'rows': res}, sys.stdout, ensure_ascii=False, indent=1)
