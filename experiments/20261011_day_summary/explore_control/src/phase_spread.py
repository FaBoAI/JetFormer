#!/usr/bin/env python3
"""ジャイロの向きによる区切り(R1 開始、ψ = −20 / +40 / +175 / +215)が、コース図の進み s [m] の上で
周ごとにどれだけばらつくか(= オンラインの区間判定の位置の精度の目安)。読み取りのみ、CPU。"""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load  # noqa: E402
TR = '/home/jetson/jetformer_reports/overnight_20261011/course/tracks_%s.npz'
SEC = json.load(open('/home/jetson/jetformer_reports/summary_20261010/explore_control/sections.json'))
res = {}
for log, name in (('20261010-172910', 'ai_run_006'), ('20261010-173445', 'ai_run_007')):
    _, d = load(log, ('IMU',))
    t_abs = d['IMU']['timestamps_us']
    t0 = t_abs[0]
    z = np.load(TR % log, allow_pickle=True)
    laps = [l for x in SEC['logs'] if x['log'] == log for l in x['laps']]
    pts = {}
    for l in laps:
        tb = l['t0']
        bounds = {'R1_start': tb}
        acc = tb
        for k in ['R1', 'GM', 'DE', 'SW', 'HS']:
            acc += l['sec'][k]['t_s']
            bounds[{'R1': 'GM_start(psi_min)', 'GM': 'psi=-20', 'DE': 'psi=+40', 'SW': 'psi=+175', 'HS': 'psi=+215'}[k]] = acc
        for k, tt in bounds.items():
            tu = t0 + tt * 1e6
            i = int(np.clip(np.searchsorted(z['t_us'], tu), 0, len(z['t_us']) - 1))
            if not z['valid'][i] or abs(z['t_us'][i] - tu) > 50_000:
                continue
            pts.setdefault(k, []).append(float(z['progress_m'][i]))
    out = {}
    for k, v in pts.items():
        v = np.array(v)
        # 周の境目付近(25.48 m)をまたぐ値は折り返して扱う
        if v.max() - v.min() > 12:
            v = np.where(v < 12.74, v + 25.48, v)
        out[k] = {'n': len(v), 's_med_m': round(float(np.median(v)) % 25.48, 2), 'iqr_m': round(float(np.subtract(*np.percentile(v, [75, 25]))), 2),
                  'std_m': round(float(np.std(v)), 2), 'range_m': round(float(v.max() - v.min()), 2)}
    res[name] = out
json.dump(res, sys.stdout, ensure_ascii=False, indent=1)
