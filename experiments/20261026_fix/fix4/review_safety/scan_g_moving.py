"""反証 4 回目・安全: 本物の記録(読むだけ)で、走っている車(前へ 1.0 m/s 以上)に水平 2.5G 以上(drive の impact_g・hold_g と
同じ hypot(ax, ay) / 9.80665)が来て、0.5 秒後・1.0 秒後もまだ前へ 1.0 m/s 以上で走っていた場面を数える。
= rec_hold_keep = 1 の日に、その衝撃が 0.5 秒より長い中立止めの間に来たら、今の assist.py(r3-r3-safety-1)では
  走っている車のまま「中止」(中立・ハンドルまっすぐ)になる場面の元。

    /home/jetson/miniconda/envs/jetracer/bin/python -I scan_g_moving.py /home/jetson/jetformer/data/raw 2026102[34]
"""
import json
import sys
from pathlib import Path

import numpy as np

KV = 0.5e-3
root = Path(sys.argv[1])
pat = sys.argv[2] if len(sys.argv) > 2 else '2026102[34]'
out = {'recs': 0, 'ai_s': 0.0, 'spikes_moving': 0, 'events_moving': 0, 'still_fast_0.5s': 0, 'still_fast_1.0s': 0,
       'examples': []}
for ai in sorted(root.glob(pat + '-*/*/AI_CMD.npz')):
    d = ai.parent
    try:
        a = np.load(ai)
        imu = np.load(d / 'IMU.npz')
        veh = np.load(d / 'VEHICLE.npz')
    except Exception:
        continue
    out['recs'] += 1
    ta = a['timestamps_us'].astype(np.int64)
    sel = a['select'].astype(bool) & ~a['dry_run'].astype(bool) & ~a['hold'].astype(bool)
    if len(ta) > 1:
        dt = np.diff(ta) / 1e6
        out['ai_s'] += float(np.sum(np.where(sel[:-1] & (dt <= 0.2), dt, 0.0)))
    ti = imu['timestamps_us'].astype(np.int64)
    g = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / 9.80665
    tv = veh['timestamps_us'].astype(np.int64)
    v = -veh['erpm'].astype(np.float64) * KV
    vsel = veh['select'].astype(bool)
    if len(tv) < 2:
        continue
    last_ev = -10 ** 18
    for j in np.where(g >= 2.5)[0]:
        t = ti[j]
        k = np.searchsorted(tv, t) - 1
        if k < 0 or not vsel[k]:
            continue
        if v[k] < 1.0:
            continue
        out['spikes_moving'] += 1
        if t - last_ev < 1_000_000:
            continue
        last_ev = t
        out['events_moving'] += 1
        def v_at(dt_us):
            kk = np.searchsorted(tv, t + dt_us) - 1
            return float(v[kk]) if 0 <= kk < len(v) and tv[-1] >= t + dt_us else None
        v05, v10 = v_at(500_000), v_at(1_000_000)
        f05 = v05 is not None and v05 >= 1.0
        f10 = v10 is not None and v10 >= 1.0
        out['still_fast_0.5s'] += int(f05)
        out['still_fast_1.0s'] += int(f10)
        if f05 and len(out['examples']) < 40:
            out['examples'].append({'rec': '%s/%s' % (d.parent.name, d.name), 't_s': round((t - ta[0]) / 1e6, 2),
                                    'g': round(float(g[j]), 2), 'v': round(float(v[k]), 2), 'v_0.5s': round(v05, 2),
                                    'v_1.0s': None if v10 is None else round(v10, 2)})
out['ai_s'] = round(out['ai_s'], 1)
print(json.dumps(out, ensure_ascii=False, indent=1))
