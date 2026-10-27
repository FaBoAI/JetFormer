"""反証 4 回目・安全: 本物の記録(読むだけ)の VEHICLE.npz の間隔(車両の値が来ない間)の分布。
drive の HOLD_VEH_GAP_S(0.06 秒)・中立止め(max_sensor_age 0.15 秒)・rev_s(0.2〜0.4 秒)と比べる。

    /home/jetson/miniconda/envs/jetracer/bin/python -I scan_veh_gaps2.py /home/jetson/jetformer/data/raw PAT
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
pat = sys.argv[2]
th = (0.06, 0.15, 0.25, 0.35, 0.5)
out = {'recs': 0, 'veh_s': 0.0, 'n_int': 0, 'gt': {str(t): 0 for t in th}, 'gt_ai_select': {str(t): 0 for t in th},
       'max_s': 0.0, 'examples': []}
for vp in sorted(root.glob(pat + '-*/*/VEHICLE.npz')):
    if not (vp.parent / 'AI_CMD.npz').exists():
        continue
    try:
        v = np.load(vp)
    except Exception:
        continue
    tv = v['timestamps_us'].astype(np.int64)
    if len(tv) < 2:
        continue
    out['recs'] += 1
    dt = np.diff(tv) / 1e6
    sel = v['select'].astype(bool)
    out['veh_s'] += float(tv[-1] - tv[0]) / 1e6
    out['n_int'] += len(dt)
    out['max_s'] = max(out['max_s'], float(dt.max()))
    for t in th:
        out['gt'][str(t)] += int((dt > t).sum())
        out['gt_ai_select'][str(t)] += int(((dt > t) & sel[:-1] & sel[1:]).sum())
    for i in np.where(dt > 0.15)[0][:2]:
        if len(out['examples']) < 20:
            out['examples'].append([vp.parent.name, round(float((tv[i] - tv[0]) / 1e6), 2), round(float(dt[i]), 3),
                                    bool(sel[i])])
out['veh_s'] = round(out['veh_s'], 1)
print(json.dumps(out, ensure_ascii=False, indent=1))
