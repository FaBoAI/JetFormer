"""反証 3 回目・safety: 本物の記録(10/23・10/24)の VEHICLE の間隔の分布(読むだけ)。
中立止めの間に HOLD_VEH_GAP_S(0.06 秒)を超える間が普通にあるか。AI_CMD の hold 行の近くも数える。"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out = {'recs': 0, 'n_int': 0, 'gt60': 0, 'gt30': 0, 'max_ms': 0.0, 'hold_rows': 0, 'hold_rows_gap60': 0, 'examples': []}
for ai in sorted(root.glob('2026102[34]-*/*/AI_CMD.npz')):
    d = ai.parent
    vp = d / 'VEHICLE.npz'
    if not vp.exists():
        continue
    try:
        v = np.load(vp)
        a = np.load(ai)
    except Exception:
        continue
    tv = v['timestamps_us'].astype(np.int64)
    if len(tv) < 2:
        continue
    out['recs'] += 1
    dt = np.diff(tv) / 1000.0
    out['n_int'] += len(dt)
    out['gt60'] += int((dt > 60).sum())
    out['gt30'] += int((dt > 30).sum())
    out['max_ms'] = max(out['max_ms'], float(dt.max()))
    # hold rows on AI side
    ta = a['timestamps_us'].astype(np.int64)
    hold = a['hold'].astype(bool)
    sel = a['select'].astype(bool)
    for t in ta[hold & sel]:
        out['hold_rows'] += 1
        j = np.searchsorted(tv, t)
        lo, hi = max(j - 15, 1), min(j + 15, len(tv))
        if hi > lo and (np.diff(tv[lo - 1:hi]) > 60000).any():
            out['hold_rows_gap60'] += 1
    big = np.where(dt > 60)[0]
    for i in big[:3]:
        if len(out['examples']) < 15:
            out['examples'].append([d.name, round(float(dt[i]), 1)])
print(json.dumps(out, ensure_ascii=False, indent=1))
