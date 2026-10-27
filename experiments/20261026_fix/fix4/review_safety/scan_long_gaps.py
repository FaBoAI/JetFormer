"""反証 4 回目・安全: 本物の記録(読むだけ)で、AI 側の処理したフレーム(hold でない・select・dry でない)どうしの間が
0.5 秒(assist.Recovery.HOLD_KEEP_MAX_S)より長い所を数え、明けたフレームの速さ・その間の IMU の水平 G の最大(drive の hold_g と
同じ hypot(ax, ay) / 9.80665)を出す。rec_hold_keep = 1 の日なら、今の assist.py がその場面で何をするかの元。

    /home/jetson/miniconda/envs/jetracer/bin/python -I scan_long_gaps.py /home/jetson/jetformer/data/raw '2026102[34]'
"""
import json
import sys
from pathlib import Path

import numpy as np

KV = 0.5e-3
root = Path(sys.argv[1])
pat = sys.argv[2] if len(sys.argv) > 2 else '2026102[34]'
out = {'recs': 0, 'ai_frames': 0, 'gaps_gt_0.5': [], 'gaps_gt_0.3': 0, 'max_gap_s': 0.0}
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
    hold = a['hold'].astype(bool)
    sel = a['select'].astype(bool) & ~a['dry_run'].astype(bool)
    proc = sel & ~hold
    out['ai_frames'] += int(proc.sum())
    ti = imu['timestamps_us'].astype(np.int64)
    g = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / 9.80665
    tv = veh['timestamps_us'].astype(np.int64)
    v = -veh['erpm'].astype(np.float64) * KV
    idx = np.where(proc)[0]
    for i0, i1 in zip(idx, idx[1:]):
        # 間に人側の行(select でない)があれば Recovery は戻る(数えない)
        if (~sel[i0:i1 + 1]).any():
            continue
        gap = (ta[i1] - ta[i0]) / 1e6
        out['max_gap_s'] = max(out['max_gap_s'], gap)
        if gap > 0.3:
            out['gaps_gt_0.3'] += 1
        if gap <= 0.5:
            continue
        m = (ti > ta[i0]) & (ti <= ta[i1])
        k0 = np.searchsorted(tv, ta[i0]) - 1
        k1 = np.searchsorted(tv, ta[i1]) - 1
        out['gaps_gt_0.5'].append({'rec': '%s/%s' % (d.parent.name, d.name), 't_s': round((ta[i0] - ta[0]) / 1e6, 2),
                                   'gap_s': round(gap, 3), 'v_before': round(float(v[k0]), 2) if k0 >= 0 else None,
                                   'v_after': round(float(v[k1]), 2) if k1 >= 0 else None,
                                   'hold_g_max': round(float(g[m].max()), 2) if m.any() else None,
                                   'hold_rows': int(hold[i0 + 1:i1].sum()),
                                   'assist_before': int(a['assist'][i0]) if 'assist' in a.files else None})
print(json.dumps(out, ensure_ascii=False, indent=1))
