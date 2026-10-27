"""反証の確かめ(r3-r3-safety-1)。本物の記録(読むだけ)で、AI 側の当たり(水平 2.5G 以上)のあと、処理したフレームの
次までの間(見ていない時間)が 0.5 秒より長いか、当たりと次の処理の間に 1 フレームでも処理したかを数える。"""
import json
import sys
from pathlib import Path

import numpy as np

G0 = 9.80665
root = Path('/home/jetson/jetformer/data/raw')
out = {}
for day in sys.argv[1:]:
    hits = []
    for ai in sorted(root.glob(day + '-*/*/AI_CMD.npz')):
        d = ai.parent
        try:
            a = np.load(ai)
            imu = np.load(d / 'IMU.npz')
        except Exception:
            continue
        ta = a['timestamps_us'].astype(np.int64)
        sel = a['select'].astype(bool)
        hold = a['hold'].astype(bool)
        dry = a['dry_run'].astype(bool) if 'dry_run' in a.files else np.zeros(len(ta), bool)
        proc = ta[sel & ~hold & ~dry]
        ti = imu['timestamps_us'].astype(np.int64)
        g = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / G0
        cam = None
        try:
            cam = np.load(d / 'CAM_FRONT.npz')['timestamps_us'].astype(np.int64)
        except Exception:
            pass
        # 当たりの山(2.5G 以上、0.3 秒以内は 1 つ)
        idx = np.where(g >= 2.5)[0]
        last_t = -10**12
        for i in idx:
            t = ti[i]
            if t - last_t < 300000:
                continue
            last_t = t
            # AI 側の区間の中か(前後 0.5 秒に AI 側の行)
            j = np.searchsorted(ta, t)
            if j == 0 or j >= len(ta) or not (sel[j - 1] and sel[min(j, len(ta) - 1)]):
                continue
            if dry[j - 1]:
                continue
            k = np.searchsorted(proc, t)
            if k == 0 or k >= len(proc):
                continue
            prev, nxt = proc[k - 1], proc[k]
            gap = (nxt - prev) / 1e6
            cam_gap = None
            if cam is not None and len(cam):
                c = np.searchsorted(cam, t)
                if 0 < c < len(cam):
                    cam_gap = round((cam[c] - cam[c - 1]) / 1e6, 3)
            hits.append({'rec': d.name, 'g': round(float(g[i]), 2), 'last_proc_before_ms': round((t - prev) / 1000, 1),
                         'next_proc_after_s': round((nxt - t) / 1e6, 3), 'unseen_s': round(gap, 3),
                         'hold_rows_between': int(((ta > prev) & (ta < nxt) & hold & sel).sum()), 'cam_gap_s': cam_gap})
    long_ = [h for h in hits if h['unseen_s'] > 0.5]
    out[day] = {'hits_ai': len(hits), 'unseen_gt_0.5': len(long_), 'long': long_,
                'max_unseen_s': max([h['unseen_s'] for h in hits], default=None)}
print(json.dumps(out, ensure_ascii=False, indent=1))
