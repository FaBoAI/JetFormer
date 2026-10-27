"""反証 2 回目・safety: 人側 → AI 側にした直後(AI 側の最初の 5 フレーム)に中立止め(hold)があるか。
hold があると発進の前進はその区間で始まらない('hold')。理由別に数える。読むだけ。0 バイトは飛ばす"""
import collections, glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
res = {'switches': 0, 'first_frame_hold': 0, 'hold_in_first5': 0, 'by_day': collections.Counter(), 'ex': []}
for sc in sorted(glob.glob(root + '/2026*/*/AI_CMD.npz')):
    if os.path.getsize(sc) == 0:
        continue
    d = os.path.dirname(sc)
    try:
        z = np.load(sc)
        sel = z['select'].astype(bool); hold = z['hold'].astype(bool); t = z['timestamps_us'].astype(np.int64)
        dry = z['dry_run'].astype(bool) if 'dry_run' in z.files else np.zeros(len(sel), bool)
    except Exception:
        continue
    vp = d + '/VEHICLE.npz'
    v = np.load(vp) if os.path.exists(vp) and os.path.getsize(vp) > 0 else None
    for i in range(1, len(sel)):
        if sel[i] and not sel[i - 1] and not dry[i] and not hold[i - 1]:
            res['switches'] += 1
            day = os.path.basename(os.path.dirname(d))[:8]
            h5 = hold[i:i + 5]
            if hold[i]:
                res['first_frame_hold'] += 1
            if h5.any():
                res['hold_in_first5'] += 1
                res['by_day'][day] += 1
                if len(res['ex']) < 10 and v is not None:
                    vt = v['timestamps_us'].astype(np.int64)
                    m = (vt > t[i] - 60_000) & (vt <= t[min(i + 5, len(t) - 1)])
                    res['ex'].append({'scene': os.path.relpath(sc, root), 'row': i, 'hold5': h5.astype(int).tolist(),
                                      'veh_sel': v['select'][m].astype(int).tolist(),
                                      'veh_valid3': v['valid'][m][:, 2].astype(int).tolist()})
res['by_day'] = dict(res['by_day'])
print(json.dumps(res, ensure_ascii=False, indent=1))
