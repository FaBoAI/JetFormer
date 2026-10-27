"""AI のフレーム(中立止めでない)で、撮像時刻までの最新の車両の値の古さ(読むだけ)。
0.1 秒超・0.15 秒以下 = 走行は続くが、速さの天井・区間の上限は動かさない(OFF と同じ)・発進の前進は終わる"""
import glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
tot = 0; mid = []; ages_all = []
for sc in sorted(glob.glob(root + '/2026102*/*/AI_CMD.npz')):
    d = os.path.dirname(sc)
    try:
        if min(os.path.getsize(os.path.join(d, f)) for f in ('AI_CMD.npz', 'VEHICLE.npz')) == 0:
            continue
        z = np.load(sc); v = np.load(os.path.join(d, 'VEHICLE.npz'))
        t = z['timestamps_us']; tv = v['timestamps_us']; e = v['erpm']
        ok = z['select'].astype(bool) & ~z['hold'].astype(bool) & ~z['dry_run'].astype(bool)
    except Exception:
        continue
    idx = np.searchsorted(tv, t, side='right') - 1
    for i in np.where(ok & (idx >= 0))[0]:
        age = (t[i] - tv[idx[i]]) / 1e3
        tot += 1
        ages_all.append(age)
        if 100 < age <= 150:
            mid.append({'scene': os.path.relpath(sc, root), 'row': int(i), 'age_ms': round(float(age), 1),
                        'v_mps': round(float(-e[idx[i]] * 0.5e-3), 2), 'out_thr': round(float(z['out'][i, 1]), 1)})
a = np.array(ages_all)
print(json.dumps({'ai_frames': tot, 'age_p50_ms': round(float(np.median(a)), 1), 'age_p99_ms': round(float(np.percentile(a, 99)), 1),
                  'age_max_ms': round(float(a.max()), 1), 'n_100_150': len(mid), 'examples': mid[:10]}, ensure_ascii=False, indent=1))
