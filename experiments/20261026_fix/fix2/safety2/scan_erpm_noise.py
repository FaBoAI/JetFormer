"""反証 2 回目・safety: 止まっている(hall_count が 0.1 秒変わらない)のに eRPM が前へ(< −150)・後ろへ(> +300)を示すサンプル。
補助の「止まっている間のモデルの後退は中立に」(eRPM ≥ −150 で判定)が外れる場面があるか。読むだけ。0 バイトは飛ばす"""
import collections, glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
cnt = collections.Counter()
ex = []
for vp in sorted(glob.glob(root + '/2026102*/*/VEHICLE.npz')):
    if os.path.getsize(vp) == 0:
        continue
    try:
        v = np.load(vp)
        t = v['timestamps_us'].astype(np.int64); e = v['erpm'].astype(int); h = v['hall_count'].astype(np.int64)
        sel = v['select'].astype(bool)
    except Exception:
        continue
    j = 0
    for i in range(len(t)):
        while t[i] - t[j] > 100_000:
            j += 1
        if t[i] - t[j] < 80_000:
            continue
        still = h[i] == h[j]
        if not still:
            continue
        cnt['still'] += 1
        if e[i] < -150:
            cnt['still_fwd_lt_-150'] += 1
            if sel[i]:
                cnt['still_fwd_ai'] += 1
            if len(ex) < 6:
                ex.append({'scene': os.path.relpath(os.path.dirname(vp), root), 'i': int(i), 'erpm': int(e[i]), 'sel': bool(sel[i])})
        if e[i] > 300:
            cnt['still_back_gt_300'] += 1
print(json.dumps({'counts': dict(cnt), 'examples': ex}, ensure_ascii=False, indent=1))
