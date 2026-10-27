"""ぶつかって止まったとき(リカバリーの始まり・リカバリー中止・人が取り返した直後で止まっている)の ToF ガードの TTC(読むだけ)。
発進の前進の「前が空いている」(直近 5 フレームの TTC の中央値 ≥ 3.3 秒)が、壁に当たって止まった車で成り立つかを見る"""
import glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
KV = 0.5e-3
rows = []
for sc in sorted(glob.glob(root + '/20261*/*/AI_CMD.npz')):
    if os.path.getsize(sc) == 0:
        continue
    try:
        z = np.load(sc)
        a = z['assist'].astype(int); hold = z['hold'].astype(bool); sel = z['select'].astype(bool)
        ttc = z['ttc'].astype(float)
    except Exception:
        continue
    if 'ttc' not in z.files:
        continue
    n = len(a)
    for i in range(1, n):
        if a[i] == 2 and a[i - 1] != 2:          # リカバリーの始まり = 止まった(ぶつかったあと)
            w = ttc[max(0, i - 4):i + 1]
            w = w[np.isfinite(w)]
            med = float(np.median(w)) if len(w) == 5 else float('nan')
            rows.append({'scene': os.path.relpath(sc, root), 'row': i, 'ttc5': [round(float(x), 2) for x in ttc[max(0, i - 4):i + 1]],
                         'med': round(med, 2), 'would_start(>=3.3)': bool(np.isfinite(med) and med >= 3.3)})
st = [r for r in rows if r['would_start(>=3.3)']]
print(json.dumps({'recovery_starts': len(rows), 'with_5_finite': sum(1 for r in rows if np.isfinite(r['med'])),
                  'median_ge_3.3': len(st), 'examples_ge_3.3': st[:12],
                  'examples_lt': [r for r in rows if np.isfinite(r['med']) and r['med'] < 3.3][:6]}, ensure_ascii=False, indent=1))
