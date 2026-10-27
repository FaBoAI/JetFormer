"""反証 2 回目・safety: 前へ走っていて衝突(水平 2.5G 以上)し、そのあと止まった(|eRPM| < 400 が 0.3 秒以上)ときの
ToF ガードの TTC の直近 5 フレームの中央値(発進の前進が「前が空いている」とみるか、≥ 3.3 秒)。読むだけ。0 バイトは飛ばす"""
import glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
G = 9.80665
cases = []
for sc in sorted(glob.glob(root + '/2026*/*/AI_CMD.npz')):
    d = os.path.dirname(sc)
    paths = [sc, d + '/VEHICLE.npz', d + '/IMU.npz']
    if any((not os.path.exists(p)) or os.path.getsize(p) == 0 for p in paths):
        continue
    try:
        z = np.load(sc)
        if 'ttc' not in z.files:
            continue
        t = z['timestamps_us'].astype(np.int64); ttc = z['ttc'].astype(float); hold = z['hold'].astype(bool)
        sel = z['select'].astype(bool); a = z['assist'].astype(int)
        v = np.load(paths[1]); vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
        im = np.load(paths[2]); it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
    except Exception:
        continue
    if not np.isfinite(ttc).any():
        continue
    hg = np.hypot(acc[:, 0], acc[:, 1]) / G
    # 衝突: 水平 2.5G 以上で、その 0.2 秒前に前へ 1000 eRPM より速く走っていた
    hits = []
    for k in np.nonzero(hg >= 2.5)[0]:
        tk = it[k]
        if hits and tk - hits[-1] < 1_500_000:
            continue
        j = np.searchsorted(vt, tk - 200_000) - 1
        if j >= 0 and ve[j] < -1000:
            hits.append(tk)
    for tk in hits:
        # 衝突から 0.3〜3 秒の間で、止まって 0.3 秒たった最初の AI のフレーム(中立止めでない)
        idx = np.nonzero((t >= tk + 300_000) & (t <= tk + 3_000_000))[0]
        for i in idx:
            jv = np.searchsorted(vt, t[i], side='right') - 1
            win = (vt > t[i] - 300_000) & (vt <= t[i])
            if jv < 0 or not win.any() or np.abs(ve[win]).max() >= 400:
                continue
            w = ttc[max(0, i - 4):i + 1]
            med = float(np.median(w)) if len(w) == 5 and np.isfinite(w).all() else float('nan')
            cases.append({'scene': os.path.relpath(sc, root), 'row': int(i), 'impact_g': round(float(hg[np.searchsorted(it, tk)]), 2),
                          's_after': round((t[i] - tk) / 1e6, 2), 'select': bool(sel[i]), 'hold': bool(hold[i]), 'assist': int(a[i]),
                          'ttc5': [round(float(x), 2) for x in w], 'med': None if not np.isfinite(med) else round(med, 2),
                          'boost_would_see_clear': bool(np.isfinite(med) and med >= 3.3)})
            break
fin = [c for c in cases if c['med'] is not None]
print(json.dumps({'impacts_then_stopped': len(cases), 'with_5_ttc': len(fin),
                  'median_ge_3.3': sum(c['boost_would_see_clear'] for c in fin),
                  'ttc_med_quantiles': [round(float(np.percentile([c['med'] for c in fin], q)), 2) for q in (0, 10, 25, 50, 75, 90, 100)] if fin else [],
                  'clear_cases': [c for c in fin if c['boost_would_see_clear']],
                  'near_cases_sample': [c for c in fin if not c['boost_would_see_clear']][:8]}, ensure_ascii=False, indent=1))
