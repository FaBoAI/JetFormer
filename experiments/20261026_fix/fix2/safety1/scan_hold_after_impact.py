"""AI 側の中立止め(hold)の前後: 直前 1 秒の水平の衝撃・止める前と後の eRPM・後のリカバリー(読むだけ)。
中立止めで Assist.reset() がリカバリーを最初から(armed = False)にするので、ぶつかって止まった直後の中立止めのあとに
リカバリーが出ないまま AI 側で止まり続けた場面を探す"""
import glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
G = 9.80665
out = []
for sc in sorted(glob.glob(root + '/20261*/*/AI_CMD.npz')):
    d = os.path.dirname(sc)
    try:
        if min(os.path.getsize(os.path.join(d, f)) for f in ('AI_CMD.npz', 'IMU.npz', 'VEHICLE.npz')) == 0:
            continue
        z = np.load(sc); imu = np.load(os.path.join(d, 'IMU.npz')); veh = np.load(os.path.join(d, 'VEHICLE.npz'))
        t = z['timestamps_us'] / 1e6; hold = z['hold'].astype(bool); sel = z['select'].astype(bool)
        a = z['assist'].astype(int); thr = z['out'][:, 1].astype(float)
        ti = imu['timestamps_us'] / 1e6; hg = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / G
        tv = veh['timestamps_us'] / 1e6; erpm = veh['erpm'].astype(float); vsel = veh['select'].astype(bool)
    except Exception:
        continue
    ai = sel & ~z['dry_run'].astype(bool)
    n, i = len(hold), 0
    while i < n:
        if ai[i] and hold[i] and i > 0 and ai[i - 1] and not hold[i - 1]:
            j = i
            while j < n and ai[j] and hold[j]:
                j += 1
            t0, t1 = t[i], t[j - 1]
            m = (ti >= t0 - 1.0) & (ti <= t0)
            imp = float(hg[m].max()) if m.any() else float('nan')
            # 後の 3 秒: AI 側のまま・止まっている(|eRPM| < 300)時間、リカバリーの有無、出したスロットル
            k = (t > t1) & (t <= t1 + 3.0)
            mv = (tv > t1) & (tv <= t1 + 3.0)
            still = float(np.mean(np.abs(erpm[mv]) < 300)) if mv.any() else float('nan')
            before_e = erpm[(tv >= t0 - 0.5) & (tv < t0)]
            out.append({'scene': os.path.relpath(sc, root), 'row': i, 'n_hold': int(j - i), 'hold_s': round(float(t1 - t0), 2),
                        'impact_1s_before_G': round(imp, 2),
                        'erpm_before_max_abs': int(np.max(np.abs(before_e))) if len(before_e) else None,
                        'after3s_ai_rows': int((k & ai).sum()), 'after3s_still_frac': round(still, 2),
                        'after3s_recovery_rows': int(((a == 2) | (a == 3))[k].sum()),
                        'after3s_thr_min': round(float(np.min(thr[k & ai & ~hold])), 1) if (k & ai & ~hold).any() else None,
                        'after3s_human_rows': int((k & ~sel).sum())})
            i = j
        else:
            i += 1
print(json.dumps(out, ensure_ascii=False, indent=1))
