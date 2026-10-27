"""safety2-1 の確かめ: 10/23・10/24 のリカバリーごとに、後ろへ下がった間・そのあと 1 秒の水平の衝撃の最大、
後退側の出力(> 中立+40)を出していた時間、下がり始めた時刻(リカバリーの始まりから)。読むだけ。0 バイトは飛ばす"""
import glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
N = 1496.0
for sc in sorted(glob.glob(root + '/2026102*/*/AI_CMD.npz')):
    d = os.path.dirname(sc)
    vp, ip = d + '/VEHICLE.npz', d + '/IMU.npz'
    if any((not os.path.exists(p)) or os.path.getsize(p) == 0 for p in (sc, vp, ip)):
        continue
    z = np.load(sc); v = np.load(vp); im = np.load(ip)
    t = z['timestamps_us'].astype(np.int64); a = z['assist'].astype(int); out = z['out'].astype(float)
    vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
    it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
    try:
        p = json.loads(open(os.path.dirname(d) + '/log.json').read())['drive']
        model = p.get('run') or p.get('model')
    except Exception:
        model = None
    n = len(t)
    for i in range(1, n):
        if not (a[i] == 2 and a[i - 1] != 2):
            continue
        j = i
        while j < n and a[j] == 2:
            j += 1
        t0, t1 = t[i], t[min(j, n - 1)]
        m = (vt >= t0) & (vt <= t1 + 300_000)
        back = vt[m][ve[m] > 300]
        tb = round((back[0] - t0) / 1e6, 2) if len(back) else None
        rev_frames = [k for k in range(i, j) if out[k, 1] > N + 40]
        mi = (it >= t0) & (it <= t1 + 1_000_000)
        # 衝撃の時刻
        gmax = float(hg[mi].max()) if mi.any() else 0.0
        tg = round((it[mi][np.argmax(hg[mi])] - t0) / 1e6, 2) if mi.any() else None
        # 始まる前 1 秒の衝撃(前にぶつかった)
        mb = (it >= t0 - 1_500_000) & (it < t0)
        gpre = float(hg[mb].max()) if mb.any() else 0.0
        print(json.dumps({'scene': os.path.relpath(d, root), 'row': i, 'model': model, 'len_s': round((t1 - t0) / 1e6, 2),
                          'start_back_s': tb, 'rev_side_s': round(len(rev_frames) / 30, 2),
                          'max_back_erpm': int(ve[m].max()) if m.any() else 0,
                          'impact_after_max_g': round(gmax, 2), 'impact_after_t_s': tg, 'impact_before_g': round(gpre, 2)},
                         ensure_ascii=False))
