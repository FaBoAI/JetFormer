"""反証 safety2-1: 記録のリカバリーごとに、直前の非中立の指令(F/R)・'brake' の 0.25 秒で下がったか・下がった速さと長さ・衝撃を並べる。
読むだけ。0 バイト・壊れた npz は飛ばす"""
import glob, json, os, sys
import numpy as np
KV = 0.5e-3
NB_LO, NB_HI = 1478.0, 1538.0
out_rows = []
for f in sorted(glob.glob('/home/jetson/jetformer/data/raw/*/*/AI_CMD.npz')):
    if os.path.getsize(f) == 0:
        continue
    d = os.path.dirname(f)
    try:
        z = np.load(f); kind = z['assist']
        if not (kind == 2).any():
            continue
        v = np.load(d + '/VEHICLE.npz'); im = np.load(d + '/IMU.npz')
    except Exception:
        continue
    L = json.load(open(os.path.dirname(d) + '/log.json'))
    n = float(L['drive']['throttle_neutral_us']); p = L['drive']['params']
    t = z['timestamps_us'].astype(np.int64); out = z['out'][:, 1].astype(float)
    vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
    it = im['timestamps_us'].astype(np.int64); hg = np.hypot(im['accel'][:, 0], im['accel'][:, 1]) / 9.80665
    starts = [i for i in range(len(kind)) if kind[i] == 2 and (i == 0 or kind[i - 1] != 2)]
    for s in starts:
        e_ = s
        while e_ + 1 < len(kind) and kind[e_ + 1] == 2:
            e_ += 1
        # 直前の非中立(中立の帯 1478〜1538 の外)
        last, last_us, last_dt = None, None, None
        for j in range(s - 1, max(-1, s - 150), -1):
            if out[j] < n - 18 or out[j] > n + 42:
                last = 'F' if out[j] < n else 'R'; last_us = round(float(out[j])); last_dt = round((t[s] - t[j]) / 1e6, 2)
                break
        t0 = t[s]
        m_brake = (vt >= t0) & (vt <= t0 + 300_000)
        e_brake = float(ve[m_brake].max()) if m_brake.any() else np.nan
        t1 = t[e_] + 33_000
        m = (vt >= t0) & (vt <= t1)
        ee = ve[m]; tt = vt[m]
        back_m = float(np.sum(np.clip(ee[:-1], 0, None) * KV * np.diff(tt) / 1e6)) if len(ee) > 1 else 0.0
        mi = (it >= t0) & (it <= t1)
        # 後退側(> n+42)を出し、後ろへ動いていた(eRPM>300)フレームの数
        rv = 0
        for j in range(s, e_ + 1):
            jv = np.searchsorted(vt, t[j], side='right') - 1
            if out[j] > n + 42 and ve[jv] > 300:
                rv += 1
        out_rows.append({'rec': os.path.basename(d), 'row': s, 'rows': e_ - s + 1, 'p': {k: p.get(k) for k in ('rev_us', 'rev_s')},
                         'erpm0': int(ve[np.searchsorted(vt, t0, side='right') - 1]),
                         'last_nonneutral': last, 'last_us': last_us, 'last_dt_s': last_dt,
                         'brake_phase_max_erpm': int(e_brake), 'backed_in_brake': bool(e_brake > 300),
                         'max_back_erpm': int(ee.max()) if len(ee) else None, 'back_m': round(back_m, 2),
                         'rev_side_while_backing_s': round(rv / 30, 2), 'max_g': round(float(hg[mi].max()), 2) if mi.any() else None})
for r in out_rows:
    print(json.dumps(r, ensure_ascii=False))
