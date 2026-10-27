"""反証 2 回目・safety: リカバリーごとに、下がった速さの最大・下がった距離(eRPM > +300 の間の ∫v dt)・
最初の「ブレーキ」の段でもう下がったか・直前の非中立の指令(F/R、>1538 を R)。読むだけ。0 バイトは飛ばす"""
import glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
KV = 0.5e-3
rows = []
for sc in sorted(glob.glob(root + '/2026102*/*/AI_CMD.npz')):
    d = os.path.dirname(sc)
    vp = d + '/VEHICLE.npz'
    if os.path.getsize(sc) == 0 or not os.path.exists(vp) or os.path.getsize(vp) == 0:
        continue
    z = np.load(sc); v = np.load(vp)
    t = z['timestamps_us'].astype(np.int64); a = z['assist'].astype(int); out = z['out'].astype(float)
    vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
    n = len(t)
    for i in range(1, n):
        if not (a[i] == 2 and a[i - 1] != 2):
            continue
        j = i
        while j < n and a[j] == 2:
            j += 1
        tb = t[i]
        k = i
        while k < j and out[k, 1] > 1650:          # 最初のブレーキの段(中立 + rev_us)
            k += 1
        m_all = (vt >= t[i]) & (vt <= t[min(j, n - 1)])
        m_br = (vt >= t[i]) & (vt <= t[min(k, n - 1)])
        e = ve[m_all]; tt = vt[m_all]
        back = e > 300
        dist = float(np.sum(np.where(back[1:], e[1:] * KV * np.diff(tt) / 1e6, 0.0))) if len(e) > 1 else 0.0
        prev = [o for o in out[max(0, i - 120):i, 1] if o < 1478 or o > 1538]
        rows.append({'scene': os.path.relpath(sc, root), 'row': i, 'len_s': round((t[j - 1] - t[i]) / 1e6, 2),
                     'back_in_brake': bool((ve[m_br] > 300).any()), 'max_back_erpm': int(e.max()) if len(e) else 0,
                     'max_back_mps': round(float(e.max()) * KV, 2) if len(e) else 0, 'dist_back_m': round(dist, 2),
                     'last_nonneutral': ('R' if prev and prev[-1] > 1538 else 'F') if prev else None})
print(json.dumps(rows, ensure_ascii=False, indent=0))
