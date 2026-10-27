"""反証 2 回目・safety: AI 側で、リカバリーでないのに車が後ろへ動いた(eRPM > +500 が 0.3 秒以上)ところ(読むだけ)。
直前 2 秒の出力(PCA9685 に書いた out)の並びを残す。0 バイトの npz は飛ばす"""
import glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
res = []
n_sc = 0
for sc in sorted(glob.glob(root + '/2026*/*/AI_CMD.npz')):
    d = os.path.dirname(sc)
    vp = os.path.join(d, 'VEHICLE.npz')
    if os.path.getsize(sc) == 0 or not os.path.exists(vp) or os.path.getsize(vp) == 0:
        continue
    try:
        z = np.load(sc); v = np.load(vp)
        t = z['timestamps_us'].astype(np.int64); sel = z['select'].astype(bool); hold = z['hold'].astype(bool)
        a = z['assist'].astype(int); out = z['out'].astype(float)
        dry = z['dry_run'].astype(bool) if 'dry_run' in z.files else np.zeros(len(t), bool)
        vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
    except Exception:
        continue
    n_sc += 1
    idx = np.searchsorted(vt, t, side='right') - 1
    ok = idx >= 0
    erpm = np.where(ok, ve[np.clip(idx, 0, len(ve) - 1)], np.nan)
    ai = sel & ~dry & ~hold
    back = ai & (erpm > 500) & (a != 2) & (a != 3)
    i = 0
    n = len(t)
    while i < n:
        if back[i]:
            j = i
            while j < n and back[j]:
                j += 1
            dur = (t[j - 1] - t[i]) / 1e6
            if dur >= 0.3:
                k0 = max(0, i - 60)
                res.append({'scene': os.path.relpath(sc, root), 'row': int(i), 'dur_s': round(dur, 2),
                            'erpm_max': float(np.nanmax(erpm[i:j])), 'assist_in': sorted(set(int(x) for x in a[i:j])),
                            'out_thr_before': [int(round(x)) for x in out[k0:i + 3, 1]],
                            'assist_before': [int(x) for x in a[k0:i + 3]],
                            'erpm_before': [int(x) if np.isfinite(x) else None for x in erpm[k0:i + 3]]})
            i = j
        else:
            i += 1
print(json.dumps({'scenes': n_sc, 'n': len(res), 'cases': res}, ensure_ascii=False))
