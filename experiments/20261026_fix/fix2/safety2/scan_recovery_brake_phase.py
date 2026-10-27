"""反証 2 回目・safety: リカバリーの最初の「ブレーキ」(中立 + rev_us を 0.25 秒。前進の直後の後退側は必ずブレーキ = 捨てる 1 回目、
という ESC の決まりが前提)の間に、車がもう後ろへ動いたか(eRPM > +300)。動いたなら、その直前の出力に
「止まる前の後退側(ブレーキ)→ 中立」があったか。読むだけ。0 バイトは飛ばす"""
import glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
rows = []
for sc in sorted(glob.glob(root + '/2026*/*/AI_CMD.npz')):
    d = os.path.dirname(sc)
    vp = d + '/VEHICLE.npz'
    if os.path.getsize(sc) == 0 or not os.path.exists(vp) or os.path.getsize(vp) == 0:
        continue
    try:
        z = np.load(sc); v = np.load(vp)
        t = z['timestamps_us'].astype(np.int64); a = z['assist'].astype(int); out = z['out'].astype(float)
        hold = z['hold'].astype(bool)
        vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
    except Exception:
        continue
    n = len(t)
    for i in range(1, n):
        if a[i] == 2 and a[i - 1] != 2:
            neutral = 1496.0 if abs(out[i, 1] - 1796) < 1 else None
            # ブレーキの間 = 出力が中立 + 200 µs 以上のフレームが続く間(最大 0.3 秒)
            j = i
            while j < n and a[j] == 2 and out[j, 1] > 1496 + 150 and t[j] - t[i] <= 300_000:
                j += 1
            m = (vt >= t[i]) & (vt <= t[min(j, n - 1)] + 30_000)
            back = bool((ve[m] > 300).any())
            # 直前 3 秒の出力: 走っている間(eRPM < -150)の後退側(> 中立 + 40)→ そのあと中立のあいだ
            k0 = np.searchsorted(t, t[i] - 3_000_000)
            prev = []
            for k in range(k0, i):
                jv = np.searchsorted(vt, t[k], side='right') - 1
                e = ve[jv] if jv >= 0 else np.nan
                o = out[k, 1]
                prev.append(('F' if o < 1456 else ('R' if o > 1536 else 'n')) + ('m' if e < -150 else 's'))
            last_motion = next((x for x in reversed(prev) if x[0] in 'FR'), None)
            rows.append({'scene': os.path.relpath(sc, root), 'row': i, 'brake_frames': int(j - i),
                         'backed_during_brake': back, 'erpm_brake': [int(x) for x in ve[m][::2][:8]],
                         'last_motion_cmd': last_motion, 'hold_before': bool(hold[k0:i].any()),
                         'prev_tail': ''.join(x[0] for x in prev[-30:])})
print(json.dumps({'recovery_starts': len(rows), 'backed_during_brake': sum(r['backed_during_brake'] for r in rows),
                  'last_motion_R': sum(r['last_motion_cmd'] is not None and r['last_motion_cmd'][0] == 'R' for r in rows),
                  'rows': rows}, ensure_ascii=False))
