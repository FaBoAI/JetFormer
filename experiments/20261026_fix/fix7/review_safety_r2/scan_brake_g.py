"""反証 7-2 回目・safety: 本物の記録(data/raw、読むだけ)で、自動リカバリー(AI_CMD の assist = 2)が始まったフレーム(前の行が 2 でない)の
あと 0〜0.5 秒の IMU の水平 G の最大と、そのときの eRPM(後退側を書き始める前後の揺れ = 'brake'・'wait'・'gap' の間の衝撃が
どれだけあるか)。rear_impact_hold は中立止めの間の水平 G ≥ 2.5 だけで後退側の段から escape(前進)に入るので、その入り口の大きさを見る。
出力 out/scan_brake_g.json
    python -I scan_brake_g.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = '/home/jetson/jetformer/data/raw'


def main():
    out = []
    for fp in sorted(glob.glob(ROOT + '/2026*/*/AI_CMD.npz')):
        d0 = os.path.dirname(fp)
        ip, vp = os.path.join(d0, 'IMU.npz'), os.path.join(d0, 'VEHICLE.npz')
        if not (os.path.exists(ip) and os.path.exists(vp)) or min(os.path.getsize(x) for x in (fp, ip, vp)) == 0:
            continue
        try:
            a = np.load(fp)
            im = np.load(ip)
            ve = np.load(vp)
            t = a['timestamps_us'].astype(np.int64)
            asst = a['assist'].astype(int)
            hold = a['hold'].astype(bool)
            out_thr = a['out'][:, 1].astype(float)
            it = im['timestamps_us'].astype(np.int64)
            g = np.hypot(im['accel'][:, 0], im['accel'][:, 1]) / 9.80665
            vt = ve['timestamps_us'].astype(np.int64)
            er = ve['erpm'].astype(float)
        except Exception:      # noqa: BLE001
            continue
        for i in range(1, len(t)):
            if asst[i] == 2 and asst[i - 1] != 2 and not hold[i]:
                t0 = t[i]
                for lo, hi, k in ((0, 300_000, 'g_0_0.3'), (300_000, 500_000, 'g_0.3_0.5')):
                    pass
                m1 = (it >= t0) & (it < t0 + 300_000)
                m2 = (it >= t0 + 300_000) & (it < t0 + 500_000)
                mb = (it >= t0 - 300_000) & (it < t0)
                ev = (vt >= t0) & (vt < t0 + 500_000)
                out.append({'rec': fp.split('/')[-2], 't': round((t0 - t[0]) / 1e6, 3), 'thr0': float(out_thr[i]),
                            'g_before_0.3': round(float(g[mb].max()), 2) if mb.any() else None,
                            'g_0_0.3': round(float(g[m1].max()), 2) if m1.any() else None,
                            'g_0.3_0.5': round(float(g[m2].max()), 2) if m2.any() else None,
                            'erpm_max_0_0.5': float(er[ev].max()) if ev.any() else None})
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'scan_brake_g.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in out:
        print(r)
    print('n', len(out), 'g>=2.5 in 0-0.5 s:', sum(1 for r in out if max(r['g_0_0.3'] or 0, r['g_0.3_0.5'] or 0) >= 2.5))


if __name__ == '__main__':
    main()
