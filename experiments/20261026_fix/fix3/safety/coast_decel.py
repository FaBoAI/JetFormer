"""直す 3 回目 safety: リカバリーの後退側をやめたあと(中立の帯)に、後ろへ惰性で動く車がどれだけで減速するか(記録から)。
10/20〜10/24 の AI の記録(0 バイトは飛ばす)で、リカバリー(assist = 2)の中で「後退側(> 中立 + 40)→ 中立の帯」に変わった
フレームから、後ろへ 1000 eRPM 以上で動いていて、そのあと 0.5 秒に 2.5G 以上の衝撃が無い場面を集め、eRPM の減り方を見る。
読むだけ(装置・API に触れない)。結果 out/coast_decel.json"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
KV = 0.5e-3                     # m/s / eRPM(safety_caps.KV)
OUT = Path(__file__).resolve().parent / 'out' / 'coast_decel.json'


def main():
    cases = []
    for sc in sorted(glob.glob(str(ROOT / 'data/raw/202610[12]*/*/AI_CMD.npz'))):
        d = os.path.dirname(sc)
        vp, ip, lp = d + '/VEHICLE.npz', d + '/IMU.npz', os.path.dirname(d) + '/log.json'
        if any(not os.path.exists(x) or os.path.getsize(x) == 0 for x in (sc, vp, ip)):
            continue
        try:
            z = np.load(sc); v = np.load(vp); im = np.load(ip)
            n_us = float(json.load(open(lp))['drive']['throttle_neutral_us']) if os.path.exists(lp) else 1496.0
        except Exception:        # noqa: BLE001  読めない記録は飛ばす
            continue
        if 'assist' not in z.files or 'out' not in z.files:
            continue                # 補助の欄が無い古い記録
        t = z['timestamps_us'].astype(np.int64); a = z['assist'].astype(int); out = z['out'].astype(float)
        hold = z['hold'].astype(bool)
        vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
        it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
        hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
        for i in range(1, len(t)):
            if not (a[i] == 2 and a[i - 1] == 2 and out[i - 1, 1] > n_us + 40 and abs(out[i, 1] - n_us) <= 40):
                continue
            t0 = t[i]
            m = (vt >= t0) & (vt <= t0 + 600_000)
            if m.sum() < 5:
                continue
            e0 = ve[vt <= t0][-1] if (vt <= t0).any() else 0.0
            if e0 < 1000:
                continue
            if (hg[(it >= t0) & (it <= t0 + 500_000)] >= 2.5).any():
                continue
            # 中立の帯が続く間だけ(次のフレームの出力が帯を出たら打ち切り)
            j = i
            while j + 1 < len(t) and abs(out[j + 1, 1] - n_us) <= 40 and not hold[j + 1] and t[j + 1] - t0 <= 600_000:
                j += 1
            t_end = t[j]
            mm = (vt >= t0) & (vt <= t_end)
            tt, ee = (vt[mm] - t0) / 1e6, ve[mm]
            if len(tt) < 4 or tt[-1] < 0.15:
                continue
            k = np.polyfit(tt, ee, 1)[0]            # eRPM/秒(負 = 減る)
            # 止まるまでの距離(この場面の記録のまま): eRPM > 300 の間の ∫v dt
            back = ee > 300
            dist = float(np.sum(ee[back][:-1] * KV * np.diff(tt[back]))) if back.sum() > 1 else 0.0
            cases.append({'scene': os.path.relpath(sc, ROOT / 'data/raw'), 'row': int(i), 'erpm0': int(e0),
                          'v0_mps': round(e0 * KV, 2), 'coast_s': round(float(tt[-1]), 3),
                          'slope_erpm_per_s': round(float(k), 0), 'decel_mps2': round(float(-k * KV), 2),
                          'erpm_end': int(ee[-1]), 'dist_in_window_m': round(dist, 3)})
    dec = [c['decel_mps2'] for c in cases]
    res = {'n': len(cases), 'decel_mps2_median': float(np.median(dec)) if dec else None,
           'decel_mps2_min': float(np.min(dec)) if dec else None, 'decel_mps2_max': float(np.max(dec)) if dec else None,
           'cases': cases}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'cases'}, ensure_ascii=False))
    for c in cases:
        print(c)


if __name__ == '__main__':
    main()
