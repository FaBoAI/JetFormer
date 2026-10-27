"""10/10 20261010-181504-0004(7.5G の衝突の 0.43 秒あとに中立止め 2 フレーム)を、今の assist.Assist に通す(読むだけ)。
A = 今の走行ループと同じ(中立止めの行で helper.reset())、B = 中立止めの行で reset しない(比べるため)。
入力: 記録の model_out(補助の前の出力)・撮像時刻までの最新の eRPM・直近 0.15 秒の水平の衝撃"""
import json, sys
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A
d = '/home/jetson/jetformer/data/raw/20261010-181504/20261010-181504-0004/'
z = np.load(d + 'AI_CMD.npz'); v = np.load(d + 'VEHICLE.npz'); imu = np.load(d + 'IMU.npz')
t = z['timestamps_us'] / 1e6; tv = v['timestamps_us'] / 1e6; ti = imu['timestamps_us'] / 1e6
hg = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / 9.80665
ref = (1443.0, 1498.0)
res = {}
NOISE = "--no-noise" in sys.argv
for mode in ("A_reset_on_hold", "B_no_reset"):
    h = A.Assist(ref)
    first_rec = None
    out = []
    for i in range(100, 200):
        if not z['select'][i]:
            break
        if z['hold'][i]:
            if mode.startswith('A'):
                h.launch_hold(); h.reset()
            out.append((i, 'hold'))
            continue
        e = float(v['erpm'][np.searchsorted(tv, t[i], side='right') - 1])
        if NOISE and i >= 143 and abs(e) >= 1000:
            e = 0.0          # what-if: 衝突のあとの eRPM の雑音(+1386 など)が無かったら
        m = (ti > t[i] - 0.15) & (ti <= t[i] + 0.01)
        g = float(hg[m].max()) if m.any() else 0.0
        o, k = h.step(tuple(float(x) for x in z['model_out'][i]), e, True, float(t[i]), esc_on=True, rec_on=True, impact_g=g)
        if k == A.ASSIST_RECOVERY and first_rec is None:
            first_rec = i
        out.append((i, int(k), round(o[1], 1)))
    res[mode] = {'first_recovery_row': first_rec,
                 'first_recovery_s_after_hold': None if first_rec is None else round(float(t[first_rec] - t[141]), 3),
                 'recorded_first_recovery_row': int(np.where(z['assist'][100:200] == 2)[0][0] + 100),
                 'rows_141_180': [x for x in out if 141 <= x[0] <= 180]}
print(json.dumps({k: {kk: vv for kk, vv in r.items() if kk != 'rows_141_180'} for k, r in res.items()}, ensure_ascii=False, indent=1))
